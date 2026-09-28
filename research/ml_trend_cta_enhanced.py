#!/usr/bin/env python3
"""
ML-Enhanced Trend CTA Strategy
================================
Tests whether LightGBM (GPU) can improve a validated simple trend CTA strategy
(6-month momentum, top-3 equal weight, monthly rebalance across 8 ETFs).

Hypothesis: ML may add value via POSITION SIZING (not timing).
Baseline: Sharpe 0.91, CAGR 11.3%, MaxDD -18.2% over 20 years.

Variants:
  A: ML ranking replaces momentum ranking (top-3 equal weight)
  B: ML ranking with inverse-vol sizing within top-3
  C: ML confidence-weighted (proportional to predicted rank gap)
  Baseline: Simple 6m momentum top-3 equal weight
"""

import os
import sys
import json
import warnings
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import yfinance as yf
from scipy import stats

warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────────────────────────
UNIVERSE = ['SPY', 'EFA', 'EEM', 'TLT', 'IEF', 'GLD', 'DBC', 'VNQ']
CASH = 'SHY'
ALL_TICKERS = UNIVERSE + [CASH]
TOP_K = 3
TRAIN_DAYS = 504   # ~2 years of trading days
TEST_DAYS = 21     # 1 month
START_DATE = '2004-01-01'  # extra history for feature warmup
END_DATE = '2026-07-18'
BACKTEST_START = '2006-01-01'
N_PERMUTATIONS = 200
OUTPUT_DIR = Path('/home/nick/Lvl3Quant/output/ml_trend_cta_enhanced')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# MLflow setup
MLFLOW_URI = 'http://jupiter:5000'
try:
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment('ml_trend_cta_enhanced')
    USE_MLFLOW = True
    print(f"[OK] MLflow connected: {MLFLOW_URI}")
except Exception as e:
    USE_MLFLOW = False
    print(f"[WARN] MLflow unavailable: {e}")


def download_data():
    """Download price data for all tickers + VIX."""
    cache_file = OUTPUT_DIR / 'price_cache.parquet'
    vix_cache = OUTPUT_DIR / 'vix_cache.parquet'
    
    if cache_file.exists() and vix_cache.exists():
        mod_time = dt.datetime.fromtimestamp(cache_file.stat().st_mtime)
        if (dt.datetime.now() - mod_time).days < 1:
            print("[CACHE] Loading cached price data")
            prices = pd.read_parquet(cache_file)
            vix = pd.read_parquet(vix_cache)
            return prices, vix
    
    print("[DL] Downloading price data...")
    prices = yf.download(ALL_TICKERS, start=START_DATE, end=END_DATE, progress=False)['Close']
    # Handle MultiIndex columns from yfinance
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.get_level_values(-1)
    prices = prices.dropna()
    
    print("[DL] Downloading VIX data...")
    vix_raw = yf.download(['^VIX', '^VIX3M'], start=START_DATE, end=END_DATE, progress=False)['Close']
    if isinstance(vix_raw.columns, pd.MultiIndex):
        vix_raw.columns = vix_raw.columns.get_level_values(-1)
    vix = pd.DataFrame(index=prices.index)
    vix['VIX'] = vix_raw['^VIX'].reindex(prices.index).ffill()
    vix['VIX3M'] = vix_raw['^VIX3M'].reindex(prices.index).ffill()
    vix['VIX_RATIO'] = vix['VIX'] / vix['VIX3M'].replace(0, np.nan)
    
    prices.to_parquet(cache_file)
    vix.to_parquet(vix_cache)
    print(f"[OK] Data: {prices.shape[0]} days, {prices.shape[1]} tickers")
    return prices, vix


def compute_features(prices, vix):
    """
    Compute features for each asset at each date. All features use T-1 data.
    Returns: DataFrame with MultiIndex (date, ticker) and feature columns.
    """
    returns = prices.pct_change()
    log_returns = np.log(prices / prices.shift(1))
    
    # Rolling correlations (63d) for cross-asset correlation feature
    corr_63d = returns[UNIVERSE].rolling(63).corr()
    
    records = []
    
    for ticker in UNIVERSE:
        r = returns[ticker]
        p = prices[ticker]
        lr = log_returns[ticker]
        
        # Momentum features
        mom_6m = p / p.shift(126) - 1
        mom_3m = p / p.shift(63) - 1
        mom_1m = p / p.shift(21) - 1
        
        # Realized vol
        vol_21d = lr.rolling(21).std() * np.sqrt(252)
        vol_63d = lr.rolling(63).std() * np.sqrt(252)
        
        # 63d Sharpe
        sharpe_63d = (lr.rolling(63).mean() * 252) / (lr.rolling(63).std() * np.sqrt(252)).replace(0, np.nan)
        
        # Price vs 200d SMA
        sma_200 = p.rolling(200).mean()
        price_vs_sma = p / sma_200.replace(0, np.nan)
        
        # Trailing 12m max drawdown
        rolling_max = p.rolling(252).max()
        drawdown = (p - rolling_max) / rolling_max.replace(0, np.nan)
        max_dd_12m = drawdown.rolling(252).min()
        
        # Cross-asset mean correlation (63d rolling)
        # For each date, get this ticker's mean absolute correlation with others
        mean_corr = pd.Series(index=prices.index, dtype=float)
        for date in prices.index:
            try:
                if date in corr_63d.index.get_level_values(0):
                    corr_matrix = corr_63d.loc[date]
                    if ticker in corr_matrix.index:
                        row = corr_matrix.loc[ticker].drop(ticker, errors='ignore')
                        mean_corr[date] = row.abs().mean()
            except:
                pass
        mean_corr = mean_corr.ffill()
        
        # Next-month return (target)
        fwd_ret_21d = r.shift(-21).rolling(21).sum().shift(-20)  # simpler: use cumulative
        fwd_ret_21d = (p.shift(-21) / p) - 1  # exact next-21d return
        
        feat_df = pd.DataFrame({
            'ticker': ticker,
            'mom_6m': mom_6m,
            'mom_3m': mom_3m,
            'mom_1m': mom_1m,
            'vol_21d': vol_21d,
            'vol_63d': vol_63d,
            'sharpe_63d': sharpe_63d,
            'price_vs_sma200': price_vs_sma,
            'vix': vix['VIX'],
            'vix_ratio': vix['VIX_RATIO'],
            'max_dd_12m': max_dd_12m,
            'mean_corr_63d': mean_corr,
            'fwd_ret_21d': fwd_ret_21d,
        }, index=prices.index)
        
        records.append(feat_df)
    
    df = pd.concat(records, ignore_index=False)
    df = df.dropna(subset=['mom_6m', 'vol_21d', 'fwd_ret_21d'])
    
    # Compute rank target: within each date, rank 1 (best) to 8 (worst) by fwd return
    df['fwd_rank'] = df.groupby(df.index)['fwd_ret_21d'].rank(ascending=False)
    
    print(f"[OK] Features computed: {len(df)} rows, {df.index.nunique()} dates")
    return df


def compute_features_fast(prices, vix):
    """
    Vectorized feature computation — much faster than the row-by-row correlation loop.
    """
    returns = prices.pct_change()
    log_returns = np.log(prices / prices.shift(1))
    
    # Pre-compute rolling correlation means for all tickers at once
    print("[FEAT] Computing rolling correlations (vectorized)...")
    corr_means = {}
    for ticker in UNIVERSE:
        # Mean absolute correlation of this ticker with all others over 63d window
        other_tickers = [t for t in UNIVERSE if t != ticker]
        pair_corrs = pd.DataFrame(index=prices.index)
        for other in other_tickers:
            pair_corrs[other] = returns[ticker].rolling(63).corr(returns[other]).abs()
        corr_means[ticker] = pair_corrs.mean(axis=1)
    
    records = []
    for ticker in UNIVERSE:
        r = returns[ticker]
        p = prices[ticker]
        lr = log_returns[ticker]
        
        mom_6m = p / p.shift(126) - 1
        mom_3m = p / p.shift(63) - 1
        mom_1m = p / p.shift(21) - 1
        vol_21d = lr.rolling(21).std() * np.sqrt(252)
        vol_63d = lr.rolling(63).std() * np.sqrt(252)
        sharpe_63d = (lr.rolling(63).mean() * 252) / (lr.rolling(63).std() * np.sqrt(252)).replace(0, np.nan)
        sma_200 = p.rolling(200).mean()
        price_vs_sma = p / sma_200.replace(0, np.nan)
        rolling_max = p.rolling(252).max()
        drawdown = (p - rolling_max) / rolling_max.replace(0, np.nan)
        max_dd_12m = drawdown.rolling(252).min()
        fwd_ret_21d = (p.shift(-21) / p) - 1
        
        feat_df = pd.DataFrame({
            'ticker': ticker,
            'mom_6m': mom_6m,
            'mom_3m': mom_3m,
            'mom_1m': mom_1m,
            'vol_21d': vol_21d,
            'vol_63d': vol_63d,
            'sharpe_63d': sharpe_63d,
            'price_vs_sma200': price_vs_sma,
            'vix': vix['VIX'],
            'vix_ratio': vix['VIX_RATIO'],
            'max_dd_12m': max_dd_12m,
            'mean_corr_63d': corr_means[ticker],
            'fwd_ret_21d': fwd_ret_21d,
        }, index=prices.index)
        records.append(feat_df)
    
    df = pd.concat(records, ignore_index=False)
    df = df.dropna(subset=['mom_6m', 'vol_21d', 'fwd_ret_21d'])
    df['fwd_rank'] = df.groupby(df.index)['fwd_ret_21d'].rank(ascending=False)
    
    print(f"[OK] Features computed: {len(df)} rows, {df.index.nunique()} dates")
    return df


FEATURE_COLS = [
    'mom_6m', 'mom_3m', 'mom_1m', 'vol_21d', 'vol_63d',
    'sharpe_63d', 'price_vs_sma200', 'vix', 'vix_ratio',
    'max_dd_12m', 'mean_corr_63d'
]


def walk_forward_train(df):
    """
    Walk-forward training with sliding window.
    504d train, 21d test, slide by 21d.
    Returns predictions DataFrame.
    """
    dates = sorted(df.index.unique())
    backtest_start = pd.Timestamp(BACKTEST_START)
    dates = [d for d in dates if d >= backtest_start]
    
    all_preds = []
    n_windows = 0
    
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'n_estimators': 300,
        'verbose': -1,
        'seed': 42,
    }
    
    i = 0
    while i + TRAIN_DAYS + TEST_DAYS <= len(dates):
        train_dates = dates[i:i + TRAIN_DAYS]
        test_dates = dates[i + TRAIN_DAYS:i + TRAIN_DAYS + TEST_DAYS]
        
        train_df = df[df.index.isin(train_dates)]
        test_df = df[df.index.isin(test_dates)]
        
        if len(test_df) == 0:
            i += TEST_DAYS
            continue
        
        X_train = train_df[FEATURE_COLS].values
        y_train = train_df['fwd_rank'].values
        X_test = test_df[FEATURE_COLS].values
        
        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(X_train, y_train, eval_set=[(X_test, test_df['fwd_rank'].values)],
                  callbacks=[lgb.log_evaluation(0)])
        
        preds = model.predict(X_test)
        
        result = test_df[['ticker', 'fwd_ret_21d', 'fwd_rank']].copy()
        result['pred_rank'] = preds
        result['window'] = n_windows
        all_preds.append(result)
        
        n_windows += 1
        i += TEST_DAYS
        
        if n_windows % 50 == 0:
            print(f"  [WF] Window {n_windows}, train: {train_dates[0].strftime('%Y-%m-%d')} to {train_dates[-1].strftime('%Y-%m-%d')}, test: {test_dates[0].strftime('%Y-%m-%d')}")
    
    print(f"[OK] Walk-forward complete: {n_windows} windows")
    preds_df = pd.concat(all_preds)
    
    # Feature importance from last model
    feat_imp = dict(zip(FEATURE_COLS, model.feature_importances_))
    
    return preds_df, feat_imp


def baseline_strategy(prices):
    """Simple 6m momentum, top-3 equal weight, monthly rebalance."""
    returns = prices[UNIVERSE].pct_change()
    monthly_dates = prices.resample('ME').last().index
    monthly_dates = monthly_dates[monthly_dates >= BACKTEST_START]
    
    portfolio_returns = []
    
    for i, date in enumerate(monthly_dates[:-1]):
        # 6m momentum ranking
        lookback_start = date - pd.DateOffset(months=6)
        mask = (prices.index >= lookback_start) & (prices.index <= date)
        period_prices = prices[UNIVERSE][mask]
        
        if len(period_prices) < 60:
            continue
        
        mom = (period_prices.iloc[-1] / period_prices.iloc[0]) - 1
        top_k = mom.nlargest(TOP_K).index.tolist()
        
        # Equal weight in top-3, hold for next month
        next_date = monthly_dates[i + 1]
        month_mask = (returns.index > date) & (returns.index <= next_date)
        month_rets = returns[month_mask]
        
        if len(month_rets) == 0:
            continue
        
        # Portfolio return = equal weight average of top-3
        port_ret = month_rets[top_k].mean(axis=1)
        portfolio_returns.append(port_ret)
    
    return pd.concat(portfolio_returns)


def ml_strategy_A(preds_df, prices):
    """ML ranking replaces momentum ranking, top-3 equal weight."""
    returns = prices[UNIVERSE].pct_change()
    
    # Get monthly rebalance dates from predictions
    # Use the first date of each test window as rebalance date
    preds_df = preds_df.copy()
    preds_df['date'] = preds_df.index
    
    # Group by window, pick top-3 by ML predicted rank (lower = better)
    portfolio_returns = []
    
    for window_id in sorted(preds_df['window'].unique()):
        window_data = preds_df[preds_df['window'] == window_id]
        rebal_date = window_data['date'].min()
        end_date = window_data['date'].max()
        
        # Get the prediction at rebalance date
        rebal_preds = window_data[window_data['date'] == rebal_date]
        if len(rebal_preds) < len(UNIVERSE):
            # Use all predictions in the window's first few days
            rebal_preds = window_data.groupby('ticker')['pred_rank'].mean().reset_index()
            rebal_preds = rebal_preds.set_index('ticker')
        else:
            rebal_preds = rebal_preds.set_index('ticker')
        
        # Top-3 by predicted rank (lower pred = better predicted performer)
        top_k = rebal_preds['pred_rank'].nsmallest(TOP_K).index.tolist()
        
        # Returns for the month
        month_mask = (returns.index >= rebal_date) & (returns.index <= end_date)
        month_rets = returns[month_mask]
        
        if len(month_rets) > 0:
            port_ret = month_rets[top_k].mean(axis=1)
            portfolio_returns.append(port_ret)
    
    if not portfolio_returns:
        return pd.Series(dtype=float)
    return pd.concat(portfolio_returns)


def ml_strategy_B(preds_df, prices):
    """ML ranking with inverse-vol position sizing within top-3."""
    returns = prices[UNIVERSE].pct_change()
    log_returns = np.log(prices[UNIVERSE] / prices[UNIVERSE].shift(1))
    
    preds_df = preds_df.copy()
    preds_df['date'] = preds_df.index
    
    portfolio_returns = []
    
    for window_id in sorted(preds_df['window'].unique()):
        window_data = preds_df[preds_df['window'] == window_id]
        rebal_date = window_data['date'].min()
        end_date = window_data['date'].max()
        
        rebal_preds = window_data[window_data['date'] == rebal_date]
        if len(rebal_preds) < len(UNIVERSE):
            rebal_preds = window_data.groupby('ticker')['pred_rank'].mean().reset_index()
            rebal_preds = rebal_preds.set_index('ticker')
        else:
            rebal_preds = rebal_preds.set_index('ticker')
        
        top_k = rebal_preds['pred_rank'].nsmallest(TOP_K).index.tolist()
        
        # Inverse-vol weighting
        vols = {}
        for ticker in top_k:
            vol_mask = (log_returns.index < rebal_date) & (log_returns.index >= rebal_date - pd.DateOffset(days=63))
            vol = log_returns[ticker][vol_mask].std() * np.sqrt(252)
            vols[ticker] = max(vol, 0.01)  # floor
        
        inv_vols = {t: 1.0 / v for t, v in vols.items()}
        total = sum(inv_vols.values())
        weights = {t: v / total for t, v in inv_vols.items()}
        
        month_mask = (returns.index >= rebal_date) & (returns.index <= end_date)
        month_rets = returns[month_mask]
        
        if len(month_rets) > 0:
            port_ret = sum(month_rets[t] * w for t, w in weights.items())
            portfolio_returns.append(port_ret)
    
    if not portfolio_returns:
        return pd.Series(dtype=float)
    return pd.concat(portfolio_returns)


def ml_strategy_C(preds_df, prices):
    """ML confidence-weighted: allocate proportional to predicted rank gap."""
    returns = prices[UNIVERSE].pct_change()
    
    preds_df = preds_df.copy()
    preds_df['date'] = preds_df.index
    
    portfolio_returns = []
    
    for window_id in sorted(preds_df['window'].unique()):
        window_data = preds_df[preds_df['window'] == window_id]
        rebal_date = window_data['date'].min()
        end_date = window_data['date'].max()
        
        rebal_preds = window_data[window_data['date'] == rebal_date]
        if len(rebal_preds) < len(UNIVERSE):
            rebal_preds = window_data.groupby('ticker')['pred_rank'].mean().reset_index()
            rebal_preds = rebal_preds.set_index('ticker')
        else:
            rebal_preds = rebal_preds.set_index('ticker')
        
        top_k = rebal_preds['pred_rank'].nsmallest(TOP_K).index.tolist()
        
        # Confidence = inverse of predicted rank (rank 1 gets highest weight)
        scores = {}
        for t in top_k:
            # Lower predicted rank = better, so use (max_rank - pred_rank) as score
            scores[t] = max(0.1, 9.0 - rebal_preds.loc[t, 'pred_rank'])
        
        total = sum(scores.values())
        weights = {t: v / total for t, v in scores.items()}
        
        month_mask = (returns.index >= rebal_date) & (returns.index <= end_date)
        month_rets = returns[month_mask]
        
        if len(month_rets) > 0:
            port_ret = sum(month_rets[t] * w for t, w in weights.items())
            portfolio_returns.append(port_ret)
    
    if not portfolio_returns:
        return pd.Series(dtype=float)
    return pd.concat(portfolio_returns)


def compute_metrics(returns, name="Strategy"):
    """Compute risk-adjusted metrics."""
    if len(returns) == 0:
        return {}
    
    returns = returns.dropna()
    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    
    downside = returns[returns < 0].std() * np.sqrt(252)
    sortino = ann_ret / downside if downside > 0 else 0
    
    # Max drawdown
    cum = (1 + returns).cumprod()
    rolling_max = cum.expanding().max()
    dd = (cum - rolling_max) / rolling_max
    max_dd = dd.min()
    
    # CAGR
    n_years = len(returns) / 252
    total_ret = cum.iloc[-1] - 1 if len(cum) > 0 else 0
    cagr = (1 + total_ret) ** (1 / max(n_years, 0.01)) - 1
    
    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    
    # Win rate (monthly)
    monthly_rets = returns.resample('ME').sum()
    wr = (monthly_rets > 0).mean()
    
    # Profit factor
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    pf = gains / losses if losses > 0 else float('inf')
    
    return {
        'name': name,
        'ann_return': ann_ret,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_dd': max_dd,
        'cagr': cagr,
        'calmar': calmar,
        'monthly_wr': wr,
        'profit_factor': pf,
        'n_days': len(returns),
        'n_years': n_years,
    }


def permutation_test(preds_df, prices, n_perms=N_PERMUTATIONS):
    """
    Permutation test: shuffle predicted ranks within each date to test significance.
    Returns p-value for Sharpe improvement over random ranking.
    """
    print(f"\n[PERM] Running {n_perms} permutations...")
    
    real_returns = ml_strategy_A(preds_df, prices)
    real_sharpe = compute_metrics(real_returns)['sharpe']
    
    perm_sharpes = []
    rng = np.random.RandomState(42)
    
    for p in range(n_perms):
        perm_preds = preds_df.copy()
        # Shuffle predicted ranks within each date
        for date in perm_preds.index.unique():
            mask = perm_preds.index == date
            vals = perm_preds.loc[mask, 'pred_rank'].values.copy()
            rng.shuffle(vals)
            perm_preds.loc[mask, 'pred_rank'] = vals
        
        perm_returns = ml_strategy_A(perm_preds, prices)
        if len(perm_returns) > 0:
            perm_sharpe = compute_metrics(perm_returns)['sharpe']
            perm_sharpes.append(perm_sharpe)
        
        if (p + 1) % 50 == 0:
            print(f"  [PERM] {p+1}/{n_perms} done")
    
    perm_sharpes = np.array(perm_sharpes)
    p_value = (perm_sharpes >= real_sharpe).mean()
    
    print(f"  [PERM] Real Sharpe: {real_sharpe:.3f}")
    print(f"  [PERM] Perm Sharpe mean: {perm_sharpes.mean():.3f} ± {perm_sharpes.std():.3f}")
    print(f"  [PERM] p-value: {p_value:.4f}")
    
    return p_value, real_sharpe, perm_sharpes


def regime_analysis(returns, prices):
    """Analyze strategy performance in different market regimes."""
    spy = prices['SPY'].pct_change().reindex(returns.index)
    
    # Bull/Bear based on SPY 200d SMA
    sma_200 = prices['SPY'].rolling(200).mean().reindex(returns.index)
    spy_price = prices['SPY'].reindex(returns.index)
    
    bull_mask = spy_price > sma_200
    bear_mask = ~bull_mask
    
    # High/Low vol based on VIX (if available, use realized vol as proxy)
    spy_vol = spy.rolling(63).std() * np.sqrt(252)
    median_vol = spy_vol.median()
    high_vol_mask = spy_vol > median_vol
    low_vol_mask = ~high_vol_mask
    
    results = {}
    for regime_name, mask in [('Bull', bull_mask), ('Bear', bear_mask), 
                               ('High Vol', high_vol_mask), ('Low Vol', low_vol_mask)]:
        regime_rets = returns[mask].dropna()
        if len(regime_rets) > 60:
            m = compute_metrics(regime_rets, regime_name)
            results[regime_name] = m
    
    return results


def sub_period_analysis(returns):
    """Split into sub-periods for stability check."""
    n = len(returns)
    third = n // 3
    
    periods = {
        'First Third': returns.iloc[:third],
        'Middle Third': returns.iloc[third:2*third],
        'Last Third': returns.iloc[2*third:],
    }
    
    results = {}
    for name, rets in periods.items():
        results[name] = compute_metrics(rets, name)
    
    return results


def lag_sensitivity(df, prices):
    """Test sensitivity to feature lag (1d, 2d, 5d)."""
    results = {}
    
    for lag in [1, 2, 5]:
        print(f"  [LAG] Testing lag={lag}d...")
        lagged_df = df.copy()
        for col in FEATURE_COLS:
            lagged_df[col] = lagged_df.groupby('ticker')[col].shift(lag - 1)  # already T-1, add more
        lagged_df = lagged_df.dropna(subset=FEATURE_COLS + ['fwd_rank'])
        
        preds, _ = walk_forward_train(lagged_df)
        rets = ml_strategy_A(preds, prices)
        results[f'lag_{lag}d'] = compute_metrics(rets, f'Lag {lag}d')
    
    return results


def rank_ic_analysis(preds_df):
    """Compute rank IC (Spearman correlation) between predicted and actual ranks."""
    ics = []
    for date in preds_df.index.unique():
        day_data = preds_df[preds_df.index == date]
        if len(day_data) >= 4:
            ic, _ = stats.spearmanr(day_data['pred_rank'], day_data['fwd_rank'])
            ics.append({'date': date, 'ic': ic})
    
    ic_df = pd.DataFrame(ics)
    if len(ic_df) == 0:
        return 0, 0, 0
    
    mean_ic = ic_df['ic'].mean()
    ic_ir = mean_ic / ic_df['ic'].std() if ic_df['ic'].std() > 0 else 0
    hit_rate = (ic_df['ic'] > 0).mean()
    
    return mean_ic, ic_ir, hit_rate


def main():
    print("=" * 70)
    print("ML-Enhanced Trend CTA Strategy")
    print("=" * 70)
    
    if USE_MLFLOW:
        run = mlflow.start_run(run_name=f"ml_trend_cta_{dt.datetime.now().strftime('%Y%m%d_%H%M')}")
    
    # Step 1: Download data
    prices, vix = download_data()
    
    # Step 2: Compute features
    print("\n[STEP 2] Computing features...")
    df = compute_features_fast(prices, vix)
    
    # Step 3: Walk-forward training
    print("\n[STEP 3] Walk-forward LightGBM training (GPU)...")
    preds_df, feat_imp = walk_forward_train(df)
    preds_df.to_parquet(OUTPUT_DIR / 'predictions.parquet')
    
    # Rank IC analysis
    mean_ic, ic_ir, ic_hit = rank_ic_analysis(preds_df)
    print(f"\n[RANK IC] Mean IC: {mean_ic:.4f}, ICIR: {ic_ir:.4f}, Hit Rate: {ic_hit:.1%}")
    
    # Step 4: Run all strategies
    print("\n[STEP 4] Running strategies...")
    
    baseline_rets = baseline_strategy(prices)
    ml_a_rets = ml_strategy_A(preds_df, prices)
    ml_b_rets = ml_strategy_B(preds_df, prices)
    ml_c_rets = ml_strategy_C(preds_df, prices)
    
    strategies = {
        'Baseline (6m Mom)': baseline_rets,
        'Variant A (ML Rank)': ml_a_rets,
        'Variant B (ML + InvVol)': ml_b_rets,
        'Variant C (ML Conf Wt)': ml_c_rets,
    }
    
    all_metrics = {}
    print("\n" + "=" * 90)
    print(f"{'Strategy':<25} {'Sharpe':>8} {'Sortino':>8} {'CAGR':>8} {'MaxDD':>8} {'MoWR':>8} {'PF':>8} {'Calmar':>8}")
    print("-" * 90)
    
    for name, rets in strategies.items():
        m = compute_metrics(rets, name)
        all_metrics[name] = m
        print(f"{name:<25} {m['sharpe']:>8.3f} {m['sortino']:>8.3f} {m['cagr']:>7.1%} {m['max_dd']:>7.1%} {m['monthly_wr']:>7.1%} {m['profit_factor']:>8.2f} {m['calmar']:>8.3f}")
    
    print("=" * 90)
    
    # Step 5: Feature importance
    print("\n[FEATURE IMPORTANCE]")
    sorted_imp = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)
    for feat, imp in sorted_imp:
        print(f"  {feat:<20} {imp:>6}")
    
    # Step 6: Permutation test
    p_value, real_sharpe, perm_sharpes = permutation_test(preds_df, prices)
    
    # Step 7: Regime analysis
    print("\n[REGIME ANALYSIS]")
    best_variant = max(all_metrics, key=lambda k: all_metrics[k].get('sharpe', 0) if k != 'Baseline (6m Mom)' else -999)
    best_rets = strategies[best_variant]
    
    regime_results = regime_analysis(best_rets, prices)
    baseline_regime = regime_analysis(baseline_rets, prices)
    
    print(f"\n  {'Regime':<15} {'ML Sharpe':>10} {'Base Sharpe':>12} {'ML CAGR':>10} {'Base CAGR':>10}")
    print("  " + "-" * 60)
    for regime in regime_results:
        ml_s = regime_results[regime]['sharpe']
        bl_s = baseline_regime.get(regime, {}).get('sharpe', 0)
        ml_c = regime_results[regime]['cagr']
        bl_c = baseline_regime.get(regime, {}).get('cagr', 0)
        print(f"  {regime:<15} {ml_s:>10.3f} {bl_s:>12.3f} {ml_c:>9.1%} {bl_c:>9.1%}")
    
    # Regime-agnostic check (HC #428)
    if len(regime_results) >= 2 and 'Bull' in regime_results and 'Bear' in regime_results:
        bull_sharpe = regime_results['Bull']['sharpe']
        bear_sharpe = regime_results['Bear']['sharpe']
        regime_gap = abs(bull_sharpe - bear_sharpe) / max(abs(bull_sharpe), abs(bear_sharpe), 0.01)
        print(f"\n  Regime Gap (|Bull-Bear|/max): {regime_gap:.3f} {'PASS' if regime_gap <= 0.50 else 'FAIL (>0.50)'}")
    
    # Step 8: Sub-period stability
    print("\n[SUB-PERIOD STABILITY]")
    sub_results = sub_period_analysis(best_rets)
    baseline_sub = sub_period_analysis(baseline_rets)
    
    print(f"  {'Period':<15} {'ML Sharpe':>10} {'Base Sharpe':>12}")
    print("  " + "-" * 40)
    for period in sub_results:
        ml_s = sub_results[period]['sharpe']
        bl_s = baseline_sub.get(period, {}).get('sharpe', 0)
        print(f"  {period:<15} {ml_s:>10.3f} {bl_s:>12.3f}")
    
    # Step 9: Lag sensitivity (skip if time-constrained, run only lag 2 and 5)
    print("\n[LAG SENSITIVITY]")
    for lag in [2, 5]:
        print(f"  Testing lag={lag}d...")
        lagged_df = df.copy()
        for col in FEATURE_COLS:
            lagged_df[col] = lagged_df.groupby('ticker')[col].shift(lag - 1)
        lagged_df = lagged_df.dropna(subset=FEATURE_COLS + ['fwd_rank'])
        
        lag_preds, _ = walk_forward_train(lagged_df)
        lag_rets = ml_strategy_A(lag_preds, prices)
        lag_m = compute_metrics(lag_rets, f'Lag {lag}d')
        print(f"  Lag {lag}d: Sharpe={lag_m['sharpe']:.3f}, CAGR={lag_m['cagr']:.1%}")
    
    # ── Summary and Verdict ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    
    bl_sharpe = all_metrics['Baseline (6m Mom)']['sharpe']
    best_ml_name = max([k for k in all_metrics if k != 'Baseline (6m Mom)'], 
                       key=lambda k: all_metrics[k]['sharpe'])
    best_ml_sharpe = all_metrics[best_ml_name]['sharpe']
    
    sharpe_improvement = best_ml_sharpe - bl_sharpe
    
    print(f"  Baseline Sharpe:     {bl_sharpe:.3f}")
    print(f"  Best ML Sharpe:      {best_ml_sharpe:.3f} ({best_ml_name})")
    print(f"  Sharpe Improvement:  {sharpe_improvement:+.3f}")
    print(f"  Permutation p-value: {p_value:.4f}")
    print(f"  Rank IC:             {mean_ic:.4f} (ICIR: {ic_ir:.4f})")
    
    if sharpe_improvement > 0.05 and p_value < 0.05:
        verdict = "ML ADDS SIGNIFICANT VALUE"
    elif sharpe_improvement > 0 and p_value < 0.10:
        verdict = "ML shows marginal improvement (borderline significant)"
    elif sharpe_improvement > 0:
        verdict = "ML shows improvement but NOT statistically significant"
    else:
        verdict = "ML does NOT improve over simple momentum (as expected)"
    
    print(f"\n  >>> {verdict} <<<")
    
    # ── Save results ─────────────────────────────────────────────────────
    results = {
        'metrics': {k: {kk: (float(vv) if isinstance(vv, (np.floating, float)) else vv) 
                        for kk, vv in v.items()} for k, v in all_metrics.items()},
        'rank_ic': {'mean': float(mean_ic), 'icir': float(ic_ir), 'hit_rate': float(ic_hit)},
        'permutation': {'p_value': float(p_value), 'n_perms': N_PERMUTATIONS},
        'feature_importance': {k: int(v) for k, v in feat_imp.items()},
        'verdict': verdict,
        'regime_results': {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv 
                                for kk, vv in v.items()} for k, v in regime_results.items()},
        'timestamp': dt.datetime.now().isoformat(),
    }
    
    with open(OUTPUT_DIR / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save equity curves
    equity_curves = pd.DataFrame({
        name: (1 + rets).cumprod() for name, rets in strategies.items()
    })
    equity_curves.to_parquet(OUTPUT_DIR / 'equity_curves.parquet')
    
    # MLflow logging
    if USE_MLFLOW:
        try:
            for name, m in all_metrics.items():
                prefix = name.replace(' ', '_').replace('(', '').replace(')', '')
                for k, v in m.items():
                    if isinstance(v, (int, float, np.floating)):
                        mlflow.log_metric(f"{prefix}_{k}", float(v))
            
            mlflow.log_metric('rank_ic_mean', float(mean_ic))
            mlflow.log_metric('rank_ic_ir', float(ic_ir))
            mlflow.log_metric('permutation_pvalue', float(p_value))
            mlflow.log_metric('sharpe_improvement', float(sharpe_improvement))
            
            mlflow.log_params({
                'universe': ','.join(UNIVERSE),
                'top_k': TOP_K,
                'train_days': TRAIN_DAYS,
                'test_days': TEST_DAYS,
                'n_features': len(FEATURE_COLS),
                'verdict': verdict,
            })
            
            mlflow.log_artifact(str(OUTPUT_DIR / 'results.json'))
            mlflow.end_run()
            print("\n[OK] MLflow run logged")
        except Exception as e:
            print(f"[WARN] MLflow logging failed: {e}")
            try:
                mlflow.end_run()
            except:
                pass
    
    print(f"\n[DONE] Results saved to {OUTPUT_DIR}")
    print(f"  - results.json")
    print(f"  - predictions.parquet")
    print(f"  - equity_curves.parquet")


if __name__ == '__main__':
    main()
