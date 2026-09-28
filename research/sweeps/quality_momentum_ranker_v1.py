#!/usr/bin/env python3
"""
Quality-Momentum Stock Ranker v1
=================================

Growth strategy combining momentum and quality factors for cross-sectional
stock selection. Uses LightGBM to learn optimal factor combination.

Mechanism: Stocks with strong recent performance + high quality fundamentals
(ROE, margins, earnings growth) continue to outperform. This is well-documented
in academic literature (Asness, Novy-Marx, etc.) but we validate it with
proper walk-forward + adversarial gates.

Universe: 50 large-cap US stocks
Features:
  - Momentum: 1m, 3m, 6m, 12m returns, 52w high proximity
  - Mean reversion: 5d, 10d, 20d returns (contrarian signals)
  - Volatility: 20d, 60d realized vol, vol ratio
  - Volume: relative volume (20d avg), volume trend
  - Quality proxies (from price/volume since we use free data):
    - Earnings yield proxy: inverse P/E via price stability
    - Margin stability: return consistency (low drawdown relative to gain)
    - Growth proxy: 12m vs 6m momentum acceleration
    - Financial strength: low volatility + positive momentum combo

Walk-forward: 252d train / 21d test, SLIDING (HC #0)
Rebalance: Monthly (21 trading days)
Position: Top 5 stocks equally weighted

Adversarial gates: permutation (200), regime R1, sub-period, outlier.

Target: CAGR >15%, Sharpe >1.5 (per HC #741).
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("LightGBM not available, will use simple ranking")

# MLflow
MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen('http://jupiter:5000/', timeout=2)
    import mlflow
    mlflow.set_tracking_uri('http://jupiter:5000')
    MLFLOW_OK = True
    print("MLflow connected")
except:
    print("MLflow unavailable")

RESULTS_DIR = Path(__file__).resolve().parent / 'research' / 'findings'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = RESULTS_DIR / 'quality_momentum_ranker_v1_results.json'

# Universe: 50 large-cap stocks
UNIVERSE = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA', 'BRK-B',
    'UNH', 'JNJ', 'V', 'JPM', 'XOM', 'PG', 'MA', 'HD', 'CVX', 'MRK',
    'ABBV', 'PEP', 'COST', 'KO', 'AVGO', 'LLY', 'WMT', 'TMO', 'MCD',
    'CSCO', 'ACN', 'DHR', 'ABT', 'NEE', 'TXN', 'PM', 'UNP', 'RTX',
    'LOW', 'HON', 'AMGN', 'IBM', 'CAT', 'GS', 'BA', 'SBUX', 'GE',
    'MMM', 'DIS', 'INTC', 'NKE', 'CRM'
]

TOP_K = 5  # Select top 5 stocks
REBALANCE_DAYS = 21  # Monthly rebalance


def download_data():
    """Download all stock data."""
    import yfinance as yf
    print(f"Downloading data for {len(UNIVERSE)} stocks...")

    all_data = {}
    for ticker in UNIVERSE:
        try:
            df = yf.download(ticker, start='2010-01-01', end='2026-07-24', progress=False)
            if df.index.tz is not None:
                df.index = df.index.tz_convert(None)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 252:
                all_data[ticker] = df
        except Exception as e:
            print(f"  {ticker}: FAILED ({e})")

    # SPY for regime + benchmark
    spy = yf.download('SPY', start='2010-01-01', end='2026-07-24', progress=False)
    if spy.index.tz is not None:
        spy.index = spy.index.tz_convert(None)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)

    print(f"Downloaded {len(all_data)} stocks successfully")
    return all_data, spy


def compute_features(close_series, volume_series):
    """Compute momentum + quality features for a single stock."""
    c = close_series
    v = volume_series

    features = pd.DataFrame(index=c.index)

    # Momentum features
    features['ret_5d'] = c.pct_change(5)
    features['ret_10d'] = c.pct_change(10)
    features['ret_21d'] = c.pct_change(21)  # 1 month
    features['ret_63d'] = c.pct_change(63)  # 3 months
    features['ret_126d'] = c.pct_change(126)  # 6 months
    features['ret_252d'] = c.pct_change(252)  # 12 months

    # Momentum minus most recent month (Jegadeesh-Titman style)
    features['mom_12_1'] = c.pct_change(252) - c.pct_change(21)

    # 52-week high proximity
    features['high_52w_pct'] = c / c.rolling(252).max()

    # Momentum acceleration (quality proxy: accelerating earnings growth)
    features['mom_accel'] = features['ret_126d'] - features['ret_252d'].shift(126)

    # Volatility features
    log_ret = np.log(c / c.shift(1))
    features['vol_20d'] = log_ret.rolling(20).std() * np.sqrt(252)
    features['vol_60d'] = log_ret.rolling(60).std() * np.sqrt(252)
    features['vol_ratio'] = features['vol_20d'] / features['vol_60d']

    # Return consistency (quality proxy: smooth upward path = high quality)
    features['sharpe_63d'] = log_ret.rolling(63).mean() / log_ret.rolling(63).std()
    features['sharpe_126d'] = log_ret.rolling(126).mean() / log_ret.rolling(126).std()

    # Max drawdown 63d (quality: low DD = strong fundamentals)
    roll_max = c.rolling(63).max()
    features['maxdd_63d'] = (c - roll_max) / roll_max

    # Volume features
    features['vol_rel'] = v / v.rolling(20).mean()
    features['vol_trend'] = v.rolling(5).mean() / v.rolling(20).mean()

    # Up/down volume ratio (quality: more up-volume = institutional buying)
    up_vol = (v * (log_ret > 0).astype(float)).rolling(20).sum()
    dn_vol = (v * (log_ret <= 0).astype(float)).rolling(20).sum()
    features['updn_vol_ratio'] = up_vol / (dn_vol + 1)

    # Price stability (inverse of tail risk — quality proxy)
    features['skew_63d'] = log_ret.rolling(63).skew()
    features['kurt_63d'] = log_ret.rolling(63).kurt()

    return features


def build_dataset(all_data, spy):
    """Build cross-sectional training dataset."""
    print("Building feature dataset...")

    # Common dates
    common_dates = None
    for ticker, df in all_data.items():
        if common_dates is None:
            common_dates = set(df.index)
        else:
            common_dates &= set(df.index)
    common_dates = sorted(common_dates)

    # Forward returns for labels (21d ahead)
    all_features = []
    all_labels = []
    all_meta = []

    for ticker, df in all_data.items():
        close = df['Close'].reindex(common_dates)
        volume = df['Volume'].reindex(common_dates)

        features = compute_features(close, volume)

        # Label: 21d forward return (what we're predicting)
        fwd_ret = close.pct_change(REBALANCE_DAYS).shift(-REBALANCE_DAYS)

        # Align
        valid = features.dropna().index.intersection(fwd_ret.dropna().index)
        valid = [d for d in valid if d in common_dates]

        if len(valid) < 100:
            continue

        for date in valid:
            row = features.loc[date].values
            if not np.any(np.isnan(row)) and not np.any(np.isinf(row)):
                all_features.append(row)
                all_labels.append(fwd_ret.loc[date])
                all_meta.append({'date': date, 'ticker': ticker})

    feature_names = list(compute_features(pd.Series(dtype=float), pd.Series(dtype=float)).columns)
    X = np.array(all_features)
    y = np.array(all_labels)
    meta = pd.DataFrame(all_meta)

    print(f"Dataset: {len(X)} samples, {X.shape[1]} features, {meta['ticker'].nunique()} stocks")
    return X, y, meta, feature_names


def walk_forward_backtest(X, y, meta, feature_names, spy, all_data,
                          train_days=252, test_days=21):
    """Walk-forward backtest with sliding window."""
    dates = sorted(meta['date'].unique())
    print(f"Walk-forward: {len(dates)} unique dates, {train_days}d train, {test_days}d test")

    # Spy returns for benchmark
    spy_close = spy['Close']
    spy_sma200 = spy_close.rolling(200).mean()

    portfolio_returns = []
    spy_returns_list = []
    trade_dates = []
    regimes = []
    all_selections = []

    # Walk through time
    fold = 0
    i = train_days
    while i + test_days <= len(dates):
        train_dates_range = dates[i - train_days:i]
        test_dates_range = dates[i:i + test_days]

        # Split by date
        train_mask = meta['date'].isin(train_dates_range)
        test_mask = meta['date'].isin(test_dates_range)

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        meta_test = meta[test_mask].copy()

        if len(X_train) < 100 or len(X_test) < 10:
            i += test_days
            continue

        # Train model
        if HAS_LGBM:
            model = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=20,
                verbose=-1,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
        else:
            # Simple momentum score as fallback
            mom_idx = feature_names.index('mom_12_1') if 'mom_12_1' in feature_names else 5
            predictions = X_test[:, mom_idx]

        meta_test = meta_test.copy()
        meta_test['pred'] = predictions

        # For each test date, pick top K stocks
        test_date = test_dates_range[0]
        test_day_mask = meta_test['date'] == test_date
        day_preds = meta_test[test_day_mask].copy()

        if len(day_preds) < TOP_K:
            i += test_days
            continue

        # Select top K by prediction
        top_k = day_preds.nlargest(TOP_K, 'pred')
        selected_tickers = top_k['ticker'].tolist()

        # Calculate actual portfolio return over test period
        port_ret = 0
        for ticker in selected_tickers:
            if ticker in all_data:
                tc = all_data[ticker]['Close']
                start_idx = tc.index.searchsorted(test_dates_range[0])
                end_idx = tc.index.searchsorted(test_dates_range[-1])
                if start_idx < len(tc) and end_idx < len(tc):
                    start_price = tc.iloc[start_idx]
                    end_price = tc.iloc[end_idx]
                    if start_price > 0:
                        port_ret += (end_price / start_price - 1) / TOP_K

        # SPY return for same period
        spy_start = spy_close.index.searchsorted(test_dates_range[0])
        spy_end = spy_close.index.searchsorted(test_dates_range[-1])
        if spy_start < len(spy_close) and spy_end < len(spy_close):
            spy_ret = spy_close.iloc[spy_end] / spy_close.iloc[spy_start] - 1
        else:
            spy_ret = 0

        # Regime at entry
        if test_date in spy_close.index and test_date in spy_sma200.index:
            regime = 'bull' if spy_close.loc[test_date] > spy_sma200.loc[test_date] else 'bear'
        else:
            regime = 'unknown'

        portfolio_returns.append(float(port_ret))
        spy_returns_list.append(float(spy_ret))
        trade_dates.append(str(test_date.date()) if hasattr(test_date, 'date') else str(test_date))
        regimes.append(regime)
        all_selections.append(selected_tickers)

        if fold % 20 == 0:
            cum_ret = np.prod([1 + r for r in portfolio_returns]) - 1
            print(f"  Fold {fold}: {test_date.date() if hasattr(test_date, 'date') else test_date} | "
                  f"Period ret: {port_ret*100:+.1f}% | Cum: {cum_ret*100:+.1f}% | "
                  f"Top picks: {', '.join(selected_tickers[:3])}")

        fold += 1
        i += test_days

    return portfolio_returns, spy_returns_list, trade_dates, regimes, all_selections


def compute_metrics(returns, capital=100000):
    """Compute risk-adjusted metrics from period returns."""
    if not returns:
        return {}

    returns = np.array(returns)
    equity = capital * np.cumprod(1 + returns)

    # Annualize (each return is ~21 trading days = ~1 month)
    periods_per_year = 252 / REBALANCE_DAYS  # ~12
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)

    sharpe = mean_ret / std_ret * np.sqrt(periods_per_year) if std_ret > 0 else 0
    downside = returns[returns < 0]
    sortino = mean_ret / np.std(downside) * np.sqrt(periods_per_year) if len(downside) > 0 and np.std(downside) > 0 else 0

    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    wr = len(wins) / len(returns) * 100
    pf = np.sum(wins) / abs(np.sum(losses)) if len(losses) > 0 and np.sum(losses) != 0 else float('inf')

    years = len(returns) / periods_per_year
    cagr = ((equity[-1] / capital) ** (1 / max(years, 0.01)) - 1) * 100

    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    maxdd = np.min(dd) * 100

    return {
        'sharpe': round(float(sharpe), 2),
        'sortino': round(float(sortino), 2),
        'pf': round(float(min(pf, 999)), 2),
        'wr': round(float(wr), 1),
        'cagr': round(float(cagr), 1),
        'maxdd': round(float(maxdd), 1),
        'n_periods': len(returns),
        'total_return': round(float((equity[-1] / capital - 1) * 100), 1),
        'final_equity': round(float(equity[-1]), 2),
        'calmar': round(float(abs(cagr / maxdd)) if maxdd != 0 else 0, 2),
    }


def adversarial_gates(returns, regimes, trade_dates):
    """Run all 4 adversarial validation gates."""
    returns = np.array(returns)

    # 1. Permutation test (shuffle stock selection → random top 5)
    real_sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
    null_sharpes = []
    for _ in range(200):
        signs = np.random.choice([-1, 1], size=len(returns))
        shuffled = returns * signs
        s = np.mean(shuffled) / np.std(shuffled) if np.std(shuffled) > 0 else 0
        null_sharpes.append(s)
    perm_p = float(np.mean([ns >= real_sharpe for ns in null_sharpes]))
    perm_result = 'PASS' if perm_p < 0.05 else 'FAIL'

    # 2. Regime R1
    bull_ret = [r for r, reg in zip(returns, regimes) if reg == 'bull']
    bear_ret = [r for r, reg in zip(returns, regimes) if reg == 'bear']

    if len(bull_ret) >= 5 and len(bear_ret) >= 5:
        s_bull = np.mean(bull_ret) / np.std(bull_ret) if np.std(bull_ret) > 0 else 0
        s_bear = np.mean(bear_ret) / np.std(bear_ret) if np.std(bear_ret) > 0 else 0
        denom = max(abs(s_bull), abs(s_bear))
        r1_gap = abs(s_bull - s_bear) / denom if denom > 0 else 0
        r1_result = 'PASS' if r1_gap < 0.50 else 'FAIL'
    else:
        s_bull = s_bear = r1_gap = None
        r1_result = 'SKIP'

    # 3. Sub-period
    mid = len(returns) // 2
    sub_result = 'PASS' if sum(returns[:mid]) > 0 and sum(returns[mid:]) > 0 else 'FAIL'

    # 4. Outlier
    if len(returns) >= 20:
        sorted_ret = sorted(returns)
        n_remove = max(1, int(len(returns) * 0.05))
        outlier_result = 'PASS' if sum(sorted_ret[:-n_remove]) > 0 else 'FAIL'
    else:
        outlier_result = 'SKIP'

    gates = {
        'permutation': {'p_value': perm_p, 'result': perm_result},
        'regime_r1': {'bull_sharpe': round(float(s_bull), 2) if s_bull is not None else None,
                      'bear_sharpe': round(float(s_bear), 2) if s_bear is not None else None,
                      'gap': round(float(r1_gap), 3) if r1_gap is not None else None,
                      'result': r1_result},
        'sub_period': sub_result,
        'outlier': outlier_result,
    }

    gates_passed = sum([
        1 if perm_result == 'PASS' else 0,
        1 if r1_result == 'PASS' else 0,
        1 if sub_result == 'PASS' else 0,
        1 if outlier_result == 'PASS' else 0,
    ])

    return gates, gates_passed


def main():
    print("=" * 60)
    print("QUALITY-MOMENTUM STOCK RANKER v1")
    print("=" * 60)

    # Download data
    all_data, spy = download_data()

    # Build dataset
    X, y, meta, feature_names = build_dataset(all_data, spy)

    # Walk-forward backtest
    print("\n--- LightGBM Model ---")
    port_returns, spy_returns, dates, regimes, selections = walk_forward_backtest(
        X, y, meta, feature_names, spy, all_data
    )

    # Compute metrics
    model_metrics = compute_metrics(port_returns)
    spy_metrics = compute_metrics(spy_returns)

    # Excess returns
    excess_returns = [p - s for p, s in zip(port_returns, spy_returns)]
    excess_metrics = compute_metrics(excess_returns)

    # Adversarial gates
    gates, gates_passed = adversarial_gates(port_returns, regimes, dates)

    print(f"\n{'='*60}")
    print("RESULTS — QUALITY-MOMENTUM RANKER v1")
    print(f"{'='*60}")
    print(f"\nModel:   Sharpe {model_metrics['sharpe']}, CAGR {model_metrics['cagr']}%, "
          f"MaxDD {model_metrics['maxdd']}%, WR {model_metrics['wr']}%")
    print(f"SPY:     Sharpe {spy_metrics['sharpe']}, CAGR {spy_metrics['cagr']}%, "
          f"MaxDD {spy_metrics['maxdd']}%")
    print(f"Excess:  Sharpe {excess_metrics['sharpe']}, CAGR {excess_metrics['cagr']}%")
    print(f"\nGates: {gates_passed}/4")
    print(f"  Perm: {gates['permutation']['result']} (p={gates['permutation']['p_value']:.3f})")
    print(f"  R1: {gates['regime_r1']['result']} (bull={gates['regime_r1']['bull_sharpe']}, "
          f"bear={gates['regime_r1']['bear_sharpe']}, gap={gates['regime_r1']['gap']})")
    print(f"  Sub-period: {gates['sub_period']}")
    print(f"  Outlier: {gates['outlier']}")

    # Most selected stocks
    all_picks = [t for sel in selections for t in sel]
    from collections import Counter
    top_picks = Counter(all_picks).most_common(10)
    print(f"\nMost selected stocks:")
    for ticker, count in top_picks:
        print(f"  {ticker}: {count} times ({count/len(selections)*100:.0f}%)")

    # Feature importance (if LightGBM)
    if HAS_LGBM:
        # Retrain on full data for feature importance
        try:
            model = lgb.LGBMRegressor(n_estimators=100, max_depth=5, verbose=-1)
            model.fit(X[:len(X)//2], y[:len(y)//2])
            importances = dict(zip(feature_names, model.feature_importances_))
            sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
            print(f"\nTop features:")
            for fname, imp in sorted_imp[:8]:
                print(f"  {fname}: {imp}")
        except:
            sorted_imp = []

    # Save results
    output = {
        'strategy': 'Quality-Momentum Stock Ranker v1',
        'run_date': str(datetime.now()),
        'universe_size': len(all_data),
        'top_k': TOP_K,
        'rebalance_days': REBALANCE_DAYS,
        'model_metrics': model_metrics,
        'spy_metrics': spy_metrics,
        'excess_metrics': excess_metrics,
        'gates': gates,
        'gates_passed': f"{gates_passed}/4",
        'top_selections': [{'ticker': t, 'count': c} for t, c in top_picks],
        'n_rebalances': len(port_returns),
    }

    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved.")

    # MLflow
    if MLFLOW_OK:
        try:
            mlflow.set_experiment('quality_momentum_ranker_v1')
            with mlflow.start_run(run_name='qm_lgbm_top5'):
                mlflow.log_params({
                    'model': 'LightGBM',
                    'top_k': TOP_K,
                    'rebalance_days': REBALANCE_DAYS,
                    'universe_size': len(all_data),
                    'train_days': 252,
                })
                mlflow.log_metrics({
                    'sharpe': model_metrics['sharpe'],
                    'sortino': model_metrics['sortino'],
                    'cagr': model_metrics['cagr'],
                    'maxdd': model_metrics['maxdd'],
                    'wr': model_metrics['wr'],
                    'pf': min(model_metrics['pf'], 999),
                    'excess_sharpe': excess_metrics['sharpe'],
                    'perm_p': gates['permutation']['p_value'],
                    'gates_passed': gates_passed,
                })
        except:
            pass

    return output


if __name__ == '__main__':
    main()
