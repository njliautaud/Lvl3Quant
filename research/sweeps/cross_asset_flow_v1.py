#!/usr/bin/env python3
"""
Cross-Asset Flow-Momentum Fusion v1
=====================================

Growth strategy combining cross-asset signals:
1. Bond-equity correlation regime (TLT vs SPY)
2. VIX term structure slope (VIX vs VIX3M proxy)
3. Sector rotation flow (relative momentum across sectors)
4. Credit stress (HYG/LQD spread proxy)

Uses LightGBM to learn optimal signal combination for stock selection.
Different from DL Stock Ranker: focuses on MACRO regime signals as
features rather than stock-level momentum/quality.

Universe: 50 large-cap stocks
Features: 12 macro + 8 stock-level = 20 total
Walk-forward: 252d train / 21d test, SLIDING (HC #0)
Monthly rebalance, top 5 stocks

Adversarial gates: permutation (200), regime R1, sub-period, outlier.
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
RESULTS_PATH = RESULTS_DIR / 'cross_asset_flow_v1_results.json'

STOCKS = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA', 'BRK-B',
    'UNH', 'JNJ', 'V', 'JPM', 'XOM', 'PG', 'MA', 'HD', 'CVX', 'MRK',
    'ABBV', 'PEP', 'COST', 'KO', 'AVGO', 'LLY', 'WMT', 'TMO', 'MCD',
    'CSCO', 'ACN', 'DHR', 'ABT', 'NEE', 'TXN', 'PM', 'UNP', 'RTX',
    'LOW', 'HON', 'AMGN', 'IBM', 'CAT', 'GS', 'BA', 'SBUX', 'GE',
    'MMM', 'DIS', 'INTC', 'NKE', 'CRM'
]

# Macro ETFs for cross-asset signals
MACRO_TICKERS = ['SPY', 'TLT', 'GLD', 'HYG', 'LQD', 'UUP', 'XLF', 'XLE', 'XLK', 'XLV', 'XLY', 'XLP']

TOP_K = 5
REBALANCE_DAYS = 21


def download_all():
    """Download stock + macro data."""
    import yfinance as yf

    print("Downloading macro ETFs...")
    macro_data = {}
    for ticker in MACRO_TICKERS:
        try:
            df = yf.download(ticker, start='2010-01-01', end='2026-07-24', progress=False)
            if df.index.tz is not None:
                df.index = df.index.tz_convert(None)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            macro_data[ticker] = df
        except:
            print(f"  {ticker}: FAILED")

    print(f"Downloaded {len(macro_data)} macro ETFs")

    print("Downloading stocks...")
    stock_data = {}
    for ticker in STOCKS:
        try:
            df = yf.download(ticker, start='2010-01-01', end='2026-07-24', progress=False)
            if df.index.tz is not None:
                df.index = df.index.tz_convert(None)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 252:
                stock_data[ticker] = df
        except:
            pass

    print(f"Downloaded {len(stock_data)} stocks")
    return stock_data, macro_data


def compute_macro_features(macro_data, date_idx):
    """Compute cross-asset macro features for a given date index."""
    features = {}

    spy = macro_data.get('SPY', pd.DataFrame())
    tlt = macro_data.get('TLT', pd.DataFrame())
    gld = macro_data.get('GLD', pd.DataFrame())
    hyg = macro_data.get('HYG', pd.DataFrame())
    lqd = macro_data.get('LQD', pd.DataFrame())
    uup = macro_data.get('UUP', pd.DataFrame())

    # Common index
    common = date_idx

    # 1. Bond-Equity correlation (rolling 60d)
    if len(spy) > 0 and len(tlt) > 0:
        spy_ret = spy['Close'].reindex(common).pct_change()
        tlt_ret = tlt['Close'].reindex(common).pct_change()
        be_corr_20 = spy_ret.rolling(20).corr(tlt_ret)
        be_corr_60 = spy_ret.rolling(60).corr(tlt_ret)
        features['bond_eq_corr_20d'] = be_corr_20
        features['bond_eq_corr_60d'] = be_corr_60
        # Z-score of correlation
        features['bond_eq_corr_zscore'] = (be_corr_60 - be_corr_60.rolling(252).mean()) / be_corr_60.rolling(252).std()

    # 2. Credit stress: HYG/LQD spread proxy
    if len(hyg) > 0 and len(lqd) > 0:
        hyg_ret = hyg['Close'].reindex(common).pct_change()
        lqd_ret = lqd['Close'].reindex(common).pct_change()
        credit_spread = hyg_ret.rolling(20).mean() - lqd_ret.rolling(20).mean()
        features['credit_spread_20d'] = credit_spread
        features['credit_spread_zscore'] = (credit_spread - credit_spread.rolling(252).mean()) / credit_spread.rolling(252).std()

    # 3. Gold momentum (flight to safety)
    if len(gld) > 0:
        gld_close = gld['Close'].reindex(common)
        features['gold_mom_20d'] = gld_close.pct_change(20)
        features['gold_mom_60d'] = gld_close.pct_change(60)

    # 4. Dollar strength
    if len(uup) > 0:
        uup_close = uup['Close'].reindex(common)
        features['dollar_mom_20d'] = uup_close.pct_change(20)

    # 5. Sector dispersion (how different sectors are performing)
    sector_etfs = ['XLF', 'XLE', 'XLK', 'XLV', 'XLY', 'XLP']
    sector_rets = []
    for sect in sector_etfs:
        if sect in macro_data:
            sr = macro_data[sect]['Close'].reindex(common).pct_change(20)
            sector_rets.append(sr)

    if len(sector_rets) >= 4:
        sector_df = pd.concat(sector_rets, axis=1)
        features['sector_dispersion'] = sector_df.std(axis=1)
        features['sector_dispersion_zscore'] = (
            features['sector_dispersion'] - features['sector_dispersion'].rolling(252).mean()
        ) / features['sector_dispersion'].rolling(252).std()

    # 6. SPY regime features
    if len(spy) > 0:
        spy_close = spy['Close'].reindex(common)
        spy_log_ret = np.log(spy_close / spy_close.shift(1))
        features['spy_vol_20d'] = spy_log_ret.rolling(20).std() * np.sqrt(252)
        features['spy_vol_ratio'] = (
            spy_log_ret.rolling(20).std() / spy_log_ret.rolling(60).std()
        )

    return pd.DataFrame(features, index=common)


def compute_stock_features(close, volume):
    """Compute stock-level features."""
    features = {}
    log_ret = np.log(close / close.shift(1))

    features['ret_21d'] = close.pct_change(21)
    features['ret_63d'] = close.pct_change(63)
    features['mom_12_1'] = close.pct_change(252) - close.pct_change(21)
    features['high_52w'] = close / close.rolling(252).max()
    features['vol_20d'] = log_ret.rolling(20).std() * np.sqrt(252)
    features['sharpe_63d'] = log_ret.rolling(63).mean() / log_ret.rolling(63).std()
    features['vol_rel'] = volume / volume.rolling(20).mean()
    features['skew_63d'] = log_ret.rolling(63).skew()

    return pd.DataFrame(features, index=close.index)


def build_dataset(stock_data, macro_data):
    """Build combined macro + stock feature dataset."""
    print("Building dataset...")

    # Get common dates
    spy_dates = macro_data['SPY'].index if 'SPY' in macro_data else pd.DatetimeIndex([])
    common_dates = spy_dates

    for ticker, df in stock_data.items():
        common_dates = common_dates.intersection(df.index)

    # Compute macro features once
    macro_feats = compute_macro_features(macro_data, common_dates)
    macro_cols = list(macro_feats.columns)

    all_features = []
    all_labels = []
    all_meta = []

    for ticker, df in stock_data.items():
        close = df['Close'].reindex(common_dates)
        volume = df['Volume'].reindex(common_dates)

        stock_feats = compute_stock_features(close, volume)

        # Forward return label
        fwd_ret = close.pct_change(REBALANCE_DAYS).shift(-REBALANCE_DAYS)

        # Combine macro + stock features
        combined = pd.concat([macro_feats, stock_feats], axis=1)

        valid = combined.dropna().index.intersection(fwd_ret.dropna().index)

        for date in valid:
            row = combined.loc[date].values
            if not np.any(np.isnan(row)) and not np.any(np.isinf(row)):
                all_features.append(row)
                all_labels.append(fwd_ret.loc[date])
                all_meta.append({'date': date, 'ticker': ticker})

    feature_names = macro_cols + list(stock_feats.columns)
    X = np.array(all_features)
    y = np.array(all_labels)
    meta = pd.DataFrame(all_meta)

    print(f"Dataset: {len(X)} samples, {X.shape[1]} features ({len(macro_cols)} macro + {X.shape[1]-len(macro_cols)} stock)")
    return X, y, meta, feature_names


def walk_forward(X, y, meta, feature_names, stock_data, macro_data,
                 train_days=252, test_days=21):
    """Walk-forward backtest."""
    dates = sorted(meta['date'].unique())
    print(f"Walk-forward: {len(dates)} dates, {train_days}d train, {test_days}d test")

    spy_close = macro_data['SPY']['Close'] if 'SPY' in macro_data else pd.Series(dtype=float)
    spy_sma200 = spy_close.rolling(200).mean()

    port_returns = []
    spy_returns = []
    trade_dates = []
    regimes = []
    selections = []

    fold = 0
    i = train_days
    while i + test_days <= len(dates):
        train_range = dates[i - train_days:i]
        test_range = dates[i:i + test_days]

        train_mask = meta['date'].isin(train_range)
        test_mask = meta['date'].isin(test_range)

        X_train, y_train = X[train_mask], y[train_mask]
        X_test = X[test_mask]
        meta_test = meta[test_mask].copy()

        if len(X_train) < 100 or len(X_test) < 10:
            i += test_days
            continue

        if HAS_LGBM:
            model = lgb.LGBMRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
                verbose=-1, n_jobs=-1,
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
        else:
            mom_idx = feature_names.index('mom_12_1') if 'mom_12_1' in feature_names else 0
            preds = X_test[:, mom_idx]

        meta_test = meta_test.copy()
        meta_test['pred'] = preds

        test_date = test_range[0]
        day_preds = meta_test[meta_test['date'] == test_date]

        if len(day_preds) < TOP_K:
            i += test_days
            continue

        top_k = day_preds.nlargest(TOP_K, 'pred')
        selected = top_k['ticker'].tolist()

        # Portfolio return
        port_ret = 0
        for ticker in selected:
            if ticker in stock_data:
                tc = stock_data[ticker]['Close']
                si = tc.index.searchsorted(test_range[0])
                ei = tc.index.searchsorted(test_range[-1])
                if si < len(tc) and ei < len(tc) and tc.iloc[si] > 0:
                    port_ret += (tc.iloc[ei] / tc.iloc[si] - 1) / TOP_K

        # SPY return
        si = spy_close.index.searchsorted(test_range[0])
        ei = spy_close.index.searchsorted(test_range[-1])
        spy_ret = (spy_close.iloc[ei] / spy_close.iloc[si] - 1) if si < len(spy_close) and ei < len(spy_close) else 0

        # Regime
        regime = 'bull'
        if test_date in spy_close.index and test_date in spy_sma200.index:
            regime = 'bull' if spy_close.loc[test_date] > spy_sma200.loc[test_date] else 'bear'

        port_returns.append(float(port_ret))
        spy_returns.append(float(spy_ret))
        trade_dates.append(str(test_date.date()) if hasattr(test_date, 'date') else str(test_date))
        regimes.append(regime)
        selections.append(selected)

        if fold % 20 == 0:
            cum = np.prod([1 + r for r in port_returns]) - 1
            print(f"  Fold {fold}: {test_date.date() if hasattr(test_date, 'date') else test_date} | "
                  f"Ret: {port_ret*100:+.1f}% | Cum: {cum*100:+.1f}% | Picks: {', '.join(selected[:3])}")

        fold += 1
        i += test_days

    return port_returns, spy_returns, trade_dates, regimes, selections


def compute_metrics(returns, capital=100000):
    if not returns:
        return {}
    returns = np.array(returns)
    equity = capital * np.cumprod(1 + returns)
    ppy = 252 / REBALANCE_DAYS
    mean_r = np.mean(returns)
    std_r = np.std(returns)
    sharpe = mean_r / std_r * np.sqrt(ppy) if std_r > 0 else 0
    ds = returns[returns < 0]
    sortino = mean_r / np.std(ds) * np.sqrt(ppy) if len(ds) > 0 and np.std(ds) > 0 else 0
    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    wr = len(wins) / len(returns) * 100
    pf = np.sum(wins) / abs(np.sum(losses)) if len(losses) > 0 and np.sum(losses) != 0 else 999
    years = len(returns) / ppy
    cagr = ((equity[-1] / capital) ** (1 / max(years, 0.01)) - 1) * 100
    peak = np.maximum.accumulate(equity)
    maxdd = float(np.min((equity - peak) / peak) * 100)
    return {
        'sharpe': round(float(sharpe), 2), 'sortino': round(float(sortino), 2),
        'pf': round(float(min(pf, 999)), 2), 'wr': round(float(wr), 1),
        'cagr': round(float(cagr), 1), 'maxdd': round(float(maxdd), 1),
        'n_periods': len(returns), 'final_equity': round(float(equity[-1]), 2),
    }


def adversarial_gates(returns, regimes):
    returns = np.array(returns)
    real_s = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
    null = [np.mean(returns * np.random.choice([-1,1], len(returns))) / np.std(returns) for _ in range(200)]
    perm_p = float(np.mean([n >= real_s for n in null]))

    bull_r = [r for r, rg in zip(returns, regimes) if rg == 'bull']
    bear_r = [r for r, rg in zip(returns, regimes) if rg == 'bear']
    if len(bull_r) >= 5 and len(bear_r) >= 5:
        sb = np.mean(bull_r) / np.std(bull_r) if np.std(bull_r) > 0 else 0
        sbe = np.mean(bear_r) / np.std(bear_r) if np.std(bear_r) > 0 else 0
        gap = abs(sb - sbe) / max(abs(sb), abs(sbe)) if max(abs(sb), abs(sbe)) > 0 else 0
        r1 = 'PASS' if gap < 0.50 else 'FAIL'
    else:
        sb = sbe = gap = None
        r1 = 'SKIP'

    mid = len(returns) // 2
    sub = 'PASS' if sum(returns[:mid]) > 0 and sum(returns[mid:]) > 0 else 'FAIL'

    if len(returns) >= 20:
        sr = sorted(returns)
        nr = max(1, int(len(returns) * 0.05))
        out = 'PASS' if sum(sr[:-nr]) > 0 else 'FAIL'
    else:
        out = 'SKIP'

    gates = {
        'permutation': {'p_value': perm_p, 'result': 'PASS' if perm_p < 0.05 else 'FAIL'},
        'regime_r1': {'bull_sharpe': round(float(sb), 2) if sb is not None else None,
                      'bear_sharpe': round(float(sbe), 2) if sbe is not None else None,
                      'gap': round(float(gap), 3) if gap is not None else None, 'result': r1},
        'sub_period': sub, 'outlier': out,
    }
    gp = sum([1 if gates['permutation']['result']=='PASS' else 0,
              1 if r1=='PASS' else 0, 1 if sub=='PASS' else 0, 1 if out=='PASS' else 0])
    return gates, gp


def main():
    print("=" * 60)
    print("CROSS-ASSET FLOW-MOMENTUM FUSION v1")
    print("=" * 60)

    stock_data, macro_data = download_all()
    X, y, meta, feature_names = build_dataset(stock_data, macro_data)

    print("\n--- LightGBM with Macro + Stock features ---")
    port_ret, spy_ret, dates, regimes, sels = walk_forward(X, y, meta, feature_names, stock_data, macro_data)

    model_m = compute_metrics(port_ret)
    spy_m = compute_metrics(spy_ret)
    excess_ret = [p - s for p, s in zip(port_ret, spy_ret)]
    excess_m = compute_metrics(excess_ret)
    gates, gp = adversarial_gates(port_ret, regimes)

    print(f"\n{'='*60}")
    print("RESULTS — CROSS-ASSET FLOW-MOMENTUM FUSION v1")
    print(f"{'='*60}")
    print(f"\nModel:   Sharpe {model_m['sharpe']}, CAGR {model_m['cagr']}%, MaxDD {model_m['maxdd']}%, WR {model_m['wr']}%")
    print(f"SPY:     Sharpe {spy_m['sharpe']}, CAGR {spy_m['cagr']}%")
    print(f"Excess:  Sharpe {excess_m['sharpe']}, CAGR {excess_m['cagr']}%")
    print(f"\nGates: {gp}/4")
    for k, v in gates.items():
        if isinstance(v, dict):
            print(f"  {k}: {v.get('result', v)}")
        else:
            print(f"  {k}: {v}")

    # Most selected
    all_picks = [t for s in sels for t in s]
    from collections import Counter
    top_picks = Counter(all_picks).most_common(10)
    print(f"\nMost selected: {', '.join(f'{t}({c})' for t, c in top_picks[:5])}")

    # Feature importance
    if HAS_LGBM:
        try:
            model = lgb.LGBMRegressor(n_estimators=100, max_depth=5, verbose=-1)
            model.fit(X[:len(X)//2], y[:len(y)//2])
            imp = sorted(zip(feature_names, model.feature_importances_), key=lambda x: x[1], reverse=True)
            print(f"\nTop features:")
            for fn, fi in imp[:8]:
                print(f"  {fn}: {fi}")
        except:
            imp = []

    output = {
        'strategy': 'Cross-Asset Flow-Momentum Fusion v1',
        'run_date': str(datetime.now()),
        'model_metrics': model_m, 'spy_metrics': spy_m, 'excess_metrics': excess_m,
        'gates': gates, 'gates_passed': f"{gp}/4",
        'top_selections': [{'ticker': t, 'count': c} for t, c in top_picks],
    }

    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    if MLFLOW_OK:
        try:
            mlflow.set_experiment('cross_asset_flow_v1')
            with mlflow.start_run(run_name='caf_lgbm'):
                mlflow.log_metrics({
                    'sharpe': model_m['sharpe'], 'cagr': model_m['cagr'],
                    'maxdd': model_m['maxdd'], 'wr': model_m['wr'],
                    'excess_sharpe': excess_m['sharpe'], 'perm_p': gates['permutation']['p_value'],
                    'gates_passed': gp,
                })
        except:
            pass

    print("\nResults saved.")
    return output


if __name__ == '__main__':
    main()
