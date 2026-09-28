#!/usr/bin/env python3
"""
Quality-Momentum Stock Ranker v2 — Macro-Enhanced
===================================================

Merges the proven stock-level features from QM v1 (Sharpe 3.20) with the
validated macro/cross-asset features from Cross-Asset Flow v1.

Key improvements over v1:
  - Adds 8 macro features (bond-eq corr, credit spread, gold/dollar momentum,
    sector dispersion, SPY vol) that proved significant in cross-asset study
  - Deducts 0.1% RT commission per stock trade at each rebalance
  - n_jobs=4 (not -1) to avoid runaway threads
  - Comprehensive feature importance analysis showing macro contribution

Universe: 50 large-cap stocks
Walk-forward: 252d train / 21d test, SLIDING (HC #0)
Monthly rebalance, top 5 stocks equally weighted
Adversarial gates: permutation (200), regime R1, sub-period, outlier
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("LightGBM not available, will use simple ranking", flush=True)

# MLflow
MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen('http://jupiter:5000/', timeout=2)
    import mlflow
    mlflow.set_tracking_uri('http://jupiter:5000')
    MLFLOW_OK = True
    print("MLflow connected", flush=True)
except Exception:
    print("MLflow unavailable — will skip logging", flush=True)

RESULTS_DIR = Path(__file__).resolve().parent / 'research' / 'findings'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = RESULTS_DIR / 'quality_momentum_v2_results.json'

# ── Universe ──────────────────────────────────────────────────────────────
UNIVERSE = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA', 'BRK-B',
    'UNH', 'JNJ', 'V', 'JPM', 'XOM', 'PG', 'MA', 'HD', 'CVX', 'MRK',
    'ABBV', 'PEP', 'COST', 'KO', 'AVGO', 'LLY', 'WMT', 'TMO', 'MCD',
    'CSCO', 'ACN', 'DHR', 'ABT', 'NEE', 'TXN', 'PM', 'UNP', 'RTX',
    'LOW', 'HON', 'AMGN', 'IBM', 'CAT', 'GS', 'BA', 'SBUX', 'GE',
    'MMM', 'DIS', 'INTC', 'NKE', 'CRM'
]

# Macro ETFs needed for cross-asset features
MACRO_TICKERS = ['SPY', 'TLT', 'GLD', 'HYG', 'LQD', 'UUP',
                 'XLF', 'XLE', 'XLK', 'XLV', 'XLY', 'XLP']

TOP_K = 5
REBALANCE_DAYS = 21
COMMISSION_RT_PCT = 0.001  # 0.1% round-trip per stock trade
DATA_START = '2010-01-01'
DATA_END = '2026-07-24'


# ── Data Download ─────────────────────────────────────────────────────────

def download_data():
    """Download stock + macro ETF data via yfinance."""
    import yfinance as yf

    print(f"Downloading {len(UNIVERSE)} stocks + {len(MACRO_TICKERS)} macro ETFs...", flush=True)

    # Download stocks
    stock_data = {}
    for ticker in UNIVERSE:
        try:
            df = yf.download(ticker, start=DATA_START, end=DATA_END, progress=False)
            if df.index.tz is not None:
                df.index = df.index.tz_convert(None)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 252:
                stock_data[ticker] = df
        except Exception as e:
            print(f"  {ticker}: FAILED ({e})", flush=True)

    # Download macro ETFs
    macro_data = {}
    for ticker in MACRO_TICKERS:
        try:
            df = yf.download(ticker, start=DATA_START, end=DATA_END, progress=False)
            if df.index.tz is not None:
                df.index = df.index.tz_convert(None)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            macro_data[ticker] = df
        except Exception as e:
            print(f"  {ticker}: FAILED ({e})", flush=True)

    print(f"Downloaded {len(stock_data)} stocks, {len(macro_data)} macro ETFs", flush=True)
    return stock_data, macro_data


# ── Feature Engineering ───────────────────────────────────────────────────

def compute_stock_features(close, volume):
    """Compute stock-level momentum + quality features (from v1)."""
    c = close
    v = volume
    feats = pd.DataFrame(index=c.index)
    log_ret = np.log(c / c.shift(1))

    # Momentum features
    feats['ret_5d'] = c.pct_change(5)
    feats['ret_21d'] = c.pct_change(21)
    feats['ret_63d'] = c.pct_change(63)
    feats['ret_126d'] = c.pct_change(126)
    feats['ret_252d'] = c.pct_change(252)
    feats['mom_12_1'] = c.pct_change(252) - c.pct_change(21)
    feats['high_52w_pct'] = c / c.rolling(252).max()
    feats['mom_accel'] = feats['ret_126d'] - feats['ret_252d'].shift(126)

    # Volatility features
    feats['vol_20d'] = log_ret.rolling(20).std() * np.sqrt(252)
    feats['vol_60d'] = log_ret.rolling(60).std() * np.sqrt(252)
    feats['vol_ratio'] = feats['vol_20d'] / feats['vol_60d']

    # Quality proxies
    feats['sharpe_63d'] = log_ret.rolling(63).mean() / log_ret.rolling(63).std()
    feats['sharpe_126d'] = log_ret.rolling(126).mean() / log_ret.rolling(126).std()
    roll_max = c.rolling(63).max()
    feats['maxdd_63d'] = (c - roll_max) / roll_max

    # Volume features
    feats['vol_rel'] = v / v.rolling(20).mean()
    feats['vol_trend'] = v.rolling(5).mean() / v.rolling(20).mean()
    up_vol = (v * (log_ret > 0).astype(float)).rolling(20).sum()
    dn_vol = (v * (log_ret <= 0).astype(float)).rolling(20).sum()
    feats['updn_vol_ratio'] = up_vol / (dn_vol + 1)

    # Higher moments
    feats['skew_63d'] = log_ret.rolling(63).skew()
    feats['kurt_63d'] = log_ret.rolling(63).kurt()

    return feats


def compute_macro_features(macro_data, date_idx):
    """Compute cross-asset macro features (validated in cross-asset flow v1)."""
    feats = {}

    spy = macro_data.get('SPY', pd.DataFrame())
    tlt = macro_data.get('TLT', pd.DataFrame())
    gld = macro_data.get('GLD', pd.DataFrame())
    hyg = macro_data.get('HYG', pd.DataFrame())
    lqd = macro_data.get('LQD', pd.DataFrame())
    uup = macro_data.get('UUP', pd.DataFrame())

    # 1. Bond-Equity correlation (key regime feature — #4 importance in cross-asset study)
    if len(spy) > 0 and len(tlt) > 0:
        spy_ret = spy['Close'].reindex(date_idx).pct_change()
        tlt_ret = tlt['Close'].reindex(date_idx).pct_change()
        be_corr_20 = spy_ret.rolling(20).corr(tlt_ret)
        feats['bond_eq_corr_20d'] = be_corr_20
        # Z-score vs 252d history
        be_corr_60 = spy_ret.rolling(60).corr(tlt_ret)
        feats['bond_eq_corr_zscore'] = (
            (be_corr_60 - be_corr_60.rolling(252).mean()) / be_corr_60.rolling(252).std()
        )

    # 2. SPY volatility regime (#6 importance)
    if len(spy) > 0:
        spy_close = spy['Close'].reindex(date_idx)
        spy_log = np.log(spy_close / spy_close.shift(1))
        feats['spy_vol_20d'] = spy_log.rolling(20).std() * np.sqrt(252)
        spy_vol_60 = spy_log.rolling(60).std() * np.sqrt(252)
        feats['spy_vol_ratio'] = feats['spy_vol_20d'] / spy_vol_60

    # 3. Credit spread proxy (HYG - TLT spread change)
    if len(hyg) > 0 and len(tlt) > 0:
        hyg_ret = hyg['Close'].reindex(date_idx).pct_change()
        tlt_ret2 = tlt['Close'].reindex(date_idx).pct_change()
        credit_spread = hyg_ret.rolling(20).mean() - tlt_ret2.rolling(20).mean()
        feats['credit_spread_proxy'] = credit_spread

    # 4. Sector dispersion (cross-sector return std)
    sector_etfs = ['XLF', 'XLE', 'XLK', 'XLV', 'XLY', 'XLP']
    sector_rets = []
    for sect in sector_etfs:
        if sect in macro_data:
            sr = macro_data[sect]['Close'].reindex(date_idx).pct_change(20)
            sector_rets.append(sr)
    if len(sector_rets) >= 4:
        sector_df = pd.concat(sector_rets, axis=1)
        feats['sector_dispersion'] = sector_df.std(axis=1)

    # 5. Gold momentum (flight-to-safety signal)
    if len(gld) > 0:
        feats['gold_mom_20d'] = gld['Close'].reindex(date_idx).pct_change(20)

    # 6. Dollar momentum
    if len(uup) > 0:
        feats['dollar_mom_20d'] = uup['Close'].reindex(date_idx).pct_change(20)

    return pd.DataFrame(feats, index=date_idx)


# ── Dataset Construction ──────────────────────────────────────────────────

def build_dataset(stock_data, macro_data):
    """Build combined stock + macro feature dataset."""
    print("Building feature dataset...", flush=True)

    # Common dates across all stocks and SPY
    spy_dates = macro_data['SPY'].index if 'SPY' in macro_data else pd.DatetimeIndex([])
    common_dates = spy_dates
    for ticker, df in stock_data.items():
        common_dates = common_dates.intersection(df.index)
    common_dates = common_dates.sort_values()

    # Compute macro features once (same for all stocks on a given day)
    macro_feats = compute_macro_features(macro_data, common_dates)
    macro_cols = list(macro_feats.columns)

    # Get stock feature names from a dummy call
    dummy_close = pd.Series(np.ones(500), index=pd.date_range('2020-01-01', periods=500))
    dummy_vol = pd.Series(np.ones(500), index=pd.date_range('2020-01-01', periods=500))
    stock_feat_names = list(compute_stock_features(dummy_close, dummy_vol).columns)

    all_features = []
    all_labels = []
    all_meta = []

    for ticker, df in stock_data.items():
        close = df['Close'].reindex(common_dates)
        volume = df['Volume'].reindex(common_dates)

        stock_feats = compute_stock_features(close, volume)

        # Forward 21d return as label
        fwd_ret = close.pct_change(REBALANCE_DAYS).shift(-REBALANCE_DAYS)

        # Combine: macro features (same for all stocks) + stock features (unique per stock)
        combined = pd.concat([macro_feats, stock_feats], axis=1)

        valid = combined.dropna().index.intersection(fwd_ret.dropna().index)

        for date in valid:
            row = combined.loc[date].values
            if not np.any(np.isnan(row)) and not np.any(np.isinf(row)):
                all_features.append(row)
                all_labels.append(fwd_ret.loc[date])
                all_meta.append({'date': date, 'ticker': ticker})

    feature_names = macro_cols + stock_feat_names
    X = np.array(all_features)
    y = np.array(all_labels)
    meta = pd.DataFrame(all_meta)

    print(f"Dataset: {len(X)} samples, {X.shape[1]} features "
          f"({len(macro_cols)} macro + {len(stock_feat_names)} stock), "
          f"{meta['ticker'].nunique()} stocks", flush=True)
    return X, y, meta, feature_names, macro_cols


# ── Walk-Forward Backtest ─────────────────────────────────────────────────

def walk_forward_backtest(X, y, meta, feature_names, stock_data, macro_data,
                          train_days=252, test_days=21):
    """Walk-forward backtest with sliding window and transaction costs."""
    dates = sorted(meta['date'].unique())
    print(f"Walk-forward: {len(dates)} unique dates, {train_days}d train, "
          f"{test_days}d test, {COMMISSION_RT_PCT*100:.1f}% RT commission", flush=True)

    spy_close = macro_data['SPY']['Close'] if 'SPY' in macro_data else pd.Series(dtype=float)
    spy_sma200 = spy_close.rolling(200).mean()

    port_returns = []
    spy_returns_list = []
    trade_dates = []
    regimes = []
    all_selections = []
    all_importances = []  # Track per-fold feature importance
    prev_selection = []

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

        # Train LightGBM
        if HAS_LGBM:
            model = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=20,
                verbose=-1,
                n_jobs=4,  # Not -1 per requirements
            )
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            # Collect feature importance every 10 folds
            if fold % 10 == 0:
                all_importances.append(model.feature_importances_.copy())
        else:
            mom_idx = feature_names.index('mom_12_1') if 'mom_12_1' in feature_names else 5
            predictions = X_test[:, mom_idx]

        meta_test = meta_test.copy()
        meta_test['pred'] = predictions

        # Select top K stocks at first test date
        test_date = test_range[0]
        day_preds = meta_test[meta_test['date'] == test_date].copy()

        if len(day_preds) < TOP_K:
            i += test_days
            continue

        top_k = day_preds.nlargest(TOP_K, 'pred')
        selected_tickers = top_k['ticker'].tolist()

        # Calculate turnover for transaction costs
        if prev_selection:
            n_new = len(set(selected_tickers) - set(prev_selection))
            turnover_fraction = n_new / TOP_K
        else:
            turnover_fraction = 1.0  # First period: all new
        prev_selection = selected_tickers

        # Actual portfolio return over test period
        port_ret = 0
        for ticker in selected_tickers:
            if ticker in stock_data:
                tc = stock_data[ticker]['Close']
                si = tc.index.searchsorted(test_range[0])
                ei = tc.index.searchsorted(test_range[-1])
                if si < len(tc) and ei < len(tc) and tc.iloc[si] > 0:
                    port_ret += (tc.iloc[ei] / tc.iloc[si] - 1) / TOP_K

        # Deduct commission: only on turned-over positions
        commission_cost = turnover_fraction * COMMISSION_RT_PCT
        port_ret -= commission_cost

        # SPY return for same period
        si = spy_close.index.searchsorted(test_range[0])
        ei = spy_close.index.searchsorted(test_range[-1])
        if si < len(spy_close) and ei < len(spy_close):
            spy_ret = spy_close.iloc[ei] / spy_close.iloc[si] - 1
        else:
            spy_ret = 0

        # Regime at entry
        regime = 'unknown'
        if test_date in spy_close.index and test_date in spy_sma200.index:
            regime = 'bull' if spy_close.loc[test_date] > spy_sma200.loc[test_date] else 'bear'

        port_returns.append(float(port_ret))
        spy_returns_list.append(float(spy_ret))
        trade_dates.append(str(test_date.date()) if hasattr(test_date, 'date') else str(test_date))
        regimes.append(regime)
        all_selections.append(selected_tickers)

        if fold % 20 == 0:
            cum_ret = np.prod([1 + r for r in port_returns]) - 1
            print(f"  Fold {fold}: {test_date.date() if hasattr(test_date, 'date') else test_date} | "
                  f"Ret: {port_ret*100:+.1f}% | Cum: {cum_ret*100:+.1f}% | "
                  f"Picks: {', '.join(selected_tickers[:3])} | "
                  f"Turnover: {turnover_fraction*100:.0f}%", flush=True)

        fold += 1
        i += test_days

    return port_returns, spy_returns_list, trade_dates, regimes, all_selections, all_importances


# ── Metrics ───────────────────────────────────────────────────────────────

def compute_metrics(returns, capital=100000):
    """Compute risk-adjusted performance metrics."""
    if not returns:
        return {}

    returns = np.array(returns)
    equity = capital * np.cumprod(1 + returns)
    ppy = 252 / REBALANCE_DAYS  # ~12 periods per year

    mean_r = np.mean(returns)
    std_r = np.std(returns)
    sharpe = mean_r / std_r * np.sqrt(ppy) if std_r > 0 else 0

    downside = returns[returns < 0]
    sortino = mean_r / np.std(downside) * np.sqrt(ppy) if len(downside) > 0 and np.std(downside) > 0 else 0

    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    wr = len(wins) / len(returns) * 100
    pf = np.sum(wins) / abs(np.sum(losses)) if len(losses) > 0 and np.sum(losses) != 0 else 999

    years = len(returns) / ppy
    cagr = ((equity[-1] / capital) ** (1 / max(years, 0.01)) - 1) * 100

    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    maxdd = float(np.min(dd) * 100)

    calmar = abs(cagr / maxdd) if maxdd != 0 else 0

    return {
        'sharpe': round(float(sharpe), 2),
        'sortino': round(float(sortino), 2),
        'pf': round(float(min(pf, 999)), 2),
        'wr': round(float(wr), 1),
        'cagr': round(float(cagr), 1),
        'maxdd': round(float(maxdd), 1),
        'calmar': round(float(calmar), 2),
        'n_periods': len(returns),
        'total_return': round(float((equity[-1] / capital - 1) * 100), 1),
        'final_equity': round(float(equity[-1]), 2),
    }


# ── Adversarial Validation Gates ──────────────────────────────────────────

def adversarial_gates(returns, regimes):
    """Run 4 adversarial validation gates."""
    returns = np.array(returns)

    # 1. Permutation test (200 shuffles)
    real_sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
    null_sharpes = []
    for _ in range(200):
        signs = np.random.choice([-1, 1], size=len(returns))
        shuffled = returns * signs
        s = np.mean(shuffled) / np.std(shuffled) if np.std(shuffled) > 0 else 0
        null_sharpes.append(s)
    perm_p = float(np.mean([ns >= real_sharpe for ns in null_sharpes]))
    perm_result = 'PASS' if perm_p < 0.05 else 'FAIL'

    # 2. Regime R1 (HC #428): |Sharpe_bull - Sharpe_bear| / max < 0.50
    bull_ret = [r for r, rg in zip(returns, regimes) if rg == 'bull']
    bear_ret = [r for r, rg in zip(returns, regimes) if rg == 'bear']

    if len(bull_ret) >= 5 and len(bear_ret) >= 5:
        s_bull = np.mean(bull_ret) / np.std(bull_ret) if np.std(bull_ret) > 0 else 0
        s_bear = np.mean(bear_ret) / np.std(bear_ret) if np.std(bear_ret) > 0 else 0
        denom = max(abs(s_bull), abs(s_bear))
        r1_gap = abs(s_bull - s_bear) / denom if denom > 0 else 0
        r1_result = 'PASS' if r1_gap < 0.50 else 'FAIL'
    else:
        s_bull = s_bear = r1_gap = None
        r1_result = 'SKIP'

    # 3. Sub-period: both halves profitable
    mid = len(returns) // 2
    sub_result = 'PASS' if sum(returns[:mid]) > 0 and sum(returns[mid:]) > 0 else 'FAIL'

    # 4. Outlier: profitable after removing top 5% returns
    if len(returns) >= 20:
        sorted_ret = sorted(returns)
        n_remove = max(1, int(len(returns) * 0.05))
        outlier_result = 'PASS' if sum(sorted_ret[:-n_remove]) > 0 else 'FAIL'
    else:
        outlier_result = 'SKIP'

    gates = {
        'permutation': {'p_value': round(perm_p, 4), 'result': perm_result},
        'regime_r1': {
            'bull_sharpe': round(float(s_bull), 2) if s_bull is not None else None,
            'bear_sharpe': round(float(s_bear), 2) if s_bear is not None else None,
            'gap': round(float(r1_gap), 3) if r1_gap is not None else None,
            'result': r1_result,
        },
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


# ── Feature Importance Analysis ───────────────────────────────────────────

def analyze_feature_importance(all_importances, feature_names, macro_cols):
    """Analyze which features (especially macro) actually contributed."""
    if not all_importances:
        return {}, {}

    avg_imp = np.mean(all_importances, axis=0)
    imp_dict = dict(zip(feature_names, avg_imp))
    sorted_imp = sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)

    # Macro vs stock contribution
    total_imp = sum(avg_imp)
    macro_imp = sum(avg_imp[i] for i, fn in enumerate(feature_names) if fn in macro_cols)
    stock_imp = total_imp - macro_imp
    macro_pct = macro_imp / total_imp * 100 if total_imp > 0 else 0

    contribution = {
        'macro_importance_pct': round(macro_pct, 1),
        'stock_importance_pct': round(100 - macro_pct, 1),
        'macro_features_ranked': [
            {'feature': fn, 'importance': round(float(imp), 1)}
            for fn, imp in sorted_imp if fn in macro_cols
        ],
    }

    return sorted_imp, contribution


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 60, flush=True)
    print("QUALITY-MOMENTUM STOCK RANKER v2 — MACRO-ENHANCED", flush=True)
    print("=" * 60, flush=True)

    # Download data
    stock_data, macro_data = download_data()

    # Build dataset
    X, y, meta, feature_names, macro_cols = build_dataset(stock_data, macro_data)

    # Walk-forward backtest
    print("\n--- LightGBM with Stock + Macro features ---", flush=True)
    port_returns, spy_returns, dates, regimes, selections, importances = \
        walk_forward_backtest(X, y, meta, feature_names, stock_data, macro_data)

    # Compute metrics
    model_metrics = compute_metrics(port_returns)
    spy_metrics = compute_metrics(spy_returns)
    excess_returns = [p - s for p, s in zip(port_returns, spy_returns)]
    excess_metrics = compute_metrics(excess_returns)

    # Adversarial gates
    gates, gates_passed = adversarial_gates(port_returns, regimes)

    # Feature importance analysis
    sorted_imp, macro_contribution = analyze_feature_importance(importances, feature_names, macro_cols)

    # ── Print Results ─────────────────────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print("RESULTS — QUALITY-MOMENTUM RANKER v2 (Macro-Enhanced)", flush=True)
    print(f"{'='*60}", flush=True)

    print(f"\nModel:   Sharpe {model_metrics.get('sharpe','?')}, "
          f"Sortino {model_metrics.get('sortino','?')}, "
          f"CAGR {model_metrics.get('cagr','?')}%, "
          f"MaxDD {model_metrics.get('maxdd','?')}%, "
          f"WR {model_metrics.get('wr','?')}%, "
          f"PF {model_metrics.get('pf','?')}, "
          f"Calmar {model_metrics.get('calmar','?')}", flush=True)
    print(f"SPY:     Sharpe {spy_metrics.get('sharpe','?')}, "
          f"CAGR {spy_metrics.get('cagr','?')}%, "
          f"MaxDD {spy_metrics.get('maxdd','?')}%", flush=True)
    print(f"Excess:  Sharpe {excess_metrics.get('sharpe','?')}, "
          f"CAGR {excess_metrics.get('cagr','?')}%", flush=True)

    # Gates
    print(f"\nAdversarial Gates: {gates_passed}/4", flush=True)
    print(f"  Permutation: {gates['permutation']['result']} "
          f"(p={gates['permutation']['p_value']})", flush=True)
    print(f"  Regime R1:   {gates['regime_r1']['result']} "
          f"(bull={gates['regime_r1']['bull_sharpe']}, "
          f"bear={gates['regime_r1']['bear_sharpe']}, "
          f"gap={gates['regime_r1']['gap']})", flush=True)
    print(f"  Sub-period:  {gates['sub_period']}", flush=True)
    print(f"  Outlier:     {gates['outlier']}", flush=True)

    # Feature importance
    if sorted_imp:
        print(f"\nFeature Importance (avg across folds):", flush=True)
        print(f"  Macro features contribute {macro_contribution['macro_importance_pct']}% "
              f"of total importance", flush=True)
        print(f"\n  Top 10 features:", flush=True)
        for fn, imp in sorted_imp[:10]:
            tag = " [MACRO]" if fn in macro_cols else ""
            print(f"    {fn}: {imp:.1f}{tag}", flush=True)
        print(f"\n  Macro features ranked:", flush=True)
        for item in macro_contribution['macro_features_ranked']:
            print(f"    {item['feature']}: {item['importance']}", flush=True)

    # Most selected stocks
    all_picks = [t for sel in selections for t in sel]
    top_picks = Counter(all_picks).most_common(10)
    print(f"\nMost selected stocks:", flush=True)
    for ticker, count in top_picks:
        pct = count / len(selections) * 100
        print(f"  {ticker}: {count} times ({pct:.0f}%)", flush=True)

    # Concentration check (survivorship bias indicator)
    top1_pct = top_picks[0][1] / len(selections) * 100 if top_picks else 0
    top3_pct = sum(c for _, c in top_picks[:3]) / (len(selections) * TOP_K) * 100 if top_picks else 0
    print(f"\nConcentration: top stock {top1_pct:.0f}%, top 3 stocks {top3_pct:.0f}% of all picks", flush=True)

    # ── v1 comparison (load v1 results if available) ──────────────────────
    v1_path = RESULTS_DIR / 'quality_momentum_ranker_v1_results.json'
    v1_comparison = {}
    if v1_path.exists():
        try:
            with open(v1_path) as f:
                v1 = json.load(f)
            v1m = v1.get('model_metrics', {})
            print(f"\n--- v2 vs v1 Comparison ---", flush=True)
            for metric in ['sharpe', 'sortino', 'cagr', 'maxdd', 'wr', 'pf']:
                v1_val = v1m.get(metric, '?')
                v2_val = model_metrics.get(metric, '?')
                print(f"  {metric:>8s}: v1={v1_val}  v2={v2_val}", flush=True)
            v1_comparison = {'v1_metrics': v1m}
        except Exception:
            pass

    # ── Save Results ──────────────────────────────────────────────────────
    output = {
        'strategy': 'Quality-Momentum Stock Ranker v2 (Macro-Enhanced)',
        'run_date': str(datetime.now()),
        'universe_size': len(stock_data),
        'top_k': TOP_K,
        'rebalance_days': REBALANCE_DAYS,
        'commission_rt_pct': COMMISSION_RT_PCT,
        'features': {
            'stock_features': [fn for fn in feature_names if fn not in macro_cols],
            'macro_features': macro_cols,
            'total': len(feature_names),
        },
        'model_metrics': model_metrics,
        'spy_metrics': spy_metrics,
        'excess_metrics': excess_metrics,
        'gates': gates,
        'gates_passed': f"{gates_passed}/4",
        'macro_contribution': macro_contribution,
        'top_selections': [{'ticker': t, 'count': c} for t, c in top_picks],
        'concentration': {
            'top1_pct': round(top1_pct, 1),
            'top3_pct': round(top3_pct, 1),
        },
        'n_rebalances': len(port_returns),
        'v1_comparison': v1_comparison,
    }

    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}", flush=True)

    # ── MLflow Logging ────────────────────────────────────────────────────
    if MLFLOW_OK:
        try:
            mlflow.set_experiment('quality_momentum_ranker_v2')
            with mlflow.start_run(run_name='qm_v2_macro_enhanced'):
                mlflow.log_params({
                    'model': 'LightGBM',
                    'top_k': TOP_K,
                    'rebalance_days': REBALANCE_DAYS,
                    'universe_size': len(stock_data),
                    'train_days': 252,
                    'n_stock_features': len(feature_names) - len(macro_cols),
                    'n_macro_features': len(macro_cols),
                    'commission_rt_pct': COMMISSION_RT_PCT,
                })
                mlflow.log_metrics({
                    'sharpe': model_metrics.get('sharpe', 0),
                    'sortino': model_metrics.get('sortino', 0),
                    'cagr': model_metrics.get('cagr', 0),
                    'maxdd': model_metrics.get('maxdd', 0),
                    'wr': model_metrics.get('wr', 0),
                    'pf': min(model_metrics.get('pf', 0), 999),
                    'calmar': model_metrics.get('calmar', 0),
                    'excess_sharpe': excess_metrics.get('sharpe', 0),
                    'perm_p': gates['permutation']['p_value'],
                    'gates_passed': gates_passed,
                    'macro_importance_pct': macro_contribution.get('macro_importance_pct', 0),
                })
                mlflow.log_artifact(str(RESULTS_PATH))
            print("MLflow run logged.", flush=True)
        except Exception as e:
            print(f"MLflow logging failed: {e}", flush=True)

    print("\nDone.", flush=True)
    return output


if __name__ == '__main__':
    main()
