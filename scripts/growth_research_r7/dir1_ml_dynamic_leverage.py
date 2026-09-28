#!/usr/bin/env python3
"""
R7 DIRECTION 1: ML-Driven Dynamic Leverage on Risk Parity

Instead of fixed 3x leverage, use LightGBM to predict optimal leverage dynamically.
Features: cross-asset correlation, VIX, yield curve slope, credit spread, realized vol, momentum breadth.
Target: forward 21-day risk parity portfolio return.
Walk-forward: 252d train, 21d test, sliding.

HC #694: Commission-free (RH/IBKR)
HC #428: Regime-agnostic validation (40+ OOT days, regime gap test)
HC #0: Sliding window walk-forward only
HC #697: No crypto, walk-forward predictive edge required
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb
import json
import os
from datetime import datetime
from scipy.stats import spearmanr

OUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_research_r7'
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_DAYS = 252
TEST_DAYS = 21
START = '2005-01-01'
END = '2026-07-14'

# Risk parity assets
RP_TICKERS = ['SPY', 'TLT', 'GLD', 'VNQ']
# Feature tickers
FEAT_TICKERS = ['^VIX', 'HYG', 'IEF', 'SHY']
ALL_TICKERS = list(set(RP_TICKERS + FEAT_TICKERS + ['SPY']))


def calc_metrics(returns, name=''):
    rets = returns.dropna()
    if len(rets) < 20:
        return {'name': name, 'sharpe': 0, 'sortino': 0, 'cagr': 0,
                'max_dd': 0, 'win_rate': 0, 'pf': 0, 'n_days': len(rets)}
    total_ret = (1 + rets).prod() - 1
    n_years = len(rets) / 252
    cagr = (1 + total_ret) ** (1 / max(n_years, 0.01)) - 1
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    downside = rets[rets < 0].std() * np.sqrt(252)
    sortino = rets.mean() * 252 / downside if downside > 0 else 0
    cum = (1 + rets).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()
    wr = (rets > 0).mean()
    gp = rets[rets > 0].sum()
    gl = abs(rets[rets < 0].sum())
    pf = gp / gl if gl > 0 else float('inf')
    calmar = abs(cagr / max_dd) if max_dd != 0 else 0
    return {'name': name, 'sharpe': round(float(sharpe), 4),
            'sortino': round(float(sortino), 4),
            'cagr': round(float(cagr), 4), 'cagr_pct': round(float(cagr * 100), 2),
            'max_dd': round(float(max_dd), 4), 'max_dd_pct': round(float(max_dd * 100), 2),
            'win_rate': round(float(wr), 4), 'pf': round(float(pf), 3),
            'calmar': round(float(calmar), 3), 'n_days': int(len(rets))}


def classify_regime(spy_ret, threshold=0.003):
    regimes = pd.Series('flat', index=spy_ret.index)
    regimes[spy_ret > threshold] = 'green'
    regimes[spy_ret < -threshold] = 'red'
    return regimes


def regime_test(strat_rets, spy_rets, name=''):
    regimes = classify_regime(spy_rets.reindex(strat_rets.index))
    results = {}
    for r in ['green', 'red', 'flat']:
        mask = regimes == r
        if mask.sum() > 10:
            m = calc_metrics(strat_rets[mask], f'{name} ({r})')
            results[r] = m
    # Regime gap
    if 'green' in results and 'red' in results:
        sg = results['green']['sharpe']
        sr = results['red']['sharpe']
        denom = max(abs(sg), abs(sr), 0.001)
        gap = abs(sg - sr) / denom
        results['regime_gap'] = round(gap, 3)
    return results


def download_data():
    """Download all needed price data."""
    tickers = RP_TICKERS + ['^VIX', 'HYG', 'IEF', 'SHY']
    tickers = list(set(tickers))
    print(f"Downloading: {tickers}")
    data = yf.download(tickers, start=START, end=END, auto_adjust=True, progress=False)
    close = data['Close'] if isinstance(data.columns, pd.MultiIndex) else data[['Close']]
    if isinstance(close.columns, pd.MultiIndex):
        close = close.droplevel(0, axis=1) if close.columns.nlevels > 1 else close
    close = close.dropna(how='all')
    close = close.ffill()
    print(f"Data: {close.index[0].date()} to {close.index[-1].date()}, {len(close)} days")
    return close


def build_risk_parity_weights(returns_df, lookback=60):
    """Inverse-volatility risk parity weights."""
    vols = returns_df.rolling(lookback).std()
    inv_vol = 1.0 / vols.clip(lower=1e-6)
    weights = inv_vol.div(inv_vol.sum(axis=1), axis=0)
    return weights


def build_features(close, rp_rets, rp_weights):
    """Build ML features for leverage prediction."""
    features = pd.DataFrame(index=close.index)

    # 1. Cross-asset rolling correlation (avg pairwise corr of RP assets)
    for lookback in [20, 60]:
        corrs = []
        rp_cols = [c for c in RP_TICKERS if c in rp_rets.columns]
        for i in range(len(rp_cols)):
            for j in range(i+1, len(rp_cols)):
                c = rp_rets[rp_cols[i]].rolling(lookback).corr(rp_rets[rp_cols[j]])
                corrs.append(c)
        if corrs:
            features[f'avg_corr_{lookback}d'] = pd.concat(corrs, axis=1).mean(axis=1)

    # 2. VIX level and changes
    if '^VIX' in close.columns:
        features['vix'] = close['^VIX']
        features['vix_20d_change'] = close['^VIX'].pct_change(20)
        features['vix_zscore'] = (close['^VIX'] - close['^VIX'].rolling(60).mean()) / close['^VIX'].rolling(60).std().clip(lower=0.01)

    # 3. Yield curve slope proxy (TLT/SHY ratio change)
    if 'TLT' in close.columns and 'SHY' in close.columns:
        yc = np.log(close['TLT']) - np.log(close['SHY'])
        features['yield_curve_slope'] = yc
        features['yield_curve_change_20d'] = yc.diff(20)

    # 4. Credit spread proxy (HYG-IEF spread)
    if 'HYG' in close.columns and 'IEF' in close.columns:
        cs = np.log(close['IEF']) - np.log(close['HYG'])  # Higher = wider spread = more stress
        features['credit_spread'] = cs
        features['credit_spread_change_20d'] = cs.diff(20)

    # 5. Realized vol of each RP asset
    for t in RP_TICKERS:
        if t in rp_rets.columns:
            features[f'rvol_{t}_20d'] = rp_rets[t].rolling(20).std() * np.sqrt(252)

    # 6. Momentum breadth (how many RP assets have positive 20d returns)
    mom_count = pd.DataFrame()
    for t in RP_TICKERS:
        if t in close.columns:
            mom_count[t] = (close[t].pct_change(20) > 0).astype(int)
    if len(mom_count.columns) > 0:
        features['momentum_breadth'] = mom_count.sum(axis=1)

    # 7. RP portfolio recent returns and vol
    rp_port_ret = (rp_rets * rp_weights).sum(axis=1)
    features['rp_ret_20d'] = rp_port_ret.rolling(20).sum()
    features['rp_vol_20d'] = rp_port_ret.rolling(20).std() * np.sqrt(252)
    features['rp_sharpe_60d'] = (rp_port_ret.rolling(60).mean() / rp_port_ret.rolling(60).std().clip(lower=1e-6)) * np.sqrt(252)

    # 8. Momentum of each asset
    for t in RP_TICKERS:
        if t in close.columns:
            features[f'mom_{t}_60d'] = close[t].pct_change(60)

    return features.replace([np.inf, -np.inf], np.nan)


def run_direction1():
    print("=" * 80)
    print("DIRECTION 1: ML-Driven Dynamic Leverage on Risk Parity")
    print("=" * 80)

    # Download data
    close = download_data()

    # RP asset returns
    rp_cols = [c for c in RP_TICKERS if c in close.columns]
    rp_rets = close[rp_cols].pct_change()
    rp_weights = build_risk_parity_weights(rp_rets, lookback=60)

    # Unlevered RP portfolio return
    rp_port_ret = (rp_rets * rp_weights).sum(axis=1)

    # Forward 21-day return as target
    fwd_ret = rp_port_ret.rolling(TEST_DAYS).sum().shift(-TEST_DAYS)

    # Build features
    features = build_features(close, rp_rets, rp_weights)
    feat_cols = features.columns.tolist()

    # Align everything
    common_idx = features.dropna().index.intersection(fwd_ret.dropna().index)
    features = features.loc[common_idx]
    fwd_ret = fwd_ret.loc[common_idx]
    rp_port_ret = rp_port_ret.reindex(common_idx)
    rp_weights = rp_weights.reindex(common_idx)

    print(f"Feature matrix: {features.shape}, target range: {fwd_ret.min():.4f} to {fwd_ret.max():.4f}")
    print(f"Features: {feat_cols}")

    # Walk-forward
    oot_preds = []
    oot_actuals = []
    oot_dates = []
    oot_leverages = []
    feature_importances = np.zeros(len(feat_cols))
    n_folds = 0

    for start_idx in range(TRAIN_DAYS, len(features) - TEST_DAYS, TEST_DAYS):
        train_end = start_idx
        test_end = min(start_idx + TEST_DAYS, len(features))

        X_train = features.iloc[train_end - TRAIN_DAYS:train_end][feat_cols].values
        y_train = fwd_ret.iloc[train_end - TRAIN_DAYS:train_end].values

        X_test = features.iloc[train_end:test_end][feat_cols].values
        y_test = fwd_ret.iloc[train_end:test_end].values
        test_dates = features.index[train_end:test_end]

        # Remove NaN rows
        train_mask = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
        X_train, y_train = X_train[train_mask], y_train[train_mask]
        test_mask = ~(np.isnan(X_test).any(axis=1) | np.isnan(y_test))
        X_test, y_test = X_test[test_mask], y_test[test_mask]
        test_dates = test_dates[test_mask]

        if len(X_train) < 50 or len(X_test) == 0:
            continue

        # Train LightGBM
        dtrain = lgb.Dataset(X_train, y_train)
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'num_leaves': 15,
            'learning_rate': 0.05,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'verbose': -1,
            'n_jobs': -1,
            'seed': 42,
        }
        model = lgb.train(params, dtrain, num_boost_round=100)

        preds = model.predict(X_test)
        feature_importances += model.feature_importance(importance_type='gain')
        n_folds += 1

        oot_preds.extend(preds)
        oot_actuals.extend(y_test)
        oot_dates.extend(test_dates)

    oot_preds = np.array(oot_preds)
    oot_actuals = np.array(oot_actuals)
    oot_dates = pd.DatetimeIndex(oot_dates)

    # IC calculation
    ic, ic_p = spearmanr(oot_preds, oot_actuals)
    print(f"\nOOT IC (Spearman): {ic:.4f} (p={ic_p:.4e})")
    print(f"OOT folds: {n_folds}")

    # Feature importance
    if n_folds > 0:
        fi = feature_importances / n_folds
        fi_df = pd.DataFrame({'feature': feat_cols, 'importance': fi}).sort_values('importance', ascending=False)
        print("\nTop features:")
        for _, row in fi_df.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.1f}")

    # Map predictions to leverage levels
    # Strategy: percentile-rank predictions, map to leverage 1x-5x
    pred_series = pd.Series(oot_preds, index=oot_dates)
    actual_series = pd.Series(oot_actuals, index=oot_dates)

    # Use expanding percentile rank (no lookahead)
    leverage_series = pd.Series(index=oot_dates, dtype=float)
    for i in range(len(pred_series)):
        if i < 21:
            leverage_series.iloc[i] = 3.0  # Default to 3x until enough history
        else:
            pctile = (pred_series.iloc[:i] < pred_series.iloc[i]).mean()
            # Map [0,1] -> [1,5] leverage
            leverage_series.iloc[i] = 1.0 + 4.0 * pctile

    # Simulate daily returns with dynamic leverage
    # For each OOT date, apply the leverage to the base RP return
    spy_rets_full = close['SPY'].pct_change()
    rp_port_ret_full = rp_port_ret.copy()

    # Dynamic leverage portfolio
    dynamic_rets = pd.Series(index=oot_dates, dtype=float)
    fixed3x_rets = pd.Series(index=oot_dates, dtype=float)

    for dt in oot_dates:
        if dt in rp_port_ret_full.index:
            base_ret = rp_port_ret_full.loc[dt]
            lev = leverage_series.loc[dt]
            dynamic_rets.loc[dt] = base_ret * lev
            fixed3x_rets.loc[dt] = base_ret * 3.0

    dynamic_rets = dynamic_rets.dropna()
    fixed3x_rets = fixed3x_rets.reindex(dynamic_rets.index).dropna()
    spy_rets_aligned = spy_rets_full.reindex(dynamic_rets.index).dropna()

    # Metrics
    m_dynamic = calc_metrics(dynamic_rets, 'ML Dynamic Leverage')
    m_fixed = calc_metrics(fixed3x_rets, 'Fixed 3x RP')
    m_spy = calc_metrics(spy_rets_aligned, 'SPY B&H')

    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)
    for m in [m_dynamic, m_fixed, m_spy]:
        print(f"\n{m['name']}:")
        print(f"  CAGR: {m['cagr_pct']:.1f}%  |  Sharpe: {m['sharpe']:.3f}  |  Sortino: {m['sortino']:.3f}")
        print(f"  MaxDD: {m['max_dd_pct']:.1f}%  |  Calmar: {m['calmar']:.3f}  |  WR: {m['win_rate']:.3f}")

    # Regime test
    print("\n--- Regime Test: Dynamic Leverage ---")
    rt = regime_test(dynamic_rets, spy_rets_aligned, 'ML Dynamic Lev')
    for r in ['green', 'red', 'flat']:
        if r in rt:
            print(f"  {r}: Sharpe={rt[r]['sharpe']:.3f}  CAGR={rt[r]['cagr_pct']:.1f}%")
    if 'regime_gap' in rt:
        print(f"  Regime gap: {rt['regime_gap']:.3f} (reject if >0.50)")

    print("\n--- Regime Test: Fixed 3x RP ---")
    rt_fixed = regime_test(fixed3x_rets, spy_rets_aligned, 'Fixed 3x RP')
    for r in ['green', 'red', 'flat']:
        if r in rt_fixed:
            print(f"  {r}: Sharpe={rt_fixed[r]['sharpe']:.3f}  CAGR={rt_fixed[r]['cagr_pct']:.1f}%")
    if 'regime_gap' in rt_fixed:
        print(f"  Regime gap: {rt_fixed['regime_gap']:.3f}")

    # Leverage stats
    print(f"\nLeverage stats: mean={leverage_series.mean():.2f}x, std={leverage_series.std():.2f}, "
          f"min={leverage_series.min():.2f}x, max={leverage_series.max():.2f}x")

    # Save results
    results = {
        'direction': 'D1_ML_Dynamic_Leverage',
        'oot_ic': round(float(ic), 4),
        'oot_ic_pval': float(ic_p),
        'n_folds': n_folds,
        'n_oot_days': len(dynamic_rets),
        'metrics': {
            'dynamic_leverage': m_dynamic,
            'fixed_3x': m_fixed,
            'spy': m_spy,
        },
        'regime_test_dynamic': rt,
        'regime_test_fixed': rt_fixed,
        'leverage_stats': {
            'mean': round(float(leverage_series.mean()), 3),
            'std': round(float(leverage_series.std()), 3),
            'min': round(float(leverage_series.min()), 3),
            'max': round(float(leverage_series.max()), 3),
        },
        'top_features': fi_df.head(10).to_dict('records') if n_folds > 0 else [],
        'honest_assessment': '',
        'timestamp': datetime.now().isoformat(),
    }

    # Honest assessment
    beats_fixed = m_dynamic['cagr'] > m_fixed['cagr']
    better_sharpe = m_dynamic['sharpe'] > m_fixed['sharpe']
    regime_ok = rt.get('regime_gap', 1.0) <= 0.50
    ic_significant = abs(ic) > 0.03 and ic_p < 0.05

    assessment = []
    if ic_significant:
        assessment.append(f"IC={ic:.4f} is statistically significant — model has some predictive power.")
    else:
        assessment.append(f"IC={ic:.4f} is weak/insignificant — model lacks robust predictive edge.")
    if beats_fixed:
        assessment.append(f"Dynamic leverage beats fixed 3x on CAGR ({m_dynamic['cagr_pct']:.1f}% vs {m_fixed['cagr_pct']:.1f}%).")
    else:
        assessment.append(f"Dynamic leverage DOES NOT beat fixed 3x on CAGR ({m_dynamic['cagr_pct']:.1f}% vs {m_fixed['cagr_pct']:.1f}%).")
    if better_sharpe:
        assessment.append(f"Risk-adjusted: dynamic Sharpe ({m_dynamic['sharpe']:.3f}) > fixed ({m_fixed['sharpe']:.3f}).")
    else:
        assessment.append(f"Risk-adjusted: dynamic Sharpe ({m_dynamic['sharpe']:.3f}) <= fixed ({m_fixed['sharpe']:.3f}).")
    if regime_ok:
        assessment.append(f"Regime gap {rt.get('regime_gap', 'N/A')} passes <0.50 threshold.")
    else:
        assessment.append(f"REJECT: Regime gap {rt.get('regime_gap', 'N/A')} fails >0.50 threshold — strategy is regime-tailored.")
    results['honest_assessment'] = ' | '.join(assessment)
    print(f"\nHONEST ASSESSMENT: {results['honest_assessment']}")

    with open(os.path.join(OUT_DIR, 'dir1_ml_dynamic_leverage.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {OUT_DIR}/dir1_ml_dynamic_leverage.json")
    return results


if __name__ == '__main__':
    run_direction1()
