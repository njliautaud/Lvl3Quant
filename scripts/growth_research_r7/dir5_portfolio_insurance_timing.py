#!/usr/bin/env python3
"""
R7 DIRECTION 5: Portfolio Insurance via Timing (Extend RP Ceiling)

Binary regime switch: toggle between growth portfolio (3-4x RP) and defensive (1x/cash).
Signal: predict whether next month is high-risk or low-risk using macro features.
Different from vol targeting — this is a discrete switch, not continuous scaling.

HC #694: Commission-free (RH/IBKR)
HC #428: Regime-agnostic validation
HC #0: Sliding window walk-forward only
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

RP_TICKERS = ['SPY', 'TLT', 'GLD', 'VNQ']
FEAT_TICKERS = ['^VIX', 'HYG', 'IEF', 'SHY', '^GSPC']


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
            results[r] = calc_metrics(strat_rets[mask], f'{name} ({r})')
    if 'green' in results and 'red' in results:
        sg, sr = results['green']['sharpe'], results['red']['sharpe']
        results['regime_gap'] = round(abs(sg - sr) / max(abs(sg), abs(sr), 0.001), 3)
    return results


def build_risk_features(close, rp_rets):
    """Build features to predict risky vs safe periods."""
    features = pd.DataFrame(index=close.index)

    # 1. VIX and VIX term structure
    if '^VIX' in close.columns:
        features['vix'] = close['^VIX']
        features['vix_20d_ma'] = close['^VIX'].rolling(20).mean()
        features['vix_above_ma'] = (close['^VIX'] > features['vix_20d_ma']).astype(float)
        features['vix_zscore'] = (close['^VIX'] - close['^VIX'].rolling(60).mean()) / close['^VIX'].rolling(60).std().clip(lower=0.01)
        features['vix_change_5d'] = close['^VIX'].pct_change(5)
        features['vix_change_20d'] = close['^VIX'].pct_change(20)

    # 2. Credit spreads (HYG-IEF)
    if 'HYG' in close.columns and 'IEF' in close.columns:
        cs = np.log(close['IEF']) - np.log(close['HYG'])
        features['credit_spread'] = cs
        features['credit_spread_z'] = (cs - cs.rolling(60).mean()) / cs.rolling(60).std().clip(lower=1e-6)
        features['credit_spread_chg_20d'] = cs.diff(20)

    # 3. Yield curve proxy
    if 'TLT' in close.columns and 'SHY' in close.columns:
        yc = np.log(close['TLT']) - np.log(close['SHY'])
        features['yield_curve'] = yc
        features['yield_curve_slope_chg'] = yc.diff(20)

    # 4. Equity momentum and vol
    if 'SPY' in close.columns:
        spy_ret = close['SPY'].pct_change()
        features['spy_mom_20d'] = close['SPY'].pct_change(20)
        features['spy_mom_60d'] = close['SPY'].pct_change(60)
        features['spy_vol_20d'] = spy_ret.rolling(20).std() * np.sqrt(252)
        features['spy_vol_60d'] = spy_ret.rolling(60).std() * np.sqrt(252)
        features['spy_above_200ma'] = (close['SPY'] > close['SPY'].rolling(200).mean()).astype(float)
        features['spy_drawdown'] = close['SPY'] / close['SPY'].rolling(252).max() - 1

    # 5. Cross-asset correlation (diversification breakdown = risk)
    rp_cols = [c for c in RP_TICKERS if c in rp_rets.columns]
    if len(rp_cols) >= 2:
        corrs = []
        for i in range(len(rp_cols)):
            for j in range(i+1, len(rp_cols)):
                c = rp_rets[rp_cols[i]].rolling(20).corr(rp_rets[rp_cols[j]])
                corrs.append(c)
        features['avg_corr_20d'] = pd.concat(corrs, axis=1).mean(axis=1)
        features['max_corr_20d'] = pd.concat(corrs, axis=1).max(axis=1)

    # 6. RP portfolio recent performance
    rp_port = rp_rets[rp_cols].mean(axis=1) if rp_cols else pd.Series(dtype=float)
    if len(rp_port) > 0:
        features['rp_ret_20d'] = rp_port.rolling(20).sum()
        features['rp_vol_20d'] = rp_port.rolling(20).std() * np.sqrt(252)
        features['rp_sharpe_60d'] = (rp_port.rolling(60).mean() / rp_port.rolling(60).std().clip(lower=1e-6)) * np.sqrt(252)

    return features.replace([np.inf, -np.inf], np.nan)


def run_direction5():
    print("=" * 80)
    print("DIRECTION 5: Portfolio Insurance via Timing")
    print("=" * 80)

    # Download data
    tickers = list(set(RP_TICKERS + FEAT_TICKERS + ['SPY']))
    print(f"Downloading: {tickers}")
    data = yf.download(tickers, start=START, end=END, auto_adjust=True, progress=False)
    close = data['Close'] if isinstance(data.columns, pd.MultiIndex) else data
    close = close.ffill().dropna(how='all')
    rets = close.pct_change()

    rp_cols = [c for c in RP_TICKERS if c in rets.columns]
    rp_rets = rets[rp_cols]
    rp_port_ret = rp_rets.mean(axis=1)

    spy_rets = rets['SPY'].dropna()

    # Build features
    features = build_risk_features(close, rp_rets)
    feat_cols = [c for c in features.columns if features[c].notna().sum() > TRAIN_DAYS]
    features = features[feat_cols]

    # Target: is the next 21 days a "bad" period for RP?
    # Binary: 1 = safe (positive forward return), 0 = risky (negative)
    fwd_ret = rp_port_ret.rolling(TEST_DAYS).sum().shift(-TEST_DAYS)
    # Also try: is forward drawdown > 5%?
    fwd_maxdd = pd.Series(index=rp_port_ret.index, dtype=float)
    for i in range(len(rp_port_ret) - TEST_DAYS):
        window = rp_port_ret.iloc[i:i+TEST_DAYS]
        cum = (1 + window).cumprod()
        dd = (cum / cum.cummax() - 1).min()
        fwd_maxdd.iloc[i] = dd

    # Align
    common_idx = features.dropna().index.intersection(fwd_ret.dropna().index)
    features = features.loc[common_idx]
    fwd_ret = fwd_ret.loc[common_idx]
    fwd_maxdd = fwd_maxdd.reindex(common_idx)
    rp_port_ret = rp_port_ret.reindex(common_idx)

    print(f"Feature matrix: {features.shape}")
    print(f"Features: {feat_cols}")

    # ── Strategy A: ML Binary Switch (Growth vs Defensive) ──
    print("\n--- Strategy A: ML Binary Switch ---")

    oot_preds_prob = []
    oot_actuals_binary = []
    oot_dates = []
    fi_total = np.zeros(len(feat_cols))
    n_folds = 0

    for start_idx in range(TRAIN_DAYS, len(features) - TEST_DAYS, TEST_DAYS):
        train_end = start_idx
        test_end = min(start_idx + TEST_DAYS, len(features))

        X_train = features.iloc[train_end - TRAIN_DAYS:train_end][feat_cols].values
        y_train = (fwd_ret.iloc[train_end - TRAIN_DAYS:train_end] > 0).astype(int).values

        X_test = features.iloc[train_end:test_end][feat_cols].values
        y_test = (fwd_ret.iloc[train_end:test_end] > 0).astype(int).values
        test_dates = features.index[train_end:test_end]

        train_mask = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
        X_train, y_train = X_train[train_mask], y_train[train_mask]
        test_mask = ~(np.isnan(X_test).any(axis=1) | np.isnan(y_test))
        X_test, y_test = X_test[test_mask], y_test[test_mask]
        test_dates = test_dates[test_mask]

        if len(X_train) < 50 or len(X_test) == 0:
            continue
        if y_train.sum() < 10 or (len(y_train) - y_train.sum()) < 10:
            continue

        dtrain = lgb.Dataset(X_train, y_train)
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
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
        fi_total += model.feature_importance(importance_type='gain')
        n_folds += 1

        oot_preds_prob.extend(preds)
        oot_actuals_binary.extend(y_test)
        oot_dates.extend(test_dates)

    oot_preds_prob = np.array(oot_preds_prob)
    oot_actuals_binary = np.array(oot_actuals_binary)
    oot_dates = pd.DatetimeIndex(oot_dates)

    # Classification accuracy
    pred_binary = (oot_preds_prob > 0.5).astype(int)
    accuracy = (pred_binary == oot_actuals_binary).mean()
    print(f"OOT accuracy: {accuracy:.3f} ({n_folds} folds)")
    print(f"Base rate (% safe): {oot_actuals_binary.mean():.3f}")

    # IC on probability vs actual return
    fwd_ret_aligned = fwd_ret.reindex(oot_dates).dropna()
    common_ic = fwd_ret_aligned.index.intersection(pd.DatetimeIndex(oot_dates))
    if len(common_ic) > 50:
        preds_ic = pd.Series(oot_preds_prob, index=oot_dates).reindex(common_ic)
        ic, ic_p = spearmanr(preds_ic.values, fwd_ret_aligned.loc[common_ic].values)
        print(f"OOT IC (prob vs fwd return): {ic:.4f} (p={ic_p:.4e})")
    else:
        ic, ic_p = 0, 1

    # Feature importance
    if n_folds > 0:
        fi = fi_total / n_folds
        fi_df = pd.DataFrame({'feature': feat_cols, 'importance': fi}).sort_values('importance', ascending=False)
        print("\nTop features:")
        for _, row in fi_df.head(8).iterrows():
            print(f"  {row['feature']}: {row['importance']:.1f}")

    # Simulate: growth vs defensive switch
    # Growth = 4x RP, Defensive = 1x RP
    # Switch based on ML prediction: prob > 0.5 → growth, else defensive
    pred_series = pd.Series(oot_preds_prob, index=oot_dates)

    strategies = {}

    for growth_lev, def_lev, threshold, label in [
        (4.0, 1.0, 0.5, 'ML_Switch_4x_1x'),
        (5.0, 1.0, 0.5, 'ML_Switch_5x_1x'),
        (4.0, 1.5, 0.5, 'ML_Switch_4x_1.5x'),
        (3.0, 1.0, 0.5, 'ML_Switch_3x_1x'),  # Conservative
    ]:
        switch_rets = pd.Series(index=oot_dates, dtype=float)
        for dt in oot_dates:
            if dt in rp_port_ret.index:
                lev = growth_lev if pred_series.loc[dt] > threshold else def_lev
                switch_rets.loc[dt] = rp_port_ret.loc[dt] * lev
        switch_rets = switch_rets.dropna()
        strategies[label] = switch_rets

    # ── Strategy B: Simple Rules-Based Switch ──
    # Use VIX + 200MA as simple risk-on/risk-off
    print("\n--- Strategy B: Rules-Based Switch ---")
    if '^VIX' in close.columns and 'SPY' in close.columns:
        vix = close['^VIX']
        spy_200ma = close['SPY'].rolling(200).mean()

        # Risk-on: VIX < 20 AND SPY above 200MA → 4x
        # Risk-off: VIX > 25 OR SPY below 200MA → 1x
        # Neutral: otherwise → 3x
        rules_lev = pd.Series(3.0, index=close.index)
        risk_on = (vix < 20) & (close['SPY'] > spy_200ma)
        risk_off = (vix > 25) | (close['SPY'] < spy_200ma)
        rules_lev[risk_on] = 4.0
        rules_lev[risk_off] = 1.0

        rules_rets = rp_port_ret * rules_lev.shift(1)
        rules_rets = rules_rets.reindex(oot_dates).dropna()
        strategies['Rules_VIX_200MA'] = rules_rets

    # Fixed 3x baseline
    fixed3x_rets = (rp_port_ret * 3.0).reindex(oot_dates).dropna()
    strategies['Fixed_3x_RP'] = fixed3x_rets

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    metrics_all = {}
    for name, ret_series in strategies.items():
        m = calc_metrics(ret_series, name)
        metrics_all[name] = m
        print(f"\n{name}:")
        print(f"  CAGR: {m['cagr_pct']:.1f}%  |  Sharpe: {m['sharpe']:.3f}  |  Sortino: {m['sortino']:.3f}")
        print(f"  MaxDD: {m['max_dd_pct']:.1f}%  |  Calmar: {m['calmar']:.3f}  |  WR: {m['win_rate']:.3f}")

    # Regime tests
    print("\n--- Regime Tests ---")
    rt_results = {}
    for name, ret_series in strategies.items():
        if len(ret_series) > 100:
            rt = regime_test(ret_series, spy_rets, name)
            rt_results[name] = rt
            for r in ['green', 'red', 'flat']:
                if r in rt:
                    print(f"  {name} {r}: Sharpe={rt[r]['sharpe']:.3f}")
            if 'regime_gap' in rt:
                print(f"  {name} regime gap: {rt['regime_gap']:.3f}")

    # Leverage stats
    print("\n--- Leverage Distribution (ML_Switch_4x_1x) ---")
    if len(pred_series) > 0:
        growth_pct = (pred_series > 0.5).mean()
        print(f"  % time in growth mode: {growth_pct*100:.1f}%")
        print(f"  % time in defensive mode: {(1-growth_pct)*100:.1f}%")

    # Honest assessment
    assessment = []
    if accuracy > 0.55:
        assessment.append(f"ML timing has some skill (accuracy={accuracy:.3f} vs base rate {oot_actuals_binary.mean():.3f}).")
    else:
        assessment.append(f"ML timing accuracy ({accuracy:.3f}) is near base rate ({oot_actuals_binary.mean():.3f}) — no real edge.")

    ml_4x1x = metrics_all.get('ML_Switch_4x_1x', {})
    fixed = metrics_all.get('Fixed_3x_RP', {})
    if ml_4x1x.get('sharpe', 0) > fixed.get('sharpe', 0):
        assessment.append(f"ML switch improves risk-adjusted returns (Sharpe {ml_4x1x['sharpe']:.3f} vs {fixed['sharpe']:.3f}).")
    else:
        assessment.append(f"ML switch does NOT improve risk-adjusted returns (Sharpe {ml_4x1x.get('sharpe',0):.3f} vs {fixed.get('sharpe',0):.3f}).")

    rules = metrics_all.get('Rules_VIX_200MA', {})
    if rules.get('sharpe', 0) > fixed.get('sharpe', 0):
        assessment.append(f"Simple VIX+200MA rules beat ML and fixed ({rules['sharpe']:.3f} Sharpe).")
    else:
        assessment.append(f"Simple VIX+200MA rules underperform fixed ({rules.get('sharpe',0):.3f} vs {fixed.get('sharpe',0):.3f}).")

    assessment.append("CRITICAL LIMITATION: Timing strategies that avoid drawdowns also miss recoveries. Net effect is often a wash.")
    assessment.append("LIMITATION: Binary switch creates whipsaw risk — switching leverage daily based on noisy predictions is costly.")

    print(f"\nHONEST ASSESSMENT: {' | '.join(assessment)}")

    results = {
        'direction': 'D5_Portfolio_Insurance_Timing',
        'ml_accuracy': round(float(accuracy), 4),
        'ml_base_rate': round(float(oot_actuals_binary.mean()), 4),
        'oot_ic': round(float(ic), 4),
        'oot_ic_pval': float(ic_p),
        'n_folds': n_folds,
        'metrics': metrics_all,
        'regime_tests': rt_results,
        'growth_mode_pct': round(float(growth_pct), 3) if len(pred_series) > 0 else 0,
        'top_features': fi_df.head(8).to_dict('records') if n_folds > 0 else [],
        'honest_assessment': ' | '.join(assessment),
        'timestamp': datetime.now().isoformat(),
    }

    with open(os.path.join(OUT_DIR, 'dir5_portfolio_insurance.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {OUT_DIR}/dir5_portfolio_insurance.json")
    return results


if __name__ == '__main__':
    run_direction5()
