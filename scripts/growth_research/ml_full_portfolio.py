#!/usr/bin/env python3
"""
Full Multi-Strategy Portfolio Construction
==========================================
Combines all validated strategies into one portfolio:
1. ML CTA Trend Following (growth) — Sharpe 2.92
2. ML Sector Rotation (growth) — Sharpe 2.47
3. ML Credit Timing (income) — Sharpe 1.19
4. VIX Leverage (tactical) — conditional

Tests: does adding credit timing to the growth portfolio improve risk-adjusted returns?
Key question: correlation between credit timing and equity strategies during stress.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import yfinance as yf
from datetime import datetime
import json, os, warnings
warnings.filterwarnings('ignore')

OUTPUT = '/home/jupiter/Lvl3Quant/output/ml_full_portfolio'
os.makedirs(OUTPUT, exist_ok=True)
np.random.seed(42)

TRAIN_WINDOW = 252

def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def build_trend_features(df, tickers):
    rows = []
    for t in tickers:
        if t not in df.columns: continue
        p = df[t]; ret = p.pct_change()
        feats = pd.DataFrame({
            'ret_1d': ret, 'ret_5d': p.pct_change(5),
            'ret_21d': p.pct_change(21), 'ret_63d': p.pct_change(63),
            'vol_21d': ret.rolling(21).std(), 'vol_63d': ret.rolling(63).std(),
            'sma_20_100': (p.rolling(20).mean() / p.rolling(100).mean() - 1),
            'sma_50_200': (p.rolling(50).mean() / p.rolling(200).mean() - 1),
            'rsi_14': compute_rsi(p, 14),
            'target': (ret.shift(-1) > 0).astype(float),
            'fwd_ret': ret.shift(-1),
        }, index=df.index)
        feats['ticker'] = t
        rows.append(feats.dropna())
    return pd.concat(rows)

def run_trend_wf(feat_df, ml_thresh=0.55):
    dates = sorted(feat_df.index.unique())
    feature_cols = [c for c in feat_df.columns if c not in ['target','ticker','fwd_ret']]
    daily_rets = []
    for i in range(TRAIN_WINDOW, len(dates)):
        test_date = dates[i]
        train_start = dates[max(0, i - TRAIN_WINDOW)]
        train = feat_df[(feat_df.index >= train_start) & (feat_df.index < test_date)]
        test = feat_df[feat_df.index == test_date]
        if len(train) < 50 or len(test) == 0: continue
        model = lgb.LGBMClassifier(n_estimators=50, max_depth=4, learning_rate=0.1,
                                    subsample=0.8, colsample_bytree=0.8, verbose=-1, min_child_samples=20)
        model.fit(train[feature_cols].values, train['target'].values)
        probs = model.predict_proba(test[feature_cols].values)[:, 1]
        day_ret = 0.0; n_pos = 0
        for j, (_, row) in enumerate(test.iterrows()):
            direction = 1 if probs[j] > ml_thresh else (-1 if probs[j] < (1 - ml_thresh) else 0)
            if direction != 0:
                next_dates = [d for d in dates if d > test_date]
                if next_dates:
                    nd = next_dates[0]
                    nd_data = feat_df[(feat_df['ticker'] == row['ticker']) & (feat_df.index == nd)]
                    if len(nd_data) > 0:
                        day_ret += direction * nd_data['ret_1d'].iloc[0]; n_pos += 1
        if n_pos > 0: day_ret /= n_pos
        daily_rets.append((test_date, day_ret))
        if (i - TRAIN_WINDOW + 1) % 500 == 0:
            print(f"      Fold {i - TRAIN_WINDOW + 1}/{len(dates) - TRAIN_WINDOW}...")
    return pd.Series([r[1] for r in daily_rets], index=[r[0] for r in daily_rets])

def run_credit_wf(close):
    """Credit timing walk-forward returning daily returns (for correlation analysis)."""
    features = pd.DataFrame(index=close.index)
    if 'HYG' in close.columns and 'LQD' in close.columns:
        spread = close['HYG'] / close['LQD']
        features['spread_level'] = spread / spread.rolling(252).mean() - 1
        features['spread_5d'] = spread.pct_change(5)
        features['spread_21d'] = spread.pct_change(21)
        features['spread_63d'] = spread.pct_change(63)
        features['spread_vol'] = spread.pct_change().rolling(21).std()
    for t in ['HYG','SPY']:
        if t in close.columns:
            p = close[t]; ret = p.pct_change()
            features[f'{t.lower()}_ret_5d'] = p.pct_change(5)
            features[f'{t.lower()}_ret_21d'] = p.pct_change(21)
            features[f'{t.lower()}_vol_21d'] = ret.rolling(21).std()
            features[f'{t.lower()}_sma_50'] = p / p.rolling(50).mean() - 1
    if 'VIX' in close.columns:
        features['vix_level'] = close['VIX']
        features['vix_sma_20'] = close['VIX'] / close['VIX'].rolling(20).mean() - 1
    if 'TLT' in close.columns:
        features['tlt_ret_21d'] = close['TLT'].pct_change(21)
    if 'HYG' in close.columns and 'SHY' in close.columns:
        features['target'] = (close['HYG'].pct_change(21).shift(-21) > close['SHY'].pct_change(21).shift(-21)).astype(float)
    features = features.dropna()
    
    feature_cols = [c for c in features.columns if c != 'target']
    dates = features.index
    daily_rets = []
    
    for i in range(TRAIN_WINDOW, len(dates)):
        date = dates[i]
        train = features.iloc[max(0,i-TRAIN_WINDOW):i]
        if len(train) < 100: continue
        model = lgb.LGBMClassifier(n_estimators=50, max_depth=4, learning_rate=0.1,
                                    subsample=0.8, colsample_bytree=0.8, verbose=-1, min_child_samples=20)
        model.fit(train[feature_cols].values, train['target'].values)
        prob = model.predict_proba(features.iloc[i:i+1][feature_cols].values)[:, 1][0]
        
        # Daily return based on allocation
        hyg_ret = close['HYG'].pct_change().iloc[close.index.get_loc(date)] if date in close.index else 0
        tlt_ret = close['TLT'].pct_change().iloc[close.index.get_loc(date)] if 'TLT' in close.columns and date in close.index else 0
        lqd_ret = close['LQD'].pct_change().iloc[close.index.get_loc(date)] if 'LQD' in close.columns and date in close.index else 0
        shy_ret = close['SHY'].pct_change().iloc[close.index.get_loc(date)] if 'SHY' in close.columns and date in close.index else 0
        
        if prob > 0.65:
            day_ret = 0.6 * hyg_ret + 0.2 * lqd_ret + 0.2 * shy_ret
        elif prob < 0.35:
            day_ret = 0.6 * tlt_ret + 0.2 * lqd_ret + 0.2 * shy_ret
        else:
            day_ret = 0.4 * lqd_ret + 0.3 * hyg_ret + 0.3 * shy_ret
        
        daily_rets.append((date, float(day_ret)))
        if (i - TRAIN_WINDOW + 1) % 500 == 0:
            print(f"      Credit fold {i - TRAIN_WINDOW + 1}/{len(dates) - TRAIN_WINDOW}...")
    
    return pd.Series([r[1] for r in daily_rets], index=[r[0] for r in daily_rets])

def sharpe(returns):
    if len(returns) < 20 or returns.std() == 0: return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(252))

def compute_metrics(returns, name=""):
    if returns is None or len(returns) < 50: return None
    ret = returns.values
    if np.std(ret) == 0: return None
    s = np.mean(ret) / np.std(ret) * np.sqrt(252)
    neg = ret[ret < 0]
    sort = np.mean(ret) / np.std(neg) * np.sqrt(252) if len(neg) > 0 and np.std(neg) > 0 else 0
    cum = (1 + pd.Series(ret)).cumprod()
    dd = (cum / cum.cummax() - 1).min() * 100
    years = len(ret) / 252
    cagr = (cum.iloc[-1] ** (1/years) - 1) * 100 if years > 0 and cum.iloc[-1] > 0 else 0
    wins = np.sum(ret > 0); trades = np.sum(ret != 0)
    wr = wins / trades * 100 if trades > 0 else 0
    pf = abs(ret[ret > 0].sum() / ret[ret < 0].sum()) if len(neg) > 0 and ret[ret < 0].sum() != 0 else 0
    return {'name': name, 'sharpe': round(float(s),3), 'sortino': round(float(sort),3),
            'cagr': round(float(cagr),1), 'max_dd': round(float(dd),1),
            'profit_factor': round(float(pf),3), 'win_rate': round(float(wr),1), 'years': round(years,1)}

print("=" * 70)
print("FULL MULTI-STRATEGY PORTFOLIO")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# Download all data
tickers_cta = ['SPY','TLT','GLD','UUP','EEM','VNQ','HYG','XLE']
tickers_sec = ['XLK','XLF','XLE','XLV','XLY','XLP','XLI','XLB','XLU','XLRE','XLC']
tickers_credit = ['HYG','LQD','TLT','SHY','AGG','^VIX']
all_tickers = list(set(tickers_cta + tickers_sec + tickers_credit))
print("\nDownloading data...")
df = yf.download(all_tickers, start='2008-01-01', progress=False)
if hasattr(df.index, 'tz') and df.index.tz is not None:
    df.index = df.index.tz_localize(None)
close = df['Close'] if isinstance(df.columns, pd.MultiIndex) else df
if '^VIX' in close.columns:
    close = close.rename(columns={'^VIX': 'VIX'})
close = close.ffill().dropna()
print(f"  {len(close)} days, {close.index[0].date()} to {close.index[-1].date()}")

# Strategy 1: CTA Trend
print("\n[1/3] CTA Trend Following...")
feat_cta = build_trend_features(close, tickers_cta)
print(f"  {len(feat_cta)} obs")
ret_cta = run_trend_wf(feat_cta)
print(f"  CTA Sharpe: {sharpe(ret_cta):.3f}")

# Strategy 2: Sector Rotation
print("\n[2/3] Sector Rotation...")
feat_sec = build_trend_features(close, tickers_sec)
print(f"  {len(feat_sec)} obs")
ret_sec = run_trend_wf(feat_sec)
print(f"  Sectors Sharpe: {sharpe(ret_sec):.3f}")

# Strategy 3: Credit Timing
print("\n[3/3] Credit Timing...")
ret_credit = run_credit_wf(close)
print(f"  Credit Sharpe: {sharpe(ret_credit):.3f}")

# Align all return series
common = ret_cta.index.intersection(ret_sec.index).intersection(ret_credit.index)
ret_cta = ret_cta[common]
ret_sec = ret_sec[common]
ret_credit = ret_credit[common]

print(f"\n  Common dates: {len(common)} ({common[0].date()} to {common[-1].date()})")

# Correlation matrix
print("\n" + "=" * 70)
print("CORRELATION ANALYSIS")
print("=" * 70)
corr_df = pd.DataFrame({'CTA': ret_cta, 'Sectors': ret_sec, 'Credit': ret_credit})
corr = corr_df.corr()
print(f"\n  Full period:")
print(f"    CTA-Sectors: {corr.loc['CTA','Sectors']:.3f}")
print(f"    CTA-Credit:  {corr.loc['CTA','Credit']:.3f}")
print(f"    Sectors-Credit: {corr.loc['Sectors','Credit']:.3f}")

# Stress correlation (VIX > 25)
if 'VIX' in close.columns:
    vix = close['VIX'].reindex(common)
    stress_mask = vix > 25
    calm_mask = vix <= 25
    if stress_mask.sum() > 20:
        corr_stress = corr_df[stress_mask].corr()
        corr_calm = corr_df[calm_mask].corr()
        print(f"\n  During stress (VIX>25, {stress_mask.sum()} days):")
        print(f"    CTA-Sectors: {corr_stress.loc['CTA','Sectors']:.3f}")
        print(f"    CTA-Credit:  {corr_stress.loc['CTA','Credit']:.3f}")
        print(f"    Sectors-Credit: {corr_stress.loc['Sectors','Credit']:.3f}")

# Portfolio combinations
print("\n" + "=" * 70)
print("PORTFOLIO COMBINATIONS")
print("=" * 70)

combos = {
    'Growth Only (70/30 CTA/Sec)': 0.7 * ret_cta + 0.3 * ret_sec,
    'Growth+Income (60/25/15)': 0.60 * ret_cta + 0.25 * ret_sec + 0.15 * ret_credit,
    'Growth+Income (50/20/30)': 0.50 * ret_cta + 0.20 * ret_sec + 0.30 * ret_credit,
    'Equal Weight (33/33/33)': (ret_cta + ret_sec + ret_credit) / 3,
    'Income Heavy (30/20/50)': 0.30 * ret_cta + 0.20 * ret_sec + 0.50 * ret_credit,
}

print(f"\n  {'Variant':40s} {'Sharpe':>7s} {'Sortino':>8s} {'CAGR':>6s} {'MaxDD':>6s} {'WR':>5s}")
print(f"  {'-'*75}")
for name, ret in combos.items():
    m = compute_metrics(ret, name)
    if m:
        print(f"  {m['name']:40s} {m['sharpe']:7.3f} {m['sortino']:8.3f} {m['cagr']:5.1f}% {m['max_dd']:5.1f}% {m['win_rate']:4.1f}%")

# SPY benchmark
spy_ret = close['SPY'].pct_change().reindex(common).fillna(0)
m_spy = compute_metrics(spy_ret, 'SPY B&H')
if m_spy:
    print(f"  {m_spy['name']:40s} {m_spy['sharpe']:7.3f} {m_spy['sortino']:8.3f} {m_spy['cagr']:5.1f}% {m_spy['max_dd']:5.1f}% {m_spy['win_rate']:4.1f}%")

# Save results
output = {
    'strategy': 'Full Multi-Strategy Portfolio',
    'timestamp': datetime.now().isoformat(),
    'individual': {
        'cta': compute_metrics(ret_cta, 'CTA'),
        'sectors': compute_metrics(ret_sec, 'Sectors'),
        'credit': compute_metrics(ret_credit, 'Credit'),
    },
    'combinations': {k: compute_metrics(v, k) for k, v in combos.items()},
    'correlations': {
        'cta_sec': round(float(corr.loc['CTA','Sectors']), 3),
        'cta_credit': round(float(corr.loc['CTA','Credit']), 3),
        'sec_credit': round(float(corr.loc['Sectors','Credit']), 3),
    },
    'benchmark': m_spy,
}
with open(os.path.join(OUTPUT, 'results.json'), 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nRuntime: {datetime.now().strftime('%H:%M:%S')}")
print(f"Output: {OUTPUT}")
