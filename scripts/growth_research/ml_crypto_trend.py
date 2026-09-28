#!/usr/bin/env python3
"""
ML Crypto Trend Following
===========================
Tests if our proven ML trend framework (Sharpe 2.9 on equities) generalizes to crypto.
Crypto has strong documented momentum effects and is uncorrelated with equities.

Universe: BTC, ETH + crypto-adjacent (MSTR, COIN, GBTC, BITO)
Features: Same trend features as CTA (momentum, vol, SMA crossovers)
Target: Next-day direction prediction
HC #714: Income + growth, ML for exploration
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import yfinance as yf
from datetime import datetime
import json, os, warnings
warnings.filterwarnings('ignore')

OUTPUT = '/home/jupiter/Lvl3Quant/output/ml_crypto_trend'
os.makedirs(OUTPUT, exist_ok=True)
np.random.seed(42)

N_PERM = 50
TRAIN_WINDOW = 252

def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def download_data():
    # Crypto + crypto-adjacent + market context
    tickers = ['BTC-USD','ETH-USD','MSTR','COIN','SPY','GLD','TLT','^VIX']
    print("  Downloading...")
    df = yf.download(tickers, start='2015-01-01', progress=False)
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    close = df['Close'] if isinstance(df.columns, pd.MultiIndex) else df
    close = close.rename(columns={'^VIX': 'VIX', 'BTC-USD': 'BTC', 'ETH-USD': 'ETH'})
    close = close.ffill()
    print(f"  {len(close)} days, cols: {close.columns.tolist()}")
    return close

def build_features(close, ticker):
    if ticker not in close.columns:
        return None
    p = close[ticker].dropna()
    ret = p.pct_change()
    
    feats = pd.DataFrame(index=p.index)
    feats['ret_1d'] = ret
    feats['ret_5d'] = p.pct_change(5)
    feats['ret_21d'] = p.pct_change(21)
    feats['ret_63d'] = p.pct_change(63)
    feats['vol_21d'] = ret.rolling(21).std()
    feats['vol_63d'] = ret.rolling(63).std()
    feats['vol_ratio'] = feats['vol_21d'] / (feats['vol_63d'] + 1e-10)
    feats['sma_20_50'] = p.rolling(20).mean() / p.rolling(50).mean() - 1
    feats['sma_50_200'] = p.rolling(50).mean() / p.rolling(200).mean() - 1
    feats['rsi_14'] = compute_rsi(p, 14)
    feats['drawdown'] = p / p.cummax() - 1
    
    # Cross-asset features
    if 'SPY' in close.columns:
        spy = close['SPY'].reindex(p.index).ffill()
        feats['spy_ret_21d'] = spy.pct_change(21)
        feats['corr_spy_21d'] = ret.rolling(21).corr(spy.pct_change())
    if 'GLD' in close.columns:
        feats['gld_ret_21d'] = close['GLD'].reindex(p.index).ffill().pct_change(21)
    if 'VIX' in close.columns:
        vix = close['VIX'].reindex(p.index).ffill()
        feats['vix_level'] = vix
        feats['vix_change'] = vix.diff(5)
    
    # Target: next day positive?
    feats['target'] = (ret.shift(-1) > 0).astype(float)
    feats['fwd_ret'] = ret.shift(-1)
    
    return feats.dropna()

def run_strategy(feats, ticker, shuffle=False):
    feature_cols = [c for c in feats.columns if c not in ['target','fwd_ret']]
    daily_rets = []
    
    for i in range(TRAIN_WINDOW, len(feats)):
        train = feats.iloc[max(0,i-TRAIN_WINDOW):i]
        if len(train) < 100:
            continue
        
        X_tr = train[feature_cols].values
        y_tr = train['target'].values
        X_te = feats.iloc[i:i+1][feature_cols].values
        
        model = lgb.LGBMClassifier(n_estimators=50, max_depth=4, learning_rate=0.1,
                                    subsample=0.8, colsample_bytree=0.8, verbose=-1, min_child_samples=20)
        model.fit(X_tr, y_tr)
        prob = model.predict_proba(X_te)[:, 1][0]
        
        if shuffle:
            prob = np.random.random()
        
        fwd = float(feats.iloc[i]['fwd_ret'])
        if prob > 0.55:
            daily_rets.append((feats.index[i], fwd))
        elif prob < 0.45:
            daily_rets.append((feats.index[i], -fwd))
        else:
            daily_rets.append((feats.index[i], 0.0))
        
        if (i - TRAIN_WINDOW + 1) % 500 == 0:
            print(f"      {ticker} fold {i-TRAIN_WINDOW+1}/{len(feats)-TRAIN_WINDOW}...")
    
    return pd.Series([r[1] for r in daily_rets], index=[r[0] for r in daily_rets])

def compute_metrics(returns, name=""):
    if returns is None or len(returns) < 50: return None
    ret = returns.values
    if np.std(ret) == 0: return None
    s = float(np.mean(ret) / np.std(ret) * np.sqrt(252))
    neg = ret[ret < 0]
    sort = float(np.mean(ret) / np.std(neg) * np.sqrt(252)) if len(neg) > 0 and np.std(neg) > 0 else 0
    cum = (1 + pd.Series(ret)).cumprod()
    dd = float((cum / cum.cummax() - 1).min() * 100)
    years = len(ret) / 252
    cagr = float((cum.iloc[-1] ** (1/years) - 1) * 100) if years > 0 and cum.iloc[-1] > 0 else 0
    wins = np.sum(ret > 0); trades = np.sum(ret != 0)
    wr = float(wins / trades * 100) if trades > 0 else 0
    pf = float(abs(ret[ret > 0].sum() / ret[ret < 0].sum())) if len(neg) > 0 and ret[ret < 0].sum() != 0 else 0
    return {'name': name, 'sharpe': round(s,3), 'sortino': round(sort,3), 'cagr': round(cagr,1),
            'max_dd': round(dd,1), 'profit_factor': round(pf,3), 'win_rate': round(wr,1), 'years': round(years,1)}

print("=" * 70)
print("ML CRYPTO TREND FOLLOWING")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

close = download_data()

# Test each crypto asset
results = {}
for ticker in ['BTC', 'ETH']:
    print(f"\n{'='*70}")
    print(f"STRATEGY: ML {ticker} Trend")
    print(f"{'='*70}")
    
    feats = build_features(close, ticker)
    if feats is None or len(feats) < 500:
        print(f"  Insufficient data for {ticker}")
        continue
    
    print(f"  {len(feats)} observations, target rate: {feats['target'].mean():.1%}")
    
    # Baseline: buy and hold
    bh_ret = close[ticker].pct_change().dropna()
    bh_ret = bh_ret[bh_ret.index >= feats.index[TRAIN_WINDOW]]
    m_bh = compute_metrics(bh_ret, f"{ticker} B&H")
    if m_bh:
        print(f"  Baseline B&H: Sharpe={m_bh['sharpe']:.3f}, CAGR={m_bh['cagr']:.1f}%, MaxDD={m_bh['max_dd']:.1f}%")
    
    # ML strategy
    print(f"  Training ML walk-forward...")
    ret_ml = run_strategy(feats, ticker)
    m_ml = compute_metrics(ret_ml, f"ML {ticker} Trend")
    if m_ml:
        print(f"  ML Result: Sharpe={m_ml['sharpe']:.3f}, Sortino={m_ml['sortino']:.3f}, CAGR={m_ml['cagr']:.1f}%, MaxDD={m_ml['max_dd']:.1f}%")
        
        # Permutation test
        print(f"  Permutation test ({N_PERM} iters)...")
        perm_sharpes = []
        for trial in range(N_PERM):
            perm_ret = run_strategy(feats, ticker, shuffle=True)
            pm = compute_metrics(perm_ret)
            if pm: perm_sharpes.append(pm['sharpe'])
            if (trial+1) % 10 == 0: print(f"    {trial+1}/{N_PERM}...")
        
        p_val = float(np.mean([s >= m_ml['sharpe'] for s in perm_sharpes])) if perm_sharpes else 1.0
        print(f"  Perm: Real={m_ml['sharpe']:.3f}, Mean={np.mean(perm_sharpes):.3f}±{np.std(perm_sharpes):.3f}, p={p_val:.4f} -> {'PASS' if p_val<0.05 else 'FAIL'}")
        
        m_ml['permutation'] = {'p': round(p_val,4), 'pass': p_val < 0.05, 'perm_mean': round(float(np.mean(perm_sharpes)),3)}
        results[ticker] = m_ml
    
    # Correlation with SPY
    spy_ret = close['SPY'].pct_change().reindex(ret_ml.index).fillna(0)
    corr_spy = float(ret_ml.corr(spy_ret))
    print(f"  Correlation with SPY: {corr_spy:.3f}")
    if ticker in results:
        results[ticker]['corr_spy'] = round(corr_spy, 3)

# Combined crypto portfolio
if 'BTC' in results and 'ETH' in results:
    print(f"\n{'='*70}")
    print("COMBINED CRYPTO PORTFOLIO")
    print(f"{'='*70}")
    
    feat_btc = build_features(close, 'BTC')
    feat_eth = build_features(close, 'ETH')
    ret_btc = run_strategy(feat_btc, 'BTC')
    ret_eth = run_strategy(feat_eth, 'ETH')
    common = ret_btc.index.intersection(ret_eth.index)
    combo = 0.6 * ret_btc[common] + 0.4 * ret_eth[common]
    m_combo = compute_metrics(combo, "60/40 BTC/ETH ML Trend")
    if m_combo:
        print(f"  Combined: {m_combo}")
        results['combo'] = m_combo

# Summary
print(f"\n{'='*70}")
print("FINAL SUMMARY")
print(f"{'='*70}")
for k, m in results.items():
    perm_str = f"  perm p={m['permutation']['p']:.3f} {'PASS' if m['permutation']['pass'] else 'FAIL'}" if 'permutation' in m else ""
    print(f"  {m['name']:25s}  Sharpe={m['sharpe']:6.3f}  CAGR={m['cagr']:5.1f}%  MaxDD={m['max_dd']:5.1f}%{perm_str}")

output = {'strategy': 'ML Crypto Trend', 'timestamp': datetime.now().isoformat(), 'results': results}
with open(os.path.join(OUTPUT, 'results.json'), 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nOutput: {OUTPUT}")
