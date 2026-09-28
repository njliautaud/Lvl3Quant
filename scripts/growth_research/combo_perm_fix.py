#!/usr/bin/env python3
"""
Fix permutation test for portfolio combo.
Train once, save daily signals, then shuffle signals (not returns).
This tests: does ML timing add value over random buy/sell decisions?
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import yfinance as yf
import json, os, warnings
warnings.filterwarnings('ignore')

OUTPUT = '/home/jupiter/Lvl3Quant/output/ml_portfolio_combo'
N_PERM = 200
TRAIN_WINDOW = 252
ML_THRESH = 0.55

def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def build_features(df, tickers):
    rows = []
    for t in tickers:
        if t not in df.columns: continue
        p = df[t]
        ret = p.pct_change()
        feats = pd.DataFrame({
            'ret_1d': ret, 'ret_5d': p.pct_change(5),
            'ret_21d': p.pct_change(21), 'ret_63d': p.pct_change(63),
            'vol_21d': ret.rolling(21).std(), 'vol_63d': ret.rolling(63).std(),
            'sma_20_100': (p.rolling(20).mean() / p.rolling(100).mean() - 1),
            'sma_50_200': (p.rolling(50).mean() / p.rolling(200).mean() - 1),
            'rsi_14': compute_rsi(p, 14),
            'target': (ret.shift(-1) > 0).astype(float),
        }, index=df.index)
        feats['ticker'] = t
        feats['fwd_ret'] = ret.shift(-1)  # Store forward return for later
        rows.append(feats.dropna())
    return pd.concat(rows)

def train_and_get_signals(feat_df):
    """Train walk-forward, return per-asset daily signals and forward returns."""
    dates = sorted(feat_df.index.unique())
    feature_cols = [c for c in feat_df.columns if c not in ['target','ticker','fwd_ret']]
    
    # Store: date -> {ticker: (prob, fwd_ret)}
    daily_signals = {}
    
    for i in range(TRAIN_WINDOW, len(dates)):
        test_date = dates[i]
        train_start = dates[max(0, i - TRAIN_WINDOW)]
        train = feat_df[(feat_df.index >= train_start) & (feat_df.index < test_date)]
        test = feat_df[feat_df.index == test_date]
        if len(train) < 50 or len(test) == 0: continue
        
        X_tr, y_tr = train[feature_cols].values, train['target'].values
        X_te = test[feature_cols].values
        
        model = lgb.LGBMClassifier(n_estimators=50, max_depth=4, learning_rate=0.1,
                                    subsample=0.8, colsample_bytree=0.8, verbose=-1,
                                    min_child_samples=20)
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_te)[:, 1]
        
        signals = {}
        for j, (_, row) in enumerate(test.iterrows()):
            signals[row['ticker']] = (float(probs[j]), float(row['fwd_ret']))
        daily_signals[test_date] = signals
        
        if (i - TRAIN_WINDOW + 1) % 500 == 0:
            print(f"    Fold {i - TRAIN_WINDOW + 1}/{len(dates) - TRAIN_WINDOW}...")
    
    return daily_signals

def compute_daily_returns(daily_signals, shuffle=False):
    """Compute strategy daily returns from signals. If shuffle, randomize signal-to-asset mapping."""
    daily_rets = []
    dates = sorted(daily_signals.keys())
    
    for date in dates:
        sigs = daily_signals[date]
        if not sigs: continue
        
        tickers = list(sigs.keys())
        probs = np.array([sigs[t][0] for t in tickers])
        fwd_rets = np.array([sigs[t][1] for t in tickers])
        
        if shuffle:
            # Shuffle which probability goes to which asset's return
            np.random.shuffle(probs)
        
        day_ret = 0.0
        n_pos = 0
        for k in range(len(tickers)):
            direction = 1 if probs[k] > ML_THRESH else (-1 if probs[k] < (1 - ML_THRESH) else 0)
            if direction != 0:
                day_ret += direction * fwd_rets[k]
                n_pos += 1
        
        if n_pos > 0:
            day_ret /= n_pos
        daily_rets.append((date, day_ret))
    
    return pd.Series([r[1] for r in daily_rets], index=[r[0] for r in daily_rets])

def sharpe(returns):
    if len(returns) < 20 or returns.std() == 0: return 0.0
    return returns.mean() / returns.std() * np.sqrt(252)

print("=" * 70)
print("COMBO SIGNAL-SHUFFLE PERMUTATION TEST")
print("=" * 70)

# Download
tickers_cta = ['SPY','TLT','GLD','UUP','EEM','VNQ','HYG','XLE']
tickers_sec = ['XLK','XLF','XLE','XLV','XLY','XLP','XLI','XLB','XLU','XLRE','XLC']
all_t = list(set(tickers_cta + tickers_sec))
print("\nDownloading data...")
df = yf.download(all_t, start='2008-01-01', progress=False)['Close'].ffill().dropna()
print(f"  {len(df)} days")

# Build features
feat_cta = build_features(df, tickers_cta)
feat_sec = build_features(df, tickers_sec)
print(f"  CTA: {len(feat_cta)} obs, Sectors: {len(feat_sec)} obs")

# Train once (slow part)
print("\nTraining CTA walk-forward (one-time)...")
sigs_cta = train_and_get_signals(feat_cta)
print(f"  {len(sigs_cta)} signal days")

print("\nTraining Sectors walk-forward (one-time)...")
sigs_sec = train_and_get_signals(feat_sec)
print(f"  {len(sigs_sec)} signal days")

# Real returns
ret_cta_real = compute_daily_returns(sigs_cta, shuffle=False)
ret_sec_real = compute_daily_returns(sigs_sec, shuffle=False)

# Align dates
common_dates = ret_cta_real.index.intersection(ret_sec_real.index)
ret_cta_real = ret_cta_real[common_dates]
ret_sec_real = ret_sec_real[common_dates]
combo_real = 0.7 * ret_cta_real + 0.3 * ret_sec_real

real_sharpe_cta = sharpe(ret_cta_real)
real_sharpe_sec = sharpe(ret_sec_real)
real_sharpe_combo = sharpe(combo_real)
print(f"\nReal CTA Sharpe:     {real_sharpe_cta:.3f}")
print(f"Real Sectors Sharpe: {real_sharpe_sec:.3f}")
print(f"Real Combo Sharpe:   {real_sharpe_combo:.3f}")

# Permutation test (fast — just shuffles, no retraining)
print(f"\nRunning {N_PERM} signal-shuffle permutations...")
perm_sharpes_combo = []
perm_sharpes_cta = []
perm_sharpes_sec = []

for trial in range(N_PERM):
    ret_cta_perm = compute_daily_returns(sigs_cta, shuffle=True)
    ret_sec_perm = compute_daily_returns(sigs_sec, shuffle=True)
    ret_cta_perm = ret_cta_perm[common_dates]
    ret_sec_perm = ret_sec_perm[common_dates]
    combo_perm = 0.7 * ret_cta_perm + 0.3 * ret_sec_perm
    
    perm_sharpes_cta.append(sharpe(ret_cta_perm))
    perm_sharpes_sec.append(sharpe(ret_sec_perm))
    perm_sharpes_combo.append(sharpe(combo_perm))
    
    if (trial + 1) % 50 == 0:
        print(f"  {trial+1}/{N_PERM} done...")

p_cta = np.mean([s >= real_sharpe_cta for s in perm_sharpes_cta])
p_sec = np.mean([s >= real_sharpe_sec for s in perm_sharpes_sec])
p_combo = np.mean([s >= real_sharpe_combo for s in perm_sharpes_combo])

print(f"\n{'='*70}")
print(f"SIGNAL-SHUFFLE PERMUTATION RESULTS")
print(f"{'='*70}")
print(f"  CTA:     Real={real_sharpe_cta:.3f}, Perm={np.mean(perm_sharpes_cta):.3f}±{np.std(perm_sharpes_cta):.3f}, p={p_cta:.4f} -> {'PASS' if p_cta<0.05 else 'FAIL'}")
print(f"  Sectors: Real={real_sharpe_sec:.3f}, Perm={np.mean(perm_sharpes_sec):.3f}±{np.std(perm_sharpes_sec):.3f}, p={p_sec:.4f} -> {'PASS' if p_sec<0.05 else 'FAIL'}")
print(f"  Combo:   Real={real_sharpe_combo:.3f}, Perm={np.mean(perm_sharpes_combo):.3f}±{np.std(perm_sharpes_combo):.3f}, p={p_combo:.4f} -> {'PASS' if p_combo<0.05 else 'FAIL'}")
print(f"{'='*70}")

result = {
    'test': 'signal_shuffle_permutation_v2',
    'method': 'Shuffle ML probability-to-asset mapping. Models trained once, signals shuffled 200x.',
    'cta': {'real_sharpe': round(real_sharpe_cta,3), 'perm_mean': round(float(np.mean(perm_sharpes_cta)),3), 'p': round(float(p_cta),4), 'pass': bool(p_cta<0.05)},
    'sectors': {'real_sharpe': round(real_sharpe_sec,3), 'perm_mean': round(float(np.mean(perm_sharpes_sec)),3), 'p': round(float(p_sec),4), 'pass': bool(p_sec<0.05)},
    'combo': {'real_sharpe': round(real_sharpe_combo,3), 'perm_mean': round(float(np.mean(perm_sharpes_combo)),3), 'p': round(float(p_combo),4), 'pass': bool(p_combo<0.05)},
    'n_permutations': N_PERM,
    'weights': '70/30 CTA/Sectors'
}
with open(os.path.join(OUTPUT, 'perm_signal_shuffle_result.json'), 'w') as f:
    json.dump(result, f, indent=2)
print(f"\nSaved to {OUTPUT}/perm_signal_shuffle_result.json")
