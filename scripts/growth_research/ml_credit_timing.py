#!/usr/bin/env python3
"""
ML Credit Spread Timing Strategy
==================================
Uses ML to predict when to be in high-yield bonds (HYG) vs investment-grade (LQD)
vs treasuries (TLT/SHY). Income strategy — bonds pay yield with alpha from timing.

Signal: ML predicts credit spread direction from macro indicators
Income: HYG yields ~5-7%, LQD ~4-5%, TLT ~3-4%
Alpha: Rotate to safety before spreads widen (avoid HY drawdowns)

HC #713: Fixed capital, no DCA
HC #714: Income + growth focus
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import yfinance as yf
from datetime import datetime
import json, os, warnings
warnings.filterwarnings('ignore')

OUTPUT = '/home/jupiter/Lvl3Quant/output/ml_credit_timing'
os.makedirs(OUTPUT, exist_ok=True)
np.random.seed(42)

N_PERM = 50
TRAIN_WINDOW = 252
REBAL_PERIOD = 21  # Only rebalance every 21 days (non-overlapping)

def download_data():
    """Download credit ETFs + macro indicators."""
    tickers = ['HYG','LQD','TLT','SHY','SPY','IEF','AGG','EMB','JNK','^VIX']
    print("  Downloading data...")
    df = yf.download(tickers, start='2008-01-01', progress=False)
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    
    close = df['Close'] if isinstance(df.columns, pd.MultiIndex) else df
    
    # Rename VIX column
    if '^VIX' in close.columns:
        close = close.rename(columns={'^VIX': 'VIX'})
    
    close = close.ffill().dropna()
    print(f"  {len(close)} days, {close.index[0].date()} to {close.index[-1].date()}")
    return close

def build_features(close):
    """Build credit timing features."""
    features = pd.DataFrame(index=close.index)
    
    # Credit spread proxy: HYG vs LQD relative performance
    if 'HYG' in close.columns and 'LQD' in close.columns:
        spread = close['HYG'] / close['LQD']
        features['spread_level'] = spread / spread.rolling(252).mean() - 1
        features['spread_5d'] = spread.pct_change(5)
        features['spread_21d'] = spread.pct_change(21)
        features['spread_63d'] = spread.pct_change(63)
        features['spread_vol'] = spread.pct_change().rolling(21).std()
    
    # HYG features
    if 'HYG' in close.columns:
        hyg = close['HYG']
        hyg_ret = hyg.pct_change()
        features['hyg_ret_5d'] = hyg.pct_change(5)
        features['hyg_ret_21d'] = hyg.pct_change(21)
        features['hyg_vol_21d'] = hyg_ret.rolling(21).std()
        features['hyg_vol_63d'] = hyg_ret.rolling(63).std()
        features['hyg_sma_50'] = hyg / hyg.rolling(50).mean() - 1
        features['hyg_sma_200'] = hyg / hyg.rolling(200).mean() - 1
        features['hyg_drawdown'] = hyg / hyg.cummax() - 1
    
    # Equity context
    if 'SPY' in close.columns:
        spy = close['SPY']
        spy_ret = spy.pct_change()
        features['spy_ret_5d'] = spy.pct_change(5)
        features['spy_ret_21d'] = spy.pct_change(21)
        features['spy_vol_21d'] = spy_ret.rolling(21).std()
        features['spy_sma_50'] = spy / spy.rolling(50).mean() - 1
        features['spy_sma_200'] = spy / spy.rolling(200).mean() - 1
    
    # VIX
    if 'VIX' in close.columns:
        vix = close['VIX']
        features['vix_level'] = vix
        features['vix_sma_20'] = vix / vix.rolling(20).mean() - 1
        features['vix_change_5d'] = vix.diff(5)
        features['vix_change_21d'] = vix.diff(21)
    
    # TLT (flight to safety signal)
    if 'TLT' in close.columns:
        features['tlt_ret_5d'] = close['TLT'].pct_change(5)
        features['tlt_ret_21d'] = close['TLT'].pct_change(21)
    
    # Cross-asset: equity-credit divergence
    if 'SPY' in close.columns and 'HYG' in close.columns:
        features['eq_credit_div'] = close['SPY'].pct_change(21) - close['HYG'].pct_change(21)
    
    # Target: is HYG going to outperform SHY over next 21 days?
    if 'HYG' in close.columns and 'SHY' in close.columns:
        hyg_fwd = close['HYG'].pct_change(21).shift(-21)
        shy_fwd = close['SHY'].pct_change(21).shift(-21)
        features['target'] = (hyg_fwd > shy_fwd).astype(float)
        features['fwd_hyg'] = hyg_fwd
        features['fwd_shy'] = shy_fwd
        features['fwd_lqd'] = close['LQD'].pct_change(21).shift(-21) if 'LQD' in close.columns else 0
        features['fwd_tlt'] = close['TLT'].pct_change(21).shift(-21) if 'TLT' in close.columns else 0
    
    return features.dropna()

def run_strategy(features, close, shuffle_signals=False):
    """Walk-forward ML credit timing."""
    feature_cols = [c for c in features.columns if c not in ['target','fwd_hyg','fwd_shy','fwd_lqd','fwd_tlt']]
    dates = features.index
    
    period_returns = []
    positions = []

    # Only rebalance every REBAL_PERIOD days (non-overlapping returns)
    rebal_indices = list(range(TRAIN_WINDOW, len(dates), REBAL_PERIOD))

    for i in rebal_indices:
        if i >= len(dates):
            break
        date = dates[i]
        train = features.iloc[max(0, i-TRAIN_WINDOW):i]

        if len(train) < 100:
            continue

        X_tr = train[feature_cols].values
        y_tr = train['target'].values
        X_te = features.iloc[i:i+1][feature_cols].values

        model = lgb.LGBMClassifier(n_estimators=50, max_depth=4, learning_rate=0.1,
                                    subsample=0.8, colsample_bytree=0.8, verbose=-1,
                                    min_child_samples=20)
        model.fit(X_tr, y_tr)
        prob = model.predict_proba(X_te)[:, 1][0]

        if shuffle_signals:
            prob = np.random.random()

        # Allocation based on ML confidence
        row = features.iloc[i]
        if prob > 0.65:
            period_ret = 0.6 * float(row.get('fwd_hyg', 0)) + 0.2 * float(row.get('fwd_lqd', 0)) + 0.2 * float(row.get('fwd_shy', 0))
            pos = 'risk_on'
        elif prob < 0.35:
            period_ret = 0.6 * float(row.get('fwd_tlt', 0)) + 0.2 * float(row.get('fwd_lqd', 0)) + 0.2 * float(row.get('fwd_shy', 0))
            pos = 'defensive'
        else:
            period_ret = 0.4 * float(row.get('fwd_lqd', 0)) + 0.3 * float(row.get('fwd_hyg', 0)) + 0.3 * float(row.get('fwd_shy', 0))
            pos = 'balanced'

        period_returns.append((date, period_ret))
        positions.append(pos)
    
    # Non-overlapping period returns (each is a 21-day return)
    ret_series = pd.Series([r[1] for r in period_returns], index=[r[0] for r in period_returns])
    return ret_series, positions

def compute_metrics(returns, name="", periods_per_year=12):
    """Compute metrics. periods_per_year=12 for monthly (21-day) returns, 252 for daily."""
    if returns is None or len(returns) < 10:
        return None
    ret = returns.values
    if np.std(ret) == 0:
        return None
    sharpe = np.mean(ret) / np.std(ret) * np.sqrt(periods_per_year)
    neg = ret[ret < 0]
    sortino = np.mean(ret) / np.std(neg) * np.sqrt(periods_per_year) if len(neg) > 0 and np.std(neg) > 0 else 0
    cum = (1 + pd.Series(ret)).cumprod()
    dd = cum / cum.cummax() - 1
    max_dd = dd.min() * 100
    years = len(ret) / periods_per_year
    cagr = (cum.iloc[-1] ** (1/years) - 1) * 100 if years > 0 and cum.iloc[-1] > 0 else 0
    total_ret = (cum.iloc[-1] - 1) * 100
    wins = np.sum(ret > 0)
    trades = np.sum(ret != 0)
    wr = wins / trades * 100 if trades > 0 else 0
    pf = abs(ret[ret > 0].sum() / ret[ret < 0].sum()) if len(neg) > 0 and ret[ret < 0].sum() != 0 else 0
    return {
        'name': name, 'sharpe': round(float(sharpe), 3), 'sortino': round(float(sortino), 3),
        'cagr': round(float(cagr), 1), 'max_dd': round(float(max_dd), 1),
        'total_return': round(float(total_ret), 1), 'profit_factor': round(float(pf), 3),
        'win_rate': round(float(wr), 1), 'n_days': len(ret), 'years': round(years, 1)
    }

def run_adversarial(returns, features, close, name="Credit"):
    """Full adversarial validation."""
    print(f"\n  ADVERSARIAL: {name}")
    print(f"  {'-'*60}")
    results = {}
    real = compute_metrics(returns, name)
    if not real:
        return {'summary': {'gates_passed': 0, 'total': 4, 'verdict': 'FAIL'}}
    real_sharpe = real['sharpe']
    
    # 1. Permutation (signal shuffle)
    print(f"    [1/4] Permutation test ({N_PERM} iters)...")
    perm_sharpes = []
    for trial in range(N_PERM):
        perm_ret, _ = run_strategy(features, close, shuffle_signals=True)
        m = compute_metrics(perm_ret)
        if m:
            perm_sharpes.append(m['sharpe'])
        if (trial + 1) % 25 == 0:
            print(f"      {trial+1}/{N_PERM}...")
    
    perm_p = np.mean([s >= real_sharpe for s in perm_sharpes]) if perm_sharpes else 1.0
    perm_pass = perm_p < 0.05
    print(f"      Real={real_sharpe:.3f}, Perm={np.mean(perm_sharpes):.3f}±{np.std(perm_sharpes):.3f}, p={perm_p:.3f} -> {'PASS' if perm_pass else 'FAIL'}")
    results['permutation'] = {'p_value': round(float(perm_p), 4), 'pass': bool(perm_pass),
                              'real': real_sharpe, 'perm_mean': round(float(np.mean(perm_sharpes)), 3)}
    
    # 2. Sub-period
    print(f"    [2/4] Sub-period consistency...")
    n = len(returns)
    block_sz = n // 4
    block_sharpes = []
    for b in range(4):
        start = b * block_sz
        end = (b+1)*block_sz if b < 3 else n
        m = compute_metrics(returns.iloc[start:end])
        if m:
            block_sharpes.append(m['sharpe'])
    cv = np.std(block_sharpes) / abs(np.mean(block_sharpes)) if block_sharpes and np.mean(block_sharpes) != 0 else 999
    sub_pass = cv < 0.50
    print(f"      Sharpes: {[f'{s:.2f}' for s in block_sharpes]}, CV={cv:.3f} -> {'PASS' if sub_pass else 'FAIL'}")
    results['sub_period'] = {'cv': round(float(cv), 3), 'pass': bool(sub_pass)}
    
    # 3. Outlier
    print(f"    [3/4] Outlier robustness...")
    ret_vals = returns.values
    nonzero = ret_vals[ret_vals != 0]
    if len(nonzero) > 10:
        p5, p95 = np.percentile(nonzero, [5, 95])
        trimmed = ret_vals.copy()
        trimmed[(trimmed < p5) | (trimmed > p95)] = 0
        m_trim = compute_metrics(pd.Series(trimmed))
        if m_trim and real_sharpe != 0:
            deg = (m_trim['sharpe'] - real_sharpe) / abs(real_sharpe)
            out_pass = deg > -0.50
            print(f"      Full={real_sharpe:.3f}, Trimmed={m_trim['sharpe']:.3f}, deg={deg:.3f} -> {'PASS' if out_pass else 'FAIL'}")
            results['outlier'] = {'degradation': round(float(deg), 3), 'pass': bool(out_pass)}
        else:
            results['outlier'] = {'pass': False}
    else:
        results['outlier'] = {'pass': False}
    
    # 4. R1 regime
    print(f"    [4/4] R1 regime test...")
    if 'SPY' in close.columns:
        spy_monthly = close['SPY'].resample('ME').last().pct_change()
        green_months = set(spy_monthly[spy_monthly > 0].index.to_period('M'))
        red_months = set(spy_monthly[spy_monthly <= 0].index.to_period('M'))
        
        green_mask = returns.index.to_period('M').isin(green_months)
        red_mask = returns.index.to_period('M').isin(red_months)
        
        m_green = compute_metrics(returns[green_mask]) if green_mask.sum() > 50 else None
        m_red = compute_metrics(returns[red_mask]) if red_mask.sum() > 50 else None
        
        if m_green and m_red and max(abs(m_green['sharpe']), abs(m_red['sharpe'])) > 0:
            gap = abs(m_green['sharpe'] - m_red['sharpe']) / max(abs(m_green['sharpe']), abs(m_red['sharpe']))
            r1_pass = gap < 0.50
            print(f"      Green={m_green['sharpe']:.3f}, Red={m_red['sharpe']:.3f}, gap={gap:.3f} -> {'PASS' if r1_pass else 'FAIL'}")
            results['r1_regime'] = {'gap': round(float(gap), 3), 'pass': bool(r1_pass),
                                    'green': m_green['sharpe'], 'red': m_red['sharpe']}
        else:
            results['r1_regime'] = {'pass': False}
    else:
        results['r1_regime'] = {'pass': False}
    
    gates = sum(1 for k in ['permutation','sub_period','outlier','r1_regime'] if results.get(k, {}).get('pass', False))
    results['summary'] = {'gates_passed': gates, 'total': 4, 'verdict': 'PASS' if gates >= 3 else 'FAIL'}
    print(f"      VERDICT: {gates}/4 -> {results['summary']['verdict']}")
    return results

# ============================================================
print("=" * 70)
print("ML CREDIT SPREAD TIMING STRATEGY")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# Step 1
close = download_data()

# Step 2: Features
print("\nBuilding features...")
features = build_features(close)
print(f"  {len(features)} observations, {len([c for c in features.columns if c not in ['target','fwd_hyg','fwd_shy','fwd_lqd','fwd_tlt']])} features")
print(f"  Target rate (HYG > SHY 21d): {features['target'].mean():.1%}")

# Step 3: Baselines
print("\n" + "=" * 70)
print("BASELINES")
print("=" * 70)

# Buy and hold each
for ticker in ['HYG', 'LQD', 'TLT', 'SHY', 'AGG']:
    if ticker in close.columns:
        ret = close[ticker].pct_change().dropna()
        ret = ret[ret.index >= features.index[TRAIN_WINDOW]]
        m = compute_metrics(ret, f"{ticker} B&H", periods_per_year=252)
        if m:
            print(f"  {m['name']:15s} Sharpe={m['sharpe']:6.3f}  CAGR={m['cagr']:5.1f}%  MaxDD={m['max_dd']:5.1f}%")

# Step 4: ML Strategy
print("\n" + "=" * 70)
print("ML CREDIT TIMING")
print("=" * 70)

print("\n  Training walk-forward...")
ret_ml, positions = run_strategy(features, close)
m_ml = compute_metrics(ret_ml, "ML Credit Timing")
if m_ml:
    print(f"  {m_ml}")
    
    # Position distribution
    from collections import Counter
    pos_counts = Counter(positions)
    total = sum(pos_counts.values())
    print(f"\n  Position distribution:")
    for pos, count in sorted(pos_counts.items()):
        print(f"    {pos:15s}: {count:5d} ({count/total*100:.1f}%)")

# Step 5: Adversarial
print("\n" + "=" * 70)
print("ADVERSARIAL VALIDATION")
print("=" * 70)

if m_ml:
    adv = run_adversarial(ret_ml, features, close, "ML Credit Timing")

# Summary
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
if m_ml:
    print(f"  ML Credit Timing: Sharpe={m_ml['sharpe']:.3f}, Sortino={m_ml['sortino']:.3f}, CAGR={m_ml['cagr']:.1f}%, MaxDD={m_ml['max_dd']:.1f}%")
    gates = adv.get('summary', {}).get('gates_passed', 0)
    total = adv.get('summary', {}).get('total', 4)
    print(f"  Adversarial: {gates}/{total} gates")

# Save
output = {
    'strategy': 'ML Credit Spread Timing',
    'timestamp': datetime.now().isoformat(),
    'metrics': m_ml,
    'adversarial': adv if m_ml else None,
}
with open(os.path.join(OUTPUT, 'results.json'), 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nRuntime: {datetime.now().strftime('%H:%M:%S')}")
print(f"Output: {OUTPUT}")
