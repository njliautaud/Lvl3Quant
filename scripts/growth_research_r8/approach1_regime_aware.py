"""
Approach 1: Regime-AWARE Allocation
- Predict regime (bull/bear/sideways) using macro indicators
- Allocate aggressively in bull, defensively in bear
- Walk-forward: 252d train, 21d test, sliding
"""
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import os, json, warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

OUT = '/home/jupiter/Lvl3Quant/output/growth_research_r8'
os.makedirs(OUT, exist_ok=True)

# Download data
print("Downloading data...")
tickers = ['SPY', 'TQQQ', 'UPRO', 'TLT', 'GLD', 'SH', 'BIL', '^VIX']
# Also get yield curve proxy, credit spread proxy, breadth proxy
macro_tickers = ['^GSPC', '^VIX', '^TNX', '^TYX', 'HYG', 'LQD']

all_tickers = list(set(tickers + macro_tickers))
data = yf.download(all_tickers, start='2011-01-01', end='2026-07-01', auto_adjust=True, progress=False)
close = data['Close'].ffill().dropna(how='all')

# TQQQ/UPRO started 2010, should have data
print(f"Data range: {close.index[0]} to {close.index[-1]}, {len(close)} days")

# Define regime based on SPY
spy = close['^GSPC'].dropna()

def classify_regime(returns_fwd_21d):
    """Bull > 2%, Bear < -2%, Sideways in between"""
    if returns_fwd_21d > 0.02:
        return 2  # bull
    elif returns_fwd_21d < -0.02:
        return 0  # bear
    else:
        return 1  # sideways

# Build features
print("Building features...")
feat_df = pd.DataFrame(index=spy.index)

# VIX level
if '^VIX' in close.columns:
    feat_df['vix'] = close['^VIX']
    feat_df['vix_ma20'] = feat_df['vix'].rolling(20).mean()
    feat_df['vix_zscore'] = (feat_df['vix'] - feat_df['vix'].rolling(60).mean()) / feat_df['vix'].rolling(60).std()

# Yield curve proxy (10Y - 30Y is wrong, use 10Y level as proxy)
if '^TNX' in close.columns:
    feat_df['tnx'] = close['^TNX']
    feat_df['tnx_chg20'] = close['^TNX'].pct_change(20)

# Credit spread proxy: HYG/LQD ratio
if 'HYG' in close.columns and 'LQD' in close.columns:
    feat_df['credit_ratio'] = close['HYG'] / close['LQD']
    feat_df['credit_ratio_chg20'] = feat_df['credit_ratio'].pct_change(20)

# SPY momentum and breadth proxies
feat_df['spy_ret_21'] = spy.pct_change(21)
feat_df['spy_ret_63'] = spy.pct_change(63)
feat_df['spy_ret_252'] = spy.pct_change(252)
feat_df['spy_above_200ma'] = (spy > spy.rolling(200).mean()).astype(float)
feat_df['spy_above_50ma'] = (spy > spy.rolling(50).mean()).astype(float)
feat_df['spy_vol_21'] = spy.pct_change().rolling(21).std() * np.sqrt(252)
feat_df['spy_vol_63'] = spy.pct_change().rolling(63).std() * np.sqrt(252)
feat_df['spy_drawdown'] = spy / spy.rolling(252).max() - 1

# Target: forward 21-day SPY return regime
spy_fwd = spy.pct_change(21).shift(-21)
feat_df['regime'] = spy_fwd.apply(classify_regime)

feat_df = feat_df.dropna()
print(f"Feature matrix: {feat_df.shape}")

# Walk-forward regime prediction
feature_cols = [c for c in feat_df.columns if c != 'regime']
train_window = 252
test_window = 21

predictions = []
actuals = []
dates = []

for i in range(train_window, len(feat_df) - test_window, test_window):
    train = feat_df.iloc[i-train_window:i]
    test = feat_df.iloc[i:i+test_window]

    X_train = train[feature_cols].values
    y_train = train['regime'].values
    X_test = test[feature_cols].values
    y_test = test['regime'].values

    clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    predictions.extend(preds)
    actuals.extend(y_test)
    dates.extend(test.index[:len(preds)])

predictions = np.array(predictions)
actuals = np.array(actuals)
dates = pd.DatetimeIndex(dates)

accuracy = accuracy_score(actuals, predictions)
print(f"\nRegime prediction accuracy: {accuracy:.3f}")
print(f"  Bull accuracy: {accuracy_score(actuals[actuals==2], predictions[actuals==2]):.3f}")
print(f"  Bear accuracy: {accuracy_score(actuals[actuals==0], predictions[actuals==0]):.3f}")
print(f"  Sideways accuracy: {accuracy_score(actuals[actuals==1], predictions[actuals==1]):.3f}")

# Now simulate portfolio
# Allocations based on predicted regime:
# Bull: 100% TQQQ (3x leveraged Nasdaq)
# Bear: 50% TLT, 30% GLD, 20% BIL (defensive)
# Sideways: 60% SPY, 40% TLT

alloc_map = {
    2: {'TQQQ': 0.7, 'UPRO': 0.3},  # bull: leveraged equities
    0: {'TLT': 0.5, 'GLD': 0.3, 'BIL': 0.2},  # bear: defensive
    1: {'SPY': 0.6, 'TLT': 0.4},  # sideways: balanced
}

# Get daily returns for allocation assets
alloc_assets = ['TQQQ', 'UPRO', 'SPY', 'TLT', 'GLD', 'BIL', 'SH']
asset_returns = close[alloc_assets].pct_change().loc[dates]

# Portfolio returns
port_returns = pd.Series(0.0, index=dates)
current_regime = predictions[0]
regime_changes = pd.Series(predictions, index=dates)

for i, dt in enumerate(dates):
    pred_regime = predictions[i]
    alloc = alloc_map[pred_regime]
    day_ret = 0.0
    for asset, weight in alloc.items():
        if asset in asset_returns.columns and pd.notna(asset_returns.loc[dt, asset]):
            day_ret += weight * asset_returns.loc[dt, asset]
    port_returns.loc[dt] = day_ret

# Also compute buy-and-hold benchmarks
spy_returns = close['SPY'].pct_change().loc[dates]
tqqq_returns = close['TQQQ'].pct_change().loc[dates]

# Cumulative returns
port_cum = (1 + port_returns).cumprod()
spy_cum = (1 + spy_returns).cumprod()
tqqq_cum = (1 + tqqq_returns).cumprod()

# Metrics
def calc_metrics(returns, name):
    ann_ret = (1 + returns.mean()) ** 252 - 1
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    # Sortino
    downside = returns[returns < 0].std() * np.sqrt(252)
    sortino = ann_ret / downside if downside > 0 else 0

    # Max drawdown
    cum = (1 + returns).cumprod()
    dd = cum / cum.cummax() - 1
    max_dd = dd.min()

    # CAGR
    years = len(returns) / 252
    cagr = (cum.iloc[-1]) ** (1/years) - 1 if years > 0 else 0

    print(f"\n{name}:")
    print(f"  CAGR: {cagr*100:.1f}%")
    print(f"  Ann Vol: {ann_vol*100:.1f}%")
    print(f"  Sharpe: {sharpe:.2f}")
    print(f"  Sortino: {sortino:.2f}")
    print(f"  Max DD: {max_dd*100:.1f}%")
    print(f"  Final cumulative: {cum.iloc[-1]:.2f}x")

    return {'name': name, 'cagr': round(cagr*100,1), 'sharpe': round(sharpe,2),
            'sortino': round(sortino,2), 'max_dd': round(max_dd*100,1),
            'final_cum': round(float(cum.iloc[-1]),2)}

results = []
results.append(calc_metrics(port_returns.dropna(), "Regime-Aware Portfolio"))
results.append(calc_metrics(spy_returns.dropna(), "SPY Buy & Hold"))
results.append(calc_metrics(tqqq_returns.dropna(), "TQQQ Buy & Hold"))

# Regime gap diagnostic
bull_mask = (actuals == 2)
bear_mask = (actuals == 0)
bull_returns = port_returns.iloc[np.where(bull_mask)[0]].dropna()
bear_returns = port_returns.iloc[np.where(bear_mask)[0]].dropna()

if len(bull_returns) > 20 and len(bear_returns) > 20:
    bull_sharpe = (bull_returns.mean() * 252) / (bull_returns.std() * np.sqrt(252)) if bull_returns.std() > 0 else 0
    bear_sharpe = (bear_returns.mean() * 252) / (bear_returns.std() * np.sqrt(252)) if bear_returns.std() > 0 else 0
    regime_gap = abs(bull_sharpe - bear_sharpe) / max(abs(bull_sharpe), abs(bear_sharpe)) if max(abs(bull_sharpe), abs(bear_sharpe)) > 0 else 0
    print(f"\nRegime Gap Diagnostic:")
    print(f"  Bull Sharpe: {bull_sharpe:.2f}")
    print(f"  Bear Sharpe: {bear_sharpe:.2f}")
    print(f"  Regime Gap: {regime_gap:.2f} {'(WARN: >0.5)' if regime_gap > 0.5 else '(OK)'}")
    results[0]['regime_gap'] = round(regime_gap, 2)
    results[0]['bull_sharpe'] = round(bull_sharpe, 2)
    results[0]['bear_sharpe'] = round(bear_sharpe, 2)

# Permutation test
print("\nRunning permutation test (100 shuffles)...")
perm_cagrs = []
for _ in range(100):
    shuffled_preds = np.random.permutation(predictions)
    perm_ret = pd.Series(0.0, index=dates)
    for i, dt in enumerate(dates):
        alloc = alloc_map[shuffled_preds[i]]
        day_ret = 0.0
        for asset, weight in alloc.items():
            if asset in asset_returns.columns and pd.notna(asset_returns.loc[dt, asset]):
                day_ret += weight * asset_returns.loc[dt, asset]
        perm_ret.loc[dt] = day_ret
    perm_cum = (1 + perm_ret.dropna()).cumprod()
    years = len(perm_ret.dropna()) / 252
    perm_cagr = (perm_cum.iloc[-1]) ** (1/years) - 1
    perm_cagrs.append(perm_cagr * 100)

actual_cagr = results[0]['cagr']
perm_p_value = np.mean([c >= actual_cagr for c in perm_cagrs])
print(f"  Actual CAGR: {actual_cagr:.1f}%")
print(f"  Permutation median CAGR: {np.median(perm_cagrs):.1f}%")
print(f"  Permutation p-value: {perm_p_value:.3f}")
results[0]['perm_p_value'] = round(perm_p_value, 3)
results[0]['perm_median_cagr'] = round(np.median(perm_cagrs), 1)

# Accuracy details
results[0]['regime_accuracy'] = round(accuracy, 3)

# Save
with open(f'{OUT}/approach1_regime_aware.json', 'w') as f:
    json.dump({'results': results, 'regime_accuracy': round(accuracy, 3),
               'n_periods': len(predictions), 'date_range': f'{dates[0]} to {dates[-1]}'}, f, indent=2)

print(f"\nResults saved to {OUT}/approach1_regime_aware.json")
print("\n=== APPROACH 1 COMPLETE ===")
