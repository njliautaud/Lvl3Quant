#!/usr/bin/env python3
"""
ML Credit Spread Timing — FIX REGIME DEPENDENCY
=================================================
Our credit timing strategy passed 2/4 gates (perm PASS, outlier PASS)
but FAILED R1 (green Sharpe 2.11 vs red 0.33 — regime-dependent).

This script attempts to fix the regime dependency by:
1. Adding regime-aware features (VIX level, credit spreads, yield curve)
2. Testing regime-conditional position sizing
3. Testing a "red market only" credit rotation variant
4. Adding tail-risk overlay (reduce when VIX spikes)

Goal: Get R1 gap below 0.50 threshold while maintaining Sharpe > 1.0
"""
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from datetime import datetime
import os, json, warnings
warnings.filterwarnings('ignore')

OUTPUT = '/home/jupiter/Lvl3Quant/output/credit_timing_v2'
os.makedirs(OUTPUT, exist_ok=True)
np.random.seed(42)

print("=" * 70)
print("ML CREDIT TIMING v2 — REGIME-AGNOSTIC FIX")
print("Target: reduce regime dependency while keeping Sharpe > 1.0")
print("=" * 70)

# Download data
tickers = ['HYG', 'LQD', 'TLT', 'IEF', 'SPY', 'AGG', 'TIPS', 'SHY',
           'EMB', 'JNK']
print(f"\nDownloading {len(tickers)} assets + VIX...")
data = yf.download(tickers + ['^VIX'], start='2007-01-01', progress=False)
if hasattr(data.index, 'tz') and data.index.tz is not None:
    data.index = data.index.tz_localize(None)

close = data['Close'] if isinstance(data.columns, pd.MultiIndex) else data
close = close.ffill().dropna(how='all')

# Separate VIX
vix = close['^VIX'] if '^VIX' in close.columns else None
close = close.drop('^VIX', axis=1, errors='ignore')
close = close.dropna()

print(f"  {len(close)} days, {close.index[0].date()} to {close.index[-1].date()}")

# ============================================================
# ENHANCED FEATURE ENGINEERING
# ============================================================
print("\nBuilding enhanced features (regime-aware)...")

spy_ret = close['SPY'].pct_change()
hyg_ret = close['HYG'].pct_change()
tlt_ret = close['TLT'].pct_change()

features = pd.DataFrame(index=close.index)

# Original features: relative momentum
for asset in ['HYG', 'LQD', 'TLT', 'IEF', 'AGG']:
    if asset not in close.columns:
        continue
    ret = close[asset].pct_change()
    for h in [5, 10, 21, 42, 63]:
        features[f'{asset}_mom_{h}d'] = close[asset].pct_change(h)
    # Relative to SPY
    features[f'{asset}_vs_spy_21d'] = close[asset].pct_change(21) - close['SPY'].pct_change(21)
    # Volatility
    features[f'{asset}_vol_21d'] = ret.rolling(21).std() * np.sqrt(252)

# Credit spread proxy: HYG-TLT spread
if 'HYG' in close.columns and 'TLT' in close.columns:
    spread = close['HYG'].pct_change(21) - close['TLT'].pct_change(21)
    features['credit_spread_21d'] = spread
    features['credit_spread_63d'] = close['HYG'].pct_change(63) - close['TLT'].pct_change(63)
    features['credit_spread_change'] = spread - spread.shift(21)

# VIX features (KEY for regime awareness)
if vix is not None:
    features['vix'] = vix
    features['vix_21d_ma'] = vix.rolling(21).mean()
    features['vix_vs_ma'] = vix / features['vix_21d_ma']
    features['vix_change_5d'] = vix.pct_change(5)
    features['vix_change_21d'] = vix.pct_change(21)
    features['vix_percentile'] = vix.rolling(252).rank(pct=True)
    # VIX regime bucket
    features['high_vix'] = (vix > 25).astype(float)

# Yield curve proxy (TLT vs IEF)
if 'TLT' in close.columns and 'IEF' in close.columns:
    features['curve_slope'] = close['TLT'].pct_change(63) - close['IEF'].pct_change(63)

# SPY regime features
features['spy_21d_ret'] = close['SPY'].pct_change(21)
features['spy_63d_ret'] = close['SPY'].pct_change(63)
features['spy_vol_21d'] = spy_ret.rolling(21).std() * np.sqrt(252)
features['spy_dd'] = close['SPY'] / close['SPY'].rolling(252).max() - 1

# Cross-asset correlations (rolling)
features['hyg_spy_corr_63d'] = hyg_ret.rolling(63).corr(spy_ret)
features['tlt_spy_corr_63d'] = tlt_ret.rolling(63).corr(spy_ret)

# ============================================================
# MULTIPLE TARGET VARIANTS
# ============================================================
REBAL_PERIOD = 21  # Non-overlapping 21-day periods

# Build non-overlapping periods
dates = close.index
period_starts = list(range(0, len(dates) - REBAL_PERIOD, REBAL_PERIOD))

print(f"  {len(features.columns)} features, {len(period_starts)} rebalancing periods")

# Target variants
targets = {}

# V1: Best absolute return (HYG vs TLT vs LQD)
# V2: Best risk-adjusted return (Sharpe over the period)
# V3: Regime-conditional (different target in different VIX regimes)

assets_to_allocate = ['HYG', 'TLT', 'LQD']
available_assets = [a for a in assets_to_allocate if a in close.columns]

# For each period, calculate forward returns
period_data = []
for i, start in enumerate(period_starts):
    end = min(start + REBAL_PERIOD, len(dates) - 1)
    period_end = min(end + REBAL_PERIOD, len(dates) - 1)

    if period_end <= end:
        break

    # Features at decision point
    feat_row = features.iloc[end] if end < len(features) else None
    if feat_row is None or feat_row.isna().sum() > len(feat_row) * 0.3:
        continue

    # Forward returns for each asset
    fwd_rets = {}
    for asset in available_assets:
        fwd_ret = (close[asset].iloc[period_end] / close[asset].iloc[end]) - 1
        fwd_rets[asset] = fwd_ret

    # Best asset (highest return)
    best_asset = max(fwd_rets, key=fwd_rets.get)
    best_ret = fwd_rets[best_asset]

    # Regime
    vix_val = vix.iloc[end] if vix is not None else 15
    spy_ret_21d = close['SPY'].pct_change(21).iloc[end]
    is_red = spy_ret_21d < -0.02  # Red regime: SPY down >2% over 21d

    period_data.append({
        'date': dates[end],
        'best_asset': best_asset,
        'best_ret': best_ret,
        'is_red': is_red,
        'vix': vix_val,
        **{f'{a}_ret': fwd_rets[a] for a in available_assets},
        'feat_idx': end
    })

period_df = pd.DataFrame(period_data)
print(f"  Valid periods: {len(period_df)}")
print(f"  Red regime periods: {period_df['is_red'].sum()} ({period_df['is_red'].mean():.1%})")
print(f"  Best asset distribution: {period_df['best_asset'].value_counts().to_dict()}")

# ============================================================
# WALK-FORWARD WITH REGIME-AWARE MODEL
# ============================================================
TRAIN_PERIODS = 24  # ~2 years of monthly periods

print(f"\n{'=' * 70}")
print("WALK-FORWARD TRAINING — REGIME-AWARE MODEL")
print(f"{'=' * 70}")

# Encode target
asset_map = {a: i for i, a in enumerate(available_assets)}
period_df['target'] = period_df['best_asset'].map(asset_map)

predictions = []
actuals = []
pred_dates = []
pred_regimes = []

for i in range(TRAIN_PERIODS, len(period_df)):
    train_slice = period_df.iloc[i-TRAIN_PERIODS:i]
    test_row = period_df.iloc[i]

    # Get features for training periods
    train_feats = features.iloc[[r['feat_idx'] for _, r in train_slice.iterrows()]]
    test_feat = features.iloc[test_row['feat_idx']:test_row['feat_idx']+1]

    # Clean
    valid_cols = train_feats.columns[train_feats.notna().all()]
    if len(valid_cols) < 10:
        continue

    X_train = train_feats[valid_cols].values
    y_train = train_slice['target'].values
    X_test = test_feat[valid_cols].values

    if np.isnan(X_train).any() or np.isnan(X_test).any():
        continue

    # Train GBM
    model = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        subsample=0.8, random_state=42
    )

    try:
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[0]

        # Regime-aware position sizing:
        # In high-VIX / red regime, upweight TLT probability
        vix_val = test_row['vix']
        if vix_val > 25:
            # Boost TLT probability in high-vol
            tlt_idx = asset_map.get('TLT', -1)
            if tlt_idx >= 0 and tlt_idx < len(probs):
                probs[tlt_idx] *= 1.5
                probs = probs / probs.sum()

        best_idx = np.argmax(probs)
        pred_asset = available_assets[best_idx]

        predictions.append(pred_asset)
        actuals.append(test_row['best_asset'])
        pred_dates.append(test_row['date'])
        pred_regimes.append('red' if test_row['is_red'] else 'green')
    except:
        continue

predictions = np.array(predictions)
actuals = np.array(actuals)
pred_dates = pd.DatetimeIndex(pred_dates)
pred_regimes = np.array(pred_regimes)

# ============================================================
# BACKTEST
# ============================================================
print(f"\n{'=' * 70}")
print("BACKTEST RESULTS")
print(f"{'=' * 70}")

# Get returns for predicted asset each period
strat_returns = []
bh_hyg_returns = []
bh_tlt_returns = []

for i, (pred, date) in enumerate(zip(predictions, pred_dates)):
    # Find this period's returns
    mask = period_df['date'] == date
    if mask.sum() == 0:
        continue
    row = period_df[mask].iloc[0]
    strat_returns.append(row[f'{pred}_ret'])
    bh_hyg_returns.append(row['HYG_ret'])
    bh_tlt_returns.append(row['TLT_ret'])

strat_returns = np.array(strat_returns)
bh_hyg_returns = np.array(bh_hyg_returns)
bh_tlt_returns = np.array(bh_tlt_returns)

def calc_metrics(rets, name):
    cum = np.cumprod(1 + rets)
    total = cum[-1] - 1
    n_periods = len(rets)
    years = n_periods * REBAL_PERIOD / 252
    cagr = (1 + total) ** (1/years) - 1 if years > 0 else 0
    vol = np.std(rets) * np.sqrt(252/REBAL_PERIOD)
    sharpe = cagr / vol if vol > 0 else 0

    # Max DD
    peak = np.maximum.accumulate(cum)
    dd = cum / peak - 1
    max_dd = dd.min()

    # Win rate
    wr = (rets > 0).mean()

    # Sortino
    downside = np.std(rets[rets < 0]) * np.sqrt(252/REBAL_PERIOD) if (rets < 0).sum() > 0 else 1e-8
    sortino = cagr / downside

    return {'name': name, 'sharpe': sharpe, 'sortino': sortino, 'cagr': cagr,
            'vol': vol, 'max_dd': max_dd, 'wr': wr, 'n_trades': n_periods}

m_strat = calc_metrics(strat_returns, 'ML Credit v2 (regime-aware)')
m_hyg = calc_metrics(bh_hyg_returns, 'Buy & Hold HYG')
m_tlt = calc_metrics(bh_tlt_returns, 'Buy & Hold TLT')

print(f"\n{'Strategy':<30} {'Sharpe':>7} {'Sortino':>8} {'CAGR':>7} {'MaxDD':>7} {'WR':>5}")
print("-" * 70)
for m in [m_strat, m_hyg, m_tlt]:
    print(f"{m['name']:<30} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} "
          f"{m['cagr']:>6.1%} {m['max_dd']:>6.1%} {m['wr']:>4.0%}")

# ============================================================
# R1 REGIME TEST — THE KEY METRIC
# ============================================================
print(f"\n{'=' * 70}")
print("R1 REGIME-AGNOSTIC TEST")
print(f"{'=' * 70}")

green_mask = pred_regimes == 'green'
red_mask = pred_regimes == 'red'

green_rets = strat_returns[green_mask[:len(strat_returns)]] if green_mask[:len(strat_returns)].sum() > 5 else np.array([0])
red_rets = strat_returns[red_mask[:len(strat_returns)]] if red_mask[:len(strat_returns)].sum() > 5 else np.array([0])

green_sharpe = np.mean(green_rets) / (np.std(green_rets) + 1e-8) * np.sqrt(252/REBAL_PERIOD) if len(green_rets) > 1 else 0
red_sharpe = np.mean(red_rets) / (np.std(red_rets) + 1e-8) * np.sqrt(252/REBAL_PERIOD) if len(red_rets) > 1 else 0

regime_gap = abs(green_sharpe - red_sharpe) / max(abs(green_sharpe), abs(red_sharpe), 0.01)

print(f"\n  Green regime: {green_mask[:len(strat_returns)].sum()} periods, Sharpe={green_sharpe:.3f}")
print(f"  Red regime:   {red_mask[:len(strat_returns)].sum()} periods, Sharpe={red_sharpe:.3f}")
print(f"  Regime gap:   {regime_gap:.3f} (threshold: 0.50)")

if regime_gap < 0.50:
    print(f"  ✅ R1 PASS — strategy works in both regimes")
else:
    print(f"  ❌ R1 FAIL — still regime-dependent (gap={regime_gap:.2f})")

# ============================================================
# PERMUTATION TEST
# ============================================================
print(f"\nRunning permutation test (50 shuffles)...")
perm_sharpes = []
for p in range(50):
    shuffled = np.random.permutation(strat_returns)
    cum = np.cumprod(1 + shuffled)
    total = cum[-1] - 1
    years = len(shuffled) * REBAL_PERIOD / 252
    cagr = (1 + total) ** (1/years) - 1 if years > 0 else 0
    vol = np.std(shuffled) * np.sqrt(252/REBAL_PERIOD)
    perm_sharpes.append(cagr / vol if vol > 0 else 0)

perm_p = np.mean([ps >= m_strat['sharpe'] for ps in perm_sharpes])
print(f"  Permutation p-value: {perm_p:.3f} (real Sharpe {m_strat['sharpe']:.3f} vs perm mean {np.mean(perm_sharpes):.3f})")
if perm_p < 0.05:
    print(f"  ✅ PERM PASS")
else:
    print(f"  ❌ PERM FAIL")

# ============================================================
# VERDICT
# ============================================================
print(f"\n{'=' * 70}")
print("VERDICT")
print(f"{'=' * 70}")

gates_passed = 0
if perm_p < 0.05: gates_passed += 1
if regime_gap < 0.50: gates_passed += 1
if m_strat['sharpe'] > 1.0: gates_passed += 1

print(f"  Gates passed: {gates_passed}/3 (perm, R1, Sharpe>1)")
if gates_passed >= 3:
    print("  ✅ CREDIT TIMING v2 — REGIME-AGNOSTIC, DEPLOYABLE")
elif gates_passed >= 2:
    print("  🟡 PARTIAL — improved but still needs work")
else:
    print("  ❌ STILL FAILING — regime dependency persists")

# Save
results = {
    'sharpe': float(m_strat['sharpe']),
    'sortino': float(m_strat['sortino']),
    'cagr': float(m_strat['cagr']),
    'max_dd': float(m_strat['max_dd']),
    'wr': float(m_strat['wr']),
    'regime_gap': float(regime_gap),
    'perm_p': float(perm_p),
    'green_sharpe': float(green_sharpe),
    'red_sharpe': float(red_sharpe),
    'gates_passed': gates_passed,
    'n_trades': int(m_strat['n_trades'])
}
with open(f'{OUTPUT}/results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nDONE")
