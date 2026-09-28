#!/usr/bin/env python3
"""
Vol Harvesting Strategy v2 — Enhanced with ML + richer features
Walk-forward sliding window (HC #0), commission-free (HC #694)
Regime-agnostic validation (HC #428 R1)
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from scipy.stats import percentileofscore
import json
import os
from datetime import datetime, timedelta

OUT_DIR = '/home/jupiter/Lvl3Quant/output/vol_harvest_v2'
os.makedirs(OUT_DIR, exist_ok=True)

###############################################################################
# 1. DATA DOWNLOAD
###############################################################################
print("=" * 80)
print("VOL HARVEST V2 — Enhanced Strategy Research")
print("=" * 80)

tickers = {
    'SVXY': 'SVXY',       # -0.5x VIX short-term futures
    '^VIX': '^VIX',       # VIX index
    '^VVIX': '^VVIX',     # VIX of VIX
    '^VIX3M': '^VIX3M',   # 3-month VIX (for term structure)
    '^VIX6M': '^VIX6M',   # 6-month VIX (for deeper term structure)
    'SPY': 'SPY',         # S&P 500 for regime classification
}

START = '2018-01-01'
END = '2026-07-11'

print("\n[1] Downloading data...")
data = {}
for name, ticker in tickers.items():
    try:
        df = yf.download(ticker, start=START, end=END, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        data[name] = df
        print(f"  {name}: {len(df)} rows ({df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')})")
    except Exception as e:
        print(f"  {name}: FAILED - {e}")

# Also get put-call ratio proxy via CBOE VIX term structure
# We'll use VIX/VIX3M ratio as a proxy since direct P/C data isn't free

###############################################################################
# 2. FEATURE ENGINEERING
###############################################################################
print("\n[2] Engineering features...")

# Build unified daily DataFrame
df = pd.DataFrame(index=data['SVXY'].index)
df['svxy_close'] = data['SVXY']['Close']
df['svxy_ret'] = df['svxy_close'].pct_change()
df['spy_close'] = data['SPY']['Close'].reindex(df.index)
df['spy_ret'] = df['spy_close'].pct_change()

# VIX features
df['vix'] = data['^VIX']['Close'].reindex(df.index)
df['vix3m'] = data['^VIX3M']['Close'].reindex(df.index) if '^VIX3M' in data else np.nan
df['vix6m'] = data['^VIX6M']['Close'].reindex(df.index) if '^VIX6M' in data else np.nan
df['vvix'] = data['^VVIX']['Close'].reindex(df.index) if '^VVIX' in data else np.nan

# Forward-fill VIX data (different trading hours)
for col in ['vix', 'vix3m', 'vix6m', 'vvix']:
    df[col] = df[col].ffill()

# ---- FEATURE SET ----

# F1: Contango ratio (VIX3M / VIX) — basic term structure
df['contango_ratio'] = df['vix3m'] / df['vix']

# F2: Deep contango (VIX6M / VIX)
df['deep_contango'] = df['vix6m'] / df['vix']

# F3: Term structure slope (VIX3M - VIX)
df['ts_slope'] = df['vix3m'] - df['vix']

# F4: VIX level
df['vix_level'] = df['vix']

# F5: VIX percentile rank (rolling 252-day)
df['vix_pctile'] = df['vix'].rolling(252).apply(
    lambda x: percentileofscore(x, x.iloc[-1]) / 100.0, raw=False
)

# F6: Realized vol (20-day) vs implied vol (VIX)
df['rv_20d'] = df['spy_ret'].rolling(20).std() * np.sqrt(252) * 100
df['rv_vs_iv'] = df['rv_20d'] - df['vix']  # Vol risk premium

# F7: VIX mean reversion signal (VIX z-score vs 60d MA)
df['vix_ma60'] = df['vix'].rolling(60).mean()
df['vix_std60'] = df['vix'].rolling(60).std()
df['vix_zscore'] = (df['vix'] - df['vix_ma60']) / df['vix_std60']

# F8: VVIX level (vol of vol — uncertainty about uncertainty)
df['vvix_level'] = df['vvix']

# F9: VVIX percentile
df['vvix_pctile'] = df['vvix'].rolling(252).apply(
    lambda x: percentileofscore(x, x.iloc[-1]) / 100.0 if len(x.dropna()) > 10 else np.nan, raw=False
)

# F10: VIX momentum (5d change)
df['vix_mom5'] = df['vix'].pct_change(5)

# F11: VIX momentum (20d change)
df['vix_mom20'] = df['vix'].pct_change(20)

# F12: SPY momentum (20d)
df['spy_mom20'] = df['spy_ret'].rolling(20).sum()

# F13: SPY drawdown from 60d high
df['spy_dd60'] = df['spy_close'] / df['spy_close'].rolling(60).max() - 1

# F14: Contango change (momentum of term structure)
df['contango_mom5'] = df['contango_ratio'].pct_change(5)

# F15: SVXY momentum (own trend)
df['svxy_mom20'] = df['svxy_ret'].rolling(20).sum()

# Target: next-day SVXY return
df['target'] = df['svxy_ret'].shift(-1)

# Regime classification (SPY close-to-close)
df['spy_daily_ret'] = df['spy_ret']
df['regime'] = 'flat'
df.loc[df['spy_daily_ret'] > 0.002, 'regime'] = 'green'
df.loc[df['spy_daily_ret'] < -0.002, 'regime'] = 'red'

FEATURES = [
    'contango_ratio', 'deep_contango', 'ts_slope',
    'vix_level', 'vix_pctile', 'rv_vs_iv', 'vix_zscore',
    'vvix_level', 'vvix_pctile',
    'vix_mom5', 'vix_mom20',
    'spy_mom20', 'spy_dd60',
    'contango_mom5', 'svxy_mom20'
]

# Drop rows with NaN in features or target
df_clean = df.dropna(subset=FEATURES + ['target'])
print(f"  Clean dataset: {len(df_clean)} rows")
print(f"  Features: {len(FEATURES)}")
print(f"  Date range: {df_clean.index[0].strftime('%Y-%m-%d')} to {df_clean.index[-1].strftime('%Y-%m-%d')}")

###############################################################################
# 3. WALK-FORWARD SLIDING WINDOW — LGBM
###############################################################################
print("\n[3] Walk-forward sliding window LightGBM...")

TRAIN_DAYS = 504  # ~2 years
OOT_DAYS = 1      # predict 1 day ahead, slide
MIN_TRAIN = 252

results = []
predictions = []

X_all = df_clean[FEATURES].values
y_all = df_clean['target'].values
dates_all = df_clean.index
regimes_all = df_clean['regime'].values
svxy_close_all = df_clean['svxy_close'].values

lgb_params = {
    'objective': 'regression',
    'metric': 'mae',
    'learning_rate': 0.03,
    'num_leaves': 15,
    'max_depth': 4,
    'min_child_samples': 30,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42,
}

n = len(X_all)
oot_start = TRAIN_DAYS

print(f"  Total samples: {n}, OOT starts at idx {oot_start}")
print(f"  Expected OOT days: {n - oot_start}")

for i in range(oot_start, n):
    # Sliding window: train on last TRAIN_DAYS
    train_start = max(0, i - TRAIN_DAYS)
    X_train = X_all[train_start:i]
    y_train = y_all[train_start:i]
    X_test = X_all[i:i+1]
    y_test = y_all[i]

    train_ds = lgb.Dataset(X_train, label=y_train, free_raw_data=False)

    model = lgb.train(
        lgb_params,
        train_ds,
        num_boost_round=200,
    )

    pred = model.predict(X_test)[0]

    predictions.append({
        'date': dates_all[i],
        'pred': pred,
        'actual': y_test,
        'regime': regimes_all[i],
        'svxy_close': svxy_close_all[i],
    })

print(f"  Generated {len(predictions)} OOT predictions")

###############################################################################
# 4. STRATEGY CONSTRUCTION
###############################################################################
print("\n[4] Constructing strategies...")

pred_df = pd.DataFrame(predictions)
pred_df.set_index('date', inplace=True)

# Strategy 1: Threshold — long SVXY when pred > 0
pred_df['signal_basic'] = (pred_df['pred'] > 0).astype(int)

# Strategy 2: Confidence-weighted — scale position by prediction magnitude
pred_std = pred_df['pred'].std()
pred_df['signal_conf'] = np.clip(pred_df['pred'] / pred_std, -1, 1)
pred_df['signal_conf'] = pred_df['signal_conf'].clip(lower=0)  # long-only for SVXY

# Strategy 3: Regime-filtered — only trade when VIX zscore < 1 (avoid spikes)
vix_zscore_oot = df_clean['vix_zscore'].reindex(pred_df.index)
pred_df['signal_regime'] = pred_df['signal_basic'].copy()
pred_df.loc[vix_zscore_oot > 1.5, 'signal_regime'] = 0  # exit during VIX spikes

# Strategy 4: Kelly-sized
# Estimate Kelly fraction from rolling win rate and avg win/loss
KELLY_WINDOW = 60
pred_df['rolling_wr'] = pred_df['signal_basic'].rolling(KELLY_WINDOW).apply(
    lambda x: np.nan, raw=True  # placeholder
)
# Proper Kelly: use rolling stats of actual returns when signal=1
signal_rets = pred_df['actual'] * pred_df['signal_basic']
pred_df['roll_mean'] = signal_rets.rolling(KELLY_WINDOW).mean()
pred_df['roll_std'] = signal_rets.rolling(KELLY_WINDOW).std()
pred_df['kelly_f'] = (pred_df['roll_mean'] / (pred_df['roll_std'] ** 2)).clip(0, 1)
pred_df['kelly_f'] = pred_df['kelly_f'].fillna(0.5)
pred_df['signal_kelly'] = pred_df['signal_basic'] * pred_df['kelly_f']

# Buy & hold benchmark
pred_df['bh_ret'] = pred_df['actual']

# Strategy returns (commission-free per HC #694)
strategies = {
    'buy_hold': pred_df['actual'],
    'lgbm_basic': pred_df['actual'] * pred_df['signal_basic'],
    'lgbm_conf': pred_df['actual'] * pred_df['signal_conf'],
    'lgbm_regime': pred_df['actual'] * pred_df['signal_regime'],
    'lgbm_kelly': pred_df['actual'] * pred_df['signal_kelly'],
}

###############################################################################
# 5. ALSO TEST SIMPLE THRESHOLD (V1 BASELINE)
###############################################################################
print("\n[5] V1 baseline (simple contango threshold)...")

contango_oot = df_clean['contango_ratio'].reindex(pred_df.index)
pred_df['v1_signal'] = (contango_oot > 1.0).astype(int)
strategies['v1_threshold'] = pred_df['actual'] * pred_df['v1_signal']

###############################################################################
# 6. PERFORMANCE ANALYSIS
###############################################################################
print("\n[6] Performance analysis...")

def calc_metrics(returns, name):
    """Calculate comprehensive performance metrics."""
    rets = returns.dropna()
    if len(rets) == 0:
        return {}

    total_ret = (1 + rets).prod() - 1
    n_years = len(rets) / 252
    cagr = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0

    ann_vol = rets.std() * np.sqrt(252)
    sharpe = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0

    downside = rets[rets < 0].std() * np.sqrt(252)
    sortino = (rets.mean() * 252) / downside if downside > 0 else 0

    # Max drawdown
    cum = (1 + rets).cumprod()
    dd = cum / cum.cummax() - 1
    max_dd = dd.min()

    # Win rate (days)
    wr = (rets > 0).mean()

    # Profit factor
    gross_profit = rets[rets > 0].sum()
    gross_loss = abs(rets[rets < 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else np.inf

    # Exposure (fraction of days in market)
    if name != 'buy_hold':
        exposure = (rets != 0).mean()
    else:
        exposure = 1.0

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    return {
        'name': name,
        'total_ret': total_ret,
        'cagr': cagr,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_dd': max_dd,
        'calmar': calmar,
        'win_rate': wr,
        'profit_factor': pf,
        'exposure': exposure,
        'n_days': len(rets),
    }

metrics = []
for name, rets in strategies.items():
    m = calc_metrics(rets, name)
    metrics.append(m)

metrics_df = pd.DataFrame(metrics).set_index('name')
print("\n  STRATEGY COMPARISON (OOT Walk-Forward):")
print("  " + "=" * 100)
for _, row in metrics_df.iterrows():
    print(f"  {row.name:20s} | CAGR={row['cagr']:+.1%} | Sharpe={row['sharpe']:.2f} | "
          f"Sortino={row['sortino']:.2f} | MaxDD={row['max_dd']:.1%} | "
          f"WR={row['win_rate']:.1%} | PF={row['profit_factor']:.2f} | "
          f"Exposure={row['exposure']:.1%}")

###############################################################################
# 7. REGIME ANALYSIS (HC #428 R1)
###############################################################################
print("\n[7] Regime analysis (HC #428 R1)...")

best_strat = 'lgbm_regime'  # we'll analyze the best strategy
best_rets = strategies[best_strat]

regime_metrics = {}
for regime in ['green', 'red', 'flat']:
    mask = pred_df['regime'] == regime
    regime_rets = best_rets[mask]
    if len(regime_rets) > 20:
        m = calc_metrics(regime_rets, f"{best_strat}_{regime}")
        regime_metrics[regime] = m
        print(f"  {regime:6s}: Sharpe={m['sharpe']:.2f}, WR={m['win_rate']:.1%}, "
              f"PF={m['profit_factor']:.2f}, N={m['n_days']}")

# Regime gap test
if 'green' in regime_metrics and 'red' in regime_metrics:
    s_green = regime_metrics['green']['sharpe']
    s_red = regime_metrics['red']['sharpe']
    regime_gap = abs(s_green - s_red) / max(abs(s_green), abs(s_red)) if max(abs(s_green), abs(s_red)) > 0 else 0
    regime_pass = regime_gap < 0.50
    print(f"\n  Regime gap: |{s_green:.2f} - {s_red:.2f}| / max = {regime_gap:.2f}")
    print(f"  HC #428 R1 regime test: {'PASS' if regime_pass else 'FAIL'} (threshold < 0.50)")

###############################################################################
# 8. OOT IC ANALYSIS
###############################################################################
print("\n[8] OOT IC analysis...")

# Overall IC
overall_ic = pred_df['pred'].corr(pred_df['actual'])
print(f"  Overall OOT IC: {overall_ic:.4f}")

# Rolling IC
ROLL_IC_WIN = 60
rolling_ic = pred_df['pred'].rolling(ROLL_IC_WIN).corr(pred_df['actual'])
print(f"  Rolling IC (60d): mean={rolling_ic.mean():.4f}, "
      f"median={rolling_ic.median():.4f}, "
      f"pct_positive={( rolling_ic > 0).mean():.1%}")

# IC by year
for year in sorted(pred_df.index.year.unique()):
    mask = pred_df.index.year == year
    if mask.sum() > 20:
        ic = pred_df.loc[mask, 'pred'].corr(pred_df.loc[mask, 'actual'])
        print(f"  IC {year}: {ic:.4f} (N={mask.sum()})")

###############################################################################
# 9. FEATURE IMPORTANCE
###############################################################################
print("\n[9] Feature importance (from last model)...")

importance = model.feature_importance(importance_type='gain')
feat_imp = sorted(zip(FEATURES, importance), key=lambda x: -x[1])
for fname, imp in feat_imp:
    print(f"  {fname:25s}: {imp:.0f}")

###############################################################################
# 10. FORWARD CAGR ESTIMATE (SVXY -0.5x adjustment)
###############################################################################
print("\n[10] Forward CAGR estimate...")

# SVXY changed from -1x to -0.5x in 2018
# Our data is already -0.5x era, so returns are realistic
# But we should note: historical VIX term structure may not persist

# Estimate: use OOT CAGR as base, apply confidence haircut
best_metrics = calc_metrics(strategies[best_strat], best_strat)
oot_cagr = best_metrics['cagr']

# Haircuts:
# 1. Overfitting risk: -20%
# 2. Regime uncertainty: -10%
# 3. SVXY tracking error: -5%
# 4. Transaction costs (even if commission-free, bid-ask on ETF): -2%
haircut = 0.63  # multiply by this factor
forward_cagr = oot_cagr * haircut

print(f"  OOT CAGR ({best_strat}): {oot_cagr:.1%}")
print(f"  Haircut factor: {haircut:.0%} (overfit -20%, regime -10%, tracking -5%, spread -2%)")
print(f"  Estimated forward CAGR: {forward_cagr:.1%}")
print(f"  Note: SVXY is -0.5x since 2018. All returns are -0.5x era.")

###############################################################################
# 11. POSITION SIZING ANALYSIS
###############################################################################
print("\n[11] Position sizing analysis...")

# Kelly criterion
signal_days = pred_df[pred_df['signal_basic'] == 1]['actual']
if len(signal_days) > 0:
    win_rate = (signal_days > 0).mean()
    avg_win = signal_days[signal_days > 0].mean() if (signal_days > 0).any() else 0
    avg_loss = abs(signal_days[signal_days < 0].mean()) if (signal_days < 0).any() else 1

    # Kelly: f* = (p * b - q) / b where b = avg_win/avg_loss
    b = avg_win / avg_loss if avg_loss > 0 else 1
    kelly_full = (win_rate * b - (1 - win_rate)) / b
    kelly_half = kelly_full / 2  # Half-Kelly for safety

    print(f"  Win rate (signal days): {win_rate:.1%}")
    print(f"  Avg win / Avg loss: {avg_win:.4f} / {avg_loss:.4f} = {b:.2f}")
    print(f"  Full Kelly fraction: {kelly_full:.1%}")
    print(f"  Half-Kelly (recommended): {kelly_half:.1%}")

###############################################################################
# 12. SAVE RESULTS
###############################################################################
print("\n[12] Saving results...")

# Save predictions
pred_df.to_csv(os.path.join(OUT_DIR, 'oot_predictions.csv'))

# Save metrics
metrics_df.to_csv(os.path.join(OUT_DIR, 'strategy_metrics.csv'))

# Save summary
summary = {
    'timestamp': datetime.now().isoformat(),
    'n_oot_days': len(pred_df),
    'date_range': f"{pred_df.index[0].strftime('%Y-%m-%d')} to {pred_df.index[-1].strftime('%Y-%m-%d')}",
    'features': FEATURES,
    'best_strategy': best_strat,
    'metrics': {name: {k: float(v) if isinstance(v, (np.floating, float)) else v
                       for k, v in m.items()}
                for name, m in zip(metrics_df.index, metrics_df.to_dict('records'))},
    'regime_analysis': {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                            for kk, vv in v.items()}
                       for k, v in regime_metrics.items()},
    'overall_ic': float(overall_ic),
    'forward_cagr_estimate': float(forward_cagr),
    'feature_importance': {f: float(i) for f, i in feat_imp},
}

with open(os.path.join(OUT_DIR, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2, default=str)

# Equity curves
eq_curves = pd.DataFrame({name: (1 + rets).cumprod() for name, rets in strategies.items()})
eq_curves.to_csv(os.path.join(OUT_DIR, 'equity_curves.csv'))

print(f"\n  All results saved to {OUT_DIR}/")
print("\n" + "=" * 80)
print("VOL HARVEST V2 — COMPLETE")
print("=" * 80)
