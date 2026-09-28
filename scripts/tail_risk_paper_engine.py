#!/usr/bin/env python3
"""
ML Tail Risk Hedging — Paper Engine
=====================================
Generates daily signal: HEDGE (tail-risk ETFs), GROWTH (equity), or BALANCED.
Uses TAIL/BTAL for hedging, QQQ/SPY for growth, TLT/SHY for safety.
PM2 cron at 4:45pm ET weekdays.
"""

import numpy as np
import pandas as pd
import warnings
import json
from pathlib import Path
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')
import yfinance as yf
import lightgbm as lgb

OUTPUT = Path("/home/jupiter/Lvl3Quant/output/ml_tail_risk_paper")
OUTPUT.mkdir(parents=True, exist_ok=True)

STATE_FILE = OUTPUT / "state.json"
SIGNAL_LOG = OUTPUT / "signals.csv"

print(f"{'='*60}")
print(f"TAIL RISK PAPER ENGINE — {datetime.now().strftime('%Y-%m-%d %H:%M ET')}")
print(f"{'='*60}")

if STATE_FILE.exists():
    with open(STATE_FILE) as f:
        state = json.load(f)
    print(f"Previous: {state.get('position', 'none')} on {state.get('signal_date', '?')}")
else:
    state = {'position': 'none', 'trades': 0}

# Config
HEDGE = ['TAIL', 'BTAL', 'SH']
GROWTH = ['QQQ', 'SPY']
SAFETY = ['TLT', 'SHY']
MACRO = ['^VIX', 'GLD', 'HYG', 'LQD', 'DBC', 'XLU', 'XLP', '^TNX', 'IEF', 'UUP']

# Download
print("\nDownloading data...")
start_date = (datetime.now() - timedelta(days=800)).strftime('%Y-%m-%d')
all_tickers = list(set(HEDGE + GROWTH + SAFETY + MACRO))
data = {}
for t in all_tickers:
    try:
        df = yf.download(t, start=start_date, progress=False, auto_adjust=True)
        if len(df) > 50:
            if isinstance(df.columns, pd.MultiIndex):
                close = df[('Close', t)].copy()
            else:
                close = df['Close'].copy()
            clean = t.replace('^', '')
            close.name = clean
            data[clean] = close
    except:
        pass

prices = pd.DataFrame(data).dropna(how='all').ffill().dropna()
returns = prices.pct_change()
print(f"Data: {len(prices)} days, latest: {prices.index[-1].date()}")

# Features
features = pd.DataFrame(index=prices.index)

# VIX features (key for tail risk)
if 'VIX' in prices.columns:
    features['vix_level'] = prices['VIX']
    features['vix_zscore_21d'] = (prices['VIX'] - prices['VIX'].rolling(21).mean()) / prices['VIX'].rolling(21).std()
    features['vix_zscore_63d'] = (prices['VIX'] - prices['VIX'].rolling(63).mean()) / prices['VIX'].rolling(63).std()
    features['vix_mom_5d'] = prices['VIX'].pct_change(5)
    features['vix_mom_21d'] = prices['VIX'].pct_change(21)

# Equity
if 'SPY' in prices.columns:
    features['spy_mom_5d'] = prices['SPY'].pct_change(5)
    features['spy_mom_21d'] = prices['SPY'].pct_change(21)
    features['spy_mom_63d'] = prices['SPY'].pct_change(63)
    features['spy_vol_21d'] = returns['SPY'].rolling(21).std()
    features['spy_drawdown'] = prices['SPY'] / prices['SPY'].rolling(252).max() - 1

# Credit
if 'HYG' in prices.columns and 'LQD' in prices.columns:
    features['credit'] = (returns['HYG'] - returns['LQD']).rolling(21).mean()
    features['credit_mom_21d'] = (prices['HYG'] / prices['LQD']).pct_change(21)

# Defensives
for d in ['XLU', 'XLP']:
    if d in prices.columns:
        features[f'{d}_mom_21d'] = prices[d].pct_change(21)

# Bonds
if 'TLT' in prices.columns:
    features['tlt_mom_21d'] = prices['TLT'].pct_change(21)

# Gold
if 'GLD' in prices.columns:
    features['gld_mom_21d'] = prices['GLD'].pct_change(21)

# USD
if 'UUP' in prices.columns:
    features['usd_mom_21d'] = prices['UUP'].pct_change(21)

# Yields
if 'TNX' in prices.columns:
    features['tnx_level'] = prices['TNX']
    features['tnx_mom_21d'] = prices['TNX'].pct_change(21)

# Commodities
if 'DBC' in prices.columns:
    features['dbc_mom_21d'] = prices['DBC'].pct_change(21)

# Hedge ETF momentum
for h in HEDGE:
    if h in prices.columns:
        features[f'{h}_mom_21d'] = prices[h].pct_change(21)

features = features.dropna()
print(f"Features: {features.shape[1]} columns")

# Train: predict SPY drawdown >5% in next 21 days
if 'SPY' in returns.columns:
    spy_fwd = returns['SPY'].rolling(21).sum().shift(-21)
    target = (spy_fwd < -0.05).astype(int)  # 1 = tail risk event
else:
    print("ERROR: no SPY data")
    exit(1)

train_features = features.iloc[:-1]
train_labels = target.reindex(train_features.index).dropna()
common = train_features.index.intersection(train_labels.index)

if len(common) < 100:
    print("ERROR: Insufficient data")
    exit(1)

X_train = np.nan_to_num(train_features.loc[common].values, nan=0, posinf=0, neginf=0)
y_train = train_labels.loc[common].values

print(f"Training: {len(X_train)} samples, tail events: {y_train.mean():.1%}")

model = lgb.LGBMClassifier(
    n_estimators=100, max_depth=4, learning_rate=0.05,
    min_child_samples=5, subsample=0.8, colsample_bytree=0.8,
    verbose=-1, n_jobs=1
)
model.fit(X_train, y_train)

# Signal
latest_X = np.nan_to_num(features.iloc[-1].values.reshape(1, -1), nan=0, posinf=0, neginf=0)
prob_tail = model.predict_proba(latest_X)[0, 1]

if prob_tail > 0.3:  # Lower threshold — tail risk is rare
    position = 'HEDGE'
    action = 'Tail risk elevated — long TAIL/BTAL/SH, reduce equity'
elif prob_tail < 0.1:
    position = 'GROWTH'
    action = 'Low tail risk — full equity exposure (QQQ/SPY)'
else:
    position = 'BALANCED'
    action = 'Moderate risk — balanced equity + some protection'

vix = prices['VIX'].iloc[-1] if 'VIX' in prices.columns else 0
dd = (prices['SPY'].iloc[-1] / prices['SPY'].rolling(252).max().iloc[-1] - 1) * 100 if 'SPY' in prices.columns else 0

print(f"\n{'='*60}")
print(f"SIGNAL: {position}")
print(f"Action: {action}")
print(f"P(tail event): {prob_tail:.1%}")
print(f"VIX: {vix:.1f}, SPY drawdown: {dd:.1f}%")
print(f"{'='*60}")

prev = state.get('position', 'none')
if prev != position:
    state['trades'] = state.get('trades', 0) + 1

state.update({
    'signal_date': str(features.index[-1].date()),
    'position': position,
    'prob_tail': round(prob_tail, 3),
    'vix': round(vix, 1),
    'spy_drawdown': round(dd, 1),
    'action': action,
    'updated': datetime.now().isoformat(),
})

with open(STATE_FILE, 'w') as f:
    json.dump(state, f, indent=2)

log_entry = {
    'date': str(features.index[-1].date()),
    'position': position,
    'prob_tail': round(prob_tail, 3),
    'vix': round(vix, 1),
    'spy_dd': round(dd, 1),
}
pd.DataFrame([log_entry]).to_csv(SIGNAL_LOG, mode='a',
    header=not SIGNAL_LOG.exists(), index=False)

print(f"\nState saved. Done.")
