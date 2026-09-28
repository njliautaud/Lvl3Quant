#!/usr/bin/env python3
"""
ML Bond Duration Timing — Paper Engine
========================================
Generates daily signal: LONG_DURATION / SHORT_DURATION / NEUTRAL
based on yield curve, inflation expectations, rate momentum.
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

OUTPUT = Path("/home/jupiter/Lvl3Quant/output/ml_bond_duration_paper")
OUTPUT.mkdir(parents=True, exist_ok=True)

STATE_FILE = OUTPUT / "state.json"
SIGNAL_LOG = OUTPUT / "signals.csv"

print(f"{'='*60}")
print(f"BOND DURATION PAPER ENGINE — {datetime.now().strftime('%Y-%m-%d %H:%M ET')}")
print(f"{'='*60}")

# Load state
if STATE_FILE.exists():
    with open(STATE_FILE) as f:
        state = json.load(f)
    print(f"Previous: {state.get('position', 'none')} on {state.get('signal_date', '?')}")
else:
    state = {'position': 'none', 'trades': 0}

# Config
LONG_DUR = ['TLT', 'EDV']
SHORT_DUR = ['SHY', 'BIL']
TIPS = ['TIP']
MACRO = ['^VIX', 'GLD', 'DBC', 'HYG', 'LQD', 'SPY', 'UUP', '^TNX', 'IEF']

# Download
print("\nDownloading data...")
start_date = (datetime.now() - timedelta(days=800)).strftime('%Y-%m-%d')
all_tickers = list(set(LONG_DUR + SHORT_DUR + TIPS + MACRO))
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

# Build features
features = pd.DataFrame(index=prices.index)

# Duration spread (long vs short bonds)
if 'TLT' in prices.columns and 'SHY' in prices.columns:
    prices['dur_ratio'] = prices['TLT'] / prices['SHY']
    features['dur_ratio'] = prices['dur_ratio']
    features['dur_zscore_21d'] = (prices['dur_ratio'] - prices['dur_ratio'].rolling(21).mean()) / prices['dur_ratio'].rolling(21).std()
    features['dur_zscore_63d'] = (prices['dur_ratio'] - prices['dur_ratio'].rolling(63).mean()) / prices['dur_ratio'].rolling(63).std()
    features['dur_mom_5d'] = prices['dur_ratio'].pct_change(5)
    features['dur_mom_21d'] = prices['dur_ratio'].pct_change(21)
    features['dur_mom_63d'] = prices['dur_ratio'].pct_change(63)

# Individual bond ETFs
for etf in ['TLT', 'SHY', 'IEF', 'TIP', 'EDV']:
    if etf in prices.columns:
        features[f'{etf}_mom_5d'] = prices[etf].pct_change(5)
        features[f'{etf}_mom_21d'] = prices[etf].pct_change(21)
        features[f'{etf}_vol_21d'] = returns[etf].rolling(21).std()

# Yields
if 'TNX' in prices.columns:
    features['tnx_level'] = prices['TNX']
    features['tnx_mom_5d'] = prices['TNX'].pct_change(5)
    features['tnx_mom_21d'] = prices['TNX'].pct_change(21)
    features['tnx_zscore'] = (prices['TNX'] - prices['TNX'].rolling(63).mean()) / prices['TNX'].rolling(63).std()

# Curve shape
if 'IEF' in prices.columns and 'TLT' in prices.columns:
    features['curve'] = prices['TLT'] / prices['IEF']
    features['curve_mom_21d'] = features['curve'].pct_change(21)

# Breakeven inflation
if 'TIP' in prices.columns and 'TLT' in prices.columns:
    features['bei'] = prices['TIP'] / prices['TLT']
    features['bei_mom_21d'] = features['bei'].pct_change(21)

# VIX
if 'VIX' in prices.columns:
    features['vix_level'] = prices['VIX']
    features['vix_zscore'] = (prices['VIX'] - prices['VIX'].rolling(63).mean()) / prices['VIX'].rolling(63).std()

# Credit
if 'HYG' in prices.columns and 'LQD' in prices.columns:
    features['credit'] = (returns['HYG'] - returns['LQD']).rolling(21).mean()

# USD
if 'UUP' in prices.columns:
    features['usd_mom_21d'] = prices['UUP'].pct_change(21)

# SPY
if 'SPY' in prices.columns:
    features['spy_mom_21d'] = prices['SPY'].pct_change(21)
    features['spy_vol_21d'] = returns['SPY'].rolling(21).std()

# Gold
if 'GLD' in prices.columns:
    features['gld_mom_21d'] = prices['GLD'].pct_change(21)

# Commodities
if 'DBC' in prices.columns:
    features['dbc_mom_21d'] = prices['DBC'].pct_change(21)

features = features.dropna()
print(f"Features: {features.shape[1]} columns")

# Train on all available data
# Target: long duration (TLT) outperforms short duration (SHY) over next 21d
if 'TLT' in returns.columns and 'SHY' in returns.columns:
    fwd = (returns['TLT'] - returns['SHY']).rolling(21).sum().shift(-21)
else:
    print("ERROR: Missing TLT or SHY")
    exit(1)

train_features = features.iloc[:-1]
train_labels = fwd.reindex(train_features.index).dropna()
common = train_features.index.intersection(train_labels.index)

if len(common) < 100:
    print("ERROR: Insufficient training data")
    exit(1)

X_train = np.nan_to_num(train_features.loc[common].values, nan=0, posinf=0, neginf=0)
y_train = (train_labels.loc[common] > 0).astype(int).values

print(f"Training: {len(X_train)} samples, long-dur wins {y_train.mean():.1%}")

model = lgb.LGBMClassifier(
    n_estimators=100, max_depth=4, learning_rate=0.05,
    min_child_samples=5, subsample=0.8, colsample_bytree=0.8,
    verbose=-1, n_jobs=1
)
model.fit(X_train, y_train)

# Generate signal
latest_X = np.nan_to_num(features.iloc[-1].values.reshape(1, -1), nan=0, posinf=0, neginf=0)
prob_long = model.predict_proba(latest_X)[0, 1]

if prob_long > 0.6:
    position = 'LONG_DURATION'
    action = 'Long TLT/EDV (rates expected to fall / bonds rally)'
elif prob_long < 0.4:
    position = 'SHORT_DURATION'
    action = 'Long SHY/BIL (rates expected to rise / avoid duration)'
else:
    position = 'NEUTRAL'
    action = 'Mixed duration (IEF) or no position'

tnx = prices['TNX'].iloc[-1] if 'TNX' in prices.columns else 0
vix = prices['VIX'].iloc[-1] if 'VIX' in prices.columns else 0
tlt_mom = prices['TLT'].pct_change(21).iloc[-1] * 100 if 'TLT' in prices.columns else 0

print(f"\n{'='*60}")
print(f"SIGNAL: {position}")
print(f"Action: {action}")
print(f"P(long duration wins): {prob_long:.1%}")
print(f"10Y yield: {tnx:.2f}%, VIX: {vix:.1f}")
print(f"TLT 21d return: {tlt_mom:.1f}%")
print(f"{'='*60}")

# Update state
prev = state.get('position', 'none')
if prev != position:
    state['trades'] = state.get('trades', 0) + 1

state.update({
    'signal_date': str(features.index[-1].date()),
    'position': position,
    'probability': round(prob_long, 3),
    'tnx': round(tnx, 2),
    'vix': round(vix, 1),
    'tlt_mom_21d': round(tlt_mom, 1),
    'action': action,
    'updated': datetime.now().isoformat(),
})

with open(STATE_FILE, 'w') as f:
    json.dump(state, f, indent=2)

log_entry = {
    'date': str(features.index[-1].date()),
    'position': position,
    'prob_long': round(prob_long, 3),
    'tnx': round(tnx, 2),
    'vix': round(vix, 1),
}
pd.DataFrame([log_entry]).to_csv(SIGNAL_LOG, mode='a',
    header=not SIGNAL_LOG.exists(), index=False)

print(f"\nState saved. Done.")
