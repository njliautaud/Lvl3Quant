#!/usr/bin/env python3
"""
ML Currency Carry Trade — Paper Engine
========================================
Generates daily signal: CARRY (high-yield FX), HAVEN (safe FX), or NEUTRAL.
Uses FX ETFs: FXA/FXB/CEW (carry), FXY/FXF/UUP (haven), FXE/FXC (neutral).
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

OUTPUT = Path("/home/jupiter/Lvl3Quant/output/ml_currency_carry_paper")
OUTPUT.mkdir(parents=True, exist_ok=True)

STATE_FILE = OUTPUT / "state.json"
SIGNAL_LOG = OUTPUT / "signals.csv"

print(f"{'='*60}")
print(f"CURRENCY CARRY PAPER ENGINE — {datetime.now().strftime('%Y-%m-%d %H:%M ET')}")
print(f"{'='*60}")

if STATE_FILE.exists():
    with open(STATE_FILE) as f:
        state = json.load(f)
    print(f"Previous: {state.get('position', 'none')} on {state.get('signal_date', '?')}")
else:
    state = {'position': 'none', 'trades': 0}

# Config
CARRY = ['FXA', 'FXB', 'CEW']
HAVEN = ['FXY', 'FXF', 'UUP']
NEUTRAL_FX = ['FXE', 'FXC']
MACRO = ['^VIX', 'GLD', 'TLT', 'HYG', 'DBC', 'SPY', 'LQD', '^TNX', 'EEM']

# Download
print("\nDownloading data...")
start_date = (datetime.now() - timedelta(days=800)).strftime('%Y-%m-%d')
all_tickers = list(set(CARRY + HAVEN + NEUTRAL_FX + MACRO))
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

# Carry vs Haven spread
carry_avail = [c for c in CARRY if c in prices.columns]
haven_avail = [h for h in HAVEN if h in prices.columns]

if carry_avail and haven_avail:
    carry_avg = prices[carry_avail].mean(axis=1)
    haven_avg = prices[haven_avail].mean(axis=1)
    prices['carry_haven_ratio'] = carry_avg / haven_avg
    features['ch_ratio'] = prices['carry_haven_ratio']
    features['ch_zscore_21d'] = (prices['carry_haven_ratio'] - prices['carry_haven_ratio'].rolling(21).mean()) / prices['carry_haven_ratio'].rolling(21).std()
    features['ch_zscore_63d'] = (prices['carry_haven_ratio'] - prices['carry_haven_ratio'].rolling(63).mean()) / prices['carry_haven_ratio'].rolling(63).std()
    features['ch_mom_5d'] = prices['carry_haven_ratio'].pct_change(5)
    features['ch_mom_21d'] = prices['carry_haven_ratio'].pct_change(21)
    features['ch_mom_63d'] = prices['carry_haven_ratio'].pct_change(63)

# Individual FX momentum
for fx in carry_avail + haven_avail:
    features[f'{fx}_mom_5d'] = prices[fx].pct_change(5)
    features[f'{fx}_mom_21d'] = prices[fx].pct_change(21)
    features[f'{fx}_vol_21d'] = returns[fx].rolling(21).std()

# VIX (risk appetite)
if 'VIX' in prices.columns:
    features['vix_level'] = prices['VIX']
    features['vix_zscore'] = (prices['VIX'] - prices['VIX'].rolling(63).mean()) / prices['VIX'].rolling(63).std()
    features['vix_mom_21d'] = prices['VIX'].pct_change(21)

# Credit
if 'HYG' in prices.columns and 'LQD' in prices.columns:
    features['credit'] = (returns['HYG'] - returns['LQD']).rolling(21).mean()

# EM (carry proxy)
if 'EEM' in prices.columns:
    features['eem_mom_21d'] = prices['EEM'].pct_change(21)

# Gold, bonds, commodities
for m in ['GLD', 'TLT', 'DBC', 'SPY']:
    if m in prices.columns:
        features[f'{m}_mom_21d'] = prices[m].pct_change(21)

if 'TNX' in prices.columns:
    features['tnx_level'] = prices['TNX']
    features['tnx_mom_21d'] = prices['TNX'].pct_change(21)

features = features.dropna()
print(f"Features: {features.shape[1]} columns")

# Train
carry_ret = returns[carry_avail].mean(axis=1) if carry_avail else pd.Series(0, index=returns.index)
haven_ret = returns[haven_avail].mean(axis=1) if haven_avail else pd.Series(0, index=returns.index)
fwd = (carry_ret - haven_ret).rolling(21).sum().shift(-21)

train_features = features.iloc[:-1]
train_labels = fwd.reindex(train_features.index).dropna()
common = train_features.index.intersection(train_labels.index)

if len(common) < 100:
    print("ERROR: Insufficient training data")
    exit(1)

X_train = np.nan_to_num(train_features.loc[common].values, nan=0, posinf=0, neginf=0)
y_train = (train_labels.loc[common] > 0).astype(int).values

print(f"Training: {len(X_train)} samples, carry wins {y_train.mean():.1%}")

model = lgb.LGBMClassifier(
    n_estimators=100, max_depth=4, learning_rate=0.05,
    min_child_samples=5, subsample=0.8, colsample_bytree=0.8,
    verbose=-1, n_jobs=1
)
model.fit(X_train, y_train)

# Signal
latest_X = np.nan_to_num(features.iloc[-1].values.reshape(1, -1), nan=0, posinf=0, neginf=0)
prob_carry = model.predict_proba(latest_X)[0, 1]

if prob_carry > 0.6:
    position = 'CARRY'
    action = f'Long carry FX ({", ".join(carry_avail)})'
elif prob_carry < 0.4:
    position = 'HAVEN'
    action = f'Long safe-haven FX ({", ".join(haven_avail)})'
else:
    position = 'NEUTRAL'
    action = 'Balanced or no position'

vix = prices['VIX'].iloc[-1] if 'VIX' in prices.columns else 0

print(f"\n{'='*60}")
print(f"SIGNAL: {position}")
print(f"Action: {action}")
print(f"P(carry wins): {prob_carry:.1%}")
print(f"VIX: {vix:.1f}")
print(f"{'='*60}")

prev = state.get('position', 'none')
if prev != position:
    state['trades'] = state.get('trades', 0) + 1

state.update({
    'signal_date': str(features.index[-1].date()),
    'position': position,
    'probability': round(prob_carry, 3),
    'vix': round(vix, 1),
    'action': action,
    'updated': datetime.now().isoformat(),
})

with open(STATE_FILE, 'w') as f:
    json.dump(state, f, indent=2)

log_entry = {
    'date': str(features.index[-1].date()),
    'position': position,
    'prob_carry': round(prob_carry, 3),
    'vix': round(vix, 1),
}
pd.DataFrame([log_entry]).to_csv(SIGNAL_LOG, mode='a',
    header=not SIGNAL_LOG.exists(), index=False)

print(f"\nState saved. Done.")
