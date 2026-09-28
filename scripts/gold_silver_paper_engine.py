#!/usr/bin/env python3
"""
ML Gold/Silver Ratio Trade — Paper Engine
==========================================
Generates daily signals for the gold/silver ratio strategy.
Runs as PM2 cron at market close on weekdays.

Strategy: ML predicts whether gold (GLD) or silver (SLV) will outperform
over the next 21 days based on the gold/silver ratio, momentum, USD, credit.
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

OUTPUT = Path("/home/jupiter/Lvl3Quant/output/ml_gold_silver_paper")
OUTPUT.mkdir(parents=True, exist_ok=True)

STATE_FILE = OUTPUT / "state.json"
SIGNAL_LOG = OUTPUT / "signals.csv"

print(f"{'='*60}")
print(f"GOLD/SILVER RATIO PAPER ENGINE — {datetime.now().strftime('%Y-%m-%d %H:%M ET')}")
print(f"{'='*60}")

# ─── Load state ───
if STATE_FILE.exists():
    with open(STATE_FILE) as f:
        state = json.load(f)
    print(f"Previous signal: {state.get('position', 'none')} on {state.get('signal_date', '?')}")
else:
    state = {'position': 'none', 'nav': 100000, 'trades': 0, 'pnl': 0}

# ─── Config ───
GOLD = 'GLD'
SILVER = 'SLV'
MACRO_TICKERS = ['^VIX', 'UUP', 'DBC', 'HYG', 'LQD', 'SPY', 'TLT', '^TNX']
TRAIN_DAYS = 252
ADVANCE_DAYS = 21

# ─── Download data (2 years for training) ───
print("\nDownloading data...")
start_date = (datetime.now() - timedelta(days=600)).strftime('%Y-%m-%d')
all_tickers = list(set([GOLD, SILVER] + MACRO_TICKERS))
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

# ─── Build features ───
features = pd.DataFrame(index=prices.index)

# Gold/Silver ratio
prices['gsr'] = prices[GOLD] / prices[SILVER]
features['gsr'] = prices['gsr']
features['gsr_zscore_63d'] = (prices['gsr'] - prices['gsr'].rolling(63).mean()) / prices['gsr'].rolling(63).std()
features['gsr_zscore_126d'] = (prices['gsr'] - prices['gsr'].rolling(126).mean()) / prices['gsr'].rolling(126).std()
features['gsr_mom_5d'] = prices['gsr'].pct_change(5)
features['gsr_mom_21d'] = prices['gsr'].pct_change(21)
features['gsr_mom_63d'] = prices['gsr'].pct_change(63)
features['gsr_vol_21d'] = prices['gsr'].pct_change().rolling(21).std()

# Individual precious metals
for pm in [GOLD, SILVER]:
    if pm in prices.columns:
        features[f'{pm}_mom_5d'] = prices[pm].pct_change(5)
        features[f'{pm}_mom_21d'] = prices[pm].pct_change(21)
        features[f'{pm}_mom_63d'] = prices[pm].pct_change(63)
        features[f'{pm}_vol_21d'] = returns[pm].rolling(21).std()

# Relative vol
features['vol_ratio'] = returns[SILVER].rolling(21).std() / returns[GOLD].rolling(21).std().clip(lower=1e-6)

# Macro
macro_names = [t.replace('^', '') for t in MACRO_TICKERS if t.replace('^', '') in prices.columns]
for m in macro_names:
    if m in prices.columns:
        features[f'{m}_ret_21d'] = prices[m].pct_change(21)
        if m in returns.columns:
            features[f'{m}_vol_21d'] = returns[m].rolling(21).std()

if 'VIX' in prices.columns:
    features['vix_level'] = prices['VIX']
    features['vix_zscore'] = (prices['VIX'] - prices['VIX'].rolling(63).mean()) / prices['VIX'].rolling(63).std()

if 'UUP' in prices.columns:
    features['usd_mom_21d'] = prices['UUP'].pct_change(21)
    features['usd_mom_63d'] = prices['UUP'].pct_change(63)

if 'HYG' in prices.columns and 'LQD' in prices.columns:
    features['credit_spread'] = (returns['HYG'] - returns['LQD']).rolling(21).mean()

features = features.dropna()
print(f"Features: {features.shape[1]} columns")

# ─── Train ───
# Target: gold outperforms silver over next 21 days
returns['gold_vs_silver'] = returns[GOLD] - returns[SILVER]
gold_fwd = returns['gold_vs_silver'].rolling(ADVANCE_DAYS).sum().shift(-ADVANCE_DAYS)

train_features = features.iloc[:-1]
train_labels = gold_fwd.reindex(train_features.index).dropna()
common = train_features.index.intersection(train_labels.index)

if len(common) < 100:
    print("ERROR: Insufficient training data")
    exit(1)

X_train = np.nan_to_num(train_features.loc[common].values, nan=0, posinf=0, neginf=0)
y_train = (train_labels.loc[common] > 0).astype(int).values

print(f"Training: {len(X_train)} samples, label balance: {y_train.mean():.1%} gold outperforms")

model = lgb.LGBMClassifier(
    n_estimators=100, max_depth=4, learning_rate=0.05,
    min_child_samples=5, subsample=0.8, colsample_bytree=0.8,
    verbose=-1, n_jobs=1
)
model.fit(X_train, y_train)

# ─── Generate signal ───
latest_date = features.index[-1]
latest_X = np.nan_to_num(features.iloc[-1].values.reshape(1, -1), nan=0, posinf=0, neginf=0)
prob_gold = model.predict_proba(latest_X)[0, 1]

if prob_gold > 0.6:
    position = 'GOLD'
    action = f'Long {GOLD} (gold outperforms silver)'
elif prob_gold < 0.4:
    position = 'SILVER'
    action = f'Long {SILVER} (silver outperforms gold)'
else:
    position = 'NEUTRAL'
    action = 'Equal weight or no position'

gsr = prices['gsr'].iloc[-1]
gsr_z = features['gsr_zscore_63d'].iloc[-1] if 'gsr_zscore_63d' in features.columns else 0
vix = prices['VIX'].iloc[-1] if 'VIX' in prices.columns else 0

print(f"\n{'='*60}")
print(f"SIGNAL: {position}")
print(f"Action: {action}")
print(f"Probability (gold outperforms): {prob_gold:.1%}")
print(f"Gold/Silver ratio: {gsr:.2f}")
print(f"GSR z-score (63d): {gsr_z:.2f}")
print(f"VIX: {vix:.1f}")
print(f"{'='*60}")

# ─── Update state ───
prev_position = state.get('position', 'none')
if prev_position != position:
    state['trades'] = state.get('trades', 0) + 1

state.update({
    'signal_date': str(latest_date.date()),
    'position': position,
    'probability': round(prob_gold, 3),
    'gsr': round(gsr, 2),
    'gsr_zscore': round(gsr_z, 2),
    'vix': round(vix, 1),
    'action': action,
    'updated': datetime.now().isoformat(),
})

with open(STATE_FILE, 'w') as f:
    json.dump(state, f, indent=2)

log_entry = {
    'date': str(latest_date.date()),
    'position': position,
    'prob_gold': round(prob_gold, 3),
    'gsr': round(gsr, 2),
    'gsr_z': round(gsr_z, 2),
    'vix': round(vix, 1),
}
pd.DataFrame([log_entry]).to_csv(SIGNAL_LOG, mode='a',
    header=not SIGNAL_LOG.exists(), index=False)

print(f"\nState saved. Done.")
