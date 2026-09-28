#!/usr/bin/env python3
"""
ML Yield Curve Trade — Paper Engine
=====================================
Generates daily signals for the yield curve steepener/flattener strategy.
Runs as PM2 cron at market close on weekdays.

Strategy: ML predicts whether curve will steepen (long SHY, short TLT)
or flatten (long TLT, short SHY) over next 21 days.
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

OUTPUT = Path("/home/jupiter/Lvl3Quant/output/ml_yield_curve_paper")
OUTPUT.mkdir(parents=True, exist_ok=True)

STATE_FILE = OUTPUT / "state.json"
SIGNAL_LOG = OUTPUT / "signals.csv"

print(f"{'='*60}")
print(f"YIELD CURVE PAPER ENGINE — {datetime.now().strftime('%Y-%m-%d %H:%M ET')}")
print(f"{'='*60}")

# ─── Load state ───
if STATE_FILE.exists():
    with open(STATE_FILE) as f:
        state = json.load(f)
    print(f"Previous signal: {state.get('position', 'none')} on {state.get('signal_date', '?')}")
else:
    state = {'position': 'none', 'nav': 100000, 'trades': 0, 'pnl': 0}

# ─── Config ───
SHORT_END = 'SHY'
LONG_END = 'TLT'
MID_END = 'IEF'
TIPS = 'TIP'
MACRO_TICKERS = ['^VIX', 'GLD', 'UUP', 'DBC', 'HYG', 'LQD', 'SPY', 'EEM', '^TNX']
TRAIN_DAYS = 252
ADVANCE_DAYS = 21

# ─── Download recent data (2 years for training) ───
print("\nDownloading data...")
start_date = (datetime.now() - timedelta(days=600)).strftime('%Y-%m-%d')
all_tickers = list(set([SHORT_END, MID_END, LONG_END, TIPS] + MACRO_TICKERS))
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

prices['curve_slope'] = prices[LONG_END] / prices[SHORT_END]
features['curve_slope'] = prices['curve_slope']
features['slope_zscore_63d'] = (prices['curve_slope'] - prices['curve_slope'].rolling(63).mean()) / prices['curve_slope'].rolling(63).std()
features['slope_zscore_126d'] = (prices['curve_slope'] - prices['curve_slope'].rolling(126).mean()) / prices['curve_slope'].rolling(126).std()
features['slope_mom_5d'] = prices['curve_slope'].pct_change(5)
features['slope_mom_21d'] = prices['curve_slope'].pct_change(21)
features['slope_mom_63d'] = prices['curve_slope'].pct_change(63)
features['slope_vol_21d'] = prices['curve_slope'].pct_change().rolling(21).std()
features['dur_vol_ratio'] = returns[LONG_END].rolling(21).std() / returns[SHORT_END].rolling(21).std().clip(lower=1e-6)

if MID_END in prices.columns:
    features['belly_vs_wings'] = prices[MID_END] / ((prices[SHORT_END] + prices[LONG_END]) / 2)
    features['belly_mom_21d'] = prices[MID_END].pct_change(21)

if TIPS in prices.columns:
    features['tips_mom_21d'] = prices[TIPS].pct_change(21)

for bond in [SHORT_END, MID_END, LONG_END]:
    if bond in prices.columns:
        features[f'{bond}_mom_5d'] = prices[bond].pct_change(5)
        features[f'{bond}_mom_21d'] = prices[bond].pct_change(21)
        features[f'{bond}_mom_63d'] = prices[bond].pct_change(63)
        features[f'{bond}_vol_21d'] = returns[bond].rolling(21).std()

macro_names = [t.replace('^', '') for t in MACRO_TICKERS if t.replace('^', '') in prices.columns]
for m in macro_names:
    if m in prices.columns:
        features[f'{m}_ret_21d'] = prices[m].pct_change(21)
        if m in returns.columns:
            features[f'{m}_vol_21d'] = returns[m].rolling(21).std()

if 'VIX' in prices.columns:
    features['vix_level'] = prices['VIX']
    features['vix_zscore'] = (prices['VIX'] - prices['VIX'].rolling(63).mean()) / prices['VIX'].rolling(63).std()

if 'HYG' in prices.columns and 'LQD' in prices.columns:
    features['credit_spread'] = (returns['HYG'] - returns['LQD']).rolling(21).mean()

features = features.dropna()
print(f"Features: {features.shape[1]} columns")

# ─── Train on recent history ───
returns['steepener'] = returns[SHORT_END] - returns[LONG_END]
steep_fwd = returns['steepener'].rolling(ADVANCE_DAYS).sum().shift(-ADVANCE_DAYS)

# Use all available training data
train_features = features.iloc[:-1]  # Exclude most recent for prediction
train_labels = steep_fwd.reindex(train_features.index).dropna()
common = train_features.index.intersection(train_labels.index)

if len(common) < 100:
    print("ERROR: Insufficient training data")
    exit(1)

X_train = np.nan_to_num(train_features.loc[common].values, nan=0, posinf=0, neginf=0)
y_train = (train_labels.loc[common] > 0).astype(int).values

print(f"Training: {len(X_train)} samples, label balance: {y_train.mean():.1%} steepener")

model = lgb.LGBMClassifier(
    n_estimators=100, max_depth=4, learning_rate=0.05,
    min_child_samples=5, subsample=0.8, colsample_bytree=0.8,
    verbose=-1, n_jobs=1
)
model.fit(X_train, y_train)

# ─── Generate signal ───
latest_date = features.index[-1]
latest_X = np.nan_to_num(features.iloc[-1].values.reshape(1, -1), nan=0, posinf=0, neginf=0)
prob_steep = model.predict_proba(latest_X)[0, 1]

if prob_steep > 0.6:
    position = 'STEEPENER'
    action = f'Long {SHORT_END}, Short {LONG_END}'
elif prob_steep < 0.4:
    position = 'FLATTENER'
    action = f'Long {LONG_END}, Short {SHORT_END}'
else:
    position = 'NEUTRAL'
    action = 'No position'

# Curve context
curve_slope = prices['curve_slope'].iloc[-1]
slope_z = features['slope_zscore_63d'].iloc[-1] if 'slope_zscore_63d' in features.columns else 0
vix = prices['VIX'].iloc[-1] if 'VIX' in prices.columns else 0

print(f"\n{'='*60}")
print(f"SIGNAL: {position}")
print(f"Action: {action}")
print(f"Probability (steepener): {prob_steep:.1%}")
print(f"Curve slope (TLT/SHY): {curve_slope:.3f}")
print(f"Slope z-score (63d): {slope_z:.2f}")
print(f"VIX: {vix:.1f}")
print(f"{'='*60}")

# ─── Update state ───
prev_position = state.get('position', 'none')
if prev_position != position:
    state['trades'] = state.get('trades', 0) + 1

state.update({
    'signal_date': str(latest_date.date()),
    'position': position,
    'probability': round(prob_steep, 3),
    'curve_slope': round(curve_slope, 3),
    'slope_zscore': round(slope_z, 2),
    'vix': round(vix, 1),
    'action': action,
    'updated': datetime.now().isoformat(),
})

with open(STATE_FILE, 'w') as f:
    json.dump(state, f, indent=2)

# Append to signal log
log_entry = {
    'date': str(latest_date.date()),
    'position': position,
    'prob_steep': round(prob_steep, 3),
    'curve_slope': round(curve_slope, 3),
    'slope_z': round(slope_z, 2),
    'vix': round(vix, 1),
}
pd.DataFrame([log_entry]).to_csv(SIGNAL_LOG, mode='a',
    header=not SIGNAL_LOG.exists(), index=False)

print(f"\nState saved. Done.")
