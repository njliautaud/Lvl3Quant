#!/usr/bin/env python3
"""
VIX Spike Probability Signal Watcher — Daily Engine
Predicts probability of VIX>30 within 5 days using cross-asset signals.

Two models:
  1. Full model (all 41 features including raw VIX) — high accuracy but partially circular
  2. Ablation model (without raw VIX features) — GENUINE early warning signal

Outputs:
  - /home/jupiter/Lvl3Quant/state/vix_spike_signal.json  (latest signal)
  - /home/jupiter/Lvl3Quant/data/vix_spike_history.csv    (append-only history)

Risk levels (based on ABLATION probability — the genuine signal):
  LOW:      < 0.15
  MODERATE: 0.15 - 0.35
  HIGH:     0.35 - 0.60
  EXTREME:  > 0.60

Designed for PM2 cron at 16:30 ET weekdays.
Usage: python3 engines/vix_spike_watcher.py
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import GradientBoostingClassifier
import warnings
import json
import os
import sys
from datetime import datetime

warnings.filterwarnings('ignore')

STATE_FILE = '/home/jupiter/Lvl3Quant/state/vix_spike_signal.json'
HISTORY_FILE = '/home/jupiter/Lvl3Quant/data/vix_spike_history.csv'

os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)

GBM_PARAMS = dict(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    min_samples_leaf=20,
    random_state=42,
)

# Ablation: these columns are removed for the "genuine early warning" model
RAW_VIX_COLS = ['vix', 'vix_pctile_252d', 'vix_pctile_63d']


def make_serializable(obj):
    """Convert numpy types to Python natives for JSON serialization."""
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_serializable(i) for i in obj]
    return obj


def download_data():
    """Download cross-asset price data."""
    tickers = {
        'VIX': '^VIX',
        'VIX3M': '^VIX3M',
        'SPY': 'SPY',
        'GLD': 'GLD',
        'SLV': 'SLV',
        'USO': 'USO',
        'UUP': 'UUP',
        'TLT': 'TLT',
        'HYG': 'HYG',
        'IEF': 'IEF',
        'CPER': 'CPER',
        'BTC-USD': 'BTC-USD',
    }

    data = {}
    for name, ticker in tickers.items():
        try:
            df = yf.download(ticker, start='2010-01-01', progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            data[name] = df['Close'].rename(name)
        except Exception as e:
            print(f"  WARNING: {name} download failed ({e})")

    prices = pd.DataFrame(data)
    prices = prices.ffill().dropna(how='all')
    return prices


def engineer_features(prices):
    """Build the 41 cross-asset features (identical to ml_vix_spike_predictor.py)."""
    feat = pd.DataFrame(index=prices.index)

    # VIX features (7)
    feat['vix'] = prices['VIX']
    feat['vix_5d_chg'] = prices['VIX'].pct_change(5)
    feat['vix_10d_chg'] = prices['VIX'].pct_change(10)
    feat['vix_20d_chg'] = prices['VIX'].pct_change(20)
    feat['vix_pctile_63d'] = prices['VIX'].rolling(63).apply(
        lambda x: (x[-1] - x.min()) / (x.max() - x.min() + 1e-8), raw=True)
    feat['vix_pctile_252d'] = prices['VIX'].rolling(252).apply(
        lambda x: (x[-1] - x.min()) / (x.max() - x.min() + 1e-8), raw=True)
    feat['vix_zscore_20d'] = (prices['VIX'] - prices['VIX'].rolling(20).mean()) / prices['VIX'].rolling(20).std()

    # VIX term structure (3)
    if 'VIX3M' in prices.columns:
        feat['vix_term_ratio'] = prices['VIX'] / prices['VIX3M']
        feat['vix_term_5d_chg'] = feat['vix_term_ratio'].pct_change(5)
        feat['vix_backwardation'] = (feat['vix_term_ratio'] > 1.0).astype(float)

    # Credit spread (4)
    if 'HYG' in prices.columns and 'IEF' in prices.columns:
        credit_spread = np.log(prices['HYG']) - np.log(prices['IEF'])
        feat['credit_spread'] = credit_spread
        feat['credit_spread_5d_chg'] = credit_spread.diff(5)
        feat['credit_spread_20d_chg'] = credit_spread.diff(20)
        feat['credit_zscore_20d'] = (credit_spread - credit_spread.rolling(20).mean()) / credit_spread.rolling(20).std()

    # SPY features (8)
    feat['spy_ret_5d'] = prices['SPY'].pct_change(5)
    feat['spy_ret_10d'] = prices['SPY'].pct_change(10)
    feat['spy_ret_20d'] = prices['SPY'].pct_change(20)
    feat['spy_vol_10d'] = prices['SPY'].pct_change().rolling(10).std() * np.sqrt(252)
    feat['spy_vol_20d'] = prices['SPY'].pct_change().rolling(20).std() * np.sqrt(252)
    feat['spy_vol_ratio'] = feat['spy_vol_10d'] / (feat['spy_vol_20d'] + 1e-8)
    feat['spy_ma_ratio_50_200'] = prices['SPY'].rolling(50).mean() / prices['SPY'].rolling(200).mean()
    feat['spy_drawdown'] = prices['SPY'] / prices['SPY'].rolling(252).max() - 1

    # Gold features (3)
    if 'GLD' in prices.columns:
        feat['gold_ret_5d'] = prices['GLD'].pct_change(5)
        feat['gold_ret_20d'] = prices['GLD'].pct_change(20)
        feat['gold_vol_10d'] = prices['GLD'].pct_change().rolling(10).std() * np.sqrt(252)

    # Silver features (2)
    if 'SLV' in prices.columns:
        feat['silver_ret_5d'] = prices['SLV'].pct_change(5)
        feat['gold_silver_ratio'] = prices['GLD'] / prices['SLV'] if 'GLD' in prices.columns else np.nan

    # Oil features (2)
    if 'USO' in prices.columns:
        feat['oil_ret_5d'] = prices['USO'].pct_change(5)
        feat['oil_ret_20d'] = prices['USO'].pct_change(20)

    # Dollar features (2)
    if 'UUP' in prices.columns:
        feat['dollar_ret_5d'] = prices['UUP'].pct_change(5)
        feat['dollar_ret_20d'] = prices['UUP'].pct_change(20)

    # Bond features (3)
    if 'TLT' in prices.columns:
        feat['bond_ret_5d'] = prices['TLT'].pct_change(5)
        feat['bond_ret_20d'] = prices['TLT'].pct_change(20)
        feat['bond_vol_10d'] = prices['TLT'].pct_change().rolling(10).std() * np.sqrt(252)

    # Copper features (2)
    if 'CPER' in prices.columns:
        feat['copper_ret_5d'] = prices['CPER'].pct_change(5)
        feat['copper_ret_20d'] = prices['CPER'].pct_change(20)

    # Bitcoin features (3)
    if 'BTC-USD' in prices.columns:
        feat['btc_ret_5d'] = prices['BTC-USD'].pct_change(5)
        feat['btc_ret_20d'] = prices['BTC-USD'].pct_change(20)
        feat['btc_vol_10d'] = prices['BTC-USD'].pct_change().rolling(10).std() * np.sqrt(252)

    # Cross-asset divergences (2)
    feat['spy_vix_diverge'] = feat.get('spy_ret_5d', 0) + feat.get('vix_5d_chg', 0)
    feat['gold_spy_diverge'] = feat.get('gold_ret_5d', 0) - feat.get('spy_ret_5d', 0)

    feat = feat.dropna()
    return feat


def build_target(prices, feat_index):
    """Target: VIX>30 at any point within next 5 days."""
    vix = prices['VIX'].reindex(feat_index)
    future_max = vix.rolling(5, min_periods=1).max().shift(-5)
    target = (future_max >= 30).astype(int)
    return target


def classify_risk(ablation_prob):
    """Classify risk level from ablation probability."""
    if ablation_prob < 0.15:
        return 'LOW'
    elif ablation_prob < 0.35:
        return 'MODERATE'
    elif ablation_prob < 0.60:
        return 'HIGH'
    else:
        return 'EXTREME'


def run():
    today_str = datetime.now().strftime('%Y-%m-%d')
    print(f"VIX Spike Watcher — {today_str}")
    print("=" * 50)

    # ── 1. Download data ─────────────────────────────────────────────
    print("\n[1/5] Downloading cross-asset data...")
    prices = download_data()
    print(f"  {len(prices)} rows, {prices.shape[1]} assets, "
          f"{prices.index[0].date()} to {prices.index[-1].date()}")

    # ── 2. Feature engineering ───────────────────────────────────────
    print("[2/5] Engineering features...")
    feat = engineer_features(prices)
    print(f"  {feat.shape[1]} features, {len(feat)} rows")

    # ── 3. Build target + train/predict split ────────────────────────
    print("[3/5] Building target & training models...")
    target = build_target(prices, feat.index)

    # Align features and target (drop rows where target is NaN = last 5 days)
    valid = target.dropna().index
    X_all = feat.loc[feat.index.isin(valid)]
    y_all = target.loc[X_all.index]

    # Walk-forward: train on everything MINUS last 252 days, predict on last day
    holdout = 252
    if len(X_all) <= holdout + 100:
        print("  ERROR: Not enough data for walk-forward split")
        sys.exit(1)

    X_train = X_all.iloc[:-holdout]
    y_train = y_all.iloc[:-holdout]

    # Today's features = last row of the FULL feature set (including days without target)
    today_features = feat.iloc[[-1]]
    today_date = feat.index[-1]

    vix_current = float(prices['VIX'].iloc[-1])

    print(f"  Training on {len(X_train)} days, predicting for {today_date.date()}")
    print(f"  VIX current: {vix_current:.2f}")
    print(f"  Spike rate in training: {y_train.mean():.1%}")

    # ── 4. Full model ────────────────────────────────────────────────
    print("[4/5] Training full model + ablation model...")
    model_full = GradientBoostingClassifier(**GBM_PARAMS)
    model_full.fit(X_train, y_train)
    spike_prob_full = float(model_full.predict_proba(today_features)[:, 1][0])

    # Feature importances (full model)
    full_importances = pd.Series(
        model_full.feature_importances_, index=X_train.columns
    ).sort_values(ascending=False)

    # Top 5 contributing features for today
    # Weight importance by feature z-score to show what's actually driving today's prediction
    today_vals = today_features.iloc[0]
    train_means = X_train.mean()
    train_stds = X_train.std().replace(0, 1)
    feature_zscores = ((today_vals - train_means) / train_stds).abs()
    feature_contribution = full_importances * feature_zscores
    top_5 = feature_contribution.sort_values(ascending=False).head(5)

    top_5_features = {}
    for fname, contrib in top_5.items():
        top_5_features[fname] = {
            'contribution_score': round(float(contrib), 4),
            'importance': round(float(full_importances[fname]), 4),
            'current_value': round(float(today_vals[fname]), 4),
            'zscore': round(float(feature_zscores[fname]), 2),
        }

    # ── 5. Ablation model (without raw VIX features) ────────────────
    ablation_cols = [c for c in X_train.columns if c not in RAW_VIX_COLS]
    X_train_ab = X_train[ablation_cols]
    today_features_ab = today_features[ablation_cols]

    model_ablation = GradientBoostingClassifier(**GBM_PARAMS)
    model_ablation.fit(X_train_ab, y_train)
    spike_prob_ablation = float(model_ablation.predict_proba(today_features_ab)[:, 1][0])

    risk_level = classify_risk(spike_prob_ablation)

    # ── Output ───────────────────────────────────────────────────────
    signal = {
        'date': today_str,
        'data_as_of': str(today_date.date()),
        'vix_current': round(vix_current, 2),
        'spike_prob_full': round(spike_prob_full, 4),
        'spike_prob_ablation': round(spike_prob_ablation, 4),
        'risk_level': risk_level,
        'top_5_features_contributing': make_serializable(top_5_features),
        'training_samples': len(X_train),
        'feature_count': feat.shape[1],
        'target': 'VIX>30 within 5 days',
        'updated_at': datetime.now().isoformat(),
    }

    with open(STATE_FILE, 'w') as f:
        json.dump(signal, f, indent=2)

    # Append to history CSV
    history_row = {
        'date': today_str,
        'vix_current': round(vix_current, 2),
        'spike_prob_full': round(spike_prob_full, 4),
        'spike_prob_ablation': round(spike_prob_ablation, 4),
        'risk_level': risk_level,
    }
    history_df = pd.DataFrame([history_row])

    if os.path.exists(HISTORY_FILE):
        existing = pd.read_csv(HISTORY_FILE)
        # Avoid duplicate dates
        existing = existing[existing['date'] != today_str]
        history_df = pd.concat([existing, history_df], ignore_index=True)

    history_df.to_csv(HISTORY_FILE, index=False)

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print(f"VIX SPIKE SIGNAL — {today_str}")
    print("=" * 50)
    print(f"  VIX Current:          {vix_current:.2f}")
    print(f"  Spike Prob (full):    {spike_prob_full:.1%}")
    print(f"  Spike Prob (ablation):{spike_prob_ablation:.1%}  ← genuine early warning")
    print(f"  Risk Level:           {risk_level}")
    print(f"\n  Top contributing features:")
    for fname, info in top_5_features.items():
        direction = "↑" if today_vals[fname] > train_means[fname] else "↓"
        print(f"    {fname:30s} z={info['zscore']:+.1f}{direction}  imp={info['importance']:.3f}")
    print(f"\n  State → {STATE_FILE}")
    print(f"  History → {HISTORY_FILE}")
    print("DONE.")

    return signal


if __name__ == '__main__':
    run()
