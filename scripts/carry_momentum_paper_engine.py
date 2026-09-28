#!/usr/bin/env python3
"""
Carry + Momentum Paper Engine
===============================
Production cron script: runs daily at 16:45 ET (market close).
Downloads latest data, trains LightGBM on sliding 252-day window,
predicts optimal allocation (income/growth/safety), saves state + signals.

Based on validated backtest: scripts/growth_research/ml_carry_momentum.py
"""

import json
import os
import sys
import csv
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')

# ─── Config ───
OUTPUT_DIR = Path('/home/jupiter/Lvl3Quant/output/ml_carry_momentum_paper')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = OUTPUT_DIR / 'state.json'
SIGNALS_FILE = OUTPUT_DIR / 'signals.csv'

# Strategy parameters (from validated backtest)
TRAIN_DAYS = 252
LABEL_HORIZON = 21  # 21-day forward prediction
DATA_DAYS = 900

# Universe
INCOME_TICKERS = ['SCHD', 'VYM', 'DVY']
GROWTH_TICKERS = ['QQQ', 'VGT', 'MTUM']
SAFETY_TICKERS = ['TLT', 'SHY', 'IEF']
ALL_TICKERS = INCOME_TICKERS + GROWTH_TICKERS + SAFETY_TICKERS
BENCHMARK = 'SPY'
REGIME_TICKERS = ['GLD', 'HYG', 'LQD', 'USO', 'DBC']

# All tickers to download
DOWNLOAD_TICKERS = ALL_TICKERS + [BENCHMARK] + REGIME_TICKERS + ['^VIX']


def load_state():
    """Load previous state or initialize."""
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        'current_allocation': None,
        'allocation_history': [],
        'last_run': None,
        'total_signals': 0
    }


def save_state(state):
    """Save state to disk."""
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, default=str)


def append_signal(row):
    """Append a signal row to signals.csv."""
    file_exists = SIGNALS_FILE.exists()
    with open(SIGNALS_FILE, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'date', 'predicted_category', 'income_weight', 'growth_weight',
            'safety_weight', 'confidence', 'top_features', 'action'
        ])
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def download_data():
    """Download last 600 days of price data for all tickers."""
    start = (datetime.now() - timedelta(days=DATA_DAYS)).strftime('%Y-%m-%d')
    print(f"Downloading {len(DOWNLOAD_TICKERS)} tickers from {start}...")

    data = {}
    for t in DOWNLOAD_TICKERS:
        try:
            df = yf.download(t, start=start, progress=False, auto_adjust=True)
            if len(df) > 100:
                if isinstance(df.columns, pd.MultiIndex):
                    close = df[('Close', t)].copy()
                else:
                    close = df['Close'].copy()
                clean = t.replace('^', '')
                close.name = clean
                data[clean] = close
        except Exception:
            pass

    prices = pd.DataFrame(data).dropna(how='all').ffill().dropna()
    print(f"  {len(prices)} trading days, {prices.shape[1]} assets")
    return prices


def build_features(prices_df, date_idx):
    """Build macro/regime features for allocation decision."""
    valid_income = [t for t in INCOME_TICKERS if t in prices_df.columns]
    valid_growth = [t for t in GROWTH_TICKERS if t in prices_df.columns]
    valid_all = valid_income + valid_growth + [t for t in SAFETY_TICKERS if t in prices_df.columns]

    feats = {}

    for ticker in valid_all + ['SPY']:
        if ticker not in prices_df.columns:
            continue
        p = prices_df[ticker].iloc[:date_idx + 1]
        if len(p) < 252:
            continue

        prefix = ticker.lower()

        # Momentum at multiple horizons
        for w in [5, 21, 63, 126, 252]:
            if len(p) > w:
                feats[f'{prefix}_ret_{w}d'] = (p.iloc[-1] / p.iloc[-w] - 1) if p.iloc[-w] > 0 else 0

        # Trend (price vs MA)
        for w in [50, 200]:
            if len(p) > w:
                ma = p.iloc[-w:].mean()
                feats[f'{prefix}_vs_ma{w}'] = (p.iloc[-1] / ma - 1) if ma > 0 else 0

        # Volatility
        rets = p.pct_change().dropna()
        if len(rets) > 21:
            feats[f'{prefix}_vol_21d'] = rets.iloc[-21:].std() * np.sqrt(252)

    # Cross-asset regime features
    for ticker in REGIME_TICKERS:
        if ticker not in prices_df.columns:
            continue
        p = prices_df[ticker].iloc[:date_idx + 1]
        if len(p) < 63:
            continue
        prefix = ticker.lower()
        feats[f'{prefix}_ret_21d'] = (p.iloc[-1] / p.iloc[-21] - 1) if p.iloc[-21] > 0 else 0
        feats[f'{prefix}_ret_63d'] = (p.iloc[-1] / p.iloc[-63] - 1) if p.iloc[-63] > 0 else 0

    # VIX
    if 'VIX' in prices_df.columns:
        vix = prices_df['VIX'].iloc[:date_idx + 1]
        if len(vix) > 63:
            feats['vix_level'] = vix.iloc[-1]
            vix_range = vix.iloc[-63:].max() - vix.iloc[-63:].min()
            feats['vix_pctile_63'] = (vix.iloc[-1] - vix.iloc[-63:].min()) / (vix_range + 1e-8)

    # Credit spread proxy (HYG vs LQD)
    if 'HYG' in prices_df.columns and 'LQD' in prices_df.columns:
        hyg = prices_df['HYG'].iloc[:date_idx + 1]
        lqd = prices_df['LQD'].iloc[:date_idx + 1]
        if len(hyg) > 21 and len(lqd) > 21:
            spread = hyg / lqd
            feats['credit_spread_21d_chg'] = (spread.iloc[-1] / spread.iloc[-21] - 1) if spread.iloc[-21] > 0 else 0

    # Income vs growth relative strength
    if valid_income and valid_growth:
        inc_ret = np.mean([(prices_df[t].iloc[date_idx] / prices_df[t].iloc[max(0, date_idx - 21)] - 1)
                           for t in valid_income if t in prices_df.columns])
        gro_ret = np.mean([(prices_df[t].iloc[date_idx] / prices_df[t].iloc[max(0, date_idx - 21)] - 1)
                           for t in valid_growth if t in prices_df.columns])
        feats['income_vs_growth_21d'] = inc_ret - gro_ret

    return feats


def train_and_predict(prices):
    """Train LightGBM on sliding window and predict today's allocation."""
    try:
        import lightgbm as lgb
        USE_LGB = True
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        USE_LGB = False
        print("WARNING: LightGBM not available, using sklearn GBM")

    from sklearn.preprocessing import LabelEncoder

    valid_income = [t for t in INCOME_TICKERS if t in prices.columns]
    valid_growth = [t for t in GROWTH_TICKERS if t in prices.columns]
    valid_safety = [t for t in SAFETY_TICKERS if t in prices.columns]

    if len(valid_income) == 0 or len(valid_growth) == 0 or len(valid_safety) == 0:
        print("ERROR: Missing tickers in at least one category")
        return None

    print(f"  Income: {valid_income}, Growth: {valid_growth}, Safety: {valid_safety}")

    # Build dataset: features + labels for training window
    print("Building feature dataset...")
    all_rows = []
    dates = prices.index.tolist()

    for i in range(TRAIN_DAYS, len(dates) - LABEL_HORIZON):
        feats = build_features(prices, i)
        if not feats:
            continue

        # Forward returns for each category
        inc_ret = np.mean([(prices[t].iloc[i + LABEL_HORIZON] / prices[t].iloc[i] - 1)
                           for t in valid_income])
        gro_ret = np.mean([(prices[t].iloc[i + LABEL_HORIZON] / prices[t].iloc[i] - 1)
                           for t in valid_growth])
        saf_ret = np.mean([(prices[t].iloc[i + LABEL_HORIZON] / prices[t].iloc[i] - 1)
                           for t in valid_safety])

        category_rets = {'income': inc_ret, 'growth': gro_ret, 'safety': saf_ret}
        best_cat = max(category_rets, key=category_rets.get)

        feats['_date'] = dates[i]
        feats['_best_category'] = best_cat
        all_rows.append(feats)

    if len(all_rows) < 50:
        print(f"ERROR: Only {len(all_rows)} training observations, need at least 50")
        return None

    df_all = pd.DataFrame(all_rows)
    feature_cols = [c for c in df_all.columns if not c.startswith('_')]
    print(f"  {len(df_all)} observations, {len(feature_cols)} features")
    print(f"  Category dist: {df_all['_best_category'].value_counts().to_dict()}")

    le = LabelEncoder()
    df_all['_label_encoded'] = le.fit_transform(df_all['_best_category'])

    # Train on last TRAIN_DAYS observations (sliding window)
    train_df = df_all.iloc[-TRAIN_DAYS:] if len(df_all) > TRAIN_DAYS else df_all

    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df['_label_encoded'].values

    n_classes = len(np.unique(y_train))
    if n_classes < 2:
        print("ERROR: Only one class in training data")
        return None

    print(f"  Training on {len(X_train)} samples...")

    if USE_LGB:
        model = lgb.LGBMClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_samples=20, verbose=-1, n_jobs=4
        )
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=20
        )

    model.fit(X_train, y_train)

    # Predict today's allocation
    today_feats = build_features(prices, len(prices) - 1)
    if not today_feats:
        print("ERROR: Could not build features for today")
        return None

    X_today = np.array([[today_feats.get(c, 0) for c in feature_cols]])
    X_today = np.nan_to_num(X_today, 0)

    pred_probs = model.predict_proba(X_today)[0]
    pred_class = model.predict(X_today)[0]
    pred_category = le.inverse_transform([pred_class])[0]

    # Build probability dict for each category
    class_names = le.classes_
    prob_dict = {}
    for i_cls, cls_name in enumerate(class_names):
        decoded = le.inverse_transform([cls_name])[0] if isinstance(cls_name, (int, np.integer)) else cls_name
        prob_dict[decoded] = float(pred_probs[i_cls])

    # Get feature importances
    if USE_LGB:
        importances = model.feature_importances_
    else:
        importances = model.feature_importances_

    top_feat_idx = np.argsort(importances)[-5:][::-1]
    top_features = [(feature_cols[i], round(float(importances[i]), 1)) for i in top_feat_idx]

    result = {
        'predicted_category': pred_category,
        'probabilities': prob_dict,
        'confidence': float(max(pred_probs)),
        'top_features': top_features,
        'n_training_samples': len(X_train),
        'n_classes': n_classes
    }

    print(f"  Prediction: {pred_category} (confidence: {max(pred_probs):.1%})")
    return result


def category_to_weights(category):
    """Convert predicted category to ETF weights."""
    if category == 'income':
        return {
            'income': 0.60,
            'growth': 0.20,
            'safety': 0.20
        }
    elif category == 'growth':
        return {
            'income': 0.20,
            'growth': 0.60,
            'safety': 0.20
        }
    elif category == 'safety':
        return {
            'income': 0.15,
            'growth': 0.15,
            'safety': 0.70
        }
    else:
        # Equal weight fallback
        return {'income': 0.33, 'growth': 0.34, 'safety': 0.33}


def main():
    print("=" * 60)
    print(f"CARRY + MOMENTUM PAPER ENGINE — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # Load state
    state = load_state()
    print(f"Previous run: {state.get('last_run', 'never')}")
    curr_alloc = state.get('current_allocation')
    prev_alloc = curr_alloc.get('category', 'none') if curr_alloc else 'none'
    print(f"Previous allocation: {prev_alloc}")

    # Download data
    prices = download_data()
    if len(prices) < TRAIN_DAYS + LABEL_HORIZON:
        print("ERROR: Insufficient data")
        sys.exit(1)

    # Train and predict
    prediction = train_and_predict(prices)
    if prediction is None:
        print("ERROR: Prediction failed")
        sys.exit(1)

    # Convert to weights
    category = prediction['predicted_category']
    weights = category_to_weights(category)
    today = prices.index[-1].strftime('%Y-%m-%d')

    # Determine action
    if prev_alloc == category:
        action = 'HOLD'
    else:
        action = f'ROTATE to {category.upper()}'

    # Build allocation details
    valid_income = [t for t in INCOME_TICKERS if t in prices.columns]
    valid_growth = [t for t in GROWTH_TICKERS if t in prices.columns]
    valid_safety = [t for t in SAFETY_TICKERS if t in prices.columns]

    positions = {}
    for t in valid_income:
        positions[t] = round(weights['income'] / len(valid_income), 4)
    for t in valid_growth:
        positions[t] = round(weights['growth'] / len(valid_growth), 4)
    for t in valid_safety:
        positions[t] = round(weights['safety'] / len(valid_safety), 4)

    # Update state
    allocation = {
        'date': today,
        'category': category,
        'weights': weights,
        'positions': positions,
        'confidence': prediction['confidence'],
        'probabilities': prediction['probabilities']
    }

    state['current_allocation'] = allocation
    state['last_run'] = datetime.now().isoformat()
    state['last_market_date'] = today
    state['total_signals'] += 1

    # Add to history (keep last 252)
    history = state.get('allocation_history', [])
    history.append({
        'date': today,
        'category': category,
        'confidence': prediction['confidence'],
        'action': action
    })
    state['allocation_history'] = history[-252:]

    # Append to signals.csv
    top_feats_str = '; '.join([f"{name}={imp}" for name, imp in prediction['top_features'][:3]])
    append_signal({
        'date': today,
        'predicted_category': category,
        'income_weight': weights['income'],
        'growth_weight': weights['growth'],
        'safety_weight': weights['safety'],
        'confidence': round(prediction['confidence'], 3),
        'top_features': top_feats_str,
        'action': action
    })

    # Print summary
    print(f"\n{'='*60}")
    print(f"DAILY SUMMARY — {today}")
    print(f"{'='*60}")
    print(f"\nPredicted regime: {category.upper()}")
    print(f"Confidence: {prediction['confidence']:.1%}")
    print(f"Action: {action}")
    print(f"\nCategory probabilities:")
    for cat, prob in sorted(prediction['probabilities'].items()):
        bar = '#' * int(prob * 30)
        print(f"  {cat:8s}: {prob:.1%} {bar}")

    print(f"\nTarget allocation:")
    print(f"  Income ({', '.join(valid_income)}): {weights['income']:.0%}")
    print(f"  Growth ({', '.join(valid_growth)}): {weights['growth']:.0%}")
    print(f"  Safety ({', '.join(valid_safety)}): {weights['safety']:.0%}")

    print(f"\nETF weights:")
    for ticker, weight in sorted(positions.items(), key=lambda x: -x[1]):
        print(f"  {ticker}: {weight:.1%}")

    print(f"\nTop features driving prediction:")
    for name, imp in prediction['top_features']:
        print(f"  {name}: importance={imp}")

    # Recent allocation history
    recent = state['allocation_history'][-5:]
    if len(recent) > 1:
        print(f"\nRecent allocations:")
        for h in recent:
            print(f"  {h['date']}: {h['category']} ({h['action']}, conf={h['confidence']:.1%})")

    print(f"\nTotal signals: {state['total_signals']}")

    # Save state
    save_state(state)
    print(f"\nState saved. Engine complete.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
