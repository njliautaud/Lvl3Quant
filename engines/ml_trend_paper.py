#!/usr/bin/env python3
"""
ML Trend Following v2 — PAPER ENGINE
=====================================
Runs daily at market close. Generates signals for next day's positions.
Based on the validated ML Trend Following v2 (Sharpe 2.90, 3/4 adversarial PASS).

Strategy: 8-asset CTA-style trend following with GBM ML filter.
- Generates dual-MA crossover signals for SPY/TLT/GLD/UUP/EEM/VNQ/HYG/XLE
- ML (GBM) trained on rolling 252d to predict trend continuation
- Only takes positions where ML confidence > 0.55
- Risk parity sizing (target 10% vol per position)

Paper tracking: logs positions, signals, and PnL to output/ml_trend_paper/
"""

import json
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import GradientBoostingClassifier

warnings.filterwarnings('ignore')

BASE = Path("/home/jupiter/Lvl3Quant")
OUTPUT = BASE / "output" / "ml_trend_paper"
OUTPUT.mkdir(parents=True, exist_ok=True)

POSITIONS_LOG = OUTPUT / "positions_log.json"
TRADES_LOG = OUTPUT / "trades_log.json"
DAILY_PNL = OUTPUT / "daily_pnl.csv"

# Strategy params (from validated backtest)
MA_SHORT = 20
MA_LONG = 100
ML_THRESHOLD = 0.55
TARGET_VOL = 0.10
TRAIN_WINDOW = 252
INITIAL_CAPITAL = 100_000
REBAL_COST_BPS = 10

UNIVERSE = {
    'SPY': 'US Equities',
    'TLT': 'Long Bonds',
    'GLD': 'Gold',
    'UUP': 'US Dollar',
    'EEM': 'Emerging Mkts',
    'VNQ': 'Real Estate',
    'HYG': 'High Yield',
    'XLE': 'Energy',
}


def get_data(lookback_days=400):
    """Download recent data for all assets."""
    tickers = list(UNIVERSE.keys()) + ['^VIX']
    start = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    closes = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw
    if '^VIX' in closes.columns:
        closes = closes.rename(columns={'^VIX': 'VIX'})
    closes = closes.ffill().dropna(thresh=len(closes.columns) - 3)
    return closes


def compute_trend_signals(df):
    """Compute dual-MA crossover signals."""
    signals = {}
    for ticker in UNIVERSE:
        if ticker not in df.columns:
            continue
        price = df[ticker]
        ma_short = price.rolling(MA_SHORT).mean()
        ma_long = price.rolling(MA_LONG).mean()
        trend = pd.Series(0.0, index=df.index)
        trend[ma_short > ma_long] = 1.0
        trend[ma_short < ma_long] = -1.0
        signals[ticker] = {
            'trend': trend,
            'ma_short': ma_short,
            'ma_long': ma_long,
            'price': price,
        }
    return signals


def build_features_for_day(df, signals, day_idx):
    """Build ML features for a single day across all assets."""
    rows = []
    for ticker, sig in signals.items():
        price = sig['price']
        trend = sig['trend']
        ma_s = sig['ma_short']
        ma_l = sig['ma_long']
        ret = price.pct_change()
        i = day_idx

        if trend.iloc[i] == 0:
            continue

        feat = {}
        feat['ticker'] = ticker
        feat['date'] = df.index[i]
        feat['direction'] = trend.iloc[i]

        # Trend strength
        feat['ma_dist'] = (ma_s.iloc[i] - ma_l.iloc[i]) / (price.iloc[i] + 1e-8)

        # Trend duration
        dur = 0
        for j in range(i, max(i - 252, 0), -1):
            if trend.iloc[j] == trend.iloc[i]:
                dur += 1
            else:
                break
        feat['trend_duration'] = dur

        # Momentum
        feat['mom_5d'] = price.pct_change(5).iloc[i]
        feat['mom_20d'] = price.pct_change(20).iloc[i]
        feat['mom_60d'] = price.pct_change(60).iloc[i]

        # Volatility
        feat['vol_20d'] = ret.iloc[max(0, i-20):i].std() * np.sqrt(252)
        feat['vol_60d'] = ret.iloc[max(0, i-60):i].std() * np.sqrt(252)

        # Drawdown
        peak = price.iloc[max(0, i-252):i+1].max()
        feat['drawdown'] = price.iloc[i] / peak - 1

        # Cross-asset alignment
        same_dir = sum(1 for t2, s2 in signals.items()
                      if t2 != ticker and s2['trend'].iloc[i] == trend.iloc[i])
        feat['cross_align'] = same_dir / max(len(signals) - 1, 1)

        # VIX
        if 'VIX' in df.columns:
            feat['vix'] = df['VIX'].iloc[i]
            feat['vix_ma20'] = df['VIX'].rolling(20).mean().iloc[i]

        # Skew/kurtosis
        if i > 20:
            feat['skew_20d'] = ret.iloc[i-20:i].skew()
            feat['kurt_20d'] = ret.iloc[i-20:i].kurt()

        rows.append(feat)
    return rows


def build_training_data(df, signals, end_idx, window=252):
    """Build training set using the last `window` days before end_idx."""
    rows = []
    start_idx = max(MA_LONG + 60, end_idx - window)

    for i in range(start_idx, end_idx):
        for ticker, sig in signals.items():
            price = sig['price']
            trend = sig['trend']
            ma_s = sig['ma_short']
            ma_l = sig['ma_long']
            ret = price.pct_change()

            if trend.iloc[i] == 0:
                continue

            feat = {}
            feat['ticker'] = ticker
            feat['direction'] = trend.iloc[i]
            feat['ma_dist'] = (ma_s.iloc[i] - ma_l.iloc[i]) / (price.iloc[i] + 1e-8)

            dur = 0
            for j in range(i, max(i - 252, 0), -1):
                if trend.iloc[j] == trend.iloc[i]:
                    dur += 1
                else:
                    break
            feat['trend_duration'] = dur
            feat['mom_5d'] = price.pct_change(5).iloc[i]
            feat['mom_20d'] = price.pct_change(20).iloc[i]
            feat['mom_60d'] = price.pct_change(60).iloc[i]
            feat['vol_20d'] = ret.iloc[max(0, i-20):i].std() * np.sqrt(252)
            feat['vol_60d'] = ret.iloc[max(0, i-60):i].std() * np.sqrt(252)
            peak = price.iloc[max(0, i-252):i+1].max()
            feat['drawdown'] = price.iloc[i] / peak - 1
            same_dir = sum(1 for t2, s2 in signals.items()
                          if t2 != ticker and s2['trend'].iloc[i] == trend.iloc[i])
            feat['cross_align'] = same_dir / max(len(signals) - 1, 1)
            if 'VIX' in df.columns:
                feat['vix'] = df['VIX'].iloc[i]
                feat['vix_ma20'] = df['VIX'].rolling(20).mean().iloc[i]
            if i > 20:
                feat['skew_20d'] = ret.iloc[i-20:i].skew()
                feat['kurt_20d'] = ret.iloc[i-20:i].kurt()

            # Target: trend continues over next 20d
            if i + 20 < len(df):
                future_ret = (price.iloc[i + 20] / price.iloc[i] - 1) * trend.iloc[i]
                feat['target'] = 1 if future_ret > 0 else 0
            else:
                feat['target'] = np.nan

            rows.append(feat)

    train_df = pd.DataFrame(rows).dropna(subset=['target'])
    return train_df


def generate_daily_signal():
    """Main: train ML on recent history, generate today's positions."""
    print(f"\n{'='*80}")
    print(f"ML TREND PAPER ENGINE — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*80}")

    # Get data
    df = get_data(lookback_days=500)
    print(f"Data: {df.shape[0]} days, {df.index[0].date()} → {df.index[-1].date()}")

    # Generate signals
    signals = compute_trend_signals(df)

    # Build training data (last 252 days before today, labeling goes back 20 more)
    today_idx = len(df) - 1
    train_df = build_training_data(df, signals, end_idx=today_idx - 20, window=TRAIN_WINDOW)

    feature_cols = [c for c in train_df.columns
                    if c not in ['ticker', 'date', 'direction', 'target']]

    print(f"Training samples: {len(train_df)} (positive rate: {train_df['target'].mean():.1%})")

    # Train GBM
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df['target'].astype(int)

    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    print(f"Train accuracy: {train_acc:.1%}")

    # Generate today's features and predictions
    today_features = build_features_for_day(df, signals, today_idx)

    if not today_features:
        print("No active signals today — all flat.")
        positions = {}
    else:
        pred_df = pd.DataFrame(today_features)
        X_today = pred_df[feature_cols].fillna(0)
        probs = model.predict_proba(X_today)[:, 1]
        pred_df['ml_prob'] = probs

        # Filter by threshold
        active = pred_df[pred_df['ml_prob'] >= ML_THRESHOLD].copy()
        print(f"\nSignals passing ML filter ({ML_THRESHOLD}): {len(active)}/{len(pred_df)}")

        # Risk parity sizing
        positions = {}
        for _, row in active.iterrows():
            ticker = row['ticker']
            vol = row['vol_20d'] if row['vol_20d'] > 0 else 0.15
            weight = TARGET_VOL / vol
            weight = min(weight, 0.25)  # Cap at 25% per position
            positions[ticker] = {
                'direction': int(row['direction']),
                'weight': round(weight, 4),
                'ml_confidence': round(row['ml_prob'], 4),
                'trend_duration': int(row['trend_duration']),
                'vol_20d': round(vol, 4),
            }

    # Normalize weights if total > 1
    total_weight = sum(abs(p['weight']) for p in positions.values())
    if total_weight > 1.0:
        for t in positions:
            positions[t]['weight'] = round(positions[t]['weight'] / total_weight, 4)

    # Log positions
    today_str = df.index[-1].strftime('%Y-%m-%d')
    log_entry = {
        'date': today_str,
        'timestamp': datetime.now().isoformat(),
        'positions': positions,
        'n_signals': len(today_features),
        'n_active': len(positions),
        'vix': float(df['VIX'].iloc[-1]) if 'VIX' in df.columns else None,
    }

    # Append to positions log
    if POSITIONS_LOG.exists():
        with open(POSITIONS_LOG, 'r') as f:
            history = json.load(f)
    else:
        history = []

    history.append(log_entry)
    with open(POSITIONS_LOG, 'w') as f:
        json.dump(history, f, indent=2, default=str)

    # Print summary
    print(f"\n{'─'*60}")
    print(f"TODAY'S POSITIONS ({today_str}):")
    print(f"{'─'*60}")
    if positions:
        for ticker, pos in sorted(positions.items(), key=lambda x: -abs(x[1]['weight'])):
            dir_str = "LONG" if pos['direction'] == 1 else "SHORT"
            print(f"  {ticker:5s} {dir_str:5s} weight={pos['weight']:.2%} "
                  f"conf={pos['ml_confidence']:.3f} dur={pos['trend_duration']}d "
                  f"vol={pos['vol_20d']:.1%}")
    else:
        print("  ALL CASH (no signals pass ML filter)")

    print(f"\nVIX: {log_entry['vix']:.1f}" if log_entry['vix'] else "")
    print(f"Net exposure: {sum(p['weight']*p['direction'] for p in positions.values()):.1%}")
    print(f"Gross exposure: {sum(abs(p['weight']) for p in positions.values()):.1%}")

    # Track PnL if we have prior day positions
    track_pnl(df, history)

    return positions


def track_pnl(df, history):
    """Calculate paper PnL from prior positions."""
    if len(history) < 2:
        return

    prev = history[-2]
    curr_date = df.index[-1]
    prev_date_str = prev['date']

    if not prev['positions']:
        return

    daily_ret = 0.0
    for ticker, pos in prev['positions'].items():
        if ticker not in df.columns:
            continue
        # Get return for today
        today_ret = df[ticker].pct_change().iloc[-1]
        position_ret = today_ret * pos['direction'] * pos['weight']
        daily_ret += position_ret

    # Subtract rebalance cost estimate
    daily_ret -= REBAL_COST_BPS / 10000

    # Append to PnL csv
    pnl_row = pd.DataFrame([{
        'date': curr_date.strftime('%Y-%m-%d'),
        'daily_return': daily_ret,
        'positions': json.dumps(prev['positions']),
    }])

    if DAILY_PNL.exists():
        existing = pd.read_csv(DAILY_PNL)
        pnl_df = pd.concat([existing, pnl_row], ignore_index=True)
    else:
        pnl_df = pnl_row

    pnl_df.to_csv(DAILY_PNL, index=False)

    # Summary stats
    if len(pnl_df) > 5:
        rets = pnl_df['daily_return'].astype(float)
        cum_ret = (1 + rets).prod() - 1
        sharpe = rets.mean() / (rets.std() + 1e-8) * np.sqrt(252)
        print(f"\n  Paper PnL ({len(pnl_df)} days): cumulative {cum_ret:.2%}, Sharpe {sharpe:.2f}")


if __name__ == '__main__':
    positions = generate_daily_signal()
