#!/usr/bin/env python3
"""
Stat Arb Pairs Trading — Paper Engine
=======================================
Production cron script: runs daily at 16:45 ET (market close).
Downloads latest data, trains ML model on sliding 252-day window,
generates today's pair trade signals, saves state + appends to signals.csv.

Based on validated backtest: scripts/growth_research/ml_stat_arb.py
"""

import json
import os
import sys
import csv
import warnings
from datetime import datetime, timedelta
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')

# ─── Config ───
OUTPUT_DIR = Path('/home/jupiter/Lvl3Quant/output/ml_stat_arb_paper')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = OUTPUT_DIR / 'state.json'
SIGNALS_FILE = OUTPUT_DIR / 'signals.csv'

# Strategy parameters (from validated backtest)
TRAIN_WINDOW = 252
COINT_LOOKBACK = 126
ENTRY_Z = 1.5
EXIT_Z = 0.3
STOP_Z = 4.0
MAX_HOLD = 42
MAX_PAIRS = 8
POS_SIZE = 1.0 / MAX_PAIRS
COST_BPS = 10

# Universe
SECTOR_ETFS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLB', 'XLU', 'XLRE', 'XLC']
CROSS_ETFS = ['GLD', 'GDX', 'TLT', 'IEF', 'HYG', 'LQD', 'SPY', 'QQQ', 'IWM', 'EEM', 'DIA']
ALL_TICKERS = list(set(SECTOR_ETFS + CROSS_ETFS))

DATA_DAYS = 900  # Download last 900 calendar days (~630 trading days)


def load_state():
    """Load previous state or initialize."""
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        'positions': {},
        'trade_history': [],
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
            'date', 'pair', 'direction', 'signal_type', 'z_score',
            'hurst', 'half_life', 'ml_confidence', 'action'
        ])
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def download_data():
    """Download last 600 days of price data."""
    start = (datetime.now() - timedelta(days=DATA_DAYS)).strftime('%Y-%m-%d')
    print(f"Downloading {len(ALL_TICKERS)} tickers from {start}...")
    df = yf.download(ALL_TICKERS, start=start, progress=False)
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    close = df['Close'] if isinstance(df.columns, pd.MultiIndex) else df
    close = close.ffill()
    valid = close.columns[close.notna().sum() > 200]
    close = close[valid].dropna()
    print(f"  {len(close)} trading days, {len(close.columns)} assets")
    return close


def rolling_cointegration(price_a, price_b, window=COINT_LOOKBACK):
    """Test cointegration over rolling window. Returns (hurst, beta, half_life)."""
    if len(price_a) < window:
        return np.nan, np.nan, np.nan

    a = price_a.values[-window:]
    b = price_b.values[-window:]
    a_norm = a / a[0]
    b_norm = b / b[0]

    X = np.column_stack([b_norm, np.ones(window)])
    try:
        beta, alpha = np.linalg.lstsq(X, a_norm, rcond=None)[0]
    except Exception:
        return np.nan, np.nan, np.nan

    residual = a_norm - beta * b_norm - alpha
    if np.std(residual) < 1e-10:
        return np.nan, np.nan, np.nan

    residual_lag = residual[:-1]
    residual_diff = np.diff(residual)
    if np.std(residual_lag) < 1e-10:
        return np.nan, np.nan, np.nan

    slope = np.polyfit(residual_lag, residual_diff, 1)[0]
    if slope >= 0:
        return np.nan, np.nan, np.nan

    half_life = -np.log(2) / slope

    lags = range(2, min(20, window // 5))
    tau = []
    for lag in lags:
        tau.append(np.std(np.subtract(residual[lag:], residual[:-lag])))
    if len(tau) < 2 or any(t <= 0 for t in tau):
        return np.nan, beta, half_life

    try:
        hurst = np.polyfit(np.log(list(lags)), np.log(tau), 1)[0]
    except Exception:
        hurst = 0.5

    return hurst, beta, half_life


def build_pair_features(close, pair, lookback_end):
    """Build ML features for a pair at a given point in time."""
    a, b = pair
    window = close.iloc[max(0, lookback_end - 252):lookback_end]
    if len(window) < 126:
        return None

    pa, pb = window[a], window[b]
    hurst, beta, half_life = rolling_cointegration(pa, pb)
    if np.isnan(hurst) or np.isnan(beta):
        return None

    spread = pa / pa.iloc[0] - beta * (pb / pb.iloc[0])
    spread_mean = spread.rolling(63).mean()
    spread_std = spread.rolling(63).std()
    if spread_std.iloc[-1] < 1e-8:
        return None

    z = (spread.iloc[-1] - spread_mean.iloc[-1]) / spread_std.iloc[-1]
    z_series = (spread - spread_mean) / spread_std.clip(lower=1e-8)

    feats = {
        'z_score': z,
        'z_score_abs': abs(z),
        'hurst': hurst,
        'half_life': half_life,
        'beta': beta,
        'z_velocity_5d': (z_series.iloc[-1] - z_series.iloc[-6]) if len(z_series) > 5 else 0,
        'z_velocity_21d': (z_series.iloc[-1] - z_series.iloc[-22]) if len(z_series) > 21 else 0,
        'z_max_21d': z_series.iloc[-21:].max() if len(z_series) > 21 else z,
        'z_min_21d': z_series.iloc[-21:].min() if len(z_series) > 21 else z,
    }

    ret_a, ret_b = pa.pct_change(), pb.pct_change()
    feats['vol_a_21d'] = ret_a.iloc[-21:].std() * np.sqrt(252) if len(ret_a) > 21 else np.nan
    feats['vol_b_21d'] = ret_b.iloc[-21:].std() * np.sqrt(252) if len(ret_b) > 21 else np.nan
    feats['vol_ratio'] = feats['vol_a_21d'] / max(feats['vol_b_21d'], 1e-8)

    if len(ret_a) > 63:
        feats['corr_21d'] = ret_a.iloc[-21:].corr(ret_b.iloc[-21:])
        feats['corr_63d'] = ret_a.iloc[-63:].corr(ret_b.iloc[-63:])
        feats['corr_change'] = feats['corr_21d'] - feats['corr_63d']
    else:
        feats['corr_21d'] = feats['corr_63d'] = feats['corr_change'] = np.nan

    spread_ret = spread.diff()
    feats['spread_vol_21d'] = spread_ret.iloc[-21:].std() if len(spread_ret) > 21 else np.nan
    feats['spread_vol_63d'] = spread_ret.iloc[-63:].std() if len(spread_ret) > 63 else np.nan

    if len(z_series) > 42:
        z_21d_ago = z_series.iloc[-22]
        feats['recent_reversion'] = 1 - abs(z) / max(abs(z_21d_ago), 0.1)
    else:
        feats['recent_reversion'] = 0

    if 'SPY' in close.columns:
        spy_window = close['SPY'].iloc[max(0, lookback_end - 252):lookback_end]
        spy_ret = spy_window.pct_change()
        feats['spy_vol_21d'] = spy_ret.iloc[-21:].std() * np.sqrt(252) if len(spy_ret) > 21 else np.nan
        feats['spy_ret_21d'] = (spy_window.iloc[-1] / spy_window.iloc[-22] - 1) if len(spy_window) > 22 else 0

    return feats


def train_ml_model(close):
    """Train LightGBM model on historical pair reversion data."""
    try:
        import lightgbm as lgb
    except ImportError:
        print("WARNING: LightGBM not available, using z-score only mode")
        return None

    print("Training ML model on historical pair data...")
    assets = list(close.columns)
    all_pairs = list(combinations(assets, 2))
    n_days = len(close)

    train_X, train_y = [], []
    feature_names = None

    # Collect training labels from historical data
    start_day = max(TRAIN_WINDOW + COINT_LOOKBACK, 400)
    # Use the most recent TRAIN_WINDOW*2 days for training data collection
    sample_start = max(start_day, n_days - TRAIN_WINDOW * 2 - MAX_HOLD)
    sample_end = n_days - MAX_HOLD  # Need forward-looking labels

    for day in range(sample_start, sample_end, 5):  # Every 5 days to avoid over-sampling
        for pair in all_pairs:
            a, b = pair
            pa = close[a].iloc[max(0, day - COINT_LOOKBACK):day + 1]
            pb = close[b].iloc[max(0, day - COINT_LOOKBACK):day + 1]

            if len(pa) < 63:
                continue

            hurst, beta, half_life = rolling_cointegration(pa, pb)
            if np.isnan(hurst) or hurst > 0.45 or np.isnan(half_life) or half_life > 42 or half_life < 2:
                continue

            spread = pa / pa.iloc[0] - beta * (pb / pb.iloc[0])
            sp_mean = spread.rolling(63).mean().iloc[-1]
            sp_std = spread.rolling(63).std().iloc[-1]
            if sp_std < 1e-8:
                continue
            z = (spread.iloc[-1] - sp_mean) / sp_std
            if abs(z) < ENTRY_Z:
                continue

            feats = build_pair_features(close, pair, day + 1)
            if feats is None:
                continue

            # Forward-looking label: did spread revert?
            future_pa = close[a].iloc[day:day + MAX_HOLD + 1]
            future_pb = close[b].iloc[day:day + MAX_HOLD + 1]
            if len(future_pa) > 5:
                future_spread = future_pa / future_pa.iloc[0] - beta * (future_pb / future_pb.iloc[0])
                initial_dev = future_spread.iloc[0] - future_spread.mean()
                end_dev = future_spread.iloc[-1] - future_spread.mean()
                reverted = int(abs(end_dev) < abs(initial_dev) * 0.5)

                if feature_names is None:
                    feature_names = sorted(feats.keys())
                feat_vals = [feats.get(k, 0) for k in feature_names]
                train_X.append(feat_vals)
                train_y.append(reverted)

    if len(train_X) < 50:
        print(f"  Only {len(train_X)} training samples, insufficient for ML")
        return None

    X_arr = np.nan_to_num(np.array(train_X), 0)
    y_arr = np.array(train_y)

    print(f"  Training on {len(X_arr)} samples (revert rate: {y_arr.mean():.1%})")

    ds = lgb.Dataset(X_arr, label=y_arr, feature_name=feature_names)
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 15,
        'learning_rate': 0.05,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'verbose': -1,
        'n_jobs': -1,
    }
    model = lgb.train(params, ds, num_boost_round=100)
    print("  Model trained successfully")
    return model


def generate_signals(close, model, state):
    """Generate today's pair trade signals."""
    today = close.index[-1].strftime('%Y-%m-%d')
    assets = list(close.columns)
    all_pairs = list(combinations(assets, 2))
    n_days = len(close)
    day = n_days - 1  # Today

    signals = []
    feature_names = None

    # Check existing positions for exits
    positions = state.get('positions', {})
    exits = []
    for pair_key, pos in list(positions.items()):
        a, b = pair_key.split('/')
        if a not in close.columns or b not in close.columns:
            exits.append({'pair': pair_key, 'action': 'EXIT', 'reason': 'ticker_unavailable'})
            continue

        pa = close[a].iloc[max(0, day - 63):day + 1]
        pb = close[b].iloc[max(0, day - 63):day + 1]

        current_z = 0
        if len(pa) > 10:
            hurst, beta, hl = rolling_cointegration(pa, pb, min(63, len(pa)))
            if not np.isnan(beta):
                spread = pa.iloc[-1] / pa.iloc[0] - beta * (pb.iloc[-1] / pb.iloc[0])
                sp_series = pa / pa.iloc[0] - beta * (pb / pb.iloc[0])
                sp_mean = sp_series.mean()
                sp_std = sp_series.std()
                if sp_std > 1e-8:
                    current_z = (spread - sp_mean) / sp_std

        days_held = (pd.Timestamp(today) - pd.Timestamp(pos['entry_date'])).days

        exit_reason = None
        if abs(current_z) < EXIT_Z:
            exit_reason = 'reverted'
        elif pos['direction'] * current_z > STOP_Z:
            exit_reason = 'stop_loss'
        elif days_held >= MAX_HOLD:
            exit_reason = 'max_hold'

        if exit_reason:
            exits.append({
                'pair': pair_key,
                'action': 'EXIT',
                'reason': exit_reason,
                'z_score': round(current_z, 3),
                'days_held': days_held
            })
            append_signal({
                'date': today,
                'pair': pair_key,
                'direction': pos['direction'],
                'signal_type': 'EXIT',
                'z_score': round(current_z, 3),
                'hurst': '',
                'half_life': '',
                'ml_confidence': '',
                'action': f'EXIT ({exit_reason})'
            })
            del positions[pair_key]

    # Scan for new entries
    entries = []
    n_coint_pass = 0
    n_zscore_pass = 0
    if len(positions) < MAX_PAIRS:
        pair_candidates = []
        for pair in all_pairs:
            a, b = pair
            pair_key = f"{a}/{b}"
            if pair_key in positions:
                continue

            pa = close[a].iloc[max(0, day - COINT_LOOKBACK):day + 1]
            pb = close[b].iloc[max(0, day - COINT_LOOKBACK):day + 1]
            if len(pa) < 63:
                continue

            hurst, beta, half_life = rolling_cointegration(pa, pb)
            if np.isnan(hurst) or hurst > 0.45 or np.isnan(half_life) or half_life > 42 or half_life < 2:
                continue
            n_coint_pass += 1

            spread = pa / pa.iloc[0] - beta * (pb / pb.iloc[0])
            sp_mean = spread.rolling(63).mean().iloc[-1]
            sp_std = spread.rolling(63).std().iloc[-1]
            if sp_std < 1e-8:
                continue
            z = (spread.iloc[-1] - sp_mean) / sp_std
            if abs(z) < ENTRY_Z:
                continue
            n_zscore_pass += 1

            feats = build_pair_features(close, pair, day + 1)
            if feats is None:
                continue

            pair_candidates.append((pair, z, beta, hurst, half_life, feats))

        # Apply ML filter
        for pair, z, beta, hurst, half_life, feats in pair_candidates:
            if feature_names is None:
                feature_names = sorted(feats.keys())

            # HC #654: Adversarial validation showed ML overlay HURTS (Sharpe 0.36
            # vs 0.81 pure z-score). Use rules-only — ML confidence logged but NOT
            # used as gate. Pure z-score mean reversion is the validated alpha.
            ml_confidence = 0.5
            if model is not None:
                feat_vals = np.array([[feats.get(k, 0) for k in feature_names]])
                feat_vals = np.nan_to_num(feat_vals, 0)
                try:
                    ml_confidence = float(model.predict(feat_vals)[0])
                except Exception:
                    ml_confidence = 0.5

            # Always enter if z-score and cointegration criteria are met
            if True:  # was: ml_confidence > 0.5 — disabled per HC #654
                a, b = pair
                pair_key = f"{a}/{b}"
                direction = -1 if z > 0 else 1
                direction_label = 'LONG' if direction == 1 else 'SHORT'

                entry = {
                    'pair': pair_key,
                    'action': 'ENTER',
                    'direction': direction,
                    'direction_label': f"{direction_label} {a} / {'SHORT' if direction == 1 else 'LONG'} {b}",
                    'z_score': round(z, 3),
                    'hurst': round(hurst, 3),
                    'half_life': round(half_life, 1),
                    'ml_confidence': round(ml_confidence, 3)
                }
                entries.append(entry)

                positions[pair_key] = {
                    'direction': direction,
                    'entry_date': today,
                    'entry_z': round(z, 3),
                    'hurst': round(hurst, 3),
                    'half_life': round(half_life, 1),
                    'ml_confidence': round(ml_confidence, 3)
                }

                append_signal({
                    'date': today,
                    'pair': pair_key,
                    'direction': direction,
                    'signal_type': 'ENTER',
                    'z_score': round(z, 3),
                    'hurst': round(hurst, 3),
                    'half_life': round(half_life, 1),
                    'ml_confidence': round(ml_confidence, 3),
                    'action': entry['direction_label']
                })

                if len(positions) >= MAX_PAIRS:
                    break

    # Diagnostic: show how many pairs survived each filter
    print(f"\n  Pair scan: {len(all_pairs)} total, {n_coint_pass} pass cointegration, {n_zscore_pass} pass z-score, {len(entries)} entries")

    # Sort entries by z-score magnitude (strongest mean reversion first)
    entries.sort(key=lambda x: abs(x['z_score']), reverse=True)

    return exits, entries, positions


def main():
    print("=" * 60)
    print(f"STAT ARB PAIRS PAPER ENGINE — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # Load state
    state = load_state()
    print(f"Previous run: {state.get('last_run', 'never')}")
    print(f"Open positions: {len(state.get('positions', {}))}")

    # Download data
    close = download_data()
    if len(close) < TRAIN_WINDOW + COINT_LOOKBACK:
        print("ERROR: Insufficient data for analysis")
        sys.exit(1)

    # Train model
    model = train_ml_model(close)

    # Generate signals
    exits, entries, positions = generate_signals(close, model, state)

    # Update state
    today = close.index[-1].strftime('%Y-%m-%d')
    state['positions'] = positions
    state['last_run'] = datetime.now().isoformat()
    state['last_market_date'] = today
    state['total_signals'] += len(exits) + len(entries)

    # Print summary
    print(f"\n{'='*60}")
    print(f"DAILY SUMMARY — {today}")
    print(f"{'='*60}")

    if exits:
        print(f"\nEXITS ({len(exits)}):")
        for e in exits:
            print(f"  {e['pair']}: {e['reason']} (z={e.get('z_score', '?')})")

    if entries:
        print(f"\nENTRIES ({len(entries)}):")
        for e in entries:
            print(f"  {e['pair']}: {e['direction_label']}")
            print(f"    z={e['z_score']}, hurst={e['hurst']}, hl={e['half_life']}d, conf={e['ml_confidence']}")
    else:
        print("\nNo new entries today.")

    print(f"\nOPEN POSITIONS ({len(positions)}):")
    if positions:
        for pair_key, pos in positions.items():
            days_open = (pd.Timestamp(today) - pd.Timestamp(pos['entry_date'])).days
            print(f"  {pair_key}: dir={pos['direction']}, z_entry={pos['entry_z']}, days={days_open}")
    else:
        print("  (none)")

    print(f"\nTotal signals generated: {state['total_signals']}")

    # Save state
    save_state(state)
    print(f"\nState saved. Engine complete.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
