#!/usr/bin/env python3
"""
Daily Stock Prediction Scanner
================================
Runs the LGBM combined-signal model on today's data for the full 71-stock universe.
Outputs ranked predictions with confidence scores and key drivers.

Walk-forward: retrains on latest 252 trading days before predicting.
Runs daily at 4:30 PM ET after market close.

Usage:
    python daily_scanner.py              # Normal run
    python daily_scanner.py --backfill 5 # Backfill last 5 days
"""

import json
import os
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Ensure dependencies
for pkg in ["yfinance", "lightgbm"]:
    try:
        __import__(pkg)
    except ImportError:
        os.system(f"{sys.executable} -m pip install {pkg} -q")

import lightgbm as lgb
import yfinance as yf

# --- Paths ---
BASE_DIR = Path("/home/jupiter/Lvl3Quant")
OUTPUT_DIR = BASE_DIR / "output/growth_research/stock_prediction"
SCAN_DIR = OUTPUT_DIR / "daily_scans"
CACHE_DIR = OUTPUT_DIR / "cache"
PAPER_STATE_DIR = BASE_DIR / "data/paper_engines/options_prediction"

for d in [SCAN_DIR, CACHE_DIR, PAPER_STATE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- Universe ---
UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "AMD", "NFLX", "CRM",
    "SHOP", "SQ", "COIN", "PLTR", "SNOW", "DDOG", "NET", "CRWD", "ZS", "PANW",
    "AVGO", "MU", "QCOM", "INTC", "ORCL", "ADBE", "NOW", "UBER", "ABNB", "DASH",
    "MELI", "SE", "BABA", "JD", "PDD", "DIS", "CMCSA", "T", "VZ",
    "JPM", "GS", "MS", "BAC", "WFC", "V", "MA", "AXP",
    "UNH", "LLY", "PFE", "ABBV", "MRK", "JNJ",
    "XOM", "CVX", "COP",
    "LMT", "BA", "CAT", "DE",
    "HD", "LOW", "TGT", "WMT", "COST",
    "NKE", "SBUX", "MCD", "KO", "PEP",
]

# --- Feature columns (must match combined_signal_predictor.py) ---
FEATURE_COLS = [
    # Technical/Momentum
    'price_vs_sma50', 'price_vs_sma200', 'above_50ma', 'above_200ma',
    'ma_50_200_ratio', 'rsi_14',
    'mom_5d', 'mom_10d', 'mom_21d', 'mom_63d', 'mom_126d', 'mom_252d',
    'vol_21d', 'vol_63d',
    'vol_ratio_20d', 'vol_ratio_50d',
    'drawdown_from_high',
    'at_20d_high', 'at_50d_high',
    'zscore_20d',
    # Earnings/PEAD
    'days_since_earnings', 'earnings_surprise', 'earnings_gap',
    'post_earnings_drift', 'pead_signal', 'pead_strong', 'pead_negative',
    # Insider (filled with 0 if no data)
    'insider_buyers_30d', 'insider_buy_value_30d', 'insider_sell_value_30d',
    'insider_buy_ratio', 'insider_cluster', 'insider_cluster_strong',
]

# Top features by importance (from model training) for driver identification
TOP_DRIVERS = [
    ('mom_126d', 'Momentum (6-month)'),
    ('vol_63d', 'Volatility (3-month)'),
    ('ma_50_200_ratio', '50/200 MA Ratio'),
    ('earnings_gap', 'Earnings Gap'),
    ('drawdown_from_high', 'Drawdown from High'),
    ('rsi_14', 'RSI'),
    ('mom_63d', 'Momentum (3-month)'),
    ('price_vs_sma200', 'Price vs 200MA'),
    ('mom_21d', 'Momentum (1-month)'),
    ('days_since_earnings', 'Days Since Earnings'),
    ('post_earnings_drift', 'Post-Earnings Drift'),
    ('vol_ratio_20d', 'Volume Surge'),
    ('at_50d_high', 'At 50-Day High'),
    ('zscore_20d', 'Z-Score (20d)'),
    ('pead_signal', 'Earnings Drift Signal'),
]


def download_prices(lookback_days=400):
    """Download recent price data for the universe. Cache for 1 day."""
    cache_file = CACHE_DIR / "scanner_prices.parquet"
    if cache_file.exists():
        mod_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
        age_hours = (datetime.now() - mod_time).total_seconds() / 3600
        if age_hours < 2:  # Use cache if less than 2 hours old
            print(f"[SCANNER] Using cached price data ({age_hours:.1f}h old)")
            return pd.read_parquet(cache_file)

    print(f"[SCANNER] Downloading price data for {len(UNIVERSE)} stocks...")
    start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")

    all_frames = []
    batch_size = 20
    for i in range(0, len(UNIVERSE), batch_size):
        batch = UNIVERSE[i:i + batch_size]
        try:
            df = yf.download(batch, start=start_date, end=end_date,
                             progress=False, group_by='ticker', threads=True)
            if isinstance(df.columns, pd.MultiIndex):
                for ticker in batch:
                    if ticker in df.columns.get_level_values(0):
                        tdf = df[ticker].copy()
                        tdf['ticker'] = ticker
                        tdf = tdf.reset_index()
                        tdf.columns = [c if c != 'Date' else 'date' for c in tdf.columns]
                        all_frames.append(tdf)
            else:
                df = df.copy()
                df['ticker'] = batch[0]
                df = df.reset_index()
                df.columns = [c if c != 'Date' else 'date' for c in df.columns]
                all_frames.append(df)
        except Exception as e:
            print(f"[SCANNER] Error downloading {batch[:3]}...: {e}")
        time.sleep(0.3)

    if not all_frames:
        print("[SCANNER] ERROR: No price data downloaded!")
        return None

    prices = pd.concat(all_frames, ignore_index=True)
    col_map = {}
    for c in prices.columns:
        cl = c.lower()
        if cl in ('open', 'high', 'low', 'close', 'volume', 'adj close', 'date', 'ticker'):
            col_map[c] = cl.replace(' ', '_')
    prices = prices.rename(columns=col_map)
    prices['date'] = pd.to_datetime(prices['date'])
    prices = prices.dropna(subset=['close'])
    prices = prices.sort_values(['ticker', 'date']).reset_index(drop=True)

    prices.to_parquet(cache_file, index=False)
    print(f"[SCANNER] Downloaded {len(prices)} rows for {prices['ticker'].nunique()} tickers")
    return prices


def get_earnings_data():
    """Get earnings data from yfinance. Cache for 3 days."""
    cache_file = CACHE_DIR / "scanner_earnings.json"
    if cache_file.exists():
        mod_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if (datetime.now() - mod_time).days < 3:
            with open(cache_file) as f:
                return json.load(f)

    print("[SCANNER] Fetching earnings data...")
    earnings_data = {}
    for ticker in UNIVERSE:
        try:
            stock = yf.Ticker(ticker)
            cal = stock.get_earnings_dates(limit=20)
            if cal is not None and len(cal) > 0:
                records = []
                for idx, row in cal.iterrows():
                    rec = {'date': idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)}
                    if 'Surprise(%)' in row.index and pd.notna(row['Surprise(%)']):
                        rec['surprise_pct'] = float(row['Surprise(%)'])
                    if 'Reported EPS' in row.index and pd.notna(row['Reported EPS']):
                        rec['eps_actual'] = float(row['Reported EPS'])
                    records.append(rec)
                earnings_data[ticker] = records
        except Exception:
            pass
        time.sleep(0.1)

    with open(cache_file, 'w') as f:
        json.dump(earnings_data, f, indent=2)
    print(f"[SCANNER] Got earnings data for {len(earnings_data)} tickers")
    return earnings_data


def compute_technical_features(df):
    """Compute all technical features for a single stock's price series."""
    df = df.copy().sort_values('date')

    # Moving averages
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_200'] = df['close'].rolling(200).mean()

    # Price relative to MAs
    df['price_vs_sma50'] = df['close'] / df['sma_50'] - 1
    df['price_vs_sma200'] = df['close'] / df['sma_200'] - 1
    df['above_50ma'] = (df['close'] > df['sma_50']).astype(float)
    df['above_200ma'] = (df['close'] > df['sma_200']).astype(float)
    df['ma_50_200_ratio'] = df['sma_50'] / df['sma_200']

    # RSI (14-day)
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # Momentum
    for lb in [5, 10, 21, 63, 126, 252]:
        df[f'mom_{lb}d'] = df['close'].pct_change(lb)

    # Volatility
    df['vol_21d'] = df['close'].pct_change().rolling(21).std() * np.sqrt(252)
    df['vol_63d'] = df['close'].pct_change().rolling(63).std() * np.sqrt(252)

    # Volume features
    df['vol_ratio_20d'] = df['volume'] / df['volume'].rolling(20).mean()
    df['vol_ratio_50d'] = df['volume'] / df['volume'].rolling(50).mean()

    # Drawdown from 52-week high
    df['high_252'] = df['close'].rolling(252).max()
    df['drawdown_from_high'] = df['close'] / df['high_252'] - 1

    # Breakout
    df['at_20d_high'] = (df['close'] >= df['close'].rolling(20).max()).astype(float)
    df['at_50d_high'] = (df['close'] >= df['close'].rolling(50).max()).astype(float)

    # Z-score
    df['zscore_20d'] = (df['close'] - df['sma_20']) / df['close'].rolling(20).std()

    return df


def compute_earnings_features_for_ticker(ticker_prices, earnings_records):
    """Compute earnings-based features for a single ticker."""
    tdf = ticker_prices.set_index('date').sort_index()

    edates = []
    for rec in earnings_records:
        try:
            dt = pd.Timestamp(rec['date'])
            if pd.isna(dt):
                continue
            surprise = rec.get('surprise_pct', None)
            edates.append((dt, surprise))
        except Exception:
            continue

    if not edates:
        return pd.DataFrame(index=tdf.index)

    feat = pd.DataFrame(index=tdf.index)
    feat['days_since_earnings'] = np.nan
    feat['earnings_surprise'] = np.nan
    feat['earnings_gap'] = np.nan
    feat['post_earnings_drift'] = np.nan

    for edt, surprise in sorted(edates, key=lambda x: x[0]):
        mask = tdf.index >= edt
        if not mask.any():
            continue
        eday_idx = tdf.index[mask][0]
        eday_loc = tdf.index.get_loc(eday_idx)

        if eday_loc > 0:
            gap = tdf['close'].iloc[eday_loc] / tdf['close'].iloc[eday_loc - 1] - 1
        else:
            gap = 0

        for j in range(eday_loc, len(tdf)):
            d = tdf.index[j]
            days_since = (d - edt).days
            if days_since > 90:
                break
            feat.loc[d, 'days_since_earnings'] = days_since
            feat.loc[d, 'earnings_surprise'] = surprise if surprise is not None else np.nan
            feat.loc[d, 'earnings_gap'] = gap
            if eday_loc > 0:
                feat.loc[d, 'post_earnings_drift'] = tdf['close'].iloc[j] / tdf['close'].iloc[eday_loc - 1] - 1
            else:
                feat.loc[d, 'post_earnings_drift'] = 0

    # PEAD signals
    feat['pead_signal'] = 0.0
    m1 = (feat['earnings_surprise'] > 0) & (feat['earnings_gap'] > 0.02) & (feat['days_since_earnings'] <= 60)
    feat.loc[m1, 'pead_signal'] = 1.0

    feat['pead_strong'] = 0.0
    m2 = (feat['earnings_surprise'] > 5) & (feat['earnings_gap'] > 0.04) & (feat['days_since_earnings'] <= 60)
    feat.loc[m2, 'pead_strong'] = 1.0

    feat['pead_negative'] = 0.0
    m3 = (feat['earnings_surprise'] < -2) & (feat['earnings_gap'] < -0.03) & (feat['days_since_earnings'] <= 60)
    feat.loc[m3, 'pead_negative'] = 1.0

    return feat


def build_features(prices, earnings_data):
    """Build feature matrix for all stocks on all dates."""
    all_frames = []
    for ticker in prices['ticker'].unique():
        tdf = prices[prices['ticker'] == ticker].copy()
        tdf = compute_technical_features(tdf)
        tdf = tdf.set_index('date')

        # Earnings features
        if ticker in earnings_data:
            ef = compute_earnings_features_for_ticker(
                prices[prices['ticker'] == ticker], earnings_data[ticker]
            )
            for col in ef.columns:
                tdf[col] = ef[col]

        # Insider features (not available in real-time, fill with 0)
        for col in ['insider_buyers_30d', 'insider_buy_value_30d', 'insider_sell_value_30d',
                     'insider_buy_ratio', 'insider_cluster', 'insider_cluster_strong']:
            tdf[col] = 0.0

        tdf = tdf.reset_index()
        all_frames.append(tdf)

    panel = pd.concat(all_frames, ignore_index=True)

    # Fill NaN earnings features with 0
    earnings_cols = ['days_since_earnings', 'earnings_surprise', 'earnings_gap',
                     'post_earnings_drift', 'pead_signal', 'pead_strong', 'pead_negative']
    for col in earnings_cols:
        if col in panel.columns:
            panel[col] = panel[col].fillna(0)

    return panel


def compute_forward_returns(prices):
    """Compute forward 60d returns for target variable (training only)."""
    results = {}
    for ticker in prices['ticker'].unique():
        tdf = prices[prices['ticker'] == ticker].sort_values('date').set_index('date')
        fwd = pd.DataFrame(index=tdf.index)
        fwd['fwd_60d'] = tdf['close'].pct_change(60).shift(-60)
        fwd['target_up5_60d'] = (fwd['fwd_60d'] > 0.05).astype(float)
        results[ticker] = fwd
    return results


def identify_key_driver(row, avail_features):
    """Identify the primary driver behind a prediction."""
    # Check features in order of importance and look for strong signals
    drivers = []

    if 'earnings_gap' in avail_features and abs(row.get('earnings_gap', 0)) > 0.03:
        gap_pct = row['earnings_gap'] * 100
        if row.get('days_since_earnings', 999) < 30:
            drivers.append(f"Earnings ({gap_pct:+.1f}% gap, {int(row.get('days_since_earnings', 0))}d ago)")

    if 'pead_strong' in avail_features and row.get('pead_strong', 0) > 0.5:
        drivers.append("Strong Post-Earnings Drift")
    elif 'pead_signal' in avail_features and row.get('pead_signal', 0) > 0.5:
        drivers.append("Post-Earnings Drift")

    if 'mom_126d' in avail_features and abs(row.get('mom_126d', 0)) > 0.15:
        mom = row['mom_126d'] * 100
        drivers.append(f"6M Momentum ({mom:+.0f}%)")

    if 'at_50d_high' in avail_features and row.get('at_50d_high', 0) > 0.5:
        drivers.append("50-Day Breakout")

    if 'ma_50_200_ratio' in avail_features:
        ratio = row.get('ma_50_200_ratio', 1.0)
        if ratio > 1.05:
            drivers.append("Golden Cross (bullish trend)")
        elif ratio < 0.95:
            drivers.append("Death Cross (bearish trend)")

    if 'vol_ratio_20d' in avail_features and row.get('vol_ratio_20d', 1.0) > 2.0:
        drivers.append(f"Volume Surge ({row['vol_ratio_20d']:.1f}x avg)")

    if 'rsi_14' in avail_features:
        rsi = row.get('rsi_14', 50)
        if rsi > 70:
            drivers.append(f"RSI Overbought ({rsi:.0f})")
        elif rsi < 30:
            drivers.append(f"RSI Oversold ({rsi:.0f})")

    if 'drawdown_from_high' in avail_features:
        dd = row.get('drawdown_from_high', 0) * 100
        if dd < -20:
            drivers.append(f"Deep Pullback ({dd:.0f}% from high)")

    if not drivers:
        # Generic momentum description
        mom_21 = row.get('mom_21d', 0) * 100
        drivers.append(f"Momentum ({mom_21:+.1f}% 1M)")

    return " | ".join(drivers[:2])  # Max 2 drivers


def train_and_predict(panel, predict_date):
    """
    Walk-forward: train on 252 trading days ending before predict_date,
    then predict for predict_date.
    Returns dict of {ticker: (confidence, direction)} for the predict_date.
    """
    avail_features = [c for c in FEATURE_COLS if c in panel.columns]

    # Get forward returns for training data
    fwd_results = compute_forward_returns(
        panel[['date', 'ticker', 'close']].drop_duplicates()
    )

    # Add target to panel
    panel_with_target = panel.copy()
    panel_with_target['target_up5_60d'] = np.nan
    for ticker, fwd_df in fwd_results.items():
        mask = panel_with_target['ticker'] == ticker
        ticker_dates = panel_with_target.loc[mask, 'date']
        for d in ticker_dates:
            if d in fwd_df.index and pd.notna(fwd_df.loc[d, 'target_up5_60d']):
                panel_with_target.loc[
                    (panel_with_target['ticker'] == ticker) & (panel_with_target['date'] == d),
                    'target_up5_60d'
                ] = fwd_df.loc[d, 'target_up5_60d']

    # Training window: last 252 trading days before predict_date
    dates_before = sorted(panel_with_target[panel_with_target['date'] < predict_date]['date'].unique())
    if len(dates_before) < 252:
        print(f"[SCANNER] Not enough history for training ({len(dates_before)} days < 252)")
        # Use what we have if at least 100 days
        if len(dates_before) < 100:
            return {}, None
        train_dates = dates_before
    else:
        train_dates = dates_before[-252:]

    train_start, train_end = train_dates[0], train_dates[-1]

    train_df = panel_with_target[
        (panel_with_target['date'] >= train_start) &
        (panel_with_target['date'] <= train_end)
    ].dropna(subset=['target_up5_60d']).copy()

    # Fill NaN features
    for col in avail_features:
        train_df[col] = train_df[col].fillna(0)

    if len(train_df) < 500:
        print(f"[SCANNER] Not enough training data ({len(train_df)} rows)")
        return {}, None

    X_train = train_df[avail_features].values
    y_train = train_df['target_up5_60d'].values

    pos_rate = y_train.mean()
    if pos_rate < 0.01 or pos_rate > 0.99:
        print(f"[SCANNER] Bad target distribution (pos_rate={pos_rate:.3f})")
        return {}, None

    scale_pos = (1 - pos_rate) / pos_rate

    # Train LGBM
    model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        random_state=42,
        verbose=-1,
        n_jobs=4,
    )
    model.fit(X_train, y_train)

    # Predict for today
    today_df = panel[panel['date'] == predict_date].copy()
    if len(today_df) == 0:
        # Try the most recent date
        most_recent = panel['date'].max()
        today_df = panel[panel['date'] == most_recent].copy()
        if len(today_df) == 0:
            print("[SCANNER] No data for prediction date!")
            return {}, model

    for col in avail_features:
        today_df[col] = today_df[col].fillna(0)

    X_today = today_df[avail_features].values
    predictions = model.predict_proba(X_today)[:, 1]

    results = {}
    for i, (_, row) in enumerate(today_df.iterrows()):
        ticker = row['ticker']
        conf = predictions[i]
        driver = identify_key_driver(row, avail_features)
        price = row['close']
        # Direction: model predicts >5% up, so direction is always UP for positive predictions
        direction = "UP" if conf > 0.5 else "FLAT/DOWN"
        results[ticker] = {
            'confidence': float(conf),
            'direction': direction,
            'price': float(price),
            'driver': driver,
            'features': {feat: float(row.get(feat, 0)) for feat in avail_features[:10]},
        }

    # Feature importance
    feat_imp = dict(zip(avail_features, model.feature_importances_))

    print(f"[SCANNER] Trained on {len(train_df)} rows ({str(train_start)[:10]} to {str(train_end)[:10]})")
    print(f"[SCANNER] Training pos rate: {pos_rate:.3f}, predictions: {len(results)}")

    return results, model


def run_scan(scan_date=None):
    """Run the daily scan. Returns list of signals."""
    if scan_date is None:
        scan_date = datetime.now().strftime("%Y-%m-%d")

    print(f"\n{'='*70}")
    print(f"[SCANNER] Daily Stock Prediction Scan — {scan_date}")
    print(f"{'='*70}")

    # Download data
    prices = download_prices(lookback_days=400)
    if prices is None:
        print("[SCANNER] FAILED: No price data")
        return []

    earnings_data = get_earnings_data()

    # Build features
    panel = build_features(prices, earnings_data)

    # Train and predict
    predict_date = pd.Timestamp(scan_date)
    # Use the most recent trading day if scan_date is not in data
    available_dates = sorted(panel['date'].unique())
    if predict_date not in available_dates:
        # Find the most recent date <= predict_date
        past_dates = [d for d in available_dates if d <= predict_date]
        if not past_dates:
            print(f"[SCANNER] No data available for or before {scan_date}")
            return []
        predict_date = past_dates[-1]
        print(f"[SCANNER] Using most recent trading day: {str(predict_date)[:10]}")

    predictions, model = train_and_predict(panel, predict_date)

    if not predictions:
        print("[SCANNER] No predictions generated")
        return []

    # Rank by confidence, filter > 0.6
    signals = []
    for ticker, pred in sorted(predictions.items(), key=lambda x: x[1]['confidence'], reverse=True):
        conf = pred['confidence']
        if conf < 0.6:
            continue
        conviction = "HIGH CONVICTION" if conf >= 0.7 else "SIGNAL"
        signals.append({
            'ticker': ticker,
            'confidence': round(conf, 4),
            'conviction': conviction,
            'direction': pred['direction'],
            'price': round(pred['price'], 2),
            'driver': pred['driver'],
            'scan_date': str(predict_date)[:10],
        })

    # Print results
    print(f"\n{'='*70}")
    print(f"SCAN RESULTS — {str(predict_date)[:10]}")
    print(f"{'='*70}")

    if not signals:
        print("No stocks above 0.6 confidence threshold today.")
    else:
        print(f"\n{'Ticker':<8} {'Conf':>6} {'Level':<16} {'Price':>10} {'Dir':<6} {'Key Signal'}")
        print("-" * 80)
        for s in signals:
            print(f"{s['ticker']:<8} {s['confidence']:>6.1%} {s['conviction']:<16} "
                  f"${s['price']:>8.2f} {s['direction']:<6} {s['driver']}")

    print(f"\nTotal signals: {len(signals)} "
          f"({sum(1 for s in signals if s['conviction'] == 'HIGH CONVICTION')} high conviction)")

    # Save scan results
    scan_file = SCAN_DIR / f"scan_{str(predict_date)[:10]}.json"
    scan_output = {
        'scan_date': str(predict_date)[:10],
        'run_time': datetime.now().isoformat(),
        'universe_size': len(UNIVERSE),
        'total_signals': len(signals),
        'high_conviction': sum(1 for s in signals if s['conviction'] == 'HIGH CONVICTION'),
        'signals': signals,
        'all_predictions': {
            ticker: {
                'confidence': round(pred['confidence'], 4),
                'price': round(pred['price'], 2),
                'direction': pred['direction'],
            }
            for ticker, pred in sorted(predictions.items(), key=lambda x: x[1]['confidence'], reverse=True)
        },
    }

    with open(scan_file, 'w') as f:
        json.dump(scan_output, f, indent=2)
    print(f"\n[SCANNER] Results saved to {scan_file}")

    return signals


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Daily Stock Prediction Scanner")
    parser.add_argument('--backfill', type=int, default=0, help="Backfill N days")
    parser.add_argument('--date', type=str, default=None, help="Specific date to scan (YYYY-MM-DD)")
    args = parser.parse_args()

    if args.date:
        run_scan(args.date)
    elif args.backfill > 0:
        for i in range(args.backfill, 0, -1):
            d = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            run_scan(d)
    else:
        run_scan()


if __name__ == "__main__":
    main()
