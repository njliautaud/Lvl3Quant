#!/usr/bin/env python3
"""
Combined Signal Stock Prediction Model
========================================
Can we predict which stocks will move >5% in 30-60 days accurately enough
to make buying options profitable?

Signals combined:
  1. Insider Buying Clusters (SEC EDGAR Form 4)
  2. Post-Earnings Drift (PEAD)
  3. Momentum + Breakout (technical)
  4. Fundamental Quality (earnings growth via yfinance)

Walk-forward: 252-day train, 63-day OOT, SLIDING window.
Target: binary classification — stock up >5% in next 60 trading days.
Key metric: PRECISION on the positive class (we want to be RIGHT when we bet).

Options profitability simulation uses Black-Scholes for ATM calls, 45 DTE.
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
from scipy.stats import norm

warnings.filterwarnings("ignore")

# --- Ensure dependencies ---
for pkg in ["yfinance", "lightgbm", "sklearn"]:
    try:
        __import__(pkg)
    except ImportError:
        os.system(f"{sys.executable} -m pip install {pkg} -q")

import lightgbm as lgb
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler

# --- Paths ---
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research/stock_prediction")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
INSIDER_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research/insider_alpha")
CACHE_DIR = OUTPUT_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# --- Universe ---
UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "AMD", "NFLX", "CRM",
    "SHOP", "SQ", "COIN", "PLTR", "SNOW", "DDOG", "NET", "CRWD", "ZS", "PANW",
    "AVGO", "MU", "QCOM", "INTC", "ORCL", "ADBE", "NOW", "UBER", "ABNB", "DASH",
    "MELI", "SE", "BABA", "JD", "PDD", "DIS", "CMCSA", "T", "VZ",
    "JPM", "GS", "MS", "BAC", "WFC", "V", "MA", "AXP", "BRK-B",
    "UNH", "LLY", "PFE", "ABBV", "MRK", "JNJ",
    "XOM", "CVX", "COP",
    "LMT", "BA", "CAT", "DE",
    "HD", "LOW", "TGT", "WMT", "COST",
    "NKE", "SBUX", "MCD", "KO", "PEP",
]

# --- Black-Scholes ---
def bs_call(S, K, T, r, sigma):
    if T <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_put(S, K, T, r, sigma):
    if T <= 0:
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# ============================================================================
# PHASE 1: DATA COLLECTION
# ============================================================================

def download_price_data(use_cache=True):
    """Download 5 years of daily price data for all universe stocks + SPY."""
    cache_file = CACHE_DIR / "price_data.parquet"
    if use_cache and cache_file.exists():
        mod_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if (datetime.now() - mod_time).days < 3:
            print(f"Loading cached price data from {cache_file}")
            return pd.read_parquet(cache_file)

    print("Downloading price data from yfinance...")
    all_tickers = UNIVERSE + ["SPY"]
    start_date = "2019-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")

    all_frames = []
    batch_size = 20
    for i in range(0, len(all_tickers), batch_size):
        batch = all_tickers[i:i+batch_size]
        print(f"  Downloading batch {i//batch_size + 1}: {batch[:5]}...")
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
                # Single ticker
                df = df.copy()
                df['ticker'] = batch[0]
                df = df.reset_index()
                df.columns = [c if c != 'Date' else 'date' for c in df.columns]
                all_frames.append(df)
        except Exception as e:
            print(f"  Error downloading batch: {e}")
        time.sleep(0.5)

    prices = pd.concat(all_frames, ignore_index=True)
    # Standardize column names
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
    print(f"Saved {len(prices)} rows for {prices['ticker'].nunique()} tickers")
    return prices


def get_earnings_data(tickers, use_cache=True):
    """Get earnings dates and surprise data from yfinance."""
    cache_file = CACHE_DIR / "earnings_data.json"
    if use_cache and cache_file.exists():
        mod_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if (datetime.now() - mod_time).days < 7:
            print(f"Loading cached earnings data")
            with open(cache_file) as f:
                return json.load(f)

    print("Fetching earnings data from yfinance...")
    earnings_data = {}
    for i, ticker in enumerate(tickers):
        if i % 10 == 0:
            print(f"  Processing {i}/{len(tickers)}...")
        try:
            stock = yf.Ticker(ticker)
            # Get earnings dates
            try:
                cal = stock.get_earnings_dates(limit=40)
                if cal is not None and len(cal) > 0:
                    records = []
                    for idx, row in cal.iterrows():
                        rec = {
                            'date': idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx),
                        }
                        if 'Surprise(%)' in row.index and pd.notna(row['Surprise(%)']):
                            rec['surprise_pct'] = float(row['Surprise(%)'])
                        if 'EPS Estimate' in row.index and pd.notna(row['EPS Estimate']):
                            rec['eps_estimate'] = float(row['EPS Estimate'])
                        if 'Reported EPS' in row.index and pd.notna(row['Reported EPS']):
                            rec['eps_actual'] = float(row['Reported EPS'])
                        records.append(rec)
                    earnings_data[ticker] = records
            except Exception:
                pass
        except Exception as e:
            pass
        time.sleep(0.15)

    with open(cache_file, 'w') as f:
        json.dump(earnings_data, f, indent=2)
    print(f"Got earnings data for {len(earnings_data)} tickers")
    return earnings_data


def load_insider_data():
    """Load cached insider transaction data."""
    insider_file = INSIDER_DIR / "insider_transactions_all.parquet"
    if insider_file.exists():
        df = pd.read_parquet(insider_file)
        print(f"Loaded {len(df)} insider transactions")
        return df
    print("WARNING: No cached insider data found")
    return pd.DataFrame()


# ============================================================================
# PHASE 2: FEATURE ENGINEERING
# ============================================================================

def compute_technical_features(prices_group):
    """Compute momentum/breakout features for a single stock's price series."""
    df = prices_group.copy().sort_values('date')

    # Moving averages
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_200'] = df['close'].rolling(200).mean()

    # Price relative to MAs
    df['price_vs_sma50'] = df['close'] / df['sma_50'] - 1
    df['price_vs_sma200'] = df['close'] / df['sma_200'] - 1
    df['above_50ma'] = (df['close'] > df['sma_50']).astype(float)
    df['above_200ma'] = (df['close'] > df['sma_200']).astype(float)

    # Golden/death cross
    df['ma_50_200_ratio'] = df['sma_50'] / df['sma_200']

    # RSI (14-day)
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # Momentum (various lookbacks)
    for lb in [5, 10, 21, 63, 126, 252]:
        df[f'mom_{lb}d'] = df['close'].pct_change(lb)

    # Volatility (realized)
    df['vol_21d'] = df['close'].pct_change().rolling(21).std() * np.sqrt(252)
    df['vol_63d'] = df['close'].pct_change().rolling(63).std() * np.sqrt(252)

    # Volume features
    df['vol_ratio_20d'] = df['volume'] / df['volume'].rolling(20).mean()
    df['vol_ratio_50d'] = df['volume'] / df['volume'].rolling(50).mean()

    # Drawdown from 52-week high
    df['high_252'] = df['close'].rolling(252).max()
    df['drawdown_from_high'] = df['close'] / df['high_252'] - 1

    # Breakout: close at N-day high
    df['at_20d_high'] = (df['close'] >= df['close'].rolling(20).max()).astype(float)
    df['at_50d_high'] = (df['close'] >= df['close'].rolling(50).max()).astype(float)

    # Mean reversion: distance from 20-day mean
    df['zscore_20d'] = (df['close'] - df['sma_20']) / df['close'].rolling(20).std()

    return df


def compute_earnings_features(prices, earnings_data):
    """Compute PEAD features for each stock-date."""
    print("Computing earnings features...")
    results = {}

    for ticker in prices['ticker'].unique():
        if ticker not in earnings_data:
            continue

        tdf = prices[prices['ticker'] == ticker].set_index('date').sort_index()
        earnings_records = earnings_data[ticker]

        # Convert earnings dates
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
            continue

        # For each trading day, compute features relative to most recent earnings
        feat = pd.DataFrame(index=tdf.index)
        feat['days_since_earnings'] = np.nan
        feat['earnings_surprise'] = np.nan
        feat['earnings_gap'] = np.nan  # 1-day return on earnings day
        feat['post_earnings_drift'] = np.nan  # cumulative return since earnings

        for edt, surprise in sorted(edates, key=lambda x: x[0]):
            # Find the actual trading day on or after earnings
            mask = tdf.index >= edt
            if not mask.any():
                continue
            eday_idx = tdf.index[mask][0]
            eday_loc = tdf.index.get_loc(eday_idx)

            # Compute gap (1-day return on earnings)
            if eday_loc > 0:
                gap = tdf['close'].iloc[eday_loc] / tdf['close'].iloc[eday_loc - 1] - 1
            else:
                gap = 0

            # For all days AFTER this earnings until next earnings
            for j in range(eday_loc, len(tdf)):
                d = tdf.index[j]
                days_since = (d - edt).days
                if days_since > 90:  # Stop after 90 days
                    break
                feat.loc[d, 'days_since_earnings'] = days_since
                feat.loc[d, 'earnings_surprise'] = surprise if surprise is not None else np.nan
                feat.loc[d, 'earnings_gap'] = gap
                # Drift since earnings
                if eday_loc > 0:
                    feat.loc[d, 'post_earnings_drift'] = tdf['close'].iloc[j] / tdf['close'].iloc[eday_loc - 1] - 1
                else:
                    feat.loc[d, 'post_earnings_drift'] = 0

        # PEAD signal: positive surprise + positive gap
        feat['pead_signal'] = 0.0
        mask = (feat['earnings_surprise'] > 0) & (feat['earnings_gap'] > 0.02) & (feat['days_since_earnings'] <= 60)
        feat.loc[mask, 'pead_signal'] = 1.0

        # Strong PEAD: big beat + big gap
        feat['pead_strong'] = 0.0
        mask2 = (feat['earnings_surprise'] > 5) & (feat['earnings_gap'] > 0.04) & (feat['days_since_earnings'] <= 60)
        feat.loc[mask2, 'pead_strong'] = 1.0

        # Negative PEAD (for puts/shorts)
        feat['pead_negative'] = 0.0
        mask3 = (feat['earnings_surprise'] < -2) & (feat['earnings_gap'] < -0.03) & (feat['days_since_earnings'] <= 60)
        feat.loc[mask3, 'pead_negative'] = 1.0

        results[ticker] = feat

    print(f"  Computed earnings features for {len(results)} tickers")
    return results


def compute_insider_features(prices, insider_df):
    """Compute insider buying cluster features."""
    print("Computing insider features...")
    if insider_df.empty:
        print("  No insider data — skipping")
        return {}

    insider_df = insider_df.copy()
    insider_df['transaction_date'] = pd.to_datetime(insider_df['transaction_date'])
    insider_df['filing_date'] = pd.to_datetime(insider_df['filing_date'])

    results = {}
    for ticker in prices['ticker'].unique():
        tdf = prices[prices['ticker'] == ticker].set_index('date').sort_index()
        idf = insider_df[insider_df['ticker'] == ticker].copy()

        if idf.empty:
            continue

        # Separate buys and sells
        buys = idf[idf['type'].str.lower().str.contains('purchase|buy|p-purchase', na=False)]
        sells = idf[idf['type'].str.lower().str.contains('sale|sell|s-sale', na=False)]

        feat = pd.DataFrame(index=tdf.index)

        for d in tdf.index:
            d_ts = pd.Timestamp(d)
            # Look back 30 days for cluster detection
            window_start = d_ts - pd.Timedelta(days=30)
            recent_buys = buys[(buys['filing_date'] >= window_start) & (buys['filing_date'] <= d_ts)]
            recent_sells = sells[(sells['filing_date'] >= window_start) & (sells['filing_date'] <= d_ts)]

            # Cluster score: number of unique insider buyers in 30d
            n_buyers = recent_buys['reporter'].nunique() if len(recent_buys) > 0 else 0
            feat.loc[d, 'insider_buyers_30d'] = n_buyers
            feat.loc[d, 'insider_buy_value_30d'] = recent_buys['value'].sum() if len(recent_buys) > 0 else 0
            feat.loc[d, 'insider_sell_value_30d'] = recent_sells['value'].sum() if len(recent_sells) > 0 else 0

            total_txns = len(recent_buys) + len(recent_sells)
            feat.loc[d, 'insider_buy_ratio'] = len(recent_buys) / max(total_txns, 1)

            # Cluster signal: 3+ insiders buying within 30 days
            feat.loc[d, 'insider_cluster'] = 1.0 if n_buyers >= 3 else 0.0

            # Strong cluster: 3+ buyers AND >$200K total
            feat.loc[d, 'insider_cluster_strong'] = 1.0 if (n_buyers >= 3 and
                feat.loc[d, 'insider_buy_value_30d'] > 200000) else 0.0

        results[ticker] = feat

    print(f"  Computed insider features for {len(results)} tickers")
    return results


def compute_forward_returns(prices):
    """Compute forward 30d and 60d returns for target variable."""
    print("Computing forward returns...")
    results = {}
    for ticker in prices['ticker'].unique():
        tdf = prices[prices['ticker'] == ticker].sort_values('date').set_index('date')

        fwd = pd.DataFrame(index=tdf.index)
        fwd['fwd_30d'] = tdf['close'].pct_change(30).shift(-30)
        fwd['fwd_60d'] = tdf['close'].pct_change(60).shift(-60)

        # Binary targets
        fwd['target_up5_30d'] = (fwd['fwd_30d'] > 0.05).astype(float)
        fwd['target_up5_60d'] = (fwd['fwd_60d'] > 0.05).astype(float)
        fwd['target_up10_60d'] = (fwd['fwd_60d'] > 0.10).astype(float)
        fwd['target_down5_60d'] = (fwd['fwd_60d'] < -0.05).astype(float)

        results[ticker] = fwd

    return results


def build_master_panel(prices, earnings_features, insider_features, forward_returns):
    """Combine all features into a single panel DataFrame."""
    print("Building master panel...")
    all_frames = []

    for ticker in prices['ticker'].unique():
        tdf = prices[prices['ticker'] == ticker].copy()
        tdf = compute_technical_features(tdf)
        tdf = tdf.set_index('date')

        # Merge earnings features
        if ticker in earnings_features:
            ef = earnings_features[ticker]
            for col in ef.columns:
                tdf[col] = ef[col]

        # Merge insider features
        if ticker in insider_features:
            inf = insider_features[ticker]
            for col in inf.columns:
                tdf[col] = inf[col]

        # Merge forward returns
        if ticker in forward_returns:
            fr = forward_returns[ticker]
            for col in fr.columns:
                tdf[col] = fr[col]

        tdf = tdf.reset_index()
        all_frames.append(tdf)

    panel = pd.concat(all_frames, ignore_index=True)

    # Fill NaN insider/earnings features with 0 (no signal = no activity)
    insider_cols = ['insider_buyers_30d', 'insider_buy_value_30d', 'insider_sell_value_30d',
                    'insider_buy_ratio', 'insider_cluster', 'insider_cluster_strong']
    earnings_cols = ['days_since_earnings', 'earnings_surprise', 'earnings_gap',
                     'post_earnings_drift', 'pead_signal', 'pead_strong', 'pead_negative']

    for col in insider_cols + earnings_cols:
        if col in panel.columns:
            panel[col] = panel[col].fillna(0)

    panel = panel.sort_values(['date', 'ticker']).reset_index(drop=True)
    print(f"Master panel: {len(panel)} rows, {len(panel.columns)} columns")
    print(f"Date range: {panel['date'].min()} to {panel['date'].max()}")
    print(f"Tickers: {panel['ticker'].nunique()}")

    return panel


# ============================================================================
# PHASE 2B: INDIVIDUAL SIGNAL TESTING
# ============================================================================

def test_individual_signals(panel):
    """Test each signal independently for predictive power."""
    print("\n" + "="*80)
    print("PHASE 2: INDIVIDUAL SIGNAL TESTING")
    print("="*80)

    results = {}

    # Unconditional base rates
    for target in ['target_up5_30d', 'target_up5_60d', 'target_up10_60d']:
        if target not in panel.columns:
            continue
        valid = panel[target].dropna()
        base_rate = valid.mean()
        print(f"\nBase rate for {target}: {base_rate:.3f} ({base_rate*100:.1f}%)")
        results[f'base_rate_{target}'] = base_rate

    # Test signals
    signals_to_test = {
        'above_200ma': ('Above 200-day MA', 'above_200ma'),
        'golden_cross': ('50MA > 200MA (golden cross)', 'ma_50_200_ratio'),
        'rsi_50_70': ('RSI between 50-70', 'rsi_14'),
        'pead_signal': ('Post-Earnings Drift (positive)', 'pead_signal'),
        'pead_strong': ('Strong PEAD (big beat)', 'pead_strong'),
        'insider_cluster': ('Insider cluster (3+ buyers)', 'insider_cluster'),
        'insider_cluster_strong': ('Strong insider cluster', 'insider_cluster_strong'),
        'breakout_50d': ('At 50-day high', 'at_50d_high'),
        'momentum_63d': ('Positive 63d momentum', 'mom_63d'),
    }

    target = 'target_up5_60d'
    if target not in panel.columns:
        print("Target column missing!")
        return results

    base_rate = panel[target].dropna().mean()

    print(f"\n{'Signal':<40} {'When ON':>10} {'When OFF':>10} {'Lift':>8} {'N signals':>10}")
    print("-" * 80)

    for name, (desc, col) in signals_to_test.items():
        if col not in panel.columns:
            print(f"  {desc:<40} {'MISSING':>10}")
            continue

        valid = panel.dropna(subset=[target, col])

        if col == 'rsi_14':
            on_mask = (valid[col] >= 50) & (valid[col] <= 70)
        elif col == 'ma_50_200_ratio':
            on_mask = valid[col] > 1.0
        elif col == 'mom_63d':
            on_mask = valid[col] > 0
        else:
            on_mask = valid[col] > 0.5

        rate_on = valid.loc[on_mask, target].mean()
        rate_off = valid.loc[~on_mask, target].mean()
        n_on = on_mask.sum()
        lift = rate_on / base_rate if base_rate > 0 else 0

        print(f"  {desc:<40} {rate_on:>9.3f} {rate_off:>9.3f} {lift:>7.2f}x {n_on:>10,}")
        results[name] = {
            'rate_on': rate_on, 'rate_off': rate_off,
            'lift': lift, 'n_signals': int(n_on)
        }

    return results


# ============================================================================
# PHASE 3: WALK-FORWARD MODEL
# ============================================================================

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
    # Insider
    'insider_buyers_30d', 'insider_buy_value_30d', 'insider_sell_value_30d',
    'insider_buy_ratio', 'insider_cluster', 'insider_cluster_strong',
]


def walk_forward_model(panel, target_col='target_up5_60d',
                       train_days=252, test_days=21,
                       model_type='lgbm'):
    """
    Walk-forward sliding-window model training.
    Train on 252 trading days, test on next 63 days, slide forward.
    """
    print(f"\n{'='*80}")
    print(f"PHASE 3: WALK-FORWARD MODEL ({model_type.upper()})")
    print(f"Target: {target_col}, Train: {train_days}d, Test: {test_days}d")
    print(f"{'='*80}")

    # Get available features
    avail_features = [c for c in FEATURE_COLS if c in panel.columns]
    print(f"Using {len(avail_features)} features: {avail_features[:10]}...")

    # Get unique dates sorted
    dates = sorted(panel['date'].unique())
    print(f"Total dates: {len(dates)}")

    # Drop rows with NaN target
    valid_panel = panel.dropna(subset=[target_col]).copy()
    # Fill NaN features with 0
    for col in avail_features:
        valid_panel[col] = valid_panel[col].fillna(0)

    # Collect OOT predictions
    all_oot_preds = []
    fold_results = []

    fold_idx = 0
    start = 0

    while start + train_days + test_days <= len(dates):
        train_dates = dates[start:start + train_days]
        test_dates = dates[start + train_days:start + train_days + test_days]

        train_start, train_end = train_dates[0], train_dates[-1]
        test_start, test_end = test_dates[0], test_dates[-1]

        train_df = valid_panel[(valid_panel['date'] >= train_start) & (valid_panel['date'] <= train_end)]
        test_df = valid_panel[(valid_panel['date'] >= test_start) & (valid_panel['date'] <= test_end)]

        if len(train_df) < 500 or len(test_df) < 100:
            start += test_days
            continue

        X_train = train_df[avail_features].values
        y_train = train_df[target_col].values
        X_test = test_df[avail_features].values
        y_test = test_df[target_col].values

        # Handle class imbalance
        pos_rate = y_train.mean()
        if pos_rate < 0.01 or pos_rate > 0.99:
            start += test_days
            continue

        scale_pos = (1 - pos_rate) / pos_rate

        if model_type == 'lgbm':
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
            model.fit(X_train, y_train,
                      eval_set=[(X_test, y_test)],
                      callbacks=[lgb.early_stopping(20, verbose=False)])
        else:
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            model = LogisticRegression(
                C=0.1, max_iter=1000, class_weight='balanced', random_state=42
            )
            model.fit(X_train_s, y_train)
            X_test = X_test_s  # Use scaled for prediction

        # Predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        fold_results.append({
            'fold': fold_idx,
            'train_start': str(train_start)[:10],
            'train_end': str(train_end)[:10],
            'test_start': str(test_start)[:10],
            'test_end': str(test_end)[:10],
            'n_train': len(train_df),
            'n_test': len(test_df),
            'pos_rate_train': pos_rate,
            'pos_rate_test': y_test.mean(),
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
        })

        # Store OOT predictions
        oot_df = test_df[['date', 'ticker', target_col]].copy()
        oot_df['pred_proba'] = y_pred_proba
        oot_df['pred'] = y_pred
        oot_df['fold'] = fold_idx
        all_oot_preds.append(oot_df)

        if fold_idx % 2 == 0:
            print(f"  Fold {fold_idx}: test {str(test_start)[:10]} to {str(test_end)[:10]} "
                  f"| Acc={acc:.3f} Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f} "
                  f"| pos_rate={y_test.mean():.3f}")

        fold_idx += 1
        start += test_days

    if not all_oot_preds:
        print("ERROR: No valid folds!")
        return None, None, None

    # Concatenate all OOT predictions
    oot_all = pd.concat(all_oot_preds, ignore_index=True)
    fold_df = pd.DataFrame(fold_results)

    # Overall OOT metrics
    print(f"\n{'='*60}")
    print(f"OVERALL OOT RESULTS ({len(fold_df)} folds)")
    print(f"{'='*60}")
    print(f"Total OOT predictions: {len(oot_all):,}")
    print(f"Target positive rate: {oot_all[target_col].mean():.3f}")
    print(f"\nAccuracy:  {fold_df['accuracy'].mean():.3f} +/- {fold_df['accuracy'].std():.3f}")
    print(f"Precision: {fold_df['precision'].mean():.3f} +/- {fold_df['precision'].std():.3f}")
    print(f"Recall:    {fold_df['recall'].mean():.3f} +/- {fold_df['recall'].std():.3f}")
    print(f"F1:        {fold_df['f1'].mean():.3f} +/- {fold_df['f1'].std():.3f}")

    # Aggregate confusion matrix
    y_true_all = oot_all[target_col].values
    y_pred_all = oot_all['pred'].values
    cm = confusion_matrix(y_true_all, y_pred_all)
    print(f"\nAggregate Confusion Matrix:")
    print(f"  TN={cm[0,0]:,}  FP={cm[0,1]:,}")
    print(f"  FN={cm[1,0]:,}  TP={cm[1,1]:,}")

    overall_prec = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
    overall_rec = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
    overall_acc = (cm[0,0] + cm[1,1]) / cm.sum()
    print(f"\nOverall Precision: {overall_prec:.3f}")
    print(f"Overall Recall: {overall_rec:.3f}")
    print(f"Overall Accuracy: {overall_acc:.3f}")

    # Feature importance (last model)
    if model_type == 'lgbm' and hasattr(model, 'feature_importances_'):
        imp = pd.DataFrame({
            'feature': avail_features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        print(f"\nTop 15 Feature Importances (last fold):")
        for _, row in imp.head(15).iterrows():
            print(f"  {row['feature']:<30} {row['importance']:>6}")

    # Per-threshold analysis (key for options trading)
    print(f"\n{'='*60}")
    print("PRECISION BY CONFIDENCE THRESHOLD")
    print(f"{'='*60}")
    print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'N Signals':>10} {'Hit Rate':>10}")
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        mask = oot_all['pred_proba'] >= thresh
        n = mask.sum()
        if n > 10:
            prec = oot_all.loc[mask, target_col].mean()
            rec = oot_all.loc[mask, target_col].sum() / max(oot_all[target_col].sum(), 1)
            print(f"  {thresh:>8.1f} {prec:>10.3f} {rec:>10.3f} {n:>10,} {prec*100:>9.1f}%")

    return oot_all, fold_df, model


# ============================================================================
# PHASE 4: OPTIONS PROFITABILITY SIMULATION
# ============================================================================

def simulate_options_trades(oot_preds, prices, target_col='target_up5_60d',
                            confidence_threshold=0.6, dte=45,
                            position_size=100, spread_pct=0.05,
                            risk_free_rate=0.05):
    """
    Simulate buying ATM calls when model predicts >5% move.
    Uses Black-Scholes for pricing. Conservative assumptions.
    """
    print(f"\n{'='*80}")
    print(f"PHASE 4: OPTIONS PROFITABILITY SIMULATION")
    print(f"Confidence threshold: {confidence_threshold}")
    print(f"DTE: {dte}, Position size: ${position_size}")
    print(f"{'='*80}")

    # Filter to high-confidence predictions
    signals = oot_preds[oot_preds['pred_proba'] >= confidence_threshold].copy()
    print(f"Total signals above threshold: {len(signals):,}")

    if len(signals) == 0:
        print("No signals above threshold!")
        return None

    # For each signal, simulate buying an ATM call
    trades = []
    prices_indexed = {}
    for ticker in prices['ticker'].unique():
        tdf = prices[prices['ticker'] == ticker].set_index('date').sort_index()
        prices_indexed[ticker] = tdf

    for _, row in signals.iterrows():
        ticker = row['ticker']
        entry_date = row['date']

        if ticker not in prices_indexed:
            continue

        tdf = prices_indexed[ticker]

        # Find entry price
        if entry_date not in tdf.index:
            # Find nearest trading day
            mask = tdf.index >= entry_date
            if not mask.any():
                continue
            entry_date = tdf.index[mask][0]

        entry_idx = tdf.index.get_loc(entry_date)
        S = tdf['close'].iloc[entry_idx]

        # Estimate historical volatility for BS pricing
        if entry_idx < 30:
            continue
        hist_returns = tdf['close'].pct_change().iloc[entry_idx-63:entry_idx].dropna()
        if len(hist_returns) < 20:
            continue
        hist_vol = hist_returns.std() * np.sqrt(252)

        # IV is typically higher than HV (volatility risk premium)
        # Use 1.15x HV as conservative IV estimate
        iv = hist_vol * 1.15

        # ATM call
        K = S
        T = dte / 365.0
        r = risk_free_rate

        entry_premium = bs_call(S, K, T, r, iv)
        if entry_premium <= 0.01:
            continue

        # Apply bid-ask spread on entry (we pay ask)
        entry_cost = entry_premium * (1 + spread_pct / 2)

        # Number of contracts (closest to position_size)
        n_contracts = max(1, round(position_size / (entry_cost * 100)))
        total_cost = n_contracts * entry_cost * 100

        # Find exit price (hold to expiry or 45 days, whichever comes first)
        exit_idx = min(entry_idx + dte, len(tdf) - 1)
        if exit_idx <= entry_idx:
            continue

        S_exit = tdf['close'].iloc[exit_idx]
        actual_days_held = exit_idx - entry_idx

        # Exit: remaining time value
        T_remaining = max((dte - actual_days_held) / 365.0, 0)

        if T_remaining > 0:
            exit_premium = bs_call(S_exit, K, T_remaining, r, iv)
        else:
            # At expiry
            exit_premium = max(S_exit - K, 0)

        # Apply bid-ask spread on exit (we get bid)
        exit_proceeds = exit_premium * (1 - spread_pct / 2)
        total_proceeds = n_contracts * exit_proceeds * 100

        pnl = total_proceeds - total_cost
        pnl_pct = pnl / total_cost if total_cost > 0 else 0
        stock_return = S_exit / S - 1

        trades.append({
            'date': entry_date,
            'ticker': ticker,
            'stock_price': S,
            'stock_return': stock_return,
            'iv': iv,
            'entry_premium': entry_cost,
            'exit_premium': exit_proceeds,
            'n_contracts': n_contracts,
            'total_cost': total_cost,
            'total_proceeds': total_proceeds,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'days_held': actual_days_held,
            'pred_proba': row['pred_proba'],
            'actual_up5': row[target_col],
        })

    if not trades:
        print("No valid trades simulated!")
        return None

    trades_df = pd.DataFrame(trades)

    # Results
    print(f"\nTotal trades: {len(trades_df):,}")
    print(f"Win rate: {(trades_df['pnl'] > 0).mean():.3f} ({(trades_df['pnl'] > 0).mean()*100:.1f}%)")
    print(f"Avg P&L per trade: ${trades_df['pnl'].mean():.2f} ({trades_df['pnl_pct'].mean()*100:.1f}%)")
    print(f"Median P&L per trade: ${trades_df['pnl'].median():.2f}")
    print(f"Total P&L: ${trades_df['pnl'].sum():.2f}")
    print(f"Avg cost per trade: ${trades_df['total_cost'].mean():.2f}")
    print(f"Max loss: ${trades_df['pnl'].min():.2f}")
    print(f"Max gain: ${trades_df['pnl'].max():.2f}")

    # When model is right vs wrong
    right = trades_df[trades_df['actual_up5'] == 1]
    wrong = trades_df[trades_df['actual_up5'] == 0]
    print(f"\nWhen model CORRECT (stock moves >5%):")
    if len(right) > 0:
        print(f"  N trades: {len(right):,}")
        print(f"  Avg P&L: ${right['pnl'].mean():.2f} ({right['pnl_pct'].mean()*100:.1f}%)")
        print(f"  Win rate: {(right['pnl'] > 0).mean():.3f}")

    print(f"\nWhen model WRONG (stock doesn't move >5%):")
    if len(wrong) > 0:
        print(f"  N trades: {len(wrong):,}")
        print(f"  Avg P&L: ${wrong['pnl'].mean():.2f} ({wrong['pnl_pct'].mean()*100:.1f}%)")
        print(f"  Win rate: {(wrong['pnl'] > 0).mean():.3f}")

    # Equity curve
    trades_df = trades_df.sort_values('date')
    trades_df['cum_pnl'] = trades_df['pnl'].cumsum()

    # Sharpe-like ratio (on trade returns)
    if len(trades_df) > 2 and trades_df['pnl_pct'].std() > 0:
        # Annualize assuming avg 45-day holding period
        trades_per_year = 252 / trades_df['days_held'].mean()
        sharpe = (trades_df['pnl_pct'].mean() / trades_df['pnl_pct'].std()) * np.sqrt(trades_per_year)
        sortino_denom = trades_df[trades_df['pnl_pct'] < 0]['pnl_pct'].std()
        sortino = (trades_df['pnl_pct'].mean() / sortino_denom * np.sqrt(trades_per_year)
                   ) if sortino_denom > 0 else 0
        print(f"\nRisk-Adjusted Metrics:")
        print(f"  Sharpe: {sharpe:.2f}")
        print(f"  Sortino: {sortino:.2f}")
        print(f"  Profit Factor: {trades_df[trades_df['pnl']>0]['pnl'].sum() / max(abs(trades_df[trades_df['pnl']<0]['pnl'].sum()), 1):.2f}")

    return trades_df


# ============================================================================
# PHASE 5: VALIDATION
# ============================================================================

def regime_test(oot_preds, prices, target_col='target_up5_60d'):
    """R1 regime-agnostic test: check if model works in both up and down markets."""
    print(f"\n{'='*80}")
    print("PHASE 5A: REGIME TEST (R1)")
    print(f"{'='*80}")

    # Get SPY returns for regime classification
    spy = prices[prices['ticker'] == 'SPY'].set_index('date').sort_index()
    if len(spy) == 0:
        print("No SPY data for regime test!")
        return

    spy['regime'] = 'flat'
    spy_monthly = spy['close'].resample('M').last().pct_change()

    # Classify each month as green/red/flat
    monthly_regimes = {}
    for dt, ret in spy_monthly.items():
        if pd.isna(ret):
            continue
        if ret > 0.02:
            monthly_regimes[dt.to_period('M')] = 'green'
        elif ret < -0.02:
            monthly_regimes[dt.to_period('M')] = 'red'
        else:
            monthly_regimes[dt.to_period('M')] = 'flat'

    # Assign regime to each prediction
    preds = oot_preds.copy()
    preds['month'] = preds['date'].dt.to_period('M')
    preds['regime'] = preds['month'].map(monthly_regimes).fillna('flat')

    for regime in ['green', 'red', 'flat']:
        subset = preds[preds['regime'] == regime]
        if len(subset) < 50:
            continue

        acc = accuracy_score(subset[target_col], subset['pred'])
        prec = precision_score(subset[target_col], subset['pred'], zero_division=0)
        hit_rate = subset.loc[subset['pred'] == 1, target_col].mean() if (subset['pred'] == 1).any() else 0

        print(f"  {regime:>5} regime: N={len(subset):,}, Acc={acc:.3f}, Prec={prec:.3f}, Hit={hit_rate:.3f}")

    # R1 check
    green_preds = preds[preds['regime'] == 'green']
    red_preds = preds[preds['regime'] == 'red']

    if len(green_preds) > 50 and len(red_preds) > 50:
        # Compute strategy returns per regime
        green_signals = green_preds[green_preds['pred'] == 1]
        red_signals = red_preds[red_preds['pred'] == 1]

        if len(green_signals) > 5 and len(red_signals) > 5:
            green_hit = green_signals[target_col].mean()
            red_hit = red_signals[target_col].mean()

            ratio = abs(green_hit - red_hit) / max(abs(green_hit), abs(red_hit), 0.001)
            print(f"\n  R1 regime skew: |green_hit - red_hit| / max = {ratio:.3f}")
            if ratio > 0.50:
                print(f"  *** R1 FAIL: Regime skew {ratio:.3f} > 0.50 — model is regime-dependent ***")
            else:
                print(f"  R1 PASS: Regime skew {ratio:.3f} <= 0.50")


def stock_portfolio_backtest(oot_preds, prices, target_col='target_up5_60d',
                             confidence_threshold=0.5, top_n=10, hold_days=42):
    """
    Backtest a long-only stock portfolio:
    Each rebalance, buy top-N predicted stocks by confidence. Hold for hold_days.
    Equal weight. Commission-free (Robinhood).
    """
    print(f"\n{'='*80}")
    print(f"STOCK PORTFOLIO BACKTEST (threshold={confidence_threshold}, top_n={top_n}, hold={hold_days}d)")
    print(f"{'='*80}")

    # Get signals above threshold, grouped by date
    signals = oot_preds[oot_preds['pred_proba'] >= confidence_threshold].copy()
    signals = signals.sort_values(['date', 'pred_proba'], ascending=[True, False])

    # Build price lookup
    price_lookup = {}
    for ticker in prices['ticker'].unique():
        tdf = prices[prices['ticker'] == ticker].set_index('date').sort_index()
        price_lookup[ticker] = tdf

    # Get SPY for benchmark
    spy_prices = price_lookup.get('SPY')

    # Group signals by rebalance date (unique dates in OOT)
    unique_dates = sorted(signals['date'].unique())

    # Rebalance monthly (every ~21 trading days)
    rebalance_dates = []
    last_rb = None
    for d in unique_dates:
        if last_rb is None or (d - last_rb).days >= 20:
            rebalance_dates.append(d)
            last_rb = d

    portfolio_returns = []
    spy_returns = []
    all_picks = []

    for rb_date in rebalance_dates:
        # Get top-N stocks on this date
        day_signals = signals[signals['date'] == rb_date].head(top_n)
        if len(day_signals) == 0:
            continue

        period_rets = []
        for _, row in day_signals.iterrows():
            ticker = row['ticker']
            if ticker not in price_lookup:
                continue
            tdf = price_lookup[ticker]
            if rb_date not in tdf.index:
                mask = tdf.index >= rb_date
                if not mask.any():
                    continue
                entry_date = tdf.index[mask][0]
            else:
                entry_date = rb_date

            entry_idx = tdf.index.get_loc(entry_date)
            exit_idx = min(entry_idx + hold_days, len(tdf) - 1)
            if exit_idx <= entry_idx:
                continue

            ret = tdf['close'].iloc[exit_idx] / tdf['close'].iloc[entry_idx] - 1
            period_rets.append(ret)
            all_picks.append({
                'date': rb_date, 'ticker': ticker,
                'confidence': row['pred_proba'], 'return': ret,
                'actual_up5': row[target_col]
            })

        if period_rets:
            avg_ret = np.mean(period_rets)
            portfolio_returns.append({'date': rb_date, 'return': avg_ret, 'n_stocks': len(period_rets)})

            # SPY return for same period
            if spy_prices is not None and rb_date in spy_prices.index:
                spy_idx = spy_prices.index.get_loc(rb_date)
                spy_exit = min(spy_idx + hold_days, len(spy_prices) - 1)
                spy_ret = spy_prices['close'].iloc[spy_exit] / spy_prices['close'].iloc[spy_idx] - 1
                spy_returns.append({'date': rb_date, 'return': spy_ret})

    if not portfolio_returns:
        print("No valid portfolio periods!")
        return None, None

    port_df = pd.DataFrame(portfolio_returns)
    spy_df = pd.DataFrame(spy_returns) if spy_returns else None
    picks_df = pd.DataFrame(all_picks) if all_picks else None

    # Compound returns
    port_cumret = (1 + port_df['return']).cumprod()
    total_return = port_cumret.iloc[-1] - 1

    # Calculate CAGR
    n_years = (port_df['date'].max() - port_df['date'].min()).days / 365.25
    if n_years > 0:
        cagr = (1 + total_return) ** (1 / n_years) - 1
    else:
        cagr = 0

    # Sharpe/Sortino (annualized, assuming ~12 rebalances/year)
    periods_per_year = 252 / hold_days
    if port_df['return'].std() > 0:
        sharpe = port_df['return'].mean() / port_df['return'].std() * np.sqrt(periods_per_year)
        downside = port_df[port_df['return'] < 0]['return'].std()
        sortino = port_df['return'].mean() / downside * np.sqrt(periods_per_year) if downside > 0 else 0
    else:
        sharpe = sortino = 0

    # Max drawdown
    running_max = port_cumret.cummax()
    drawdowns = port_cumret / running_max - 1
    max_dd = drawdowns.min()
    calmar = cagr / abs(max_dd) if max_dd < 0 else 0

    # Win rate
    win_rate = (port_df['return'] > 0).mean()
    profit_factor = (port_df[port_df['return'] > 0]['return'].sum() /
                     abs(port_df[port_df['return'] < 0]['return'].sum())
                     if (port_df['return'] < 0).any() else float('inf'))

    print(f"\nPortfolio Results ({len(port_df)} periods, {n_years:.1f} years):")
    print(f"  Total Return: {total_return*100:.1f}%")
    print(f"  CAGR: {cagr*100:.1f}%")
    print(f"  Sharpe: {sharpe:.2f}")
    print(f"  Sortino: {sortino:.2f}")
    print(f"  Max Drawdown: {max_dd*100:.1f}%")
    print(f"  Calmar: {calmar:.2f}")
    print(f"  Win Rate: {win_rate*100:.1f}%")
    print(f"  Profit Factor: {profit_factor:.2f}")
    print(f"  Avg stocks/period: {port_df['n_stocks'].mean():.1f}")

    if spy_df is not None and len(spy_df) > 0:
        spy_total = (1 + spy_df['return']).cumprod().iloc[-1] - 1
        spy_cagr = (1 + spy_total) ** (1 / n_years) - 1 if n_years > 0 else 0
        print(f"\n  SPY Buy-Hold: Total={spy_total*100:.1f}%, CAGR={spy_cagr*100:.1f}%")
        print(f"  Alpha vs SPY: {(cagr - spy_cagr)*100:.1f}% annualized")

    return port_df, picks_df


def permutation_test(panel, oot_preds, target_col='target_up5_60d', n_perms=100):
    """Permutation test: shuffle signal dates to check for spurious results."""
    print(f"\n{'='*80}")
    print(f"PHASE 5B: PERMUTATION TEST ({n_perms} trials)")
    print(f"{'='*80}")

    # Real model precision
    real_signals = oot_preds[oot_preds['pred'] == 1]
    if len(real_signals) == 0:
        print("No positive predictions!")
        return None

    real_precision = real_signals[target_col].mean()
    real_n = len(real_signals)

    # Permutation: randomly select same number of stock-dates
    valid = panel.dropna(subset=[target_col])
    perm_precisions = []

    for i in range(n_perms):
        # Random sample of same size
        sample = valid.sample(n=min(real_n, len(valid)), replace=False, random_state=i)
        perm_prec = sample[target_col].mean()
        perm_precisions.append(perm_prec)

    perm_mean = np.mean(perm_precisions)
    perm_std = np.std(perm_precisions)
    z_score = (real_precision - perm_mean) / max(perm_std, 0.001)
    p_value = 1 - norm.cdf(z_score)

    print(f"Real model precision: {real_precision:.4f} (N={real_n:,})")
    print(f"Permutation mean:     {perm_mean:.4f} +/- {perm_std:.4f}")
    print(f"Z-score: {z_score:.2f}")
    print(f"P-value: {p_value:.4f}")
    if p_value < 0.05:
        print("*** SIGNIFICANT: Model precision is significantly better than random ***")
    else:
        print("*** NOT SIGNIFICANT: Model precision is NOT better than random ***")

    return {'real_precision': real_precision, 'perm_mean': perm_mean,
            'z_score': z_score, 'p_value': p_value}


def per_year_breakdown(oot_preds, target_col='target_up5_60d'):
    """Per-year breakdown of model performance."""
    print(f"\n{'='*80}")
    print("PHASE 5C: PER-YEAR BREAKDOWN")
    print(f"{'='*80}")

    preds = oot_preds.copy()
    preds['year'] = preds['date'].dt.year

    print(f"{'Year':>6} {'N':>8} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'Signals':>8} {'Hit%':>8}")
    print("-" * 60)

    for year in sorted(preds['year'].unique()):
        subset = preds[preds['year'] == year]
        if len(subset) < 50:
            continue

        acc = accuracy_score(subset[target_col], subset['pred'])
        prec = precision_score(subset[target_col], subset['pred'], zero_division=0)
        rec = recall_score(subset[target_col], subset['pred'], zero_division=0)
        n_signals = (subset['pred'] == 1).sum()
        hit = subset.loc[subset['pred'] == 1, target_col].mean() if n_signals > 0 else 0

        print(f"  {year:>4} {len(subset):>8,} {acc:>8.3f} {prec:>8.3f} {rec:>8.3f} {n_signals:>8,} {hit*100:>7.1f}%")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("COMBINED SIGNAL STOCK PREDICTION MODEL")
    print("Can we predict 5%+ moves accurately enough to buy options profitably?")
    print("="*80)
    t0 = time.time()

    # Phase 1: Data
    print("\n" + "="*80)
    print("PHASE 1: DATA COLLECTION")
    print("="*80)

    prices = download_price_data(use_cache=True)
    earnings_data = get_earnings_data(UNIVERSE, use_cache=True)
    insider_df = load_insider_data()

    # Phase 1b: Forward returns & features
    forward_returns = compute_forward_returns(prices)
    earnings_features = compute_earnings_features(prices, earnings_data)
    insider_features = compute_insider_features(prices, insider_df)

    # Build master panel
    panel = build_master_panel(prices, earnings_features, insider_features, forward_returns)
    panel.to_parquet(OUTPUT_DIR / "master_panel.parquet", index=False)

    # Phase 2: Individual signal testing
    signal_results = test_individual_signals(panel)
    with open(OUTPUT_DIR / "individual_signal_results.json", 'w') as f:
        json.dump({k: v if isinstance(v, (int, float, str)) else
                   {kk: float(vv) if isinstance(vv, (int, float, np.floating, np.integer)) else vv
                    for kk, vv in v.items()}
                   for k, v in signal_results.items()}, f, indent=2)

    # Phase 3: Walk-forward LGBM
    oot_preds_lgbm, fold_results_lgbm, model_lgbm = walk_forward_model(
        panel, target_col='target_up5_60d', model_type='lgbm'
    )

    # Also try logistic regression as baseline
    oot_preds_lr, fold_results_lr, model_lr = walk_forward_model(
        panel, target_col='target_up5_60d', model_type='logreg'
    )

    # Also try 30-day target
    oot_preds_30d, fold_results_30d, _ = walk_forward_model(
        panel, target_col='target_up5_30d', model_type='lgbm'
    )

    # Use best model for options simulation
    best_oot = oot_preds_lgbm  # LGBM is typically better
    if best_oot is None:
        print("\nERROR: No valid OOT predictions. Cannot proceed.")
        return

    # Save OOT predictions
    best_oot.to_parquet(OUTPUT_DIR / "oot_predictions_lgbm.parquet", index=False)
    if oot_preds_lr is not None:
        oot_preds_lr.to_parquet(OUTPUT_DIR / "oot_predictions_logreg.parquet", index=False)

    # Phase 3b: Stock portfolio backtest
    port_results = {}
    for thresh in [0.4, 0.5, 0.6, 0.7]:
        port_df, picks_df = stock_portfolio_backtest(
            best_oot, prices, confidence_threshold=thresh, top_n=10, hold_days=42
        )
        if port_df is not None:
            port_results[thresh] = port_df

    # Phase 4: Options simulation at different thresholds
    print("\n" + "="*80)
    print("OPTIONS SIMULATION AT DIFFERENT CONFIDENCE THRESHOLDS")
    print("="*80)

    for thresh in [0.4, 0.5, 0.6, 0.7, 0.8]:
        trades = simulate_options_trades(
            best_oot, prices,
            confidence_threshold=thresh,
            dte=45, position_size=100
        )
        if trades is not None:
            trades.to_parquet(OUTPUT_DIR / f"options_trades_thresh{int(thresh*100)}.parquet", index=False)

    # Phase 5: Validation
    regime_test(best_oot, prices)
    perm_result = permutation_test(panel, best_oot)
    per_year_breakdown(best_oot)

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    elapsed = time.time() - t0
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Runtime: {elapsed/60:.1f} minutes")
    print(f"Universe: {panel['ticker'].nunique()} stocks")
    print(f"Date range: {panel['date'].min().strftime('%Y-%m-%d')} to {panel['date'].max().strftime('%Y-%m-%d')}")
    print(f"\nLGBM Model (target: >5% in 60d):")
    if fold_results_lgbm is not None:
        print(f"  Folds: {len(fold_results_lgbm)}")
        print(f"  Avg Precision: {fold_results_lgbm['precision'].mean():.3f}")
        print(f"  Avg Accuracy:  {fold_results_lgbm['accuracy'].mean():.3f}")
    print(f"\nLogistic Regression Baseline:")
    if fold_results_lr is not None:
        print(f"  Avg Precision: {fold_results_lr['precision'].mean():.3f}")
        print(f"  Avg Accuracy:  {fold_results_lr['accuracy'].mean():.3f}")

    # Save full results
    summary = {
        'runtime_minutes': elapsed / 60,
        'n_stocks': int(panel['ticker'].nunique()),
        'date_range': f"{panel['date'].min().strftime('%Y-%m-%d')} to {panel['date'].max().strftime('%Y-%m-%d')}",
        'lgbm_avg_precision': float(fold_results_lgbm['precision'].mean()) if fold_results_lgbm is not None else None,
        'lgbm_avg_accuracy': float(fold_results_lgbm['accuracy'].mean()) if fold_results_lgbm is not None else None,
        'lgbm_avg_recall': float(fold_results_lgbm['recall'].mean()) if fold_results_lgbm is not None else None,
        'lgbm_avg_f1': float(fold_results_lgbm['f1'].mean()) if fold_results_lgbm is not None else None,
        'logreg_avg_precision': float(fold_results_lr['precision'].mean()) if fold_results_lr is not None else None,
        'logreg_avg_accuracy': float(fold_results_lr['accuracy'].mean()) if fold_results_lr is not None else None,
        'base_rate_up5_60d': float(signal_results.get('base_rate_target_up5_60d', 0)),
    }

    if perm_result:
        summary['permutation_p_value'] = perm_result['p_value']
        summary['permutation_z_score'] = perm_result['z_score']

    with open(OUTPUT_DIR / "model_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # MLflow logging
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("stock_prediction_v1")
        with mlflow.start_run(run_name=f"combined_signal_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            mlflow.log_params({
                'universe_size': len(UNIVERSE),
                'train_window': 252,
                'test_window': 21,
                'target': 'up5_60d',
                'model_type': 'lgbm',
                'n_features': len([c for c in FEATURE_COLS if c in panel.columns]),
            })
            for k, v in summary.items():
                if v is not None and isinstance(v, (int, float)):
                    mlflow.log_metric(k, v)
            mlflow.log_artifact(str(OUTPUT_DIR / "model_summary.json"))
            if (OUTPUT_DIR / "individual_signal_results.json").exists():
                mlflow.log_artifact(str(OUTPUT_DIR / "individual_signal_results.json"))
        print("Logged to MLflow experiment 'stock_prediction_v1'")
    except Exception as e:
        print(f"MLflow logging failed (non-fatal): {e}")

    print(f"\nAll results saved to {OUTPUT_DIR}")
    print("DONE.")


if __name__ == "__main__":
    main()
