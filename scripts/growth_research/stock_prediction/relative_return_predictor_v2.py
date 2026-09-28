#!/usr/bin/env python3
"""
Stock Prediction v2 — RELATIVE Returns (Regime-Neutral)
========================================================
v1 predicted ABSOLUTE returns (>5% in 60d) and failed R1 regime test because
in bull markets everything goes up — the model just rode beta.

v2 predicts EXCESS returns vs SPY:
  - "Will NVDA outperform SPY by 3%+ in 60 days?"
  - This strips out market beta and should work in both bull and bear markets.

Key differences from v1:
  1. Target = forward excess return (stock - SPY) instead of absolute return
  2. Cross-sectional rank features (momentum rank within sector, etc.)
  3. Relative strength vs SPY features
  4. 60-day embargo between train and test (prevents label overlap)
  5. Long-short portfolio backtest (long outperformers, short underperformers)
  6. Enhanced R1 regime validation

Walk-forward: 252d sliding train, 21d test steps, 60d embargo.
Model: LGBM classifier.
Universe: S&P 500 representative sample (~70 stocks).
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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

# --- Paths ---
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research/stock_prediction/v2_relative")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research/stock_prediction/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
INSIDER_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research/insider_alpha")

# --- Universe (same as v1 + sector labels) ---
SECTOR_MAP = {
    # Tech
    "AAPL": "Tech", "MSFT": "Tech", "GOOGL": "Tech", "META": "Tech", "NVDA": "Tech",
    "AMD": "Tech", "AVGO": "Tech", "MU": "Tech", "QCOM": "Tech", "INTC": "Tech",
    "ORCL": "Tech", "ADBE": "Tech",
    # Software/Cloud
    "CRM": "Software", "SHOP": "Software", "SNOW": "Software", "DDOG": "Software",
    "NET": "Software", "NOW": "Software",
    # Cybersecurity
    "CRWD": "Cyber", "ZS": "Cyber", "PANW": "Cyber",
    # Consumer Internet
    "AMZN": "ConsumerInternet", "TSLA": "ConsumerInternet", "NFLX": "ConsumerInternet",
    "UBER": "ConsumerInternet", "ABNB": "ConsumerInternet", "DASH": "ConsumerInternet",
    # Fintech/Crypto
    "SQ": "Fintech", "COIN": "Fintech", "PLTR": "Fintech",
    # Intl eCommerce
    "MELI": "IntlEcom", "SE": "IntlEcom", "BABA": "IntlEcom", "JD": "IntlEcom", "PDD": "IntlEcom",
    # Telecom/Media
    "DIS": "Media", "CMCSA": "Media", "T": "Telecom", "VZ": "Telecom",
    # Financials
    "JPM": "Financials", "GS": "Financials", "MS": "Financials", "BAC": "Financials",
    "WFC": "Financials", "V": "Financials", "MA": "Financials", "AXP": "Financials", "BRK-B": "Financials",
    # Healthcare
    "UNH": "Healthcare", "LLY": "Healthcare", "PFE": "Healthcare", "ABBV": "Healthcare",
    "MRK": "Healthcare", "JNJ": "Healthcare",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    # Industrials
    "LMT": "Industrials", "BA": "Industrials", "CAT": "Industrials", "DE": "Industrials",
    # Consumer Staples/Discretionary
    "HD": "ConsumerDisc", "LOW": "ConsumerDisc", "TGT": "ConsumerDisc",
    "WMT": "ConsumerStaples", "COST": "ConsumerStaples",
    "NKE": "ConsumerDisc", "SBUX": "ConsumerDisc", "MCD": "ConsumerStaples",
    "KO": "ConsumerStaples", "PEP": "ConsumerStaples",
}

UNIVERSE = list(SECTOR_MAP.keys())


# ============================================================================
# PHASE 1: DATA COLLECTION (reuses v1 cache)
# ============================================================================

def download_price_data(use_cache=True):
    """Download 5+ years of daily price data for all universe stocks + SPY."""
    cache_file = CACHE_DIR / "price_data.parquet"
    if use_cache and cache_file.exists():
        mod_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if (datetime.now() - mod_time).days < 3:
            print(f"Loading cached price data")
            return pd.read_parquet(cache_file)

    print("Downloading price data from yfinance...")
    all_tickers = list(set(UNIVERSE + ["SPY"]))
    start_date = "2019-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")

    all_frames = []
    batch_size = 20
    for i in range(0, len(all_tickers), batch_size):
        batch = all_tickers[i:i + batch_size]
        print(f"  Downloading batch {i // batch_size + 1}: {batch[:5]}...")
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
            print(f"  Error downloading batch: {e}")
        time.sleep(0.5)

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
    print(f"Saved {len(prices)} rows for {prices['ticker'].nunique()} tickers")
    return prices


def get_earnings_data(tickers, use_cache=True):
    """Get earnings dates and surprise data from yfinance (reuses v1 cache)."""
    cache_file = CACHE_DIR / "earnings_data.json"
    if use_cache and cache_file.exists():
        mod_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if (datetime.now() - mod_time).days < 7:
            print("Loading cached earnings data")
            with open(cache_file) as f:
                return json.load(f)

    print("Fetching earnings data from yfinance...")
    earnings_data = {}
    for i, ticker in enumerate(tickers):
        if i % 10 == 0:
            print(f"  Processing {i}/{len(tickers)}...")
        try:
            stock = yf.Ticker(ticker)
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
        except Exception:
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
# PHASE 2: FEATURE ENGINEERING (enhanced for v2)
# ============================================================================

def compute_spy_returns(prices):
    """Extract SPY daily returns for computing excess returns."""
    spy = prices[prices['ticker'] == 'SPY'].copy().sort_values('date').set_index('date')
    spy_ret = spy['close'].pct_change()
    return spy


def compute_technical_features(prices_group):
    """Same technicals as v1."""
    df = prices_group.copy().sort_values('date')

    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_200'] = df['close'].rolling(200).mean()

    df['price_vs_sma50'] = df['close'] / df['sma_50'] - 1
    df['price_vs_sma200'] = df['close'] / df['sma_200'] - 1
    df['above_50ma'] = (df['close'] > df['sma_50']).astype(float)
    df['above_200ma'] = (df['close'] > df['sma_200']).astype(float)
    df['ma_50_200_ratio'] = df['sma_50'] / df['sma_200']

    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi_14'] = 100 - (100 / (1 + rs))

    for lb in [5, 10, 21, 63, 126, 252]:
        df[f'mom_{lb}d'] = df['close'].pct_change(lb)

    df['vol_21d'] = df['close'].pct_change().rolling(21).std() * np.sqrt(252)
    df['vol_63d'] = df['close'].pct_change().rolling(63).std() * np.sqrt(252)

    df['vol_ratio_20d'] = df['volume'] / df['volume'].rolling(20).mean()
    df['vol_ratio_50d'] = df['volume'] / df['volume'].rolling(50).mean()

    df['high_252'] = df['close'].rolling(252).max()
    df['drawdown_from_high'] = df['close'] / df['high_252'] - 1

    df['at_20d_high'] = (df['close'] >= df['close'].rolling(20).max()).astype(float)
    df['at_50d_high'] = (df['close'] >= df['close'].rolling(50).max()).astype(float)

    df['zscore_20d'] = (df['close'] - df['sma_20']) / df['close'].rolling(20).std()

    return df


def compute_relative_strength_features(prices):
    """
    NEW v2 features: relative strength vs SPY at multiple horizons.
    Also computes cross-sectional ranks within sector.
    """
    print("Computing relative strength features...")
    spy = prices[prices['ticker'] == 'SPY'].copy().sort_values('date').set_index('date')

    # SPY momentum at various lookbacks
    spy_mom = {}
    for lb in [21, 63, 126, 252]:
        spy_mom[lb] = spy['close'].pct_change(lb)

    results = {}
    for ticker in prices['ticker'].unique():
        if ticker == 'SPY':
            continue
        tdf = prices[prices['ticker'] == ticker].copy().sort_values('date').set_index('date')

        feat = pd.DataFrame(index=tdf.index)

        # Relative strength vs SPY at multiple horizons
        for lb in [21, 63, 126, 252]:
            stock_mom = tdf['close'].pct_change(lb)
            # Align with SPY
            spy_m = spy_mom[lb].reindex(tdf.index)
            feat[f'rs_vs_spy_{lb}d'] = stock_mom - spy_m

        # Relative volume vs SPY
        stock_vol_ratio = tdf['volume'] / tdf['volume'].rolling(20).mean()
        spy_vol_ratio = (spy['volume'] / spy['volume'].rolling(20).mean()).reindex(tdf.index)
        feat['rel_volume_vs_spy'] = stock_vol_ratio - spy_vol_ratio

        # Relative volatility vs SPY
        stock_vol = tdf['close'].pct_change().rolling(21).std() * np.sqrt(252)
        spy_vol = (spy['close'].pct_change().rolling(21).std() * np.sqrt(252)).reindex(tdf.index)
        feat['rel_vol_vs_spy'] = stock_vol - spy_vol

        # Beta (rolling 63d)
        stock_ret = tdf['close'].pct_change()
        spy_ret = spy['close'].pct_change().reindex(tdf.index)
        # Rolling correlation and beta
        roll_cov = stock_ret.rolling(63).cov(spy_ret)
        roll_var = spy_ret.rolling(63).var()
        feat['beta_63d'] = roll_cov / roll_var.replace(0, np.nan)

        # Idiosyncratic momentum (residual after removing beta * SPY)
        feat['idio_mom_63d'] = tdf['close'].pct_change(63) - feat['beta_63d'] * spy['close'].pct_change(63).reindex(tdf.index)

        results[ticker] = feat

    print(f"  Computed relative strength for {len(results)} tickers")
    return results


def compute_cross_sectional_ranks(panel):
    """
    NEW v2: Cross-sectional rank features computed per date.
    Ranks momentum, volume, volatility within sector and across all stocks.
    """
    print("Computing cross-sectional rank features...")
    panel = panel.copy()

    # Add sector
    panel['sector'] = panel['ticker'].map(SECTOR_MAP).fillna('Other')

    rank_cols = ['mom_21d', 'mom_63d', 'mom_126d', 'vol_21d', 'vol_ratio_20d', 'rsi_14']

    for col in rank_cols:
        if col not in panel.columns:
            continue

        # Universe-wide percentile rank per date
        panel[f'{col}_rank'] = panel.groupby('date')[col].rank(pct=True)

        # Sector rank per date
        panel[f'{col}_sector_rank'] = panel.groupby(['date', 'sector'])[col].rank(pct=True)

    # Composite momentum rank (average of 21d, 63d, 126d momentum ranks)
    mom_rank_cols = [f'mom_{lb}d_rank' for lb in [21, 63, 126] if f'mom_{lb}d_rank' in panel.columns]
    if mom_rank_cols:
        panel['composite_mom_rank'] = panel[mom_rank_cols].mean(axis=1)

    print(f"  Added {sum(1 for c in panel.columns if '_rank' in c)} rank features")
    return panel


def compute_earnings_features(prices, earnings_data):
    """Same as v1 — compute PEAD features."""
    print("Computing earnings features...")
    results = {}

    for ticker in prices['ticker'].unique():
        if ticker == 'SPY' or ticker not in earnings_data:
            continue

        tdf = prices[prices['ticker'] == ticker].set_index('date').sort_index()
        earnings_records = earnings_data[ticker]

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

        feat['pead_signal'] = 0.0
        m = (feat['earnings_surprise'] > 0) & (feat['earnings_gap'] > 0.02) & (feat['days_since_earnings'] <= 60)
        feat.loc[m, 'pead_signal'] = 1.0

        feat['pead_strong'] = 0.0
        m2 = (feat['earnings_surprise'] > 5) & (feat['earnings_gap'] > 0.04) & (feat['days_since_earnings'] <= 60)
        feat.loc[m2, 'pead_strong'] = 1.0

        feat['pead_negative'] = 0.0
        m3 = (feat['earnings_surprise'] < -2) & (feat['earnings_gap'] < -0.03) & (feat['days_since_earnings'] <= 60)
        feat.loc[m3, 'pead_negative'] = 1.0

        results[ticker] = feat

    print(f"  Computed earnings features for {len(results)} tickers")
    return results


def compute_insider_features(prices, insider_df):
    """Same as v1."""
    print("Computing insider features...")
    if insider_df.empty:
        print("  No insider data — skipping")
        return {}

    insider_df = insider_df.copy()
    insider_df['transaction_date'] = pd.to_datetime(insider_df['transaction_date'])
    insider_df['filing_date'] = pd.to_datetime(insider_df['filing_date'])

    results = {}
    for ticker in prices['ticker'].unique():
        if ticker == 'SPY':
            continue
        tdf = prices[prices['ticker'] == ticker].set_index('date').sort_index()
        idf = insider_df[insider_df['ticker'] == ticker].copy()

        if idf.empty:
            continue

        buys = idf[idf['type'].str.lower().str.contains('purchase|buy|p-purchase', na=False)]
        sells = idf[idf['type'].str.lower().str.contains('sale|sell|s-sale', na=False)]

        feat = pd.DataFrame(index=tdf.index)
        for d in tdf.index:
            d_ts = pd.Timestamp(d)
            window_start = d_ts - pd.Timedelta(days=30)
            recent_buys = buys[(buys['filing_date'] >= window_start) & (buys['filing_date'] <= d_ts)]
            recent_sells = sells[(sells['filing_date'] >= window_start) & (sells['filing_date'] <= d_ts)]

            n_buyers = recent_buys['reporter'].nunique() if len(recent_buys) > 0 else 0
            feat.loc[d, 'insider_buyers_30d'] = n_buyers
            feat.loc[d, 'insider_buy_value_30d'] = recent_buys['value'].sum() if len(recent_buys) > 0 else 0
            feat.loc[d, 'insider_sell_value_30d'] = recent_sells['value'].sum() if len(recent_sells) > 0 else 0
            total_txns = len(recent_buys) + len(recent_sells)
            feat.loc[d, 'insider_buy_ratio'] = len(recent_buys) / max(total_txns, 1)
            feat.loc[d, 'insider_cluster'] = 1.0 if n_buyers >= 3 else 0.0
            feat.loc[d, 'insider_cluster_strong'] = 1.0 if (n_buyers >= 3 and
                feat.loc[d, 'insider_buy_value_30d'] > 200000) else 0.0

        results[ticker] = feat

    print(f"  Computed insider features for {len(results)} tickers")
    return results


def compute_forward_excess_returns(prices):
    """
    KEY v2 CHANGE: Compute forward EXCESS returns (stock - SPY).
    This is the core difference from v1.
    """
    print("Computing forward EXCESS returns (stock - SPY)...")
    spy = prices[prices['ticker'] == 'SPY'].copy().sort_values('date').set_index('date')

    # SPY forward returns
    spy_fwd_30d = spy['close'].pct_change(30).shift(-30)
    spy_fwd_60d = spy['close'].pct_change(60).shift(-60)

    results = {}
    for ticker in prices['ticker'].unique():
        if ticker == 'SPY':
            continue
        tdf = prices[prices['ticker'] == ticker].sort_values('date').set_index('date')

        fwd = pd.DataFrame(index=tdf.index)

        # Stock forward returns
        stock_fwd_30d = tdf['close'].pct_change(30).shift(-30)
        stock_fwd_60d = tdf['close'].pct_change(60).shift(-60)

        # Absolute returns (for reference)
        fwd['fwd_30d_abs'] = stock_fwd_30d
        fwd['fwd_60d_abs'] = stock_fwd_60d

        # EXCESS returns (stock - SPY)
        spy_30 = spy_fwd_30d.reindex(tdf.index)
        spy_60 = spy_fwd_60d.reindex(tdf.index)

        fwd['fwd_30d_excess'] = stock_fwd_30d - spy_30
        fwd['fwd_60d_excess'] = stock_fwd_60d - spy_60

        # Binary targets for excess returns
        fwd['target_excess_60d_3pct'] = (fwd['fwd_60d_excess'] > 0.03).astype(float)
        fwd['target_excess_60d_5pct'] = (fwd['fwd_60d_excess'] > 0.05).astype(float)
        fwd['target_excess_30d_2pct'] = (fwd['fwd_30d_excess'] > 0.02).astype(float)

        # Also: underperform target (for short side of long-short)
        fwd['target_underperform_60d_3pct'] = (fwd['fwd_60d_excess'] < -0.03).astype(float)

        results[ticker] = fwd

    return results


def build_master_panel(prices, earnings_features, insider_features,
                       forward_returns, relative_strength):
    """Combine all features into master panel."""
    print("Building master panel...")
    all_frames = []

    for ticker in prices['ticker'].unique():
        if ticker == 'SPY':
            continue

        tdf = prices[prices['ticker'] == ticker].copy()
        tdf = compute_technical_features(tdf)
        tdf = tdf.set_index('date')

        # Merge relative strength features
        if ticker in relative_strength:
            rs = relative_strength[ticker]
            for col in rs.columns:
                tdf[col] = rs[col]

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

    # Fill NaN features with 0 for insider/earnings (no signal = no activity)
    fill_zero_cols = [
        'insider_buyers_30d', 'insider_buy_value_30d', 'insider_sell_value_30d',
        'insider_buy_ratio', 'insider_cluster', 'insider_cluster_strong',
        'days_since_earnings', 'earnings_surprise', 'earnings_gap',
        'post_earnings_drift', 'pead_signal', 'pead_strong', 'pead_negative',
    ]
    for col in fill_zero_cols:
        if col in panel.columns:
            panel[col] = panel[col].fillna(0)

    # Add sector
    panel['sector'] = panel['ticker'].map(SECTOR_MAP).fillna('Other')

    # Compute cross-sectional ranks
    panel = compute_cross_sectional_ranks(panel)

    # Compute sector momentum (average momentum of sector stocks)
    print("Computing sector momentum features...")
    sector_mom = panel.groupby(['date', 'sector'])['mom_63d'].transform('mean')
    panel['sector_mom_63d'] = sector_mom
    panel['stock_vs_sector_mom'] = panel['mom_63d'] - panel['sector_mom_63d']

    panel = panel.sort_values(['date', 'ticker']).reset_index(drop=True)
    print(f"Master panel: {len(panel)} rows, {len(panel.columns)} columns")
    print(f"Date range: {panel['date'].min()} to {panel['date'].max()}")
    print(f"Tickers: {panel['ticker'].nunique()}")

    return panel


# ============================================================================
# FEATURE COLUMNS
# ============================================================================

FEATURE_COLS = [
    # --- v1 Technical/Momentum ---
    'price_vs_sma50', 'price_vs_sma200', 'above_50ma', 'above_200ma',
    'ma_50_200_ratio', 'rsi_14',
    'mom_5d', 'mom_10d', 'mom_21d', 'mom_63d', 'mom_126d', 'mom_252d',
    'vol_21d', 'vol_63d',
    'vol_ratio_20d', 'vol_ratio_50d',
    'drawdown_from_high',
    'at_20d_high', 'at_50d_high',
    'zscore_20d',
    # --- v1 Earnings/PEAD ---
    'days_since_earnings', 'earnings_surprise', 'earnings_gap',
    'post_earnings_drift', 'pead_signal', 'pead_strong', 'pead_negative',
    # --- v1 Insider ---
    'insider_buyers_30d', 'insider_buy_value_30d', 'insider_sell_value_30d',
    'insider_buy_ratio', 'insider_cluster', 'insider_cluster_strong',
    # --- NEW v2: Relative Strength vs SPY ---
    'rs_vs_spy_21d', 'rs_vs_spy_63d', 'rs_vs_spy_126d', 'rs_vs_spy_252d',
    'rel_volume_vs_spy', 'rel_vol_vs_spy',
    'beta_63d', 'idio_mom_63d',
    # --- NEW v2: Cross-sectional Ranks ---
    'mom_21d_rank', 'mom_63d_rank', 'mom_126d_rank',
    'vol_21d_rank', 'vol_ratio_20d_rank', 'rsi_14_rank',
    'mom_21d_sector_rank', 'mom_63d_sector_rank', 'mom_126d_sector_rank',
    'composite_mom_rank',
    # --- NEW v2: Sector features ---
    'sector_mom_63d', 'stock_vs_sector_mom',
]


# ============================================================================
# PHASE 3: WALK-FORWARD MODEL (with embargo)
# ============================================================================

def walk_forward_model(panel, target_col='target_excess_60d_3pct',
                       train_days=252, test_days=21, embargo_days=60,
                       model_type='lgbm'):
    """
    Walk-forward with EMBARGO between train and test.
    The embargo prevents label leakage from overlapping forward windows.
    """
    print(f"\n{'=' * 80}")
    print(f"PHASE 3: WALK-FORWARD MODEL ({model_type.upper()})")
    print(f"Target: {target_col}, Train: {train_days}d, Test: {test_days}d, Embargo: {embargo_days}d")
    print(f"{'=' * 80}")

    avail_features = [c for c in FEATURE_COLS if c in panel.columns]
    print(f"Using {len(avail_features)} features")

    dates = sorted(panel['date'].unique())
    print(f"Total dates: {len(dates)}")

    valid_panel = panel.dropna(subset=[target_col]).copy()
    for col in avail_features:
        valid_panel[col] = valid_panel[col].fillna(0)

    all_oot_preds = []
    fold_results = []
    fold_idx = 0
    start = 0

    while start + train_days + embargo_days + test_days <= len(dates):
        train_dates = dates[start:start + train_days]
        # EMBARGO: skip embargo_days between train end and test start
        test_start_idx = start + train_days + embargo_days
        test_dates = dates[test_start_idx:test_start_idx + test_days]

        train_start, train_end = train_dates[0], train_dates[-1]
        test_start, test_end = test_dates[0], test_dates[-1]

        train_df = valid_panel[(valid_panel['date'] >= train_start) & (valid_panel['date'] <= train_end)]
        test_df = valid_panel[(valid_panel['date'] >= test_start) & (valid_panel['date'] <= test_end)]

        if len(train_df) < 500 or len(test_df) < 50:
            start += test_days
            continue

        X_train = train_df[avail_features].values
        y_train = train_df[target_col].values
        X_test = test_df[avail_features].values
        y_test = test_df[target_col].values

        pos_rate = y_train.mean()
        if pos_rate < 0.01 or pos_rate > 0.99:
            start += test_days
            continue

        scale_pos = (1 - pos_rate) / pos_rate

        model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.7,
            scale_pos_weight=scale_pos,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1,
            n_jobs=4,
        )
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  callbacks=[lgb.early_stopping(30, verbose=False)])

        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)

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
            'pos_rate_test': float(y_test.mean()),
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
        })

        oot_df = test_df[['date', 'ticker', 'sector', target_col]].copy()
        # Also grab excess return for portfolio sim
        if 'fwd_60d_excess' in test_df.columns:
            oot_df['fwd_60d_excess'] = test_df['fwd_60d_excess'].values
        if 'fwd_30d_excess' in test_df.columns:
            oot_df['fwd_30d_excess'] = test_df['fwd_30d_excess'].values
        if 'fwd_60d_abs' in test_df.columns:
            oot_df['fwd_60d_abs'] = test_df['fwd_60d_abs'].values
        oot_df['pred_proba'] = y_pred_proba
        oot_df['pred'] = y_pred
        oot_df['fold'] = fold_idx
        all_oot_preds.append(oot_df)

        if fold_idx % 5 == 0:
            print(f"  Fold {fold_idx}: test {str(test_start)[:10]} to {str(test_end)[:10]} "
                  f"| Acc={acc:.3f} Prec={prec:.3f} Rec={rec:.3f} "
                  f"| pos_rate={y_test.mean():.3f}")

        fold_idx += 1
        start += test_days

    if not all_oot_preds:
        print("ERROR: No valid folds!")
        return None, None, None

    oot_all = pd.concat(all_oot_preds, ignore_index=True)
    fold_df = pd.DataFrame(fold_results)

    # Overall results
    print(f"\n{'=' * 60}")
    print(f"OVERALL OOT RESULTS ({len(fold_df)} folds)")
    print(f"{'=' * 60}")
    print(f"Total OOT predictions: {len(oot_all):,}")
    print(f"Target positive rate: {oot_all[target_col].mean():.3f}")

    print(f"\nAccuracy:  {fold_df['accuracy'].mean():.3f} +/- {fold_df['accuracy'].std():.3f}")
    print(f"Precision: {fold_df['precision'].mean():.3f} +/- {fold_df['precision'].std():.3f}")
    print(f"Recall:    {fold_df['recall'].mean():.3f} +/- {fold_df['recall'].std():.3f}")
    print(f"F1:        {fold_df['f1'].mean():.3f} +/- {fold_df['f1'].std():.3f}")

    y_true_all = oot_all[target_col].values
    y_pred_all = oot_all['pred'].values
    cm = confusion_matrix(y_true_all, y_pred_all)
    print(f"\nAggregate Confusion Matrix:")
    print(f"  TN={cm[0, 0]:,}  FP={cm[0, 1]:,}")
    print(f"  FN={cm[1, 0]:,}  TP={cm[1, 1]:,}")

    overall_prec = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0
    overall_rec = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
    print(f"\nOverall Precision: {overall_prec:.3f}")
    print(f"Overall Recall: {overall_rec:.3f}")

    # Feature importance
    if hasattr(model, 'feature_importances_'):
        imp = pd.DataFrame({
            'feature': avail_features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        print(f"\nTop 20 Feature Importances (last fold):")
        for _, row in imp.head(20).iterrows():
            print(f"  {row['feature']:<30} {row['importance']:>6}")
        imp.to_csv(OUTPUT_DIR / f"feature_importance_{target_col}.csv", index=False)

    # Precision by threshold
    print(f"\n{'=' * 60}")
    print("PRECISION BY CONFIDENCE THRESHOLD (for excess returns)")
    print(f"{'=' * 60}")
    base_rate = oot_all[target_col].mean()
    print(f"Base rate: {base_rate:.3f}")
    print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'N Signals':>10} {'Lift':>8}")
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        mask = oot_all['pred_proba'] >= thresh
        n = mask.sum()
        if n > 10:
            prec = oot_all.loc[mask, target_col].mean()
            rec = oot_all.loc[mask, target_col].sum() / max(oot_all[target_col].sum(), 1)
            lift = prec / base_rate if base_rate > 0 else 0
            print(f"  {thresh:>8.1f} {prec:>10.3f} {rec:>10.3f} {n:>10,} {lift:>7.2f}x")

    return oot_all, fold_df, model


# ============================================================================
# PHASE 4: LONG-SHORT PORTFOLIO BACKTEST
# ============================================================================

def long_short_portfolio(oot_preds, prices, target_col='target_excess_60d_3pct',
                         top_n=10, hold_days=42):
    """
    Long-short portfolio: long top-N predicted outperformers, short bottom-N.
    This should be approximately market-neutral.
    """
    print(f"\n{'=' * 80}")
    print(f"PHASE 4: LONG-SHORT PORTFOLIO (top/bottom {top_n}, hold {hold_days}d)")
    print(f"{'=' * 80}")

    spy = prices[prices['ticker'] == 'SPY'].set_index('date').sort_index()

    price_lookup = {}
    for ticker in prices['ticker'].unique():
        tdf = prices[prices['ticker'] == ticker].set_index('date').sort_index()
        price_lookup[ticker] = tdf

    # Group predictions by date
    unique_dates = sorted(oot_preds['date'].unique())

    # Rebalance monthly
    rebalance_dates = []
    last_rb = None
    for d in unique_dates:
        if last_rb is None or (d - last_rb).days >= 20:
            rebalance_dates.append(d)
            last_rb = d

    portfolio_returns = []
    long_returns = []
    short_returns = []
    all_picks = []

    for rb_date in rebalance_dates:
        day_preds = oot_preds[oot_preds['date'] == rb_date].copy()
        if len(day_preds) < 2 * top_n:
            continue

        # Sort by predicted probability
        day_preds = day_preds.sort_values('pred_proba', ascending=False)

        # Long: top-N highest confidence outperformers
        longs = day_preds.head(top_n)
        # Short: bottom-N (predicted underperformers = lowest outperform probability)
        shorts = day_preds.tail(top_n)

        # Compute returns
        period_long_rets = []
        period_short_rets = []

        for _, row in longs.iterrows():
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
            period_long_rets.append(ret)
            all_picks.append({
                'date': rb_date, 'ticker': ticker, 'side': 'long',
                'confidence': row['pred_proba'], 'return': ret,
            })

        for _, row in shorts.iterrows():
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
            period_short_rets.append(-ret)  # Short = negative of stock return
            all_picks.append({
                'date': rb_date, 'ticker': ticker, 'side': 'short',
                'confidence': row['pred_proba'], 'return': -ret,
            })

        if period_long_rets and period_short_rets:
            avg_long = np.mean(period_long_rets)
            avg_short = np.mean(period_short_rets)
            # Equal weight long-short
            ls_ret = (avg_long + avg_short) / 2

            # SPY return for same period (benchmark)
            spy_ret = 0
            if rb_date in spy.index:
                spy_idx = spy.index.get_loc(rb_date)
                spy_exit = min(spy_idx + hold_days, len(spy) - 1)
                spy_ret = spy['close'].iloc[spy_exit] / spy['close'].iloc[spy_idx] - 1

            portfolio_returns.append({
                'date': rb_date,
                'ls_return': ls_ret,
                'long_return': avg_long,
                'short_return': avg_short,
                'spy_return': spy_ret,
                'n_longs': len(period_long_rets),
                'n_shorts': len(period_short_rets),
            })

    if not portfolio_returns:
        print("No valid portfolio periods!")
        return None, None

    port_df = pd.DataFrame(portfolio_returns)
    picks_df = pd.DataFrame(all_picks)

    # Compound returns
    ls_cumret = (1 + port_df['ls_return']).cumprod()
    long_cumret = (1 + port_df['long_return']).cumprod()
    spy_cumret = (1 + port_df['spy_return']).cumprod()

    total_ls = ls_cumret.iloc[-1] - 1
    total_long = long_cumret.iloc[-1] - 1
    total_spy = spy_cumret.iloc[-1] - 1

    n_years = (port_df['date'].max() - port_df['date'].min()).days / 365.25
    if n_years > 0:
        cagr_ls = (1 + total_ls) ** (1 / n_years) - 1
        cagr_long = (1 + total_long) ** (1 / n_years) - 1
        cagr_spy = (1 + total_spy) ** (1 / n_years) - 1
    else:
        cagr_ls = cagr_long = cagr_spy = 0

    periods_per_year = 252 / hold_days
    if port_df['ls_return'].std() > 0:
        sharpe_ls = port_df['ls_return'].mean() / port_df['ls_return'].std() * np.sqrt(periods_per_year)
        downside = port_df[port_df['ls_return'] < 0]['ls_return'].std()
        sortino_ls = port_df['ls_return'].mean() / downside * np.sqrt(periods_per_year) if downside > 0 else 0
    else:
        sharpe_ls = sortino_ls = 0

    # Long-only metrics
    if port_df['long_return'].std() > 0:
        sharpe_long = port_df['long_return'].mean() / port_df['long_return'].std() * np.sqrt(periods_per_year)
    else:
        sharpe_long = 0

    # Max drawdown
    running_max = ls_cumret.cummax()
    drawdowns = ls_cumret / running_max - 1
    max_dd = drawdowns.min()

    win_rate = (port_df['ls_return'] > 0).mean()
    profit_factor = (port_df[port_df['ls_return'] > 0]['ls_return'].sum() /
                     abs(port_df[port_df['ls_return'] < 0]['ls_return'].sum())
                     if (port_df['ls_return'] < 0).any() else float('inf'))

    # Correlation with SPY (key test — should be near 0 for market-neutral)
    spy_corr = port_df['ls_return'].corr(port_df['spy_return'])

    print(f"\nLong-Short Portfolio ({len(port_df)} periods, {n_years:.1f} years):")
    print(f"  Total Return:   LS={total_ls * 100:.1f}%  Long-only={total_long * 100:.1f}%  SPY={total_spy * 100:.1f}%")
    print(f"  CAGR:           LS={cagr_ls * 100:.1f}%  Long-only={cagr_long * 100:.1f}%  SPY={cagr_spy * 100:.1f}%")
    print(f"  Sharpe:         LS={sharpe_ls:.2f}  Long-only={sharpe_long:.2f}")
    print(f"  Sortino (LS):   {sortino_ls:.2f}")
    print(f"  Max DD (LS):    {max_dd * 100:.1f}%")
    print(f"  Win Rate (LS):  {win_rate * 100:.1f}%")
    print(f"  Profit Factor:  {profit_factor:.2f}")
    print(f"  SPY Correlation:{spy_corr:.3f}  (want near 0 for market-neutral)")

    # Per-year breakdown
    port_df['year'] = port_df['date'].dt.year
    print(f"\n  Per-Year Long-Short Returns:")
    for year, grp in port_df.groupby('year'):
        yr_ret = (1 + grp['ls_return']).prod() - 1
        yr_spy = (1 + grp['spy_return']).prod() - 1
        yr_sharpe = grp['ls_return'].mean() / grp['ls_return'].std() * np.sqrt(periods_per_year) if grp['ls_return'].std() > 0 else 0
        print(f"    {year}: LS={yr_ret * 100:+.1f}%  SPY={yr_spy * 100:+.1f}%  Sharpe={yr_sharpe:.2f}  N={len(grp)}")

    return port_df, picks_df


# ============================================================================
# PHASE 5: VALIDATION
# ============================================================================

def regime_test(oot_preds, prices, target_col='target_excess_60d_3pct'):
    """
    R1 regime-agnostic test: check if model works in both up and down markets.
    This is the KEY test that v1 failed.
    """
    print(f"\n{'=' * 80}")
    print("PHASE 5A: REGIME TEST (R1) — KEY GATE")
    print(f"{'=' * 80}")

    spy = prices[prices['ticker'] == 'SPY'].set_index('date').sort_index()
    if len(spy) == 0:
        print("No SPY data for regime test!")
        return None

    spy_monthly = spy['close'].resample('ME').last().pct_change()

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

    preds = oot_preds.copy()
    preds['month'] = preds['date'].dt.to_period('M')
    preds['regime'] = preds['month'].map(monthly_regimes).fillna('flat')

    regime_results = {}
    for regime in ['green', 'red', 'flat']:
        subset = preds[preds['regime'] == regime]
        if len(subset) < 50:
            continue

        acc = accuracy_score(subset[target_col], subset['pred'])
        prec = precision_score(subset[target_col], subset['pred'], zero_division=0)
        rec = recall_score(subset[target_col], subset['pred'], zero_division=0)

        signals = subset[subset['pred'] == 1]
        hit_rate = signals[target_col].mean() if len(signals) > 0 else 0
        n_signals = len(signals)

        # If we have excess return data, compute avg excess return of signals
        avg_excess = 0
        if 'fwd_60d_excess' in signals.columns and len(signals) > 0:
            avg_excess = signals['fwd_60d_excess'].mean()

        regime_results[regime] = {
            'n': len(subset), 'n_signals': n_signals,
            'acc': acc, 'prec': prec, 'rec': rec,
            'hit_rate': hit_rate, 'avg_excess': avg_excess,
        }

        print(f"  {regime:>5} regime: N={len(subset):,}, Signals={n_signals:,}, "
              f"Prec={prec:.3f}, Hit={hit_rate:.3f}, AvgExcess={avg_excess:.4f}")

    # R1 check: compare green vs red hit rates
    if 'green' in regime_results and 'red' in regime_results:
        green_hit = regime_results['green']['hit_rate']
        red_hit = regime_results['red']['hit_rate']
        green_excess = regime_results['green']['avg_excess']
        red_excess = regime_results['red']['avg_excess']

        if max(abs(green_hit), abs(red_hit)) > 0:
            hit_skew = abs(green_hit - red_hit) / max(abs(green_hit), abs(red_hit), 0.001)
        else:
            hit_skew = 0

        print(f"\n  R1 Regime Skew Analysis:")
        print(f"    Green hit rate: {green_hit:.3f}  |  Red hit rate: {red_hit:.3f}")
        print(f"    Green avg excess: {green_excess:.4f}  |  Red avg excess: {red_excess:.4f}")
        print(f"    Hit rate skew: {hit_skew:.3f}")

        if hit_skew > 0.50:
            print(f"    *** R1 FAIL: Regime skew {hit_skew:.3f} > 0.50 — still regime-dependent ***")
            return False
        else:
            print(f"    *** R1 PASS: Regime skew {hit_skew:.3f} <= 0.50 — regime-neutral! ***")
            return True
    else:
        print("  Cannot compute R1 — insufficient data for one or more regimes")
        return None


def permutation_test(panel, oot_preds, target_col='target_excess_60d_3pct', n_perms=100):
    """Permutation test: is the model better than random?"""
    print(f"\n{'=' * 80}")
    print(f"PHASE 5B: PERMUTATION TEST ({n_perms} trials)")
    print(f"{'=' * 80}")

    real_signals = oot_preds[oot_preds['pred'] == 1]
    if len(real_signals) == 0:
        print("No positive predictions!")
        return None

    real_precision = real_signals[target_col].mean()
    real_n = len(real_signals)

    valid = panel.dropna(subset=[target_col])
    perm_precisions = []
    for i in range(n_perms):
        sample = valid.sample(n=min(real_n, len(valid)), replace=False, random_state=i)
        perm_precisions.append(sample[target_col].mean())

    perm_mean = np.mean(perm_precisions)
    perm_std = np.std(perm_precisions)
    z_score = (real_precision - perm_mean) / max(perm_std, 0.001)
    p_value = 1 - norm.cdf(z_score)

    print(f"Real model precision: {real_precision:.4f} (N={real_n:,})")
    print(f"Permutation mean:     {perm_mean:.4f} +/- {perm_std:.4f}")
    print(f"Z-score: {z_score:.2f}")
    print(f"P-value: {p_value:.6f}")
    if p_value < 0.05:
        print("*** SIGNIFICANT: Model beats random ***")
    elif p_value < 0.10:
        print("*** MARGINAL: p < 0.10 but not < 0.05 ***")
    else:
        print("*** NOT SIGNIFICANT ***")

    return {'real_precision': real_precision, 'perm_mean': perm_mean,
            'z_score': z_score, 'p_value': p_value}


def per_year_breakdown(oot_preds, target_col='target_excess_60d_3pct'):
    """Per-year model performance."""
    print(f"\n{'=' * 80}")
    print("PHASE 5C: PER-YEAR BREAKDOWN")
    print(f"{'=' * 80}")

    preds = oot_preds.copy()
    preds['year'] = preds['date'].dt.year

    print(f"{'Year':>6} {'N':>8} {'BaseRate':>9} {'Acc':>8} {'Prec':>8} {'Signals':>8} {'Hit%':>8}")
    print("-" * 65)

    for year in sorted(preds['year'].unique()):
        subset = preds[preds['year'] == year]
        if len(subset) < 50:
            continue

        base_rate = subset[target_col].mean()
        acc = accuracy_score(subset[target_col], subset['pred'])
        prec = precision_score(subset[target_col], subset['pred'], zero_division=0)
        n_signals = (subset['pred'] == 1).sum()
        hit = subset.loc[subset['pred'] == 1, target_col].mean() if n_signals > 0 else 0

        print(f"  {year:>4} {len(subset):>8,} {base_rate:>9.3f} {acc:>8.3f} {prec:>8.3f} {n_signals:>8,} {hit * 100:>7.1f}%")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("STOCK PREDICTION v2 — RELATIVE RETURNS (Regime-Neutral)")
    print("Predicting excess returns vs SPY to strip out market beta")
    print("=" * 80)
    t0 = time.time()

    # Phase 1: Data
    print(f"\n{'=' * 80}")
    print("PHASE 1: DATA COLLECTION")
    print(f"{'=' * 80}")

    prices = download_price_data(use_cache=True)
    earnings_data = get_earnings_data(UNIVERSE, use_cache=True)
    insider_df = load_insider_data()

    # Phase 2: Feature engineering
    print(f"\n{'=' * 80}")
    print("PHASE 2: FEATURE ENGINEERING")
    print(f"{'=' * 80}")

    forward_returns = compute_forward_excess_returns(prices)
    relative_strength = compute_relative_strength_features(prices)
    earnings_features = compute_earnings_features(prices, earnings_data)
    insider_features = compute_insider_features(prices, insider_df)

    panel = build_master_panel(prices, earnings_features, insider_features,
                               forward_returns, relative_strength)
    panel.to_parquet(OUTPUT_DIR / "master_panel_v2.parquet", index=False)

    # Print base rates for all targets
    print("\nBase rates for excess return targets:")
    for target in ['target_excess_60d_3pct', 'target_excess_60d_5pct', 'target_excess_30d_2pct']:
        if target in panel.columns:
            valid = panel[target].dropna()
            print(f"  {target}: {valid.mean():.3f} ({valid.mean() * 100:.1f}%)")

    # Phase 3: Walk-forward models for each target
    all_results = {}

    targets = [
        ('target_excess_60d_3pct', '60d excess > 3%'),
        ('target_excess_60d_5pct', '60d excess > 5%'),
        ('target_excess_30d_2pct', '30d excess > 2%'),
    ]

    best_oot = None
    best_target = None
    best_fold_df = None

    for target_col, desc in targets:
        if target_col not in panel.columns:
            print(f"\nSkipping {target_col} — not in panel")
            continue

        print(f"\n{'#' * 80}")
        print(f"# TARGET: {desc} ({target_col})")
        print(f"{'#' * 80}")

        oot, fold_df, model = walk_forward_model(
            panel, target_col=target_col,
            train_days=252, test_days=21, embargo_days=60,
        )

        if oot is not None:
            oot.to_parquet(OUTPUT_DIR / f"oot_predictions_{target_col}.parquet", index=False)
            fold_df.to_csv(OUTPUT_DIR / f"fold_results_{target_col}.csv", index=False)

            all_results[target_col] = {
                'desc': desc,
                'n_folds': len(fold_df),
                'avg_precision': float(fold_df['precision'].mean()),
                'avg_accuracy': float(fold_df['accuracy'].mean()),
                'avg_recall': float(fold_df['recall'].mean()),
                'avg_f1': float(fold_df['f1'].mean()),
                'base_rate': float(oot[target_col].mean()),
            }

            # Use 60d/3% as primary
            if target_col == 'target_excess_60d_3pct':
                best_oot = oot
                best_target = target_col
                best_fold_df = fold_df

    if best_oot is None:
        print("\nERROR: No valid OOT predictions for any target!")
        return

    # Phase 4: Long-short portfolio
    port_df, picks_df = long_short_portfolio(
        best_oot, prices, target_col=best_target,
        top_n=10, hold_days=42,
    )
    if port_df is not None:
        port_df.to_csv(OUTPUT_DIR / "long_short_portfolio.csv", index=False)
    if picks_df is not None:
        picks_df.to_csv(OUTPUT_DIR / "portfolio_picks.csv", index=False)

    # Also test with top 5
    port_df_5, _ = long_short_portfolio(
        best_oot, prices, target_col=best_target,
        top_n=5, hold_days=42,
    )

    # Phase 5: Validation
    r1_result = regime_test(best_oot, prices, target_col=best_target)
    perm_result = permutation_test(panel, best_oot, target_col=best_target)
    per_year_breakdown(best_oot, target_col=best_target)

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    elapsed = time.time() - t0
    print(f"\n{'=' * 80}")
    print("FINAL SUMMARY — v2 RELATIVE RETURN PREDICTOR")
    print(f"{'=' * 80}")
    print(f"Runtime: {elapsed / 60:.1f} minutes")
    print(f"Universe: {panel['ticker'].nunique()} stocks (ex-SPY)")
    print(f"Date range: {panel['date'].min().strftime('%Y-%m-%d')} to {panel['date'].max().strftime('%Y-%m-%d')}")
    print(f"Features: {len([c for c in FEATURE_COLS if c in panel.columns])}")
    print(f"Embargo: 60 trading days between train and test")

    print(f"\nResults by target:")
    for target_col, info in all_results.items():
        print(f"\n  {info['desc']} (base rate: {info['base_rate']:.3f}):")
        print(f"    Avg Precision: {info['avg_precision']:.3f}")
        print(f"    Avg Accuracy:  {info['avg_accuracy']:.3f}")
        print(f"    Avg Recall:    {info['avg_recall']:.3f}")
        print(f"    Precision lift vs base: {info['avg_precision'] / info['base_rate']:.2f}x" if info['base_rate'] > 0 else "")

    if r1_result is True:
        print(f"\n  R1 REGIME TEST: PASS")
    elif r1_result is False:
        print(f"\n  R1 REGIME TEST: FAIL")
    else:
        print(f"\n  R1 REGIME TEST: INCONCLUSIVE")

    if perm_result:
        print(f"  Permutation test p-value: {perm_result['p_value']:.6f}")

    # Save summary
    summary = {
        'version': 'v2_relative',
        'runtime_minutes': elapsed / 60,
        'n_stocks': int(panel['ticker'].nunique()),
        'n_features': len([c for c in FEATURE_COLS if c in panel.columns]),
        'date_range': f"{panel['date'].min().strftime('%Y-%m-%d')} to {panel['date'].max().strftime('%Y-%m-%d')}",
        'embargo_days': 60,
        'train_window': 252,
        'test_step': 21,
        'targets': all_results,
        'r1_regime_pass': r1_result,
    }

    if perm_result:
        summary['permutation_p_value'] = perm_result['p_value']
        summary['permutation_z_score'] = perm_result['z_score']

    with open(OUTPUT_DIR / "model_summary_v2.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # MLflow logging
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("stock_prediction_v2_relative")
        with mlflow.start_run(run_name=f"v2_relative_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            mlflow.log_params({
                'version': 'v2_relative',
                'universe_size': len(UNIVERSE),
                'train_window': 252,
                'test_step': 21,
                'embargo_days': 60,
                'target': 'excess_60d_3pct',
                'n_features': len([c for c in FEATURE_COLS if c in panel.columns]),
            })
            for target_col, info in all_results.items():
                for k, v in info.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(f"{target_col}_{k}", v)
            if r1_result is not None:
                mlflow.log_metric("r1_regime_pass", 1.0 if r1_result else 0.0)
            if perm_result:
                mlflow.log_metric("permutation_p_value", perm_result['p_value'])
                mlflow.log_metric("permutation_z_score", perm_result['z_score'])
            mlflow.log_artifact(str(OUTPUT_DIR / "model_summary_v2.json"))
        print("\nLogged to MLflow experiment 'stock_prediction_v2_relative'")
    except Exception as e:
        print(f"\nMLflow logging failed (non-fatal): {e}")

    print(f"\nAll results saved to {OUTPUT_DIR}")
    print("DONE.")


if __name__ == "__main__":
    main()
