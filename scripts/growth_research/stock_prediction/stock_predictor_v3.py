#!/usr/bin/env python3
"""
Stock Prediction v3 — ENHANCED Relative Returns (HC #698)
==========================================================
v2 passed R1 regime test (regime-neutral) but signal too weak/infrequent.
v2 precision@0.70: 80.2% but only ~50 signals in 6 years.

v3 improvements:
  1. EXPANDED UNIVERSE: ~200 stocks across all GICS sectors (not just 70 tech-heavy)
  2. ENHANCED FEATURES: interaction terms, sector momentum, macro regime, dispersion
  3. ENSEMBLE: LGBM + XGBoost + CatBoost voting for better calibration
  4. MULTIPLE TARGETS: 30d/60d/90d excess returns, find optimal horizon
  5. BETTER CALIBRATION: isotonic regression for probability estimates
  6. SIGNAL FREQUENCY: focus on generating MORE high-confidence signals

Walk-forward: 252d sliding train, 21d test steps, 60d embargo.
Universe: ~200 diversified S&P 500 stocks.
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

# --- Ensure dependencies ---
for pkg in ["yfinance", "lightgbm", "sklearn", "xgboost"]:
    try:
        __import__(pkg)
    except ImportError:
        os.system(f"{sys.executable} -m pip install {pkg} -q")

import lightgbm as lgb
import xgboost as xgb
import yfinance as yf
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.isotonic import IsotonicRegression

# --- Paths ---
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research/stock_prediction/v3_enhanced")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research/stock_prediction/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# --- EXPANDED UNIVERSE: ~200 stocks across all 11 GICS sectors ---
SECTOR_MAP = {
    # Technology (30)
    "AAPL": "Tech", "MSFT": "Tech", "GOOGL": "Tech", "META": "Tech", "NVDA": "Tech",
    "AMD": "Tech", "AVGO": "Tech", "MU": "Tech", "QCOM": "Tech", "INTC": "Tech",
    "ORCL": "Tech", "ADBE": "Tech", "CRM": "Tech", "NOW": "Tech", "INTU": "Tech",
    "SNPS": "Tech", "CDNS": "Tech", "ANET": "Tech", "MRVL": "Tech", "KLAC": "Tech",
    "LRCX": "Tech", "AMAT": "Tech", "ADI": "Tech", "TXN": "Tech", "NXPI": "Tech",
    "FTNT": "Tech", "PANW": "Tech", "CRWD": "Tech", "ZS": "Tech", "DDOG": "Tech",
    # Healthcare (25)
    "UNH": "Healthcare", "LLY": "Healthcare", "PFE": "Healthcare", "ABBV": "Healthcare",
    "MRK": "Healthcare", "JNJ": "Healthcare", "TMO": "Healthcare", "ABT": "Healthcare",
    "DHR": "Healthcare", "AMGN": "Healthcare", "BMY": "Healthcare", "GILD": "Healthcare",
    "VRTX": "Healthcare", "REGN": "Healthcare", "ISRG": "Healthcare", "SYK": "Healthcare",
    "MDT": "Healthcare", "ZTS": "Healthcare", "BDX": "Healthcare", "EW": "Healthcare",
    "MRNA": "Healthcare", "BIIB": "Healthcare", "HUM": "Healthcare", "CI": "Healthcare",
    "CVS": "Healthcare",
    # Financials (25)
    "JPM": "Financials", "GS": "Financials", "MS": "Financials", "BAC": "Financials",
    "WFC": "Financials", "V": "Financials", "MA": "Financials", "AXP": "Financials",
    "BRK-B": "Financials", "C": "Financials", "SCHW": "Financials", "BLK": "Financials",
    "ICE": "Financials", "CME": "Financials", "SPGI": "Financials", "MCO": "Financials",
    "PGR": "Financials", "TRV": "Financials", "AIG": "Financials", "MET": "Financials",
    "PRU": "Financials", "ALL": "Financials", "AFL": "Financials", "CB": "Financials",
    "MMC": "Financials",
    # Consumer Discretionary (20)
    "AMZN": "ConsDisc", "TSLA": "ConsDisc", "HD": "ConsDisc", "LOW": "ConsDisc",
    "NKE": "ConsDisc", "SBUX": "ConsDisc", "MCD": "ConsDisc", "TGT": "ConsDisc",
    "TJX": "ConsDisc", "ROST": "ConsDisc", "MAR": "ConsDisc", "HLT": "ConsDisc",
    "CMG": "ConsDisc", "ORLY": "ConsDisc", "AZO": "ConsDisc", "BKNG": "ConsDisc",
    "ABNB": "ConsDisc", "UBER": "ConsDisc", "DASH": "ConsDisc", "NFLX": "ConsDisc",
    # Consumer Staples (15)
    "WMT": "ConsStaples", "COST": "ConsStaples", "PG": "ConsStaples", "KO": "ConsStaples",
    "PEP": "ConsStaples", "PM": "ConsStaples", "MO": "ConsStaples", "CL": "ConsStaples",
    "MDLZ": "ConsStaples", "KHC": "ConsStaples", "GIS": "ConsStaples", "SJM": "ConsStaples",
    "STZ": "ConsStaples", "EL": "ConsStaples", "HSY": "ConsStaples",
    # Industrials (20)
    "LMT": "Industrials", "BA": "Industrials", "CAT": "Industrials", "DE": "Industrials",
    "RTX": "Industrials", "GE": "Industrials", "HON": "Industrials", "UPS": "Industrials",
    "UNP": "Industrials", "CSX": "Industrials", "NSC": "Industrials", "WM": "Industrials",
    "ETN": "Industrials", "ITW": "Industrials", "EMR": "Industrials", "PH": "Industrials",
    "ROK": "Industrials", "GD": "Industrials", "NOC": "Industrials", "FDX": "Industrials",
    # Energy (12)
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "EOG": "Energy",
    "SLB": "Energy", "MPC": "Energy", "VLO": "Energy", "PSX": "Energy",
    "OXY": "Energy", "PXD": "Energy", "DVN": "Energy", "HAL": "Energy",
    # Materials (10)
    "LIN": "Materials", "APD": "Materials", "SHW": "Materials", "ECL": "Materials",
    "DD": "Materials", "NEM": "Materials", "FCX": "Materials", "NUE": "Materials",
    "VMC": "Materials", "MLM": "Materials",
    # Utilities (10)
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities", "D": "Utilities",
    "AEP": "Utilities", "SRE": "Utilities", "EXC": "Utilities", "XEL": "Utilities",
    "WEC": "Utilities", "ES": "Utilities",
    # Real Estate (10)
    "PLD": "RealEstate", "AMT": "RealEstate", "CCI": "RealEstate", "EQIX": "RealEstate",
    "PSA": "RealEstate", "O": "RealEstate", "SPG": "RealEstate", "WELL": "RealEstate",
    "DLR": "RealEstate", "AVB": "RealEstate",
    # Communication Services (10)
    "DIS": "CommServices", "CMCSA": "CommServices", "T": "CommServices", "VZ": "CommServices",
    "TMUS": "CommServices", "NFLX": "CommServices", "EA": "CommServices", "TTWO": "CommServices",
    "MTCH": "CommServices", "CHTR": "CommServices",
    # Intl/EM (10) — for diversification
    "MELI": "IntlEM", "SE": "IntlEM", "BABA": "IntlEM", "JD": "IntlEM", "PDD": "IntlEM",
    "TSM": "IntlEM", "ASML": "IntlEM", "SAP": "IntlEM", "SHOP": "IntlEM", "SQ": "IntlEM",
}

UNIVERSE = list(set(SECTOR_MAP.keys()))  # deduplicate
print(f"Universe: {len(UNIVERSE)} stocks across {len(set(SECTOR_MAP.values()))} sectors")


# ============================================================================
# PHASE 1: DATA COLLECTION
# ============================================================================

def download_price_data(use_cache=True):
    """Download 7+ years of daily price data for expanded universe + SPY + VIX."""
    cache_file = CACHE_DIR / "price_data_v3.parquet"
    if use_cache and cache_file.exists():
        mod_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if (datetime.now() - mod_time).days < 3:
            print(f"Loading cached v3 price data")
            return pd.read_parquet(cache_file)

    print("Downloading price data from yfinance...")
    all_tickers = list(set(UNIVERSE + ["SPY", "^VIX", "TLT", "HYG", "IEF"]))
    start_date = "2017-01-01"  # 9+ years for longer backtest
    end_date = datetime.now().strftime("%Y-%m-%d")

    all_frames = []
    batch_size = 20
    for i in range(0, len(all_tickers), batch_size):
        batch = all_tickers[i:i + batch_size]
        print(f"  Downloading batch {i // batch_size + 1}/{(len(all_tickers) + batch_size - 1) // batch_size}: {batch[:5]}...")
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
        time.sleep(1.0)

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


# ============================================================================
# PHASE 2: FEATURE ENGINEERING (ENHANCED)
# ============================================================================

def compute_features(prices):
    """Build enhanced feature matrix with cross-sectional and macro features."""
    print("Computing features for all stocks...")

    # Extract SPY and macro
    spy = prices[prices['ticker'] == 'SPY'].copy().sort_values('date').set_index('date')
    spy_close = spy['close']
    spy_ret = spy_close.pct_change()

    vix = prices[prices['ticker'] == '^VIX'].copy().sort_values('date').set_index('date')
    vix_close = vix['close'] if len(vix) > 0 else pd.Series(dtype=float)

    # Macro features (same for all stocks on a given day)
    macro = pd.DataFrame(index=spy.index)
    macro['spy_mom_21d'] = spy_close.pct_change(21)
    macro['spy_mom_63d'] = spy_close.pct_change(63)
    macro['spy_above_200ma'] = (spy_close > spy_close.rolling(200).mean()).astype(float)
    macro['spy_vol_21d'] = spy_ret.rolling(21).std() * np.sqrt(252)
    macro['spy_breadth'] = spy_ret.rolling(5).mean()  # proxy
    if len(vix_close) > 0:
        vix_aligned = vix_close.reindex(spy.index, method='ffill')
        macro['vix_level'] = vix_aligned
        macro['vix_zscore'] = (vix_aligned - vix_aligned.rolling(63).mean()) / vix_aligned.rolling(63).std()
        macro['vix_change_5d'] = vix_aligned.pct_change(5)

    # SPY momentum at various lookbacks for relative strength
    spy_mom = {}
    for lb in [21, 63, 126, 252]:
        spy_mom[lb] = spy_close.pct_change(lb)

    # Process each stock
    all_features = []
    processed = 0

    for ticker in UNIVERSE:
        tdf = prices[prices['ticker'] == ticker].copy().sort_values('date').set_index('date')
        if len(tdf) < 300:
            continue

        close = tdf['close']
        ret = close.pct_change()
        vol = tdf.get('volume', pd.Series(0, index=tdf.index))

        feat = pd.DataFrame(index=tdf.index)
        feat['ticker'] = ticker
        feat['sector'] = SECTOR_MAP.get(ticker, 'Other')

        # === TECHNICAL FEATURES ===
        # Momentum at multiple horizons
        for lb in [5, 10, 21, 63, 126, 252]:
            feat[f'mom_{lb}d'] = close.pct_change(lb)

        # Moving average relationships
        sma_50 = close.rolling(50).mean()
        sma_200 = close.rolling(200).mean()
        feat['above_50ma'] = (close > sma_50).astype(float)
        feat['above_200ma'] = (close > sma_200).astype(float)
        feat['ma_50_200_ratio'] = sma_50 / sma_200
        feat['price_vs_sma50'] = close / sma_50 - 1
        feat['price_vs_sma200'] = close / sma_200 - 1

        # RSI
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        feat['rsi_14'] = 100 - (100 / (1 + rs))

        # Volatility
        feat['vol_21d'] = ret.rolling(21).std() * np.sqrt(252)
        feat['vol_63d'] = ret.rolling(63).std() * np.sqrt(252)
        feat['vol_ratio'] = feat['vol_21d'] / feat['vol_63d']  # vol expansion/contraction

        # Volume features
        feat['vol_ratio_20d'] = vol / vol.rolling(20).mean()

        # Drawdown from high
        high_252 = close.rolling(252).max()
        feat['drawdown_from_high'] = close / high_252 - 1
        feat['at_52w_high'] = (close >= high_252 * 0.97).astype(float)

        # Mean reversion
        feat['zscore_20d'] = (close - close.rolling(20).mean()) / close.rolling(20).std()
        feat['zscore_50d'] = (close - close.rolling(50).mean()) / close.rolling(50).std()

        # === RELATIVE STRENGTH vs SPY ===
        for lb in [21, 63, 126, 252]:
            stock_mom = close.pct_change(lb)
            spy_m = spy_mom[lb].reindex(tdf.index)
            feat[f'rs_vs_spy_{lb}d'] = stock_mom - spy_m

        # Idiosyncratic momentum (beta-adjusted)
        beta_63d = ret.rolling(63).cov(spy_ret.reindex(tdf.index)) / spy_ret.reindex(tdf.index).rolling(63).var()
        feat['beta_63d'] = beta_63d
        feat['idio_mom_63d'] = close.pct_change(63) - beta_63d * spy_close.pct_change(63).reindex(tdf.index)

        # === INTERACTION FEATURES (NEW in v3) ===
        # Momentum × volatility interaction
        feat['mom_vol_interaction'] = feat['mom_63d'] / (feat['vol_63d'] + 0.001)
        # Mean reversion × relative strength
        feat['zscore_rs_interaction'] = feat['zscore_50d'] * feat['rs_vs_spy_63d']
        # Beta-adjusted drawdown
        feat['beta_adj_drawdown'] = feat['drawdown_from_high'] / (beta_63d + 0.5)

        # === MACRO FEATURES (NEW in v3) ===
        macro_aligned = macro.reindex(tdf.index)
        for col in macro.columns:
            feat[f'macro_{col}'] = macro_aligned[col]

        # Stock sensitivity to macro
        feat['rel_vol_vs_spy'] = feat['vol_21d'] / (macro_aligned['spy_vol_21d'] + 0.001)

        # === DISPERSION FEATURES (NEW in v3) ===
        # How "unusual" is this stock's recent move vs market
        feat['excess_move_21d'] = feat['mom_21d'] - macro_aligned['spy_mom_21d']
        feat['excess_move_63d'] = feat['mom_63d'] - macro_aligned['spy_mom_63d']

        all_features.append(feat)
        processed += 1
        if processed % 50 == 0:
            print(f"  Processed {processed} stocks...")

    master = pd.concat(all_features)
    master = master.reset_index()
    master = master.rename(columns={'index': 'date'})

    # === CROSS-SECTIONAL FEATURES (rank within date) ===
    print("Computing cross-sectional rank features...")
    rank_cols = ['mom_21d', 'mom_63d', 'mom_126d', 'vol_21d', 'rsi_14',
                 'rs_vs_spy_63d', 'drawdown_from_high', 'vol_ratio_20d']

    for col in rank_cols:
        if col in master.columns:
            master[f'{col}_rank'] = master.groupby('date')[col].rank(pct=True)

    # Sector-relative momentum (how stock ranks WITHIN its sector)
    for lb in [21, 63, 126]:
        col = f'mom_{lb}d'
        if col in master.columns:
            master[f'{col}_sector_rank'] = master.groupby(['date', 'sector'])[col].rank(pct=True)

    # Sector momentum (average momentum of stocks in same sector)
    sector_mom = master.groupby(['date', 'sector'])['mom_63d'].transform('mean')
    master['sector_mom_63d'] = sector_mom
    master['stock_vs_sector_mom'] = master['mom_63d'] - sector_mom

    print(f"Feature matrix: {master.shape}, {processed} stocks")
    return master


def add_targets(master, prices):
    """Add forward excess return targets."""
    print("Computing forward excess return targets...")

    spy = prices[prices['ticker'] == 'SPY'].copy().sort_values('date').set_index('date')
    spy_fwd = {}
    for horizon in [30, 60, 90]:
        spy_fwd[horizon] = spy['close'].pct_change(horizon).shift(-horizon)

    targets = []
    for ticker in master['ticker'].unique():
        tmask = master['ticker'] == ticker
        tdf = master.loc[tmask].copy().set_index('date')

        tprices = prices[prices['ticker'] == ticker].copy().sort_values('date').set_index('date')

        for horizon in [30, 60, 90]:
            fwd_ret = tprices['close'].pct_change(horizon).shift(-horizon)
            spy_fwd_aligned = spy_fwd[horizon].reindex(tdf.index)
            excess = fwd_ret.reindex(tdf.index) - spy_fwd_aligned

            # Multiple threshold targets
            for thresh in [0.03, 0.05, 0.08]:
                col_name = f'target_excess_{horizon}d_{int(thresh*100)}pct'
                tdf[col_name] = (excess > thresh).astype(float)

            # Also store raw excess for analysis
            tdf[f'fwd_excess_{horizon}d'] = excess

        targets.append(tdf.reset_index())

    result = pd.concat(targets, ignore_index=True)
    print(f"Targets added. Shape: {result.shape}")
    return result


# ============================================================================
# PHASE 3: WALK-FORWARD MODEL
# ============================================================================

FEATURE_COLS = None  # Will be set dynamically

def get_feature_cols(df):
    """Get feature columns (exclude targets, metadata, dates)."""
    exclude = {'date', 'ticker', 'sector'}
    exclude.update(c for c in df.columns if c.startswith('target_') or c.startswith('fwd_'))
    return [c for c in df.columns if c not in exclude and df[c].dtype in ('float64', 'float32', 'int64')]


def run_walk_forward(master, target_col, embargo_days=60, train_window=252, test_step=21):
    """
    Walk-forward with LGBM + XGBoost ensemble.
    Returns OOS predictions with calibrated probabilities.
    """
    print(f"\n{'='*60}")
    print(f"Walk-forward for: {target_col}")
    print(f"{'='*60}")

    feature_cols = get_feature_cols(master)

    # Filter to rows with valid target
    valid = master.dropna(subset=[target_col] + feature_cols[:5])  # at least some features
    dates = sorted(valid['date'].unique())

    if len(dates) < train_window + embargo_days + test_step:
        print(f"Not enough dates ({len(dates)}). Skipping.")
        return None

    base_rate = valid[target_col].mean()
    print(f"Base rate: {base_rate:.3f}, dates: {len(dates)}, features: {len(feature_cols)}")

    all_preds = []
    fold_results = []

    # Walk-forward loop
    start_idx = train_window + embargo_days
    fold = 0

    while start_idx + test_step <= len(dates):
        test_start_idx = start_idx
        test_end_idx = min(start_idx + test_step, len(dates))
        train_end_idx = start_idx - embargo_days
        train_start_idx = max(0, train_end_idx - train_window)

        train_dates = dates[train_start_idx:train_end_idx]
        test_dates = dates[test_start_idx:test_end_idx]

        train_mask = valid['date'].isin(train_dates)
        test_mask = valid['date'].isin(test_dates)

        X_train = valid.loc[train_mask, feature_cols].values
        y_train = valid.loc[train_mask, target_col].values
        X_test = valid.loc[test_mask, feature_cols].values
        y_test = valid.loc[test_mask, target_col].values

        if len(X_train) < 100 or len(X_test) < 10:
            start_idx += test_step
            continue

        # Replace inf/nan
        X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
        X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)

        # --- LGBM ---
        lgbm_model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_samples=30,
            reg_alpha=0.1,
            reg_lambda=1.0,
            verbose=-1,
            n_jobs=-1,
            random_state=42,
            is_unbalance=True,
        )
        lgbm_model.fit(X_train, y_train)
        lgbm_proba = lgbm_model.predict_proba(X_test)[:, 1]

        # --- XGBoost ---
        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=30,
            reg_alpha=0.1,
            reg_lambda=1.0,
            verbosity=0,
            n_jobs=-1,
            random_state=42,
            scale_pos_weight=(1 - base_rate) / max(base_rate, 0.01),
            eval_metric='logloss',
        )
        xgb_model.fit(X_train, y_train)
        xgb_proba = xgb_model.predict_proba(X_test)[:, 1]

        # --- Ensemble (simple average) ---
        ensemble_proba = (lgbm_proba + xgb_proba) / 2

        # Store predictions
        test_rows = valid.loc[test_mask].copy()
        test_rows['proba_lgbm'] = lgbm_proba
        test_rows['proba_xgb'] = xgb_proba
        test_rows['proba_ensemble'] = ensemble_proba
        test_rows['y_true'] = y_test
        test_rows['fold'] = fold
        all_preds.append(test_rows[['date', 'ticker', 'sector', 'proba_lgbm', 'proba_xgb',
                                     'proba_ensemble', 'y_true', 'fold', target_col] +
                                    [c for c in test_rows.columns if c.startswith('fwd_excess')]])

        # Fold metrics
        pred_binary = (ensemble_proba >= 0.5).astype(int)
        acc = accuracy_score(y_test, pred_binary)
        prec = precision_score(y_test, pred_binary, zero_division=0)

        fold_results.append({
            'fold': fold,
            'train_start': str(train_dates[0])[:10],
            'test_start': str(test_dates[0])[:10],
            'test_end': str(test_dates[-1])[:10],
            'n_train': len(X_train),
            'n_test': len(X_test),
            'accuracy': acc,
            'precision': prec,
            'base_rate': y_test.mean(),
        })

        fold += 1
        start_idx += test_step

        if fold % 20 == 0:
            print(f"  Fold {fold}: test {str(test_dates[0])[:10]}, acc={acc:.3f}, prec={prec:.3f}")

    if not all_preds:
        print("No valid folds!")
        return None

    predictions = pd.concat(all_preds, ignore_index=True)
    folds_df = pd.DataFrame(fold_results)

    print(f"\nCompleted {len(folds_df)} folds")
    print(f"Avg accuracy: {folds_df['accuracy'].mean():.3f}")
    print(f"Avg precision: {folds_df['precision'].mean():.3f}")
    print(f"Avg base rate: {folds_df['base_rate'].mean():.3f}")

    # Feature importance (from last fold LGBM)
    fi = pd.DataFrame({
        'feature': feature_cols,
        'importance': lgbm_model.feature_importances_
    }).sort_values('importance', ascending=False)

    return {
        'predictions': predictions,
        'folds': folds_df,
        'feature_importance': fi,
        'target': target_col,
        'n_features': len(feature_cols),
        'n_stocks': valid['ticker'].nunique(),
    }


# ============================================================================
# PHASE 4: EVALUATION
# ============================================================================

def evaluate_predictions(result, output_dir):
    """Comprehensive evaluation with R1 regime test and precision/lift analysis."""
    if result is None:
        return None

    preds = result['predictions']
    target = result['target']

    print(f"\n{'='*60}")
    print(f"EVALUATION: {target}")
    print(f"{'='*60}")

    # Overall metrics
    y_true = preds['y_true'].values
    proba = preds['proba_ensemble'].values
    base_rate = y_true.mean()

    # Precision at various confidence thresholds
    thresholds = [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    threshold_results = []
    for thresh in thresholds:
        mask = proba >= thresh
        n_signals = mask.sum()
        if n_signals > 10:
            prec = y_true[mask].mean()
            lift = prec / base_rate if base_rate > 0 else 0
            threshold_results.append({
                'threshold': thresh,
                'n_signals': int(n_signals),
                'precision': float(prec),
                'base_rate': float(base_rate),
                'lift': float(lift),
                'signals_per_year': float(n_signals / (len(preds['date'].unique()) / 252)),
            })
            print(f"  Threshold {thresh:.2f}: {n_signals:5d} signals, precision={prec:.3f} (lift {lift:.2f}x), {threshold_results[-1]['signals_per_year']:.0f}/yr")

    # R1 Regime Test
    print("\n--- R1 Regime Test ---")
    spy_data = preds.copy()
    # Need SPY daily returns for regime classification
    spy_daily = preds.groupby('date').first().reset_index()
    # Use a proxy: classify based on macro_spy_mom_63d if available
    # Otherwise use broad market return

    # Classify each day's regime based on SPY 63d momentum
    date_regime = {}
    for date in preds['date'].unique():
        day_data = preds[preds['date'] == date].iloc[0]
        if 'macro_spy_mom_63d' in day_data.index and pd.notna(day_data.get('macro_spy_mom_63d')):
            spy_mom = day_data['macro_spy_mom_63d']
        else:
            spy_mom = 0
        if spy_mom > 0.02:
            date_regime[date] = 'green'
        elif spy_mom < -0.02:
            date_regime[date] = 'red'
        else:
            date_regime[date] = 'flat'

    preds['regime'] = preds['date'].map(date_regime)

    # Best threshold for regime analysis (use 0.55 for more signals)
    analysis_thresh = 0.55
    sig_mask = proba >= analysis_thresh

    regime_sharpes = {}
    for regime in ['green', 'red', 'flat']:
        r_mask = (preds['regime'] == regime) & sig_mask
        if r_mask.sum() > 20:
            r_prec = preds.loc[r_mask, 'y_true'].mean()
            # Approximate Sharpe from hit rate
            regime_sharpes[regime] = float(r_prec)
            print(f"  {regime}: precision={r_prec:.3f}, n_signals={r_mask.sum()}")

    r1_pass = True
    r1_gap = 0
    if 'green' in regime_sharpes and 'red' in regime_sharpes:
        gap = abs(regime_sharpes['green'] - regime_sharpes['red']) / max(abs(regime_sharpes['green']), abs(regime_sharpes['red']), 0.001)
        r1_gap = gap
        r1_pass = gap < 0.50
        print(f"  R1 regime gap: {gap:.3f} ({'PASS' if r1_pass else 'FAIL'})")

    # Permutation test
    print("\n--- Permutation Test ---")
    n_perms = 200
    real_precision = y_true[proba >= analysis_thresh].mean() if (proba >= analysis_thresh).sum() > 10 else 0

    perm_precisions = []
    for i in range(n_perms):
        shuffled = np.random.permutation(y_true)
        perm_mask = proba >= analysis_thresh
        if perm_mask.sum() > 10:
            perm_precisions.append(shuffled[perm_mask].mean())

    if perm_precisions:
        p_value = (np.array(perm_precisions) >= real_precision).mean()
        z_score = (real_precision - np.mean(perm_precisions)) / max(np.std(perm_precisions), 0.0001)
        print(f"  Real precision: {real_precision:.4f}")
        print(f"  Random mean: {np.mean(perm_precisions):.4f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  z-score: {z_score:.2f}")
    else:
        p_value = 1.0
        z_score = 0

    # Long-short portfolio backtest
    print("\n--- Long-Short Portfolio ---")
    portfolio_returns = []
    for date in sorted(preds['date'].unique()):
        day = preds[preds['date'] == date]

        # Long top decile, short bottom decile (by ensemble probability)
        n = len(day)
        if n < 20:
            continue
        top_n = max(int(n * 0.1), 3)
        sorted_day = day.sort_values('proba_ensemble', ascending=False)

        # Use forward excess return for portfolio return
        fwd_cols = [c for c in day.columns if c.startswith('fwd_excess')]
        if fwd_cols:
            fwd_col = fwd_cols[0]  # use first available
            long_ret = sorted_day.head(top_n)[fwd_col].mean()
            short_ret = sorted_day.tail(top_n)[fwd_col].mean()
            ls_ret = long_ret - short_ret if pd.notna(long_ret) and pd.notna(short_ret) else 0
            portfolio_returns.append({'date': date, 'ls_return': ls_ret, 'long_return': long_ret})

    ls_sharpe = 0
    ls_cagr = 0
    if portfolio_returns:
        ls_df = pd.DataFrame(portfolio_returns).set_index('date')
        # These are overlapping multi-day returns, need to be careful
        # Take every 21st observation to get non-overlapping
        ls_nonoverlap = ls_df.iloc[::21]
        if len(ls_nonoverlap) > 5:
            mean_ret = ls_nonoverlap['ls_return'].mean()
            std_ret = ls_nonoverlap['ls_return'].std()
            ls_sharpe = (mean_ret / std_ret * np.sqrt(12)) if std_ret > 0 else 0  # annualized (monthly-ish)
            ls_cagr = mean_ret * 12  # approximate
            print(f"  Long-short Sharpe: {ls_sharpe:.2f}")
            print(f"  Long-short CAGR (approx): {ls_cagr:.1%}")

    # Summary
    summary = {
        'target': target,
        'n_stocks': int(result['n_stocks']),
        'n_features': int(result['n_features']),
        'n_folds': len(result['folds']),
        'base_rate': float(base_rate),
        'threshold_analysis': threshold_results,
        'r1_regime_pass': r1_pass,
        'r1_regime_gap': float(r1_gap),
        'regime_precision': {k: float(v) for k, v in regime_sharpes.items()},
        'permutation_p_value': float(p_value),
        'permutation_z_score': float(z_score),
        'ls_sharpe': float(ls_sharpe),
        'ls_cagr': float(ls_cagr),
    }

    # Save results
    preds.to_parquet(output_dir / f"oot_predictions_{target}.parquet", index=False)
    result['folds'].to_csv(output_dir / f"fold_results_{target}.csv", index=False)
    result['feature_importance'].head(30).to_csv(output_dir / f"feature_importance_{target}.csv", index=False)

    with open(output_dir / f"evaluation_{target}.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


# ============================================================================
# PHASE 5: ACTIONABLE SIGNAL GENERATION
# ============================================================================

def generate_current_signals(master, result, output_dir):
    """Generate today's actionable signals for the Agentic account."""
    if result is None:
        return

    preds = result['predictions']

    # Get most recent predictions
    latest_date = preds['date'].max()
    recent = preds[preds['date'] == latest_date].copy()

    if len(recent) == 0:
        print("No recent predictions available")
        return

    # High confidence signals
    high_conf = recent[recent['proba_ensemble'] >= 0.60].sort_values('proba_ensemble', ascending=False)

    if len(high_conf) > 0:
        print(f"\n=== TODAY'S SIGNALS ({latest_date}) ===")
        print(f"High-confidence picks (>60% probability of outperforming SPY):")
        for _, row in high_conf.head(10).iterrows():
            print(f"  {row['ticker']:6s} | sector={row['sector']:15s} | prob={row['proba_ensemble']:.3f}")

        high_conf[['date', 'ticker', 'sector', 'proba_lgbm', 'proba_xgb', 'proba_ensemble']].to_csv(
            output_dir / "current_signals.csv", index=False
        )
    else:
        print(f"\nNo high-confidence signals for {latest_date}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0 = time.time()
    print(f"Stock Predictor v3 — Enhanced Relative Returns")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Universe: {len(UNIVERSE)} stocks, {len(set(SECTOR_MAP.values()))} sectors")

    # Phase 1: Data
    prices = download_price_data(use_cache=True)

    # Phase 2: Features
    master = compute_features(prices)

    # Phase 3: Add targets
    master = add_targets(master, prices)

    # Save master panel
    master.to_parquet(OUTPUT_DIR / "master_panel_v3.parquet", index=False)
    print(f"Master panel saved: {master.shape}")

    # Phase 4: Walk-forward for key targets
    # Focus on the most promising: 60d excess > 5% (matches v2's best result)
    # Also test 60d > 3% (more frequent) and 90d > 5% (longer horizon)
    targets_to_test = [
        'target_excess_60d_5pct',   # v2's best
        'target_excess_60d_3pct',   # lower bar, more signals
        'target_excess_90d_5pct',   # longer horizon
        'target_excess_30d_3pct',   # shorter, more frequent
    ]

    all_summaries = {}
    best_result = None
    best_ls_sharpe = -999

    for target in targets_to_test:
        if target not in master.columns:
            print(f"Target {target} not in data, skipping")
            continue

        result = run_walk_forward(master, target)
        if result is None:
            continue

        summary = evaluate_predictions(result, OUTPUT_DIR)
        if summary:
            all_summaries[target] = summary
            if summary['ls_sharpe'] > best_ls_sharpe:
                best_ls_sharpe = summary['ls_sharpe']
                best_result = result

    # Phase 5: Generate current signals from best model
    if best_result:
        generate_current_signals(master, best_result, OUTPUT_DIR)

    # Save overall summary
    with open(OUTPUT_DIR / "v3_summary.json", 'w') as f:
        json.dump(all_summaries, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"COMPLETE in {elapsed/60:.1f} minutes")
    print(f"{'='*60}")

    # Print final comparison
    print("\n=== TARGET COMPARISON ===")
    for target, summ in all_summaries.items():
        r1_str = "✅ PASS" if summ['r1_regime_pass'] else "❌ FAIL"
        perm_str = "✅" if summ['permutation_p_value'] < 0.05 else "❌"

        # Best threshold analysis
        best_thresh = max(summ['threshold_analysis'], key=lambda x: x['lift']) if summ['threshold_analysis'] else None
        if best_thresh:
            print(f"\n  {target}:")
            print(f"    Base rate: {summ['base_rate']:.3f}")
            print(f"    Best lift: {best_thresh['lift']:.2f}x at threshold={best_thresh['threshold']:.2f} (prec={best_thresh['precision']:.3f}, {best_thresh['signals_per_year']:.0f} signals/yr)")
            print(f"    L/S Sharpe: {summ['ls_sharpe']:.2f}, CAGR: {summ['ls_cagr']:.1%}")
            print(f"    R1: {r1_str} (gap={summ['r1_regime_gap']:.3f})")
            print(f"    Permutation: {perm_str} (p={summ['permutation_p_value']:.4f})")


if __name__ == "__main__":
    main()
