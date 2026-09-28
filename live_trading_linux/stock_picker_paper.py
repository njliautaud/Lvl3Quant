#!/usr/bin/env python3
"""
Stock Picker Paper Engine (HC #700 — Agentic Robinhood account growth)
=======================================================================
Weekly paper engine driven by the v3 LGBM+XGBoost ensemble prediction model.
Picks top 5 stocks with >60% confidence of outperforming SPY by 3%+ over 60 days.

Behaviour:
  - Monday 9:45 AM ET: retrain on latest data, pick top 5, add to portfolio
  - Daily: mark-to-market, check stop-losses (10% from entry)
  - 60-day rolling hold: each week's picks roll off after 60 days
  - Equal-weight within the 5-pick cohort added each Monday
  - Commission-free (Robinhood, HC #694)

Starting NAV: $10,000
State: /home/jupiter/Lvl3Quant/data/paper_engines/stock_picker/state.json
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
import warnings
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
ROOT = Path("/home/jupiter/Lvl3Quant")
STATE_DIR = ROOT / "data" / "paper_engines" / "stock_picker"
STATE_DIR.mkdir(parents=True, exist_ok=True)

STATE_FILE   = STATE_DIR / "state.json"
TRADES_LOG   = STATE_DIR / "trades.jsonl"
EQUITY_LOG   = STATE_DIR / "equity_curve.jsonl"
PICKS_LOG    = STATE_DIR / "weekly_picks.jsonl"

V3_CACHE     = ROOT / "output/growth_research/stock_prediction/cache/price_data_v3.parquet"
V3_OUTPUT    = ROOT / "output/growth_research/stock_prediction/v3_enhanced"

WEBHOOK      = "/home/jupiter/teleclaude-main/utils/webhook_notifier.js"

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
STARTING_NAV       = 10_000.0
TOP_K              = 5          # picks per week
HOLD_DAYS          = 60         # calendar days to hold each cohort
STOP_LOSS_PCT      = 0.10       # stop-loss: 10% from entry
CONFIDENCE_THRESH  = 0.60       # minimum ensemble probability
TARGET_COL         = "target_excess_60d_3pct"   # best L/S Sharpe 0.715
CACHE_STALE_HOURS  = 24         # re-download if cache older than this
TRAIN_WINDOW_DAYS  = 252        # ~1yr sliding train
EMBARGO_DAYS       = 60
TEST_STEP_DAYS     = 21

# ---------------------------------------------------------------------------
# v3 UNIVERSE (same as stock_predictor_v3.py)
# ---------------------------------------------------------------------------
SECTOR_MAP = {
    "AAPL": "Tech", "MSFT": "Tech", "GOOGL": "Tech", "META": "Tech", "NVDA": "Tech",
    "AMD": "Tech", "AVGO": "Tech", "MU": "Tech", "QCOM": "Tech", "INTC": "Tech",
    "ORCL": "Tech", "ADBE": "Tech", "CRM": "Tech", "NOW": "Tech", "INTU": "Tech",
    "SNPS": "Tech", "CDNS": "Tech", "ANET": "Tech", "MRVL": "Tech", "KLAC": "Tech",
    "LRCX": "Tech", "AMAT": "Tech", "ADI": "Tech", "TXN": "Tech", "NXPI": "Tech",
    "FTNT": "Tech", "PANW": "Tech", "CRWD": "Tech", "ZS": "Tech", "DDOG": "Tech",
    "UNH": "Healthcare", "LLY": "Healthcare", "PFE": "Healthcare", "ABBV": "Healthcare",
    "MRK": "Healthcare", "JNJ": "Healthcare", "TMO": "Healthcare", "ABT": "Healthcare",
    "DHR": "Healthcare", "AMGN": "Healthcare", "BMY": "Healthcare", "GILD": "Healthcare",
    "VRTX": "Healthcare", "REGN": "Healthcare", "ISRG": "Healthcare", "SYK": "Healthcare",
    "MDT": "Healthcare", "ZTS": "Healthcare", "BDX": "Healthcare", "EW": "Healthcare",
    "MRNA": "Healthcare", "BIIB": "Healthcare", "HUM": "Healthcare", "CI": "Healthcare",
    "CVS": "Healthcare",
    "JPM": "Financials", "GS": "Financials", "MS": "Financials", "BAC": "Financials",
    "WFC": "Financials", "V": "Financials", "MA": "Financials", "AXP": "Financials",
    "BRK-B": "Financials", "C": "Financials", "SCHW": "Financials", "BLK": "Financials",
    "ICE": "Financials", "CME": "Financials", "SPGI": "Financials", "MCO": "Financials",
    "PGR": "Financials", "TRV": "Financials", "AIG": "Financials", "MET": "Financials",
    "PRU": "Financials", "ALL": "Financials", "AFL": "Financials", "CB": "Financials",
    "MMC": "Financials",
    "AMZN": "ConsDisc", "TSLA": "ConsDisc", "HD": "ConsDisc", "LOW": "ConsDisc",
    "NKE": "ConsDisc", "SBUX": "ConsDisc", "MCD": "ConsDisc", "TGT": "ConsDisc",
    "TJX": "ConsDisc", "ROST": "ConsDisc", "MAR": "ConsDisc", "HLT": "ConsDisc",
    "CMG": "ConsDisc", "ORLY": "ConsDisc", "AZO": "ConsDisc", "BKNG": "ConsDisc",
    "ABNB": "ConsDisc", "UBER": "ConsDisc", "DASH": "ConsDisc", "NFLX": "ConsDisc",
    "WMT": "ConsStaples", "COST": "ConsStaples", "PG": "ConsStaples", "KO": "ConsStaples",
    "PEP": "ConsStaples", "PM": "ConsStaples", "MO": "ConsStaples", "CL": "ConsStaples",
    "MDLZ": "ConsStaples", "KHC": "ConsStaples", "GIS": "ConsStaples", "SJM": "ConsStaples",
    "STZ": "ConsStaples", "EL": "ConsStaples", "HSY": "ConsStaples",
    "LMT": "Industrials", "BA": "Industrials", "CAT": "Industrials", "DE": "Industrials",
    "RTX": "Industrials", "GE": "Industrials", "HON": "Industrials", "UPS": "Industrials",
    "UNP": "Industrials", "CSX": "Industrials", "NSC": "Industrials", "WM": "Industrials",
    "ETN": "Industrials", "ITW": "Industrials", "EMR": "Industrials", "PH": "Industrials",
    "ROK": "Industrials", "GD": "Industrials", "NOC": "Industrials", "FDX": "Industrials",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "EOG": "Energy",
    "SLB": "Energy", "MPC": "Energy", "VLO": "Energy", "PSX": "Energy",
    "OXY": "Energy", "PXD": "Energy", "DVN": "Energy", "HAL": "Energy",
    "LIN": "Materials", "APD": "Materials", "SHW": "Materials", "ECL": "Materials",
    "DD": "Materials", "NEM": "Materials", "FCX": "Materials", "NUE": "Materials",
    "VMC": "Materials", "MLM": "Materials",
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities", "D": "Utilities",
    "AEP": "Utilities", "SRE": "Utilities", "EXC": "Utilities", "XEL": "Utilities",
    "WEC": "Utilities", "ES": "Utilities",
    "PLD": "RealEstate", "AMT": "RealEstate", "CCI": "RealEstate", "EQIX": "RealEstate",
    "PSA": "RealEstate", "O": "RealEstate", "SPG": "RealEstate", "WELL": "RealEstate",
    "DLR": "RealEstate", "AVB": "RealEstate",
    "DIS": "CommServices", "CMCSA": "CommServices", "T": "CommServices", "VZ": "CommServices",
    "TMUS": "CommServices", "EA": "CommServices", "TTWO": "CommServices",
    "MTCH": "CommServices", "CHTR": "CommServices",
    "MELI": "IntlEM", "SE": "IntlEM", "BABA": "IntlEM", "JD": "IntlEM", "PDD": "IntlEM",
    "TSM": "IntlEM", "ASML": "IntlEM", "SAP": "IntlEM", "SHOP": "IntlEM", "SQ": "IntlEM",
}
UNIVERSE = sorted(set(SECTOR_MAP.keys()))


# ---------------------------------------------------------------------------
# IMPORTS (install if needed)
# ---------------------------------------------------------------------------
for pkg in ["yfinance", "lightgbm", "sklearn", "xgboost"]:
    try:
        __import__(pkg)
    except ImportError:
        os.system(f"{sys.executable} -m pip install {pkg} -q")

import lightgbm as lgb
import xgboost as xgb
import yfinance as yf


# ---------------------------------------------------------------------------
# DISCORD NOTIFICATION
# ---------------------------------------------------------------------------
def notify(msg: str):
    """Send a brief message to Discord via webhook notifier."""
    safe = msg.replace('"', "'")
    os.system(f'node {WEBHOOK} "{safe}" 2>/dev/null')


# ---------------------------------------------------------------------------
# STATE
# ---------------------------------------------------------------------------
def _load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "nav_usd": STARTING_NAV,
        "cash_usd": STARTING_NAV,
        "positions": {},     # {ticker: {shares, entry_px, entry_date, cohort_id, alloc_usd}}
        "closed_positions": [],
        "n_rebalances": 0,
        "last_rebal_date": None,
        "last_mtm_date": None,
        "total_realized_pnl": 0.0,
        "created": str(datetime.now().date()),
    }


def _save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def _log_trade(rec: dict):
    with open(TRADES_LOG, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def _log_equity(nav: float, date_str: str, notes: str = ""):
    with open(EQUITY_LOG, "a") as f:
        f.write(json.dumps({"date": date_str, "nav_usd": nav, "notes": notes}, default=str) + "\n")


def _log_picks(picks: list, date_str: str, cohort_id: str):
    with open(PICKS_LOG, "a") as f:
        f.write(json.dumps({"date": date_str, "cohort_id": cohort_id, "picks": picks}, default=str) + "\n")


# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------
def load_price_data(force_refresh: bool = False) -> pd.DataFrame:
    """Load price data from cache or re-download if stale."""
    if V3_CACHE.exists() and not force_refresh:
        age_hours = (time.time() - V3_CACHE.stat().st_mtime) / 3600
        if age_hours < CACHE_STALE_HOURS:
            print(f"[stock-picker] Using cached price data (age={age_hours:.1f}h)")
            return pd.read_parquet(V3_CACHE)

    print("[stock-picker] Downloading fresh price data from yfinance...")
    all_tickers = list(set(UNIVERSE + ["SPY", "^VIX", "TLT", "HYG", "IEF"]))
    start_date = "2017-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")

    all_frames = []
    batch_size = 20
    for i in range(0, len(all_tickers), batch_size):
        batch = all_tickers[i:i + batch_size]
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
            print(f"  Batch error: {e}")
        time.sleep(0.5)

    prices = pd.concat(all_frames, ignore_index=True)
    col_map = {}
    for c in prices.columns:
        cl = c.lower()
        if cl in ('open', 'high', 'low', 'close', 'volume', 'adj close', 'date', 'ticker'):
            col_map[c] = cl.replace(' ', '_')
    prices = prices.rename(columns=col_map)
    prices['date'] = pd.to_datetime(prices['date'])
    prices = prices.dropna(subset=['close']).sort_values(['ticker', 'date']).reset_index(drop=True)

    V3_CACHE.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(V3_CACHE, index=False)
    print(f"[stock-picker] Cached {len(prices)} rows, {prices['ticker'].nunique()} tickers")
    return prices


# ---------------------------------------------------------------------------
# FEATURE ENGINEERING (identical to v3 stock_predictor_v3.py)
# ---------------------------------------------------------------------------
def compute_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Build v3 feature matrix (53 features) for all stocks."""
    spy = prices[prices['ticker'] == 'SPY'].copy().sort_values('date').set_index('date')
    spy_close = spy['close']
    spy_ret = spy_close.pct_change()

    vix = prices[prices['ticker'] == '^VIX'].copy().sort_values('date').set_index('date')
    vix_close = vix['close'] if len(vix) > 0 else pd.Series(dtype=float)

    macro = pd.DataFrame(index=spy.index)
    macro['spy_mom_21d'] = spy_close.pct_change(21)
    macro['spy_mom_63d'] = spy_close.pct_change(63)
    macro['spy_above_200ma'] = (spy_close > spy_close.rolling(200).mean()).astype(float)
    macro['spy_vol_21d'] = spy_ret.rolling(21).std() * np.sqrt(252)
    macro['spy_breadth'] = spy_ret.rolling(5).mean()
    if len(vix_close) > 0:
        vix_aligned = vix_close.reindex(spy.index, method='ffill')
        macro['vix_level'] = vix_aligned
        macro['vix_zscore'] = (vix_aligned - vix_aligned.rolling(63).mean()) / vix_aligned.rolling(63).std()
        macro['vix_change_5d'] = vix_aligned.pct_change(5)

    spy_mom = {lb: spy_close.pct_change(lb) for lb in [21, 63, 126, 252]}

    all_features = []
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

        for lb in [5, 10, 21, 63, 126, 252]:
            feat[f'mom_{lb}d'] = close.pct_change(lb)

        sma_50 = close.rolling(50).mean()
        sma_200 = close.rolling(200).mean()
        feat['above_50ma'] = (close > sma_50).astype(float)
        feat['above_200ma'] = (close > sma_200).astype(float)
        feat['ma_50_200_ratio'] = sma_50 / sma_200
        feat['price_vs_sma50'] = close / sma_50 - 1
        feat['price_vs_sma200'] = close / sma_200 - 1

        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        feat['rsi_14'] = 100 - (100 / (1 + rs))

        feat['vol_21d'] = ret.rolling(21).std() * np.sqrt(252)
        feat['vol_63d'] = ret.rolling(63).std() * np.sqrt(252)
        feat['vol_ratio'] = feat['vol_21d'] / feat['vol_63d']

        feat['vol_ratio_20d'] = vol / vol.rolling(20).mean()

        high_252 = close.rolling(252).max()
        feat['drawdown_from_high'] = close / high_252 - 1
        feat['at_52w_high'] = (close >= high_252 * 0.97).astype(float)

        feat['zscore_20d'] = (close - close.rolling(20).mean()) / close.rolling(20).std()
        feat['zscore_50d'] = (close - close.rolling(50).mean()) / close.rolling(50).std()

        for lb in [21, 63, 126, 252]:
            stock_mom = close.pct_change(lb)
            spy_m = spy_mom[lb].reindex(tdf.index)
            feat[f'rs_vs_spy_{lb}d'] = stock_mom - spy_m

        beta_63d = ret.rolling(63).cov(spy_ret.reindex(tdf.index)) / spy_ret.reindex(tdf.index).rolling(63).var()
        feat['beta_63d'] = beta_63d
        feat['idio_mom_63d'] = close.pct_change(63) - beta_63d * spy_close.pct_change(63).reindex(tdf.index)

        feat['mom_vol_interaction'] = feat['mom_63d'] / (feat['vol_63d'] + 0.001)
        feat['zscore_rs_interaction'] = feat['zscore_50d'] * feat['rs_vs_spy_63d']
        feat['beta_adj_drawdown'] = feat['drawdown_from_high'] / (beta_63d + 0.5)

        macro_aligned = macro.reindex(tdf.index)
        for col in macro.columns:
            feat[f'macro_{col}'] = macro_aligned[col]

        feat['rel_vol_vs_spy'] = feat['vol_21d'] / (macro_aligned['spy_vol_21d'] + 0.001)
        feat['excess_move_21d'] = feat['mom_21d'] - macro_aligned['spy_mom_21d']
        feat['excess_move_63d'] = feat['mom_63d'] - macro_aligned['spy_mom_63d']

        all_features.append(feat)

    master = pd.concat(all_features).reset_index().rename(columns={'index': 'date'})

    rank_cols = ['mom_21d', 'mom_63d', 'mom_126d', 'vol_21d', 'rsi_14',
                 'rs_vs_spy_63d', 'drawdown_from_high', 'vol_ratio_20d']
    for col in rank_cols:
        if col in master.columns:
            master[f'{col}_rank'] = master.groupby('date')[col].rank(pct=True)

    for lb in [21, 63, 126]:
        col = f'mom_{lb}d'
        if col in master.columns:
            master[f'{col}_sector_rank'] = master.groupby(['date', 'sector'])[col].rank(pct=True)

    sector_mom = master.groupby(['date', 'sector'])['mom_63d'].transform('mean')
    master['sector_mom_63d'] = sector_mom
    master['stock_vs_sector_mom'] = master['mom_63d'] - sector_mom

    return master


def add_target(master: pd.DataFrame, prices: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Add a single forward excess return target."""
    # target_col format: "target_excess_60d_3pct"
    parts = target_col.split('_')   # ['target', 'excess', '60d', '3pct']
    horizon = int(parts[2].replace('d', ''))
    thresh = int(parts[3].replace('pct', '')) / 100.0

    spy = prices[prices['ticker'] == 'SPY'].copy().sort_values('date').set_index('date')
    spy_fwd = spy['close'].pct_change(horizon).shift(-horizon)

    targets = []
    for ticker in master['ticker'].unique():
        tmask = master['ticker'] == ticker
        tdf = master.loc[tmask].copy().set_index('date')
        tprices = prices[prices['ticker'] == ticker].copy().sort_values('date').set_index('date')
        fwd_ret = tprices['close'].pct_change(horizon).shift(-horizon)
        spy_fwd_aligned = spy_fwd.reindex(tdf.index)
        excess = fwd_ret.reindex(tdf.index) - spy_fwd_aligned
        tdf[target_col] = (excess > thresh).astype(float)
        tdf[f'fwd_excess_{horizon}d'] = excess
        targets.append(tdf.reset_index())

    return pd.concat(targets, ignore_index=True)


def get_feature_cols(df: pd.DataFrame) -> list:
    exclude = {'date', 'ticker', 'sector'}
    exclude.update(c for c in df.columns if c.startswith('target_') or c.startswith('fwd_'))
    return [c for c in df.columns if c not in exclude and df[c].dtype in ('float64', 'float32', 'int64')]


# ---------------------------------------------------------------------------
# MODEL: TRAIN ON RECENT DATA + PREDICT LATEST ROW
# ---------------------------------------------------------------------------
def train_and_predict(master: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Train LGBM+XGBoost on the most recent TRAIN_WINDOW_DAYS of labeled data
    (excluding EMBARGO_DAYS of the most recent period to avoid label leakage),
    then predict on the most recent date's stocks.

    Returns DataFrame with ticker, sector, proba_lgbm, proba_xgb, proba_ensemble
    for the latest date.
    """
    feature_cols = get_feature_cols(master)
    valid = master.dropna(subset=[target_col] + feature_cols[:5])
    dates = sorted(valid['date'].unique())

    # Latest date is the prediction target
    latest_date = dates[-1]
    pred_date = latest_date

    # Training window: [train_start, train_end] with embargo
    train_end_date = latest_date - pd.Timedelta(days=EMBARGO_DAYS)
    train_start_date = train_end_date - pd.Timedelta(days=TRAIN_WINDOW_DAYS)

    train_mask = (valid['date'] >= train_start_date) & (valid['date'] <= train_end_date)
    pred_mask  = valid['date'] == pred_date

    X_train = valid.loc[train_mask, feature_cols].values
    y_train = valid.loc[train_mask, target_col].values
    X_pred  = valid.loc[pred_mask, feature_cols].values

    if len(X_train) < 200:
        raise ValueError(f"Not enough training rows: {len(X_train)}")
    if len(X_pred) == 0:
        raise ValueError(f"No stocks to predict for {pred_date}")

    X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
    X_pred  = np.nan_to_num(X_pred,  nan=0, posinf=0, neginf=0)

    base_rate = y_train.mean()
    print(f"[stock-picker] Training on {len(X_train)} rows, base_rate={base_rate:.3f}")
    print(f"[stock-picker] Predicting {len(X_pred)} stocks for {pred_date.date()}")

    lgbm_model = lgb.LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, min_child_samples=30,
        reg_alpha=0.1, reg_lambda=1.0, verbose=-1, n_jobs=-1, random_state=42,
        is_unbalance=True,
    )
    lgbm_model.fit(X_train, y_train)
    lgbm_proba = lgbm_model.predict_proba(X_pred)[:, 1]

    xgb_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=30,
        reg_alpha=0.1, reg_lambda=1.0, verbosity=0, n_jobs=-1, random_state=42,
        scale_pos_weight=(1 - base_rate) / max(base_rate, 0.01),
        eval_metric='logloss',
    )
    xgb_model.fit(X_train, y_train)
    xgb_proba = xgb_model.predict_proba(X_pred)[:, 1]

    ensemble_proba = (lgbm_proba + xgb_proba) / 2

    result_df = valid.loc[pred_mask, ['date', 'ticker', 'sector']].copy()
    result_df['proba_lgbm']     = lgbm_proba
    result_df['proba_xgb']      = xgb_proba
    result_df['proba_ensemble'] = ensemble_proba

    return result_df.sort_values('proba_ensemble', ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# PRICE FETCH (for mark-to-market and entry prices)
# ---------------------------------------------------------------------------
def get_current_prices(tickers: list) -> dict:
    """Get latest close prices for a list of tickers."""
    if not tickers:
        return {}
    try:
        df = yf.download(tickers, period='5d', progress=False, group_by='ticker', threads=True)
        prices = {}
        for ticker in tickers:
            try:
                if isinstance(df.columns, pd.MultiIndex):
                    px = df[ticker]['Close'].dropna().iloc[-1]
                else:
                    px = df['Close'].dropna().iloc[-1]
                prices[ticker] = float(px)
            except Exception:
                pass
        return prices
    except Exception as e:
        print(f"[stock-picker] Price fetch error: {e}")
        return {}


# ---------------------------------------------------------------------------
# WEEKLY REBALANCE (Monday 9:45 AM ET)
# ---------------------------------------------------------------------------
def run_weekly_rebalance(state: dict, prices_df: pd.DataFrame, today: Optional[date] = None) -> dict:
    """
    Core weekly logic:
    1. Retrain model on latest data
    2. Pick top K stocks with ensemble prob > CONFIDENCE_THRESH
    3. Add as new cohort with equal weight
    4. Roll off cohorts older than HOLD_DAYS
    """
    today = today or datetime.now().date()
    cohort_id = str(today)

    print(f"[stock-picker] === WEEKLY REBALANCE {today} ===")

    # Build features
    master = compute_features(prices_df)
    master = add_target(master, prices_df, TARGET_COL)

    # Train & predict
    predictions = train_and_predict(master, TARGET_COL)

    # Filter high-confidence
    high_conf = predictions[predictions['proba_ensemble'] >= CONFIDENCE_THRESH]
    picks = high_conf.head(TOP_K)[['ticker', 'sector', 'proba_ensemble']].to_dict('records')

    print(f"[stock-picker] High-confidence picks ({len(high_conf)} total, taking top {TOP_K}):")
    for p in picks:
        print(f"  {p['ticker']:6s} | {p['sector']:15s} | prob={p['proba_ensemble']:.3f}")

    if not picks:
        print(f"[stock-picker] WARNING: No picks above {CONFIDENCE_THRESH:.0%} threshold. Skipping entry.")
        state['last_rebal_date'] = str(today)
        return state

    # Get current prices for picks
    tickers = [p['ticker'] for p in picks]
    px_now = get_current_prices(tickers)
    missing = [t for t in tickers if t not in px_now]
    if missing:
        print(f"[stock-picker] No price for {missing}, removing from picks")
        picks = [p for p in picks if p['ticker'] in px_now]

    if not picks:
        print("[stock-picker] All picks had missing prices. Skipping.")
        state['last_rebal_date'] = str(today)
        return state

    # Allocation: equal-weight across this week's picks
    nav = state['nav_usd']
    alloc_per_pick = (nav * 0.80 / len(picks))  # use 80% of NAV, keep 20% cash buffer

    # Open positions for new cohort
    for p in picks:
        ticker = p['ticker']
        if ticker in state['positions']:
            print(f"  [skip] {ticker} already held (existing position)")
            continue
        px = px_now[ticker]
        shares = alloc_per_pick / px
        state['positions'][ticker] = {
            'shares': shares,
            'entry_px': px,
            'entry_date': str(today),
            'cohort_id': cohort_id,
            'alloc_usd': alloc_per_pick,
            'sector': p['sector'],
            'confidence': p['proba_ensemble'],
        }
        state['cash_usd'] -= alloc_per_pick
        _log_trade({
            'action': 'BUY',
            'date': str(today),
            'ticker': ticker,
            'shares': shares,
            'px': px,
            'alloc_usd': alloc_per_pick,
            'cohort_id': cohort_id,
            'confidence': p['proba_ensemble'],
        })
        print(f"  BUY {shares:.2f} sh {ticker} @ ${px:.2f} = ${alloc_per_pick:.0f}")

    # Roll off cohorts older than HOLD_DAYS
    state = _roll_off_old_cohorts(state, px_now, today)

    # Mark to market
    state = _mark_to_market(state, today)

    state['last_rebal_date'] = str(today)
    state['n_rebalances'] = state.get('n_rebalances', 0) + 1

    # Log picks
    _log_picks([{
        'ticker': p['ticker'],
        'sector': p['sector'],
        'confidence': p['proba_ensemble'],
        'entry_px': px_now.get(p['ticker']),
    } for p in picks], str(today), cohort_id)

    return state


def _roll_off_old_cohorts(state: dict, px_now: dict, today: date) -> dict:
    """Exit positions from cohorts older than HOLD_DAYS."""
    to_close = []
    for ticker, pos in state['positions'].items():
        entry = date.fromisoformat(pos['entry_date'])
        age_days = (today - entry).days
        if age_days >= HOLD_DAYS:
            to_close.append(ticker)
            print(f"  ROLL_OFF {ticker} (held {age_days} days)")

    for ticker in to_close:
        pos = state['positions'].pop(ticker)
        px = px_now.get(ticker) or pos['entry_px']  # fallback to entry if price unavailable
        proceeds = pos['shares'] * px
        cost     = pos['alloc_usd']
        pnl      = proceeds - cost
        state['cash_usd'] += proceeds
        state['total_realized_pnl'] += pnl
        closed = {**pos, 'exit_date': str(today), 'exit_px': px, 'pnl_usd': pnl}
        state['closed_positions'].append(closed)
        _log_trade({
            'action': 'SELL_ROLLOFF',
            'date': str(today),
            'ticker': ticker,
            'shares': pos['shares'],
            'exit_px': px,
            'entry_px': pos['entry_px'],
            'pnl_usd': pnl,
            'cohort_id': pos['cohort_id'],
        })

    return state


def _check_stop_losses(state: dict, px_now: dict, today: date) -> dict:
    """Exit any position that has dropped >STOP_LOSS_PCT from entry."""
    to_close = []
    for ticker, pos in state['positions'].items():
        px = px_now.get(ticker)
        if px is None:
            continue
        entry_px = pos['entry_px']
        drop = (px - entry_px) / entry_px
        if drop <= -STOP_LOSS_PCT:
            print(f"  STOP_LOSS {ticker}: drop={drop:.1%} (entry=${entry_px:.2f} now=${px:.2f})")
            to_close.append(ticker)

    for ticker in to_close:
        pos = state['positions'].pop(ticker)
        px = px_now[ticker]
        proceeds = pos['shares'] * px
        pnl = proceeds - pos['alloc_usd']
        state['cash_usd'] += proceeds
        state['total_realized_pnl'] += pnl
        closed = {**pos, 'exit_date': str(today), 'exit_px': px, 'pnl_usd': pnl, 'exit_reason': 'STOP_LOSS'}
        state['closed_positions'].append(closed)
        _log_trade({
            'action': 'SELL_STOPLOSS',
            'date': str(today),
            'ticker': ticker,
            'shares': pos['shares'],
            'exit_px': px,
            'entry_px': pos['entry_px'],
            'pnl_usd': pnl,
            'drop_pct': (px - pos['entry_px']) / pos['entry_px'],
        })
        notify(f"STOP-LOSS: {ticker} hit -10% from entry. Closed paper position.")

    return state


def _mark_to_market(state: dict, today: date, px_now: Optional[dict] = None) -> dict:
    """Update NAV from market prices."""
    if not state['positions']:
        state['nav_usd'] = state['cash_usd']
        return state

    if px_now is None:
        tickers = list(state['positions'].keys())
        px_now = get_current_prices(tickers)

    position_value = 0.0
    for ticker, pos in state['positions'].items():
        px = px_now.get(ticker, pos['entry_px'])  # fallback to entry
        position_value += pos['shares'] * px

    state['nav_usd'] = state['cash_usd'] + position_value
    return state


# ---------------------------------------------------------------------------
# DAILY MTM (non-Monday days)
# ---------------------------------------------------------------------------
def run_daily_mtm(state: dict, today: Optional[date] = None) -> dict:
    """Daily mark-to-market + stop-loss check."""
    today = today or datetime.now().date()
    print(f"[stock-picker] === DAILY MTM {today} ===")

    if not state['positions']:
        print("[stock-picker] No positions. Nothing to mark.")
        state['last_mtm_date'] = str(today)
        _log_equity(state['nav_usd'], str(today), "no_positions")
        return state

    tickers = list(state['positions'].keys())
    px_now = get_current_prices(tickers)

    # Stop-loss check first
    state = _check_stop_losses(state, px_now, today)

    # Mark to market
    state = _mark_to_market(state, today, px_now)

    state['last_mtm_date'] = str(today)

    # Print position summary
    start_nav = STARTING_NAV
    pnl_pct = (state['nav_usd'] - start_nav) / start_nav
    print(f"[stock-picker] NAV=${state['nav_usd']:,.2f} (vs start ${start_nav:,.0f}, {pnl_pct:+.1%})")
    print(f"[stock-picker] Cash=${state['cash_usd']:,.2f}, Positions={len(state['positions'])}")

    for ticker, pos in state['positions'].items():
        px = px_now.get(ticker, pos['entry_px'])
        chg = (px - pos['entry_px']) / pos['entry_px']
        age = (today - date.fromisoformat(pos['entry_date'])).days
        print(f"  {ticker:6s} {chg:+.1%} age={age}d px=${px:.2f} entry=${pos['entry_px']:.2f}")

    _log_equity(state['nav_usd'], str(today))

    return state


# ---------------------------------------------------------------------------
# STATUS REPORT
# ---------------------------------------------------------------------------
def build_status_report(state: dict, prices_df: Optional[pd.DataFrame] = None) -> str:
    """Build a plain-English status report for Discord."""
    today = datetime.now().date()
    nav = state['nav_usd']
    start_nav = STARTING_NAV
    pnl_pct = (nav - start_nav) / start_nav
    realized = state.get('total_realized_pnl', 0)

    positions = state.get('positions', {})
    n_pos = len(positions)

    lines = [
        f"Stock Picker Paper — {today}",
        f"NAV: ${nav:,.0f} ({pnl_pct:+.1%} vs ${start_nav:,.0f} start)",
        f"Realized P&L: ${realized:+,.0f} | Rebalances: {state.get('n_rebalances', 0)}",
        f"Open positions: {n_pos}",
    ]

    if positions:
        # Get current prices
        tickers = list(positions.keys())
        px_map = get_current_prices(tickers)

        lines.append("")
        lines.append("Current picks:")
        for ticker, pos in positions.items():
            px = px_map.get(ticker, pos['entry_px'])
            chg = (px - pos['entry_px']) / pos['entry_px']
            age = (today - date.fromisoformat(pos['entry_date'])).days
            days_left = max(0, HOLD_DAYS - age)
            conf = pos.get('confidence', 0)
            lines.append(
                f"  {ticker} {chg:+.1%} ({days_left}d left, {conf:.0%} conf)"
            )

    if state.get('closed_positions'):
        n_closed = len(state['closed_positions'])
        winning = sum(1 for p in state['closed_positions'] if p.get('pnl_usd', 0) > 0)
        lines.append(f"Closed: {n_closed} trades, {winning}/{n_closed} profitable")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    today = datetime.now().date()
    is_monday = today.weekday() == 0

    print(f"[stock-picker] Stock Picker Paper Engine — {today} ({'Monday REBALANCE' if is_monday else 'Daily MTM'})")

    # Load state
    state = _load_state()

    # Check idempotency: skip if already ran today
    last_run = state.get('last_mtm_date') or state.get('last_rebal_date')
    if last_run == str(today):
        print(f"[stock-picker] Already ran today ({today}). Printing status only.")
        report = build_status_report(state)
        print(report)
        return

    try:
        if is_monday:
            # Load/refresh price data
            prices_df = load_price_data()
            state = run_weekly_rebalance(state, prices_df, today)
            _save_state(state)

            # Build report and notify
            report = build_status_report(state)
            print(report)
            notify(report)

        else:
            # Daily MTM only
            state = run_daily_mtm(state, today)
            _save_state(state)

            # Only notify on Fridays (weekly wrap) or if stop-loss fired
            if today.weekday() == 4:  # Friday
                report = build_status_report(state)
                print(report)
                notify(report)

    except Exception as e:
        err_msg = f"Stock picker paper engine error: {e}"
        print(f"[stock-picker] ERROR: {err_msg}")
        traceback.print_exc()
        notify(err_msg)
        sys.exit(1)


def run_dry_test():
    """
    Run an immediate dry test: load prices, retrain, generate picks.
    Does NOT modify state. Used for --dry-run mode.
    """
    print("[stock-picker] === DRY RUN — no state changes ===")
    today = datetime.now().date()

    prices_df = load_price_data()
    print(f"[stock-picker] Price data: {prices_df['ticker'].nunique()} tickers through {prices_df['date'].max().date()}")

    master = compute_features(prices_df)
    master = add_target(master, prices_df, TARGET_COL)
    print(f"[stock-picker] Feature matrix: {master.shape}, feature cols: {len(get_feature_cols(master))}")

    predictions = train_and_predict(master, TARGET_COL)

    print(f"\n[stock-picker] === THIS WEEK'S PICKS (as of {today}) ===")
    print(f"Target: {TARGET_COL} | Threshold: >{CONFIDENCE_THRESH:.0%}")
    print()

    high_conf = predictions[predictions['proba_ensemble'] >= CONFIDENCE_THRESH]
    top_picks = high_conf.head(TOP_K)

    if len(top_picks) == 0:
        print(f"  No picks above {CONFIDENCE_THRESH:.0%} confidence today.")
    else:
        tickers = top_picks['ticker'].tolist()
        px_now = get_current_prices(tickers)
        for _, row in top_picks.iterrows():
            px = px_now.get(row['ticker'], 0)
            print(f"  #{_+1}  {row['ticker']:6s} | {row['sector']:15s} | "
                  f"prob={row['proba_ensemble']:.3f} | px=${px:.2f}")

    print(f"\nAll stocks with >50% confidence:")
    med_conf = predictions[predictions['proba_ensemble'] >= 0.50].head(20)
    for _, row in med_conf.iterrows():
        print(f"  {row['ticker']:6s} {row['proba_ensemble']:.3f}  ({row['sector']})")

    return top_picks


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--dry-run':
        run_dry_test()
    elif len(sys.argv) > 1 and sys.argv[1] == '--force-rebalance':
        # Force a full rebalance even on non-Monday
        print("[stock-picker] FORCE REBALANCE mode")
        state = _load_state()
        prices_df = load_price_data()
        state = run_weekly_rebalance(state, prices_df)
        _save_state(state)
        report = build_status_report(state)
        print(report)
        notify(report)
    else:
        main()
