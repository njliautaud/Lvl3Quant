#!/usr/bin/env python3
"""
ML Asymmetric Stock Ranker — Paper Trading Engine
==================================================
Monthly rebalance paper engine based on the validated ML stock ranker
(Neptune run: Strategy B, 20.3% ann, perm p=0.000).

Strategy B: LightGBM ranks stocks by predicted 1-month forward return,
but ONLY trades when the asymmetric filter fires:
  - vol_20d_pctrank >= 0.80 (high short-term vol vs peers)
  - mom_1m < 0 (negative recent momentum — beaten down)
  - vol_surge > 1.5 (volume surge — smart money entering)

When no stocks pass the filter → stay 100% cash. This happens ~74% of months.
The edge is concentrated in distressed conditions (bear IC=0.069 > bull IC=0.050).

Rebalance: 1st trading day of each month, 9:45 AM ET
Starting NAV: $10,000
State: /home/jupiter/Lvl3Quant/data/paper_engines/ml_asymmetric_ranker/

HC #725/#726: Signal-first asymmetric upside research
HC #724: Anti-lookahead (all features T-1)
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path("/home/jupiter/Lvl3Quant")
STATE_DIR = ROOT / "data" / "paper_engines" / "ml_asymmetric_ranker"
STATE_DIR.mkdir(parents=True, exist_ok=True)

STATE_FILE   = STATE_DIR / "state.json"
TRADES_LOG   = STATE_DIR / "trades.jsonl"
EQUITY_LOG   = STATE_DIR / "equity_curve.jsonl"
PICKS_LOG    = STATE_DIR / "monthly_picks.jsonl"
MODEL_DIR    = STATE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

WEBHOOK = "/home/jupiter/teleclaude-main/utils/webhook_notifier.js"
LOG_FILE = ROOT / "logs" / "ml_asymmetric_ranker_paper.log"

# ── Config ───────────────────────────────────────────────────────────────────
STARTING_NAV       = 10_000.0
TOP_K              = 5          # picks per rebalance
TRAIN_DAYS         = 504        # ~2yr sliding train window
COST_BPS           = 25         # per leg (Robinhood = 0, but keep conservative)
STOP_LOSS_PCT      = 0.15       # 15% stop-loss per position

# Asymmetric filter thresholds (from backtest validation)
FILTER_VOL_PCTRANK = 0.80
FILTER_MOM_1M      = 0.0        # must be negative
FILTER_VOL_SURGE   = 1.5

# ── Universe (same 50 stocks as backtest) ────────────────────────────────────
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "GS", "BAC",
    "V", "MA", "UNH", "JNJ", "PG", "KO", "PEP", "MRK", "ABBV", "LLY",
    "HD", "COST", "WMT", "CRM", "AMD", "NFLX", "ADBE", "INTC", "CSCO", "QCOM",
    "XOM", "CVX", "PFE", "TMO", "ABT", "AVGO", "TXN", "MCD", "NKE", "DIS",
    "CMCSA", "T", "VZ", "NEE", "SO", "SHW", "LMT", "RTX", "CAT", "DE",
]

SECTOR_ETFS = ["XLK", "XLF", "XLV", "XLP", "XLE", "XLI", "XLY", "XLU", "XLC", "XLRE", "XLB"]

STOCK_SECTOR = {
    "AAPL": "XLK", "MSFT": "XLK", "GOOGL": "XLC", "AMZN": "XLY", "META": "XLC",
    "NVDA": "XLK", "TSLA": "XLY", "JPM": "XLF", "GS": "XLF", "BAC": "XLF",
    "V": "XLK", "MA": "XLK", "UNH": "XLV", "JNJ": "XLV", "PG": "XLP",
    "KO": "XLP", "PEP": "XLP", "MRK": "XLV", "ABBV": "XLV", "LLY": "XLV",
    "HD": "XLY", "COST": "XLP", "WMT": "XLP", "CRM": "XLK", "AMD": "XLK",
    "NFLX": "XLC", "ADBE": "XLK", "INTC": "XLK", "CSCO": "XLK", "QCOM": "XLK",
    "XOM": "XLE", "CVX": "XLE", "PFE": "XLV", "TMO": "XLV", "ABT": "XLV",
    "AVGO": "XLK", "TXN": "XLK", "MCD": "XLY", "NKE": "XLY", "DIS": "XLC",
    "CMCSA": "XLC", "T": "XLC", "VZ": "XLC", "NEE": "XLU", "SO": "XLU",
    "SHW": "XLB", "LMT": "XLI", "RTX": "XLI", "CAT": "XLI", "DE": "XLI",
}

FEATURE_COLS = [
    "rsi_14", "dist_52w_high", "vol_20d", "vol_63d", "vol_surge",
    "mom_1m", "mom_3m", "mom_6m", "mr_zscore", "price_vs_sma200",
    "max_dd_52w", "vol_ratio", "mom_accel",
    "vix", "vix_term_structure", "breadth", "sector_ret_1m", "cs_rank_1m",
    "vol_20d_pctrank", "vol_63d_pctrank",
]

# ── Logging ──────────────────────────────────────────────────────────────────
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ── Data ─────────────────────────────────────────────────────────────────────
def download_data() -> pd.DataFrame:
    """Download price data for all tickers."""
    import yfinance as yf

    cache = STATE_DIR / "price_cache.parquet"

    # Check cache freshness
    if cache.exists():
        df = pd.read_parquet(cache)
        last_date = df.index.max()
        if last_date >= pd.Timestamp.now() - pd.Timedelta(days=3):
            log.info(f"Using cached data through {last_date.strftime('%Y-%m-%d')}")
            return df

    all_tickers = UNIVERSE + SECTOR_ETFS + ["^VIX", "^VIX3M", "SPY"]
    log.info(f"Downloading {len(all_tickers)} tickers...")

    data = yf.download(
        all_tickers,
        start="2022-01-01",  # 2+ years for 504d training
        end=datetime.now().strftime("%Y-%m-%d"),
        auto_adjust=True,
        threads=True,
    )

    close = data["Close"] if "Close" in data.columns.get_level_values(0) else data["Adj Close"]
    volume = data["Volume"]
    combined = pd.concat({"Close": close, "Volume": volume}, axis=1)
    combined.to_parquet(cache)
    log.info(f"Downloaded: {combined.shape}, {combined.index[0]} to {combined.index[-1]}")
    return combined


# ── Features ─────────────────────────────────────────────────────────────────
def compute_features_single(data: pd.DataFrame, date_idx, ticker: str) -> Optional[dict]:
    """Compute features for one stock at one date using T-1 data."""
    close_col = ("Close", ticker)
    vol_col = ("Volume", ticker)

    if close_col not in data.columns or vol_col not in data.columns:
        return None

    loc = data.index.get_loc(date_idx)
    if loc < 252:
        return None

    prices = data[close_col].iloc[:loc].values.astype(float)
    volumes = data[vol_col].iloc[:loc].values.astype(float)

    if len(prices) < 252 or np.isnan(prices[-1]):
        return None

    p = prices[-1]
    feats = {}

    # RSI(14)
    deltas = np.diff(prices[-15:])
    gains = np.maximum(deltas, 0).mean()
    losses = np.abs(np.minimum(deltas, 0)).mean()
    feats["rsi_14"] = 100 - 100 / (1 + gains / (losses + 1e-10))

    # Distance from 52-week high
    high_252 = np.nanmax(prices[-252:])
    feats["dist_52w_high"] = (p / high_252 - 1) * 100

    # Realized vol
    rets = np.diff(np.log(prices[-64:] + 1e-10))
    feats["vol_20d"] = np.std(rets[-20:]) * np.sqrt(252) if len(rets) >= 20 else np.nan
    feats["vol_63d"] = np.std(rets[-63:]) * np.sqrt(252) if len(rets) >= 63 else np.nan

    # Volume surge
    vol_5d = np.nanmean(volumes[-5:])
    vol_63d_avg = np.nanmean(volumes[-63:])
    feats["vol_surge"] = vol_5d / (vol_63d_avg + 1e-10)

    # Momentum
    feats["mom_1m"] = p / prices[-21] - 1 if len(prices) >= 21 else np.nan
    feats["mom_3m"] = p / prices[-63] - 1 if len(prices) >= 63 else np.nan
    feats["mom_6m"] = p / prices[-126] - 1 if len(prices) >= 126 else np.nan

    # Mean reversion z-score
    mean_63 = np.mean(prices[-63:])
    std_63 = np.std(prices[-63:])
    feats["mr_zscore"] = (p - mean_63) / (std_63 + 1e-10)

    # Price vs 200d SMA
    if len(prices) >= 200:
        sma_200 = np.mean(prices[-200:])
        feats["price_vs_sma200"] = p / sma_200 - 1
    else:
        feats["price_vs_sma200"] = np.nan

    # Max drawdown 52w
    rolling_max = np.maximum.accumulate(prices[-252:])
    drawdowns = prices[-252:] / rolling_max - 1
    feats["max_dd_52w"] = np.min(drawdowns)

    # Vol ratio
    feats["vol_ratio"] = feats["vol_20d"] / feats["vol_63d"] if feats["vol_63d"] > 0 else np.nan

    # Momentum acceleration
    if not np.isnan(feats.get("mom_1m", np.nan)) and not np.isnan(feats.get("mom_3m", np.nan)):
        feats["mom_accel"] = feats["mom_1m"] - feats["mom_3m"] / 3
    else:
        feats["mom_accel"] = np.nan

    return feats


def build_current_features(data: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix for the most recent date (for live prediction)."""
    latest_date = data.index[-1]
    log.info(f"Building features for {latest_date.strftime('%Y-%m-%d')} (T-1 data)")

    loc = data.index.get_loc(latest_date)

    # Market features
    vix_col = ("Close", "^VIX") if ("Close", "^VIX") in data.columns else None
    vix3m_col = ("Close", "^VIX3M") if ("Close", "^VIX3M") in data.columns else None
    vix_val = float(data[vix_col].iloc[loc]) if vix_col else np.nan
    vix3m_val = float(data[vix3m_col].iloc[loc]) if vix3m_col else np.nan
    vix_term = vix_val / (vix3m_val + 1e-10) if not np.isnan(vix_val) and not np.isnan(vix3m_val) else np.nan

    # Breadth
    breadth = 0.0
    n_etfs = 0
    for etf in SECTOR_ETFS:
        col = ("Close", etf)
        if col in data.columns:
            sma50 = data[col].iloc[max(0, loc-50):loc+1].mean()
            if data[col].iloc[loc] > sma50:
                breadth += 1
            n_etfs += 1
    breadth = breadth / max(n_etfs, 1)

    # Sector returns
    sector_rets = {}
    for etf in SECTOR_ETFS:
        col = ("Close", etf)
        if col in data.columns and loc >= 21:
            p_now = data[col].iloc[loc]
            p_prev = data[col].iloc[loc - 21]
            if not np.isnan(p_now) and not np.isnan(p_prev) and p_prev > 0:
                sector_rets[etf] = p_now / p_prev - 1

    # Cross-sectional momentum ranks
    cs_rets = {}
    for ticker in UNIVERSE:
        col = ("Close", ticker)
        if col in data.columns and loc >= 21:
            p_now = data[col].iloc[loc]
            p_prev = data[col].iloc[loc - 21]
            if not np.isnan(p_now) and not np.isnan(p_prev) and p_prev > 0:
                cs_rets[ticker] = p_now / p_prev - 1

    sorted_tickers = sorted(cs_rets.keys(), key=lambda t: cs_rets[t])
    cs_rank = {t: rank / len(sorted_tickers) for rank, t in enumerate(sorted_tickers)}

    rows = []
    for ticker in UNIVERSE:
        feats = compute_features_single(data, latest_date, ticker)
        if feats is None:
            continue

        feats["vix"] = vix_val
        feats["vix_term_structure"] = vix_term
        feats["breadth"] = breadth
        feats["sector_ret_1m"] = sector_rets.get(STOCK_SECTOR.get(ticker, "XLK"), np.nan)
        feats["cs_rank_1m"] = cs_rank.get(ticker, np.nan)
        feats["ticker"] = ticker
        feats["date"] = latest_date
        rows.append(feats)

    df = pd.DataFrame(rows)

    # Cross-sectional vol pctranks
    for col in ["vol_20d", "vol_63d"]:
        if col in df.columns:
            df[f"{col}_pctrank"] = df[col].rank(pct=True)

    # Asymmetric filter
    df["asymmetric_filter"] = (
        (df["vol_20d_pctrank"] >= FILTER_VOL_PCTRANK) &
        (df["mom_1m"] < FILTER_MOM_1M) &
        (df["vol_surge"] > FILTER_VOL_SURGE)
    ).astype(int)

    return df


def build_training_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """Build walk-forward training matrix from historical data."""
    dates = data.index
    start_idx = max(504, 252)
    rebal_dates = dates[start_idx::21]  # Monthly

    log.info(f"Building training matrix: {len(rebal_dates)} dates")

    rows = []
    for dt in rebal_dates:
        loc = data.index.get_loc(dt)
        if loc + 21 >= len(dates):
            continue

        # Market features
        vix_col = ("Close", "^VIX") if ("Close", "^VIX") in data.columns else None
        vix3m_col = ("Close", "^VIX3M") if ("Close", "^VIX3M") in data.columns else None
        vix_val = float(data[vix_col].iloc[loc]) if vix_col else np.nan
        vix3m_val = float(data[vix3m_col].iloc[loc]) if vix3m_col else np.nan
        vix_term = vix_val / (vix3m_val + 1e-10) if not np.isnan(vix_val) and not np.isnan(vix3m_val) else np.nan

        # Breadth
        breadth = 0.0
        n_etfs = 0
        for etf in SECTOR_ETFS:
            col = ("Close", etf)
            if col in data.columns:
                sma50 = data[col].iloc[max(0, loc-50):loc+1].mean()
                if data[col].iloc[loc] > sma50:
                    breadth += 1
                n_etfs += 1
        breadth = breadth / max(n_etfs, 1)

        # Sector returns
        sector_rets = {}
        for etf in SECTOR_ETFS:
            col = ("Close", etf)
            if col in data.columns and loc >= 21:
                p_now = data[col].iloc[loc]
                p_prev = data[col].iloc[loc - 21]
                if not np.isnan(p_now) and not np.isnan(p_prev) and p_prev > 0:
                    sector_rets[etf] = p_now / p_prev - 1

        # CS ranks
        cs_rets = {}
        for ticker in UNIVERSE:
            col = ("Close", ticker)
            if col in data.columns and loc >= 21:
                p_now = data[col].iloc[loc]
                p_prev = data[col].iloc[loc - 21]
                if not np.isnan(p_now) and not np.isnan(p_prev) and p_prev > 0:
                    cs_rets[ticker] = p_now / p_prev - 1

        sorted_tickers = sorted(cs_rets.keys(), key=lambda t: cs_rets[t])
        cs_rank = {t: rank / len(sorted_tickers) for rank, t in enumerate(sorted_tickers)}

        for ticker in UNIVERSE:
            feats = compute_features_single(data, dt, ticker)
            if feats is None:
                continue

            feats["vix"] = vix_val
            feats["vix_term_structure"] = vix_term
            feats["breadth"] = breadth
            feats["sector_ret_1m"] = sector_rets.get(STOCK_SECTOR.get(ticker, "XLK"), np.nan)
            feats["cs_rank_1m"] = cs_rank.get(ticker, np.nan)
            feats["ticker"] = ticker
            feats["date"] = dt

            # Forward return (target)
            fwd_col = ("Close", ticker)
            p_now = data[fwd_col].iloc[loc]
            p_fwd = data[fwd_col].iloc[loc + 21]
            feats["fwd_ret_1m"] = p_fwd / p_now - 1 if not np.isnan(p_now) and not np.isnan(p_fwd) and p_now > 0 else np.nan

            rows.append(feats)

    df = pd.DataFrame(rows)

    # Cross-sectional vol pctranks per date
    for col in ["vol_20d", "vol_63d"]:
        if col in df.columns:
            df[f"{col}_pctrank"] = df.groupby("date")[col].rank(pct=True)

    # Asymmetric filter
    df["asymmetric_filter"] = (
        (df["vol_20d_pctrank"] >= FILTER_VOL_PCTRANK) &
        (df["mom_1m"] < FILTER_MOM_1M) &
        (df["vol_surge"] > FILTER_VOL_SURGE)
    ).astype(int)

    return df


# ── Model ────────────────────────────────────────────────────────────────────
def train_model(train_df: pd.DataFrame):
    """Train LightGBM on the training matrix (CPU mode)."""
    import lightgbm as lgb

    params = {
        "objective": "regression",
        "metric": "mse",
        "num_leaves": 63,
        "learning_rate": 0.03,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
    }

    # Use last 504d worth of rows for training
    unique_dates = sorted(train_df["date"].unique())
    n_train_periods = min(TRAIN_DAYS // 21, len(unique_dates) - 1)
    train_dates = unique_dates[-n_train_periods:]

    subset = train_df[train_df["date"].isin(train_dates)].copy()
    subset = subset.dropna(subset=["fwd_ret_1m"] + FEATURE_COLS)

    if len(subset) < 100:
        log.warning(f"Insufficient training data: {len(subset)} rows")
        return None

    X = subset[FEATURE_COLS].values.astype(np.float32)
    y = subset["fwd_ret_1m"].values.astype(np.float32)

    # Remove NaN rows
    valid = ~np.isnan(y) & ~np.any(np.isnan(X), axis=1)
    X, y = X[valid], y[valid]

    log.info(f"Training LightGBM: {len(X)} rows, {len(FEATURE_COLS)} features, "
             f"date range: {train_dates[0].strftime('%Y-%m-%d')} to {train_dates[-1].strftime('%Y-%m-%d')}")

    dtrain = lgb.Dataset(X, label=y)
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=500,
    )

    # Save model
    model_path = MODEL_DIR / f"lgbm_{datetime.now().strftime('%Y%m%d')}.txt"
    model.save_model(str(model_path))
    log.info(f"Model saved: {model_path.name}")

    return model


# ── State Management ─────────────────────────────────────────────────────────
def load_state() -> dict:
    """Load paper engine state."""
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "nav": STARTING_NAV,
        "cash": STARTING_NAV,
        "positions": {},  # {ticker: {shares, entry_price, entry_date}}
        "last_rebalance": None,
        "trades": [],
        "created": datetime.now().isoformat(),
    }


def save_state(state: dict):
    """Save paper engine state."""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def log_equity(state: dict, prices: dict):
    """Log equity curve point."""
    nav = state["cash"]
    for ticker, pos in state["positions"].items():
        price = prices.get(ticker, pos["entry_price"])
        nav += pos["shares"] * price

    entry = {
        "timestamp": datetime.now().isoformat(),
        "nav": nav,
        "cash": state["cash"],
        "n_positions": len(state["positions"]),
        "positions": {t: p["shares"] * prices.get(t, p["entry_price"])
                      for t, p in state["positions"].items()},
    }
    with open(EQUITY_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")

    return nav


def log_trade(action: str, ticker: str, shares: float, price: float, reason: str):
    """Log a trade."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "action": action,
        "ticker": ticker,
        "shares": shares,
        "price": price,
        "reason": reason,
    }
    with open(TRADES_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


def log_picks(picks: list, filter_stats: dict):
    """Log monthly picks."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "picks": picks,
        "filter_stats": filter_stats,
    }
    with open(PICKS_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ── Portfolio Operations ─────────────────────────────────────────────────────
def get_current_prices(tickers: list) -> dict:
    """Get current market prices via yfinance."""
    import yfinance as yf
    prices = {}
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="2d")
            if len(hist) > 0:
                prices[ticker] = float(hist["Close"].iloc[-1])
        except Exception:
            pass
    return prices


def should_rebalance(state: dict) -> bool:
    """Check if we should rebalance today (1st trading day of month)."""
    now = datetime.now()
    today = now.date()

    # Skip weekends
    if today.weekday() >= 5:
        return False

    # Check if 1st trading day of month
    if today.day > 5:
        return False

    # Check if already rebalanced this month
    last_rebal = state.get("last_rebalance")
    if last_rebal:
        last_dt = datetime.fromisoformat(last_rebal).date()
        if last_dt.year == today.year and last_dt.month == today.month:
            return False

    return True


def rebalance(state: dict, model, current_features: pd.DataFrame, prices: dict):
    """Execute monthly rebalance."""
    log.info("=" * 60)
    log.info("MONTHLY REBALANCE")
    log.info("=" * 60)

    # Predict returns for all stocks
    valid_features = current_features.dropna(subset=FEATURE_COLS)
    if len(valid_features) == 0:
        log.warning("No valid features — staying in cash")
        return

    X = valid_features[FEATURE_COLS].values.astype(np.float32)
    preds = model.predict(X)
    valid_features = valid_features.copy()
    valid_features["prediction"] = preds

    # Apply asymmetric filter
    filtered = valid_features[valid_features["asymmetric_filter"] == 1]

    filter_stats = {
        "total_stocks": len(valid_features),
        "passing_filter": len(filtered),
        "filter_pct": len(filtered) / len(valid_features) * 100,
        "vix": float(valid_features["vix"].iloc[0]) if "vix" in valid_features.columns else None,
    }

    log.info(f"Filter: {filter_stats['passing_filter']}/{filter_stats['total_stocks']} stocks "
             f"pass asymmetric filter ({filter_stats['filter_pct']:.0f}%)")

    # Close all existing positions first
    for ticker, pos in list(state["positions"].items()):
        price = prices.get(ticker, pos["entry_price"])
        proceeds = pos["shares"] * price
        state["cash"] += proceeds
        ret_pct = (price / pos["entry_price"] - 1) * 100
        log.info(f"  SELL {ticker}: {pos['shares']:.1f} shares @ ${price:.2f} "
                 f"(return: {ret_pct:+.1f}%)")
        log_trade("SELL", ticker, pos["shares"], price, "monthly_rebalance")

    state["positions"] = {}

    if len(filtered) == 0:
        log.info("No stocks pass filter → 100% CASH this month")
        log_picks([], filter_stats)
        state["last_rebalance"] = datetime.now().isoformat()
        save_state(state)
        return

    # Select top-K by prediction
    picks = filtered.nlargest(min(TOP_K, len(filtered)), "prediction")
    pick_tickers = picks["ticker"].tolist()

    log.info(f"Selected {len(pick_tickers)} stocks: {pick_tickers}")
    for _, row in picks.iterrows():
        log.info(f"  {row['ticker']}: pred={row['prediction']:.4f}, "
                 f"mom_1m={row['mom_1m']:.3f}, vol_surge={row['vol_surge']:.2f}, "
                 f"vol_pctrank={row['vol_20d_pctrank']:.2f}")

    # Buy equal weight
    position_value = state["cash"] / len(pick_tickers)
    cost_per_trade = position_value * COST_BPS / 10000

    for ticker in pick_tickers:
        price = prices.get(ticker)
        if price is None or price <= 0:
            log.warning(f"No price for {ticker}, skipping")
            continue

        shares = (position_value - cost_per_trade) / price
        state["positions"][ticker] = {
            "shares": shares,
            "entry_price": price,
            "entry_date": datetime.now().isoformat(),
        }
        state["cash"] -= position_value
        log.info(f"  BUY {ticker}: {shares:.1f} shares @ ${price:.2f} "
                 f"(${position_value:.0f} position)")
        log_trade("BUY", ticker, shares, price, "monthly_rebalance_filtered")

    log_picks(pick_tickers, filter_stats)
    state["last_rebalance"] = datetime.now().isoformat()
    save_state(state)

    nav = log_equity(state, prices)
    log.info(f"Post-rebalance NAV: ${nav:,.2f} | Cash: ${state['cash']:,.2f} | "
             f"Positions: {len(state['positions'])}")


def check_stop_losses(state: dict, prices: dict):
    """Check stop-losses on current positions."""
    for ticker, pos in list(state["positions"].items()):
        price = prices.get(ticker, pos["entry_price"])
        ret = price / pos["entry_price"] - 1

        if ret < -STOP_LOSS_PCT:
            proceeds = pos["shares"] * price
            state["cash"] += proceeds
            log.info(f"STOP-LOSS {ticker}: {ret*100:.1f}% loss (threshold: {-STOP_LOSS_PCT*100:.0f}%)")
            log_trade("SELL", ticker, pos["shares"], price, f"stop_loss_{ret*100:.1f}pct")
            del state["positions"][ticker]
            save_state(state)


# ── Main ─────────────────────────────────────────────────────────────────────
def run():
    """Main paper engine loop."""
    log.info("=" * 60)
    log.info("ML ASYMMETRIC STOCK RANKER — PAPER ENGINE")
    log.info("=" * 60)

    state = load_state()
    log.info(f"State: NAV=${state['nav']:,.2f}, Cash=${state['cash']:,.2f}, "
             f"Positions={len(state['positions'])}, "
             f"Last rebalance={state.get('last_rebalance', 'never')}")

    # Download data
    data = download_data()

    # Get current prices for held positions
    all_tickers = list(state["positions"].keys()) + UNIVERSE[:10]  # Always price top 10
    prices = get_current_prices(list(set(all_tickers)))
    log.info(f"Got prices for {len(prices)} tickers")

    # Check stop-losses
    if state["positions"]:
        check_stop_losses(state, prices)

    # Log daily equity
    nav = log_equity(state, prices)
    log.info(f"Current NAV: ${nav:,.2f}")

    # Check if rebalance needed
    if should_rebalance(state):
        log.info("Rebalance day — training model and picking stocks...")

        # Build training matrix
        train_df = build_training_matrix(data)
        log.info(f"Training matrix: {len(train_df)} rows")

        # Train model
        model = train_model(train_df)
        if model is None:
            log.error("Model training failed!")
            return

        # Build current features
        current_features = build_current_features(data)
        log.info(f"Current features: {len(current_features)} stocks")

        # Get prices for all universe stocks
        all_prices = get_current_prices(UNIVERSE)
        prices.update(all_prices)

        # Rebalance
        rebalance(state, model, current_features, prices)
    else:
        log.info("Not a rebalance day — daily MTM only")
        state["nav"] = nav
        save_state(state)

    log.info("Done.")


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        log.error(f"FATAL: {e}", exc_info=True)
        sys.exit(1)
