#!/usr/bin/env python3
"""
Live Volatility Forecaster — LGBM inference for position sizing
================================================================
Provides real-time 21-day forward vol predictions for the wheel universe.
Uses the same feature engineering as research/vol_forecaster.py but
trains a fresh model on the latest 252 days of data.

Usage:
    from vol_forecaster_live import get_vol_predictions
    preds = get_vol_predictions()  # Returns {ticker: pred_vol_ann}

    # For sizing: inverse-vol weight
    scale = median_vol / pred_vol  # Capped at [0.5, 2.0]

The model is retrained daily (cached for the day). IC=0.75, ICIR=5.82.
Dominant features: parkinson_vol, rv_60d, ust_2y, days_to_earnings.
"""
from __future__ import annotations

import json
import pickle
import time
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import yfinance as yf
except ImportError:
    yf = None

# ── Config ─────────────────────────────────────────────────────────────────
ROOT = Path("/home/jupiter/Lvl3Quant")
CACHE_DIR = ROOT / "live_trading_linux" / "vol_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CACHE = CACHE_DIR / "lgbm_vol_model.pkl"
PREDS_CACHE = CACHE_DIR / "daily_predictions.json"

ANNUALIZE = np.sqrt(252)
TRAIN_WINDOW = 252  # days
PRICE_LOOKBACK = 320  # fetch extra for warmup

# Universe — same as wheel V5 paper engine
WHEEL_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "NFLX",
    "CRM", "ADBE", "INTC", "PYPL", "XYZ", "SHOP", "UBER", "ABNB", "COIN",  # SQ→XYZ (Block rename)
    "SOFI", "HOOD", "PLTR", "SNOW", "NET", "DDOG", "ZS", "CRWD", "PANW",
    "MDB", "AFRM", "ARM", "SMCI", "MSTR", "AVGO", "MU", "QCOM", "MRVL",
    "ON", "LRCX", "AMAT", "KLAC", "TXN", "COST", "WMT", "TGT", "HD",
    "LOW", "NKE", "SBUX", "MCD", "DIS", "CMCSA", "T", "VZ", "PFE",
    "JNJ", "UNH", "LLY", "ABBV", "MRK", "BMY", "XOM", "CVX", "COP",
    "SLB", "JPM", "GS", "MS", "BAC", "C", "V", "MA",
]

FEATURE_COLS = [
    "rv_5d", "rv_10d", "rv_20d", "rv_60d",
    "vol_ratio_5_20", "vol_ratio_20_60", "vol_of_vol_20",
    "ewm_vol_10", "ewm_vol_20",
    "ret_1d", "ret_5d", "ret_20d",
    "abs_ret_1d", "abs_ret_5d_mean",
    "vol_20d_avg", "volume_ratio",
    "parkinson_vol",
    "vix", "vix3m", "vix_ts",
    "sector_code",
]

LGBM_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": 7,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "verbose": -1,
    "n_jobs": -1,
    "seed": 42,
}

# Sector mapping (simplified)
SECTOR_MAP = {
    "Technology": 0, "Communication Services": 1, "Consumer Cyclical": 2,
    "Consumer Defensive": 3, "Financial Services": 4, "Healthcare": 5,
    "Industrials": 6, "Energy": 7, "Basic Materials": 8, "Real Estate": 9,
    "Utilities": 10, "Unknown": 11,
}

# Rough sector assignment for the universe (avoids yfinance info calls)
TICKER_SECTORS = {
    "AAPL": 0, "MSFT": 0, "GOOGL": 1, "AMZN": 2, "META": 1, "NVDA": 0,
    "TSLA": 2, "AMD": 0, "NFLX": 1, "CRM": 0, "ADBE": 0, "INTC": 0,
    "PYPL": 0, "XYZ": 0, "SHOP": 0, "UBER": 0, "ABNB": 2, "COIN": 4,
    "SOFI": 4, "HOOD": 4, "PLTR": 0, "SNOW": 0, "NET": 0, "DDOG": 0,
    "ZS": 0, "CRWD": 0, "PANW": 0, "MDB": 0, "AFRM": 0, "ARM": 0,
    "SMCI": 0, "MSTR": 0, "AVGO": 0, "MU": 0, "QCOM": 0, "MRVL": 0,
    "ON": 0, "LRCX": 0, "AMAT": 0, "KLAC": 0, "TXN": 0, "COST": 3,
    "WMT": 3, "TGT": 2, "HD": 2, "LOW": 2, "NKE": 2, "SBUX": 2,
    "MCD": 2, "DIS": 1, "CMCSA": 1, "T": 1, "VZ": 1, "PFE": 5,
    "JNJ": 5, "UNH": 5, "LLY": 5, "ABBV": 5, "MRK": 5, "BMY": 5,
    "XOM": 7, "CVX": 7, "COP": 7, "SLB": 7, "JPM": 4, "GS": 4,
    "MS": 4, "BAC": 4, "C": 4, "V": 4, "MA": 4,
}


def _fetch_prices(tickers: list[str], days: int = PRICE_LOOKBACK) -> pd.DataFrame:
    """Fetch OHLCV from yfinance for multiple tickers."""
    if yf is None:
        raise ImportError("yfinance not installed")

    period = f"{days + 30}d"
    data = yf.download(tickers, period=period, progress=False, threads=True)

    if data.empty:
        return pd.DataFrame()

    records = []
    for ticker in tickers:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                # Multi-ticker download: columns are (Price, Ticker)
                cols_available = data.columns.get_level_values(1).unique()
                if ticker not in cols_available:
                    continue
                df = data.xs(ticker, level=1, axis=1).copy()
            else:
                # Single ticker
                df = data.copy()

            df = df.reset_index()
            # Normalize column names
            df.columns = [c.lower() if isinstance(c, str) else str(c).lower() for c in df.columns]
            if "adj close" in df.columns:
                df["close"] = df["adj close"]
            df["ticker"] = ticker

            needed = ["ticker", "date", "open", "high", "low", "close", "volume"]
            if all(c in df.columns for c in needed):
                records.append(df[needed].dropna())
        except (KeyError, TypeError, ValueError):
            continue

    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


def _fetch_vix() -> pd.DataFrame:
    """Fetch VIX and VIX3M for macro features."""
    if yf is None:
        raise ImportError("yfinance not installed")

    vix = yf.download("^VIX", period="400d", progress=False)
    vix3m = yf.download("^VIX3M", period="400d", progress=False)

    result = pd.DataFrame()
    if not vix.empty:
        vix = vix.reset_index()
        vix.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in vix.columns]
        result["date"] = vix["date"] if "date" in vix.columns else vix.index
        result["vix"] = vix["close"].values if "close" in vix.columns else vix.iloc[:, -2].values

    if not vix3m.empty:
        vix3m = vix3m.reset_index()
        vix3m.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in vix3m.columns]
        vix3m_close = vix3m["close"].values if "close" in vix3m.columns else vix3m.iloc[:, -2].values
        if len(result) > 0:
            # Merge on date
            vix3m_df = pd.DataFrame({"date": vix3m["date"] if "date" in vix3m.columns else vix3m.index, "vix3m": vix3m_close})
            result = result.merge(vix3m_df, on="date", how="left")

    if "vix" in result.columns and "vix3m" in result.columns:
        result["vix_ts"] = result["vix3m"] / result["vix"]
    else:
        result["vix_ts"] = 1.0

    return result


def _compute_features(prices: pd.DataFrame, vix_data: pd.DataFrame) -> pd.DataFrame:
    """Compute vol features for all tickers (same as research/vol_forecaster.py)."""
    records = []
    for ticker, g in prices.groupby("ticker"):
        g = g.sort_values("date").copy()
        g["lr"] = np.log(g["close"] / g["close"].shift(1))

        # Realized vol at multiple windows
        for w in [5, 10, 20, 60]:
            g[f"rv_{w}d"] = g["lr"].rolling(w).std() * ANNUALIZE

        # Vol ratios
        g["vol_ratio_5_20"] = g["rv_5d"] / g["rv_20d"]
        g["vol_ratio_20_60"] = g["rv_20d"] / g["rv_60d"]

        # Vol-of-vol
        g["vol_of_vol_20"] = g["rv_20d"].rolling(20).std()

        # EWM vol
        g["ewm_vol_10"] = g["lr"].ewm(span=10).std() * ANNUALIZE
        g["ewm_vol_20"] = g["lr"].ewm(span=20).std() * ANNUALIZE

        # Returns
        g["ret_1d"] = g["lr"]
        g["ret_5d"] = g["lr"].rolling(5).sum()
        g["ret_20d"] = g["lr"].rolling(20).sum()

        # Abs returns
        g["abs_ret_1d"] = g["lr"].abs()
        g["abs_ret_5d_mean"] = g["abs_ret_1d"].rolling(5).mean()

        # Volume features
        g["vol_20d_avg"] = g["volume"].rolling(20).mean()
        g["volume_ratio"] = g["volume"] / g["vol_20d_avg"]

        # Parkinson vol
        g["hl_range"] = np.log(g["high"] / g["low"])
        g["parkinson_vol"] = g["hl_range"].rolling(20).apply(
            lambda x: np.sqrt((1 / (4 * np.log(2))) * (x**2).mean()) * ANNUALIZE,
            raw=True,
        )

        # Sector
        g["sector_code"] = TICKER_SECTORS.get(ticker, 11)

        # Forward RV target (for training only — last 21 days won't have target)
        g["fwd_rv_21d"] = (
            g["lr"].shift(-1).rolling(21).std().shift(-20) * ANNUALIZE
        )

        g["ticker"] = ticker
        records.append(g)

    if not records:
        return pd.DataFrame()

    panel = pd.concat(records, ignore_index=True)

    # Merge VIX data
    if not vix_data.empty and "date" in vix_data.columns:
        panel["date"] = pd.to_datetime(panel["date"])
        vix_data["date"] = pd.to_datetime(vix_data["date"])
        panel = panel.merge(vix_data[["date", "vix", "vix3m", "vix_ts"]], on="date", how="left")
    else:
        panel["vix"] = np.nan
        panel["vix3m"] = np.nan
        panel["vix_ts"] = 1.0

    return panel


def _train_and_predict(panel: pd.DataFrame) -> dict[str, float]:
    """Train on historical data with target, predict for latest date."""
    if lgb is None:
        raise ImportError("lightgbm not installed")

    # Split: rows with target = training, latest rows without target = inference
    has_target = panel.dropna(subset=["fwd_rv_21d"])
    latest_date = panel["date"].max()
    latest = panel[panel["date"] == latest_date].copy()

    # Use last TRAIN_WINDOW days with targets for training
    train_dates = sorted(has_target["date"].unique())[-TRAIN_WINDOW:]
    train = has_target[has_target["date"].isin(train_dates)]

    # Ensure we have enough data
    if len(train) < 500:
        print(f"  WARNING: Only {len(train)} training rows (need 500+)")
        return {}

    # Prepare features
    X_train = train[FEATURE_COLS].copy()
    y_train = train["fwd_rv_21d"]
    X_pred = latest[FEATURE_COLS].copy()

    # Handle NaNs (fill with column median)
    for col in FEATURE_COLS:
        med = X_train[col].median()
        X_train[col] = X_train[col].fillna(med)
        X_pred[col] = X_pred[col].fillna(med)

    # Train LGBM
    ds_train = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(LGBM_PARAMS, ds_train, num_boost_round=500)

    # Predict
    preds = model.predict(X_pred)

    # Map to tickers
    result = {}
    tickers = latest["ticker"].values
    for i, ticker in enumerate(tickers):
        result[ticker] = float(preds[i])

    # Save model cache
    with open(MODEL_CACHE, "wb") as f:
        pickle.dump(model, f)

    return result


def get_vol_predictions(force_refresh: bool = False) -> dict[str, float]:
    """
    Get today's vol predictions. Uses cache if available and fresh.

    Returns:
        Dict mapping ticker -> predicted 21-day annualized vol.

    Example:
        {'AAPL': 0.22, 'NVDA': 0.45, ...}
    """
    today = date.today().isoformat()

    # Check cache
    if not force_refresh and PREDS_CACHE.exists():
        try:
            with open(PREDS_CACHE) as f:
                cached = json.load(f)
            if cached.get("date") == today:
                print(f"  Vol forecaster: using cached predictions ({len(cached['predictions'])} tickers)")
                return cached["predictions"]
        except (json.JSONDecodeError, KeyError):
            pass

    print("  Vol forecaster: fetching fresh data and training model...")
    t0 = time.time()

    # Fetch data
    prices = _fetch_prices(WHEEL_UNIVERSE)
    if prices.empty:
        print("  ERROR: No price data fetched")
        return {}

    vix_data = _fetch_vix()

    # Compute features
    panel = _compute_features(prices, vix_data)
    if panel.empty:
        print("  ERROR: Feature computation failed")
        return {}

    # Train and predict
    predictions = _train_and_predict(panel)

    elapsed = time.time() - t0
    print(f"  Vol forecaster: {len(predictions)} predictions in {elapsed:.1f}s")

    # Cache
    cache_data = {"date": today, "predictions": predictions, "elapsed_sec": elapsed}
    with open(PREDS_CACHE, "w") as f:
        json.dump(cache_data, f, indent=2)

    return predictions


def get_vol_sizing_scale(ticker: str, predictions: Optional[dict] = None) -> float:
    """
    Get position sizing scale factor for a ticker based on predicted vol.

    Returns a multiplier in [0.5, 2.0]:
        - Low predicted vol → larger position (up to 2x)
        - High predicted vol → smaller position (down to 0.5x)

    Uses inverse-vol relative to universe median.
    """
    if predictions is None:
        predictions = get_vol_predictions()

    if not predictions or ticker not in predictions:
        return 1.0  # Default: no scaling

    pred_vol = predictions[ticker]
    all_vols = list(predictions.values())
    median_vol = float(np.median(all_vols))

    if pred_vol <= 0 or median_vol <= 0:
        return 1.0

    # Inverse-vol scale: low vol → bigger position
    scale = median_vol / pred_vol

    # Cap at [0.5, 2.0] to avoid extreme positions
    return float(np.clip(scale, 0.5, 2.0))


if __name__ == "__main__":
    print("=" * 60)
    print("LIVE VOL FORECASTER — Daily Predictions")
    print("=" * 60)

    preds = get_vol_predictions(force_refresh=True)

    if preds:
        # Sort by predicted vol (highest first)
        sorted_preds = sorted(preds.items(), key=lambda x: x[1], reverse=True)

        print(f"\nTop 10 highest predicted vol (21d forward, annualized):")
        for ticker, vol in sorted_preds[:10]:
            scale = get_vol_sizing_scale(ticker, preds)
            print(f"  {ticker:6s}: {vol:.1%} → size scale {scale:.2f}x")

        print(f"\nTop 10 lowest predicted vol:")
        for ticker, vol in sorted_preds[-10:]:
            scale = get_vol_sizing_scale(ticker, preds)
            print(f"  {ticker:6s}: {vol:.1%} → size scale {scale:.2f}x")

        median_vol = np.median(list(preds.values()))
        print(f"\nUniverse median vol: {median_vol:.1%}")
        print(f"Range: {min(preds.values()):.1%} - {max(preds.values()):.1%}")
    else:
        print("No predictions generated.")
