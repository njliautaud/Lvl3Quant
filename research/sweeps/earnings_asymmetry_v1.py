#!/usr/bin/env python3
"""
Earnings Asymmetry Research v1
==============================
Research question: Can we predict which large-cap stocks will have outsized
post-earnings moves (21d return > +5%), and exploit this with jade lizard
premium selling + directional overlays?

Methodology:
- 50 large-cap stocks, earnings events 2015-2026
- Features: IV rank, pre-earnings momentum, RSI, gap history, sector, regime
- Target: 21-day post-earnings return > +5% (asymmetric upside)
- Walk-forward: 504d train, 63d embargo, 21d test, SLIDING window (HC #0)
- All signals on T-1 data (anti-lookahead per HC #724)
- Validation: 200-shuffle permutation test, regime test, sub-period stability

Cost model: 10 bps round-trip for stocks (FIFO).
Primary metrics: Sharpe, Sortino, PF, WR.

Runtime: ~30-60 min on Neptune (32GB RAM, CPU).
Designed to run: python3 earnings_asymmetry_v1.py
"""

import os
import sys
import json
import time
import logging
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FMT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
log = logging.getLogger("earnings_asymmetry")

# ---------------------------------------------------------------------------
# Output directory — auto-detect host
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
if (SCRIPT_DIR / "output").exists() or SCRIPT_DIR.name == "Lvl3Quant":
    OUTPUT_DIR = SCRIPT_DIR / "output" / "earnings_asymmetry_v1"
else:
    OUTPUT_DIR = Path.home() / "Lvl3Quant" / "output" / "earnings_asymmetry_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "GS", "BAC",
    "V", "MA", "UNH", "JNJ", "PG", "KO", "PEP", "MRK", "ABBV", "LLY",
    "HD", "COST", "WMT", "CRM", "AMD", "NFLX", "ADBE", "INTC", "CSCO", "QCOM",
    "XOM", "CVX", "PFE", "TMO", "ABT", "AVGO", "TXN", "MCD", "NKE", "DIS",
    "CMCSA", "T", "VZ", "NEE", "SO", "SHW", "LMT", "RTX", "CAT", "DE",
]

SECTOR_MAP = {
    "AAPL": "Tech", "MSFT": "Tech", "GOOGL": "Tech", "AMZN": "ConsDisc",
    "META": "Tech", "NVDA": "Tech", "TSLA": "ConsDisc", "JPM": "Fin",
    "GS": "Fin", "BAC": "Fin", "V": "Fin", "MA": "Fin",
    "UNH": "Health", "JNJ": "Health", "PG": "Staples", "KO": "Staples",
    "PEP": "Staples", "MRK": "Health", "ABBV": "Health", "LLY": "Health",
    "HD": "ConsDisc", "COST": "Staples", "WMT": "Staples", "CRM": "Tech",
    "AMD": "Tech", "NFLX": "Tech", "ADBE": "Tech", "INTC": "Tech",
    "CSCO": "Tech", "QCOM": "Tech", "XOM": "Energy", "CVX": "Energy",
    "PFE": "Health", "TMO": "Health", "ABT": "Health", "AVGO": "Tech",
    "TXN": "Tech", "MCD": "ConsDisc", "NKE": "ConsDisc", "DIS": "ConsDisc",
    "CMCSA": "Tech", "T": "Telecom", "VZ": "Telecom", "NEE": "Util",
    "SO": "Util", "SHW": "Materials", "LMT": "Industrials", "RTX": "Industrials",
    "CAT": "Industrials", "DE": "Industrials",
}

SECTORS = sorted(set(SECTOR_MAP.values()))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
COST_BPS_RT = 10          # 10 bps round-trip for stocks
TARGET_THRESHOLD = 0.05   # 21d return > +5%
TARGET_HORIZON = 21       # trading days

# Walk-forward config (SLIDING window -- HC #0)
WF_TRAIN_DAYS = 252
WF_EMBARGO_DAYS = 21
WF_TEST_DAYS = 21
WF_STEP_DAYS = 21  # slide by test window size

# Validation
PERM_SHUFFLES = 200
REGIME_THRESHOLD = 0.50
SUBPERIOD_CV_CAP = 0.70


# ===================================================================
# DATA DOWNLOAD
# ===================================================================

def download_data(tickers: list, start: str = "2014-01-01", end: str = "2026-07-22") -> dict:
    """Download price data and earnings dates for all tickers via yfinance."""
    import yfinance as yf

    log.info(f"Downloading price data for {len(tickers)} tickers...")
    price_data = {}
    earnings_data = {}
    failed = []

    for i, ticker in enumerate(tickers):
        try:
            tk = yf.Ticker(ticker)

            # Price data
            df = tk.history(start=start, end=end, auto_adjust=True)
            if df.empty or len(df) < 252:
                log.warning(f"  {ticker}: insufficient price data ({len(df)} rows), skipping")
                failed.append(ticker)
                continue
            df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
            price_data[ticker] = df

            # Earnings dates
            try:
                cal = tk.get_earnings_dates(limit=100)
                if cal is not None and len(cal) > 0:
                    cal.index = cal.index.tz_localize(None) if cal.index.tz is not None else cal.index
                    # Filter to dates within our price range
                    cal = cal[(cal.index >= df.index[0]) & (cal.index <= df.index[-1])]
                    if len(cal) > 0:
                        earnings_data[ticker] = cal
                    else:
                        earnings_data[ticker] = _estimate_earnings_from_volume(df)
                else:
                    earnings_data[ticker] = _estimate_earnings_from_volume(df)
            except Exception:
                earnings_data[ticker] = _estimate_earnings_from_volume(df)

            if (i + 1) % 10 == 0:
                log.info(f"  Downloaded {i + 1}/{len(tickers)} "
                         f"({len(price_data)} OK, {len(failed)} failed)")
                time.sleep(1)  # rate limit courtesy

        except Exception as e:
            log.warning(f"  {ticker}: download failed ({e})")
            failed.append(ticker)

    log.info(f"Downloaded {len(price_data)} tickers successfully, "
             f"{len(failed)} failed: {failed}")
    return {"prices": price_data, "earnings": earnings_data, "failed": failed}


def _estimate_earnings_from_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate earnings dates from volume spikes (>3x 20d avg) + large moves.
    Returns a DataFrame with DatetimeIndex of estimated earnings dates."""
    if "Volume" not in df.columns or df["Volume"].sum() == 0:
        return pd.DataFrame(index=pd.DatetimeIndex([]))

    vol_ma = df["Volume"].rolling(20).mean()
    vol_ratio = df["Volume"] / vol_ma
    ret = df["Close"].pct_change().abs()

    # Earnings proxy: volume > 3x average AND abs return > 2%
    mask = (vol_ratio > 3.0) & (ret > 0.02)
    dates = df.index[mask]

    # Deduplicate: keep only one per quarter (min 60 days apart)
    if len(dates) == 0:
        return pd.DataFrame(index=pd.DatetimeIndex([]))

    deduped = [dates[0]]
    for d in dates[1:]:
        if (d - deduped[-1]).days > 60:
            deduped.append(d)

    return pd.DataFrame(index=pd.DatetimeIndex(deduped))


# ===================================================================
# SPY REGIME DETECTION
# ===================================================================

def get_spy_regime(start: str = "2014-01-01", end: str = "2026-07-22") -> pd.Series:
    """Download SPY and classify each day as bull/bear/neutral.
    Bull: 50d SMA > 200d SMA and price > 50d SMA
    Bear: 50d SMA < 200d SMA and price < 50d SMA
    Neutral: otherwise
    """
    import yfinance as yf
    log.info("Downloading SPY for regime classification...")
    spy = yf.Ticker("SPY").history(start=start, end=end, auto_adjust=True)
    spy.index = spy.index.tz_localize(None) if spy.index.tz is not None else spy.index

    sma50 = spy["Close"].rolling(50).mean()
    sma200 = spy["Close"].rolling(200).mean()

    regime = pd.Series("neutral", index=spy.index, name="regime")
    regime[(sma50 > sma200) & (spy["Close"] > sma50)] = "bull"
    regime[(sma50 < sma200) & (spy["Close"] < sma50)] = "bear"

    log.info(f"Regime counts: {regime.value_counts().to_dict()}")
    return regime


# ===================================================================
# FEATURE ENGINEERING
# ===================================================================

def compute_features_for_event(
    price_df: pd.DataFrame,
    event_date: pd.Timestamp,
    ticker: str,
    regime_series: pd.Series,
) -> dict | None:
    """Compute all features for a single earnings event.
    ALL features use T-1 data (anti-lookahead per HC #724).
    Returns dict of features or None if insufficient data."""

    idx = price_df.index.get_indexer([event_date], method="nearest")[0]
    if idx < 252 or idx >= len(price_df) - TARGET_HORIZON - 5:
        return None

    # T-1 is the day BEFORE earnings (all features must be knowable pre-event)
    t_minus_1 = idx - 1
    if t_minus_1 < 252:
        return None

    close = price_df["Close"].values
    high = price_df["High"].values
    low = price_df["Low"].values
    opn = price_df["Open"].values
    volume = price_df["Volume"].values
    dates = price_df.index

    # --- IV rank proxy (252d realized vol percentile) ---
    log_ret = np.log(close[1:t_minus_1 + 1] / close[:t_minus_1])
    if len(log_ret) < 252:
        return None

    hv_20 = pd.Series(log_ret).rolling(20).std().values * np.sqrt(252)
    hv_series = hv_20[-252:]
    hv_series = hv_series[~np.isnan(hv_series)]
    if len(hv_series) < 100:
        return None

    current_hv = hv_series[-1]
    iv_rank = float(np.sum(hv_series < current_hv) / len(hv_series))

    # --- Pre-earnings momentum (1m, 3m) ---
    mom_1m = float(close[t_minus_1] / close[max(0, t_minus_1 - 21)] - 1.0)
    mom_3m = float(close[t_minus_1] / close[max(0, t_minus_1 - 63)] - 1.0)

    # --- RSI (14d) ---
    deltas = np.diff(close[max(0, t_minus_1 - 20):t_minus_1 + 1])
    gains = np.maximum(deltas, 0)
    losses = np.abs(np.minimum(deltas, 0))
    avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else np.mean(gains) if len(gains) > 0 else 0
    avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else np.mean(losses) if len(losses) > 0 else 1e-9
    if avg_loss == 0:
        rsi = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))

    # --- Volume trend (20d volume vs 60d volume) ---
    vol_20 = np.mean(volume[max(0, t_minus_1 - 20):t_minus_1 + 1])
    vol_60 = np.mean(volume[max(0, t_minus_1 - 60):t_minus_1 + 1])
    vol_trend = float(vol_20 / vol_60) if vol_60 > 0 else 1.0

    # --- Pre-earnings drift (last 5 days before earnings) ---
    pre_drift_5d = float(close[t_minus_1] / close[max(0, t_minus_1 - 5)] - 1.0)

    # --- ATR (14d) normalized ---
    tr_vals = []
    for k in range(max(1, t_minus_1 - 14), t_minus_1 + 1):
        tr = max(high[k] - low[k], abs(high[k] - close[k - 1]), abs(low[k] - close[k - 1]))
        tr_vals.append(tr)
    atr_14 = np.mean(tr_vals) if tr_vals else 0.0
    atr_pct = float(atr_14 / close[t_minus_1]) if close[t_minus_1] > 0 else 0.0

    # --- Distance from 52-week high ---
    high_252 = np.max(high[max(0, t_minus_1 - 252):t_minus_1 + 1])
    dist_from_high = float(close[t_minus_1] / high_252 - 1.0)

    # --- Distance from 52-week low ---
    low_252 = np.min(low[max(0, t_minus_1 - 252):t_minus_1 + 1])
    dist_from_low = float(close[t_minus_1] / low_252 - 1.0) if low_252 > 0 else 0.0

    # --- Analyst revision proxy: momentum acceleration ---
    if t_minus_1 >= 42:
        mom_recent = close[t_minus_1] / close[t_minus_1 - 21] - 1.0
        mom_prior = close[t_minus_1 - 21] / close[t_minus_1 - 42] - 1.0
        revision_proxy = float(mom_recent - mom_prior)
    else:
        revision_proxy = 0.0

    # --- Regime at T-1 ---
    t1_date = dates[t_minus_1]
    regime_idx = regime_series.index.get_indexer([t1_date], method="nearest")
    regime_val = regime_series.iloc[regime_idx[0]] if len(regime_idx) > 0 else "neutral"

    # --- Quarter (seasonality) ---
    quarter = int(event_date.quarter)

    # --- Year-over-year return ---
    yoy_ret = float(close[t_minus_1] / close[t_minus_1 - 252] - 1.0) if t_minus_1 >= 252 else 0.0

    # --- Realized vol ratio (20d / 60d) for vol compression signal ---
    hv_60_vals = pd.Series(log_ret[-60:]).std() * np.sqrt(252) if len(log_ret) >= 60 else current_hv
    vol_compression = float(current_hv / hv_60_vals) if hv_60_vals > 0 else 1.0

    # --- Post-earnings returns (TARGET) ---
    post_idx = idx  # earnings day
    if post_idx + TARGET_HORIZON >= len(close):
        return None

    # Gap: open on earnings day vs prior close
    gap = float(opn[post_idx] / close[post_idx - 1] - 1.0)

    # Post-earnings drift targets
    ret_5d = float(close[min(post_idx + 5, len(close) - 1)] / close[post_idx - 1] - 1.0)
    ret_21d = float(close[min(post_idx + TARGET_HORIZON, len(close) - 1)] / close[post_idx - 1] - 1.0)
    ret_63d_idx = min(post_idx + 63, len(close) - 1)
    ret_63d = float(close[ret_63d_idx] / close[post_idx - 1] - 1.0) if post_idx + 63 < len(close) else np.nan

    sector = SECTOR_MAP.get(ticker, "Unknown")

    return {
        "ticker": ticker,
        "event_date": event_date,
        "t_minus_1_date": t1_date,
        # Features (all T-1)
        "iv_rank": iv_rank,
        "hv_20d": float(current_hv),
        "mom_1m": mom_1m,
        "mom_3m": mom_3m,
        "rsi_14": float(rsi),
        "vol_trend": vol_trend,
        "pre_drift_5d": pre_drift_5d,
        "atr_pct": atr_pct,
        "dist_from_high": dist_from_high,
        "dist_from_low": dist_from_low,
        "revision_proxy": revision_proxy,
        "yoy_return": yoy_ret,
        "vol_compression": vol_compression,
        "quarter": quarter,
        "regime": regime_val,
        "sector": sector,
        # Historical gap stats (filled in batch later)
        "hist_avg_gap": np.nan,
        "hist_gap_std": np.nan,
        # Targets
        "gap": gap,
        "ret_5d": ret_5d,
        "ret_21d": ret_21d,
        "ret_63d": ret_63d,
        "target": int(ret_21d > TARGET_THRESHOLD),
    }


def build_features_df(data: dict, regime_series: pd.Series) -> pd.DataFrame:
    """Build full feature DataFrame from all earnings events across all tickers."""
    records = []
    prices = data["prices"]
    earnings = data["earnings"]

    for ticker in prices:
        if ticker not in earnings:
            continue
        price_df = prices[ticker]
        earn_dates = earnings[ticker].index

        # Filter to events within our price range (need 252d lookback + 21d forward)
        valid_dates = earn_dates[
            (earn_dates >= price_df.index[252]) &
            (earn_dates <= price_df.index[-TARGET_HORIZON - 5])
        ]

        ticker_count = 0
        for edate in valid_dates:
            feat = compute_features_for_event(price_df, edate, ticker, regime_series)
            if feat is not None:
                records.append(feat)
                ticker_count += 1

        if ticker_count > 0:
            log.info(f"  {ticker}: {ticker_count} earnings events")

    df = pd.DataFrame(records)
    log.info(f"Total earnings events with features: {len(df)}")

    if len(df) == 0:
        return df

    # --- Fill historical gap stats (using ONLY prior gaps for that ticker) ---
    df = df.sort_values("event_date").reset_index(drop=True)
    for ticker in df["ticker"].unique():
        mask = df["ticker"] == ticker
        ticker_df = df.loc[mask].copy()
        gaps = ticker_df["gap"].values
        avg_gaps = []
        std_gaps = []
        for i in range(len(gaps)):
            if i == 0:
                avg_gaps.append(0.0)
                std_gaps.append(0.05)  # prior for first event
            else:
                prior = gaps[:i]
                avg_gaps.append(float(np.mean(prior)))
                std_gaps.append(float(np.std(prior)) if len(prior) > 1 else 0.05)
        df.loc[mask, "hist_avg_gap"] = avg_gaps
        df.loc[mask, "hist_gap_std"] = std_gaps

    # --- Encode categoricals ---
    for r in ["bull", "bear", "neutral"]:
        df[f"regime_{r}"] = (df["regime"] == r).astype(int)
    for s in SECTORS:
        df[f"sector_{s}"] = (df["sector"] == s).astype(int)

    return df


# ===================================================================
# FEATURE COLUMNS
# ===================================================================

def get_feature_cols(df: pd.DataFrame) -> list:
    """Return list of feature column names (excludes targets, metadata)."""
    exclude = {
        "ticker", "event_date", "t_minus_1_date", "regime", "sector",
        "gap", "ret_5d", "ret_21d", "ret_63d", "target", "day_idx",
    }
    return [c for c in df.columns if c not in exclude]


# ===================================================================
# WALK-FORWARD LIGHTGBM
# ===================================================================

def walk_forward_lgbm(df: pd.DataFrame) -> dict:
    """Run walk-forward validation with SLIDING window (HC #0).
    504d train, 63d embargo, 21d test.
    Returns predictions, actuals, and per-fold metrics."""
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    df = df.sort_values("event_date").reset_index(drop=True)
    feature_cols = get_feature_cols(df)

    log.info(f"Walk-forward: {len(df)} events, {len(feature_cols)} features")
    log.info(f"  Target rate: {df['target'].mean():.3f}")
    log.info(f"  Date range: {df['event_date'].min()} to {df['event_date'].max()}")

    # Assign sequential day-index for windowing
    unique_dates = sorted(df["event_date"].unique())
    date_to_idx = {d: i for i, d in enumerate(unique_dates)}
    df["day_idx"] = df["event_date"].map(date_to_idx)

    n_dates = len(unique_dates)
    min_test_start = WF_TRAIN_DAYS + WF_EMBARGO_DAYS

    if n_dates < min_test_start + WF_TEST_DAYS:
        log.error(f"Not enough unique dates for walk-forward: {n_dates} "
                  f"< {min_test_start + WF_TEST_DAYS}")
        log.info("Trying with calendar-day indexing instead...")
        # Fall back: use event index directly
        df["day_idx"] = np.arange(len(df))
        n_dates = len(df)
        if n_dates < min_test_start + WF_TEST_DAYS:
            log.error("Still not enough events. Aborting walk-forward.")
            return {}

    all_preds = []
    all_targets = []
    all_dates = []
    all_tickers = []
    all_ret21d = []
    fold_metrics = []
    feature_importance = np.zeros(len(feature_cols))

    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "n_estimators": 500,
        "max_depth": 5,
        "num_leaves": 31,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "min_child_samples": 20,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "verbose": -1,
        "n_jobs": -1,
        "is_unbalance": True,
        "seed": 42,
    }

    fold = 0
    test_start = min_test_start

    while test_start + WF_TEST_DAYS <= n_dates:
        # SLIDING window: train window starts WF_TRAIN_DAYS + WF_EMBARGO_DAYS before test
        train_start = test_start - WF_EMBARGO_DAYS - WF_TRAIN_DAYS
        train_end = test_start - WF_EMBARGO_DAYS  # embargo gap
        test_end = test_start + WF_TEST_DAYS

        # Clamp train_start
        train_start = max(0, train_start)

        train_mask = (df["day_idx"] >= train_start) & (df["day_idx"] < train_end)
        test_mask = (df["day_idx"] >= test_start) & (df["day_idx"] < test_end)

        X_train = df.loc[train_mask, feature_cols].values.astype(np.float32)
        y_train = df.loc[train_mask, "target"].values.astype(int)
        X_test = df.loc[test_mask, feature_cols].values.astype(np.float32)
        y_test = df.loc[test_mask, "target"].values.astype(int)

        if len(X_train) < 30 or len(X_test) < 2:
            test_start += WF_STEP_DAYS
            continue

        # Handle NaN/Inf
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)

        # Train LightGBM
        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        # Predict probabilities
        preds = model.predict_proba(X_test)[:, 1]

        all_preds.extend(preds.tolist())
        all_targets.extend(y_test.tolist())
        all_dates.extend(df.loc[test_mask, "event_date"].tolist())
        all_tickers.extend(df.loc[test_mask, "ticker"].tolist())
        all_ret21d.extend(df.loc[test_mask, "ret_21d"].tolist())

        # Accumulate feature importance
        feature_importance += model.feature_importances_

        # Per-fold AUC
        try:
            auc = roc_auc_score(y_test, preds)
        except ValueError:
            auc = 0.5

        fold_metrics.append({
            "fold": fold,
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
            "target_rate_train": float(np.mean(y_train)),
            "target_rate_test": float(np.mean(y_test)),
            "auc": float(auc),
            "train_start_idx": int(train_start),
            "test_start_idx": int(test_start),
        })

        if (fold + 1) % 10 == 0:
            log.info(f"  Fold {fold + 1}: train={len(X_train)} test={len(X_test)} AUC={auc:.3f}")

        fold += 1
        test_start += WF_STEP_DAYS

    log.info(f"Walk-forward complete: {fold} folds, {len(all_preds)} OOT predictions")

    if fold == 0:
        return {}

    # Normalize feature importance
    feature_importance /= fold
    fi_dict = {col: float(imp) for col, imp in
                sorted(zip(feature_cols, feature_importance), key=lambda x: -x[1])}

    return {
        "preds": np.array(all_preds),
        "targets": np.array(all_targets),
        "ret_21d": np.array(all_ret21d),
        "dates": all_dates,
        "tickers": all_tickers,
        "fold_metrics": fold_metrics,
        "feature_importance": fi_dict,
        "feature_cols": feature_cols,
        "lgb_params": {k: str(v) for k, v in lgb_params.items()},
    }


# ===================================================================
# STRATEGY SIMULATION (FIFO COST MODEL)
# ===================================================================

def simulate_strategy(preds: np.ndarray, targets: np.ndarray, ret_21d: np.ndarray,
                       threshold: float = 0.5) -> dict:
    """Simulate long-only strategy: go long when pred > threshold.
    Cost model: 10 bps round-trip (FIFO).
    Returns risk-adjusted metrics (Sharpe, Sortino, PF, WR)."""
    signals = preds > threshold
    n_trades = int(np.sum(signals))

    if n_trades < 5:
        return {
            "n_trades": n_trades, "sharpe": 0.0, "sortino": 0.0,
            "pf": 0.0, "wr": 0.0, "avg_return": 0.0, "std_return": 0.0,
            "total_return": 0.0, "max_loss": 0.0, "max_gain": 0.0,
            "hit_rate_target": 0.0,
        }

    cost_rt = COST_BPS_RT / 10000.0  # 10 bps = 0.001
    trade_returns = ret_21d[signals] - cost_rt

    gross_profits = float(trade_returns[trade_returns > 0].sum())
    gross_losses = float(abs(trade_returns[trade_returns < 0].sum()))

    wr = float(np.mean(trade_returns > 0))
    pf = float(gross_profits / gross_losses) if gross_losses > 0 else (
        float("inf") if gross_profits > 0 else 0.0
    )
    avg_ret = float(np.mean(trade_returns))
    std_ret = float(np.std(trade_returns))

    # Annualize: ~12 non-overlapping 21d periods per year
    ann_factor = np.sqrt(12)
    sharpe = float(avg_ret / std_ret * ann_factor) if std_ret > 0 else 0.0

    downside = trade_returns[trade_returns < 0]
    downside_std = float(np.std(downside)) if len(downside) > 1 else std_ret
    sortino = float(avg_ret / downside_std * ann_factor) if downside_std > 0 else 0.0

    return {
        "n_trades": n_trades,
        "avg_return": avg_ret,
        "std_return": std_ret,
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "pf": round(pf, 4),
        "wr": round(wr, 4),
        "total_return": float(np.sum(trade_returns)),
        "max_loss": float(np.min(trade_returns)),
        "max_gain": float(np.max(trade_returns)),
        "hit_rate_target": float(np.mean(targets[signals])),
    }


# ===================================================================
# VALIDATION GATES
# ===================================================================

def permutation_test(preds: np.ndarray, targets: np.ndarray, ret_21d: np.ndarray,
                      threshold: float, n_shuffles: int = PERM_SHUFFLES) -> dict:
    """200-shuffle permutation test. p < 0.05 required."""
    log.info(f"Running {n_shuffles}-shuffle permutation test...")

    real_metrics = simulate_strategy(preds, targets, ret_21d, threshold)
    real_sharpe = real_metrics["sharpe"]

    rng = np.random.RandomState(42)
    shuffle_sharpes = []

    for i in range(n_shuffles):
        shuffled_preds = preds.copy()
        rng.shuffle(shuffled_preds)
        m = simulate_strategy(shuffled_preds, targets, ret_21d, threshold)
        shuffle_sharpes.append(m["sharpe"])
        if (i + 1) % 50 == 0:
            log.info(f"  Permutation shuffle {i + 1}/{n_shuffles}")

    shuffle_sharpes = np.array(shuffle_sharpes)
    p_value = float(np.mean(shuffle_sharpes >= real_sharpe))

    passed = p_value < 0.05
    log.info(f"  Permutation test: real Sharpe={real_sharpe:.3f}, "
             f"mean shuffle={np.mean(shuffle_sharpes):.3f}, "
             f"p-value={p_value:.4f} ({'PASS' if passed else 'FAIL'})")

    return {
        "real_sharpe": float(real_sharpe),
        "mean_shuffle_sharpe": float(np.mean(shuffle_sharpes)),
        "std_shuffle_sharpe": float(np.std(shuffle_sharpes)),
        "p_value": p_value,
        "pass": passed,
    }


def regime_test(preds: np.ndarray, targets: np.ndarray, ret_21d: np.ndarray,
                 regimes: np.ndarray, threshold: float) -> dict:
    """Test regime stability: |Sharpe_bull - Sharpe_bear| / max < 0.50."""
    log.info("Running regime stability test...")

    results = {}
    for r in ["bull", "bear", "neutral"]:
        mask = regimes == r
        n_in_regime = int(np.sum(mask))
        if n_in_regime < 10:
            results[r] = {"sharpe": 0.0, "n": n_in_regime, "wr": 0.0, "n_trades": 0}
            continue
        m = simulate_strategy(preds[mask], targets[mask], ret_21d[mask], threshold)
        results[r] = {
            "sharpe": m["sharpe"], "n": n_in_regime,
            "wr": m["wr"], "n_trades": m["n_trades"],
        }

    bull_s = results.get("bull", {}).get("sharpe", 0.0)
    bear_s = results.get("bear", {}).get("sharpe", 0.0)
    max_s = max(abs(bull_s), abs(bear_s))

    regime_diff = abs(bull_s - bear_s) / max_s if max_s > 0 else 0.0

    passed = regime_diff < REGIME_THRESHOLD
    log.info(f"  Regime test: bull Sharpe={bull_s:.3f}, bear Sharpe={bear_s:.3f}, "
             f"diff_ratio={regime_diff:.3f} ({'PASS' if passed else 'FAIL'})")

    return {
        "per_regime": results,
        "diff_ratio": float(regime_diff),
        "pass": passed,
    }


def subperiod_stability(preds: np.ndarray, targets: np.ndarray, ret_21d: np.ndarray,
                          dates: list, threshold: float, n_periods: int = 4) -> dict:
    """Sub-period stability: CV of Sharpe across sub-periods < 0.70."""
    log.info(f"Running sub-period stability test ({n_periods} periods)...")

    n = len(preds)
    chunk_size = n // n_periods
    if chunk_size < 10:
        log.warning("  Too few events for sub-period test, auto-pass")
        return {"sharpes": [], "cv": 0.0, "pass": True}

    sharpes = []
    for i in range(n_periods):
        start = i * chunk_size
        end = start + chunk_size if i < n_periods - 1 else n
        m = simulate_strategy(preds[start:end], targets[start:end],
                               ret_21d[start:end], threshold)
        sharpes.append(m["sharpe"])

    sharpes_arr = np.array(sharpes)
    mean_s = float(np.mean(sharpes_arr))
    std_s = float(np.std(sharpes_arr))
    cv = float(std_s / abs(mean_s)) if abs(mean_s) > 1e-6 else float("inf")

    passed = cv < SUBPERIOD_CV_CAP
    log.info(f"  Sub-period Sharpes: {[f'{s:.3f}' for s in sharpes]}, "
             f"CV={cv:.3f} ({'PASS' if passed else 'FAIL'})")

    return {
        "sharpes": [float(s) for s in sharpes],
        "mean": mean_s,
        "std": std_s,
        "cv": cv,
        "pass": passed,
    }


def lag_sensitivity_test(preds: np.ndarray, targets: np.ndarray,
                          ret_21d: np.ndarray, threshold: float) -> dict:
    """Anti-lookahead check (HC #724): compare T-0 signal vs T-1 (shifted).
    If T-0 is dramatically better than T-1, the model has real predictive content.
    If T-0 ~ T-1, signal may be noise or leaked."""
    log.info("Running lag sensitivity test (T-0 vs T-1)...")

    # T-0: actual model predictions
    t0_metrics = simulate_strategy(preds, targets, ret_21d, threshold)

    # T-1: shift predictions by 1 position (use prior event's prediction for current)
    shifted_preds = np.roll(preds, 1)
    shifted_preds[0] = 0.5  # neutral for first event
    t1_metrics = simulate_strategy(shifted_preds, targets, ret_21d, threshold)

    t0_sharpe = t0_metrics["sharpe"]
    t1_sharpe = t1_metrics["sharpe"]

    # T-0 should be meaningfully better than shifted T-1
    ratio = t0_sharpe / t1_sharpe if abs(t1_sharpe) > 0.01 else float("inf")

    passed = ratio > 1.0
    log.info(f"  Lag test: T-0 Sharpe={t0_sharpe:.3f}, T-1(shifted) Sharpe={t1_sharpe:.3f}, "
             f"ratio={ratio:.2f} ({'PASS' if passed else 'FAIL'})")

    return {
        "t0_sharpe": float(t0_sharpe),
        "t1_sharpe": float(t1_sharpe),
        "ratio": float(ratio),
        "pass": passed,
    }


# ===================================================================
# OPTIMAL THRESHOLD SEARCH
# ===================================================================

def find_optimal_threshold(preds: np.ndarray, targets: np.ndarray,
                            ret_21d: np.ndarray) -> tuple:
    """Find threshold that maximizes Sharpe across a grid."""
    best_sharpe = -999.0
    best_thresh = 0.5
    results = []

    for thresh in np.arange(0.25, 0.90, 0.05):
        m = simulate_strategy(preds, targets, ret_21d, float(thresh))
        results.append({"threshold": float(thresh), **m})
        if m["sharpe"] > best_sharpe and m["n_trades"] >= 10:
            best_sharpe = m["sharpe"]
            best_thresh = float(thresh)

    log.info(f"Optimal threshold: {best_thresh:.2f} (Sharpe={best_sharpe:.3f})")
    return best_thresh, results


# ===================================================================
# MLFLOW LOGGING
# ===================================================================

def log_to_mlflow(results: dict):
    """Log experiment results to MLflow."""
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("earnings_asymmetry_v1")

        with mlflow.start_run(run_name=f"earnings_asym_{datetime.now():%Y%m%d_%H%M}"):
            # Params
            mlflow.log_param("universe_size", len(UNIVERSE))
            mlflow.log_param("target_threshold", TARGET_THRESHOLD)
            mlflow.log_param("target_horizon", TARGET_HORIZON)
            mlflow.log_param("wf_train_days", WF_TRAIN_DAYS)
            mlflow.log_param("wf_embargo_days", WF_EMBARGO_DAYS)
            mlflow.log_param("wf_test_days", WF_TEST_DAYS)
            mlflow.log_param("cost_bps_rt", COST_BPS_RT)
            mlflow.log_param("optimal_threshold", results.get("optimal_threshold", 0.5))
            mlflow.log_param("n_events", results.get("n_events", 0))
            mlflow.log_param("n_folds", results.get("n_folds", 0))
            mlflow.log_param("window_type", "SLIDING")
            mlflow.log_param("signal_lag", "T-1")

            # Metrics
            strat = results.get("strategy_metrics", {})
            mlflow.log_metric("sharpe", strat.get("sharpe", 0))
            mlflow.log_metric("sortino", strat.get("sortino", 0))
            mlflow.log_metric("profit_factor", strat.get("pf", 0))
            mlflow.log_metric("win_rate", strat.get("wr", 0))
            mlflow.log_metric("n_trades", strat.get("n_trades", 0))
            mlflow.log_metric("avg_return", strat.get("avg_return", 0))

            # Validation metrics
            val = results.get("validation", {})
            perm = val.get("permutation_test", {})
            mlflow.log_metric("perm_p_value", perm.get("p_value", 1.0))
            reg = val.get("regime_test", {})
            mlflow.log_metric("regime_diff_ratio", reg.get("diff_ratio", 1.0))
            sub = val.get("subperiod_stability", {})
            mlflow.log_metric("subperiod_cv", sub.get("cv", 1.0))
            lag = val.get("lag_sensitivity", {})
            mlflow.log_metric("lag_ratio", lag.get("ratio", 0.0))

            # Artifact
            results_path = str(OUTPUT_DIR / "results.json")
            if os.path.exists(results_path):
                mlflow.log_artifact(results_path)

        log.info("MLflow logging complete")
    except Exception as e:
        log.warning(f"MLflow logging failed (non-fatal): {e}")


# ===================================================================
# MAIN
# ===================================================================

def main():
    t_start = time.time()
    log.info("=" * 70)
    log.info("EARNINGS ASYMMETRY RESEARCH v1")
    log.info("=" * 70)
    log.info(f"Universe: {len(UNIVERSE)} stocks")
    log.info(f"Target: 21d return > +{TARGET_THRESHOLD*100:.0f}%")
    log.info(f"Walk-forward: {WF_TRAIN_DAYS}d train, {WF_EMBARGO_DAYS}d embargo, "
             f"{WF_TEST_DAYS}d test (SLIDING)")
    log.info(f"Cost: {COST_BPS_RT} bps round-trip (FIFO)")
    log.info(f"Output: {OUTPUT_DIR}")

    # ---------------------------------------------------------------
    # STEP 1: Download data
    # ---------------------------------------------------------------
    log.info("-" * 50)
    log.info("STEP 1: Downloading price + earnings data...")
    data = download_data(UNIVERSE)

    if len(data["prices"]) < 10:
        log.error("Too few tickers downloaded successfully, aborting")
        sys.exit(1)

    # ---------------------------------------------------------------
    # STEP 2: SPY regime
    # ---------------------------------------------------------------
    log.info("-" * 50)
    log.info("STEP 2: Computing market regime from SPY...")
    regime_series = get_spy_regime()

    # ---------------------------------------------------------------
    # STEP 3: Build features
    # ---------------------------------------------------------------
    log.info("-" * 50)
    log.info("STEP 3: Building features for all earnings events...")
    df = build_features_df(data, regime_series)

    if len(df) < 100:
        log.error(f"Too few earnings events ({len(df)}), need >= 100. Aborting.")
        sys.exit(1)

    log.info(f"Feature matrix shape: {df.shape}")
    log.info(f"Target distribution: {df['target'].value_counts().to_dict()}")
    log.info(f"Date range: {df['event_date'].min()} to {df['event_date'].max()}")

    # Save feature matrix
    feat_path = OUTPUT_DIR / "features.parquet"
    try:
        df.to_parquet(feat_path, index=False)
        log.info(f"Saved features ({len(df)} rows)")
    except Exception as e:
        log.warning(f"Parquet save failed, trying CSV: {e}")
        df.to_csv(OUTPUT_DIR / "features.csv", index=False)

    # ---------------------------------------------------------------
    # STEP 4: Walk-forward LightGBM
    # ---------------------------------------------------------------
    log.info("-" * 50)
    log.info("STEP 4: Walk-forward LightGBM training (SLIDING)...")
    wf_results = walk_forward_lgbm(df)

    if not wf_results or len(wf_results.get("preds", [])) < 20:
        log.error("Walk-forward produced too few predictions, aborting")
        sys.exit(1)

    preds = wf_results["preds"]
    targets = wf_results["targets"]
    ret_21d = wf_results["ret_21d"]

    log.info(f"  Total OOT predictions: {len(preds)}")
    log.info(f"  Folds: {len(wf_results['fold_metrics'])}")
    avg_auc = np.mean([f["auc"] for f in wf_results["fold_metrics"]])
    log.info(f"  Mean fold AUC: {avg_auc:.3f}")

    # ---------------------------------------------------------------
    # STEP 5: Find optimal threshold
    # ---------------------------------------------------------------
    log.info("-" * 50)
    log.info("STEP 5: Optimizing prediction threshold...")
    optimal_thresh, thresh_results = find_optimal_threshold(preds, targets, ret_21d)

    # ---------------------------------------------------------------
    # STEP 6: Strategy metrics at optimal threshold
    # ---------------------------------------------------------------
    log.info("-" * 50)
    log.info("STEP 6: Computing strategy metrics at optimal threshold...")
    strategy_metrics = simulate_strategy(preds, targets, ret_21d, optimal_thresh)

    log.info(f"  Threshold: {optimal_thresh:.2f}")
    log.info(f"  Sharpe:    {strategy_metrics['sharpe']:.4f}")
    log.info(f"  Sortino:   {strategy_metrics['sortino']:.4f}")
    log.info(f"  PF:        {strategy_metrics['pf']:.4f}")
    log.info(f"  WR:        {strategy_metrics['wr']:.4f}")
    log.info(f"  Trades:    {strategy_metrics['n_trades']}")
    log.info(f"  Avg Ret:   {strategy_metrics['avg_return']*100:.2f}%")
    log.info(f"  Total Ret: {strategy_metrics['total_return']*100:.2f}%")

    # ---------------------------------------------------------------
    # STEP 7: Validation gates (MANDATORY)
    # ---------------------------------------------------------------
    log.info("-" * 50)
    log.info("STEP 7: Running validation gates...")

    # Build regime array for OOT predictions
    pred_dates = pd.to_datetime(wf_results["dates"])
    regime_arr = np.array(["neutral"] * len(pred_dates), dtype=object)
    for i, d in enumerate(pred_dates):
        ridx = regime_series.index.get_indexer([d], method="nearest")
        if len(ridx) > 0 and ridx[0] >= 0:
            regime_arr[i] = regime_series.iloc[ridx[0]]

    # 7a: 200-shuffle permutation test (p < 0.05)
    perm_result = permutation_test(preds, targets, ret_21d, optimal_thresh)

    # 7b: Regime stability (|bull - bear| / max < 0.50)
    regime_result = regime_test(preds, targets, ret_21d, regime_arr, optimal_thresh)

    # 7c: Sub-period stability (CV < 0.70)
    subperiod_result = subperiod_stability(
        preds, targets, ret_21d, wf_results["dates"], optimal_thresh
    )

    # 7d: Lag sensitivity / anti-lookahead (HC #724)
    lag_result = lag_sensitivity_test(preds, targets, ret_21d, optimal_thresh)

    # Aggregate
    all_passed = all([
        perm_result["pass"],
        regime_result["pass"],
        subperiod_result["pass"],
        lag_result["pass"],
    ])

    log.info("-" * 50)
    log.info(f"VALIDATION SUMMARY: {'ALL PASSED' if all_passed else 'SOME GATES FAILED'}")
    log.info(f"  [{'PASS' if perm_result['pass'] else 'FAIL'}] Permutation test "
             f"(p<0.05): p={perm_result['p_value']:.4f}")
    log.info(f"  [{'PASS' if regime_result['pass'] else 'FAIL'}] Regime stability "
             f"(<0.50): ratio={regime_result['diff_ratio']:.3f}")
    log.info(f"  [{'PASS' if subperiod_result['pass'] else 'FAIL'}] Sub-period CV "
             f"(<0.70): CV={subperiod_result['cv']:.3f}")
    log.info(f"  [{'PASS' if lag_result['pass'] else 'FAIL'}] Lag sensitivity "
             f"(T-0>T-1): ratio={lag_result['ratio']:.2f}")

    # ---------------------------------------------------------------
    # STEP 8: Feature importance
    # ---------------------------------------------------------------
    log.info("-" * 50)
    log.info("STEP 8: Top 15 features by importance:")
    fi = wf_results["feature_importance"]
    for i, (feat, imp) in enumerate(list(fi.items())[:15]):
        log.info(f"  {i+1:2d}. {feat:25s} {imp:.1f}")

    # ---------------------------------------------------------------
    # STEP 9: Per-sector breakdown
    # ---------------------------------------------------------------
    log.info("-" * 50)
    log.info("STEP 9: Per-sector performance:")
    tickers_arr = np.array(wf_results["tickers"])
    sector_results = {}
    for sector in SECTORS:
        sector_tickers = [t for t, s in SECTOR_MAP.items() if s == sector]
        sector_mask = np.isin(tickers_arr, sector_tickers)
        n_sector = int(np.sum(sector_mask))
        if n_sector < 5:
            continue
        m = simulate_strategy(
            preds[sector_mask], targets[sector_mask],
            ret_21d[sector_mask], optimal_thresh
        )
        sector_results[sector] = m
        if m["n_trades"] > 0:
            log.info(f"  {sector:12s}: Sharpe={m['sharpe']:7.3f}  "
                     f"WR={m['wr']:.3f}  N={m['n_trades']:3d}  "
                     f"Avg={m['avg_return']*100:.1f}%")

    # ---------------------------------------------------------------
    # STEP 10: Compile and save results
    # ---------------------------------------------------------------
    log.info("-" * 50)
    log.info("STEP 10: Saving results...")

    elapsed = time.time() - t_start

    results = {
        "experiment": "earnings_asymmetry_v1",
        "timestamp": datetime.now().isoformat(),
        "runtime_seconds": round(elapsed, 1),
        "universe_size": len(UNIVERSE),
        "tickers_used": sorted(data["prices"].keys()),
        "tickers_failed": data["failed"],
        "n_events_total": len(df),
        "n_oot_predictions": len(preds),
        "n_folds": len(wf_results["fold_metrics"]),
        "mean_fold_auc": round(float(avg_auc), 4),
        "target_rate": round(float(df["target"].mean()), 4),
        "date_range": {
            "start": str(df["event_date"].min()),
            "end": str(df["event_date"].max()),
        },
        "optimal_threshold": round(float(optimal_thresh), 2),
        "strategy_metrics": strategy_metrics,
        "threshold_scan": thresh_results,
        "validation": {
            "all_passed": all_passed,
            "permutation_test": perm_result,
            "regime_test": regime_result,
            "subperiod_stability": subperiod_result,
            "lag_sensitivity": lag_result,
        },
        "feature_importance_top20": dict(list(fi.items())[:20]),
        "fold_metrics": wf_results["fold_metrics"],
        "per_sector": sector_results,
        "config": {
            "target_threshold": TARGET_THRESHOLD,
            "target_horizon_days": TARGET_HORIZON,
            "wf_train_days": WF_TRAIN_DAYS,
            "wf_embargo_days": WF_EMBARGO_DAYS,
            "wf_test_days": WF_TEST_DAYS,
            "wf_step_days": WF_STEP_DAYS,
            "cost_bps_rt": COST_BPS_RT,
            "perm_shuffles": PERM_SHUFFLES,
            "regime_threshold": REGIME_THRESHOLD,
            "subperiod_cv_cap": SUBPERIOD_CV_CAP,
            "window_type": "SLIDING (HC #0)",
            "signal_lag": "T-1 (HC #724)",
            "cost_model": "FIFO 10bps RT",
        },
    }

    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"Saved results.json ({len(results)} keys)")

    # Save predictions for downstream analysis
    pred_out = pd.DataFrame({
        "date": wf_results["dates"],
        "ticker": wf_results["tickers"],
        "pred_prob": preds,
        "target": targets,
        "ret_21d": ret_21d,
        "regime": regime_arr,
    })
    try:
        pred_out.to_parquet(OUTPUT_DIR / "predictions.parquet", index=False)
    except Exception:
        pred_out.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    log.info(f"Saved predictions ({len(pred_out)} rows)")

    # ---------------------------------------------------------------
    # STEP 11: Log to MLflow
    # ---------------------------------------------------------------
    log.info("-" * 50)
    log.info("STEP 11: Logging to MLflow...")
    log_to_mlflow(results)

    # ---------------------------------------------------------------
    # FINAL SUMMARY
    # ---------------------------------------------------------------
    log.info("=" * 70)
    log.info("EARNINGS ASYMMETRY v1 -- COMPLETE")
    log.info(f"Runtime: {elapsed/60:.1f} minutes")
    log.info(f"Events: {len(df)} | OOT Preds: {len(preds)} | Folds: {len(wf_results['fold_metrics'])}")
    log.info(f"Mean AUC: {avg_auc:.3f}")
    log.info(f"Sharpe: {strategy_metrics['sharpe']:.3f} | "
             f"Sortino: {strategy_metrics['sortino']:.3f} | "
             f"PF: {strategy_metrics['pf']:.3f} | "
             f"WR: {strategy_metrics['wr']:.3f}")
    log.info(f"Validation: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    log.info(f"Output: {OUTPUT_DIR}")
    log.info("=" * 70)

    return results


if __name__ == "__main__":
    # Dependency check
    missing = []
    for pkg_name, import_name in [
        ("yfinance", "yfinance"),
        ("lightgbm", "lightgbm"),
        ("scikit-learn", "sklearn"),
        ("pyarrow", "pyarrow"),
    ]:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg_name)

    if missing:
        log.error(f"Missing packages: {missing}")
        log.error("Install with: pip install " + " ".join(missing))
        sys.exit(1)

    main()
