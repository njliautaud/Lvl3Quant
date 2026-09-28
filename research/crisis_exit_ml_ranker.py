#!/usr/bin/env python3
"""
Crisis Exit + ML Asymmetric Stock Ranker — Combined Strategy Research (HC #724)

Combines two validated findings:
1. Crisis Exit Signal: VIX drops below 30 after being above 30 → buy aggressively (79% WR at 3m/12m)
2. ML Asymmetric Stock Ranker: LightGBM picks top 5 stocks when asymmetric filter fires (perm p=0.000)

Strategy variants:
A. ML ranker alone (baseline)
B. Crisis exit + SPY (buy SPY when VIX crosses below 30, hold 3 months)
C. Crisis exit + ML top 5 (buy ML-ranked top 5 when VIX crosses below 30, hold 3 months)
D. Combined: ML ranker with asymmetric filter + 2x allocation when crisis exit fires
E. Regime-aware ML: add crisis_exit as a feature to the ML model

All signals T-1 only (no same-day lookahead).
Walk-forward: train LightGBM on rolling 504d window, predict next 21d.
Transaction costs: 25 bps per leg.
Run period: 2006-2026.
"""

import os
import sys
import json
import time
import warnings
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats

warnings.filterwarnings("ignore")

# ── Setup ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("/home/nick/Lvl3Quant/output/crisis_exit_ml_ranker")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "run.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── MLflow ───────────────────────────────────────────────────────────────────
try:
    import mlflow
    mlflow.set_tracking_uri("http://jupiter:5000")
    mlflow.set_experiment("crisis_exit_ml_ranker")
    MLFLOW_OK = True
    log.info("MLflow connected: http://jupiter:5000")
except Exception as e:
    MLFLOW_OK = False
    log.warning(f"MLflow unavailable: {e}")

# ── Constants ────────────────────────────────────────────────────────────────
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

TRAIN_DAYS = 504
TEST_DAYS = 21
COST_BPS = 25
START_YEAR = 2006  # Longer history to capture multiple crises
CRISIS_HOLD_MONTHS = 3  # Hold period after crisis exit signal (in months ~ 63 trading days)
CRISIS_HOLD_DAYS = 63
VIX_THRESHOLD = 30


# ── Data Download ────────────────────────────────────────────────────────────
def download_data():
    """Download all required price data via yfinance."""
    import yfinance as yf

    cache_file = OUTPUT_DIR / "price_cache.parquet"
    if cache_file.exists():
        log.info("Loading cached price data")
        df = pd.read_parquet(cache_file)
        if df.index.max() >= pd.Timestamp.now() - pd.Timedelta(days=7):
            return df

    all_tickers = UNIVERSE + SECTOR_ETFS + ["^VIX", "^VIX3M", "SPY"]
    log.info(f"Downloading {len(all_tickers)} tickers from yfinance...")

    data = yf.download(
        all_tickers,
        start=f"{START_YEAR - 3}-01-01",  # Extra history for lookback
        end=datetime.now().strftime("%Y-%m-%d"),
        auto_adjust=True,
        threads=True,
    )

    close = data["Close"] if "Close" in data.columns.get_level_values(0) else data["Adj Close"]
    volume = data["Volume"]

    combined = pd.concat({"Close": close, "Volume": volume}, axis=1)
    combined.to_parquet(cache_file)
    log.info(f"Data shape: {combined.shape}, range: {combined.index[0]} to {combined.index[-1]}")
    return combined


# ── Crisis Exit Signal ───────────────────────────────────────────────────────
def compute_crisis_exit_signals(data):
    """
    Detect VIX crossing below 30 after being above 30.
    Returns a Series of dates where the signal fires (T-1: using yesterday's VIX).
    """
    log.info("Computing crisis exit signals...")

    vix_col = ("Close", "^VIX") if ("Close", "^VIX") in data.columns else None
    if vix_col is None:
        log.error("VIX data not found!")
        return pd.Series(dtype=bool, index=data.index)

    vix = data[vix_col].copy()
    vix = vix.ffill()

    # T-1: use YESTERDAY's VIX to determine today's signal
    # Signal fires when VIX_yesterday < 30 AND VIX_day_before_yesterday >= 30
    vix_lag1 = vix.shift(1)  # yesterday
    vix_lag2 = vix.shift(2)  # day before yesterday

    # Crisis exit = VIX crossed below 30 (was >= 30, now < 30)
    crisis_exit = (vix_lag1 < VIX_THRESHOLD) & (vix_lag2 >= VIX_THRESHOLD)

    n_signals = crisis_exit.sum()
    signal_dates = data.index[crisis_exit]
    log.info(f"Crisis exit signals: {n_signals} in {data.index[0].year}-{data.index[-1].year}")
    for dt in signal_dates:
        log.info(f"  {dt.strftime('%Y-%m-%d')}: VIX {vix_lag2.loc[dt]:.1f} -> {vix_lag1.loc[dt]:.1f}")

    return crisis_exit


def compute_crisis_active(crisis_exit_signals, dates):
    """
    Return a boolean Series: True on any date within CRISIS_HOLD_DAYS of a crisis exit signal.
    """
    crisis_active = pd.Series(False, index=dates)
    signal_dates = dates[crisis_exit_signals]

    for sig_dt in signal_dates:
        loc = dates.get_loc(sig_dt)
        end_loc = min(loc + CRISIS_HOLD_DAYS, len(dates))
        crisis_active.iloc[loc:end_loc] = True

    return crisis_active


# ── Feature Engineering ──────────────────────────────────────────────────────
def compute_features(data, date_idx, ticker):
    """Compute all features for a single stock at a single date (using T-1 data)."""
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

    # 1. RSI(14)
    deltas = np.diff(prices[-15:])
    gains = np.maximum(deltas, 0).mean()
    losses = np.abs(np.minimum(deltas, 0)).mean()
    feats["rsi_14"] = 100 - 100 / (1 + gains / (losses + 1e-10))

    # 2. Distance from 52-week high
    high_252 = np.nanmax(prices[-252:])
    feats["dist_52w_high"] = (p / high_252 - 1) * 100

    # 3. Realized vol (20d, 63d)
    rets = np.diff(np.log(prices[-64:] + 1e-10))
    feats["vol_20d"] = np.std(rets[-20:]) * np.sqrt(252) if len(rets) >= 20 else np.nan
    feats["vol_63d"] = np.std(rets[-63:]) * np.sqrt(252) if len(rets) >= 63 else np.nan

    # 4. Volume surge
    vol_5d = np.nanmean(volumes[-5:])
    vol_63d = np.nanmean(volumes[-63:])
    feats["vol_surge"] = vol_5d / (vol_63d + 1e-10)

    # 5. Momentum (1m, 3m, 6m, 12m)
    feats["mom_1m"] = p / prices[-21] - 1 if len(prices) >= 21 else np.nan
    feats["mom_3m"] = p / prices[-63] - 1 if len(prices) >= 63 else np.nan
    feats["mom_6m"] = p / prices[-126] - 1 if len(prices) >= 126 else np.nan
    feats["mom_12m"] = p / prices[-252] - 1 if len(prices) >= 252 else np.nan

    # 6. Mean reversion z-score
    mean_63 = np.mean(prices[-63:])
    std_63 = np.std(prices[-63:])
    feats["mr_zscore"] = (p - mean_63) / (std_63 + 1e-10)

    # 7. Price vs 200d SMA
    if len(prices) >= 200:
        sma_200 = np.mean(prices[-200:])
        feats["price_vs_sma200"] = p / sma_200 - 1
    else:
        feats["price_vs_sma200"] = np.nan

    # 8. Trailing max drawdown from 52-week high
    rolling_max = np.maximum.accumulate(prices[-252:])
    drawdowns = prices[-252:] / rolling_max - 1
    feats["max_dd_52w"] = np.min(drawdowns)

    # 9. Vol ratio (short/long term vol)
    if feats["vol_63d"] and feats["vol_63d"] > 0:
        feats["vol_ratio"] = feats["vol_20d"] / feats["vol_63d"]
    else:
        feats["vol_ratio"] = np.nan

    # 10. Momentum acceleration
    if not np.isnan(feats.get("mom_1m", np.nan)) and not np.isnan(feats.get("mom_3m", np.nan)):
        feats["mom_accel"] = feats["mom_1m"] - feats["mom_3m"] / 3
    else:
        feats["mom_accel"] = np.nan

    # 11. Beta to SPY (63d rolling)
    spy_col = ("Close", "SPY")
    if spy_col in data.columns and len(prices) >= 63:
        spy_prices = data[spy_col].iloc[:loc].values.astype(float)
        if len(spy_prices) >= 63:
            stock_rets_63 = np.diff(np.log(prices[-64:] + 1e-10))
            spy_rets_63 = np.diff(np.log(spy_prices[-64:] + 1e-10))
            if len(stock_rets_63) >= 63 and len(spy_rets_63) >= 63:
                cov = np.cov(stock_rets_63[-63:], spy_rets_63[-63:])
                feats["beta_63d"] = cov[0, 1] / (cov[1, 1] + 1e-10)
            else:
                feats["beta_63d"] = np.nan
        else:
            feats["beta_63d"] = np.nan
    else:
        feats["beta_63d"] = np.nan

    return feats


def build_feature_matrix(data, crisis_exit_signals, crisis_active):
    """Build the full feature matrix with walk-forward dates."""
    log.info("Building feature matrix...")

    dates = data.index

    # VIX and VIX3M
    vix_col = ("Close", "^VIX") if ("Close", "^VIX") in data.columns else None
    vix3m_col = ("Close", "^VIX3M") if ("Close", "^VIX3M") in data.columns else None

    # Sector ETF returns
    sector_rets_1m = {}
    for etf in SECTOR_ETFS:
        col = ("Close", etf)
        if col in data.columns:
            sector_rets_1m[etf] = data[col].pct_change(21)

    # Breadth
    breadth = pd.Series(0.0, index=dates)
    for etf in SECTOR_ETFS:
        col = ("Close", etf)
        if col in data.columns:
            sma50 = data[col].rolling(50).mean()
            breadth += (data[col] > sma50).astype(float)
    breadth = breadth / len(SECTOR_ETFS)

    # VIX features for crisis context
    vix_series = data[vix_col] if vix_col else pd.Series(np.nan, index=dates)
    vix_sma20 = vix_series.rolling(20).mean()
    vix_pctile_252 = vix_series.rolling(252).apply(lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100, raw=False)

    rows = []
    start_idx = max(504, 252)
    rebal_dates = dates[start_idx::TEST_DAYS]
    log.info(f"Total rebalance dates: {len(rebal_dates)} from {rebal_dates[0]} to {rebal_dates[-1]}")

    for i, dt in enumerate(rebal_dates):
        if i % 50 == 0:
            log.info(f"  Processing date {i}/{len(rebal_dates)}: {dt.strftime('%Y-%m-%d')}")

        loc = data.index.get_loc(dt)

        if loc + TEST_DAYS >= len(dates):
            continue

        # Market-level features (all T-1: use loc-1 for VIX etc.)
        vix_val = vix_series.iloc[loc - 1] if loc > 0 else np.nan  # T-1
        vix3m_val = data[vix3m_col].iloc[loc - 1] if vix3m_col and loc > 0 else np.nan
        vix_term = vix_val / (vix3m_val + 1e-10) if not np.isnan(vix_val) and not np.isnan(vix3m_val) else np.nan
        breadth_val = breadth.iloc[loc - 1] if loc > 0 else np.nan

        # Crisis-specific features (T-1)
        is_crisis_active = bool(crisis_active.iloc[loc - 1]) if loc > 0 else False
        is_crisis_exit_today = bool(crisis_exit_signals.iloc[loc - 1]) if loc > 0 else False

        # Days since last crisis exit
        past_signals = crisis_exit_signals.iloc[:loc]
        if past_signals.any():
            last_signal_loc = past_signals.values.nonzero()[0][-1]
            days_since_crisis_exit = loc - last_signal_loc
        else:
            days_since_crisis_exit = 9999

        # VIX context features
        vix_vs_sma20 = vix_val / (vix_sma20.iloc[loc - 1] + 1e-10) if loc > 0 else np.nan
        vix_pctile = vix_pctile_252.iloc[loc - 1] if loc > 0 else np.nan

        # Cross-sectional returns for ranking
        cs_rets = {}
        for ticker in UNIVERSE:
            col = ("Close", ticker)
            if col in data.columns:
                p_now = data[col].iloc[loc - 1]  # T-1
                p_prev = data[col].iloc[loc - 22] if loc >= 22 else np.nan
                if not np.isnan(p_now) and not np.isnan(p_prev) and p_prev > 0:
                    cs_rets[ticker] = p_now / p_prev - 1

        if cs_rets:
            sorted_tickers = sorted(cs_rets.keys(), key=lambda t: cs_rets[t])
            cs_rank = {t: rank / len(sorted_tickers) for rank, t in enumerate(sorted_tickers)}
        else:
            cs_rank = {}

        for ticker in UNIVERSE:
            feats = compute_features(data, dt, ticker)
            if feats is None:
                continue

            # Market-level features
            feats["vix"] = vix_val
            feats["vix_term_structure"] = vix_term
            feats["breadth"] = breadth_val

            # Sector ETF return
            sector = STOCK_SECTOR.get(ticker, "XLK")
            feats["sector_ret_1m"] = sector_rets_1m.get(sector, pd.Series(dtype=float)).iloc[loc - 1] if sector in sector_rets_1m and loc > 0 else np.nan

            # Cross-sectional rank
            feats["cs_rank_1m"] = cs_rank.get(ticker, np.nan)

            # Crisis features (for Strategy E — regime-aware ML)
            feats["crisis_exit_active"] = int(is_crisis_active)
            feats["crisis_exit_today"] = int(is_crisis_exit_today)
            feats["days_since_crisis_exit"] = days_since_crisis_exit
            feats["vix_vs_sma20"] = vix_vs_sma20
            feats["vix_pctile_252"] = vix_pctile
            feats["vix_above_30"] = int(vix_val >= 30) if not np.isnan(vix_val) else 0

            # Forward return (target)
            fwd_col = ("Close", ticker)
            p_now = data[fwd_col].iloc[loc]
            p_fwd = data[fwd_col].iloc[loc + TEST_DAYS]
            fwd_ret = p_fwd / p_now - 1 if not np.isnan(p_now) and not np.isnan(p_fwd) and p_now > 0 else np.nan

            feats["ticker"] = ticker
            feats["date"] = dt
            feats["fwd_ret_1m"] = fwd_ret
            rows.append(feats)

    df = pd.DataFrame(rows)
    log.info(f"Feature matrix: {df.shape[0]} rows, {df.shape[1]} cols")

    # Cross-sectional percentile ranks
    for col in ["vol_20d", "vol_63d"]:
        df[f"{col}_pctrank"] = df.groupby("date")[col].rank(pct=True)

    # Asymmetric filter: high vol + negative momentum + volume surge
    df["asymmetric_filter"] = (
        (df["vol_20d_pctrank"] >= 0.80) &
        (df["mom_1m"] < 0) &
        (df["vol_surge"] > 1.5)
    ).astype(int)

    return df


# ── Walk-Forward Training ────────────────────────────────────────────────────
# Base feature cols (without crisis features — used for Strategy A)
BASE_FEATURE_COLS = [
    "rsi_14", "dist_52w_high", "vol_20d", "vol_63d", "vol_surge",
    "mom_1m", "mom_3m", "mom_6m", "mom_12m", "mr_zscore", "price_vs_sma200",
    "max_dd_52w", "vol_ratio", "mom_accel", "beta_63d",
    "vix", "vix_term_structure", "breadth", "sector_ret_1m", "cs_rank_1m",
    "vol_20d_pctrank", "vol_63d_pctrank",
]

# Extended feature cols (with crisis features — used for Strategy E)
EXTENDED_FEATURE_COLS = BASE_FEATURE_COLS + [
    "crisis_exit_active", "crisis_exit_today", "days_since_crisis_exit",
    "vix_vs_sma20", "vix_pctile_252", "vix_above_30",
]

LGB_PARAMS = {
    "objective": "regression",
    "metric": "mse",
    "device": "gpu",
    "gpu_use_dp": False,
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


def walk_forward_train(df, feature_cols, model_name="base"):
    """Walk-forward LightGBM training with GPU acceleration."""
    log.info(f"Walk-forward training [{model_name}] with {len(feature_cols)} features...")

    unique_dates = sorted(df["date"].unique())
    log.info(f"Unique rebalance dates: {len(unique_dates)}")

    train_periods = TRAIN_DAYS // TEST_DAYS  # 24 months

    all_predictions = []
    fold_metrics = []
    last_model = None

    for i in range(train_periods, len(unique_dates)):
        test_date = unique_dates[i]
        train_dates = unique_dates[i - train_periods:i]

        train_df = df[df["date"].isin(train_dates)].copy()
        test_df = df[df["date"] == test_date].copy()

        if len(test_df) < 10 or len(train_df) < 100:
            continue

        X_train = train_df[feature_cols].values.astype(np.float32)
        y_train = train_df["fwd_ret_1m"].values.astype(np.float32)
        X_test = test_df[feature_cols].values.astype(np.float32)
        y_test = test_df["fwd_ret_1m"].values.astype(np.float32)

        valid_train = ~np.isnan(y_train) & ~np.any(np.isnan(X_train), axis=1)
        valid_test = ~np.isnan(y_test) & ~np.any(np.isnan(X_test), axis=1)

        if valid_train.sum() < 50 or valid_test.sum() < 5:
            continue

        X_train, y_train = X_train[valid_train], y_train[valid_train]
        X_test_v, y_test_v = X_test[valid_test], y_test[valid_test]

        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_test_v, label=y_test_v, reference=dtrain)

        model = lgb.train(
            LGB_PARAMS,
            dtrain,
            num_boost_round=500,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        last_model = model

        preds = model.predict(X_test_v)

        if len(preds) >= 5:
            rank_ic = stats.spearmanr(preds, y_test_v)[0]
        else:
            rank_ic = np.nan

        test_subset = test_df[valid_test].copy()
        test_subset["prediction"] = preds
        test_subset["fold"] = i
        all_predictions.append(test_subset)

        fold_metrics.append({
            "fold": i,
            "date": str(test_date)[:10],
            "n_stocks": len(preds),
            "rank_ic": rank_ic,
            "mse": float(np.mean((preds - y_test_v) ** 2)),
        })

        if i % 20 == 0:
            recent_ics = [m["rank_ic"] for m in fold_metrics[-20:] if not np.isnan(m["rank_ic"])]
            avg_ic = np.mean(recent_ics) if recent_ics else 0
            log.info(f"  [{model_name}] Fold {i}/{len(unique_dates)}, date={str(test_date)[:10]}, "
                     f"Rank IC={rank_ic:.4f}, Avg IC(20)={avg_ic:.4f}")

    importance = dict(zip(feature_cols, last_model.feature_importance(importance_type="gain"))) if last_model else {}

    all_preds_df = pd.concat(all_predictions, ignore_index=True)
    log.info(f"[{model_name}] Walk-forward complete: {len(fold_metrics)} folds, {len(all_preds_df)} predictions")

    return all_preds_df, fold_metrics, importance


# ── Portfolio Construction ───────────────────────────────────────────────────
def build_all_strategy_portfolios(preds_base_df, preds_extended_df, data, crisis_exit_signals, crisis_active):
    """Build all 5 strategy variants and compute returns."""
    log.info("Building strategy portfolios...")

    cost_per_trade = COST_BPS / 10000

    spy_col = ("Close", "SPY")
    dates = data.index

    # Get rebalance dates from predictions
    rebal_dates_base = sorted(preds_base_df["date"].unique())
    rebal_dates_ext = sorted(preds_extended_df["date"].unique())

    results = {}

    # ── Strategy A: ML ranker alone (baseline, top 5) ────────────────────────
    log.info("Strategy A: ML ranker alone (top 5)...")
    a_returns, a_dates, a_holdings = [], [], []
    for dt in rebal_dates_base:
        group = preds_base_df[preds_base_df["date"] == dt].dropna(subset=["fwd_ret_1m", "prediction"])
        if len(group) < 5:
            a_returns.append(0.0)
            a_dates.append(dt)
            a_holdings.append([])
            continue
        top5 = group.nlargest(5, "prediction")
        port_ret = top5["fwd_ret_1m"].mean() - 5 * 2 * cost_per_trade
        a_returns.append(port_ret)
        a_dates.append(dt)
        a_holdings.append(top5["ticker"].tolist())
    results["A_ml_ranker"] = {"returns": np.array(a_returns), "dates": a_dates, "holdings": a_holdings}

    # ── Strategy B: Crisis exit + SPY ────────────────────────────────────────
    log.info("Strategy B: Crisis exit + SPY...")
    b_returns, b_dates = [], []
    # For each rebalance date, check if within crisis exit hold period
    for dt in rebal_dates_base:
        loc = dates.get_loc(dt)
        if loc + TEST_DAYS >= len(dates):
            continue

        # Check if crisis is active (using T-1)
        is_active = bool(crisis_active.iloc[loc - 1]) if loc > 0 else False

        if is_active and spy_col in data.columns:
            p0 = data[spy_col].iloc[loc]
            p1 = data[spy_col].iloc[loc + TEST_DAYS]
            ret = (p1 / p0 - 1) - 2 * cost_per_trade  # Single position
            b_returns.append(ret)
        else:
            b_returns.append(0.0)  # Cash when no crisis exit
        b_dates.append(dt)
    results["B_crisis_spy"] = {"returns": np.array(b_returns), "dates": b_dates, "holdings": []}

    # ── Strategy C: Crisis exit + ML top 5 ───────────────────────────────────
    log.info("Strategy C: Crisis exit + ML top 5...")
    c_returns, c_dates, c_holdings = [], [], []
    for dt in rebal_dates_base:
        loc = dates.get_loc(dt)
        if loc + TEST_DAYS >= len(dates):
            continue

        is_active = bool(crisis_active.iloc[loc - 1]) if loc > 0 else False

        if is_active:
            group = preds_base_df[preds_base_df["date"] == dt].dropna(subset=["fwd_ret_1m", "prediction"])
            if len(group) >= 5:
                top5 = group.nlargest(5, "prediction")
                port_ret = top5["fwd_ret_1m"].mean() - 5 * 2 * cost_per_trade
                c_returns.append(port_ret)
                c_holdings.append(top5["ticker"].tolist())
            else:
                c_returns.append(0.0)
                c_holdings.append([])
        else:
            c_returns.append(0.0)
            c_holdings.append([])
        c_dates.append(dt)
    results["C_crisis_ml5"] = {"returns": np.array(c_returns), "dates": c_dates, "holdings": c_holdings}

    # ── Strategy D: ML ranker (asymmetric filter) + 2x when crisis fires ─────
    log.info("Strategy D: ML asymmetric filter + 2x crisis boost...")
    d_returns, d_dates, d_holdings = [], [], []
    for dt in rebal_dates_base:
        loc = dates.get_loc(dt)
        if loc + TEST_DAYS >= len(dates):
            continue

        group = preds_base_df[preds_base_df["date"] == dt].dropna(subset=["fwd_ret_1m", "prediction"])
        is_active = bool(crisis_active.iloc[loc - 1]) if loc > 0 else False

        # Filter: only stocks passing asymmetric filter
        filtered = group[group["asymmetric_filter"] == 1]

        if len(filtered) >= 1:
            n_picks = min(5, len(filtered))
            top = filtered.nlargest(n_picks, "prediction")
            port_ret = top["fwd_ret_1m"].mean() - n_picks * 2 * cost_per_trade

            # 2x allocation during crisis exit
            if is_active:
                port_ret *= 2.0
                # Extra cost for the additional position sizing
                port_ret -= n_picks * 2 * cost_per_trade

            d_returns.append(port_ret)
            d_holdings.append(top["ticker"].tolist())
        else:
            d_returns.append(0.0)
            d_holdings.append([])
        d_dates.append(dt)
    results["D_asymmetric_crisis2x"] = {"returns": np.array(d_returns), "dates": d_dates, "holdings": d_holdings}

    # ── Strategy E: Regime-aware ML (crisis features in model) ───────────────
    log.info("Strategy E: Regime-aware ML (crisis features in model)...")
    e_returns, e_dates, e_holdings = [], [], []
    for dt in rebal_dates_ext:
        group = preds_extended_df[preds_extended_df["date"] == dt].dropna(subset=["fwd_ret_1m", "prediction"])
        if len(group) < 5:
            e_returns.append(0.0)
            e_dates.append(dt)
            e_holdings.append([])
            continue
        top5 = group.nlargest(5, "prediction")
        port_ret = top5["fwd_ret_1m"].mean() - 5 * 2 * cost_per_trade
        e_returns.append(port_ret)
        e_dates.append(dt)
        e_holdings.append(top5["ticker"].tolist())
    results["E_regime_aware_ml"] = {"returns": np.array(e_returns), "dates": e_dates, "holdings": e_holdings}

    # ── Benchmarks ───────────────────────────────────────────────────────────
    # SPY buy & hold (monthly returns)
    log.info("Computing SPY benchmark...")
    spy_returns, spy_dates = [], []
    for dt in rebal_dates_base:
        loc = dates.get_loc(dt)
        if loc + TEST_DAYS < len(dates) and spy_col in data.columns:
            p0 = data[spy_col].iloc[loc]
            p1 = data[spy_col].iloc[loc + TEST_DAYS]
            spy_returns.append(p1 / p0 - 1 if p0 > 0 else 0)
        else:
            spy_returns.append(0)
        spy_dates.append(dt)
    results["SPY_benchmark"] = {"returns": np.array(spy_returns), "dates": spy_dates, "holdings": []}

    # Equal weight all 50 stocks
    log.info("Computing equal-weight 50 benchmark...")
    ew_returns, ew_dates = [], []
    for dt in rebal_dates_base:
        group = preds_base_df[preds_base_df["date"] == dt].dropna(subset=["fwd_ret_1m"])
        if len(group) > 0:
            ew_returns.append(group["fwd_ret_1m"].mean() - len(group) * 2 * cost_per_trade)
        else:
            ew_returns.append(0.0)
        ew_dates.append(dt)
    results["EW50_benchmark"] = {"returns": np.array(ew_returns), "dates": ew_dates, "holdings": []}

    return results


# ── Performance Metrics ──────────────────────────────────────────────────────
def compute_metrics(returns, name="Strategy"):
    """Compute comprehensive performance metrics."""
    returns = np.array(returns)
    returns = returns[~np.isnan(returns)]

    if len(returns) < 2:
        return {"name": name, "error": "insufficient data"}

    ann_ret = np.mean(returns) * 12
    ann_vol = np.std(returns) * np.sqrt(12)
    sharpe = ann_ret / (ann_vol + 1e-10)

    downside = returns[returns < 0]
    downside_vol = np.std(downside) * np.sqrt(12) if len(downside) > 0 else 1e-10
    sortino = ann_ret / (downside_vol + 1e-10)

    cum = np.cumprod(1 + returns)
    rolling_max = np.maximum.accumulate(cum)
    drawdowns = cum / rolling_max - 1
    max_dd = np.min(drawdowns)

    wr = np.mean(returns > 0)

    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    pf = gains / (losses + 1e-10)

    calmar = ann_ret / (abs(max_dd) + 1e-10)

    # Active months (non-zero returns) for strategies that sit in cash
    active_months = np.sum(returns != 0)

    return {
        "name": name,
        "ann_return": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_dd": float(max_dd),
        "win_rate": float(wr),
        "profit_factor": float(pf),
        "calmar": float(calmar),
        "n_months": int(len(returns)),
        "active_months": int(active_months),
        "total_return": float(np.prod(1 + returns) - 1),
    }


# ── Validation Tests ─────────────────────────────────────────────────────────
def permutation_test(preds_df, n_perms=200):
    """Test if signal ranking is statistically significant."""
    log.info(f"Running permutation test ({n_perms} permutations)...")

    actual_ics = []
    for dt, group in preds_df.groupby("date"):
        group = group.dropna(subset=["fwd_ret_1m", "prediction"])
        if len(group) >= 5:
            ic = stats.spearmanr(group["prediction"], group["fwd_ret_1m"])[0]
            if not np.isnan(ic):
                actual_ics.append(ic)

    actual_mean_ic = np.mean(actual_ics)

    perm_mean_ics = []
    rng = np.random.RandomState(42)
    for p in range(n_perms):
        perm_ics = []
        for dt, group in preds_df.groupby("date"):
            group = group.dropna(subset=["fwd_ret_1m", "prediction"])
            if len(group) >= 5:
                shuffled = rng.permutation(group["fwd_ret_1m"].values)
                ic = stats.spearmanr(group["prediction"], shuffled)[0]
                if not np.isnan(ic):
                    perm_ics.append(ic)
        perm_mean_ics.append(np.mean(perm_ics))

    p_value = np.mean(np.array(perm_mean_ics) >= actual_mean_ic)

    log.info(f"  Actual Mean Rank IC: {actual_mean_ic:.4f}")
    log.info(f"  Permutation p-value: {p_value:.4f}")

    return {
        "actual_mean_ic": float(actual_mean_ic),
        "p_value": float(p_value),
        "perm_mean_ic": float(np.mean(perm_mean_ics)),
        "perm_std_ic": float(np.std(perm_mean_ics)),
        "significant_5pct": bool(p_value < 0.05),
    }


def regime_test(portfolios, data):
    """Test performance in different market regimes (bull vs bear)."""
    log.info("Running regime test...")

    spy_col = ("Close", "SPY")
    if spy_col not in data.columns:
        return {"error": "SPY data not available"}

    spy_prices = data[spy_col]
    spy_sma200 = spy_prices.rolling(200).mean()

    results = {}
    for strat_name, port in portfolios.items():
        bull_rets, bear_rets = [], []
        for j, dt in enumerate(port["dates"]):
            try:
                is_bull = spy_prices.loc[dt] > spy_sma200.loc[dt]
            except (KeyError, TypeError):
                continue
            if is_bull:
                bull_rets.append(port["returns"][j])
            else:
                bear_rets.append(port["returns"][j])

        bull_sharpe = np.mean(bull_rets) * 12 / (np.std(bull_rets) * np.sqrt(12) + 1e-10) if len(bull_rets) > 1 else 0
        bear_sharpe = np.mean(bear_rets) * 12 / (np.std(bear_rets) * np.sqrt(12) + 1e-10) if len(bear_rets) > 1 else 0

        gap = abs(bull_sharpe - bear_sharpe) / (max(abs(bull_sharpe), abs(bear_sharpe)) + 1e-10)

        results[strat_name] = {
            "bull_sharpe": float(bull_sharpe),
            "bear_sharpe": float(bear_sharpe),
            "bull_n": len(bull_rets),
            "bear_n": len(bear_rets),
            "regime_gap": float(gap),
            "passes_regime_gate": bool(gap < 0.50),
        }
        log.info(f"  {strat_name}: bull_sharpe={bull_sharpe:.2f}, bear_sharpe={bear_sharpe:.2f}, gap={gap:.2f} {'PASS' if gap < 0.50 else 'FAIL'}")

    return results


def subperiod_stability(returns, dates, name="Strategy"):
    """Test return stability across sub-periods."""
    log.info(f"Sub-period stability for {name}...")

    n = len(returns)
    if n < 6:
        return {"error": "insufficient data"}

    periods = {
        "early": (0, n // 3),
        "middle": (n // 3, 2 * n // 3),
        "late": (2 * n // 3, n),
    }

    sub_sharpes = []
    results = {}
    for pname, (s, e) in periods.items():
        sub = returns[s:e]
        sub_ann_ret = np.mean(sub) * 12
        sub_ann_vol = np.std(sub) * np.sqrt(12)
        sub_sharpe = sub_ann_ret / (sub_ann_vol + 1e-10)
        sub_sharpes.append(sub_sharpe)
        results[pname] = {
            "sharpe": float(sub_sharpe),
            "ann_return": float(sub_ann_ret),
            "n": e - s,
        }

    cv = np.std(sub_sharpes) / (np.mean(np.abs(sub_sharpes)) + 1e-10) if sub_sharpes else 999
    results["cv_sharpe"] = float(cv)
    results["passes_stability_gate"] = bool(cv < 0.70)

    log.info(f"  CV of sub-period Sharpe: {cv:.2f} {'PASS' if cv < 0.70 else 'FAIL'}")

    return results


def lag_sensitivity_test(preds_df):
    """Test T-0 vs T-1 signal sensitivity."""
    log.info("Running lag sensitivity test...")
    ics_by_shift = {}
    dates = sorted(preds_df["date"].unique())

    for shift_name, shift_val in [("T+0 (actual)", 0), ("T+1 (1mo lag)", 1)]:
        shifted_ics = []
        for i, dt in enumerate(dates):
            if i + shift_val >= len(dates):
                break
            current = preds_df[preds_df["date"] == dt].dropna(subset=["prediction"])
            shifted_dt = dates[min(i + shift_val, len(dates) - 1)]
            shifted = preds_df[preds_df["date"] == shifted_dt].dropna(subset=["fwd_ret_1m"])

            merged = current[["ticker", "prediction"]].merge(
                shifted[["ticker", "fwd_ret_1m"]], on="ticker"
            )
            if len(merged) >= 5:
                ic = stats.spearmanr(merged["prediction"], merged["fwd_ret_1m"])[0]
                if not np.isnan(ic):
                    shifted_ics.append(ic)

        ics_by_shift[shift_name] = float(np.mean(shifted_ics)) if shifted_ics else 0.0
        log.info(f"  {shift_name}: Mean IC = {ics_by_shift[shift_name]:.4f}")

    # Check for lookahead: T-0 IC should NOT be dramatically higher than T+1
    t0_ic = ics_by_shift.get("T+0 (actual)", 0)
    t1_ic = ics_by_shift.get("T+1 (1mo lag)", 0)
    ratio = t0_ic / (t1_ic + 1e-10) if t1_ic != 0 else 999
    ics_by_shift["t0_t1_ratio"] = float(ratio)
    ics_by_shift["likely_lookahead"] = bool(ratio > 3.0)

    return ics_by_shift


# ── Crisis Exit Analysis ────────────────────────────────────────────────────
def analyze_crisis_exits(data, crisis_exit_signals):
    """Detailed analysis of crisis exit signal performance."""
    log.info("Analyzing crisis exit signals...")

    spy_col = ("Close", "SPY")
    if spy_col not in data.columns:
        return {"error": "SPY not available"}

    dates = data.index
    signal_dates = dates[crisis_exit_signals]

    results = []
    for dt in signal_dates:
        loc = dates.get_loc(dt)
        if loc + 252 >= len(dates):
            continue

        p0 = data[spy_col].iloc[loc]
        entry = {"date": str(dt)[:10], "spy_price": float(p0)}

        for horizon_name, horizon_days in [("1m", 21), ("3m", 63), ("6m", 126), ("12m", 252)]:
            if loc + horizon_days < len(dates):
                p1 = data[spy_col].iloc[loc + horizon_days]
                entry[f"spy_ret_{horizon_name}"] = float(p1 / p0 - 1)

        results.append(entry)

    if not results:
        return {"error": "no valid signals"}

    df = pd.DataFrame(results)

    summary = {}
    for col in [c for c in df.columns if c.startswith("spy_ret_")]:
        horizon = col.replace("spy_ret_", "")
        rets = df[col].dropna()
        summary[horizon] = {
            "mean_return": float(rets.mean()),
            "median_return": float(rets.median()),
            "win_rate": float((rets > 0).mean()),
            "n_signals": int(len(rets)),
            "best": float(rets.max()),
            "worst": float(rets.min()),
        }
        log.info(f"  Crisis exit {horizon}: mean={rets.mean():.1%}, WR={((rets > 0).mean()):.0%}, n={len(rets)}")

    return {"signals": results, "summary": summary}


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("Crisis Exit + ML Asymmetric Stock Ranker — Combined Strategy Research")
    log.info("HC #724 — All signals T-1, Walk-forward 504d, 25 bps costs")
    log.info("=" * 70)

    # Start MLflow run
    mlflow_run = None
    if MLFLOW_OK:
        mlflow_run = mlflow.start_run(run_name=f"crisis_exit_ml_{datetime.now().strftime('%Y%m%d_%H%M')}")
        mlflow.log_params({
            "train_days": TRAIN_DAYS,
            "test_days": TEST_DAYS,
            "n_stocks": len(UNIVERSE),
            "cost_bps": COST_BPS,
            "start_year": START_YEAR,
            "vix_threshold": VIX_THRESHOLD,
            "crisis_hold_months": CRISIS_HOLD_MONTHS,
            "device": "gpu",
            "model": "lightgbm",
            "n_base_features": len(BASE_FEATURE_COLS),
            "n_extended_features": len(EXTENDED_FEATURE_COLS),
        })

    # 1. Download data
    data = download_data()

    # 2. Compute crisis exit signals
    crisis_exit_signals = compute_crisis_exit_signals(data)
    crisis_active = compute_crisis_active(crisis_exit_signals, data.index)

    # 3. Analyze crisis exit on its own
    crisis_analysis = analyze_crisis_exits(data, crisis_exit_signals)

    # 4. Build features
    df = build_feature_matrix(data, crisis_exit_signals, crisis_active)
    df.to_parquet(OUTPUT_DIR / "feature_matrix.parquet")
    log.info(f"Feature matrix saved: {len(df)} rows")

    # 5. Walk-forward training — BASE model (Strategies A, B, C, D)
    preds_base, folds_base, importance_base = walk_forward_train(df, BASE_FEATURE_COLS, "base")
    preds_base.to_parquet(OUTPUT_DIR / "predictions_base.parquet")

    # 6. Walk-forward training — EXTENDED model (Strategy E)
    preds_ext, folds_ext, importance_ext = walk_forward_train(df, EXTENDED_FEATURE_COLS, "extended")
    preds_ext.to_parquet(OUTPUT_DIR / "predictions_extended.parquet")

    # 7. Build all strategy portfolios
    portfolios = build_all_strategy_portfolios(preds_base, preds_ext, data, crisis_exit_signals, crisis_active)

    # 8. Compute metrics for all strategies
    log.info("\n" + "=" * 70)
    log.info("STRATEGY PERFORMANCE COMPARISON (net of 25 bps/leg)")
    log.info("=" * 70)

    all_metrics = {}
    for strat_name, port in portfolios.items():
        m = compute_metrics(port["returns"], strat_name)
        all_metrics[strat_name] = m
        log.info(f"\n{strat_name}:")
        log.info(f"  Ann Return: {m['ann_return']:.1%} | Ann Vol: {m['ann_vol']:.1%}")
        log.info(f"  Sharpe: {m['sharpe']:.2f} | Sortino: {m['sortino']:.2f}")
        log.info(f"  Max DD: {m['max_dd']:.1%} | Win Rate: {m['win_rate']:.1%}")
        log.info(f"  Profit Factor: {m['profit_factor']:.2f} | Calmar: {m['calmar']:.2f}")
        log.info(f"  Total Return: {m['total_return']:.1%} | Active Months: {m['active_months']}/{m['n_months']}")

    # 9. Rank IC summaries
    log.info("\n" + "=" * 70)
    log.info("RANK IC SUMMARY")
    log.info("=" * 70)

    ic_summaries = {}
    for model_name, fold_metrics in [("base", folds_base), ("extended", folds_ext)]:
        ic_values = [m["rank_ic"] for m in fold_metrics if not np.isnan(m["rank_ic"])]
        mean_ic = np.mean(ic_values)
        ic_ir = mean_ic / (np.std(ic_values) + 1e-10)
        ic_summaries[model_name] = {
            "mean_ic": float(mean_ic),
            "ic_ir": float(ic_ir),
            "ic_std": float(np.std(ic_values)),
            "pct_positive": float(np.mean(np.array(ic_values) > 0)),
            "n_folds": len(ic_values),
        }
        log.info(f"\n  {model_name} model: IC={mean_ic:.4f}, IR={ic_ir:.4f}, "
                 f"pct+={np.mean(np.array(ic_values) > 0):.1%}, n={len(ic_values)}")

    # 10. Feature importance
    log.info("\nFeature Importance — Base Model (top 10):")
    sorted_imp_base = sorted(importance_base.items(), key=lambda x: x[1], reverse=True)
    for feat, imp in sorted_imp_base[:10]:
        log.info(f"  {feat}: {imp:.0f}")

    log.info("\nFeature Importance — Extended Model (top 10):")
    sorted_imp_ext = sorted(importance_ext.items(), key=lambda x: x[1], reverse=True)
    for feat, imp in sorted_imp_ext[:10]:
        log.info(f"  {feat}: {imp:.0f}")

    # 11. Validation tests
    log.info("\n" + "=" * 70)
    log.info("VALIDATION TESTS")
    log.info("=" * 70)

    # Permutation test on base model
    perm_base = permutation_test(preds_base, n_perms=200)
    perm_ext = permutation_test(preds_ext, n_perms=200)

    # Regime test on all strategies
    regime_results = regime_test(portfolios, data)

    # Sub-period stability for each strategy
    stability_results = {}
    for strat_name, port in portfolios.items():
        stability_results[strat_name] = subperiod_stability(port["returns"], port["dates"], strat_name)

    # Lag sensitivity
    lag_base = lag_sensitivity_test(preds_base)
    lag_ext = lag_sensitivity_test(preds_ext)

    # 12. Summary: which strategy is best?
    log.info("\n" + "=" * 70)
    log.info("VALIDATION GATE SUMMARY")
    log.info("=" * 70)

    gate_summary = {}
    for strat_name in portfolios:
        m = all_metrics[strat_name]
        regime = regime_results.get(strat_name, {})
        stab = stability_results.get(strat_name, {})

        passes_regime = regime.get("passes_regime_gate", False)
        passes_stability = stab.get("passes_stability_gate", False)

        gate_summary[strat_name] = {
            "sharpe": m.get("sharpe", 0),
            "sortino": m.get("sortino", 0),
            "regime_gap": regime.get("regime_gap", 999),
            "passes_regime": passes_regime,
            "stability_cv": stab.get("cv_sharpe", 999),
            "passes_stability": passes_stability,
            "all_gates_pass": passes_regime and passes_stability,
        }

        status = "PASS ALL" if gate_summary[strat_name]["all_gates_pass"] else "FAIL"
        log.info(f"  {strat_name}: Sharpe={m.get('sharpe', 0):.2f}, "
                 f"Regime={'PASS' if passes_regime else 'FAIL'} (gap={regime.get('regime_gap', 999):.2f}), "
                 f"Stability={'PASS' if passes_stability else 'FAIL'} (CV={stab.get('cv_sharpe', 999):.2f}) "
                 f"=> {status}")

    # 13. Log to MLflow
    if MLFLOW_OK and mlflow_run:
        for model_name, summary in ic_summaries.items():
            for k, v in summary.items():
                mlflow.log_metric(f"{model_name}_{k}", v)

        mlflow.log_metric("perm_p_base", perm_base["p_value"])
        mlflow.log_metric("perm_p_ext", perm_ext["p_value"])

        for strat_name, m in all_metrics.items():
            if isinstance(m, dict) and "sharpe" in m:
                mlflow.log_metrics({
                    f"{strat_name}_sharpe": m["sharpe"],
                    f"{strat_name}_sortino": m["sortino"],
                    f"{strat_name}_ann_ret": m["ann_return"],
                    f"{strat_name}_max_dd": m["max_dd"],
                    f"{strat_name}_win_rate": m["win_rate"],
                    f"{strat_name}_total_ret": m["total_return"],
                })

        for strat_name, g in gate_summary.items():
            mlflow.log_metric(f"{strat_name}_regime_gap", g["regime_gap"])
            mlflow.log_metric(f"{strat_name}_stability_cv", g["stability_cv"])
            mlflow.log_metric(f"{strat_name}_all_gates", int(g["all_gates_pass"]))

        mlflow.log_metric("n_crisis_signals", int(crisis_exit_signals.sum()))

        mlflow.end_run()

    # 14. Save full results
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Timestamp):
                return str(obj)
            elif isinstance(obj, (np.bool_,)):
                return bool(obj)
            return super().default(obj)

    full_results = {
        "run_time": datetime.now().isoformat(),
        "duration_sec": time.time() - t0,
        "config": {
            "train_days": TRAIN_DAYS,
            "test_days": TEST_DAYS,
            "n_stocks": len(UNIVERSE),
            "cost_bps": COST_BPS,
            "start_year": START_YEAR,
            "vix_threshold": VIX_THRESHOLD,
            "crisis_hold_months": CRISIS_HOLD_MONTHS,
        },
        "crisis_exit_analysis": crisis_analysis,
        "strategy_metrics": all_metrics,
        "ic_summaries": ic_summaries,
        "feature_importance_base": sorted_imp_base,
        "feature_importance_extended": sorted_imp_ext,
        "permutation_test_base": perm_base,
        "permutation_test_extended": perm_ext,
        "regime_test": regime_results,
        "stability_test": stability_results,
        "lag_sensitivity_base": lag_base,
        "lag_sensitivity_extended": lag_ext,
        "gate_summary": gate_summary,
        "fold_metrics_base": folds_base,
        "fold_metrics_extended": folds_ext,
    }

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(full_results, f, indent=2, cls=NumpyEncoder)

    elapsed = time.time() - t0
    log.info(f"\n{'=' * 70}")
    log.info(f"COMPLETE in {elapsed / 60:.1f} minutes")
    log.info(f"Results saved to {OUTPUT_DIR}")
    log.info(f"{'=' * 70}")


if __name__ == "__main__":
    main()
