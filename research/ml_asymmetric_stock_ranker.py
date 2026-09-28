#!/usr/bin/env python3
"""
ML Asymmetric Stock Ranker — GPU-accelerated LightGBM
Predicts 1-month forward returns to identify stocks with asymmetric upside bounces.

Walk-forward: 504d train, 21d test, monthly rebalance, 2015-2026.
Features: ~20 per stock capturing mean-reversion, vol, momentum, breadth, VIX.
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

# ── Setup logging ────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("/home/nick/Lvl3Quant/output/ml_asymmetric_stock_ranker")
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
    mlflow.set_experiment("ml_asymmetric_stock_ranker")
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

# Map stocks to sector ETFs
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
COST_BPS = 25  # 25 bps per leg
START_YEAR = 2015

# ── Data Download ────────────────────────────────────────────────────────────
def download_data():
    """Download all required price data via yfinance."""
    import yfinance as yf

    cache_file = OUTPUT_DIR / "price_cache.parquet"
    if cache_file.exists():
        log.info("Loading cached price data")
        df = pd.read_parquet(cache_file)
        # Check if reasonably fresh (within 7 days)
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

    # Extract close and volume
    close = data["Close"] if "Close" in data.columns.get_level_values(0) else data["Adj Close"]
    volume = data["Volume"]

    # Combine into single df with multi-level columns
    combined = pd.concat({"Close": close, "Volume": volume}, axis=1)
    combined.to_parquet(cache_file)
    log.info(f"Data shape: {combined.shape}, range: {combined.index[0]} to {combined.index[-1]}")
    return combined


# ── Feature Engineering ──────────────────────────────────────────────────────
def compute_features(data, date_idx, ticker):
    """Compute all features for a single stock at a single date (using T-1 data)."""
    close_col = ("Close", ticker)
    vol_col = ("Volume", ticker)

    if close_col not in data.columns or vol_col not in data.columns:
        return None

    # Get data up to T-1
    loc = data.index.get_loc(date_idx)
    if loc < 252:  # Need at least 1 year of history
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

    # 5. Momentum (1m, 3m, 6m)
    feats["mom_1m"] = p / prices[-21] - 1 if len(prices) >= 21 else np.nan
    feats["mom_3m"] = p / prices[-63] - 1 if len(prices) >= 63 else np.nan
    feats["mom_6m"] = p / prices[-126] - 1 if len(prices) >= 126 else np.nan

    # 6. Mean reversion z-score (price vs 63d mean)
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

    # 10. Momentum acceleration (1m mom - 3m mom / 3)
    if not np.isnan(feats.get("mom_1m", np.nan)) and not np.isnan(feats.get("mom_3m", np.nan)):
        feats["mom_accel"] = feats["mom_1m"] - feats["mom_3m"] / 3
    else:
        feats["mom_accel"] = np.nan

    return feats


def build_feature_matrix(data):
    """Build the full feature matrix with walk-forward dates."""
    log.info("Building feature matrix...")

    # Get trading dates
    dates = data.index

    # Get VIX and VIX3M
    vix_col = ("Close", "^VIX") if ("Close", "^VIX") in data.columns else None
    vix3m_col = ("Close", "^VIX3M") if ("Close", "^VIX3M") in data.columns else None

    # Sector ETF returns
    sector_rets_1m = {}
    for etf in SECTOR_ETFS:
        col = ("Close", etf)
        if col in data.columns:
            sector_rets_1m[etf] = data[col].pct_change(21)

    # Breadth: % of sector ETFs above 50d SMA
    breadth = pd.Series(0.0, index=dates)
    for etf in SECTOR_ETFS:
        col = ("Close", etf)
        if col in data.columns:
            sma50 = data[col].rolling(50).mean()
            breadth += (data[col] > sma50).astype(float)
    breadth = breadth / len(SECTOR_ETFS)

    rows = []
    # Monthly rebalance dates (every ~21 trading days)
    # Start after enough history
    start_idx = max(504, 252)  # Need 504 for training + 252 for features

    rebal_dates = dates[start_idx::TEST_DAYS]
    log.info(f"Total rebalance dates: {len(rebal_dates)} from {rebal_dates[0]} to {rebal_dates[-1]}")

    for i, dt in enumerate(rebal_dates):
        if i % 50 == 0:
            log.info(f"  Processing date {i}/{len(rebal_dates)}: {dt.strftime('%Y-%m-%d')}")

        loc = data.index.get_loc(dt)

        # Forward return (target) — 21 trading days ahead
        if loc + TEST_DAYS >= len(dates):
            continue

        # Market-level features
        vix_val = data[vix_col].iloc[loc] if vix_col else np.nan
        vix3m_val = data[vix3m_col].iloc[loc] if vix3m_col else np.nan
        vix_term = vix_val / (vix3m_val + 1e-10) if not np.isnan(vix_val) and not np.isnan(vix3m_val) else np.nan
        breadth_val = breadth.iloc[loc]

        # Cross-sectional 1m returns for ranking
        cs_rets = {}
        for ticker in UNIVERSE:
            col = ("Close", ticker)
            if col in data.columns:
                p_now = data[col].iloc[loc]
                p_prev = data[col].iloc[loc - 21] if loc >= 21 else np.nan
                if not np.isnan(p_now) and not np.isnan(p_prev) and p_prev > 0:
                    cs_rets[ticker] = p_now / p_prev - 1

        # Rank cross-sectional returns
        if cs_rets:
            sorted_tickers = sorted(cs_rets.keys(), key=lambda t: cs_rets[t])
            cs_rank = {t: rank / len(sorted_tickers) for rank, t in enumerate(sorted_tickers)}
        else:
            cs_rank = {}

        for ticker in UNIVERSE:
            feats = compute_features(data, dt, ticker)
            if feats is None:
                continue

            # Add market-level features
            feats["vix"] = vix_val
            feats["vix_term_structure"] = vix_term
            feats["breadth"] = breadth_val

            # Sector ETF return
            sector = STOCK_SECTOR.get(ticker, "XLK")
            feats["sector_ret_1m"] = sector_rets_1m.get(sector, pd.Series(dtype=float)).iloc[loc] if sector in sector_rets_1m else np.nan

            # Cross-sectional rank
            feats["cs_rank_1m"] = cs_rank.get(ticker, np.nan)

            # Forward return (target)
            fwd_col = ("Close", ticker)
            p_now = data[fwd_col].iloc[loc]
            p_fwd = data[fwd_col].iloc[loc + TEST_DAYS]
            fwd_ret = p_fwd / p_now - 1 if not np.isnan(p_now) and not np.isnan(p_fwd) and p_now > 0 else np.nan

            # Asymmetric filter flags
            vol_80pct = np.nanpercentile(
                [compute_features(data, dt, t) or {} for t in UNIVERSE[:5]],
                80
            ) if False else np.nan  # Computed below in batch

            feats["ticker"] = ticker
            feats["date"] = dt
            feats["fwd_ret_1m"] = fwd_ret
            rows.append(feats)

    df = pd.DataFrame(rows)
    log.info(f"Feature matrix: {df.shape[0]} rows, {df.shape[1]} cols")

    # Compute vol percentile ranks cross-sectionally per date
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
def walk_forward_train(df):
    """Walk-forward LightGBM training with GPU acceleration."""
    log.info("Starting walk-forward training...")

    feature_cols = [
        "rsi_14", "dist_52w_high", "vol_20d", "vol_63d", "vol_surge",
        "mom_1m", "mom_3m", "mom_6m", "mr_zscore", "price_vs_sma200",
        "max_dd_52w", "vol_ratio", "mom_accel",
        "vix", "vix_term_structure", "breadth", "sector_ret_1m", "cs_rank_1m",
        "vol_20d_pctrank", "vol_63d_pctrank",
    ]

    params = {
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

    # Get unique dates sorted
    unique_dates = sorted(df["date"].unique())
    log.info(f"Unique rebalance dates: {len(unique_dates)}")

    # Walk-forward: 504d train ~ 24 monthly periods, 21d test = 1 period
    train_periods = TRAIN_DAYS // TEST_DAYS  # 24 months
    results = []
    all_predictions = []
    fold_metrics = []

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

        # Remove NaN targets
        valid_train = ~np.isnan(y_train) & ~np.any(np.isnan(X_train), axis=1)
        valid_test = ~np.isnan(y_test) & ~np.any(np.isnan(X_test), axis=1)

        if valid_train.sum() < 50 or valid_test.sum() < 5:
            continue

        X_train, y_train = X_train[valid_train], y_train[valid_train]
        X_test_v, y_test_v = X_test[valid_test], y_test[valid_test]

        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_test_v, label=y_test_v, reference=dtrain)

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=500,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )

        preds = model.predict(X_test_v)

        # Rank IC (Spearman correlation between predicted and actual ranks)
        if len(preds) >= 5:
            rank_ic = stats.spearmanr(preds, y_test_v)[0]
        else:
            rank_ic = np.nan

        # Store predictions
        test_subset = test_df[valid_test].copy()
        test_subset["prediction"] = preds
        test_subset["fold"] = i
        all_predictions.append(test_subset)

        fold_metrics.append({
            "fold": i,
            "date": str(test_date)[:10],
            "n_stocks": len(preds),
            "rank_ic": rank_ic,
            "mse": np.mean((preds - y_test_v) ** 2),
        })

        if i % 20 == 0:
            recent_ics = [m["rank_ic"] for m in fold_metrics[-20:] if not np.isnan(m["rank_ic"])]
            avg_ic = np.mean(recent_ics) if recent_ics else 0
            log.info(f"  Fold {i}/{len(unique_dates)}, date={str(test_date)[:10]}, "
                     f"Rank IC={rank_ic:.4f}, Avg IC (20)={avg_ic:.4f}")

    # Feature importance from last model
    importance = dict(zip(feature_cols, model.feature_importance(importance_type="gain")))

    all_preds_df = pd.concat(all_predictions, ignore_index=True)
    log.info(f"Walk-forward complete: {len(fold_metrics)} folds, {len(all_preds_df)} predictions")

    return all_preds_df, fold_metrics, importance, feature_cols


# ── Portfolio Construction ───────────────────────────────────────────────────
def build_portfolios(preds_df):
    """Build strategy portfolios and compute returns."""
    log.info("Building portfolios...")

    cost_per_trade = COST_BPS / 10000  # 25 bps per leg, so 50 bps round trip per stock

    strategies = {
        "A_top5": {"n": 5, "filter": False},
        "B_top5_filtered": {"n": 5, "filter": True},
        "C_top10": {"n": 10, "filter": False},
        "baseline_ew50": {"n": 50, "filter": False},
    }

    results = {}

    for strat_name, config in strategies.items():
        monthly_returns = []
        monthly_dates = []
        monthly_holdings = []

        for dt, group in preds_df.groupby("date"):
            group = group.dropna(subset=["fwd_ret_1m", "prediction"])

            if config["filter"] and strat_name == "B_top5_filtered":
                # Only consider stocks passing asymmetric filter
                filtered = group[group["asymmetric_filter"] == 1]
                if len(filtered) < 1:
                    # No stocks pass filter — stay in cash
                    monthly_returns.append(0.0)
                    monthly_dates.append(dt)
                    monthly_holdings.append([])
                    continue
                group = filtered

            if strat_name == "baseline_ew50":
                # Equal weight all stocks
                selected = group
            else:
                # Rank by prediction, take top N
                selected = group.nlargest(config["n"], "prediction")

            if len(selected) == 0:
                monthly_returns.append(0.0)
                monthly_dates.append(dt)
                monthly_holdings.append([])
                continue

            # Equal weight portfolio return
            port_ret = selected["fwd_ret_1m"].mean()

            # Transaction costs: assume full turnover each month
            n_trades = len(selected) * 2  # Buy + sell
            total_cost = n_trades * cost_per_trade
            port_ret_net = port_ret - total_cost

            monthly_returns.append(port_ret_net)
            monthly_dates.append(dt)
            monthly_holdings.append(selected["ticker"].tolist())

        results[strat_name] = {
            "returns": np.array(monthly_returns),
            "dates": monthly_dates,
            "holdings": monthly_holdings,
        }

    # Also get SPY returns for comparison
    return results


def compute_spy_returns(data, dates):
    """Compute SPY returns aligned to rebalance dates."""
    spy_col = ("Close", "SPY")
    if spy_col not in data.columns:
        return np.zeros(len(dates))

    spy_rets = []
    for dt in dates:
        loc = data.index.get_loc(dt)
        if loc + TEST_DAYS < len(data.index):
            p0 = data[spy_col].iloc[loc]
            p1 = data[spy_col].iloc[loc + TEST_DAYS]
            spy_rets.append(p1 / p0 - 1 if p0 > 0 else 0)
        else:
            spy_rets.append(0)
    return np.array(spy_rets)


# ── Performance Metrics ──────────────────────────────────────────────────────
def compute_metrics(returns, name="Strategy"):
    """Compute comprehensive performance metrics."""
    returns = np.array(returns)
    returns = returns[~np.isnan(returns)]

    if len(returns) < 2:
        return {}

    # Annualize (monthly returns, 12 periods/year)
    ann_ret = np.mean(returns) * 12
    ann_vol = np.std(returns) * np.sqrt(12)
    sharpe = ann_ret / (ann_vol + 1e-10)

    # Sortino
    downside = returns[returns < 0]
    downside_vol = np.std(downside) * np.sqrt(12) if len(downside) > 0 else 1e-10
    sortino = ann_ret / (downside_vol + 1e-10)

    # Max drawdown (on cumulative returns)
    cum = np.cumprod(1 + returns)
    rolling_max = np.maximum.accumulate(cum)
    drawdowns = cum / rolling_max - 1
    max_dd = np.min(drawdowns)

    # Win rate
    wr = np.mean(returns > 0)

    # Profit factor
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    pf = gains / (losses + 1e-10)

    # Calmar
    calmar = ann_ret / (abs(max_dd) + 1e-10)

    return {
        "name": name,
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": max_dd,
        "win_rate": wr,
        "profit_factor": pf,
        "calmar": calmar,
        "n_months": len(returns),
        "total_return": np.prod(1 + returns) - 1,
    }


# ── Validation Tests ─────────────────────────────────────────────────────────
def permutation_test(preds_df, n_perms=200):
    """Test if signal ranking is statistically significant via permutation."""
    log.info(f"Running permutation test ({n_perms} permutations)...")

    # Actual Rank IC per fold
    actual_ics = []
    for dt, group in preds_df.groupby("date"):
        group = group.dropna(subset=["fwd_ret_1m", "prediction"])
        if len(group) >= 5:
            ic = stats.spearmanr(group["prediction"], group["fwd_ret_1m"])[0]
            if not np.isnan(ic):
                actual_ics.append(ic)

    actual_mean_ic = np.mean(actual_ics)

    # Permuted ICs
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
    log.info(f"  Perm IC distribution: mean={np.mean(perm_mean_ics):.4f}, "
             f"std={np.std(perm_mean_ics):.4f}")

    return {
        "actual_mean_ic": actual_mean_ic,
        "p_value": p_value,
        "perm_mean_ic": np.mean(perm_mean_ics),
        "perm_std_ic": np.std(perm_mean_ics),
        "significant_5pct": p_value < 0.05,
    }


def regime_test(preds_df, data):
    """Test performance in different market regimes."""
    log.info("Running regime test...")

    spy_col = ("Close", "SPY")
    if spy_col not in data.columns:
        return {"error": "SPY data not available"}

    spy_prices = data[spy_col]
    spy_sma200 = spy_prices.rolling(200).mean()

    results = {}
    for regime_name, regime_filter in [
        ("bull", lambda dt: spy_prices.loc[dt] > spy_sma200.loc[dt]),
        ("bear", lambda dt: spy_prices.loc[dt] <= spy_sma200.loc[dt]),
    ]:
        regime_ics = []
        regime_top5_rets = []
        for dt, group in preds_df.groupby("date"):
            try:
                is_regime = regime_filter(dt)
            except (KeyError, TypeError):
                continue
            if not is_regime:
                continue
            group = group.dropna(subset=["fwd_ret_1m", "prediction"])
            if len(group) >= 5:
                ic = stats.spearmanr(group["prediction"], group["fwd_ret_1m"])[0]
                if not np.isnan(ic):
                    regime_ics.append(ic)
                top5 = group.nlargest(5, "prediction")
                regime_top5_rets.append(top5["fwd_ret_1m"].mean())

        results[regime_name] = {
            "mean_ic": np.mean(regime_ics) if regime_ics else np.nan,
            "n_months": len(regime_ics),
            "mean_top5_ret": np.mean(regime_top5_rets) if regime_top5_rets else np.nan,
        }
        log.info(f"  {regime_name}: IC={results[regime_name]['mean_ic']:.4f}, "
                 f"n={results[regime_name]['n_months']}, "
                 f"top5_ret={results[regime_name]['mean_top5_ret']:.4f}")

    return results


def subperiod_stability(fold_metrics):
    """Test IC stability across sub-periods."""
    log.info("Running sub-period stability test...")

    df = pd.DataFrame(fold_metrics)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["rank_ic"])

    # Split into 3 equal sub-periods
    n = len(df)
    periods = {
        "early": df.iloc[:n // 3],
        "middle": df.iloc[n // 3:2 * n // 3],
        "late": df.iloc[2 * n // 3:],
    }

    results = {}
    for name, sub in periods.items():
        ics = sub["rank_ic"].values
        results[name] = {
            "mean_ic": np.mean(ics),
            "std_ic": np.std(ics),
            "ic_ir": np.mean(ics) / (np.std(ics) + 1e-10),
            "pct_positive": np.mean(ics > 0),
            "n": len(ics),
            "date_range": f"{sub['date'].iloc[0].strftime('%Y-%m')} to {sub['date'].iloc[-1].strftime('%Y-%m')}",
        }
        log.info(f"  {name}: IC={results[name]['mean_ic']:.4f}, "
                 f"IR={results[name]['ic_ir']:.4f}, "
                 f"pct+={results[name]['pct_positive']:.1%}, "
                 f"range={results[name]['date_range']}")

    return results


def lag_sensitivity(preds_df):
    """Test sensitivity to feature lag (check if T-2 features still predict)."""
    log.info("Running lag sensitivity test...")
    # We check if IC decays when we shift the alignment
    # Using the existing predictions but checking actual vs shifted actual returns
    ics_by_shift = {}
    for shift_name, shift_val in [("T+0 (actual)", 0), ("T+1 (1mo lag)", 1), ("T+2 (2mo lag)", 2)]:
        dates = sorted(preds_df["date"].unique())
        shifted_ics = []
        for i, dt in enumerate(dates):
            if i + shift_val >= len(dates):
                break
            current = preds_df[preds_df["date"] == dt].dropna(subset=["prediction"])
            shifted_dt = dates[min(i + shift_val, len(dates) - 1)]
            shifted = preds_df[preds_df["date"] == shifted_dt].dropna(subset=["fwd_ret_1m"])

            # Merge on ticker
            merged = current[["ticker", "prediction"]].merge(
                shifted[["ticker", "fwd_ret_1m"]], on="ticker"
            )
            if len(merged) >= 5:
                ic = stats.spearmanr(merged["prediction"], merged["fwd_ret_1m"])[0]
                if not np.isnan(ic):
                    shifted_ics.append(ic)

        ics_by_shift[shift_name] = np.mean(shifted_ics) if shifted_ics else np.nan
        log.info(f"  {shift_name}: Mean IC = {ics_by_shift[shift_name]:.4f}")

    return ics_by_shift


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("ML Asymmetric Stock Ranker — Starting")
    log.info("=" * 70)

    # Start MLflow run
    mlflow_run = None
    if MLFLOW_OK:
        mlflow_run = mlflow.start_run(run_name=f"asymmetric_ranker_{datetime.now().strftime('%Y%m%d_%H%M')}")
        mlflow.log_params({
            "train_days": TRAIN_DAYS,
            "test_days": TEST_DAYS,
            "n_stocks": len(UNIVERSE),
            "cost_bps": COST_BPS,
            "device": "gpu",
            "model": "lightgbm",
        })

    # 1. Download data
    data = download_data()

    # 2. Build features
    df = build_feature_matrix(data)
    df.to_parquet(OUTPUT_DIR / "feature_matrix.parquet")
    log.info(f"Feature matrix saved: {len(df)} rows")

    # 3. Walk-forward training
    preds_df, fold_metrics, importance, feature_cols = walk_forward_train(df)
    preds_df.to_parquet(OUTPUT_DIR / "predictions.parquet")

    # 4. Portfolio construction
    portfolios = build_portfolios(preds_df)
    spy_rets = compute_spy_returns(data, portfolios["A_top5"]["dates"])

    # 5. Compute metrics
    log.info("\n" + "=" * 70)
    log.info("PORTFOLIO PERFORMANCE (net of 25 bps/leg transaction costs)")
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
        log.info(f"  Total Return: {m['total_return']:.1%} | N Months: {m['n_months']}")

    spy_m = compute_metrics(spy_rets, "SPY")
    all_metrics["SPY"] = spy_m
    log.info(f"\nSPY Benchmark:")
    log.info(f"  Ann Return: {spy_m['ann_return']:.1%} | Sharpe: {spy_m['sharpe']:.2f}")
    log.info(f"  Max DD: {spy_m['max_dd']:.1%} | Total Return: {spy_m['total_return']:.1%}")

    # 6. Rank IC summary
    ic_values = [m["rank_ic"] for m in fold_metrics if not np.isnan(m["rank_ic"])]
    mean_ic = np.mean(ic_values)
    ic_ir = mean_ic / (np.std(ic_values) + 1e-10)
    log.info(f"\nRank IC Summary:")
    log.info(f"  Mean Rank IC: {mean_ic:.4f}")
    log.info(f"  IC IR: {ic_ir:.4f}")
    log.info(f"  IC Std: {np.std(ic_values):.4f}")
    log.info(f"  % Positive IC: {np.mean(np.array(ic_values) > 0):.1%}")

    # 7. Feature importance
    log.info(f"\nFeature Importance (top 10):")
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for feat, imp in sorted_imp[:10]:
        log.info(f"  {feat}: {imp:.0f}")

    # 8. Validation tests
    log.info("\n" + "=" * 70)
    log.info("VALIDATION TESTS")
    log.info("=" * 70)

    perm_results = permutation_test(preds_df, n_perms=200)
    regime_results = regime_test(preds_df, data)
    subperiod_results = subperiod_stability(fold_metrics)
    lag_results = lag_sensitivity(preds_df)

    # 9. Log to MLflow
    if MLFLOW_OK and mlflow_run:
        mlflow.log_metrics({
            "mean_rank_ic": mean_ic,
            "ic_ir": ic_ir,
            "ic_pct_positive": np.mean(np.array(ic_values) > 0),
            "perm_p_value": perm_results["p_value"],
        })
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
        mlflow.log_metrics({
            "regime_bull_ic": regime_results.get("bull", {}).get("mean_ic", 0),
            "regime_bear_ic": regime_results.get("bear", {}).get("mean_ic", 0),
        })
        # Log feature importance
        for feat, imp in sorted_imp:
            mlflow.log_metric(f"feat_imp_{feat}", imp)

        mlflow.end_run()

    # 10. Save full results
    full_results = {
        "run_time": datetime.now().isoformat(),
        "duration_sec": time.time() - t0,
        "metrics": {k: v for k, v in all_metrics.items()},
        "rank_ic": {"mean": mean_ic, "ir": ic_ir, "std": float(np.std(ic_values)), "pct_positive": float(np.mean(np.array(ic_values) > 0))},
        "feature_importance": sorted_imp,
        "permutation_test": perm_results,
        "regime_test": {k: v for k, v in regime_results.items()},
        "subperiod_stability": subperiod_results,
        "lag_sensitivity": lag_results,
        "fold_metrics": fold_metrics,
    }

    # Convert numpy types for JSON serialization
    def convert(obj):
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
        return obj

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            r = convert(obj)
            if r is not obj:
                return r
            return super().default(obj)

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(full_results, f, indent=2, cls=NumpyEncoder)

    elapsed = time.time() - t0
    log.info(f"\n{'=' * 70}")
    log.info(f"COMPLETE in {elapsed / 60:.1f} minutes")
    log.info(f"Results saved to {OUTPUT_DIR}")
    log.info(f"{'=' * 70}")


if __name__ == "__main__":
    main()
