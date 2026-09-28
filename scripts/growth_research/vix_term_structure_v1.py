#!/usr/bin/env python3
"""
VIX Term Structure Research v1
===============================

Research priority #8: VIX term structure (contango/backwardation) as a trading signal.

VIX futures term structure interpretation:
  - Contango (VIX < VIX3M): market expects vol to increase or stay flat — normal state
  - Backwardation (VIX > VIX3M): market expects vol to DECREASE from crisis — fear mode

Features constructed:
  - VIX_ratio = VIX / VIX3M (< 1 = contango, > 1 = backwardation)
  - VIX_slope = (VIX3M - VIX) / VIX (normalized slope, positive = contango)
  - VIX_slope_change_5d = 5-day change in slope (momentum of term structure)
  - VIX_regime = categorical: deep_contango / contango / flat / backwardation / deep_backwardation

6 backtest variants:
  A. Baseline (regime>0.4, legacy 18 features)
  B. Baseline + 3 VIX term structure features in LGBM
  C. Regime>0.4 AND VIX_ratio > 0.90
  D. Regime>0.4 AND VIX_ratio > 0.95
  E. Regime>0.4 AND VIX slope flattening (5d change < 0)
  F. Adaptive position sizing based on VIX term structure

All variants: walk-forward LGBM, $645 capital, 3% bull call spreads, DTE=21,
hold to expiry, 15% entry haircut, $2.60 commission.
Full 5-gate adversarial validation.

Uses standardized tools:
  - research.tools.options_pricer for pricing
  - research.tools.adversarial_validator for validation
"""
from __future__ import annotations

import sys
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, "/home/jupiter/Lvl3Quant")

from research.tools.options_pricer import (
    price_bull_call_spread,
    exit_spread_value,
    spread_pnl,
    estimate_iv,
    compute_atr,
    COMMISSION_RT_SPREAD,
    DEFAULT_HAIRCUT,
)
from research.tools.adversarial_validator import validate_trades

BASE = Path("/home/jupiter/Lvl3Quant")


def fprint(*args, **kwargs):
    print(*args, **kwargs, flush=True)


# ─── MLflow Setup ──────────────────────────────────────────────────

MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen("http://jupiter:5000/", timeout=2)
    import mlflow
    mlflow.set_tracking_uri("http://jupiter:5000")
    MLFLOW_OK = True
    fprint("[MLflow] Connected")
except Exception:
    fprint("[MLflow] Not available, skipping logging")


# ─── Configuration ─────────────────────────────────────────────────

CONFIG = {
    # Universe
    "sector_etfs": [
        "XLE", "XLK", "XLF", "XLV", "XLI", "XLP", "XLU", "XLRE", "XLB", "XLC", "XLY",
    ],
    "benchmark": "SPY",
    "macro_tickers": ["TLT", "HYG", "GLD"],
    "vix_tickers": ["^VIX", "^VIX3M", "^VIX9D"],

    # Dates
    "start_date": "2011-01-01",
    "end_date": "2026-07-25",

    # Capital
    "initial_capital": 645.0,
    "max_trade_pct": 0.40,

    # Options
    "target_dte": 21,
    "spread_width_pct": 0.03,
    "top_n": 3,
    "rebalance_freq_days": 10,
    "haircut": DEFAULT_HAIRCUT,
    "commission_rt": COMMISSION_RT_SPREAD,

    # VIX term structure regimes
    "vix_regime_thresholds": {
        "deep_contango": 0.85,
        "contango": 0.95,
        "flat": 1.05,
        "backwardation": 1.15,
        # > 1.15 = deep_backwardation
    },

    # Walk-forward LGBM
    "train_months": 12,
    "test_months": 1,
    "lgbm_params": {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "max_depth": 6,
        "verbosity": -1,
        "seed": 42,
        "n_jobs": 4,
    },
    "num_boost_round": 300,
    "early_stopping_rounds": 30,

    # Validation
    "n_perms": 500,

    # GRU regime
    "regime_threshold": 0.4,
}


# =====================================================================
# 1. DATA LOADING
# =====================================================================

def load_market_data(config):
    """Load price data, VIX spot, VIX3M, VIX9D from yfinance."""
    import yfinance as yf

    all_equity_tickers = config["sector_etfs"] + [config["benchmark"]] + config["macro_tickers"]
    all_equity_tickers = list(set(all_equity_tickers))

    fprint(f"\n[1/8] Loading market data...")
    fprint(f"  Equity tickers: {len(all_equity_tickers)}")

    # Download equity data
    data = yf.download(
        all_equity_tickers,
        start=config["start_date"],
        end=config["end_date"],
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )

    prices = {}
    for ticker in all_equity_tickers:
        try:
            if len(all_equity_tickers) > 1:
                df = data[ticker][["Close", "High", "Low", "Volume"]].dropna()
            else:
                df = data[["Close", "High", "Low", "Volume"]].dropna()
            df.columns = ["close", "high", "low", "volume"]
            if len(df) > 60:
                prices[ticker] = df
        except Exception:
            pass

    # Download VIX family
    vix_data = {}
    for vix_ticker in config["vix_tickers"]:
        try:
            vd = yf.download(
                vix_ticker, start=config["start_date"], end=config["end_date"],
                auto_adjust=True, progress=False,
            )
            series = vd["Close"].dropna()
            if hasattr(series, "columns"):
                series = series.iloc[:, 0]
            if len(series) > 60:
                vix_data[vix_ticker] = series
                fprint(f"  {vix_ticker}: {len(series)} days ({series.index[0].date()} to {series.index[-1].date()})")
        except Exception as e:
            fprint(f"  WARNING: Failed to load {vix_ticker}: {e}")

    fprint(f"  Got {len(prices)} equity tickers, {len(vix_data)} VIX tickers")
    return prices, vix_data


# =====================================================================
# 2. VIX TERM STRUCTURE FEATURES
# =====================================================================

def build_vix_term_structure(vix_data, config):
    """Build VIX term structure features from VIX spot and VIX3M."""
    fprint("\n[2/8] Building VIX term structure features...")

    vix_spot = vix_data.get("^VIX")
    vix3m = vix_data.get("^VIX3M")
    vix9d = vix_data.get("^VIX9D")

    if vix_spot is None or vix3m is None:
        fprint("  ERROR: Need both ^VIX and ^VIX3M for term structure")
        return None

    # Align dates
    common_idx = vix_spot.index.intersection(vix3m.index)
    vix_spot = vix_spot.loc[common_idx]
    vix3m = vix3m.loc[common_idx]

    ts = pd.DataFrame(index=common_idx)
    ts["vix_spot"] = vix_spot.values
    ts["vix3m"] = vix3m.values

    if vix9d is not None:
        vix9d_aligned = vix9d.reindex(common_idx)
        ts["vix9d"] = vix9d_aligned.values
        fprint(f"  VIX9D available: {vix9d_aligned.notna().sum()} days")

    # Core term structure features
    ts["vix_ratio"] = ts["vix_spot"] / ts["vix3m"]
    ts["vix_slope"] = (ts["vix3m"] - ts["vix_spot"]) / ts["vix_spot"]
    ts["vix_slope_change_5d"] = ts["vix_slope"].diff(5)

    # Additional features
    ts["vix_ratio_sma_10"] = ts["vix_ratio"].rolling(10).mean()
    ts["vix_ratio_std_10"] = ts["vix_ratio"].rolling(10).std()

    # Categorical regime
    thresholds = config["vix_regime_thresholds"]
    conditions = [
        ts["vix_ratio"] < thresholds["deep_contango"],
        (ts["vix_ratio"] >= thresholds["deep_contango"]) & (ts["vix_ratio"] < thresholds["contango"]),
        (ts["vix_ratio"] >= thresholds["contango"]) & (ts["vix_ratio"] < thresholds["flat"]),
        (ts["vix_ratio"] >= thresholds["flat"]) & (ts["vix_ratio"] < thresholds["backwardation"]),
        ts["vix_ratio"] >= thresholds["backwardation"],
    ]
    choices = ["deep_contango", "contango", "flat", "backwardation", "deep_backwardation"]
    ts["vix_regime"] = np.select(conditions, choices, default="unknown")

    fprint(f"  Term structure: {len(ts)} days, {ts.index[0].date()} to {ts.index[-1].date()}")
    fprint(f"  VIX_ratio: mean={ts['vix_ratio'].mean():.3f}, "
           f"std={ts['vix_ratio'].std():.3f}, "
           f"min={ts['vix_ratio'].min():.3f}, max={ts['vix_ratio'].max():.3f}")

    return ts


# =====================================================================
# 3. DESCRIPTIVE ANALYSIS
# =====================================================================

def descriptive_analysis(ts, prices, config):
    """Analyze VIX term structure regime distribution and forward returns."""
    fprint("\n[3/8] Descriptive analysis of VIX term structure...")

    # Regime distribution
    regime_counts = pd.Series(ts["vix_regime"]).value_counts()
    fprint("\n  VIX Term Structure Regime Distribution:")
    fprint("  " + "-" * 50)
    total = len(ts)
    for regime in ["deep_contango", "contango", "flat", "backwardation", "deep_backwardation"]:
        count = regime_counts.get(regime, 0)
        pct = count / total * 100
        fprint(f"  {regime:>22s}: {count:5d} days ({pct:5.1f}%)")

    # VIX_ratio histogram stats
    fprint(f"\n  VIX_ratio percentiles:")
    for p in [5, 10, 25, 50, 75, 90, 95]:
        val = ts["vix_ratio"].quantile(p / 100)
        fprint(f"    p{p:02d}: {val:.3f}")

    # Forward returns by regime
    spy = prices.get(config["benchmark"])
    if spy is not None:
        spy_close = spy["close"]
        fprint("\n  SPY Forward Returns by VIX Term Structure Regime:")
        fprint("  " + "-" * 75)
        fprint(f"  {'Regime':>22s} | {'5d':>8s} | {'10d':>8s} | {'21d':>8s} | {'N':>6s}")
        fprint("  " + "-" * 75)

        for horizon in [5, 10, 21]:
            ts[f"spy_fwd_{horizon}d"] = spy_close.reindex(ts.index).pct_change(horizon).shift(-horizon)

        for regime in ["deep_contango", "contango", "flat", "backwardation", "deep_backwardation"]:
            mask = ts["vix_regime"] == regime
            if mask.sum() < 10:
                continue
            r5 = ts.loc[mask, "spy_fwd_5d"].mean() * 100
            r10 = ts.loc[mask, "spy_fwd_10d"].mean() * 100
            r21 = ts.loc[mask, "spy_fwd_21d"].mean() * 100
            n = mask.sum()
            fprint(f"  {regime:>22s} | {r5:+7.3f}% | {r10:+7.3f}% | {r21:+7.3f}% | {n:6d}")

        # Sector forward returns by regime
        fprint("\n  Sector 21d Forward Returns by VIX Regime (mean %):")
        fprint("  " + "-" * 120)
        header = f"  {'Regime':>22s}"
        for etf in config["sector_etfs"]:
            header += f" | {etf:>5s}"
        fprint(header)
        fprint("  " + "-" * 120)

        for regime in ["deep_contango", "contango", "flat", "backwardation", "deep_backwardation"]:
            mask = ts["vix_regime"] == regime
            if mask.sum() < 10:
                continue
            row = f"  {regime:>22s}"
            for etf in config["sector_etfs"]:
                if etf in prices:
                    etf_close = prices[etf]["close"]
                    fwd = etf_close.reindex(ts.index).pct_change(21).shift(-21)
                    val = fwd.loc[mask].mean() * 100
                    row += f" | {val:+5.2f}"
                else:
                    row += f" |   N/A"
            fprint(row)

    return ts


# =====================================================================
# 4. LOAD GRU REGIME PREDICTIONS
# =====================================================================

def load_regime_predictions():
    """Load GRU regime predictions from disk."""
    fprint("\n[4/8] Loading GRU regime predictions...")
    regime_path = BASE / 'output' / 'regime_detector_v1' / 'regime_predictions_v1.npz'
    if not regime_path.exists():
        fprint(f"  WARNING: Regime predictions not found at {regime_path}")
        fprint("  Will run without GRU regime filter")
        return None

    data = np.load(regime_path, allow_pickle=True)
    dates = pd.to_datetime(data['dates'])
    scores = data['regime_scores']

    regime_series = pd.Series(scores, index=dates, name='regime_score')
    if regime_series.index.duplicated().any():
        regime_series = regime_series[~regime_series.index.duplicated(keep='last')]
    fprint(f"  Loaded {len(regime_series)} regime scores, {dates[0].date()} to {dates[-1].date()}")
    fprint(f"  Score range: {scores.min():.3f} to {scores.max():.3f}, mean={scores.mean():.3f}")
    fprint(f"  Days above 0.4: {(scores > 0.4).sum()} ({(scores > 0.4).mean()*100:.1f}%)")
    return regime_series


# =====================================================================
# 5. FEATURE ENGINEERING
# =====================================================================

def _compute_atr_series(high, low, close, period=14):
    """Compute ATR as a full series."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period).mean()


def build_features(prices, vix_data, vix_ts, config, include_vix_ts_features=False):
    """
    Build feature panel for LGBM sector ranking.

    18 legacy features + optionally 3 VIX term structure features.
    """
    sector_etfs = config["sector_etfs"]
    benchmark = config["benchmark"]
    spy_close = prices[benchmark]["close"]
    vix_series = vix_data["^VIX"]

    frames = []
    for ticker in sector_etfs:
        if ticker not in prices:
            continue
        df = prices[ticker].copy()
        close = df["close"]
        volume = df["volume"]

        feat = pd.DataFrame(index=df.index)
        feat["ticker"] = ticker

        # Momentum features (4)
        for lookback in [5, 10, 21, 63]:
            feat[f"mom_{lookback}d"] = close.pct_change(lookback)

        # Relative strength vs SPY (2)
        for lookback in [21, 63]:
            spy_ret = spy_close.pct_change(lookback).reindex(close.index)
            ticker_ret = close.pct_change(lookback)
            feat[f"rs_vs_spy_{lookback}d"] = ticker_ret - spy_ret

        # Volatility features (3)
        feat["vol_21d"] = close.pct_change().rolling(21).std() * np.sqrt(252)
        feat["vol_63d"] = close.pct_change().rolling(63).std() * np.sqrt(252)
        feat["vol_ratio"] = feat["vol_21d"] / (feat["vol_63d"] + 1e-8)

        # Volume feature (1)
        feat["vol_sma_ratio"] = volume / volume.rolling(21).mean()

        # RSI (1)
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / 14, min_periods=14).mean()
        avg_loss = loss.ewm(alpha=1 / 14, min_periods=14).mean()
        feat["rsi_14"] = 100 - (100 / (1 + avg_gain / (avg_loss + 1e-8)))

        # Moving average distance (2)
        feat["sma_50_dist"] = close / close.rolling(50).mean() - 1
        feat["sma_200_dist"] = close / close.rolling(200).mean() - 1

        # VIX features (already in baseline: 3)
        feat["vix_level"] = vix_series.reindex(df.index, method="ffill")
        feat["vix_sma_ratio"] = feat["vix_level"] / feat["vix_level"].rolling(21).mean()
        feat["vix_pctile_63d"] = feat["vix_level"].rolling(63).rank(pct=True)

        # === VIX TERM STRUCTURE FEATURES (3 additional) ===
        if include_vix_ts_features and vix_ts is not None:
            feat["vix_ratio"] = vix_ts["vix_ratio"].reindex(df.index, method="ffill")
            feat["vix_slope"] = vix_ts["vix_slope"].reindex(df.index, method="ffill")
            feat["vix_slope_change_5d"] = vix_ts["vix_slope_change_5d"].reindex(df.index, method="ffill")

        # Target + meta columns
        feat["target"] = close.pct_change(21).shift(-21)
        feat["close"] = close
        feat["atr_14"] = _compute_atr_series(df["high"], df["low"], close)
        feat["vix"] = vix_series.reindex(df.index, method="ffill")

        frames.append(feat)

    panel = pd.concat(frames)
    panel = panel.dropna(subset=["target"])
    panel = panel.reset_index().rename(columns={"index": "date", "Date": "date"})
    if "date" not in panel.columns:
        panel["date"] = panel.index

    feature_cols = [c for c in panel.columns if c not in
                    ["date", "ticker", "target", "close", "atr_14", "vix"]]
    fprint(f"  Feature panel: {len(panel)} rows, {len(feature_cols)} features")
    if include_vix_ts_features:
        fprint(f"  VIX term structure features included: vix_ratio, vix_slope, vix_slope_change_5d")

    return panel


# =====================================================================
# 6. WALK-FORWARD LGBM
# =====================================================================

def walk_forward_lgbm(panel, config):
    """Walk-forward LGBM ranking with sliding window."""
    import lightgbm as lgb

    feature_cols = [c for c in panel.columns if c not in
                    ["date", "ticker", "target", "close", "atr_14", "vix"]]

    panel = panel.sort_values("date").reset_index(drop=True)
    dates = sorted(panel["date"].unique())

    date_to_idx = {d: i for i, d in enumerate(dates)}
    panel["_date_idx"] = panel["date"].map(date_to_idx)
    panel = panel.sort_values("_date_idx")

    train_months = config.get("train_months", 12)
    test_months = config.get("test_months", 1)
    train_days = train_months * 21
    test_days = test_months * 21

    feat_vals = np.nan_to_num(panel[feature_cols].values.astype(np.float32),
                               nan=0, posinf=0, neginf=0)
    target_vals = np.nan_to_num(panel["target"].values.astype(np.float32),
                                 nan=0, posinf=0, neginf=0)
    date_idx_vals = panel["_date_idx"].values

    predictions = []
    n_folds = 0
    feature_importance = np.zeros(len(feature_cols))

    for i in range(train_days, len(dates), max(1, test_days)):
        if i >= len(dates):
            break

        train_start = max(0, i - train_days)
        test_end = min(len(dates), i + test_days)

        if i - train_start < 100:
            continue

        train_mask = (date_idx_vals >= train_start) & (date_idx_vals < i)
        test_mask = (date_idx_vals >= i) & (date_idx_vals < test_end)

        if not test_mask.any():
            continue

        X_train = feat_vals[train_mask]
        y_train = target_vals[train_mask]
        X_test = feat_vals[test_mask]

        try:
            dtrain = lgb.Dataset(X_train, label=y_train)
            val_start = max(0, len(X_train) - len(X_train) // 5)
            dval = lgb.Dataset(X_train[val_start:], label=y_train[val_start:])

            model = lgb.train(
                config.get("lgbm_params", CONFIG["lgbm_params"]),
                dtrain,
                num_boost_round=config.get("num_boost_round", 300),
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(
                    config.get("early_stopping_rounds", 30),
                    verbose=False,
                )],
            )

            preds = model.predict(X_test)
            test_data = panel.loc[test_mask, ["date", "ticker", "close", "atr_14", "vix"]].copy()
            test_data["pred"] = preds
            predictions.append(test_data)
            feature_importance += model.feature_importance(importance_type="gain")
            n_folds += 1

            if n_folds % 20 == 0:
                fprint(f"    ... fold {n_folds}, date up to {dates[min(test_end-1, len(dates)-1)]}")

        except Exception:
            continue

    panel.drop(columns=["_date_idx"], inplace=True, errors="ignore")

    if not predictions:
        return pd.DataFrame(), {}

    result = pd.concat(predictions, ignore_index=True)
    fprint(f"  Walk-forward: {n_folds} folds, {len(result)} predictions")

    # Feature importance
    if n_folds > 0:
        fi = feature_importance / n_folds
        fi_dict = dict(sorted(zip(feature_cols, fi), key=lambda x: -x[1]))
        fprint(f"\n  Top 10 feature importances:")
        for i, (fname, fval) in enumerate(list(fi_dict.items())[:10]):
            fprint(f"    {i+1}. {fname}: {fval:.1f}")
    else:
        fi_dict = {}

    return result, fi_dict


# =====================================================================
# 7. COMPUTE IC (Information Coefficient)
# =====================================================================

def compute_sector_ic(predictions):
    """Compute rank IC (Spearman) for sector ranking predictions."""
    from scipy.stats import spearmanr

    predictions = predictions.copy()
    predictions["fwd_21d"] = np.nan

    # Group by date, compute rank correlation
    ics = []
    for date, group in predictions.groupby("date"):
        if len(group) < 5:
            continue
        # Actual 21d forward return (already in target via price)
        # We need to compute it from close prices
        # Skip — IC will be computed from the walk-forward predictions vs actuals
        pass

    # Simple cross-sectional IC per date
    dates = sorted(predictions["date"].unique())
    for d in dates:
        grp = predictions[predictions["date"] == d]
        if len(grp) < 5:
            continue
        # For now, just return prediction stats
        pass

    return None


# =====================================================================
# 8. TRADE GENERATION
# =====================================================================

def generate_trades(
    predictions,
    prices,
    vix_data,
    vix_ts,
    regime_series,
    config,
    variant="A",
):
    """
    Generate bull call spread trades from LGBM predictions.

    Variants:
      A: Baseline — regime>0.4 filter only
      B: Same as A (features already in LGBM)
      C: Regime>0.4 AND VIX_ratio > 0.90
      D: Regime>0.4 AND VIX_ratio > 0.95
      E: Regime>0.4 AND VIX slope flattening (5d change < 0)
      F: Adaptive position sizing based on VIX term structure
    """
    fprint(f"\n  Generating trades for variant {variant}...")

    vix_series = vix_data["^VIX"]
    spy_close = prices[config["benchmark"]]["close"]
    top_n = config["top_n"]
    dte = config["target_dte"]
    rebal_days = config["rebalance_freq_days"]
    spread_pct = config["spread_width_pct"]
    initial_capital = config["initial_capital"]
    max_trade_pct = config["max_trade_pct"]

    predictions = predictions.sort_values("date")
    trade_dates = sorted(predictions["date"].unique())

    trades = []
    equity = initial_capital
    last_trade_date = None

    for trade_date in trade_dates:
        # Rebalance frequency
        if last_trade_date is not None:
            days_since = (pd.Timestamp(trade_date) - pd.Timestamp(last_trade_date)).days
            if days_since < rebal_days:
                continue

        # GRU regime filter (all variants use it)
        if regime_series is not None:
            regime_date = pd.Timestamp(trade_date)
            if regime_date in regime_series.index:
                regime_score = regime_series.loc[regime_date]
            else:
                # Find closest previous date
                valid_dates = regime_series.index[regime_series.index <= regime_date]
                if len(valid_dates) == 0:
                    continue
                regime_score = regime_series.loc[valid_dates[-1]]

            if regime_score < config["regime_threshold"]:
                continue

        # VIX term structure filters (variant-specific)
        ts_date = pd.Timestamp(trade_date)
        if vix_ts is not None and ts_date in vix_ts.index:
            vix_ratio = vix_ts.loc[ts_date, "vix_ratio"]
            vix_slope_change = vix_ts.loc[ts_date, "vix_slope_change_5d"]
        elif vix_ts is not None:
            valid = vix_ts.index[vix_ts.index <= ts_date]
            if len(valid) > 0:
                vix_ratio = vix_ts.loc[valid[-1], "vix_ratio"]
                vix_slope_change = vix_ts.loc[valid[-1], "vix_slope_change_5d"]
            else:
                vix_ratio = 0.9
                vix_slope_change = 0.0
        else:
            vix_ratio = 0.9
            vix_slope_change = 0.0

        # Apply variant-specific filters
        if variant == "C":
            if np.isnan(vix_ratio) or vix_ratio <= 0.90:
                continue
        elif variant == "D":
            if np.isnan(vix_ratio) or vix_ratio <= 0.95:
                continue
        elif variant == "E":
            if np.isnan(vix_slope_change) or vix_slope_change >= 0:
                continue

        # Variant F: adaptive sizing
        if variant == "F":
            if np.isnan(vix_ratio):
                size_mult = 1.0
            elif vix_ratio > 1.10:
                size_mult = 1.5  # bigger in backwardation (fear = opportunity)
            elif vix_ratio > 1.0:
                size_mult = 1.25
            elif vix_ratio > 0.90:
                size_mult = 1.0
            else:
                size_mult = 0.75  # smaller in deep contango
        else:
            size_mult = 1.0

        # Get top-N ranked sectors
        day_preds = predictions[predictions["date"] == trade_date].copy()
        day_preds = day_preds.sort_values("pred", ascending=False)
        top_sectors = day_preds.head(top_n)

        if len(top_sectors) == 0:
            continue

        # Get VIX level for pricing
        vix_val = float(vix_series.reindex([pd.Timestamp(trade_date)], method="ffill").iloc[0]) \
            if pd.Timestamp(trade_date) in vix_series.index or len(vix_series) > 0 else 20.0
        try:
            vix_val = float(vix_series.asof(pd.Timestamp(trade_date)))
        except Exception:
            vix_val = 20.0

        for _, row in top_sectors.iterrows():
            ticker = row["ticker"]
            spot = row["close"]
            atr = row["atr_14"]

            if pd.isna(spot) or pd.isna(atr) or spot <= 0:
                continue

            # Bull call spread: ATM to ATM+3%
            K1 = round(spot, 2)
            K2 = round(spot * (1 + spread_pct), 2)

            try:
                entry_cost, max_profit = price_bull_call_spread(
                    S=spot, K1=K1, K2=K2, dte=dte, atr=atr, vix=vix_val,
                    haircut=config["haircut"],
                )
            except Exception:
                continue

            # Position sizing
            cost_per_contract = entry_cost * 100 + config["commission_rt"]
            if cost_per_contract <= 0:
                continue

            max_trade_capital = equity * max_trade_pct * size_mult
            contracts = max(1, int(max_trade_capital / cost_per_contract))
            contracts = min(contracts, 5)  # cap

            # Expiry price
            entry_dt = pd.Timestamp(trade_date)
            exit_dt = entry_dt + pd.Timedelta(days=dte)

            if ticker not in prices:
                continue

            ticker_close = prices[ticker]["close"]
            exit_prices = ticker_close[ticker_close.index <= exit_dt]
            if len(exit_prices) == 0:
                continue
            expiry_price = float(exit_prices.iloc[-1])
            actual_exit_dt = exit_prices.index[-1]

            # PnL at expiry (hold to expiry)
            exit_val = exit_spread_value(
                S=expiry_price, K1=K1, K2=K2,
                remaining_dte=0, original_dte=dte,
                atr=atr, vix=vix_val,
            )

            pnl = spread_pnl(entry_cost, exit_val, contracts=contracts,
                              commission_rt=config["commission_rt"])

            equity += pnl

            # Determine regime for trade
            spy_entry = spy_close.asof(entry_dt)
            spy_exit = spy_close.asof(actual_exit_dt)
            if spy_exit >= spy_entry:
                regime_label = "bull"
            else:
                regime_label = "bear"

            trades.append({
                "pnl": pnl,
                "entry_date": str(entry_dt.date()),
                "exit_date": str(actual_exit_dt.date()),
                "ticker": ticker,
                "spot": spot,
                "K1": K1,
                "K2": K2,
                "entry_cost": entry_cost,
                "exit_val": exit_val,
                "contracts": contracts,
                "vix": vix_val,
                "vix_ratio": vix_ratio,
                "regime": regime_label,
            })

        last_trade_date = trade_date

    fprint(f"  Variant {variant}: {len(trades)} trades, final equity: ${equity:.2f}")
    return trades


# =====================================================================
# 9. RUN ALL VARIANTS
# =====================================================================

def run_all_variants(prices, vix_data, vix_ts, regime_series, config):
    """Run all 6 backtest variants."""
    fprint("\n" + "=" * 70)
    fprint("  RUNNING 6 BACKTEST VARIANTS")
    fprint("=" * 70)

    spy_close = prices[config["benchmark"]]["close"]

    results = {}

    # --- Variants A, C, D, E, F use baseline features (18 legacy) ---
    fprint("\n[5/8] Building baseline features (18 legacy)...")
    panel_baseline = build_features(prices, vix_data, vix_ts, config, include_vix_ts_features=False)

    fprint("\n[6/8] Walk-forward LGBM (baseline features)...")
    preds_baseline, fi_baseline = walk_forward_lgbm(panel_baseline, config)

    if len(preds_baseline) == 0:
        fprint("  ERROR: No predictions from baseline LGBM")
        return {}

    for variant in ["A", "C", "D", "E", "F"]:
        trades = generate_trades(
            preds_baseline, prices, vix_data, vix_ts, regime_series, config,
            variant=variant,
        )

        if len(trades) < 10:
            fprint(f"  Variant {variant}: too few trades ({len(trades)}), skipping validation")
            results[variant] = {
                "trades": trades,
                "validation": None,
                "feature_importance": fi_baseline,
            }
            continue

        val_result = validate_trades(
            trades=trades,
            initial_capital=config["initial_capital"],
            spy_prices=spy_close,
            strategy_name=f"VIX_TS_v1_{variant}",
            n_perms=config["n_perms"],
        )
        val_result.print_summary()
        results[variant] = {
            "trades": trades,
            "validation": val_result,
            "feature_importance": fi_baseline,
        }

    # --- Variant B: baseline + VIX term structure features ---
    fprint("\n[7/8] Building enhanced features (18 legacy + 3 VIX TS)...")
    panel_enhanced = build_features(prices, vix_data, vix_ts, config, include_vix_ts_features=True)

    fprint("\n  Walk-forward LGBM (enhanced features)...")
    preds_enhanced, fi_enhanced = walk_forward_lgbm(panel_enhanced, config)

    if len(preds_enhanced) > 0:
        trades_b = generate_trades(
            preds_enhanced, prices, vix_data, vix_ts, regime_series, config,
            variant="A",  # same filtering as A, but with enhanced features
        )

        if len(trades_b) >= 10:
            val_b = validate_trades(
                trades=trades_b,
                initial_capital=config["initial_capital"],
                spy_prices=spy_close,
                strategy_name="VIX_TS_v1_B (enhanced features)",
                n_perms=config["n_perms"],
            )
            val_b.print_summary()
            results["B"] = {
                "trades": trades_b,
                "validation": val_b,
                "feature_importance": fi_enhanced,
            }
        else:
            fprint(f"  Variant B: too few trades ({len(trades_b)})")
            results["B"] = {"trades": trades_b, "validation": None, "feature_importance": fi_enhanced}
    else:
        fprint("  Variant B: no predictions")
        results["B"] = {"trades": [], "validation": None, "feature_importance": fi_enhanced}

    return results


# =====================================================================
# 10. VIX TS vs GRU REGIME — INDEPENDENCE CHECK
# =====================================================================

def check_vix_gru_independence(vix_ts, regime_series):
    """Check if VIX term structure adds info beyond GRU regime score."""
    fprint("\n[8/8] VIX Term Structure vs GRU Regime Independence...")

    if regime_series is None or vix_ts is None:
        fprint("  Cannot check — missing one or both signals")
        return

    # Align
    common = vix_ts.index.intersection(regime_series.index)
    if len(common) < 100:
        fprint(f"  Only {len(common)} overlapping days, insufficient")
        return

    vr = vix_ts["vix_ratio"].reindex(common)
    rs = regime_series.reindex(common)

    # Correlation
    corr = vr.corr(rs)
    fprint(f"  Correlation(VIX_ratio, GRU_regime_score): {corr:.3f}")

    # Rank correlation
    from scipy.stats import spearmanr
    rank_corr, p_val = spearmanr(vr.dropna(), rs.reindex(vr.dropna().index).dropna())
    fprint(f"  Spearman rank correlation: {rank_corr:.3f} (p={p_val:.4f})")

    # Joint regime table
    fprint("\n  Joint Distribution (VIX regime vs GRU regime active/inactive):")
    fprint("  " + "-" * 65)
    gru_active = rs > 0.4
    for vix_regime in ["deep_contango", "contango", "flat", "backwardation", "deep_backwardation"]:
        mask = vix_ts["vix_regime"].reindex(common) == vix_regime
        if mask.sum() < 5:
            continue
        gru_active_pct = gru_active.loc[mask].mean() * 100
        n = mask.sum()
        fprint(f"  {vix_regime:>22s}: GRU active in {gru_active_pct:5.1f}% of {n} days")

    # Information content: what if we require BOTH?
    both_active_count = ((vr > 0.90) & gru_active).sum()
    gru_only_count = gru_active.sum()
    fprint(f"\n  GRU active days: {gru_only_count}")
    fprint(f"  GRU active AND VIX_ratio > 0.90: {both_active_count} "
           f"({both_active_count / max(gru_only_count, 1) * 100:.1f}% of GRU active)")
    fprint(f"  GRU active AND VIX_ratio > 0.95: {((vr > 0.95) & gru_active).sum()}")
    fprint(f"  GRU active AND slope flattening: {((vix_ts['vix_slope_change_5d'].reindex(common) < 0) & gru_active).sum()}")


# =====================================================================
# 11. COMPARISON SUMMARY
# =====================================================================

def comparison_summary(results, config):
    """Print side-by-side comparison of all variants."""
    fprint("\n" + "=" * 90)
    fprint("  VARIANT COMPARISON SUMMARY")
    fprint("=" * 90)

    variant_names = {
        "A": "Baseline (regime>0.4)",
        "B": "+3 VIX TS features in LGBM",
        "C": "+VIX_ratio > 0.90 filter",
        "D": "+VIX_ratio > 0.95 filter",
        "E": "+VIX slope flattening filter",
        "F": "Adaptive sizing (VIX TS)",
    }

    header = f"  {'Variant':>6s} | {'Description':>32s} | {'Trades':>6s} | {'Sharpe':>7s} | {'Sortino':>7s} | {'PF':>5s} | {'WR':>5s} | {'MaxDD':>7s} | {'CAGR':>7s} | {'Gates':>5s} | {'Final$':>8s}"
    fprint(header)
    fprint("  " + "-" * (len(header) - 2))

    best_sharpe = -999
    best_variant = None

    for var in ["A", "B", "C", "D", "E", "F"]:
        if var not in results:
            continue
        val = results[var].get("validation")
        trades = results[var].get("trades", [])
        desc = variant_names.get(var, var)

        if val is None:
            fprint(f"  {var:>6s} | {desc:>32s} | {len(trades):>6d} | {'N/A':>7s} | {'N/A':>7s} | {'N/A':>5s} | {'N/A':>5s} | {'N/A':>7s} | {'N/A':>7s} | {'N/A':>5s} | {'N/A':>8s}")
            continue

        gates = f"{val.gates_passed}/{val.gates_total}"
        fprint(f"  {var:>6s} | {desc:>32s} | {val.n_trades:>6d} | {val.sharpe:>7.2f} | {val.sortino:>7.2f} | {val.profit_factor:>5.2f} | {val.win_rate*100:>4.1f}% | {val.max_dd*100:>6.1f}% | {val.cagr*100:>6.1f}% | {gates:>5s} | ${val.final_equity:>7.0f}")

        if val.sharpe > best_sharpe:
            best_sharpe = val.sharpe
            best_variant = var

    fprint("  " + "-" * (len(header) - 2))
    if best_variant:
        fprint(f"  BEST: Variant {best_variant} ({variant_names.get(best_variant, '')}) — Sharpe {best_sharpe:.2f}")

    return best_variant


# =====================================================================
# 12. MLFLOW LOGGING
# =====================================================================

def log_to_mlflow(results, vix_ts, config):
    """Log all results to MLflow."""
    if not MLFLOW_OK:
        fprint("\n  MLflow not available, skipping logging")
        return

    import mlflow

    try:
        mlflow.set_experiment("vix_term_structure_v1")
    except Exception as e:
        fprint(f"  MLflow experiment setup failed: {e}")
        return

    variant_names = {
        "A": "baseline_regime_only",
        "B": "enhanced_vix_ts_features",
        "C": "vix_ratio_gt_0.90",
        "D": "vix_ratio_gt_0.95",
        "E": "slope_flattening",
        "F": "adaptive_sizing",
    }

    for var in ["A", "B", "C", "D", "E", "F"]:
        if var not in results:
            continue
        val = results[var].get("validation")
        trades = results[var].get("trades", [])

        with mlflow.start_run(run_name=f"variant_{var}_{variant_names.get(var, '')}"):
            mlflow.log_param("variant", var)
            mlflow.log_param("variant_description", variant_names.get(var, ""))
            mlflow.log_param("n_trades", len(trades))
            mlflow.log_param("initial_capital", config["initial_capital"])
            mlflow.log_param("target_dte", config["target_dte"])
            mlflow.log_param("spread_width_pct", config["spread_width_pct"])
            mlflow.log_param("regime_threshold", config["regime_threshold"])
            mlflow.log_param("haircut", config["haircut"])

            if val is not None:
                mlflow.log_metric("sharpe", val.sharpe)
                mlflow.log_metric("sortino", val.sortino)
                mlflow.log_metric("cagr", val.cagr)
                mlflow.log_metric("max_dd", val.max_dd)
                mlflow.log_metric("win_rate", val.win_rate)
                mlflow.log_metric("profit_factor", val.profit_factor)
                mlflow.log_metric("final_equity", val.final_equity)
                mlflow.log_metric("gates_passed", val.gates_passed)
                mlflow.log_metric("gates_total", val.gates_total)
                mlflow.log_metric("all_passed", 1 if val.all_passed else 0)

            # Log feature importance for this variant
            fi = results[var].get("feature_importance", {})
            for fname, fval in list(fi.items())[:20]:
                safe_name = fname.replace("/", "_")
                mlflow.log_metric(f"fi_{safe_name}", fval)

    fprint("  MLflow logging complete")


# =====================================================================
# MAIN
# =====================================================================

def main():
    fprint("=" * 70)
    fprint("  VIX TERM STRUCTURE RESEARCH v1")
    fprint(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 70)

    config = CONFIG.copy()

    # 1. Load data
    prices, vix_data = load_market_data(config)

    if "^VIX" not in vix_data or "^VIX3M" not in vix_data:
        fprint("FATAL: Cannot proceed without ^VIX and ^VIX3M data")
        return

    # 2. Build VIX term structure
    vix_ts = build_vix_term_structure(vix_data, config)
    if vix_ts is None:
        fprint("FATAL: Failed to build VIX term structure")
        return

    # 3. Descriptive analysis
    vix_ts = descriptive_analysis(vix_ts, prices, config)

    # 4. Load GRU regime
    regime_series = load_regime_predictions()

    # 5-7. Run all variants
    results = run_all_variants(prices, vix_data, vix_ts, regime_series, config)

    if not results:
        fprint("FATAL: No results from any variant")
        return

    # 8. Independence check
    check_vix_gru_independence(vix_ts, regime_series)

    # 9. Comparison summary
    best = comparison_summary(results, config)

    # 10. MLflow logging
    log_to_mlflow(results, vix_ts, config)

    # 11. Save results to disk
    output_dir = BASE / "output" / "vix_term_structure_v1"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for var in ["A", "B", "C", "D", "E", "F"]:
        if var not in results:
            continue
        val = results[var].get("validation")
        if val is not None:
            summary[var] = val.to_dict()
        else:
            summary[var] = {"n_trades": len(results[var].get("trades", [])), "error": "too few trades"}

    with open(output_dir / "variant_comparison.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Save VIX term structure data
    vix_ts.to_csv(output_dir / "vix_term_structure_data.csv")

    fprint(f"\n  Results saved to {output_dir}")
    fprint(f"\n  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 70)


if __name__ == "__main__":
    main()
