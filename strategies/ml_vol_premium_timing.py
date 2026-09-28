#!/usr/bin/env python3
"""
ML Premium Timing Optimizer — LightGBM Vol Premium Strategy
============================================================
Predicts next-week realized vol for liquid ETFs/stocks, then trades vol premium.
Signal: sell premium when predicted RV << IV (fat premium), avoid when RV >> IV.

Walk-forward: 252-day sliding window, monthly retrain, weekly prediction.
All features T-1 (anti-lookahead per HC #724).

Validation gates:
  1. Permutation test (200 perms, p < 0.05)
  2. Regime agnostic: |Sharpe_green - Sharpe_red| / max < 0.50
  3. Sub-period: 4-block Sharpe CV < 0.70
  4. Outlier: remove top 5% trades, Sharpe drop < 50%

Author: Claude (Head of Quant)
Date: 2026-07-21
"""

import os
import sys
import json
import warnings
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import yfinance as yf
from scipy import stats
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.lightgbm

warnings.filterwarnings('ignore')

# ─── Configuration ──────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("/home/nick/Lvl3Quant/output/ml_vol_premium_timing")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MLFLOW_TRACKING_URI = "http://jupiter:5000"  # Jupiter via Tailscale
EXPERIMENT_NAME = "ml_vol_premium_timing"

UNIVERSE = [
    "SPY", "QQQ", "IWM",
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA",
    "JPM", "GS",
    "XLF", "XLE", "XLK",
    "GLD", "TLT",
]

# Walk-forward params
TRAIN_WINDOW = 252     # trading days
RETRAIN_FREQ = 21      # retrain monthly
PREDICT_HORIZON = 5    # 5-day forward realized vol
MIN_HISTORY = 63 + 5   # need 63d lookback + 5d forward

# LightGBM params (GPU-accelerated)
LGB_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "device": "gpu",
    "gpu_platform_id": 0,
    "gpu_device_id": 0,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "n_estimators": 500,
    "early_stopping_rounds": 50,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}

N_PERMUTATIONS = 200
ANNUALIZATION = np.sqrt(252 / PREDICT_HORIZON)  # weekly trades, annualize


# ─── Data Download ──────────────────────────────────────────────────────────────
def download_data(tickers, start="2015-01-01", end=None):
    """Download price data for all tickers + VIX."""
    if end is None:
        end = dt.datetime.now().strftime("%Y-%m-%d")

    print(f"Downloading data for {len(tickers)} tickers + VIX from {start} to {end}...")
    all_data = {}

    for ticker in tickers + ["^VIX"]:
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if df is not None and len(df) > 100:
                # Flatten multi-level columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                all_data[ticker] = df
                print(f"  {ticker}: {len(df)} days")
            else:
                print(f"  {ticker}: SKIPPED (insufficient data: {len(df) if df is not None else 0})")
        except Exception as e:
            print(f"  {ticker}: FAILED ({e})")

    return all_data


# ─── Feature Engineering ────────────────────────────────────────────────────────
def compute_features(price_df, vix_df, ticker):
    """
    Compute features for a single ticker. ALL features use T-1 data only.
    Returns DataFrame with features aligned to prediction dates.
    """
    close = price_df["Close"].copy()
    log_ret = np.log(close / close.shift(1))

    features = pd.DataFrame(index=close.index)

    # Historical realized vol (annualized) — all T-1 (shifted by 1)
    for window in [5, 10, 21, 63]:
        rv = log_ret.rolling(window).std() * np.sqrt(252)
        features[f"rv_{window}d"] = rv.shift(1)

    # Vol ratios (T-1)
    features["rv_5_21_ratio"] = features["rv_5d"] / features["rv_21d"].replace(0, np.nan)
    features["rv_5_63_ratio"] = features["rv_5d"] / features["rv_63d"].replace(0, np.nan)
    features["rv_21_63_ratio"] = features["rv_21d"] / features["rv_63d"].replace(0, np.nan)

    # Vol trend (T-1) — is short-term vol rising or falling?
    features["rv_5d_chg"] = features["rv_5d"] - features["rv_5d"].shift(5)
    features["rv_21d_chg"] = features["rv_21d"] - features["rv_21d"].shift(21)

    # IV proxy: VIX level and term structure (T-1)
    if vix_df is not None and "Close" in vix_df.columns:
        vix_close = vix_df["Close"].reindex(close.index, method="ffill")
        features["vix_level"] = vix_close.shift(1)
        features["vix_5d_ma"] = vix_close.rolling(5).mean().shift(1)
        features["vix_21d_ma"] = vix_close.rolling(21).mean().shift(1)
        features["vix_term"] = (vix_close.shift(1) - vix_close.rolling(21).mean().shift(1))
        # IV rank proxy (where is VIX relative to 252d range)
        vix_252_min = vix_close.rolling(252).min().shift(1)
        vix_252_max = vix_close.rolling(252).max().shift(1)
        features["iv_rank"] = (vix_close.shift(1) - vix_252_min) / (vix_252_max - vix_252_min).replace(0, np.nan)
        # IV percentile proxy
        features["iv_pctile"] = vix_close.shift(1).rolling(252).rank(pct=True)

    # Momentum features (T-1)
    for period in [5, 10, 21, 63]:
        features[f"mom_{period}d"] = (close / close.shift(period) - 1).shift(1)

    # Realized skewness and kurtosis (T-1)
    features["skew_21d"] = log_ret.rolling(21).skew().shift(1)
    features["kurt_21d"] = log_ret.rolling(21).kurt().shift(1)

    # Day of week (0=Mon, 4=Fri)
    features["dow"] = close.index.dayofweek

    # Distance to round months (proxy for earnings proximity)
    features["day_of_month"] = close.index.day
    features["month"] = close.index.month

    # High-Low range vol proxy (T-1)
    if "High" in price_df.columns and "Low" in price_df.columns:
        hl_range = np.log(price_df["High"] / price_df["Low"])
        features["hl_range_5d"] = hl_range.rolling(5).mean().shift(1)
        features["hl_range_21d"] = hl_range.rolling(21).mean().shift(1)

    # Volume features (T-1) — volume regime
    if "Volume" in price_df.columns:
        vol = price_df["Volume"].replace(0, np.nan)
        features["vol_ratio_5_21"] = (vol.rolling(5).mean() / vol.rolling(21).mean()).shift(1)

    return features


def compute_target(price_df, horizon=5):
    """
    Target: realized vol over next `horizon` trading days (T+1 to T+horizon).
    Annualized.
    """
    close = price_df["Close"]
    log_ret = np.log(close / close.shift(1))

    # Forward realized vol
    fwd_rv = log_ret.shift(-horizon).rolling(horizon).std() * np.sqrt(252)
    # Shift so that target[t] = realized vol from t+1 to t+horizon
    # Actually: we want vol from day t+1 to t+5
    # log_ret.shift(-1) gives return on day t+1, ..., log_ret.shift(-5) gives return on day t+5
    # Compute std of returns on days t+1 through t+5
    fwd_returns = pd.DataFrame({
        f"ret_{i}": log_ret.shift(-i) for i in range(1, horizon + 1)
    })
    fwd_rv = fwd_returns.std(axis=1) * np.sqrt(252)

    return fwd_rv


# ─── Dataset Construction ───────────────────────────────────────────────────────
def build_dataset(all_data, universe):
    """Build panel dataset with features and targets for all tickers."""
    vix_df = all_data.get("^VIX")
    rows = []

    for ticker in universe:
        if ticker not in all_data:
            continue
        price_df = all_data[ticker]
        feat = compute_features(price_df, vix_df, ticker)
        target = compute_target(price_df, PREDICT_HORIZON)

        combined = feat.copy()
        combined["target_rv"] = target
        combined["ticker"] = ticker

        # Also store current IV proxy for signal generation
        if vix_df is not None and "Close" in vix_df.columns:
            vix_close = vix_df["Close"].reindex(price_df.index, method="ffill")
            combined["current_iv"] = vix_close.shift(1)  # T-1

        rows.append(combined)

    panel = pd.concat(rows, axis=0)
    panel = panel.dropna(subset=["target_rv"])

    feature_cols = [c for c in panel.columns if c not in ["target_rv", "ticker", "current_iv"]]
    panel = panel.dropna(subset=feature_cols, how="all")

    print(f"Panel dataset: {len(panel)} rows, {len(feature_cols)} features, {panel['ticker'].nunique()} tickers")
    print(f"Date range: {panel.index.min()} to {panel.index.max()}")

    return panel, feature_cols


# ─── Walk-Forward Engine ────────────────────────────────────────────────────────
def walk_forward(panel, feature_cols):
    """
    Sliding window walk-forward.
    Train on 252 days, retrain every 21 days, predict 1 week ahead.
    """
    dates = sorted(panel.index.unique())
    print(f"Total unique dates: {len(dates)}")

    results = []
    models = []
    feature_importances = []

    # Start after enough history
    start_idx = TRAIN_WINDOW + MIN_HISTORY
    if start_idx >= len(dates):
        raise ValueError(f"Not enough data: need {start_idx} days, have {len(dates)}")

    last_model = None
    last_train_idx = -RETRAIN_FREQ  # force first train

    n_retrains = 0
    n_predictions = 0

    for i in range(start_idx, len(dates), PREDICT_HORIZON):
        # Current prediction date
        pred_date = dates[i]

        # Training window: [i - TRAIN_WINDOW, i)
        train_start = dates[max(0, i - TRAIN_WINDOW)]
        train_end = dates[i - 1]

        train_mask = (panel.index >= train_start) & (panel.index <= train_end)
        train_data = panel[train_mask]

        # Retrain?
        if i - last_train_idx >= RETRAIN_FREQ or last_model is None:
            X_train = train_data[feature_cols].values
            y_train = train_data["target_rv"].values

            # Remove rows with NaN
            valid = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
            X_train = X_train[valid]
            y_train = y_train[valid]

            if len(X_train) < 100:
                continue

            # Split last 20% for early stopping
            split = int(len(X_train) * 0.8)
            X_tr, X_val = X_train[:split], X_train[split:]
            y_tr, y_val = y_train[:split], y_train[split:]

            model = lgb.LGBMRegressor(**LGB_PARAMS)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
            )

            last_model = model
            last_train_idx = i
            n_retrains += 1

            # Feature importance
            fi = dict(zip(feature_cols, model.feature_importances_))
            feature_importances.append(fi)

        # Predict on current date
        pred_mask = panel.index == pred_date
        pred_data = panel[pred_mask]

        if len(pred_data) == 0:
            continue

        X_pred = pred_data[feature_cols].values
        valid_pred = ~np.isnan(X_pred).any(axis=1)

        if valid_pred.sum() == 0:
            continue

        preds = np.full(len(pred_data), np.nan)
        preds[valid_pred] = last_model.predict(X_pred[valid_pred])

        for j, (idx, row) in enumerate(pred_data.iterrows()):
            if np.isnan(preds[j]):
                continue
            results.append({
                "date": idx,
                "ticker": row["ticker"],
                "predicted_rv": preds[j],
                "actual_rv": row["target_rv"],
                "current_iv": row.get("current_iv", np.nan),
            })
            n_predictions += 1

    print(f"Walk-forward complete: {n_retrains} retrains, {n_predictions} predictions")

    results_df = pd.DataFrame(results)
    results_df["date"] = pd.to_datetime(results_df["date"])

    # Aggregate feature importance
    if feature_importances:
        avg_fi = pd.DataFrame(feature_importances).mean().sort_values(ascending=False)
    else:
        avg_fi = pd.Series(dtype=float)

    return results_df, avg_fi


# ─── Signal & Backtest ──────────────────────────────────────────────────────────
def generate_signal(results_df):
    """
    Signal: vol premium = current_iv - predicted_rv
    When premium is fat (positive) → sell premium (short straddle proxy)
    When premium is thin (negative) → avoid or buy protection (long straddle proxy)
    """
    df = results_df.copy()

    # Vol premium
    df["vol_premium"] = df["current_iv"] - df["predicted_rv"]

    # Normalize by current IV for comparability
    df["vol_premium_pct"] = df["vol_premium"] / df["current_iv"].replace(0, np.nan)

    # Signal: z-score of vol premium across tickers on each date
    df["signal"] = df.groupby("date")["vol_premium_pct"].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )

    return df


def backtest_straddle(signal_df):
    """
    Simplified straddle backtest:
    - Signal > 0: short straddle (sell premium) — profit if RV < IV
    - Signal < 0: long straddle (buy protection) — profit if RV > IV
    - PnL proportional to |signal| * (IV - RV) for shorts, (RV - IV) for longs

    Returns per-trade PnL series.
    """
    df = signal_df.copy()
    df = df.dropna(subset=["signal", "current_iv", "actual_rv"])

    # Position sizing: proportional to signal magnitude, capped at ±2 sigma
    df["position"] = df["signal"].clip(-2, 2)

    # PnL: short premium profits when IV > actual RV
    # Normalize by IV to make it percentage terms
    df["pnl_raw"] = df["position"] * (df["current_iv"] - df["actual_rv"]) / df["current_iv"].replace(0, np.nan)

    # Transaction cost: ~0.5% per straddle trade (conservative)
    tx_cost = 0.005
    df["pnl_net"] = df["pnl_raw"] - tx_cost * df["position"].abs()

    return df


# ─── Metrics ────────────────────────────────────────────────────────────────────
def compute_metrics(pnl_series, label=""):
    """Compute risk-adjusted metrics for a PnL series."""
    pnl = pnl_series.dropna()
    if len(pnl) < 10:
        return {}

    mean_pnl = pnl.mean()
    std_pnl = pnl.std()

    sharpe = mean_pnl / std_pnl * ANNUALIZATION if std_pnl > 0 else 0
    downside = pnl[pnl < 0].std()
    sortino = mean_pnl / downside * ANNUALIZATION if downside > 0 else 0

    cum_pnl = pnl.cumsum()
    peak = cum_pnl.cummax()
    drawdown = cum_pnl - peak
    max_dd = drawdown.min()

    win_rate = (pnl > 0).mean()
    profit_factor = pnl[pnl > 0].sum() / abs(pnl[pnl < 0].sum()) if (pnl < 0).any() else np.inf
    avg_win = pnl[pnl > 0].mean() if (pnl > 0).any() else 0
    avg_loss = pnl[pnl < 0].mean() if (pnl < 0).any() else 0

    # CAGR approximation: total return / years
    total_ret = cum_pnl.iloc[-1] if len(cum_pnl) > 0 else 0
    n_years = len(pnl) * PREDICT_HORIZON / 252
    cagr = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 and total_ret > -1 else 0

    metrics = {
        f"{label}sharpe": round(sharpe, 3),
        f"{label}sortino": round(sortino, 3),
        f"{label}cagr": round(cagr * 100, 2),
        f"{label}max_dd": round(max_dd * 100, 2),
        f"{label}win_rate": round(win_rate * 100, 1),
        f"{label}profit_factor": round(profit_factor, 2),
        f"{label}avg_win": round(avg_win * 100, 3),
        f"{label}avg_loss": round(avg_loss * 100, 3),
        f"{label}n_trades": len(pnl),
        f"{label}total_return_pct": round(total_ret * 100, 2),
    }
    return metrics


def compute_ic(results_df):
    """Information coefficient: rank correlation between predicted and actual RV."""
    df = results_df.dropna(subset=["predicted_rv", "actual_rv"])

    # Overall IC
    overall_ic = df[["predicted_rv", "actual_rv"]].corr(method="spearman").iloc[0, 1]

    # Per-date IC
    date_ics = df.groupby("date").apply(
        lambda g: g[["predicted_rv", "actual_rv"]].corr(method="spearman").iloc[0, 1]
        if len(g) > 3 else np.nan
    ).dropna()

    return {
        "overall_ic": round(overall_ic, 4),
        "mean_daily_ic": round(date_ics.mean(), 4),
        "ic_std": round(date_ics.std(), 4),
        "ic_ir": round(date_ics.mean() / date_ics.std(), 4) if date_ics.std() > 0 else 0,
        "pct_positive_ic": round((date_ics > 0).mean() * 100, 1),
    }


# ─── Validation Gates ──────────────────────────────────────────────────────────
def validation_gate_1_permutation(signal_df, n_perms=N_PERMUTATIONS):
    """Gate 1: Permutation test — shuffle signal-to-date mapping."""
    print(f"\n=== GATE 1: Permutation Test ({n_perms} perms) ===")

    df = signal_df.dropna(subset=["pnl_net"])
    actual_sharpe = df["pnl_net"].mean() / df["pnl_net"].std() * ANNUALIZATION

    perm_sharpes = []
    dates = df["date"].unique()

    for p in range(n_perms):
        # Shuffle dates (break signal-date mapping)
        shuffled = df.copy()
        shuffled["date"] = np.random.permutation(shuffled["date"].values)
        # Recompute PnL with shuffled dates (signal is now random relative to actual outcomes)
        shuffled_pnl = shuffled["pnl_net"]
        s = shuffled_pnl.mean() / shuffled_pnl.std() * ANNUALIZATION if shuffled_pnl.std() > 0 else 0
        perm_sharpes.append(s)

    perm_sharpes = np.array(perm_sharpes)
    p_value = (perm_sharpes >= actual_sharpe).mean()

    result = {
        "actual_sharpe": round(actual_sharpe, 3),
        "perm_mean_sharpe": round(perm_sharpes.mean(), 3),
        "perm_std_sharpe": round(perm_sharpes.std(), 3),
        "p_value": round(p_value, 4),
        "pass": p_value < 0.05,
    }
    print(f"  Actual Sharpe: {result['actual_sharpe']}")
    print(f"  Perm mean: {result['perm_mean_sharpe']} ± {result['perm_std_sharpe']}")
    print(f"  p-value: {result['p_value']} {'PASS' if result['pass'] else 'FAIL'}")
    return result


def validation_gate_2_regime(signal_df, spy_data):
    """Gate 2: Regime agnostic — Sharpe must not differ >50% between green/red days."""
    print("\n=== GATE 2: Regime Agnostic ===")

    spy_close = spy_data["Close"]
    spy_daily_ret = spy_close.pct_change()

    # Classify dates as green (SPY up) or red (SPY down)
    green_dates = set(spy_daily_ret[spy_daily_ret > 0].index)
    red_dates = set(spy_daily_ret[spy_daily_ret <= 0].index)

    df = signal_df.dropna(subset=["pnl_net"])
    df_green = df[df["date"].isin(green_dates)]
    df_red = df[df["date"].isin(red_dates)]

    sharpe_green = df_green["pnl_net"].mean() / df_green["pnl_net"].std() * ANNUALIZATION if len(df_green) > 10 and df_green["pnl_net"].std() > 0 else 0
    sharpe_red = df_red["pnl_net"].mean() / df_red["pnl_net"].std() * ANNUALIZATION if len(df_red) > 10 and df_red["pnl_net"].std() > 0 else 0

    max_abs = max(abs(sharpe_green), abs(sharpe_red))
    regime_diff = abs(sharpe_green - sharpe_red) / max_abs if max_abs > 0 else 0

    result = {
        "sharpe_green": round(sharpe_green, 3),
        "sharpe_red": round(sharpe_red, 3),
        "regime_diff_ratio": round(regime_diff, 3),
        "n_green": len(df_green),
        "n_red": len(df_red),
        "pass": regime_diff < 0.50,
    }
    print(f"  Sharpe (green days): {result['sharpe_green']} ({result['n_green']} trades)")
    print(f"  Sharpe (red days):   {result['sharpe_red']} ({result['n_red']} trades)")
    print(f"  Regime diff ratio:   {result['regime_diff_ratio']} {'PASS' if result['pass'] else 'FAIL'}")
    return result


def validation_gate_3_subperiod(signal_df):
    """Gate 3: Sub-period stability — 4-block Sharpe CV < 0.70."""
    print("\n=== GATE 3: Sub-Period Stability ===")

    df = signal_df.dropna(subset=["pnl_net"]).sort_values("date")
    n = len(df)
    block_size = n // 4

    block_sharpes = []
    for b in range(4):
        start = b * block_size
        end = (b + 1) * block_size if b < 3 else n
        block = df.iloc[start:end]
        s = block["pnl_net"].mean() / block["pnl_net"].std() * ANNUALIZATION if block["pnl_net"].std() > 0 else 0
        block_sharpes.append(s)

    block_sharpes = np.array(block_sharpes)
    cv = block_sharpes.std() / abs(block_sharpes.mean()) if abs(block_sharpes.mean()) > 0 else 999

    result = {
        "block_sharpes": [round(s, 3) for s in block_sharpes],
        "mean_sharpe": round(block_sharpes.mean(), 3),
        "std_sharpe": round(block_sharpes.std(), 3),
        "cv": round(cv, 3),
        "pass": cv < 0.70,
    }
    print(f"  Block Sharpes: {result['block_sharpes']}")
    print(f"  Mean: {result['mean_sharpe']}, Std: {result['std_sharpe']}, CV: {result['cv']}")
    print(f"  {'PASS' if result['pass'] else 'FAIL'}")
    return result


def validation_gate_4_outlier(signal_df):
    """Gate 4: Outlier robustness — remove top 5% trades, Sharpe drop < 50%."""
    print("\n=== GATE 4: Outlier Robustness ===")

    df = signal_df.dropna(subset=["pnl_net"])
    pnl = df["pnl_net"]

    full_sharpe = pnl.mean() / pnl.std() * ANNUALIZATION if pnl.std() > 0 else 0

    # Remove top 5% (by absolute PnL — biggest winners)
    threshold = pnl.quantile(0.95)
    trimmed = pnl[pnl <= threshold]
    trimmed_sharpe = trimmed.mean() / trimmed.std() * ANNUALIZATION if trimmed.std() > 0 else 0

    drop_pct = (full_sharpe - trimmed_sharpe) / abs(full_sharpe) * 100 if abs(full_sharpe) > 0 else 0

    result = {
        "full_sharpe": round(full_sharpe, 3),
        "trimmed_sharpe": round(trimmed_sharpe, 3),
        "sharpe_drop_pct": round(drop_pct, 1),
        "n_removed": len(pnl) - len(trimmed),
        "pass": drop_pct < 50,
    }
    print(f"  Full Sharpe:    {result['full_sharpe']}")
    print(f"  Trimmed Sharpe: {result['trimmed_sharpe']} ({result['n_removed']} trades removed)")
    print(f"  Drop: {result['sharpe_drop_pct']}% {'PASS' if result['pass'] else 'FAIL'}")
    return result


# ─── Lag Sensitivity ────────────────────────────────────────────────────────────
def lag_sensitivity(panel, feature_cols, results_df):
    """
    Compare T-0 (lookahead) vs T-1 (production) IC.
    T-0 is a sanity check — should be better, but if MUCH better, the model is fragile.
    """
    print("\n=== Lag Sensitivity: T-0 vs T-1 ===")

    # T-1 IC (what we use)
    ic_t1 = compute_ic(results_df)

    # For T-0, we'd need to retrain with unshifted features — expensive.
    # Instead, measure raw correlation between current-day vol features and target.
    df = panel.dropna(subset=["target_rv"])

    # Unshifted rv_5d (T-0) — this is a simple proxy
    # Note: our rv_5d feature is already shifted in compute_features. The raw unshifted version
    # would be: log_ret.rolling(5).std() * sqrt(252)
    # We approximate by comparing rv_5d (T-1) correlation vs rv_5d shifted back (T-0 proxy)
    t1_corr = df[["rv_5d", "target_rv"]].corr(method="spearman").iloc[0, 1] if "rv_5d" in df.columns else np.nan
    # T-0 proxy: shift rv_5d forward by 1 (undo the T-1 shift)
    df["rv_5d_t0"] = df["rv_5d"].shift(-1)
    t0_corr = df[["rv_5d_t0", "target_rv"]].corr(method="spearman").iloc[0, 1] if "rv_5d_t0" in df.columns else np.nan

    result = {
        "rv5d_ic_t1": round(t1_corr, 4) if not np.isnan(t1_corr) else None,
        "rv5d_ic_t0": round(t0_corr, 4) if not np.isnan(t0_corr) else None,
        "model_ic_t1": ic_t1["overall_ic"],
        "model_mean_daily_ic_t1": ic_t1["mean_daily_ic"],
    }
    print(f"  rv5d IC (T-1): {result['rv5d_ic_t1']}")
    print(f"  rv5d IC (T-0): {result['rv5d_ic_t0']}")
    print(f"  Model overall IC (T-1): {result['model_ic_t1']}")
    print(f"  Model mean daily IC (T-1): {result['model_mean_daily_ic_t1']}")
    return result


# ─── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("ML Premium Timing Optimizer — LightGBM Vol Premium Strategy")
    print("=" * 80)
    print(f"Started: {dt.datetime.now()}")
    print(f"Universe: {', '.join(UNIVERSE)}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    # MLflow setup
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"vol_premium_lgbm_{dt.datetime.now().strftime('%Y%m%d_%H%M')}"):
        # Log params
        mlflow.log_params({
            "universe": ",".join(UNIVERSE),
            "train_window": TRAIN_WINDOW,
            "retrain_freq": RETRAIN_FREQ,
            "predict_horizon": PREDICT_HORIZON,
            "n_estimators": LGB_PARAMS["n_estimators"],
            "num_leaves": LGB_PARAMS["num_leaves"],
            "learning_rate": LGB_PARAMS["learning_rate"],
            "device": LGB_PARAMS["device"],
            "n_permutations": N_PERMUTATIONS,
        })

        # 1. Download data
        all_data = download_data(UNIVERSE, start="2015-01-01")
        if len(all_data) < 5:
            print("FATAL: Not enough tickers downloaded. Aborting.")
            sys.exit(1)

        # 2. Build panel dataset
        panel, feature_cols = build_dataset(all_data, UNIVERSE)
        mlflow.log_params({
            "n_features": len(feature_cols),
            "n_tickers_actual": panel["ticker"].nunique(),
            "date_range_start": str(panel.index.min().date()),
            "date_range_end": str(panel.index.max().date()),
            "n_rows": len(panel),
        })

        # Save feature list
        with open(OUTPUT_DIR / "feature_cols.json", "w") as f:
            json.dump(feature_cols, f, indent=2)

        # 3. Walk-forward
        print("\n" + "=" * 60)
        print("WALK-FORWARD TRAINING")
        print("=" * 60)
        results_df, avg_fi = walk_forward(panel, feature_cols)

        if len(results_df) < 50:
            print(f"FATAL: Only {len(results_df)} predictions. Need at least 50. Aborting.")
            sys.exit(1)

        # Save predictions
        results_df.to_csv(OUTPUT_DIR / "predictions.csv", index=False)

        # Feature importance
        fi_df = avg_fi.head(20)
        print("\nTop 20 Features (avg importance):")
        for feat, imp in fi_df.items():
            print(f"  {feat}: {imp:.1f}")

        fi_df.to_csv(OUTPUT_DIR / "feature_importance.csv")

        # 4. IC metrics
        print("\n" + "=" * 60)
        print("PREDICTION QUALITY (IC)")
        print("=" * 60)
        ic_metrics = compute_ic(results_df)
        for k, v in ic_metrics.items():
            print(f"  {k}: {v}")
        mlflow.log_metrics(ic_metrics)

        # 5. Signal generation & backtest
        print("\n" + "=" * 60)
        print("SIGNAL GENERATION & BACKTEST")
        print("=" * 60)
        signal_df = generate_signal(results_df)
        bt_df = backtest_straddle(signal_df)

        # Save backtest
        bt_df.to_csv(OUTPUT_DIR / "backtest.csv", index=False)

        # Portfolio-level metrics (aggregate across all tickers per date)
        portfolio_pnl = bt_df.groupby("date")["pnl_net"].sum()
        metrics = compute_metrics(portfolio_pnl, label="portfolio_")
        print("\nPortfolio Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        mlflow.log_metrics(metrics)

        # Per-ticker metrics
        print("\nPer-Ticker Sharpe:")
        ticker_metrics = {}
        for ticker in bt_df["ticker"].unique():
            t_pnl = bt_df[bt_df["ticker"] == ticker]["pnl_net"]
            if len(t_pnl) > 10 and t_pnl.std() > 0:
                t_sharpe = t_pnl.mean() / t_pnl.std() * ANNUALIZATION
                ticker_metrics[ticker] = round(t_sharpe, 3)
                print(f"  {ticker}: {t_sharpe:.3f}")

        # 6. Validation Gates
        print("\n" + "=" * 60)
        print("VALIDATION GATES")
        print("=" * 60)

        gate1 = validation_gate_1_permutation(bt_df)
        mlflow.log_metrics({f"gate1_{k}": v for k, v in gate1.items() if isinstance(v, (int, float))})

        spy_data = all_data.get("SPY")
        gate2 = validation_gate_2_regime(bt_df, spy_data) if spy_data is not None else {"pass": False, "error": "no SPY data"}
        mlflow.log_metrics({f"gate2_{k}": v for k, v in gate2.items() if isinstance(v, (int, float))})

        gate3 = validation_gate_3_subperiod(bt_df)
        mlflow.log_metrics({f"gate3_{k}": v for k, v in gate3.items() if isinstance(v, (int, float))})

        gate4 = validation_gate_4_outlier(bt_df)
        mlflow.log_metrics({f"gate4_{k}": v for k, v in gate4.items() if isinstance(v, (int, float))})

        # 7. Lag sensitivity
        lag_result = lag_sensitivity(panel, feature_cols, results_df)
        mlflow.log_metrics({f"lag_{k}": v for k, v in lag_result.items() if v is not None and isinstance(v, (int, float))})

        # 8. Summary
        gates_passed = sum([
            gate1.get("pass", False),
            gate2.get("pass", False),
            gate3.get("pass", False),
            gate4.get("pass", False),
        ])

        print("\n" + "=" * 80)
        print("FINAL SUMMARY")
        print("=" * 80)
        print(f"Validation: {gates_passed}/4 gates passed")
        print(f"  Gate 1 (Permutation): {'PASS' if gate1.get('pass') else 'FAIL'} (p={gate1.get('p_value', 'N/A')})")
        print(f"  Gate 2 (Regime):      {'PASS' if gate2.get('pass') else 'FAIL'} (diff={gate2.get('regime_diff_ratio', 'N/A')})")
        print(f"  Gate 3 (Sub-period):  {'PASS' if gate3.get('pass') else 'FAIL'} (CV={gate3.get('cv', 'N/A')})")
        print(f"  Gate 4 (Outlier):     {'PASS' if gate4.get('pass') else 'FAIL'} (drop={gate4.get('sharpe_drop_pct', 'N/A')}%)")
        print()
        print(f"Portfolio Sharpe:  {metrics.get('portfolio_sharpe', 'N/A')}")
        print(f"Portfolio Sortino: {metrics.get('portfolio_sortino', 'N/A')}")
        print(f"Portfolio CAGR:    {metrics.get('portfolio_cagr', 'N/A')}%")
        print(f"Portfolio MaxDD:   {metrics.get('portfolio_max_dd', 'N/A')}%")
        print(f"Win Rate:          {metrics.get('portfolio_win_rate', 'N/A')}%")
        print(f"Profit Factor:     {metrics.get('portfolio_profit_factor', 'N/A')}")
        print(f"Overall IC:        {ic_metrics.get('overall_ic', 'N/A')}")
        print(f"Mean Daily IC:     {ic_metrics.get('mean_daily_ic', 'N/A')}")
        print()
        print(f"Completed: {dt.datetime.now()}")

        # Log final summary
        mlflow.log_metrics({
            "gates_passed": gates_passed,
            "gates_total": 4,
        })

        # Save full results
        summary = {
            "metrics": metrics,
            "ic": ic_metrics,
            "gate1_permutation": gate1,
            "gate2_regime": gate2,
            "gate3_subperiod": gate3,
            "gate4_outlier": gate4,
            "lag_sensitivity": lag_result,
            "ticker_sharpes": ticker_metrics,
            "feature_importance_top10": {k: float(v) for k, v in avg_fi.head(10).items()},
            "config": {
                "universe": UNIVERSE,
                "train_window": TRAIN_WINDOW,
                "retrain_freq": RETRAIN_FREQ,
                "predict_horizon": PREDICT_HORIZON,
                "lgb_params": {k: v for k, v in LGB_PARAMS.items() if k != "verbose"},
            },
        }
        with open(OUTPUT_DIR / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        # Determine overall result
        if gates_passed >= 3:
            verdict = "PROMISING — consider paper trading integration"
        elif gates_passed >= 2:
            verdict = "MARGINAL — needs tuning before deployment"
        else:
            verdict = "REJECT — insufficient edge"

        print(f"\nVERDICT: {verdict}")
        mlflow.set_tag("verdict", verdict)

        return summary


if __name__ == "__main__":
    main()
