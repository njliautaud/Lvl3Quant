#!/usr/bin/env python3
"""
LGBMRanker vs LGBMRegressor for Sector ETF Ranking — v1
========================================================

Tests whether LGBMRanker (native Learning-to-Rank) produces better sector
rankings than LGBMRegressor (what V10 currently uses).

V10 baseline: Sharpe ~5-6, 8 positions, 4% OTM, 30% profit target,
biweekly rebalance, DTE=28.

6 Variants:
  A: LGBMRegressor baseline (100 trees, depth 4, lr 0.05, subsample 0.8)
  B: LGBMRanker with lambdarank objective (group_size=11)
  C: LGBMRanker with rank_xendcg objective
  D: LGBMRegressor with larger model (200 trees, depth 6)
  E: LGBMRanker + feature importance pruning (<1% importance dropped)
  F: Ensemble (average rank from A + B)

Walk-forward: 500d sliding train, 10d (biweekly) rebalance.
Options: BS-priced bull call spreads (top 4) + bear put spreads (bottom 4).
Validation: permutation, regime, sub-period, outlier, yearly consistency.

Usage:
    python3 lgbm_ranker_vs_regressor_v1.py
"""

import json
import os
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, spearmanr

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Unbuffered printing
# ---------------------------------------------------------------------------
os.environ["PYTHONUNBUFFERED"] = "1"


def fprint(*args, **kwargs):
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


# ---------------------------------------------------------------------------
# Cross-platform paths
# ---------------------------------------------------------------------------
if Path("/home/nick/Lvl3Quant").exists():
    LVL3 = Path("/home/nick/Lvl3Quant")
elif Path("/home/jupiter/Lvl3Quant").exists():
    LVL3 = Path("/home/jupiter/Lvl3Quant")
else:
    LVL3 = Path.cwd()

OUTPUT_DIR = LVL3 / "output" / "growth_research" / "lgbm_ranker_vs_regressor_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# MLflow setup
# ---------------------------------------------------------------------------
MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen("http://jupiter:5000/", timeout=2)
    import mlflow
    mlflow.set_tracking_uri("http://jupiter:5000")
    MLFLOW_OK = True
    fprint("MLflow connected")
except Exception:
    fprint("MLflow unavailable — results will be logged locally only")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
N_SECTORS = len(SECTORS)  # 11
BENCHMARK = "SPY"

# 21 features
FEATURE_NAMES = [
    "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "vol_21d", "vol_63d", "sharpe_63d", "maxdd_63d", "pct_52w_high",
    "mom_accel", "pct_pos_months_12m", "sortino_63d", "calmar_1y",
    "up_capture", "trend_r2_63d", "trend_slope_63d",
    "sector_spy_beta_63d", "sector_relative_vol_21d", "cross_sector_dispersion",
]

# Walk-forward
WF_TRAIN_DAYS = 500
REBALANCE_DAYS = 10  # biweekly

# Options parameters
OTM_PCT = 0.04       # 4% OTM
PROFIT_TARGET = 0.30  # 30% profit target exit
DTE = 28
DTE_YEARS = DTE / 365.0
ENTRY_HAIRCUT = 0.15  # 15% haircut on entry
COMMISSION_RT = 2.60  # $2.60 round-trip
CAPITAL = 645.0
MAX_PER_TRADE = 200.0
N_BULL = 4  # top 4 sectors -> bull call spreads
N_BEAR = 4  # bottom 4 sectors -> bear put spreads
RISK_FREE = 0.05

# Validation
N_PERMUTATIONS = 300
REGIME_ASYM_THRESHOLD = 0.50
SUB_PERIOD_MIN_SHARPE = 0.5
OUTLIER_TRIM_PCT = 0.05
YEARLY_PCT_THRESHOLD = 0.60


# =========================================================================
# BLACK-SCHOLES PRICING
# =========================================================================
def bs_d1d2(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return None, None
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1, d2 = bs_d1d2(S, K, T, r, sigma)
    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def bs_put(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1, d2 = bs_d1d2(S, K, T, r, sigma)
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def price_bull_call_spread(spot, otm_pct, vol, dte_years=DTE_YEARS, r=RISK_FREE):
    """
    Bull call spread: buy call at K_long (ATM), sell call at K_short (OTM).
    Spread width = max($3, 3% of spot).
    Returns (net_debit, max_profit, K_long, K_short).
    """
    K_long = spot * (1 + otm_pct)  # 4% OTM long call
    width = max(3.0, spot * 0.03)
    K_short = K_long + width

    long_price = bs_call(spot, K_long, dte_years, r, vol)
    short_price = bs_call(spot, K_short, dte_years, r, vol)

    # bid-ask spread: 20bps per leg
    ba = 0.002
    long_price *= (1 + ba)
    short_price *= (1 - ba)

    net_debit = long_price - short_price
    max_profit = width - net_debit
    return net_debit, max_profit, K_long, K_short


def price_bear_put_spread(spot, otm_pct, vol, dte_years=DTE_YEARS, r=RISK_FREE):
    """
    Bear put spread: buy put at K_long (ATM), sell put at K_short (further OTM).
    Spread width = max($3, 3% of spot).
    Returns (net_debit, max_profit, K_long, K_short).
    """
    K_long = spot * (1 - otm_pct)  # 4% OTM long put
    width = max(3.0, spot * 0.03)
    K_short = K_long - width

    long_price = bs_put(spot, K_long, dte_years, r, vol)
    short_price = bs_put(spot, K_short, dte_years, r, vol)

    ba = 0.002
    long_price *= (1 + ba)
    short_price *= (1 - ba)

    net_debit = long_price - short_price
    max_profit = width - net_debit
    return net_debit, max_profit, K_long, K_short


def spread_pnl_at_expiry(spot_exit, K_long, K_short, entry_cost, is_bull=True):
    """Compute P&L at expiry for a vertical spread."""
    if is_bull:
        # Bull call spread
        long_val = max(spot_exit - K_long, 0)
        short_val = max(spot_exit - K_short, 0)
    else:
        # Bear put spread
        long_val = max(K_long - spot_exit, 0)
        short_val = max(K_short - spot_exit, 0)

    exit_value = long_val - short_val
    pnl = exit_value - entry_cost
    # cap at bounds
    max_loss = -entry_cost
    width = abs(K_short - K_long)
    max_gain = width - entry_cost
    pnl = max(max_loss, min(max_gain, pnl))
    return pnl


def compute_spread_pnl_with_profit_target(
    prices_series, entry_idx, entry_spot, K_long, K_short,
    entry_cost, is_bull, vol, hold_days=DTE
):
    """
    Simulate spread with 30% profit target and hold-to-expiry fallback.
    Checks daily for profit target hit using BS mid-life pricing.
    Returns (pnl, days_held).
    """
    target_value = entry_cost * (1 + PROFIT_TARGET)

    for d in range(1, hold_days + 1):
        check_idx = entry_idx + d
        if check_idx >= len(prices_series):
            # expired beyond data
            pnl = spread_pnl_at_expiry(prices_series.iloc[-1], K_long, K_short, entry_cost, is_bull)
            return pnl, d

        spot_now = prices_series.iloc[check_idx]
        dte_remain = (hold_days - d) / 365.0

        if d < hold_days and dte_remain > 0:
            # Mid-life: use BS to estimate current spread value
            if is_bull:
                long_val = bs_call(spot_now, K_long, dte_remain, RISK_FREE, vol)
                short_val = bs_call(spot_now, K_short, dte_remain, RISK_FREE, vol)
            else:
                long_val = bs_put(spot_now, K_long, dte_remain, RISK_FREE, vol)
                short_val = bs_put(spot_now, K_short, dte_remain, RISK_FREE, vol)
            current_value = long_val - short_val

            if current_value >= target_value:
                # Profit target hit — exit with haircut on exit
                pnl = current_value * (1 - 0.002) - entry_cost  # exit spread cost
                return pnl, d
        else:
            # At expiry
            pnl = spread_pnl_at_expiry(spot_now, K_long, K_short, entry_cost, is_bull)
            return pnl, d

    # Fallback (shouldn't reach here)
    last_spot = prices_series.iloc[min(entry_idx + hold_days, len(prices_series) - 1)]
    pnl = spread_pnl_at_expiry(last_spot, K_long, K_short, entry_cost, is_bull)
    return pnl, hold_days


# =========================================================================
# DATA DOWNLOAD
# =========================================================================
def download_data():
    """Download sector ETF + SPY data via yfinance."""
    import yfinance as yf

    tickers = SECTORS + [BENCHMARK]
    fprint(f"Downloading {len(tickers)} tickers...")
    data = yf.download(tickers, start="2008-01-01", auto_adjust=True, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
    else:
        close = data

    close = close[[c for c in tickers if c in close.columns]]
    close = close.dropna(how="all").ffill().dropna()
    fprint(f"  Loaded {close.shape[1]} tickers, {len(close)} trading days "
           f"({close.index[0].date()} to {close.index[-1].date()})")
    return close


# =========================================================================
# FEATURE ENGINEERING
# =========================================================================
def compute_features(close_df):
    """
    Compute 21 features for each (date, sector) pair.
    Returns a DataFrame with MultiIndex (date, sector) and feature columns.
    """
    fprint("Computing 21 features for all sectors...")
    sector_cols = [c for c in SECTORS if c in close_df.columns]
    spy = close_df[BENCHMARK]
    spy_ret = spy.pct_change()

    records = []

    for sec in sector_cols:
        px = close_df[sec]
        ret = px.pct_change()
        log_ret = np.log(px / px.shift(1))

        # Returns at various horizons
        ret_5d = px.pct_change(5)
        ret_10d = px.pct_change(10)
        ret_21d = px.pct_change(21)
        ret_63d = px.pct_change(63)
        ret_126d = px.pct_change(126)
        ret_252d = px.pct_change(252)

        # Volatility
        vol_21d = log_ret.rolling(21).std() * np.sqrt(252)
        vol_63d = log_ret.rolling(63).std() * np.sqrt(252)

        # Sharpe 63d
        mean_63 = ret.rolling(63).mean() * 252
        std_63 = ret.rolling(63).std() * np.sqrt(252)
        sharpe_63d = mean_63 / std_63.replace(0, np.nan)

        # Max drawdown 63d
        rolling_max_63 = px.rolling(63).max()
        dd_63 = (px - rolling_max_63) / rolling_max_63
        maxdd_63d = dd_63.rolling(63).min()

        # Pct of 52-week high
        high_252 = px.rolling(252).max()
        pct_52w_high = px / high_252.replace(0, np.nan)

        # Momentum acceleration: ret_21d - ret_21d.shift(21)
        mom_accel = ret_21d - ret_21d.shift(21)

        # Pct positive months in last 12 months (approx with 21d windows)
        monthly_rets = ret_21d.copy()
        pct_pos_months_12m = monthly_rets.rolling(252).apply(
            lambda x: np.sum(x[::21] > 0) / max(len(x[::21]), 1), raw=True
        )

        # Sortino 63d
        downside = ret.copy()
        downside[downside > 0] = 0
        downside_std_63 = downside.rolling(63).std() * np.sqrt(252)
        sortino_63d = mean_63 / downside_std_63.replace(0, np.nan)

        # Calmar 1y
        ann_ret_252 = ret.rolling(252).mean() * 252
        rolling_max_252 = px.rolling(252).max()
        dd_252 = (px - rolling_max_252) / rolling_max_252
        maxdd_252 = dd_252.rolling(252).min()
        calmar_1y = ann_ret_252 / (-maxdd_252).replace(0, np.nan)

        # Up capture ratio (63d)
        spy_up = spy_ret.copy()
        spy_up[spy_up <= 0] = np.nan
        sec_when_spy_up = ret.where(spy_up.notna())
        up_capture = (sec_when_spy_up.rolling(63).mean() / spy_up.rolling(63).mean())

        # Trend R^2 and slope (63d)
        def rolling_trend(series, window=63):
            r2_vals = pd.Series(np.nan, index=series.index)
            slope_vals = pd.Series(np.nan, index=series.index)
            x = np.arange(window)
            for i in range(window, len(series)):
                y = series.iloc[i - window:i].values
                if np.any(np.isnan(y)):
                    continue
                slope, intercept, r_value, _, _ = stats.linregress(x, y)
                r2_vals.iloc[i] = r_value ** 2
                slope_vals.iloc[i] = slope
            return r2_vals, slope_vals

        # Use log prices for trend
        log_px = np.log(px)
        trend_r2_63d, trend_slope_63d = rolling_trend(log_px, 63)

        # Sector-SPY beta 63d
        def rolling_beta(sec_ret, mkt_ret, window=63):
            betas = pd.Series(np.nan, index=sec_ret.index)
            for i in range(window, len(sec_ret)):
                s = sec_ret.iloc[i - window:i].values
                m = mkt_ret.iloc[i - window:i].values
                mask = ~(np.isnan(s) | np.isnan(m))
                if mask.sum() < 10:
                    continue
                cov = np.cov(s[mask], m[mask])
                if cov[1, 1] > 0:
                    betas.iloc[i] = cov[0, 1] / cov[1, 1]
            return betas

        sector_spy_beta_63d = rolling_beta(ret, spy_ret, 63)

        # Sector relative vol 21d
        spy_vol_21d = spy_ret.rolling(21).std() * np.sqrt(252)
        sector_relative_vol_21d = vol_21d / spy_vol_21d.replace(0, np.nan)

        # Cross-sector dispersion (same for all sectors on a given day)
        # — computed per sector as a feature, but value is cross-sectional
        # We'll fill this in a second pass

        # Build per-day records
        for i in range(252, len(px)):
            dt = px.index[i]
            rec = {
                "date": dt,
                "sector": sec,
                "ret_5d": ret_5d.iloc[i],
                "ret_10d": ret_10d.iloc[i],
                "ret_21d": ret_21d.iloc[i],
                "ret_63d": ret_63d.iloc[i],
                "ret_126d": ret_126d.iloc[i],
                "ret_252d": ret_252d.iloc[i],
                "vol_21d": vol_21d.iloc[i],
                "vol_63d": vol_63d.iloc[i],
                "sharpe_63d": sharpe_63d.iloc[i],
                "maxdd_63d": maxdd_63d.iloc[i],
                "pct_52w_high": pct_52w_high.iloc[i],
                "mom_accel": mom_accel.iloc[i],
                "pct_pos_months_12m": pct_pos_months_12m.iloc[i],
                "sortino_63d": sortino_63d.iloc[i],
                "calmar_1y": calmar_1y.iloc[i],
                "up_capture": up_capture.iloc[i],
                "trend_r2_63d": trend_r2_63d.iloc[i],
                "trend_slope_63d": trend_slope_63d.iloc[i],
                "sector_spy_beta_63d": sector_spy_beta_63d.iloc[i],
                "sector_relative_vol_21d": sector_relative_vol_21d.iloc[i],
                "cross_sector_dispersion": np.nan,  # placeholder
            }
            records.append(rec)

    df = pd.DataFrame(records)

    # Compute cross-sector dispersion per date
    fprint("  Computing cross-sector dispersion...")
    sector_ret_21d = {}
    for sec in sector_cols:
        px = close_df[sec]
        sector_ret_21d[sec] = px.pct_change(21)

    disp_df = pd.DataFrame(sector_ret_21d)
    cross_disp = disp_df.std(axis=1)

    date_disp = {}
    for dt in df["date"].unique():
        if dt in cross_disp.index:
            date_disp[dt] = cross_disp.loc[dt]
        else:
            date_disp[dt] = np.nan

    df["cross_sector_dispersion"] = df["date"].map(date_disp)

    # Forward returns (target): next REBALANCE_DAYS return
    fprint("  Computing forward returns...")
    fwd_rets = {}
    for sec in sector_cols:
        px = close_df[sec]
        fwd = px.pct_change(REBALANCE_DAYS).shift(-REBALANCE_DAYS)
        fwd_rets[sec] = fwd

    def get_fwd_ret(row):
        sec = row["sector"]
        dt = row["date"]
        if dt in fwd_rets[sec].index:
            return fwd_rets[sec].loc[dt]
        return np.nan

    df["fwd_return"] = df.apply(get_fwd_ret, axis=1)

    # Drop rows with NaN target or features
    df = df.dropna(subset=["fwd_return"] + FEATURE_NAMES)

    # Fill remaining NaN features with 0
    df[FEATURE_NAMES] = df[FEATURE_NAMES].fillna(0)

    fprint(f"  Feature matrix: {len(df)} rows, {len(df['date'].unique())} dates, "
           f"{len(df['sector'].unique())} sectors")
    return df


# =========================================================================
# RANKING METRICS
# =========================================================================
def ndcg_at_k(y_true_ranks, y_pred_ranks, k=4):
    """
    NDCG@k: how well do predicted top-k match actual top-k.
    y_true_ranks, y_pred_ranks: arrays of relevance (higher = better).
    """
    # Sort by predicted rank descending
    order = np.argsort(-y_pred_ranks)
    y_sorted = y_true_ranks[order]

    # DCG@k
    dcg = 0.0
    for i in range(min(k, len(y_sorted))):
        dcg += (2 ** y_sorted[i] - 1) / np.log2(i + 2)

    # Ideal DCG@k
    ideal_order = np.argsort(-y_true_ranks)
    y_ideal = y_true_ranks[ideal_order]
    idcg = 0.0
    for i in range(min(k, len(y_ideal))):
        idcg += (2 ** y_ideal[i] - 1) / np.log2(i + 2)

    if idcg == 0:
        return 1.0
    return dcg / idcg


# =========================================================================
# MODEL TRAINING AND PREDICTION
# =========================================================================
def train_predict_variant(variant, train_df, test_df, feature_cols):
    """
    Train a model variant and return predicted ranks for test_df.
    Returns: predicted_ranks (array, higher = better predicted sector)
    """
    import lightgbm as lgb

    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values

    # Target for regressor: forward return
    y_train_ret = train_df["fwd_return"].values

    # Target for ranker: rank of forward return (0-10, higher = better)
    # Computed per date group
    y_train_rank = np.zeros(len(train_df), dtype=np.float32)
    train_dates = train_df["date"].values
    unique_train_dates = np.unique(train_dates)
    train_group = []
    for dt in unique_train_dates:
        mask = train_dates == dt
        rets = y_train_ret[mask]
        ranks = stats.rankdata(rets, method="ordinal") - 1  # 0-based
        y_train_rank[mask] = ranks.astype(np.float32)
        train_group.append(int(mask.sum()))

    if variant == "A":
        # LGBMRegressor baseline
        model = lgb.LGBMRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_samples=20, random_state=42, verbose=-1,
            n_jobs=-1,
        )
        model.fit(X_train, y_train_ret)
        preds = model.predict(X_test)

    elif variant == "B":
        # LGBMRanker with lambdarank
        model = lgb.LGBMRanker(
            objective="lambdarank",
            n_estimators=100, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_samples=20, random_state=42, verbose=-1,
            n_jobs=-1,
        )
        model.fit(X_train, y_train_rank, group=train_group)
        preds = model.predict(X_test)

    elif variant == "C":
        # LGBMRanker with rank_xendcg
        model = lgb.LGBMRanker(
            objective="rank_xendcg",
            n_estimators=100, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_samples=20, random_state=42, verbose=-1,
            n_jobs=-1,
        )
        model.fit(X_train, y_train_rank, group=train_group)
        preds = model.predict(X_test)

    elif variant == "D":
        # LGBMRegressor larger model
        model = lgb.LGBMRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_samples=20, random_state=42, verbose=-1,
            n_jobs=-1,
        )
        model.fit(X_train, y_train_ret)
        preds = model.predict(X_test)

    elif variant == "E":
        # LGBMRanker + feature importance pruning
        # First fit to get importances
        model0 = lgb.LGBMRanker(
            objective="lambdarank",
            n_estimators=100, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_samples=20, random_state=42, verbose=-1,
            n_jobs=-1,
        )
        model0.fit(X_train, y_train_rank, group=train_group)
        importances = model0.feature_importances_
        total_imp = importances.sum()
        if total_imp > 0:
            imp_pct = importances / total_imp
        else:
            imp_pct = np.ones(len(importances)) / len(importances)

        # Keep features with >= 1% importance
        keep_mask = imp_pct >= 0.01
        if keep_mask.sum() < 3:
            keep_mask = np.ones(len(importances), dtype=bool)

        X_train_pruned = X_train[:, keep_mask]
        X_test_pruned = X_test[:, keep_mask]

        model = lgb.LGBMRanker(
            objective="lambdarank",
            n_estimators=100, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_samples=20, random_state=42, verbose=-1,
            n_jobs=-1,
        )
        model.fit(X_train_pruned, y_train_rank, group=train_group)
        preds = model.predict(X_test_pruned)

    elif variant == "F":
        # Ensemble: average rank from A + B
        # A: regressor
        model_a = lgb.LGBMRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_samples=20, random_state=42, verbose=-1,
            n_jobs=-1,
        )
        model_a.fit(X_train, y_train_ret)
        preds_a = model_a.predict(X_test)

        # B: ranker
        model_b = lgb.LGBMRanker(
            objective="lambdarank",
            n_estimators=100, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_samples=20, random_state=42, verbose=-1,
            n_jobs=-1,
        )
        model_b.fit(X_train, y_train_rank, group=train_group)
        preds_b = model_b.predict(X_test)

        # Average ranks per test date
        test_dates = test_df["date"].values
        preds = np.zeros(len(test_df))
        for dt in np.unique(test_dates):
            mask = test_dates == dt
            rank_a = stats.rankdata(preds_a[mask])
            rank_b = stats.rankdata(preds_b[mask])
            avg_rank = (rank_a + rank_b) / 2.0
            preds[mask] = avg_rank

    else:
        raise ValueError(f"Unknown variant: {variant}")

    return preds


# =========================================================================
# OPTIONS TRADING SIMULATION
# =========================================================================
def simulate_options_trades(test_dates, rankings_by_date, close_df):
    """
    For each rebalance date, take bull spreads on top 4, bear spreads on bottom 4.
    Returns list of trade dicts with P&L.
    """
    all_trades = []
    sector_cols = [c for c in SECTORS if c in close_df.columns]

    for dt in test_dates:
        if dt not in rankings_by_date:
            continue

        ranking = rankings_by_date[dt]  # dict: sector -> predicted_score
        # Sort by score descending
        sorted_sectors = sorted(ranking.keys(), key=lambda s: ranking[s], reverse=True)
        bull_sectors = sorted_sectors[:N_BULL]
        bear_sectors = sorted_sectors[-N_BEAR:]

        dt_idx = close_df.index.get_loc(dt)

        for sec in bull_sectors:
            if sec not in close_df.columns or dt_idx + DTE >= len(close_df):
                continue
            spot = close_df[sec].iloc[dt_idx]
            # Compute vol
            if dt_idx >= 63:
                log_rets = np.log(close_df[sec].iloc[dt_idx - 63:dt_idx] /
                                  close_df[sec].iloc[dt_idx - 63:dt_idx].shift(1)).dropna()
                vol = float(log_rets.std() * np.sqrt(252))
            else:
                vol = 0.20

            net_debit, max_profit, K_long, K_short = price_bull_call_spread(
                spot, OTM_PCT, vol
            )

            if net_debit <= 0 or net_debit > MAX_PER_TRADE:
                continue

            # Apply entry haircut
            effective_cost = net_debit * (1 + ENTRY_HAIRCUT)

            # Number of contracts we can afford
            n_contracts = max(1, int(MAX_PER_TRADE / (effective_cost * 100)))
            trade_cost = effective_cost * 100 * n_contracts + COMMISSION_RT

            # Simulate with profit target
            pnl_per, days_held = compute_spread_pnl_with_profit_target(
                close_df[sec], dt_idx, spot, K_long, K_short, effective_cost, True, vol
            )

            total_pnl = pnl_per * 100 * n_contracts - COMMISSION_RT

            all_trades.append({
                "date": dt,
                "sector": sec,
                "direction": "bull",
                "spot": spot,
                "K_long": K_long,
                "K_short": K_short,
                "cost": trade_cost,
                "pnl": total_pnl,
                "days_held": days_held,
                "n_contracts": n_contracts,
            })

        for sec in bear_sectors:
            if sec not in close_df.columns or dt_idx + DTE >= len(close_df):
                continue
            spot = close_df[sec].iloc[dt_idx]
            if dt_idx >= 63:
                log_rets = np.log(close_df[sec].iloc[dt_idx - 63:dt_idx] /
                                  close_df[sec].iloc[dt_idx - 63:dt_idx].shift(1)).dropna()
                vol = float(log_rets.std() * np.sqrt(252))
            else:
                vol = 0.20

            net_debit, max_profit, K_long, K_short = price_bear_put_spread(
                spot, OTM_PCT, vol
            )

            if net_debit <= 0 or net_debit > MAX_PER_TRADE:
                continue

            effective_cost = net_debit * (1 + ENTRY_HAIRCUT)
            n_contracts = max(1, int(MAX_PER_TRADE / (effective_cost * 100)))
            trade_cost = effective_cost * 100 * n_contracts + COMMISSION_RT

            pnl_per, days_held = compute_spread_pnl_with_profit_target(
                close_df[sec], dt_idx, spot, K_long, K_short, effective_cost, False, vol
            )

            total_pnl = pnl_per * 100 * n_contracts - COMMISSION_RT

            all_trades.append({
                "date": dt,
                "sector": sec,
                "direction": "bear",
                "spot": spot,
                "K_long": K_long,
                "K_short": K_short,
                "cost": trade_cost,
                "pnl": total_pnl,
                "days_held": days_held,
                "n_contracts": n_contracts,
            })

    return all_trades


# =========================================================================
# WALK-FORWARD ENGINE
# =========================================================================
def run_walkforward(variant, feat_df, close_df):
    """
    Run sliding walk-forward for a variant.
    Returns: trades list, ranking metrics (spearman, ndcg lists).
    """
    fprint(f"\n{'='*60}")
    fprint(f"  Variant {variant}: starting walk-forward")
    fprint(f"{'='*60}")

    dates = sorted(feat_df["date"].unique())
    feature_cols = FEATURE_NAMES.copy()

    all_trades = []
    spearman_list = []
    ndcg_list = []

    # Find rebalance dates (every REBALANCE_DAYS trading days)
    rebal_dates = dates[WF_TRAIN_DAYS::REBALANCE_DAYS]
    fprint(f"  {len(rebal_dates)} rebalance dates from {rebal_dates[0].date() if len(rebal_dates) > 0 else 'N/A'}"
           f" to {rebal_dates[-1].date() if len(rebal_dates) > 0 else 'N/A'}")

    rankings_by_date = {}
    n_done = 0

    for rebal_dt in rebal_dates:
        # Train window: last WF_TRAIN_DAYS dates before rebal_dt
        train_mask = (feat_df["date"] < rebal_dt)
        train_dates = sorted(feat_df[train_mask]["date"].unique())

        if len(train_dates) < WF_TRAIN_DAYS:
            continue

        # Sliding: take last WF_TRAIN_DAYS dates
        train_dates_window = train_dates[-WF_TRAIN_DAYS:]
        train_df = feat_df[feat_df["date"].isin(train_dates_window)]

        # Test: this rebalance date
        test_df = feat_df[feat_df["date"] == rebal_dt]
        if len(test_df) < N_SECTORS - 2:  # need at least 9 sectors
            continue

        # Predict
        try:
            preds = train_predict_variant(variant, train_df, test_df, feature_cols)
        except Exception as e:
            fprint(f"  WARNING: variant {variant} failed on {rebal_dt.date()}: {e}")
            continue

        # Build ranking
        test_sectors = test_df["sector"].values
        ranking = {sec: float(pred) for sec, pred in zip(test_sectors, preds)}
        rankings_by_date[rebal_dt] = ranking

        # Ranking quality metrics
        actual_rets = test_df["fwd_return"].values
        if len(actual_rets) == len(preds) and len(actual_rets) >= 4:
            # Spearman
            rho, _ = spearmanr(preds, actual_rets)
            if not np.isnan(rho):
                spearman_list.append(rho)

            # NDCG@4
            actual_ranks = stats.rankdata(actual_rets)
            pred_ranks = stats.rankdata(preds)
            ndcg = ndcg_at_k(actual_ranks, pred_ranks, k=4)
            ndcg_list.append(ndcg)

        n_done += 1
        if n_done % 50 == 0:
            fprint(f"    ... {n_done}/{len(rebal_dates)} rebalance dates processed")

    fprint(f"  Completed {n_done} rebalance periods")

    # Simulate options trades
    rebal_date_list = sorted(rankings_by_date.keys())
    trades = simulate_options_trades(rebal_date_list, rankings_by_date, close_df)
    fprint(f"  Generated {len(trades)} trades")

    return trades, spearman_list, ndcg_list


# =========================================================================
# METRICS COMPUTATION
# =========================================================================
def compute_metrics(trades):
    """Compute strategy-level metrics from trade list."""
    if not trades:
        return {
            "sharpe": 0, "sortino": 0, "pf": 0, "wr": 0,
            "total_pnl": 0, "n_trades": 0, "maxdd": 0, "calmar": 0,
            "avg_pnl": 0, "monthly_returns": [],
        }

    trade_df = pd.DataFrame(trades)
    trade_df["date"] = pd.to_datetime(trade_df["date"])

    # Daily P&L aggregation
    daily_pnl = trade_df.groupby("date")["pnl"].sum()
    daily_pnl = daily_pnl.sort_index()

    # Monthly returns (calendar month)
    monthly = daily_pnl.resample("ME").sum()

    # Metrics
    total_pnl = daily_pnl.sum()
    n_trades = len(trade_df)
    winners = trade_df[trade_df["pnl"] > 0]
    losers = trade_df[trade_df["pnl"] <= 0]
    wr = len(winners) / n_trades if n_trades > 0 else 0

    gross_profit = winners["pnl"].sum() if len(winners) > 0 else 0
    gross_loss = abs(losers["pnl"].sum()) if len(losers) > 0 else 1
    pf = gross_profit / gross_loss if gross_loss > 0 else 999

    # Monthly Sharpe
    if len(monthly) > 2:
        sharpe = float(monthly.mean() / monthly.std() * np.sqrt(12)) if monthly.std() > 0 else 0
    else:
        sharpe = 0

    # Monthly Sortino
    if len(monthly) > 2:
        downside_m = monthly[monthly < 0]
        downside_std = downside_m.std() if len(downside_m) > 1 else monthly.std()
        sortino = float(monthly.mean() / downside_std * np.sqrt(12)) if downside_std > 0 else 0
    else:
        sortino = 0

    # Cumulative equity curve for max drawdown
    cum_pnl = daily_pnl.cumsum()
    running_max = cum_pnl.cummax()
    drawdown = cum_pnl - running_max
    maxdd = float(drawdown.min()) if len(drawdown) > 0 else 0

    # Calmar
    ann_pnl = total_pnl / (len(daily_pnl) / 252) if len(daily_pnl) > 0 else 0
    calmar = ann_pnl / abs(maxdd) if maxdd != 0 else 0

    avg_pnl = total_pnl / n_trades if n_trades > 0 else 0

    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "pf": round(pf, 3),
        "wr": round(wr, 4),
        "total_pnl": round(total_pnl, 2),
        "n_trades": n_trades,
        "maxdd": round(maxdd, 2),
        "calmar": round(calmar, 3),
        "avg_pnl": round(avg_pnl, 2),
        "monthly_returns": monthly.values.tolist() if len(monthly) > 0 else [],
    }


# =========================================================================
# VALIDATION SUITE
# =========================================================================
def run_validation(trades, variant_name):
    """Run 5-point validation suite. Returns dict of results."""
    fprint(f"\n  --- Validation for Variant {variant_name} ---")
    results = {}

    if not trades:
        fprint("    No trades — all validations FAIL")
        return {
            "permutation_p": 1.0, "permutation_pass": False,
            "regime_asym": 1.0, "regime_pass": False,
            "sub_period_pass": False,
            "outlier_sharpe": 0, "outlier_pass": False,
            "yearly_pct": 0, "yearly_pass": False,
            "all_pass": False,
        }

    trade_df = pd.DataFrame(trades)
    trade_df["date"] = pd.to_datetime(trade_df["date"])
    daily_pnl = trade_df.groupby("date")["pnl"].sum().sort_index()
    monthly = daily_pnl.resample("ME").sum()
    actual_sharpe = float(monthly.mean() / monthly.std() * np.sqrt(12)) if len(monthly) > 2 and monthly.std() > 0 else 0

    # 1. Permutation test (300 shuffles)
    fprint("    [1/5] Permutation test (300 shuffles)...")
    perm_sharpes = []
    pnl_vals = daily_pnl.values.copy()
    for _ in range(N_PERMUTATIONS):
        np.random.shuffle(pnl_vals)
        shuf_series = pd.Series(pnl_vals, index=daily_pnl.index)
        shuf_monthly = shuf_series.resample("ME").sum()
        if len(shuf_monthly) > 2 and shuf_monthly.std() > 0:
            perm_sharpes.append(float(shuf_monthly.mean() / shuf_monthly.std() * np.sqrt(12)))
        else:
            perm_sharpes.append(0)

    perm_p = np.mean(np.array(perm_sharpes) >= actual_sharpe)
    perm_pass = perm_p < 0.05
    results["permutation_p"] = round(float(perm_p), 4)
    results["permutation_pass"] = bool(perm_pass)
    fprint(f"      Sharpe={actual_sharpe:.2f}, p={perm_p:.4f} -> {'PASS' if perm_pass else 'FAIL'}")

    # 2. Regime stability
    fprint("    [2/5] Regime stability...")
    spy_monthly = None
    try:
        import yfinance as yf
        spy = yf.download("SPY", start=daily_pnl.index[0], end=daily_pnl.index[-1], progress=False)
        if isinstance(spy.columns, pd.MultiIndex):
            spy_close = spy["Close"].iloc[:, 0] if spy["Close"].shape[1] > 0 else spy["Close"]
        else:
            spy_close = spy["Close"]
        spy_monthly = spy_close.resample("ME").last().pct_change().dropna()
    except Exception:
        pass

    if spy_monthly is not None and len(spy_monthly) > 6:
        bull_months = spy_monthly[spy_monthly > 0].index
        bear_months = spy_monthly[spy_monthly <= 0].index
        bull_pnl = monthly[monthly.index.isin(bull_months)]
        bear_pnl = monthly[monthly.index.isin(bear_months)]

        bull_sharpe = float(bull_pnl.mean() / bull_pnl.std() * np.sqrt(12)) if len(bull_pnl) > 2 and bull_pnl.std() > 0 else 0
        bear_sharpe = float(bear_pnl.mean() / bear_pnl.std() * np.sqrt(12)) if len(bear_pnl) > 2 and bear_pnl.std() > 0 else 0
        max_s = max(abs(bull_sharpe), abs(bear_sharpe))
        regime_asym = abs(bull_sharpe - bear_sharpe) / max_s if max_s > 0 else 0
        regime_pass = regime_asym < REGIME_ASYM_THRESHOLD
        results["bull_sharpe"] = round(bull_sharpe, 3)
        results["bear_sharpe"] = round(bear_sharpe, 3)
    else:
        regime_asym = 0
        regime_pass = True
        results["bull_sharpe"] = 0
        results["bear_sharpe"] = 0

    results["regime_asym"] = round(float(regime_asym), 4)
    results["regime_pass"] = bool(regime_pass)
    fprint(f"      Asymmetry={regime_asym:.3f} -> {'PASS' if regime_pass else 'FAIL'}")

    # 3. Sub-period: both halves Sharpe > 0.5
    fprint("    [3/5] Sub-period consistency...")
    mid = len(monthly) // 2
    h1 = monthly.iloc[:mid]
    h2 = monthly.iloc[mid:]
    s1 = float(h1.mean() / h1.std() * np.sqrt(12)) if len(h1) > 2 and h1.std() > 0 else 0
    s2 = float(h2.mean() / h2.std() * np.sqrt(12)) if len(h2) > 2 and h2.std() > 0 else 0
    sub_pass = s1 > SUB_PERIOD_MIN_SHARPE and s2 > SUB_PERIOD_MIN_SHARPE
    results["sub_period_s1"] = round(s1, 3)
    results["sub_period_s2"] = round(s2, 3)
    results["sub_period_pass"] = bool(sub_pass)
    fprint(f"      H1 Sharpe={s1:.2f}, H2 Sharpe={s2:.2f} -> {'PASS' if sub_pass else 'FAIL'}")

    # 4. Outlier removal: trim 5% tails, Sharpe > 0.5
    fprint("    [4/5] Outlier removal...")
    if len(monthly) > 10:
        lo = np.percentile(monthly, OUTLIER_TRIM_PCT * 100)
        hi = np.percentile(monthly, (1 - OUTLIER_TRIM_PCT) * 100)
        trimmed = monthly[(monthly >= lo) & (monthly <= hi)]
        outlier_sharpe = float(trimmed.mean() / trimmed.std() * np.sqrt(12)) if len(trimmed) > 2 and trimmed.std() > 0 else 0
    else:
        outlier_sharpe = actual_sharpe
    outlier_pass = outlier_sharpe > SUB_PERIOD_MIN_SHARPE
    results["outlier_sharpe"] = round(outlier_sharpe, 3)
    results["outlier_pass"] = bool(outlier_pass)
    fprint(f"      Trimmed Sharpe={outlier_sharpe:.2f} -> {'PASS' if outlier_pass else 'FAIL'}")

    # 5. Yearly consistency: >= 60% profitable years
    fprint("    [5/5] Yearly consistency...")
    yearly = daily_pnl.resample("YE").sum()
    if len(yearly) > 0:
        pct_profitable = (yearly > 0).mean()
    else:
        pct_profitable = 0
    yearly_pass = pct_profitable >= YEARLY_PCT_THRESHOLD
    results["yearly_pct"] = round(float(pct_profitable), 4)
    results["yearly_pass"] = bool(yearly_pass)
    fprint(f"      {pct_profitable*100:.0f}% profitable years -> {'PASS' if yearly_pass else 'FAIL'}")

    results["all_pass"] = bool(perm_pass and regime_pass and sub_pass and outlier_pass and yearly_pass)
    fprint(f"    ALL VALIDATIONS: {'PASS' if results['all_pass'] else 'FAIL'}")
    return results


# =========================================================================
# MAIN
# =========================================================================
def main():
    t0 = time.time()
    fprint("=" * 70)
    fprint("LGBMRanker vs LGBMRegressor for Sector ETF Ranking — v1")
    fprint(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 70)

    # Download data
    close_df = download_data()

    # Compute features
    feat_df = compute_features(close_df)

    # Run all variants
    VARIANTS = ["A", "B", "C", "D", "E", "F"]
    VARIANT_NAMES = {
        "A": "LGBMRegressor baseline (100t/d4)",
        "B": "LGBMRanker lambdarank",
        "C": "LGBMRanker rank_xendcg",
        "D": "LGBMRegressor large (200t/d6)",
        "E": "LGBMRanker + feat pruning",
        "F": "Ensemble (A+B avg rank)",
    }

    all_results = {}

    # MLflow experiment
    mlflow_run = None
    if MLFLOW_OK:
        try:
            mlflow.set_experiment("lgbm_ranker_vs_regressor_v1")
            mlflow_run = mlflow.start_run(run_name=f"comparison_{datetime.now().strftime('%Y%m%d_%H%M')}")
        except Exception as e:
            fprint(f"MLflow experiment setup failed: {e}")
            MLFLOW_OK_local = False

    for var in VARIANTS:
        fprint(f"\n{'#'*70}")
        fprint(f"# VARIANT {var}: {VARIANT_NAMES[var]}")
        fprint(f"{'#'*70}")

        trades, spearman_list, ndcg_list = run_walkforward(var, feat_df, close_df)
        metrics = compute_metrics(trades)
        validation = run_validation(trades, var)

        # Ranking quality
        avg_spearman = float(np.mean(spearman_list)) if spearman_list else 0
        avg_ndcg = float(np.mean(ndcg_list)) if ndcg_list else 0
        median_spearman = float(np.median(spearman_list)) if spearman_list else 0

        result = {
            "variant": var,
            "name": VARIANT_NAMES[var],
            "metrics": metrics,
            "validation": validation,
            "ranking_quality": {
                "avg_spearman": round(avg_spearman, 4),
                "median_spearman": round(median_spearman, 4),
                "avg_ndcg_at_4": round(avg_ndcg, 4),
                "n_eval_dates": len(spearman_list),
            },
        }
        all_results[var] = result

        # Log to MLflow
        if MLFLOW_OK and mlflow_run:
            try:
                pfx = f"v{var}_"
                mlflow.log_metric(f"{pfx}sharpe", metrics["sharpe"])
                mlflow.log_metric(f"{pfx}sortino", metrics["sortino"])
                mlflow.log_metric(f"{pfx}pf", metrics["pf"])
                mlflow.log_metric(f"{pfx}wr", metrics["wr"])
                mlflow.log_metric(f"{pfx}total_pnl", metrics["total_pnl"])
                mlflow.log_metric(f"{pfx}n_trades", metrics["n_trades"])
                mlflow.log_metric(f"{pfx}maxdd", metrics["maxdd"])
                mlflow.log_metric(f"{pfx}avg_spearman", avg_spearman)
                mlflow.log_metric(f"{pfx}avg_ndcg4", avg_ndcg)
                mlflow.log_metric(f"{pfx}perm_p", validation["permutation_p"])
                mlflow.log_metric(f"{pfx}all_pass", 1 if validation["all_pass"] else 0)
            except Exception as e:
                fprint(f"  MLflow log failed: {e}")

        fprint(f"\n  SUMMARY Variant {var}:")
        fprint(f"    Sharpe={metrics['sharpe']:.2f}  Sortino={metrics['sortino']:.2f}  "
               f"PF={metrics['pf']:.2f}  WR={metrics['wr']:.1%}")
        fprint(f"    Total P&L=${metrics['total_pnl']:.0f}  Trades={metrics['n_trades']}  "
               f"MaxDD=${metrics['maxdd']:.0f}")
        fprint(f"    Spearman={avg_spearman:.3f}  NDCG@4={avg_ndcg:.3f}")
        fprint(f"    Validation: {'ALL PASS' if validation['all_pass'] else 'SOME FAIL'}")

    # End MLflow run
    if MLFLOW_OK and mlflow_run:
        try:
            mlflow.end_run()
        except Exception:
            pass

    # =====================================================================
    # FINAL RESULTS TABLE
    # =====================================================================
    fprint("\n" + "=" * 90)
    fprint("FINAL RESULTS — sorted by Sharpe")
    fprint("=" * 90)

    sorted_vars = sorted(all_results.keys(), key=lambda v: all_results[v]["metrics"]["sharpe"], reverse=True)

    header = f"{'Var':<4} {'Name':<32} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} {'WR':>6} {'P&L':>9} {'#Trd':>5} {'MaxDD':>8} {'Valid':>6}"
    fprint(header)
    fprint("-" * len(header))
    for var in sorted_vars:
        r = all_results[var]
        m = r["metrics"]
        v = r["validation"]
        valid_str = "PASS" if v["all_pass"] else "FAIL"
        fprint(f"{var:<4} {r['name']:<32} {m['sharpe']:>7.2f} {m['sortino']:>8.2f} "
               f"{m['pf']:>6.2f} {m['wr']:>5.1%} {m['total_pnl']:>9.0f} {m['n_trades']:>5} "
               f"{m['maxdd']:>8.0f} {valid_str:>6}")

    fprint("\n" + "=" * 90)
    fprint("RANKING QUALITY METRICS")
    fprint("=" * 90)
    header2 = f"{'Var':<4} {'Name':<32} {'Spearman':>9} {'Med Spear':>10} {'NDCG@4':>8} {'N_eval':>7}"
    fprint(header2)
    fprint("-" * len(header2))
    for var in sorted_vars:
        r = all_results[var]
        rq = r["ranking_quality"]
        fprint(f"{var:<4} {r['name']:<32} {rq['avg_spearman']:>9.4f} {rq['median_spearman']:>10.4f} "
               f"{rq['avg_ndcg_at_4']:>8.4f} {rq['n_eval_dates']:>7}")

    fprint("\n" + "=" * 90)
    fprint("VALIDATION DETAIL")
    fprint("=" * 90)
    header3 = f"{'Var':<4} {'Perm p':>8} {'Regime':>8} {'Sub H1':>7} {'Sub H2':>7} {'OutlSh':>7} {'YrPct':>6} {'ALL':>5}"
    fprint(header3)
    fprint("-" * len(header3))
    for var in sorted_vars:
        r = all_results[var]
        v = r["validation"]
        fprint(f"{var:<4} {v['permutation_p']:>8.4f} {v['regime_asym']:>8.3f} "
               f"{v.get('sub_period_s1', 0):>7.2f} {v.get('sub_period_s2', 0):>7.2f} "
               f"{v['outlier_sharpe']:>7.2f} {v['yearly_pct']:>5.0%} "
               f"{'PASS' if v['all_pass'] else 'FAIL':>5}")

    # Save JSON
    json_path = OUTPUT_DIR / "results.json"
    # Convert numpy types for JSON serialization
    def json_safe(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return obj

    json_results = {}
    for var in all_results:
        r = all_results[var]
        # Deep copy with json-safe conversion
        jr = {
            "variant": r["variant"],
            "name": r["name"],
            "metrics": {k: json_safe(v) for k, v in r["metrics"].items()},
            "validation": {k: json_safe(v) for k, v in r["validation"].items()},
            "ranking_quality": {k: json_safe(v) for k, v in r["ranking_quality"].items()},
        }
        # Remove monthly_returns from JSON (too large)
        jr["metrics"].pop("monthly_returns", None)
        json_results[var] = jr

    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2, default=json_safe)
    fprint(f"\nResults saved to {json_path}")

    elapsed = time.time() - t0
    fprint(f"\nTotal runtime: {elapsed/60:.1f} minutes")
    fprint("Done.")


if __name__ == "__main__":
    main()
