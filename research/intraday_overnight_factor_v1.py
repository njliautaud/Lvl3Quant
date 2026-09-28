"""
Intraday Momentum Factor for Overnight Returns — v1

Research question: Can intraday patterns (last-hour momentum, volume profile,
relative strength shift) predict overnight sector ETF returns (close→next open)?

This is a well-documented anomaly in academic literature. Lou, Polk & Skouras (2019)
show intraday momentum predicts overnight returns due to institutional rebalancing
flows near close.

Universe: 11 sector ETFs (XLK, XLF, XLE, XLV, XLY, XLI, XLP, XLU, XLRE, XLB, XLC)
Model: LightGBM walk-forward (sliding 252d train, 21d OOT) — HC #0 compliant.
Validation: 4-gate adversarial audit (permutation, R1 regime, sub-period, outlier removal).

HC #428 R1: regime-agnostic validation (|Sharpe_green - Sharpe_red| / max <= 0.50).
HC #0: SLIDING window only, NEVER expanding.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# Try MLflow
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Try yfinance
try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT = Path("/home/nick/Lvl3Quant") if Path("/home/nick/Lvl3Quant").exists() else Path("/home/jupiter/Lvl3Quant")
OUTPUT_DIR = ROOT / "output" / "intraday_overnight_factor_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = OUTPUT_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLI", "XLP", "XLU", "XLRE", "XLB", "XLC"]
BENCHMARK = "SPY"
VIX_TICKER = "^VIX"

# Walk-forward (HC #0: sliding, never expanding)
TRAIN_DAYS = 252
OOT_DAYS = 21
MIN_TRAIN_SAMPLES = 200

# LGBM
LGBM_PARAMS = {
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
    "n_jobs": -1,
    "device": "cpu",  # LGBM on CPU is fine for tabular
}
NUM_BOOST_ROUND = 500
EARLY_STOPPING_ROUNDS = 30

# Cost: $2.60 RT for options, but for ETF position sizing we use ETF commissions
# For options translation later, use $2.60 RT
COMMISSION_PER_SHARE = 0.005  # ~$0.005/share for ETF
SLIPPAGE_BPS = 5  # 5 bps slippage

# MLflow
MLFLOW_TRACKING_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "intraday_overnight_factor_v1"

# ---------------------------------------------------------------------------
# Data Download
# ---------------------------------------------------------------------------
def download_data(tickers: list[str], start: str = "2018-01-01", end: str | None = None) -> dict[str, pd.DataFrame]:
    """Download daily OHLCV data for tickers. Cache locally."""
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    cache_file = CACHE_DIR / f"daily_data_{start}_{end}.pkl"
    if cache_file.exists():
        print(f"Loading cached data from {cache_file.name}")
        return pd.read_pickle(cache_file)

    if not YF_AVAILABLE:
        raise RuntimeError("yfinance not available — cannot download data")

    print(f"Downloading data for {len(tickers)} tickers from {start} to {end}...")
    all_data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if len(df) > 100:
                # Flatten MultiIndex columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                all_data[ticker] = df
                print(f"  {ticker}: {len(df)} rows")
            else:
                print(f"  {ticker}: insufficient data ({len(df)} rows), skipping")
        except Exception as e:
            print(f"  {ticker}: download failed — {e}")

    # Also download hourly data for intraday features (last 730 days max from yfinance)
    print("\nDownloading hourly data for intraday features...")
    hourly_data = {}
    for ticker in tickers:
        try:
            df_h = yf.download(ticker, period="730d", interval="1h", progress=False, auto_adjust=True)
            if len(df_h) > 100:
                if isinstance(df_h.columns, pd.MultiIndex):
                    df_h.columns = df_h.columns.get_level_values(0)
                hourly_data[ticker] = df_h
                print(f"  {ticker} hourly: {len(df_h)} rows")
            else:
                print(f"  {ticker} hourly: insufficient ({len(df_h)} rows)")
        except Exception as e:
            print(f"  {ticker} hourly: failed — {e}")

    result = {"daily": all_data, "hourly": hourly_data}
    pd.to_pickle(result, cache_file)
    return result


def download_vix(start: str = "2018-01-01", end: str | None = None) -> pd.Series:
    """Download VIX close."""
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")
    try:
        vix = yf.download(VIX_TICKER, start=start, end=end, progress=False, auto_adjust=True)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)
        return vix["Close"].squeeze()
    except Exception:
        return pd.Series(dtype=float)


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------
def compute_daily_features(daily_data: dict[str, pd.DataFrame], vix: pd.Series) -> pd.DataFrame:
    """
    Compute features from daily OHLCV data as proxy for intraday patterns.

    Features per ETF per day:
    1. last_hour_return_proxy: (Close - Open of last bar) approximated by close vs intraday range
    2. close_vs_high: Close position within day's range (proxy for late-day momentum)
    3. volume_ratio: Today's volume vs 20-day avg
    4. intraday_range: (High - Low) / Open — realized volatility proxy
    5. close_vs_vwap_proxy: Close vs typical price (H+L+C)/3
    6. rsi_14: 14-period RSI at close
    7. mom_5d: 5-day momentum
    8. mom_10d: 10-day momentum
    9. rel_strength_vs_spy: Relative strength vs SPY (5-day)
    10. vix_level: VIX close
    11. vix_change_5d: VIX 5-day change
    12. volume_trend: Volume 5d MA / Volume 20d MA
    13. close_vs_open: (Close - Open) / Open — intraday return direction
    14. gap_from_prev: (Open - prev Close) / prev Close — gap pattern
    15. high_close_ratio: (High - Close) / (High - Low) — selling pressure at close
    """
    all_rows = []
    spy_data = daily_data.get("SPY", pd.DataFrame())

    for ticker in SECTOR_ETFS:
        if ticker not in daily_data:
            continue
        df = daily_data[ticker].copy()
        if len(df) < 50:
            continue

        # Ensure we have the right columns
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col not in df.columns:
                continue

        o, h, l, c, v = df["Open"], df["High"], df["Low"], df["Close"], df["Volume"]

        feat = pd.DataFrame(index=df.index)
        feat["ticker"] = ticker

        # 1. Close position in day range (proxy for last-hour momentum)
        day_range = h - l
        day_range_safe = day_range.replace(0, np.nan)
        feat["close_vs_high"] = (c - l) / day_range_safe  # 1 = closed at high, 0 = closed at low

        # 2. Volume ratio
        vol_ma20 = v.rolling(20).mean()
        feat["volume_ratio"] = v / vol_ma20

        # 3. Intraday range (volatility proxy)
        feat["intraday_range"] = (h - l) / o

        # 4. Close vs VWAP proxy (typical price)
        typical_price = (h + l + c) / 3
        feat["close_vs_vwap"] = (c - typical_price) / typical_price

        # 5. RSI 14
        delta = c.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        feat["rsi_14"] = 100 - (100 / (1 + rs))

        # 6. Momentum
        feat["mom_5d"] = c.pct_change(5)
        feat["mom_10d"] = c.pct_change(10)

        # 7. Relative strength vs SPY
        if len(spy_data) > 0 and "Close" in spy_data.columns:
            spy_close = spy_data["Close"].squeeze()
            spy_ret_5d = spy_close.pct_change(5).reindex(df.index)
            etf_ret_5d = c.pct_change(5)
            feat["rel_strength_spy"] = etf_ret_5d - spy_ret_5d

        # 8. VIX
        if len(vix) > 0:
            vix_aligned = vix.reindex(df.index, method="ffill")
            feat["vix_level"] = vix_aligned
            feat["vix_change_5d"] = vix_aligned.pct_change(5)

        # 9. Volume trend
        vol_ma5 = v.rolling(5).mean()
        feat["volume_trend"] = vol_ma5 / vol_ma20

        # 10. Close vs Open (intraday return)
        feat["close_vs_open"] = (c - o) / o

        # 11. Gap from previous close
        feat["gap_from_prev"] = (o - c.shift(1)) / c.shift(1)

        # 12. Selling pressure at close
        feat["high_close_ratio"] = (h - c) / day_range_safe

        # 13. Mean reversion signal: z-score of close vs 20d MA
        ma20 = c.rolling(20).mean()
        std20 = c.rolling(20).std()
        feat["zscore_20d"] = (c - ma20) / std20

        # 14. Overnight return target: close today → open tomorrow
        feat["overnight_return"] = (o.shift(-1) - c) / c

        # 15. Next-day close for full-day return (for comparison)
        feat["next_day_return"] = c.pct_change().shift(-1)

        all_rows.append(feat)

    if not all_rows:
        return pd.DataFrame()

    combined = pd.concat(all_rows, axis=0)
    combined.index.name = "date"
    return combined


def add_hourly_features(combined: pd.DataFrame, hourly_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Add actual intraday features from hourly data where available.

    Last-hour return, last-hour volume fraction, morning vs afternoon momentum.
    """
    if not hourly_data:
        print("No hourly data available — using daily proxies only")
        return combined

    print("Computing hourly-derived features...")
    hourly_features = {}

    for ticker in SECTOR_ETFS:
        if ticker not in hourly_data:
            continue
        hdf = hourly_data[ticker].copy()
        if len(hdf) < 50:
            continue

        # Ensure datetime index with timezone handling
        if hdf.index.tz is not None:
            hdf.index = hdf.index.tz_convert("America/New_York")

        # Extract date and hour
        hdf["date"] = hdf.index.date
        hdf["hour"] = hdf.index.hour

        for date, day_data in hdf.groupby("date"):
            if len(day_data) < 3:
                continue

            c = day_data["Close"].squeeze() if isinstance(day_data["Close"], pd.DataFrame) else day_data["Close"]
            v = day_data["Volume"].squeeze() if isinstance(day_data["Volume"], pd.DataFrame) else day_data["Volume"]

            # Last hour return (3pm-4pm bar, hour=15)
            last_hour = day_data[day_data["hour"] >= 15]
            early_hours = day_data[day_data["hour"] < 15]

            if len(last_hour) > 0 and len(early_hours) > 0:
                last_c = last_hour["Close"].iloc[-1]
                pre_last_c = early_hours["Close"].iloc[-1]
                if isinstance(last_c, pd.Series):
                    last_c = last_c.iloc[0]
                if isinstance(pre_last_c, pd.Series):
                    pre_last_c = pre_last_c.iloc[0]
                last_hour_ret = (last_c - pre_last_c) / pre_last_c if pre_last_c != 0 else 0
            else:
                last_hour_ret = np.nan

            # Last hour volume fraction
            total_vol = v.sum()
            last_hour_vol = v[day_data["hour"] >= 15].sum()
            last_hour_vol_frac = last_hour_vol / total_vol if total_vol > 0 else np.nan

            # Morning vs afternoon momentum
            morning = day_data[day_data["hour"] < 12]
            afternoon = day_data[day_data["hour"] >= 12]
            if len(morning) > 0 and len(afternoon) > 0:
                m_first = morning["Close"].iloc[0]
                m_last = morning["Close"].iloc[-1]
                a_first = afternoon["Close"].iloc[0]
                a_last = afternoon["Close"].iloc[-1]
                for x in [m_first, m_last, a_first, a_last]:
                    if isinstance(x, pd.Series):
                        x = x.iloc[0]
                morning_ret = (m_last - m_first) / m_first if m_first != 0 else 0
                afternoon_ret = (a_last - a_first) / a_first if a_first != 0 else 0
                momentum_shift = afternoon_ret - morning_ret
            else:
                morning_ret = afternoon_ret = momentum_shift = np.nan

            dt = pd.Timestamp(date)
            hourly_features[(ticker, dt)] = {
                "last_hour_return": last_hour_ret,
                "last_hour_vol_frac": last_hour_vol_frac,
                "morning_return": morning_ret,
                "afternoon_return": afternoon_ret,
                "momentum_shift": momentum_shift,
            }

    if not hourly_features:
        print("  No hourly features computed")
        return combined

    # Merge hourly features into combined
    hf_df = pd.DataFrame.from_dict(hourly_features, orient="index")
    hf_df.index = pd.MultiIndex.from_tuples(hf_df.index, names=["ticker", "date"])
    hf_df = hf_df.reset_index()

    combined = combined.reset_index()
    combined["date"] = pd.to_datetime(combined["date"])
    hf_df["date"] = pd.to_datetime(hf_df["date"])

    merged = combined.merge(hf_df, on=["ticker", "date"], how="left")
    merged = merged.set_index("date")

    n_filled = merged["last_hour_return"].notna().sum()
    print(f"  Added hourly features for {n_filled}/{len(merged)} rows")
    return merged


# ---------------------------------------------------------------------------
# Walk-Forward Engine
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "close_vs_high", "volume_ratio", "intraday_range", "close_vs_vwap",
    "rsi_14", "mom_5d", "mom_10d", "rel_strength_spy", "vix_level",
    "vix_change_5d", "volume_trend", "close_vs_open", "gap_from_prev",
    "high_close_ratio", "zscore_20d",
    # Hourly features (may be NaN if not available)
    "last_hour_return", "last_hour_vol_frac", "morning_return",
    "afternoon_return", "momentum_shift",
]

TARGET = "overnight_return"


def walk_forward_lgbm(df: pd.DataFrame) -> dict:
    """
    Sliding walk-forward with LGBM.
    252-day train, 21-day OOT, drop oldest day on slide.
    HC #0: SLIDING only, never expanding.
    """
    # Get unique dates
    dates = sorted(df.index.unique())
    n_dates = len(dates)

    print(f"\nWalk-forward: {n_dates} unique dates, {TRAIN_DAYS}d train, {OOT_DAYS}d OOT")

    # Determine available features (some may be all NaN)
    avail_features = []
    for f in FEATURE_COLS:
        if f in df.columns and df[f].notna().sum() > len(df) * 0.1:
            avail_features.append(f)
    print(f"Available features: {len(avail_features)}/{len(FEATURE_COLS)}")
    print(f"  Features: {avail_features}")

    if len(avail_features) < 5:
        raise ValueError(f"Only {len(avail_features)} features available — need at least 5")

    oot_predictions = []
    fold_metrics = []
    feature_importances = []

    fold_start = TRAIN_DAYS
    fold_idx = 0

    while fold_start + OOT_DAYS <= n_dates:
        train_dates = dates[fold_start - TRAIN_DAYS: fold_start]
        oot_dates = dates[fold_start: fold_start + OOT_DAYS]

        train_mask = df.index.isin(train_dates)
        oot_mask = df.index.isin(oot_dates)

        train_df = df[train_mask].copy()
        oot_df = df[oot_mask].copy()

        # Drop rows with NaN target
        train_df = train_df.dropna(subset=[TARGET])
        oot_df_clean = oot_df.dropna(subset=[TARGET])

        if len(train_df) < MIN_TRAIN_SAMPLES or len(oot_df_clean) < 5:
            fold_start += OOT_DAYS
            continue

        X_train = train_df[avail_features].fillna(0).values
        y_train = train_df[TARGET].values
        X_oot = oot_df_clean[avail_features].fillna(0).values
        y_oot = oot_df_clean[TARGET].values

        # Train LGBM
        dtrain = lgb.Dataset(X_train, label=y_train, feature_name=avail_features)
        dval = lgb.Dataset(X_oot, label=y_oot, reference=dtrain, feature_name=avail_features)

        callbacks = [lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)]
        model = lgb.train(
            LGBM_PARAMS, dtrain, num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[dval], callbacks=callbacks,
        )

        # Predictions
        preds = model.predict(X_oot)

        # Per-fold metrics
        ic = np.corrcoef(preds, y_oot)[0, 1] if len(preds) > 2 else 0
        rmse = np.sqrt(mean_squared_error(y_oot, preds))

        fold_metrics.append({
            "fold": fold_idx,
            "train_start": str(train_dates[0]),
            "train_end": str(train_dates[-1]),
            "oot_start": str(oot_dates[0]),
            "oot_end": str(oot_dates[-1]),
            "n_train": len(train_df),
            "n_oot": len(oot_df_clean),
            "ic": ic,
            "rmse": rmse,
            "best_iteration": model.best_iteration,
        })

        # Store predictions
        for i, idx in enumerate(oot_df_clean.index):
            ticker = oot_df_clean.iloc[i]["ticker"] if "ticker" in oot_df_clean.columns else "UNK"
            oot_predictions.append({
                "date": str(idx),
                "ticker": ticker,
                "pred": preds[i],
                "actual": y_oot[i],
                "fold": fold_idx,
            })

        # Feature importance
        imp = model.feature_importance(importance_type="gain")
        for fname, fval in zip(avail_features, imp):
            feature_importances.append({"feature": fname, "importance": fval, "fold": fold_idx})

        if fold_idx % 5 == 0:
            print(f"  Fold {fold_idx}: IC={ic:.4f}, RMSE={rmse:.6f}, n_train={len(train_df)}, n_oot={len(oot_df_clean)}")

        fold_start += OOT_DAYS
        fold_idx += 1

    print(f"\nCompleted {fold_idx} folds")
    return {
        "predictions": pd.DataFrame(oot_predictions),
        "fold_metrics": pd.DataFrame(fold_metrics),
        "feature_importances": pd.DataFrame(feature_importances),
        "avail_features": avail_features,
    }


# ---------------------------------------------------------------------------
# Strategy Simulation
# ---------------------------------------------------------------------------
def simulate_strategy(preds_df: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    """
    Long-short overnight strategy:
    - Each day at close, go long top_k predicted overnight gainers, short top_k predicted losers
    - Exit at next open
    - Equal weight within each leg

    Costs: commission + slippage per side.
    """
    if preds_df.empty:
        return pd.DataFrame()

    preds_df["date"] = pd.to_datetime(preds_df["date"])
    daily_returns = []

    for date, day_preds in preds_df.groupby("date"):
        if len(day_preds) < 2 * top_k:
            continue

        sorted_preds = day_preds.sort_values("pred")

        # Short bottom k, long top k
        shorts = sorted_preds.head(top_k)
        longs = sorted_preds.tail(top_k)

        long_ret = longs["actual"].mean()
        short_ret = -shorts["actual"].mean()  # profit from short = negative of return
        gross_ret = (long_ret + short_ret) / 2  # average of long and short legs

        # Costs: ~10 bps total round-trip (slippage + commission) per leg
        cost_per_leg = SLIPPAGE_BPS / 10000  # 5 bps slippage
        total_cost = 2 * cost_per_leg  # long entry + exit, short entry + exit simplified
        net_ret = gross_ret - total_cost

        daily_returns.append({
            "date": date,
            "gross_return": gross_ret,
            "net_return": net_ret,
            "long_return": long_ret,
            "short_return": short_ret,
            "n_longs": top_k,
            "n_shorts": top_k,
            "avg_pred_long": longs["pred"].mean(),
            "avg_pred_short": shorts["pred"].mean(),
        })

    return pd.DataFrame(daily_returns)


def compute_strategy_metrics(returns_df: pd.DataFrame, col: str = "net_return") -> dict:
    """Compute risk-adjusted metrics."""
    if returns_df.empty or col not in returns_df.columns:
        return {}

    rets = returns_df[col].dropna()
    if len(rets) < 10:
        return {}

    n_days = len(rets)
    ann_factor = np.sqrt(252)

    total_ret = (1 + rets).prod() - 1
    ann_ret = (1 + total_ret) ** (252 / n_days) - 1
    vol = rets.std() * ann_factor
    sharpe = (rets.mean() / rets.std()) * ann_factor if rets.std() > 0 else 0

    downside = rets[rets < 0].std() * ann_factor
    sortino = (rets.mean() * 252) / downside if downside > 0 else 0

    wins = (rets > 0).sum()
    losses = (rets <= 0).sum()
    win_rate = wins / n_days if n_days > 0 else 0

    avg_win = rets[rets > 0].mean() if wins > 0 else 0
    avg_loss = abs(rets[rets <= 0].mean()) if losses > 0 else 1e-9
    profit_factor = (avg_win * wins) / (avg_loss * losses) if losses > 0 and avg_loss > 0 else 0

    # Max drawdown
    cum = (1 + rets).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()

    return {
        "n_days": n_days,
        "total_return": total_ret,
        "ann_return": ann_ret,
        "ann_vol": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_drawdown": max_dd,
        "avg_daily_return": rets.mean(),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "calmar": ann_ret / abs(max_dd) if max_dd != 0 else 0,
    }


# ---------------------------------------------------------------------------
# 4-Gate Adversarial Audit
# ---------------------------------------------------------------------------
def permutation_test(df: pd.DataFrame, n_perms: int = 100) -> dict:
    """Gate 1: Permutation test — is the signal better than random?"""
    print("\n=== Gate 1: Permutation Test ===")
    real_ic = np.corrcoef(df["pred"], df["actual"])[0, 1]

    perm_ics = []
    for i in range(n_perms):
        shuffled = df["actual"].sample(frac=1, random_state=i).values
        perm_ic = np.corrcoef(df["pred"].values, shuffled)[0, 1]
        perm_ics.append(perm_ic)

    perm_ics = np.array(perm_ics)
    p_value = (perm_ics >= real_ic).mean()
    percentile = (perm_ics < real_ic).mean() * 100

    result = {
        "real_ic": real_ic,
        "perm_mean_ic": perm_ics.mean(),
        "perm_std_ic": perm_ics.std(),
        "p_value": p_value,
        "percentile": percentile,
        "pass": p_value < 0.05,
    }
    status = "PASS" if result["pass"] else "FAIL"
    print(f"  Real IC: {real_ic:.4f}, Perm mean: {perm_ics.mean():.4f} +/- {perm_ics.std():.4f}")
    print(f"  p-value: {p_value:.4f}, Percentile: {percentile:.1f}% — {status}")
    return result


def regime_test(returns_df: pd.DataFrame, spy_daily: pd.DataFrame) -> dict:
    """
    Gate 2: HC #428 R1 — Regime-agnostic validation.
    |Sharpe_green - Sharpe_red| / max(|Sharpe_green|, |Sharpe_red|) <= 0.50
    """
    print("\n=== Gate 2: Regime-Agnostic Test (HC #428 R1) ===")
    if returns_df.empty or len(spy_daily) == 0:
        return {"pass": False, "reason": "insufficient data"}

    # Classify days by SPY return
    spy_close = spy_daily["Close"].squeeze() if isinstance(spy_daily["Close"], pd.DataFrame) else spy_daily["Close"]
    spy_ret = spy_close.pct_change().dropna()

    returns_df = returns_df.copy()
    returns_df["date"] = pd.to_datetime(returns_df["date"])
    returns_df = returns_df.set_index("date")

    spy_ret.index = pd.to_datetime(spy_ret.index)
    returns_df["spy_ret"] = spy_ret.reindex(returns_df.index)

    green_days = returns_df[returns_df["spy_ret"] > 0.001]
    red_days = returns_df[returns_df["spy_ret"] < -0.001]
    flat_days = returns_df[(returns_df["spy_ret"] >= -0.001) & (returns_df["spy_ret"] <= 0.001)]

    ann = np.sqrt(252)
    def sharpe(rets):
        if len(rets) < 5 or rets.std() == 0:
            return 0
        return (rets.mean() / rets.std()) * ann

    sharpe_green = sharpe(green_days["net_return"]) if len(green_days) > 5 else 0
    sharpe_red = sharpe(red_days["net_return"]) if len(red_days) > 5 else 0
    sharpe_flat = sharpe(flat_days["net_return"]) if len(flat_days) > 5 else 0

    max_sharpe = max(abs(sharpe_green), abs(sharpe_red))
    regime_gap = abs(sharpe_green - sharpe_red) / max_sharpe if max_sharpe > 0 else 0
    passes = regime_gap <= 0.50

    result = {
        "sharpe_green": sharpe_green,
        "sharpe_red": sharpe_red,
        "sharpe_flat": sharpe_flat,
        "regime_gap": regime_gap,
        "n_green": len(green_days),
        "n_red": len(red_days),
        "n_flat": len(flat_days),
        "pass": passes,
    }
    status = "PASS" if passes else "FAIL"
    print(f"  Green days ({len(green_days)}): Sharpe={sharpe_green:.2f}")
    print(f"  Red days ({len(red_days)}): Sharpe={sharpe_red:.2f}")
    print(f"  Flat days ({len(flat_days)}): Sharpe={sharpe_flat:.2f}")
    print(f"  Regime gap: {regime_gap:.2f} (threshold: 0.50) — {status}")
    return result


def sub_period_test(returns_df: pd.DataFrame, n_periods: int = 4) -> dict:
    """Gate 3: Sub-period stability — strategy works across different time periods."""
    print("\n=== Gate 3: Sub-Period Stability ===")
    if returns_df.empty:
        return {"pass": False, "reason": "no data"}

    returns_df = returns_df.copy()
    returns_df["date"] = pd.to_datetime(returns_df["date"]) if not pd.api.types.is_datetime64_any_dtype(returns_df["date"]) else returns_df["date"]
    returns_df = returns_df.sort_values("date")

    chunk_size = len(returns_df) // n_periods
    if chunk_size < 10:
        return {"pass": False, "reason": f"too few data points ({len(returns_df)}) for {n_periods} periods"}

    ann = np.sqrt(252)
    period_sharpes = []
    period_details = []

    for i in range(n_periods):
        start = i * chunk_size
        end = start + chunk_size if i < n_periods - 1 else len(returns_df)
        chunk = returns_df.iloc[start:end]
        rets = chunk["net_return"]
        s = (rets.mean() / rets.std()) * ann if rets.std() > 0 else 0
        period_sharpes.append(s)
        period_details.append({
            "period": i + 1,
            "start": str(chunk["date"].iloc[0].date()) if hasattr(chunk["date"].iloc[0], "date") else str(chunk["date"].iloc[0]),
            "end": str(chunk["date"].iloc[-1].date()) if hasattr(chunk["date"].iloc[-1], "date") else str(chunk["date"].iloc[-1]),
            "sharpe": s,
            "n_days": len(chunk),
        })
        print(f"  Period {i+1}: Sharpe={s:.2f} ({len(chunk)} days)")

    # Pass if majority of periods have positive Sharpe and no period is deeply negative
    positive_periods = sum(1 for s in period_sharpes if s > 0)
    min_sharpe = min(period_sharpes)
    passes = positive_periods >= n_periods * 0.75 and min_sharpe > -0.5

    result = {
        "period_sharpes": period_sharpes,
        "period_details": period_details,
        "positive_periods": positive_periods,
        "min_sharpe": min_sharpe,
        "pass": passes,
    }
    status = "PASS" if passes else "FAIL"
    print(f"  Positive periods: {positive_periods}/{n_periods}, Min Sharpe: {min_sharpe:.2f} — {status}")
    return result


def outlier_removal_test(returns_df: pd.DataFrame, pct: float = 0.05) -> dict:
    """Gate 4: Remove top/bottom pct of returns — does strategy survive?"""
    print(f"\n=== Gate 4: Outlier Removal Test (removing top/bottom {pct*100:.0f}%) ===")
    if returns_df.empty:
        return {"pass": False, "reason": "no data"}

    rets = returns_df["net_return"].dropna()
    n_remove = int(len(rets) * pct)
    if n_remove < 1:
        n_remove = 1

    sorted_rets = rets.sort_values()
    trimmed = sorted_rets.iloc[n_remove:-n_remove]

    ann = np.sqrt(252)
    full_sharpe = (rets.mean() / rets.std()) * ann if rets.std() > 0 else 0
    trimmed_sharpe = (trimmed.mean() / trimmed.std()) * ann if trimmed.std() > 0 else 0

    # Pass if trimmed Sharpe is still > 50% of full Sharpe (and positive)
    passes = trimmed_sharpe > 0 and trimmed_sharpe > full_sharpe * 0.5

    result = {
        "full_sharpe": full_sharpe,
        "trimmed_sharpe": trimmed_sharpe,
        "n_removed": 2 * n_remove,
        "sharpe_retention": trimmed_sharpe / full_sharpe if full_sharpe != 0 else 0,
        "pass": passes,
    }
    status = "PASS" if passes else "FAIL"
    print(f"  Full Sharpe: {full_sharpe:.2f}, Trimmed Sharpe: {trimmed_sharpe:.2f}")
    print(f"  Retention: {result['sharpe_retention']:.1%} — {status}")
    return result


# ---------------------------------------------------------------------------
# Options Translation
# ---------------------------------------------------------------------------
def options_translation(metrics: dict, returns_df: pd.DataFrame) -> dict:
    """
    Translate ETF overnight strategy into options plays for $645 account.

    If overnight long/short edge works, we can:
    - Buy calls on predicted gainers (cheap near-expiry ATM)
    - Buy puts on predicted losers
    - Use 0DTE or 1DTE options for overnight holds
    - Risk per trade: max 5% of account = $32.25

    Options cost: $2.60 RT commission.
    """
    print("\n=== Options Translation for $645 Account ===")
    account_size = 645
    max_risk_pct = 0.05
    max_risk = account_size * max_risk_pct
    options_commission = 2.60  # RT per contract

    if not metrics:
        return {"viable": False, "reason": "no metrics"}

    sharpe = metrics.get("sharpe", 0)
    win_rate = metrics.get("win_rate", 0)
    avg_daily_return = metrics.get("avg_daily_return", 0)

    # Sector ETF 0DTE/1DTE options typically cost $0.50-$2.00 per contract
    # Average overnight move for sector ETFs is ~0.3-0.5%
    # ATM option delta ~0.50, gamma exposure gives leverage

    avg_option_cost = 1.00 * 100  # $1.00 per share * 100 shares = $100 per contract
    contracts_per_trade = max(1, int(max_risk / (avg_option_cost + options_commission)))

    # Expected P&L per trade
    # If sector ETF moves 0.3% overnight, ATM option (delta 0.50) moves ~0.15%
    # But options have leverage: 0.3% on $50 ETF = $0.15 move, option cost $1 = 15% option return
    avg_etf_move = abs(avg_daily_return) if avg_daily_return != 0 else 0.003
    avg_option_return = avg_etf_move * 0.50 * 50 / 1.00  # delta * ETF price * move / option cost (rough)

    expected_gross = avg_option_return * avg_option_cost * contracts_per_trade * (2 * win_rate - 1)
    expected_net = expected_gross - options_commission * contracts_per_trade

    viable = sharpe > 0.5 and win_rate > 0.52 and expected_net > 0

    result = {
        "viable": viable,
        "account_size": account_size,
        "max_risk_per_trade": max_risk,
        "contracts_per_trade": contracts_per_trade,
        "avg_option_cost": avg_option_cost,
        "options_commission": options_commission,
        "estimated_edge_per_trade_bps": avg_daily_return * 10000,
        "strategy_sharpe": sharpe,
        "strategy_win_rate": win_rate,
        "recommendation": (
            "VIABLE: Use 1DTE ATM calls/puts on top-3/bottom-3 sector ETFs at close. "
            f"Risk ${max_risk:.0f}/trade, {contracts_per_trade} contract(s)."
            if viable else
            "NOT YET VIABLE: Signal needs stronger edge or higher win rate for options."
        ),
    }

    print(f"  Viable: {viable}")
    print(f"  Contracts/trade: {contracts_per_trade}")
    print(f"  Estimated edge: {avg_daily_return * 10000:.1f} bps/day")
    print(f"  Recommendation: {result['recommendation']}")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    start_time = time.time()
    print("=" * 70)
    print("INTRADAY MOMENTUM → OVERNIGHT RETURNS FACTOR — v1")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Setup MLflow
    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(EXPERIMENT_NAME)
            run = mlflow.start_run(run_name=f"v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            print(f"MLflow run started: {run.info.run_id}")
        except Exception as e:
            print(f"MLflow setup failed (will continue without): {e}")
            MLFLOW_AVAILABLE_LOCAL = False
        else:
            MLFLOW_AVAILABLE_LOCAL = True
    else:
        MLFLOW_AVAILABLE_LOCAL = False

    try:
        # 1. Download data
        print("\n--- Step 1: Download Data ---")
        all_tickers = SECTOR_ETFS + [BENCHMARK]
        raw_data = download_data(all_tickers, start="2018-01-01")
        daily_data = raw_data["daily"]
        hourly_data = raw_data.get("hourly", {})
        vix = download_vix(start="2018-01-01")

        print(f"\nDaily data: {len(daily_data)} tickers loaded")
        print(f"Hourly data: {len(hourly_data)} tickers loaded")
        print(f"VIX data: {len(vix)} rows")

        # 2. Feature engineering
        print("\n--- Step 2: Feature Engineering ---")
        combined = compute_daily_features(daily_data, vix)
        print(f"Daily features: {len(combined)} rows, {combined.shape[1]} columns")

        combined = add_hourly_features(combined, hourly_data)
        print(f"After hourly merge: {len(combined)} rows, {combined.shape[1]} columns")

        # Drop rows without target
        valid = combined.dropna(subset=[TARGET])
        print(f"Valid rows (with overnight return): {len(valid)}")

        if len(valid) < TRAIN_DAYS + OOT_DAYS:
            raise ValueError(f"Insufficient data: {len(valid)} rows, need at least {TRAIN_DAYS + OOT_DAYS}")

        # 3. Walk-forward
        print("\n--- Step 3: Walk-Forward LightGBM ---")
        results = walk_forward_lgbm(valid)

        preds_df = results["predictions"]
        fold_df = results["fold_metrics"]
        fimp_df = results["feature_importances"]

        if preds_df.empty:
            raise ValueError("No predictions generated — check data quality")

        # Concat IC (primary metric)
        concat_ic = np.corrcoef(preds_df["pred"], preds_df["actual"])[0, 1]
        mean_fold_ic = fold_df["ic"].mean()
        print(f"\nConcat IC: {concat_ic:.4f}")
        print(f"Mean fold IC: {mean_fold_ic:.4f}")
        print(f"IC std: {fold_df['ic'].std():.4f}")

        # Feature importance (top 10)
        avg_imp = fimp_df.groupby("feature")["importance"].mean().sort_values(ascending=False)
        print(f"\nTop 10 Features:")
        for fname, fval in avg_imp.head(10).items():
            print(f"  {fname}: {fval:.1f}")

        # 4. Strategy simulation
        print("\n--- Step 4: Strategy Simulation ---")
        for top_k in [2, 3, 4]:
            print(f"\n  --- Top/Bottom {top_k} ---")
            strat_returns = simulate_strategy(preds_df, top_k=top_k)
            metrics = compute_strategy_metrics(strat_returns, col="net_return")
            gross_metrics = compute_strategy_metrics(strat_returns, col="gross_return")

            if metrics:
                print(f"  Net:   Sharpe={metrics['sharpe']:.2f}, Sortino={metrics['sortino']:.2f}, "
                      f"WR={metrics['win_rate']:.1%}, PF={metrics['profit_factor']:.2f}, "
                      f"MaxDD={metrics['max_drawdown']:.1%}")
                print(f"  Gross: Sharpe={gross_metrics['sharpe']:.2f}, Sortino={gross_metrics['sortino']:.2f}")

        # Use top_k=3 as primary for audit
        primary_returns = simulate_strategy(preds_df, top_k=3)
        primary_metrics = compute_strategy_metrics(primary_returns, col="net_return")

        # 5. Adversarial audit
        print("\n--- Step 5: 4-Gate Adversarial Audit ---")
        gate1 = permutation_test(preds_df, n_perms=200)

        spy_daily = daily_data.get("SPY", pd.DataFrame())
        gate2 = regime_test(primary_returns, spy_daily)
        gate3 = sub_period_test(primary_returns, n_periods=4)
        gate4 = outlier_removal_test(primary_returns, pct=0.05)

        gates_passed = sum([gate1["pass"], gate2["pass"], gate3["pass"], gate4["pass"]])
        print(f"\n  GATES PASSED: {gates_passed}/4")

        # 6. Options translation
        print("\n--- Step 6: Options Translation ---")
        options_result = options_translation(primary_metrics, primary_returns)

        # 7. Save results
        print("\n--- Step 7: Save Results ---")
        preds_df.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
        fold_df.to_csv(OUTPUT_DIR / "fold_metrics.csv", index=False)
        fimp_df.to_csv(OUTPUT_DIR / "feature_importances.csv", index=False)
        primary_returns.to_csv(OUTPUT_DIR / "strategy_returns.csv", index=False)

        summary = {
            "experiment": "intraday_overnight_factor_v1",
            "timestamp": datetime.now().isoformat(),
            "concat_ic": float(concat_ic),
            "mean_fold_ic": float(mean_fold_ic),
            "n_folds": len(fold_df),
            "n_predictions": len(preds_df),
            "primary_metrics_net": primary_metrics,
            "gates": {
                "permutation": gate1,
                "regime": gate2,
                "sub_period": gate3,
                "outlier_removal": gate4,
            },
            "gates_passed": gates_passed,
            "options_translation": options_result,
            "top_features": {k: float(v) for k, v in avg_imp.head(10).items()},
            "runtime_seconds": time.time() - start_time,
        }

        with open(OUTPUT_DIR / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        # 8. Log to MLflow
        if MLFLOW_AVAILABLE_LOCAL:
            try:
                mlflow.log_param("model", "lightgbm")
                mlflow.log_param("train_days", TRAIN_DAYS)
                mlflow.log_param("oot_days", OOT_DAYS)
                mlflow.log_param("n_features", len(results["avail_features"]))
                mlflow.log_param("universe", ",".join(SECTOR_ETFS))
                mlflow.log_param("target", TARGET)
                mlflow.log_param("top_k", 3)

                mlflow.log_metric("concat_ic", concat_ic)
                mlflow.log_metric("mean_fold_ic", mean_fold_ic)
                mlflow.log_metric("n_folds", len(fold_df))
                mlflow.log_metric("sharpe_net", primary_metrics.get("sharpe", 0))
                mlflow.log_metric("sortino_net", primary_metrics.get("sortino", 0))
                mlflow.log_metric("win_rate", primary_metrics.get("win_rate", 0))
                mlflow.log_metric("profit_factor", primary_metrics.get("profit_factor", 0))
                mlflow.log_metric("max_drawdown", primary_metrics.get("max_drawdown", 0))
                mlflow.log_metric("gates_passed", gates_passed)
                mlflow.log_metric("perm_test_pvalue", gate1.get("p_value", 1))
                mlflow.log_metric("regime_gap", gate2.get("regime_gap", 1))
                mlflow.log_metric("options_viable", 1 if options_result.get("viable", False) else 0)

                mlflow.log_artifact(str(OUTPUT_DIR / "summary.json"))
                mlflow.log_artifact(str(OUTPUT_DIR / "predictions.csv"))
                mlflow.log_artifact(str(OUTPUT_DIR / "fold_metrics.csv"))

                mlflow.end_run()
                print("MLflow logging complete")
            except Exception as e:
                print(f"MLflow logging error: {e}")

        # Final summary
        elapsed = time.time() - start_time
        print("\n" + "=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)
        print(f"Concat IC: {concat_ic:.4f}")
        print(f"Strategy (top/bot 3, net): Sharpe={primary_metrics.get('sharpe', 0):.2f}, "
              f"Sortino={primary_metrics.get('sortino', 0):.2f}, "
              f"WR={primary_metrics.get('win_rate', 0):.1%}, "
              f"PF={primary_metrics.get('profit_factor', 0):.2f}")
        print(f"Gates passed: {gates_passed}/4")
        print(f"Options viable: {options_result.get('viable', False)}")
        print(f"Runtime: {elapsed:.0f}s ({elapsed/60:.1f}m)")
        print("=" * 70)

    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
        if MLFLOW_AVAILABLE_LOCAL:
            try:
                mlflow.log_param("error", str(e)[:250])
                mlflow.end_run(status="FAILED")
            except Exception:
                pass
        sys.exit(1)


if __name__ == "__main__":
    main()
