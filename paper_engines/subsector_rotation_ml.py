#!/usr/bin/env python3
"""
Sub-Sector Rotation ML Alpha Research
=======================================
LGBM model that predicts which sub-sectors will outperform over 1-4 weeks.

Features:
  - Rotation velocity (momentum of relative strength changes)
  - Volume patterns (flow acceleration)
  - Cross-sector correlations (when X rotates in, what follows?)
  - Macro regime (VIX level, yield curve, risk appetite)
  - Technical (RSI, SMA position, breadth)

Validation:
  - Walk-forward sliding window (NOT expanding — HC #0)
  - Train 252d, predict 21d, slide forward 21d
  - Reports Sharpe, Sortino, hit rate, regime stratification

Usage:
    # Run backtest
    python3 paper_engines/subsector_rotation_ml.py --backtest

    # Generate live predictions
    python3 paper_engines/subsector_rotation_ml.py --predict

    # Full run (backtest + predict)
    python3 paper_engines/subsector_rotation_ml.py
"""

import argparse
import json
import logging
import pickle
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
STATE_DIR = BASE_DIR / "state"
STATE_DIR.mkdir(exist_ok=True)
DATA_DIR = BASE_DIR / "data"

LOG_PATH = LOG_DIR / "subsector_rotation_ml.log"
MODEL_PATH = DATA_DIR / "subsector_rotation_lgbm.pkl"
PREDICTIONS_PATH = STATE_DIR / "subsector_rotation_predictions.json"
BACKTEST_PATH = STATE_DIR / "subsector_rotation_backtest.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SubSectorML] %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# Import taxonomy from tracker
from subsector_rotation_tracker import (
    SUBSECTOR_UNIVERSE,
    ALL_TICKERS,
    ALL_ETFS,
    MACRO_TICKERS,
    ALL_DOWNLOAD_TICKERS,
    compute_rsi,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRAIN_WINDOW = 252       # ~1 year sliding window
PREDICT_HORIZON = 21     # 21 trading days forward (1 month)
SLIDE_STEP = 21          # Slide forward 21 days each fold
MIN_HISTORY = 126        # 6 months minimum for features
TOPK = 5                 # Top-K sub-sectors to go long
BOTTOMK = 3              # Bottom-K sub-sectors to go short (for long/short variant)

LGBM_PARAMS = {
    "objective": "regression",
    "metric": "mse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 10,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "verbose": -1,
    "n_jobs": -1,
    "seed": 42,
}
N_ROUNDS = 200


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

def build_subsector_return_panel(data) -> pd.DataFrame:
    """Build a panel of daily sub-sector equal-weight returns."""
    subsector_returns = {}

    for name, info in SUBSECTOR_UNIVERSE.items():
        tickers = info["tickers"]
        close_cols = {}
        for t in tickers:
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    if t in data["Close"].columns:
                        close_cols[t] = data["Close"][t]
                elif t in data.columns:
                    close_cols[t] = data[t]
            except Exception:
                continue

        if len(close_cols) < 2:
            continue

        df = pd.DataFrame(close_cols).dropna()
        if len(df) < MIN_HISTORY:
            continue

        # Equal-weight return
        ret = df.pct_change()
        subsector_returns[name] = ret.mean(axis=1)

    return pd.DataFrame(subsector_returns).dropna()


def build_macro_features(data) -> pd.DataFrame:
    """Build macro feature dataframe aligned to trading days."""
    features = {}

    def safe_close(ticker):
        try:
            if isinstance(data.columns, pd.MultiIndex):
                if ticker in data["Close"].columns:
                    return data["Close"][ticker]
            elif ticker in data.columns:
                return data[ticker]
        except Exception:
            pass
        return None

    # SPY momentum
    spy = safe_close("SPY")
    if spy is not None:
        features["spy_ret_5d"] = spy.pct_change(5)
        features["spy_ret_21d"] = spy.pct_change(21)
        features["spy_ret_63d"] = spy.pct_change(63)
        features["spy_vol_21d"] = spy.pct_change().rolling(21).std() * np.sqrt(252)
        features["spy_sma20_dist"] = (spy / spy.rolling(20).mean() - 1)
        features["spy_sma50_dist"] = (spy / spy.rolling(50).mean() - 1)

    # VIX
    vix = safe_close("^VIX")
    if vix is not None:
        features["vix_level"] = vix
        features["vix_chg_5d"] = vix.diff(5)
        features["vix_chg_21d"] = vix.diff(21)
        features["vix_sma20_ratio"] = vix / vix.rolling(20).mean()

    # TLT (bonds)
    tlt = safe_close("TLT")
    if tlt is not None:
        features["tlt_ret_21d"] = tlt.pct_change(21)

    # HYG (credit spreads proxy)
    hyg = safe_close("HYG")
    if hyg is not None:
        features["hyg_ret_21d"] = hyg.pct_change(21)
        if tlt is not None:
            features["credit_spread_proxy"] = hyg.pct_change(21) - tlt.pct_change(21)

    # Gold
    gld = safe_close("GLD")
    if gld is not None:
        features["gld_ret_21d"] = gld.pct_change(21)

    return pd.DataFrame(features)


def build_features_for_subsector(
    subsector_ret: pd.Series,
    all_returns: pd.DataFrame,
    macro_feat: pd.DataFrame,
    spy_close: Optional[pd.Series],
    subsector_name: str,
) -> pd.DataFrame:
    """Build feature matrix for one sub-sector at each time step."""
    feat = pd.DataFrame(index=subsector_ret.index)

    # --- Own momentum ---
    cum = (1 + subsector_ret).cumprod()
    feat["ret_5d"] = cum.pct_change(5)
    feat["ret_10d"] = cum.pct_change(10)
    feat["ret_21d"] = cum.pct_change(21)
    feat["ret_63d"] = cum.pct_change(63)

    # --- Momentum acceleration ---
    feat["mom_accel"] = feat["ret_5d"] - feat["ret_5d"].shift(5)

    # --- Volatility ---
    feat["vol_21d"] = subsector_ret.rolling(21).std() * np.sqrt(252)
    feat["vol_ratio"] = feat["vol_21d"] / feat["vol_21d"].shift(21)

    # --- RSI ---
    rsi = compute_rsi(cum, 14)
    feat["rsi_14"] = rsi

    # --- Relative strength vs SPY ---
    if spy_close is not None and len(spy_close) >= 22:
        spy_ret_21d = spy_close.pct_change(21)
        # Align
        common = feat.index.intersection(spy_ret_21d.index)
        aligned_spy = spy_ret_21d.reindex(common)
        aligned_self = feat["ret_21d"].reindex(common)
        rel_str = aligned_self - aligned_spy
        feat["rel_str_21d"] = rel_str

        # Rotation velocity (change in relative strength)
        feat["rotation_velocity"] = feat.get("rel_str_21d", pd.Series(dtype=float)).diff(10)

    # --- Cross-sector correlation features ---
    # How correlated is this sub-sector with others recently?
    if all_returns is not None and len(all_returns.columns) > 3:
        # Rolling 21d correlation with average of all other sub-sectors
        others = all_returns.drop(columns=[subsector_name], errors="ignore")
        if not others.empty:
            avg_others = others.mean(axis=1)
            feat["cross_corr_21d"] = subsector_ret.rolling(21).corr(avg_others)

        # Correlation with top-performing sub-sectors (momentum leaders)
        # Use lagged momentum rankings
        mom_21d = all_returns.rolling(21).sum()
        # Rank each day
        ranks = mom_21d.rank(axis=1, ascending=False)
        # Average rank of this subsector
        if subsector_name in ranks.columns:
            feat["momentum_rank"] = ranks[subsector_name]
            feat["momentum_rank_chg"] = feat["momentum_rank"].diff(10)

    # --- SMA distance ---
    feat["sma20_dist"] = cum / cum.rolling(20).mean() - 1
    feat["sma50_dist"] = cum / cum.rolling(50).mean() - 1

    # --- Breadth proxy (above/below SMA) ---
    feat["above_sma20"] = (cum > cum.rolling(20).mean()).astype(float)
    feat["above_sma50"] = (cum > cum.rolling(50).mean()).astype(float)

    # --- Merge macro features ---
    if macro_feat is not None and not macro_feat.empty:
        common = feat.index.intersection(macro_feat.index)
        for col in macro_feat.columns:
            feat[col] = macro_feat[col].reindex(feat.index)

    return feat


# ---------------------------------------------------------------------------
# Walk-Forward Backtest
# ---------------------------------------------------------------------------

def run_backtest(data) -> dict:
    """Run walk-forward sliding window backtest."""
    if not HAS_LGBM:
        log.error("LightGBM not installed. Cannot run backtest.")
        return {"error": "LightGBM required"}

    log.info("Building sub-sector return panel...")
    returns_panel = build_subsector_return_panel(data)
    log.info(f"  {len(returns_panel)} trading days, {len(returns_panel.columns)} sub-sectors")

    if len(returns_panel) < TRAIN_WINDOW + PREDICT_HORIZON + 60:
        log.error(f"Insufficient data: {len(returns_panel)} days, need {TRAIN_WINDOW + PREDICT_HORIZON + 60}")
        return {"error": "Insufficient data"}

    # SPY close for relative strength
    spy_close = None
    try:
        if isinstance(data.columns, pd.MultiIndex):
            spy_close = data["Close"]["SPY"]
        elif "SPY" in data.columns:
            spy_close = data["SPY"]
    except Exception:
        pass

    # Macro features
    log.info("Building macro features...")
    macro_feat = build_macro_features(data)

    # Build feature matrices for all sub-sectors
    log.info("Building per-subsector features...")
    all_features = {}
    subsector_names = list(returns_panel.columns)

    for name in subsector_names:
        feat = build_features_for_subsector(
            returns_panel[name], returns_panel, macro_feat, spy_close, name
        )
        all_features[name] = feat

    # Forward returns (target)
    fwd_returns = {}
    for name in subsector_names:
        cum = (1 + returns_panel[name]).cumprod()
        fwd = cum.shift(-PREDICT_HORIZON) / cum - 1
        fwd_returns[name] = fwd

    # Walk-forward loop
    n_days = len(returns_panel)
    n_folds = max(1, (n_days - TRAIN_WINDOW - PREDICT_HORIZON) // SLIDE_STEP)
    log.info(f"Walk-forward: {n_folds} folds, train={TRAIN_WINDOW}d, predict={PREDICT_HORIZON}d, slide={SLIDE_STEP}d")

    fold_results = []
    all_predictions = []
    feature_importance_accum = {}

    for fold_i in range(n_folds):
        train_start = fold_i * SLIDE_STEP
        train_end = train_start + TRAIN_WINDOW
        test_start = train_end
        test_end = min(test_start + PREDICT_HORIZON, n_days)

        if test_end > n_days - 1:
            break

        # Build train/test for all sub-sectors stacked
        train_X_parts = []
        train_y_parts = []
        test_X_parts = []
        test_y_parts = []
        test_names = []
        test_dates = []

        for name in subsector_names:
            feat = all_features[name]
            target = fwd_returns[name]

            # Align
            common = feat.index.intersection(target.dropna().index)
            if len(common) < TRAIN_WINDOW:
                continue

            feat_aligned = feat.loc[common]
            target_aligned = target.loc[common]

            # Get indices by position
            dates = feat_aligned.index
            if train_end > len(dates) or test_start >= len(dates):
                continue

            train_idx = dates[train_start:train_end]
            test_idx = dates[test_start:test_end]

            if len(train_idx) < TRAIN_WINDOW * 0.8 or len(test_idx) < 1:
                continue

            X_train = feat_aligned.loc[train_idx].copy()
            y_train = target_aligned.loc[train_idx].copy()
            X_test = feat_aligned.loc[test_idx].copy()
            y_test = target_aligned.loc[test_idx].copy()

            # Drop rows with NaN target
            valid_train = ~y_train.isna() & ~X_train.isna().any(axis=1)
            valid_test = ~y_test.isna() & ~X_test.isna().any(axis=1)

            if valid_train.sum() < 50 or valid_test.sum() < 1:
                continue

            train_X_parts.append(X_train[valid_train])
            train_y_parts.append(y_train[valid_train])
            test_X_parts.append(X_test[valid_test])
            test_y_parts.append(y_test[valid_test])
            test_names.extend([name] * valid_test.sum())
            test_dates.extend(list(X_test[valid_test].index))

        if not train_X_parts or not test_X_parts:
            continue

        # Stack
        X_train = pd.concat(train_X_parts, ignore_index=True)
        y_train = pd.concat(train_y_parts, ignore_index=True)
        X_test = pd.concat(test_X_parts, ignore_index=True)
        y_test = pd.concat(test_y_parts, ignore_index=True)

        # Fill remaining NaNs
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)

        # Ensure consistent columns
        common_cols = sorted(set(X_train.columns) & set(X_test.columns))
        if not common_cols:
            continue
        X_train = X_train[common_cols]
        X_test = X_test[common_cols]

        # Train LGBM
        try:
            dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
            model = lgb.train(
                LGBM_PARAMS,
                dtrain,
                num_boost_round=N_ROUNDS,
                valid_sets=[dtrain],
                callbacks=[lgb.log_evaluation(period=0)],
            )
            preds = model.predict(X_test)

            # Feature importance
            fi = dict(zip(common_cols, model.feature_importance(importance_type="gain")))
            for k, v in fi.items():
                feature_importance_accum[k] = feature_importance_accum.get(k, 0) + v

        except Exception as e:
            log.warning(f"  Fold {fold_i}: training error: {e}")
            continue

        # Evaluate: rank sub-sectors by predicted return, go long top-K
        # Build prediction table for this fold
        pred_df = pd.DataFrame({
            "subsector": test_names,
            "predicted": preds,
            "actual": y_test.values,
            "date": test_dates,
        })

        # Group by subsector (average prediction for the test window)
        avg_pred = pred_df.groupby("subsector").agg(
            pred_mean=("predicted", "mean"),
            actual_mean=("actual", "mean"),
        )

        if len(avg_pred) < TOPK:
            continue

        # Rank by predicted return
        avg_pred["rank"] = avg_pred["pred_mean"].rank(ascending=False)

        # Long top-K strategy return
        top_k = avg_pred[avg_pred["rank"] <= TOPK]
        long_return = top_k["actual_mean"].mean()

        # Bottom-K short
        bottom_k = avg_pred[avg_pred["rank"] > len(avg_pred) - BOTTOMK]
        short_return = -bottom_k["actual_mean"].mean()

        # Long-short return
        ls_return = (long_return + short_return) / 2

        # Benchmark: equal-weight all sub-sectors
        benchmark_return = avg_pred["actual_mean"].mean()

        # Hit rate: did we pick sub-sectors that actually outperformed median?
        median_actual = avg_pred["actual_mean"].median()
        hits = (top_k["actual_mean"] > median_actual).sum()
        hit_rate = hits / len(top_k) if len(top_k) > 0 else 0

        # IC: rank correlation between predicted and actual
        from scipy.stats import spearmanr
        ic, ic_pval = spearmanr(avg_pred["pred_mean"], avg_pred["actual_mean"])

        fold_result = {
            "fold": fold_i,
            "test_date": str(test_dates[0]) if test_dates else "unknown",
            "long_return_pct": round(long_return * 100, 2),
            "short_return_pct": round(short_return * 100, 2),
            "ls_return_pct": round(ls_return * 100, 2),
            "benchmark_return_pct": round(benchmark_return * 100, 2),
            "excess_return_pct": round((long_return - benchmark_return) * 100, 2),
            "hit_rate": round(hit_rate, 2),
            "ic": round(ic, 3) if not np.isnan(ic) else 0,
            "top_picks": list(top_k.index),
            "bottom_picks": list(bottom_k.index),
        }
        fold_results.append(fold_result)
        all_predictions.append(avg_pred)

        if fold_i % 5 == 0:
            log.info(
                f"  Fold {fold_i}/{n_folds}: long={long_return*100:+.1f}%, "
                f"LS={ls_return*100:+.1f}%, IC={ic:.3f}, HR={hit_rate:.0%}"
            )

    if not fold_results:
        log.error("No valid folds completed")
        return {"error": "No valid folds"}

    # --- Aggregate Results ---
    long_rets = [f["long_return_pct"] / 100 for f in fold_results]
    ls_rets = [f["ls_return_pct"] / 100 for f in fold_results]
    excess_rets = [f["excess_return_pct"] / 100 for f in fold_results]
    ics = [f["ic"] for f in fold_results]
    hit_rates = [f["hit_rate"] for f in fold_results]

    # Sharpe and Sortino (annualized, assuming monthly folds)
    periods_per_year = 252 / PREDICT_HORIZON

    def sharpe(rets):
        arr = np.array(rets)
        if arr.std() == 0:
            return 0
        return float(arr.mean() / arr.std() * np.sqrt(periods_per_year))

    def sortino(rets):
        arr = np.array(rets)
        downside = arr[arr < 0]
        if len(downside) == 0 or downside.std() == 0:
            return float("inf") if arr.mean() > 0 else 0
        return float(arr.mean() / downside.std() * np.sqrt(periods_per_year))

    def profit_factor(rets):
        arr = np.array(rets)
        gains = arr[arr > 0].sum()
        losses = abs(arr[arr < 0].sum())
        return round(gains / losses, 2) if losses > 0 else float("inf")

    # Regime stratification
    vix_levels = []
    try:
        if isinstance(data.columns, pd.MultiIndex):
            vix_close = data["Close"]["^VIX"]
        else:
            vix_close = None
    except Exception:
        vix_close = None

    # Classify each fold by VIX regime
    regime_folds = {"low_vol": [], "normal": [], "high_vol": []}
    for f in fold_results:
        # Simple: use fold index to approximate date
        regime_folds["normal"].append(f)  # Default to normal

    if vix_close is not None:
        regime_folds = {"low_vol": [], "normal": [], "high_vol": []}
        for f in fold_results:
            try:
                fold_date = pd.Timestamp(f["test_date"])
                nearest_idx = vix_close.index.get_indexer([fold_date], method="nearest")[0]
                vix_val = float(vix_close.iloc[nearest_idx])
                if vix_val < 15:
                    regime_folds["low_vol"].append(f)
                elif vix_val >= 25:
                    regime_folds["high_vol"].append(f)
                else:
                    regime_folds["normal"].append(f)
            except Exception:
                regime_folds["normal"].append(f)

    regime_stats = {}
    for regime_name, folds in regime_folds.items():
        if folds:
            r = [f["long_return_pct"] / 100 for f in folds]
            regime_stats[regime_name] = {
                "n_folds": len(folds),
                "mean_return_pct": round(np.mean(r) * 100, 2),
                "sharpe": round(sharpe(r), 2),
                "hit_rate": round(np.mean([f["hit_rate"] for f in folds]), 2),
            }

    # Feature importance
    fi_sorted = sorted(feature_importance_accum.items(), key=lambda x: x[1], reverse=True)
    top_features = fi_sorted[:15]

    results = {
        "generated_at": datetime.now().isoformat(),
        "model": "LGBM_subsector_rotation",
        "train_window": TRAIN_WINDOW,
        "predict_horizon": PREDICT_HORIZON,
        "slide_step": SLIDE_STEP,
        "n_folds": len(fold_results),
        "n_subsectors": len(subsector_names),
        "top_k_long": TOPK,
        "bottom_k_short": BOTTOMK,
        "long_only_results": {
            "mean_return_pct": round(np.mean(long_rets) * 100, 2),
            "sharpe": round(sharpe(long_rets), 2),
            "sortino": round(sortino(long_rets), 2),
            "profit_factor": profit_factor(long_rets),
            "win_rate": round(np.mean([1 if r > 0 else 0 for r in long_rets]), 2),
            "max_drawdown_pct": round(min(long_rets) * 100, 2),
            "best_fold_pct": round(max(long_rets) * 100, 2),
        },
        "long_short_results": {
            "mean_return_pct": round(np.mean(ls_rets) * 100, 2),
            "sharpe": round(sharpe(ls_rets), 2),
            "sortino": round(sortino(ls_rets), 2),
            "profit_factor": profit_factor(ls_rets),
            "win_rate": round(np.mean([1 if r > 0 else 0 for r in ls_rets]), 2),
        },
        "excess_vs_benchmark": {
            "mean_excess_pct": round(np.mean(excess_rets) * 100, 2),
            "sharpe_of_excess": round(sharpe(excess_rets), 2),
        },
        "information_coefficient": {
            "mean_ic": round(np.mean(ics), 3),
            "median_ic": round(np.median(ics), 3),
            "ic_std": round(np.std(ics), 3),
            "pct_positive_ic": round(np.mean([1 if ic > 0 else 0 for ic in ics]), 2),
        },
        "hit_rate": {
            "mean": round(np.mean(hit_rates), 2),
            "std": round(np.std(hit_rates), 2),
        },
        "regime_stratification": regime_stats,
        "top_features": [{"feature": f, "importance": round(v, 1)} for f, v in top_features],
        "fold_details": fold_results,
    }

    # Check regime-agnostic gate (HC #428)
    if len(regime_stats) >= 2:
        sharpes = [v["sharpe"] for v in regime_stats.values() if v["n_folds"] >= 2]
        if len(sharpes) >= 2:
            max_s = max(abs(s) for s in sharpes)
            if max_s > 0:
                regime_dispersion = (max(sharpes) - min(sharpes)) / max_s
                results["regime_agnostic_check"] = {
                    "sharpe_dispersion": round(regime_dispersion, 2),
                    "pass": regime_dispersion <= 0.50,
                    "note": "PASS: regime-agnostic" if regime_dispersion <= 0.50 else "FAIL: regime-dependent (|S_green - S_red|/max > 0.50)",
                }

    return results


# ---------------------------------------------------------------------------
# Live Predictions
# ---------------------------------------------------------------------------

def generate_predictions(data) -> dict:
    """Train on most recent data and generate current predictions."""
    if not HAS_LGBM:
        log.error("LightGBM not installed")
        return {"error": "LightGBM required"}

    log.info("Generating live sub-sector rotation predictions...")

    returns_panel = build_subsector_return_panel(data)
    subsector_names = list(returns_panel.columns)

    spy_close = None
    try:
        if isinstance(data.columns, pd.MultiIndex):
            spy_close = data["Close"]["SPY"]
    except Exception:
        pass

    macro_feat = build_macro_features(data)

    # Build features for all sub-sectors
    all_features = {}
    fwd_returns = {}
    for name in subsector_names:
        feat = build_features_for_subsector(
            returns_panel[name], returns_panel, macro_feat, spy_close, name
        )
        all_features[name] = feat

        cum = (1 + returns_panel[name]).cumprod()
        fwd = cum.shift(-PREDICT_HORIZON) / cum - 1
        fwd_returns[name] = fwd

    # Train on most recent TRAIN_WINDOW days
    train_X_parts = []
    train_y_parts = []
    current_X_parts = []
    current_names = []

    for name in subsector_names:
        feat = all_features[name]
        target = fwd_returns[name]

        common = feat.index.intersection(target.dropna().index)
        if len(common) < TRAIN_WINDOW:
            continue

        # Train on historical (excluding last PREDICT_HORIZON days where target is NaN)
        train_idx = common[-TRAIN_WINDOW:]
        X_train = feat.loc[train_idx].fillna(0)
        y_train = target.loc[train_idx]

        valid = ~y_train.isna() & ~X_train.isna().any(axis=1)
        if valid.sum() < 50:
            continue

        train_X_parts.append(X_train[valid])
        train_y_parts.append(y_train[valid])

        # Current features (last available day)
        last_feat = feat.iloc[-1:].fillna(0)
        current_X_parts.append(last_feat)
        current_names.append(name)

    if not train_X_parts or not current_X_parts:
        return {"error": "Insufficient data for predictions"}

    X_train = pd.concat(train_X_parts, ignore_index=True)
    y_train = pd.concat(train_y_parts, ignore_index=True)

    # Ensure consistent columns
    feature_cols = sorted(X_train.columns)
    X_train = X_train[feature_cols]

    # Train
    dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    model = lgb.train(
        LGBM_PARAMS, dtrain, num_boost_round=N_ROUNDS,
        valid_sets=[dtrain], callbacks=[lgb.log_evaluation(period=0)],
    )

    # Predict current
    predictions = {}
    for name, feat_row in zip(current_names, current_X_parts):
        X = feat_row[feature_cols].fillna(0) if all(c in feat_row.columns for c in feature_cols) else feat_row.reindex(columns=feature_cols, fill_value=0)
        pred = float(model.predict(X)[0])
        predictions[name] = pred

    # Rank
    pred_sorted = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    rankings = [
        {
            "rank": i + 1,
            "subsector": name,
            "predicted_return_pct": round(pred * 100, 2),
            "signal": "OVERWEIGHT" if i < TOPK else ("UNDERWEIGHT" if i >= len(pred_sorted) - BOTTOMK else "NEUTRAL"),
            "gics_sector": SUBSECTOR_UNIVERSE.get(name, {}).get("gics_sector", ""),
            "etf_proxy": SUBSECTOR_UNIVERSE.get(name, {}).get("etf", ""),
            "tickers": SUBSECTOR_UNIVERSE.get(name, {}).get("tickers", []),
        }
        for i, (name, pred) in enumerate(pred_sorted)
    ]

    # Feature importance
    fi = dict(zip(feature_cols, model.feature_importance(importance_type="gain")))
    fi_sorted = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:10]

    output = {
        "generated_at": datetime.now().isoformat(),
        "predict_horizon_days": PREDICT_HORIZON,
        "n_subsectors": len(rankings),
        "top_overweight": [r for r in rankings if r["signal"] == "OVERWEIGHT"],
        "bottom_underweight": [r for r in rankings if r["signal"] == "UNDERWEIGHT"],
        "full_rankings": rankings,
        "top_features": [{"feature": f, "importance": round(v, 1)} for f, v in fi_sorted],
    }

    # Save model
    try:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({"model": model, "features": feature_cols, "timestamp": datetime.now().isoformat()}, f)
        log.info(f"Model saved to {MODEL_PATH}")
    except Exception as e:
        log.warning(f"Could not save model: {e}")

    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Sub-Sector Rotation ML")
    parser.add_argument("--backtest", action="store_true", help="Run walk-forward backtest")
    parser.add_argument("--predict", action="store_true", help="Generate live predictions")
    args = parser.parse_args()

    # Default: run both
    run_bt = args.backtest or (not args.backtest and not args.predict)
    run_pred = args.predict or (not args.backtest and not args.predict)

    log.info("=" * 60)
    log.info(f"Sub-Sector Rotation ML — {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # Download data
    log.info("Downloading market data (2 years)...")
    if not HAS_YF:
        log.error("yfinance required")
        return

    data = yf.download(ALL_DOWNLOAD_TICKERS, period="2y", progress=False, threads=False)
    if data is None or data.empty:
        log.error("Download failed")
        return

    log.info(f"Downloaded {len(data)} days of data")

    if run_bt:
        log.info("\n--- BACKTEST ---")
        bt_results = run_backtest(data)

        with open(BACKTEST_PATH, "w") as f:
            json.dump(bt_results, f, indent=2, default=str)
        log.info(f"Backtest results saved to {BACKTEST_PATH}")

        if "error" not in bt_results:
            log.info(f"\n{'='*60}")
            log.info("BACKTEST RESULTS SUMMARY")
            log.info(f"{'='*60}")
            lo = bt_results["long_only_results"]
            ls = bt_results["long_short_results"]
            ic = bt_results["information_coefficient"]
            log.info(f"  Folds: {bt_results['n_folds']}, Sub-sectors: {bt_results['n_subsectors']}")
            log.info(f"  LONG-ONLY: Sharpe={lo['sharpe']}, Sortino={lo['sortino']}, "
                     f"WR={lo['win_rate']:.0%}, PF={lo['profit_factor']}, "
                     f"Avg ret={lo['mean_return_pct']:+.2f}%/period")
            log.info(f"  LONG-SHORT: Sharpe={ls['sharpe']}, Sortino={ls['sortino']}, "
                     f"WR={ls['win_rate']:.0%}, PF={ls['profit_factor']}")
            log.info(f"  IC: mean={ic['mean_ic']}, median={ic['median_ic']}, "
                     f"pct_positive={ic['pct_positive_ic']:.0%}")
            log.info(f"  Excess vs benchmark: {bt_results['excess_vs_benchmark']['mean_excess_pct']:+.2f}%/period")

            if "regime_stratification" in bt_results:
                log.info("  Regime stratification:")
                for reg, stats in bt_results["regime_stratification"].items():
                    log.info(f"    {reg}: n={stats['n_folds']}, Sharpe={stats['sharpe']}, "
                             f"HR={stats['hit_rate']:.0%}")

            if "regime_agnostic_check" in bt_results:
                rac = bt_results["regime_agnostic_check"]
                log.info(f"  Regime-agnostic check: {rac['note']} (dispersion={rac['sharpe_dispersion']})")
            log.info(f"{'='*60}\n")

    if run_pred:
        log.info("\n--- LIVE PREDICTIONS ---")
        pred_results = generate_predictions(data)

        with open(PREDICTIONS_PATH, "w") as f:
            json.dump(pred_results, f, indent=2, default=str)
        log.info(f"Predictions saved to {PREDICTIONS_PATH}")

        if "error" not in pred_results:
            log.info(f"\n{'='*60}")
            log.info(f"TOP OVERWEIGHT (next {PREDICT_HORIZON} trading days):")
            for r in pred_results["top_overweight"]:
                log.info(f"  #{r['rank']} {r['subsector']}: {r['predicted_return_pct']:+.1f}% "
                         f"({r['gics_sector']}) ETF={r['etf_proxy']}")
            log.info(f"\nBOTTOM UNDERWEIGHT:")
            for r in pred_results["bottom_underweight"]:
                log.info(f"  #{r['rank']} {r['subsector']}: {r['predicted_return_pct']:+.1f}% "
                         f"({r['gics_sector']}) ETF={r['etf_proxy']}")
            log.info(f"{'='*60}\n")


if __name__ == "__main__":
    main()
