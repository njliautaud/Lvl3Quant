#!/usr/bin/env python3
"""
ML Signal Weight Optimizer
==========================
Learns which signal COMBINATIONS predict winning vs losing trades
in the sector ETF options system.

Data sources:
  - Agentic trade log (closed trades with outcomes)
  - Paper engine trade logs (sector_combined_v* trades)
  - Active options closed trades
  - Paper state closed trades
  - Signal aggregator backtest results (for historical signal scoring)
  - Market data from yfinance (for forward returns / context features)

Models: LogReg, RandomForest, LightGBM
Validation: Sliding window walk-forward (HC #0)
"""

import json
import glob
import datetime as dt
import warnings
import sys
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    brier_score_loss, classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit

import lightgbm as lgb

warnings.filterwarnings("ignore")

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE = Path("/home/jupiter/Lvl3Quant")
STATE_DIR = BASE / "state"
DATA_DIR = BASE / "data"
LOGS_DIR = BASE / "paper_engines" / "logs"
OUTPUT_FILE = BASE / "output" / "ml_signal_weights_results.json"

SECTOR_ETFS = ["XLK", "XLC", "XLY", "XLE", "XLF", "XLV", "XLI", "XLP", "XLB", "XLRE", "XLU"]

# All known signal source names (canonical list)
ALL_SIGNAL_SOURCES = [
    "sector_spreads", "quality_momentum", "rsi_bullish", "rsi_bearish",
    "sector_etf_momentum", "cta_trend", "strong_momentum", "mom_accelerating",
    "mom_decelerating", "v91_monthly", "v93_combined", "v10_optimal",
    "subsector_rotation", "pead_drift", "vol_crush", "bond_yield_inflow",
    "equity_rotation_rank", "vix_elevated", "vix_calm", "weak_momentum",
    "regime_bullish", "volume_spike", "breadth_sector", "trend_50sma",
    "iv_runup", "contrarian_reversion", "earnings_jade_lizard",
    "extreme_idio", "liquidity_signal", "market_neutral_ls",
    "cross_asset_trend", "v93_profit_target", "lgbm_score_high",
    "dl_stock_ranker",
]


def load_agentic_trades():
    """Load trades from agentic trade log."""
    path = STATE_DIR / "agentic_trade_log.json"
    if not path.exists():
        return []
    d = json.load(open(path))
    trades = d.get("trades", [])
    # Only keep trades with outcomes
    result = []
    for t in trades:
        status = t.get("status", "")
        if status in ("WIN", "LOSS"):
            result.append({
                "ticker": t.get("ticker"),
                "direction": t.get("direction"),
                "entry_date": t.get("entry_date"),
                "exit_date": t.get("exit_date"),
                "confidence": t.get("confidence", 0),
                "setup_type": t.get("setup_type", ""),
                "source": t.get("source", ""),
                "status": status,
                "pnl_pct": t.get("exit_pnl", 0),
                "origin": "agentic",
            })
    return result


def load_active_options_closed():
    """Load closed trades from active_options.json."""
    path = DATA_DIR / "active_options.json"
    if not path.exists():
        return []
    d = json.load(open(path))
    closed = d.get("closed", [])
    result = []
    for t in closed:
        pnl_pct = t.get("pnl_pct", 0)
        result.append({
            "ticker": t.get("ticker"),
            "direction": "call" if t.get("type") == "call" else "put",
            "entry_date": t.get("entry_date"),
            "exit_date": t.get("exit_date"),
            "confidence": 0,
            "setup_type": "",
            "source": "active_options",
            "status": "WIN" if pnl_pct > 0 else "LOSS",
            "pnl_pct": pnl_pct,
            "origin": "active_options",
        })
    return result


def load_paper_state_closed():
    """Load closed trades from all paper state files."""
    result = []
    for f in sorted(glob.glob(str(STATE_DIR / "*_paper_state.json"))):
        try:
            d = json.load(open(f))
            closed = d.get("closed_trades", [])
            engine_name = Path(f).stem.replace("_paper_state", "")
            for t in closed:
                pnl = t.get("pnl", t.get("pnl_pct", 0))
                entry_date = t.get("entry_date", "")
                if isinstance(entry_date, str) and "T" in entry_date:
                    entry_date = entry_date.split("T")[0]
                exit_date = t.get("exit_date", "")
                if isinstance(exit_date, str) and "T" in exit_date:
                    exit_date = exit_date.split("T")[0]
                ticker = t.get("ticker", t.get("sector", ""))
                result.append({
                    "ticker": ticker,
                    "direction": t.get("direction", "call"),
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "confidence": 0,
                    "setup_type": engine_name,
                    "source": engine_name,
                    "status": "WIN" if pnl > 0 else "LOSS",
                    "pnl_pct": pnl,
                    "origin": "paper_state",
                })
        except Exception:
            continue
    return result


def load_paper_engine_trades():
    """Load trades from paper engine JSONL logs."""
    result = []
    for f in sorted(glob.glob(str(LOGS_DIR / "*_trades.jsonl"))):
        engine_name = Path(f).stem.replace("_trades", "")
        try:
            opens = {}
            with open(f) as fh:
                for line in fh:
                    try:
                        t = json.loads(line.strip())
                    except json.JSONDecodeError:
                        continue
                    action = t.get("action", "")
                    ticker = t.get("ticker", "")
                    if action == "OPEN":
                        opens[ticker] = t
                    elif action == "CLOSE" and ticker in opens:
                        o = opens.pop(ticker)
                        pnl = t.get("pnl", 0)
                        entry_date = o.get("date", "")
                        exit_date = t.get("date", "")
                        mode = o.get("mode", "bull")
                        lgbm_score = o.get("lgbm_score", 0)
                        vix = o.get("vix", 0)
                        result.append({
                            "ticker": ticker,
                            "direction": "call" if mode == "bull" else "put",
                            "entry_date": entry_date,
                            "exit_date": exit_date,
                            "confidence": lgbm_score,
                            "setup_type": engine_name,
                            "source": engine_name,
                            "status": "WIN" if pnl > 0 else "LOSS",
                            "pnl_pct": pnl,
                            "vix_at_entry": vix,
                            "origin": "paper_engine",
                        })
        except Exception:
            continue
    return result


def load_signal_calibration():
    """Load signal performance data from calibration report."""
    path = STATE_DIR / "signal_calibration_report.json"
    if not path.exists():
        return {}
    d = json.load(open(path))
    return d.get("signal_performance", {})


def load_agentic_signals():
    """Load current signal snapshot for source list."""
    path = STATE_DIR / "agentic_signals.json"
    if not path.exists():
        return {}
    return json.load(open(path))


def get_market_data(tickers, start_date, end_date):
    """Fetch market data for context features."""
    try:
        all_tickers = list(set(tickers + ["SPY", "^VIX"]))
        data = yf.download(
            all_tickers,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
        )
        if isinstance(data.columns, pd.MultiIndex):
            close = data["Close"].copy()
            volume = data["Volume"].copy()
        else:
            close = data
            volume = pd.DataFrame()
        if isinstance(close.columns, pd.MultiIndex):
            close.columns = close.columns.get_level_values(-1)
        if isinstance(volume.columns, pd.MultiIndex):
            volume.columns = volume.columns.get_level_values(-1)
        return close, volume
    except Exception as e:
        print(f"Warning: market data fetch failed: {e}")
        return pd.DataFrame(), pd.DataFrame()


def build_feature_matrix(all_trades, close_df, volume_df):
    """
    Build feature matrix from trade data + market context.

    For each trade, create features:
    - Binary: which signal sources were active (inferred from setup/source)
    - Continuous: confidence, VIX level, market context
    - Target: WIN=1, LOSS=0
    """
    rows = []

    # Map setup types and sources to signal indicators
    source_to_signals = {
        "sector_combined_v7": ["sector_spreads", "quality_momentum", "v93_combined"],
        "sector_combined_v8": ["sector_spreads", "quality_momentum", "v93_combined", "mom_accelerating"],
        "sector_combined_v9": ["sector_spreads", "v93_combined", "sector_etf_momentum", "cta_trend"],
        "sector_combined_v91": ["sector_spreads", "v91_monthly", "v93_combined"],
        "sector_combined_v92": ["sector_spreads", "v93_combined", "lgbm_score_high"],
        "sector_combined_v93": ["sector_spreads", "v93_combined", "lgbm_score_high", "v93_profit_target"],
        "sector_combined_v10": ["v10_optimal", "sector_spreads", "v93_combined"],
        "sector_combined_v10_optimal": ["v10_optimal", "sector_spreads", "v93_combined", "lgbm_score_high"],
        "sector_spreads": ["sector_spreads"],
        "momentum_options": ["strong_momentum", "sector_etf_momentum", "mom_accelerating"],
        "sector_earnings_standalone": ["pead_drift", "vol_crush"],
        "sector_pairs": ["sector_spreads", "subsector_rotation"],
        "pead_ml": ["pead_drift"],
        "iv_runup": ["iv_runup", "vol_crush"],
        "market_neutral_ls": ["market_neutral_ls", "quality_momentum"],
        "contrarian_reversion": ["contrarian_reversion", "rsi_bearish"],
        "earnings_jade_lizard": ["vol_crush", "pead_drift"],
        "extreme_idio": ["extreme_idio"],
        "liquidity_signal": ["liquidity_signal"],
        "cross_asset_trend": ["cross_asset_trend", "cta_trend"],
        "bond_yield": ["bond_yield_inflow"],
        "A_momentum_continuation": ["strong_momentum", "sector_etf_momentum", "cta_trend", "mom_accelerating"],
        "B_oversold_bounce": ["rsi_bearish", "contrarian_reversion"],
        "C_flow_divergence": ["liquidity_signal", "sector_spreads"],
        "D_post_earnings_bounce": ["pead_drift", "vol_crush"],
    }

    for t in all_trades:
        entry_date = t.get("entry_date", "")
        if not entry_date:
            continue

        try:
            edate = pd.Timestamp(entry_date)
        except Exception:
            continue

        ticker = t.get("ticker", "")
        if not ticker:
            continue

        # Build signal binary features
        setup = t.get("setup_type", "")
        source = t.get("source", "")
        active_signals = set()

        # Map from setup type
        if setup in source_to_signals:
            active_signals.update(source_to_signals[setup])
        # Map from source
        if source in source_to_signals:
            active_signals.update(source_to_signals[source])

        # Direction features
        direction = t.get("direction", "call")
        is_bullish = 1 if direction in ("call", "bull", "UP") else 0

        row = {
            "ticker": ticker,
            "entry_date": entry_date,
            "direction": direction,
            "is_bullish": is_bullish,
            "confidence": t.get("confidence", 0),
            "pnl_pct": t.get("pnl_pct", 0),
            "target": 1 if t.get("status") == "WIN" else 0,
            "origin": t.get("origin", ""),
            "setup_type": setup,
        }

        # Binary signal features
        for sig in ALL_SIGNAL_SOURCES:
            row[f"sig_{sig}"] = 1 if sig in active_signals else 0

        # Count of active signals
        row["n_active_signals"] = len(active_signals)

        # Market context features (if data available)
        if not close_df.empty and edate in close_df.index:
            idx = close_df.index.get_loc(edate)

            # SPY context
            if "SPY" in close_df.columns and idx >= 20:
                spy = close_df["SPY"]
                row["spy_5d_return"] = (spy.iloc[idx] / spy.iloc[max(0, idx-5)] - 1) * 100
                row["spy_20d_return"] = (spy.iloc[idx] / spy.iloc[max(0, idx-20)] - 1) * 100
                sma200_val = spy.iloc[max(0, idx-199):idx+1].mean()
                row["spy_above_sma200"] = 1 if spy.iloc[idx] > sma200_val else 0

            # VIX context
            vix_col = "^VIX" if "^VIX" in close_df.columns else ("VIX" if "VIX" in close_df.columns else None)
            if vix_col and vix_col in close_df.columns:
                vix_val = close_df[vix_col].iloc[idx]
                if not pd.isna(vix_val):
                    row["vix_level"] = vix_val
                    row["vix_below_20"] = 1 if vix_val < 20 else 0
                    row["vix_above_25"] = 1 if vix_val > 25 else 0

            # Ticker-specific context
            if ticker in close_df.columns and idx >= 20:
                tk = close_df[ticker]
                row["ticker_5d_return"] = (tk.iloc[idx] / tk.iloc[max(0, idx-5)] - 1) * 100
                row["ticker_20d_return"] = (tk.iloc[idx] / tk.iloc[max(0, idx-20)] - 1) * 100

                # Relative strength vs SPY
                if "SPY" in close_df.columns:
                    spy_5d = (close_df["SPY"].iloc[idx] / close_df["SPY"].iloc[max(0, idx-5)] - 1)
                    tk_5d = (tk.iloc[idx] / tk.iloc[max(0, idx-5)] - 1)
                    row["rel_strength_5d"] = (tk_5d - spy_5d) * 100
        elif t.get("vix_at_entry"):
            row["vix_level"] = t["vix_at_entry"]
            row["vix_below_20"] = 1 if t["vix_at_entry"] < 20 else 0
            row["vix_above_25"] = 1 if t["vix_at_entry"] > 25 else 0

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df = df.sort_values("entry_date").reset_index(drop=True)
    return df


def analyze_signal_cooccurrence(df):
    """Analyze which signal combinations co-occur with wins vs losses."""
    sig_cols = [c for c in df.columns if c.startswith("sig_")]
    active_sig_cols = [c for c in sig_cols if df[c].sum() > 0]

    results = {}

    # Individual signal performance
    individual = {}
    for col in active_sig_cols:
        sig_name = col.replace("sig_", "")
        mask = df[col] == 1
        n = mask.sum()
        if n < 2:
            continue
        wr = df.loc[mask, "target"].mean()
        avg_pnl = df.loc[mask, "pnl_pct"].mean()
        individual[sig_name] = {
            "n_trades": int(n),
            "win_rate": round(float(wr), 3),
            "avg_pnl_pct": round(float(avg_pnl), 2),
        }
    results["individual_signals"] = dict(sorted(individual.items(), key=lambda x: -x[1]["win_rate"]))

    # Pairwise co-occurrence analysis
    pairs = {}
    for i, c1 in enumerate(active_sig_cols):
        for c2 in active_sig_cols[i+1:]:
            mask = (df[c1] == 1) & (df[c2] == 1)
            n = mask.sum()
            if n < 2:
                continue
            wr = df.loc[mask, "target"].mean()
            s1 = c1.replace("sig_", "")
            s2 = c2.replace("sig_", "")

            # Compare to individual rates
            wr1 = df.loc[df[c1] == 1, "target"].mean() if df[c1].sum() > 0 else 0
            wr2 = df.loc[df[c2] == 1, "target"].mean() if df[c2].sum() > 0 else 0
            synergy = wr - max(wr1, wr2)

            pairs[f"{s1}+{s2}"] = {
                "n_trades": int(n),
                "combined_wr": round(float(wr), 3),
                "individual_wr_1": round(float(wr1), 3),
                "individual_wr_2": round(float(wr2), 3),
                "synergy": round(float(synergy), 3),
            }

    # Sort by synergy
    golden = {k: v for k, v in sorted(pairs.items(), key=lambda x: -x[1]["synergy"]) if v["synergy"] > 0.05}
    toxic = {k: v for k, v in sorted(pairs.items(), key=lambda x: x[1]["synergy"]) if v["synergy"] < -0.05}

    results["golden_combinations"] = golden
    results["toxic_combinations"] = toxic
    results["all_pairs"] = pairs

    # Optimal number of confirming sources
    n_sig_perf = {}
    for n in sorted(df["n_active_signals"].unique()):
        mask = df["n_active_signals"] == n
        count = mask.sum()
        if count < 2:
            continue
        wr = df.loc[mask, "target"].mean()
        avg_pnl = df.loc[mask, "pnl_pct"].mean()
        n_sig_perf[int(n)] = {
            "n_trades": int(count),
            "win_rate": round(float(wr), 3),
            "avg_pnl_pct": round(float(avg_pnl), 2),
        }
    results["optimal_n_sources"] = n_sig_perf

    return results


def compute_mutual_information(df):
    """Identify redundant signals via mutual information."""
    sig_cols = [c for c in df.columns if c.startswith("sig_") and df[c].sum() > 0]

    redundancy = {}
    for i, c1 in enumerate(sig_cols):
        for c2 in sig_cols[i+1:]:
            # Simple correlation for binary variables (phi coefficient)
            if df[c1].std() == 0 or df[c2].std() == 0:
                continue
            corr = df[c1].corr(df[c2])
            if abs(corr) > 0.5:
                s1 = c1.replace("sig_", "")
                s2 = c2.replace("sig_", "")
                redundancy[f"{s1}+{s2}"] = {
                    "correlation": round(float(corr), 3),
                    "assessment": "highly_redundant" if abs(corr) > 0.8 else "moderately_redundant",
                }

    # Check incremental value of each signal
    incremental = {}
    for col in sig_cols:
        sig_name = col.replace("sig_", "")
        # Correlation with target
        if df[col].std() > 0 and df["target"].std() > 0:
            corr_target = df[col].corr(df["target"])

            # Partial correlation: controlling for other signals
            other_sigs = [c for c in sig_cols if c != col and df[c].std() > 0]
            if other_sigs and df[col].sum() >= 3:
                from sklearn.linear_model import LinearRegression
                X_other = df[other_sigs].values
                lr = LinearRegression()
                lr.fit(X_other, df[col].values)
                resid_sig = df[col].values - lr.predict(X_other)
                lr.fit(X_other, df["target"].values)
                resid_target = df["target"].values - lr.predict(X_other)
                if np.std(resid_sig) > 0 and np.std(resid_target) > 0:
                    partial_corr = np.corrcoef(resid_sig, resid_target)[0, 1]
                else:
                    partial_corr = 0.0
            else:
                partial_corr = corr_target

            incremental[sig_name] = {
                "raw_corr_with_target": round(float(corr_target), 3),
                "partial_corr_with_target": round(float(partial_corr), 3),
                "incremental_value": "high" if abs(partial_corr) > 0.15 else ("moderate" if abs(partial_corr) > 0.05 else "low"),
                "n_fires": int(df[col].sum()),
            }

    return {
        "redundant_pairs": redundancy,
        "incremental_value": dict(sorted(incremental.items(), key=lambda x: -abs(x[1]["partial_corr_with_target"]))),
    }


def train_models_sliding(df, feature_cols):
    """
    Train models using SLIDING window walk-forward validation.
    HC #0: NEVER expanding window.
    """
    n = len(df)
    if n < 15:
        print(f"  WARNING: Only {n} samples — too few for reliable ML. Using statistical fallback.")
        return None

    # Sliding window: use 80% for training window size, slide 1 step
    train_window = max(10, int(n * 0.7))

    X = df[feature_cols].fillna(0).values
    y = df["target"].values

    scaler = StandardScaler()

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, C=0.1, penalty="l2", random_state=42,
            class_weight="balanced"
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100, max_depth=3, min_samples_leaf=3,
            random_state=42, class_weight="balanced"
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=50, max_depth=2, min_samples_leaf=3,
            learning_rate=0.05, random_state=42
        ),
        "lightgbm": lgb.LGBMClassifier(
            n_estimators=50, max_depth=2, min_child_samples=3,
            learning_rate=0.05, random_state=42, is_unbalance=True,
            verbose=-1, num_threads=1
        ),
    }

    results = {}
    all_preds = {name: {"y_true": [], "y_pred": [], "y_prob": []} for name in models}

    n_folds = 0
    for start in range(0, n - train_window):
        end_train = start + train_window
        if end_train >= n:
            break

        # Test on next sample(s) — use 1-3 depending on data
        end_test = min(end_train + max(1, (n - train_window) // 5), n)
        if end_test <= end_train:
            break

        X_train = X[start:end_train]
        y_train = y[start:end_train]
        X_test = X[end_train:end_test]
        y_test = y[end_train:end_test]

        # Skip if only one class in training
        if len(set(y_train)) < 2:
            continue

        # Scale
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)

        for name, model in models.items():
            try:
                if name == "lightgbm":
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)
                    prob = model.predict_proba(X_test)[:, 1]
                elif name in ("random_forest", "gradient_boosting"):
                    model.fit(X_train_sc, y_train)
                    pred = model.predict(X_test_sc)
                    prob = model.predict_proba(X_test_sc)[:, 1]
                else:
                    model.fit(X_train_sc, y_train)
                    pred = model.predict(X_test_sc)
                    prob = model.predict_proba(X_test_sc)[:, 1]

                all_preds[name]["y_true"].extend(y_test.tolist())
                all_preds[name]["y_pred"].extend(pred.tolist())
                all_preds[name]["y_prob"].extend(prob.tolist())
            except Exception as e:
                pass

        n_folds += 1
        # Move forward by test window size to avoid overlap
        # (the for loop handles this by incrementing start by 1,
        # but we only use non-overlapping test windows)

    print(f"  Walk-forward folds: {n_folds}")

    # Compute aggregate metrics
    for name in models:
        preds = all_preds[name]
        if len(preds["y_true"]) < 5:
            results[name] = {"error": "insufficient_predictions"}
            continue

        yt = np.array(preds["y_true"])
        yp = np.array(preds["y_pred"])
        yprob = np.array(preds["y_prob"])

        metrics = {
            "accuracy": round(float(accuracy_score(yt, yp)), 3),
            "precision": round(float(precision_score(yt, yp, zero_division=0)), 3),
            "recall": round(float(recall_score(yt, yp, zero_division=0)), 3),
            "brier_score": round(float(brier_score_loss(yt, yprob)), 4),
            "n_predictions": len(yt),
            "baseline_rate": round(float(yt.mean()), 3),
        }

        # AUC requires both classes
        if len(set(yt)) >= 2:
            metrics["auc_roc"] = round(float(roc_auc_score(yt, yprob)), 3)
        else:
            metrics["auc_roc"] = None

        results[name] = metrics

    # Train final model on all data for feature importance
    feature_importance = {}
    for name, model in models.items():
        try:
            X_sc = scaler.fit_transform(X)
            if name == "lightgbm":
                model.fit(X, y)
                importances = model.feature_importances_
            else:
                model.fit(X_sc, y)
                if hasattr(model, "feature_importances_"):
                    importances = model.feature_importances_
                elif hasattr(model, "coef_"):
                    importances = np.abs(model.coef_[0])
                else:
                    continue

            # Permutation importance (more reliable, avoids Gini artifacts)
            if name == "lightgbm":
                perm_imp = permutation_importance(model, X, y, n_repeats=10, random_state=42)
            else:
                perm_imp = permutation_importance(model, X_sc, y, n_repeats=10, random_state=42)

            imp_dict = {}
            for i, col in enumerate(feature_cols):
                imp_dict[col] = {
                    "model_importance": round(float(importances[i]), 4),
                    "perm_importance_mean": round(float(perm_imp.importances_mean[i]), 4),
                    "perm_importance_std": round(float(perm_imp.importances_std[i]), 4),
                }

            # Sort by permutation importance
            feature_importance[name] = dict(
                sorted(imp_dict.items(), key=lambda x: -x[1]["perm_importance_mean"])
            )
        except Exception as e:
            feature_importance[name] = {"error": str(e)}

    return {
        "model_metrics": results,
        "feature_importance": feature_importance,
        "n_folds": n_folds,
        "n_samples": n,
        "class_balance": {"wins": int(y.sum()), "losses": int((1-y).sum())},
    }


def compute_optimal_weights(df, feature_cols):
    """Compute recommended signal weights from best model."""
    sig_cols = [c for c in feature_cols if c.startswith("sig_") and df[c].sum() > 0]

    if len(df) < 10:
        return {"error": "insufficient_data"}

    X = df[sig_cols].fillna(0).values
    y = df["target"].values

    if len(set(y)) < 2:
        return {"error": "single_class"}

    # Use L1-regularized logistic regression for sparse weights
    lr = LogisticRegression(max_iter=1000, C=1.0, penalty="l1", solver="liblinear",
                            random_state=42, class_weight="balanced")
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    lr.fit(X_sc, y)

    coefs = lr.coef_[0]

    weights = {}
    for i, col in enumerate(sig_cols):
        sig_name = col.replace("sig_", "")
        w = float(coefs[i])
        weights[sig_name] = {
            "raw_weight": round(w, 4),
            "direction": "positive" if w > 0 else ("negative" if w < 0 else "zero"),
            "magnitude": round(abs(w), 4),
        }

    # Normalize to recommended weights (0-2 scale)
    max_abs = max(abs(w["raw_weight"]) for w in weights.values()) if weights else 1
    if max_abs > 0:
        for sig in weights:
            norm_w = weights[sig]["raw_weight"] / max_abs
            weights[sig]["normalized_weight"] = round(1.0 + norm_w, 3)  # 0 to 2 scale

    return dict(sorted(weights.items(), key=lambda x: -x[1]["magnitude"]))


def statistical_fallback(df):
    """When sample too small for ML, do pure statistical analysis."""
    print("\n  Running statistical fallback analysis (small sample)...")

    sig_cols = [c for c in df.columns if c.startswith("sig_") and df[c].sum() > 0]

    results = {
        "method": "statistical_fallback",
        "reason": f"Only {len(df)} samples — too few for reliable ML",
    }

    # Fisher exact test for each signal vs outcome
    signal_tests = {}
    for col in sig_cols:
        sig_name = col.replace("sig_", "")
        # 2x2 contingency table
        a = ((df[col] == 1) & (df["target"] == 1)).sum()  # sig+win
        b = ((df[col] == 1) & (df["target"] == 0)).sum()  # sig+loss
        c = ((df[col] == 0) & (df["target"] == 1)).sum()  # no_sig+win
        d = ((df[col] == 0) & (df["target"] == 0)).sum()  # no_sig+loss

        if a + b < 2:
            continue

        try:
            odds_ratio, p_value = stats.fisher_exact([[a, b], [c, d]])
        except Exception:
            odds_ratio, p_value = 1.0, 1.0

        wr_with = a / (a + b) if (a + b) > 0 else 0
        wr_without = c / (c + d) if (c + d) > 0 else 0

        signal_tests[sig_name] = {
            "n_with_signal": int(a + b),
            "wr_with_signal": round(float(wr_with), 3),
            "wr_without_signal": round(float(wr_without), 3),
            "odds_ratio": round(float(odds_ratio), 3) if not np.isinf(odds_ratio) else "inf",
            "p_value": round(float(p_value), 4),
            "significant": p_value < 0.1,  # relaxed threshold for small samples
        }

    results["signal_tests"] = dict(sorted(signal_tests.items(), key=lambda x: x[1]["p_value"]))

    # Confidence vs outcome
    if "confidence" in df.columns and df["confidence"].std() > 0:
        corr, p = stats.pointbiserialr(df["target"], df["confidence"])
        results["confidence_correlation"] = {
            "correlation": round(float(corr), 3),
            "p_value": round(float(p), 4),
        }

    # Direction analysis
    if "is_bullish" in df.columns:
        bull_mask = df["is_bullish"] == 1
        bear_mask = df["is_bullish"] == 0
        if bull_mask.sum() >= 2 and bear_mask.sum() >= 2:
            results["direction_analysis"] = {
                "bullish_wr": round(float(df.loc[bull_mask, "target"].mean()), 3),
                "bearish_wr": round(float(df.loc[bear_mask, "target"].mean()), 3),
                "bullish_n": int(bull_mask.sum()),
                "bearish_n": int(bear_mask.sum()),
            }

    return results


def compare_ml_vs_fixed_rule(df, ml_results):
    """Compare ML-selected trades vs current fixed rule (5+ sources, 78%+ conf)."""
    # Current fixed rule
    fixed_mask = (df["n_active_signals"] >= 5) & (df["confidence"] >= 0.78)
    n_fixed = fixed_mask.sum()

    comparison = {
        "fixed_rule": {
            "rule": "n_sources >= 5 AND confidence >= 0.78",
            "n_trades": int(n_fixed),
        },
        "all_trades": {
            "n_trades": len(df),
            "win_rate": round(float(df["target"].mean()), 3),
            "avg_pnl": round(float(df["pnl_pct"].mean()), 2),
        }
    }

    if n_fixed >= 2:
        comparison["fixed_rule"]["win_rate"] = round(float(df.loc[fixed_mask, "target"].mean()), 3)
        comparison["fixed_rule"]["avg_pnl"] = round(float(df.loc[fixed_mask, "pnl_pct"].mean()), 2)
    else:
        comparison["fixed_rule"]["note"] = "Too few trades match fixed rule to evaluate"

    # Alternative thresholds
    for n_thresh in [2, 3, 4, 5]:
        mask = df["n_active_signals"] >= n_thresh
        n = mask.sum()
        if n >= 2:
            comparison[f"threshold_{n_thresh}_sources"] = {
                "n_trades": int(n),
                "win_rate": round(float(df.loc[mask, "target"].mean()), 3),
                "avg_pnl": round(float(df.loc[mask, "pnl_pct"].mean()), 2),
            }

    return comparison


def main():
    print("=" * 70)
    print("ML SIGNAL WEIGHT OPTIMIZER")
    print("=" * 70)

    # ─── 1. Gather Training Data ─────────────────────────────────────────
    print("\n[1] Loading trade data from all sources...")

    agentic = load_agentic_trades()
    print(f"  Agentic trade log: {len(agentic)} closed trades")

    active_opts = load_active_options_closed()
    print(f"  Active options closed: {len(active_opts)} trades")

    paper_states = load_paper_state_closed()
    print(f"  Paper state closed: {len(paper_states)} trades")

    paper_engine = load_paper_engine_trades()
    print(f"  Paper engine logs: {len(paper_engine)} trades")

    all_trades = agentic + active_opts + paper_states + paper_engine
    print(f"  TOTAL TRADES: {len(all_trades)}")

    if not all_trades:
        print("\nERROR: No trades found. Cannot train ML models.")
        return

    # ─── 2. Fetch Market Data for Context ─────────────────────────────────
    print("\n[2] Fetching market data for context features...")

    all_tickers = list(set(
        [t["ticker"] for t in all_trades if t.get("ticker")]
        + SECTOR_ETFS + ["SPY", "QQQ"]
    ))

    # Get date range
    dates = [t["entry_date"] for t in all_trades if t.get("entry_date")]
    if dates:
        min_date = min(dates)
        max_date = max(dates)
        # Add buffer for lookback
        start_date = (pd.Timestamp(min_date) - pd.Timedelta(days=250)).strftime("%Y-%m-%d")
        end_date = (pd.Timestamp(max_date) + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
        print(f"  Date range: {min_date} to {max_date}")
    else:
        start_date = "2025-01-01"
        end_date = dt.datetime.now().strftime("%Y-%m-%d")

    close_df, volume_df = get_market_data(all_tickers, start_date, end_date)
    if not close_df.empty:
        print(f"  Market data: {len(close_df)} days, {len(close_df.columns)} tickers")
    else:
        print("  WARNING: No market data available, proceeding without context features")

    # ─── 3. Build Feature Matrix ──────────────────────────────────────────
    print("\n[3] Building feature matrix...")

    df = build_feature_matrix(all_trades, close_df, volume_df)
    print(f"  Feature matrix: {len(df)} rows x {len(df.columns)} columns")
    print(f"  Win rate: {df['target'].mean():.1%}")
    print(f"  Wins: {df['target'].sum()}, Losses: {(1-df['target']).sum()}")

    if len(df) < 5:
        print("\nERROR: Too few trades to analyze.")
        return

    # Print data summary
    print(f"\n  Trades by origin:")
    for origin, count in df["origin"].value_counts().items():
        wr = df.loc[df["origin"] == origin, "target"].mean()
        print(f"    {origin}: {count} trades, {wr:.0%} WR")

    print(f"\n  Trades by ticker:")
    for ticker, count in df["ticker"].value_counts().head(10).items():
        wr = df.loc[df["ticker"] == ticker, "target"].mean()
        print(f"    {ticker}: {count} trades, {wr:.0%} WR")

    # ─── 4. Signal Co-occurrence Analysis ─────────────────────────────────
    print("\n[4] Analyzing signal co-occurrence patterns...")

    cooccurrence = analyze_signal_cooccurrence(df)

    print(f"\n  Individual signal performance:")
    for sig, perf in list(cooccurrence["individual_signals"].items())[:10]:
        print(f"    {sig}: {perf['n_trades']} trades, {perf['win_rate']:.0%} WR, avg PnL {perf['avg_pnl_pct']:.1f}%")

    if cooccurrence.get("golden_combinations"):
        print(f"\n  GOLDEN combinations (synergy > 5%):")
        for pair, data in list(cooccurrence["golden_combinations"].items())[:5]:
            print(f"    {pair}: {data['n_trades']}t, {data['combined_wr']:.0%} WR, synergy +{data['synergy']:.0%}")

    if cooccurrence.get("toxic_combinations"):
        print(f"\n  TOXIC combinations (synergy < -5%):")
        for pair, data in list(cooccurrence["toxic_combinations"].items())[:5]:
            print(f"    {pair}: {data['n_trades']}t, {data['combined_wr']:.0%} WR, synergy {data['synergy']:.0%}")

    if cooccurrence.get("optimal_n_sources"):
        print(f"\n  Optimal number of confirming sources:")
        for n, perf in cooccurrence["optimal_n_sources"].items():
            print(f"    {n} sources: {perf['n_trades']} trades, {perf['win_rate']:.0%} WR, avg PnL {perf['avg_pnl_pct']:.1f}%")

    # ─── 5. Mutual Information / Redundancy ───────────────────────────────
    print("\n[5] Checking signal redundancy...")

    mi_results = compute_mutual_information(df)

    if mi_results.get("redundant_pairs"):
        print(f"  Redundant signal pairs:")
        for pair, data in mi_results["redundant_pairs"].items():
            print(f"    {pair}: r={data['correlation']:.2f} ({data['assessment']})")
    else:
        print("  No highly redundant signal pairs found")

    print(f"\n  Signal incremental value (sorted by partial correlation with target):")
    for sig, data in list(mi_results["incremental_value"].items())[:10]:
        print(f"    {sig}: partial_r={data['partial_corr_with_target']:.3f}, value={data['incremental_value']}, n={data['n_fires']}")

    # ─── 6. Feature columns for ML ────────────────────────────────────────
    sig_feature_cols = [c for c in df.columns if c.startswith("sig_") and df[c].sum() > 0]
    context_cols = [c for c in df.columns if c in [
        "confidence", "is_bullish", "n_active_signals",
        "spy_5d_return", "spy_20d_return", "spy_above_sma200",
        "vix_level", "vix_below_20", "vix_above_25",
        "ticker_5d_return", "ticker_20d_return", "rel_strength_5d",
    ] and c in df.columns and df[c].notna().sum() > len(df) * 0.3]

    feature_cols = sig_feature_cols + context_cols
    print(f"\n[6] Feature set: {len(sig_feature_cols)} signal features + {len(context_cols)} context features = {len(feature_cols)} total")

    # ─── 7. Train Models ─────────────────────────────────────────────────
    print("\n[7] Training models (sliding window walk-forward)...")

    ml_results = train_models_sliding(df, feature_cols)

    if ml_results:
        print(f"\n  Model comparison (walk-forward OOT):")
        print(f"  {'Model':<25} {'AUC':>6} {'Acc':>6} {'Prec':>6} {'Recall':>6} {'Brier':>7}")
        print(f"  {'-'*60}")
        for name, metrics in ml_results["model_metrics"].items():
            if "error" in metrics:
                print(f"  {name:<25} {metrics['error']}")
                continue
            auc = metrics.get("auc_roc")
            auc_str = f"{auc:.3f}" if auc is not None else "N/A"
            print(f"  {name:<25} {auc_str:>6} {metrics['accuracy']:.3f} {metrics['precision']:.3f} {metrics['recall']:.3f} {metrics['brier_score']:.4f}")

        # Feature importance from best model
        best_model = None
        best_auc = 0
        for name, metrics in ml_results["model_metrics"].items():
            if "error" not in metrics and metrics.get("auc_roc") is not None:
                if metrics["auc_roc"] > best_auc:
                    best_auc = metrics["auc_roc"]
                    best_model = name

        if best_model and best_model in ml_results["feature_importance"]:
            print(f"\n  Top features ({best_model}, permutation importance):")
            fi = ml_results["feature_importance"][best_model]
            if "error" not in fi:
                for feat, imp in list(fi.items())[:15]:
                    if imp["perm_importance_mean"] > 0.001:
                        print(f"    {feat:<35} perm_imp={imp['perm_importance_mean']:.4f} ± {imp['perm_importance_std']:.4f}")
    else:
        print("  ML training skipped (insufficient data)")

    # ─── 8. Statistical Fallback ──────────────────────────────────────────
    stat_results = statistical_fallback(df)

    if stat_results.get("signal_tests"):
        print(f"\n  Statistical tests (Fisher exact):")
        for sig, test in list(stat_results["signal_tests"].items())[:10]:
            star = " *" if test["significant"] else ""
            print(f"    {sig:<25} WR_with={test['wr_with_signal']:.0%} WR_without={test['wr_without_signal']:.0%} OR={test['odds_ratio']} p={test['p_value']:.3f}{star}")

    if stat_results.get("confidence_correlation"):
        cc = stat_results["confidence_correlation"]
        print(f"\n  Confidence → Win correlation: r={cc['correlation']:.3f}, p={cc['p_value']:.3f}")

    if stat_results.get("direction_analysis"):
        da = stat_results["direction_analysis"]
        print(f"\n  Direction analysis: Bull WR={da['bullish_wr']:.0%} (n={da['bullish_n']}), Bear WR={da['bearish_wr']:.0%} (n={da['bearish_n']})")

    # ─── 9. Compare ML vs Fixed Rule ──────────────────────────────────────
    print("\n[8] Comparing ML-selected vs fixed-rule trades...")

    comparison = compare_ml_vs_fixed_rule(df, ml_results)
    for rule_name, data in comparison.items():
        if "n_trades" in data and data["n_trades"] > 0:
            wr = data.get("win_rate", "N/A")
            pnl = data.get("avg_pnl", "N/A")
            print(f"  {rule_name}: {data['n_trades']} trades, WR={wr}, avg PnL={pnl}%")

    # ─── 10. Compute Optimal Weights ──────────────────────────────────────
    print("\n[9] Computing optimal signal weights...")

    weights = compute_optimal_weights(df, feature_cols)

    if "error" not in weights:
        print(f"\n  Recommended signal weights (0-2 scale, 1.0 = neutral):")
        for sig, w in weights.items():
            direction_marker = "+" if w["direction"] == "positive" else ("-" if w["direction"] == "negative" else " ")
            bar = "█" * int(w["normalized_weight"] * 10)
            print(f"    {sig:<30} {w['normalized_weight']:.3f} {direction_marker} {bar}")
    else:
        print(f"  {weights['error']}")

    # ─── 11. Save Results ─────────────────────────────────────────────────
    print(f"\n[10] Saving results...")

    output = {
        "metadata": {
            "run_date": dt.datetime.now().isoformat(),
            "n_trades_total": len(df),
            "n_wins": int(df["target"].sum()),
            "n_losses": int((1 - df["target"]).sum()),
            "overall_win_rate": round(float(df["target"].mean()), 3),
            "date_range": {
                "earliest": str(df["entry_date"].min()),
                "latest": str(df["entry_date"].max()),
            },
            "data_sources": dict(df["origin"].value_counts().to_dict()),
            "validation_method": "sliding_window_walk_forward",
        },
        "signal_cooccurrence": cooccurrence,
        "mutual_information": mi_results,
        "ml_results": ml_results if ml_results else {"skipped": "insufficient_data"},
        "statistical_analysis": stat_results,
        "comparison_vs_fixed_rule": comparison,
        "optimal_weights": weights,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"  Results saved to {OUTPUT_FILE}")

    # ─── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nData: {len(df)} trades ({df['target'].sum():.0f}W / {(1-df['target']).sum():.0f}L = {df['target'].mean():.0%} WR)")

    # Best ML model
    if ml_results:
        best_name = None
        best_auc = 0
        for name, m in ml_results["model_metrics"].items():
            if "error" not in m and m.get("auc_roc") is not None and m["auc_roc"] > best_auc:
                best_auc = m["auc_roc"]
                best_name = name
        if best_name:
            bm = ml_results["model_metrics"][best_name]
            print(f"\nBest ML model: {best_name}")
            print(f"  AUC: {bm['auc_roc']:.3f}, Accuracy: {bm['accuracy']:.3f}, Precision: {bm['precision']:.3f}")
            if best_auc < 0.55:
                print(f"  NOTE: AUC {best_auc:.3f} is near random — ML adds little value with current sample size.")
                print(f"  Recommend: collect more trades before trusting ML weights.")
            elif best_auc > 0.65:
                print(f"  ML shows meaningful predictive power. Weights can be trusted cautiously.")

    # Top signals
    if cooccurrence.get("individual_signals"):
        top = list(cooccurrence["individual_signals"].items())[:3]
        if top:
            print(f"\nTop signals by win rate:")
            for sig, p in top:
                print(f"  {sig}: {p['win_rate']:.0%} WR ({p['n_trades']} trades)")

    # Optimal N
    if cooccurrence.get("optimal_n_sources"):
        best_n = max(cooccurrence["optimal_n_sources"].items(), key=lambda x: x[1]["win_rate"])
        print(f"\nOptimal confirming sources: {best_n[0]} (WR={best_n[1]['win_rate']:.0%}, n={best_n[1]['n_trades']})")
        print(f"  Current rule uses 5+. {'This looks right.' if best_n[0] >= 5 else f'Consider lowering to {best_n[0]}.'}")

    if cooccurrence.get("golden_combinations"):
        print(f"\nGolden combos: {', '.join(list(cooccurrence['golden_combinations'].keys())[:3])}")
    if cooccurrence.get("toxic_combinations"):
        print(f"Toxic combos: {', '.join(list(cooccurrence['toxic_combinations'].keys())[:3])}")

    print()


if __name__ == "__main__":
    main()
