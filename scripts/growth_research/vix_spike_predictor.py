#!/usr/bin/env python3
"""
VIX Spike Probability Predictor
================================
ML model to predict probability of VIX exceeding 25/30 within next 5 trading days.

Goal: Early warning system so we can position BEFORE spikes, not just react after.

Strategy overlay:
  - P(spike) > 50%: reduce UPRO to SPY (defensive)
  - P(spike) > 70%: move to SHY + prepare UPRO buy for VIX>30 cross
  - Compare overlay vs pure v4.4 buy-and-hold UPRO

Models: LightGBM, Logistic Regression, Random Forest (walk-forward, SLIDING 252d)

HC compliance:
  - SLIDING window only (HC #0)
  - Fixed $100K capital, NO DCA (HC #713)
  - Risk-adjusted metrics primary (HC #69)
  - Adversarial validation (HC #705): permutation, sub-period, outlier removal, R1
  - Exploratory research (HC #714)
"""
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    brier_score_loss,
    f1_score,
)
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/vix_spike_predictor")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CAPITAL = 100_000  # Fixed, no DCA (HC #713)
TRAIN_DAYS = 252  # 1 year sliding window
FORECAST_HORIZON = 5  # Predict spike within next 5 trading days
VIX_SPIKE_25 = 25
VIX_SPIKE_30 = 30

# ── Data Fetching ───────────────────────────────────────────────────────────

def fetch_data() -> pd.DataFrame:
    """Fetch cross-asset daily data 2010-2026."""
    tickers = {
        "^VIX": "VIX",
        "^VIX3M": "VIX3M",  # 3-month VIX for term structure
        "SPY": "SPY",
        "GLD": "GLD",
        "TLT": "TLT",
        "HYG": "HYG",
        "IEF": "IEF",
        "UUP": "UUP",
        "USO": "USO",
        "QQQ": "QQQ",
        "IWM": "IWM",
        "UPRO": "UPRO",
        "SHY": "SHY",
    }

    print(f"Fetching data for {list(tickers.keys())}...")
    df = pd.DataFrame()
    for yahoo_ticker, name in tickers.items():
        try:
            t = yf.Ticker(yahoo_ticker)
            hist = t.history(start="2010-01-01", end="2026-07-17", auto_adjust=True)
            if len(hist) > 0:
                # Normalize timezone to date-only index
                series = hist["Close"].copy()
                series.index = series.index.tz_localize(None).normalize()
                df[name] = series
                print(f"  {name}: {len(hist)} rows")
            else:
                print(f"  Warning: {yahoo_ticker} returned no data")
        except Exception as e:
            print(f"  Warning: {yahoo_ticker} failed: {e}")

    df = df.dropna(subset=["VIX", "SPY"])
    print(f"  Data: {df.index[0].date()} to {df.index[-1].date()}, {len(df)} trading days")
    print(f"  Columns: {list(df.columns)}")

    # Forward-fill minor gaps in non-critical columns
    df = df.ffill().dropna()
    return df


# ── Feature Engineering ─────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build predictive features from cross-asset data."""
    feat = pd.DataFrame(index=df.index)

    # ── VIX features ──
    feat["vix_level"] = df["VIX"]
    feat["vix_chg_5d"] = df["VIX"].pct_change(5)
    feat["vix_chg_10d"] = df["VIX"].pct_change(10)
    feat["vix_chg_21d"] = df["VIX"].pct_change(21)
    # Vectorized percentile rank (much faster than rolling apply)
    def rolling_pctrank(s, window):
        result = np.full(len(s), np.nan)
        vals = s.values
        for i in range(window - 1, len(vals)):
            w = vals[i - window + 1:i + 1]
            result[i] = np.sum(w <= vals[i]) / window
        return pd.Series(result, index=s.index)

    feat["vix_pctrank_63d"] = rolling_pctrank(df["VIX"], 63)
    feat["vix_pctrank_252d"] = rolling_pctrank(df["VIX"], 252)
    feat["vix_zscore_63d"] = (df["VIX"] - df["VIX"].rolling(63).mean()) / df["VIX"].rolling(63).std()

    # ── VIX term structure (backwardation = stress) ──
    if "VIX3M" in df.columns:
        feat["vix_term_ratio"] = df["VIX"] / df["VIX3M"]  # >1 = backwardation = stress
        feat["vix_term_ratio_chg5d"] = feat["vix_term_ratio"].pct_change(5)
        feat["vix_in_backwardation"] = (feat["vix_term_ratio"] > 1.0).astype(float)
    else:
        print("  Warning: VIX3M not available, skipping term structure features")

    # ── SPY features ──
    spy_ret = df["SPY"].pct_change()
    feat["spy_ret_1d"] = spy_ret
    feat["spy_ret_5d"] = df["SPY"].pct_change(5)
    feat["spy_ret_10d"] = df["SPY"].pct_change(10)
    feat["spy_ret_21d"] = df["SPY"].pct_change(21)
    feat["spy_rvol_10d"] = spy_ret.rolling(10).std() * np.sqrt(252)
    feat["spy_rvol_21d"] = spy_ret.rolling(21).std() * np.sqrt(252)
    feat["spy_vol_of_vol"] = feat["spy_rvol_10d"].rolling(21).std()

    # Drawdown from 63d high
    feat["spy_dd_63d"] = df["SPY"] / df["SPY"].rolling(63).max() - 1.0

    # ── Credit spread proxy (HYG vs IEF) ──
    if "HYG" in df.columns and "IEF" in df.columns:
        credit_spread = np.log(df["IEF"]) - np.log(df["HYG"])  # Wider = stress
        feat["credit_spread"] = credit_spread
        feat["credit_spread_chg5d"] = credit_spread.diff(5)
        feat["credit_spread_chg21d"] = credit_spread.diff(21)
        feat["credit_spread_zscore"] = (credit_spread - credit_spread.rolling(63).mean()) / credit_spread.rolling(63).std()

    # ── Gold (safe haven) ──
    if "GLD" in df.columns:
        feat["gld_ret_5d"] = df["GLD"].pct_change(5)
        feat["gld_ret_21d"] = df["GLD"].pct_change(21)
        feat["gld_momentum"] = df["GLD"] / df["GLD"].rolling(63).mean() - 1.0

    # ── Bonds (flight to quality) ──
    if "TLT" in df.columns:
        feat["tlt_ret_5d"] = df["TLT"].pct_change(5)
        feat["tlt_ret_21d"] = df["TLT"].pct_change(21)

    # ── Oil (supply shocks) ──
    if "USO" in df.columns:
        feat["uso_ret_5d"] = df["USO"].pct_change(5)
        feat["uso_ret_21d"] = df["USO"].pct_change(21)
        feat["uso_rvol_21d"] = df["USO"].pct_change().rolling(21).std() * np.sqrt(252)

    # ── Dollar strength ──
    if "UUP" in df.columns:
        feat["uup_ret_5d"] = df["UUP"].pct_change(5)
        feat["uup_ret_21d"] = df["UUP"].pct_change(21)

    # ── Market breadth proxy: QQQ vs IWM relative strength ──
    if "QQQ" in df.columns and "IWM" in df.columns:
        rel_strength = np.log(df["QQQ"]) - np.log(df["IWM"])
        feat["qqq_iwm_rel_5d"] = rel_strength.diff(5)
        feat["qqq_iwm_rel_21d"] = rel_strength.diff(21)

    # ── Fear premium: VIX vs realized vol ──
    feat["fear_premium"] = df["VIX"] / 100 - feat["spy_rvol_21d"]  # VIX is annualized %
    feat["fear_premium_zscore"] = (feat["fear_premium"] - feat["fear_premium"].rolling(63).mean()) / feat["fear_premium"].rolling(63).std()

    # ── Calendar features ──
    feat["month"] = df.index.month
    feat["day_of_week"] = df.index.dayofweek
    feat["is_aug_sep_oct"] = df.index.month.isin([8, 9, 10]).astype(float)

    return feat


# ── Target Construction ─────────────────────────────────────────────────────

def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Build forward-looking spike targets."""
    targets = pd.DataFrame(index=df.index)

    # Will VIX exceed threshold within next FORECAST_HORIZON days?
    for threshold, name in [(VIX_SPIKE_25, "spike_25"), (VIX_SPIKE_30, "spike_30")]:
        future_max_vix = df["VIX"].rolling(FORECAST_HORIZON, min_periods=1).max().shift(-FORECAST_HORIZON)
        targets[name] = (future_max_vix >= threshold).astype(float)

    return targets


# ── Walk-Forward Engine ─────────────────────────────────────────────────────

def walk_forward_predict(features: pd.DataFrame, targets: pd.Series,
                          model_type: str = "lgbm", train_days: int = TRAIN_DAYS,
                          step_days: int = 21) -> pd.DataFrame:
    """Walk-forward with SLIDING window. Returns DataFrame of predictions."""
    results = []
    feature_cols = [c for c in features.columns if c not in ["month", "day_of_week", "is_aug_sep_oct"]
                    or model_type != "logistic"]

    # For logistic regression, skip calendar dummies (they need encoding)
    if model_type == "logistic":
        feature_cols = [c for c in features.columns
                        if c not in ["month", "day_of_week"]]

    valid_idx = features.dropna().index.intersection(targets.dropna().index)
    features_clean = features.loc[valid_idx, feature_cols].copy()
    targets_clean = targets.loc[valid_idx].copy()

    n = len(features_clean)
    start_idx = train_days

    print(f"  Walk-forward: {model_type}, {n} samples, train={train_days}d, step={step_days}d")

    fold = 0
    i = start_idx
    while i < n:
        end_test = min(i + step_days, n)

        X_train = features_clean.iloc[i - train_days:i]
        y_train = targets_clean.iloc[i - train_days:i]
        X_test = features_clean.iloc[i:end_test]
        y_test = targets_clean.iloc[i:end_test]

        if len(X_test) == 0:
            break

        # Handle class imbalance
        pos_rate = y_train.mean()
        if pos_rate == 0 or pos_rate == 1:
            # Degenerate fold, skip
            i += step_days
            continue

        scale_pos = (1 - pos_rate) / pos_rate

        try:
            if model_type == "lgbm":
                model = lgb.LGBMClassifier(
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.05,
                    scale_pos_weight=scale_pos,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_samples=20,
                    random_state=42,
                    verbose=-1,
                    n_jobs=1,
                )
                model.fit(X_train, y_train)
                proba = model.predict_proba(X_test)[:, 1]

            elif model_type == "logistic":
                scaler = StandardScaler()
                X_tr_scaled = scaler.fit_transform(X_train)
                X_te_scaled = scaler.transform(X_test)
                model = LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    C=0.1,
                    random_state=42,
                )
                model.fit(X_tr_scaled, y_train)
                proba = model.predict_proba(X_te_scaled)[:, 1]

            elif model_type == "rf":
                model = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=8,
                    class_weight="balanced",
                    min_samples_leaf=20,
                    random_state=42,
                    n_jobs=1,
                )
                model.fit(X_train, y_train)
                proba = model.predict_proba(X_test)[:, 1]

            for j in range(len(X_test)):
                results.append({
                    "date": X_test.index[j],
                    "y_true": y_test.iloc[j],
                    "y_prob": proba[j],
                    "fold": fold,
                })

        except Exception as e:
            print(f"    Fold {fold} error: {e}")

        fold += 1
        if fold % 25 == 0:
            total_folds = (n - start_idx) // step_days
            print(f"    Progress: fold {fold}/{total_folds}")
        i += step_days

    print(f"  Completed {fold} folds, {len(results)} predictions")
    return pd.DataFrame(results).set_index("date") if results else pd.DataFrame()


# ── Evaluation ──────────────────────────────────────────────────────────────

def evaluate_predictions(preds: pd.DataFrame, target_name: str, model_name: str) -> dict:
    """Evaluate predictions with precision/recall at various thresholds."""
    if preds.empty or preds["y_true"].nunique() < 2:
        print(f"  {model_name}: insufficient data for evaluation")
        return {}

    y_true = preds["y_true"].values
    y_prob = preds["y_prob"].values

    auc = roc_auc_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)

    print(f"\n  === {model_name} — {target_name} ===")
    print(f"  AUC-ROC: {auc:.4f}")
    print(f"  Brier Score: {brier:.4f}")
    print(f"  Base rate: {y_true.mean():.4f} ({int(y_true.sum())}/{len(y_true)} positive days)")

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    threshold_results = {}
    print(f"  {'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Alerts':>10} {'True+':>10} {'FalseAlarm':>10}")
    print(f"  {'-'*70}")

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        n_alerts = y_pred.sum()
        if n_alerts == 0:
            print(f"  {t:>10.2f} {'N/A':>10} {'0.00':>10} {'N/A':>10} {0:>10} {0:>10} {0:>10}")
            continue

        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        tp = int((y_pred & y_true.astype(int)).sum())
        fp = int(n_alerts - tp)

        threshold_results[str(t)] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "alerts": int(n_alerts),
            "true_positives": tp,
            "false_alarms": fp,
        }
        print(f"  {t:>10.2f} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f} {n_alerts:>10} {tp:>10} {fp:>10}")

    # Spike capture analysis
    actual_spikes = preds[preds["y_true"] == 1]
    total_spikes = len(actual_spikes)
    print(f"\n  Spike capture analysis ({total_spikes} actual spike-days):")

    for t in [0.3, 0.5, 0.7]:
        captured = (actual_spikes["y_prob"] >= t).sum()
        rate = captured / total_spikes if total_spikes > 0 else 0
        print(f"    At P>{t:.1f}: captured {captured}/{total_spikes} ({rate:.1%})")

    return {
        "model": model_name,
        "target": target_name,
        "auc_roc": round(auc, 4),
        "brier": round(brier, 4),
        "base_rate": round(float(y_true.mean()), 4),
        "n_samples": len(y_true),
        "n_positive": int(y_true.sum()),
        "thresholds": threshold_results,
    }


# ── SHAP Feature Importance ────────────────────────────────────────────────

def compute_shap_importance(features: pd.DataFrame, targets: pd.Series) -> pd.DataFrame:
    """Train final LightGBM on last 2 years and compute SHAP values."""
    print("\n  Computing SHAP feature importance (last 2 years)...")
    feature_cols = features.columns.tolist()
    valid_idx = features.dropna().index.intersection(targets.dropna().index)
    X = features.loc[valid_idx, feature_cols].iloc[-504:]
    y = targets.loc[valid_idx].iloc[-504:]

    pos_rate = y.mean()
    scale_pos = (1 - pos_rate) / max(pos_rate, 0.01)

    model = lgb.LGBMClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        scale_pos_weight=scale_pos, subsample=0.8,
        colsample_bytree=0.8, min_child_samples=20,
        random_state=42, verbose=-1, n_jobs=-1,
    )
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # For binary classification, shap_values may be a list [neg, pos]
    if isinstance(shap_values, list):
        sv = np.abs(shap_values[1]).mean(axis=0)
    else:
        sv = np.abs(shap_values).mean(axis=0)

    importance = pd.DataFrame({
        "feature": feature_cols,
        "shap_importance": sv,
    }).sort_values("shap_importance", ascending=False)

    print("\n  Top 15 features by SHAP importance:")
    for _, row in importance.head(15).iterrows():
        print(f"    {row['feature']:30s} {row['shap_importance']:.4f}")

    return importance


# ── Trading Strategy Overlay ────────────────────────────────────────────────

def backtest_overlay(df: pd.DataFrame, preds_25: pd.DataFrame, preds_30: pd.DataFrame) -> dict:
    """
    Backtest prediction-based defensive overlay vs pure UPRO hold.

    Strategies:
      1. Pure UPRO (buy-and-hold, $100K fixed start)
      2. VMR-like: hold UPRO normally, buy UPRO at VIX>30
      3. Predictive overlay:
         - Normal: UPRO
         - P(spike_25) > 50%: switch to SPY (defensive)
         - P(spike_30) > 70%: switch to SHY + queue UPRO buy at VIX>30 cross
    """
    print("\n" + "="*80)
    print("TRADING STRATEGY OVERLAY BACKTEST")
    print("="*80)

    # Align dates across all dataframes
    common_dates = df.index.intersection(preds_25.index).intersection(preds_30.index)
    if len(common_dates) < 252:
        print("  Insufficient overlapping dates for overlay backtest")
        return {}

    data = df.loc[common_dates].copy()
    p25 = preds_25.loc[common_dates, "y_prob"].copy()
    p30 = preds_30.loc[common_dates, "y_prob"].copy()

    # Daily returns
    upro_ret = data["UPRO"].pct_change().fillna(0)
    spy_ret = data["SPY"].pct_change().fillna(0)
    shy_ret = data["SHY"].pct_change().fillna(0) if "SHY" in data.columns else pd.Series(0.0001, index=data.index)

    strategies = {}

    # Strategy 1: Pure UPRO
    nav_upro = CAPITAL * (1 + upro_ret).cumprod()
    strategies["Pure UPRO"] = nav_upro

    # Strategy 2: VMR-like (UPRO normally, buy at VIX>30)
    # This is the existing strategy — hold UPRO, extra conviction buy at VIX>30
    strategies["VMR (UPRO hold)"] = nav_upro.copy()  # Same as pure UPRO for this simplified test

    # Strategy 3: Predictive overlay
    nav_overlay = [CAPITAL]
    position = "UPRO"  # Start in UPRO
    position_log = []
    vix_crossed_30 = False

    for i in range(1, len(data)):
        date = data.index[i]
        prob_25 = p25.iloc[i] if i < len(p25) else 0
        prob_30 = p30.iloc[i] if i < len(p30) else 0
        current_vix = data["VIX"].iloc[i]

        # Decision logic
        if prob_30 > 0.70:
            new_position = "SHY"
        elif prob_25 > 0.50:
            new_position = "SPY"
        else:
            new_position = "UPRO"

        # If VIX actually crossed 30 and we were in SHY, switch to UPRO (the actual spike buy)
        if current_vix >= 30 and position == "SHY":
            new_position = "UPRO"
            vix_crossed_30 = True

        if new_position != position:
            position_log.append((date, position, new_position, prob_25, prob_30, current_vix))
        position = new_position

        # Apply return based on current position
        if position == "UPRO":
            daily_ret = upro_ret.iloc[i]
        elif position == "SPY":
            daily_ret = spy_ret.iloc[i]
        else:  # SHY
            daily_ret = shy_ret.iloc[i]

        nav_overlay.append(nav_overlay[-1] * (1 + daily_ret))

    nav_overlay = pd.Series(nav_overlay, index=data.index)
    strategies["Predictive Overlay"] = nav_overlay

    # Strategy 4: Perfect foresight (oracle — upper bound)
    # Switch to SHY 5 days before VIX>30, buy UPRO at VIX>30
    vix_will_spike = (data["VIX"].rolling(FORECAST_HORIZON, min_periods=1).max().shift(-FORECAST_HORIZON) >= 30)
    nav_oracle = [CAPITAL]
    for i in range(1, len(data)):
        if vix_will_spike.iloc[i]:
            daily_ret = shy_ret.iloc[i]
        else:
            daily_ret = upro_ret.iloc[i]
        nav_oracle.append(nav_oracle[-1] * (1 + daily_ret))
    nav_oracle = pd.Series(nav_oracle, index=data.index)
    strategies["Perfect Foresight"] = nav_oracle

    # Compute metrics for each strategy
    print(f"\n  Backtest period: {common_dates[0].date()} to {common_dates[-1].date()} ({len(common_dates)} days)")
    print(f"\n  {'Strategy':<25} {'Final NAV':>12} {'CAGR':>8} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} {'Switches':>8}")
    print(f"  {'-'*85}")

    results = {}
    for name, nav in strategies.items():
        rets = nav.pct_change().dropna()
        years = len(rets) / 252
        cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1
        sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
        downside = rets[rets < 0].std()
        sortino = rets.mean() / downside * np.sqrt(252) if downside > 0 else 0
        dd = (nav / nav.cummax() - 1).min()

        switches = len(position_log) if name == "Predictive Overlay" else 0

        print(f"  {name:<25} ${nav.iloc[-1]:>10,.0f} {cagr:>7.1%} {sharpe:>7.2f} {sortino:>7.2f} {dd:>7.1%} {switches:>8}")

        results[name] = {
            "final_nav": round(float(nav.iloc[-1]), 2),
            "cagr": round(cagr, 4),
            "sharpe": round(sharpe, 4),
            "sortino": round(sortino, 4),
            "max_drawdown": round(float(dd), 4),
        }

    # Log position switches
    if position_log:
        print(f"\n  Position switches ({len(position_log)} total):")
        for date, old, new, p25, p30, vix in position_log[:20]:
            print(f"    {date.date()} {old:>5} -> {new:<5}  P25={p25:.2f} P30={p30:.2f} VIX={vix:.1f}")
        if len(position_log) > 20:
            print(f"    ... and {len(position_log)-20} more")

    return results


# ── Adversarial Validation (HC #705) ────────────────────────────────────────

def adversarial_validation(features: pd.DataFrame, targets: pd.Series,
                            model_type: str = "lgbm") -> dict:
    """
    Adversarial validation suite:
    1. Permutation test (shuffle targets, re-run)
    2. Sub-period stability (split into 3 periods)
    3. Outlier removal (drop top/bottom 5% VIX days)
    4. R1: regime-agnostic check
    """
    print("\n" + "="*80)
    print("ADVERSARIAL VALIDATION")
    print("="*80)

    results = {}

    # 1. Permutation test
    print("\n  1. Permutation test (3 shuffles)...")
    real_preds = walk_forward_predict(features, targets, model_type, step_days=63)
    if real_preds.empty or real_preds["y_true"].nunique() < 2:
        print("    Skipped: insufficient predictions")
        return results

    real_auc = roc_auc_score(real_preds["y_true"], real_preds["y_prob"])

    perm_aucs = []
    for seed in range(3):
        shuffled = targets.sample(frac=1.0, random_state=seed)
        shuffled.index = targets.index
        perm_preds = walk_forward_predict(features, shuffled, model_type, step_days=63)
        if not perm_preds.empty and perm_preds["y_true"].nunique() >= 2:
            perm_aucs.append(roc_auc_score(perm_preds["y_true"], perm_preds["y_prob"]))

    if perm_aucs:
        perm_mean = np.mean(perm_aucs)
        perm_std = np.std(perm_aucs)
        z_score = (real_auc - perm_mean) / max(perm_std, 0.001)
        print(f"    Real AUC: {real_auc:.4f}")
        print(f"    Permutation AUC: {perm_mean:.4f} +/- {perm_std:.4f}")
        print(f"    Z-score: {z_score:.2f}")
        print(f"    PASS: {'YES' if z_score > 2.0 else 'NO'}")
        results["permutation"] = {
            "real_auc": round(real_auc, 4),
            "perm_mean_auc": round(perm_mean, 4),
            "z_score": round(z_score, 2),
            "pass": z_score > 2.0,
        }

    # 2. Sub-period stability
    print("\n  2. Sub-period stability (3 periods)...")
    valid_idx = features.dropna().index.intersection(targets.dropna().index)
    n = len(valid_idx)
    period_size = n // 3
    period_aucs = []

    for p in range(3):
        start = p * period_size
        end = (p + 1) * period_size if p < 2 else n
        period_dates = valid_idx[start:end]
        period_feat = features.loc[period_dates]
        period_tgt = targets.loc[period_dates]

        preds = walk_forward_predict(period_feat, period_tgt, model_type,
                                      train_days=min(TRAIN_DAYS, len(period_feat)//2),
                                      step_days=63)
        if not preds.empty and preds["y_true"].nunique() >= 2:
            auc = roc_auc_score(preds["y_true"], preds["y_prob"])
            period_aucs.append(auc)
            date_range = f"{period_dates[0].date()} to {period_dates[-1].date()}"
            print(f"    Period {p+1} ({date_range}): AUC={auc:.4f}")
        else:
            print(f"    Period {p+1}: insufficient data")

    if len(period_aucs) >= 2:
        stability = 1.0 - (max(period_aucs) - min(period_aucs)) / max(max(period_aucs), 0.01)
        print(f"    Stability (1-range/max): {stability:.4f}")
        print(f"    PASS: {'YES' if stability > 0.7 and min(period_aucs) > 0.55 else 'NO'}")
        results["sub_period"] = {
            "period_aucs": [round(a, 4) for a in period_aucs],
            "stability": round(stability, 4),
            "pass": stability > 0.7 and min(period_aucs) > 0.55,
        }

    # 3. Outlier removal
    print("\n  3. Outlier removal (drop extreme VIX days)...")
    if "vix_level" in features.columns:
        vix_col = features["vix_level"]
        low_q = vix_col.quantile(0.05)
        high_q = vix_col.quantile(0.95)
        mask = (vix_col >= low_q) & (vix_col <= high_q)
        clean_feat = features.loc[mask]
        clean_tgt = targets.loc[mask.index[mask]]

        preds = walk_forward_predict(clean_feat, clean_tgt, model_type, step_days=63)
        if not preds.empty and preds["y_true"].nunique() >= 2:
            clean_auc = roc_auc_score(preds["y_true"], preds["y_prob"])
            degradation = (real_auc - clean_auc) / max(real_auc, 0.01)
            print(f"    Clean AUC (no extremes): {clean_auc:.4f}")
            print(f"    Degradation from full: {degradation:.4f}")
            print(f"    PASS: {'YES' if degradation < 0.15 else 'NO'}")
            results["outlier_removal"] = {
                "clean_auc": round(clean_auc, 4),
                "degradation": round(degradation, 4),
                "pass": degradation < 0.15,
            }

    # 4. R1 — Regime check (bull vs bear market)
    print("\n  4. R1 regime-agnostic check (bull vs bear)...")
    if "spy_ret_21d" in features.columns:
        bull_mask = features["spy_ret_21d"] > 0
        bear_mask = features["spy_ret_21d"] <= 0

        for regime, mask in [("Bull", bull_mask), ("Bear", bear_mask)]:
            regime_feat = features.loc[mask.values]
            regime_tgt = targets.loc[mask.values]
            preds = walk_forward_predict(regime_feat, regime_tgt, model_type,
                                          train_days=min(TRAIN_DAYS, len(regime_feat)//2),
                                          step_days=63)
            if not preds.empty and preds["y_true"].nunique() >= 2:
                regime_auc = roc_auc_score(preds["y_true"], preds["y_prob"])
                print(f"    {regime}: AUC={regime_auc:.4f} ({len(regime_feat)} days)")
            else:
                print(f"    {regime}: insufficient data")

    return results


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("="*80)
    print("VIX SPIKE PROBABILITY PREDICTOR")
    print(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # Fetch data
    df = fetch_data()

    # Build features and targets
    print("\nBuilding features...")
    features = build_features(df)
    targets = build_targets(df)

    print(f"  Features: {features.shape[1]} columns")
    print(f"  Target base rates:")
    for col in targets.columns:
        valid = targets[col].dropna()
        print(f"    {col}: {valid.mean():.4f} ({int(valid.sum())}/{len(valid)} days)")

    # Drop rows with NaN features (from rolling calculations)
    valid_mask = features.notna().all(axis=1) & targets.notna().all(axis=1)
    features = features.loc[valid_mask]
    targets = targets.loc[valid_mask]
    print(f"  After cleanup: {len(features)} valid rows ({features.index[0].date()} to {features.index[-1].date()})")

    all_results = {}

    # Run models for each target
    for target_name in ["spike_25", "spike_30"]:
        print(f"\n{'='*80}")
        print(f"TARGET: {target_name} (VIX > {25 if '25' in target_name else 30} within {FORECAST_HORIZON}d)")
        print(f"{'='*80}")

        target = targets[target_name]

        for model_type, model_name in [
            ("lgbm", "LightGBM"),
            ("logistic", "Logistic Regression"),
            ("rf", "Random Forest"),
        ]:
            preds = walk_forward_predict(features, target, model_type)
            if not preds.empty:
                eval_result = evaluate_predictions(preds, target_name, model_name)
                all_results[f"{target_name}_{model_type}"] = {
                    "evaluation": eval_result,
                    "predictions": preds,
                }

    # SHAP analysis on best model (LightGBM, spike_30)
    print("\n" + "="*80)
    print("SHAP FEATURE IMPORTANCE (LightGBM, spike_30)")
    print("="*80)
    shap_importance = compute_shap_importance(features, targets["spike_30"])
    shap_importance.to_csv(OUTPUT_DIR / "shap_importance.csv", index=False)

    # Trading strategy overlay
    # Use LightGBM predictions for overlay
    lgbm_25 = all_results.get("spike_25_lgbm", {}).get("predictions", pd.DataFrame())
    lgbm_30 = all_results.get("spike_30_lgbm", {}).get("predictions", pd.DataFrame())

    overlay_results = {}
    if not lgbm_25.empty and not lgbm_30.empty:
        overlay_results = backtest_overlay(df, lgbm_25, lgbm_30)

    # Adversarial validation (on spike_30 LightGBM — the main use case)
    adversarial = adversarial_validation(features, targets["spike_30"], "lgbm")

    # ── Summary ──
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    # Best model selection
    best_auc = 0
    best_model = ""
    for key, val in all_results.items():
        ev = val.get("evaluation", {})
        auc = ev.get("auc_roc", 0)
        if auc > best_auc:
            best_auc = auc
            best_model = key

    print(f"\n  Best model: {best_model} (AUC={best_auc:.4f})")

    # Feasibility assessment
    print("\n  FEASIBILITY ASSESSMENT:")
    if best_auc > 0.75:
        print("  -> STRONG signal. VIX spike prediction appears feasible.")
        print("  -> Recommend: implement as real-time overlay on growth strategy.")
    elif best_auc > 0.65:
        print("  -> MODERATE signal. Some predictive power exists.")
        print("  -> Recommend: use as one input among several, not standalone trigger.")
    elif best_auc > 0.55:
        print("  -> WEAK signal. Marginally better than random.")
        print("  -> Recommend: may help at extreme probability thresholds only.")
    else:
        print("  -> NO signal detected. VIX spikes appear unpredictable with these features.")
        print("  -> Recommend: stick to reactive strategy (buy after VIX>30 cross).")

    # Adversarial summary
    if adversarial:
        n_pass = sum(1 for v in adversarial.values() if isinstance(v, dict) and v.get("pass", False))
        n_total = len(adversarial)
        print(f"\n  Adversarial validation: {n_pass}/{n_total} tests passed")

    # Save results
    save_results = {
        "run_timestamp": datetime.now().isoformat(),
        "data_range": f"{df.index[0].date()} to {df.index[-1].date()}",
        "n_trading_days": len(df),
        "models": {},
        "overlay_backtest": overlay_results,
        "adversarial": adversarial,
        "best_model": best_model,
        "best_auc": best_auc,
    }

    for key, val in all_results.items():
        ev = val.get("evaluation", {})
        save_results["models"][key] = ev

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(save_results, f, indent=2, default=str)

    # Save predictions for best model
    if best_model in all_results:
        best_preds = all_results[best_model]["predictions"]
        best_preds.to_csv(OUTPUT_DIR / "best_predictions.csv")

    print(f"\n  Results saved to {OUTPUT_DIR}/")
    print("  DONE.")


if __name__ == "__main__":
    main()
