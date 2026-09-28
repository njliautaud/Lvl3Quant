#!/usr/bin/env python3
"""
ML Macro Risk Scaler — GBM-based continuous position sizing for UPRO/SHY/GLD
=============================================================================
Replaces the binary danger-signal scaling in v4.4 (#472) with a continuous
ML probability of VIX>30 within 5 days, trained walk-forward.

Base direction signal: VIX rolling percentile (63d lookback, 20/80 thresholds)
Position scaling: GBM P(VIX>30 in 5d) -> 4-tier leverage schedule

HC #0  : Sliding walk-forward (3yr train, 1yr test, slide 1yr)
HC #705: Permutation test, sub-period consistency, walk-forward excess check
HC #713: Fixed $100K, no DCA, next-day execution
"""

import os
import sys
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/ml_macro_risk_scaler")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "2008-01-01"  # extra history for feature warmup
END_DATE = "2026-07-17"
BACKTEST_START = "2012-01-01"  # actual backtest starts here (after warmup)
INITIAL_CAPITAL = 100_000

# Walk-forward (sliding)
TRAIN_YEARS = 3
TEST_YEARS = 1

# GBM hyperparameters (per spec)
GBM_PARAMS = dict(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    min_samples_leaf=20,
    random_state=42,
)

# VIX spike target
VIX_SPIKE_THRESHOLD = 30
VIX_SPIKE_HORIZON = 5  # trading days

# ML scaling tiers
# p = P(VIX>30 in 5d) from GBM
ML_TIERS = [
    (0.00, 0.15, 0.67, 0.33, 0.00, 0.00),  # low risk: 67% UPRO / 33% SPY
    (0.15, 0.35, 0.50, 0.50, 0.00, 0.00),  # moderate: 50/50
    (0.35, 0.60, 0.33, 0.67, 0.00, 0.00),  # high: 33/67
    (0.60, 1.01, 0.00, 0.00, 1.00, 0.00),  # extreme: 100% SHY
]

# v4.4 baseline tiers (binary danger signals — for comparison)
V44_DANGER_TIERS = [
    (0, 0.67, 0.33, 0.00, 0.00),  # 0 dangers: 67/33
    (1, 0.50, 0.50, 0.00, 0.00),  # 1 danger: 50/50
    (2, 0.00, 0.00, 1.00, 0.00),  # 2+ dangers: 100% SHY
]

# Permutation test
N_PERMUTATIONS = 100
PERM_P_THRESHOLD = 0.05

# Sub-period consistency
CV_SHARPE_THRESHOLD = 0.60


# ============================================================================
# DATA LOADING
# ============================================================================
def load_data():
    """Download all required tickers from yfinance."""
    import yfinance as yf

    tickers = [
        "^VIX", "^VIX3M", "SPY", "UPRO", "SHY", "GLD", "TLT",
        "HYG", "IEF", "SLV", "USO", "UUP", "CPER", "BTC-USD",
    ]
    print(f"Downloading {len(tickers)} tickers from {START_DATE} to {END_DATE}...")
    raw = yf.download(tickers, start=START_DATE, end=END_DATE,
                      auto_adjust=True, threads=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        closes = raw["Close"].copy()
    else:
        closes = raw.copy()

    # Flatten any remaining multi-index
    if hasattr(closes.columns, "droplevel"):
        try:
            closes.columns = closes.columns.droplevel(1)
        except Exception:
            pass

    renames = {"^VIX": "VIX", "^VIX3M": "VIX3M", "BTC-USD": "BTC"}
    closes = closes.rename(columns=renames)
    closes = closes.ffill().dropna(subset=["SPY"])
    print(f"  Loaded {len(closes)} rows, {closes.columns.tolist()}")
    return closes


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
def build_features(closes):
    """Build ablation feature set (no raw VIX level)."""
    feat = pd.DataFrame(index=closes.index)
    vix = closes["VIX"]
    spy = closes["SPY"]
    spy_ret = spy.pct_change()

    # --- VIX term structure ---
    if "VIX3M" in closes.columns:
        vix3m = closes["VIX3M"]
        feat["vix_term_ratio"] = vix / vix3m.replace(0, np.nan)
    else:
        feat["vix_term_ratio"] = np.nan

    # --- VIX changes (NOT raw level) ---
    feat["vix_chg_5d"] = vix.pct_change(5)
    feat["vix_chg_20d"] = vix.pct_change(20)
    feat["vix_chg_1d"] = vix.pct_change(1)

    # --- Credit spread proxy: HYG vs IEF ---
    if "HYG" in closes.columns and "IEF" in closes.columns:
        hyg = closes["HYG"]
        ief = closes["IEF"]
        credit = (ief.pct_change(5) - hyg.pct_change(5))  # widening = positive
        feat["credit_spread_5d"] = credit
        feat["credit_spread_20d"] = (ief.pct_change(20) - hyg.pct_change(20))
        feat["credit_chg_5d"] = credit - credit.shift(5)
    else:
        feat["credit_spread_5d"] = np.nan
        feat["credit_spread_20d"] = np.nan
        feat["credit_chg_5d"] = np.nan

    # --- SPY features ---
    feat["spy_ret_5d"] = spy_ret.rolling(5).sum()
    feat["spy_ret_20d"] = spy_ret.rolling(20).sum()
    feat["spy_vol_10d"] = spy_ret.rolling(10).std() * np.sqrt(252)
    feat["spy_vol_20d"] = spy_ret.rolling(20).std() * np.sqrt(252)
    feat["spy_vol_ratio"] = feat["spy_vol_10d"] / feat["spy_vol_20d"].replace(0, np.nan)

    # SPY drawdown from 63d high
    spy_high_63 = spy.rolling(63).max()
    feat["spy_drawdown"] = (spy / spy_high_63) - 1.0

    # SPY MA ratio
    feat["spy_ma_ratio_50_200"] = spy.rolling(50).mean() / spy.rolling(200).mean()

    # --- Gold, Silver, Oil, Dollar, Bond momentum ---
    for col, name in [("GLD", "gold"), ("SLV", "silver"), ("USO", "oil"),
                       ("UUP", "dollar"), ("TLT", "bond")]:
        if col in closes.columns:
            p = closes[col]
            feat[f"{name}_mom_5d"] = p.pct_change(5)
            feat[f"{name}_mom_20d"] = p.pct_change(20)
        else:
            feat[f"{name}_mom_5d"] = np.nan
            feat[f"{name}_mom_20d"] = np.nan

    # --- Gold/Silver ratio ---
    if "GLD" in closes.columns and "SLV" in closes.columns:
        feat["gold_silver_ratio"] = closes["GLD"] / closes["SLV"].replace(0, np.nan)
    else:
        feat["gold_silver_ratio"] = np.nan

    # --- Copper ---
    if "CPER" in closes.columns:
        feat["copper_mom_5d"] = closes["CPER"].pct_change(5)
        feat["copper_mom_20d"] = closes["CPER"].pct_change(20)
    else:
        feat["copper_mom_5d"] = np.nan
        feat["copper_mom_20d"] = np.nan

    # --- BTC ---
    if "BTC" in closes.columns:
        feat["btc_mom_5d"] = closes["BTC"].pct_change(5)
        feat["btc_mom_20d"] = closes["BTC"].pct_change(20)
    else:
        feat["btc_mom_5d"] = np.nan
        feat["btc_mom_20d"] = np.nan

    # --- Cross-asset divergences ---
    # SPY vs gold divergence (risk-on vs risk-off)
    if "GLD" in closes.columns:
        feat["spy_gold_div_20d"] = spy.pct_change(20) - closes["GLD"].pct_change(20)

    # SPY vs bonds divergence
    if "TLT" in closes.columns:
        feat["spy_bond_div_20d"] = spy.pct_change(20) - closes["TLT"].pct_change(20)

    # Credit vs equity divergence
    if "HYG" in closes.columns:
        feat["equity_credit_div"] = spy.pct_change(20) - closes["HYG"].pct_change(20)

    return feat


def build_target(closes):
    """Target: VIX > 30 within next 5 trading days."""
    vix = closes["VIX"]
    # Forward-looking max VIX over next 5 days
    fwd_max_vix = vix.rolling(VIX_SPIKE_HORIZON, min_periods=1).max().shift(-VIX_SPIKE_HORIZON)
    target = (fwd_max_vix >= VIX_SPIKE_THRESHOLD).astype(int)
    return target


# ============================================================================
# VIX PERCENTILE SIGNAL (v4.4 base direction)
# ============================================================================
def vix_percentile_signal(closes):
    """
    VIX rolling percentile with 63-day lookback.
    <20th pctile = risk-on (UPRO)
    >80th pctile = risk-off (SHY/GLD)
    Between = neutral (reduced exposure)
    Returns: Series of allocation regime: 'risk_on', 'neutral', 'risk_off'
    """
    vix = closes["VIX"]
    pctile = vix.rolling(63).apply(
        lambda x: (x.iloc[-1] > x.iloc[:-1]).sum() / (len(x) - 1) * 100,
        raw=False
    )
    regime = pd.Series("neutral", index=closes.index)
    regime[pctile <= 20] = "risk_on"
    regime[pctile >= 80] = "risk_off"
    return regime, pctile


# ============================================================================
# BINARY DANGER SIGNALS (v4.4 / #472 comparison)
# ============================================================================
def binary_danger_signals(closes):
    """
    Binary macro danger signals for the #472 comparison strategy.
    Returns: Series of danger count (0, 1, 2+)
    """
    vix = closes["VIX"]
    spy = closes["SPY"]

    dangers = pd.DataFrame(index=closes.index)

    # 1. Credit stress: HYG underperforming IEF over 20d
    if "HYG" in closes.columns and "IEF" in closes.columns:
        dangers["credit"] = (closes["HYG"].pct_change(20) < closes["IEF"].pct_change(20)).astype(int)
    else:
        dangers["credit"] = 0

    # 2. VIX elevated: above 63d 80th percentile
    pctile = vix.rolling(63).apply(
        lambda x: (x.iloc[-1] > x.iloc[:-1]).sum() / (len(x) - 1) * 100 if len(x) > 1 else 50,
        raw=False
    )
    dangers["vix_high"] = (pctile >= 80).astype(int)

    # 3. Gold outperforming SPY (risk-off flow)
    if "GLD" in closes.columns:
        dangers["gold_flight"] = (closes["GLD"].pct_change(20) > spy.pct_change(20)).astype(int)
    else:
        dangers["gold_flight"] = 0

    # 4. Bond rally (TLT up = flight to safety)
    if "TLT" in closes.columns:
        dangers["bond_rally"] = (closes["TLT"].pct_change(20) > 0.02).astype(int)
    else:
        dangers["bond_rally"] = 0

    # 5. Dollar strength (UUP up = risk-off)
    if "UUP" in closes.columns:
        dangers["dollar_strong"] = (closes["UUP"].pct_change(20) > 0.01).astype(int)
    else:
        dangers["dollar_strong"] = 0

    danger_count = dangers.sum(axis=1)
    return danger_count


# ============================================================================
# WALK-FORWARD GBM TRAINING
# ============================================================================
def walk_forward_gbm(features, target, closes):
    """
    Sliding walk-forward GBM training.
    3-year train, 1-year test, sliding by 1 year.
    Returns: Series of OOS predicted probabilities.
    """
    # Align features and target, drop NaN rows
    valid = features.dropna().index.intersection(target.dropna().index)
    features = features.loc[valid]
    target = target.loc[valid]

    train_days = TRAIN_YEARS * 252
    test_days = TEST_YEARS * 252

    predictions = pd.Series(np.nan, index=features.index)
    fold_aucs = []

    feature_names = features.columns.tolist()
    feature_importance_sum = np.zeros(len(feature_names))
    n_folds = 0

    i = 0
    while i + train_days + test_days <= len(features):
        train_idx = features.index[i:i + train_days]
        test_idx = features.index[i + train_days:i + train_days + test_days]

        X_train = features.loc[train_idx]
        y_train = target.loc[train_idx]
        X_test = features.loc[test_idx]
        y_test = target.loc[test_idx]

        # Handle NaN in features
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)

        # Train GBM
        model = GradientBoostingClassifier(**GBM_PARAMS)
        model.fit(X_train, y_train)

        # Predict
        proba = model.predict_proba(X_test)[:, 1]
        predictions.loc[test_idx] = proba

        # AUC
        if y_test.nunique() > 1:
            auc = roc_auc_score(y_test, proba)
        else:
            auc = np.nan
        fold_aucs.append(auc)
        feature_importance_sum += model.feature_importances_
        n_folds += 1

        fold_start = test_idx[0].strftime("%Y-%m-%d")
        fold_end = test_idx[-1].strftime("%Y-%m-%d")
        print(f"  Fold {n_folds}: {fold_start} -> {fold_end}  AUC={auc:.3f}  "
              f"pos_rate={y_test.mean():.3f}  mean_p={proba.mean():.3f}")

        i += test_days  # slide by test window

    # Feature importance
    if n_folds > 0:
        avg_imp = feature_importance_sum / n_folds
        fi = pd.Series(avg_imp, index=feature_names).sort_values(ascending=False)
        print(f"\n  Top 10 features:")
        for fname, imp in fi.head(10).items():
            print(f"    {fname}: {imp:.4f}")
    else:
        fi = pd.Series(dtype=float)

    mean_auc = np.nanmean(fold_aucs)
    print(f"\n  Walk-forward complete: {n_folds} folds, mean AUC = {mean_auc:.3f}")

    return predictions, fold_aucs, fi


# ============================================================================
# STRATEGY BACKTESTS
# ============================================================================
def get_daily_returns(closes):
    """Get daily returns for all assets."""
    rets = closes[["SPY", "UPRO", "SHY", "GLD"]].pct_change()
    # Fill NaN returns with 0 for missing assets
    rets = rets.fillna(0)
    return rets


def apply_ml_scaling(ml_proba, regime):
    """
    Apply ML-based continuous scaling.
    Risk-off regime overrides to SHY regardless of ML.
    Risk-on/neutral: use ML tiers.
    """
    alloc = pd.DataFrame(0.0, index=ml_proba.index,
                         columns=["UPRO", "SPY", "SHY", "GLD"])

    for i, idx in enumerate(ml_proba.index):
        r = regime.loc[idx] if idx in regime.index else "neutral"
        p = ml_proba.loc[idx]

        if pd.isna(p):
            # No ML prediction: default to moderate
            p = 0.25

        if r == "risk_off":
            # Direction signal says risk-off: go to SHY regardless
            alloc.loc[idx, "SHY"] = 0.70
            alloc.loc[idx, "GLD"] = 0.30
        else:
            # Use ML tier for sizing
            for lo, hi, w_upro, w_spy, w_shy, w_gld in ML_TIERS:
                if lo <= p < hi:
                    alloc.loc[idx, "UPRO"] = w_upro
                    alloc.loc[idx, "SPY"] = w_spy
                    alloc.loc[idx, "SHY"] = w_shy
                    alloc.loc[idx, "GLD"] = w_gld
                    break

    return alloc


def apply_v44_baseline(regime):
    """v4.4 baseline: fixed 67/33 UPRO/SPY in risk-on, SHY in risk-off."""
    alloc = pd.DataFrame(0.0, index=regime.index,
                         columns=["UPRO", "SPY", "SHY", "GLD"])
    for idx in regime.index:
        r = regime.loc[idx]
        if r == "risk_off":
            alloc.loc[idx, "SHY"] = 0.70
            alloc.loc[idx, "GLD"] = 0.30
        else:
            alloc.loc[idx, "UPRO"] = 0.67
            alloc.loc[idx, "SPY"] = 0.33
    return alloc


def apply_binary_macro_scaling(regime, danger_count):
    """#472 binary macro-scaled: reduce leverage based on danger count."""
    alloc = pd.DataFrame(0.0, index=regime.index,
                         columns=["UPRO", "SPY", "SHY", "GLD"])
    for idx in regime.index:
        r = regime.loc[idx]
        d = danger_count.loc[idx] if idx in danger_count.index else 0

        if r == "risk_off":
            alloc.loc[idx, "SHY"] = 0.70
            alloc.loc[idx, "GLD"] = 0.30
        elif d >= 2:
            alloc.loc[idx, "SHY"] = 1.00
        elif d == 1:
            alloc.loc[idx, "UPRO"] = 0.50
            alloc.loc[idx, "SPY"] = 0.50
        else:
            alloc.loc[idx, "UPRO"] = 0.67
            alloc.loc[idx, "SPY"] = 0.33
    return alloc


def run_backtest(alloc, daily_returns, label="Strategy"):
    """
    Run backtest with next-day execution.
    Signal at close T -> trade at open T+1 (approximated as close T+1).
    """
    # Shift allocation by 1 day (next-day execution)
    alloc_shifted = alloc.shift(1)

    # Align
    common = alloc_shifted.dropna().index.intersection(daily_returns.dropna().index)
    alloc_shifted = alloc_shifted.loc[common]
    rets = daily_returns.loc[common]

    # Portfolio return = sum of weighted asset returns
    port_ret = (alloc_shifted * rets[["UPRO", "SPY", "SHY", "GLD"]]).sum(axis=1)

    # Build equity curve
    equity = INITIAL_CAPITAL * (1 + port_ret).cumprod()

    return port_ret, equity


def compute_metrics(returns, equity, label="Strategy"):
    """Compute all required performance metrics."""
    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    downside = returns[returns < 0].std() * np.sqrt(252)
    sortino = ann_ret / downside if downside > 0 else 0

    total_ret = equity.iloc[-1] / equity.iloc[0] - 1
    years = len(returns) / 252
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else 0

    rolling_max = equity.cummax()
    drawdown = (equity / rolling_max) - 1
    max_dd = drawdown.min()

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Win rate (daily)
    wr = (returns > 0).mean()

    # Trades per year (approximate: count days where allocation changes)
    trades_per_year = 252  # approximate, always invested

    return {
        "label": label,
        "sharpe": sharpe,
        "sortino": sortino,
        "cagr": cagr,
        "max_dd": max_dd,
        "calmar": calmar,
        "win_rate": wr,
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "total_return": total_ret,
        "years": years,
    }


# ============================================================================
# ADVERSARIAL VALIDATION
# ============================================================================
def permutation_test(ml_proba, regime, daily_returns, real_sharpe, closes):
    """
    Shuffle ML predictions, recalculate strategy returns.
    If real > 95% of permuted -> ML adds value.
    """
    print(f"\nPermutation test ({N_PERMUTATIONS} shuffles)...")
    perm_sharpes = []

    for i in range(N_PERMUTATIONS):
        # Shuffle ML predictions
        shuffled_p = ml_proba.copy()
        valid_mask = ~shuffled_p.isna()
        vals = shuffled_p[valid_mask].values.copy()
        np.random.shuffle(vals)
        shuffled_p[valid_mask] = vals

        # Run strategy with shuffled predictions
        alloc = apply_ml_scaling(shuffled_p, regime)
        rets_daily = get_daily_returns(closes)
        port_ret, equity = run_backtest(alloc, rets_daily, f"Perm_{i}")

        # Filter to backtest period
        bt_mask = port_ret.index >= BACKTEST_START
        port_ret_bt = port_ret[bt_mask]

        if len(port_ret_bt) > 0:
            ann_ret = port_ret_bt.mean() * 252
            ann_vol = port_ret_bt.std() * np.sqrt(252)
            s = ann_ret / ann_vol if ann_vol > 0 else 0
        else:
            s = 0
        perm_sharpes.append(s)

    perm_sharpes = np.array(perm_sharpes)
    p_value = (perm_sharpes >= real_sharpe).mean()
    print(f"  Real Sharpe: {real_sharpe:.3f}")
    print(f"  Permuted Sharpe: mean={perm_sharpes.mean():.3f}, "
          f"std={perm_sharpes.std():.3f}, max={perm_sharpes.max():.3f}")
    print(f"  p-value: {p_value:.3f} ({'PASS' if p_value < PERM_P_THRESHOLD else 'FAIL'})")

    return p_value, perm_sharpes


def sub_period_consistency(returns, n_blocks=3):
    """Split into n equal blocks, check CV of Sharpe < threshold."""
    block_size = len(returns) // n_blocks
    block_sharpes = []

    for b in range(n_blocks):
        start = b * block_size
        end = (b + 1) * block_size if b < n_blocks - 1 else len(returns)
        block_ret = returns.iloc[start:end]

        ann_ret = block_ret.mean() * 252
        ann_vol = block_ret.std() * np.sqrt(252)
        s = ann_ret / ann_vol if ann_vol > 0 else 0
        block_sharpes.append(s)
        period_start = returns.index[start].strftime("%Y-%m-%d")
        period_end = returns.index[min(end - 1, len(returns) - 1)].strftime("%Y-%m-%d")
        print(f"  Block {b + 1} ({period_start} to {period_end}): Sharpe = {s:.3f}")

    cv = np.std(block_sharpes) / abs(np.mean(block_sharpes)) if np.mean(block_sharpes) != 0 else 999
    passed = cv < CV_SHARPE_THRESHOLD
    print(f"  CV of Sharpe: {cv:.3f} ({'PASS' if passed else 'FAIL'}, threshold={CV_SHARPE_THRESHOLD})")
    return cv, block_sharpes, passed


def walk_forward_excess_check(ml_returns, spy_returns):
    """Check that all OOS windows have positive excess vs SPY."""
    # Split into yearly windows
    years = sorted(ml_returns.index.year.unique())
    all_positive = True
    print(f"\nWalk-forward excess vs SPY by year:")
    for yr in years:
        mask = ml_returns.index.year == yr
        if mask.sum() < 20:
            continue
        ml_yr = ml_returns[mask]
        spy_yr = spy_returns.reindex(ml_yr.index).fillna(0)
        excess = (ml_yr.mean() - spy_yr.mean()) * 252
        status = "+" if excess > 0 else "NEGATIVE"
        if excess <= 0:
            all_positive = False
        print(f"  {yr}: excess ann return = {excess:+.4f} ({status})")

    print(f"  All windows positive: {'PASS' if all_positive else 'FAIL'}")
    return all_positive


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 80)
    print("ML MACRO RISK SCALER BACKTEST")
    print("=" * 80)

    # --- Load data ---
    closes = load_data()

    # --- Build features and target ---
    print("\nBuilding features...")
    features = build_features(closes)
    target = build_target(closes)

    print(f"  Features: {features.shape[1]} columns")
    print(f"  Target positive rate: {target.mean():.3f}")

    # --- Walk-forward GBM ---
    print("\n" + "=" * 60)
    print("WALK-FORWARD GBM TRAINING")
    print("=" * 60)
    ml_proba, fold_aucs, feature_importance = walk_forward_gbm(features, target, closes)

    # --- Direction signal ---
    regime, vix_pctile = vix_percentile_signal(closes)
    danger_count = binary_danger_signals(closes)

    # --- Daily returns ---
    daily_rets = get_daily_returns(closes)

    # --- Filter to backtest period ---
    bt_mask = closes.index >= BACKTEST_START

    # ========================================================
    # STRATEGY A: ML Macro Risk Scaler (this strategy)
    # ========================================================
    print("\n" + "=" * 60)
    print("STRATEGY A: ML Macro Risk Scaler")
    print("=" * 60)
    alloc_ml = apply_ml_scaling(ml_proba, regime)
    ret_ml, eq_ml = run_backtest(alloc_ml, daily_rets, "ML Risk Scaler")
    ret_ml_bt = ret_ml[ret_ml.index >= BACKTEST_START]
    eq_ml_bt = INITIAL_CAPITAL * (1 + ret_ml_bt).cumprod()
    metrics_ml = compute_metrics(ret_ml_bt, eq_ml_bt, "ML Risk Scaler")

    # ========================================================
    # STRATEGY B: v4.4 Baseline (fixed 67/33, no ML)
    # ========================================================
    print("\n" + "=" * 60)
    print("STRATEGY B: v4.4 Baseline (fixed 67/33)")
    print("=" * 60)
    alloc_v44 = apply_v44_baseline(regime)
    ret_v44, eq_v44 = run_backtest(alloc_v44, daily_rets, "v4.4 Baseline")
    ret_v44_bt = ret_v44[ret_v44.index >= BACKTEST_START]
    eq_v44_bt = INITIAL_CAPITAL * (1 + ret_v44_bt).cumprod()
    metrics_v44 = compute_metrics(ret_v44_bt, eq_v44_bt, "v4.4 Baseline")

    # ========================================================
    # STRATEGY C: Binary Macro-Scaled (#472)
    # ========================================================
    print("\n" + "=" * 60)
    print("STRATEGY C: Binary Macro-Scaled (#472)")
    print("=" * 60)
    alloc_bin = apply_binary_macro_scaling(regime, danger_count)
    ret_bin, eq_bin = run_backtest(alloc_bin, daily_rets, "Binary Macro-Scaled")
    ret_bin_bt = ret_bin[ret_bin.index >= BACKTEST_START]
    eq_bin_bt = INITIAL_CAPITAL * (1 + ret_bin_bt).cumprod()
    metrics_bin = compute_metrics(ret_bin_bt, eq_bin_bt, "Binary Macro-Scaled")

    # ========================================================
    # BENCHMARK: SPY Buy & Hold
    # ========================================================
    print("\n" + "=" * 60)
    print("BENCHMARK: SPY Buy & Hold")
    print("=" * 60)
    spy_ret = closes["SPY"].pct_change()
    spy_ret_bt = spy_ret[spy_ret.index >= BACKTEST_START].fillna(0)
    eq_spy_bt = INITIAL_CAPITAL * (1 + spy_ret_bt).cumprod()
    metrics_spy = compute_metrics(spy_ret_bt, eq_spy_bt, "SPY Buy & Hold")

    # ========================================================
    # RESULTS TABLE
    # ========================================================
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)

    all_metrics = [metrics_ml, metrics_v44, metrics_bin, metrics_spy]
    header = f"{'Strategy':<25} {'Sharpe':>7} {'Sortino':>8} {'CAGR':>7} {'MaxDD':>8} {'Calmar':>7} {'WR':>6} {'AnnRet':>8} {'AnnVol':>8}"
    print(header)
    print("-" * len(header))
    for m in all_metrics:
        print(f"{m['label']:<25} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} "
              f"{m['cagr']:>6.1%} {m['max_dd']:>7.1%} {m['calmar']:>7.3f} "
              f"{m['win_rate']:>5.1%} {m['ann_return']:>7.1%} {m['ann_vol']:>7.1%}")

    # ========================================================
    # ADVERSARIAL VALIDATION
    # ========================================================
    print("\n" + "=" * 80)
    print("ADVERSARIAL VALIDATION")
    print("=" * 80)

    # 1. Permutation test
    perm_p, perm_sharpes = permutation_test(
        ml_proba, regime, daily_rets, metrics_ml["sharpe"], closes
    )

    # 2. Sub-period consistency
    print(f"\nSub-period consistency (3 blocks):")
    cv_sharpe, block_sharpes, cv_pass = sub_period_consistency(ret_ml_bt)

    # 3. Walk-forward excess
    wf_pass = walk_forward_excess_check(ret_ml_bt, spy_ret_bt)

    # 4. ML vs random check
    print(f"\n--- ML Value-Add Check ---")
    ml_excess_sharpe = metrics_ml["sharpe"] - metrics_v44["sharpe"]
    ml_excess_cagr = metrics_ml["cagr"] - metrics_v44["cagr"]
    ml_excess_dd = metrics_ml["max_dd"] - metrics_v44["max_dd"]  # less negative = better
    print(f"  ML vs v4.4 baseline: Sharpe diff = {ml_excess_sharpe:+.3f}, "
          f"CAGR diff = {ml_excess_cagr:+.1%}, MaxDD diff = {ml_excess_dd:+.1%}")

    if perm_p >= PERM_P_THRESHOLD:
        print(f"\n  *** REJECT: Permutation test FAILED (p={perm_p:.3f} >= {PERM_P_THRESHOLD})")
        print(f"  *** Random ML predictions produce similar returns -> ML adds no value")
        verdict = "REJECT"
    elif not cv_pass:
        print(f"\n  *** REJECT: Sub-period consistency FAILED (CV={cv_sharpe:.3f})")
        verdict = "REJECT"
    elif ml_excess_sharpe <= 0:
        print(f"\n  *** REJECT: ML does not improve Sharpe over v4.4 baseline")
        verdict = "REJECT"
    else:
        print(f"\n  *** PASS: ML adds genuine value over baseline")
        verdict = "PASS"

    # ========================================================
    # ALLOCATION STATISTICS
    # ========================================================
    print("\n" + "=" * 80)
    print("ALLOCATION STATISTICS (ML Strategy)")
    print("=" * 80)
    alloc_bt = alloc_ml[alloc_ml.index >= BACKTEST_START]
    for col in ["UPRO", "SPY", "SHY", "GLD"]:
        mean_w = alloc_bt[col].mean()
        print(f"  Mean {col} weight: {mean_w:.1%}")

    # Tier distribution
    ml_proba_bt = ml_proba[ml_proba.index >= BACKTEST_START].dropna()
    if len(ml_proba_bt) > 0:
        print(f"\n  ML probability distribution:")
        print(f"    Low risk (p<0.15):    {(ml_proba_bt < 0.15).mean():.1%} of days")
        print(f"    Moderate (0.15-0.35): {((ml_proba_bt >= 0.15) & (ml_proba_bt < 0.35)).mean():.1%}")
        print(f"    High (0.35-0.60):     {((ml_proba_bt >= 0.35) & (ml_proba_bt < 0.60)).mean():.1%}")
        print(f"    Extreme (>0.60):      {(ml_proba_bt >= 0.60).mean():.1%}")

    # ========================================================
    # SAVE OUTPUTS
    # ========================================================
    print("\n" + "=" * 80)
    print("SAVING OUTPUTS")
    print("=" * 80)

    # Save equity curves
    eq_df = pd.DataFrame({
        "ML_Risk_Scaler": eq_ml_bt,
        "v44_Baseline": eq_v44_bt,
        "Binary_Macro": eq_bin_bt,
        "SPY_BH": eq_spy_bt,
    })
    eq_df.to_csv(OUTPUT_DIR / "equity_curves.csv")
    print(f"  Saved equity curves to {OUTPUT_DIR / 'equity_curves.csv'}")

    # Save ML predictions
    pred_df = pd.DataFrame({
        "ml_proba": ml_proba,
        "regime": regime,
        "vix_pctile": vix_pctile,
    })
    pred_df.to_csv(OUTPUT_DIR / "ml_predictions.csv")

    # Save results summary
    results = {
        "run_date": datetime.now().isoformat(),
        "backtest_start": BACKTEST_START,
        "backtest_end": str(closes.index[-1].date()),
        "strategies": {m["label"]: {k: float(v) if isinstance(v, (np.floating, float)) else v
                                     for k, v in m.items()} for m in all_metrics},
        "adversarial": {
            "permutation_p_value": float(perm_p),
            "permutation_pass": perm_p < PERM_P_THRESHOLD,
            "subperiod_cv": float(cv_sharpe),
            "subperiod_pass": cv_pass,
            "walkforward_excess_pass": wf_pass,
            "verdict": verdict,
        },
        "gbm": {
            "mean_oos_auc": float(np.nanmean(fold_aucs)),
            "fold_aucs": [float(a) for a in fold_aucs],
            "n_folds": len(fold_aucs),
            "top_features": feature_importance.head(10).to_dict() if len(feature_importance) > 0 else {},
        },
        "ml_vs_baseline": {
            "sharpe_diff": float(ml_excess_sharpe),
            "cagr_diff": float(ml_excess_cagr),
            "maxdd_diff": float(ml_excess_dd),
        },
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved results to {OUTPUT_DIR / 'results.json'}")

    # Save feature importance
    if len(feature_importance) > 0:
        feature_importance.to_csv(OUTPUT_DIR / "feature_importance.csv")

    # ========================================================
    # FINAL VERDICT
    # ========================================================
    print("\n" + "=" * 80)
    print(f"FINAL VERDICT: {verdict}")
    print("=" * 80)
    if verdict == "PASS":
        print("ML Macro Risk Scaler adds genuine value over the v4.4 baseline.")
        print(f"Sharpe improvement: {ml_excess_sharpe:+.3f}")
        print(f"MaxDD improvement: {ml_excess_dd:+.1%}")
    else:
        print("ML Macro Risk Scaler does NOT reliably improve over simpler approaches.")
        print("Recommendation: stick with v4.4 binary macro-scaled (#472).")

    return results


if __name__ == "__main__":
    results = main()
