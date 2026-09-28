#!/usr/bin/env python3
"""
Drawdown Predictor v1 — HC #0 compliant sliding walk-forward
Goal: predict whether SPY will have a max drawdown > 5% in next 30 trading days.
Use as a "risk switch" for income / wheel / options income strategies.

Features: VIX level + term structure, SPY momentum, credit spreads,
          realized vs implied vol gap, yield curve slope, put/call ratio proxy
Model:    LGBM + XGBoost ensemble (soft-voted probabilities)
Validation: permutation test (100 shuffles), regime-stratified AUC (HC #428 R1)
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import mlflow
import mlflow.sklearn
import json
from datetime import datetime, timedelta
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)
from sklearn.utils import shuffle as sk_shuffle
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
OUTPUT_DIR = "/home/nick/Lvl3Quant/output/drawdown_predictor_v1"
MLFLOW_URI = "http://jupiter:5000"   # Jupiter MLflow
EXPERIMENT_NAME = "drawdown_predictor_v1"

TRAIN_DAYS   = 252          # sliding window train size
TEST_DAYS    = 63           # OOT window per fold
STEP_DAYS    = 21           # step between folds (quarterly steps)
HORIZON      = 30           # predict N-day forward drawdown
DD_THRESH    = 0.05         # 5% drawdown threshold
PERM_ITERS   = 100          # permutation test shuffles
START_DATE   = "2003-01-01" # capture 2003+ data → test folds from ~2005
END_DATE     = datetime.today().strftime("%Y-%m-%d")

TICKERS = {
    "spy":  "SPY",
    "vix":  "^VIX",
    "vix3m":"^VIX3M",
    "hyg":  "HYG",
    "tlt":  "TLT",
    "ief":  "IEF",
    "lqd":  "LQD",
    "tnx":  "^TNX",   # 10Y yield
    "irx":  "^IRX",   # 3M yield (13-week T-bill)
    "spx":  "^GSPC",  # SPX for cross-check
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ──────────────────────────────────────────────
# DATA DOWNLOAD
# ──────────────────────────────────────────────
def download_data():
    print("[DATA] Downloading price history …")
    raw = {}
    for key, ticker in TICKERS.items():
        try:
            df = yf.download(ticker, start=START_DATE, end=END_DATE,
                             auto_adjust=True, progress=False)
            if len(df) > 100:
                raw[key] = df["Close"].squeeze()
                print(f"  {ticker}: {len(df)} rows  ({df.index[0].date()} – {df.index[-1].date()})")
            else:
                print(f"  {ticker}: SKIPPED (too short: {len(df)} rows)")
        except Exception as e:
            print(f"  {ticker}: FAILED ({e})")
    return raw


# ──────────────────────────────────────────────
# FEATURE ENGINEERING
# ──────────────────────────────────────────────
def build_features(raw: dict) -> pd.DataFrame:
    spy   = raw["spy"]
    vix   = raw.get("vix")
    vix3m = raw.get("vix3m")
    hyg   = raw.get("hyg")
    tlt   = raw.get("tlt")
    ief   = raw.get("ief")
    lqd   = raw.get("lqd")
    tnx   = raw.get("tnx")
    irx   = raw.get("irx")

    # Align all series to SPY trading days
    idx = spy.index
    f = pd.DataFrame(index=idx)

    # ── SPY momentum ──
    for w in [1, 5, 10, 21, 63, 126, 252]:
        f[f"spy_ret_{w}d"] = spy.pct_change(w)

    # ── SPY realized volatility ──
    log_ret = np.log(spy / spy.shift(1))
    for w in [5, 10, 21, 63]:
        f[f"spy_rvol_{w}d"] = log_ret.rolling(w).std() * np.sqrt(252)

    # ── Drawdown from rolling peak (how far into drawdown now?) ──
    roll_peak_63  = spy.rolling(63).max()
    roll_peak_252 = spy.rolling(252).max()
    f["spy_dd_63d"]  = (spy / roll_peak_63) - 1
    f["spy_dd_252d"] = (spy / roll_peak_252) - 1

    # ── SPY distance from 200d MA ──
    ma200 = spy.rolling(200).mean()
    f["spy_vs_ma200"] = (spy / ma200) - 1

    # ── VIX features ──
    if vix is not None:
        v = vix.reindex(idx, method="ffill")
        f["vix_level"]       = v
        f["vix_log"]         = np.log(v + 1e-6)
        f["vix_change_5d"]   = v.pct_change(5)
        f["vix_change_21d"]  = v.pct_change(21)
        f["vix_vs_ma20"]     = (v / v.rolling(20).mean()) - 1
        f["vix_vs_ma63"]     = (v / v.rolling(63).mean()) - 1
        # VIX regime bucketing
        f["vix_low"]    = (v < 15).astype(int)
        f["vix_medium"] = ((v >= 15) & (v < 25)).astype(int)
        f["vix_high"]   = ((v >= 25) & (v < 40)).astype(int)
        f["vix_crisis"] = (v >= 40).astype(int)

        if vix3m is not None:
            v3 = vix3m.reindex(idx, method="ffill")
            f["vix_term_ratio"]   = v / (v3 + 1e-6)    # >1 = backwardation (fear)
            f["vix3m_level"]      = v3
            f["vix3m_change_5d"]  = v3.pct_change(5)
            f["vix3m_change_21d"] = v3.pct_change(21)

        # VIX vs realized vol gap (implied - realized)
        if "spy_rvol_21d" in f.columns:
            f["iv_rv_gap_21d"] = v / 100 - f["spy_rvol_21d"]

    # ── Credit spreads ──
    # HYG/TLT ratio as junk credit proxy (lower = wider spreads)
    if hyg is not None and tlt is not None:
        h = hyg.reindex(idx, method="ffill")
        t = tlt.reindex(idx, method="ffill")
        ratio = h / t
        f["hyg_tlt_ratio"]         = ratio
        f["hyg_tlt_change_5d"]     = ratio.pct_change(5)
        f["hyg_tlt_change_21d"]    = ratio.pct_change(21)
        f["hyg_tlt_vs_ma63"]       = (ratio / ratio.rolling(63).mean()) - 1

    # LQD/TLT ratio as IG credit proxy
    if lqd is not None and tlt is not None:
        l = lqd.reindex(idx, method="ffill")
        t = tlt.reindex(idx, method="ffill")
        ratio_ig = l / t
        f["lqd_tlt_ratio"]      = ratio_ig
        f["lqd_tlt_change_21d"] = ratio_ig.pct_change(21)

    # ── Yield curve ──
    if tnx is not None:
        t10 = tnx.reindex(idx, method="ffill")
        f["yield_10y"]        = t10
        f["yield_10y_chg_5d"] = t10.diff(5)
        f["yield_10y_chg_21d"]= t10.diff(21)

        if irx is not None:
            t3m = irx.reindex(idx, method="ffill")
            f["yield_curve_slope"]     = t10 - t3m      # negative = inverted
            f["yield_curve_chg_21d"]   = (t10 - t3m).diff(21)
            f["yield_inverted"]        = (t10 < t3m).astype(int)

    # ── TLT momentum (flight-to-quality signal) ──
    if tlt is not None:
        t = tlt.reindex(idx, method="ffill")
        f["tlt_ret_5d"]  = t.pct_change(5)
        f["tlt_ret_21d"] = t.pct_change(21)
        f["tlt_ret_63d"] = t.pct_change(63)

    # ── Cross-asset momentum divergence ──
    # SPY vs TLT 21d momentum divergence (risk-off signal when TLT > SPY)
    if tlt is not None:
        t = tlt.reindex(idx, method="ffill")
        spy_21 = spy.pct_change(21)
        tlt_21 = t.pct_change(21)
        f["spy_vs_tlt_21d_mom"] = spy_21 - tlt_21

    # ── Skew proxy: rolling asymmetry of daily returns ──
    for w in [21, 63]:
        f[f"spy_skew_{w}d"]  = log_ret.rolling(w).skew()
        f[f"spy_kurt_{w}d"]  = log_ret.rolling(w).kurt()

    # ── Regime: SPY trend direction ──
    f["spy_above_ma50"]  = (spy > spy.rolling(50).mean()).astype(int)
    f["spy_above_ma200"] = (spy > spy.rolling(200).mean()).astype(int)

    # ── Time-of-year features (seasonality) ──
    f["month"]          = idx.month
    f["quarter"]        = idx.quarter
    f["is_q4"]          = (idx.quarter == 4).astype(int)
    f["is_sept_oct"]    = idx.month.isin([9, 10]).astype(int)

    return f


# ──────────────────────────────────────────────
# TARGET VARIABLE
# ──────────────────────────────────────────────
def build_target(spy: pd.Series, horizon: int = HORIZON, thresh: float = DD_THRESH) -> pd.Series:
    """
    For each day t, look forward `horizon` trading days.
    Return 1 if max drawdown from t's close exceeds thresh.
    Max drawdown from t = min(spy[t+1:t+horizon+1]) / spy[t] - 1
    """
    targets = []
    prices  = spy.values
    dates   = spy.index

    for i in range(len(prices)):
        end = i + horizon + 1
        if end > len(prices):
            targets.append(np.nan)
        else:
            fwd_prices = prices[i+1:end]
            peak_price = prices[i]
            min_fwd    = np.min(fwd_prices)
            drawdown   = (min_fwd / peak_price) - 1
            targets.append(1 if drawdown <= -thresh else 0)

    return pd.Series(targets, index=dates, name="target")


# ──────────────────────────────────────────────
# MODEL BUILDERS
# ──────────────────────────────────────────────
def make_lgbm(scale_pos_weight=None):
    params = dict(
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=4,
        verbose=-1,
    )
    if scale_pos_weight:
        params["scale_pos_weight"] = scale_pos_weight
    return lgb.LGBMClassifier(**params)


def make_xgb(scale_pos_weight=None):
    params = dict(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=4,
        verbosity=0,
    )
    if scale_pos_weight:
        params["scale_pos_weight"] = scale_pos_weight
    return xgb.XGBClassifier(**params)


def ensemble_predict(lgbm_model, xgb_model, X):
    p_lgbm = lgbm_model.predict_proba(X)[:, 1]
    p_xgb  = xgb_model.predict_proba(X)[:, 1]
    return 0.5 * p_lgbm + 0.5 * p_xgb


# ──────────────────────────────────────────────
# WALK-FORWARD (SLIDING WINDOW)
# ──────────────────────────────────────────────
def walk_forward(X: pd.DataFrame, y: pd.Series):
    """
    HC #0 compliant: sliding window (oldest day dropped as new added).
    TRAIN_DAYS train, TEST_DAYS test, STEP_DAYS between folds.
    Returns list of fold result dicts.
    """
    dates = X.index
    n     = len(dates)
    folds = []

    start_idx = TRAIN_DAYS  # first fold train ends at TRAIN_DAYS
    fold_num  = 0

    while True:
        train_end = start_idx
        test_end  = train_end + TEST_DAYS

        if test_end > n:
            break

        # SLIDING: train window starts at train_end - TRAIN_DAYS
        train_start = train_end - TRAIN_DAYS

        X_train = X.iloc[train_start:train_end]
        y_train = y.iloc[train_start:train_end]
        X_test  = X.iloc[train_end:test_end]
        y_test  = y.iloc[train_end:test_end]

        # Drop NaN rows
        mask_train = ~(X_train.isna().any(axis=1) | y_train.isna())
        mask_test  = ~(X_test.isna().any(axis=1)  | y_test.isna())

        if mask_train.sum() < 50 or mask_test.sum() < 5:
            start_idx += STEP_DAYS
            fold_num  += 1
            continue

        Xtr = X_train[mask_train]
        ytr = y_train[mask_train]
        Xte = X_test[mask_test]
        yte = y_test[mask_test]

        if ytr.nunique() < 2 or yte.nunique() < 2:
            start_idx += STEP_DAYS
            fold_num  += 1
            continue

        spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
        spw = min(max(spw, 1.0), 5.0)  # cap at 5x

        lgbm_m = make_lgbm(scale_pos_weight=spw)
        xgb_m  = make_xgb(scale_pos_weight=spw)
        lgbm_m.fit(Xtr, ytr)
        xgb_m.fit(Xtr, ytr)

        probs = ensemble_predict(lgbm_m, xgb_m, Xte)
        auc   = roc_auc_score(yte, probs)
        ap    = average_precision_score(yte, probs)

        # Regime: green/red/flat based on SPY return over test window
        spy_ret_test = X_test["spy_ret_21d"].iloc[mask_test.values].mean() if "spy_ret_21d" in X_test.columns else 0.0
        if spy_ret_test > 0.01:
            regime = "green"
        elif spy_ret_test < -0.01:
            regime = "red"
        else:
            regime = "flat"

        fold_result = {
            "fold": fold_num,
            "train_start": str(dates[train_start].date()),
            "train_end":   str(dates[train_end - 1].date()),
            "test_start":  str(dates[train_end].date()),
            "test_end":    str(dates[test_end - 1].date()),
            "n_train":     int(mask_train.sum()),
            "n_test":      int(mask_test.sum()),
            "pos_rate_train": float(ytr.mean()),
            "pos_rate_test":  float(yte.mean()),
            "auc":         float(auc),
            "avg_precision": float(ap),
            "regime":      regime,
            "probs":       probs.tolist(),
            "labels":      yte.values.tolist(),
            "dates_test":  [str(d.date()) for d in yte.index],
            "lgbm_model":  lgbm_m,
            "xgb_model":   xgb_m,
            "feature_names": list(Xtr.columns),
        }
        folds.append(fold_result)

        print(f"  Fold {fold_num:3d} | {dates[train_end].date()} – {dates[test_end-1].date()} | "
              f"AUC={auc:.3f} AP={ap:.3f} | regime={regime} pos_rate={yte.mean():.2f}")

        start_idx += STEP_DAYS
        fold_num  += 1

    return folds


# ──────────────────────────────────────────────
# PERMUTATION TEST
# ──────────────────────────────────────────────
def permutation_test(folds, n_iters=PERM_ITERS):
    """
    Shuffle labels within each fold's test set, recompute AUC.
    p-value = fraction of permuted AUCs >= observed concat AUC.
    """
    print(f"\n[PERM] Running {n_iters} permutation shuffles …")

    # Gather all OOT labels + probs
    all_labels = []
    all_probs  = []
    for fold in folds:
        all_labels.extend(fold["labels"])
        all_probs.extend(fold["probs"])

    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    if len(np.unique(all_labels)) < 2:
        print("  Cannot run perm test — only one class in concat OOT labels")
        return {"p_value": None, "observed_auc": None, "n_iters": n_iters}

    observed_auc = roc_auc_score(all_labels, all_probs)
    perm_aucs    = []

    for i in range(n_iters):
        shuffled = sk_shuffle(all_labels, random_state=i)
        try:
            perm_aucs.append(roc_auc_score(shuffled, all_probs))
        except Exception:
            pass

    perm_aucs = np.array(perm_aucs)
    p_value   = (perm_aucs >= observed_auc).mean()

    print(f"  Observed AUC = {observed_auc:.4f}")
    print(f"  Perm AUC mean = {perm_aucs.mean():.4f} ± {perm_aucs.std():.4f}")
    print(f"  p-value = {p_value:.4f}  ({(perm_aucs >= observed_auc).sum()}/{n_iters} shuffles beat observed)")

    return {
        "p_value":        float(p_value),
        "observed_auc":   float(observed_auc),
        "perm_auc_mean":  float(perm_aucs.mean()),
        "perm_auc_std":   float(perm_aucs.std()),
        "n_iters":        n_iters,
    }


# ──────────────────────────────────────────────
# REGIME TEST (HC #428 R1)
# ──────────────────────────────────────────────
def regime_test(folds):
    """
    HC #428 R1: stratified AUC by regime.
    Reject if |AUC_green - AUC_red| / max(|AUC_green|, |AUC_red|) > 0.50
    """
    print("\n[REGIME] Running HC #428 R1 regime stratification test …")

    regime_data = {"green": ([], []), "red": ([], []), "flat": ([], [])}
    for fold in folds:
        r = fold["regime"]
        if r in regime_data:
            regime_data[r][0].extend(fold["labels"])
            regime_data[r][1].extend(fold["probs"])

    regime_aucs = {}
    for regime, (labels, probs) in regime_data.items():
        labels = np.array(labels)
        probs  = np.array(probs)
        if len(np.unique(labels)) < 2 or len(labels) < 10:
            print(f"  {regime}: insufficient data ({len(labels)} samples)")
            continue
        auc = roc_auc_score(labels, probs)
        regime_aucs[regime] = float(auc)
        print(f"  {regime}: AUC={auc:.4f} (n={len(labels)}, pos_rate={labels.mean():.2f})")

    # HC #428 R1 gate
    passed = True
    gap    = None
    if "green" in regime_aucs and "red" in regime_aucs:
        auc_g = regime_aucs["green"]
        auc_r = regime_aucs["red"]
        gap   = abs(auc_g - auc_r) / max(abs(auc_g), abs(auc_r))
        passed = gap <= 0.50
        print(f"\n  HC #428 R1 gate: |AUC_green - AUC_red| / max = {gap:.4f}")
        print(f"  Threshold = 0.50 → {'PASS' if passed else 'FAIL'}")
    else:
        print("  Regime gate: skipped (missing green or red regime data)")

    return {
        "regime_aucs": regime_aucs,
        "gap":         gap,
        "hc428_passed": passed,
    }


# ──────────────────────────────────────────────
# THRESHOLD ANALYSIS
# ──────────────────────────────────────────────
def threshold_analysis(folds, thresholds=None):
    """
    Precision / recall / FPR at various confidence thresholds.
    We care more about recall (catching drawdowns) than precision.
    """
    if thresholds is None:
        thresholds = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

    all_labels = []
    all_probs  = []
    for fold in folds:
        all_labels.extend(fold["labels"])
        all_probs.extend(fold["probs"])

    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    results = []
    print("\n[THRESH] Precision / Recall / FPR at confidence thresholds:")
    print(f"  {'Threshold':>10} | {'Precision':>10} | {'Recall':>10} | {'FPR':>10} | {'F1':>10} | {'Coverage':>10}")
    print("  " + "-"*72)

    for t in thresholds:
        preds = (all_probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(all_labels, preds, labels=[0, 1]).ravel() \
            if len(np.unique(preds)) > 1 else (int((all_labels==0).sum()), 0, int((all_labels==1).sum()), 0)

        total_pos = tp + fn
        total_neg = tn + fp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        coverage  = preds.mean()

        print(f"  {t:>10.2f} | {precision:>10.3f} | {recall:>10.3f} | {fpr:>10.3f} | {f1:>10.3f} | {coverage:>10.3f}")

        results.append({
            "threshold": t,
            "precision": float(precision),
            "recall":    float(recall),
            "fpr":       float(fpr),
            "f1":        float(f1),
            "coverage":  float(coverage),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        })

    # Baseline: predict=1 always
    baseline_recall    = float(all_labels.mean())
    baseline_precision = float(all_labels.mean())
    print(f"\n  Baseline (always predict drawdown): precision={baseline_precision:.3f} recall=1.000")
    print(f"  Positive rate (actual drawdown days): {all_labels.mean():.3f}")

    return results


# ──────────────────────────────────────────────
# FEATURE IMPORTANCE
# ──────────────────────────────────────────────
def aggregate_feature_importance(folds, top_n=20):
    """Average LGBM gain importance across last 10 folds."""
    imp_agg = {}
    recent_folds = folds[-10:]

    for fold in recent_folds:
        model = fold.get("lgbm_model")
        names = fold.get("feature_names", [])
        if model is None:
            continue
        imp = model.feature_importances_
        for name, val in zip(names, imp):
            imp_agg[name] = imp_agg.get(name, 0.0) + float(val)

    if not imp_agg:
        return {}

    n = len(recent_folds)
    imp_avg = {k: v / n for k, v in imp_agg.items()}
    sorted_imp = sorted(imp_avg.items(), key=lambda x: -x[1])

    print(f"\n[IMPORTANCE] Top {top_n} features (avg LGBM gain, last 10 folds):")
    for rank, (name, val) in enumerate(sorted_imp[:top_n], 1):
        print(f"  {rank:2d}. {name:<35} {val:.1f}")

    return dict(sorted_imp)


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  DRAWDOWN PREDICTOR v1 — HC #0 Sliding Walk-Forward")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Config: TRAIN={TRAIN_DAYS}d, TEST={TEST_DAYS}d, STEP={STEP_DAYS}d")
    print(f"  Target: SPY max drawdown > {DD_THRESH*100:.0f}% in next {HORIZON}d")
    print("=" * 70)

    # ── 1. Data ──
    raw = download_data()
    if "spy" not in raw:
        print("FATAL: could not download SPY data")
        sys.exit(1)

    # ── 2. Features + Target ──
    print("\n[FEAT] Building features …")
    X = build_features(raw)
    y = build_target(raw["spy"])
    print(f"  Feature matrix: {X.shape}, date range {X.index[0].date()} – {X.index[-1].date()}")
    print(f"  Positive rate (all): {y.mean():.3f}  ({int(y.sum())} / {int(y.notna().sum())} days)")

    # Align X and y (drop rows where target is NaN — last HORIZON rows)
    valid = y.notna() & ~X.isna().all(axis=1)
    X = X[valid]
    y = y[valid]

    # ── 3. Walk-forward ──
    print(f"\n[WF] Walk-forward (sliding, {TRAIN_DAYS}d train, {TEST_DAYS}d test, step {STEP_DAYS}d) …")
    folds = walk_forward(X, y)
    print(f"\n  Completed {len(folds)} folds")

    if not folds:
        print("FATAL: no folds completed")
        sys.exit(1)

    # ── 4. Concat OOT metrics ──
    all_auc = [f["auc"] for f in folds]
    all_ap  = [f["avg_precision"] for f in folds]
    concat_auc = roc_auc_score(
        [l for f in folds for l in f["labels"]],
        [p for f in folds for p in f["probs"]]
    )
    concat_ap  = average_precision_score(
        [l for f in folds for l in f["labels"]],
        [p for f in folds for p in f["probs"]]
    )
    print(f"\n[RESULTS] Concat OOT AUC={concat_auc:.4f}  AP={concat_ap:.4f}")
    print(f"          Per-fold AUC: mean={np.mean(all_auc):.4f} ± {np.std(all_auc):.4f}")

    # ── 5. Threshold analysis ──
    thresh_results = threshold_analysis(folds)

    # ── 6. Permutation test ──
    perm_results = permutation_test(folds)

    # ── 7. Regime test (HC #428 R1) ──
    regime_results = regime_test(folds)

    # ── 8. Feature importance ──
    feat_imp = aggregate_feature_importance(folds)

    # ── 9. Save results ──
    summary = {
        "run_date":         datetime.now().isoformat(),
        "config": {
            "train_days":   TRAIN_DAYS,
            "test_days":    TEST_DAYS,
            "step_days":    STEP_DAYS,
            "horizon":      HORIZON,
            "dd_thresh":    DD_THRESH,
            "start_date":   START_DATE,
            "end_date":     END_DATE,
        },
        "n_folds":          len(folds),
        "concat_auc":       float(concat_auc),
        "concat_ap":        float(concat_ap),
        "fold_auc_mean":    float(np.mean(all_auc)),
        "fold_auc_std":     float(np.std(all_auc)),
        "fold_ap_mean":     float(np.mean(all_ap)),
        "threshold_analysis": thresh_results,
        "permutation_test": perm_results,
        "regime_test":      regime_results,
        "feature_importance": {k: v for k, v in list(feat_imp.items())[:30]},
    }

    results_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[SAVE] Results → {results_path}")

    # Save per-fold OOT predictions
    oot_rows = []
    for fold in folds:
        for date, label, prob in zip(fold["dates_test"], fold["labels"], fold["probs"]):
            oot_rows.append({
                "date":   date,
                "fold":   fold["fold"],
                "label":  label,
                "prob":   prob,
                "regime": fold["regime"],
            })
    oot_df = pd.DataFrame(oot_rows).sort_values("date")
    oot_path = os.path.join(OUTPUT_DIR, "oot_predictions.csv")
    oot_df.to_csv(oot_path, index=False)
    print(f"[SAVE] OOT predictions → {oot_path}")

    # ── 10. MLflow ──
    print("\n[MLFLOW] Logging to MLflow …")
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        with mlflow.start_run(run_name=f"drawdown_predictor_v1_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            mlflow.log_params({
                "train_days":  TRAIN_DAYS,
                "test_days":   TEST_DAYS,
                "step_days":   STEP_DAYS,
                "horizon":     HORIZON,
                "dd_thresh":   DD_THRESH,
                "start_date":  START_DATE,
                "n_folds":     len(folds),
            })
            mlflow.log_metrics({
                "concat_auc":       float(concat_auc),
                "concat_ap":        float(concat_ap),
                "fold_auc_mean":    float(np.mean(all_auc)),
                "fold_auc_std":     float(np.std(all_auc)),
                "perm_p_value":     float(perm_results.get("p_value") or -1),
                "perm_observed_auc":float(perm_results.get("observed_auc") or -1),
                "hc428_regime_gap": float(regime_results.get("gap") or -1),
            })
            # Log precision/recall at key thresholds
            for tr in thresh_results:
                t = tr["threshold"]
                mlflow.log_metrics({
                    f"precision_at_{t:.2f}": tr["precision"],
                    f"recall_at_{t:.2f}":    tr["recall"],
                    f"fpr_at_{t:.2f}":       tr["fpr"],
                })
            mlflow.log_artifact(results_path)
            mlflow.log_artifact(oot_path)
        print("  MLflow logging complete")
    except Exception as e:
        print(f"  MLflow logging failed: {e} (results still saved locally)")

    # ── 11. Final summary ──
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    print(f"  Folds: {len(folds)}")
    print(f"  Concat OOT AUC:    {concat_auc:.4f}")
    print(f"  Concat OOT AP:     {concat_ap:.4f}")
    print(f"  Perm test p-value: {perm_results.get('p_value', 'N/A')}")
    print(f"  HC #428 regime gap: {regime_results.get('gap', 'N/A'):.4f if regime_results.get('gap') else 'N/A'}")
    print(f"  HC #428 R1 PASS:   {regime_results.get('hc428_passed', 'N/A')}")
    print()
    print("  Threshold summary (key operating points):")
    for tr in thresh_results:
        if tr["threshold"] in [0.30, 0.50, 0.70]:
            print(f"    threshold={tr['threshold']:.2f}  prec={tr['precision']:.3f}  "
                  f"recall={tr['recall']:.3f}  FPR={tr['fpr']:.3f}")
    print("=" * 70)
    print("DONE")


if __name__ == "__main__":
    main()
