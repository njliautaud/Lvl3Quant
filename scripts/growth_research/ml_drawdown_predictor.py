#!/usr/bin/env python3
"""
ML Drawdown Predictor — v4.4 + ML Vol-Targeting Defensive Overlay
=================================================================
Goal: Predict the probability of a >=5% SPY drawdown within next 10 trading
      days, then use that signal to override the v4.4 regime strategy to SPY
      when the risk is elevated.

Architecture:
  - LightGBM with class_weight='balanced' (drawdowns are rare ~5-8% of days)
  - Sliding 252-day walk-forward (HC #0 compliant)
  - 30+ cross-asset features (SPY, VIX, GLD, TLT, HYG, IEF, UUP, XLF, EEM, IWM, QQQ, DBC)
  - Target: binary — will SPY drop >=5% from current close within next 10 trading days?
  - Overlay thresholds tested: 0.3, 0.5, 0.7

Adversarial validation:
  - Permutation test (shuffle SIGNALS, keep returns in order — HC #0 / HC #428)
  - Sub-period (4 blocks)
  - Outlier removal
  - R1 regime-stratified evaluation (HC #428 R1)

Fixed $100K, no DCA (HC #713)
Outputs: output/ml_drawdown_predictor/results.json
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Optional MLflow (non-fatal if unavailable) ────────────────────────────────
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("[WARN] mlflow not installed — skipping experiment tracking")

# ── LightGBM ──────────────────────────────────────────────────────────────────
try:
    import lightgbm as lgb
except ImportError:
    print("Installing lightgbm...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm", "-q"])
    import lgb  # type: ignore

from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    average_precision_score, confusion_matrix
)

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR     = Path("/home/jupiter/Lvl3Quant/output/ml_drawdown_predictor")
MLFLOW_URI     = "http://localhost:5000"
EXPERIMENT_NAME= "ml_drawdown_predictor"

START_DATE     = "2003-01-01"   # capture 2003+ so test folds start ~2005
END_DATE       = datetime.today().strftime("%Y-%m-%d")

TRAIN_DAYS     = 252            # sliding window train size (1 year, HC #0)
STEP_DAYS      = 21             # monthly steps between folds
HORIZON        = 10             # predict 10-day forward drawdown
DD_THRESH      = 0.05           # 5% drawdown threshold
PERM_ITERS     = 500            # permutation test shuffles
INITIAL_CAPITAL= 100_000        # fixed capital, no DCA (HC #713)

# Overlay thresholds to test
THRESHOLDS     = [0.3, 0.5, 0.7]

TICKERS = {
    "spy":  "SPY",
    "vix":  "^VIX",
    "vix3m":"^VIX3M",   # VIX 3-month (term structure)
    "gld":  "GLD",
    "tlt":  "TLT",
    "hyg":  "HYG",
    "ief":  "IEF",
    "uup":  "UUP",      # Dollar index
    "xlf":  "XLF",      # Financials
    "eem":  "EEM",      # Emerging markets
    "iwm":  "IWM",      # Small caps
    "qqq":  "QQQ",      # Tech/growth
    "dbc":  "DBC",      # Commodities
    "upro": "UPRO",     # 3x SPY (for v4.4 strategy)
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# DATA DOWNLOAD
# ──────────────────────────────────────────────────────────────────────────────
def download_data() -> dict:
    print("[DATA] Downloading price history ...")
    raw = {}
    for key, ticker in TICKERS.items():
        try:
            df = yf.download(ticker, start=START_DATE, end=END_DATE,
                             auto_adjust=True, progress=False)
            if len(df) > 100:
                raw[key] = df["Close"].squeeze()
                print(f"  {ticker}: {len(df)} rows "
                      f"({df.index[0].date()} – {df.index[-1].date()})")
            else:
                print(f"  {ticker}: SKIPPED (too short: {len(df)} rows)")
        except Exception as e:
            print(f"  {ticker}: FAILED ({e})")
    return raw


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING  (30+ features)
# ──────────────────────────────────────────────────────────────────────────────
def build_features(raw: dict) -> pd.DataFrame:
    spy = raw["spy"]
    idx = spy.index
    f   = pd.DataFrame(index=idx)

    log_ret = np.log(spy / spy.shift(1))

    # ── SPY realized volatility ──────────────────────────────────────────────
    for w in [5, 10, 20]:
        f[f"spy_rvol_{w}d"] = log_ret.rolling(w).std() * np.sqrt(252)

    # ── SPY drawdown from ATH and 20d high ───────────────────────────────────
    ath    = spy.expanding().max()
    high20 = spy.rolling(20).max()
    f["spy_dd_from_ath"]   = (spy / ath) - 1
    f["spy_dd_from_20d_high"] = (spy / high20) - 1

    # ── SPY distance from 200 SMA ────────────────────────────────────────────
    ma200 = spy.rolling(200).mean()
    f["spy_vs_ma200"] = (spy / ma200) - 1
    f["spy_above_ma200"] = (spy > ma200).astype(int)

    # ── SPY momentum ─────────────────────────────────────────────────────────
    for w in [5, 20, 60]:
        f[f"spy_mom_{w}d"] = spy.pct_change(w)

    # ── RSI (14) ─────────────────────────────────────────────────────────────
    delta  = spy.diff()
    up     = delta.clip(lower=0).rolling(14).mean()
    down   = (-delta.clip(upper=0)).rolling(14).mean()
    rs     = up / (down + 1e-9)
    f["spy_rsi14"] = 100 - (100 / (1 + rs))

    # ── Bollinger %B (20d) ───────────────────────────────────────────────────
    bb_mid = spy.rolling(20).mean()
    bb_std = spy.rolling(20).std()
    f["spy_bb_pct_b"] = (spy - (bb_mid - 2 * bb_std)) / (4 * bb_std + 1e-9)

    # ── Vol-of-vol, skew, kurtosis ───────────────────────────────────────────
    f["spy_vol_of_vol_20d"] = log_ret.rolling(20).std().rolling(20).std() * np.sqrt(252)
    f["spy_skew_20d"]       = log_ret.rolling(20).skew()
    f["spy_kurt_20d"]       = log_ret.rolling(20).kurt()

    # ── Max daily loss in past 5 days ────────────────────────────────────────
    f["spy_max_loss_5d"] = log_ret.rolling(5).min()

    # ── VIX features ─────────────────────────────────────────────────────────
    vix = raw.get("vix")
    vix3m = raw.get("vix3m")
    if vix is not None:
        v = vix.reindex(idx, method="ffill")
        f["vix_level"]       = v
        f["vix_pct_rank_63d"]= v.rolling(63).rank(pct=True)
        f["vix_change_5d"]   = v.pct_change(5)
        f["vix_change_10d"]  = v.pct_change(10)
        f["vix_vs_ma20"]     = (v / v.rolling(20).mean()) - 1
        f["vix_above_25"]    = (v > 25).astype(int)

        # VIX term structure: VIX / VIX3M — >1 means backwardation (fear)
        if vix3m is not None:
            v3 = vix3m.reindex(idx, method="ffill")
            f["vix_term_ratio"] = v / (v3 + 1e-6)
            f["vix3m_change_5d"] = v3.pct_change(5)
        else:
            # Proxy: VIX vs its own 20d SMA as term structure substitute
            f["vix_term_ratio"] = v / (v.rolling(20).mean() + 1e-6)

    # ── Cross-asset features ──────────────────────────────────────────────────
    def _mom(key, w):
        """Safe momentum for an asset."""
        s = raw.get(key)
        if s is None:
            return None
        return s.reindex(idx, method="ffill").pct_change(w)

    # GLD momentum (flight-to-safety)
    for w in [5, 20]:
        m = _mom("gld", w)
        if m is not None:
            f[f"gld_mom_{w}d"] = m

    # TLT momentum (bond rally = risk-off)
    for w in [5, 20]:
        m = _mom("tlt", w)
        if m is not None:
            f[f"tlt_mom_{w}d"] = m

    # HYG-IEF spread proxy (credit risk)
    hyg_s = raw.get("hyg")
    ief_s = raw.get("ief")
    if hyg_s is not None and ief_s is not None:
        hyg_r = hyg_s.reindex(idx, method="ffill")
        ief_r = ief_s.reindex(idx, method="ffill")
        credit_spread = hyg_r / ief_r
        f["hyg_ief_spread"]       = credit_spread
        f["hyg_ief_spread_chg_5d"] = credit_spread.pct_change(5)
        f["hyg_ief_spread_chg_20d"]= credit_spread.pct_change(20)

    # UUP momentum (dollar strength = emerging market risk)
    for w in [5, 20]:
        m = _mom("uup", w)
        if m is not None:
            f[f"uup_mom_{w}d"] = m

    # XLF momentum (financials stress leading indicator)
    for w in [5, 20]:
        m = _mom("xlf", w)
        if m is not None:
            f[f"xlf_mom_{w}d"] = m

    # IWM/SPY ratio — small cap vs large cap (risk appetite)
    iwm_s = raw.get("iwm")
    if iwm_s is not None:
        iwm_r = iwm_s.reindex(idx, method="ffill")
        ratio = iwm_r / spy
        f["iwm_spy_ratio"]       = ratio
        f["iwm_spy_ratio_chg_20d"] = ratio.pct_change(20)
        # Breadth proxy: IWM relative strength vs SPY
        f["iwm_spy_rel_str_5d"] = iwm_r.pct_change(5) - spy.pct_change(5)

    # EEM/SPY ratio — emerging market risk on/off
    eem_s = raw.get("eem")
    if eem_s is not None:
        eem_r = eem_s.reindex(idx, method="ffill")
        ratio_eem = eem_r / spy
        f["eem_spy_ratio"]       = ratio_eem
        f["eem_spy_ratio_chg_20d"] = ratio_eem.pct_change(20)

    # QQQ/SPY ratio — tech leadership
    qqq_s = raw.get("qqq")
    if qqq_s is not None:
        qqq_r = qqq_s.reindex(idx, method="ffill")
        ratio_qqq = qqq_r / spy
        f["qqq_spy_ratio"]       = ratio_qqq
        f["qqq_spy_ratio_chg_20d"] = ratio_qqq.pct_change(20)

    # DBC momentum (commodities as inflation/risk signal)
    for w in [5, 20]:
        m = _mom("dbc", w)
        if m is not None:
            f[f"dbc_mom_{w}d"] = m

    # ── Seasonality ──────────────────────────────────────────────────────────
    f["month"]       = idx.month
    f["is_sept_oct"] = idx.month.isin([9, 10]).astype(int)

    return f


# ──────────────────────────────────────────────────────────────────────────────
# TARGET VARIABLE
# ──────────────────────────────────────────────────────────────────────────────
def build_target(spy: pd.Series,
                 horizon: int = HORIZON,
                 thresh: float = DD_THRESH) -> pd.Series:
    """
    For each day t, look forward `horizon` trading days.
    Returns 1 if min(spy[t+1 : t+horizon+1]) / spy[t] - 1 <= -thresh.
    The last `horizon` rows will be NaN (future unknown at evaluation time).
    """
    prices  = spy.values
    n       = len(prices)
    targets = np.full(n, np.nan)

    for i in range(n - horizon):
        fwd        = prices[i + 1 : i + horizon + 1]
        drawdown   = np.min(fwd) / prices[i] - 1
        targets[i] = 1 if drawdown <= -thresh else 0

    return pd.Series(targets, index=spy.index, name="target")


# ──────────────────────────────────────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────────────────────────────────────
def make_lgbm(n_pos: int, n_neg: int) -> lgb.LGBMClassifier:
    """
    LightGBM with class_weight handling for rare positive events.
    scale_pos_weight = n_neg / n_pos balances the classes.
    """
    scale = n_neg / max(n_pos, 1)
    return lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.02,
        num_leaves=31,
        max_depth=5,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale,
        random_state=42,
        n_jobs=4,
        verbose=-1,
    )


# ──────────────────────────────────────────────────────────────────────────────
# WALK-FORWARD (HC #0: sliding 252-day window)
# ──────────────────────────────────────────────────────────────────────────────
def walk_forward(features: pd.DataFrame,
                 target: pd.Series) -> pd.Series:
    """
    Sliding 252-day training window, step = STEP_DAYS.
    Returns out-of-sample predicted probabilities aligned to the original index.
    """
    aligned  = features.join(target, how="inner").dropna(subset=["target"])
    X_all    = aligned.drop(columns=["target"])
    y_all    = aligned["target"]
    dates    = aligned.index

    n        = len(aligned)
    probs    = pd.Series(np.nan, index=dates, name="ml_prob")

    fold_idx = 0
    start    = 0

    while start + TRAIN_DAYS < n:
        train_end = start + TRAIN_DAYS
        test_end  = min(train_end + STEP_DAYS, n)

        X_train = X_all.iloc[start:train_end]
        y_train = y_all.iloc[start:train_end]
        X_test  = X_all.iloc[train_end:test_end]

        # Drop columns with >50% NaN in training window
        valid_cols = X_train.columns[X_train.isna().mean() < 0.5]
        X_train    = X_train[valid_cols].fillna(X_train[valid_cols].median())
        X_test     = X_test[valid_cols].fillna(X_train[valid_cols].median())

        n_pos = int(y_train.sum())
        n_neg = int((y_train == 0).sum())

        if n_pos < 5 or n_neg < 5:
            # Not enough signal in this window, skip
            start += STEP_DAYS
            fold_idx += 1
            continue

        model = make_lgbm(n_pos, n_neg)
        try:
            model.fit(X_train, y_train)
            p = model.predict_proba(X_test)[:, 1]
            probs.iloc[train_end:test_end] = p
        except Exception as e:
            print(f"  [FOLD {fold_idx}] FAILED: {e}")

        fold_idx += 1
        # Slide window: drop oldest day, add new day
        start += STEP_DAYS

    print(f"  Walk-forward complete: {fold_idx} folds, "
          f"{probs.notna().sum()} OOT predictions")
    return probs


# ──────────────────────────────────────────────────────────────────────────────
# CLASSIFICATION METRICS
# ──────────────────────────────────────────────────────────────────────────────
def classification_report_dict(y_true: np.ndarray,
                                y_prob: np.ndarray,
                                threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total_pos = tp + fn
    total_neg = tn + fp
    return {
        "auc":          float(roc_auc_score(y_true, y_prob)),
        "avg_precision":float(average_precision_score(y_true, y_prob)),
        "precision":    float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":       float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":           float(f1_score(y_true, y_pred, zero_division=0)),
        "tp":           int(tp),
        "fp":           int(fp),
        "tn":           int(tn),
        "fn":           int(fn),
        "total_drawdowns": int(total_pos),
        "total_non_drawdown_days": int(total_neg),
        "drawdowns_caught": int(tp),
        "drawdowns_missed": int(fn),
        "false_alarm_rate": float(fp / max(total_neg, 1)),
        "threshold":    threshold,
    }


# ──────────────────────────────────────────────────────────────────────────────
# PERFORMANCE METRICS
# ──────────────────────────────────────────────────────────────────────────────
def compute_metrics(daily_rets: pd.Series, name: str = "") -> dict:
    """Annualized Sharpe, Sortino, CAGR, MaxDD, Calmar from daily returns."""
    r   = daily_rets.dropna()
    n   = len(r)
    if n == 0:
        return {}

    ann_factor = 252
    total_ret  = (1 + r).prod() - 1
    years      = n / ann_factor
    cagr       = (1 + total_ret) ** (1 / max(years, 0.01)) - 1

    excess     = r - 0.0
    sharpe     = (excess.mean() / (excess.std() + 1e-9)) * np.sqrt(ann_factor)

    downside   = r[r < 0]
    sortino_denom = np.sqrt((downside ** 2).mean() + 1e-9) * np.sqrt(ann_factor)
    sortino    = (excess.mean() * ann_factor) / max(sortino_denom, 1e-9)

    cum        = (1 + r).cumprod()
    roll_max   = cum.cummax()
    dd         = (cum / roll_max) - 1
    max_dd     = float(dd.min())

    calmar     = cagr / max(abs(max_dd), 1e-9)

    return {
        "name":    name,
        "sharpe":  round(float(sharpe), 3),
        "sortino": round(float(sortino), 3),
        "cagr":    round(float(cagr), 4),
        "max_dd":  round(float(max_dd), 4),
        "calmar":  round(float(calmar), 3),
        "n_days":  n,
        "start":   str(r.index[0].date()),
        "end":     str(r.index[-1].date()),
    }


# ──────────────────────────────────────────────────────────────────────────────
# V4.4 REGIME SIGNAL
# ──────────────────────────────────────────────────────────────────────────────
def v44_signal(vix: pd.Series) -> pd.Series:
    """
    v4.4 regime: VIX 20d SMA < VIX 200d SMA AND VIX < 25 → use UPRO (risk on)
                 else → use SPY (risk off)
    Returns pd.Series of {"upro", "spy"} aligned to vix.index.
    """
    vix_ma20  = vix.rolling(20).mean()
    vix_ma200 = vix.rolling(200).mean()
    risk_on   = (vix_ma20 < vix_ma200) & (vix < 25)
    signal    = pd.Series("spy", index=vix.index, name="v44_signal")
    signal[risk_on] = "upro"
    return signal


# ──────────────────────────────────────────────────────────────────────────────
# BACKTESTER
# ──────────────────────────────────────────────────────────────────────────────
def backtest(signal: pd.Series,
             spy_ret: pd.Series,
             upro_ret: pd.Series,
             name: str = "strategy") -> pd.Series:
    """
    signal: pd.Series of {"upro", "spy"} — today's EOD signal, applied to
            NEXT DAY's return (no lookahead).
    Returns daily strategy returns.
    """
    # Shift signal by 1 to avoid lookahead — today's signal acts on tomorrow
    sig_lag = signal.shift(1)

    # Align
    idx = signal.index.intersection(spy_ret.index).intersection(upro_ret.index)
    sig_lag   = sig_lag.reindex(idx)
    spy_ret   = spy_ret.reindex(idx)
    upro_ret  = upro_ret.reindex(idx)

    # Where UPRO data is missing (pre-2009), fall back to SPY
    strat_ret = np.where(
        sig_lag == "upro",
        upro_ret.fillna(spy_ret * 3),   # synthetic 3x before UPRO launch
        spy_ret
    )
    return pd.Series(strat_ret, index=idx, name=name)


def apply_ml_overlay(base_signal: pd.Series,
                     ml_prob: pd.Series,
                     threshold: float) -> pd.Series:
    """
    When ML P(drawdown) > threshold → override to SPY regardless of base signal.
    Returns new signal pd.Series.
    """
    overlay = base_signal.copy()
    override_mask = ml_prob.reindex(base_signal.index).fillna(0) > threshold
    overlay[override_mask] = "spy"
    return overlay


# ──────────────────────────────────────────────────────────────────────────────
# SIMPLE RULE BASELINES
# ──────────────────────────────────────────────────────────────────────────────
def simple_rule_signals(spy: pd.Series,
                        vix: pd.Series,
                        log_ret: pd.Series) -> dict:
    """
    Three simple rules for comparison:
    1. VIX > 25
    2. SPY below 200 SMA
    3. RSI < 30 (oversold = already in trouble)
    Returns dict of {rule_name: prob_series} (1 = high risk day, 0 = low risk).
    """
    # Rule 1: VIX above 25
    r1 = (vix > 25).astype(float)

    # Rule 2: SPY below 200d SMA
    ma200 = spy.rolling(200).mean()
    r2    = (spy < ma200).astype(float)

    # Rule 3: RSI(14) < 30
    delta  = spy.diff()
    up     = delta.clip(lower=0).rolling(14).mean()
    down   = (-delta.clip(upper=0)).rolling(14).mean()
    rs     = up / (down + 1e-9)
    rsi    = 100 - (100 / (1 + rs))
    r3     = (rsi < 30).astype(float)

    return {"rule_vix25": r1, "rule_below_200sma": r2, "rule_rsi30": r3}


# ──────────────────────────────────────────────────────────────────────────────
# PERMUTATION TEST  —  shuffle SIGNALS, keep returns in order
# ──────────────────────────────────────────────────────────────────────────────
def permutation_test(base_signal: pd.Series,
                     ml_prob: pd.Series,
                     spy_ret: pd.Series,
                     upro_ret: pd.Series,
                     threshold: float = 0.5,
                     n_iters: int = PERM_ITERS,
                     metric: str = "sharpe") -> dict:
    """
    Proper permutation test per HC #428 R4:
    - Shuffle the ML OVERRIDE DECISIONS (which days to switch to SPY)
    - Keep market returns in their ORIGINAL time order
    - Metric: Sharpe of the permuted strategy

    Returns {observed, perm_mean, perm_std, p_value, perm_sharpes[:100]}.
    """
    # Observed overlay strategy
    overlay_sig  = apply_ml_overlay(base_signal, ml_prob, threshold)
    obs_ret      = backtest(overlay_sig, spy_ret, upro_ret, "observed")
    obs_metrics  = compute_metrics(obs_ret)
    observed_val = obs_metrics.get(metric, 0.0)

    # The override mask is what we shuffle
    override_mask = (ml_prob.reindex(base_signal.index).fillna(0) > threshold).values

    perm_vals = []
    rng       = np.random.default_rng(seed=42)

    for _ in range(n_iters):
        shuffled_mask  = rng.permutation(override_mask)
        perm_sig       = base_signal.copy()
        perm_sig[shuffled_mask] = "spy"

        perm_ret    = backtest(perm_sig, spy_ret, upro_ret, "perm")
        perm_metrics= compute_metrics(perm_ret)
        perm_vals.append(perm_metrics.get(metric, 0.0))

    perm_arr = np.array(perm_vals)
    p_value  = float((perm_arr >= observed_val).mean())

    return {
        "observed":   float(observed_val),
        "perm_mean":  float(perm_arr.mean()),
        "perm_std":   float(perm_arr.std()),
        "p_value":    p_value,
        "n_iters":    n_iters,
        "metric":     metric,
        "perm_samples": perm_arr[:100].tolist(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# SUB-PERIOD ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────
def sub_period_analysis(strat_ret: pd.Series, n_blocks: int = 4) -> list:
    """Split daily returns into n equal time blocks, compute metrics for each."""
    n     = len(strat_ret)
    block = n // n_blocks
    results = []
    for i in range(n_blocks):
        start_i = i * block
        end_i   = (i + 1) * block if i < n_blocks - 1 else n
        chunk   = strat_ret.iloc[start_i:end_i]
        m       = compute_metrics(chunk, name=f"block_{i+1}")
        results.append(m)
    return results


# ──────────────────────────────────────────────────────────────────────────────
# OUTLIER REMOVAL
# ──────────────────────────────────────────────────────────────────────────────
def outlier_removal_analysis(strat_ret: pd.Series) -> dict:
    """Test how much performance depends on extreme single-day returns."""
    # Remove top 5% by absolute return magnitude
    abs_ret    = strat_ret.abs()
    pct_95     = abs_ret.quantile(0.95)
    trimmed_5pct = strat_ret[abs_ret <= pct_95]

    # Remove top 10 days by absolute return
    top10_idx  = abs_ret.nlargest(10).index
    trimmed_10 = strat_ret.drop(index=top10_idx)

    return {
        "full":       compute_metrics(strat_ret, "full"),
        "trim_top5pct": compute_metrics(trimmed_5pct, "trim_top5pct"),
        "trim_top10days": compute_metrics(trimmed_10, "trim_top10days"),
    }


# ──────────────────────────────────────────────────────────────────────────────
# REGIME STRATIFICATION  (HC #428 R1)
# ──────────────────────────────────────────────────────────────────────────────
def regime_stratified_sharpe(strat_ret: pd.Series,
                              spy_ret: pd.Series) -> dict:
    """
    Classify each day as green / red / flat based on SPY close-to-close.
    Compute Sharpe for each regime.
    Check if |Sharpe_green - Sharpe_red| / max(|Sg|, |Sr|) > 0.50 (reject threshold).
    """
    spy_daily = spy_ret.reindex(strat_ret.index)
    green = spy_daily > 0.001
    red   = spy_daily < -0.001
    flat  = (~green) & (~red)

    def sharpe_of(mask: pd.Series) -> float:
        r = strat_ret[mask]
        if len(r) < 20:
            return np.nan
        return float((r.mean() / (r.std() + 1e-9)) * np.sqrt(252))

    s_green = sharpe_of(green)
    s_red   = sharpe_of(red)
    s_flat  = sharpe_of(flat)

    max_abs = max(abs(s_green) if not np.isnan(s_green) else 0,
                  abs(s_red)   if not np.isnan(s_red)   else 0)
    divergence = abs(s_green - s_red) / max(max_abs, 1e-9) if max_abs > 0 else np.nan

    return {
        "sharpe_green_days": round(float(s_green), 3) if not np.isnan(s_green) else None,
        "sharpe_red_days":   round(float(s_red),   3) if not np.isnan(s_red)   else None,
        "sharpe_flat_days":  round(float(s_flat),  3) if not np.isnan(s_flat)  else None,
        "regime_divergence": round(float(divergence), 3) if not np.isnan(divergence) else None,
        "r1_reject":         bool(divergence > 0.50) if not np.isnan(divergence) else None,
        "n_green_days":      int(green.sum()),
        "n_red_days":        int(red.sum()),
        "n_flat_days":       int(flat.sum()),
    }


# ──────────────────────────────────────────────────────────────────────────────
# SIMPLE RULE AUC
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_simple_rules(rules: dict,
                           y_true: np.ndarray) -> dict:
    out = {}
    for name, pred in rules.items():
        p = pred.reindex(pd.RangeIndex(len(y_true))).fillna(0).values \
            if isinstance(pred.index, pd.DatetimeIndex) else pred.values
        p = p[:len(y_true)]
        try:
            out[name] = float(roc_auc_score(y_true[:len(p)], p))
        except Exception:
            out[name] = None
    return out


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("ML DRAWDOWN PREDICTOR — v4.4 Defensive Overlay")
    print(f"Horizon: {HORIZON}d | DD threshold: {DD_THRESH*100:.0f}% | "
          f"Train window: {TRAIN_DAYS}d | Step: {STEP_DAYS}d")
    print("=" * 70)

    # ── MLflow setup ──────────────────────────────────────────────────────────
    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        run = mlflow.start_run(run_name=f"ml_dd_{datetime.now().strftime('%Y%m%d_%H%M')}")
        print(f"[MLFLOW] Run ID: {run.info.run_id}")
    else:
        run = None

    try:
        # ── Download data ─────────────────────────────────────────────────────
        raw     = download_data()
        spy     = raw["spy"]
        vix     = raw.get("vix")
        upro    = raw.get("upro")

        # Daily returns
        spy_ret  = spy.pct_change().rename("spy_ret")
        upro_ret = upro.pct_change().rename("upro_ret") if upro is not None \
                   else (spy_ret * 3).rename("upro_ret")   # synthetic pre-UPRO

        # ── Build features + target ───────────────────────────────────────────
        print("\n[FEATURES] Engineering features ...")
        features = build_features(raw)
        target   = build_target(spy, HORIZON, DD_THRESH)

        # Align feature/target to common index (post-NaN warmup)
        # Require at least 252 days of history for the first feature to be valid
        valid_start = features.dropna(thresh=int(len(features.columns) * 0.5)).index[0]
        features    = features.loc[valid_start:]
        target      = target.reindex(features.index)

        n_pos = int(target.sum())
        n_neg = int((target == 0).sum())
        pos_rate = n_pos / max(n_pos + n_neg, 1)
        print(f"  Positive rate: {pos_rate:.2%} "
              f"({n_pos} drawdown events / {n_pos + n_neg} labeled days)")

        # ── Walk-forward OOT predictions ──────────────────────────────────────
        print("\n[WF] Running sliding walk-forward ...")
        ml_prob = walk_forward(features, target)

        # Restrict evaluation to OOT period
        oot_mask  = ml_prob.notna()
        oot_dates = ml_prob[oot_mask].index
        y_true_oot= target.reindex(oot_dates).dropna()
        y_prob_oot= ml_prob.reindex(y_true_oot.index)

        # ── Classification metrics at each threshold ──────────────────────────
        print("\n[EVAL] Classification metrics ...")
        clf_results = {}
        for thr in THRESHOLDS:
            clf_results[f"threshold_{thr}"] = classification_report_dict(
                y_true_oot.values, y_prob_oot.values, threshold=thr
            )
            r = clf_results[f"threshold_{thr}"]
            print(f"  Threshold={thr}: AUC={r['auc']:.3f} "
                  f"Prec={r['precision']:.3f} Recall={r['recall']:.3f} "
                  f"F1={r['f1']:.3f} | "
                  f"Caught {r['drawdowns_caught']}/{r['total_drawdowns']} drawdowns, "
                  f"False alarm rate={r['false_alarm_rate']:.2%}")

        # ── Simple rule baselines ─────────────────────────────────────────────
        print("\n[RULES] Evaluating simple rule baselines ...")
        log_ret = np.log(spy / spy.shift(1))
        rules   = simple_rule_signals(spy, vix if vix is not None else spy * 0, log_ret)
        # Align rules to OOT dates
        rules_oot  = {k: v.reindex(y_true_oot.index).fillna(0) for k, v in rules.items()}
        rule_aucs  = {}
        for name, r_series in rules_oot.items():
            try:
                auc = float(roc_auc_score(y_true_oot.values, r_series.values))
            except Exception:
                auc = None
            rule_aucs[name] = auc
            print(f"  {name}: AUC = {auc:.3f}" if auc else f"  {name}: AUC = N/A")
        print(f"  ml_lgbm:        AUC = {clf_results['threshold_0.5']['auc']:.3f}")

        # ── Strategy backtests ────────────────────────────────────────────────
        print("\n[BACKTEST] Running strategy comparisons ...")

        # V4.4 baseline (no ML overlay)
        if vix is not None:
            sig_v44 = v44_signal(vix)
        else:
            sig_v44 = pd.Series("spy", index=spy.index, name="v44_signal")

        ret_v44 = backtest(sig_v44, spy_ret, upro_ret, "v44_base")

        strat_results = {}
        strat_rets    = {"v44_base": ret_v44}

        m_v44 = compute_metrics(ret_v44, "v44_base")
        strat_results["v44_base"] = m_v44
        print(f"  v4.4 base: Sharpe={m_v44['sharpe']} CAGR={m_v44['cagr']:.1%} "
              f"MaxDD={m_v44['max_dd']:.1%}")

        # v4.4 + ML overlay at each threshold
        for thr in THRESHOLDS:
            name      = f"v44_ml_{thr}"
            overlay   = apply_ml_overlay(sig_v44, ml_prob, thr)
            ret_strat = backtest(overlay, spy_ret, upro_ret, name)
            m         = compute_metrics(ret_strat, name)
            strat_results[name]  = m
            strat_rets[name]     = ret_strat
            print(f"  {name}: Sharpe={m['sharpe']} CAGR={m['cagr']:.1%} "
                  f"MaxDD={m['max_dd']:.1%} Calmar={m['calmar']}")

        # SPY buy-and-hold reference
        spy_bh = spy_ret.copy()
        spy_bh.name = "spy_bh"
        m_spy = compute_metrics(spy_bh, "spy_buy_hold")
        strat_results["spy_buy_hold"] = m_spy
        strat_rets["spy_buy_hold"]    = spy_bh
        print(f"  SPY B&H: Sharpe={m_spy['sharpe']} CAGR={m_spy['cagr']:.1%} "
              f"MaxDD={m_spy['max_dd']:.1%}")

        # ── Adversarial: pick primary overlay (threshold=0.5) ─────────────────
        primary_thr  = 0.5
        primary_name = f"v44_ml_{primary_thr}"
        primary_ret  = strat_rets[primary_name]

        # 1. Permutation test (shuffle SIGNALS, not returns — HC #428)
        print(f"\n[PERM] Permutation test on {primary_name} ({PERM_ITERS} iters) ...")
        perm_results = permutation_test(
            sig_v44, ml_prob, spy_ret, upro_ret,
            threshold=primary_thr, n_iters=PERM_ITERS, metric="sharpe"
        )
        print(f"  Observed Sharpe={perm_results['observed']:.3f} | "
              f"Perm mean={perm_results['perm_mean']:.3f} ± {perm_results['perm_std']:.3f} | "
              f"p-value={perm_results['p_value']:.4f}")

        # 2. Sub-period analysis (4 blocks)
        print("\n[SUB-PERIOD] 4-block analysis ...")
        sub_period_res = {}
        for name, ret_s in strat_rets.items():
            if name in ["spy_buy_hold", "v44_base", primary_name]:
                sub_period_res[name] = sub_period_analysis(ret_s, n_blocks=4)
        for block_idx in range(4):
            v44_sp = sub_period_res["v44_base"][block_idx]
            ml_sp  = sub_period_res.get(primary_name, [{}]*4)[block_idx]
            print(f"  Block {block_idx+1} ({v44_sp.get('start','?')} – {v44_sp.get('end','?')}): "
                  f"v44 Sharpe={v44_sp.get('sharpe','?')} | "
                  f"ML Sharpe={ml_sp.get('sharpe','?')}")

        # 3. Outlier removal
        print("\n[OUTLIERS] Outlier removal analysis ...")
        outlier_res = {}
        for name in ["v44_base", primary_name]:
            outlier_res[name] = outlier_removal_analysis(strat_rets[name])
            o = outlier_res[name]
            print(f"  {name}: Full Sharpe={o['full']['sharpe']} | "
                  f"Trim 5%: {o['trim_top5pct']['sharpe']} | "
                  f"Trim top10 days: {o['trim_top10days']['sharpe']}")

        # 4. Regime stratification (HC #428 R1)
        print("\n[REGIME] R1 regime-stratified Sharpe ...")
        regime_res = {}
        for name in ["v44_base", primary_name]:
            regime_res[name] = regime_stratified_sharpe(strat_rets[name], spy_ret)
            r = regime_res[name]
            reject = r.get('r1_reject', None)
            print(f"  {name}: Green={r['sharpe_green_days']} "
                  f"Red={r['sharpe_red_days']} Flat={r['sharpe_flat_days']} | "
                  f"Divergence={r['regime_divergence']} → R1_REJECT={reject}")

        # ── Assemble results ──────────────────────────────────────────────────
        results = {
            "run_date":      datetime.now().isoformat(),
            "config": {
                "horizon":     HORIZON,
                "dd_thresh":   DD_THRESH,
                "train_days":  TRAIN_DAYS,
                "step_days":   STEP_DAYS,
                "perm_iters":  PERM_ITERS,
                "start_date":  START_DATE,
                "end_date":    END_DATE,
                "thresholds":  THRESHOLDS,
            },
            "classification": clf_results,
            "simple_rule_aucs": rule_aucs,
            "strategy_performance": strat_results,
            "adversarial": {
                "permutation_test": perm_results,
                "sub_period":       sub_period_res,
                "outlier_removal":  outlier_res,
                "regime_r1":        regime_res,
            },
        }

        # Save
        out_path = OUTPUT_DIR / "results.json"
        with open(out_path, "w") as fp:
            json.dump(results, fp, indent=2, default=str)
        print(f"\n[SAVED] {out_path}")

        # ── MLflow log ────────────────────────────────────────────────────────
        if MLFLOW_AVAILABLE and run is not None:
            mlflow.log_params({
                "horizon":    HORIZON,
                "dd_thresh":  DD_THRESH,
                "train_days": TRAIN_DAYS,
                "step_days":  STEP_DAYS,
            })
            # Log primary strategy metrics
            pm = strat_results[primary_name]
            mlflow.log_metrics({
                "sharpe":    pm["sharpe"],
                "sortino":   pm["sortino"],
                "cagr":      pm["cagr"],
                "max_dd":    abs(pm["max_dd"]),
                "calmar":    pm["calmar"],
                "ml_auc":    clf_results["threshold_0.5"]["auc"],
                "perm_pvalue": perm_results["p_value"],
            })
            mlflow.log_artifact(str(out_path))

        # ── Final summary ─────────────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)
        print(f"{'Strategy':<25} {'Sharpe':>8} {'Sortino':>8} "
              f"{'CAGR':>8} {'MaxDD':>8} {'Calmar':>8}")
        print("-" * 70)
        for name, m in strat_results.items():
            print(f"{name:<25} {m['sharpe']:>8.3f} {m['sortino']:>8.3f} "
                  f"{m['cagr']:>8.1%} {m['max_dd']:>8.1%} {m['calmar']:>8.3f}")
        print("-" * 70)
        print(f"\nML model (threshold=0.5):")
        r5 = clf_results["threshold_0.5"]
        print(f"  AUC: {r5['auc']:.3f} | Avg Precision: {r5['avg_precision']:.3f}")
        print(f"  Drawdowns caught: {r5['drawdowns_caught']} / {r5['total_drawdowns']} "
              f"| False alarm rate: {r5['false_alarm_rate']:.1%}")
        print(f"\nPermutation test (shuffle signals, keep returns): p={perm_results['p_value']:.4f}")

        return results

    finally:
        if MLFLOW_AVAILABLE and run is not None:
            mlflow.end_run()


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
