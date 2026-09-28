#!/usr/bin/env python3
"""
ML Factor Timing — Rotating Between Smart-Beta Factor ETFs
============================================================
Tests whether LightGBM can predict which factor ETF will outperform
SPY each month, rotating across: MTUM (momentum), VLUE (value),
QUAL (quality), USMV (low vol), or SHY (cash/safety).

Baseline target: v4.4+ML vol targeting (Sharpe 1.26).

HC compliance:
  - Sliding 12-month walk-forward, monthly rebalance (HC #0)
  - No DCA, fixed $100K (HC #713)
  - Regime-agnostic adversarial validation (HC #428 R1)
  - Signals shuffled (not returns) in permutation test
"""

import json
import warnings
import datetime as dt
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

import lightgbm as lgb

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/ml_factor_timing")
CACHE_DIR   = Path("/home/jupiter/Lvl3Quant/output/growth_research/cache")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "2013-01-01"   # Factor ETFs launched ~2013
END_DATE   = dt.date.today().isoformat()

INITIAL_CAPITAL = 100_000.0  # Fixed, no DCA (HC #713)

# Universe
FACTOR_ETFS   = ["MTUM", "VLUE", "QUAL", "USMV"]
CASH_ASSET    = "SHY"
BENCH_ASSET   = "SPY"
ALL_ACTIONS   = FACTOR_ETFS + [CASH_ASSET]  # "none" maps to SHY

# Walk-forward: sliding 12-month train, 1-month test (monthly rebalance)
TRAIN_MONTHS  = 12
TEST_MONTHS   = 1

# LightGBM
LGBM_PARAMS = {
    "objective":        "multiclass",
    "num_class":        len(ALL_ACTIONS),
    "metric":           "multi_logloss",
    "n_estimators":     300,
    "learning_rate":    0.05,
    "num_leaves":       31,
    "min_child_samples": 5,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.1,
    "reg_lambda":       0.1,
    "random_state":     42,
    "verbosity":        -1,
    "n_jobs":           -1,
}

N_PERMUTATIONS = 500

# ══════════════════════════════════════════════════════════════
# 1. DATA DOWNLOAD
# ══════════════════════════════════════════════════════════════

def download_all_data():
    """Download price data; use disk cache if available."""
    cache_file = CACHE_DIR / f"ml_factor_timing_{END_DATE}.parquet"
    if cache_file.exists():
        print(f"[DATA] Loading cached data ({cache_file.name})")
        return pd.read_parquet(cache_file)

    tickers = {
        # Factor ETFs (targets + features)
        "MTUM":  "MTUM",
        "VLUE":  "VLUE",
        "QUAL":  "QUAL",
        "USMV":  "USMV",
        # Benchmarks / safety
        "SPY":   "SPY",
        "SHY":   "SHY",
        # Macro context
        "VIX":   "^VIX",
        "TLT":   "TLT",
        "GLD":   "GLD",
        "UUP":   "UUP",
        "HYG":   "HYG",
        "IEF":   "IEF",
    }

    prices = {}
    for name, ticker in tickers.items():
        print(f"  Downloading {name} ({ticker})...")
        try:
            df = yf.download(ticker, start=START_DATE, end=END_DATE,
                             progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 100:
                prices[name] = df["Close"]
            else:
                print(f"    WARNING: {name} only {len(df)} rows — skipping")
        except Exception as e:
            print(f"    ERROR downloading {name}: {e}")

    price_df = pd.DataFrame(prices).ffill(limit=5)
    price_df.to_parquet(cache_file)
    print(f"[DATA] Saved {len(price_df)} rows × {len(price_df.columns)} cols to cache")
    return price_df


# ══════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING  (daily → resampled to month-end)
# ══════════════════════════════════════════════════════════════

def build_features(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Build month-end feature rows from daily price data.

    All features are computed as of the last trading day of each calendar
    month so there is zero look-ahead: we rebalance at month-open using
    features known at prior month-close.
    """
    feat = pd.DataFrame(index=daily.index)

    spy = daily["SPY"]
    vix = daily["VIX"] if "VIX" in daily.columns else None

    # ── VIX features ──────────────────────────────────────────
    if vix is not None:
        feat["vix_level"]       = vix
        feat["vix_pct_63d"]     = vix.rolling(63).rank(pct=True)
        feat["vix_chg_21d"]     = vix.diff(21)
        # VIX term structure proxy: short-run vs medium-run realized vol
        spy_ret = spy.pct_change()
        rv10    = spy_ret.rolling(10).std() * np.sqrt(252) * 100
        rv21    = spy_ret.rolling(21).std() * np.sqrt(252) * 100
        feat["vix_vs_rv21"]         = vix / (rv21 + 1e-10)
        feat["rv_term_structure"]    = rv10 / (rv21 + 1e-10)

    # ── Credit spread ─────────────────────────────────────────
    if "HYG" in daily.columns and "IEF" in daily.columns:
        cr = daily["HYG"] / daily["IEF"]
        feat["credit_spread_ratio"]   = cr
        feat["credit_spread_21d_chg"] = cr.pct_change(21)
        feat["credit_trend"]          = (cr.rolling(10).mean()
                                          - cr.rolling(50).mean())

    # ── Yield curve / TLT ────────────────────────────────────
    if "TLT" in daily.columns:
        feat["tlt_ret_21d"]   = daily["TLT"].pct_change(21)
        feat["tlt_ret_63d"]   = daily["TLT"].pct_change(63)
        feat["tlt_mom_sign"]  = np.sign(feat["tlt_ret_21d"])

    # ── Dollar / safe-haven ───────────────────────────────────
    if "UUP" in daily.columns:
        feat["uup_mom_21d"]   = daily["UUP"].pct_change(21)
        feat["uup_mom_63d"]   = daily["UUP"].pct_change(63)
    if "GLD" in daily.columns:
        feat["gld_mom_21d"]   = daily["GLD"].pct_change(21)
        feat["gld_mom_63d"]   = daily["GLD"].pct_change(63)

    # ── SPY momentum (multi-horizon) ──────────────────────────
    for n, lbl in [(21, "1m"), (63, "3m"), (126, "6m"), (252, "12m")]:
        feat[f"spy_mom_{lbl}"]  = spy.pct_change(n)
    feat["spy_above_200ma"]     = (spy > spy.rolling(200).mean()).astype(float)

    # ── SPY volatility ────────────────────────────────────────
    spy_ret = spy.pct_change()
    feat["spy_vol_10d"]   = spy_ret.rolling(10).std() * np.sqrt(252)
    feat["spy_vol_20d"]   = spy_ret.rolling(21).std() * np.sqrt(252)
    # Vol-of-vol: how much is vol itself changing?
    feat["vol_of_vol"]    = feat["spy_vol_10d"].rolling(21).std()

    # ── Per-factor trailing returns & alpha vs SPY ────────────
    for etf in FACTOR_ETFS:
        if etf not in daily.columns:
            continue
        etf_ret = daily[etf].pct_change()
        spy_ret_d = spy.pct_change()

        # Trailing returns
        feat[f"{etf}_ret_1m"]  = daily[etf].pct_change(21)
        feat[f"{etf}_ret_3m"]  = daily[etf].pct_change(63)

        # Trailing alpha vs SPY (excess return, not regression alpha)
        feat[f"{etf}_alpha_1m"] = feat[f"{etf}_ret_1m"] - feat["spy_mom_1m"]
        feat[f"{etf}_alpha_3m"] = feat[f"{etf}_ret_3m"] - feat["spy_mom_3m"]

    # ── Resample to month-end ─────────────────────────────────
    monthly = feat.resample("ME").last()
    return monthly


# ══════════════════════════════════════════════════════════════
# 3. TARGET CONSTRUCTION  (which factor beats SPY next month)
# ══════════════════════════════════════════════════════════════

def build_targets(daily: pd.DataFrame) -> pd.DataFrame:
    """
    For each month-end date, compute NEXT-MONTH excess return of each factor
    vs SPY.  Label = factor with highest positive excess return, or SHY if all
    factors underperform SPY.

    Returns a DataFrame with columns:
      - 'target_label': string label (one of ALL_ACTIONS)
      - 'target_idx':   integer index (for LightGBM)
      - 'next_spy_ret': SPY return next month (for P&L)
      - one column per factor ETF: their next-month raw return
    """
    monthly_prices = daily.resample("ME").last()
    monthly_returns = monthly_prices.pct_change()

    rows = []
    dates = monthly_returns.index[:-1]  # drop last (no forward return)

    for date in dates:
        next_date_idx = monthly_returns.index.get_loc(date) + 1
        if next_date_idx >= len(monthly_returns):
            break
        next_date = monthly_returns.index[next_date_idx]

        spy_fwd = monthly_returns.loc[next_date, "SPY"] if "SPY" in monthly_returns.columns else 0.0

        # Excess returns of each factor vs SPY
        excess = {}
        raw_rets = {}
        for etf in FACTOR_ETFS:
            if etf in monthly_returns.columns:
                etf_ret = monthly_returns.loc[next_date, etf]
                excess[etf]   = etf_ret - spy_fwd
                raw_rets[etf] = etf_ret
            else:
                excess[etf]   = np.nan
                raw_rets[etf] = np.nan

        # Best factor (highest excess return, must be > 0)
        valid = {k: v for k, v in excess.items() if not np.isnan(v)}
        best_factor  = max(valid, key=lambda k: valid[k]) if valid else None
        best_excess  = valid[best_factor] if best_factor else -1.0

        if best_excess > 0:
            label = best_factor
        else:
            label = CASH_ASSET  # all factors underperform SPY → hold SHY

        row = {
            "date":        date,
            "target_label": label,
            "next_spy_ret": spy_fwd,
        }
        for etf in FACTOR_ETFS:
            row[f"next_{etf}_ret"] = raw_rets.get(etf, np.nan)
        if CASH_ASSET in monthly_returns.columns:
            row[f"next_{CASH_ASSET}_ret"] = monthly_returns.loc[next_date, CASH_ASSET]
        rows.append(row)

    targets = pd.DataFrame(rows).set_index("date")

    # Encode labels
    le = LabelEncoder()
    le.fit(ALL_ACTIONS)
    targets["target_idx"] = le.transform(targets["target_label"])
    targets.attrs["label_encoder"] = le  # stash for later
    return targets, le


# ══════════════════════════════════════════════════════════════
# 4. WALK-FORWARD SLIDING BACKTEST
# ══════════════════════════════════════════════════════════════

def run_walkforward(features_monthly: pd.DataFrame,
                    targets: pd.DataFrame,
                    le: LabelEncoder):
    """
    Sliding 12-month train / 1-month test walk-forward.

    Returns per-period results: date, predicted label, actual label,
    and returns for all instruments that month.
    """
    # Align index
    common_idx = features_monthly.index.intersection(targets.index)
    X_all = features_monthly.loc[common_idx].copy()
    y_all = targets.loc[common_idx].copy()

    X_all = X_all.replace([np.inf, -np.inf], np.nan)
    X_all = X_all.ffill().bfill()

    feature_cols = X_all.columns.tolist()
    n = len(X_all)
    dates = X_all.index

    results = []
    fold_accs = []

    print(f"\n[WF] {n} monthly obs. Train={TRAIN_MONTHS}m, Test={TEST_MONTHS}m")

    for test_start_i in range(TRAIN_MONTHS, n):
        train_start_i = max(0, test_start_i - TRAIN_MONTHS)
        train_idx = list(range(train_start_i, test_start_i))
        test_idx  = [test_start_i]

        if len(train_idx) < TRAIN_MONTHS:
            continue

        X_train = X_all.iloc[train_idx][feature_cols].values
        y_train = y_all.iloc[train_idx]["target_idx"].values.astype(int)
        X_test  = X_all.iloc[test_idx][feature_cols].values
        y_test  = y_all.iloc[test_idx]["target_idx"].values.astype(int)

        # Drop rows with NaN in training
        valid_mask = ~np.isnan(X_train).any(axis=1)
        X_train = X_train[valid_mask]
        y_train = y_train[valid_mask]

        if len(X_train) < 6 or np.isnan(X_test).any():
            continue

        # Train LightGBM
        try:
            model = lgb.LGBMClassifier(**LGBM_PARAMS)
            model.fit(X_train, y_train,
                      callbacks=[lgb.early_stopping(30, verbose=False),
                                 lgb.log_evaluation(period=-1)],
                      eval_set=[(X_train, y_train)])
        except Exception as e:
            # Fall back without early stopping if val set too small
            model = lgb.LGBMClassifier(
                **{k: v for k, v in LGBM_PARAMS.items()
                   if k not in ("n_estimators",)},
                n_estimators=100
            )
            model.fit(X_train, y_train)

        pred_idx  = model.predict(X_test)[0]
        pred_label = le.inverse_transform([pred_idx])[0]
        actual_label = le.inverse_transform(y_test)[0]

        fold_accs.append(int(pred_idx == y_test[0]))

        test_date = dates[test_idx[0]]
        row_target = y_all.iloc[test_idx[0]]

        # Build return lookup for this month
        ret_lookup = {"SPY": row_target["next_spy_ret"]}
        for etf in FACTOR_ETFS:
            col = f"next_{etf}_ret"
            if col in row_target.index:
                ret_lookup[etf] = row_target[col]
        if f"next_{CASH_ASSET}_ret" in row_target.index:
            ret_lookup[CASH_ASSET] = row_target[f"next_{CASH_ASSET}_ret"]
        else:
            ret_lookup[CASH_ASSET] = 0.0015  # ~SHY monthly carry if missing

        results.append({
            "date":         test_date,
            "pred_label":   pred_label,
            "actual_label": actual_label,
            "correct":      int(pred_label == actual_label),
            "ret_lookup":   ret_lookup,
        })

    print(f"[WF] {len(results)} test periods. Avg fold accuracy: "
          f"{np.mean(fold_accs):.3f}")
    return results


# ══════════════════════════════════════════════════════════════
# 5. EQUITY CURVE BUILDER
# ══════════════════════════════════════════════════════════════

def build_equity_curve(results, signal_key="pred_label",
                       initial=INITIAL_CAPITAL):
    """
    Build a monthly equity curve from walk-forward results.
    signal_key: column to use for the trade signal ('pred_label' or
                any permuted label column).
    """
    equity  = initial
    values  = [initial]
    monthly = []

    for r in results:
        label   = r[signal_key]
        ret_map = r["ret_lookup"]
        ret     = ret_map.get(label, 0.0)
        if np.isnan(ret):
            ret = 0.0
        equity  *= (1 + ret)
        values.append(equity)
        monthly.append({
            "date":   r["date"],
            "label":  label,
            "ret":    ret,
            "equity": equity,
        })

    return pd.DataFrame(monthly)


def compute_metrics(monthly_df: pd.DataFrame, label: str = "") -> dict:
    """Compute Sharpe, Sortino, CAGR, MaxDD, Calmar, WR vs SPY, turnover."""
    rets = monthly_df["ret"].values
    if len(rets) == 0:
        return {}

    ann  = 12
    mean = np.mean(rets) * ann
    std  = np.std(rets, ddof=1) * np.sqrt(ann)
    sharpe = mean / (std + 1e-10)

    downside = rets[rets < 0]
    sortino_den = np.sqrt(np.mean(downside**2) * ann) if len(downside) > 0 else 1e-10
    sortino = mean / sortino_den

    n_years = len(rets) / 12
    start_val = INITIAL_CAPITAL
    end_val   = monthly_df["equity"].iloc[-1]
    cagr = (end_val / start_val) ** (1 / n_years) - 1 if n_years > 0 else 0.0

    roll_max = monthly_df["equity"].cummax()
    dd = (monthly_df["equity"] - roll_max) / roll_max
    max_dd = dd.min()

    calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-6 else 0.0

    # Turnover: fraction of months where holding changes
    labels = monthly_df["label"].tolist()
    turns  = sum(1 for i in range(1, len(labels)) if labels[i] != labels[i-1])
    turnover = turns / max(len(labels) - 1, 1)

    return {
        "label":    label,
        "sharpe":   round(sharpe, 3),
        "sortino":  round(sortino, 3),
        "cagr":     round(cagr * 100, 2),
        "max_dd":   round(max_dd * 100, 2),
        "calmar":   round(calmar, 3),
        "turnover": round(turnover, 3),
        "n_months": len(rets),
        "end_value": round(end_val, 0),
    }


# ══════════════════════════════════════════════════════════════
# 6. BASELINES
# ══════════════════════════════════════════════════════════════

def build_baseline(results, asset_label):
    """Buy-and-hold one asset throughout (ignores ML signal)."""
    monthly = []
    equity  = INITIAL_CAPITAL
    for r in results:
        ret = r["ret_lookup"].get(asset_label, 0.0)
        if np.isnan(ret):
            ret = 0.0
        equity *= (1 + ret)
        monthly.append({"date": r["date"], "label": asset_label,
                         "ret": ret, "equity": equity})
    return pd.DataFrame(monthly)


def build_equal_weight_baseline(results):
    """Equal-weight all four factor ETFs each month (rebalanced monthly)."""
    monthly = []
    equity  = INITIAL_CAPITAL
    for r in results:
        avg_ret = np.nanmean([r["ret_lookup"].get(etf, 0.0) for etf in FACTOR_ETFS])
        equity *= (1 + avg_ret)
        monthly.append({"date": r["date"], "label": "EW_FACTORS",
                         "ret": avg_ret, "equity": equity})
    return pd.DataFrame(monthly)


# ══════════════════════════════════════════════════════════════
# 7. ADVERSARIAL TESTS
# ══════════════════════════════════════════════════════════════

def permutation_test(results, n_perms=N_PERMUTATIONS):
    """
    Shuffle ML SIGNALS (pred_label) while keeping factor returns in order.
    Measures how often random signal achieves the observed Sharpe.

    IMPORTANT: we shuffle the picks (which factor to hold each month),
    NOT the factor returns. This isolates signal skill from structural
    risk premia.
    """
    ml_curve   = build_equity_curve(results, signal_key="pred_label")
    ml_sharpe  = compute_metrics(ml_curve, "ML")["sharpe"]

    labels      = [r["pred_label"] for r in results]
    perm_sharpes = []

    rng = np.random.default_rng(seed=42)
    for _ in range(n_perms):
        shuffled = rng.permutation(labels).tolist()
        # Inject shuffled labels into results copies (no mutation)
        perm_results = [dict(r, pred_label=shuffled[i])
                        for i, r in enumerate(results)]
        perm_curve   = build_equity_curve(perm_results, signal_key="pred_label")
        perm_sharpes.append(compute_metrics(perm_curve, "perm")["sharpe"])

    p_value  = np.mean(np.array(perm_sharpes) >= ml_sharpe)
    pct_beat = (1 - p_value) * 100

    return {
        "observed_sharpe":   ml_sharpe,
        "perm_mean_sharpe":  round(float(np.mean(perm_sharpes)), 3),
        "perm_p95_sharpe":   round(float(np.percentile(perm_sharpes, 95)), 3),
        "p_value":           round(float(p_value), 4),
        "pct_perms_beaten":  round(pct_beat, 1),
        "n_perms":           n_perms,
        "verdict":           "PASS" if p_value < 0.10 else "FAIL",
    }


def sub_period_test(results):
    """Split into 4 equal sub-periods; check if ML beats SPY B&H in each."""
    n = len(results)
    block = n // 4
    sub_results = []

    for i in range(4):
        start = i * block
        end   = (i + 1) * block if i < 3 else n
        chunk = results[start:end]

        ml_curve  = build_equity_curve(chunk, "pred_label")
        spy_curve = build_baseline(chunk, "SPY")

        ml_m  = compute_metrics(ml_curve,  f"ML_sub{i+1}")
        spy_m = compute_metrics(spy_curve, f"SPY_sub{i+1}")

        date_start = chunk[0]["date"].strftime("%Y-%m")
        date_end   = chunk[-1]["date"].strftime("%Y-%m")
        sub_results.append({
            "period":        f"{date_start}→{date_end}",
            "ml_sharpe":     ml_m["sharpe"],
            "spy_sharpe":    spy_m["sharpe"],
            "ml_cagr":       ml_m["cagr"],
            "spy_cagr":      spy_m["cagr"],
            "ml_beats_spy":  ml_m["sharpe"] > spy_m["sharpe"],
        })
    return sub_results


def outlier_removal_test(results, trim_pct=0.10):
    """
    Remove the best and worst trim_pct months of SPY to stress-test
    whether ML alpha is driven by outlier months.
    """
    spy_rets = np.array([r["ret_lookup"].get("SPY", 0.0) for r in results])
    lo = np.percentile(spy_rets, trim_pct * 100)
    hi = np.percentile(spy_rets, (1 - trim_pct) * 100)

    trimmed = [r for r in results
               if lo <= r["ret_lookup"].get("SPY", 0.0) <= hi]

    ml_curve  = build_equity_curve(trimmed, "pred_label")
    spy_curve = build_baseline(trimmed, "SPY")

    return {
        "n_months_after_trim": len(trimmed),
        "pct_removed":         round((1 - len(trimmed) / len(results)) * 100, 1),
        "ml_sharpe":           compute_metrics(ml_curve,  "ML_trimmed")["sharpe"],
        "spy_sharpe":          compute_metrics(spy_curve, "SPY_trimmed")["sharpe"],
        "verdict":             ("PASS"
                                if compute_metrics(ml_curve, "ml")["sharpe"] >
                                   compute_metrics(spy_curve, "spy")["sharpe"]
                                else "FAIL"),
    }


def regime_test(results):
    """
    HC #428 R1: Stratify by SPY regime (green/red/flat) and check if
    Sharpe disparity is within the 50% tolerance band.
    """
    spy_rets = np.array([r["ret_lookup"].get("SPY", 0.0) for r in results])
    # Classify months
    green = spy_rets > 0.005
    red   = spy_rets < -0.005
    flat  = ~green & ~red

    regime_labels = []
    for g, r_, f in zip(green, red, flat):
        if g:
            regime_labels.append("green")
        elif r_:
            regime_labels.append("red")
        else:
            regime_labels.append("flat")

    regime_sharpes = {}
    for regime in ["green", "red", "flat"]:
        idxs = [i for i, rl in enumerate(regime_labels) if rl == regime]
        if len(idxs) < 3:
            continue
        chunk = [results[i] for i in idxs]
        ml_curve = build_equity_curve(chunk, "pred_label")
        regime_sharpes[regime] = compute_metrics(ml_curve, f"ML_{regime}")["sharpe"]

    # Check HC #428 criterion
    sh_values = list(regime_sharpes.values())
    if len(sh_values) >= 2:
        max_abs = max(abs(v) for v in sh_values)
        disparities = []
        regimes_list = list(regime_sharpes.keys())
        for i in range(len(regimes_list)):
            for j in range(i + 1, len(regimes_list)):
                ri, rj = regimes_list[i], regimes_list[j]
                disparity = abs(regime_sharpes[ri] - regime_sharpes[rj]) / (max_abs + 1e-10)
                disparities.append((f"{ri}_vs_{rj}", round(disparity, 3)))

        max_disparity = max(d[1] for d in disparities)
        verdict = "PASS" if max_disparity <= 0.50 else "FAIL"
    else:
        disparities = []
        max_disparity = 0.0
        verdict = "INSUFFICIENT_DATA"

    return {
        "regime_sharpes":  {k: round(v, 3) for k, v in regime_sharpes.items()},
        "pairwise_disparity": disparities,
        "max_disparity":   round(max_disparity, 3),
        "threshold":       0.50,
        "verdict":         verdict,
    }


def win_rate_vs_spy(results):
    """Fraction of months where ML strategy beat SPY."""
    ml_rets  = [r["ret_lookup"].get(r["pred_label"], 0.0) for r in results]
    spy_rets = [r["ret_lookup"].get("SPY", 0.0) for r in results]
    beats    = sum(1 for m, s in zip(ml_rets, spy_rets) if m > s)
    return round(beats / max(len(results), 1), 3)


# ══════════════════════════════════════════════════════════════
# 8. MAIN
# ══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("ML FACTOR TIMING RESEARCH")
    print(f"Universe: {FACTOR_ETFS} | Cash: {CASH_ASSET}")
    print(f"Period: {START_DATE} → {END_DATE}")
    print("=" * 60)

    # ── 1. Data ───────────────────────────────────────────────
    daily = download_all_data()
    print(f"\n[DATA] Shape: {daily.shape}")
    print(f"[DATA] Date range: {daily.index[0].date()} → {daily.index[-1].date()}")

    # Drop rows where all factor ETFs are missing
    core_cols = FACTOR_ETFS + [BENCH_ASSET]
    daily = daily.dropna(subset=[c for c in core_cols if c in daily.columns],
                          how="all")

    # ── 2. Features & targets ─────────────────────────────────
    print("\n[FEAT] Building monthly features...")
    features_monthly = build_features(daily)
    features_monthly = features_monthly.dropna(how="all")

    print("[TARGET] Building monthly targets...")
    targets, le = build_targets(daily)

    print(f"\n[TARGET] Label distribution:")
    label_counts = targets["target_label"].value_counts()
    for lbl, cnt in label_counts.items():
        print(f"  {lbl}: {cnt} months ({cnt/len(targets)*100:.1f}%)")

    # ── 3. Walk-forward ───────────────────────────────────────
    print("\n[WF] Running sliding walk-forward...")
    wf_results = run_walkforward(features_monthly, targets, le)

    if len(wf_results) < 12:
        print("[ERROR] Insufficient walk-forward results. Check data coverage.")
        return

    # ── 4. Equity curves ──────────────────────────────────────
    print("\n[PERF] Building equity curves...")
    ml_curve  = build_equity_curve(wf_results, "pred_label")
    spy_curve = build_baseline(wf_results, "SPY")
    ew_curve  = build_equal_weight_baseline(wf_results)
    mtum_curve= build_baseline(wf_results, "MTUM")

    # ── 5. Metrics ────────────────────────────────────────────
    wr_vs_spy = win_rate_vs_spy(wf_results)

    ml_m   = compute_metrics(ml_curve,   "ML Factor Rotation")
    spy_m  = compute_metrics(spy_curve,  "SPY B&H")
    ew_m   = compute_metrics(ew_curve,   "Equal Weight Factors")
    mtum_m = compute_metrics(mtum_curve, "MTUM B&H")

    ml_m["wr_months_beat_spy"]   = wr_vs_spy
    spy_m["wr_months_beat_spy"]  = 0.5
    ew_m["wr_months_beat_spy"]   = win_rate_vs_spy(
        [{**r, "pred_label": "EW"} for r in wf_results])  # placeholder
    mtum_m["wr_months_beat_spy"] = win_rate_vs_spy(
        [{**r, "pred_label": "MTUM"} for r in wf_results])

    print("\n── PERFORMANCE SUMMARY ──────────────────────────────")
    for m in [ml_m, spy_m, ew_m, mtum_m]:
        print(f"\n  {m['label']}")
        print(f"    Sharpe:   {m['sharpe']:.3f}")
        print(f"    Sortino:  {m['sortino']:.3f}")
        print(f"    CAGR:     {m['cagr']:.1f}%")
        print(f"    MaxDD:    {m['max_dd']:.1f}%")
        print(f"    Calmar:   {m['calmar']:.3f}")
        print(f"    Turnover: {m.get('turnover', 0):.2f} (monthly churn rate)")
        print(f"    WR vs SPY:{m.get('wr_months_beat_spy', 0):.3f}")

    # ── 6. Adversarial tests ──────────────────────────────────
    print("\n[ADV] Running permutation test...")
    perm_result = permutation_test(wf_results)
    print(f"  Observed Sharpe: {perm_result['observed_sharpe']:.3f}")
    print(f"  Perm p95 Sharpe: {perm_result['perm_p95_sharpe']:.3f}")
    print(f"  p-value: {perm_result['p_value']:.4f}  → {perm_result['verdict']}")

    print("\n[ADV] Running sub-period test...")
    sub_results = sub_period_test(wf_results)
    n_sub_beats = sum(1 for s in sub_results if s["ml_beats_spy"])
    for s in sub_results:
        beat_str = "BEAT" if s["ml_beats_spy"] else "MISS"
        print(f"  {s['period']}: ML Sharpe={s['ml_sharpe']:.2f}, "
              f"SPY={s['spy_sharpe']:.2f} [{beat_str}]")
    print(f"  ML beats SPY in {n_sub_beats}/4 sub-periods")

    print("\n[ADV] Running outlier removal test...")
    outlier_result = outlier_removal_test(wf_results)
    print(f"  Removed {outlier_result['pct_removed']:.1f}% of months")
    print(f"  ML Sharpe after trim: {outlier_result['ml_sharpe']:.3f}")
    print(f"  SPY Sharpe after trim: {outlier_result['spy_sharpe']:.3f}")
    print(f"  Verdict: {outlier_result['verdict']}")

    print("\n[ADV] Regime test (HC #428 R1)...")
    regime_result = regime_test(wf_results)
    print(f"  Regime Sharpes: {regime_result['regime_sharpes']}")
    print(f"  Max pairwise disparity: {regime_result['max_disparity']:.3f} "
          f"(threshold 0.50)")
    print(f"  Verdict: {regime_result['verdict']}")

    # ── 7. Factor allocation summary ─────────────────────────
    picks = [r["pred_label"] for r in wf_results]
    pick_counts = defaultdict(int)
    for p in picks:
        pick_counts[p] += 1

    print("\n[PICKS] ML Factor allocation history:")
    for k, v in sorted(pick_counts.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v} months ({v/len(picks)*100:.1f}%)")

    # ── 8. Save results ───────────────────────────────────────
    output = {
        "meta": {
            "run_date":   dt.datetime.now().isoformat(),
            "start_date": START_DATE,
            "end_date":   END_DATE,
            "initial_capital": INITIAL_CAPITAL,
            "universe":   FACTOR_ETFS + [CASH_ASSET],
            "train_months": TRAIN_MONTHS,
            "n_wf_periods": len(wf_results),
        },
        "performance": {
            "ml_factor_rotation": ml_m,
            "spy_bah":            spy_m,
            "equal_weight":       ew_m,
            "mtum_bah":           mtum_m,
        },
        "adversarial": {
            "permutation":   perm_result,
            "sub_period":    sub_results,
            "sub_beats":     n_sub_beats,
            "outlier_removal": outlier_result,
            "regime":        regime_result,
        },
        "factor_allocation": {k: int(v) for k, v in pick_counts.items()},
        "monthly_detail": [
            {
                "date":        r["date"].strftime("%Y-%m"),
                "pred_label":  r["pred_label"],
                "actual_label": r["actual_label"],
                "correct":     r["correct"],
                "ml_ret":      round(r["ret_lookup"].get(r["pred_label"], 0.0), 5),
                "spy_ret":     round(r["ret_lookup"].get("SPY", 0.0), 5),
            }
            for r in wf_results
        ],
    }

    out_path = OUTPUT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n[SAVE] Results saved to {out_path}")

    # ── 9. Final verdict ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL VERDICT")
    print("=" * 60)
    print(f"  ML Sharpe:   {ml_m['sharpe']:.3f}  "
          f"(target: beat 1.26 from v4.4+ML)")
    print(f"  SPY Sharpe:  {spy_m['sharpe']:.3f}")
    print(f"  Permutation: {perm_result['verdict']} "
          f"(p={perm_result['p_value']:.3f})")
    print(f"  Regime R1:   {regime_result['verdict']}")
    print(f"  Sub-periods: {n_sub_beats}/4 beat SPY")

    all_pass = (
        perm_result["verdict"] == "PASS"
        and regime_result["verdict"] in ("PASS", "INSUFFICIENT_DATA")
        and n_sub_beats >= 2
    )
    overall = "EDGE CONFIRMED" if all_pass else "NEEDS REVIEW"
    print(f"\n  OVERALL: {overall}")
    output["overall_verdict"] = overall

    # Re-save with verdict
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    return output


if __name__ == "__main__":
    main()
