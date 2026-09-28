#!/usr/bin/env python3
"""
Realized Volatility Forecaster — LGBM Walk-Forward
====================================================
Predicts 21-day forward annualized realized vol for the wheel universe.
Walk-forward: 252d train, 21d OOS, sliding window.

Usage:
    python3 /home/jupiter/Lvl3Quant/research/vol_forecaster.py

Output:
    /home/jupiter/Lvl3Quant/output/vol_forecaster/
        - wf_predictions.parquet    (all OOS predictions)
        - wf_metrics.json           (aggregate metrics)
        - feature_importance.parquet
        - summary.txt               (human-readable report)
"""

import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ──────────────────────────────────────────────────────────────────
CACHE = Path("/home/jupiter/Lvl3Quant/wheel_strategy_v1/data/cache")
OUTPUT = Path("/home/jupiter/Lvl3Quant/output/vol_forecaster")
OUTPUT.mkdir(parents=True, exist_ok=True)

ANNUALIZE = np.sqrt(252)
TARGET_HORIZON = 21  # trading days
TRAIN_WINDOW = 252
OOS_STEP = 21


# ── Data Loading ───────────────────────────────────────────────────────────

def load_prices() -> pd.DataFrame:
    """Load price data with basic columns."""
    df = pd.read_parquet(CACHE / "prices.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


def load_macro() -> pd.DataFrame:
    """Load VIX + term structure."""
    m = pd.read_parquet(CACHE / "macro.parquet")
    m["date"] = pd.to_datetime(m["date"])
    m = m[["date", "vix", "vix3m", "vix_ts"]].drop_duplicates("date")
    return m


def load_macro_extra() -> pd.DataFrame:
    """Load yield curve and macro extras."""
    me = pd.read_parquet(CACHE / "macro_extra.parquet")
    me["date"] = pd.to_datetime(me["date"])
    cols = ["date", "ust_10y", "ust_2y", "yc_2s10s"]
    me = me[[c for c in cols if c in me.columns]].drop_duplicates("date")
    return me


def load_sectors() -> pd.DataFrame:
    """Load sector mapping from fundamentals."""
    f = pd.read_parquet(CACHE / "fundamentals.parquet")
    sectors = f[["ticker", "sector"]].drop_duplicates("ticker")
    return sectors


def load_earnings() -> pd.DataFrame:
    """Load earnings dates."""
    e = pd.read_parquet(CACHE / "earnings_dates.parquet")
    e["earnings_date"] = pd.to_datetime(e["earnings_date"])
    return e


# ── Feature Engineering ────────────────────────────────────────────────────

def compute_vol_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-ticker vol and return features from price data."""
    records = []
    for ticker, g in df.groupby("ticker"):
        g = g.sort_values("date").copy()

        # Log returns (use existing or compute)
        if "log_ret" in g.columns:
            g["lr"] = g["log_ret"]
        else:
            g["lr"] = np.log(g["close"] / g["close"].shift(1))

        # Realized vol at multiple windows (annualized)
        for w in [5, 10, 20, 60]:
            g[f"rv_{w}d"] = g["lr"].rolling(w).std() * ANNUALIZE

        # Vol ratios (regime indicators)
        g["vol_ratio_5_20"] = g["rv_5d"] / g["rv_20d"]
        g["vol_ratio_20_60"] = g["rv_20d"] / g["rv_60d"]

        # Vol-of-vol: rolling std of rv_20d
        g["vol_of_vol_20"] = g["rv_20d"].rolling(20).std()

        # Exponentially weighted vol
        g["ewm_vol_10"] = g["lr"].ewm(span=10).std() * ANNUALIZE
        g["ewm_vol_20"] = g["lr"].ewm(span=20).std() * ANNUALIZE

        # Log returns at multiple horizons
        g["ret_1d"] = g["lr"]
        g["ret_5d"] = g["lr"].rolling(5).sum()
        g["ret_20d"] = g["lr"].rolling(20).sum()

        # Absolute return (recent shock indicator)
        g["abs_ret_1d"] = g["lr"].abs()
        g["abs_ret_5d_mean"] = g["abs_ret_1d"].rolling(5).mean()

        # Volume features
        g["vol_20d_avg"] = g["volume"].rolling(20).mean()
        g["volume_ratio"] = g["volume"] / g["vol_20d_avg"]

        # High-low range as vol proxy
        g["hl_range"] = np.log(g["high"] / g["low"])
        g["parkinson_vol"] = g["hl_range"].rolling(20).apply(
            lambda x: np.sqrt((1 / (4 * np.log(2))) * (x**2).mean()) * ANNUALIZE,
            raw=True,
        )

        # Forward realized vol (TARGET)
        g["fwd_rv_21d"] = (
            g["lr"]
            .shift(-1)  # start from next day
            .rolling(TARGET_HORIZON)
            .std()
            .shift(-TARGET_HORIZON + 1)  # align to today
            * ANNUALIZE
        )

        g["ticker"] = ticker
        records.append(g)

    return pd.concat(records, ignore_index=True)


def add_earnings_proximity(df: pd.DataFrame, earnings: pd.DataFrame) -> pd.DataFrame:
    """Add days-to-next-earnings feature."""
    df = df.copy()
    df["days_to_earnings"] = np.nan

    for ticker in df["ticker"].unique():
        mask = df["ticker"] == ticker
        edates = earnings.loc[earnings["ticker"] == ticker, "earnings_date"].sort_values()
        if edates.empty:
            continue

        ticker_dates = df.loc[mask, "date"].values
        edates_arr = edates.values

        # For each date, find days to next earnings
        idx = np.searchsorted(edates_arr, ticker_dates, side="left")
        idx = np.clip(idx, 0, len(edates_arr) - 1)

        days_fwd = (edates_arr[idx] - ticker_dates).astype("timedelta64[D]").astype(float)
        # If the found date is in the past, set NaN (no upcoming earnings found)
        days_fwd[days_fwd < 0] = np.nan

        df.loc[mask, "days_to_earnings"] = days_fwd

    return df


def build_panel(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    macro_extra: pd.DataFrame,
    sectors: pd.DataFrame,
    earnings: pd.DataFrame,
) -> pd.DataFrame:
    """Build the full feature panel."""
    print("Computing vol features...")
    df = compute_vol_features(prices)

    # Merge macro
    print("Merging macro data...")
    df = df.merge(macro, on="date", how="left")
    df = df.merge(macro_extra, on="date", how="left")

    # Sector encoding (label encode)
    print("Adding sector features...")
    df = df.merge(sectors, on="ticker", how="left")
    df["sector"] = df["sector"].fillna("Unknown")
    sector_map = {s: i for i, s in enumerate(sorted(df["sector"].unique()))}
    df["sector_code"] = df["sector"].map(sector_map)

    # Earnings proximity
    print("Adding earnings proximity...")
    df = add_earnings_proximity(df, earnings)

    return df


# ── Walk-Forward Engine ────────────────────────────────────────────────────

FEATURE_COLS = [
    "rv_5d", "rv_10d", "rv_20d", "rv_60d",
    "vol_ratio_5_20", "vol_ratio_20_60", "vol_of_vol_20",
    "ewm_vol_10", "ewm_vol_20",
    "ret_1d", "ret_5d", "ret_20d",
    "abs_ret_1d", "abs_ret_5d_mean",
    "vol_20d_avg", "volume_ratio",
    "parkinson_vol",
    "vix", "vix3m", "vix_ts",
    "ust_10y", "ust_2y", "yc_2s10s",
    "sector_code",
    "days_to_earnings",
]

LGBM_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": 7,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "verbose": -1,
    "n_jobs": -1,
    "seed": 42,
}


def walk_forward(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """
    Walk-forward LGBM training.
    252d train, 21d OOS, sliding window.
    Returns (predictions_df, metrics_dict, feature_importance_df).
    """
    # Get sorted unique dates
    all_dates = sorted(panel["date"].unique())
    n_dates = len(all_dates)

    # Drop rows where target is NaN
    valid = panel.dropna(subset=["fwd_rv_21d"]).copy()
    valid_dates = sorted(valid["date"].unique())
    n_valid = len(valid_dates)

    print(f"Total dates: {n_dates}, Valid dates (with target): {n_valid}")

    # Walk-forward splits
    oos_preds = []
    importances = []
    fold = 0

    import time as _time
    total_folds = (n_valid - TRAIN_WINDOW) // OOS_STEP
    print(f"Expected folds: ~{total_folds}", flush=True)

    # Pre-index for faster filtering
    valid_date_idx = valid.set_index("date")

    start_idx = TRAIN_WINDOW  # first OOS window starts after train window
    while start_idx + OOS_STEP <= n_valid:
        t0 = _time.time()
        train_dates = valid_dates[start_idx - TRAIN_WINDOW : start_idx]
        oos_dates = valid_dates[start_idx : start_idx + OOS_STEP]

        # Use .loc with date index for speed
        train_df = valid_date_idx.loc[valid_date_idx.index.isin(train_dates)]
        oos_df = valid_date_idx.loc[valid_date_idx.index.isin(oos_dates)]

        X_train = train_df[FEATURE_COLS]
        y_train = train_df["fwd_rv_21d"]
        X_oos = oos_df[FEATURE_COLS]
        y_oos = oos_df["fwd_rv_21d"]

        if len(X_train) < 100 or len(X_oos) == 0:
            start_idx += OOS_STEP
            continue

        # Train
        ds_train = lgb.Dataset(X_train, label=y_train)
        model = lgb.train(
            LGBM_PARAMS,
            ds_train,
            num_boost_round=500,
        )

        # Predict
        preds = model.predict(X_oos)

        oos_slice = oos_df[["ticker", "fwd_rv_21d", "rv_20d", "ewm_vol_20"]].copy()
        oos_slice["date"] = oos_slice.index
        oos_slice["pred_vol"] = preds
        oos_slice["fold"] = fold
        oos_preds.append(oos_slice.reset_index(drop=True))

        # Feature importance
        imp = pd.DataFrame({
            "feature": FEATURE_COLS,
            "importance": model.feature_importance(importance_type="gain"),
            "fold": fold,
        })
        importances.append(imp)

        fold += 1
        start_idx += OOS_STEP
        elapsed = _time.time() - t0

        if fold % 5 == 0:
            print(f"  Fold {fold}/{total_folds}: {elapsed:.1f}s, "
                  f"train={len(X_train)}, oos={len(X_oos)}", flush=True)

        if fold == 1:
            print(f"  Fold 1 took {elapsed:.1f}s — est total: {elapsed * total_folds / 60:.1f} min", flush=True)

    print(f"  Fold {fold}: train {str(train_dates[0])[:10]}"
                  f"→{str(train_dates[-1])[:10]}, "
                  f"OOS {str(oos_dates[0])[:10]}"
                  f"→{str(oos_dates[-1])[:10]}")

    print(f"Completed {fold} walk-forward folds")

    if not oos_preds:
        raise RuntimeError("No OOS predictions generated — check data availability")

    preds_df = pd.concat(oos_preds, ignore_index=True)
    imp_df = pd.concat(importances, ignore_index=True)

    # Aggregate feature importance
    imp_agg = imp_df.groupby("feature")["importance"].mean().sort_values(ascending=False).reset_index()

    # Compute metrics
    metrics = compute_metrics(preds_df)

    return preds_df, metrics, imp_agg


# ── Metrics ────────────────────────────────────────────────────────────────

def compute_metrics(preds_df: pd.DataFrame) -> dict:
    """Compute MAE, RMSE, directional accuracy, IC vs baselines."""
    actual = preds_df["fwd_rv_21d"]
    predicted = preds_df["pred_vol"]
    naive_20d = preds_df["rv_20d"]
    naive_ewm = preds_df["ewm_vol_20"]

    metrics = {}

    # --- LGBM metrics ---
    metrics["lgbm_mae"] = float(np.mean(np.abs(actual - predicted)))
    metrics["lgbm_rmse"] = float(np.sqrt(np.mean((actual - predicted) ** 2)))

    # --- Naive 20d trailing vol ---
    metrics["naive_20d_mae"] = float(np.mean(np.abs(actual - naive_20d)))
    metrics["naive_20d_rmse"] = float(np.sqrt(np.mean((actual - naive_20d) ** 2)))

    # --- Naive EWM vol ---
    metrics["naive_ewm_mae"] = float(np.mean(np.abs(actual - naive_ewm)))
    metrics["naive_ewm_rmse"] = float(np.sqrt(np.mean((actual - naive_ewm) ** 2)))

    # --- MAE improvement ---
    metrics["mae_improvement_vs_20d_pct"] = float(
        (metrics["naive_20d_mae"] - metrics["lgbm_mae"]) / metrics["naive_20d_mae"] * 100
    )
    metrics["mae_improvement_vs_ewm_pct"] = float(
        (metrics["naive_ewm_mae"] - metrics["lgbm_mae"]) / metrics["naive_ewm_mae"] * 100
    )

    # --- Directional accuracy ---
    # Did vol go up or down relative to current 20d rv?
    actual_direction = (actual > naive_20d).astype(int)
    pred_direction = (predicted > naive_20d).astype(int)
    naive_direction = 0  # naive always predicts "stay same" (direction = 0)

    metrics["lgbm_directional_accuracy"] = float((actual_direction == pred_direction).mean())
    # Naive baseline: always predict "no change" = current 20d rv
    metrics["naive_directional_accuracy"] = float(
        (actual_direction == 0).mean()  # naive says "vol stays same"
    )

    # --- Cross-sectional IC (rank correlation) per date ---
    ics_lgbm = []
    ics_naive = []
    for dt, g in preds_df.groupby("date"):
        if len(g) < 5:
            continue
        ic_l, _ = spearmanr(g["pred_vol"], g["fwd_rv_21d"])
        ic_n, _ = spearmanr(g["rv_20d"], g["fwd_rv_21d"])
        if not np.isnan(ic_l):
            ics_lgbm.append(ic_l)
        if not np.isnan(ic_n):
            ics_naive.append(ic_n)

    metrics["lgbm_mean_ic"] = float(np.mean(ics_lgbm)) if ics_lgbm else 0.0
    metrics["lgbm_ic_std"] = float(np.std(ics_lgbm)) if ics_lgbm else 0.0
    metrics["lgbm_icir"] = (
        float(np.mean(ics_lgbm) / np.std(ics_lgbm))
        if ics_lgbm and np.std(ics_lgbm) > 0 else 0.0
    )
    metrics["naive_mean_ic"] = float(np.mean(ics_naive)) if ics_naive else 0.0

    metrics["n_predictions"] = len(preds_df)
    metrics["n_oos_dates"] = int(preds_df["date"].nunique())
    metrics["n_tickers"] = int(preds_df["ticker"].nunique())

    return metrics


# ── Reporting ──────────────────────────────────────────────────────────────

def print_report(metrics: dict, imp_df: pd.DataFrame) -> str:
    """Generate human-readable summary report."""
    lines = [
        "=" * 70,
        "REALIZED VOLATILITY FORECASTER — WALK-FORWARD RESULTS",
        "=" * 70,
        f"Predictions: {metrics['n_predictions']:,} "
        f"({metrics['n_tickers']} tickers × {metrics['n_oos_dates']} OOS dates)",
        "",
        "─── Accuracy (lower = better) ───",
        f"  {'Model':<20} {'MAE':>10} {'RMSE':>10}",
        f"  {'LGBM':<20} {metrics['lgbm_mae']:>10.4f} {metrics['lgbm_rmse']:>10.4f}",
        f"  {'Naive 20d RV':<20} {metrics['naive_20d_mae']:>10.4f} {metrics['naive_20d_rmse']:>10.4f}",
        f"  {'Naive EWM':<20} {metrics['naive_ewm_mae']:>10.4f} {metrics['naive_ewm_rmse']:>10.4f}",
        "",
        f"  MAE improvement vs 20d trailing: {metrics['mae_improvement_vs_20d_pct']:+.1f}%",
        f"  MAE improvement vs EWM:          {metrics['mae_improvement_vs_ewm_pct']:+.1f}%",
        "",
        "─── Directional Accuracy ───",
        f"  LGBM:  {metrics['lgbm_directional_accuracy']:.1%}",
        f"  Naive: {metrics['naive_directional_accuracy']:.1%}",
        "",
        "─── Cross-Sectional IC (rank correlation) ───",
        f"  LGBM mean IC:  {metrics['lgbm_mean_ic']:.4f} (std={metrics['lgbm_ic_std']:.4f}, ICIR={metrics['lgbm_icir']:.2f})",
        f"  Naive mean IC: {metrics['naive_mean_ic']:.4f}",
        "",
        "─── Top 10 Features (gain importance) ───",
    ]
    for _, row in imp_df.head(10).iterrows():
        lines.append(f"  {row['feature']:<25} {row['importance']:>12.1f}")

    lines.extend(["", "=" * 70])
    report = "\n".join(lines)
    return report


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    prices = load_prices()
    macro = load_macro()
    macro_extra = load_macro_extra()
    sectors = load_sectors()
    earnings = load_earnings()

    panel = build_panel(prices, macro, macro_extra, sectors, earnings)

    # Drop rows with insufficient feature data
    min_feature_cols = ["rv_60d", "vol_of_vol_20", "ret_20d", "parkinson_vol"]
    before = len(panel)
    panel = panel.dropna(subset=min_feature_cols)
    print(f"Dropped {before - len(panel):,} rows with missing core features "
          f"({len(panel):,} remaining)")

    print(f"\nStarting walk-forward (train={TRAIN_WINDOW}d, OOS={OOS_STEP}d, sliding)...")
    preds_df, metrics, imp_df = walk_forward(panel)

    # Report
    report = print_report(metrics, imp_df)
    print(report)

    # Save outputs
    preds_df.to_parquet(OUTPUT / "wf_predictions.parquet", index=False)
    imp_df.to_parquet(OUTPUT / "feature_importance.parquet", index=False)
    with open(OUTPUT / "wf_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(OUTPUT / "summary.txt", "w") as f:
        f.write(report)

    print(f"\nResults saved to {OUTPUT}/")


if __name__ == "__main__":
    main()
