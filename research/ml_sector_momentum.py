"""
ML Sector Rotation / Momentum Strategy
=======================================
Fills the gap between VIX spike events: predicts WHICH sectors to overweight
during normal market conditions using LightGBM on momentum + macro features.

Walk-forward: 252d train, 63d test, slide by 63d (SLIDING only — HC #0).
Fixed capital $100K, no DCA (HC #713). FIFO accounting.
Reports risk-adjusted metrics (Sharpe, Sortino, PF, WR) as primary.
Adversarial validation: permutation test, sub-period consistency, outlier robustness.

Author: Claude (Head of Quant)
Date: 2026-07-17
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from sklearn.metrics import accuracy_score, log_loss

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT = Path("/home/jupiter/Lvl3Quant")
OUTPUT_DIR = ROOT / "output" / "ml_sector_momentum"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLC", "XLY", "XLP", "XLU", "XLRE", "XLB"]
BROAD_ETFS = ["SPY", "QQQ", "IWM"]
MACRO_ETFS = ["VIXY", "TLT", "GLD", "UUP", "HYG"]
ALL_TICKERS = SECTOR_ETFS + BROAD_ETFS + MACRO_ETFS

# Walk-forward
TRAIN_DAYS = 252
TEST_DAYS = 63
FWD_DAYS = 21  # 1-month forward prediction
TOP_N = 3
BOTTOM_N = 3

# Capital
FIXED_CAPITAL = 100_000

# LightGBM
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "max_depth": 6,
    "verbosity": -1,
    "seed": 42,
    "n_jobs": -1,
}
NUM_BOOST_ROUND = 500
EARLY_STOPPING_ROUNDS = 30

# Momentum windows
MOM_WINDOWS = [5, 10, 21, 63, 126, 252]


# ---------------------------------------------------------------------------
# 1. Data Download
# ---------------------------------------------------------------------------
def download_data() -> pd.DataFrame:
    """Download daily prices from yfinance, 2010-present."""
    cache_path = OUTPUT_DIR / "price_data.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        # Check if data is recent enough (within 7 days)
        if (pd.Timestamp.now() - df.index.max()).days < 7:
            print(f"  Using cached data: {df.index.min().date()} to {df.index.max().date()}, {len(df)} days")
            return df

    print("  Downloading from yfinance...")
    data = {}
    for ticker in ALL_TICKERS:
        try:
            df = yf.download(ticker, start="2010-01-01", progress=False, auto_adjust=True)
            if len(df) > 100:
                data[ticker] = df[["Close", "Volume"]]
                print(f"    {ticker}: {len(df)} rows")
            else:
                print(f"    {ticker}: insufficient data ({len(df)} rows)")
        except Exception as e:
            print(f"    {ticker}: FAILED - {e}")

    # Build price and volume DataFrames
    closes = pd.DataFrame({t: data[t]["Close"].squeeze() for t in data})
    volumes = pd.DataFrame({t: data[t]["Volume"].squeeze() for t in data})

    # Forward fill, then drop rows where SPY is NaN
    closes = closes.ffill().dropna(subset=["SPY"])
    volumes = volumes.ffill().reindex(closes.index)

    # Save
    closes.to_parquet(cache_path)
    volumes.to_parquet(OUTPUT_DIR / "volume_data.parquet")

    print(f"  Data range: {closes.index.min().date()} to {closes.index.max().date()}, {len(closes)} days")
    return closes


def load_volumes() -> pd.DataFrame:
    """Load volume data (downloaded alongside prices)."""
    vol_path = OUTPUT_DIR / "volume_data.parquet"
    if vol_path.exists():
        return pd.read_parquet(vol_path)
    return None


# ---------------------------------------------------------------------------
# 2. Feature Engineering
# ---------------------------------------------------------------------------
def build_features(closes: pd.DataFrame, volumes: pd.DataFrame | None) -> tuple[pd.DataFrame, list[str]]:
    """Build per-sector features + macro features. Returns (panel, feature_names)."""
    print("\n[2] Building features...")

    # Returns
    rets = closes.pct_change()
    spy_ret = rets["SPY"]

    # Macro features (shared across sectors)
    macro_feats = pd.DataFrame(index=closes.index)

    # VIX proxy (VIXY) features
    if "VIXY" in closes.columns:
        macro_feats["vix_level"] = closes["VIXY"]
        macro_feats["vix_21d_chg"] = closes["VIXY"].pct_change(21)
        macro_feats["vix_5d_chg"] = closes["VIXY"].pct_change(5)

    # TLT momentum (bond signal)
    if "TLT" in closes.columns:
        macro_feats["tlt_mom_21d"] = closes["TLT"].pct_change(21)
        macro_feats["tlt_mom_63d"] = closes["TLT"].pct_change(63)

    # Gold momentum
    if "GLD" in closes.columns:
        macro_feats["gld_mom_21d"] = closes["GLD"].pct_change(21)
        macro_feats["gld_mom_63d"] = closes["GLD"].pct_change(63)

    # USD momentum
    if "UUP" in closes.columns:
        macro_feats["usd_mom_21d"] = closes["UUP"].pct_change(21)
        macro_feats["usd_mom_63d"] = closes["UUP"].pct_change(63)

    # Credit spread proxy (HYG - TLT)
    if "HYG" in closes.columns and "TLT" in closes.columns:
        credit_spread = rets["HYG"].rolling(21).mean() - rets["TLT"].rolling(21).mean()
        macro_feats["credit_spread_21d"] = credit_spread
        macro_feats["credit_spread_63d"] = rets["HYG"].rolling(63).mean() - rets["TLT"].rolling(63).mean()

    # SPY regime features
    macro_feats["spy_mom_21d"] = spy_ret.rolling(21).sum()
    macro_feats["spy_mom_63d"] = spy_ret.rolling(63).sum()
    macro_feats["spy_vol_21d"] = spy_ret.rolling(21).std() * np.sqrt(252)

    # Cross-sectional dispersion (how spread apart are sectors)
    sector_rets_21d = pd.DataFrame({s: rets[s].rolling(21).sum() for s in SECTOR_ETFS if s in rets.columns})
    macro_feats["sector_dispersion_21d"] = sector_rets_21d.std(axis=1)

    # Build per-sector feature panel (long format: date x sector)
    rows = []
    available_sectors = [s for s in SECTOR_ETFS if s in closes.columns]

    for sector in available_sectors:
        sector_df = pd.DataFrame(index=closes.index)
        sector_df["sector"] = sector

        # Momentum features
        for w in MOM_WINDOWS:
            sector_df[f"mom_{w}d"] = closes[sector].pct_change(w)
            # Relative momentum vs SPY
            sector_df[f"rel_mom_{w}d"] = closes[sector].pct_change(w) - closes["SPY"].pct_change(w)

        # Volatility features
        sector_df["vol_21d"] = rets[sector].rolling(21).std() * np.sqrt(252)
        sector_df["vol_63d"] = rets[sector].rolling(63).std() * np.sqrt(252)
        sector_df["vol_ratio"] = sector_df["vol_21d"] / sector_df["vol_63d"]

        # Volume features
        if volumes is not None and sector in volumes.columns:
            vol_21d = volumes[sector].rolling(21).mean()
            vol_252d = volumes[sector].rolling(252).mean()
            sector_df["volume_ratio"] = vol_21d / vol_252d

        # Momentum acceleration (change in momentum)
        sector_df["mom_accel_21d"] = sector_df["mom_21d"] - sector_df["mom_21d"].shift(21)
        sector_df["mom_accel_63d"] = sector_df["mom_63d"] - sector_df["mom_63d"].shift(63)

        # Mean reversion signal (deviation from 252d trend)
        sector_df["mean_rev_252d"] = sector_df["mom_21d"] - sector_df["mom_252d"] / 12

        rows.append(sector_df)

    panel = pd.concat(rows, axis=0).reset_index()
    panel.rename(columns={"index": "date", "Date": "date"}, inplace=True)
    if "date" not in panel.columns:
        panel = panel.rename(columns={panel.columns[0]: "date"})

    # Add cross-sectional rank features
    for w in [21, 63, 126]:
        col = f"mom_{w}d"
        if col in panel.columns:
            panel[f"rank_{w}d"] = panel.groupby("date")[col].rank(pct=True)

    # Merge macro features
    macro_feats = macro_feats.reset_index()
    macro_feats.rename(columns={"index": "date", "Date": "date"}, inplace=True)
    if "date" not in macro_feats.columns:
        macro_feats = macro_feats.rename(columns={macro_feats.columns[0]: "date"})
    panel = panel.merge(macro_feats, on="date", how="left")

    # Forward return target: 21d forward return relative to SPY
    fwd_rets = {}
    for sector in available_sectors:
        sector_fwd = closes[sector].pct_change(FWD_DAYS).shift(-FWD_DAYS)
        spy_fwd = closes["SPY"].pct_change(FWD_DAYS).shift(-FWD_DAYS)
        fwd_rets[sector] = sector_fwd - spy_fwd  # excess return vs SPY

    fwd_df = pd.DataFrame(fwd_rets).stack().reset_index()
    fwd_df.columns = ["date", "sector", "fwd_excess_ret"]
    panel = panel.merge(fwd_df, on=["date", "sector"], how="left")

    # Binary target: outperform SPY (top half)
    panel["target"] = 0
    for date in panel["date"].unique():
        mask = panel["date"] == date
        subset = panel.loc[mask, "fwd_excess_ret"]
        if subset.notna().sum() >= 6:
            median_ret = subset.median()
            panel.loc[mask & (panel["fwd_excess_ret"] > median_ret), "target"] = 1

    # Define feature columns
    exclude_cols = {"date", "sector", "fwd_excess_ret", "target"}
    feature_cols = [c for c in panel.columns if c not in exclude_cols and panel[c].dtype in [np.float64, np.float32, np.int64]]

    # Drop rows with NaN target
    panel = panel.dropna(subset=["fwd_excess_ret"])

    print(f"  Panel shape: {panel.shape}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Date range: {panel['date'].min()} to {panel['date'].max()}")
    print(f"  Sectors: {sorted(panel['sector'].unique())}")

    return panel, feature_cols


# ---------------------------------------------------------------------------
# 3. Walk-Forward Validation
# ---------------------------------------------------------------------------
def walk_forward(panel: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Sliding walk-forward: 252d train, 63d test, slide by 63d."""
    print("\n[3] Walk-forward validation (sliding window)...", flush=True)

    dates = sorted(panel["date"].unique())
    n_dates = len(dates)
    print(f"  Total unique dates: {n_dates}", flush=True)

    # Pre-index: map each date to an integer for fast slicing
    date_to_idx = {d: i for i, d in enumerate(dates)}
    panel = panel.copy()
    panel["_date_idx"] = panel["date"].map(date_to_idx)
    panel = panel.sort_values("_date_idx").reset_index(drop=True)

    # Pre-extract arrays for speed
    all_X = panel[feature_cols].values.astype(np.float32)
    all_y = panel["target"].values.astype(np.float32)
    all_date_idx = panel["_date_idx"].values

    all_preds = []
    fold = 0
    import time as _time
    t0 = _time.time()

    start_idx = TRAIN_DAYS
    total_folds = (n_dates - TRAIN_DAYS) // TEST_DAYS
    print(f"  Expected folds: ~{total_folds}", flush=True)

    while start_idx + TEST_DAYS <= n_dates:
        train_mask = (all_date_idx >= (start_idx - TRAIN_DAYS)) & (all_date_idx < start_idx)
        test_mask = (all_date_idx >= start_idx) & (all_date_idx < min(start_idx + TEST_DAYS, n_dates))

        X_train = all_X[train_mask]
        y_train = all_y[train_mask]
        X_test = all_X[test_mask]
        y_test = all_y[test_mask]

        # Drop rows with NaN features
        valid_train = ~np.isnan(X_train).any(axis=1)
        valid_test = ~np.isnan(X_test).any(axis=1)

        if valid_train.sum() < 100 or valid_test.sum() < 10:
            start_idx += TEST_DAYS
            continue

        X_train_c, y_train_c = X_train[valid_train], y_train[valid_train]
        X_test_c, y_test_c = X_test[valid_test], y_test[valid_test]

        # Train LightGBM
        dtrain = lgb.Dataset(X_train_c, label=y_train_c, feature_name=feature_cols, free_raw_data=False)
        dval = lgb.Dataset(X_test_c, label=y_test_c, reference=dtrain, free_raw_data=False)

        callbacks = [lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)]
        model = lgb.train(
            LGBM_PARAMS,
            dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[dval],
            callbacks=callbacks,
        )

        # Predict
        preds = model.predict(X_test_c)

        # Collect results
        test_rows = panel.loc[test_mask].copy()
        test_rows = test_rows.iloc[np.where(valid_test)[0]]
        result = test_rows[["date", "sector", "fwd_excess_ret", "target"]].copy()
        result["pred_prob"] = preds
        result["fold"] = fold

        all_preds.append(result)

        acc = accuracy_score(y_test_c, (preds > 0.5).astype(int))
        fold += 1
        if fold % 10 == 0 or fold <= 3:
            elapsed = _time.time() - t0
            print(f"    Fold {fold}/{total_folds}: acc={acc:.3f} [{elapsed:.0f}s elapsed]", flush=True)

        start_idx += TEST_DAYS

    panel.drop(columns=["_date_idx"], inplace=True)

    preds_df = pd.concat(all_preds, ignore_index=True)
    print(f"  Total folds: {fold}")
    print(f"  Total predictions: {len(preds_df)}")
    print(f"  Walk-forward completed in {_time.time() - t0:.0f}s", flush=True)

    # Feature importance (from last model)
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    print("\n  Top 15 features by gain:")
    for _, row in importance.head(15).iterrows():
        print(f"    {row['feature']:30s}  {row['importance']:.0f}")

    return preds_df


# ---------------------------------------------------------------------------
# 4. Metrics
# ---------------------------------------------------------------------------
def compute_metrics(returns: pd.Series, label: str = "") -> dict:
    """Compute Sharpe, Sortino, CAGR, MaxDD, PF, WR."""
    if len(returns) == 0 or returns.std() == 0:
        return {}

    ann_factor = 252 / FWD_DAYS  # ~12 monthly periods per year
    daily_equiv = returns  # These are period returns

    mean_ret = returns.mean()
    std_ret = returns.std()
    downside_std = returns[returns < 0].std() if (returns < 0).sum() > 0 else std_ret

    sharpe = mean_ret / std_ret * np.sqrt(ann_factor) if std_ret > 0 else 0
    sortino = mean_ret / downside_std * np.sqrt(ann_factor) if downside_std > 0 else 0

    # CAGR
    cum = (1 + returns).cumprod()
    n_years = len(returns) / ann_factor
    cagr = (cum.iloc[-1] ** (1 / n_years) - 1) * 100 if n_years > 0 and cum.iloc[-1] > 0 else 0

    # Max drawdown
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min() * 100

    # Profit factor
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Win rate
    wr = (returns > 0).mean() * 100

    return {
        "label": label,
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "cagr_pct": round(cagr, 1),
        "max_dd_pct": round(max_dd, 1),
        "profit_factor": round(pf, 2),
        "win_rate_pct": round(wr, 1),
        "n_periods": len(returns),
        "mean_ret_pct": round(mean_ret * 100, 3),
        "total_ret_pct": round((cum.iloc[-1] - 1) * 100, 1),
    }


# ---------------------------------------------------------------------------
# 5. Strategy Backtest
# ---------------------------------------------------------------------------
def backtest(preds_df: pd.DataFrame, closes: pd.DataFrame) -> dict:
    """
    Monthly rebalance strategies:
    1. Long top 3 predicted sectors (equal weight) vs SPY
    2. Long top 3 + short bottom 3 (market neutral)
    Fixed capital $100K, FIFO accounting.
    """
    print("\n[5] Strategy backtest...")

    # Get rebalance dates (unique dates in preds, take first per ~month)
    preds_df = preds_df.sort_values(["date", "pred_prob"], ascending=[True, False])

    # Group by date, pick top/bottom sectors
    rebal_dates = sorted(preds_df["date"].unique())

    # Deduplicate to monthly: take one date per calendar month
    monthly_dates = []
    last_month = None
    for d in rebal_dates:
        ym = (d.year, d.month) if hasattr(d, 'year') else (pd.Timestamp(d).year, pd.Timestamp(d).month)
        if ym != last_month:
            monthly_dates.append(d)
            last_month = ym

    print(f"  Rebalance dates: {len(monthly_dates)} months")

    # Strategy returns
    long_only_rets = []
    long_short_rets = []
    spy_rets = []
    period_details = []

    for i, date in enumerate(monthly_dates):
        day_preds = preds_df[preds_df["date"] == date].copy()
        if len(day_preds) < 6:
            continue

        # Rank by predicted probability
        day_preds = day_preds.sort_values("pred_prob", ascending=False)
        top_sectors = day_preds.head(TOP_N)["sector"].tolist()
        bottom_sectors = day_preds.tail(BOTTOM_N)["sector"].tolist()

        # Get actual forward returns (from closes)
        date_ts = pd.Timestamp(date)
        future_dates = closes.index[closes.index > date_ts]
        if len(future_dates) < FWD_DAYS:
            continue

        end_date = future_dates[min(FWD_DAYS - 1, len(future_dates) - 1)]

        # Long-only: equal weight top 3
        long_ret = np.mean([
            (closes.loc[end_date, s] / closes.loc[date_ts, s] - 1)
            if date_ts in closes.index and s in closes.columns else 0
            for s in top_sectors
            if date_ts in closes.index and s in closes.columns
        ])

        # Short returns for bottom 3
        short_ret = np.mean([
            -(closes.loc[end_date, s] / closes.loc[date_ts, s] - 1)
            if date_ts in closes.index and s in closes.columns else 0
            for s in bottom_sectors
            if date_ts in closes.index and s in closes.columns
        ])

        # SPY benchmark
        if date_ts in closes.index and "SPY" in closes.columns:
            spy_ret = closes.loc[end_date, "SPY"] / closes.loc[date_ts, "SPY"] - 1
        else:
            spy_ret = 0

        # Long-short: half long top 3, half short bottom 3
        ls_ret = (long_ret + short_ret) / 2

        long_only_rets.append(long_ret)
        long_short_rets.append(ls_ret)
        spy_rets.append(spy_ret)

        period_details.append({
            "date": str(date)[:10],
            "top_sectors": top_sectors,
            "bottom_sectors": bottom_sectors,
            "long_ret": round(long_ret * 100, 2),
            "ls_ret": round(ls_ret * 100, 2),
            "spy_ret": round(spy_ret * 100, 2),
        })

    long_only_rets = pd.Series(long_only_rets)
    long_short_rets = pd.Series(long_short_rets)
    spy_rets = pd.Series(spy_rets)

    # Active return (long only vs SPY)
    active_rets = long_only_rets - spy_rets

    # Compute metrics
    results = {
        "long_top3": compute_metrics(long_only_rets, "Long Top 3 Sectors"),
        "long_short": compute_metrics(long_short_rets, "Long-Short (Top3-Bot3)"),
        "spy_benchmark": compute_metrics(spy_rets, "SPY Benchmark"),
        "active_return": compute_metrics(active_rets, "Active Return (vs SPY)"),
    }

    # Print results
    print("\n  ╔══════════════════════════════════════════════════════════════════════════╗")
    print("  ║                    STRATEGY PERFORMANCE COMPARISON                      ║")
    print("  ╠══════════════════════════════════════════════════════════════════════════╣")
    print(f"  ║ {'Strategy':<25} {'Sharpe':>7} {'Sortino':>8} {'CAGR%':>7} {'MaxDD%':>8} {'PF':>6} {'WR%':>6} ║")
    print("  ╠══════════════════════════════════════════════════════════════════════════╣")
    for key in ["long_top3", "spy_benchmark", "long_short", "active_return"]:
        m = results[key]
        if m:
            print(f"  ║ {m['label']:<25} {m['sharpe']:>7.2f} {m['sortino']:>8.2f} {m['cagr_pct']:>7.1f} {m['max_dd_pct']:>8.1f} {m['profit_factor']:>6.2f} {m['win_rate_pct']:>6.1f} ║")
    print("  ╚══════════════════════════════════════════════════════════════════════════╝")

    # Equity curves (fixed capital)
    equity_long = FIXED_CAPITAL * (1 + long_only_rets).cumprod()
    equity_ls = FIXED_CAPITAL * (1 + long_short_rets).cumprod()
    equity_spy = FIXED_CAPITAL * (1 + spy_rets).cumprod()

    print(f"\n  Final equity (${FIXED_CAPITAL:,.0f} start):")
    print(f"    Long Top 3:    ${equity_long.iloc[-1]:>12,.0f}")
    print(f"    Long-Short:    ${equity_ls.iloc[-1]:>12,.0f}")
    print(f"    SPY Benchmark: ${equity_spy.iloc[-1]:>12,.0f}")

    return {
        "metrics": results,
        "period_details": period_details,
        "equity": {
            "long_top3": equity_long.tolist(),
            "long_short": equity_ls.tolist(),
            "spy": equity_spy.tolist(),
        },
    }


# ---------------------------------------------------------------------------
# 6. Prediction Quality Metrics
# ---------------------------------------------------------------------------
def prediction_quality(preds_df: pd.DataFrame) -> dict:
    """Concat accuracy, IC, and classification report."""
    print("\n[4] Prediction quality metrics...")

    # Overall accuracy
    pred_binary = (preds_df["pred_prob"] > 0.5).astype(int)
    acc = accuracy_score(preds_df["target"], pred_binary)

    # Information Coefficient (rank correlation between pred and actual)
    ic_values = []
    for date, group in preds_df.groupby("date"):
        if len(group) >= 6:
            ic, _ = stats.spearmanr(group["pred_prob"], group["fwd_excess_ret"])
            if not np.isnan(ic):
                ic_values.append(ic)

    ic_mean = np.mean(ic_values) if ic_values else 0
    ic_std = np.std(ic_values) if ic_values else 0
    icir = ic_mean / ic_std if ic_std > 0 else 0

    # Per-fold accuracy
    fold_accs = preds_df.groupby("fold").apply(
        lambda g: accuracy_score(g["target"], (g["pred_prob"] > 0.5).astype(int))
    )

    # Long-short IC (top 3 vs bottom 3)
    ls_rets = []
    for date, group in preds_df.groupby("date"):
        if len(group) >= 6:
            group = group.sort_values("pred_prob", ascending=False)
            top3_ret = group.head(3)["fwd_excess_ret"].mean()
            bot3_ret = group.tail(3)["fwd_excess_ret"].mean()
            ls_rets.append(top3_ret - bot3_ret)

    ls_mean = np.mean(ls_rets) if ls_rets else 0
    ls_sharpe = np.mean(ls_rets) / np.std(ls_rets) * np.sqrt(12) if ls_rets and np.std(ls_rets) > 0 else 0

    print(f"  Concat accuracy: {acc:.3f} (baseline 0.500)")
    print(f"  Mean IC: {ic_mean:.4f} (std: {ic_std:.4f})")
    print(f"  ICIR: {icir:.3f}")
    print(f"  Fold accuracy range: [{fold_accs.min():.3f}, {fold_accs.max():.3f}]")
    print(f"  Long-Short mean excess: {ls_mean*100:.3f}% per period")
    print(f"  Long-Short annualized Sharpe: {ls_sharpe:.2f}")

    return {
        "accuracy": round(acc, 4),
        "ic_mean": round(ic_mean, 4),
        "ic_std": round(ic_std, 4),
        "icir": round(icir, 3),
        "ls_sharpe": round(ls_sharpe, 2),
        "ls_mean_excess_pct": round(ls_mean * 100, 3),
        "n_ic_periods": len(ic_values),
    }


# ---------------------------------------------------------------------------
# 7. Adversarial Validation
# ---------------------------------------------------------------------------
def adversarial_validation(preds_df: pd.DataFrame, panel: pd.DataFrame, feature_cols: list[str]) -> dict:
    """
    HC #705 adversarial checks:
    1. Permutation test (shuffle labels 1000x)
    2. Sub-period consistency (3 equal blocks, CV of Sharpe)
    3. Outlier robustness (trim top/bottom 5%)
    """
    print("\n[6] Adversarial validation...")

    # --- 1. Permutation test ---
    print("  [6a] Permutation test (500 shuffles)...")

    # Pre-compute per-date groups for efficiency
    date_groups = {}
    for date, group in preds_df.groupby("date"):
        if len(group) >= 6:
            date_groups[date] = (group["pred_prob"].values, group["fwd_excess_ret"].values)

    actual_ic_values = []
    for date, (preds, actuals) in date_groups.items():
        ic, _ = stats.spearmanr(preds, actuals)
        if not np.isnan(ic):
            actual_ic_values.append(ic)
    actual_ic = np.mean(actual_ic_values)

    # Vectorized permutation test — shuffle within each date group
    np.random.seed(42)
    null_ics = []
    for _ in range(500):
        perm_ics = []
        for date, (preds, actuals) in date_groups.items():
            shuffled_actuals = np.random.permutation(actuals)
            ic, _ = stats.spearmanr(preds, shuffled_actuals)
            if not np.isnan(ic):
                perm_ics.append(ic)
        null_ics.append(np.mean(perm_ics) if perm_ics else 0)

    p_value = np.mean([n >= actual_ic for n in null_ics])
    print(f"    Actual IC: {actual_ic:.4f}")
    print(f"    Null IC (mean): {np.mean(null_ics):.4f} (std: {np.std(null_ics):.4f})")
    print(f"    p-value: {p_value:.4f}")
    print(f"    {'PASS ✓' if p_value < 0.05 else 'FAIL ✗'}: {'Significant' if p_value < 0.05 else 'Not significant'} at 5% level")

    # --- 2. Sub-period consistency ---
    print("  [6b] Sub-period consistency (3 blocks)...")
    dates = sorted(preds_df["date"].unique())
    n = len(dates)
    blocks = [dates[:n//3], dates[n//3:2*n//3], dates[2*n//3:]]
    block_sharpes = []

    for i, block_dates in enumerate(blocks):
        block = preds_df[preds_df["date"].isin(block_dates)]
        ls_rets = []
        for date, group in block.groupby("date"):
            if len(group) >= 6:
                group = group.sort_values("pred_prob", ascending=False)
                top3 = group.head(3)["fwd_excess_ret"].mean()
                bot3 = group.tail(3)["fwd_excess_ret"].mean()
                ls_rets.append(top3 - bot3)

        if ls_rets and np.std(ls_rets) > 0:
            sharpe = np.mean(ls_rets) / np.std(ls_rets) * np.sqrt(12)
        else:
            sharpe = 0
        block_sharpes.append(sharpe)
        date_range = f"{str(block_dates[0])[:10]} to {str(block_dates[-1])[:10]}"
        print(f"    Block {i+1} ({date_range}): Sharpe = {sharpe:.2f}")

    # CV of Sharpe
    if np.mean([abs(s) for s in block_sharpes]) > 0:
        cv_sharpe = np.std(block_sharpes) / np.mean([abs(s) for s in block_sharpes])
    else:
        cv_sharpe = float("inf")
    print(f"    CV of Sharpe across blocks: {cv_sharpe:.2f}")
    print(f"    {'PASS ✓' if cv_sharpe < 1.0 else 'CAUTION'}: {'Consistent' if cv_sharpe < 1.0 else 'Inconsistent'} across sub-periods")

    # Check sign consistency
    all_positive = all(s > 0 for s in block_sharpes)
    print(f"    All blocks positive Sharpe: {'Yes ✓' if all_positive else 'No ✗'}")

    # --- 3. Outlier robustness ---
    print("  [6c] Outlier robustness (trim 5% tails)...")
    ls_rets_all = []
    for date, group in preds_df.groupby("date"):
        if len(group) >= 6:
            group = group.sort_values("pred_prob", ascending=False)
            top3 = group.head(3)["fwd_excess_ret"].mean()
            bot3 = group.tail(3)["fwd_excess_ret"].mean()
            ls_rets_all.append(top3 - bot3)

    ls_rets_all = np.array(ls_rets_all)
    lower = np.percentile(ls_rets_all, 5)
    upper = np.percentile(ls_rets_all, 95)
    trimmed = ls_rets_all[(ls_rets_all >= lower) & (ls_rets_all <= upper)]

    full_sharpe = np.mean(ls_rets_all) / np.std(ls_rets_all) * np.sqrt(12) if np.std(ls_rets_all) > 0 else 0
    trim_sharpe = np.mean(trimmed) / np.std(trimmed) * np.sqrt(12) if np.std(trimmed) > 0 else 0

    print(f"    Full Sharpe: {full_sharpe:.2f}")
    print(f"    Trimmed Sharpe (5-95%): {trim_sharpe:.2f}")
    degradation = (full_sharpe - trim_sharpe) / abs(full_sharpe) * 100 if abs(full_sharpe) > 0 else 0
    print(f"    Degradation: {degradation:.1f}%")
    print(f"    {'PASS ✓' if abs(degradation) < 50 else 'CAUTION'}: {'Robust' if abs(degradation) < 50 else 'Outlier-dependent'}")

    return {
        "permutation_p_value": round(p_value, 4),
        "permutation_pass": p_value < 0.05,
        "block_sharpes": [round(s, 2) for s in block_sharpes],
        "cv_sharpe": round(cv_sharpe, 2),
        "sub_period_consistent": cv_sharpe < 1.0,
        "all_blocks_positive": all_positive,
        "full_sharpe": round(full_sharpe, 2),
        "trimmed_sharpe": round(trim_sharpe, 2),
        "outlier_robust": abs(degradation) < 50,
    }


# ---------------------------------------------------------------------------
# 8. Regime Analysis (HC #428 R1)
# ---------------------------------------------------------------------------
def regime_analysis(preds_df: pd.DataFrame, closes: pd.DataFrame) -> dict:
    """
    HC #428 R1: regime-agnostic validation.
    Classify days by SPY close-to-close as green/red/flat.
    Check |Sharpe_green - Sharpe_red| / max <= 0.50.
    """
    print("\n[7] Regime analysis (HC #428 R1)...")

    spy_daily = closes["SPY"].pct_change()

    # Classify each prediction date by SPY regime (21d trailing)
    spy_21d = spy_daily.rolling(21).sum()

    regime_map = {}
    for d in preds_df["date"].unique():
        ts = pd.Timestamp(d)
        if ts in spy_21d.index and not np.isnan(spy_21d.loc[ts]):
            ret = spy_21d.loc[ts]
            if ret > 0.02:
                regime_map[d] = "green"
            elif ret < -0.02:
                regime_map[d] = "red"
            else:
                regime_map[d] = "flat"

    preds_df = preds_df.copy()
    preds_df["regime"] = preds_df["date"].map(regime_map)

    regime_sharpes = {}
    for regime in ["green", "red", "flat"]:
        regime_data = preds_df[preds_df["regime"] == regime]
        ls_rets = []
        for date, group in regime_data.groupby("date"):
            if len(group) >= 6:
                group = group.sort_values("pred_prob", ascending=False)
                top3 = group.head(3)["fwd_excess_ret"].mean()
                bot3 = group.tail(3)["fwd_excess_ret"].mean()
                ls_rets.append(top3 - bot3)

        if ls_rets and np.std(ls_rets) > 0:
            sharpe = np.mean(ls_rets) / np.std(ls_rets) * np.sqrt(12)
        else:
            sharpe = 0
        regime_sharpes[regime] = sharpe
        print(f"    {regime:>5} regime: Sharpe = {sharpe:>6.2f} ({len(ls_rets)} periods)")

    # HC #428 R1 check
    s_green = regime_sharpes.get("green", 0)
    s_red = regime_sharpes.get("red", 0)
    max_s = max(abs(s_green), abs(s_red))
    if max_s > 0:
        regime_divergence = abs(s_green - s_red) / max_s
    else:
        regime_divergence = 0

    print(f"    Regime divergence: {regime_divergence:.2f} (threshold: 0.50)")
    passed = regime_divergence <= 0.50
    print(f"    {'PASS ✓' if passed else 'FAIL ✗'}: {'Regime-agnostic' if passed else 'Regime-dependent'}")

    return {
        "regime_sharpes": {k: round(v, 2) for k, v in regime_sharpes.items()},
        "regime_divergence": round(regime_divergence, 2),
        "regime_agnostic": passed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Force unbuffered output
    sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

    print("=" * 78)
    print("  ML SECTOR ROTATION / MOMENTUM STRATEGY")
    print("  LightGBM on momentum + macro features, 21d forward prediction")
    print("  Walk-forward: 252d train, 63d test, sliding window")
    print("=" * 78)
    sys.stdout.flush()

    # 1. Data
    print("\n[1] Downloading data...")
    closes = download_data()
    volumes = load_volumes()

    # 2. Features
    panel, feature_cols = build_features(closes, volumes)

    # 3. Walk-forward
    preds_df = walk_forward(panel, feature_cols)

    # Save predictions
    preds_df.to_parquet(OUTPUT_DIR / "predictions.parquet", index=False)
    print(f"\n  Predictions saved to {OUTPUT_DIR / 'predictions.parquet'}")

    # 4. Prediction quality
    quality = prediction_quality(preds_df)

    # 5. Backtest
    bt_results = backtest(preds_df, closes)

    # 6. Adversarial
    adv_results = adversarial_validation(preds_df, panel, feature_cols)

    # 7. Regime
    regime_results = regime_analysis(preds_df, closes)

    # Summary
    print("\n" + "=" * 78)
    print("  FINAL SUMMARY")
    print("=" * 78)
    print(f"\n  Prediction Quality:")
    print(f"    Accuracy:          {quality['accuracy']:.3f}")
    print(f"    Mean IC:           {quality['ic_mean']:.4f}")
    print(f"    ICIR:              {quality['icir']:.3f}")
    print(f"    L/S Sharpe:        {quality['ls_sharpe']:.2f}")

    print(f"\n  Adversarial Checks:")
    print(f"    Permutation test:  {'PASS' if adv_results['permutation_pass'] else 'FAIL'} (p={adv_results['permutation_p_value']:.4f})")
    print(f"    Sub-period CV:     {'PASS' if adv_results['sub_period_consistent'] else 'FAIL'} (CV={adv_results['cv_sharpe']:.2f})")
    print(f"    All blocks +Sharpe:{'PASS' if adv_results['all_blocks_positive'] else 'FAIL'}")
    print(f"    Outlier robust:    {'PASS' if adv_results['outlier_robust'] else 'FAIL'}")

    print(f"\n  Regime Analysis (HC #428 R1):")
    print(f"    Regime-agnostic:   {'PASS' if regime_results['regime_agnostic'] else 'FAIL'} (divergence={regime_results['regime_divergence']:.2f})")
    for regime, sharpe in regime_results["regime_sharpes"].items():
        print(f"    {regime:>5} Sharpe:       {sharpe:.2f}")

    # Overall verdict
    checks = [
        adv_results["permutation_pass"],
        adv_results["sub_period_consistent"],
        adv_results["outlier_robust"],
        regime_results["regime_agnostic"],
    ]
    passed = sum(checks)
    total = len(checks)

    print(f"\n  VERDICT: {passed}/{total} adversarial checks passed")
    if passed == total:
        print("  >>> STRATEGY IS ROBUST — consider for paper trading <<<")
    elif passed >= 3:
        print("  >>> STRATEGY SHOWS PROMISE — needs refinement <<<")
    else:
        print("  >>> STRATEGY NEEDS WORK — adversarial checks failing <<<")

    # Save full results
    full_results = {
        "timestamp": datetime.now().isoformat(),
        "quality": quality,
        "backtest_metrics": bt_results["metrics"],
        "adversarial": adv_results,
        "regime": regime_results,
        "config": {
            "train_days": TRAIN_DAYS,
            "test_days": TEST_DAYS,
            "fwd_days": FWD_DAYS,
            "top_n": TOP_N,
            "sectors": SECTOR_ETFS,
            "fixed_capital": FIXED_CAPITAL,
        },
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(full_results, f, indent=2, default=str)

    print(f"\n  Full results saved to {OUTPUT_DIR / 'results.json'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
