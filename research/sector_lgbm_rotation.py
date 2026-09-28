"""
Sector Rotation Predictor — LightGBM on tabular sector flow features.

Predicts 21-day forward sector ETF returns using:
  - Sector flow features (dollar volume z-scores, relative strength, macro correlations)
  - Cross-sector flow dispersion
  - Momentum acceleration (not level)
  - Regime features (yield curve, VIX term structure, fed funds)
  - Factor loadings, theme basket flows
  - 87 features total (same panel as MLP v2)

Walk-forward: 252d train, 21d OOS, sliding window (HC #0: NEVER expanding).
MLflow logging mandatory.
HC #428 R1: regime-agnostic validation (|Sharpe_green - Sharpe_red| / max <= 0.50).
HC #670: anti-concentration check.

Context: Heuristic v3 Sharpe=2.39, MLP v1=0.95, MLP v2=1.45. LGBM better for tabular.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path("/home/nick/Lvl3Quant") if Path("/home/nick/Lvl3Quant").exists() else Path("/home/jupiter/Lvl3Quant")
DATA = ROOT / "data"
FEATURE_STORE = DATA / "feature_store" / "v2"
PRICE_PATH = ROOT / "wheel_strategy_v1" / "data" / "cache" / "prices_v2.parquet"
MACRO_EXTRA_PATH = ROOT / "wheel_strategy_v1" / "data" / "cache" / "macro_extra.parquet"
SECTOR_ROTATION_PATH = DATA / "feature_store" / "sector_rotation" / "daily.parquet"
OUTPUT_DIR = ROOT / "output" / "sector_flow_predictor"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLY", "XLP", "XLU", "XLI", "XLV", "XLB", "XLC", "XLRE"]
BENCHMARK = "SPY"

# Walk-forward config
TRAIN_DAYS = 252
OOS_DAYS = 21
FWD_DAYS = 21  # predict 21-day forward return
MIN_TRAIN_SAMPLES = 200

# LGBM config
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
}
NUM_BOOST_ROUND = 500
EARLY_STOPPING_ROUNDS = 30

# Cost
TXN_COST_BPS = 5.0
TOP_K = 3  # top-K equal weight portfolio


# ============================================================================
# DATA LOADING (same as MLP v2)
# ============================================================================

def load_prices() -> pd.DataFrame:
    """Load sector ETF + SPY daily prices."""
    df = pd.read_parquet(PRICE_PATH)
    universe = SECTOR_ETFS + [BENCHMARK]
    df = df[df["ticker"].isin(universe)].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


def compute_forward_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute 21-day forward returns for each sector ETF."""
    records = []
    for etf in SECTOR_ETFS:
        sub = prices[prices["ticker"] == etf][["date", "close"]].copy()
        sub = sub.sort_values("date").reset_index(drop=True)
        sub[f"fwd_ret_{FWD_DAYS}d"] = sub["close"].pct_change(FWD_DAYS).shift(-FWD_DAYS)
        sub["ticker"] = etf
        records.append(sub[["ticker", "date", f"fwd_ret_{FWD_DAYS}d"]])
    return pd.concat(records, ignore_index=True)


def compute_spy_regime(prices: pd.DataFrame) -> pd.DataFrame:
    """Classify each day as green/red/flat based on SPY 21d return."""
    spy = prices[prices["ticker"] == BENCHMARK][["date", "close"]].copy()
    spy = spy.sort_values("date").reset_index(drop=True)
    spy["spy_ret_21d"] = spy["close"].pct_change(21).shift(-21)
    spy["regime"] = "flat"
    spy.loc[spy["spy_ret_21d"] > 0.01, "regime"] = "green"
    spy.loc[spy["spy_ret_21d"] < -0.01, "regime"] = "red"
    return spy[["date", "regime", "spy_ret_21d"]]


def load_sector_flow_features() -> pd.DataFrame:
    """Load sector flow z-score features."""
    path = FEATURE_STORE / "sector_flow_features.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_flow_features() -> pd.DataFrame:
    """Load sector AUM/return features."""
    path = FEATURE_STORE / "flow_features.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_regime_features() -> pd.DataFrame:
    """Load macro regime features (yield curve, VIX, fed funds)."""
    path = FEATURE_STORE / "regime_features.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_theme_features() -> pd.DataFrame:
    """Load theme basket features."""
    path = FEATURE_STORE / "theme_features.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_factor_features() -> pd.DataFrame:
    """Load factor exposure features."""
    path = FEATURE_STORE / "factor_features.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_sector_rotation_features() -> pd.DataFrame:
    """Load sector rotation features."""
    if SECTOR_ROTATION_PATH.exists():
        df = pd.read_parquet(SECTOR_ROTATION_PATH)
        df["date"] = pd.to_datetime(df["date"])
        df = df.rename(columns={"etf": "ticker"})
        return df
    return pd.DataFrame()


def load_macro_extra() -> pd.DataFrame:
    """Load macro extra features."""
    if MACRO_EXTRA_PATH.exists():
        df = pd.read_parquet(MACRO_EXTRA_PATH)
        df["date"] = pd.to_datetime(df["date"])
        return df
    return pd.DataFrame()


def compute_cross_sector_features(panel: pd.DataFrame, flow_cols: list[str]) -> pd.DataFrame:
    """Compute cross-sector dispersion and range features per date."""
    result = pd.DataFrame({"date": panel["date"].unique()})
    for col in flow_cols:
        grp = panel.groupby("date")[col]
        result = result.merge(
            grp.std().rename(f"xs_disp_{col}").reset_index(),
            on="date", how="left",
        )
        result = result.merge(
            (grp.max() - grp.min()).rename(f"xs_range_{col}").reset_index(),
            on="date", how="left",
        )
    return result


def compute_momentum_acceleration(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute momentum acceleration (change in momentum, not level)."""
    records = []
    for etf in SECTOR_ETFS:
        sub = prices[prices["ticker"] == etf][["date", "close"]].copy()
        sub = sub.sort_values("date").reset_index(drop=True)
        sub["mom_20d"] = sub["close"].pct_change(20)
        sub["mom_60d"] = sub["close"].pct_change(60)
        sub["mom_accel_20d"] = sub["mom_20d"].diff(5)
        sub["mom_accel_60d"] = sub["mom_60d"].diff(5)
        sub["mom_ratio"] = sub["mom_20d"] / sub["mom_60d"].replace(0, np.nan)
        sub["ticker"] = etf
        records.append(sub[["ticker", "date", "mom_accel_20d", "mom_accel_60d", "mom_ratio"]])
    return pd.concat(records, ignore_index=True)


# ============================================================================
# PANEL CONSTRUCTION (same as MLP v2)
# ============================================================================

def build_panel() -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Build master panel: merge all feature sources + forward returns."""
    print("Loading data sources...")
    prices = load_prices()
    print(f"  Prices: {prices.shape}")

    fwd = compute_forward_returns(prices)
    print(f"  Forward returns: {fwd.shape}")

    regime_df = compute_spy_regime(prices)

    # Sector flow features
    sf = load_sector_flow_features()
    fl_raw = load_flow_features()

    # Map individual tickers to sector ETFs
    ticker_sector = pd.DataFrame()
    sf_feature_cols = []
    if not sf.empty and not fl_raw.empty:
        ticker_sector = fl_raw[["ticker", "sector_etf"]].drop_duplicates()
        sf = sf.merge(ticker_sector, on="ticker", how="left")
        sf = sf.dropna(subset=["sector_etf"])
        sf_feature_cols = [c for c in sf.columns if c not in ["ticker", "date", "sector_etf"]]
        sf_agg = sf.groupby(["sector_etf", "date"])[sf_feature_cols].mean().reset_index()
        sf_agg = sf_agg.rename(columns={"sector_etf": "ticker"})
        print(f"  Sector flow features (aggregated): {sf_agg.shape}")
    else:
        sf_agg = pd.DataFrame()

    # Flow features
    if not fl_raw.empty:
        fl_feat_cols = [c for c in fl_raw.columns if c not in ["ticker", "date", "sector_etf"]]
        fl_agg = fl_raw.groupby(["sector_etf", "date"])[fl_feat_cols].mean().reset_index()
        fl_agg = fl_agg.rename(columns={"sector_etf": "ticker"})
        print(f"  Flow features (aggregated): {fl_agg.shape}")
    else:
        fl_agg = pd.DataFrame()

    # Regime features (date-level)
    reg = load_regime_features()
    if not reg.empty:
        print(f"  Regime features: {reg.shape}")

    # Theme features
    theme = load_theme_features()
    if not theme.empty and not ticker_sector.empty:
        theme = theme.merge(ticker_sector, on="ticker", how="left")
        theme = theme.dropna(subset=["sector_etf"])
        theme_feat_cols = [c for c in theme.columns if c not in ["ticker", "date", "sector_etf"]
                           and theme[c].dtype in ("float64", "float32", "int64", "int32")]
        theme_agg = theme.groupby(["sector_etf", "date"])[theme_feat_cols].mean().reset_index()
        theme_agg = theme_agg.rename(columns={"sector_etf": "ticker"})
        print(f"  Theme features (aggregated): {theme_agg.shape}")
    else:
        theme_agg = pd.DataFrame()

    # Factor features
    factor = load_factor_features()
    if not factor.empty and not ticker_sector.empty:
        factor = factor.merge(ticker_sector, on="ticker", how="left")
        factor = factor.dropna(subset=["sector_etf"])
        factor_feat_raw = [c for c in factor.columns if c not in ["ticker", "date", "sector_etf"]]
        factor_agg = factor.groupby(["sector_etf", "date"])[factor_feat_raw].mean().reset_index()
        factor_agg = factor_agg.rename(columns={"sector_etf": "ticker"})
        print(f"  Factor features (aggregated): {factor_agg.shape}")
    else:
        factor_agg = pd.DataFrame()

    # Sector rotation features
    sr = load_sector_rotation_features()
    if not sr.empty:
        sr = sr[sr["ticker"].isin(SECTOR_ETFS)]
        print(f"  Sector rotation features: {sr.shape}")

    # Momentum acceleration
    mom_accel = compute_momentum_acceleration(prices)
    print(f"  Momentum acceleration: {mom_accel.shape}")

    # Macro extra (date-level)
    macro = load_macro_extra()
    if not macro.empty:
        macro_cols = ["date", "yc_2s10s", "be_10y", "fed_funds", "umich_sent",
                      "jobless_claims", "unrate", "cap_util"]
        macro_cols = [c for c in macro_cols if c in macro.columns]
        macro = macro[macro_cols]
        print(f"  Macro extra: {macro.shape}")

    # === MERGE ===
    panel = fwd.copy()

    if not sf_agg.empty:
        panel = panel.merge(sf_agg, on=["ticker", "date"], how="left")
    if not fl_agg.empty:
        panel = panel.merge(fl_agg, on=["ticker", "date"], how="left")
    if not theme_agg.empty:
        panel = panel.merge(theme_agg, on=["ticker", "date"], how="left")
    if not factor_agg.empty:
        panel = panel.merge(factor_agg, on=["ticker", "date"], how="left")
    if not sr.empty:
        sr_cols = [c for c in sr.columns if c not in ["ticker", "date"]]
        panel = panel.merge(sr[["ticker", "date"] + sr_cols], on=["ticker", "date"], how="left")

    panel = panel.merge(mom_accel, on=["ticker", "date"], how="left")

    if not reg.empty:
        panel = panel.merge(reg, on="date", how="left")
    if not macro.empty:
        panel = panel.merge(macro, on="date", how="left")

    panel = panel.merge(regime_df[["date", "regime"]], on="date", how="left")

    # Cross-sector dispersion features
    flow_z_cols = [c for c in sf_feature_cols if "_z" in c]
    if flow_z_cols:
        xs = compute_cross_sector_features(panel, flow_z_cols[:4])
        if not xs.empty:
            panel = panel.merge(xs, on="date", how="left")

    # Identify feature columns
    exclude = {"ticker", "date", f"fwd_ret_{FWD_DAYS}d", "regime", "spy_ret_21d", "close", "sector_etf"}
    feature_cols = sorted([c for c in panel.columns if c not in exclude])

    # Drop rows where target is NaN
    panel = panel.dropna(subset=[f"fwd_ret_{FWD_DAYS}d"])

    # Fill NaN features with 0 (z-scores => 0 = neutral)
    for c in feature_cols:
        panel[c] = panel[c].fillna(0.0)

    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)

    print(f"\nFinal panel: {panel.shape}")
    print(f"Feature columns ({len(feature_cols)}): {feature_cols[:10]}...")
    print(f"Date range: {panel['date'].min()} to {panel['date'].max()}")
    print(f"Unique tickers: {panel['ticker'].nunique()}")
    print(f"Unique dates: {panel['date'].nunique()}")

    return panel, feature_cols, regime_df


# ============================================================================
# WALK-FORWARD ENGINE (LGBM)
# ============================================================================

def walk_forward(panel: pd.DataFrame, feature_cols: list[str]):
    """
    Sliding walk-forward: 252d train, 21d OOS.
    Returns all OOS predictions with metadata.
    """
    target_col = f"fwd_ret_{FWD_DAYS}d"
    dates = sorted(panel["date"].unique())
    n_dates = len(dates)

    print(f"\nWalk-forward: {n_dates} unique dates, {TRAIN_DAYS}d train, {OOS_DAYS}d OOS")

    all_oos = []
    fold_metrics = []
    fold_idx = 0
    feature_importance_sum = np.zeros(len(feature_cols))

    i = TRAIN_DAYS
    while i + OOS_DAYS <= n_dates:
        train_dates = dates[i - TRAIN_DAYS: i]
        oos_dates = dates[i: i + OOS_DAYS]

        train_mask = panel["date"].isin(train_dates)
        oos_mask = panel["date"].isin(oos_dates)

        train_data = panel[train_mask]
        oos_data = panel[oos_mask]

        if len(train_data) < MIN_TRAIN_SAMPLES or len(oos_data) < 10:
            i += OOS_DAYS
            continue

        X_train = train_data[feature_cols].values
        y_train = train_data[target_col].values
        X_oos = oos_data[feature_cols].values
        y_oos = oos_data[target_col].values

        # Use last 21 days of training as validation for early stopping
        val_split = max(1, int(len(X_train) * 0.9))
        X_tr, X_val = X_train[:val_split], X_train[val_split:]
        y_tr, y_val = y_train[:val_split], y_train[val_split:]

        train_set = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_cols)
        val_set = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols, reference=train_set)

        callbacks = [
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=0),  # suppress iteration logs
        ]

        model = lgb.train(
            LGBM_PARAMS,
            train_set,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[val_set],
            callbacks=callbacks,
        )

        # Predict OOS
        preds = model.predict(X_oos, num_iteration=model.best_iteration)

        # Accumulate feature importance
        feature_importance_sum += model.feature_importance(importance_type="gain")

        # Compute fold IC
        if len(np.unique(preds)) > 1 and len(np.unique(y_oos)) > 1:
            ic = np.corrcoef(preds, y_oos)[0, 1]
            rank_ic = stats.spearmanr(preds, y_oos).statistic
        else:
            ic = 0.0
            rank_ic = 0.0

        fold_metrics.append({
            "fold": fold_idx,
            "train_start": str(train_dates[0])[:10],
            "oos_start": str(oos_dates[0])[:10],
            "oos_end": str(oos_dates[-1])[:10],
            "n_train": len(train_data),
            "n_oos": len(oos_data),
            "ic": float(ic),
            "rank_ic": float(rank_ic),
            "best_iteration": model.best_iteration,
        })

        # Save OOS predictions
        oos_df = oos_data[["ticker", "date", target_col, "regime"]].copy()
        oos_df["pred"] = preds
        all_oos.append(oos_df)

        if fold_idx % 10 == 0:
            print(f"  Fold {fold_idx:3d} | OOS {str(oos_dates[0])[:10]} | "
                  f"IC={ic:.4f} | RankIC={rank_ic:.4f} | trees={model.best_iteration}")

        fold_idx += 1
        i += OOS_DAYS

    if not all_oos:
        raise ValueError("No OOS predictions generated. Check data coverage.")

    oos_all = pd.concat(all_oos, ignore_index=True)

    # Feature importance
    fi = pd.DataFrame({
        "feature": feature_cols,
        "importance": feature_importance_sum / max(fold_idx, 1),
    }).sort_values("importance", ascending=False)

    print(f"\nCompleted {fold_idx} folds")
    print(f"OOS predictions: {oos_all.shape}")

    return oos_all, fold_metrics, fi


# ============================================================================
# EVALUATION
# ============================================================================

def compute_concat_ic(oos: pd.DataFrame) -> dict:
    """Compute concatenated IC across all OOS folds."""
    valid = oos.dropna(subset=["pred", f"fwd_ret_{FWD_DAYS}d"])
    if len(valid) < 20:
        return {"concat_ic": 0.0, "concat_rank_ic": 0.0}

    ic = np.corrcoef(valid["pred"].values, valid[f"fwd_ret_{FWD_DAYS}d"].values)[0, 1]
    rank_ic = stats.spearmanr(valid["pred"].values, valid[f"fwd_ret_{FWD_DAYS}d"].values).statistic
    return {"concat_ic": float(ic), "concat_rank_ic": float(rank_ic)}


def compute_strategy_sharpe(oos: pd.DataFrame) -> dict:
    """
    Compute strategy Sharpe: top-K equal weight, 21d rebalance.
    Returns Sharpe, Sortino, PF, WR, and per-period returns.
    """
    target_col = f"fwd_ret_{FWD_DAYS}d"
    rebal_dates = sorted(oos["date"].unique())

    # Group by rebal date, pick top-K predicted sectors
    period_returns = []
    period_dates = []
    holdings_per_period = []

    for dt in rebal_dates:
        day_data = oos[oos["date"] == dt].copy()
        if len(day_data) < TOP_K:
            continue

        # Rank by prediction, pick top K
        day_data = day_data.sort_values("pred", ascending=False)
        top_k = day_data.head(TOP_K)

        # Equal-weight return (realized)
        avg_ret = top_k[target_col].mean()

        # Subtract transaction costs (rebalance cost)
        avg_ret -= TXN_COST_BPS / 10000.0

        period_returns.append(avg_ret)
        period_dates.append(dt)
        holdings_per_period.append(list(top_k["ticker"].values))

    if len(period_returns) < 5:
        return {"sharpe": 0.0, "sortino": 0.0, "pf": 0.0, "wr": 0.0,
                "n_periods": 0, "period_returns": [], "period_dates": [],
                "holdings": []}

    rets = np.array(period_returns)
    mean_ret = np.mean(rets)
    std_ret = np.std(rets, ddof=1)

    # Annualize: ~12 periods per year (252/21)
    ann_factor = np.sqrt(252.0 / OOS_DAYS)
    sharpe = (mean_ret / std_ret) * ann_factor if std_ret > 0 else 0.0

    # Sortino
    downside = rets[rets < 0]
    down_std = np.std(downside, ddof=1) if len(downside) > 1 else std_ret
    sortino = (mean_ret / down_std) * ann_factor if down_std > 0 else 0.0

    # Profit factor
    gross_profit = np.sum(rets[rets > 0])
    gross_loss = np.abs(np.sum(rets[rets < 0]))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Win rate
    wr = np.mean(rets > 0) * 100.0

    return {
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "pf": float(pf),
        "wr": float(wr),
        "n_periods": len(rets),
        "total_return_pct": float(np.sum(rets) * 100),
        "mean_period_return_bps": float(mean_ret * 10000),
        "period_returns": [float(r) for r in rets],
        "period_dates": [str(d)[:10] for d in period_dates],
        "holdings": holdings_per_period,
    }


def compute_regime_gap(oos: pd.DataFrame, strategy_result: dict) -> dict:
    """
    HC #428 R1: regime-agnostic validation.
    |Sharpe_green - Sharpe_red| / max(|Sharpe_green|, |Sharpe_red|) <= 0.50
    """
    target_col = f"fwd_ret_{FWD_DAYS}d"

    regime_sharpes = {}
    for regime in ["green", "red", "flat"]:
        regime_data = oos[oos["regime"] == regime]
        if len(regime_data) < 20:
            regime_sharpes[regime] = None
            continue

        # Compute strategy returns for this regime only
        rebal_dates = sorted(regime_data["date"].unique())
        rets = []
        for dt in rebal_dates:
            day_data = regime_data[regime_data["date"] == dt].copy()
            if len(day_data) < TOP_K:
                continue
            day_data = day_data.sort_values("pred", ascending=False)
            top_k = day_data.head(TOP_K)
            avg_ret = top_k[target_col].mean() - TXN_COST_BPS / 10000.0
            rets.append(avg_ret)

        if len(rets) < 3:
            regime_sharpes[regime] = None
            continue

        rets = np.array(rets)
        ann = np.sqrt(252.0 / OOS_DAYS)
        sharpe = (np.mean(rets) / np.std(rets, ddof=1)) * ann if np.std(rets, ddof=1) > 0 else 0.0
        regime_sharpes[regime] = float(sharpe)

    # Compute gap
    sg = regime_sharpes.get("green")
    sr = regime_sharpes.get("red")

    if sg is not None and sr is not None:
        denom = max(abs(sg), abs(sr))
        gap = abs(sg - sr) / denom if denom > 0 else 0.0
        pass_r1 = gap <= 0.50
    else:
        gap = None
        pass_r1 = None

    return {
        "sharpe_green": sg,
        "sharpe_red": sr,
        "sharpe_flat": regime_sharpes.get("flat"),
        "regime_gap": float(gap) if gap is not None else None,
        "pass_r1": pass_r1,
    }


def compute_anti_concentration(strategy_result: dict) -> dict:
    """
    HC #670: anti-concentration check.
    No single sector should dominate holdings.
    """
    if not strategy_result.get("holdings"):
        return {"max_sector_freq": 0.0, "pass_concentration": True}

    from collections import Counter
    all_picks = []
    for h in strategy_result["holdings"]:
        all_picks.extend(h)

    counts = Counter(all_picks)
    total = len(all_picks)
    freq = {k: v / total for k, v in counts.items()}
    max_freq = max(freq.values()) if freq else 0.0

    # Day-level concentration: what fraction of periods is one sector in top-K?
    sector_presence = {etf: 0 for etf in SECTOR_ETFS}
    n_periods = len(strategy_result["holdings"])
    for h in strategy_result["holdings"]:
        for etf in set(h):
            sector_presence[etf] += 1

    presence_pct = {k: v / n_periods for k, v in sector_presence.items()}
    max_presence = max(presence_pct.values()) if presence_pct else 0.0

    # HC #344 day-conc cap <= 0.70
    pass_conc = max_presence <= 0.70

    return {
        "sector_frequency": freq,
        "max_sector_freq_pct": float(max_freq * 100),
        "sector_presence_pct": {k: round(v * 100, 1) for k, v in sorted(presence_pct.items(), key=lambda x: -x[1])},
        "max_sector_presence_pct": float(max_presence * 100),
        "pass_concentration": pass_conc,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Sector LGBM Rotation Predictor")
    parser.add_argument("--mlflow-tracking-uri", type=str, default="http://jupiter:5000")
    parser.add_argument("--experiment-name", type=str, default="sector_lgbm_rotation")
    args = parser.parse_args()

    print("=" * 80)
    print("SECTOR LGBM ROTATION PREDICTOR")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 80)

    # MLflow setup
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=f"lgbm_rotation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log params
        mlflow.log_params({
            "model_type": "lightgbm",
            "train_days": TRAIN_DAYS,
            "oos_days": OOS_DAYS,
            "fwd_days": FWD_DAYS,
            "top_k": TOP_K,
            "txn_cost_bps": TXN_COST_BPS,
            "num_boost_round": NUM_BOOST_ROUND,
            "early_stopping": EARLY_STOPPING_ROUNDS,
            **{f"lgbm_{k}": v for k, v in LGBM_PARAMS.items() if k != "verbosity"},
        })

        # Build panel
        t0 = time.time()
        panel, feature_cols, regime_df = build_panel()
        build_time = time.time() - t0
        print(f"Panel build time: {build_time:.1f}s")

        mlflow.log_params({
            "n_features": len(feature_cols),
            "n_samples": len(panel),
            "n_dates": panel["date"].nunique(),
        })

        # Walk-forward
        t0 = time.time()
        oos_all, fold_metrics, feature_importance = walk_forward(panel, feature_cols)
        wf_time = time.time() - t0
        print(f"Walk-forward time: {wf_time:.1f}s")

        # === EVALUATION ===
        print("\n" + "=" * 80)
        print("EVALUATION")
        print("=" * 80)

        # 1. Concat IC
        ic_result = compute_concat_ic(oos_all)
        print(f"\nConcat IC:      {ic_result['concat_ic']:.4f}")
        print(f"Concat Rank IC: {ic_result['concat_rank_ic']:.4f}")

        # 2. Strategy Sharpe
        strat = compute_strategy_sharpe(oos_all)
        print(f"\nStrategy (top-{TOP_K} equal weight, 21d rebal):")
        print(f"  Sharpe:  {strat['sharpe']:.2f}")
        print(f"  Sortino: {strat['sortino']:.2f}")
        print(f"  PF:      {strat['pf']:.2f}")
        print(f"  WR:      {strat['wr']:.1f}%")
        print(f"  Periods: {strat['n_periods']}")
        print(f"  Total Return: {strat['total_return_pct']:.2f}%")
        print(f"  Mean Period Return: {strat['mean_period_return_bps']:.1f} bps")

        # 3. Regime gap (HC #428 R1)
        regime = compute_regime_gap(oos_all, strat)
        print(f"\nRegime Analysis (HC #428 R1):")
        print(f"  Sharpe (green): {regime['sharpe_green']}")
        print(f"  Sharpe (red):   {regime['sharpe_red']}")
        print(f"  Sharpe (flat):  {regime['sharpe_flat']}")
        print(f"  Regime Gap:     {regime['regime_gap']}")
        print(f"  PASS R1:        {regime['pass_r1']}")

        # 4. Anti-concentration (HC #670)
        conc = compute_anti_concentration(strat)
        print(f"\nAnti-Concentration (HC #670):")
        print(f"  Max Sector Presence: {conc['max_sector_presence_pct']:.1f}%")
        print(f"  PASS:                {conc['pass_concentration']}")
        if conc.get("sector_presence_pct"):
            print(f"  Top sectors: {dict(list(conc['sector_presence_pct'].items())[:5])}")

        # 5. Feature importance
        print(f"\nTop 15 Features:")
        for _, row in feature_importance.head(15).iterrows():
            print(f"  {row['feature']:40s} {row['importance']:.1f}")

        # 6. Per-fold IC stats
        ics = [f["ic"] for f in fold_metrics]
        rank_ics = [f["rank_ic"] for f in fold_metrics]
        print(f"\nPer-Fold IC Stats:")
        print(f"  Mean IC:      {np.mean(ics):.4f} (std {np.std(ics):.4f})")
        print(f"  Mean RankIC:  {np.mean(rank_ics):.4f} (std {np.std(rank_ics):.4f})")
        print(f"  IC > 0:       {np.mean(np.array(ics) > 0)*100:.1f}%")

        # === LOG TO MLFLOW ===
        mlflow.log_metrics({
            "concat_ic": ic_result["concat_ic"],
            "concat_rank_ic": ic_result["concat_rank_ic"],
            "sharpe": strat["sharpe"],
            "sortino": strat["sortino"],
            "profit_factor": strat["pf"],
            "win_rate": strat["wr"],
            "n_periods": strat["n_periods"],
            "total_return_pct": strat["total_return_pct"],
            "mean_period_return_bps": strat["mean_period_return_bps"],
            "mean_fold_ic": float(np.mean(ics)),
            "mean_fold_rank_ic": float(np.mean(rank_ics)),
            "ic_positive_pct": float(np.mean(np.array(ics) > 0) * 100),
            "max_sector_presence_pct": conc["max_sector_presence_pct"],
            "pass_concentration": int(conc["pass_concentration"]),
            "build_time_s": build_time,
            "wf_time_s": wf_time,
        })

        if regime["regime_gap"] is not None:
            mlflow.log_metrics({
                "sharpe_green": regime["sharpe_green"] or 0.0,
                "sharpe_red": regime["sharpe_red"] or 0.0,
                "regime_gap": regime["regime_gap"],
                "pass_r1": int(regime["pass_r1"]),
            })

        # Save OOS predictions
        oos_path = OUTPUT_DIR / "lgbm_oos_predictions.parquet"
        oos_all.to_parquet(oos_path, index=False)
        print(f"\nSaved OOS predictions: {oos_path}")

        # Save fold metrics
        metrics_path = OUTPUT_DIR / "lgbm_fold_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(fold_metrics, f, indent=2)

        # Save feature importance
        fi_path = OUTPUT_DIR / "lgbm_feature_importance.json"
        feature_importance.to_json(fi_path, orient="records", indent=2)

        # Save results summary
        summary = {
            "model": "lightgbm",
            "timestamp": datetime.now().isoformat(),
            "concat_ic": ic_result,
            "strategy": {k: v for k, v in strat.items()
                         if k not in ["period_returns", "period_dates", "holdings"]},
            "regime": regime,
            "concentration": {k: v for k, v in conc.items() if k != "sector_frequency"},
            "fold_ic_stats": {
                "mean": float(np.mean(ics)),
                "std": float(np.std(ics)),
                "pct_positive": float(np.mean(np.array(ics) > 0) * 100),
            },
            "lgbm_params": LGBM_PARAMS,
        }
        summary_path = OUTPUT_DIR / "lgbm_results_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Log artifacts to MLflow (HC: try/except to avoid permission crash)
        for artifact_path in [oos_path, metrics_path, fi_path, summary_path]:
            try:
                mlflow.log_artifact(str(artifact_path))
            except Exception as e:
                print(f"WARNING: Failed to log artifact {artifact_path.name}: {e}")

        # === VERDICT ===
        print("\n" + "=" * 80)
        print("VERDICT")
        print("=" * 80)

        heuristic_sharpe = 2.39
        mlp_v2_sharpe = 1.45

        print(f"  Heuristic v3 Sharpe: {heuristic_sharpe}")
        print(f"  MLP v2 Sharpe:       {mlp_v2_sharpe}")
        print(f"  LGBM Sharpe:         {strat['sharpe']:.2f}")

        if strat["sharpe"] > heuristic_sharpe:
            print("  >> LGBM BEATS HEURISTIC - worth pursuing")
        elif strat["sharpe"] > mlp_v2_sharpe:
            print("  >> LGBM beats MLP v2 but not heuristic - incremental improvement")
        else:
            print("  >> LGBM underperforms - tabular data may not help here")

        if regime.get("pass_r1") is False:
            print("  >> FAILS HC #428 R1 regime-agnostic test - regime-tailored, not real edge")
        if not conc.get("pass_concentration", True):
            print("  >> FAILS HC #670 anti-concentration - too concentrated in one sector")

        print(f"\nFinished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
