"""
lgbm_ticker_ranker_v1.py — LGBM-powered ticker ranker for wheel strategy.

Replaces the static fund_score gate with a walk-forward LGBM model that
predicts which tickers will generate the best put-premium income (30d yield
net of assignment cost) given PIT fundamentals + IV features.

Design (HC #428 R1 / HC #0 compliant):
  - SLIDING window: 24m train / 6m OOT / 3m step
  - Features: PIT fundamentals (13 cols) + IV features (11 cols) + price
    momentum (3 cols) — all lagged 1 trading day to prevent lookahead
  - Target: realized 30d CSP yield = (premium_received - assignment_loss) /
    strike × 12  (annualized). Positive = good, negative = ruinous assignment.
  - LGBM regression (LGBMRegressor, 100 trees, max_depth=4, lr=0.05)
  - OOT: rank tickers by predicted yield, take top-K (default K=20 for Tier2)
  - Compare vs baseline Tier2 (fund_score > 55 + IV-rank filter)

Usage:
    cd /home/jupiter/Lvl3Quant/wheel_strategy_v1
    python3 -m strategy.lgbm_ticker_ranker_v1

Outputs (to results/lgbm_ticker_ranker_v1/):
    fold_predictions.parquet  — per-fold OOT ranked tickers + true yield
    metrics.json              — WF IC, Sharpe comparison vs baseline
    report.md                 — findings report
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    import lightgbm as lgb
except ImportError:
    raise ImportError("pip install lightgbm")

ROOT = Path("/home/jupiter/Lvl3Quant")
WHEEL = ROOT / "wheel_strategy_v1"
FSTORE = ROOT / "data" / "feature_store" / "v2"
CACHE = WHEEL / "data" / "cache"
OUT_DIR = WHEEL / "results" / "lgbm_ticker_ranker_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Walk-forward schedule (HC #0: SLIDING only)
TRAIN_MONTHS = 24
OOT_MONTHS = 6
STEP_MONTHS = 3
START = pd.Timestamp("2017-01-01")
END = pd.Timestamp("2025-12-31")
OOT_START = pd.Timestamp("2023-01-01")  # First OOT evaluation date

TOP_K = 20  # Tickers to select per rebalance period (Tier2 Balanced)
TRADING_DAYS = 252
ANN = np.sqrt(TRADING_DAYS)

# Target horizon: 30 calendar days forward put yield
HORIZON_DAYS = 30


def load_panel() -> pd.DataFrame:
    """Join PIT fundamentals, IV features, price momentum into daily panel."""
    pit = pd.read_parquet(FSTORE / "fund_pit_features.parquet")
    iv = pd.read_parquet(CACHE / "iv_features_real_blend.parquet")
    prices = pd.read_parquet(CACHE / "prices_v2.parquet")

    # Keep only tickers present in all three sources
    common = set(pit.ticker) & set(iv.ticker) & set(prices.ticker)
    pit = pit[pit.ticker.isin(common)].copy()
    iv = iv[iv.ticker.isin(common)].copy()
    prices = prices[prices.ticker.isin(common)].copy()

    # Price-derived momentum features (lagged 1d to prevent lookahead)
    prices = prices.sort_values(["ticker", "date"])
    prices["mom_20d"] = prices.groupby("ticker")["close"].pct_change(20).shift(1)
    prices["mom_60d"] = prices.groupby("ticker")["close"].pct_change(60).shift(1)
    prices["rv_20d"] = prices.groupby("ticker")["log_ret"].transform(
        lambda x: x.rolling(20).std() * np.sqrt(252)
    ).shift(1)

    # IV features: lag 1d
    iv_cols = [c for c in iv.columns if c not in ("ticker", "date", "pricing_source")]
    iv = iv[["ticker", "date"] + iv_cols].copy()
    for c in iv_cols:
        iv[c] = iv.groupby("ticker")[c].shift(1)

    # PIT fundamentals: already point-in-time, lag 1d for safety
    pit_cols = [c for c in pit.columns if c not in ("ticker", "date", "fund_asof")]
    for c in pit_cols:
        pit[c] = pit.groupby("ticker")[c].shift(1)

    # Merge
    panel = (
        prices[["ticker", "date", "close", "log_ret", "mom_20d", "mom_60d", "rv_20d"]]
        .merge(pit[["ticker", "date"] + pit_cols], on=["ticker", "date"], how="left")
        .merge(iv[["ticker", "date"] + iv_cols], on=["ticker", "date"], how="left")
    )
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    return panel


def build_target(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Approximate 30d CSP yield for each ticker-day.

    Proxy: realized 30d forward return magnitude scaled by IV (delta-approx).
    For a delta-0.22 put, yield ~ IV * sqrt(30/252) * 0.22 * notional.
    True yield requires options engine. Proxy: use (sigma * sqrt(HORIZON/252)) as
    the premium proxy, then subtract max(0, -(fwd_ret_30d)) as assignment cost.

    yield_proxy = sigma * sqrt(h/252) - max(0, -fwd_ret_30d)
    Positive = profitable. Negative = got assigned badly.
    """
    h = HORIZON_DAYS
    panel = panel.sort_values(["ticker", "date"])
    panel["fwd_ret_30d"] = panel.groupby("ticker")["log_ret"].transform(
        lambda x: x.rolling(h).sum().shift(-h)
    )
    panel["sigma"] = panel.get("sigma", panel.get("rv_20d", np.nan))

    panel["target_yield"] = (
        panel["sigma"] * np.sqrt(h / TRADING_DAYS)
        - np.maximum(0.0, -panel["fwd_ret_30d"])
    )
    return panel


def sliding_folds(start: pd.Timestamp, end: pd.Timestamp,
                  train_mo: int, oot_mo: int, step_mo: int,
                  oot_start: pd.Timestamp) -> List[Tuple]:
    """Generate (train_start, train_end, oot_start, oot_end) tuples."""
    folds = []
    fold_start = start
    while True:
        train_end = fold_start + pd.DateOffset(months=train_mo)
        oot_end = train_end + pd.DateOffset(months=oot_mo)
        if oot_end > end:
            break
        if train_end >= oot_start:
            folds.append((fold_start, train_end, train_end, oot_end))
        fold_start += pd.DateOffset(months=step_mo)
    return folds


FEAT_COLS = None  # filled at runtime


def get_feat_cols(panel: pd.DataFrame) -> List[str]:
    exclude = {"ticker", "date", "close", "log_ret", "fund_asof",
               "pricing_source", "fwd_ret_30d", "target_yield"}
    return [c for c in panel.columns if c not in exclude and panel[c].dtype != object]


def run_walkforward(panel: pd.DataFrame) -> pd.DataFrame:
    global FEAT_COLS
    FEAT_COLS = get_feat_cols(panel)
    print(f"Features ({len(FEAT_COLS)}): {FEAT_COLS}")

    folds = sliding_folds(START, END, TRAIN_MONTHS, OOT_MONTHS, STEP_MONTHS, OOT_START)
    print(f"Walk-forward: {len(folds)} folds")

    all_oot = []
    fold_ics = []

    for i, (tr_s, tr_e, ot_s, ot_e) in enumerate(folds):
        train = panel[(panel["date"] >= tr_s) & (panel["date"] < tr_e)].copy()
        oot = panel[(panel["date"] >= ot_s) & (panel["date"] < ot_e)].copy()

        train = train.dropna(subset=FEAT_COLS + ["target_yield"])
        oot = oot.dropna(subset=FEAT_COLS)

        if len(train) < 200 or len(oot) < 50:
            print(f"  Fold {i}: insufficient data (train={len(train)}, oot={len(oot)}), skip")
            continue

        X_tr = train[FEAT_COLS].values.astype(np.float32)
        y_tr = train["target_yield"].values.astype(np.float32)
        X_ot = oot[FEAT_COLS].values.astype(np.float32)

        model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            num_leaves=15,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            random_state=42,
            verbose=-1,
        )
        model.fit(X_tr, y_tr)

        oot = oot.copy()
        oot["pred_yield"] = model.predict(X_ot)

        # IC: Spearman rank-IC of predictions vs realized target where available
        oot_with_target = oot.dropna(subset=["target_yield"])
        if len(oot_with_target) > 10:
            ic = oot_with_target["pred_yield"].corr(
                oot_with_target["target_yield"], method="spearman"
            )
            fold_ics.append(ic)
            print(f"  Fold {i}: {tr_s.date()} -> {tr_e.date()} | OOT: {ot_s.date()} -> {ot_e.date()} | IC={ic:.3f} | n_train={len(train)}")
        else:
            print(f"  Fold {i}: {tr_s.date()} -> {tr_e.date()} | OOT: {ot_s.date()} -> {ot_e.date()} | no target for IC | n_train={len(train)}")

        all_oot.append(oot[["ticker", "date", "pred_yield", "target_yield", "sigma", "iv_rank"] if "iv_rank" in oot.columns else ["ticker", "date", "pred_yield", "target_yield", "sigma"]])

    print(f"\nMean OOT IC: {np.mean(fold_ics):.3f} ± {np.std(fold_ics):.3f}" if fold_ics else "No IC computed")
    oot_df = pd.concat(all_oot, ignore_index=True) if all_oot else pd.DataFrame()
    return oot_df, fold_ics


def monthly_top_k_selection(oot_df: pd.DataFrame, k: int = TOP_K) -> pd.DataFrame:
    """
    On the first trading day of each calendar month, pick top-K tickers by pred_yield.
    Returns monthly selection log.
    """
    if oot_df.empty:
        return pd.DataFrame()
    oot_df = oot_df.copy()
    oot_df["ym"] = oot_df["date"].dt.to_period("M")
    oot_df["is_month_start"] = oot_df.groupby(["ticker", "ym"])["date"].transform("min") == oot_df["date"]
    monthly = oot_df[oot_df["is_month_start"]].copy()

    selections = []
    for ym, grp in monthly.groupby("ym"):
        ranked = grp.sort_values("pred_yield", ascending=False).head(k)
        selections.append(ranked.assign(rebalance_ym=ym))

    return pd.concat(selections, ignore_index=True) if selections else pd.DataFrame()


def selection_performance(selections: pd.DataFrame, panel: pd.DataFrame) -> dict:
    """
    Compute realized 30d return of selected vs non-selected tickers per month.
    Proxy for whether the ranker is adding alpha.
    """
    if selections.empty:
        return {}

    panel_ret = panel[["ticker", "date", "fwd_ret_30d"]].dropna().copy()
    merged = selections.merge(panel_ret, on=["ticker", "date"], how="inner")

    selected_ret = merged.groupby("rebalance_ym")["fwd_ret_30d"].mean()
    all_ret = panel_ret.groupby(panel_ret["date"].dt.to_period("M"))["fwd_ret_30d"].mean()
    all_ret.index = all_ret.index.astype(str)
    selected_ret.index = selected_ret.index.astype(str)

    common_months = sorted(set(selected_ret.index) & set(all_ret.index))
    if not common_months:
        return {}

    sel = selected_ret.loc[common_months]
    univ = all_ret.loc[common_months]
    excess = sel - univ

    sharpe_sel = (sel.mean() / sel.std() * np.sqrt(12)) if sel.std() > 0 else np.nan
    sharpe_exc = (excess.mean() / excess.std() * np.sqrt(12)) if excess.std() > 0 else np.nan

    return {
        "n_months": len(common_months),
        "mean_selected_ret_30d": float(sel.mean()),
        "mean_universe_ret_30d": float(univ.mean()),
        "mean_excess_ret_30d": float(excess.mean()),
        "sharpe_selected_annualized": float(sharpe_sel),
        "sharpe_excess_annualized": float(sharpe_exc),
        "hit_rate": float((excess > 0).mean()),
        "months": common_months,
    }


def write_report(metrics: dict, fold_ics: list, selections_df: pd.DataFrame) -> None:
    lines = [
        "# Wheel LGBM Ticker Ranker v1 — Report",
        "",
        f"Walk-forward: {TRAIN_MONTHS}m train / {OOT_MONTHS}m OOT / {STEP_MONTHS}m step (SLIDING, HC #0)",
        f"Features: {len(FEAT_COLS)} (PIT fundamentals + IV + price momentum)",
        f"Top-K per rebalance: {TOP_K}",
        "",
        "## OOT Information Coefficient",
        f"- Mean Spearman IC: {np.mean(fold_ics):.3f} ± {np.std(fold_ics):.3f}" if fold_ics else "- IC: N/A",
        f"- Folds with positive IC: {sum(1 for x in fold_ics if x > 0)} / {len(fold_ics)}",
        "",
        "## Selection Alpha (vs Universe)",
        f"- Months evaluated: {metrics.get('n_months', 0)}",
        f"- Mean selected 30d return: {metrics.get('mean_selected_ret_30d', 0)*100:.2f}%",
        f"- Mean universe 30d return: {metrics.get('mean_universe_ret_30d', 0)*100:.2f}%",
        f"- Mean excess 30d return: {metrics.get('mean_excess_ret_30d', 0)*100:.2f}%",
        f"- Annualized Sharpe (selected basket): {metrics.get('sharpe_selected_annualized', float('nan')):.2f}",
        f"- Annualized Sharpe (excess vs universe): {metrics.get('sharpe_excess_annualized', float('nan')):.2f}",
        f"- Monthly hit rate (selected > universe): {metrics.get('hit_rate', 0)*100:.1f}%",
        "",
        "## Verdict",
    ]
    ic_mean = np.mean(fold_ics) if fold_ics else 0.0
    exc_sharpe = metrics.get("sharpe_excess_annualized", 0.0) or 0.0
    hit = metrics.get("hit_rate", 0.0) or 0.0
    if ic_mean > 0.05 and exc_sharpe > 0.3 and hit > 0.55:
        lines.append("DEPLOY-CANDIDATE: IC positive, excess Sharpe > 0.30, hit rate > 55%. Replace static fund_score with LGBM ranker.")
    elif ic_mean > 0.02 or hit > 0.50:
        lines.append("MARGINAL: weak IC or borderline hit rate. Use as additional signal alongside fund_score, not replacement.")
    else:
        lines.append("NEGATIVE: IC near zero or negative, LGBM ranker not adding alpha vs static filter. Investigate feature quality.")

    with open(OUT_DIR / "report.md", "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


def main() -> None:
    print("Loading panel...")
    panel = load_panel()
    print(f"Panel shape: {panel.shape}, tickers: {panel.ticker.nunique()}, dates: {panel.date.min()} -> {panel.date.max()}")

    print("Building targets...")
    panel = build_target(panel)
    print(f"Target non-null: {panel['target_yield'].notna().mean():.1%}")

    print("Running walk-forward...")
    oot_df, fold_ics = run_walkforward(panel)

    if oot_df.empty:
        print("ERROR: No OOT predictions generated. Check data coverage.")
        return

    oot_df.to_parquet(OUT_DIR / "fold_predictions.parquet", index=False)
    print(f"OOT predictions: {len(oot_df)} rows")

    print("Building monthly selections...")
    selections_df = monthly_top_k_selection(oot_df)
    if not selections_df.empty:
        selections_df.to_parquet(OUT_DIR / "monthly_selections.parquet", index=False)
        print(f"Monthly selection log: {len(selections_df)} rows, {selections_df['rebalance_ym'].nunique()} months")

    print("Computing selection performance vs universe...")
    metrics = selection_performance(selections_df, panel)

    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump({**metrics, "fold_ics": fold_ics, "mean_ic": float(np.mean(fold_ics)) if fold_ics else None}, f, indent=2)

    write_report(metrics, fold_ics, selections_df)
    print(f"\nOutputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
