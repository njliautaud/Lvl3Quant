"""
lgbm_assignment_risk_v1.py — LGBM assignment-risk ranker for wheel CSP strategy.

Motivation: The yield-optimized ranker (v1) selected high-IV volatile tickers that
bled on red market days. This model flips the objective: predict which tickers are
SAFEST (lowest assignment risk) during the next 30 days, so we can sell CSPs on
stocks less likely to gap through the strike.

Target (binary): did the stock's close drop below the 30-delta CSP strike within
the next 30 calendar days?
  strike_approx = close * exp(-sigma * sqrt(30/252) * N_inv(0.30))
    where N_inv(0.30) ≈ -0.5244  →  strike ~ close * exp(+0.5244 * sigma * sqrt(30/252))
  assignment_event = 1 if any close in [t+1, t+30] falls below strike_approx

  Fallback continuous target when binary has low prevalence: max adverse drawdown
  over next 30 days = max(0, -(min_close_fwd - close) / close)

Walk-forward (HC #0 — SLIDING only):
  24-month train / 1-month OOS / 1-month step
  OOT starts 2023-01-01 (matches regime validation window)

Features:
  - PIT fundamentals (13 z-score cols)
  - IV features (sigma, iv_rank, iv_rv_ratio, term_ratio, r_1m, sigma_atm_30d)
  - Price momentum (mom_20d, mom_60d, rv_20, rv_60)
  - Beta to SPY (rolling 60d)
  - Max historical drawdown (rolling 252d)
  - VIX level at entry (absolute level + z-score)
  - Sector dummy (one-hot, dropped if collinear)

Outputs (to results/lgbm_assignment_risk_v1/):
  fold_predictions.parquet  — per-fold OOT predictions + realized targets
  monthly_selections.parquet — monthly top-K safest tickers
  metrics.json              — IC, hit rate, regime split
  report.md                 — findings report

Usage:
    cd /home/jupiter/Lvl3Quant/wheel_strategy_v1
    python3 -m strategy.lgbm_assignment_risk_v1
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

try:
    import lightgbm as lgb
except ImportError:
    raise ImportError("pip install lightgbm")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT       = Path("/home/jupiter/Lvl3Quant")
WHEEL      = ROOT / "wheel_strategy_v1"
FSTORE     = ROOT / "data" / "feature_store" / "v2"
CACHE      = WHEEL / "data" / "cache"
OUT_DIR    = WHEEL / "results" / "lgbm_assignment_risk_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Walk-forward schedule (HC #0: SLIDING only)
# ---------------------------------------------------------------------------
TRAIN_MONTHS = 24
OOT_MONTHS   = 1
STEP_MONTHS  = 1
START        = pd.Timestamp("2017-01-01")
END          = pd.Timestamp("2025-12-31")
OOT_START    = pd.Timestamp("2023-01-01")

TOP_K        = 20
TRADING_DAYS = 252
ANN_MONTHLY  = np.sqrt(12)   # annualize monthly Sharpe

# Strike approximation: 30-delta put
HORIZON_DAYS = 30
DELTA_TARGET = 0.30
N_INV_30     = norm.ppf(DELTA_TARGET)   # ≈ -0.5244

FEAT_COLS: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_spy_returns() -> pd.Series:
    spy = pd.read_parquet(CACHE / "spy_prices.parquet")[["date", "close"]].copy()
    spy = spy.sort_values("date")
    spy["spy_ret"] = spy["close"].pct_change()
    return spy.set_index("date")["spy_ret"]


def load_vix() -> pd.DataFrame:
    vix = pd.read_parquet(CACHE / "vix_history.parquet")[["date", "close"]].copy()
    vix = vix.rename(columns={"close": "vix_close"})
    vix = vix.sort_values("date")
    vix["vix_z60"] = (
        (vix["vix_close"] - vix["vix_close"].rolling(60).mean())
        / vix["vix_close"].rolling(60).std()
    )
    return vix


def load_universe_sectors() -> pd.DataFrame:
    u = pd.read_parquet(CACHE / "universe_v2.parquet")[["ticker", "sector"]].copy()
    return u


def load_panel() -> pd.DataFrame:
    """
    Join PIT fundamentals + IV features + price data + macro into daily panel.
    All features lagged 1 trading day (anti-lookahead, HC #428).
    """
    pit    = pd.read_parquet(FSTORE / "fund_pit_features.parquet")
    iv     = pd.read_parquet(CACHE / "iv_features_real_blend.parquet")
    prices = pd.read_parquet(CACHE / "prices_v2.parquet")

    # Restrict to tickers present in all three core sources
    common = set(pit.ticker) & set(iv.ticker) & set(prices.ticker)
    pit    = pit[pit.ticker.isin(common)].copy()
    iv     = iv[iv.ticker.isin(common)].copy()
    prices = prices[prices.ticker.isin(common)].copy()
    print(f"Common tickers: {len(common)}")

    # ---- SPY returns for rolling beta ----
    spy_ret = load_spy_returns()

    # ---- VIX ----
    vix = load_vix()

    # ---- Sector dummies ----
    sectors = load_universe_sectors()

    # ---- Price features ----
    prices = prices.sort_values(["ticker", "date"]).copy()
    prices["mom_20d"] = prices.groupby("ticker")["ret"].transform(
        lambda x: x.rolling(20).sum()
    ).shift(1)
    prices["mom_60d"] = prices.groupby("ticker")["ret"].transform(
        lambda x: x.rolling(60).sum()
    ).shift(1)
    # Max adverse drawdown over trailing 252 days (risk proxy)
    prices["max_dd_252"] = prices.groupby("ticker")["close"].transform(
        lambda x: x.rolling(252).apply(
            lambda w: (w[-1] - w.max()) / w.max() if w.max() > 0 else 0.0,
            raw=True
        )
    ).shift(1)
    # Rolling beta to SPY (60d)
    prices["spy_ret_aligned"] = prices["date"].map(spy_ret).fillna(0.0)

    def rolling_beta(g: pd.DataFrame, window: int = 60) -> pd.Series:
        cov = g["ret"].rolling(window).cov(g["spy_ret_aligned"])
        var = g["spy_ret_aligned"].rolling(window).var()
        return (cov / var.replace(0, np.nan)).shift(1)

    prices["beta_spy_60d"] = prices.groupby("ticker", group_keys=False).apply(rolling_beta)

    # ---- IV features (lag 1d) ----
    iv_cols = [c for c in iv.columns if c not in ("ticker", "date", "pricing_source")]
    iv = iv[["ticker", "date"] + iv_cols].sort_values(["ticker", "date"]).copy()
    for c in iv_cols:
        iv[c] = iv.groupby("ticker")[c].shift(1)

    # ---- PIT fundamentals (lag 1d for safety) ----
    pit_cols = [c for c in pit.columns if c not in ("ticker", "date", "fund_asof")]
    pit = pit[["ticker", "date"] + pit_cols].sort_values(["ticker", "date"]).copy()
    for c in pit_cols:
        pit[c] = pit.groupby("ticker")[c].shift(1)

    # ---- Merge all ----
    price_cols = ["ticker", "date", "close", "ret", "log_ret",
                  "rv_20", "rv_60", "mom_20d", "mom_60d",
                  "max_dd_252", "beta_spy_60d"]
    panel = (
        prices[price_cols]
        .merge(pit[["ticker", "date"] + pit_cols], on=["ticker", "date"], how="left")
        .merge(iv[["ticker", "date"] + iv_cols],   on=["ticker", "date"], how="left")
        .merge(vix[["date", "vix_close", "vix_z60"]], on="date", how="left")
        .merge(sectors, on="ticker", how="left")
    )

    # ---- Sector one-hot (drop first to avoid collinearity) ----
    sector_dummies = pd.get_dummies(panel["sector"], prefix="sec", drop_first=True, dtype=float)
    panel = pd.concat([panel.drop(columns=["sector"]), sector_dummies], axis=1)

    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    return panel


# ---------------------------------------------------------------------------
# Target construction
# ---------------------------------------------------------------------------

def build_targets(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Two targets:
      1. assignment_event (binary): any close in next 30 days below 30-delta strike
      2. max_adverse_30d (continuous): max adverse % move in next 30 days

    Strike approximation:
      strike = close * exp(N_inv(delta) * sigma * sqrt(h/252))
      For delta=0.30: N_inv ≈ -0.5244 → strike = close * exp(-0.5244 * sigma * sqrt(h/252))
      i.e., strike is BELOW current close.
    """
    h = HORIZON_DAYS
    panel = panel.sort_values(["ticker", "date"]).copy()

    # Use sigma from IV features; fall back to rv_20
    sigma = panel["sigma"].fillna(panel["rv_20"])
    strike_offset = np.exp(N_INV_30 * sigma * np.sqrt(h / TRADING_DAYS))
    # strike is below close (N_INV_30 < 0 => strike < close)
    panel["strike_approx"] = panel["close"] * strike_offset

    # Forward min close over next h trading days (per ticker, lagged -h)
    panel["fwd_min_close_30d"] = panel.groupby("ticker")["close"].transform(
        lambda x: x.rolling(h).min().shift(-h)
    )
    # Max adverse drawdown (positive = loss)
    panel["max_adverse_30d"] = np.maximum(
        0.0,
        -(panel["fwd_min_close_30d"] - panel["close"]) / panel["close"]
    )
    # Binary assignment event
    panel["assignment_event"] = (
        panel["fwd_min_close_30d"] < panel["strike_approx"]
    ).astype(float)

    # Also compute forward return for regime split analysis
    panel["fwd_ret_30d"] = panel.groupby("ticker")["ret"].transform(
        lambda x: x.rolling(h).sum().shift(-h)
    )

    prevalence = panel["assignment_event"].mean()
    print(f"Assignment event prevalence: {prevalence:.1%}")
    print(f"Mean max adverse drawdown: {panel['max_adverse_30d'].mean():.2%}")

    return panel


# ---------------------------------------------------------------------------
# Walk-forward machinery
# ---------------------------------------------------------------------------

def sliding_folds(
    start: pd.Timestamp, end: pd.Timestamp,
    train_mo: int, oot_mo: int, step_mo: int,
    oot_start: pd.Timestamp,
) -> List[Tuple]:
    folds = []
    fold_start = start
    while True:
        train_end = fold_start + pd.DateOffset(months=train_mo)
        oot_end   = train_end  + pd.DateOffset(months=oot_mo)
        if oot_end > end:
            break
        if train_end >= oot_start:
            folds.append((fold_start, train_end, train_end, oot_end))
        fold_start += pd.DateOffset(months=step_mo)
    return folds


def get_feat_cols(panel: pd.DataFrame) -> List[str]:
    exclude = {
        "ticker", "date", "close", "ret", "log_ret", "fund_asof", "pricing_source",
        "fwd_min_close_30d", "strike_approx", "assignment_event", "max_adverse_30d",
        "fwd_ret_30d",
    }
    return [c for c in panel.columns if c not in exclude and panel[c].dtype != object]


def run_walkforward(panel: pd.DataFrame) -> Tuple[pd.DataFrame, List[float], List[float]]:
    global FEAT_COLS
    FEAT_COLS = get_feat_cols(panel)
    print(f"\nFeatures ({len(FEAT_COLS)}): {FEAT_COLS}")

    folds = sliding_folds(START, END, TRAIN_MONTHS, OOT_MONTHS, STEP_MONTHS, OOT_START)
    print(f"Walk-forward: {len(folds)} folds (24m train / 1m OOT / 1m step, SLIDING)")

    all_oot   = []
    ics_binary   = []   # Spearman IC: pred_risk vs assignment_event
    ics_continu  = []   # Spearman IC: pred_risk vs max_adverse_30d

    for i, (tr_s, tr_e, ot_s, ot_e) in enumerate(folds):
        train = panel[(panel["date"] >= tr_s) & (panel["date"] < tr_e)].copy()
        oot   = panel[(panel["date"] >= ot_s) & (panel["date"] < ot_e)].copy()

        # Drop rows missing features or target
        target_col = "assignment_event"  # primary target
        train = train.dropna(subset=FEAT_COLS + [target_col])
        oot   = oot.dropna(subset=FEAT_COLS)

        if len(train) < 200 or len(oot) < 20:
            print(f"  Fold {i:02d}: skip (train={len(train)}, oot={len(oot)})")
            continue

        X_tr = train[FEAT_COLS].values.astype(np.float32)
        y_tr = train[target_col].values.astype(np.float32)
        X_ot = oot[FEAT_COLS].values.astype(np.float32)

        # LGBM binary classifier — outputs probability of assignment
        model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.03,
            num_leaves=15,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            class_weight="balanced",   # handles imbalanced assignment events
            random_state=42,
            verbose=-1,
        )
        model.fit(X_tr, y_tr)

        oot = oot.copy()
        oot["pred_assignment_prob"] = model.predict_proba(X_ot)[:, 1]

        # IC vs binary target
        oot_with_target = oot.dropna(subset=["assignment_event"])
        ic_bin = ic_cont = np.nan
        if len(oot_with_target) >= 10:
            ic_bin = oot_with_target["pred_assignment_prob"].corr(
                oot_with_target["assignment_event"], method="spearman"
            )
            ics_binary.append(ic_bin)

        # IC vs continuous drawdown
        oot_with_dd = oot.dropna(subset=["max_adverse_30d"])
        if len(oot_with_dd) >= 10:
            ic_cont = oot_with_dd["pred_assignment_prob"].corr(
                oot_with_dd["max_adverse_30d"], method="spearman"
            )
            ics_continu.append(ic_cont)

        print(
            f"  Fold {i:02d}: {tr_s.date()} -> {tr_e.date()} | "
            f"OOT: {ot_s.date()} -> {ot_e.date()} | "
            f"IC_binary={ic_bin:.3f}  IC_drawdown={ic_cont:.3f} | "
            f"n_train={len(train)}  n_oot={len(oot)}"
        )

        keep_cols = [
            "ticker", "date", "pred_assignment_prob",
            "assignment_event", "max_adverse_30d", "fwd_ret_30d",
        ]
        if "iv_rank" in oot.columns:
            keep_cols.append("iv_rank")
        all_oot.append(oot[[c for c in keep_cols if c in oot.columns]])

    if ics_binary:
        print(f"\nMean IC (binary assignment):  {np.mean(ics_binary):.3f} ± {np.std(ics_binary):.3f}")
    if ics_continu:
        print(f"Mean IC (max drawdown):        {np.mean(ics_continu):.3f} ± {np.std(ics_continu):.3f}")

    oot_df = pd.concat(all_oot, ignore_index=True) if all_oot else pd.DataFrame()
    return oot_df, ics_binary, ics_continu


# ---------------------------------------------------------------------------
# Monthly selection: pick safest tickers (LOWEST predicted assignment prob)
# ---------------------------------------------------------------------------

def monthly_safe_selection(oot_df: pd.DataFrame, k: int = TOP_K) -> pd.DataFrame:
    if oot_df.empty:
        return pd.DataFrame()
    oot_df = oot_df.copy()
    oot_df["ym"] = oot_df["date"].dt.to_period("M")
    oot_df["is_month_start"] = (
        oot_df.groupby(["ticker", "ym"])["date"].transform("min") == oot_df["date"]
    )
    monthly = oot_df[oot_df["is_month_start"]].copy()

    selections = []
    for ym, grp in monthly.groupby("ym"):
        # Rank ASCENDING by assignment probability → safest first
        ranked = grp.sort_values("pred_assignment_prob", ascending=True).head(k)
        selections.append(ranked.assign(rebalance_ym=ym))

    return pd.concat(selections, ignore_index=True) if selections else pd.DataFrame()


# ---------------------------------------------------------------------------
# Hit rate: do selected tickers actually have fewer assignment events?
# ---------------------------------------------------------------------------

def compute_hit_rate(selections: pd.DataFrame, oot_df: pd.DataFrame) -> dict:
    if selections.empty or oot_df.empty:
        return {}

    oot_with_ev = oot_df.dropna(subset=["assignment_event"]).copy()
    oot_with_ev["ym"] = oot_with_ev["date"].dt.to_period("M")
    oot_with_ev["is_month_start"] = (
        oot_with_ev.groupby(["ticker", "ym"])["date"].transform("min") == oot_with_ev["date"]
    )
    monthly_universe = oot_with_ev[oot_with_ev["is_month_start"]].copy()

    # Selected basket assignment rate vs universe rate per month
    sel_keys = set(zip(selections["rebalance_ym"].astype(str), selections["ticker"]))
    monthly_universe["is_selected"] = monthly_universe.apply(
        lambda r: (str(r["ym"]), r["ticker"]) in sel_keys, axis=1
    )

    sel = monthly_universe[monthly_universe["is_selected"]]
    non_sel = monthly_universe[~monthly_universe["is_selected"]]

    sel_rate  = sel["assignment_event"].mean() if len(sel) > 0 else np.nan
    univ_rate = monthly_universe["assignment_event"].mean()
    non_rate  = non_sel["assignment_event"].mean() if len(non_sel) > 0 else np.nan

    return {
        "selected_assignment_rate":   float(sel_rate),
        "universe_assignment_rate":   float(univ_rate),
        "non_selected_assignment_rate": float(non_rate),
        "assignment_rate_reduction":  float(univ_rate - sel_rate),
        "n_selected_obs":  int(len(sel)),
        "n_universe_obs":  int(len(monthly_universe)),
    }


# ---------------------------------------------------------------------------
# Regime split: green / red / flat SPY days
# ---------------------------------------------------------------------------

def regime_split(oot_df: pd.DataFrame, selections: pd.DataFrame) -> dict:
    """
    Classify each OOT date as green / red / flat (ES/SPY close-to-close >= +0.25% / <= -0.25%).
    Compute per-regime assignment rate for selected basket vs universe.
    This directly tests HC #428 R1 regime-symmetry gate.
    """
    spy = pd.read_parquet(CACHE / "spy_prices.parquet")[["date", "close"]].copy()
    spy = spy.sort_values("date")
    spy["spy_ret_day"] = spy["close"].pct_change()
    spy["regime"] = "flat"
    spy.loc[spy["spy_ret_day"] >= 0.0025,  "regime"] = "green"
    spy.loc[spy["spy_ret_day"] <= -0.0025, "regime"] = "red"

    if oot_df.empty or selections.empty:
        return {}

    oot_with_ev = oot_df.dropna(subset=["assignment_event"]).copy()
    oot_with_ev["ym"] = oot_with_ev["date"].dt.to_period("M")
    oot_with_ev["is_month_start"] = (
        oot_with_ev.groupby(["ticker", "ym"])["date"].transform("min") == oot_with_ev["date"]
    )
    monthly = oot_with_ev[oot_with_ev["is_month_start"]].merge(
        spy[["date", "regime"]], on="date", how="left"
    )

    sel_keys = set(zip(selections["rebalance_ym"].astype(str), selections["ticker"]))
    monthly["is_selected"] = monthly.apply(
        lambda r: (str(r["ym"]), r["ticker"]) in sel_keys, axis=1
    )

    results = {}
    for regime in ["green", "red", "flat"]:
        sub = monthly[monthly["regime"] == regime]
        sel_sub = sub[sub["is_selected"]]
        univ_rate = sub["assignment_event"].mean() if len(sub) > 0 else np.nan
        sel_rate  = sel_sub["assignment_event"].mean() if len(sel_sub) > 0 else np.nan
        results[regime] = {
            "universe_assignment_rate": float(univ_rate) if not np.isnan(univ_rate) else None,
            "selected_assignment_rate": float(sel_rate) if not np.isnan(sel_rate) else None,
            "n_obs_universe":  int(len(sub)),
            "n_obs_selected":  int(len(sel_sub)),
        }
        print(
            f"  {regime.upper():5s}: universe rate={univ_rate:.1%}  "
            f"selected rate={sel_rate:.1%}  n_universe={len(sub)}  n_selected={len(sel_sub)}"
        )

    # Regime skew on selected basket (HC #428 R1 gate)
    red_rate   = results.get("red",   {}).get("selected_assignment_rate")
    green_rate = results.get("green", {}).get("selected_assignment_rate")
    if red_rate is not None and green_rate is not None:
        denom = max(abs(red_rate), abs(green_rate))
        skew  = abs(red_rate - green_rate) / denom if denom > 0 else 0.0
        results["regime_skew"] = float(skew)
        print(f"\n  Regime skew |red - green| / max = {skew:.3f}  [PASS if <= 0.50]")
    return results


# ---------------------------------------------------------------------------
# Comparison vs yield ranker (v1 results if present)
# ---------------------------------------------------------------------------

def compare_vs_yield_ranker(oot_df: pd.DataFrame, selections: pd.DataFrame) -> dict:
    """
    If yield-ranker v1 results exist, compare per-month assignment rates:
    yield-basket vs safety-basket.
    """
    v1_path = WHEEL / "results" / "lgbm_ticker_ranker_v1" / "monthly_selections.parquet"
    if not v1_path.exists():
        print("  Yield ranker v1 monthly selections not found — skipping comparison.")
        return {}

    v1_sel = pd.read_parquet(v1_path)
    print(f"  Yield-ranker v1 selections: {len(v1_sel)} rows, {v1_sel['rebalance_ym'].nunique()} months")

    # Get assignment events for common months/tickers
    oot_ev = oot_df.dropna(subset=["assignment_event"]).copy()
    oot_ev["ym"] = oot_ev["date"].dt.to_period("M")
    oot_ev["is_month_start"] = (
        oot_ev.groupby(["ticker", "ym"])["date"].transform("min") == oot_ev["date"]
    )
    monthly = oot_ev[oot_ev["is_month_start"]].copy()

    v1_keys   = set(zip(v1_sel["rebalance_ym"].astype(str), v1_sel["ticker"]))
    safe_keys = set(zip(selections["rebalance_ym"].astype(str), selections["ticker"]))

    monthly["in_v1"]   = monthly.apply(lambda r: (str(r["ym"]), r["ticker"]) in v1_keys,   axis=1)
    monthly["in_safe"] = monthly.apply(lambda r: (str(r["ym"]), r["ticker"]) in safe_keys, axis=1)

    v1_rate   = monthly.loc[monthly["in_v1"],   "assignment_event"].mean()
    safe_rate = monthly.loc[monthly["in_safe"], "assignment_event"].mean()
    univ_rate = monthly["assignment_event"].mean()

    print(f"\n  Yield-basket  assignment rate: {v1_rate:.1%}")
    print(f"  Safety-basket assignment rate: {safe_rate:.1%}")
    print(f"  Universe      assignment rate: {univ_rate:.1%}")
    improvement = v1_rate - safe_rate
    print(f"  Safety improvement vs yield basket: {improvement:.1%} pp reduction")

    return {
        "v1_yield_basket_assignment_rate":  float(v1_rate),
        "safety_basket_assignment_rate":    float(safe_rate),
        "universe_assignment_rate":         float(univ_rate),
        "assignment_rate_improvement_pp":   float(improvement),
    }


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def compute_feature_importance(panel: pd.DataFrame) -> dict:
    """Quick single-model importance on all available training data."""
    global FEAT_COLS
    if FEAT_COLS is None:
        return {}
    train = panel.dropna(subset=FEAT_COLS + ["assignment_event"])
    X = train[FEAT_COLS].values.astype(np.float32)
    y = train["assignment_event"].values.astype(np.float32)

    model = lgb.LGBMClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.03,
        num_leaves=15, subsample=0.8, colsample_bytree=0.8,
        min_child_samples=20, class_weight="balanced",
        random_state=42, verbose=-1,
    )
    model.fit(X, y)
    imp = dict(zip(FEAT_COLS, model.feature_importances_))
    top10 = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\nTop-10 features by importance (full-sample model):")
    for feat, score in top10:
        print(f"  {feat:35s}  {score:6.0f}")
    return dict(top10)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(
    fold_ics_binary: list,
    fold_ics_cont: list,
    hit_rate_metrics: dict,
    regime: dict,
    comparison: dict,
    top_features: dict,
) -> None:
    global FEAT_COLS
    n_feats = len(FEAT_COLS) if FEAT_COLS else "?"

    ic_bin_mean = np.mean(fold_ics_binary) if fold_ics_binary else float("nan")
    ic_bin_std  = np.std(fold_ics_binary)  if fold_ics_binary else float("nan")
    ic_con_mean = np.mean(fold_ics_cont)   if fold_ics_cont   else float("nan")

    sel_rate  = hit_rate_metrics.get("selected_assignment_rate", float("nan"))
    univ_rate = hit_rate_metrics.get("universe_assignment_rate", float("nan"))
    reduction = hit_rate_metrics.get("assignment_rate_reduction", float("nan"))

    red_rate   = (regime.get("red",   {}) or {}).get("selected_assignment_rate", float("nan"))
    green_rate = (regime.get("green", {}) or {}).get("selected_assignment_rate", float("nan"))
    skew       = regime.get("regime_skew", float("nan"))

    compare_imp = comparison.get("assignment_rate_improvement_pp", float("nan"))

    # Verdict logic
    ic_ok    = ic_bin_mean > 0.05
    hit_ok   = (not np.isnan(reduction)) and reduction > 0.03
    skew_ok  = (not np.isnan(skew)) and skew <= 0.50
    all_pass = ic_ok and hit_ok

    if all_pass and skew_ok:
        verdict = "DEPLOY-CANDIDATE: positive IC, assignment rate reduction confirmed, regime symmetry PASS. Replace or supplement yield ranker."
    elif all_pass:
        verdict = "CONDITIONAL: IC and hit rate pass but regime skew fails. Investigate sector concentration. Do NOT deploy until R1 fixed."
    elif ic_ok:
        verdict = "MARGINAL: IC positive but assignment rate reduction weak. Investigate feature quality or increase TOP_K."
    else:
        verdict = "NEGATIVE: IC near zero, model not predicting assignment risk reliably. Revisit target definition or data coverage."

    lines = [
        "# Wheel LGBM Assignment Risk Ranker v1 — Report",
        "",
        f"Walk-forward: {TRAIN_MONTHS}m train / {OOT_MONTHS}m OOT / {STEP_MONTHS}m step (SLIDING, HC #0)",
        f"Features: {n_feats} (PIT + IV + momentum + beta + max_dd + VIX + sector dummies)",
        f"Top-K safe tickers per rebalance: {TOP_K}  (ranked by LOWEST predicted assignment probability)",
        "",
        "## OOT Information Coefficient",
        f"- Mean Spearman IC (vs binary assignment event): {ic_bin_mean:.3f} ± {ic_bin_std:.3f}" if fold_ics_binary else "- IC (binary): N/A",
        f"- Mean Spearman IC (vs max adverse drawdown):   {ic_con_mean:.3f}" if fold_ics_cont else "- IC (drawdown): N/A",
        f"- Folds with positive IC: {sum(1 for x in fold_ics_binary if x > 0)} / {len(fold_ics_binary)}",
        "",
        "## Assignment Hit Rate",
        f"- Universe assignment rate:  {univ_rate:.1%}",
        f"- Selected basket rate:      {sel_rate:.1%}",
        f"- Reduction vs universe:     {reduction:.1%} pp  ({'PASS' if hit_ok else 'FAIL'} — need >3pp)",
        "",
        "## Regime Split (HC #428 R1)",
        f"- Green days — selected basket assignment rate: {green_rate:.1%}",
        f"- Red days   — selected basket assignment rate: {red_rate:.1%}",
        f"- Regime skew |red-green|/max = {skew:.3f}  [{'PASS' if skew_ok else 'FAIL'} — threshold 0.50]",
        "",
    ]

    if comparison:
        v1_rate = comparison.get("v1_yield_basket_assignment_rate", float("nan"))
        s_rate  = comparison.get("safety_basket_assignment_rate",   float("nan"))
        lines += [
            "## Comparison vs Yield Ranker v1",
            f"- Yield-basket assignment rate:  {v1_rate:.1%}",
            f"- Safety-basket assignment rate: {s_rate:.1%}",
            f"- Improvement: {compare_imp:.1%} pp reduction in assignment events",
            "",
        ]

    if top_features:
        lines += ["## Top Features (full-sample importance)"]
        for feat, score in top_features.items():
            lines.append(f"  {feat:35s}  {score:.0f}")
        lines.append("")

    lines += ["## Verdict", verdict]

    report_text = "\n".join(lines)
    with open(OUT_DIR / "report.md", "w") as f:
        f.write(report_text)
    print("\n" + "=" * 60)
    print(report_text)
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("LGBM Assignment Risk Ranker v1")
    print("=" * 60)

    print("\n[1/6] Loading panel...")
    panel = load_panel()
    print(f"Panel shape: {panel.shape}, tickers: {panel.ticker.nunique()}, "
          f"dates: {panel.date.min().date()} -> {panel.date.max().date()}")

    print("\n[2/6] Building assignment risk targets...")
    panel = build_targets(panel)

    print("\n[3/6] Running walk-forward (SLIDING, HC #0)...")
    oot_df, ics_binary, ics_cont = run_walkforward(panel)

    if oot_df.empty:
        print("ERROR: No OOT predictions — check data coverage.")
        return

    oot_df.to_parquet(OUT_DIR / "fold_predictions.parquet", index=False)
    print(f"\nOOT predictions: {len(oot_df)} rows, "
          f"{oot_df.date.min().date()} -> {oot_df.date.max().date()}")

    print("\n[4/6] Building monthly safe-ticker selections...")
    selections = monthly_safe_selection(oot_df, k=TOP_K)
    if not selections.empty:
        selections.to_parquet(OUT_DIR / "monthly_selections.parquet", index=False)
        print(f"Monthly selections: {len(selections)} rows, "
              f"{selections['rebalance_ym'].nunique()} months")

    print("\n[5/6] Computing hit rate and regime split...")
    hit_rate_metrics = compute_hit_rate(selections, oot_df)
    print(f"  Selected assignment rate: {hit_rate_metrics.get('selected_assignment_rate', 'N/A'):.1%}")
    print(f"  Universe assignment rate: {hit_rate_metrics.get('universe_assignment_rate', 'N/A'):.1%}")

    print("\n  Regime split:")
    regime_metrics = regime_split(oot_df, selections)

    print("\n  Comparison vs yield ranker:")
    comparison = compare_vs_yield_ranker(oot_df, selections)

    print("\n[6/6] Feature importance + writing report...")
    top_feats = compute_feature_importance(panel)

    # Save all metrics
    all_metrics = {
        "fold_ics_binary":  ics_binary,
        "fold_ics_cont":    ics_cont,
        "mean_ic_binary":   float(np.mean(ics_binary)) if ics_binary else None,
        "mean_ic_cont":     float(np.mean(ics_cont))   if ics_cont   else None,
        "hit_rate":         hit_rate_metrics,
        "regime":           regime_metrics,
        "comparison_vs_v1": comparison,
        "top_features":     top_feats,
    }
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    write_report(ics_binary, ics_cont, hit_rate_metrics, regime_metrics, comparison, top_feats)
    print(f"\nOutputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
