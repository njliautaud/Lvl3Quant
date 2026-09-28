"""
K=6 Megacap-Tech HEDGE OVERLAY v3 — SPY short hedge to close the regime gap.

v1 (sign-classifier skip gate) and v2 (regression skip gate) both REJECTED:
skipping days cannot close the structural HC #428 R1 regime gap (the book is
pure long beta on red days). v3 HEDGES instead of abstaining: overlay a short
SPY position on the long-only K=6 book.

Variants (5 cells):
  (a) static_beta : short SPY sized to trailing 60d realized beta, EVERY day
                    (the symmetric baseline).
  (b) cond_m10bps : beta-sized SPY short only when LGBM E[ret] < -10 bps.
      cond_0bps   : same with threshold 0 bps.
  (c) scaled_k25  : hedge ratio = clip(-E[ret]/25bps, 0, 1) * beta (continuous).
      scaled_k50  : same with k = 50 bps.

Machinery reused from v2: LGBM regression on forward 1d K=6 return, cached
18-feature matrix (already lagged 1d at build time), SLIDING 24m/6m/3m
walk-forward (HC #0 — NEVER expanding).

HEDGE COST: 1bp of book notional charged on every day the hedge ratio changes
(SPY one-way cost, conservative flat charge per adjustment day).

LEAKAGE AUDIT (explicit):
  - E[ret] for day t comes from features as-of t-1 (v1 build shifted every
    feature column by 1 before caching) and a model trained strictly before
    the OOT window. Decision for day t uses NOTHING from day t.
  - Trailing beta for day t: rolling 60d cov/var of (book, SPY) returns,
    then shift(1) -> uses returns through t-1 only.
  - Hedge ratio for day t is therefore fully known at the open of day t.
  - Day-t SPY return is used ONLY for (i) hedge P&L realization and
    (ii) EVAL green/red/flat stratification (HC #428 R1 reporting). Never
    for the hedge decision.
  - Degenerate-fold guard: v2 fold 3 (OOT 2020-10-01, n_train=165) produced
    best_iter=1 constant predictions -> NaN IC. v3 detects constant/NaN
    prediction folds (std < 1e-12) and DROPS them (days fall back to
    "no prediction" handling) instead of silently including garbage.

No-prediction days (pre-first-OOT or dropped folds): conditional/scaled
variants do NOT hedge (no signal -> no action); static variant hedges
whenever beta is available (it needs no prediction).

MLflow: experiment k6_hedge_overlay_v3 @ http://localhost:5000
        (parent run + one nested run per variant; logs regime_gap and
        passes_hc428_r1 per variant).

Outputs: output/macro_picker/k6_hedge_overlay_v3/
Live K=6 paper state: UNTOUCHED (research only).
"""
from __future__ import annotations
import json
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "strategy" / "macro_picker"))

from walk_forward import _metrics  # type: ignore
from megacap_tech_rotation import (  # type: ignore
    BENCH_SPY, TRAIN_MONTHS, OOT_MONTHS, STEP_MONTHS,
    WF_START, WF_END, _load_prices, _iter_wf,
    regime_stratification, deploy_verdict,
)
from megacap_tech_extended_v1 import tail_dd_metric  # type: ignore

V1_DIR = ROOT / "output/macro_picker/k6_meta_classifier_v1"
FEAT_PATH = V1_DIR / "feature_matrix.parquet"
K6_BOOK_PATH = ROOT / "output/macro_picker/megacap_tech_extended_v1_20260609_171903/book_K6_mom60.parquet"

OUT_DIR = ROOT / "output/macro_picker/k6_hedge_overlay_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BETA_WINDOW = 60            # trailing realized-beta lookback (days)
BETA_CLIP = (0.0, 2.0)      # sanity clip on hedge beta
HEDGE_COST = 1e-4           # 1bp per hedge ADJUSTMENT day (SPY one-way)
MIN_POS_RATE = 0.15         # v2 early-fold garbage filter (kept)
EVAL_REGIME_THRESH = 0.001  # day-t SPY classification, EVAL ONLY

VARIANTS = [
    {"name": "static_beta", "mode": "static"},
    {"name": "cond_m10bps", "mode": "cond", "thresh": -0.0010},
    {"name": "cond_0bps",   "mode": "cond", "thresh": 0.0},
    {"name": "scaled_k25",  "mode": "scaled", "k": 0.0025},
    {"name": "scaled_k50",  "mode": "scaled", "k": 0.0050},
]


def load_feature_matrix() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_parquet(FEAT_PATH)
    feat_cols = df["feature_cols"].dropna().iloc[0].split(",")
    df.index = pd.to_datetime(df.index)
    return df, feat_cols


def train_wf_regressor(df: pd.DataFrame, feat_cols: list[str]) -> tuple[pd.DataFrame, dict]:
    """1d-horizon LGBM regression, identical harness to v2 h1, plus a
    degenerate-fold guard (drops constant-prediction folds like v2 fold 3)."""
    import lightgbm as lgb
    data = df.dropna(subset=feat_cols + ["y_reg"])
    windows = _iter_wf(WF_START, WF_END)
    print(f"[v3] WF windows: {len(windows)} (SLIDING {TRAIN_MONTHS}m/{OOT_MONTHS}m/{STEP_MONTHS}m, HC #0)")

    all_preds, fold_diag = [], []
    n_skipped_posrate, n_dropped_degenerate = 0, 0

    for fold_i, (tr_s, tr_e, os_, oe) in enumerate(windows):
        train = data[(data.index >= tr_s) & (data.index < tr_e)]
        oot = data[(data.index >= os_) & (data.index < oe)]
        if len(train) < 60 or len(oot) < 5:
            continue
        pos_rate = float(train["y_pos"].mean())
        if pos_rate < MIN_POS_RATE:
            n_skipped_posrate += 1
            fold_diag.append({"fold": fold_i, "oot_start": str(os_.date()),
                              "pos_rate_train": pos_rate, "skipped": "pos_rate"})
            continue

        cut = int(len(train) * 0.9)
        tr, val = train.iloc[:cut], train.iloc[cut:]
        params = dict(objective="regression", metric="l2", learning_rate=0.03,
                      num_leaves=15, max_depth=4, min_data_in_leaf=20,
                      feature_fraction=0.85, bagging_fraction=0.85,
                      bagging_freq=5, lambda_l2=1.0, verbose=-1)
        dtr = lgb.Dataset(tr[feat_cols].values, label=tr["y_reg"].values, feature_name=feat_cols)
        dval = lgb.Dataset(val[feat_cols].values, label=val["y_reg"].values,
                           feature_name=feat_cols, reference=dtr)
        booster = lgb.train(params, dtr, num_boost_round=500, valid_sets=[dval],
                            callbacks=[lgb.early_stopping(30, verbose=False),
                                       lgb.log_evaluation(0)])
        e_ret = booster.predict(oot[feat_cols].values, num_iteration=booster.best_iteration)

        # DEGENERATE-FOLD GUARD (v2 fold-3 NaN-IC issue): constant preds = no info
        if not np.all(np.isfinite(e_ret)) or float(np.std(e_ret)) < 1e-12:
            n_dropped_degenerate += 1
            fold_diag.append({"fold": fold_i, "oot_start": str(os_.date()),
                              "n_train": int(len(tr)), "pos_rate_train": pos_rate,
                              "best_iter": int(booster.best_iteration or 0),
                              "skipped": "degenerate_constant_preds"})
            print(f"[v3] fold {fold_i} OOT {os_.date()} DROPPED (degenerate: "
                  f"pred_std={np.std(e_ret):.2e}, best_iter={booster.best_iteration})")
            continue

        ic = float(np.corrcoef(e_ret, oot["y_reg"].values)[0, 1]) if len(oot) > 3 else float("nan")
        all_preds.append(pd.DataFrame({"date": oot.index, "e_ret": e_ret,
                                       "y_ret": oot["y_reg"].values, "fold": fold_i}))
        fold_diag.append({"fold": fold_i, "oot_start": str(os_.date()),
                          "n_train": int(len(tr)), "n_oot": int(len(oot)),
                          "pos_rate_train": pos_rate, "skipped": False,
                          "best_iter": int(booster.best_iteration or 0),
                          "oot_pearson_ic": ic})
        print(f"[v3] fold {fold_i} OOT {os_.date()}: n_train={len(tr)} IC={ic:.3f} "
              f"best_iter={booster.best_iteration}")

    if not all_preds:
        raise RuntimeError("no fold produced predictions")
    preds = pd.concat(all_preds, ignore_index=True)
    preds = preds.sort_values(["date", "fold"]).drop_duplicates("date", keep="first").reset_index(drop=True)
    return preds, {"fold_diag": fold_diag, "n_skipped_posrate": n_skipped_posrate,
                   "n_dropped_degenerate": n_dropped_degenerate}


def trailing_beta(book_ret: pd.Series, spy_ret: pd.Series) -> pd.Series:
    """Trailing 60d realized beta of K=6 book vs SPY, shifted 1d so the value
    for day t uses ONLY returns through t-1 (leakage-free)."""
    spy = spy_ret.reindex(book_ret.index)
    cov = book_ret.rolling(BETA_WINDOW).cov(spy)
    var = spy.rolling(BETA_WINDOW).var()
    beta = (cov / var).shift(1)  # <= t-1 info only
    return beta.clip(*BETA_CLIP)


def build_hedged_book(k6_book: pd.DataFrame, preds: pd.DataFrame,
                      spy_ret: pd.Series, beta: pd.Series, spec: dict) -> pd.DataFrame:
    book = k6_book.copy().sort_values("date").reset_index(drop=True)
    book["date"] = pd.to_datetime(book["date"])
    book["e_ret"] = book["date"].map(preds.set_index("date")["e_ret"])
    book["beta_lag"] = book["date"].map(beta)
    book["spy_ret_t"] = book["date"].map(spy_ret)  # P&L realization only

    b = book["beta_lag"].fillna(0.0)
    mode = spec["mode"]
    if mode == "static":
        ratio = b
    elif mode == "cond":
        ratio = np.where(book["e_ret"] < spec["thresh"], b, 0.0)
        ratio = np.where(book["e_ret"].isna(), 0.0, ratio)  # no signal -> no hedge
    elif mode == "scaled":
        scale = (-book["e_ret"] / spec["k"]).clip(0.0, 1.0).fillna(0.0)
        ratio = scale * b
    else:
        raise ValueError(mode)
    book["hedge_ratio"] = np.asarray(ratio, dtype=float)

    # Cost: 1bp on every day the hedge ratio changes (adjustment day)
    delta = book["hedge_ratio"].diff().fillna(book["hedge_ratio"])
    book["hedge_cost"] = np.where(np.abs(delta) > 1e-9, HEDGE_COST, 0.0)

    book["daily_ret_orig"] = book["daily_ret"]
    book["hedge_pnl"] = -book["hedge_ratio"] * book["spy_ret_t"].fillna(0.0)
    book["daily_ret"] = book["daily_ret_orig"] + book["hedge_pnl"] - book["hedge_cost"]
    return book


def summarize(book: pd.DataFrame, spy_ret: pd.Series, spy_close: pd.Series,
              label: str) -> dict:
    s = book.set_index("date")["daily_ret"]
    m = _metrics(s)
    strat = regime_stratification(book, spy_ret)  # day-t SPY, EVAL ONLY
    gates = deploy_verdict(m, strat, book)
    tail = tail_dd_metric(book, spy_close)
    pos, neg = s[s > 0], s[s < 0]
    return {
        "label": label,
        "cagr_pct": (m.get("cagr") or 0) * 100,
        "sharpe": m.get("sharpe"), "sortino": m.get("sortino"),
        "calmar": m.get("calmar"),
        "max_dd_pct": (m.get("max_dd") or 0) * 100,
        "pf": m.get("pf"), "wr_pct": (m.get("wr") or 0) * 100,
        "n_pos_days": int(len(pos)), "n_neg_days": int(len(neg)),
        "regime_strat": strat.to_dict(orient="records"),
        "regime_gap": gates.get("regime_imbalance"),
        "regime_gap_pass_le_0_50": bool(gates.get("regime_balance_ok")),
        "regime_green_sh": gates.get("regime_green_sharpe"),
        "regime_red_sh": gates.get("regime_red_sharpe"),
        "passes_hc428_r1": bool(gates["PASSES_DEPLOY_GATES"]),
        "day_conc": gates.get("day_concentration"),
        "day_conc_pass_le_0_70": bool(gates.get("day_concentration_ok")),
        "n_oot_days": int(s.dropna().shape[0]),
        "worst_red_quarter_dd_pct": tail["worst_red_quarter_dd_pct"],
        "passes_tail_dd_25pct": tail["passes_tail_dd_25pct"],
    }


def main():
    print("[v3] === K=6 Hedge Overlay v3 (SPY short overlay) ===")
    t0 = time.time()

    feat_df, feat_cols = load_feature_matrix()
    k6_book = pd.read_parquet(K6_BOOK_PATH)
    k6_book["date"] = pd.to_datetime(k6_book["date"])
    print(f"[v3] features: {feat_df.shape[0]} days x {len(feat_cols)}; book: {len(k6_book)} days")

    prices = _load_prices()
    spy_close = prices[prices["ticker"] == BENCH_SPY].sort_values("date").set_index("date")["close"]
    spy_ret = spy_close.pct_change().dropna()

    # 1d-horizon target (hedge decision is daily)
    k6 = k6_book.set_index("date")["daily_ret"].sort_index()
    tdf = feat_df.copy()
    tdf["y_reg"] = k6.reindex(tdf.index)
    tdf["y_pos"] = (tdf["y_reg"] > 0).astype(float)

    preds, meta = train_wf_regressor(tdf, feat_cols)
    preds.to_parquet(OUT_DIR / "oot_predictions.parquet", index=False)
    print(f"[v3] preds: {len(preds)} days | folds dropped degenerate: "
          f"{meta['n_dropped_degenerate']}, pos_rate: {meta['n_skipped_posrate']}")

    beta = trailing_beta(k6, spy_ret)

    import mlflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("k6_hedge_overlay_v3")

    baseline = summarize(k6_book, spy_ret, spy_close, "K6_unhedged_baseline")
    print(f"[v3] baseline: Sharpe={baseline['sharpe']:.2f} Calmar={baseline['calmar']:.2f} "
          f"gap={baseline['regime_gap']:.2f}")

    report = {"strategy": "k6_hedge_overlay_v3",
              "beta_window_d": BETA_WINDOW, "hedge_cost_bp_per_adj_day": 1.0,
              "wf_spec": {"train_months": TRAIN_MONTHS, "oot_months": OOT_MONTHS,
                          "step_months": STEP_MONTHS, "window_type": "SLIDING (HC #0)"},
              "fold_diag": meta["fold_diag"],
              "n_dropped_degenerate_folds": meta["n_dropped_degenerate"],
              "unhedged_baseline": baseline, "variants": {}}

    with mlflow.start_run(run_name="k6_hedge_overlay_v3_parent"):
        mlflow.log_params({"beta_window": BETA_WINDOW, "hedge_cost_bp": 1.0,
                           "n_features": len(feat_cols), "wf": "sliding_24m_6m_3m"})
        for k, v in baseline.items():
            if isinstance(v, (int, float)) and v is not None and np.isfinite(float(v)):
                mlflow.log_metric(f"baseline_{k}", float(v))

        for spec in VARIANTS:
            name = spec["name"]
            print(f"\n[v3] ===== VARIANT {name} ({spec}) =====")
            hb = build_hedged_book(k6_book, preds, spy_ret, beta, spec)
            hb.to_parquet(OUT_DIR / f"hedged_book_{name}.parquet", index=False)

            summ = summarize(hb[["date", "daily_ret"]], spy_ret, spy_close, name)
            summ["n_hedge_days"] = int((hb["hedge_ratio"] > 1e-9).sum())
            summ["mean_hedge_ratio"] = float(hb["hedge_ratio"].mean())
            summ["total_hedge_cost_pct"] = float(hb["hedge_cost"].sum() * 100)
            report["variants"][name] = summ

            with mlflow.start_run(run_name=name, nested=True):
                mlflow.log_params({k: v for k, v in spec.items()})
                for k, v in summ.items():
                    if isinstance(v, (bool, int, float)) and v is not None:
                        try:
                            if np.isfinite(float(v)):
                                mlflow.log_metric(k, float(v))
                        except (TypeError, ValueError):
                            pass
                mlflow.log_metric("delta_sharpe_vs_baseline",
                                  float((summ["sharpe"] or 0) - (baseline["sharpe"] or 0)))
            print(f"[v3:{name}] Sharpe={summ['sharpe']:.2f} (base {baseline['sharpe']:.2f}) "
                  f"Calmar={summ['calmar']:.2f} MaxDD={summ['max_dd_pct']:.1f}% "
                  f"gap={summ['regime_gap']:.2f} R1_PASS={summ['passes_hc428_r1']} "
                  f"hedge_days={summ['n_hedge_days']} cost={summ['total_hedge_cost_pct']:.2f}%")

        report["wall_seconds"] = round(time.time() - t0, 1)
        (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, default=str))
        mlflow.log_artifact(str(OUT_DIR / "report.json"))

    print("\n[v3] === FINAL SUMMARY ===")
    print(f"  baseline    : Sharpe={baseline['sharpe']:.2f} Calmar={baseline['calmar']:.2f} "
          f"gap={baseline['regime_gap']:.2f} R1={baseline['passes_hc428_r1']}")
    for name, s in report["variants"].items():
        print(f"  {name:12s}: Sharpe={s['sharpe']:.2f} Calmar={s['calmar']:.2f} "
              f"MaxDD={s['max_dd_pct']:.1f}% gap={s['regime_gap']:.2f} R1={s['passes_hc428_r1']}")
    print(f"[v3] done in {report['wall_seconds']}s -> {OUT_DIR}")
    return report


if __name__ == "__main__":
    main()
