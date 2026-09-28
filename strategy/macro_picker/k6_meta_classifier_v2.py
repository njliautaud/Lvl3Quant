"""
K=6 Megacap-Tech Meta-Gate v2 — REGIME-ASYMMETRIC REGRESSION GATE.

Follow-up to k6_meta_classifier_v1 (verdict NEGATIVE — symmetric sign-classifier
gate destroyed Sharpe/Calmar without closing the HC #428 R1 regime gap).
Implements next-iteration candidates (1)-(4) from the v1 findings doc as a
2x2 variant sweep:

  (a) LGBM REGRESSION on forward K=6 return; gate (sit in cash) when
      E[ret] < -10 bps/day  (asymmetric: skip predicted LOSERS only).
  (b) Regime-conditioned gate: apply the gate ONLY on red SPY days.
      Red is defined LEAKAGE-FREE as SPY day t-1 return < -0.1%
      (the day-t SPY close is unknown at decision time).
  (c) Horizon variant: target = forward 5d mean daily return (K=6 is a
      holding strategy; day-1 sign is ~80% noise per v1 diagnosis).
  (d) Fold filter: skip folds where train positive-rate < 0.15 (v1 early
      folds 0-3 trained on pos_rate 0-5% garbage labels).

Variants (2x2 = horizon {1d, 5d} x gate-scope {alldays, redonly}),
(d) applied to all:
  h1_alldays, h1_redonly, h5_alldays, h5_redonly

Walk-forward harness IDENTICAL to v1: SLIDING 24m train / 6m OOT / 3m step
(HC #0 — NEVER expanding), 23 folds, same cached 18-feature matrix.

LEAKAGE AUDIT:
  - Features were lagged 1d at build time (v1 build_feature_matrix shifts
    every feat col by 1 before saving) -> features are as-of t-1.
  - 1d target: K=6 return ON day t. Train rows require t < train_end.
  - 5d target: mean K=6 daily_ret over days t..t+4. Train rows are kept
    only if the FULL target window ends strictly before train_end
    (t_plus_h_date < tr_e), so no train target peeks into OOT.
  - Red-day gate condition uses SPY return at t-1 (already known at open
    of day t). Day-t SPY classification is used for EVAL stratification
    only (standard HC #428 R1 reporting), never for the trade decision.
  - OOT predictions deduped keep="first" by date (same as v1).

MLflow: experiment k6_meta_classifier_v2 @ http://localhost:5000
        (parent run + one nested run per variant).

Outputs: output/macro_picker/k6_meta_classifier_v2/
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

TRADING_DAYS = 252
K6_UNIVERSE = ["META", "AVGO", "TSLA", "MSFT", "AAPL", "NVDA"]

# Reuse v1 cached feature matrix (1953 days x 18 features, already lagged 1d)
V1_DIR = ROOT / "output/macro_picker/k6_meta_classifier_v1"
FEAT_PATH = V1_DIR / "feature_matrix.parquet"
K6_BOOK_PATH = ROOT / "output/macro_picker/megacap_tech_extended_v1_20260609_171903/book_K6_mom60.parquet"

OUT_DIR = ROOT / "output/macro_picker/k6_meta_classifier_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GATE_BPS = -0.0010          # gate when E[ret] < -10 bps/day
RED_THRESH = -0.001         # SPY t-1 ret < -0.1% => "red" decision regime
MIN_POS_RATE = 0.15         # (d) skip folds with train pos rate below this
EVAL_REGIME_THRESH = 0.001  # day-t SPY classification for EVAL stratification

VARIANTS = [
    {"name": "h1_alldays", "horizon": 1, "red_only": False},
    {"name": "h1_redonly", "horizon": 1, "red_only": True},
    {"name": "h5_alldays", "horizon": 5, "red_only": False},
    {"name": "h5_redonly", "horizon": 5, "red_only": True},
]


def load_feature_matrix() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_parquet(FEAT_PATH)
    feat_cols = df["feature_cols"].dropna().iloc[0].split(",")
    df.index = pd.to_datetime(df.index)
    return df, feat_cols


def build_targets(feat_df: pd.DataFrame, k6_book: pd.DataFrame,
                  horizon: int) -> pd.DataFrame:
    """Attach forward-return target. horizon=1: day-t ret. horizon=5: mean
    daily ret over t..t+4. Also record the END DATE of the target window so
    the WF loop can exclude train rows whose target leaks past train_end."""
    k6 = k6_book.set_index("date")["daily_ret"].sort_index()
    df = feat_df.copy()
    ret_t = k6.reindex(df.index)
    if horizon == 1:
        df["y_reg"] = ret_t
        df["target_end_date"] = df.index
    else:
        # forward mean over t..t+h-1 computed on the K6 book's own calendar
        fwd = k6.rolling(horizon).mean().shift(-(horizon - 1))
        df["y_reg"] = fwd.reindex(df.index)
        end_dates = pd.Series(k6.index, index=k6.index).shift(-(horizon - 1))
        df["target_end_date"] = end_dates.reindex(df.index)
    df["y_pos"] = (ret_t > 0).astype(float)  # day-t sign, for fold pos-rate filter
    return df


def train_wf_regressor(df: pd.DataFrame, feat_cols: list[str],
                       variant: str) -> tuple[pd.DataFrame, dict]:
    import lightgbm as lgb
    data = df.dropna(subset=feat_cols + ["y_reg", "target_end_date"])
    windows = _iter_wf(WF_START, WF_END)
    print(f"[v2:{variant}] WF windows: {len(windows)} (SLIDING {TRAIN_MONTHS}m/{OOT_MONTHS}m/{STEP_MONTHS}m, HC #0)")

    all_preds, fold_diag = [], []
    importance_acc = pd.Series(0.0, index=feat_cols)
    n_skipped_posrate = 0

    for fold_i, (tr_s, tr_e, os_, oe) in enumerate(windows):
        train = data[(data.index >= tr_s) & (data.index < tr_e)]
        # LEAKAGE GUARD: drop train rows whose forward target window crosses train_end
        train = train[train["target_end_date"] < tr_e]
        oot = data[(data.index >= os_) & (data.index < oe)]
        if len(train) < 60 or len(oot) < 5:
            continue

        pos_rate = float(train["y_pos"].mean())
        if pos_rate < MIN_POS_RATE:  # (d) early-fold garbage filter
            n_skipped_posrate += 1
            print(f"[v2:{variant}] fold {fold_i} SKIPPED (train pos_rate={pos_rate:.3f} < {MIN_POS_RATE})")
            fold_diag.append({"fold": fold_i, "oot_start": str(os_.date()),
                              "pos_rate_train": pos_rate, "skipped": True})
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
        all_preds.append(pd.DataFrame({
            "date": oot.index, "e_ret": e_ret, "y_ret": oot["y_reg"].values,
            "fold": fold_i, "oot_start": str(os_.date()),
        }))
        imp = booster.feature_importance(importance_type="gain")
        for c, g in zip(feat_cols, imp):
            importance_acc[c] += float(g)
        ic = float(np.corrcoef(e_ret, oot["y_reg"].values)[0, 1]) if len(oot) > 3 else float("nan")
        fold_diag.append({"fold": fold_i, "oot_start": str(os_.date()),
                          "n_train": int(len(tr)), "n_oot": int(len(oot)),
                          "pos_rate_train": pos_rate, "skipped": False,
                          "best_iter": int(booster.best_iteration or 0),
                          "oot_pearson_ic": ic})
        print(f"[v2:{variant}] fold {fold_i} OOT {os_.date()}: n_train={len(tr)} IC={ic:.3f} best_iter={booster.best_iteration}")

    if not all_preds:
        raise RuntimeError(f"{variant}: no fold produced predictions")
    preds = pd.concat(all_preds, ignore_index=True)
    preds = preds.sort_values(["date", "fold"]).drop_duplicates("date", keep="first").reset_index(drop=True)
    importance_acc /= max(importance_acc.sum(), 1e-9)
    return preds, {"fold_diag": fold_diag, "n_skipped_posrate": n_skipped_posrate,
                   "feature_importance": importance_acc.sort_values(ascending=False).to_dict()}


def gate_book_v2(k6_book: pd.DataFrame, preds: pd.DataFrame,
                 spy_ret_lag1: pd.Series, red_only: bool) -> pd.DataFrame:
    """Gate = sit in cash on day t when E[ret] < GATE_BPS.
    red_only: gate eligible only when SPY ret at t-1 < RED_THRESH (no look-ahead).
    No prediction available -> trade (gate=1)."""
    book = k6_book.copy().sort_values("date").reset_index(drop=True)
    book["date"] = pd.to_datetime(book["date"])
    book["e_ret"] = book["date"].map(preds.set_index("date")["e_ret"])
    book["spy_ret_lag1"] = book["date"].map(spy_ret_lag1)
    bad = book["e_ret"] < GATE_BPS
    if red_only:
        bad = bad & (book["spy_ret_lag1"] < RED_THRESH)
    book["meta_gate"] = (~bad).astype(float)
    book.loc[book["e_ret"].isna(), "meta_gate"] = 1.0
    book["daily_ret_orig"] = book["daily_ret"]
    book["daily_ret"] = book["daily_ret_orig"] * book["meta_gate"]
    return book


def per_day_table(book: pd.DataFrame, spy_ret: pd.Series) -> pd.DataFrame:
    b = book.set_index("date")["daily_ret"]
    spy = spy_ret.reindex(b.index).fillna(0.0)
    cls = pd.Series("flat", index=b.index, dtype=object)
    cls[spy > EVAL_REGIME_THRESH] = "green"
    cls[spy < -EVAL_REGIME_THRESH] = "red"
    return pd.DataFrame({"date": b.index, "daily_ret": b.values,
                         "spy_ret": spy.values, "regime": cls.values})


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
        "n_oot_days": int(s.dropna().shape[0]),
        "worst_red_quarter_dd_pct": tail["worst_red_quarter_dd_pct"],
        "passes_tail_dd_25pct": tail["passes_tail_dd_25pct"],
    }


def main():
    print("[v2] === K=6 Meta-Gate v2 (regime-asymmetric regression) ===")
    t0 = time.time()

    feat_df, feat_cols = load_feature_matrix()
    print(f"[v2] feature matrix: {feat_df.shape[0]} days x {len(feat_cols)} feats (cached from v1)")

    k6_book = pd.read_parquet(K6_BOOK_PATH)
    k6_book["date"] = pd.to_datetime(k6_book["date"])

    prices = _load_prices()
    spy_close = prices[prices["ticker"] == BENCH_SPY].sort_values("date").set_index("date")["close"]
    spy_ret = spy_close.pct_change().dropna()
    spy_ret_lag1 = spy_ret.shift(1)  # known at open of day t

    import mlflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("k6_meta_classifier_v2")

    baseline = summarize(k6_book, spy_ret, spy_close, "K6_ungated_baseline")
    print(f"[v2] baseline: Sharpe={baseline['sharpe']:.2f} Calmar={baseline['calmar']:.2f} gap={baseline['regime_gap']:.2f}")

    report = {"strategy": "k6_meta_classifier_v2",
              "gate_bps_per_day": GATE_BPS, "red_thresh_lag1": RED_THRESH,
              "min_pos_rate_fold_filter": MIN_POS_RATE,
              "wf_spec": {"train_months": TRAIN_MONTHS, "oot_months": OOT_MONTHS,
                          "step_months": STEP_MONTHS, "window_type": "SLIDING (HC #0)"},
              "feature_cols": feat_cols, "ungated_baseline": baseline,
              "variants": {}}

    with mlflow.start_run(run_name="k6_meta_gate_v2_parent") as parent:
        mlflow.log_params({"gate_bps": GATE_BPS, "red_thresh": RED_THRESH,
                           "min_pos_rate": MIN_POS_RATE, "n_features": len(feat_cols),
                           "wf": "sliding_24m_6m_3m"})
        for k, v in baseline.items():
            if isinstance(v, (int, float)) and v is not None and np.isfinite(float(v)):
                mlflow.log_metric(f"baseline_{k}", float(v))

        for spec in VARIANTS:
            name, h, red_only = spec["name"], spec["horizon"], spec["red_only"]
            print(f"\n[v2] ===== VARIANT {name} (horizon={h}d, red_only={red_only}) =====")
            tdf = build_targets(feat_df, k6_book, h)
            preds, meta = train_wf_regressor(tdf, feat_cols, name)
            preds.to_parquet(OUT_DIR / f"oot_predictions_{name}.parquet", index=False)
            gated = gate_book_v2(k6_book, preds, spy_ret_lag1, red_only)
            gated.to_parquet(OUT_DIR / f"gated_book_{name}.parquet", index=False)
            per_day_table(gated, spy_ret).to_parquet(OUT_DIR / f"per_day_{name}.parquet", index=False)

            summ = summarize(gated[["date", "daily_ret"]], spy_ret, spy_close, name)
            n_skip = int((gated["meta_gate"] == 0).sum())
            summ["n_skip_days"] = n_skip
            summ["n_folds_used"] = len([f for f in meta["fold_diag"] if not f.get("skipped")])
            summ["n_folds_skipped_posrate"] = meta["n_skipped_posrate"]
            summ["fold_diag"] = meta["fold_diag"]
            summ["feature_importance_top5"] = dict(list(meta["feature_importance"].items())[:5])
            report["variants"][name] = summ

            with mlflow.start_run(run_name=name, nested=True):
                mlflow.log_params({"horizon_d": h, "red_only": red_only,
                                   "gate_bps": GATE_BPS})
                for k, v in summ.items():
                    if isinstance(v, (bool, int, float)) and v is not None:
                        try:
                            if np.isfinite(float(v)):
                                mlflow.log_metric(k, float(v))
                        except (TypeError, ValueError):
                            pass
                mlflow.log_metric("delta_sharpe_vs_baseline",
                                  float((summ["sharpe"] or 0) - (baseline["sharpe"] or 0)))
            gap = summ["regime_gap"]
            print(f"[v2:{name}] Sharpe={summ['sharpe']:.2f} (base {baseline['sharpe']:.2f}) "
                  f"Calmar={summ['calmar']:.2f} MaxDD={summ['max_dd_pct']:.1f}% "
                  f"gap={gap:.2f} R1_PASS={summ['passes_hc428_r1']} skip_days={n_skip}")

        report["wall_seconds"] = round(time.time() - t0, 1)
        (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, default=str))
        mlflow.log_artifact(str(OUT_DIR / "report.json"))

    print("\n[v2] === FINAL SUMMARY ===")
    print(f"  baseline: Sharpe={baseline['sharpe']:.2f} Calmar={baseline['calmar']:.2f} gap={baseline['regime_gap']:.2f} R1={baseline['passes_hc428_r1']}")
    for name, s in report["variants"].items():
        print(f"  {name:12s}: Sharpe={s['sharpe']:.2f} Calmar={s['calmar']:.2f} "
              f"MaxDD={s['max_dd_pct']:.1f}% gap={s['regime_gap']:.2f} R1={s['passes_hc428_r1']}")
    print(f"[v2] done in {report['wall_seconds']}s -> {OUT_DIR}")
    return report


if __name__ == "__main__":
    main()
