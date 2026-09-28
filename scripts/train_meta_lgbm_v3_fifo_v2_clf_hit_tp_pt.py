#!/usr/bin/env python3
"""
Meta-LGBM v3 FIFO v2 — rank-norm + confluence-stack pivot (HC #280(B), #275(A), #285).

Pivot from v1 (ic_mean ~+0.007, near-zero edge):
  1. Per-day rank normalization on ALL features (HC #280(B)) so vol-regime
     shifts don't destroy LGBM splits.
  2. Add confluence-stack features (HC #275(A)): book imbalance L3,
     book depth imbalance 5lvl, signal persistence (5 & 20 evals),
     volatility regime, time-of-day regime bin.

Target: tp4sl3_short_hit_tp (HC #69: short side has best edge).
Features per row:
    pred_1s, pred_5s, pred_10s,
    ms_l3_imb, ms_depth_imb_5,
    signal_persist_5, signal_persist_20,
    vol_30s_day_pct, tod_regime_bin
  -> per-day rank-normalized to [0,1] BEFORE concat across days.

Source data:
  enriched parquets: /home/jupiter/Lvl3Quant/output/meta_lgbm_features/<DATE>_signals_enriched.parquet
  FIFO labels:       /home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3_fifo_labels/<DATE>_fifo_labels.npz
  alignment: parquet.pred_idx -> FIFO row index (1:1, verified).

CV: walk-forward by chronological date. For each split S>=2, train on dates[0..S-1],
eval on dates[S]. Filled-only rows for train+eval.
"""
import os, sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
import mlflow

ENRICHED_DIR = Path("/home/jupiter/Lvl3Quant/output/meta_lgbm_features")
FIFO_DIR     = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3_fifo_labels")
OUT_DIR      = Path("/home/jupiter/Lvl3Quant/output/meta_lgbm_v3_fifo_v2_clf_hit_tp")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CAP_TICKS = 20.0
TARGET_KEY = "tp4sl3_short_hit_tp"
FILLED_KEY = "tp4sl3_short_filled"

# HC #289 v3.1 hypothesis test — PatchTST predictions as input features on clf gate.
# Compare vs A1 baseline (ic_mean=+0.0530 on same 5 raw + same tp4sl3_short_hit_tp target).
RAW_FEATURES = [
    "pred_1s", "pred_5s", "pred_10s",
    "vol_30s_day_pct",    # vol regime (intraday percentile)
    "tod_regime_bin",     # time-of-day regime bin
    "pt_pred_1s", "pt_pred_5s", "pt_pred_10s",  # HC #289 — PatchTST preds as inputs
]

# Dates where ms_* book features were broken (zip read failed in merge log)
DATE_BLACKLIST = {"20260429"}

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("MetaLGBM_v3_FIFO_v2_clf_hit_tp_pt")


def per_day_rank_norm(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Per-day rank-normalize each feature into [0,1].

    Uses pandas rank(method='average', pct=True). NaN stays NaN.
    LGBM handles NaN natively, so this is safe.
    Within a single day all rows share the same 'date', so this is just
    a vectorized rank+divide per column.
    """
    out = df.copy()
    n = len(df)
    if n == 0:
        return out
    for c in cols:
        v = df[c]
        # rank: ties get average, pct=True gives values in (0,1]
        out[c] = v.rank(method="average", pct=True)
    return out


def load_date(date_str: str):
    """Load one date's enriched parquet + FIFO labels, return per-day rank-normed frame.

    Returns DataFrame with columns: RAW_FEATURES (rank-normed) + ['y'] + ['date'].
    Filtered to filled-only rows.
    """
    ep = ENRICHED_DIR / f"{date_str}_signals_enriched.parquet"
    fp = FIFO_DIR / f"{date_str}_fifo_labels.npz"
    if not ep.exists() or not fp.exists():
        return None
    if date_str in DATE_BLACKLIST:
        return None
    df = pd.read_parquet(ep)
    fifo = np.load(fp)
    # Verify all required cols
    missing = [c for c in RAW_FEATURES if c not in df.columns]
    if missing:
        print(f"[{date_str}] MISSING cols {missing} — skipping")
        return None
    # Resolve target via pred_idx -> FIFO row
    idx = df["pred_idx"].values.astype(np.int64)
    n_fifo = len(fifo["window_k"])
    if idx.max() >= n_fifo:
        keep = idx < n_fifo
        df = df.loc[keep].reset_index(drop=True)
        idx = df["pred_idx"].values.astype(np.int64)
    filled = fifo[FILLED_KEY][idx].astype(bool)
    if filled.sum() == 0:
        return None
    target = fifo[TARGET_KEY][idx].astype(np.float32)
    target = np.clip(target, -CAP_TICKS, CAP_TICKS)
    df = df.loc[filled].reset_index(drop=True)
    target = target[filled]
    # Per-day rank normalization on RAW_FEATURES
    df_rn = per_day_rank_norm(df, RAW_FEATURES)
    out = df_rn[RAW_FEATURES].copy()
    out["y"] = target
    out["date"] = date_str
    return out


def main():
    dates = sorted({p.name.split("_")[0] for p in ENRICHED_DIR.glob("*_signals_enriched.parquet")})
    dates = [d for d in dates if d not in DATE_BLACKLIST]
    print(f"Found {len(dates)} candidate enriched dates (after blacklist)")
    per_day = []
    for d in dates:
        r = load_date(d)
        if r is None:
            continue
        per_day.append((d, r))
        print(f"  {d}: n_filled={len(r):>6}")
    print(f"Loaded {len(per_day)} usable dates")

    if len(per_day) < 3:
        print("ERROR: need >=3 dates for walk-forward")
        sys.exit(1)

    feat_names = list(RAW_FEATURES)

    with mlflow.start_run(run_name=f"meta_lgbm_v3_fifo_v2_clf_hit_tp_{int(time.time())}"):
        mlflow.log_params({
            "target": TARGET_KEY,
            "cap_ticks": CAP_TICKS,
            "n_dates": len(per_day),
            "n_features": len(feat_names),
            "features": ",".join(feat_names),
            "per_day_rank_norm": True,
            "confluence_features": "ms_l3_imb,ms_depth_imb_5,signal_persist_5,signal_persist_20,vol_30s_day_pct,tod_regime_bin",
            "model": "LightGBM regressor",
            "node": "Jupiter CPU",
            "filled_only": True,
            "pivot_from": "MetaLGBM_v3_FIFO (raw preds + 96-dim embed)",
        })
        all_results = []
        for split_idx in range(2, len(per_day)):
            train_dates = per_day[:split_idx]
            eval_date, eval_df = per_day[split_idx]
            train_df = pd.concat([d[1] for d in train_dates], ignore_index=True)
            X_tr = train_df[feat_names].values.astype(np.float32)
            y_tr = train_df["y"].values.astype(np.float32)
            X_ev = eval_df[feat_names].values.astype(np.float32)
            y_ev = eval_df["y"].values.astype(np.float32)
            params = {
                "objective": "binary", "metric": "binary_logloss",
                "num_leaves": 31, "learning_rate": 0.05,
                "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
                "min_data_in_leaf": 100, "verbose": -1, "num_threads": 16,
            }
            ds_tr = lgb.Dataset(X_tr, label=y_tr, feature_name=feat_names)
            ds_ev = lgb.Dataset(X_ev, label=y_ev, reference=ds_tr)
            model = lgb.train(params, ds_tr, num_boost_round=500,
                              valid_sets=[ds_ev], valid_names=["eval"],
                              callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
            preds = model.predict(X_ev)
            ic, _ = spearmanr(preds, y_ev)
            top10_mask = preds >= np.quantile(preds, 0.9)
            top10_mean = float(y_ev[top10_mask].mean())
            top10_n = int(top10_mask.sum())
            bottom10_mask = preds <= np.quantile(preds, 0.1)
            bottom10_mean = float(y_ev[bottom10_mask].mean())
            bottom10_n = int(bottom10_mask.sum())
            rmse = float(np.sqrt(np.mean((preds - y_ev) ** 2)))
            print(f"split={split_idx:2d}  eval_date={eval_date}  "
                  f"n_tr={len(y_tr):>6}  n_ev={len(y_ev):>5}  "
                  f"IC={ic:+.4f}  top10_mean={top10_mean:+.3f}t (n={top10_n})  "
                  f"bottom10_mean={bottom10_mean:+.3f}t (n={bottom10_n})  RMSE={rmse:.3f}")
            mlflow.log_metric(f"ic_split_{split_idx}", float(ic), step=split_idx)
            mlflow.log_metric(f"top10_split_{split_idx}", top10_mean, step=split_idx)
            mlflow.log_metric(f"bottom10_split_{split_idx}", bottom10_mean, step=split_idx)
            mlflow.log_metric(f"rmse_split_{split_idx}", rmse, step=split_idx)
            all_results.append({"split": split_idx, "eval_date": eval_date,
                                "ic": float(ic), "top10_mean": top10_mean,
                                "bottom10_mean": bottom10_mean, "rmse": rmse,
                                "n_train": int(len(y_tr)), "n_eval": int(len(y_ev))})
            model.save_model(str(OUT_DIR / f"meta_lgbm_split_{split_idx:02d}.txt"))
        with open(OUT_DIR / "results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        ics = [r["ic"] for r in all_results]
        top10s = [r["top10_mean"] for r in all_results]
        bottom10s = [r["bottom10_mean"] for r in all_results]
        mlflow.log_metric("ic_mean", float(np.mean(ics)))
        mlflow.log_metric("ic_median", float(np.median(ics)))
        mlflow.log_metric("top10_mean", float(np.mean(top10s)))
        mlflow.log_metric("bottom10_mean", float(np.mean(bottom10s)))
        print(f"\nDONE  ic_mean={np.mean(ics):+.4f}  ic_median={np.median(ics):+.4f}  "
              f"top10_mean={np.mean(top10s):+.3f}t  bottom10_mean={np.mean(bottom10s):+.3f}t")
        mlflow.log_artifact(str(OUT_DIR / "results.json"))


if __name__ == "__main__":
    main()
