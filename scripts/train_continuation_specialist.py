#!/usr/bin/env python3
"""
train_continuation_specialist.py — HC #470 R5 continuation-specialist smoke.

Train a small model that, given v4's per-event prediction heads at time t,
predicts whether the prediction's stream-life will be LONG (>= 4 strides)
— i.e. will the v4 sign call hold for >=4 consecutive future predictions.

This is the smoke version: lightweight XGBoost + sklearn MLP, walk-forward
15-train / 1-test sliding, on the 32 champion-overlap OOT days that have
both v4 multi-head predictions and HC #470 R1 stream-native labels.

Goal: prove that the v4 heads carry information about FUTURE STREAM
COHERENCE (continuation), not just future return at a fixed horizon.

This is the Phase C smoke from HC #470 R8 — proves the concept before
the full CNN-Mamba retrain. Pure CPU on Jupiter, ~2-3 min total.

Outputs:
  output/continuation_specialist_smoke/preds_<model>_<date>.npz
  output/continuation_specialist_smoke/REPORT.md
  output/continuation_specialist_smoke/continuation_specialist.DONE
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/jupiter/Lvl3Quant")
V4_DIR = ROOT / "output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate"
LABEL_DIR = ROOT / "output/stream_native_labels"
OUT_DIR = ROOT / "output/continuation_specialist_smoke"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STREAM_LIFE_BINARY_THRESHOLD = 4   # >=4 strides confirmed = "long stream-life"
K_TRAIN_DAYS = 15
PRED_FEAT_PREFIXES = ("pred_",)


def load_day_features_and_labels(date_str: str) -> pd.DataFrame:
    v4_path = V4_DIR / f"oot_{date_str}.npz"
    lab_path = LABEL_DIR / f"labels_{date_str}.npz"
    if not v4_path.exists() or not lab_path.exists():
        return pd.DataFrame()
    v = np.load(v4_path, allow_pickle=True)
    L = np.load(lab_path, allow_pickle=True)
    if "stream_life_log_ret_1s" not in L.files:
        return pd.DataFrame()
    n = v["pred_log_ret_1s"].shape[0]
    cols = {}
    for k in v.files:
        if any(k.startswith(p) for p in PRED_FEAT_PREFIXES) and v[k].shape == (n,):
            cols[k] = v[k].astype(np.float32)
    cols["stream_life_1s"] = L["stream_life_log_ret_1s"].astype(np.int32)
    cols["stream_int_return_1s"] = L["stream_int_return_log_ret_1s"].astype(np.float32)
    cols["target_log_ret_1s"] = v["target_log_ret_1s"].astype(np.float32)
    cols["date"] = date_str
    df = pd.DataFrame(cols)
    df["y_long_stream"] = (df["stream_life_1s"] >= STREAM_LIFE_BINARY_THRESHOLD).astype(np.int8)
    # Drop rows where the basic 1s target is NaN (no ground truth)
    df = df.dropna(subset=["target_log_ret_1s"]).reset_index(drop=True)
    return df


def main():
    t0 = time.time()
    print("=" * 78)
    print("HC #470 R5 — CONTINUATION-SPECIALIST SMOKE (Phase C)")
    print("=" * 78)

    label_npzs = sorted(LABEL_DIR.glob("labels_*.npz"))
    dates = [p.stem.replace("labels_", "") for p in label_npzs]
    print(f"[setup] {len(dates)} dates with stream labels.")

    day_dfs = {}
    for d in dates:
        df = load_day_features_and_labels(d)
        if not df.empty:
            day_dfs[d] = df
    avail = sorted(day_dfs.keys())
    print(f"[setup] aligned per-day frames: {len(avail)}")
    if len(avail) < K_TRAIN_DAYS + 1:
        print(f"[FATAL] not enough days for {K_TRAIN_DAYS}-day train (have {len(avail)})")
        return

    feat_cols = [c for c in day_dfs[avail[0]].columns
                 if c.startswith("pred_") and c not in {"pred_k"}]
    print(f"[setup] feature dim = {len(feat_cols)}")
    print(f"[setup] target = y_long_stream  (stream_life_1s >= {STREAM_LIFE_BINARY_THRESHOLD})")

    # Walk-forward
    folds = []
    for i in range(K_TRAIN_DAYS, len(avail)):
        train_dates = avail[i - K_TRAIN_DAYS:i]
        test_date = avail[i]
        folds.append((train_dates, test_date))
    print(f"[setup] {len(folds)} walk-forward folds (K_train_days={K_TRAIN_DAYS})")

    # Compute pos balance
    pos_rates = []
    for d in avail:
        df = day_dfs[d]
        pos_rates.append(float(df["y_long_stream"].mean()))
    print(f"[setup] per-day positive rate: mean={np.mean(pos_rates):.2%} min={np.min(pos_rates):.2%} max={np.max(pos_rates):.2%}")

    all_results = {"xgb": [], "mlp": []}

    # --- XGBoost ---
    import xgboost as xgb
    print("\n-- XGBoost (CPU) --")
    for i, (tr_dates, te_date) in enumerate(folds, 1):
        Xtr = pd.concat([day_dfs[d][feat_cols] for d in tr_dates], ignore_index=True).values.astype(np.float32)
        ytr = pd.concat([day_dfs[d]["y_long_stream"] for d in tr_dates], ignore_index=True).values.astype(np.int32)
        Xte = day_dfs[te_date][feat_cols].values.astype(np.float32)
        yte = day_dfs[te_date]["y_long_stream"].values.astype(np.int32)
        pos_w = max(1.0, (1 - ytr.mean()) / max(1e-6, ytr.mean()))
        dtr = xgb.DMatrix(Xtr, label=ytr)
        dte = xgb.DMatrix(Xte, label=yte)
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 5,
            "eta": 0.07,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "scale_pos_weight": float(pos_w),
            "tree_method": "hist",
            "verbosity": 0,
        }
        t1 = time.time()
        booster = xgb.train(params, dtr, num_boost_round=300, evals=[(dte, "test")], verbose_eval=False)
        prob = booster.predict(dte)
        dur = time.time() - t1
        # Lift at top-5% / 10% / 20% confidence
        top_lift = {}
        for top_q in (0.95, 0.90, 0.80):
            thr = np.quantile(prob, top_q)
            mask = prob >= thr
            if mask.sum() > 0:
                base = yte.mean()
                lift = float(yte[mask].mean() / max(base, 1e-6))
                top_lift[f"top_{int((1-top_q)*100)}pct_lift"] = lift
        np.savez_compressed(
            OUT_DIR / f"preds_xgb_{te_date}.npz",
            prob=prob.astype(np.float32),
            y_true=yte.astype(np.int8),
            stream_life=day_dfs[te_date]["stream_life_1s"].values.astype(np.int32),
            stream_int=day_dfs[te_date]["stream_int_return_1s"].values.astype(np.float32),
            date=np.array([te_date], dtype="<U8"),
        )
        result = {
            "fold": i, "test_date": te_date, "n_test": int(len(yte)),
            "pos_rate_train": float(ytr.mean()), "pos_rate_test": float(yte.mean()),
            "prob_mean": float(prob.mean()), "dur_s": dur, **top_lift,
        }
        all_results["xgb"].append(result)
        print(f"  [xgb] fold {i}/{len(folds)} test={te_date} pos_tr={ytr.mean():.3f} pos_te={yte.mean():.3f} "
              f"top5%_lift={top_lift.get('top_5pct_lift',0):.2f} t={dur:.1f}s")

    # --- sklearn MLP ---
    print("\n-- sklearn MLP --")
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    for i, (tr_dates, te_date) in enumerate(folds, 1):
        Xtr = pd.concat([day_dfs[d][feat_cols] for d in tr_dates], ignore_index=True).values.astype(np.float32)
        ytr = pd.concat([day_dfs[d]["y_long_stream"] for d in tr_dates], ignore_index=True).values.astype(np.int32)
        Xte = day_dfs[te_date][feat_cols].values.astype(np.float32)
        yte = day_dfs[te_date]["y_long_stream"].values.astype(np.int32)
        scaler = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
        t1 = time.time()
        clf = MLPClassifier(
            hidden_layer_sizes=(128, 64), activation="relu",
            solver="adam", alpha=1e-4, batch_size=4096,
            learning_rate_init=2e-3, max_iter=12, random_state=0,
            verbose=False,
        )
        clf.fit(Xtr_s, ytr)
        prob = clf.predict_proba(Xte_s)[:, 1]
        dur = time.time() - t1
        top_lift = {}
        for top_q in (0.95, 0.90, 0.80):
            thr = np.quantile(prob, top_q)
            mask = prob >= thr
            if mask.sum() > 0:
                base = yte.mean()
                top_lift[f"top_{int((1-top_q)*100)}pct_lift"] = float(yte[mask].mean() / max(base, 1e-6))
        np.savez_compressed(
            OUT_DIR / f"preds_mlp_{te_date}.npz",
            prob=prob.astype(np.float32),
            y_true=yte.astype(np.int8),
            stream_life=day_dfs[te_date]["stream_life_1s"].values.astype(np.int32),
            stream_int=day_dfs[te_date]["stream_int_return_1s"].values.astype(np.float32),
            date=np.array([te_date], dtype="<U8"),
        )
        result = {
            "fold": i, "test_date": te_date, "n_test": int(len(yte)),
            "pos_rate_train": float(ytr.mean()), "pos_rate_test": float(yte.mean()),
            "prob_mean": float(prob.mean()), "dur_s": dur, **top_lift,
        }
        all_results["mlp"].append(result)
        print(f"  [mlp] fold {i}/{len(folds)} test={te_date} pos_tr={ytr.mean():.3f} pos_te={yte.mean():.3f} "
              f"top5%_lift={top_lift.get('top_5pct_lift',0):.2f} t={dur:.1f}s")

    # Aggregate
    summary_rows = []
    for model_name, results in all_results.items():
        if not results:
            continue
        lifts5 = [r.get("top_5pct_lift", np.nan) for r in results]
        lifts10 = [r.get("top_10pct_lift", np.nan) for r in results]
        lifts20 = [r.get("top_20pct_lift", np.nan) for r in results]
        summary_rows.append({
            "model": model_name,
            "n_folds": len(results),
            "mean_top_5pct_lift": float(np.nanmean(lifts5)),
            "mean_top_10pct_lift": float(np.nanmean(lifts10)),
            "mean_top_20pct_lift": float(np.nanmean(lifts20)),
            "median_top_5pct_lift": float(np.nanmedian(lifts5)),
            "mean_dur_s": float(np.mean([r["dur_s"] for r in results])),
        })

    sum_df = pd.DataFrame(summary_rows)
    print("\n=== SUMMARY ===")
    print(sum_df.to_string(index=False))

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump({"all_results": all_results, "summary": summary_rows,
                   "config": {
                       "k_train_days": K_TRAIN_DAYS,
                       "stream_life_threshold": STREAM_LIFE_BINARY_THRESHOLD,
                       "n_features": len(feat_cols),
                   }}, f, indent=2)

    with open(OUT_DIR / "REPORT.md", "w") as f:
        f.write("# Continuation-Specialist Smoke (HC #470 R5 / Phase C)\n\n")
        f.write(f"Goal: prove v4 heads carry info about FUTURE STREAM COHERENCE — not just future return.\n\n")
        f.write(f"- Target: y_long_stream = stream_life_log_ret_1s >= {STREAM_LIFE_BINARY_THRESHOLD}.\n")
        f.write(f"- Walk-forward: {K_TRAIN_DAYS}-day-train / 1-day-test sliding, {len(folds)} folds.\n")
        f.write(f"- Features: {len(feat_cols)} v4 prediction heads.\n")
        f.write(f"- Per-day positive rate: mean={np.mean(pos_rates):.2%} min={np.min(pos_rates):.2%} max={np.max(pos_rates):.2%}.\n\n")
        f.write("## Summary lift (predicted top-q% confidence vs base rate)\n\n")
        f.write(sum_df.to_markdown(index=False))
        f.write("\n\nLift >1.0 means the top-confidence band contains more long-stream events than the base rate. "
                ">2.0 means roughly 2x concentration — the head is a meaningful continuation predictor.\n")

    (OUT_DIR / "continuation_specialist.DONE").touch()
    print(f"\n[done] total {time.time()-t0:.1f}s. output -> {OUT_DIR}")


if __name__ == "__main__":
    main()
