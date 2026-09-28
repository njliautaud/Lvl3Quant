#!/usr/bin/env python3
"""LGBM with magnitude-weighted samples. Hypothesis: weighting by |label|^0.5
improves high-confidence IC and directional accuracy by focusing on large moves.

Data: mbo_features_cache (340 features) + dl_events_cache (mid_prices).
Label: 1s forward return = (mid[t+10] - mid[t]) / mid[t]  (10 bars @ 100ms = 1s).
"""

import json, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
import lightgbm as lgb

FEAT_DIR   = Path("/home/jupiter/lvl3quant/data/processed/mbo_features_cache")
EVT_DIR    = Path("/home/jupiter/lvl3quant/data/processed/dl_events_cache")
RESULTS    = Path("/home/jupiter/Lvl3Quant/alpha_discovery/results/lgbm_magnitude_weighted")
RESULTS.mkdir(parents=True, exist_ok=True)

BARS_1S    = 10     # 10 bars x 100ms = 1 second forward
WEIGHT_EXP = 0.5   # sample_weight = |label|^WEIGHT_EXP
TRAIN_DAYS = 45
TEST_DAYS  = 15
WARMUP     = 5000  # skip first N bars of each day (rolling feature warmup)
MLFLOW_URI = "http://localhost:5002"


def init_mlflow():
    """Try to connect to MLflow with a hard 10s timeout. Returns (mlflow_module, ok)."""
    import threading, importlib
    result = {"ok": False, "mlflow": None, "err": None}

    def _try():
        try:
            import mlflow as mf
            mf.set_tracking_uri(MLFLOW_URI)
            mf.set_experiment("LGBM_MagnitudeWeighted")
            result["ok"] = True
            result["mlflow"] = mf
        except Exception as e:
            result["err"] = str(e)

    t = threading.Thread(target=_try, daemon=True)
    t.start()
    t.join(timeout=10)
    if t.is_alive():
        result["err"] = "MLflow init timed out after 10s"
    return result["mlflow"], result["ok"], result["err"]


def compute_labels_1s(mid_prices: np.ndarray) -> np.ndarray:
    """Forward return at +BARS_1S bars. Last BARS_1S rows set to NaN."""
    N = len(mid_prices)
    labels = np.full(N, np.nan, dtype=np.float32)
    valid = mid_prices > 0
    labels[:N - BARS_1S] = np.where(
        valid[:N - BARS_1S] & valid[BARS_1S:],
        (mid_prices[BARS_1S:].astype(np.float32) - mid_prices[:N - BARS_1S].astype(np.float32))
        / mid_prices[:N - BARS_1S].astype(np.float32),
        np.nan
    )
    return labels


def load_one_day(feat_path: Path, evt_path: Path):
    fd = np.load(feat_path, allow_pickle=False)
    ed = np.load(evt_path,  allow_pickle=False)
    feats = fd["mbo_features"]
    mids  = ed["mid_prices"].astype(np.float32)
    N = min(len(feats), len(mids))
    feats  = feats[:N][WARMUP:]
    labels = compute_labels_1s(mids[:N])[WARMUP:]
    return feats, labels


def load_day_range(feat_files, evt_files):
    all_X, all_y = [], []
    for ff, ef in zip(feat_files, evt_files):
        try:
            X, y = load_one_day(ff, ef)
            all_X.append(X)
            all_y.append(y)
        except Exception as e:
            print(f"  skip {ff.name}: {e}", flush=True)
    if not all_X:
        return np.empty((0, 340), dtype=np.float32), np.empty(0, dtype=np.float32)
    return np.concatenate(all_X, axis=0), np.concatenate(all_y, axis=0)


def confidence_eval(y_true, y_pred, label=""):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    abs_p  = np.abs(y_pred)
    tiers  = [("all", 0), ("top50", 50), ("top25", 75), ("top10", 90)]
    results = {}
    for name, pct in tiers:
        mask = abs_p >= np.percentile(abs_p, pct) if pct > 0 else np.ones(len(y_pred), dtype=bool)
        n = mask.sum()
        if n < 10:
            continue
        ic = float(spearmanr(y_pred[mask], y_true[mask])[0])
        # DirAcc: only evaluate on rows where y_true != 0.
        # ~67% of labels are exactly zero (mid price flat in 10 bars at float32 precision).
        # sign(0) == 0 which never matches sign(pred), so including zeros artificially
        # deflates DirAcc even when IC is strongly positive. We exclude them here.
        nz_mask = mask & (y_true != 0)
        n_nz = nz_mask.sum()
        da = float(np.mean(np.sign(y_pred[nz_mask]) == np.sign(y_true[nz_mask]))) if n_nz > 10 else float('nan')
        lm = nz_mask & (y_pred > 0)
        sm = nz_mask & (y_pred < 0)
        dl = float(np.mean(np.sign(y_pred[lm]) == np.sign(y_true[lm]))) if lm.sum() > 10 else float('nan')
        ds = float(np.mean(np.sign(y_pred[sm]) == np.sign(y_true[sm]))) if sm.sum() > 10 else float('nan')
        results[name] = dict(ic=ic, dir_acc=da, dir_long=dl, dir_short=ds, n=int(n), n_dirac=int(n_nz))
        tag = f"[{label} {name}]" if label else f"[{name}]"
        print(f"  {tag:24s} n={n:7d} (nz={n_nz:7d})  IC={ic:+.4f}  DirAcc={da:.3f}  Long={dl:.3f}  Short={ds:.3f}", flush=True)
    return results


def main():
    # Match feature and event files by date
    feat_dates = {f.name.replace("_mbo_features.npz", ""): f
                  for f in sorted(FEAT_DIR.glob("*_mbo_features.npz"))}
    evt_dates  = {f.name.replace("_event_tokens.npz",  ""): f
                  for f in sorted(EVT_DIR.glob("*_event_tokens.npz"))}
    common = sorted(set(feat_dates) & set(evt_dates))

    feat_files = [feat_dates[d] for d in common]
    evt_files  = [evt_dates[d]  for d in common]

    print(f"Found {len(common)} matched days: {common[0]} -> {common[-1]}", flush=True)

    n = len(common)
    folds = []
    i = 0
    while i + TRAIN_DAYS + TEST_DAYS <= n:
        folds.append((i, i + TRAIN_DAYS, i + TRAIN_DAYS + TEST_DAYS))
        i += TEST_DAYS
    print(f"Walk-forward: {len(folds)} folds, {TRAIN_DAYS}d train / {TEST_DAYS}d test", flush=True)

    # MLflow init with timeout
    print("Connecting to MLflow...", flush=True)
    mlflow, mlflow_ok, mlflow_err = init_mlflow()
    if mlflow_ok:
        print(f"MLflow connected at {MLFLOW_URI}", flush=True)
    else:
        print(f"MLflow unavailable ({mlflow_err}) -- logging locally only", flush=True)

    all_results = []

    for fold_i, (ts, te, end) in enumerate(folds):
        print(f"\n=== Fold {fold_i:02d} | train [{ts}:{te}] ({common[ts]}..{common[te-1]}) "
              f"test [{te}:{end}] ({common[te]}..{common[end-1]}) ===", flush=True)

        print(f"  Loading train ({te-ts} days)...", flush=True)
        Xtr, ytr = load_day_range(feat_files[ts:te], evt_files[ts:te])
        print(f"  Loading test ({end-te} days)...", flush=True)
        Xte, yte = load_day_range(feat_files[te:end], evt_files[te:end])

        if Xtr.shape[0] == 0 or Xte.shape[0] == 0:
            print("  Empty split, skipping.", flush=True)
            continue

        # Remove NaN labels
        ok_tr = np.isfinite(ytr);  Xtr, ytr = Xtr[ok_tr], ytr[ok_tr]
        ok_te = np.isfinite(yte);  Xte, yte = Xte[ok_te], yte[ok_te]

        # Remove NaN features
        ok_tr2 = np.all(np.isfinite(Xtr), axis=1); Xtr, ytr = Xtr[ok_tr2], ytr[ok_tr2]
        ok_te2 = np.all(np.isfinite(Xte), axis=1); Xte, yte = Xte[ok_te2], yte[ok_te2]

        # Magnitude weights: |label|^0.5, normalized to mean=1
        sample_weight = np.abs(ytr) ** WEIGHT_EXP
        sw_mean = sample_weight.mean()
        if sw_mean > 0:
            sample_weight /= sw_mean

        print(f"  Train: {len(Xtr):,}  Test: {len(Xte):,}  Features: {Xtr.shape[1]}", flush=True)
        print(f"  Weight range: {sample_weight.min():.4f} - {sample_weight.max():.4f}  mean={sample_weight.mean():.4f}", flush=True)

        print(f"  Training LGBM...", flush=True)
        params = dict(
            objective="regression", metric="rmse", verbosity=1,
            n_estimators=200, learning_rate=0.1, num_leaves=31,
            min_child_samples=100, subsample=0.8, colsample_bytree=0.8,
            n_jobs=12
        )
        model = lgb.LGBMRegressor(**params)
        model.fit(Xtr, ytr, sample_weight=sample_weight)
        print(f"  Training done. Running confidence eval...", flush=True)

        preds = model.predict(Xte)
        conf  = confidence_eval(yte, preds, label=f"fold{fold_i:02d}")

        fold_result = {"fold": fold_i, "n_train": int(len(Xtr)),
                       "n_test": int(len(Xte)), "conf_ic": conf}
        all_results.append(fold_result)

        if mlflow_ok and mlflow is not None:
            try:
                with mlflow.start_run(run_name=f"fold_{fold_i:02d}"):
                    mlflow.log_param("weight_exp", WEIGHT_EXP)
                    mlflow.log_param("fold", fold_i)
                    mlflow.log_param("n_train", len(Xtr))
                    for tier, m in conf.items():
                        mlflow.log_metric(f"{tier}_ic", m["ic"])
                        mlflow.log_metric(f"{tier}_dir_acc", m["dir_acc"])
                        if not np.isnan(m["dir_long"]):
                            mlflow.log_metric(f"{tier}_dir_long", m["dir_long"])
                        if not np.isnan(m["dir_short"]):
                            mlflow.log_metric(f"{tier}_dir_short", m["dir_short"])
                print(f"  MLflow logged fold {fold_i}.", flush=True)
            except Exception as e:
                print(f"  MLflow log error: {e}", flush=True)

        with open(RESULTS / f"fold_{fold_i:02d}.json", "w") as f:
            json.dump(fold_result, f, indent=2)

        # Save predictions for fill simulation
        np.savez_compressed(
            RESULTS / f"fold_{fold_i:02d}_predictions.npz",
            predictions=preds.astype(np.float32),
            labels=yte.astype(np.float32)
        )
        print(f"  Predictions saved: fold_{fold_i:02d}_predictions.npz", flush=True)

    # Summary
    print("\n=== SUMMARY -- LGBM Magnitude-Weighted (weight_exp=0.5) ===", flush=True)
    for tier in ["all", "top50", "top25", "top10"]:
        ics = [r["conf_ic"][tier]["ic"]       for r in all_results if tier in r["conf_ic"]]
        das = [r["conf_ic"][tier]["dir_acc"]  for r in all_results if tier in r["conf_ic"]]
        dls = [r["conf_ic"][tier]["dir_long"] for r in all_results if tier in r["conf_ic"]
               and not np.isnan(r["conf_ic"][tier]["dir_long"])]
        dss = [r["conf_ic"][tier]["dir_short"] for r in all_results if tier in r["conf_ic"]
               and not np.isnan(r["conf_ic"][tier]["dir_short"])]
        if ics:
            dl_str = f"{np.mean(dls):.3f}" if dls else "n/a"
            ds_str = f"{np.mean(dss):.3f}" if dss else "n/a"
            print(f"  {tier:6s}  folds={len(ics)}  avg IC={np.mean(ics):+.4f}  "
                  f"avg DirAcc={np.mean(das):.3f}  Long={dl_str}  Short={ds_str}", flush=True)

    with open(RESULTS / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
