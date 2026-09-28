"""
XGBoost Continuation Model — Multi-minute direction prediction.
Experiment: continuation_multimin_v1

Walk-forward sliding window: 60-day train, 1-day OOT (HC #0).
Target: binary — did price move favorably by >= K ticks within horizon h?
  For each event, we create TWO targets:
    - long_hit: MFE >= K (price went UP by K+ ticks within h)
    - short_hit: |MAE| >= K (price went DOWN by K+ ticks within h)
  We train on the "directional MFE" — max of long_hit, short_hit — predicting
  "will there be a K+ tick move in EITHER direction?" Then direction is inferred
  from the sign of (mfe - |mae|).

Actually simpler and more useful: predict P(MFE >= K) — i.e., is there a tradeable
move of K+ ticks to the UPSIDE within h? Separately predict P(|MAE| >= K) for downside.
We combine into a single model predicting "continuation" = max(mfe, |mae|) >= K.

Grid: h in {60s, 120s, 300s}, K in {3, 5, 8} ticks.
Kill: AUC < 0.55 on first 3 OOT folds.

Features: smart_v3 event features + book features.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

warnings.filterwarnings("ignore")

# Paths
DATA_ROOT = Path("/home/nick/Lvl3Quant/data")
EVENTS_DIR = DATA_ROOT / "processed/mbo_events_smart_v3"
BOOK_DIR = DATA_ROOT / "processed/mbo_book_features"
RELABEL_DIR = DATA_ROOT / "relabel"
OUTPUT_DIR = DATA_ROOT / "models/continuation_v1"

# Walk-forward params
TRAIN_DAYS = 60
OOT_DAYS = 1

# XGBoost params (GPU)
XGB_PARAMS = {
    "device": "cuda",
    "tree_method": "hist",
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 100,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_estimators": 500,
    "early_stopping_rounds": 30,
    "verbosity": 0,
}

# Feature columns from smart_v3 events
# The .npz files have 'features' array and 'feature_names'
# Book features have 'features' with bid/ask levels


def load_date_features(date_str: str) -> np.ndarray | None:
    """Load and concatenate event features + book features for a date."""
    ev_path = EVENTS_DIR / f"{date_str}_mbo_events.npz"
    bk_path = BOOK_DIR / f"{date_str}_book_features.npz"
    if not ev_path.exists() or not bk_path.exists():
        return None

    ev = np.load(ev_path, allow_pickle=True)
    bk = np.load(bk_path, allow_pickle=True)

    ev_feats = ev["features"].astype(np.float32)
    bk_feats = bk["features"].astype(np.float32)

    N_ev = ev_feats.shape[0]
    N_bk = bk_feats.shape[0]

    if N_ev == N_bk:
        return np.hstack([ev_feats, bk_feats])
    else:
        # Align book to events via timestamps
        ev_ts = ev["timestamps"].astype(np.int64)
        bk_ts = bk["timestamps"].astype(np.int64)
        idx = np.searchsorted(bk_ts, ev_ts, side="right") - 1
        idx = np.clip(idx, 0, N_bk - 1)
        bk_aligned = bk_feats[idx]
        return np.hstack([ev_feats, bk_aligned])


def load_date_labels(date_str: str, horizon: str, K: int) -> np.ndarray | None:
    """Load MFE/MAE labels and create binary target: max(MFE, |MAE|) >= K ticks."""
    label_path = RELABEL_DIR / f"mfe_mae_h{horizon}_{date_str}.parquet"
    if not label_path.exists():
        return None

    df = pd.read_parquet(label_path)
    mfe = df["mfe_ticks"].values
    mae = df["mae_ticks"].values  # mae is negative (price went down)

    # Target: is there a K+ tick move in either direction?
    # MFE >= K means upside move of K+ ticks
    # |MAE| >= K means downside move of K+ ticks
    # "Continuation" = there's a tradeable move of K+ ticks
    target = ((mfe >= K) | (np.abs(mae) >= K)).astype(np.float32)

    # Mark NaN labels as -1 (to filter out)
    nan_mask = np.isnan(mfe) | np.isnan(mae)
    target[nan_mask] = -1.0

    return target


def get_available_dates() -> list[str]:
    """Get sorted list of dates with events, book features, AND relabel data."""
    ev_dates = {p.stem.replace("_mbo_events", "") for p in EVENTS_DIR.glob("2026*_mbo_events.npz")}
    bk_dates = {p.stem.replace("_book_features", "") for p in BOOK_DIR.glob("2026*_book_features.npz")}
    # Check which dates have relabel for any horizon
    rl_dates = set()
    for p in RELABEL_DIR.glob("mfe_mae_h*_2026*.parquet"):
        # Extract date from filename: mfe_mae_h60s_20260101.parquet
        parts = p.stem.split("_")
        date_str = parts[-1]
        rl_dates.add(date_str)

    common = sorted(ev_dates & bk_dates & rl_dates)
    return common


def run_walkforward(horizon: str, K: int, dates: list[str],
                    kill_threshold: float = 0.55, kill_folds: int = 3) -> dict:
    """Run sliding window walk-forward for one (horizon, K) cell."""
    print(f"\n{'='*60}")
    print(f"  h={horizon}, K={K} ticks — Walk-Forward ({TRAIN_DAYS}d train, {OOT_DAYS}d OOT)")
    print(f"{'='*60}")

    # Filter dates that have labels for this horizon
    valid_dates = []
    for d in dates:
        lp = RELABEL_DIR / f"mfe_mae_h{horizon}_{d}.parquet"
        if lp.exists():
            valid_dates.append(d)

    if len(valid_dates) < TRAIN_DAYS + kill_folds:
        return {"horizon": horizon, "K": K, "status": "insufficient_dates",
                "n_dates": len(valid_dates)}

    # Walk-forward folds
    fold_results = []
    killed = False

    n_folds = len(valid_dates) - TRAIN_DAYS
    print(f"  Available folds: {n_folds} (dates: {len(valid_dates)})")

    for fold_idx in range(n_folds):
        train_dates = valid_dates[fold_idx:fold_idx + TRAIN_DAYS]
        oot_date = valid_dates[fold_idx + TRAIN_DAYS]

        # Load training data
        X_train_list, y_train_list = [], []
        for d in train_dates:
            X = load_date_features(d)
            y = load_date_labels(d, horizon, K)
            if X is None or y is None:
                continue
            # Filter out NaN labels
            valid_mask = y >= 0
            if valid_mask.sum() < 100:
                continue
            X_train_list.append(X[valid_mask])
            y_train_list.append(y[valid_mask])

        if not X_train_list:
            fold_results.append({"fold": fold_idx, "oot_date": oot_date, "auc": np.nan, "status": "no_train_data"})
            continue

        X_train = np.vstack(X_train_list)
        y_train = np.concatenate(y_train_list)

        # Load OOT data
        X_oot = load_date_features(oot_date)
        y_oot = load_date_labels(oot_date, horizon, K)
        if X_oot is None or y_oot is None:
            fold_results.append({"fold": fold_idx, "oot_date": oot_date, "auc": np.nan, "status": "no_oot_data"})
            continue

        valid_oot = y_oot >= 0
        if valid_oot.sum() < 50:
            fold_results.append({"fold": fold_idx, "oot_date": oot_date, "auc": np.nan, "status": "insufficient_oot"})
            continue

        X_oot_valid = X_oot[valid_oot]
        y_oot_valid = y_oot[valid_oot]

        # Check class balance
        pos_rate_train = y_train.mean()
        pos_rate_oot = y_oot_valid.mean()

        # Handle NaN/Inf in features
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_oot_valid = np.nan_to_num(X_oot_valid, nan=0.0, posinf=0.0, neginf=0.0)

        # Train XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_oot_valid, label=y_oot_valid)

        params = {k: v for k, v in XGB_PARAMS.items()
                  if k not in ("n_estimators", "early_stopping_rounds")}

        try:
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=XGB_PARAMS["n_estimators"],
                evals=[(dval, "oot")],
                early_stopping_rounds=XGB_PARAMS["early_stopping_rounds"],
                verbose_eval=False,
            )

            # Predict OOT
            y_pred = model.predict(dval)
            auc = roc_auc_score(y_oot_valid, y_pred)
        except Exception as e:
            auc = np.nan
            print(f"    Fold {fold_idx} ({oot_date}): ERROR - {e}")
            fold_results.append({"fold": fold_idx, "oot_date": oot_date, "auc": np.nan, "status": f"error: {e}"})
            continue

        fold_results.append({
            "fold": fold_idx,
            "oot_date": oot_date,
            "auc": round(float(auc), 4),
            "pos_rate_train": round(float(pos_rate_train), 4),
            "pos_rate_oot": round(float(pos_rate_oot), 4),
            "n_train": int(len(y_train)),
            "n_oot": int(len(y_oot_valid)),
            "best_iteration": int(model.best_iteration) if hasattr(model, 'best_iteration') else -1,
        })

        print(f"    Fold {fold_idx:3d} | OOT {oot_date} | AUC={auc:.4f} | "
              f"pos_rate={pos_rate_oot:.3f} | n_train={len(y_train):,} | n_oot={len(y_oot_valid):,}")

        # Kill criterion: if first kill_folds all have AUC < threshold
        if fold_idx == kill_folds - 1:
            early_aucs = [r["auc"] for r in fold_results if not np.isnan(r.get("auc", np.nan))]
            if len(early_aucs) >= kill_folds and all(a < kill_threshold for a in early_aucs[:kill_folds]):
                print(f"  ** KILLED: first {kill_folds} folds all AUC < {kill_threshold} "
                      f"(avg={np.mean(early_aucs[:kill_folds]):.4f})")
                killed = True
                break

    # Summary
    valid_aucs = [r["auc"] for r in fold_results if not np.isnan(r.get("auc", np.nan))]
    result = {
        "horizon": horizon,
        "K": K,
        "n_folds_run": len(fold_results),
        "n_valid_folds": len(valid_aucs),
        "mean_auc": round(float(np.mean(valid_aucs)), 4) if valid_aucs else None,
        "std_auc": round(float(np.std(valid_aucs)), 4) if valid_aucs else None,
        "median_auc": round(float(np.median(valid_aucs)), 4) if valid_aucs else None,
        "min_auc": round(float(np.min(valid_aucs)), 4) if valid_aucs else None,
        "max_auc": round(float(np.max(valid_aucs)), 4) if valid_aucs else None,
        "killed": killed,
        "verdict": "GO" if valid_aucs and np.mean(valid_aucs) >= kill_threshold else "NO-GO",
        "fold_details": fold_results,
    }

    print(f"\n  RESULT: h={horizon} K={K} | mean_AUC={result['mean_auc']} | "
          f"verdict={result['verdict']} | folds={len(valid_aucs)}")

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", nargs="+", default=["60s", "120s", "300s"])
    ap.add_argument("--k-values", nargs="+", type=int, default=[3, 5, 8])
    ap.add_argument("--kill-threshold", type=float, default=0.55)
    ap.add_argument("--kill-folds", type=int, default=3)
    ap.add_argument("--mlflow-uri", type=str, default="http://localhost:5000")
    ap.add_argument("--experiment-name", type=str, default="continuation_multimin_v1")
    args = ap.parse_args()

    print("=" * 70)
    print("  XGBoost Continuation Model — Multi-Minute Direction Prediction")
    print("  Sliding Window Walk-Forward (HC #0)")
    print("=" * 70)

    # Setup MLflow
    if HAS_MLFLOW:
        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment(args.experiment_name)
        print(f"  MLflow: {args.mlflow_uri} / {args.experiment_name}")

    # Get available dates
    dates = get_available_dates()
    print(f"  Available dates: {len(dates)} (need {TRAIN_DAYS}+ for walk-forward)")
    if len(dates) < TRAIN_DAYS + 3:
        print(f"  ERROR: Need at least {TRAIN_DAYS + 3} dates, have {len(dates)}")
        sys.exit(1)

    print(f"  Date range: {dates[0]} to {dates[-1]}")
    print(f"  Grid: h in {args.horizons}, K in {args.k_values}")
    print(f"  Kill: AUC < {args.kill_threshold} on first {args.kill_folds} folds")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    t_start = time.time()

    for h in args.horizons:
        for K in args.k_values:
            # MLflow run per (h, K) cell
            if HAS_MLFLOW:
                with mlflow.start_run(run_name=f"h{h}_K{K}"):
                    mlflow.log_params({
                        "horizon": h,
                        "K_ticks": K,
                        "train_days": TRAIN_DAYS,
                        "oot_days": OOT_DAYS,
                        "kill_threshold": args.kill_threshold,
                        "n_dates": len(dates),
                        **{f"xgb_{k}": v for k, v in XGB_PARAMS.items()},
                    })

                    result = run_walkforward(h, K, dates, args.kill_threshold, args.kill_folds)
                    all_results.append(result)

                    # Log metrics
                    if result["mean_auc"] is not None:
                        mlflow.log_metrics({
                            "mean_auc": result["mean_auc"],
                            "std_auc": result["std_auc"],
                            "median_auc": result["median_auc"],
                            "min_auc": result["min_auc"],
                            "max_auc": result["max_auc"],
                            "n_folds": result["n_valid_folds"],
                        })
                    mlflow.log_params({
                        "verdict": result["verdict"],
                        "killed": result["killed"],
                    })
            else:
                result = run_walkforward(h, K, dates, args.kill_threshold, args.kill_folds)
                all_results.append(result)

    elapsed = round(time.time() - t_start, 1)

    # Final summary
    print("\n" + "=" * 70)
    print("  FINAL RESULTS MATRIX")
    print("=" * 70)
    print(f"{'Horizon':<10} {'K(ticks)':<10} {'Mean AUC':<12} {'Std':<8} {'Folds':<8} {'Verdict':<10}")
    print("-" * 60)
    for r in all_results:
        auc_str = f"{r['mean_auc']:.4f}" if r['mean_auc'] is not None else "N/A"
        std_str = f"{r['std_auc']:.4f}" if r['std_auc'] is not None else "N/A"
        print(f"{r['horizon']:<10} {r['K']:<10} {auc_str:<12} {std_str:<8} "
              f"{r['n_valid_folds']:<8} {r['verdict']:<10}")

    print(f"\nTotal elapsed: {elapsed}s")

    # Save results
    results_path = OUTPUT_DIR / "grid_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_s": elapsed,
            "params": {
                "horizons": args.horizons,
                "k_values": args.k_values,
                "train_days": TRAIN_DAYS,
                "xgb_params": XGB_PARAMS,
            },
            "results": all_results,
        }, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
