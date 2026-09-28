#!/usr/bin/env python3
"""
Fill Probability XGBoost Model — Traditional/Rules-Based Execution Research
===========================================================================
Trains XGBoost models to predict whether a limit order at the BBO would get
filled within 1s, 3s, and 10s, given market microstructure state.

Walk-forward training: 60-day sliding window, 1-day OOT, oldest-day-drop.
Feeds into Neptune's AI execution models (HC #77).

Output: /home/jupiter/Lvl3Quant/output/fill_prob_xgb/
"""

import argparse
import glob
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    average_precision_score,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FEATURES_DIR = Path("/home/jupiter/Lvl3Quant/output/exec_features_v1")
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/fill_prob_xgb")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_WINDOW = 60  # sliding window days
HORIZONS = ["1s", "3s", "10s"]  # fill horizons to model
HORIZON_COL_MAP = {"1s": 0, "3s": 1, "10s": 2}  # column indices in features array

# Features to use (indices into the 44-feature array)
# Exclude fill_prob targets (0,1,2) — those are our labels
FEATURE_INDICES = list(range(3, 44))  # 41 microstructure features

# Binarize threshold: fill_prob > threshold => filled=1
FILL_THRESHOLD = 0.5

# XGBoost params — tuned for tabular fill probability
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "nthread": 1,  # per-model thread; parallelism via ProcessPool
    "verbosity": 0,
    "seed": 42,
}
NUM_BOOST_ROUNDS = 500
EARLY_STOPPING_ROUNDS = 30

# Regime definitions based on spread_ticks (col 20) and price_volatility (col 43)
REGIME_SPREAD_THRESH = 1.5  # ticks; <=1 = tight, >1.5 = wide
REGIME_VOL_THRESH_HIGH = 0.75  # quantile for high-vol classification

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUTPUT_DIR / "training.log"),
    ],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@dataclass
class DateData:
    date: str
    features: np.ndarray  # (n_windows, 44)
    n_windows: int


def load_all_dates() -> List[DateData]:
    """Load all exec feature files, sorted by date."""
    files = sorted(glob.glob(str(FEATURES_DIR / "*_exec_features.npz")))
    dates = []
    for f in files:
        try:
            d = np.load(f, allow_pickle=True)
            date_str = str(d["date"])
            feats = d["features"].astype(np.float32)
            n = feats.shape[0]
            if n < 10:
                continue
            dates.append(DateData(date=date_str, features=feats, n_windows=n))
        except Exception as e:
            log.warning(f"Failed to load {f}: {e}")
    log.info(f"Loaded {len(dates)} dates with exec features")
    return dates


def get_feature_names() -> List[str]:
    """Get feature names from first available file."""
    files = sorted(glob.glob(str(FEATURES_DIR / "*_exec_features.npz")))
    if not files:
        return [f"feat_{i}" for i in FEATURE_INDICES]
    d = np.load(files[0], allow_pickle=True)
    names = list(d["feature_names"])
    return [names[i] for i in FEATURE_INDICES]


# ---------------------------------------------------------------------------
# Walk-forward engine
# ---------------------------------------------------------------------------
@dataclass
class FoldResult:
    fold_idx: int
    test_date: str
    horizon: str
    n_train: int
    n_test: int
    auc: float
    brier: float
    logloss: float
    avg_precision: float
    # Calibration buckets
    calibration_bins: List[Tuple[float, float]]  # (predicted_mean, actual_rate)
    # Regime metrics
    regime_auc: Dict[str, float]
    # Feature importances (gain-based)
    feature_importances: np.ndarray


def compute_calibration(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10):
    """Compute calibration: predicted prob vs actual fill rate in bins."""
    bins = np.linspace(0, 1, n_bins + 1)
    result = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_pred >= lo) & (y_pred < hi)
        if mask.sum() < 5:
            continue
        pred_mean = y_pred[mask].mean()
        actual_rate = y_true[mask].mean()
        result.append((float(pred_mean), float(actual_rate)))
    return result


def classify_regimes(features: np.ndarray) -> np.ndarray:
    """Classify each row into a regime string based on spread and vol."""
    spread_col = FEATURE_INDICES.index(20)  # spread_ticks in our feature subset
    vol_col = FEATURE_INDICES.index(43)    # price_volatility_window

    spread = features[:, spread_col]
    vol = features[:, vol_col]
    vol_thresh = np.quantile(vol[~np.isnan(vol)], REGIME_VOL_THRESH_HIGH) if len(vol) > 0 else 1.0

    regimes = np.empty(len(features), dtype=object)
    for i in range(len(features)):
        s = "tight" if spread[i] <= 1.0 else ("mid" if spread[i] <= REGIME_SPREAD_THRESH else "wide")
        v = "hivol" if vol[i] >= vol_thresh else "lovol"
        regimes[i] = f"{s}_{v}"
    return regimes


def train_fold(
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    fold_idx: int,
    test_date: str,
    horizon: str,
) -> Optional[Tuple[FoldResult, xgb.Booster]]:
    """Train one XGBoost fold and evaluate."""
    # Skip if degenerate
    if len(np.unique(train_y)) < 2:
        return None
    if len(np.unique(test_y)) < 2:
        return None

    dtrain = xgb.DMatrix(train_X, label=train_y)
    dtest = xgb.DMatrix(test_X, label=test_y)

    # Scale pos weight for imbalanced fill rates
    pos_rate = train_y.mean()
    if pos_rate > 0 and pos_rate < 1:
        scale_pos = (1 - pos_rate) / pos_rate
    else:
        scale_pos = 1.0

    params = {**XGB_PARAMS, "scale_pos_weight": min(scale_pos, 10.0)}

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=NUM_BOOST_ROUNDS,
        evals=[(dtest, "test")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
    )

    y_pred = model.predict(dtest)

    # Metrics
    try:
        auc = roc_auc_score(test_y, y_pred)
    except ValueError:
        auc = 0.5
    brier = brier_score_loss(test_y, y_pred)
    ll = log_loss(test_y, y_pred, labels=[0, 1])
    ap = average_precision_score(test_y, y_pred)

    # Calibration
    cal = compute_calibration(test_y, y_pred)

    # Regime AUC
    regimes = classify_regimes(test_X)
    regime_auc = {}
    for regime in np.unique(regimes):
        mask = regimes == regime
        if mask.sum() < 20 or len(np.unique(test_y[mask])) < 2:
            continue
        try:
            regime_auc[regime] = float(roc_auc_score(test_y[mask], y_pred[mask]))
        except ValueError:
            pass

    # Feature importances
    importance = model.get_score(importance_type="gain")
    fi = np.zeros(len(FEATURE_INDICES))
    for k, v in importance.items():
        idx = int(k.replace("f", ""))
        if idx < len(fi):
            fi[idx] = v

    result = FoldResult(
        fold_idx=fold_idx,
        test_date=test_date,
        horizon=horizon,
        n_train=len(train_y),
        n_test=len(test_y),
        auc=auc,
        brier=brier,
        logloss=ll,
        avg_precision=ap,
        calibration_bins=cal,
        regime_auc=regime_auc,
        feature_importances=fi,
    )
    return result, model


def run_walk_forward_horizon(
    all_dates: List[DateData], horizon: str
) -> Tuple[List[FoldResult], Optional[xgb.Booster]]:
    """Run walk-forward for one fill horizon."""
    target_col = HORIZON_COL_MAP[horizon]
    results = []
    best_model = None
    best_auc = 0.0

    n_dates = len(all_dates)
    start_fold = TRAIN_WINDOW  # first test date index

    log.info(f"[{horizon}] Starting walk-forward: {n_dates - start_fold} folds")

    for fold_idx, test_idx in enumerate(range(start_fold, n_dates)):
        train_start = test_idx - TRAIN_WINDOW
        train_dates = all_dates[train_start:test_idx]
        test_date_data = all_dates[test_idx]

        # Assemble training data
        train_X_parts = []
        train_y_parts = []
        for dd in train_dates:
            X = dd.features[:, FEATURE_INDICES]
            y_raw = dd.features[:, target_col]
            y = (y_raw > FILL_THRESHOLD).astype(np.float32)
            train_X_parts.append(X)
            train_y_parts.append(y)

        train_X = np.vstack(train_X_parts)
        train_y = np.concatenate(train_y_parts)

        # Test data
        test_X = test_date_data.features[:, FEATURE_INDICES]
        test_y_raw = test_date_data.features[:, target_col]
        test_y = (test_y_raw > FILL_THRESHOLD).astype(np.float32)

        # Handle NaN
        train_nan_mask = np.isnan(train_X).any(axis=1)
        test_nan_mask = np.isnan(test_X).any(axis=1)
        train_X = train_X[~train_nan_mask]
        train_y = train_y[~train_nan_mask]
        test_X = test_X[~test_nan_mask]
        test_y = test_y[~test_nan_mask]

        if len(train_y) < 100 or len(test_y) < 10:
            continue

        result = train_fold(train_X, train_y, test_X, test_y, fold_idx, test_date_data.date, horizon)
        if result is None:
            continue

        fold_result, model = result
        results.append(fold_result)

        if fold_result.auc > best_auc:
            best_auc = fold_result.auc
            best_model = model

        if fold_idx % 20 == 0:
            avg_auc = np.mean([r.auc for r in results[-20:]])
            log.info(
                f"  [{horizon}] Fold {fold_idx}/{n_dates - start_fold} "
                f"date={test_date_data.date} AUC={fold_result.auc:.4f} "
                f"rolling20_AUC={avg_auc:.4f} brier={fold_result.brier:.4f}"
            )

    return results, best_model


# ---------------------------------------------------------------------------
# Parallel walk-forward across horizons
# ---------------------------------------------------------------------------
def process_horizon(args):
    """Worker function for parallel horizon processing."""
    dates_path, horizon = args
    # Reload dates in subprocess
    all_dates = load_all_dates()
    results, best_model = run_walk_forward_horizon(all_dates, horizon)

    # Save model
    if best_model is not None:
        model_path = OUTPUT_DIR / f"fill_prob_xgb_{horizon}.json"
        best_model.save_model(str(model_path))

    # Serialize results
    serialized = []
    for r in results:
        serialized.append({
            "fold_idx": r.fold_idx,
            "test_date": r.test_date,
            "horizon": r.horizon,
            "n_train": r.n_train,
            "n_test": r.n_test,
            "auc": r.auc,
            "brier": r.brier,
            "logloss": r.logloss,
            "avg_precision": r.avg_precision,
            "calibration_bins": r.calibration_bins,
            "regime_auc": r.regime_auc,
            "feature_importances": r.feature_importances.tolist(),
        })
    return horizon, serialized


def main():
    parser = argparse.ArgumentParser(description="Fill Probability XGBoost Model")
    parser.add_argument("--workers", type=int, default=16, help="Parallel workers")
    parser.add_argument("--horizons", nargs="+", default=HORIZONS, help="Fill horizons")
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("Fill Probability XGBoost — Walk-Forward Training")
    log.info(f"Workers: {args.workers}, Horizons: {args.horizons}")
    log.info(f"Train window: {TRAIN_WINDOW} days, Output: {OUTPUT_DIR}")
    log.info("=" * 70)

    t0 = time.time()

    # Load dates once to get metadata
    all_dates = load_all_dates()
    feature_names = get_feature_names()
    log.info(f"Date range: {all_dates[0].date} to {all_dates[-1].date}")
    log.info(f"Total windows across all dates: {sum(d.n_windows for d in all_dates):,}")
    log.info(f"Features ({len(feature_names)}): {feature_names[:10]}...")

    # Compute fill rate stats
    for h in args.horizons:
        col = HORIZON_COL_MAP[h]
        all_vals = np.concatenate([d.features[:, col] for d in all_dates])
        fill_rate = (all_vals > FILL_THRESHOLD).mean()
        log.info(f"  {h} fill rate (>{FILL_THRESHOLD}): {fill_rate:.3f}")

    # Run walk-forward for each horizon
    # Use ProcessPoolExecutor for parallel date loading/training
    all_results = {}

    if len(args.horizons) > 1 and args.workers >= 3:
        # Parallel across horizons
        log.info("Running horizons in parallel...")
        with ProcessPoolExecutor(max_workers=min(len(args.horizons), args.workers)) as pool:
            futures = {
                pool.submit(process_horizon, (str(FEATURES_DIR), h)): h
                for h in args.horizons
            }
            for future in as_completed(futures):
                horizon, serialized = future.result()
                all_results[horizon] = serialized
                log.info(f"Completed horizon {horizon}: {len(serialized)} folds")
    else:
        for h in args.horizons:
            results, best_model = run_walk_forward_horizon(all_dates, h)
            if best_model is not None:
                model_path = OUTPUT_DIR / f"fill_prob_xgb_{h}.json"
                best_model.save_model(str(model_path))
            all_results[h] = [
                {
                    "fold_idx": r.fold_idx,
                    "test_date": r.test_date,
                    "horizon": r.horizon,
                    "n_train": r.n_train,
                    "n_test": r.n_test,
                    "auc": r.auc,
                    "brier": r.brier,
                    "logloss": r.logloss,
                    "avg_precision": r.avg_precision,
                    "calibration_bins": r.calibration_bins,
                    "regime_auc": r.regime_auc,
                    "feature_importances": r.feature_importances.tolist(),
                }
                for r in results
            ]

    elapsed = time.time() - t0

    # ---------------------------------------------------------------------------
    # Summary & save
    # ---------------------------------------------------------------------------
    summary = {
        "train_window": TRAIN_WINDOW,
        "fill_threshold": FILL_THRESHOLD,
        "xgb_params": XGB_PARAMS,
        "num_boost_rounds": NUM_BOOST_ROUNDS,
        "n_features": len(FEATURE_INDICES),
        "feature_names": feature_names,
        "elapsed_seconds": elapsed,
        "horizons": {},
    }

    log.info("")
    log.info("=" * 70)
    log.info("RESULTS SUMMARY")
    log.info("=" * 70)

    for h in args.horizons:
        if h not in all_results or not all_results[h]:
            log.warning(f"No results for horizon {h}")
            continue

        folds = all_results[h]
        aucs = [f["auc"] for f in folds]
        briers = [f["brier"] for f in folds]
        lls = [f["logloss"] for f in folds]
        aps = [f["avg_precision"] for f in folds]

        # Aggregate feature importances
        fi_matrix = np.array([f["feature_importances"] for f in folds])
        fi_mean = fi_matrix.mean(axis=0)
        fi_rank = np.argsort(-fi_mean)
        top_features = [(feature_names[i], float(fi_mean[i])) for i in fi_rank[:15]]

        # Aggregate regime AUCs
        regime_aucs_agg = {}
        for f in folds:
            for regime, auc_val in f["regime_auc"].items():
                regime_aucs_agg.setdefault(regime, []).append(auc_val)
        regime_summary = {k: {"mean": np.mean(v), "std": np.std(v), "n": len(v)}
                         for k, v in regime_aucs_agg.items()}

        # Aggregate calibration
        cal_all = {}
        for f in folds:
            for pred_mean, actual_rate in f["calibration_bins"]:
                bucket = round(pred_mean, 1)
                cal_all.setdefault(bucket, {"pred": [], "actual": []})
                cal_all[bucket]["pred"].append(pred_mean)
                cal_all[bucket]["actual"].append(actual_rate)
        calibration_summary = {
            str(k): {
                "mean_predicted": np.mean(v["pred"]),
                "mean_actual": np.mean(v["actual"]),
                "n_folds": len(v["pred"]),
            }
            for k, v in sorted(cal_all.items())
        }

        h_summary = {
            "n_folds": len(folds),
            "auc_mean": float(np.mean(aucs)),
            "auc_std": float(np.std(aucs)),
            "auc_median": float(np.median(aucs)),
            "brier_mean": float(np.mean(briers)),
            "logloss_mean": float(np.mean(lls)),
            "avg_precision_mean": float(np.mean(aps)),
            "top_features": top_features,
            "regime_auc": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in regime_summary.items()},
            "calibration": calibration_summary,
        }
        summary["horizons"][h] = h_summary

        log.info(f"\n--- Horizon: {h} ---")
        log.info(f"  Folds: {len(folds)}")
        log.info(f"  AUC: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f} (median {np.median(aucs):.4f})")
        log.info(f"  Brier: {np.mean(briers):.4f}")
        log.info(f"  Log Loss: {np.mean(lls):.4f}")
        log.info(f"  Avg Precision: {np.mean(aps):.4f}")
        log.info(f"  Top 10 Features:")
        for name, gain in top_features[:10]:
            log.info(f"    {name}: {gain:.1f}")
        log.info(f"  Regime AUC:")
        for regime, stats in sorted(regime_summary.items()):
            log.info(f"    {regime}: {stats['mean']:.4f} +/- {stats['std']:.4f} (n={stats['n']})")
        log.info(f"  Calibration (predicted -> actual):")
        for bucket, vals in sorted(cal_all.items()):
            log.info(f"    {np.mean(vals['pred']):.2f} -> {np.mean(vals['actual']):.2f}")

    # Save summary JSON
    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"\nSummary saved to {summary_path}")

    # Save per-fold results
    for h in args.horizons:
        if h in all_results:
            folds_path = OUTPUT_DIR / f"folds_{h}.json"
            with open(folds_path, "w") as f:
                json.dump(all_results[h], f, indent=2)

    log.info(f"\nTotal elapsed: {elapsed:.1f}s ({elapsed/60:.1f}m)")
    log.info("Done.")


if __name__ == "__main__":
    main()
