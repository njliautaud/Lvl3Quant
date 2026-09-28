#!/usr/bin/env python3
r"""
Pressure XGBoost v2 — Walk-Forward Regression on Smooth Pressure Targets (GPU).

Trains XGBoost GPU regressors to predict smooth pressure targets (EOFI, PDI,
NTPS, TIA) at various horizons using smart_v3 event features + rolling stats.

Walk-forward: 60-day sliding window, 1-day OOT, drop oldest day (HC #0).
Subsample training data (default 10%) for memory; predict ALL OOT events.

Usage:
    python train_pressure_xgb_v2.py --target eofi --horizon 10s
    python train_pressure_xgb_v2.py --target pdi --horizon 30s --subsample-ratio 0.05

Inputs:
    data/processed/mbo_events_smart_v3/*.npz      (events, timestamps)
    data/processed/smooth_pressure_targets/*_pressure.npz (pressure labels)

Outputs:
    output/pressure_xgb_v2_{target}_{horizon}/fold_NN_oot_YYYYMMDD.npz
    output/pressure_xgb_v2_{target}_{horizon}/concat_oot.npz
    output/pressure_xgb_v2_{target}_{horizon}/summary.json
"""
from __future__ import annotations

import argparse
import gc
import glob
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Parse args early so we can set up paths
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Pressure XGBoost v2 Walk-Forward")
    p.add_argument("--data-dir", type=str,
                   default="data/processed/mbo_events_smart_v3",
                   help="Event data directory (relative to LVL3_ROOT)")
    p.add_argument("--pressure-dir", type=str,
                   default="data/processed/smooth_pressure_targets",
                   help="Pressure target directory (relative to LVL3_ROOT)")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Output directory (default: auto from target/horizon)")
    p.add_argument("--target", type=str, default="eofi",
                   choices=["eofi", "pdi", "ntps", "tia"],
                   help="Target type (default: eofi)")
    p.add_argument("--horizon", type=str, default="10s",
                   choices=["1s", "5s", "10s", "30s"],
                   help="Target horizon (default: 10s)")
    p.add_argument("--train-days", type=int, default=60,
                   help="Sliding window train days (default: 60)")
    p.add_argument("--subsample-ratio", type=float, default=0.10,
                   help="Fraction of training events to use (default: 0.10)")
    p.add_argument("--mlflow-uri", type=str,
                   default="http://localhost:5000",
                   help="MLflow tracking URI")
    p.add_argument("--n-estimators", type=int, default=800,
                   help="XGBoost n_estimators (default: 800)")
    p.add_argument("--max-depth", type=int, default=6,
                   help="XGBoost max_depth (default: 6)")
    p.add_argument("--learning-rate", type=float, default=0.05,
                   help="XGBoost learning_rate (default: 0.05)")
    p.add_argument("--early-stopping", type=int, default=50,
                   help="Early stopping rounds (default: 50)")
    p.add_argument("--rolling-windows", type=str, default="50,200",
                   help="Rolling stat windows comma-separated (default: 50,200)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Constants / setup
# ---------------------------------------------------------------------------
# Auto-detect root: works on Jupiter (/home/jupiter) and Neptune (/home/nick)
_script_dir = Path(__file__).resolve().parent
LVL3_ROOT = _script_dir.parent  # scripts/ -> Lvl3Quant/

args = parse_args()

TARGET_KEY = f"{args.target}_label_{args.horizon}"
ROLLING_WINDOWS = [int(x) for x in args.rolling_windows.split(",") if x.strip()] if args.rolling_windows.strip() else []

# Output dir
if args.output_dir:
    OUT_DIR = Path(args.output_dir)
else:
    OUT_DIR = LVL3_ROOT / "output" / f"pressure_xgb_v2_{args.target}_{args.horizon}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = LVL3_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOG_DIR / f"pressure_xgb_v2_{args.target}_{args.horizon}_{RUN_TS}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("pressure_xgb_v2")

DATA_DIR = LVL3_ROOT / args.data_dir
PRESSURE_DIR = LVL3_ROOT / args.pressure_dir


# ---------------------------------------------------------------------------
# Feature engineering: rolling statistics
# ---------------------------------------------------------------------------
def build_rolling_features(events: np.ndarray, windows: list[int]) -> np.ndarray:
    """
    Given events (N, 25), compute rolling mean/std/min/max for each feature
    over specified windows. Returns (N, 25 + 25*4*len(windows)) features.

    Uses cumsum tricks for speed on large arrays.
    """
    N, F = events.shape
    # Pre-allocate output: base features + rolling stats
    n_rolling = F * 4 * len(windows)  # mean, std, min, max per feature per window
    out = np.empty((N, F + n_rolling), dtype=np.float32)
    out[:, :F] = events

    col = F
    for w in windows:
        if N < w:
            # Not enough data, fill with NaN
            out[:, col:col + F * 4] = np.nan
            col += F * 4
            continue

        # Rolling mean via cumsum
        cumsum = np.cumsum(events, axis=0)
        cumsum_padded = np.vstack([np.zeros((1, F), dtype=np.float64), cumsum])

        # Rolling mean
        rmean = np.empty((N, F), dtype=np.float32)
        rmean[:w - 1] = np.nan
        rmean[w - 1:] = ((cumsum_padded[w:] - cumsum_padded[:N - w + 1]) / w).astype(np.float32)
        out[:, col:col + F] = rmean
        col += F

        # Rolling std via cumsum of squares
        cumsum2 = np.cumsum(events.astype(np.float64) ** 2, axis=0)
        cumsum2_padded = np.vstack([np.zeros((1, F), dtype=np.float64), cumsum2])
        rvar = np.empty((N, F), dtype=np.float32)
        rvar[:w - 1] = np.nan
        mean_sq = (cumsum_padded[w:] - cumsum_padded[:N - w + 1]) / w
        sq_mean = (cumsum2_padded[w:] - cumsum2_padded[:N - w + 1]) / w
        variance = sq_mean - mean_sq ** 2
        # Clamp negative variance from floating point
        variance = np.maximum(variance, 0)
        rvar[w - 1:] = np.sqrt(variance).astype(np.float32)
        out[:, col:col + F] = rvar
        col += F

        del cumsum2, cumsum2_padded, mean_sq, sq_mean, variance

        # Rolling min/max — use strided view for moderate windows
        # For large windows, sliding approach with scipy or numpy
        if w <= 250:
            # Strided approach: create (N-w+1, w, F) view
            from numpy.lib.stride_tricks import sliding_window_view
            sw = sliding_window_view(events, window_shape=w, axis=0)  # (N-w+1, F, w)
            rmin = np.empty((N, F), dtype=np.float32)
            rmax = np.empty((N, F), dtype=np.float32)
            rmin[:w - 1] = np.nan
            rmax[:w - 1] = np.nan
            rmin[w - 1:] = sw.min(axis=-1).astype(np.float32)
            rmax[w - 1:] = sw.max(axis=-1).astype(np.float32)
            del sw
        else:
            # Fallback for very large windows — compute in chunks
            rmin = np.full((N, F), np.nan, dtype=np.float32)
            rmax = np.full((N, F), np.nan, dtype=np.float32)
            for i in range(w - 1, N):
                chunk = events[i - w + 1:i + 1]
                rmin[i] = chunk.min(axis=0)
                rmax[i] = chunk.max(axis=0)

        out[:, col:col + F] = rmin
        col += F
        out[:, col:col + F] = rmax
        col += F

        del cumsum, cumsum_padded, rmean, rmin, rmax

    gc.collect()
    return out


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def discover_matched_days() -> list[str]:
    """Find dates that exist in BOTH event and pressure directories."""
    event_files = sorted(glob.glob(str(DATA_DIR / "*_mbo_events.npz")))
    pressure_files = sorted(glob.glob(str(PRESSURE_DIR / "*_pressure.npz")))

    event_dates = set()
    for f in event_files:
        base = os.path.basename(f)
        date_str = base.split("_")[0]
        event_dates.add(date_str)

    matched = []
    for f in pressure_files:
        base = os.path.basename(f)
        date_str = base.split("_")[0]
        if date_str in event_dates:
            matched.append(date_str)

    matched.sort()
    return matched


def load_day(date_str: str, need_target: bool = True) -> dict | None:
    """Load events + pressure target for a single day. Returns dict or None."""
    event_path = DATA_DIR / f"{date_str}_mbo_events.npz"
    pressure_path = PRESSURE_DIR / f"{date_str}_pressure.npz"

    if not event_path.exists():
        return None
    if need_target and not pressure_path.exists():
        return None

    ef = np.load(str(event_path))
    events = ef["events"]  # (N, 25) float32
    timestamps = ef["timestamps"]  # (N,) int64

    if need_target:
        pf = np.load(str(pressure_path))
        if TARGET_KEY not in pf:
            log.warning(f"Target {TARGET_KEY} not in {pressure_path}")
            return None
        target = pf[TARGET_KEY]  # (N,) float32
        pts = pf["timestamps"]

        # Verify alignment
        if len(events) != len(target):
            log.warning(
                f"Length mismatch {date_str}: events={len(events)}, "
                f"pressure={len(target)}. Skipping."
            )
            return None
        if not np.array_equal(timestamps, pts):
            log.warning(f"Timestamp mismatch {date_str}. Skipping.")
            return None
    else:
        target = None

    return {
        "date": date_str,
        "events": events,
        "timestamps": timestamps,
        "target": target,
    }


def load_and_featurize_day(date_str: str, need_target: bool = True) -> dict | None:
    """Load a day and compute rolling features."""
    day = load_day(date_str, need_target=need_target)
    if day is None:
        return None

    if ROLLING_WINDOWS:
        features = build_rolling_features(day["events"], ROLLING_WINDOWS)
    else:
        features = day["events"].astype(np.float32)
    day["features"] = features
    del day["events"]
    gc.collect()
    return day


# ---------------------------------------------------------------------------
# Walk-forward engine
# ---------------------------------------------------------------------------
def run_walk_forward(dates: list[str]):
    """Main walk-forward loop."""
    import xgboost as xgb

    train_days = args.train_days
    n_dates = len(dates)

    if n_dates <= train_days:
        log.error(
            f"Only {n_dates} matched days, need > {train_days} for walk-forward. Aborting."
        )
        sys.exit(1)

    n_folds = n_dates - train_days
    log.info(
        f"Walk-forward: {n_dates} days, {train_days}-day window, {n_folds} OOT folds"
    )
    log.info(f"Target: {TARGET_KEY}, subsample ratio: {args.subsample_ratio}")

    # MLflow setup
    mlflow_ok = False
    try:
        import mlflow

        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment("pressure_xgb_v2")
        mlflow_ok = True
        log.info(f"MLflow tracking at {args.mlflow_uri}")
    except Exception as e:
        log.warning(f"MLflow unavailable ({e}), proceeding without tracking")

    run_ctx = None
    if mlflow_ok:
        run_ctx = mlflow.start_run(
            run_name=f"pressure_{args.target}_{args.horizon}_{RUN_TS}"
        )
        mlflow.log_params({
            "target": args.target,
            "horizon": args.horizon,
            "target_key": TARGET_KEY,
            "train_days": train_days,
            "n_folds": n_folds,
            "subsample_ratio": args.subsample_ratio,
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "early_stopping": args.early_stopping,
            "rolling_windows": str(ROLLING_WINDOWS),
            "seed": args.seed,
        })

    # XGBoost params
    xgb_params = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "device": "cuda",
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 100,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "seed": args.seed,
        "verbosity": 0,
    }

    # Results collectors
    fold_results = []
    all_oot_preds = []
    all_oot_targets = []
    all_oot_timestamps = []
    total_t0 = time.time()

    # Pre-load feature names
    n_base = 25
    n_rolling_per_window = n_base * 4  # mean, std, min, max
    n_total_features = n_base + n_rolling_per_window * len(ROLLING_WINDOWS)
    feature_names = [f"f{i}" for i in range(n_base)]
    for w in ROLLING_WINDOWS:
        for stat in ["mean", "std", "min", "max"]:
            for i in range(n_base):
                feature_names.append(f"f{i}_{stat}_{w}")

    # Cache: keep loaded days in memory (LRU-ish: drop oldest when not in window)
    day_cache: dict[str, dict] = {}

    for fold_idx in range(n_folds):
        fold_t0 = time.time()
        train_dates = dates[fold_idx: fold_idx + train_days]
        oot_date = dates[fold_idx + train_days]

        log.info(
            f"Fold {fold_idx:03d}/{n_folds}: "
            f"train [{train_dates[0]}..{train_dates[-1]}] -> OOT {oot_date}"
        )

        # --- Load OOT day (ALL events, with features) ---
        if oot_date in day_cache:
            oot_day = day_cache[oot_date]
        else:
            oot_day = load_and_featurize_day(oot_date)
        if oot_day is None:
            log.warning(f"Fold {fold_idx}: OOT day {oot_date} failed to load. Skip.")
            continue
        day_cache[oot_date] = oot_day

        X_oot = oot_day["features"]
        y_oot = oot_day["target"]
        ts_oot = oot_day["timestamps"]

        # Filter NaN targets in OOT
        oot_valid = np.isfinite(y_oot)
        # Also filter NaN features (from rolling warmup)
        oot_feat_valid = np.all(np.isfinite(X_oot), axis=1)
        oot_mask = oot_valid & oot_feat_valid

        if oot_mask.sum() < 100:
            log.warning(f"Fold {fold_idx}: OOT {oot_date} too few valid rows ({oot_mask.sum()}). Skip.")
            continue

        # --- Assemble training data (subsampled) ---
        rng = np.random.RandomState(args.seed + fold_idx)
        train_X_chunks = []
        train_y_chunks = []
        n_train_total = 0
        n_train_sampled = 0

        for td in train_dates:
            if td in day_cache:
                tday = day_cache[td]
            else:
                tday = load_and_featurize_day(td)
                if tday is not None:
                    day_cache[td] = tday
            if tday is None:
                continue

            X_day = tday["features"]
            y_day = tday["target"]
            N = len(y_day)
            n_train_total += N

            # Valid mask: finite target + finite features
            valid = np.isfinite(y_day) & np.all(np.isfinite(X_day), axis=1)
            valid_idx = np.where(valid)[0]

            if len(valid_idx) == 0:
                continue

            # Subsample
            n_sample = max(1, int(len(valid_idx) * args.subsample_ratio))
            chosen = rng.choice(valid_idx, size=n_sample, replace=False)
            chosen.sort()  # preserve temporal order

            train_X_chunks.append(X_day[chosen])
            train_y_chunks.append(y_day[chosen])
            n_train_sampled += n_sample

        # Evict days no longer in the window from cache
        active_dates = set(train_dates) | {oot_date}
        # Also keep next few OOT days if already cached
        for cached_date in list(day_cache.keys()):
            if cached_date not in active_dates:
                del day_cache[cached_date]
                gc.collect()

        if len(train_X_chunks) == 0:
            log.warning(f"Fold {fold_idx}: No valid training data. Skip.")
            continue

        X_train = np.vstack(train_X_chunks)
        y_train = np.concatenate(train_y_chunks)
        del train_X_chunks, train_y_chunks
        gc.collect()

        log.info(
            f"  Train: {n_train_sampled:,} rows (subsampled from {n_train_total:,}), "
            f"OOT: {oot_mask.sum():,} rows"
        )

        # --- Train XGBoost ---
        # Use 10% of training data as eval set for early stopping
        n_eval = max(100, int(len(X_train) * 0.1))
        # Take last portion as eval (most recent data)
        X_eval_es = X_train[-n_eval:]
        y_eval_es = y_train[-n_eval:]
        X_train_fit = X_train[:-n_eval]
        y_train_fit = y_train[:-n_eval]

        dtrain = xgb.DMatrix(X_train_fit, label=y_train_fit, feature_names=feature_names)
        deval = xgb.DMatrix(X_eval_es, label=y_eval_es, feature_names=feature_names)
        doot = xgb.DMatrix(X_oot[oot_mask], feature_names=feature_names)

        evals_result: dict = {}
        model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=args.n_estimators,
            evals=[(dtrain, "train"), (deval, "eval")],
            early_stopping_rounds=args.early_stopping,
            evals_result=evals_result,
            verbose_eval=False,
        )

        best_iteration = model.best_iteration
        train_rmse = evals_result["train"]["rmse"][-1]
        eval_rmse = evals_result["eval"]["rmse"][-1]

        # --- Predict OOT ---
        preds_valid = model.predict(doot)

        # Map back to full array
        preds_full = np.full(len(y_oot), np.nan, dtype=np.float32)
        preds_full[oot_mask] = preds_valid

        # --- Compute IC on valid OOT ---
        ic, ic_pval = stats.spearmanr(preds_valid, y_oot[oot_mask])
        if np.isnan(ic):
            ic = 0.0

        fold_time = time.time() - fold_t0

        log.info(
            f"  IC={ic:.4f} (p={ic_pval:.2e}), "
            f"best_iter={best_iteration}, "
            f"train_rmse={train_rmse:.6f}, eval_rmse={eval_rmse:.6f}, "
            f"time={fold_time:.1f}s"
        )

        # Save fold predictions
        fold_path = OUT_DIR / f"fold_{fold_idx:03d}_oot_{oot_date}.npz"
        np.savez_compressed(
            str(fold_path),
            predictions=preds_full,
            targets=y_oot,
            timestamps=ts_oot,
            valid_mask=oot_mask,
        )

        fold_results.append({
            "fold": fold_idx,
            "oot_date": oot_date,
            "train_start": train_dates[0],
            "train_end": train_dates[-1],
            "n_train": int(n_train_sampled),
            "n_oot": int(oot_mask.sum()),
            "ic": float(ic),
            "ic_pval": float(ic_pval),
            "best_iteration": int(best_iteration),
            "train_rmse": float(train_rmse),
            "eval_rmse": float(eval_rmse),
            "fold_time_s": round(fold_time, 1),
        })

        # Collect for concat IC
        all_oot_preds.append(preds_valid)
        all_oot_targets.append(y_oot[oot_mask])
        all_oot_timestamps.append(ts_oot[oot_mask])

        # Log to MLflow per fold
        if mlflow_ok:
            mlflow.log_metrics({
                f"fold_{fold_idx:03d}_ic": float(ic),
                f"fold_{fold_idx:03d}_eval_rmse": float(eval_rmse),
            }, step=fold_idx)

        # Cleanup
        del X_train, y_train, X_train_fit, y_train_fit, X_eval_es, y_eval_es
        del dtrain, deval, doot, model, preds_valid, preds_full
        gc.collect()

    # --- Aggregate results ---
    total_time = time.time() - total_t0
    log.info(f"\n{'='*60}")
    log.info(f"Walk-forward complete: {len(fold_results)}/{n_folds} folds in {total_time:.0f}s")

    if len(all_oot_preds) == 0:
        log.error("No valid folds produced. Aborting.")
        if mlflow_ok and run_ctx:
            mlflow.end_run(status="FAILED")
        sys.exit(1)

    # Concat IC (THE primary metric per HC)
    concat_preds = np.concatenate(all_oot_preds)
    concat_targets = np.concatenate(all_oot_targets)
    concat_timestamps = np.concatenate(all_oot_timestamps)

    concat_ic, concat_ic_pval = stats.spearmanr(concat_preds, concat_targets)
    if np.isnan(concat_ic):
        concat_ic = 0.0

    # Per-fold IC stats
    ics = [r["ic"] for r in fold_results]
    mean_ic = float(np.mean(ics))
    median_ic = float(np.median(ics))
    std_ic = float(np.std(ics))
    pct_positive = float(np.mean([ic > 0 for ic in ics]) * 100)

    log.info(f"CONCAT IC: {concat_ic:.4f} (p={concat_ic_pval:.2e})")
    log.info(f"Per-fold IC: mean={mean_ic:.4f}, median={median_ic:.4f}, std={std_ic:.4f}")
    log.info(f"Positive IC folds: {pct_positive:.1f}%")
    log.info(f"Total OOT events: {len(concat_preds):,}")

    # Save concat predictions
    concat_path = OUT_DIR / "concat_oot.npz"
    np.savez_compressed(
        str(concat_path),
        predictions=concat_preds,
        targets=concat_targets,
        timestamps=concat_timestamps,
    )
    log.info(f"Saved concat OOT predictions: {concat_path}")

    # Summary
    summary = {
        "target": args.target,
        "horizon": args.horizon,
        "target_key": TARGET_KEY,
        "train_days": train_days,
        "n_folds_total": n_folds,
        "n_folds_valid": len(fold_results),
        "subsample_ratio": args.subsample_ratio,
        "xgb_params": xgb_params,
        "concat_ic": float(concat_ic),
        "concat_ic_pval": float(concat_ic_pval),
        "mean_fold_ic": mean_ic,
        "median_fold_ic": median_ic,
        "std_fold_ic": std_ic,
        "pct_positive_ic_folds": pct_positive,
        "total_oot_events": int(len(concat_preds)),
        "total_time_s": round(total_time, 1),
        "avg_fold_time_s": round(total_time / max(len(fold_results), 1), 1),
        "folds": fold_results,
        "run_timestamp": RUN_TS,
    }

    summary_path = OUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Saved summary: {summary_path}")

    # MLflow final metrics
    if mlflow_ok:
        mlflow.log_metrics({
            "concat_ic": float(concat_ic),
            "mean_fold_ic": mean_ic,
            "median_fold_ic": median_ic,
            "std_fold_ic": std_ic,
            "pct_positive_ic_folds": pct_positive,
            "n_folds_valid": len(fold_results),
            "total_oot_events": int(len(concat_preds)),
            "total_time_s": total_time,
        })
        mlflow.log_artifact(str(summary_path))
        mlflow.log_artifact(str(LOG_FILE))
        mlflow.end_run()
        log.info("MLflow run logged and closed.")

    # Print final table
    log.info(f"\n{'='*60}")
    log.info("FOLD RESULTS:")
    log.info(f"{'Fold':>5} {'OOT Date':>10} {'IC':>8} {'N_OOT':>10} {'Time':>6}")
    log.info("-" * 45)
    for r in fold_results:
        log.info(
            f"{r['fold']:5d} {r['oot_date']:>10} {r['ic']:8.4f} "
            f"{r['n_oot']:10,} {r['fold_time_s']:6.1f}s"
        )
    log.info("-" * 45)
    log.info(
        f"CONCAT IC = {concat_ic:.4f} | "
        f"Mean IC = {mean_ic:.4f} | "
        f"Positive folds = {pct_positive:.1f}%"
    )

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    log.info(f"Pressure XGBoost v2 — {args.target}_{args.horizon}")
    log.info(f"Data dir: {DATA_DIR}")
    log.info(f"Pressure dir: {PRESSURE_DIR}")
    log.info(f"Output dir: {OUT_DIR}")
    log.info(f"Target key: {TARGET_KEY}")
    log.info(f"Subsample ratio: {args.subsample_ratio}")
    log.info(f"Rolling windows: {ROLLING_WINDOWS}")

    # Discover matched days
    dates = discover_matched_days()
    log.info(f"Found {len(dates)} matched days (event + pressure)")

    if len(dates) == 0:
        log.error("No matched days found. Check data directories.")
        sys.exit(1)

    log.info(f"Date range: {dates[0]} to {dates[-1]}")

    summary = run_walk_forward(dates)

    log.info("Done.")
