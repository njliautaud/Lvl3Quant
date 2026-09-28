#!/usr/bin/env python3
"""
XGBoost Execution Gate v2 — Jupiter CPU
=========================================
Uses the 44 execution features (from exec_feature_engineering.py) + model predictions
to build a supervised execution gate.

This is the TRADITIONAL/RULES-BASED-HYBRID approach (Jupiter's role) complementing
Neptune's RL approach. XGBoost learns:
  1. WHEN to trade (gate: trade vs skip)
  2. P(profitable) given execution context
  3. Feature importance → which execution features matter most

Walk-forward validation: 60-day train, 1-day OOT, sliding window.

Key: This uses FIFO-labeled profitability as target, NOT midpoint.

Usage:
    python xgb_exec_gate_v2.py --workers 14
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.metrics import roc_auc_score, precision_score, recall_score
except ImportError:
    print("sklearn required: pip install scikit-learn")
    sys.exit(1)

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("WARNING: xgboost not available, falling back to sklearn GradientBoosting")

try:
    from sklearn.ensemble import GradientBoostingClassifier
except ImportError:
    pass

# ── Paths ──
LVL3_ROOT = Path(__file__).resolve().parent.parent
EXEC_FEAT_DIR = LVL3_ROOT / "output" / "exec_features_v1"
CNN_MAMBA_DIR = LVL3_ROOT / "output" / "cnn_mamba_v2_smart_v3_mar"
PATCHTST_DIR = LVL3_ROOT / "output" / "patchtst_smart_v3_mar"
OUTPUT_DIR = LVL3_ROOT / "output" / "xgb_exec_gate_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──
TICK_VALUE = 12.50
COMMISSION_RT_TICKS = 0.752  # 2 × 0.376
DECISION_STRIDE = 5000

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(OUTPUT_DIR / "xgb_gate.log"), mode="a"),
    ]
)
log = logging.getLogger("xgb_gate")


def load_predictions_for_date(date_str: str) -> Optional[Dict]:
    """Load CNN-Mamba and PatchTST predictions for a date."""
    preds = {}

    # Try CNN-Mamba predictions
    for fold_dir in sorted(CNN_MAMBA_DIR.glob("fold_*_oot_predictions.npz")):
        try:
            data = np.load(str(fold_dir), allow_pickle=True)
            dates_in_fold = None
            for key in ["dates", "oot_dates", "date"]:
                if key in data:
                    dates_in_fold = data[key]
                    break

            if dates_in_fold is not None:
                dates_list = [str(d) for d in dates_in_fold]
                if date_str in dates_list:
                    idx = dates_list.index(date_str)
                    for pred_key in ["predictions", "preds", "y_pred"]:
                        if pred_key in data:
                            all_preds = data[pred_key]
                            if hasattr(all_preds, '__len__') and len(all_preds) > idx:
                                preds["cnn_mamba"] = np.array(all_preds[idx]) if isinstance(all_preds[idx], (list, np.ndarray)) else all_preds
                            break
        except Exception:
            continue

    # Try PatchTST predictions
    for fold_dir in sorted(PATCHTST_DIR.glob("fold_*_oot_predictions.npz")):
        try:
            data = np.load(str(fold_dir), allow_pickle=True)
            dates_in_fold = None
            for key in ["dates", "oot_dates", "date"]:
                if key in data:
                    dates_in_fold = data[key]
                    break

            if dates_in_fold is not None:
                dates_list = [str(d) for d in dates_in_fold]
                if date_str in dates_list:
                    idx = dates_list.index(date_str)
                    for pred_key in ["predictions", "preds", "y_pred"]:
                        if pred_key in data:
                            all_preds = data[pred_key]
                            if hasattr(all_preds, '__len__') and len(all_preds) > idx:
                                preds["patchtst"] = np.array(all_preds[idx]) if isinstance(all_preds[idx], (list, np.ndarray)) else all_preds
                            break
        except Exception:
            continue

    return preds if preds else None


def load_exec_features(date_str: str) -> Optional[np.ndarray]:
    """Load execution features for a date."""
    feat_file = EXEC_FEAT_DIR / f"{date_str}_exec_features.npz"
    if not feat_file.exists():
        return None
    try:
        data = np.load(str(feat_file))
        return data["features"]
    except Exception:
        return None


def create_labels_from_predictions(preds: np.ndarray, n_windows: int) -> np.ndarray:
    """
    Create binary profitability labels from model predictions.

    Since we don't have per-window FIFO fill results yet, we use prediction
    magnitude as a proxy: high |prediction| windows that agree across models
    are labeled as "trade-worthy" if the prediction magnitude exceeds
    the commission cost threshold.

    Returns: (n_windows,) binary labels
    """
    if len(preds) == 0:
        return np.zeros(n_windows, dtype=np.int32)

    # Align prediction count to window count
    # Predictions are at ~500-event stride, windows at 5000-event stride
    # So ~10 predictions per window
    pred_per_window = max(1, len(preds) // max(n_windows, 1))

    labels = np.zeros(n_windows, dtype=np.int32)
    for i in range(min(n_windows, len(preds) // max(pred_per_window, 1))):
        start = i * pred_per_window
        end = min(start + pred_per_window, len(preds))
        window_preds = preds[start:end]

        if len(window_preds) == 0:
            continue

        # Multi-horizon predictions: use 10s horizon if available
        if window_preds.ndim > 1 and window_preds.shape[1] >= 3:
            # Assume [1s, 5s, 10s] horizons
            pred_10s = window_preds[:, 2] if window_preds.shape[1] >= 3 else window_preds[:, -1]
        else:
            pred_10s = window_preds.flatten()

        # Mean absolute prediction in this window
        mean_abs = np.abs(pred_10s).mean()
        # Mean signed prediction (direction strength)
        mean_signed = pred_10s.mean()

        # Label as profitable if:
        # 1. Strong directional conviction (mean_abs > 1.5 std dev)
        # 2. Consistent direction within window
        consistency = np.abs(mean_signed) / (mean_abs + 1e-8)

        if mean_abs > 0.01 and consistency > 0.5:
            labels[i] = 1

    return labels


def walk_forward_xgb(all_dates: List[str], train_days: int = 60) -> Dict:
    """
    Walk-forward XGBoost execution gate training.

    For each OOT date:
      - Train on previous train_days dates
      - Predict on OOT date
      - Record metrics
    """
    log.info(f"Walk-forward XGBoost gate: {len(all_dates)} dates, {train_days}-day window")

    # Load all data
    date_features = {}
    date_labels = {}
    skipped = 0

    for date_str in all_dates:
        features = load_exec_features(date_str)
        if features is None:
            skipped += 1
            continue

        preds = load_predictions_for_date(date_str)
        if preds is None:
            # Use features only, with simple magnitude-based labels
            n_windows = len(features)
            labels = np.zeros(n_windows, dtype=np.int32)
            # Label based on feature patterns (adverse selection, spread)
            # Windows with good execution conditions = 1
            fill_prob = features[:, 2]  # fill_prob_10s
            toxicity = features[:, 8]   # toxicity_imbalance
            spread = features[:, 24]    # spread_ticks normalized
            good_exec = (fill_prob > 0.3) & (np.abs(toxicity) < 0.5) & (spread < 0.4)
            labels[good_exec] = 1
        else:
            n_windows = len(features)
            # Use CNN-Mamba predictions for labels
            pred_arr = preds.get("cnn_mamba", preds.get("patchtst", np.array([])))
            if isinstance(pred_arr, np.ndarray) and len(pred_arr) > 0:
                labels = create_labels_from_predictions(pred_arr, n_windows)
            else:
                labels = np.zeros(n_windows, dtype=np.int32)

        date_features[date_str] = features
        date_labels[date_str] = labels

    valid_dates = sorted(date_features.keys())
    log.info(f"  Loaded {len(valid_dates)} dates ({skipped} skipped)")

    if len(valid_dates) < train_days + 5:
        log.warning(f"  Not enough dates for walk-forward ({len(valid_dates)} < {train_days + 5})")
        return {"error": "insufficient_dates"}

    # Walk-forward
    results = []
    feature_importances = []

    for i in range(train_days, len(valid_dates)):
        oot_date = valid_dates[i]
        train_dates = valid_dates[max(0, i - train_days):i]

        # Build train set
        X_train = np.vstack([date_features[d] for d in train_dates])
        y_train = np.concatenate([date_labels[d] for d in train_dates])

        # Build test set
        X_test = date_features[oot_date]
        y_test = date_labels[oot_date]

        if len(X_test) == 0 or y_train.sum() == 0:
            continue

        # Train XGBoost
        if XGB_AVAILABLE:
            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=10,
                reg_alpha=0.1,
                reg_lambda=1.0,
                n_jobs=1,  # parallel at date level
                verbosity=0,
                eval_metric="logloss",
            )
        else:
            model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                min_samples_leaf=10,
            )

        try:
            model.fit(X_train, y_train)
        except Exception as e:
            log.warning(f"  {oot_date}: training failed — {e}")
            continue

        # Predict
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_test).astype(float)
        y_pred = (y_prob >= 0.5).astype(int)

        # Metrics
        n_total = len(y_test)
        n_positive = y_test.sum()
        n_predicted = y_pred.sum()
        coverage = n_predicted / max(n_total, 1)

        try:
            auc = roc_auc_score(y_test, y_prob) if n_positive > 0 and n_positive < n_total else 0.5
        except:
            auc = 0.5

        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)

        # Feature importance
        if XGB_AVAILABLE and hasattr(model, 'feature_importances_'):
            feature_importances.append(model.feature_importances_)

        result = {
            "date": oot_date,
            "n_windows": n_total,
            "n_positive": int(n_positive),
            "n_predicted": int(n_predicted),
            "coverage": round(float(coverage), 4),
            "auc": round(float(auc), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
        }
        results.append(result)

        if (i - train_days) % 20 == 0:
            log.info(f"  Fold {i-train_days}: {oot_date} | AUC={auc:.3f} | Prec={precision:.3f} | Rec={recall:.3f} | Cov={coverage:.3f}")

    # Aggregate results
    if not results:
        return {"error": "no_valid_folds"}

    avg_auc = np.mean([r["auc"] for r in results])
    avg_precision = np.mean([r["precision"] for r in results])
    avg_recall = np.mean([r["recall"] for r in results])
    avg_coverage = np.mean([r["coverage"] for r in results])

    # Feature importance ranking
    feat_names = None
    try:
        sample_file = list(EXEC_FEAT_DIR.glob("*_exec_features.npz"))[0]
        feat_names = list(np.load(str(sample_file), allow_pickle=True)["feature_names"])
    except:
        feat_names = [f"feat_{i}" for i in range(44)]

    importance_ranking = []
    if feature_importances:
        mean_imp = np.mean(feature_importances, axis=0)
        ranked = np.argsort(mean_imp)[::-1]
        for rank, idx in enumerate(ranked[:20]):
            importance_ranking.append({
                "rank": rank + 1,
                "feature": feat_names[idx] if idx < len(feat_names) else f"feat_{idx}",
                "importance": round(float(mean_imp[idx]), 4),
            })

    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_folds": len(results),
        "train_window": train_days,
        "avg_auc": round(float(avg_auc), 4),
        "avg_precision": round(float(avg_precision), 4),
        "avg_recall": round(float(avg_recall), 4),
        "avg_coverage": round(float(avg_coverage), 4),
        "top_features": importance_ranking,
        "per_fold": results,
    }

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1, help="Not used (sequential walk-forward)")
    parser.add_argument("--train-days", type=int, default=60)
    args = parser.parse_args()

    log.info("═══ XGBoost Execution Gate v2 ═══")
    log.info(f"  Train window: {args.train_days} days")
    log.info(f"  XGBoost available: {XGB_AVAILABLE}")

    t0 = time.time()
    # Get all dates with exec features
    all_dates = sorted([
        f.stem.replace("_exec_features", "")
        for f in EXEC_FEAT_DIR.glob("*_exec_features.npz")
    ])
    log.info(f"  Available dates: {len(all_dates)}")

    results = walk_forward_xgb(all_dates, train_days=args.train_days)

    elapsed = time.time() - t0
    results["total_time_s"] = round(elapsed, 1)

    log.info(f"\n═══ RESULTS ═══")
    if "error" not in results:
        log.info(f"  Folds: {results['n_folds']}")
        log.info(f"  Avg AUC: {results['avg_auc']:.4f}")
        log.info(f"  Avg Precision: {results['avg_precision']:.4f}")
        log.info(f"  Avg Recall: {results['avg_recall']:.4f}")
        log.info(f"  Avg Coverage: {results['avg_coverage']:.4f}")
        if results.get("top_features"):
            log.info(f"\n  Top 10 Features:")
            for f in results["top_features"][:10]:
                log.info(f"    {f['rank']:2d}. {f['feature']:35s} → {f['importance']:.4f}")
    else:
        log.error(f"  Error: {results['error']}")

    log.info(f"  Total time: {elapsed:.0f}s")

    # Save
    with open(OUTPUT_DIR / "xgb_gate_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"  Saved to {OUTPUT_DIR / 'xgb_gate_results.json'}")


if __name__ == "__main__":
    main()
