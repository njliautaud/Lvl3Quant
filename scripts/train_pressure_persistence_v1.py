"""
Pressure-Persistence XGBoost GPU Model (v1)
============================================
Predicts: Does 1s order-flow pressure direction persist at 30s? (binary)
Features: smart_v3 event features (25) + book features (30) = 55 features
Method: Sliding window walk-forward (60-day train, 1-day OOT)
GPU: XGBoost gpu_hist on RTX 3070

HC #0: Sliding window ONLY
HC #500: 2026 data only
HC #504: XGBoost whitelisted for Razer
"""

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import mlflow
import mlflow.xgboost
import glob
import os
import json
import gc
import time
from pathlib import Path

# === CONFIG ===
import platform
IS_WINDOWS = platform.system() == "Windows"

if IS_WINDOWS:
    DATA_ROOT = r"C:\Users\claude\Lvl3Quant\data\processed"
    OUTPUT_DIR = r"C:\Users\claude\Lvl3Quant\models\pressure_persistence_v1"
else:
    DATA_ROOT = "/home/jupiter/Lvl3Quant/data/processed"
    OUTPUT_DIR = "/home/jupiter/Lvl3Quant/models/pressure_persistence_v1"

SMART_V3_DIR = os.path.join(DATA_ROOT, "mbo_events_smart_v3")
PRESSURE_DIR = os.path.join(DATA_ROOT, "mbo_events_smart_v3_pressure_labels")
BOOK_DIR = os.path.join(DATA_ROOT, "mbo_book_features")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_WINDOW = 60  # days
TARGET_COL = "persistence_1s_30s"
MAX_SAMPLES_PER_DAY = 50000  # subsample large days to prevent OOM
if IS_WINDOWS:
    MLFLOW_URI = r"file:///C:/Users/claude/Lvl3Quant/mlruns"
else:
    MLFLOW_URI = "file:///home/jupiter/Lvl3Quant/mlruns"
EXPERIMENT_NAME = "pressure_persistence_razer_v1"

# XGBoost params - auto-detect GPU
try:
    import torch
    HAS_GPU = torch.cuda.is_available()
except ImportError:
    HAS_GPU = False

XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "gpu_hist" if HAS_GPU else "hist",
    "device": "cuda" if HAS_GPU else "cpu",
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 50,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_estimators": 500,
    "early_stopping_rounds": 30,
    "random_state": 42,
}

# Feature names for smart_v3 (25 features)
SMART_V3_FEATURES = [
    f"smart_v3_{i}" for i in range(25)
]

def get_available_dates():
    """Find all dates where smart_v3 + pressure labels exist (2026 only)."""
    smart_files = glob.glob(os.path.join(SMART_V3_DIR, "2026*_mbo_events.npz"))
    pressure_files = glob.glob(os.path.join(PRESSURE_DIR, "2026*_pressure.npz"))

    smart_dates = {os.path.basename(f)[:8] for f in smart_files}
    pressure_dates = {os.path.basename(f)[:8] for f in pressure_files}

    # Book features are optional (enhance if available)
    book_files = glob.glob(os.path.join(BOOK_DIR, "2026*_book_features.npz"))
    book_dates = {os.path.basename(f)[:8] for f in book_files}

    # Require smart_v3 + pressure; book is optional
    valid_dates = sorted(smart_dates & pressure_dates)
    print(f"Found {len(valid_dates)} dates with smart_v3 + pressure labels")
    print(f"  Book features available for {len(book_dates & set(valid_dates))} of those dates")
    return valid_dates, book_dates


def load_day(date_str, book_dates, max_samples=None, is_test=False):
    """Load features and target for a single day. Subsample if needed."""
    smart_path = os.path.join(SMART_V3_DIR, f"{date_str}_mbo_events.npz")
    pressure_path = os.path.join(PRESSURE_DIR, f"{date_str}_pressure.npz")
    book_path = os.path.join(BOOK_DIR, f"{date_str}_book_features.npz")

    smart_data = np.load(smart_path)
    pressure_data = np.load(pressure_path)

    events = smart_data["events"]  # (N, 25)
    target = pressure_data[TARGET_COL]  # (N,)

    # Ensure same length (pressure labels must match events)
    min_len = min(len(events), len(target))
    events = events[:min_len]
    target = target[:min_len]

    # Add book features if available
    if date_str in book_dates and os.path.exists(book_path):
        book_data = np.load(book_path, allow_pickle=True)
        book_feats = book_data["features"]  # (N, 30)
        if book_feats.shape[0] >= min_len:
            features = np.hstack([events, book_feats[:min_len]])
        else:
            features = events
    else:
        features = events

    # Filter out target == -1 (undefined)
    valid_mask = target >= 0
    features = features[valid_mask]
    target = target[valid_mask]

    # Subsample training data (keep all test data for accurate evaluation)
    if max_samples and not is_test and len(features) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(features), max_samples, replace=False)
        idx.sort()  # preserve temporal order
        features = features[idx]
        target = target[idx]

    return features, target.astype(np.int32)


def get_feature_names(has_book):
    """Get feature names list."""
    names = SMART_V3_FEATURES.copy()
    if has_book:
        book_names = [
            'bid_price_1', 'bid_price_2', 'bid_price_3', 'bid_price_4', 'bid_price_5',
            'ask_price_1', 'ask_price_2', 'ask_price_3', 'ask_price_4', 'ask_price_5',
            'bid_size_1', 'bid_size_2', 'bid_size_3', 'bid_size_4', 'bid_size_5',
            'ask_size_1', 'ask_size_2', 'ask_size_3', 'ask_size_4', 'ask_size_5',
            'cum_delta', 'rolling_imbalance_100', 'trade_intensity_100',
            'depth_imbalance_5', 'spread_ticks', 'bid_size_change',
            'ask_size_change', 'mid_price_change_ticks', 'spread_change_ticks',
            'net_order_flow'
        ]
        names.extend(book_names)
    return names


def train_fold(train_dates, test_date, book_dates, fold_idx):
    """Train one WF fold: train on train_dates, test on test_date."""
    # Load training data (subsample large days)
    X_train_list, y_train_list = [], []
    has_book_all = True
    for d in train_dates:
        X, y = load_day(d, book_dates, max_samples=MAX_SAMPLES_PER_DAY, is_test=False)
        if X.shape[1] == 25:
            has_book_all = False
        X_train_list.append(X)
        y_train_list.append(y)

    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)

    # Load test data (subsample test too for very large days to avoid OOM)
    X_test, y_test = load_day(test_date, book_dates, max_samples=MAX_SAMPLES_PER_DAY * 2, is_test=False)

    # Ensure consistent feature count
    n_feat_train = X_train.shape[1]
    n_feat_test = X_test.shape[1]
    if n_feat_train != n_feat_test:
        # Use minimum (smart_v3 only = 25)
        min_feat = min(n_feat_train, n_feat_test)
        X_train = X_train[:, :min_feat]
        X_test = X_test[:, :min_feat]
        has_book_all = (min_feat > 25)

    # Handle NaN/inf
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    # Scale pos weight for imbalanced classes
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / max(n_pos, 1)

    # Train XGBoost
    params = XGB_PARAMS.copy()
    params["scale_pos_weight"] = scale_pos_weight
    n_estimators = params.pop("n_estimators")
    early_stopping = params.pop("early_stopping_rounds")

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping,
        **params
    )

    # Use last 20% of train as eval set for early stopping
    split_idx = int(len(X_train) * 0.8)
    X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
    y_tr, y_val = y_train[:split_idx], y_train[split_idx:]

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # Predict
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    # Metrics
    if len(np.unique(y_test)) > 1:
        auc = roc_auc_score(y_test, y_prob)
    else:
        auc = 0.5  # undefined

    acc = accuracy_score(y_test, y_pred)
    pos_rate = y_test.mean()

    # Feature importance
    importance = model.feature_importances_

    return {
        "fold": fold_idx,
        "test_date": test_date,
        "auc": auc,
        "accuracy": acc,
        "pos_rate": pos_rate,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "best_iteration": model.best_iteration,
        "y_prob": y_prob,
        "y_test": y_test,
        "importance": importance,
        "has_book": has_book_all,
        "scale_pos_weight": scale_pos_weight,
    }


def main():
    print("=" * 60)
    print("PRESSURE-PERSISTENCE XGBoost GPU v1")
    print("Target: persistence_1s_30s (does 1s pressure persist at 30s?)")
    print("Method: Sliding WF, 60-day train, 1-day OOT, 2026 only")
    print("=" * 60)

    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Get dates
    all_dates, book_dates = get_available_dates()

    if len(all_dates) < TRAIN_WINDOW + 1:
        print(f"ERROR: Need at least {TRAIN_WINDOW + 1} dates, have {len(all_dates)}")
        return

    # Sliding window walk-forward
    results = []
    all_y_prob = []
    all_y_test = []

    n_folds = len(all_dates) - TRAIN_WINDOW
    print(f"\nRunning {n_folds} OOT folds...")
    print(f"Train window: {TRAIN_WINDOW} days, OOT: 1 day")

    with mlflow.start_run(run_name=f"pressure_persist_xgb_gpu_v1_{time.strftime('%Y%m%d_%H%M')}"):
        # Log params
        mlflow.log_params({
            "model_type": "XGBoost_GPU",
            "target": TARGET_COL,
            "train_window": TRAIN_WINDOW,
            "n_folds": n_folds,
            "tree_method": "gpu_hist",
            "max_depth": XGB_PARAMS["max_depth"],
            "learning_rate": XGB_PARAMS["learning_rate"],
            "n_estimators": XGB_PARAMS["n_estimators"],
            "data_dates": f"{all_dates[0]}-{all_dates[-1]}",
        })

        t_start = time.time()

        for i in range(n_folds):
            train_dates = all_dates[i:i + TRAIN_WINDOW]
            test_date = all_dates[i + TRAIN_WINDOW]

            fold_result = train_fold(train_dates, test_date, book_dates, i)
            results.append(fold_result)
            all_y_prob.append(fold_result["y_prob"])
            all_y_test.append(fold_result["y_test"])

            # Progress
            if (i + 1) % 5 == 0 or i == n_folds - 1:
                elapsed = time.time() - t_start
                avg_auc = np.mean([r["auc"] for r in results])
                print(f"  Fold {i+1}/{n_folds} | Date: {test_date} | "
                      f"AUC: {fold_result['auc']:.4f} | Avg AUC: {avg_auc:.4f} | "
                      f"Time: {elapsed:.0f}s")

            # Log per-fold metrics
            mlflow.log_metric("fold_auc", fold_result["auc"], step=i)
            mlflow.log_metric("fold_accuracy", fold_result["accuracy"], step=i)

            # Free memory
            gc.collect()

        # === AGGREGATE RESULTS ===
        elapsed_total = time.time() - t_start

        # Concat AUC (primary metric)
        all_y_prob_concat = np.concatenate(all_y_prob)
        all_y_test_concat = np.concatenate(all_y_test)
        concat_auc = roc_auc_score(all_y_test_concat, all_y_prob_concat)

        # Per-fold stats
        fold_aucs = [r["auc"] for r in results]
        mean_auc = np.mean(fold_aucs)
        std_auc = np.std(fold_aucs)
        min_auc = np.min(fold_aucs)
        max_auc = np.max(fold_aucs)

        # Accuracy at various thresholds
        concat_pred_50 = (all_y_prob_concat >= 0.5).astype(int)
        concat_acc = accuracy_score(all_y_test_concat, concat_pred_50)

        # Top decile precision (most confident predictions)
        top_10_thresh = np.percentile(all_y_prob_concat, 90)
        top_10_mask = all_y_prob_concat >= top_10_thresh
        top_10_precision = all_y_test_concat[top_10_mask].mean() if top_10_mask.sum() > 0 else 0

        # Feature importance (average across folds)
        avg_importance = np.mean([r["importance"] for r in results], axis=0)
        n_features = len(avg_importance)
        feature_names = get_feature_names(n_features > 25)[:n_features]

        # Top features
        top_idx = np.argsort(avg_importance)[::-1][:10]
        top_features = [(feature_names[i], float(avg_importance[i])) for i in top_idx]

        # Print results
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        print(f"Concat AUC (primary):    {concat_auc:.4f}")
        print(f"Mean fold AUC:           {mean_auc:.4f} +/- {std_auc:.4f}")
        print(f"Min/Max fold AUC:        {min_auc:.4f} / {max_auc:.4f}")
        print(f"Concat accuracy:         {concat_acc:.4f}")
        print(f"Top-10% precision:       {top_10_precision:.4f}")
        print(f"Baseline (pos rate):     {all_y_test_concat.mean():.4f}")
        print(f"Total OOT samples:       {len(all_y_test_concat):,}")
        print(f"Total training time:     {elapsed_total:.0f}s")
        print(f"Folds completed:         {n_folds}")
        print(f"\nTop 10 features:")
        for name, imp in top_features:
            print(f"  {name}: {imp:.4f}")

        # Log to MLflow
        mlflow.log_metrics({
            "concat_auc": concat_auc,
            "mean_fold_auc": mean_auc,
            "std_fold_auc": std_auc,
            "min_fold_auc": min_auc,
            "max_fold_auc": max_auc,
            "concat_accuracy": concat_acc,
            "top_10_precision": top_10_precision,
            "baseline_pos_rate": float(all_y_test_concat.mean()),
            "total_oot_samples": len(all_y_test_concat),
            "training_time_s": elapsed_total,
            "n_folds": n_folds,
            "n_features": n_features,
        })

        # Log feature importance
        fi_dict = {name: float(imp) for name, imp in zip(feature_names, avg_importance)}
        mlflow.log_dict(fi_dict, "feature_importance.json")
        mlflow.log_dict(
            {"top_features": top_features, "concat_auc": concat_auc, "mean_auc": mean_auc},
            "summary.json"
        )

        # Save OOS predictions
        np.savez(
            os.path.join(OUTPUT_DIR, "oos_predictions_v1.npz"),
            y_prob=all_y_prob_concat,
            y_test=all_y_test_concat,
            fold_dates=[r["test_date"] for r in results],
            fold_aucs=fold_aucs,
        )

        # Save per-fold results
        fold_summary = [{
            "fold": r["fold"],
            "test_date": r["test_date"],
            "auc": r["auc"],
            "accuracy": r["accuracy"],
            "pos_rate": r["pos_rate"],
            "n_test": r["n_test"],
            "best_iteration": r["best_iteration"],
        } for r in results]

        with open(os.path.join(OUTPUT_DIR, "fold_results_v1.json"), "w") as f:
            json.dump(fold_summary, f, indent=2)

        mlflow.log_artifact(os.path.join(OUTPUT_DIR, "fold_results_v1.json"))

        print(f"\nResults saved to: {OUTPUT_DIR}")
        print(f"MLflow run logged to: {MLFLOW_URI}")
        print("DONE.")


if __name__ == "__main__":
    main()
