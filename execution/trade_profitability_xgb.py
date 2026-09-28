#!/usr/bin/env python3
"""
Trade Profitability Predictor — XGBoost
========================================
Predicts P(profitable trade | AI signal + microstructure state).

Combines:
  - CNN-Mamba v2 OOT predictions (signal strength/direction)
  - PatchTST OOT predictions (ensemble signal)
  - Exec features (microstructure: spread, vol, fill_prob, queue depth, etc.)

Target: was the trade profitable after realistic costs?
  - All order types: 0.376 ticks RT commission only (HC #231(A): no spread cost)

Walk-forward: 60-day sliding window, 1-day OOT.
Output: /home/jupiter/Lvl3Quant/output/trade_profitability_xgb/

Cost constants (HC #52):
  ES_TICK_VALUE = $12.50
  ES_RT_COMMISSION = $4.70 = 0.376 ticks
"""

import argparse
import glob
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import xgboost as xgb
except ImportError:
    print("ERROR: xgboost not installed. Run: pip install xgboost")
    sys.exit(1)

from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    average_precision_score,
    precision_score,
)
from sklearn.calibration import calibration_curve

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FEATURES_DIR = Path("/home/jupiter/Lvl3Quant/output/exec_features_v1")
CNN_MAMBA_DIR = Path("/home/jupiter/Lvl3Quant/output/cnn_mamba_v2_smart_v3_mar")
PATCHTST_DIR = Path("/home/jupiter/Lvl3Quant/output/patchtst_smart_v3_mar")
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/trade_profitability_xgb")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Cost thresholds in ticks. HC #231(A): no spread crossing cost — only commission.
COST_LIMIT = 0.376       # RT commission only
COST_MID = 0.376          # HC #231(A): commission only (no spread cost)
COST_MARKET = 0.376       # HC #231(A): commission only (no spread cost)
ES_TICK_VALUE = 12.50

# Walk-forward
TRAIN_WINDOW = 60  # days

# Exec feature columns (44 total, first 3 are fill_prob targets)
EXEC_FEATURE_NAMES = [
    'fill_prob_1s', 'fill_prob_3s', 'fill_prob_10s',
    'queue_consumption_velocity_bid', 'queue_consumption_velocity_ask',
    'queue_replenish_ratio', 'post_trade_drift_bid_1k',
    'post_trade_drift_ask_1k', 'toxicity_imbalance', 'large_trade_fraction',
    'informed_flow_score', 'adverse_select_asymmetry', 'trade_imb_momentum',
    'trade_imb_acceleration', 'volume_weighted_imbalance',
    'small_vs_large_imbalance', 'cancel_velocity_bid', 'cancel_velocity_ask',
    'add_cancel_ratio_bid', 'add_cancel_ratio_ask', 'spread_ticks',
    'spread_volatility', 'time_in_spread', 'spread_state', 'spread_mean_10k',
    'spread_widening_trend', 'bid_depth_l1', 'ask_depth_l1', 'bid_depth_l2_l5',
    'ask_depth_l2_l5', 'depth_restoration_speed', 'depth_imbalance_l1',
    'layering_score_bid', 'layering_score_ask', 'tod_sin', 'tod_cos',
    'minutes_from_open', 'session_progress', 'event_rate', 'trade_rate',
    'cancel_rate', 'add_rate', 'book_turnover', 'price_volatility_window',
]

# Indices for key exec features
IDX_FILL_PROB_1S = 0
IDX_FILL_PROB_3S = 1
IDX_FILL_PROB_10S = 2
IDX_SPREAD_TICKS = 20
IDX_SPREAD_STATE = 23
IDX_PRICE_VOL = 43
IDX_TOD_SIN = 34
IDX_TOD_COS = 35
IDX_SESSION_PROGRESS = 37
IDX_DEPTH_IMBALANCE = 31
IDX_TRADE_IMB_MOMENTUM = 12

# Selected exec feature indices for the model (skip raw fill_prob — they're separate features)
EXEC_FEAT_INDICES = list(range(3, 44))  # 41 microstructure features

# XGBoost params
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 20,
    "gamma": 0.1,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "nthread": 1,  # per-model; parallelism via ProcessPool
    "verbosity": 0,
    "seed": 42,
}
NUM_BOOST_ROUNDS = 800
EARLY_STOPPING_ROUNDS = 40

# Regime thresholds
REGIME_SPREAD_THRESH = 1.5   # ticks
REGIME_VOL_QUANTILE = 0.75   # high vol threshold

# Label horizon to use (10s = our standard reporting horizon)
PRIMARY_HORIZON = 2  # index into (1s, 5s, 10s) predictions/labels
HORIZON_NAMES = ['1s', '5s', '10s']

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(OUTPUT_DIR / "training.log"),
    ],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def extract_date_from_oot(oot_files_arr) -> Optional[str]:
    """Extract YYYYMMDD date from oot_files path."""
    path = str(oot_files_arr[0]) if hasattr(oot_files_arr, '__len__') else str(oot_files_arr)
    m = re.search(r'(\d{8})_mbo', path)
    return m.group(1) if m else None


def load_prediction_files(pred_dir: Path) -> Dict[str, dict]:
    """Load all fold prediction files, indexed by date."""
    date_preds = {}
    for fpath in sorted(pred_dir.glob("fold_*_oot_predictions.npz")):
        try:
            data = np.load(fpath, allow_pickle=True)
            date = extract_date_from_oot(data['oot_files'])
            if date is None:
                continue
            date_preds[date] = {
                'predictions': data['predictions'],  # (N, 3)
                'labels': data['labels'],              # (N, 3)
                'file': str(fpath),
            }
            # Include embeddings if available
            if 'embeddings' in data:
                date_preds[date]['embeddings'] = data['embeddings']
        except Exception as e:
            log.warning(f"Failed to load {fpath}: {e}")
    return date_preds


def load_exec_features(date: str) -> Optional[np.ndarray]:
    """Load exec features for a given date. Returns (N_windows, 44) array."""
    fpath = FEATURES_DIR / f"{date}_exec_features.npz"
    if not fpath.exists():
        return None
    try:
        data = np.load(fpath, allow_pickle=True)
        return data['features']  # (N_windows, 44)
    except Exception as e:
        log.warning(f"Failed to load exec features for {date}: {e}")
        return None


def align_predictions_to_exec(pred_n: int, exec_n: int) -> np.ndarray:
    """
    Map prediction indices to exec feature window indices.
    Both are ordered chronologically within the trading day.
    We use proportional mapping: pred_idx / pred_n -> exec_idx / exec_n.
    Returns array of exec indices for each prediction.
    """
    pred_positions = np.arange(pred_n, dtype=np.float64) / max(pred_n - 1, 1)
    exec_indices = (pred_positions * (exec_n - 1)).astype(np.int64)
    exec_indices = np.clip(exec_indices, 0, exec_n - 1)
    return exec_indices


def build_date_features(
    date: str,
    cnn_mamba_preds: Optional[dict],
    patchtst_preds: Optional[dict],
    exec_features: np.ndarray,
    cost_type: str = "limit",
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Build feature matrix and target for a single date.

    Returns: (X, y, metadata) where metadata contains regime info.
    """
    # Use CNN-Mamba as primary (most available), fallback to PatchTST
    if cnn_mamba_preds is not None:
        primary_preds = cnn_mamba_preds['predictions']  # (N, 3)
        primary_labels = cnn_mamba_preds['labels']      # (N, 3)
        n_samples = primary_preds.shape[0]
    elif patchtst_preds is not None:
        primary_preds = patchtst_preds['predictions']
        primary_labels = patchtst_preds['labels']
        n_samples = primary_preds.shape[0]
    else:
        return None

    exec_n = exec_features.shape[0]
    if exec_n < 10:
        return None

    # Skip very small folds (e.g., partial days with <100 samples)
    if n_samples < 100:
        return None

    # Align predictions to exec feature windows
    exec_map = align_predictions_to_exec(n_samples, exec_n)
    aligned_exec = exec_features[exec_map]  # (n_samples, 44)

    # Cost threshold based on order type
    cost_map = {"limit": COST_LIMIT, "mid": COST_MID, "market": COST_MARKET}
    cost_threshold = cost_map.get(cost_type, COST_LIMIT)

    # Build target: profitable trade at 10s horizon
    # Label is price change in ticks. Long trade profitable if label > cost.
    # We consider BOTH directions: max(|label|) > cost for optimal direction.
    labels_10s = primary_labels[:, PRIMARY_HORIZON]  # 10s horizon

    # Directional profitability: take trade in predicted direction
    pred_direction = np.sign(primary_preds[:, PRIMARY_HORIZON])
    # Actual PnL if we follow the prediction direction
    directional_pnl = pred_direction * labels_10s  # positive = correct direction
    y = (directional_pnl > cost_threshold).astype(np.float32)

    # --- Feature engineering ---
    feature_list = []
    feature_names = []

    # 1. CNN-Mamba prediction features
    for h, hname in enumerate(HORIZON_NAMES):
        pred_h = primary_preds[:, h]
        feature_list.append(pred_h.reshape(-1, 1))
        feature_names.append(f"pred_{hname}")

        feature_list.append(np.abs(pred_h).reshape(-1, 1))
        feature_names.append(f"pred_abs_{hname}")

    # Prediction direction (sign)
    feature_list.append(pred_direction.reshape(-1, 1))
    feature_names.append("pred_direction")

    # Prediction magnitude (z-score of absolute prediction)
    pred_abs = np.abs(primary_preds[:, PRIMARY_HORIZON])
    pred_zscore = (pred_abs - np.mean(pred_abs)) / (np.std(pred_abs) + 1e-8)
    feature_list.append(pred_zscore.reshape(-1, 1))
    feature_names.append("pred_magnitude_zscore")

    # Cross-horizon consistency: do 1s/5s/10s agree?
    signs = np.sign(primary_preds)
    consistency = np.mean(signs == signs[:, [PRIMARY_HORIZON]], axis=1)
    feature_list.append(consistency.reshape(-1, 1))
    feature_names.append("pred_cross_horizon_consistency")

    # Prediction confidence: ratio of 10s to 1s prediction
    safe_denom = np.abs(primary_preds[:, 0]) + 1e-8
    pred_ratio = np.abs(primary_preds[:, PRIMARY_HORIZON]) / safe_denom
    feature_list.append(pred_ratio.reshape(-1, 1))
    feature_names.append("pred_10s_1s_ratio")

    # 2. PatchTST ensemble features (if available for this date)
    if patchtst_preds is not None and cnn_mamba_preds is not None:
        # Align PatchTST to same sample count as CNN-Mamba
        pt_preds = patchtst_preds['predictions']
        pt_n = pt_preds.shape[0]
        if pt_n != n_samples:
            # Resample PatchTST to match CNN-Mamba count
            pt_indices = (np.arange(n_samples, dtype=np.float64) / max(n_samples - 1, 1) * (pt_n - 1)).astype(np.int64)
            pt_indices = np.clip(pt_indices, 0, pt_n - 1)
            pt_preds_aligned = pt_preds[pt_indices]
        else:
            pt_preds_aligned = pt_preds

        for h, hname in enumerate(HORIZON_NAMES):
            feature_list.append(pt_preds_aligned[:, h].reshape(-1, 1))
            feature_names.append(f"patchtst_pred_{hname}")

        # Model agreement: do CNN-Mamba and PatchTST agree on direction?
        agreement = (np.sign(primary_preds[:, PRIMARY_HORIZON]) ==
                     np.sign(pt_preds_aligned[:, PRIMARY_HORIZON])).astype(np.float32)
        feature_list.append(agreement.reshape(-1, 1))
        feature_names.append("model_agreement_10s")

        # Ensemble mean prediction
        ens_mean = (primary_preds[:, PRIMARY_HORIZON] + pt_preds_aligned[:, PRIMARY_HORIZON]) / 2
        feature_list.append(ens_mean.reshape(-1, 1))
        feature_names.append("ensemble_mean_10s")
    else:
        # Pad with zeros when PatchTST not available
        for hname in HORIZON_NAMES:
            feature_list.append(np.zeros((n_samples, 1), dtype=np.float32))
            feature_names.append(f"patchtst_pred_{hname}")
        feature_list.append(np.zeros((n_samples, 1), dtype=np.float32))
        feature_names.append("model_agreement_10s")
        feature_list.append(np.zeros((n_samples, 1), dtype=np.float32))
        feature_names.append("ensemble_mean_10s")

    # 3. Exec microstructure features (41 features, indices 3-43)
    exec_subset = aligned_exec[:, EXEC_FEAT_INDICES]
    feature_list.append(exec_subset)
    for idx in EXEC_FEAT_INDICES:
        feature_names.append(f"exec_{EXEC_FEATURE_NAMES[idx]}")

    # 4. Interaction features: prediction x microstructure
    spread = aligned_exec[:, IDX_SPREAD_TICKS]
    vol = aligned_exec[:, IDX_PRICE_VOL]
    fill_prob = aligned_exec[:, IDX_FILL_PROB_10S]
    depth_imb = aligned_exec[:, IDX_DEPTH_IMBALANCE]

    pred_10s = primary_preds[:, PRIMARY_HORIZON]

    # Prediction x spread state
    feature_list.append((pred_10s * spread).reshape(-1, 1))
    feature_names.append("pred_x_spread")

    # Prediction x volatility
    feature_list.append((pred_10s * vol).reshape(-1, 1))
    feature_names.append("pred_x_vol")

    # Prediction x fill probability
    feature_list.append((pred_10s * fill_prob).reshape(-1, 1))
    feature_names.append("pred_x_fill_prob")

    # Prediction x depth imbalance (alignment = good)
    feature_list.append((pred_10s * depth_imb).reshape(-1, 1))
    feature_names.append("pred_x_depth_imbalance")

    # |Prediction| x spread (wider spread needs stronger signal)
    feature_list.append((pred_abs * spread).reshape(-1, 1))
    feature_names.append("pred_abs_x_spread")

    # Signal-to-cost ratio: |prediction| / spread
    safe_spread = spread + 1e-8
    feature_list.append((pred_abs / safe_spread).reshape(-1, 1))
    feature_names.append("signal_to_cost_ratio")

    # Prediction magnitude x cross-horizon consistency
    feature_list.append((pred_abs * consistency).reshape(-1, 1))
    feature_names.append("pred_abs_x_consistency")

    # Spread regime: tight (<=1), normal (1-1.5), wide (>1.5)
    spread_regime = np.where(spread <= 1.0, 0, np.where(spread <= 1.5, 1, 2)).astype(np.float32)
    feature_list.append(spread_regime.reshape(-1, 1))
    feature_names.append("spread_regime")

    # Vol regime: compute from data
    vol_regime = np.where(vol <= np.nanquantile(vol, 0.25), 0,
                  np.where(vol <= np.nanquantile(vol, 0.75), 1, 2)).astype(np.float32)
    feature_list.append(vol_regime.reshape(-1, 1))
    feature_names.append("vol_regime")

    # Concatenate all features
    X = np.hstack(feature_list).astype(np.float32)

    # Replace NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Metadata for regime analysis
    metadata = np.column_stack([
        spread_regime,
        vol_regime,
        aligned_exec[:, IDX_SESSION_PROGRESS],  # time of day
    ]).astype(np.float32)

    return X, y, metadata


# ---------------------------------------------------------------------------
# Walk-forward training
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    fold_idx: int
    date: str
    n_train: int
    n_test: int
    auc: float
    avg_precision: float
    precision_at_5pct: float
    precision_at_10pct: float
    brier: float
    base_rate: float
    predictions: np.ndarray
    labels: np.ndarray
    regime_results: dict
    feature_importances: Optional[np.ndarray] = None


def train_fold(
    fold_idx: int,
    train_dates: List[str],
    test_date: str,
    all_date_data: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    feature_names: List[str],
) -> Optional[FoldResult]:
    """Train single walk-forward fold."""
    try:
        # Assemble training data
        X_train_parts, y_train_parts = [], []
        for d in train_dates:
            if d in all_date_data:
                X_train_parts.append(all_date_data[d][0])
                y_train_parts.append(all_date_data[d][1])

        if not X_train_parts:
            return None

        X_train = np.vstack(X_train_parts)
        y_train = np.concatenate(y_train_parts)

        if test_date not in all_date_data:
            return None

        X_test, y_test, metadata_test = all_date_data[test_date]

        if len(X_test) < 50 or len(X_train) < 500:
            return None

        # Skip if target is degenerate
        if y_train.sum() < 10 or (y_train.sum() / len(y_train)) > 0.99:
            log.warning(f"Fold {fold_idx} ({test_date}): degenerate target, skipping")
            return None

        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
        dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)

        # Train with early stopping
        model = xgb.train(
            XGB_PARAMS,
            dtrain,
            num_boost_round=NUM_BOOST_ROUNDS,
            evals=[(dtest, "oot")],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=False,
        )

        # Predict
        y_pred = model.predict(dtest)

        # Metrics
        try:
            auc = roc_auc_score(y_test, y_pred)
        except ValueError:
            auc = 0.5

        avg_prec = average_precision_score(y_test, y_pred)
        brier = brier_score_loss(y_test, y_pred)
        base_rate = y_test.mean()

        # Precision at top-K%
        n_test = len(y_test)
        sorted_idx = np.argsort(-y_pred)

        top_5pct = max(1, int(0.05 * n_test))
        top_10pct = max(1, int(0.10 * n_test))
        prec_5 = y_test[sorted_idx[:top_5pct]].mean()
        prec_10 = y_test[sorted_idx[:top_10pct]].mean()

        # Feature importances
        importance = model.get_score(importance_type='gain')
        imp_arr = np.zeros(len(feature_names))
        for fname, gain in importance.items():
            if fname in feature_names:
                imp_arr[feature_names.index(fname)] = gain

        # Per-regime breakdown
        regime_results = {}
        spread_regimes = metadata_test[:, 0]
        vol_regimes = metadata_test[:, 1]
        session_progress = metadata_test[:, 2]

        # By spread regime
        for regime_val, regime_name in [(0, "tight_spread"), (1, "normal_spread"), (2, "wide_spread")]:
            mask = spread_regimes == regime_val
            if mask.sum() > 20:
                try:
                    r_auc = roc_auc_score(y_test[mask], y_pred[mask])
                except ValueError:
                    r_auc = 0.5
                regime_results[regime_name] = {
                    "n": int(mask.sum()),
                    "auc": float(r_auc),
                    "base_rate": float(y_test[mask].mean()),
                    "mean_pred": float(y_pred[mask].mean()),
                }

        # By vol regime
        for regime_val, regime_name in [(0, "low_vol"), (1, "normal_vol"), (2, "high_vol")]:
            mask = vol_regimes == regime_val
            if mask.sum() > 20:
                try:
                    r_auc = roc_auc_score(y_test[mask], y_pred[mask])
                except ValueError:
                    r_auc = 0.5
                regime_results[regime_name] = {
                    "n": int(mask.sum()),
                    "auc": float(r_auc),
                    "base_rate": float(y_test[mask].mean()),
                    "mean_pred": float(y_pred[mask].mean()),
                }

        # By session period
        for lo, hi, name in [(0, 0.15, "open_15min"), (0.15, 0.5, "morning"),
                              (0.5, 0.85, "midday"), (0.85, 1.0, "close_15min")]:
            mask = (session_progress >= lo) & (session_progress < hi)
            if mask.sum() > 20:
                try:
                    r_auc = roc_auc_score(y_test[mask], y_pred[mask])
                except ValueError:
                    r_auc = 0.5
                regime_results[name] = {
                    "n": int(mask.sum()),
                    "auc": float(r_auc),
                    "base_rate": float(y_test[mask].mean()),
                    "mean_pred": float(y_pred[mask].mean()),
                }

        log.info(
            f"Fold {fold_idx:3d} | {test_date} | AUC={auc:.4f} | "
            f"AP={avg_prec:.4f} | P@5%={prec_5:.3f} | P@10%={prec_10:.3f} | "
            f"base_rate={base_rate:.3f} | n_train={len(X_train):,d} | n_test={n_test:,d}"
        )

        return FoldResult(
            fold_idx=fold_idx,
            date=test_date,
            n_train=len(X_train),
            n_test=n_test,
            auc=auc,
            avg_precision=avg_prec,
            precision_at_5pct=prec_5,
            precision_at_10pct=prec_10,
            brier=brier,
            base_rate=base_rate,
            predictions=y_pred,
            labels=y_test,
            regime_results=regime_results,
            feature_importances=imp_arr,
        )

    except Exception as e:
        log.error(f"Fold {fold_idx} ({test_date}) failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_walk_forward(
    all_date_data: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    feature_names: List[str],
    workers: int = 16,
    cost_type: str = "limit",
) -> List[FoldResult]:
    """Run walk-forward training across all available dates."""
    sorted_dates = sorted(all_date_data.keys())
    n_dates = len(sorted_dates)
    log.info(f"Walk-forward: {n_dates} dates, cost_type={cost_type}")

    # Adaptive window strategy:
    # - If we have >= TRAIN_WINDOW+1 dates: sliding window of TRAIN_WINDOW
    # - If we have fewer: expanding window, minimum 3 training dates
    MIN_TRAIN_DATES = 3

    if n_dates < MIN_TRAIN_DATES + 1:
        log.error(f"Not enough dates ({n_dates}) — need at least {MIN_TRAIN_DATES + 1}")
        return []

    folds = []
    if n_dates >= TRAIN_WINDOW + 1:
        # Standard sliding window
        log.info(f"Using {TRAIN_WINDOW}-day sliding window")
        for i in range(TRAIN_WINDOW, n_dates):
            test_date = sorted_dates[i]
            train_dates = sorted_dates[i - TRAIN_WINDOW:i]
            folds.append((len(folds), train_dates, test_date))
    else:
        # Expanding window: start with MIN_TRAIN_DATES, expand each fold
        log.info(f"Using expanding window (min {MIN_TRAIN_DATES} train dates, {n_dates - MIN_TRAIN_DATES} folds)")
        for i in range(MIN_TRAIN_DATES, n_dates):
            test_date = sorted_dates[i]
            train_dates = sorted_dates[:i]
            folds.append((len(folds), train_dates, test_date))

    log.info(f"Total folds to train: {len(folds)}")

    results = []
    if workers <= 1 or len(folds) <= 2:
        for fold_idx, train_dates, test_date in folds:
            r = train_fold(fold_idx, train_dates, test_date, all_date_data, feature_names)
            if r is not None:
                results.append(r)
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(folds))) as executor:
            futures = {}
            for fold_idx, train_dates, test_date in folds:
                fut = executor.submit(
                    train_fold, fold_idx, train_dates, test_date, all_date_data, feature_names
                )
                futures[fut] = fold_idx

            for fut in as_completed(futures):
                r = fut.result()
                if r is not None:
                    results.append(r)

    results.sort(key=lambda r: r.date)
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def compute_calibration(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10):
    """Compute calibration curve data."""
    try:
        frac_pos, mean_pred = calibration_curve(y_true, y_pred, n_bins=n_bins, strategy='uniform')
        return {
            "fraction_positives": frac_pos.tolist(),
            "mean_predicted": mean_pred.tolist(),
        }
    except Exception:
        return {}


def generate_report(results: List[FoldResult], feature_names: List[str], cost_type: str):
    """Generate comprehensive evaluation report."""
    if not results:
        log.error("No results to report!")
        return {}

    aucs = [r.auc for r in results]
    aps = [r.avg_precision for r in results]
    p5s = [r.precision_at_5pct for r in results]
    p10s = [r.precision_at_10pct for r in results]
    briers = [r.brier for r in results]
    base_rates = [r.base_rate for r in results]

    # Aggregate regime results
    regime_agg = defaultdict(lambda: {"aucs": [], "base_rates": [], "n_total": 0})
    for r in results:
        for regime, data in r.regime_results.items():
            regime_agg[regime]["aucs"].append(data["auc"])
            regime_agg[regime]["base_rates"].append(data["base_rate"])
            regime_agg[regime]["n_total"] += data["n"]

    regime_summary = {}
    for regime, data in regime_agg.items():
        regime_summary[regime] = {
            "mean_auc": float(np.mean(data["aucs"])),
            "std_auc": float(np.std(data["aucs"])),
            "mean_base_rate": float(np.mean(data["base_rates"])),
            "n_total": data["n_total"],
            "n_folds": len(data["aucs"]),
        }

    # Feature importance aggregation
    all_importances = np.array([r.feature_importances for r in results if r.feature_importances is not None])
    if len(all_importances) > 0:
        mean_imp = np.mean(all_importances, axis=0)
        top_features = sorted(
            zip(feature_names, mean_imp.tolist()),
            key=lambda x: x[1], reverse=True
        )[:20]
    else:
        top_features = []

    # Concatenated calibration
    all_preds = np.concatenate([r.predictions for r in results])
    all_labels = np.concatenate([r.labels for r in results])
    calibration = compute_calibration(all_labels, all_preds)

    report = {
        "cost_type": cost_type,
        "cost_threshold_ticks": {"limit": COST_LIMIT, "mid": COST_MID, "market": COST_MARKET}[cost_type],
        "n_folds": len(results),
        "date_range": f"{results[0].date} - {results[-1].date}",
        "total_samples": sum(r.n_test for r in results),
        "metrics": {
            "auc_mean": float(np.mean(aucs)),
            "auc_std": float(np.std(aucs)),
            "auc_min": float(np.min(aucs)),
            "auc_max": float(np.max(aucs)),
            "avg_precision_mean": float(np.mean(aps)),
            "avg_precision_std": float(np.std(aps)),
            "precision_at_5pct_mean": float(np.mean(p5s)),
            "precision_at_5pct_std": float(np.std(p5s)),
            "precision_at_10pct_mean": float(np.mean(p10s)),
            "precision_at_10pct_std": float(np.std(p10s)),
            "brier_mean": float(np.mean(briers)),
            "base_rate_mean": float(np.mean(base_rates)),
        },
        "per_fold": [
            {
                "fold": r.fold_idx,
                "date": r.date,
                "auc": float(r.auc),
                "avg_precision": float(r.avg_precision),
                "precision_at_5pct": float(r.precision_at_5pct),
                "precision_at_10pct": float(r.precision_at_10pct),
                "brier": float(r.brier),
                "base_rate": float(r.base_rate),
                "n_train": int(r.n_train),
                "n_test": int(r.n_test),
            }
            for r in results
        ],
        "regime_analysis": regime_summary,
        "top_20_features": [{"name": n, "mean_gain": g} for n, g in top_features],
        "calibration": calibration,
    }

    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Trade Profitability XGBoost Predictor")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--cost-type", choices=["limit", "mid", "market"], default="limit",
                        help="Cost model: all 0.376t commission only (HC #231(A) — no spread cost)")
    parser.add_argument("--all-costs", action="store_true",
                        help="Run for all three cost types")
    args = parser.parse_args()

    log.info("=" * 80)
    log.info("Trade Profitability XGBoost Predictor")
    log.info("=" * 80)
    t0 = time.time()

    # Step 1: Load all prediction files
    log.info("Loading CNN-Mamba v2 predictions...")
    cnn_mamba_by_date = load_prediction_files(CNN_MAMBA_DIR)
    log.info(f"  Loaded {len(cnn_mamba_by_date)} dates from CNN-Mamba")

    log.info("Loading PatchTST predictions...")
    patchtst_by_date = load_prediction_files(PATCHTST_DIR)
    log.info(f"  Loaded {len(patchtst_by_date)} dates from PatchTST")

    # Step 2: Find dates where we have BOTH predictions AND exec features
    all_pred_dates = set(cnn_mamba_by_date.keys()) | set(patchtst_by_date.keys())
    log.info(f"Total unique prediction dates: {len(all_pred_dates)}")

    # Also get ALL exec feature dates (for training window padding)
    exec_dates = set()
    for f in FEATURES_DIR.glob("*_exec_features.npz"):
        date = f.stem.replace("_exec_features", "")
        exec_dates.add(date)
    log.info(f"Total exec feature dates: {len(exec_dates)}")

    # Dates with both predictions and exec features
    matched_dates = sorted(all_pred_dates & exec_dates)
    log.info(f"Matched dates (predictions + exec features): {len(matched_dates)}")
    for d in matched_dates:
        src = []
        if d in cnn_mamba_by_date:
            src.append(f"CNN-Mamba({cnn_mamba_by_date[d]['predictions'].shape[0]})")
        if d in patchtst_by_date:
            src.append(f"PatchTST({patchtst_by_date[d]['predictions'].shape[0]})")
        log.info(f"  {d}: {', '.join(src)}")

    if not matched_dates:
        log.error("No matched dates found! Cannot train.")
        sys.exit(1)

    # For walk-forward we need a broader date set. Use ALL exec feature dates
    # as potential training dates (even without AI predictions - we'll build
    # features using exec features only + synthetic zero predictions).
    # But for TEST dates we need real predictions.
    # Strategy: build data for matched dates, use expanding/sliding window within them.

    cost_types = ["limit", "mid", "market"] if args.all_costs else [args.cost_type]

    for cost_type in cost_types:
        log.info(f"\n{'='*60}")
        log.info(f"Training for cost_type={cost_type}")
        log.info(f"{'='*60}")

        # Step 3: Build feature matrices for ALL matched dates
        log.info("Building feature matrices...")
        all_date_data = {}
        feature_names = None

        for date in matched_dates:
            exec_feats = load_exec_features(date)
            if exec_feats is None:
                continue

            result = build_date_features(
                date,
                cnn_mamba_by_date.get(date),
                patchtst_by_date.get(date),
                exec_feats,
                cost_type=cost_type,
            )
            if result is None:
                continue

            X, y, metadata = result
            all_date_data[date] = (X, y, metadata)

            if feature_names is None:
                # Build feature names list
                feature_names = []
                for h in HORIZON_NAMES:
                    feature_names.append(f"pred_{h}")
                    feature_names.append(f"pred_abs_{h}")
                feature_names.append("pred_direction")
                feature_names.append("pred_magnitude_zscore")
                feature_names.append("pred_cross_horizon_consistency")
                feature_names.append("pred_10s_1s_ratio")
                for h in HORIZON_NAMES:
                    feature_names.append(f"patchtst_pred_{h}")
                feature_names.append("model_agreement_10s")
                feature_names.append("ensemble_mean_10s")
                for idx in EXEC_FEAT_INDICES:
                    feature_names.append(f"exec_{EXEC_FEATURE_NAMES[idx]}")
                interaction_names = [
                    "pred_x_spread", "pred_x_vol", "pred_x_fill_prob",
                    "pred_x_depth_imbalance", "pred_abs_x_spread",
                    "signal_to_cost_ratio", "pred_abs_x_consistency",
                    "spread_regime", "vol_regime",
                ]
                feature_names.extend(interaction_names)

            log.info(
                f"  {date}: X={X.shape}, y_mean={y.mean():.3f} "
                f"(profitable={y.sum():.0f}/{len(y)})"
            )

        log.info(f"Built data for {len(all_date_data)} dates, {sum(d[0].shape[0] for d in all_date_data.values()):,d} total samples")
        log.info(f"Feature count: {len(feature_names)}")

        if len(all_date_data) < 2:
            log.error("Not enough dates for walk-forward. Need at least 2.")
            continue

        # Step 4: Walk-forward training
        results = run_walk_forward(
            all_date_data, feature_names,
            workers=args.workers, cost_type=cost_type,
        )

        if not results:
            log.error("No valid fold results!")
            continue

        # Step 5: Generate report
        report = generate_report(results, feature_names, cost_type)

        # Save report
        report_path = OUTPUT_DIR / f"report_{cost_type}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, cls=NumpyEncoder)
        log.info(f"Report saved to {report_path}")

        # Save predictions
        all_preds = np.concatenate([r.predictions for r in results])
        all_labels = np.concatenate([r.labels for r in results])
        all_dates_arr = np.concatenate([
            np.full(r.n_test, r.date) for r in results
        ])
        np.savez_compressed(
            OUTPUT_DIR / f"predictions_{cost_type}.npz",
            predictions=all_preds,
            labels=all_labels,
            dates=all_dates_arr,
        )

        # Print summary
        m = report["metrics"]
        log.info("\n" + "=" * 60)
        log.info(f"RESULTS SUMMARY — {cost_type} order cost")
        log.info("=" * 60)
        log.info(f"  Folds:            {report['n_folds']}")
        log.info(f"  Date range:       {report['date_range']}")
        log.info(f"  Total samples:    {report['total_samples']:,d}")
        log.info(f"  Base rate:        {m['base_rate_mean']:.3f}")
        log.info(f"  AUC:              {m['auc_mean']:.4f} ± {m['auc_std']:.4f}")
        log.info(f"  Avg Precision:    {m['avg_precision_mean']:.4f} ± {m['avg_precision_std']:.4f}")
        log.info(f"  Precision@5%:     {m['precision_at_5pct_mean']:.3f} ± {m['precision_at_5pct_std']:.3f}")
        log.info(f"  Precision@10%:    {m['precision_at_10pct_mean']:.3f} ± {m['precision_at_10pct_std']:.3f}")
        log.info(f"  Brier score:      {m['brier_mean']:.4f}")

        log.info("\nRegime Analysis:")
        for regime, data in sorted(report["regime_analysis"].items()):
            log.info(
                f"  {regime:20s}: AUC={data['mean_auc']:.4f}±{data['std_auc']:.4f} | "
                f"base_rate={data['mean_base_rate']:.3f} | n={data['n_total']:,d}"
            )

        log.info("\nTop 10 Features:")
        for feat in report["top_20_features"][:10]:
            log.info(f"  {feat['name']:40s}: gain={feat['mean_gain']:.1f}")

    elapsed = time.time() - t0
    log.info(f"\nTotal elapsed: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    log.info("Done.")


if __name__ == "__main__":
    main()
