"""
Hybrid Ensemble — Combine BookSpatialCNN, EventTransformer, and LightGBM predictions.

Five combination approaches, all evaluated with walk-forward (no lookahead):

  1. Optimized Weighted Average — weights optimized on rolling training window
  2. Stacking Meta-Learner — Ridge/Lasso on [cnn_pred, et_pred, lgbm_pred]
  3. Gated Mixture of Experts — neural net learns when each model is strongest
  4. Confidence-Weighted Blend — weight by rolling IC (last K folds)
  5. Feature Concatenation — penultimate-layer features into LightGBM (if available)

Walk-forward protocol matches existing 94-fold structure:
  - For each test date, use only predictions from PRIOR dates for meta-training
  - No future information leakage

Prediction files:
  - BookSpatialCNN:    deep_models/results/oos_predictions_book_20260303_234725.npz
  - EventTransformer:  deep_models/results/oos_predictions_event_20260228_123832.npz
  - LightGBM:          Re-generated inline from MBO feature cache (walk-forward)

Usage:
  # Full run (all 5 approaches)
  python alpha_discovery/hybrid_ensemble.py

  # Quick test (first 20 folds only)
  python alpha_discovery/hybrid_ensemble.py --max-folds 20

  # Skip LightGBM regeneration (2-model ensemble only)
  python alpha_discovery/hybrid_ensemble.py --skip-lgbm

  # Custom lookback for confidence-weighted
  python alpha_discovery/hybrid_ensemble.py --lookback 10
"""

import argparse
import gc
import json
import logging
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr, rankdata
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
LVL3_ROOT = Path(__file__).parent.parent
ALPHA_DIR = Path(__file__).parent

BOOK_PREDS_FILE = ALPHA_DIR / "deep_models" / "results" / "oos_predictions_book_20260303_234725.npz"
EVENT_PREDS_FILE = ALPHA_DIR / "deep_models" / "results" / "oos_predictions_event_20260228_123832.npz"

if platform.system() == "Windows":
    FEAT_CACHE = LVL3_ROOT / "data" / "processed" / "mbo_features_cache"
else:
    FEAT_CACHE = Path.home() / "lvl3quant" / "data" / "processed" / "mbo_features_cache"

RESULTS_DIR = ALPHA_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TICK = 0.25
TICK_VAL = 12.50

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
_log_file = RESULTS_DIR / f"hybrid_ensemble_{_ts}.log"
_root = logging.getLogger()
_root.setLevel(logging.INFO)
for _h in _root.handlers[:]:
    _root.removeHandler(_h)
_fmt = logging.Formatter("%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S")
_fh = logging.FileHandler(str(_log_file), mode="w")
_fh.setFormatter(_fmt)
_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(_fmt)
_root.addHandler(_fh)
_root.addHandler(_sh)
logger = logging.getLogger("hybrid")


# ============================================================================
# Data Loading
# ============================================================================

def load_deep_predictions(npz_path: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Load per-date OOS predictions from deep model NPZ file.

    Returns:
        {date_str: (predictions, targets)} sorted by date
    """
    data = np.load(str(npz_path))
    keys = sorted(data.keys())
    dates = sorted(set(k.rsplit("_", 1)[0] for k in keys))

    result = {}
    for date in dates:
        preds = data[f"{date}_preds"]
        targets = data[f"{date}_targets"]
        result[date] = (preds.astype(np.float32), targets.astype(np.float32))

    return result


def generate_lgbm_predictions(dates: List[str]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Re-generate LightGBM walk-forward predictions for specified dates.

    Matches the methodology in trade_test_walkforward.py:
      - Expanding window (min 5 days, max 30 days)
      - 100-bar (10s) horizon forward return as target
      - Subsample training by 5x
      - Predict ALL bars on test day

    Returns:
        {date_str: (predictions, targets)}
    """
    try:
        import lightgbm as lgb
    except (ImportError, AttributeError, OSError) as e:
        logger.error(f"lightgbm not available: {e}")
        logger.error("Run on Jupiter/Saturn where lightgbm works, or fix numpy/dask compat")
        return {}

    logger.info("Generating LightGBM walk-forward predictions...")

    # Load feature data for all available dates
    # We need dates BEFORE the first test date for training
    all_cache_dates = sorted(
        f.stem.replace("_mbo_features", "")
        for f in FEAT_CACHE.glob("*_mbo_features.npz")
    )
    logger.info(f"  Feature cache has {len(all_cache_dates)} dates")

    # Load features lazily
    day_features = {}
    day_targets = {}

    needed_dates = sorted(set(all_cache_dates) & (set(dates) | set(all_cache_dates)))
    for date_str in needed_dates:
        fpath = FEAT_CACHE / f"{date_str}_mbo_features.npz"
        if not fpath.exists():
            continue
        try:
            feats = np.load(str(fpath))["mbo_features"]
            mid = feats[:, 0].astype(np.float32)

            # 100-bar forward return in ticks (same target as deep models)
            horizon = 100
            n = len(mid)
            fwd_ret = np.full(n, np.nan, dtype=np.float32)
            fwd_ret[: n - horizon] = (mid[horizon:] - mid[: n - horizon]) / TICK

            day_features[date_str] = feats.copy()
            day_targets[date_str] = fwd_ret
            del feats
        except Exception as e:
            logger.warning(f"  Skip {date_str}: {e}")

    logger.info(f"  Loaded features for {len(day_features)} dates")

    # LightGBM params (matching arch_benchmark / trade_test_walkforward)
    lgbm_params = {
        "objective": "regression",
        "metric": "mse",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 6,
        "min_child_samples": 200,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
    }

    min_train_days = 5
    max_train_days = 30
    subsample_step = 5

    sorted_feat_dates = sorted(day_features.keys())
    result = {}

    for test_idx in range(min_train_days, len(sorted_feat_dates)):
        test_date = sorted_feat_dates[test_idx]
        if test_date not in dates:
            continue  # Only generate for requested dates

        # Training window
        train_start = max(0, test_idx - max_train_days)
        train_dates = sorted_feat_dates[train_start:test_idx]

        # Build training set (subsampled)
        X_trains, y_trains = [], []
        for td in train_dates:
            if td not in day_features:
                continue
            feats = day_features[td]
            tgt = day_targets[td]
            n = len(feats)
            indices = np.arange(n)
            mask = (indices % subsample_step == 0) & np.isfinite(tgt) & (indices >= 5000)
            if mask.sum() > 0:
                X_trains.append(feats[mask])
                y_trains.append(tgt[mask])

        if not X_trains:
            continue

        X_train = np.vstack(X_trains)
        y_train = np.concatenate(y_trains)

        # Remove NaN/inf
        finite = np.all(np.isfinite(X_train), axis=1) & np.isfinite(y_train)
        X_train = X_train[finite]
        y_train = y_train[finite]

        if len(X_train) < 1000:
            continue

        # Cap training size
        if len(X_train) > 2_000_000:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X_train), 2_000_000, replace=False)
            X_train = X_train[idx]
            y_train = y_train[idx]

        # Normalize target
        y_mean = np.mean(y_train)
        y_std = np.std(y_train)
        if y_std < 1e-10:
            continue
        y_train_norm = (y_train - y_mean) / y_std

        try:
            train_ds = lgb.Dataset(X_train, label=y_train_norm, free_raw_data=True)
            model = lgb.train(
                lgbm_params, train_ds, num_boost_round=200,
                valid_sets=[train_ds], callbacks=[lgb.log_evaluation(0)],
            )

            # Predict on full test day
            test_feats = day_features[test_date]
            valid_mask = np.all(np.isfinite(test_feats), axis=1)
            predictions = np.zeros(len(test_feats), dtype=np.float32)
            if np.any(valid_mask):
                preds_raw = model.predict(test_feats[valid_mask])
                predictions[valid_mask] = (preds_raw * y_std + y_mean).astype(np.float32)

            # Target for this day
            targets = day_targets[test_date]

            result[test_date] = (predictions, targets)

            n_done = len(result)
            if n_done % 10 == 0 or n_done <= 3:
                # Quick IC check
                sub_mask = np.isfinite(predictions) & np.isfinite(targets) & (predictions != 0)
                if sub_mask.sum() > 100:
                    ic = spearmanr(predictions[sub_mask], targets[sub_mask])[0]
                else:
                    ic = 0.0
                logger.info(f"  LightGBM [{n_done}] {test_date} IC={ic:+.4f} train={len(X_train):,}")

            del model, train_ds, X_train, y_train

        except Exception as e:
            logger.warning(f"  LightGBM {test_date}: ERROR: {e}")
            continue

        gc.collect()

    logger.info(f"  LightGBM: generated predictions for {len(result)} dates")
    return result


def align_predictions(
    book_preds: Dict[str, Tuple[np.ndarray, np.ndarray]],
    event_preds: Dict[str, Tuple[np.ndarray, np.ndarray]],
    lgbm_preds: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
) -> Tuple[List[str], Dict[str, Dict[str, np.ndarray]], Dict[str, np.ndarray]]:
    """Align predictions across models by truncating to minimum length per date.

    The models have slightly different sample counts per date:
      - BookSpatialCNN: ~233880 (uses windowed input, loses some bars)
      - EventTransformer: ~233899
      - LightGBM: ~234000 (full day)

    We truncate all to the minimum length and z-score normalize per day.

    Returns:
        common_dates: sorted list of dates with all models present
        aligned_preds: {model_name: {date: z-scored_predictions}}
        aligned_targets: {date: targets}  (from book model, most conservative)
    """
    # Find common dates
    common = set(book_preds.keys()) & set(event_preds.keys())
    if lgbm_preds:
        common = common & set(lgbm_preds.keys())
    common_dates = sorted(common)

    if not common_dates:
        raise ValueError("No common dates found across models!")

    logger.info(f"Common dates across all models: {len(common_dates)}")

    aligned_preds = {"book_cnn": {}, "event_tf": {}}
    if lgbm_preds:
        aligned_preds["lgbm"] = {}
    aligned_targets = {}

    for date in common_dates:
        bp, bt = book_preds[date]
        ep, et = event_preds[date]

        # Minimum length across models for this date
        min_len = min(len(bp), len(ep))
        if lgbm_preds and date in lgbm_preds:
            lp, lt = lgbm_preds[date]
            min_len = min(min_len, len(lp))

        # Truncate and z-score normalize predictions per day
        def zscore(arr):
            arr = arr[:min_len].copy()
            std = np.nanstd(arr)
            if std > 1e-10:
                return ((arr - np.nanmean(arr)) / std).astype(np.float32)
            return np.zeros(min_len, dtype=np.float32)

        aligned_preds["book_cnn"][date] = zscore(bp)
        aligned_preds["event_tf"][date] = zscore(ep)
        if lgbm_preds and date in lgbm_preds:
            lp, _ = lgbm_preds[date]
            aligned_preds["lgbm"][date] = zscore(lp)

        # Use book targets (most conservative length)
        aligned_targets[date] = bt[:min_len].astype(np.float32)

    return common_dates, aligned_preds, aligned_targets


# ============================================================================
# IC Computation Helpers
# ============================================================================

def compute_ic(preds: np.ndarray, targets: np.ndarray) -> float:
    """Spearman IC between predictions and targets, handling NaN."""
    valid = np.isfinite(preds) & np.isfinite(targets)
    if valid.sum() < 50:
        return 0.0
    return float(spearmanr(preds[valid], targets[valid])[0])


def ic_stats(ics: List[float]) -> Dict:
    """Compute IC, ICIR, t-stat, pct positive from list of per-fold ICs."""
    if not ics:
        return {"ic": 0.0, "icir": 0.0, "tstat": 0.0, "pct_pos": 0.0, "n": 0}
    arr = np.array(ics)
    m = float(arr.mean())
    s = float(arr.std())
    n = len(arr)
    return {
        "ic": m,
        "icir": m / s if s > 1e-10 else 0.0,
        "tstat": m / s * np.sqrt(n) if s > 1e-10 else 0.0,
        "pct_pos": float(100 * np.mean(arr > 0)),
        "n": n,
    }


# ============================================================================
# Approach 1: Optimized Weighted Average
# ============================================================================

def approach_1_weighted_avg(
    dates: List[str],
    preds: Dict[str, Dict[str, np.ndarray]],
    targets: Dict[str, np.ndarray],
    lookback: int = 10,
) -> Tuple[List[float], str]:
    """Optimize weights [w_cnn, w_et, w_lgbm] on rolling window, predict next fold.

    For each test date:
      - Use the last `lookback` dates as optimization window
      - Find weights that maximize mean IC on that window
      - Apply those weights to the test date
    """
    logger.info("=" * 60)
    logger.info("APPROACH 1: Optimized Weighted Average")
    logger.info(f"  Lookback window: {lookback} folds")

    model_names = sorted(preds.keys())
    n_models = len(model_names)
    fold_ics = []

    for i, test_date in enumerate(dates):
        if i < lookback:
            # Not enough history -- use equal weights
            weights = np.ones(n_models) / n_models
        else:
            # Optimize weights on lookback window
            window_dates = dates[i - lookback: i]

            def neg_mean_ic(w):
                w = np.abs(w)
                w = w / (w.sum() + 1e-10)
                ics_window = []
                for wd in window_dates:
                    combined = np.zeros_like(targets[wd])
                    for j, mn in enumerate(model_names):
                        combined += w[j] * preds[mn][wd]
                    ic_val = compute_ic(combined, targets[wd])
                    ics_window.append(ic_val)
                return -np.mean(ics_window)

            # Start from equal weights
            w0 = np.ones(n_models) / n_models
            result = minimize(neg_mean_ic, w0, method="Nelder-Mead",
                              options={"maxiter": 200, "xatol": 0.01})
            weights = np.abs(result.x)
            weights = weights / (weights.sum() + 1e-10)

        # Apply weights to test date
        combined = np.zeros_like(targets[test_date])
        for j, mn in enumerate(model_names):
            combined += weights[j] * preds[mn][test_date]

        ic = compute_ic(combined, targets[test_date])
        fold_ics.append(ic)

        if (i + 1) % 20 == 0 or i == len(dates) - 1:
            w_str = " ".join(f"{mn}={weights[j]:.3f}" for j, mn in enumerate(model_names))
            logger.info(f"  Fold {i+1}/{len(dates)}: IC={ic:+.4f}  weights=[{w_str}]")

    stats = ic_stats(fold_ics)
    summary = (f"  Optimized Weighted Avg: IC={stats['ic']:+.4f} "
               f"ICIR={stats['icir']:.2f} t={stats['tstat']:.2f} "
               f"pct_pos={stats['pct_pos']:.1f}%")
    logger.info(summary)
    return fold_ics, "optimized_weighted_avg"


# ============================================================================
# Approach 2: Stacking Meta-Learner
# ============================================================================

def approach_2_stacking(
    dates: List[str],
    preds: Dict[str, Dict[str, np.ndarray]],
    targets: Dict[str, np.ndarray],
    min_train_folds: int = 5,
    subsample: int = 10,
) -> Tuple[List[float], str]:
    """Train Ridge meta-learner on [cnn_pred, et_pred, lgbm_pred] features.

    Walk-forward: train on first N folds, predict fold N+1.
    Subsamples training data to reduce autocorrelation.
    """
    from sklearn.linear_model import Ridge, Lasso

    logger.info("=" * 60)
    logger.info("APPROACH 2: Stacking Meta-Learner (Ridge)")
    logger.info(f"  Min train folds: {min_train_folds}, subsample: 1/{subsample}")

    model_names = sorted(preds.keys())
    fold_ics = []

    for i, test_date in enumerate(dates):
        if i < min_train_folds:
            # Not enough history -- fallback to equal weight
            combined = np.zeros_like(targets[test_date])
            for mn in model_names:
                combined += preds[mn][test_date] / len(model_names)
            ic = compute_ic(combined, targets[test_date])
            fold_ics.append(ic)
            continue

        # Build meta-training set from all prior folds
        X_meta_list, y_meta_list = [], []
        for prior_date in dates[:i]:
            # Stack model predictions as features
            n_samples = len(targets[prior_date])
            x_row = np.column_stack([preds[mn][prior_date] for mn in model_names])
            y_row = targets[prior_date]

            # Subsample to reduce autocorrelation
            indices = np.arange(n_samples)
            mask = (indices % subsample == 0) & np.all(np.isfinite(x_row), axis=1) & np.isfinite(y_row)
            if mask.sum() > 0:
                X_meta_list.append(x_row[mask])
                y_meta_list.append(y_row[mask])

        X_meta = np.vstack(X_meta_list)
        y_meta = np.concatenate(y_meta_list)

        # Also add squared predictions and interaction terms for richer meta-features
        X_sq = X_meta ** 2
        # Pairwise products
        X_interact = []
        for j in range(len(model_names)):
            for k in range(j + 1, len(model_names)):
                X_interact.append(X_meta[:, j] * X_meta[:, k])
        if X_interact:
            X_interact = np.column_stack(X_interact)
            X_meta_full = np.hstack([X_meta, X_sq, X_interact])
        else:
            X_meta_full = np.hstack([X_meta, X_sq])

        # Train Ridge
        try:
            meta_model = Ridge(alpha=10.0)  # Strong regularization to avoid overfit
            meta_model.fit(X_meta_full, y_meta)

            # Predict on test date
            x_test = np.column_stack([preds[mn][test_date] for mn in model_names])
            x_test_sq = x_test ** 2
            x_test_interact = []
            for j in range(len(model_names)):
                for k in range(j + 1, len(model_names)):
                    x_test_interact.append(x_test[:, j] * x_test[:, k])
            if x_test_interact:
                x_test_interact = np.column_stack(x_test_interact)
                x_test_full = np.hstack([x_test, x_test_sq, x_test_interact])
            else:
                x_test_full = np.hstack([x_test, x_test_sq])

            valid_mask = np.all(np.isfinite(x_test_full), axis=1)
            combined = np.zeros(len(targets[test_date]), dtype=np.float32)
            if valid_mask.sum() > 0:
                combined[valid_mask] = meta_model.predict(x_test_full[valid_mask]).astype(np.float32)

            ic = compute_ic(combined, targets[test_date])
            fold_ics.append(ic)

            if (i + 1) % 20 == 0 or i == len(dates) - 1:
                coefs = meta_model.coef_[:len(model_names)]
                c_str = " ".join(f"{mn}={coefs[j]:+.3f}" for j, mn in enumerate(model_names))
                logger.info(f"  Fold {i+1}/{len(dates)}: IC={ic:+.4f}  linear_coefs=[{c_str}]")

        except Exception as e:
            logger.warning(f"  Fold {i+1}: Ridge failed: {e}")
            # Fallback to equal weight
            combined = np.zeros_like(targets[test_date])
            for mn in model_names:
                combined += preds[mn][test_date] / len(model_names)
            ic = compute_ic(combined, targets[test_date])
            fold_ics.append(ic)

    stats = ic_stats(fold_ics)
    summary = (f"  Stacking (Ridge): IC={stats['ic']:+.4f} "
               f"ICIR={stats['icir']:.2f} t={stats['tstat']:.2f} "
               f"pct_pos={stats['pct_pos']:.1f}%")
    logger.info(summary)
    return fold_ics, "stacking_ridge"


# ============================================================================
# Approach 3: Gated Mixture of Experts
# ============================================================================

def approach_3_gated_moe(
    dates: List[str],
    preds: Dict[str, Dict[str, np.ndarray]],
    targets: Dict[str, np.ndarray],
    min_train_folds: int = 10,
    subsample: int = 20,
) -> Tuple[List[float], str]:
    """Small neural network that learns gating weights from regime features.

    Regime features (computed per-sample from the predictions themselves):
      - Absolute prediction magnitude from each model (confidence proxy)
      - Agreement between models (pairwise correlation in local window)
      - Prediction variance across models

    The gating network outputs softmax weights that sum to 1.
    Final prediction = sum(gate_i * pred_i).

    Uses PyTorch for the small gating network, trained on CPU.
    """
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        logger.error("PyTorch not available, skipping Gated MoE")
        return [], "gated_moe"

    logger.info("=" * 60)
    logger.info("APPROACH 3: Gated Mixture of Experts")
    logger.info(f"  Min train folds: {min_train_folds}, subsample: 1/{subsample}")

    model_names = sorted(preds.keys())
    n_models = len(model_names)

    class GatingNetwork(nn.Module):
        """Small MLP that produces softmax gating weights."""
        def __init__(self, n_features: int, n_experts: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_features, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, n_experts),
            )

        def forward(self, x):
            return torch.softmax(self.net(x), dim=-1)

    def build_regime_features(date: str) -> np.ndarray:
        """Build regime/context features for the gating network.

        Features per sample:
          - |pred_i| for each model (confidence proxy)
          - pred_i * pred_j agreement for each pair
          - std(preds) across models
          - mean(|preds|) across models
        """
        model_preds = [preds[mn][date] for mn in model_names]
        n = len(model_preds[0])

        feats = []
        # Absolute predictions (confidence)
        for p in model_preds:
            feats.append(np.abs(p).reshape(-1, 1))

        # Pairwise agreement (product of predictions)
        for j in range(n_models):
            for k in range(j + 1, n_models):
                feats.append((model_preds[j] * model_preds[k]).reshape(-1, 1))

        # Cross-model stats
        stacked = np.column_stack(model_preds)
        feats.append(np.std(stacked, axis=1).reshape(-1, 1))
        feats.append(np.mean(np.abs(stacked), axis=1).reshape(-1, 1))

        return np.hstack(feats).astype(np.float32)

    n_regime_features = n_models + (n_models * (n_models - 1)) // 2 + 2
    fold_ics = []

    for i, test_date in enumerate(dates):
        if i < min_train_folds:
            # Fallback to equal weight
            combined = np.zeros_like(targets[test_date])
            for mn in model_names:
                combined += preds[mn][test_date] / n_models
            fold_ics.append(compute_ic(combined, targets[test_date]))
            continue

        # Build training data from prior folds
        X_gate_list, X_pred_list, y_list = [], [], []
        for prior_date in dates[max(0, i - 30): i]:  # Use last 30 folds max
            regime_feats = build_regime_features(prior_date)
            model_pred_stack = np.column_stack([preds[mn][prior_date] for mn in model_names])
            tgt = targets[prior_date]

            n = len(tgt)
            indices = np.arange(n)
            mask = (indices % subsample == 0) & np.isfinite(tgt)
            mask &= np.all(np.isfinite(regime_feats), axis=1)
            mask &= np.all(np.isfinite(model_pred_stack), axis=1)

            if mask.sum() > 0:
                X_gate_list.append(regime_feats[mask])
                X_pred_list.append(model_pred_stack[mask])
                y_list.append(tgt[mask])

        if not X_gate_list:
            fold_ics.append(0.0)
            continue

        X_gate = np.vstack(X_gate_list)
        X_pred = np.vstack(X_pred_list)
        y = np.concatenate(y_list)

        # Normalize gate features
        gate_mean = X_gate.mean(axis=0)
        gate_std = X_gate.std(axis=0) + 1e-8
        X_gate_norm = (X_gate - gate_mean) / gate_std

        # Train gating network
        gate_net = GatingNetwork(n_regime_features, n_models)
        optimizer_g = optim.Adam(gate_net.parameters(), lr=1e-3, weight_decay=1e-4)

        X_gate_t = torch.from_numpy(X_gate_norm)
        X_pred_t = torch.from_numpy(X_pred)
        y_t = torch.from_numpy(y)

        gate_net.train()
        batch_size = min(2048, len(y))
        n_epochs = 20

        for epoch in range(n_epochs):
            perm = torch.randperm(len(y))
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, len(y), batch_size):
                idx = perm[start: start + batch_size]
                gates = gate_net(X_gate_t[idx])  # (B, n_models)
                expert_preds = X_pred_t[idx]  # (B, n_models)
                combined_pred = (gates * expert_preds).sum(dim=1)
                loss = torch.nn.functional.mse_loss(combined_pred, y_t[idx])
                optimizer_g.zero_grad()
                loss.backward()
                optimizer_g.step()
                epoch_loss += loss.item()
                n_batches += 1

        # Predict on test date
        gate_net.eval()
        test_regime = build_regime_features(test_date)
        test_regime_norm = (test_regime - gate_mean) / gate_std
        test_model_preds = np.column_stack([preds[mn][test_date] for mn in model_names])

        valid_mask = (np.all(np.isfinite(test_regime_norm), axis=1) &
                      np.all(np.isfinite(test_model_preds), axis=1))

        combined = np.zeros(len(targets[test_date]), dtype=np.float32)
        if valid_mask.sum() > 0:
            with torch.no_grad():
                gates_test = gate_net(torch.from_numpy(test_regime_norm[valid_mask]))
                expert_test = torch.from_numpy(test_model_preds[valid_mask])
                combined_valid = (gates_test * expert_test).sum(dim=1).numpy()
            combined[valid_mask] = combined_valid.astype(np.float32)

        ic = compute_ic(combined, targets[test_date])
        fold_ics.append(ic)

        if (i + 1) % 20 == 0 or i == len(dates) - 1:
            # Report average gate weights on test
            if valid_mask.sum() > 0:
                with torch.no_grad():
                    avg_gates = gate_net(torch.from_numpy(test_regime_norm[valid_mask])).mean(dim=0).numpy()
                g_str = " ".join(f"{mn}={avg_gates[j]:.3f}" for j, mn in enumerate(model_names))
            else:
                g_str = "N/A"
            logger.info(f"  Fold {i+1}/{len(dates)}: IC={ic:+.4f}  avg_gates=[{g_str}]")

        del gate_net, optimizer_g
        gc.collect()

    stats = ic_stats(fold_ics)
    summary = (f"  Gated MoE: IC={stats['ic']:+.4f} "
               f"ICIR={stats['icir']:.2f} t={stats['tstat']:.2f} "
               f"pct_pos={stats['pct_pos']:.1f}%")
    logger.info(summary)
    return fold_ics, "gated_moe"


# ============================================================================
# Approach 4: Confidence-Weighted Blend
# ============================================================================

def approach_4_confidence_weighted(
    dates: List[str],
    preds: Dict[str, Dict[str, np.ndarray]],
    targets: Dict[str, np.ndarray],
    lookback: int = 5,
) -> Tuple[List[float], str]:
    """Weight each model by its rolling IC over the last K folds.

    Models with negative recent IC get zero weight.
    Simple, no extra parameters to overfit.
    """
    logger.info("=" * 60)
    logger.info("APPROACH 4: Confidence-Weighted Blend")
    logger.info(f"  Lookback: {lookback} folds")

    model_names = sorted(preds.keys())

    # Pre-compute per-fold IC for each model
    per_fold_ic = {mn: [] for mn in model_names}
    for date in dates:
        for mn in model_names:
            ic = compute_ic(preds[mn][date], targets[date])
            per_fold_ic[mn].append(ic)

    fold_ics = []

    for i, test_date in enumerate(dates):
        if i < lookback:
            # Not enough history -- equal weight
            weights = {mn: 1.0 / len(model_names) for mn in model_names}
        else:
            # Weight by mean IC over last K folds (zero out negatives)
            weights = {}
            for mn in model_names:
                recent = per_fold_ic[mn][i - lookback: i]
                w = max(0.0, np.mean(recent))
                weights[mn] = w

            total = sum(weights.values())
            if total < 1e-10:
                weights = {mn: 1.0 / len(model_names) for mn in model_names}
            else:
                weights = {mn: w / total for mn, w in weights.items()}

        # Combine
        combined = np.zeros_like(targets[test_date])
        for mn in model_names:
            combined += weights[mn] * preds[mn][test_date]

        ic = compute_ic(combined, targets[test_date])
        fold_ics.append(ic)

        if (i + 1) % 20 == 0 or i == len(dates) - 1:
            w_str = " ".join(f"{mn}={weights[mn]:.3f}" for mn in model_names)
            logger.info(f"  Fold {i+1}/{len(dates)}: IC={ic:+.4f}  weights=[{w_str}]")

    stats = ic_stats(fold_ics)
    summary = (f"  Confidence-Weighted: IC={stats['ic']:+.4f} "
               f"ICIR={stats['icir']:.2f} t={stats['tstat']:.2f} "
               f"pct_pos={stats['pct_pos']:.1f}%")
    logger.info(summary)
    return fold_ics, "confidence_weighted"


# ============================================================================
# Approach 5: Feature Concatenation
# ============================================================================

def approach_5_feature_concat(
    dates: List[str],
    preds: Dict[str, Dict[str, np.ndarray]],
    targets: Dict[str, np.ndarray],
) -> Tuple[List[float], str]:
    """Extract penultimate-layer features from CNN and Transformer, feed into LightGBM.

    NOTE: This requires saved intermediate representations (penultimate layer activations)
    from the CNN and Transformer models. These are NOT currently saved in the existing
    NPZ files -- the files only contain final scalar predictions.

    As a practical alternative, we construct DERIVED features from the predictions:
      - Raw predictions from each model
      - Squared predictions (non-linear transform)
      - Pairwise products (interaction features)
      - Rolling statistics (local mean/std of predictions in 100-bar windows)
      - Prediction ranks (ordinal transform)

    This is then fed into a LightGBM meta-model with walk-forward validation.
    """
    try:
        import lightgbm as lgb
        _USE_LGBM_META = True
    except (ImportError, AttributeError, OSError):
        _USE_LGBM_META = False

    if not _USE_LGBM_META:
        # Fallback: use sklearn GradientBoosting instead of LightGBM
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            logger.info("  LightGBM unavailable, using sklearn GradientBoosting fallback")
        except ImportError:
            logger.error("Neither lightgbm nor sklearn available, skipping Feature Concat")
            return [], "feature_concat"

    logger.info("=" * 60)
    logger.info("APPROACH 5: Feature Concatenation (Derived Features -> Meta-Model)")
    logger.info("  NOTE: Penultimate-layer features not available in saved NPZ files.")
    logger.info("  Using derived features from scalar predictions instead.")

    model_names = sorted(preds.keys())
    n_models = len(model_names)
    min_train_folds = 5
    subsample = 10 if _USE_LGBM_META else 50  # Higher subsample for slower sklearn

    def build_meta_features(date: str) -> np.ndarray:
        """Build rich feature set from model predictions."""
        raw = [preds[mn][date] for mn in model_names]
        n = len(raw[0])
        feats = []

        # Raw predictions
        for p in raw:
            feats.append(p.reshape(-1, 1))

        # Squared (captures non-linear magnitude)
        for p in raw:
            feats.append((p ** 2).reshape(-1, 1))

        # Pairwise products (interaction)
        for j in range(n_models):
            for k in range(j + 1, n_models):
                feats.append((raw[j] * raw[k]).reshape(-1, 1))

        # Cross-model stats
        stacked = np.column_stack(raw)
        feats.append(np.std(stacked, axis=1).reshape(-1, 1))
        feats.append(np.mean(np.abs(stacked), axis=1).reshape(-1, 1))
        feats.append(np.max(stacked, axis=1).reshape(-1, 1))
        feats.append(np.min(stacked, axis=1).reshape(-1, 1))

        # Rolling features (100-bar window ~ 10 seconds)
        window = 100
        for p in raw:
            # Pad for rolling
            padded = np.pad(p, (window, 0), mode="edge")
            # Cumsum trick for rolling mean
            cs = np.cumsum(padded)
            roll_mean = (cs[window:] - cs[:-window]) / window
            feats.append(roll_mean[:n].reshape(-1, 1))

            # Rolling std (approximate via E[X^2] - E[X]^2)
            cs2 = np.cumsum(padded ** 2)
            roll_sq_mean = (cs2[window:] - cs2[:-window]) / window
            roll_var = np.maximum(roll_sq_mean[:n] - roll_mean[:n] ** 2, 0)
            feats.append(np.sqrt(roll_var).reshape(-1, 1))

        # Rank transform (within day)
        for p in raw:
            valid = np.isfinite(p)
            ranks = np.full(n, 0.5, dtype=np.float32)
            if valid.sum() > 0:
                ranks[valid] = rankdata(p[valid]).astype(np.float32) / valid.sum()
            feats.append(ranks.reshape(-1, 1))

        return np.hstack(feats).astype(np.float32)

    # LightGBM meta-model params (lighter than base model)
    if _USE_LGBM_META:
        lgbm_params = {
            "objective": "regression",
            "metric": "mse",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 4,
            "min_child_samples": 500,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "reg_alpha": 0.5,
            "reg_lambda": 5.0,
            "verbose": -1,
            "n_jobs": -1,
            "seed": 42,
        }

    fold_ics = []

    for i, test_date in enumerate(dates):
        if i < min_train_folds:
            # Fallback
            combined = np.zeros_like(targets[test_date])
            for mn in model_names:
                combined += preds[mn][test_date] / n_models
            fold_ics.append(compute_ic(combined, targets[test_date]))
            continue

        # Build meta-training set
        X_list, y_list = [], []
        for prior_date in dates[max(0, i - 30): i]:  # Last 30 folds max
            meta_feats = build_meta_features(prior_date)
            tgt = targets[prior_date]
            n = len(tgt)
            indices = np.arange(n)
            mask = (indices % subsample == 0) & np.isfinite(tgt)
            mask &= np.all(np.isfinite(meta_feats), axis=1)
            if mask.sum() > 0:
                X_list.append(meta_feats[mask])
                y_list.append(tgt[mask])

        if not X_list:
            fold_ics.append(0.0)
            continue

        X_train = np.vstack(X_list)
        y_train = np.concatenate(y_list)

        # Normalize target
        y_mean = np.mean(y_train)
        y_std = np.std(y_train)
        if y_std < 1e-10:
            fold_ics.append(0.0)
            continue
        y_norm = (y_train - y_mean) / y_std

        try:
            if _USE_LGBM_META:
                train_ds = lgb.Dataset(X_train, label=y_norm, free_raw_data=True)
                model = lgb.train(
                    lgbm_params, train_ds, num_boost_round=100,
                    valid_sets=[train_ds], callbacks=[lgb.log_evaluation(0)],
                )
            else:
                # Cap training size for sklearn (much slower than LightGBM)
                max_sklearn = 200_000
                if len(X_train) > max_sklearn:
                    rng = np.random.RandomState(42)
                    idx = rng.choice(len(X_train), max_sklearn, replace=False)
                    X_fit, y_fit = X_train[idx], y_norm[idx]
                else:
                    X_fit, y_fit = X_train, y_norm
                model = GradientBoostingRegressor(
                    n_estimators=100, max_depth=4, learning_rate=0.05,
                    subsample=0.7, min_samples_leaf=200, random_state=42,
                )
                model.fit(X_fit, y_fit)

            # Predict on test
            test_feats = build_meta_features(test_date)
            valid_mask = np.all(np.isfinite(test_feats), axis=1)
            combined = np.zeros(len(targets[test_date]), dtype=np.float32)
            if valid_mask.sum() > 0:
                raw_pred = model.predict(test_feats[valid_mask])
                combined[valid_mask] = (raw_pred * y_std + y_mean).astype(np.float32)

            ic = compute_ic(combined, targets[test_date])
            fold_ics.append(ic)

            if (i + 1) % 20 == 0 or i == len(dates) - 1:
                if _USE_LGBM_META:
                    imp = model.feature_importance(importance_type="gain")
                    top_idx = np.argsort(imp)[-3:][::-1]
                    extra = f"top_feats={list(top_idx)}"
                else:
                    imp = model.feature_importances_
                    top_idx = np.argsort(imp)[-3:][::-1]
                    extra = f"top_feats={list(top_idx)}"
                logger.info(f"  Fold {i+1}/{len(dates)}: IC={ic:+.4f}  "
                            f"{extra}  train={len(X_train):,}")

            if _USE_LGBM_META:
                del model, train_ds
            else:
                del model
        except Exception as e:
            logger.warning(f"  Fold {i+1}: Meta-model failed: {e}")
            fold_ics.append(0.0)

        gc.collect()

    stats = ic_stats(fold_ics)
    meta_label = "LightGBM" if _USE_LGBM_META else "sklearn GB"
    summary = (f"  Feature Concat ({meta_label} meta): IC={stats['ic']:+.4f} "
               f"ICIR={stats['icir']:.2f} t={stats['tstat']:.2f} "
               f"pct_pos={stats['pct_pos']:.1f}%")
    logger.info(summary)
    return fold_ics, "feature_concat_lgbm"


# ============================================================================
# Simple Baselines
# ============================================================================

def baseline_equal_weight(
    dates: List[str],
    preds: Dict[str, Dict[str, np.ndarray]],
    targets: Dict[str, np.ndarray],
) -> Tuple[List[float], str]:
    """Simple equal-weight average (no optimization needed)."""
    logger.info("=" * 60)
    logger.info("BASELINE: Equal Weight Average")

    model_names = sorted(preds.keys())
    fold_ics = []

    for date in dates:
        combined = np.zeros_like(targets[date])
        for mn in model_names:
            combined += preds[mn][date] / len(model_names)
        fold_ics.append(compute_ic(combined, targets[date]))

    stats = ic_stats(fold_ics)
    logger.info(f"  Equal Weight: IC={stats['ic']:+.4f} "
                f"ICIR={stats['icir']:.2f} t={stats['tstat']:.2f} "
                f"pct_pos={stats['pct_pos']:.1f}%")
    return fold_ics, "equal_weight"


def baseline_rank_avg(
    dates: List[str],
    preds: Dict[str, Dict[str, np.ndarray]],
    targets: Dict[str, np.ndarray],
) -> Tuple[List[float], str]:
    """Rank-based average (convert to ranks first, then average)."""
    logger.info("=" * 60)
    logger.info("BASELINE: Rank Average")

    model_names = sorted(preds.keys())
    fold_ics = []

    for date in dates:
        ranked = []
        for mn in model_names:
            p = preds[mn][date]
            valid = np.isfinite(p)
            r = np.full_like(p, 0.5)
            if valid.sum() > 0:
                r[valid] = rankdata(p[valid]).astype(np.float32) / valid.sum()
            ranked.append(r)
        combined = np.mean(np.column_stack(ranked), axis=1)
        fold_ics.append(compute_ic(combined, targets[date]))

    stats = ic_stats(fold_ics)
    logger.info(f"  Rank Avg: IC={stats['ic']:+.4f} "
                f"ICIR={stats['icir']:.2f} t={stats['tstat']:.2f} "
                f"pct_pos={stats['pct_pos']:.1f}%")
    return fold_ics, "rank_avg"


# ============================================================================
# Individual Model Baselines
# ============================================================================

def individual_model_ics(
    dates: List[str],
    preds: Dict[str, Dict[str, np.ndarray]],
    targets: Dict[str, np.ndarray],
) -> Dict[str, List[float]]:
    """Compute per-fold IC for each individual model."""
    result = {}
    for mn in sorted(preds.keys()):
        ics = []
        for date in dates:
            ics.append(compute_ic(preds[mn][date], targets[date]))
        result[mn] = ics
    return result


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Hybrid Ensemble Evaluation")
    parser.add_argument("--max-folds", type=int, default=None,
                        help="Limit to first N folds (for quick testing)")
    parser.add_argument("--skip-lgbm", action="store_true",
                        help="Skip LightGBM regeneration (2-model ensemble only)")
    parser.add_argument("--lookback", type=int, default=5,
                        help="Lookback window for confidence-weighted (default: 5)")
    parser.add_argument("--opt-lookback", type=int, default=10,
                        help="Lookback window for weight optimization (default: 10)")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("HYBRID ENSEMBLE EVALUATION")
    logger.info("=" * 70)
    logger.info(f"Timestamp: {_ts}")

    # ---- Step 1: Load deep model predictions ----
    logger.info("\nStep 1: Loading deep model predictions...")
    t0 = time.time()

    if not BOOK_PREDS_FILE.exists():
        logger.error(f"Book CNN predictions not found: {BOOK_PREDS_FILE}")
        sys.exit(1)
    if not EVENT_PREDS_FILE.exists():
        logger.error(f"Event Transformer predictions not found: {EVENT_PREDS_FILE}")
        sys.exit(1)

    book_preds = load_deep_predictions(BOOK_PREDS_FILE)
    event_preds = load_deep_predictions(EVENT_PREDS_FILE)
    logger.info(f"  Book CNN: {len(book_preds)} dates loaded ({time.time()-t0:.1f}s)")
    logger.info(f"  Event TF: {len(event_preds)} dates loaded")

    # ---- Step 2: Generate or skip LightGBM predictions ----
    lgbm_preds = None
    if not args.skip_lgbm:
        logger.info("\nStep 2: Generating LightGBM walk-forward predictions...")
        t1 = time.time()
        # We need predictions for all dates that deep models have
        needed_dates = sorted(set(book_preds.keys()) & set(event_preds.keys()))
        lgbm_preds = generate_lgbm_predictions(needed_dates)
        if lgbm_preds:
            logger.info(f"  LightGBM: {len(lgbm_preds)} dates ({time.time()-t1:.1f}s)")
        else:
            logger.warning("  LightGBM generation failed, proceeding with 2-model ensemble")
    else:
        logger.info("\nStep 2: Skipping LightGBM (--skip-lgbm)")

    # ---- Step 3: Align predictions ----
    logger.info("\nStep 3: Aligning predictions across models...")
    common_dates, aligned_preds, aligned_targets = align_predictions(
        book_preds, event_preds, lgbm_preds
    )

    if args.max_folds:
        common_dates = common_dates[:args.max_folds]
        logger.info(f"  Limited to first {args.max_folds} folds")

    logger.info(f"  Using {len(common_dates)} dates, models: {sorted(aligned_preds.keys())}")
    logger.info(f"  Date range: {common_dates[0]} to {common_dates[-1]}")

    # Verify sample counts
    for mn in sorted(aligned_preds.keys()):
        sample_date = common_dates[0]
        logger.info(f"  {mn}: {len(aligned_preds[mn][sample_date])} samples/day (day 1)")

    # ---- Step 4: Individual model baselines ----
    logger.info("\nStep 4: Computing individual model ICs...")
    indiv_ics = individual_model_ics(common_dates, aligned_preds, aligned_targets)
    for mn, ics in sorted(indiv_ics.items()):
        stats = ic_stats(ics)
        logger.info(f"  {mn:12s}: IC={stats['ic']:+.4f} ICIR={stats['icir']:.2f} "
                     f"t={stats['tstat']:.2f} pct_pos={stats['pct_pos']:.1f}% "
                     f"({stats['n']} folds)")

    # ---- Step 5: Run all approaches ----
    logger.info("\nStep 5: Running ensemble approaches...")
    all_results = {}

    # Baselines
    ics, name = baseline_equal_weight(common_dates, aligned_preds, aligned_targets)
    all_results[name] = ics

    ics, name = baseline_rank_avg(common_dates, aligned_preds, aligned_targets)
    all_results[name] = ics

    # Approach 1: Optimized Weighted Average
    ics, name = approach_1_weighted_avg(
        common_dates, aligned_preds, aligned_targets, lookback=args.opt_lookback
    )
    all_results[name] = ics

    # Approach 2: Stacking Meta-Learner
    ics, name = approach_2_stacking(common_dates, aligned_preds, aligned_targets)
    all_results[name] = ics

    # Approach 3: Gated Mixture of Experts
    ics, name = approach_3_gated_moe(common_dates, aligned_preds, aligned_targets)
    all_results[name] = ics

    # Approach 4: Confidence-Weighted Blend
    ics, name = approach_4_confidence_weighted(
        common_dates, aligned_preds, aligned_targets, lookback=args.lookback
    )
    all_results[name] = ics

    # Approach 5: Feature Concatenation
    ics, name = approach_5_feature_concat(common_dates, aligned_preds, aligned_targets)
    all_results[name] = ics

    # ---- Step 6: Comparison Table ----
    logger.info("\n")
    logger.info("=" * 80)
    logger.info("FINAL COMPARISON TABLE")
    logger.info("=" * 80)

    # Header
    logger.info(f"{'Method':<30s} {'IC':>8s} {'ICIR':>8s} {'t-stat':>8s} "
                f"{'Pct+':>8s} {'Folds':>6s} {'vs Best':>8s}")
    logger.info("-" * 80)

    # Find best individual model IC for comparison
    best_indiv_name = max(indiv_ics.keys(), key=lambda mn: np.mean(indiv_ics[mn]))
    best_indiv_ic = np.mean(indiv_ics[best_indiv_name])

    # Individual models first
    for mn in sorted(indiv_ics.keys()):
        stats = ic_stats(indiv_ics[mn])
        delta = stats["ic"] - best_indiv_ic
        marker = " <-- best" if mn == best_indiv_name else ""
        logger.info(f"  {mn:<28s} {stats['ic']:+8.4f} {stats['icir']:8.2f} "
                     f"{stats['tstat']:8.2f} {stats['pct_pos']:7.1f}% "
                     f"{stats['n']:5d}  {delta:+7.4f}{marker}")

    logger.info("-" * 80)

    # Ensemble methods
    ranked_methods = sorted(all_results.keys(), key=lambda k: np.mean(all_results[k]) if all_results[k] else -1, reverse=True)
    for name in ranked_methods:
        ics = all_results[name]
        if not ics:
            continue
        stats = ic_stats(ics)
        delta = stats["ic"] - best_indiv_ic
        beat = "BEAT" if delta > 0.001 else ("TIE" if abs(delta) < 0.001 else "")
        logger.info(f"  {name:<28s} {stats['ic']:+8.4f} {stats['icir']:8.2f} "
                     f"{stats['tstat']:8.2f} {stats['pct_pos']:7.1f}% "
                     f"{stats['n']:5d}  {delta:+7.4f}  {beat}")

    logger.info("=" * 80)
    logger.info(f"Best individual model: {best_indiv_name} (IC={best_indiv_ic:+.4f})")

    # Find best ensemble
    best_ens_name = max(all_results.keys(),
                        key=lambda k: np.mean(all_results[k]) if all_results[k] else -1)
    best_ens_ic = np.mean(all_results[best_ens_name]) if all_results[best_ens_name] else 0
    improvement = best_ens_ic - best_indiv_ic
    logger.info(f"Best ensemble method:  {best_ens_name} (IC={best_ens_ic:+.4f})")
    logger.info(f"Improvement over best individual: {improvement:+.4f} "
                f"({'YES' if improvement > 0 else 'NO'} improvement)")

    # ---- Step 7: Save results ----
    results_file = RESULTS_DIR / f"hybrid_ensemble_{_ts}.json"
    save_data = {
        "timestamp": _ts,
        "n_dates": len(common_dates),
        "date_range": [common_dates[0], common_dates[-1]],
        "models": sorted(aligned_preds.keys()),
        "individual": {
            mn: ic_stats(ics) for mn, ics in indiv_ics.items()
        },
        "ensemble": {
            name: ic_stats(ics) for name, ics in all_results.items() if ics
        },
        "best_individual": {"name": best_indiv_name, "ic": best_indiv_ic},
        "best_ensemble": {"name": best_ens_name, "ic": best_ens_ic},
        "improvement": improvement,
        "per_fold_ics": {
            **{f"indiv_{mn}": [float(x) for x in ics] for mn, ics in indiv_ics.items()},
            **{name: [float(x) for x in ics] for name, ics in all_results.items() if ics},
        },
    }

    with open(results_file, "w") as f:
        json.dump(save_data, f, indent=2)
    logger.info(f"\nResults saved: {results_file}")
    logger.info(f"Log saved: {_log_file}")

    # ---- Step 8: Correlation analysis between ensemble methods ----
    logger.info("\n")
    logger.info("ENSEMBLE METHOD CORRELATION (per-fold IC vectors):")
    method_names = [n for n in ranked_methods if all_results.get(n)]
    if len(method_names) >= 2:
        for j in range(len(method_names)):
            for k in range(j + 1, len(method_names)):
                mj, mk = method_names[j], method_names[k]
                min_len = min(len(all_results[mj]), len(all_results[mk]))
                if min_len > 5:
                    corr = np.corrcoef(
                        all_results[mj][:min_len],
                        all_results[mk][:min_len]
                    )[0, 1]
                    logger.info(f"  {mj:25s} vs {mk:25s}: r={corr:.3f}")

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
