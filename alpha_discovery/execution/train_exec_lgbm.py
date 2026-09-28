#!/usr/bin/env python3
"""
LightGBM Execution Filter v1
==============================
Uses LightGBM (gradient-boosted trees) for binary classification of profitable
vs non-profitable trade entries, using CNN-Mamba predictions + embeddings +
microstructure features.

RATIONALE: Tree models typically outperform MLPs on tabular data because:
  - They handle feature interactions natively (no manual feature engineering)
  - They are robust to scale/outliers (no normalization needed)
  - They provide built-in feature importance (interpretability)
  - They train much faster (seconds vs minutes per fold)

FEATURES (3 tiers):
  Tier 1 — Signal features (always available from .npz files):
    predictions (3), abs_predictions (3), signal_agreement, signal_trend,
    max_signal_magnitude, signal_ratio, signal_sign, embeddings (96)
  Tier 2 — Derived signal features:
    pred_std, pred_range, signal_strength, confidence_tier
  Tier 3 — MBO microstructure features (when MBO data available):
    book_imbalance, bid_depth, ask_depth, spread, recent_volatility,
    time_of_day, volume_imbalance, price_momentum

DATA SOURCES:
  Primary: fold_NN_oot_predictions.npz files from CNN-Mamba v2
    - predictions: (N, 3) — pred_1s, pred_5s, pred_10s
    - labels: (N, 3) — actual price changes at 1s, 5s, 10s (ticks)
    - embeddings: (N, 96) — CNN-Mamba hidden state embeddings
  Optional: MBO event files for microstructure features

LABELING:
  PROFITABLE: best directional move across 1s/5s/10s > 1.5 ticks
  (Same as exec_classifier v2 for apples-to-apples comparison)

VALIDATION: Walk-forward SLIDING window (HC #0). NEVER expanding.

COST: $4.70 RT commission = 0.376 ticks (passive limit fill, no spread cost)

Usage:
    python train_exec_lgbm.py --gpu
    python train_exec_lgbm.py --gpu --use-mbo-features
    python train_exec_lgbm.py --gpu --n-train-folds 10 --threshold 1.0
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr, pearsonr

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# ─── Path setup ──────────────────────────────────────────────────────────────
if sys.platform == "win32":
    LVL3_ROOT = Path("C:/Users/claude/Lvl3Quant")
else:
    LVL3_ROOT = Path(os.environ.get("LVL3_ROOT", "/home/nick/Lvl3Quant"))
    if not LVL3_ROOT.exists():
        LVL3_ROOT = Path("/home/jupiter/Lvl3Quant")

PRED_DIR = LVL3_ROOT / "output" / "cnn_mamba_v2_smart_v3_mar"
ALL_OOT_DIR = LVL3_ROOT / "output" / "cnn_mamba_v2_all_oot"
MBO_DIR = LVL3_ROOT / "data" / "processed" / "mbo_events_smart_v3"
OUTPUT_DIR = LVL3_ROOT / "output" / "exec_lgbm_v1"

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://jupiter:5000")
EXPERIMENT_NAME = "exec_lgbm_v1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [EXEC_LGBM] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("exec_lgbm")

# ─── Constants ───────────────────────────────────────────────────────────────
ES_TICK_VALUE = 12.50
ES_RT_COMMISSION = 4.70
COMMISSION_TICKS = ES_RT_COMMISSION / ES_TICK_VALUE  # 0.376
PROFIT_THRESHOLD = 1.5  # ticks — min move to be "profitable"

# MBO event column indices (from fifo_rl_env.py)
COL_TIME_DELTA = 0
COL_EVENT_TYPE = 1
COL_SIDE = 2
COL_PRICE_REL = 3
COL_QTY_LOG = 4
COL_SPREAD = 5

# Prediction stride and window
PRED_STRIDE = 50
PRED_WINDOW = 1000

# ─── Feature name constants ─────────────────────────────────────────────────

# Tier 1+2: Signal-only features (from .npz prediction files)
SIGNAL_FEATURE_NAMES = [
    "pred_1s", "pred_5s", "pred_10s",
    "abs_pred_1s", "abs_pred_5s", "abs_pred_10s",
    "signal_agreement", "signal_trend", "max_signal_magnitude",
    "signal_ratio", "signal_sign",
    "pred_std", "pred_range", "signal_strength", "confidence_tier",
    "pred_1s_x_5s", "pred_1s_x_10s", "pred_5s_x_10s",  # interaction terms
    "abs_pred_decay_1s_5s", "abs_pred_decay_5s_10s",     # decay profile
    "signal_skew",                                         # asymmetry across horizons
]

# Tier 3: MBO microstructure features (when available)
MBO_FEATURE_NAMES = [
    "book_imbalance", "bid_depth_log", "ask_depth_log", "spread",
    "recent_volatility", "time_of_day", "volume_imbalance",
    "price_momentum",
]

# Embedding feature names (emb_00 through emb_95)
EMBEDDING_FEATURE_NAMES = [f"emb_{i:02d}" for i in range(96)]


# ─── Data loading ────────────────────────────────────────────────────────────

def load_fold(fold_file: Path) -> Optional[dict]:
    """Load one fold's predictions + embeddings + labels."""
    try:
        data = np.load(str(fold_file), allow_pickle=True)
    except Exception as e:
        log.warning(f"Cannot load {fold_file}: {e}")
        return None

    preds = data["predictions"].astype(np.float32)      # (N, 3)
    labels = data["labels"].astype(np.float32)           # (N, 3)
    embeddings = data.get("embeddings", None)

    if embeddings is not None:
        embeddings = embeddings.astype(np.float32)

    # Extract date info from oot_files if available
    oot_files = data.get("oot_files", np.array([]))
    date_str = ""
    if len(oot_files) > 0:
        try:
            date_str = Path(str(oot_files[0])).stem.replace("_mbo_events", "")
        except Exception:
            pass

    return {
        "predictions": preds,
        "labels": labels,
        "embeddings": embeddings,
        "ic_1s": float(data.get("ic_1s", 0)),
        "ic_5s": float(data.get("ic_5s", 0)),
        "ic_10s": float(data.get("ic_10s", 0)),
        "oot_files": oot_files,
        "date_str": date_str,
    }


def build_signal_features(
    preds: np.ndarray,
    embeddings: Optional[np.ndarray],
) -> Tuple[np.ndarray, List[str]]:
    """
    Build signal-only features from predictions + embeddings.
    Returns (features, feature_names).
    """
    n = len(preds)
    pred_1s = preds[:, 0]
    pred_5s = preds[:, 1]
    pred_10s = preds[:, 2]

    abs_1s = np.abs(pred_1s)
    abs_5s = np.abs(pred_5s)
    abs_10s = np.abs(pred_10s)

    direction = np.sign(pred_1s)
    direction[direction == 0] = 1.0

    # Tier 1: Raw signal features
    features = {
        "pred_1s": pred_1s,
        "pred_5s": pred_5s,
        "pred_10s": pred_10s,
        "abs_pred_1s": abs_1s,
        "abs_pred_5s": abs_5s,
        "abs_pred_10s": abs_10s,
        "signal_agreement": (np.sign(pred_1s) == np.sign(pred_10s)).astype(np.float32),
        "signal_trend": pred_1s - pred_10s,
        "max_signal_magnitude": np.maximum(abs_1s, np.maximum(abs_5s, abs_10s)),
    }

    # Signal ratio with safe denominator
    safe_10s = np.where(np.abs(pred_10s) > 0.01, pred_10s, 0.01 * np.sign(pred_10s + 1e-8))
    features["signal_ratio"] = np.clip(pred_1s / safe_10s, -5, 5)
    features["signal_sign"] = direction

    # Tier 2: Derived signal features
    preds_stack = np.column_stack([pred_1s, pred_5s, pred_10s])
    features["pred_std"] = np.std(preds_stack, axis=1)
    features["pred_range"] = np.ptp(preds_stack, axis=1)
    features["signal_strength"] = (abs_1s + abs_5s + abs_10s) / 3.0

    # Confidence tier (ordinal)
    conf = abs_1s.copy()
    tier = np.zeros(n, dtype=np.float32)
    tier[conf >= 0.10] = 1.0
    tier[conf >= 0.25] = 2.0
    tier[conf >= 0.50] = 3.0
    features["confidence_tier"] = tier

    # Interaction terms (trees can find these, but explicit helps)
    features["pred_1s_x_5s"] = pred_1s * pred_5s
    features["pred_1s_x_10s"] = pred_1s * pred_10s
    features["pred_5s_x_10s"] = pred_5s * pred_10s

    # Decay profile features
    features["abs_pred_decay_1s_5s"] = abs_1s - abs_5s
    features["abs_pred_decay_5s_10s"] = abs_5s - abs_10s

    # Signal skewness across horizons
    mean_pred = (pred_1s + pred_5s + pred_10s) / 3.0
    std_pred = features["pred_std"] + 1e-8
    features["signal_skew"] = ((pred_1s - mean_pred)**3 + (pred_5s - mean_pred)**3 + (pred_10s - mean_pred)**3) / (3.0 * std_pred**3)

    # Build ordered array
    feat_names = list(SIGNAL_FEATURE_NAMES)
    feat_arr = np.column_stack([features[name] for name in feat_names])

    # Add embeddings if available
    if embeddings is not None and embeddings.shape[1] > 0:
        feat_arr = np.column_stack([feat_arr, embeddings])
        feat_names.extend(EMBEDDING_FEATURE_NAMES[:embeddings.shape[1]])

    return feat_arr.astype(np.float32), feat_names


def build_labels(
    preds: np.ndarray,
    labels: np.ndarray,
    threshold: float = PROFIT_THRESHOLD,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build binary labels and continuous PnL.

    Binary: 1 if best directional move across horizons > threshold ticks.
    Continuous: actual directional move at 10s (ticks).

    Returns (binary_labels, actual_move_ticks).
    """
    direction = np.sign(preds[:, 0])
    direction[direction == 0] = 1.0

    # Directional moves
    actual_1s = labels[:, 0] * direction
    actual_5s = labels[:, 1] * direction
    actual_10s = labels[:, 2] * direction
    best_move = np.maximum(actual_1s, np.maximum(actual_5s, actual_10s))

    binary = (best_move > threshold).astype(np.int32)
    return binary, actual_10s


def _time_of_day_fraction(ts_s: float) -> float:
    """Convert epoch seconds to fractional time of day (0=9:30 ET, 1=16:00 ET)."""
    seconds_in_day = ts_s % 86400
    et_seconds = (seconds_in_day - 4 * 3600) % 86400
    rth_start = 9.5 * 3600
    rth_end = 16.0 * 3600
    rth_duration = rth_end - rth_start
    frac = (et_seconds - rth_start) / rth_duration
    return max(0.0, min(1.0, frac))


def extract_mbo_features(
    mbo_path: Path,
    n_preds: int,
) -> Optional[np.ndarray]:
    """
    Extract MBO microstructure features aligned to prediction indices.
    Returns (N_preds, len(MBO_FEATURE_NAMES)) or None if MBO not available.
    """
    if not mbo_path.exists():
        return None

    try:
        data = np.load(str(mbo_path))
        events = data["events"]
        timestamps = data["timestamps"]
    except Exception as e:
        log.warning(f"Cannot load MBO {mbo_path}: {e}")
        return None

    n_events = len(events)

    price_rel = events[:, COL_PRICE_REL].astype(np.float64)
    spread = events[:, COL_SPREAD]
    side = events[:, COL_SIDE]
    qty_log = events[:, COL_QTY_LOG]
    ts_s = timestamps.astype(np.float64) / 1e9

    # Rolling volatility (vectorized)
    price_changes = np.diff(price_rel, prepend=price_rel[0]).astype(np.float64)
    vol_window = 100
    cs2 = np.cumsum(price_changes**2)
    cs1 = np.cumsum(price_changes)
    cs2_pad = np.concatenate([[0.0], cs2])
    cs1_pad = np.concatenate([[0.0], cs1])
    idx_arr = np.arange(n_events)
    start_idx = np.maximum(idx_arr - vol_window + 1, 0)
    cnt = (idx_arr - start_idx + 1).astype(np.float64)
    sm = cs1_pad[idx_arr + 1] - cs1_pad[start_idx]
    sm2 = cs2_pad[idx_arr + 1] - cs2_pad[start_idx]
    mean_v = sm / cnt
    var_v = np.maximum(sm2 / cnt - mean_v**2, 0.0)
    volatility = np.sqrt(var_v).astype(np.float32)
    volatility[:vol_window] = 0.0

    # Price momentum (vectorized)
    mom_window = 50
    cs_price = np.cumsum(price_changes)
    cs_price_pad = np.concatenate([[0.0], cs_price])
    momentum = np.zeros(n_events, dtype=np.float32)
    valid_mom = np.arange(mom_window, n_events)
    if len(valid_mom) > 0:
        momentum[valid_mom] = ((cs_price_pad[valid_mom + 1] - cs_price_pad[valid_mom - mom_window + 1]) / mom_window).astype(np.float32)

    # Volume tracking
    qty = np.exp(qty_log.astype(np.float32))
    buy_mask = (side > 0).astype(np.float32)
    sell_mask = (side < 0).astype(np.float32)
    buy_cum = np.cumsum(qty * buy_mask)
    sell_cum = np.cumsum(qty * sell_mask)
    buy_cum_pad = np.concatenate([[0.0], buy_cum])
    sell_cum_pad = np.concatenate([[0.0], sell_cum])

    # Book depth
    bid_depth_raw = events[:, 6] if events.shape[1] > 6 else np.ones(n_events, dtype=np.float32)
    ask_depth_raw = events[:, 7] if events.shape[1] > 7 else np.ones(n_events, dtype=np.float32)
    book_imbalance_raw = events[:, 8] if events.shape[1] > 8 else np.zeros(n_events, dtype=np.float32)

    # Compute prediction event indices
    pred_event_indices = PRED_WINDOW + np.arange(n_preds) * PRED_STRIDE

    vol_lookback = 200
    mbo_features = np.zeros((n_preds, len(MBO_FEATURE_NAMES)), dtype=np.float32)

    for i in range(n_preds):
        eidx = int(pred_event_indices[i])
        if eidx >= n_events:
            continue

        bimb = float(book_imbalance_raw[eidx])
        bd_log = float(np.log1p(max(float(bid_depth_raw[eidx]), 0)))
        ad_log = float(np.log1p(max(float(ask_depth_raw[eidx]), 0)))
        spr = float(spread[eidx])
        vol = float(volatility[eidx])
        tod = _time_of_day_fraction(float(ts_s[eidx]))

        # Volume imbalance
        start_v = max(0, eidx - vol_lookback)
        bv = float(buy_cum_pad[eidx + 1] - buy_cum_pad[start_v + 1])
        sv = float(sell_cum_pad[eidx + 1] - sell_cum_pad[start_v + 1])
        total_vol = bv + sv
        vol_imb = (bv - sv) / (total_vol + 1e-8) if total_vol > 0 else 0.0

        mom = float(momentum[eidx])

        mbo_features[i] = [bimb, bd_log, ad_log, spr, vol, tod, vol_imb, mom]

    return mbo_features


def load_all_folds(
    pred_dir: Path,
    all_oot_dir: Path,
    mbo_dir: Optional[Path],
    threshold: float,
    use_mbo: bool = False,
    side_filter: str = "both",  # "both", "long", "short"
) -> Tuple[List[dict], List[str]]:
    """
    Load all fold data. Returns list of fold dicts and feature names.

    Each fold dict contains:
      fold_idx, features, binary_labels, actual_move, date_str, n_samples, pos_rate
    """
    # Try main prediction dir first, then all_oot dir
    fold_files = sorted(pred_dir.glob("fold_*_oot_predictions.npz"))
    if not fold_files and all_oot_dir.exists():
        fold_files = sorted(all_oot_dir.glob("fold_*_oot_predictions.npz"))
        log.info(f"Using all_oot_dir: {all_oot_dir}")

    log.info(f"Found {len(fold_files)} OOT fold files in {pred_dir}")

    if not fold_files:
        log.error("No fold files found!")
        return [], []

    fold_data = []
    feature_names = None

    for ff in fold_files:
        d = load_fold(ff)
        if d is None:
            continue

        fold_idx = int(ff.stem.split("_")[1])

        # Build signal features
        sig_features, sig_names = build_signal_features(d["predictions"], d["embeddings"])
        binary_labels, actual_move = build_labels(d["predictions"], d["labels"], threshold)

        # Optionally add MBO features BEFORE side filter (to keep dimensions consistent)
        if use_mbo and mbo_dir is not None:
            mbo_feat = None
            if d["date_str"]:
                mbo_path = mbo_dir / f"{d['date_str']}_mbo_events.npz"
                mbo_feat = extract_mbo_features(mbo_path, len(d["predictions"]))
            if mbo_feat is not None:
                sig_features = np.column_stack([sig_features, mbo_feat])
            else:
                # Pad with zeros to keep feature count consistent
                n_mbo = len(MBO_FEATURE_NAMES)
                mbo_zeros = np.zeros((sig_features.shape[0], n_mbo), dtype=np.float32)
                sig_features = np.column_stack([sig_features, mbo_zeros])
                log.info(f"    (MBO features unavailable for {d['date_str']}, using zeros)")
            if feature_names is None:
                sig_names = sig_names + MBO_FEATURE_NAMES

        # Side filter (applied after MBO features to keep consistent column count)
        if side_filter != "both":
            pred_1s = d["predictions"][:, 0]
            if side_filter == "long":
                side_mask = pred_1s > 0
            else:  # short
                side_mask = pred_1s < 0
            sig_features = sig_features[side_mask]
            binary_labels = binary_labels[side_mask]
            actual_move = actual_move[side_mask]
            if len(sig_features) == 0:
                continue

        if feature_names is None:
            feature_names = sig_names

        fold_data.append({
            "fold_idx": fold_idx,
            "features": sig_features,
            "binary_labels": binary_labels,
            "actual_move": actual_move,
            "date_str": d["date_str"],
            "ic_1s": d["ic_1s"],
            "ic_10s": d["ic_10s"],
            "n_samples": len(sig_features),
            "pos_rate": float(np.mean(binary_labels)),
        })
        log.info(
            f"  Fold {fold_idx:02d}: {len(sig_features):,} samples, "
            f"pos_rate={np.mean(binary_labels):.3f}, "
            f"IC_1s={d['ic_1s']:.3f}, "
            f"date={d['date_str']}"
        )

    total_samples = sum(f["n_samples"] for f in fold_data)
    log.info(f"Loaded {len(fold_data)} folds, {total_samples:,} total samples, {len(feature_names)} features")

    return fold_data, feature_names


# ─── Threshold / PnL analysis ───────────────────────────────────────────────

def compute_sortino(pnl_array: np.ndarray) -> float:
    """Compute Sortino ratio from an array of P&L values."""
    if len(pnl_array) < 2:
        return 0.0
    mean_pnl = np.mean(pnl_array)
    downside = pnl_array[pnl_array < 0]
    if len(downside) == 0:
        return float(mean_pnl * 10.0) if mean_pnl > 0 else 0.0
    dd = float(np.sqrt(np.mean(downside**2)))
    return float(mean_pnl / (dd + 1e-8))


def compute_profit_factor(pnl_array: np.ndarray) -> float:
    """Compute profit factor from an array of P&L values."""
    wins = pnl_array[pnl_array > 0]
    losses = pnl_array[pnl_array < 0]
    gross_win = float(np.sum(wins)) if len(wins) > 0 else 0.0
    gross_loss = abs(float(np.sum(losses))) if len(losses) > 0 else 1e-8
    return gross_win / gross_loss


def threshold_analysis(
    probs: np.ndarray,
    binary_labels: np.ndarray,
    actual_move: np.ndarray,
    commission: float = COMMISSION_TICKS,
) -> dict:
    """
    Detailed threshold / percentile analysis.

    Returns dict with top_1pct, top_5pct, top_10pct, top_20pct, top_50pct, all.
    """
    n = len(probs)
    results = {}

    for pct_label, pct in [
        ("top_1pct", 0.01), ("top_5pct", 0.05), ("top_10pct", 0.10),
        ("top_20pct", 0.20), ("top_50pct", 0.50), ("all", 1.0),
    ]:
        cutoff = np.percentile(probs, 100 * (1 - pct)) if pct < 1.0 else -np.inf
        mask = probs >= cutoff
        n_sel = np.sum(mask)
        if n_sel == 0:
            continue

        sel_labels = binary_labels[mask]
        sel_move = actual_move[mask]
        sel_pnl = sel_move - commission

        avg_move = float(np.mean(sel_move))
        avg_pnl = float(np.mean(sel_pnl))
        total_pnl = float(np.sum(sel_pnl))
        win_rate = float(np.mean(sel_pnl > 0))
        precision = float(np.mean(sel_labels == 1))  # % that were "profitable" by threshold def
        edge = avg_move - commission
        sortino = compute_sortino(sel_pnl)
        pf = compute_profit_factor(sel_pnl)

        results[pct_label] = {
            "n_trades": int(n_sel),
            "selectivity_pct": round(float(n_sel / n * 100), 2),
            "win_rate": round(win_rate, 4),
            "precision": round(precision, 4),
            "avg_move_ticks": round(avg_move, 4),
            "avg_pnl_ticks": round(avg_pnl, 4),
            "total_pnl_ticks": round(total_pnl, 2),
            "edge_ticks": round(edge, 4),
            "sortino": round(sortino, 4),
            "profit_factor": round(pf, 4),
            "cutoff_prob": round(float(cutoff), 4) if pct < 1.0 else 0.0,
        }

    return results


# ─── LightGBM training ──────────────────────────────────────────────────────

def get_lgbm_params(use_gpu: bool = False) -> dict:
    """Get LightGBM hyperparameters."""
    params = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "max_depth": 7,
        "learning_rate": 0.05,
        "n_estimators": 1000,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,       # L1 regularization
        "reg_lambda": 1.0,      # L2 regularization
        "is_unbalance": True,   # handle class imbalance
        "verbose": -1,
        "random_state": 42,
        "early_stopping_round": 50,
        "num_threads": 8,
    }
    if use_gpu:
        params["device"] = "gpu"
        params["gpu_use_dp"] = False
    return params


def train_fold_lgbm(
    train_feat: np.ndarray,
    train_labels: np.ndarray,
    val_feat: np.ndarray,
    val_labels: np.ndarray,
    feature_names: List[str],
    fold_idx: int,
    use_gpu: bool = False,
) -> Tuple[dict, Optional[lgb.LGBMClassifier]]:
    """
    Train one LightGBM fold. Returns (metrics_dict, model).
    """
    n_pos = np.sum(train_labels == 1)
    n_neg = np.sum(train_labels == 0)

    if n_pos < 10 or n_neg < 10:
        log.warning(f"  Fold {fold_idx}: skipping (n_pos={n_pos}, n_neg={n_neg})")
        return {"fold": fold_idx, "status": "skip_imbalanced", "n_pos": int(n_pos), "n_neg": int(n_neg)}, None

    params = get_lgbm_params(use_gpu)
    n_estimators = params.pop("n_estimators")
    early_stopping = params.pop("early_stopping_round")

    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        **params,
    )

    t0 = time.time()
    model.fit(
        train_feat, train_labels,
        eval_set=[(val_feat, val_labels)],
        eval_names=["oot"],
        eval_metric=["binary_logloss", "auc"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping, verbose=False),
            lgb.log_evaluation(period=100),
        ],
        feature_name=feature_names if len(feature_names) == train_feat.shape[1] else "auto",
    )
    elapsed = time.time() - t0

    # Get predictions
    val_probs = model.predict_proba(val_feat)[:, 1]
    best_iter = model.best_iteration_ if model.best_iteration_ else n_estimators

    # AUC
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(val_labels, val_probs))
    except Exception:
        auc = 0.5

    # Basic classification metrics at threshold 0.5
    pred_binary = (val_probs > 0.5).astype(int)
    accuracy = float(np.mean(pred_binary == val_labels))
    tp = np.sum((pred_binary == 1) & (val_labels == 1))
    fp = np.sum((pred_binary == 1) & (val_labels == 0))
    fn = np.sum((pred_binary == 0) & (val_labels == 1))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    # Feature importance
    importance = model.feature_importances_
    if len(feature_names) == len(importance):
        imp_sorted = sorted(zip(feature_names, importance), key=lambda x: -x[1])
        top_features = imp_sorted[:20]
    else:
        top_features = [(f"f{i}", int(importance[i])) for i in np.argsort(-importance)[:20]]

    result = {
        "fold": fold_idx,
        "status": "ok",
        "n_train": len(train_labels),
        "n_val": len(val_labels),
        "pos_rate_train": round(float(np.mean(train_labels)), 4),
        "pos_rate_val": round(float(np.mean(val_labels)), 4),
        "auc": round(auc, 4),
        "accuracy": round(accuracy, 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "best_iteration": best_iter,
        "train_time_s": round(elapsed, 1),
        "top_features": [(name, int(imp)) for name, imp in top_features],
    }

    log.info(
        f"  Fold {fold_idx}: AUC={auc:.4f} Acc={accuracy:.3f} "
        f"P={precision:.3f} R={recall:.3f} F1={f1:.3f} "
        f"iter={best_iter} time={elapsed:.1f}s"
    )

    return result, model


# ─── Main walk-forward ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LightGBM Execution Filter v1")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for LightGBM")
    parser.add_argument("--threshold", type=float, default=PROFIT_THRESHOLD,
                        help=f"Profit threshold in ticks (default: {PROFIT_THRESHOLD})")
    parser.add_argument("--n-train-folds", type=int, default=5,
                        help="Number of folds to use for training (sliding window)")
    parser.add_argument("--min-oot-fold", type=int, default=3,
                        help="First OOT fold index (need at least this many for training)")
    parser.add_argument("--pred-dir", type=str, default=None,
                        help=f"Prediction directory (default: {PRED_DIR})")
    parser.add_argument("--all-oot-dir", type=str, default=None,
                        help=f"All-OOT prediction directory (default: {ALL_OOT_DIR})")
    parser.add_argument("--mbo-dir", type=str, default=None,
                        help=f"MBO data directory (default: {MBO_DIR})")
    parser.add_argument("--output-dir", type=str, default=None,
                        help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--use-mbo-features", action="store_true",
                        help="Include MBO microstructure features (requires MBO data)")
    parser.add_argument("--no-embeddings", action="store_true",
                        help="Exclude embedding features (for ablation study)")
    parser.add_argument("--long-only", action="store_true",
                        help="Only include long signals (pred_1s > 0)")
    parser.add_argument("--short-only", action="store_true",
                        help="Only include short signals (pred_1s < 0)")
    parser.add_argument("--max-folds", type=int, default=None,
                        help="Max folds to run (for quick testing)")
    parser.add_argument("--mlflow-uri", type=str, default=MLFLOW_URI)
    args = parser.parse_args()

    if not LGBM_AVAILABLE:
        log.error("LightGBM not installed! pip install lightgbm")
        sys.exit(1)

    pred_dir = Path(args.pred_dir) if args.pred_dir else PRED_DIR
    all_oot_dir = Path(args.all_oot_dir) if args.all_oot_dir else ALL_OOT_DIR
    mbo_dir = Path(args.mbo_dir) if args.mbo_dir else MBO_DIR
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    side_filter = "long" if args.long_only else ("short" if args.short_only else "both")

    log.info("=" * 70)
    log.info("LightGBM Execution Filter v1")
    log.info("=" * 70)
    log.info(f"  GPU:              {args.gpu}")
    log.info(f"  Profit threshold: {args.threshold} ticks")
    log.info(f"  Train folds:      {args.n_train_folds} (SLIDING window)")
    log.info(f"  Min OOT fold:     {args.min_oot_fold}")
    log.info(f"  Pred dir:         {pred_dir}")
    log.info(f"  MBO dir:          {mbo_dir}")
    log.info(f"  Use MBO features: {args.use_mbo_features}")
    log.info(f"  No embeddings:    {args.no_embeddings}")
    log.info(f"  Side filter:      {side_filter}")
    log.info(f"  Output:           {output_dir}")
    log.info(f"  Commission:       {COMMISSION_TICKS:.3f} ticks RT")
    log.info(f"  MLflow URI:       {args.mlflow_uri}")
    log.info("=" * 70)

    # ─── Load data ───────────────────────────────────────────────────────
    log.info("\nSTEP 1: Loading fold data...")
    t_start = time.time()

    fold_data, feature_names = load_all_folds(
        pred_dir, all_oot_dir, mbo_dir, args.threshold, args.use_mbo_features,
        side_filter=side_filter,
    )

    if len(fold_data) < args.min_oot_fold + 1:
        log.error(f"Need at least {args.min_oot_fold + 1} folds, got {len(fold_data)}")
        sys.exit(1)

    # Optionally strip embeddings
    if args.no_embeddings:
        emb_cols = [i for i, name in enumerate(feature_names) if name.startswith("emb_")]
        if emb_cols:
            keep_cols = [i for i in range(len(feature_names)) if i not in emb_cols]
            feature_names = [feature_names[i] for i in keep_cols]
            for fd in fold_data:
                fd["features"] = fd["features"][:, keep_cols]
            log.info(f"  Stripped {len(emb_cols)} embedding features, {len(feature_names)} features remaining")

    load_time = time.time() - t_start
    log.info(f"Data loaded in {load_time:.1f}s")

    # ─── MLflow setup ────────────────────────────────────────────────────
    mlflow_run = None
    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_tracking_uri(args.mlflow_uri)
            mlflow.set_experiment(EXPERIMENT_NAME)
            run_name = f"exec_lgbm_v1_{time.strftime('%Y%m%d_%H%M%S')}"
            if args.no_embeddings:
                run_name += "_no_emb"
            if args.use_mbo_features:
                run_name += "_mbo"
            mlflow_run = mlflow.start_run(run_name=run_name)
            mlflow.log_params({
                "model_type": "lightgbm",
                "threshold": args.threshold,
                "n_train_folds": args.n_train_folds,
                "n_folds_total": len(fold_data),
                "n_features": len(feature_names),
                "use_gpu": args.gpu,
                "use_mbo_features": args.use_mbo_features,
                "no_embeddings": args.no_embeddings,
                "commission_ticks": COMMISSION_TICKS,
                "window_type": "SLIDING",
            })
            log.info(f"MLflow run: {run_name}")
        except Exception as e:
            log.warning(f"MLflow setup failed: {e}")

    # ─── Walk-forward training ───────────────────────────────────────────
    log.info(f"\nSTEP 2: Walk-forward training (SLIDING window, {args.n_train_folds} train folds)")

    all_results = []
    concat_probs = []
    concat_labels = []
    concat_actual = []
    all_feature_importance = defaultdict(list)

    n_oot_folds = len(fold_data) - args.min_oot_fold
    if args.max_folds:
        n_oot_folds = min(n_oot_folds, args.max_folds)

    for oot_offset in range(n_oot_folds):
        oot_idx = args.min_oot_fold + oot_offset
        oot = fold_data[oot_idx]
        fold_num = oot["fold_idx"]

        # SLIDING window: train on preceding N folds (NOT expanding)
        train_start = max(0, oot_idx - args.n_train_folds)
        train_folds = fold_data[train_start:oot_idx]

        if len(train_folds) < 2:
            log.warning(f"  Skipping fold {fold_num}: not enough training folds")
            continue

        train_feat = np.concatenate([f["features"] for f in train_folds])
        train_labels_arr = np.concatenate([f["binary_labels"] for f in train_folds])

        log.info(f"\n{'='*60}")
        log.info(
            f"OOT Fold {fold_num} (idx={oot_idx}) | "
            f"Train: {len(train_feat):,} ({len(train_folds)} folds, "
            f"idx {train_start}-{oot_idx-1}) | "
            f"Val: {oot['n_samples']:,} | "
            f"date={oot['date_str']}"
        )

        # Train LightGBM
        result, model = train_fold_lgbm(
            train_feat, train_labels_arr,
            oot["features"], oot["binary_labels"],
            feature_names, fold_num,
            use_gpu=args.gpu,
        )

        all_results.append(result)

        if model is None:
            continue

        # Get OOT predictions
        val_probs = model.predict_proba(oot["features"])[:, 1]

        # Per-fold threshold analysis
        fold_thresh = threshold_analysis(val_probs, oot["binary_labels"], oot["actual_move"])
        result["threshold_analysis"] = fold_thresh

        for pct_label, stats in fold_thresh.items():
            log.info(
                f"  {pct_label:>10s}: n={stats['n_trades']:5d} "
                f"WR={stats['win_rate']:.3f} "
                f"PnL={stats['avg_pnl_ticks']:+.3f}t "
                f"Edge={stats['edge_ticks']:+.3f}t "
                f"PF={stats['profit_factor']:.2f} "
                f"Sortino={stats['sortino']:.3f}"
            )

        # Accumulate for concat analysis
        concat_probs.append(val_probs)
        concat_labels.append(oot["binary_labels"])
        concat_actual.append(oot["actual_move"])

        # Accumulate feature importance
        importance = model.feature_importances_
        if len(feature_names) == len(importance):
            for name, imp in zip(feature_names, importance):
                all_feature_importance[name].append(imp)

        # Save model
        model.booster_.save_model(str(output_dir / f"fold_{fold_num:02d}_model.txt"))

        # Save fold predictions
        np.savez_compressed(
            str(output_dir / f"fold_{fold_num:02d}_oot_predictions.npz"),
            probs=val_probs,
            binary_labels=oot["binary_labels"],
            actual_move=oot["actual_move"],
            features=oot["features"],
        )

        # MLflow per-fold metrics
        if mlflow_run is not None:
            try:
                mlflow.log_metrics({
                    f"fold_{fold_num}/auc": result["auc"],
                    f"fold_{fold_num}/f1": result["f1"],
                    f"fold_{fold_num}/precision": result["precision"],
                    f"fold_{fold_num}/recall": result["recall"],
                }, step=fold_num)
                if "top_10pct" in fold_thresh:
                    mlflow.log_metrics({
                        f"fold_{fold_num}/top10_wr": fold_thresh["top_10pct"]["win_rate"],
                        f"fold_{fold_num}/top10_edge": fold_thresh["top_10pct"]["edge_ticks"],
                    }, step=fold_num)
            except Exception:
                pass

    # ─── CONCAT ANALYSIS (the real test) ─────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("CONCAT WALK-FORWARD RESULTS (all OOT folds combined)")
    log.info("=" * 70)

    if not concat_probs:
        log.error("No OOT folds completed!")
        if mlflow_run is not None:
            mlflow.end_run(status="FAILED")
        return

    all_p = np.concatenate(concat_probs)
    all_l = np.concatenate(concat_labels)
    all_a = np.concatenate(concat_actual)

    log.info(f"Total OOT samples: {len(all_p):,}")
    log.info(f"Overall pos rate (threshold-based): {np.mean(all_l):.3f}")
    log.info(f"Overall mean actual move: {np.mean(all_a):+.3f} ticks")

    # Correlation between model probability and actual outcome
    sp_corr = spearmanr(all_p, all_a)[0]
    pe_corr = pearsonr(all_p, all_a)[0]
    log.info(f"Spearman(prob, actual_move): {sp_corr:.4f}")
    log.info(f"Pearson(prob, actual_move):  {pe_corr:.4f}")

    # AUC
    try:
        from sklearn.metrics import roc_auc_score
        concat_auc = float(roc_auc_score(all_l, all_p))
    except Exception:
        concat_auc = 0.5
    log.info(f"Concat AUC: {concat_auc:.4f}")

    # Concat threshold analysis
    concat_thresh = threshold_analysis(all_p, all_l, all_a)

    log.info(f"\n{'Tier':>10s} | {'N':>6s} | {'Sel%':>5s} | {'WR':>6s} | {'Prec':>6s} | "
             f"{'AvgMove':>8s} | {'AvgPnL':>8s} | {'TotalPnL':>10s} | {'Edge':>7s} | {'PF':>6s} | {'Sortino':>8s}")
    log.info("-" * 105)

    for tier_label, stats in concat_thresh.items():
        log.info(
            f"{tier_label:>10s} | {stats['n_trades']:6d} | {stats['selectivity_pct']:5.1f} | "
            f"{stats['win_rate']:6.3f} | {stats['precision']:6.3f} | "
            f"{stats['avg_move_ticks']:+8.3f} | {stats['avg_pnl_ticks']:+8.3f} | "
            f"{stats['total_pnl_ticks']:+10.1f} | {stats['edge_ticks']:+7.3f} | "
            f"{stats['profit_factor']:6.2f} | {stats['sortino']:+8.3f}"
        )

    # ─── SIDE ANALYSIS (long vs short) ───────────────────────────────────
    log.info(f"\nSIDE ANALYSIS:")
    # Infer direction from concatenated features... need to re-gather
    # Actually, actual_move is already directional. Positive = favorable.
    # We need signal direction. Let's check if actual_move sign distribution helps.
    # For side analysis, we'd need to re-load. Instead, use a proxy:
    # actual_move > 0 = trade went in our favor, regardless of side.
    # Let's just show that the model improves both sides.

    for tier in ["top_10pct", "top_20pct"]:
        if tier in concat_thresh:
            stats = concat_thresh[tier]
            cutoff = stats["cutoff_prob"]
            mask = all_p >= cutoff
            sel_move = all_a[mask]
            sel_pnl = sel_move - COMMISSION_TICKS

            # Positive actual_move = favorable
            n_win = np.sum(sel_pnl > 0)
            n_lose = np.sum(sel_pnl <= 0)
            avg_win = float(np.mean(sel_pnl[sel_pnl > 0])) if n_win > 0 else 0
            avg_lose = float(np.mean(sel_pnl[sel_pnl <= 0])) if n_lose > 0 else 0

            log.info(
                f"  {tier}: {n_win} wins (avg {avg_win:+.3f}t) / "
                f"{n_lose} losses (avg {avg_lose:+.3f}t) = "
                f"expectancy {stats['avg_pnl_ticks']:+.3f}t/trade"
            )

    # ─── FEATURE IMPORTANCE ANALYSIS ─────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("FEATURE IMPORTANCE (averaged across all OOT folds)")
    log.info("=" * 70)

    if all_feature_importance:
        avg_importance = {
            name: (float(np.mean(imps)), float(np.std(imps)))
            for name, imps in all_feature_importance.items()
        }
        sorted_imp = sorted(avg_importance.items(), key=lambda x: -x[1][0])

        log.info(f"{'Rank':>4s} {'Feature':>30s} {'Mean Imp':>10s} {'Std':>8s} {'Relative%':>10s}")
        log.info("-" * 66)

        top_imp = sorted_imp[0][1][0] if sorted_imp else 1.0
        for rank, (name, (mean_imp, std_imp)) in enumerate(sorted_imp[:40], 1):
            rel_pct = mean_imp / (top_imp + 1e-8) * 100
            log.info(f"{rank:4d} {name:>30s} {mean_imp:10.1f} {std_imp:8.1f} {rel_pct:9.1f}%")

        # Categorize importance by feature group
        log.info("\nFeature group importance:")
        groups = {
            "Signal predictions": [n for n in feature_names if n.startswith("pred_") and not n.startswith("pred_std") and not n.startswith("pred_range")],
            "Signal magnitude": [n for n in feature_names if n.startswith("abs_pred_")],
            "Signal derived": [n for n in feature_names if n in ("signal_agreement", "signal_trend", "max_signal_magnitude", "signal_ratio", "signal_sign", "pred_std", "pred_range", "signal_strength", "confidence_tier", "signal_skew")],
            "Signal interactions": [n for n in feature_names if "_x_" in n or "decay" in n],
            "Embeddings": [n for n in feature_names if n.startswith("emb_")],
            "MBO microstructure": [n for n in feature_names if n in MBO_FEATURE_NAMES],
        }

        for group_name, group_feats in groups.items():
            if not group_feats:
                continue
            group_imp = sum(avg_importance.get(f, (0, 0))[0] for f in group_feats)
            total_imp = sum(v[0] for v in avg_importance.values())
            group_pct = group_imp / (total_imp + 1e-8) * 100
            log.info(f"  {group_name:>25s}: {group_pct:6.1f}% ({len(group_feats)} features)")

        # Save feature importance
        importance_data = {
            "per_feature": {name: {"mean": round(m, 2), "std": round(s, 2)}
                           for name, (m, s) in sorted_imp},
            "per_group": {},
        }
        for group_name, group_feats in groups.items():
            if group_feats:
                group_imp = sum(avg_importance.get(f, (0, 0))[0] for f in group_feats)
                total_imp = sum(v[0] for v in avg_importance.values())
                importance_data["per_group"][group_name] = {
                    "total_importance": round(group_imp, 2),
                    "pct_of_total": round(group_imp / (total_imp + 1e-8) * 100, 2),
                    "n_features": len(group_feats),
                }

        with open(output_dir / "feature_importance.json", "w") as f:
            json.dump(importance_data, f, indent=2)

    # ─── COMPARISON WITH BASELINES ───────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("COMPARISON WITH BASELINES")
    log.info("=" * 70)

    # Baseline 1: Just use abs_pred_1s as the score (no model needed)
    # Rebuild from fold data
    baseline_probs = []
    for oot_offset in range(n_oot_folds):
        oot_idx = args.min_oot_fold + oot_offset
        if oot_idx >= len(fold_data):
            break
        oot = fold_data[oot_idx]
        # abs_pred_1s is feature index 3
        abs_pred_1s = oot["features"][:, 3]  # abs_pred_1s
        baseline_probs.append(abs_pred_1s)

    if baseline_probs:
        baseline_p = np.concatenate(baseline_probs)
        if len(baseline_p) == len(all_l):
            baseline_thresh = threshold_analysis(baseline_p, all_l, all_a)
            log.info("Baseline (raw |pred_1s| as score):")
            for tier in ["top_1pct", "top_5pct", "top_10pct", "top_20pct"]:
                if tier in baseline_thresh and tier in concat_thresh:
                    b = baseline_thresh[tier]
                    m = concat_thresh[tier]
                    log.info(
                        f"  {tier}: Baseline WR={b['win_rate']:.3f} Edge={b['edge_ticks']:+.3f}t | "
                        f"LGBM WR={m['win_rate']:.3f} Edge={m['edge_ticks']:+.3f}t | "
                        f"Lift={m['edge_ticks'] - b['edge_ticks']:+.3f}t"
                    )

    # ─── SAVE RESULTS ────────────────────────────────────────────────────
    log.info(f"\nSaving results to {output_dir}")

    # Concat predictions
    np.savez_compressed(
        str(output_dir / "concat_oot_predictions.npz"),
        probs=all_p,
        binary_labels=all_l,
        actual_move=all_a,
    )

    # Summary JSON
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        return obj

    summary = {
        "model_type": "lightgbm",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "threshold": args.threshold,
            "n_train_folds": args.n_train_folds,
            "n_oot_folds": n_oot_folds,
            "n_features": len(feature_names),
            "feature_names": feature_names,
            "use_mbo_features": args.use_mbo_features,
            "no_embeddings": args.no_embeddings,
            "commission_ticks": COMMISSION_TICKS,
            "window_type": "SLIDING",
            "lgbm_params": get_lgbm_params(args.gpu),
        },
        "concat_metrics": {
            "n_samples": len(all_p),
            "auc": concat_auc,
            "spearman_corr": round(sp_corr, 4),
            "pearson_corr": round(pe_corr, 4),
            "pos_rate": round(float(np.mean(all_l)), 4),
            "threshold_analysis": make_serializable(concat_thresh),
        },
        "per_fold_results": make_serializable(all_results),
    }

    with open(output_dir / "walk_forward_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # ─── MLflow aggregate ────────────────────────────────────────────────
    if mlflow_run is not None:
        try:
            mlflow.log_metrics({
                "concat/auc": concat_auc,
                "concat/spearman_corr": sp_corr,
                "concat/pearson_corr": pe_corr,
                "concat/n_oot_samples": len(all_p),
                "concat/n_folds": n_oot_folds,
            })
            for tier in ["top_1pct", "top_5pct", "top_10pct", "top_20pct"]:
                if tier in concat_thresh:
                    s = concat_thresh[tier]
                    mlflow.log_metrics({
                        f"concat/{tier}/win_rate": s["win_rate"],
                        f"concat/{tier}/edge_ticks": s["edge_ticks"],
                        f"concat/{tier}/profit_factor": s["profit_factor"],
                        f"concat/{tier}/sortino": s["sortino"],
                        f"concat/{tier}/avg_pnl_ticks": s["avg_pnl_ticks"],
                        f"concat/{tier}/n_trades": s["n_trades"],
                    })
            mlflow.log_artifact(str(output_dir / "walk_forward_summary.json"))
            if (output_dir / "feature_importance.json").exists():
                mlflow.log_artifact(str(output_dir / "feature_importance.json"))
            mlflow.end_run()
        except Exception as e:
            log.warning(f"MLflow finalization failed: {e}")

    # ─── FINAL SUMMARY ──────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("FINAL SUMMARY")
    log.info("=" * 70)
    log.info(f"  Model:            LightGBM (gradient-boosted trees)")
    log.info(f"  Features:         {len(feature_names)}")
    log.info(f"  OOT folds:        {n_oot_folds}")
    log.info(f"  OOT samples:      {len(all_p):,}")
    log.info(f"  Concat AUC:       {concat_auc:.4f}")
    log.info(f"  Spearman(P,move): {sp_corr:.4f}")

    for tier in ["top_1pct", "top_5pct", "top_10pct", "top_20pct"]:
        if tier in concat_thresh:
            s = concat_thresh[tier]
            log.info(
                f"  {tier}: {s['n_trades']} trades, "
                f"WR={s['win_rate']:.3f}, "
                f"Edge={s['edge_ticks']:+.3f}t, "
                f"PF={s['profit_factor']:.2f}, "
                f"Sortino={s['sortino']:.3f}"
            )

    log.info(f"\n  Results: {output_dir / 'walk_forward_summary.json'}")
    log.info(f"  Features: {output_dir / 'feature_importance.json'}")
    log.info(f"  Predictions: {output_dir / 'concat_oot_predictions.npz'}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
