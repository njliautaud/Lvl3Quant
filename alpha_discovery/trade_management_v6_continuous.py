#!/usr/bin/env python3
"""
Trade Management v6 — Continuous Edge Scoring (NOT Binary Cut/Hold)
===================================================================

Evolution from v2_tick (Sharpe 3.43, +30% over static) and mid-trade_enhanced_v2
(MLP AUC 0.687, IC 0.322). Key insight from v5: binary cut/hold is NEGATIVE at
all thresholds because exit cost (1.376 ticks) exceeds savings. The signal is
better used as a CONTINUOUS CONFIDENCE SCORE for position sizing.

What's new in v6:
  1. Loads pre-built tick_samples.parquet from v2 (52K rows, 33 cols)
  2. Engineers ~15 NEW derived features: velocity, regime, cross-feature interactions,
     spread proxy, time-decay
  3. Target: label_remaining_favorable_ticks (CONTINUOUS regression, not binary)
  4. Models: BOTH LightGBM regressor AND small MLP (64/32 ReLU dropout 0.2)
  5. Walk-forward: 20d train / 10d val / 5d slide (more folds than v2's 2)
  6. Simulation: continuous output as POSITION SIZING SIGNAL
     - High remaining_edge -> hold full position
     - Low/negative -> scale to 0.5x or exit
     - Compare: static, edge-scaled, threshold-based
  7. MLflow: experiment "trade_management_v6_continuous"
  8. Stability: per-fold Sharpe, segment consistency ratio

Cost: passive entry 0.376 ticks, market exit 1.376 ticks (FIFO).

Usage:
  cd /home/nick/Lvl3Quant && \\
  /home/nick/miniconda3/envs/py311-train/bin/python \\
      alpha_discovery/trade_management_v6_continuous.py

Author: Claude (autonomous research)
"""

import gc
import json
import logging
import os
import pickle
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "output" / "trade_management_v2_tick"
OUTPUT_DIR = ROOT / "output" / "trade_management_v6_continuous"
LOG_DIR = ROOT / "logs"
MODEL_DIR = OUTPUT_DIR / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [TM-v6] %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_DIR / "trade_management_v6_continuous.log")),
    ],
)
log = logging.getLogger("TM-v6")

# ─────────────────────────────────────────────
#  COST CONSTANTS (ES Futures — AMP/Rithmic)
# ─────────────────────────────────────────────
ES_TICK_VALUE = 12.50
# Passive limit entry (0.376 commission) + market exit (0.376 commission + 1.0 spread)
COST_RT_TICKS = 1.376
COST_PARTIAL_EXIT_TICKS = 1.376  # market exit for partial position reduction

# Walk-forward config: 20d/10d/5d for MORE folds than v2's 2
TRAIN_DAYS = 20
VAL_DAYS = 10
SLIDE_DAYS = 5

# MLP config
MLP_HIDDEN_1 = 64
MLP_HIDDEN_2 = 32
MLP_DROPOUT = 0.2
MLP_LR = 1e-3
MLP_EPOCHS = 100
MLP_BATCH_SIZE = 512
MLP_PATIENCE = 15

# LightGBM regressor params (tuned from v2 edge model)
LGBM_REG_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_child_samples": 50,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "max_depth": 7,
    "verbose": -1,
    "n_jobs": -1,
}

# Deferred imports
lgb = None
torch = None
nn = None


def _import_lightgbm():
    global lgb
    if lgb is not None:
        return
    import lightgbm as _lgb
    lgb = _lgb


def _import_torch():
    global torch, nn
    if torch is not None:
        return
    import torch as _torch
    import torch.nn as _nn
    torch = _torch
    nn = _nn


# ═══════════════════════════════════════════════════════════════════
#  SECTION 1: DATA LOADING & FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════


def load_tick_samples() -> pd.DataFrame:
    """Load pre-built tick_samples.parquet from v2_tick output."""
    fpath = INPUT_DIR / "tick_samples.parquet"
    if not fpath.exists():
        raise FileNotFoundError(
            f"tick_samples.parquet not found at {fpath}. "
            f"Run trade_management_v2_tick.py first to generate it."
        )
    df = pd.read_parquet(fpath)
    log.info(f"Loaded tick_samples: {len(df):,} rows, {len(df.columns)} columns")
    log.info(f"  Columns: {sorted(df.columns.tolist())}")
    log.info(f"  Dates: {sorted(df['date'].unique())}")
    log.info(f"  Trades: {df['trade_idx'].nunique()}")
    return df


def engineer_v6_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer NEW derived features on top of existing 33 columns.

    Groups:
      - Velocity: rate of change of key signals
      - Regime: drawdown/gain speed
      - Cross-feature: interaction terms
      - Spread proxy: queue imbalance trend
      - Time-decay: confidence decay
    """
    df = df.copy()

    # Sort by trade then time for within-trade computations
    df = df.sort_values(["trade_idx", "time_in_trade_seconds"]).reset_index(drop=True)

    log.info("Engineering v6 features...")

    # ── Velocity features ──
    # queue_ratio_velocity: delta over last 2 samples (within same trade)
    df["queue_ratio_velocity"] = df.groupby("trade_idx")["queue_ratio"].diff(2)
    df["queue_ratio_velocity"] = df["queue_ratio_velocity"].fillna(0.0)

    # ofi_acceleration: delta of ofi_recent_5s (rate of change of flow)
    df["ofi_acceleration"] = df.groupby("trade_idx")["ofi_recent_5s"].diff(1)
    df["ofi_acceleration"] = df["ofi_acceleration"].fillna(0.0)

    # microprice_velocity: rate of change of microprice trend
    df["microprice_velocity"] = df.groupby("trade_idx")["microprice_trend"].diff(1)
    df["microprice_velocity"] = df["microprice_velocity"].fillna(0.0)

    # imbalance_velocity: rate of change of order book imbalance
    df["imbalance_velocity"] = df.groupby("trade_idx")["imbalance_now"].diff(1)
    df["imbalance_velocity"] = df["imbalance_velocity"].fillna(0.0)

    # ── Regime / speed features ──
    # Drawdown speed: how fast are we drawing down from peak?
    time_safe = df["time_in_trade_seconds"].clip(lower=1.0)
    df["drawdown_speed"] = df["drawdown_from_peak"] / time_safe

    # Gain speed: how fast did we reach MFE?
    df["gain_speed"] = df["mfe_so_far_ticks"] / time_safe

    # P&L momentum: unrealized P&L normalized by time
    df["pnl_per_second"] = df["unrealized_pnl_ticks"] / time_safe

    # MAE recovery: how much have we recovered from worst point?
    mae_range = df["mfe_so_far_ticks"] - df["mae_so_far_ticks"]
    df["mae_recovery_ratio"] = np.where(
        mae_range.abs() > 0.01,
        (df["unrealized_pnl_ticks"] - df["mae_so_far_ticks"]) / mae_range.clip(lower=0.01),
        0.5,
    )
    df["mae_recovery_ratio"] = df["mae_recovery_ratio"].clip(-2, 2)

    # ── Cross-feature interactions ──
    # Flow agreement: queue_ratio * ofi_alignment
    df["flow_agreement"] = df["queue_ratio"] * df["ofi_alignment"]

    # Pressure momentum: imbalance change * microprice_trend
    df["pressure_momentum"] = df["imbalance_vs_entry"] * df["microprice_trend"]

    # Queue-flow divergence: our side building but flow going against us (danger signal)
    df["queue_flow_divergence"] = df["queue_ratio_change"] * (1 - df["ofi_alignment"])

    # Cancel pressure vs add rate: net queue drain
    df["net_queue_drain"] = df["against_side_cancel_rate"] - df["our_side_add_rate"]

    # ── Spread proxy ──
    # against_side_qty / our_side_qty ratio (inverse of queue_ratio for clarity)
    our_safe = df["our_side_qty"].clip(lower=1.0)
    df["against_our_ratio"] = df["against_side_qty"] / our_safe

    # Trend of against/our ratio: 3-sample moving average vs current
    df["against_our_ratio_ma3"] = (
        df.groupby("trade_idx")["against_our_ratio"]
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )
    df["spread_proxy_trend"] = df["against_our_ratio"] - df["against_our_ratio_ma3"]

    # ── Time-decay of entry confidence ──
    # Signal decays exponentially: entry_confidence * exp(-time / 300)
    df["decayed_confidence"] = df["entry_confidence"] * np.exp(
        -df["time_in_trade_seconds"] / 300.0
    )

    # Confidence-weighted edge: remaining edge expectation weighted by signal freshness
    df["confidence_weighted_pnl"] = df["unrealized_pnl_ticks"] * df["decayed_confidence"]

    # ── Time features (normalized) ──
    df["time_fraction"] = df["time_in_trade_seconds"] / df.groupby("trade_idx")[
        "time_in_trade_seconds"
    ].transform("max").clip(lower=1.0)

    # Count how many features we added
    new_cols = [
        "queue_ratio_velocity", "ofi_acceleration", "microprice_velocity",
        "imbalance_velocity", "drawdown_speed", "gain_speed", "pnl_per_second",
        "mae_recovery_ratio", "flow_agreement", "pressure_momentum",
        "queue_flow_divergence", "net_queue_drain", "against_our_ratio",
        "spread_proxy_trend", "decayed_confidence", "confidence_weighted_pnl",
        "time_fraction",
    ]
    log.info(f"Engineered {len(new_cols)} new features: {new_cols}")

    # Replace inf/nan
    for col in new_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return df


def get_v6_feature_columns() -> List[str]:
    """All feature columns for v6 models (original 27 + 17 new)."""
    # Original features from v2 (excluding labels, identity cols)
    original = [
        "time_in_trade_seconds",
        "unrealized_pnl_ticks",
        "mfe_so_far_ticks",
        "mae_so_far_ticks",
        "drawdown_from_peak",
        "our_side_qty",
        "against_side_qty",
        "queue_ratio",
        "queue_ratio_change",
        "our_side_add_rate",
        "against_side_cancel_rate",
        "ofi_since_entry",
        "ofi_recent_5s",
        "ofi_alignment",
        "imbalance_now",
        "imbalance_vs_entry",
        "microprice_offset_now",
        "microprice_trend",
        "trade_rate_our_side",
        "trade_rate_against_side",
        "level_age_our_side",
        "cancel_spike",
        "our_side_n_orders",
        "against_side_n_orders",
        "entry_confidence",
        "entry_direction",
    ]
    # New v6 features
    new = [
        "queue_ratio_velocity",
        "ofi_acceleration",
        "microprice_velocity",
        "imbalance_velocity",
        "drawdown_speed",
        "gain_speed",
        "pnl_per_second",
        "mae_recovery_ratio",
        "flow_agreement",
        "pressure_momentum",
        "queue_flow_divergence",
        "net_queue_drain",
        "against_our_ratio",
        "spread_proxy_trend",
        "decayed_confidence",
        "confidence_weighted_pnl",
        "time_fraction",
    ]
    return original + new


# ═══════════════════════════════════════════════════════════════════
#  SECTION 2: MLP MODEL DEFINITION
# ═══════════════════════════════════════════════════════════════════


def _build_mlp(n_features: int, device: str = "cpu"):
    """Build a small MLP for continuous edge prediction."""
    _import_torch()

    class EdgeMLP(nn.Module):
        def __init__(self, n_in):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_in, MLP_HIDDEN_1),
                nn.ReLU(),
                nn.Dropout(MLP_DROPOUT),
                nn.Linear(MLP_HIDDEN_1, MLP_HIDDEN_2),
                nn.ReLU(),
                nn.Dropout(MLP_DROPOUT),
                nn.Linear(MLP_HIDDEN_2, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    model = EdgeMLP(n_features).to(device)
    return model


def _train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: str = "cpu",
) -> Tuple[Any, float, List[float]]:
    """Train MLP with early stopping. Returns (model, best_val_mae, loss_history)."""
    _import_torch()

    n_features = X_train.shape[1]
    model = _build_mlp(n_features, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=MLP_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7, min_lr=1e-5
    )
    criterion = nn.SmoothL1Loss()

    # Normalize features (compute on train, apply to both)
    train_mean = np.nanmean(X_train, axis=0)
    train_std = np.nanstd(X_train, axis=0)
    train_std[train_std < 1e-8] = 1.0

    X_tr_norm = (X_train - train_mean) / train_std
    X_v_norm = (X_val - train_mean) / train_std

    X_tr_t = torch.tensor(X_tr_norm, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_v_t = torch.tensor(X_v_norm, dtype=torch.float32, device=device)
    y_v_t = torch.tensor(y_val, dtype=torch.float32, device=device)

    dataset = torch.utils.data.TensorDataset(X_tr_t, y_tr_t)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=MLP_BATCH_SIZE, shuffle=True, drop_last=False
    )

    best_val_mae = float("inf")
    best_state = None
    patience_counter = 0
    loss_history = []

    for epoch in range(MLP_EPOCHS):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        loss_history.append(avg_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_v_t)
            val_mae = torch.mean(torch.abs(val_pred - y_v_t)).item()

        scheduler.step(val_mae)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= MLP_PATIENCE:
            break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # Store normalization params on model for inference
    model._norm_mean = train_mean
    model._norm_std = train_std

    return model, best_val_mae, loss_history


def _predict_mlp(model, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Run MLP inference."""
    _import_torch()
    X_norm = (X - model._norm_mean) / model._norm_std
    X_t = torch.tensor(X_norm, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        preds = model(X_t).cpu().numpy()
    return preds


# ═══════════════════════════════════════════════════════════════════
#  SECTION 3: WALK-FORWARD TRAINING
# ═══════════════════════════════════════════════════════════════════


def train_models_wf(
    samples_df: pd.DataFrame,
) -> Tuple[List[Dict], pd.DataFrame]:
    """
    Walk-forward train BOTH LightGBM regressor and MLP on continuous target.

    20d train / 10d val / 5d slide — SLIDING only (HC #0).
    Target: label_remaining_favorable_ticks (continuous).
    """
    _import_lightgbm()

    feature_cols = get_v6_feature_columns()
    # Filter to columns that actually exist
    feature_cols = [c for c in feature_cols if c in samples_df.columns]
    target_col = "label_remaining_favorable_ticks"

    dates = sorted(samples_df["date"].unique())
    log.info(f"Walk-forward: {len(dates)} dates, {len(feature_cols)} features")
    log.info(f"  WF config: {TRAIN_DAYS}d train / {VAL_DAYS}d val / {SLIDE_DAYS}d slide")
    expected_folds = max(0, (len(dates) - TRAIN_DAYS - VAL_DAYS) // SLIDE_DAYS + 1)
    log.info(f"  Expected folds: ~{expected_folds}")

    # Detect GPU for MLP
    _import_torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"  MLP device: {device}")

    # Output prediction columns
    samples_df["pred_lgbm_edge"] = np.nan
    samples_df["pred_mlp_edge"] = np.nan

    fold_results = []
    fold_idx = 0
    start = 0

    while start + TRAIN_DAYS + VAL_DAYS <= len(dates):
        train_dates = dates[start:start + TRAIN_DAYS]
        val_dates = dates[start + TRAIN_DAYS:start + TRAIN_DAYS + VAL_DAYS]

        train_mask = samples_df["date"].isin(train_dates)
        val_mask = samples_df["date"].isin(val_dates)

        X_train = samples_df.loc[train_mask, feature_cols].values.astype(np.float32)
        y_train = samples_df.loc[train_mask, target_col].values.astype(np.float32)
        X_val = samples_df.loc[val_mask, feature_cols].values.astype(np.float32)
        y_val = samples_df.loc[val_mask, target_col].values.astype(np.float32)

        # Clean NaN/inf
        valid_train = (
            ~np.isnan(y_train) &
            ~np.isinf(y_train) &
            ~np.isnan(X_train).any(axis=1) &
            ~np.isinf(X_train).any(axis=1)
        )
        valid_val = (
            ~np.isnan(y_val) &
            ~np.isinf(y_val) &
            ~np.isnan(X_val).any(axis=1) &
            ~np.isinf(X_val).any(axis=1)
        )

        X_tr = X_train[valid_train]
        y_tr = y_train[valid_train]
        X_v = X_val[valid_val]
        y_v = y_val[valid_val]

        if len(X_tr) < 200 or len(X_v) < 20:
            log.warning(
                f"Fold {fold_idx}: insufficient data "
                f"(train={len(X_tr)}, val={len(X_v)}), skipping"
            )
            start += SLIDE_DAYS
            continue

        # ── LightGBM regressor ──
        dtrain = lgb.Dataset(
            X_tr, label=y_tr, feature_name=feature_cols, free_raw_data=False
        )
        dval = lgb.Dataset(
            X_v, label=y_v, feature_name=feature_cols, free_raw_data=False
        )
        lgbm_model = lgb.train(
            LGBM_REG_PARAMS,
            dtrain,
            num_boost_round=800,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
        )
        lgbm_preds = lgbm_model.predict(X_v)
        lgbm_mae = float(np.mean(np.abs(y_v - lgbm_preds)))
        lgbm_ic = float(np.corrcoef(y_v, lgbm_preds)[0, 1]) if len(y_v) > 5 else 0.0

        # Feature importance
        imp_gain = lgbm_model.feature_importance(importance_type="gain")
        imp_dict = dict(zip(feature_cols, imp_gain.tolist()))

        # ── MLP regressor ──
        mlp_model, mlp_mae, mlp_loss_hist = _train_mlp(X_tr, y_tr, X_v, y_v, device)
        mlp_preds = _predict_mlp(mlp_model, X_v, device)
        mlp_ic = float(np.corrcoef(y_v, mlp_preds)[0, 1]) if len(y_v) > 5 else 0.0

        # Store OOT predictions
        val_indices = samples_df.index[val_mask]
        # Map back to valid indices only
        valid_val_positions = np.where(valid_val)[0]
        for i, pos in enumerate(valid_val_positions):
            if pos < len(val_indices):
                samples_df.loc[val_indices[pos], "pred_lgbm_edge"] = lgbm_preds[i]
                samples_df.loc[val_indices[pos], "pred_mlp_edge"] = mlp_preds[i]

        fold_results.append({
            "fold": fold_idx,
            "train_start": train_dates[0],
            "train_end": train_dates[-1],
            "val_start": val_dates[0],
            "val_end": val_dates[-1],
            "n_train": len(X_tr),
            "n_val": len(X_v),
            "lgbm_mae": lgbm_mae,
            "lgbm_ic": lgbm_ic,
            "mlp_mae": mlp_mae,
            "mlp_ic": mlp_ic,
            "mlp_epochs": len(mlp_loss_hist),
            "feature_importance": imp_dict,
        })

        log.info(
            f"Fold {fold_idx}: "
            f"LGBM MAE={lgbm_mae:.3f} IC={lgbm_ic:.4f} | "
            f"MLP MAE={mlp_mae:.3f} IC={mlp_ic:.4f} | "
            f"val={val_dates[0]}..{val_dates[-1]} "
            f"n_tr={len(X_tr)} n_v={len(X_v)}"
        )

        # Save last fold's models
        if start + SLIDE_DAYS + TRAIN_DAYS + VAL_DAYS > len(dates):
            # Save LGBM
            lgbm_path = MODEL_DIR / "lgbm_edge_latest.txt"
            lgbm_model.save_model(str(lgbm_path))
            # Save MLP
            mlp_path = MODEL_DIR / "mlp_edge_latest.pt"
            torch.save({
                "state_dict": mlp_model.state_dict(),
                "norm_mean": mlp_model._norm_mean,
                "norm_std": mlp_model._norm_std,
                "n_features": len(feature_cols),
                "feature_cols": feature_cols,
            }, str(mlp_path))
            # Save LGBM as pickle too for easy loading
            lgbm_pkl_path = MODEL_DIR / "lgbm_edge_latest.pkl"
            with open(lgbm_pkl_path, "wb") as f:
                pickle.dump(lgbm_model, f)
            log.info(f"Saved final fold models to {MODEL_DIR}")

        fold_idx += 1
        start += SLIDE_DAYS

        del lgbm_model, mlp_model, dtrain, dval
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    log.info(f"\nTraining complete: {fold_idx} folds")
    if fold_results:
        avg_lgbm_mae = np.mean([f["lgbm_mae"] for f in fold_results])
        avg_lgbm_ic = np.mean([f["lgbm_ic"] for f in fold_results])
        avg_mlp_mae = np.mean([f["mlp_mae"] for f in fold_results])
        avg_mlp_ic = np.mean([f["mlp_ic"] for f in fold_results])
        log.info(f"  LGBM avg: MAE={avg_lgbm_mae:.3f}, IC={avg_lgbm_ic:.4f}")
        log.info(f"  MLP  avg: MAE={avg_mlp_mae:.3f}, IC={avg_mlp_ic:.4f}")

    return fold_results, samples_df


# ═══════════════════════════════════════════════════════════════════
#  SECTION 4: SIMULATION — CONTINUOUS EDGE SCORING
# ═══════════════════════════════════════════════════════════════════


def simulate_continuous_strategies(
    samples_df: pd.DataFrame,
) -> Dict:
    """
    Simulate three strategy families using continuous edge predictions:

    (a) Static exit at 30 min (baseline)
    (b) Edge-scaled dynamic: scale position based on predicted remaining edge
    (c) Threshold-based exit: exit when remaining_edge < threshold

    For each, compute Sharpe, Sortino, PF, WR.
    Uses BOTH LGBM and MLP predictions.
    """
    results = {}

    # Get unique trades from samples
    trade_groups = samples_df.groupby("trade_idx")
    trade_ids = sorted(samples_df["trade_idx"].unique())

    # Build per-trade metadata
    trade_meta = []
    for tid in trade_ids:
        grp = trade_groups.get_group(tid)
        trade_meta.append({
            "trade_idx": tid,
            "date": grp["date"].iloc[0],
            "direction": grp["direction"].iloc[0] if "direction" in grp.columns else grp["entry_direction"].iloc[0],
            "final_pnl_ticks": grp["label_final_pnl_ticks"].iloc[0] if "label_final_pnl_ticks" in grp.columns else np.nan,
            "max_time_s": grp["time_in_trade_seconds"].max(),
        })
    meta_df = pd.DataFrame(trade_meta)

    # ── (a) Static baseline ──
    static_pnl = meta_df["final_pnl_ticks"].values - COST_RT_TICKS
    static_pnl = static_pnl[~np.isnan(static_pnl)]
    results["static"] = _compute_strategy_metrics(static_pnl, "static")
    results["static"]["avg_hold_sec"] = float(meta_df["max_time_s"].mean())
    log.info(
        f"STATIC: Sharpe={results['static']['sharpe']:.3f} "
        f"Sortino={results['static']['sortino']:.3f} "
        f"WR={results['static']['win_rate']:.1%} "
        f"PF={results['static']['profit_factor']:.2f} "
        f"mean={results['static']['mean_pnl_ticks']:.2f}t n={results['static']['n_trades']}"
    )

    # Valid samples only (have predictions)
    for model_name, pred_col in [("lgbm", "pred_lgbm_edge"), ("mlp", "pred_mlp_edge")]:
        valid = samples_df.dropna(subset=[pred_col])
        if len(valid) == 0:
            log.warning(f"No valid {model_name} predictions for simulation")
            continue

        log.info(f"\n--- Simulating with {model_name.upper()} predictions ({len(valid):,} samples) ---")

        # ── (b) Edge-scaled position sizing ──
        # Scale position based on predicted remaining edge:
        #   pred >= 3.0 ticks -> hold 1.0x
        #   pred in [1.0, 3.0) -> hold 0.75x
        #   pred in [0.0, 1.0) -> hold 0.5x
        #   pred < 0.0 -> exit (scale to 0)
        for scale_config_name, thresholds in [
            ("conservative", {"full": 3.0, "mid": 1.5, "low": 0.5}),
            ("moderate", {"full": 2.0, "mid": 1.0, "low": 0.0}),
            ("aggressive", {"full": 1.5, "mid": 0.5, "low": -0.5}),
        ]:
            config_key = f"{model_name}_scaled_{scale_config_name}"
            trade_pnls = _simulate_edge_scaled(
                valid, meta_df, pred_col,
                full_thresh=thresholds["full"],
                mid_thresh=thresholds["mid"],
                low_thresh=thresholds["low"],
            )
            if len(trade_pnls) > 0:
                results[config_key] = _compute_strategy_metrics(
                    np.array([t["pnl"] for t in trade_pnls]), config_key
                )
                results[config_key]["avg_hold_sec"] = float(
                    np.mean([t["hold_sec"] for t in trade_pnls])
                )
                results[config_key]["n_scale_events"] = sum(
                    1 for t in trade_pnls if t.get("scaled", False)
                )
                results[config_key]["n_exits"] = sum(
                    1 for t in trade_pnls if t.get("early_exit", False)
                )
                log.info(
                    f"  {config_key}: Sharpe={results[config_key]['sharpe']:.3f} "
                    f"Sortino={results[config_key]['sortino']:.3f} "
                    f"WR={results[config_key]['win_rate']:.1%} "
                    f"PF={results[config_key]['profit_factor']:.2f} "
                    f"exits={results[config_key]['n_exits']}"
                )

        # ── (c) Threshold-based exit ──
        for exit_thresh in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
            config_key = f"{model_name}_thresh_{exit_thresh:.1f}"
            trade_pnls = _simulate_threshold_exit(
                valid, meta_df, pred_col, threshold=exit_thresh
            )
            if len(trade_pnls) > 0:
                results[config_key] = _compute_strategy_metrics(
                    np.array([t["pnl"] for t in trade_pnls]), config_key
                )
                results[config_key]["avg_hold_sec"] = float(
                    np.mean([t["hold_sec"] for t in trade_pnls])
                )
                results[config_key]["exit_rate"] = float(
                    np.mean([1 if t.get("early_exit", False) else 0 for t in trade_pnls])
                )
                log.info(
                    f"  {config_key}: Sharpe={results[config_key]['sharpe']:.3f} "
                    f"Sortino={results[config_key]['sortino']:.3f} "
                    f"WR={results[config_key]['win_rate']:.1%} "
                    f"exit_rate={results[config_key]['exit_rate']:.1%}"
                )

    # Find best config
    best_key = "static"
    best_sharpe = results["static"]["sharpe"]
    for key, val in results.items():
        if key == "static" or not isinstance(val, dict):
            continue
        if val.get("sharpe", -999) > best_sharpe and val.get("n_trades", 0) >= 5:
            best_sharpe = val["sharpe"]
            best_key = key

    results["best_config"] = best_key
    log.info(f"\nBEST CONFIG: {best_key} (Sharpe={best_sharpe:.3f})")
    if best_key != "static":
        improvement = best_sharpe - results["static"]["sharpe"]
        log.info(f"  Sharpe improvement over static: {improvement:+.3f}")

    return results, meta_df


def _simulate_edge_scaled(
    valid_samples: pd.DataFrame,
    meta_df: pd.DataFrame,
    pred_col: str,
    full_thresh: float,
    mid_thresh: float,
    low_thresh: float,
) -> List[Dict]:
    """
    Simulate edge-scaled position sizing.

    At each 10s checkpoint:
      - pred >= full_thresh -> hold 1.0x (no change)
      - pred in [mid_thresh, full_thresh) -> hold 0.75x
      - pred in [low_thresh, mid_thresh) -> hold 0.5x
      - pred < low_thresh -> exit entirely

    The P&L is accumulated in segments with position-size weighting.
    """
    trade_pnls = []

    for _, trade_info in meta_df.iterrows():
        tid = trade_info["trade_idx"]
        final_pnl = trade_info["final_pnl_ticks"]
        if np.isnan(final_pnl):
            continue

        trade_samples = valid_samples[
            valid_samples["trade_idx"] == tid
        ].sort_values("time_in_trade_seconds")

        if len(trade_samples) == 0:
            # No predictions for this trade -> static exit
            pnl = final_pnl - COST_RT_TICKS
            trade_pnls.append({
                "trade_idx": tid,
                "pnl": pnl,
                "hold_sec": trade_info["max_time_s"],
                "scaled": False,
                "early_exit": False,
            })
            continue

        # Walk through samples, tracking position size
        position = 1.0  # start at full size
        prev_unrealized = 0.0
        accumulated_pnl = 0.0
        exit_time = trade_info["max_time_s"]
        early_exit = False
        ever_scaled = False
        scale_cost_incurred = 0.0

        for _, sample in trade_samples.iterrows():
            pred_edge = sample[pred_col]
            unrealized = sample["unrealized_pnl_ticks"]

            if np.isnan(pred_edge):
                continue

            # Determine target position size
            if pred_edge >= full_thresh:
                target_pos = 1.0
            elif pred_edge >= mid_thresh:
                target_pos = 0.75
            elif pred_edge >= low_thresh:
                target_pos = 0.5
            else:
                target_pos = 0.0  # exit

            # If scaling down, accumulate P&L on the portion being exited
            if target_pos < position:
                # Portion being exited
                portion_exiting = position - target_pos
                # P&L on that portion: (current_unrealized - prev_unrealized) * old_position
                # Plus the realized P&L of the exiting portion at current mark
                segment_pnl = (unrealized - prev_unrealized) * position
                accumulated_pnl += segment_pnl
                # Cost of partial exit (market order)
                scale_cost_incurred += portion_exiting * COST_PARTIAL_EXIT_TICKS
                prev_unrealized = unrealized
                position = target_pos
                ever_scaled = True

                if position <= 0:
                    exit_time = sample["time_in_trade_seconds"]
                    early_exit = True
                    break

        # Final settlement: remaining position at end
        if position > 0:
            # Remaining position held to static exit
            final_segment_pnl = (final_pnl - prev_unrealized) * position
            accumulated_pnl += final_segment_pnl
            # Exit cost for remaining position
            scale_cost_incurred += position * COST_RT_TICKS
        # Entry cost (always paid, passive limit)
        entry_cost = 0.376  # commission only for passive entry
        total_cost = entry_cost + scale_cost_incurred

        net_pnl = accumulated_pnl - total_cost

        trade_pnls.append({
            "trade_idx": tid,
            "pnl": net_pnl,
            "hold_sec": exit_time,
            "scaled": ever_scaled,
            "early_exit": early_exit,
        })

    return trade_pnls


def _simulate_threshold_exit(
    valid_samples: pd.DataFrame,
    meta_df: pd.DataFrame,
    pred_col: str,
    threshold: float,
    min_hold_sec: float = 30.0,
) -> List[Dict]:
    """
    Simple threshold exit: if predicted remaining edge < threshold, exit.
    Minimum hold time of 30s to avoid noise exits.
    """
    trade_pnls = []

    for _, trade_info in meta_df.iterrows():
        tid = trade_info["trade_idx"]
        final_pnl = trade_info["final_pnl_ticks"]
        if np.isnan(final_pnl):
            continue

        trade_samples = valid_samples[
            valid_samples["trade_idx"] == tid
        ].sort_values("time_in_trade_seconds")

        if len(trade_samples) == 0:
            pnl = final_pnl - COST_RT_TICKS
            trade_pnls.append({
                "trade_idx": tid,
                "pnl": pnl,
                "hold_sec": trade_info["max_time_s"],
                "early_exit": False,
            })
            continue

        exited_early = False
        exit_pnl = None
        exit_time = trade_info["max_time_s"]

        for _, sample in trade_samples.iterrows():
            t = sample["time_in_trade_seconds"]
            pred_edge = sample[pred_col]

            if np.isnan(pred_edge) or t < min_hold_sec:
                continue

            if pred_edge < threshold:
                # Exit now
                exit_pnl = sample["unrealized_pnl_ticks"] - COST_RT_TICKS
                exit_time = t
                exited_early = True
                break

        if not exited_early:
            exit_pnl = final_pnl - COST_RT_TICKS

        trade_pnls.append({
            "trade_idx": tid,
            "pnl": exit_pnl,
            "hold_sec": exit_time,
            "early_exit": exited_early,
        })

    return trade_pnls


def _compute_strategy_metrics(pnl: np.ndarray, name: str) -> Dict:
    """Compute all performance metrics for a strategy."""
    pnl = pnl[~np.isnan(pnl)]
    if len(pnl) == 0:
        return {"n_trades": 0, "sharpe": 0, "sortino": 0, "win_rate": 0,
                "profit_factor": 0, "mean_pnl_ticks": 0, "total_pnl_ticks": 0,
                "max_drawdown_ticks": 0}
    return {
        "n_trades": len(pnl),
        "mean_pnl_ticks": float(np.mean(pnl)),
        "total_pnl_ticks": float(np.sum(pnl)),
        "sharpe": _compute_sharpe(pnl),
        "sortino": _compute_sortino(pnl),
        "win_rate": float((pnl > 0).mean()),
        "profit_factor": _compute_profit_factor(pnl),
        "max_drawdown_ticks": float(_max_drawdown(pnl)),
    }


# ═══════════════════════════════════════════════════════════════════
#  SECTION 5: FEATURE IMPORTANCE & ANALYSIS
# ═══════════════════════════════════════════════════════════════════


def analyze_feature_importance(fold_results: List[Dict]) -> Dict:
    """Aggregate feature importance across folds."""
    if not fold_results:
        return {}

    all_imp = {}
    for fr in fold_results:
        for feat, imp in fr.get("feature_importance", {}).items():
            if feat not in all_imp:
                all_imp[feat] = []
            all_imp[feat].append(imp)

    avg_imp = {feat: float(np.mean(vals)) for feat, vals in all_imp.items()}
    sorted_imp = sorted(avg_imp.items(), key=lambda x: x[1], reverse=True)

    log.info("\n=== TOP 20 FEATURES (avg gain importance) ===")
    for i, (feat, imp) in enumerate(sorted_imp[:20]):
        marker = " [NEW]" if feat in [
            "queue_ratio_velocity", "ofi_acceleration", "microprice_velocity",
            "imbalance_velocity", "drawdown_speed", "gain_speed", "pnl_per_second",
            "mae_recovery_ratio", "flow_agreement", "pressure_momentum",
            "queue_flow_divergence", "net_queue_drain", "against_our_ratio",
            "spread_proxy_trend", "decayed_confidence", "confidence_weighted_pnl",
            "time_fraction",
        ] else ""
        log.info(f"  {i+1:2d}. {feat:40s} {imp:10.1f}{marker}")

    return dict(sorted_imp)


def analyze_fold_stability(
    fold_results: List[Dict],
    sim_results: Dict,
    samples_df: pd.DataFrame,
    meta_df: pd.DataFrame,
) -> Dict:
    """
    Per-fold Sharpe analysis and segment consistency ratio.
    HC #649: sample size awareness.
    """
    stability = {}

    # Model metrics stability
    if fold_results:
        lgbm_ics = [f["lgbm_ic"] for f in fold_results]
        mlp_ics = [f["mlp_ic"] for f in fold_results]
        lgbm_maes = [f["lgbm_mae"] for f in fold_results]

        stability["lgbm_ic_mean"] = float(np.mean(lgbm_ics))
        stability["lgbm_ic_std"] = float(np.std(lgbm_ics))
        stability["lgbm_ic_min"] = float(np.min(lgbm_ics))
        stability["lgbm_ic_max"] = float(np.max(lgbm_ics))
        stability["mlp_ic_mean"] = float(np.mean(mlp_ics))
        stability["mlp_ic_std"] = float(np.std(mlp_ics))
        stability["lgbm_mae_cv"] = float(np.std(lgbm_maes) / max(np.mean(lgbm_maes), 1e-6))

        log.info("\n=== FOLD STABILITY ===")
        log.info(f"LGBM IC: {stability['lgbm_ic_mean']:.4f} +/- {stability['lgbm_ic_std']:.4f} "
                 f"[{stability['lgbm_ic_min']:.4f}, {stability['lgbm_ic_max']:.4f}]")
        log.info(f"MLP  IC: {stability['mlp_ic_mean']:.4f} +/- {stability['mlp_ic_std']:.4f}")
        log.info(f"LGBM MAE CV: {stability['lgbm_mae_cv']:.3f}")

    # Per-fold Sharpe from the best strategy
    best_key = sim_results.get("best_config", "static")
    if best_key != "static" and "lgbm" in best_key:
        pred_col = "pred_lgbm_edge"
    elif best_key != "static" and "mlp" in best_key:
        pred_col = "pred_mlp_edge"
    else:
        pred_col = None

    # Compute daily Sharpe for consistency ratio
    if "label_final_pnl_ticks" in samples_df.columns:
        # Get per-trade P&L (one per trade, using first sample's final_pnl)
        trade_pnl = samples_df.groupby("trade_idx").first()[
            ["date", "label_final_pnl_ticks"]
        ].reset_index()
        trade_pnl["net_pnl"] = trade_pnl["label_final_pnl_ticks"] - COST_RT_TICKS
        daily_pnl = trade_pnl.groupby("date")["net_pnl"].sum()

        if len(daily_pnl) > 5:
            n_positive_days = (daily_pnl > 0).sum()
            n_total_days = len(daily_pnl)
            stability["daily_consistency"] = float(n_positive_days / n_total_days)
            stability["daily_sharpe"] = float(
                daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)
            ) if daily_pnl.std() > 0 else 0
            log.info(f"Daily consistency: {stability['daily_consistency']:.1%} "
                     f"({n_positive_days}/{n_total_days} positive days)")
            log.info(f"Daily Sharpe: {stability['daily_sharpe']:.3f}")

    # Segment consistency: what fraction of WF folds have positive IC?
    if fold_results:
        n_positive_ic = sum(1 for f in fold_results if f["lgbm_ic"] > 0)
        stability["segment_consistency_lgbm"] = float(n_positive_ic / len(fold_results))
        n_positive_ic_mlp = sum(1 for f in fold_results if f["mlp_ic"] > 0)
        stability["segment_consistency_mlp"] = float(n_positive_ic_mlp / len(fold_results))
        log.info(f"Segment consistency LGBM: {stability['segment_consistency_lgbm']:.1%} "
                 f"({n_positive_ic}/{len(fold_results)} folds IC>0)")
        log.info(f"Segment consistency MLP:  {stability['segment_consistency_mlp']:.1%} "
                 f"({n_positive_ic_mlp}/{len(fold_results)} folds IC>0)")

    return stability


def regime_analysis(samples_df: pd.DataFrame, meta_df: pd.DataFrame) -> Dict:
    """
    Stratify results by day regime. HC #428: reject if regime gap > 0.50.
    """
    if "date" not in meta_df.columns or "final_pnl_ticks" not in meta_df.columns:
        log.warning("Cannot run regime analysis: missing required columns")
        return {}

    # Classify days based on aggregate trade P&L direction (proxy for market regime)
    daily_pnl = meta_df.groupby("date")["final_pnl_ticks"].sum()

    regime_map = {}
    for date, dpnl in daily_pnl.items():
        if dpnl > 2.0:
            regime_map[date] = "green"
        elif dpnl < -2.0:
            regime_map[date] = "red"
        else:
            regime_map[date] = "flat"

    meta = meta_df.copy()
    meta["regime"] = meta["date"].map(regime_map).fillna("flat")

    results = {}
    log.info("\n=== REGIME ANALYSIS (HC #428) ===")
    for regime in ["green", "red", "flat"]:
        regime_trades = meta[meta["regime"] == regime]
        if len(regime_trades) < 3:
            continue
        pnl = regime_trades["final_pnl_ticks"].values - COST_RT_TICKS
        results[regime] = {
            "n_trades": len(regime_trades),
            "n_days": regime_trades["date"].nunique(),
            "sharpe": _compute_sharpe(pnl),
            "sortino": _compute_sortino(pnl),
            "win_rate": float((pnl > 0).mean()),
            "profit_factor": _compute_profit_factor(pnl),
            "mean_pnl_ticks": float(np.mean(pnl)),
        }
        log.info(
            f"  {regime:5s}: Sharpe={results[regime]['sharpe']:.3f}, "
            f"WR={results[regime]['win_rate']:.1%}, "
            f"n={len(regime_trades)} trades, {results[regime]['n_days']} days"
        )

    if "green" in results and "red" in results:
        s_g = results["green"]["sharpe"]
        s_r = results["red"]["sharpe"]
        denom = max(abs(s_g), abs(s_r), 0.01)
        regime_gap = abs(s_g - s_r) / denom
        results["regime_gap"] = float(regime_gap)
        results["regime_pass"] = regime_gap <= 0.50
        log.info(f"  Regime gap: {regime_gap:.2f} ({'PASS' if regime_gap <= 0.50 else 'FAIL'} HC #428)")

    return results


# ═══════════════════════════════════════════════════════════════════
#  SECTION 6: MLFLOW LOGGING
# ═══════════════════════════════════════════════════════════════════


def log_to_mlflow(
    fold_results: List[Dict],
    sim_results: Dict,
    regime_results: Dict,
    feat_importance: Dict,
    stability: Dict,
    n_trades: int,
    n_samples: int,
    n_features: int,
):
    """Log experiment to MLflow."""
    try:
        import mlflow
        mlflow.set_tracking_uri("http://jupiter:5000")
        mlflow.set_experiment("trade_management_v6_continuous")

        with mlflow.start_run(
            run_name=f"tm_v6_cont_{datetime.now().strftime('%Y%m%d_%H%M')}"
        ):
            # Params
            mlflow.log_param("model_type", "LightGBM_regressor + MLP")
            mlflow.log_param("target", "remaining_favorable_ticks (continuous)")
            mlflow.log_param("train_days", TRAIN_DAYS)
            mlflow.log_param("val_days", VAL_DAYS)
            mlflow.log_param("slide_days", SLIDE_DAYS)
            mlflow.log_param("cost_rt_ticks", COST_RT_TICKS)
            mlflow.log_param("n_features", n_features)
            mlflow.log_param("n_trades", n_trades)
            mlflow.log_param("n_samples", n_samples)
            mlflow.log_param("mlp_arch", f"{MLP_HIDDEN_1}/{MLP_HIDDEN_2}")
            mlflow.log_param("mlp_dropout", MLP_DROPOUT)
            mlflow.log_param("mlp_lr", MLP_LR)
            mlflow.log_param("wf_type", "SLIDING (HC #0)")

            # Fold-level metrics
            if fold_results:
                mlflow.log_metric("n_folds", len(fold_results))
                mlflow.log_metric("avg_lgbm_mae", np.mean([f["lgbm_mae"] for f in fold_results]))
                mlflow.log_metric("avg_lgbm_ic", np.mean([f["lgbm_ic"] for f in fold_results]))
                mlflow.log_metric("avg_mlp_mae", np.mean([f["mlp_mae"] for f in fold_results]))
                mlflow.log_metric("avg_mlp_ic", np.mean([f["mlp_ic"] for f in fold_results]))

                # Per-fold ICs
                for i, fr in enumerate(fold_results):
                    mlflow.log_metric(f"lgbm_ic_fold{i}", fr["lgbm_ic"])
                    mlflow.log_metric(f"mlp_ic_fold{i}", fr["mlp_ic"])

            # Static baseline
            if "static" in sim_results:
                st = sim_results["static"]
                mlflow.log_metric("static_sharpe", st["sharpe"])
                mlflow.log_metric("static_sortino", st["sortino"])
                mlflow.log_metric("static_wr", st["win_rate"])
                mlflow.log_metric("static_pf", st["profit_factor"])
                mlflow.log_metric("static_mean_pnl", st["mean_pnl_ticks"])

            # Best dynamic config
            best_key = sim_results.get("best_config", "static")
            mlflow.log_param("best_config", best_key)
            if best_key != "static" and best_key in sim_results:
                best = sim_results[best_key]
                mlflow.log_metric("best_dynamic_sharpe", best["sharpe"])
                mlflow.log_metric("best_dynamic_sortino", best["sortino"])
                mlflow.log_metric("best_dynamic_wr", best["win_rate"])
                mlflow.log_metric("best_dynamic_pf", best["profit_factor"])
                mlflow.log_metric("best_dynamic_mean_pnl", best["mean_pnl_ticks"])
                if "static" in sim_results:
                    mlflow.log_metric(
                        "sharpe_improvement",
                        best["sharpe"] - sim_results["static"]["sharpe"],
                    )

            # Stability metrics
            for k, v in stability.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"stability_{k}", v)

            # Regime results
            if regime_results:
                if "regime_gap" in regime_results:
                    mlflow.log_metric("regime_gap", regime_results["regime_gap"])
                    mlflow.log_param("regime_pass", regime_results.get("regime_pass", False))

            # Save full results as artifact
            results_path = OUTPUT_DIR / "full_results.json"

            # Sanitize fold_results (remove feature_importance for JSON)
            clean_folds = []
            for fr in fold_results:
                clean = {k: v for k, v in fr.items() if k != "feature_importance"}
                clean_folds.append(clean)

            # Sanitize sim_results (ensure all values are JSON-serializable)
            clean_sim = {}
            for k, v in sim_results.items():
                if isinstance(v, dict):
                    clean_sim[k] = {
                        sk: float(sv) if isinstance(sv, (np.floating, np.integer)) else sv
                        for sk, sv in v.items()
                    }
                else:
                    clean_sim[k] = v

            with open(results_path, "w") as f:
                json.dump({
                    "fold_results": clean_folds,
                    "sim_results": clean_sim,
                    "regime_results": regime_results,
                    "feature_importance": feat_importance,
                    "stability": stability,
                }, f, indent=2, default=str)
            mlflow.log_artifact(str(results_path))

            # Save feature importance as artifact
            fi_path = OUTPUT_DIR / "feature_importance.json"
            with open(fi_path, "w") as f:
                json.dump(feat_importance, f, indent=2)
            mlflow.log_artifact(str(fi_path))

            # Log model artifacts
            for model_file in MODEL_DIR.glob("*"):
                try:
                    mlflow.log_artifact(str(model_file), "models")
                except Exception:
                    pass

            log.info("MLflow logging complete")

    except Exception as e:
        log.warning(f"MLflow logging failed: {e}")
        import traceback
        traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════


def _compute_sharpe(pnl: np.ndarray) -> float:
    pnl = pnl[~np.isnan(pnl)]
    if len(pnl) < 2 or np.std(pnl) == 0:
        return 0.0
    return float(np.mean(pnl) / np.std(pnl) * np.sqrt(252))


def _compute_sortino(pnl: np.ndarray) -> float:
    pnl = pnl[~np.isnan(pnl)]
    if len(pnl) < 2:
        return 0.0
    downside = pnl[pnl < 0]
    if len(downside) < 2 or np.std(downside) == 0:
        return float(np.mean(pnl) * np.sqrt(252)) if np.mean(pnl) > 0 else 0.0
    return float(np.mean(pnl) / np.std(downside) * np.sqrt(252))


def _compute_profit_factor(pnl: np.ndarray) -> float:
    pnl = pnl[~np.isnan(pnl)]
    gross_profit = pnl[pnl > 0].sum()
    gross_loss = abs(pnl[pnl < 0].sum())
    if gross_loss == 0:
        return 99.0 if gross_profit > 0 else 0.0
    return float(gross_profit / gross_loss)


def _max_drawdown(pnl: np.ndarray) -> float:
    pnl = pnl[~np.isnan(pnl)]
    if len(pnl) == 0:
        return 0.0
    cumsum = np.cumsum(pnl)
    peak = np.maximum.accumulate(cumsum)
    dd = peak - cumsum
    return float(dd.max()) if len(dd) > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════


def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("Trade Management v6 — Continuous Edge Scoring")
    log.info("=" * 70)
    log.info(f"Input:  {INPUT_DIR}")
    log.info(f"Output: {OUTPUT_DIR}")

    # ── Step 1: Load pre-built tick samples ──
    log.info("\n>>> STEP 1: Loading tick samples from v2 <<<")
    samples_df = load_tick_samples()

    # Validate expected columns
    required_cols = ["trade_idx", "date", "time_in_trade_seconds",
                     "label_remaining_favorable_ticks"]
    missing = [c for c in required_cols if c not in samples_df.columns]
    if missing:
        log.error(f"Missing required columns: {missing}")
        return

    n_trades = samples_df["trade_idx"].nunique()
    n_dates = samples_df["date"].nunique()
    log.info(f"  {len(samples_df):,} samples, {n_trades} trades, {n_dates} dates")

    # ── Step 2: Feature engineering ──
    log.info("\n>>> STEP 2: Engineering v6 features <<<")
    samples_df = engineer_v6_features(samples_df)

    feature_cols = get_v6_feature_columns()
    available_features = [c for c in feature_cols if c in samples_df.columns]
    missing_features = [c for c in feature_cols if c not in samples_df.columns]
    if missing_features:
        log.warning(f"Missing features (will be excluded): {missing_features}")
    log.info(f"Using {len(available_features)} features for modeling")

    # Save engineered dataset
    samples_df.to_parquet(OUTPUT_DIR / "tick_samples_v6.parquet", index=False)
    log.info(f"Saved engineered dataset: {OUTPUT_DIR / 'tick_samples_v6.parquet'}")

    # ── Step 3: Walk-forward training ──
    log.info("\n>>> STEP 3: Walk-forward training (LGBM + MLP) <<<")
    fold_results, samples_df = train_models_wf(samples_df)

    if not fold_results:
        log.error("No WF folds completed! Insufficient data.")
        return

    # ── Step 4: Feature importance ──
    log.info("\n>>> STEP 4: Feature importance analysis <<<")
    feat_importance = analyze_feature_importance(fold_results)

    # ── Step 5: Simulation ──
    log.info("\n>>> STEP 5: Simulating continuous edge strategies <<<")
    sim_results, meta_df = simulate_continuous_strategies(samples_df)

    # ── Step 6: Fold stability ──
    log.info("\n>>> STEP 6: Walk-forward stability analysis <<<")
    stability = analyze_fold_stability(fold_results, sim_results, samples_df, meta_df)

    # ── Step 7: Regime analysis ──
    log.info("\n>>> STEP 7: Regime analysis (HC #428) <<<")
    regime_results = regime_analysis(samples_df, meta_df)

    # ── Step 8: MLflow logging ──
    log.info("\n>>> STEP 8: Logging to MLflow <<<")
    log_to_mlflow(
        fold_results, sim_results, regime_results,
        feat_importance, stability,
        n_trades=n_trades,
        n_samples=len(samples_df),
        n_features=len(available_features),
    )

    # Save predictions
    pred_cols = ["trade_idx", "date", "time_in_trade_seconds",
                 "pred_lgbm_edge", "pred_mlp_edge",
                 "label_remaining_favorable_ticks"]
    pred_cols = [c for c in pred_cols if c in samples_df.columns]
    samples_df[pred_cols].to_parquet(
        OUTPUT_DIR / "predictions_oot.parquet", index=False
    )

    # ── Final summary ──
    elapsed = time.time() - t0
    log.info("\n" + "=" * 70)
    log.info("FINAL SUMMARY — Trade Management v6 Continuous Edge")
    log.info("=" * 70)
    log.info(f"Trades: {n_trades} | Samples: {len(samples_df):,} | Features: {len(available_features)}")
    log.info(f"WF folds: {len(fold_results)} ({TRAIN_DAYS}d/{VAL_DAYS}d/{SLIDE_DAYS}d)")

    if fold_results:
        avg_lgbm_ic = np.mean([f["lgbm_ic"] for f in fold_results])
        avg_mlp_ic = np.mean([f["mlp_ic"] for f in fold_results])
        avg_lgbm_mae = np.mean([f["lgbm_mae"] for f in fold_results])
        avg_mlp_mae = np.mean([f["mlp_mae"] for f in fold_results])
        log.info(f"LGBM: IC={avg_lgbm_ic:.4f}, MAE={avg_lgbm_mae:.3f}")
        log.info(f"MLP:  IC={avg_mlp_ic:.4f}, MAE={avg_mlp_mae:.3f}")

    if "static" in sim_results:
        st = sim_results["static"]
        log.info(f"\nStatic baseline: Sharpe={st['sharpe']:.3f}, "
                 f"Sortino={st['sortino']:.3f}, "
                 f"WR={st['win_rate']:.1%}, PF={st['profit_factor']:.2f}")

    best_key = sim_results.get("best_config", "static")
    if best_key != "static" and best_key in sim_results:
        best = sim_results[best_key]
        log.info(f"Best dynamic ({best_key}):")
        log.info(f"  Sharpe={best['sharpe']:.3f}, Sortino={best['sortino']:.3f}, "
                 f"WR={best['win_rate']:.1%}, PF={best['profit_factor']:.2f}")
        improvement = best["sharpe"] - sim_results["static"]["sharpe"]
        log.info(f"  Sharpe improvement over static: {improvement:+.3f}")
    else:
        log.info("No dynamic config beat static exit")

    if stability:
        log.info(f"\nStability:")
        if "segment_consistency_lgbm" in stability:
            log.info(f"  Segment consistency LGBM: {stability['segment_consistency_lgbm']:.1%}")
        if "segment_consistency_mlp" in stability:
            log.info(f"  Segment consistency MLP:  {stability['segment_consistency_mlp']:.1%}")
        if "daily_consistency" in stability:
            log.info(f"  Daily consistency: {stability['daily_consistency']:.1%}")

    if regime_results and "regime_gap" in regime_results:
        log.info(f"\nRegime gap: {regime_results['regime_gap']:.2f} "
                 f"({'PASS' if regime_results['regime_pass'] else 'FAIL'} HC #428)")

    log.info(f"\nCompleted in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    log.info(f"Results saved to {OUTPUT_DIR}")

    # Save summary JSON
    summary = {
        "completed_at": datetime.now().isoformat(),
        "elapsed_seconds": elapsed,
        "n_trades": n_trades,
        "n_samples": len(samples_df),
        "n_features": len(available_features),
        "n_folds": len(fold_results),
        "avg_lgbm_ic": float(np.mean([f["lgbm_ic"] for f in fold_results])) if fold_results else 0,
        "avg_mlp_ic": float(np.mean([f["mlp_ic"] for f in fold_results])) if fold_results else 0,
        "static_sharpe": sim_results.get("static", {}).get("sharpe", 0),
        "best_config": best_key,
        "best_dynamic_sharpe": sim_results.get(best_key, {}).get("sharpe", 0) if best_key != "static" else None,
        "regime_pass": regime_results.get("regime_pass", None),
    }
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    log.info("Done.")


if __name__ == "__main__":
    main()
