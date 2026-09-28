#!/usr/bin/env python3
"""
optuna_exec_mlp_v2.py — Optuna HP Optimization for Execution MLP v2 (Both Sides)
==================================================================================
Runs on Neptune RTX 3090 (24GB VRAM, Ubuntu).

Key enhancements over v1:
  1. Trains on BOTH long and short signals (not short-only).
     Evaluates per-side (long vs short) AND combined metrics.
  2. Signal flip features — detect high-confidence signal reversals
     (e.g. signal going from +2sigma to -1.5sigma).
  3. Lower gate thresholds explored: [0.30..0.60] to capture
     top 10-20% of trades, not just top 1-5%.
  4. Cancel timing features: signal decay rate, time since signal peak.
  5. Spread condition features: spread_regime, signal*spread interaction.
  6. Optuna optimizes: lr, hidden_dim, n_layers, dropout, batch_size,
     weight_decay, gate_loss_weight, label_smoothing PLUS new params:
     signal_flip_threshold, cancel_decay_window.

Cost basis: 0.376 ticks RT ($4.70 AMP/Rithmic, HC #52).
All results are MIDPOINT-BASED (no FIFO fill sim, HC #57).
Reports Sharpe AND Sortino. Separate metrics for long-side and short-side.
Sliding window walk-forward (HC #0).
MLflow logging mandatory for each trial.
"""

import os
import sys
import gc
import time
import json
import logging
import argparse
import warnings
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Add project root and local directory for imports
LVL3_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LVL3_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import everything we need from the base training script
from train_exec_mlp_gpu import (
    ExecMLP,
    ExecDataset,
    load_fold_data,
    build_all_features,
    build_targets,
    evaluate_gated_trading,
    compute_sharpe,
    compute_sortino,
    COST_TICKS_RT,
    TICK_VAL,
    TRAIN_FOLDS,
    VAL_FOLDS,
    N_FOLDS,
    NUM_WORKERS,
    PIN_MEMORY,
    DEVICE,
)

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# ============================================================
# Logging
# ============================================================
logging.basicConfig(
    force=True,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ============================================================
# Constants
# ============================================================
OUTPUT_DIR = Path("/home/nick/Lvl3Quant/output/optuna_exec_mlp_v2")
MLFLOW_URI = "http://localhost:5000"
EXPERIMENT_NAME = "optuna_exec_mlp_v2_both_sides"

# Optuna config
N_TRIALS = 100
MAX_EPOCHS = 60
PATIENCE = 15           # early stopping patience per trial
PRUNER_STARTUP = 15     # MedianPruner starts after this many epochs

# Lower gate thresholds — explore top 10-20% not just top 1-5%
GATE_THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]


# ============================================================
# New Feature Engineering (appended to base features)
# ============================================================

def compute_signal_flip_features(
    predictions: np.ndarray,
    flip_threshold: float = 1.5,
    decay_window: int = 20,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute signal flip, decay, and cancel-timing features.

    These features capture signal dynamics that matter for execution:
    - Signal reversals indicate regime changes (don't enter on a fading signal)
    - Decay rate tells us how fast the signal is losing strength
    - Time since peak helps decide when to cancel a pending order

    Args:
        predictions: Raw signal predictions array (n_samples,) — typically pred_10s.
        flip_threshold: Z-score threshold for detecting signal flips (Optuna-tuned).
        decay_window: Lookback window for decay rate computation (Optuna-tuned).

    Returns:
        (feature_matrix, feature_names) — n_samples x n_new_features
    """
    n = len(predictions)
    features = []
    names = []

    # --- 1. Signal flip strength ---
    # Detect when signal crosses from one extreme to the opposite.
    # Compute rolling z-score of predictions, then detect crossings.
    # A flip from +2sigma to -1.5sigma is a strong reversal signal.
    if n > decay_window:
        # Rolling mean and std for z-score
        rolling_mean = np.full(n, np.nan)
        rolling_std = np.full(n, np.nan)
        for i in range(decay_window, n):
            window = predictions[i - decay_window:i]
            rolling_mean[i] = window.mean()
            rolling_std[i] = window.std()
        # Fill initial values with global stats
        global_mean = predictions.mean()
        global_std = max(predictions.std(), 1e-8)
        rolling_mean[:decay_window] = global_mean
        rolling_std[:decay_window] = global_std
        rolling_std = np.where(rolling_std < 1e-8, global_std, rolling_std)

        zscore = (predictions - rolling_mean) / rolling_std

        # Signal flip strength: change in z-score (large negative = flip from bullish to bearish)
        zscore_diff = np.diff(zscore, prepend=zscore[0])
        signal_flip_strength = np.abs(zscore_diff)

        # Binary flip indicator: did z-score cross the threshold magnitude?
        flip_occurred = np.zeros(n, dtype=np.float32)
        for i in range(1, n):
            # Flip from positive extreme to negative (or vice versa)
            if (zscore[i - 1] > flip_threshold and zscore[i] < -flip_threshold * 0.75) or \
               (zscore[i - 1] < -flip_threshold and zscore[i] > flip_threshold * 0.75):
                flip_occurred[i] = 1.0
    else:
        signal_flip_strength = np.zeros(n, dtype=np.float32)
        flip_occurred = np.zeros(n, dtype=np.float32)
        zscore = np.zeros(n, dtype=np.float32)

    features.append(signal_flip_strength.astype(np.float32))
    names.append("signal_flip_strength")
    features.append(flip_occurred.astype(np.float32))
    names.append("signal_flip_occurred")

    # --- 2. Signal decay rate ---
    # Rate of signal magnitude decrease: d(|signal|)/dt
    # Negative decay rate = signal is fading = bad time to enter
    abs_signal = np.abs(predictions)
    if n > 1:
        # Use a short window for smoothed derivative
        decay_rate = np.zeros(n, dtype=np.float32)
        short_window = min(5, n - 1)
        for i in range(short_window, n):
            # Linear slope of |signal| over last short_window points
            y = abs_signal[i - short_window:i + 1]
            x = np.arange(len(y), dtype=np.float32)
            # Simple slope: (y[-1] - y[0]) / window_size
            decay_rate[i] = (y[-1] - y[0]) / short_window
        # Fill initial values
        decay_rate[:short_window] = 0.0
    else:
        decay_rate = np.zeros(n, dtype=np.float32)

    features.append(decay_rate)
    names.append("signal_decay_rate")

    # --- 3. Time since signal peak ---
    # How many events since |signal| was at its local maximum?
    # Larger values = signal peaked a while ago = consider canceling
    time_since_peak = np.zeros(n, dtype=np.float32)
    running_max = abs_signal[0] if n > 0 else 0
    last_peak_idx = 0
    for i in range(n):
        if abs_signal[i] >= running_max:
            running_max = abs_signal[i]
            last_peak_idx = i
        time_since_peak[i] = i - last_peak_idx
        # Reset running max periodically to track local peaks
        if i - last_peak_idx > decay_window * 2:
            running_max = abs_signal[i]
            last_peak_idx = i

    # Normalize to [0, 1] range using decay_window as reference
    time_since_peak_norm = np.clip(time_since_peak / max(decay_window, 1), 0, 5).astype(np.float32)
    features.append(time_since_peak_norm)
    names.append("time_since_signal_peak")

    # --- 4. Signal z-score (useful as a feature itself) ---
    features.append(zscore.astype(np.float32))
    names.append("signal_zscore")

    return np.column_stack(features), names


def compute_spread_features(
    features_matrix: np.ndarray,
    feature_names: List[str],
    predictions: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute spread regime and signal-spread interaction features.

    Args:
        features_matrix: Existing feature matrix from build_all_features.
        feature_names: Names of existing features.
        predictions: Raw signal predictions (pred_10s).

    Returns:
        (new_features, new_names) to append to existing features.
    """
    n = len(features_matrix)
    new_features = []
    new_names = []

    # Find spread_zscore in existing features
    spread_zscore_idx = None
    for i, name in enumerate(feature_names):
        if name == "spread_zscore":
            spread_zscore_idx = i
            break

    if spread_zscore_idx is not None:
        spread_zscore = features_matrix[:, spread_zscore_idx]
    else:
        # If spread_zscore not found, create a dummy (shouldn't happen with MBO data)
        logger.warning("spread_zscore not found in features — using zeros")
        spread_zscore = np.zeros(n, dtype=np.float32)

    # --- 1. Spread regime ---
    # Categorize: tight (<-0.5 z), normal (-0.5 to 0.5), wide (0.5 to 1.5), very_wide (>1.5)
    # Encoded as ordinal: 0=tight, 1=normal, 2=wide, 3=very_wide
    spread_regime = np.zeros(n, dtype=np.float32)
    spread_regime = np.where(spread_zscore < -0.5, 0.0, spread_regime)  # tight
    spread_regime = np.where((spread_zscore >= -0.5) & (spread_zscore < 0.5), 1.0, spread_regime)  # normal
    spread_regime = np.where((spread_zscore >= 0.5) & (spread_zscore < 1.5), 2.0, spread_regime)  # wide
    spread_regime = np.where(spread_zscore >= 1.5, 3.0, spread_regime)  # very wide

    # Normalize to [0, 1]
    spread_regime_norm = (spread_regime / 3.0).astype(np.float32)
    new_features.append(spread_regime_norm)
    new_names.append("spread_regime")

    # --- 2. Signal-spread interaction ---
    # Stronger signal + tighter spread = better execution opportunity
    # signal_strength * (1 / (1 + spread_zscore_clipped)) — higher when spread is tight
    signal_strength = np.abs(predictions)
    # Clip spread_zscore to avoid division issues, shift so tight spread = high value
    spread_inv = 1.0 / (1.0 + np.clip(spread_zscore, -1.0, 5.0))
    signal_spread_interaction = (signal_strength * spread_inv).astype(np.float32)
    new_features.append(signal_spread_interaction)
    new_names.append("signal_spread_interaction")

    # --- 3. Tight spread indicator (binary) ---
    # When spread is tight, execution is cheaper — this is a direct go/no-go signal
    tight_spread = (spread_zscore < -0.3).astype(np.float32)
    new_features.append(tight_spread)
    new_names.append("tight_spread_indicator")

    return np.column_stack(new_features), new_names


# ============================================================
# Per-side evaluation
# ============================================================

def evaluate_per_side(
    gate_probs: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> Dict[str, dict]:
    """
    Evaluate gated trading separately for long and short signals.

    Returns dict with keys: 'combined', 'long', 'short', each containing
    the standard metrics from evaluate_gated_trading.
    """
    results = {}

    # Combined (both sides)
    results['combined'] = evaluate_gated_trading(gate_probs, predictions, labels, threshold=threshold)

    # Use 10s prediction (column 2) for direction if multi-column
    if predictions.ndim > 1:
        pred_direction = predictions[:, 2]
    else:
        pred_direction = predictions

    # Long side: predictions > 0
    long_mask = pred_direction > 0
    if long_mask.sum() >= 10:
        results['long'] = evaluate_gated_trading(
            gate_probs[long_mask], predictions[long_mask],
            labels[long_mask], threshold=threshold,
        )
    else:
        results['long'] = {'n_trades': 0, 'sharpe': 0.0, 'sortino': 0.0, 'win_rate': 0.0, 'pf': 0.0, 'avg_pnl': 0.0}

    # Short side: predictions < 0
    short_mask = pred_direction < 0
    if short_mask.sum() >= 10:
        results['short'] = evaluate_gated_trading(
            gate_probs[short_mask], predictions[short_mask],
            labels[short_mask], threshold=threshold,
        )
    else:
        results['short'] = {'n_trades': 0, 'sharpe': 0.0, 'sortino': 0.0, 'win_rate': 0.0, 'pf': 0.0, 'avg_pnl': 0.0}

    return results


# ============================================================
# Data cache (loaded once, reused across trials)
# ============================================================
_DATA_CACHE = {}


def load_and_cache_data(
    signal_flip_threshold: float = 1.5,
    cancel_decay_window: int = 20,
) -> dict:
    """
    Load all fold data, build features (base + new v2 features) and targets.
    Cache globally. Re-creates if flip/decay params change.

    The v2-specific features are appended AFTER base features so that
    the ExecMLP input_dim grows but the architecture stays the same.
    """
    global _DATA_CACHE
    cache_key = f"{signal_flip_threshold:.2f}_{cancel_decay_window}"
    if _DATA_CACHE and _DATA_CACHE.get('_cache_key') == cache_key:
        return _DATA_CACHE

    logger.info("=" * 80)
    logger.info(f"Loading and caching fold data (flip_thresh={signal_flip_threshold:.2f}, "
                f"decay_window={cancel_decay_window})")
    logger.info("=" * 80)

    all_data = {}
    for fold in range(N_FOLDS):
        d = load_fold_data(fold)
        if d is not None:
            all_data[fold] = d
            has_mbo = "yes" if d['mbo_data'] is not None else "no"
            has_vol = "yes" if d['vol_pred'] is not None else "no"
            logger.info(f"  Fold {fold:02d}: date={d['date']}, n={d['n_samples']:,}, MBO={has_mbo}, vol={has_vol}")

    if len(all_data) < 3:
        raise RuntimeError("Not enough folds loaded. Need at least 3.")

    # Build features for all folds — base + v2 enhancements
    logger.info("\nBuilding features (base + v2 enhancements)...")
    fold_features = {}
    fold_targets = {}
    feature_names = None

    for fold_idx, data in all_data.items():
        logger.info(f"  Fold {fold_idx:02d} ({data['date']}):")

        # Base features from train_exec_mlp_gpu
        feats, names = build_all_features(data)
        tgts = build_targets(data)

        # Get raw predictions for signal-based features
        preds = data['predictions']  # (N, 3) = 1s/5s/10s
        pred_10s = preds[:, 2]  # Use 10s horizon for flip/decay features

        # V2 Feature 1: Signal flip + decay + cancel timing features
        flip_feats, flip_names = compute_signal_flip_features(
            pred_10s,
            flip_threshold=signal_flip_threshold,
            decay_window=cancel_decay_window,
        )

        # V2 Feature 2: Spread regime + signal-spread interaction
        spread_feats, spread_names = compute_spread_features(feats, names, pred_10s)

        # Append new features to base
        feats = np.concatenate([feats, flip_feats, spread_feats], axis=1)
        if feature_names is None:
            feature_names = names + flip_names + spread_names
            logger.info(f"    Base features: {len(names)}, "
                        f"Flip features: {len(flip_names)}, "
                        f"Spread features: {len(spread_names)}, "
                        f"Total: {len(feature_names)}")

        fold_features[fold_idx] = feats
        fold_targets[fold_idx] = tgts

    # Concatenate train folds
    train_X_parts = []
    train_tgt_parts = {k: [] for k in ['gate', 'mfe', 'mae', 'hold_time']}
    for fold_idx in TRAIN_FOLDS:
        if fold_idx not in fold_features:
            continue
        train_X_parts.append(fold_features[fold_idx])
        for k in train_tgt_parts:
            train_tgt_parts[k].append(fold_targets[fold_idx][k])

    train_X = np.concatenate(train_X_parts)
    train_tgts = {k: np.concatenate(v) for k, v in train_tgt_parts.items()}

    # Concatenate val folds
    val_X_parts = []
    val_tgt_parts = {k: [] for k in ['gate', 'mfe', 'mae', 'hold_time']}
    val_preds_parts = []
    val_labels_parts = []
    for fold_idx in VAL_FOLDS:
        if fold_idx not in fold_features:
            continue
        val_X_parts.append(fold_features[fold_idx])
        for k in val_tgt_parts:
            val_tgt_parts[k].append(fold_targets[fold_idx][k])
        val_preds_parts.append(all_data[fold_idx]['predictions'])
        val_labels_parts.append(all_data[fold_idx]['labels'])

    val_X = np.concatenate(val_X_parts)
    val_tgts = {k: np.concatenate(v) for k, v in val_tgt_parts.items()}
    val_preds = np.concatenate(val_preds_parts)
    val_labels = np.concatenate(val_labels_parts)

    # Precompute normalization from train set
    mean = train_X.mean(axis=0)
    std = train_X.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    train_norm = (train_X - mean) / std
    val_norm = (val_X - mean) / std

    # Replace any NaN/Inf with 0 (safety for new features)
    train_norm = np.nan_to_num(train_norm, nan=0.0, posinf=0.0, neginf=0.0)
    val_norm = np.nan_to_num(val_norm, nan=0.0, posinf=0.0, neginf=0.0)

    logger.info(f"\nTrain samples: {len(train_X):,}")
    logger.info(f"Val samples:   {len(val_X):,}")
    logger.info(f"Features:      {train_X.shape[1]} (base + v2)")

    # Per-side stats (use 10s prediction column for direction)
    val_pred_10s = val_preds[:, 2] if val_preds.ndim > 1 else val_preds
    logger.info(f"Val long signals:  {(val_pred_10s > 0).sum():,}")
    logger.info(f"Val short signals: {(val_pred_10s < 0).sum():,}")

    _DATA_CACHE = {
        'train_norm': train_norm,
        'train_tgts': train_tgts,
        'val_norm': val_norm,
        'val_tgts': val_tgts,
        'val_preds': val_preds,
        'val_labels': val_labels,
        'feature_names': feature_names,
        'input_dim': train_X.shape[1],
        'norm_mean': mean,
        'norm_std': std,
        'gate_positive_ratio': train_tgts['gate'].mean(),
        'signal_flip_threshold': signal_flip_threshold,
        'cancel_decay_window': cancel_decay_window,
        '_cache_key': cache_key,
    }
    return _DATA_CACHE


# ============================================================
# Optuna Objective
# ============================================================

def objective(trial: optuna.Trial) -> float:
    """
    Single Optuna trial: train ExecMLP with sampled hyperparameters,
    return best validation Sharpe across gate thresholds.

    Trains on BOTH long and short signals. Reports per-side metrics.
    """

    # --- Sample hyperparameters ---
    # Standard params (same as v1)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    hidden_dim = trial.suggest_int("hidden_dim", 128, 512, step=64)
    n_layers = trial.suggest_int("n_layers", 2, 6)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    batch_size = trial.suggest_int("batch_size", 512, 4096, step=256)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    gate_loss_weight = trial.suggest_float("gate_loss_weight", 0.3, 0.7)
    label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2)

    # NEW v2 params: signal flip detection and cancel timing
    signal_flip_threshold = trial.suggest_float("signal_flip_threshold", 1.0, 3.0, step=0.25)
    cancel_decay_window = trial.suggest_int("cancel_decay_window", 10, 50, step=5)

    # Load data with the v2-specific feature params
    # Note: data will be rebuilt if flip/decay params differ from cache
    data = load_and_cache_data(
        signal_flip_threshold=signal_flip_threshold,
        cancel_decay_window=cancel_decay_window,
    )

    trial_name = f"trial_{trial.number:03d}"
    logger.info(f"\n{'='*70}")
    logger.info(f"Trial {trial.number}: lr={lr:.2e}, hidden={hidden_dim}, layers={n_layers}, "
                f"drop={dropout:.2f}, bs={batch_size}, wd={weight_decay:.2e}, "
                f"gate_w={gate_loss_weight:.2f}, ls={label_smoothing:.2f}")
    logger.info(f"  v2 params: flip_thresh={signal_flip_threshold:.2f}, "
                f"decay_window={cancel_decay_window}")
    logger.info(f"{'='*70}")

    # --- MLflow logging for this trial ---
    mlflow_run = None
    if MLFLOW_AVAILABLE:
        try:
            mlflow_run = mlflow.start_run(
                run_name=f"optuna_v2_{trial_name}_{time.strftime('%H%M%S')}",
                nested=True,
            )
            mlflow.log_params({
                'trial_number': trial.number,
                'lr': lr,
                'hidden_dim': hidden_dim,
                'n_layers': n_layers,
                'dropout': dropout,
                'batch_size': batch_size,
                'weight_decay': weight_decay,
                'gate_loss_weight': gate_loss_weight,
                'label_smoothing': label_smoothing,
                'signal_flip_threshold': signal_flip_threshold,
                'cancel_decay_window': cancel_decay_window,
                'max_epochs': MAX_EPOCHS,
                'cost_ticks_rt': COST_TICKS_RT,
                'execution_basis': 'midpoint (no FIFO fill sim)',
                'signal_sides': 'both (long + short)',
            })
        except Exception as e:
            logger.warning(f"MLflow trial logging failed: {e}")

    try:
        best_sharpe = _run_trial(
            trial, data, lr, hidden_dim, n_layers, dropout,
            batch_size, weight_decay, gate_loss_weight, label_smoothing,
        )
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        best_sharpe = float('-inf')
    finally:
        if mlflow_run is not None:
            try:
                mlflow.log_metrics({
                    'final_best_sharpe': best_sharpe if best_sharpe > float('-inf') else -999,
                })
                mlflow.end_run()
            except Exception:
                pass

    gc.collect()
    torch.cuda.empty_cache()
    return best_sharpe


def _run_trial(
    trial: optuna.Trial,
    data: dict,
    lr: float,
    hidden_dim: int,
    n_layers: int,
    dropout: float,
    batch_size: int,
    weight_decay: float,
    gate_loss_weight: float,
    label_smoothing: float,
) -> float:
    """Execute a single trial's training loop. Returns best val Sharpe."""

    input_dim = data['input_dim']

    # Create datasets — trains on ALL signals (both long and short)
    train_ds = ExecDataset(data['train_norm'], data['train_tgts'])
    val_ds = ExecDataset(data['val_norm'], data['val_tgts'])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    )

    # Model
    model = ExecMLP(input_dim, hidden_dim, n_layers, dropout).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Model params: {n_params:,} (input_dim={input_dim})")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    # Loss functions
    huber_loss = nn.SmoothL1Loss()

    # Gate class weight
    gate_positive_ratio = data['gate_positive_ratio']
    gate_class_weight = max(0.5, min(2.0, (1 - gate_positive_ratio) / max(gate_positive_ratio, 0.01)))

    # Apply label smoothing to gate targets
    def smooth_gate_target(target: torch.Tensor) -> torch.Tensor:
        if label_smoothing > 0:
            return target * (1.0 - label_smoothing) + 0.5 * label_smoothing
        return target

    # Early stopping
    best_val_sharpe = float('-inf')
    best_state = None
    best_epoch = 0
    no_improve = 0

    for epoch in range(MAX_EPOCHS):
        # === Training ===
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            feats = batch['features'].to(DEVICE)
            gate_tgt = smooth_gate_target(batch['gate'].to(DEVICE))
            mfe_tgt = batch['mfe'].to(DEVICE)
            mae_tgt = batch['mae'].to(DEVICE)
            hold_tgt = batch['hold_time'].to(DEVICE)

            outputs = model(feats)

            # Gate loss: weighted BCE with label smoothing applied to targets
            pos_weight = torch.where(gate_tgt > 0.5, gate_class_weight, 1.0)
            gate_l = F.binary_cross_entropy(outputs['gate'], gate_tgt, weight=pos_weight)

            # Regression heads: Huber loss
            mfe_l = huber_loss(outputs['mfe'], mfe_tgt)
            mae_l = huber_loss(outputs['mae'], mae_tgt)
            hold_l = huber_loss(outputs['hold_time'], hold_tgt)

            # Total loss: gate_loss_weight controls gate vs regression balance
            reg_weight = (1.0 - gate_loss_weight) * (4.0 / 3.0)
            total_loss = (gate_loss_weight * 4.0) * gate_l + reg_weight * (mfe_l + mae_l + 0.6 * hold_l)

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += total_loss.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = epoch_loss / max(n_batches, 1)

        # === Validation ===
        model.eval()
        all_gate_probs = []

        with torch.no_grad():
            for batch in val_loader:
                feats = batch['features'].to(DEVICE)
                outputs = model(feats)
                all_gate_probs.append(outputs['gate'].cpu().numpy())

        gate_probs = np.concatenate(all_gate_probs)

        # Find best Sharpe across gate thresholds (combined both sides)
        best_thresh_sharpe = float('-inf')
        best_thresh = 0.5
        best_thresh_result = {}
        best_per_side = {}

        for thresh in GATE_THRESHOLDS:
            per_side = evaluate_per_side(
                gate_probs, data['val_preds'], data['val_labels'], threshold=thresh,
            )
            combined = per_side['combined']
            if combined['n_trades'] >= 10 and combined.get('sharpe', float('-inf')) > best_thresh_sharpe:
                best_thresh_sharpe = combined['sharpe']
                best_thresh = thresh
                best_thresh_result = combined
                best_per_side = per_side

        val_sharpe = best_thresh_sharpe if best_thresh_sharpe > float('-inf') else 0.0
        val_sortino = best_thresh_result.get('sortino', 0.0)

        # Log every 5 epochs or at end
        if epoch % 5 == 0 or epoch == MAX_EPOCHS - 1:
            long_r = best_per_side.get('long', {})
            short_r = best_per_side.get('short', {})
            logger.info(
                f"  Epoch {epoch:>3d}/{MAX_EPOCHS} | "
                f"Loss: {avg_train_loss:.4f} | "
                f"Val Sharpe: {val_sharpe:>7.2f} (t={best_thresh:.2f}) | "
                f"Sortino: {val_sortino:>7.2f} | "
                f"Trades: {best_thresh_result.get('n_trades', 0)} | "
                f"WR: {best_thresh_result.get('win_rate', 0):.1%} | "
                f"Long: {long_r.get('n_trades', 0)}t/{long_r.get('sharpe', 0):.1f}S | "
                f"Short: {short_r.get('n_trades', 0)}t/{short_r.get('sharpe', 0):.1f}S"
            )

        # MLflow per-epoch metrics
        if MLFLOW_AVAILABLE:
            try:
                long_r = best_per_side.get('long', {})
                short_r = best_per_side.get('short', {})
                mlflow.log_metrics({
                    'train_loss': avg_train_loss,
                    'val_sharpe': val_sharpe,
                    'val_sortino': val_sortino,
                    'val_best_threshold': best_thresh,
                    'val_n_trades': best_thresh_result.get('n_trades', 0),
                    'val_win_rate': best_thresh_result.get('win_rate', 0),
                    'val_pf': min(best_thresh_result.get('pf', 0), 99.99),
                    'val_avg_pnl': best_thresh_result.get('avg_pnl', 0),
                    # Per-side metrics
                    'val_long_n_trades': long_r.get('n_trades', 0),
                    'val_long_sharpe': long_r.get('sharpe', 0),
                    'val_long_sortino': long_r.get('sortino', 0),
                    'val_long_win_rate': long_r.get('win_rate', 0),
                    'val_short_n_trades': short_r.get('n_trades', 0),
                    'val_short_sharpe': short_r.get('sharpe', 0),
                    'val_short_sortino': short_r.get('sortino', 0),
                    'val_short_win_rate': short_r.get('win_rate', 0),
                    'lr_current': optimizer.param_groups[0]['lr'],
                }, step=epoch)
            except Exception:
                pass

        # Report to Optuna for pruning
        trial.report(val_sharpe, epoch)
        if trial.should_prune():
            logger.info(f"  Trial {trial.number} pruned at epoch {epoch}")
            raise optuna.TrialPruned()

        # Early stopping
        if val_sharpe > best_val_sharpe:
            best_val_sharpe = val_sharpe
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            logger.info(f"  Early stopping at epoch {epoch} (best epoch: {best_epoch})")
            break

    logger.info(f"  Trial {trial.number} done: best Sharpe={best_val_sharpe:.2f} at epoch {best_epoch}")

    # Save best model state for this trial
    if best_state is not None:
        trial.set_user_attr('best_epoch', best_epoch)

    return best_val_sharpe


# ============================================================
# Post-sweep: save best trial's model and run full evaluation
# ============================================================

def save_best_trial(study: optuna.Study):
    """Save the best trial's model weights and run comprehensive evaluation."""

    best_trial = study.best_trial
    p = best_trial.params

    # Reload data with the best trial's feature params
    data = load_and_cache_data(
        signal_flip_threshold=p['signal_flip_threshold'],
        cancel_decay_window=p['cancel_decay_window'],
    )

    logger.info("\n" + "=" * 80)
    logger.info("BEST TRIAL RESULTS")
    logger.info("=" * 80)
    logger.info(f"  Trial number: {best_trial.number}")
    logger.info(f"  Best Sharpe:  {best_trial.value:.2f}")
    logger.info(f"  Params:")
    for k, v in best_trial.params.items():
        logger.info(f"    {k}: {v}")

    # Retrain the best model
    logger.info("\nRetraining best trial for final model save...")

    input_dim = data['input_dim']

    train_ds = ExecDataset(data['train_norm'], data['train_tgts'])
    val_ds = ExecDataset(data['val_norm'], data['val_tgts'])

    train_loader = DataLoader(
        train_ds, batch_size=p['batch_size'], shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=p['batch_size'] * 2, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    )

    model = ExecMLP(input_dim, p['hidden_dim'], p['n_layers'], p['dropout']).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=p['lr'], weight_decay=p['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    huber_loss = nn.SmoothL1Loss()

    gate_positive_ratio = data['gate_positive_ratio']
    gate_class_weight = max(0.5, min(2.0, (1 - gate_positive_ratio) / max(gate_positive_ratio, 0.01)))
    ls = p['label_smoothing']
    gate_w = p['gate_loss_weight']

    best_val_sharpe = float('-inf')
    best_state = None
    best_epoch = 0
    no_improve = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        for batch in train_loader:
            feats = batch['features'].to(DEVICE)
            gate_tgt = batch['gate'].to(DEVICE)
            if ls > 0:
                gate_tgt = gate_tgt * (1.0 - ls) + 0.5 * ls
            mfe_tgt = batch['mfe'].to(DEVICE)
            mae_tgt = batch['mae'].to(DEVICE)
            hold_tgt = batch['hold_time'].to(DEVICE)

            outputs = model(feats)
            pos_weight = torch.where(gate_tgt > 0.5, gate_class_weight, 1.0)
            gate_l = F.binary_cross_entropy(outputs['gate'], gate_tgt, weight=pos_weight)
            mfe_l = huber_loss(outputs['mfe'], mfe_tgt)
            mae_l = huber_loss(outputs['mae'], mae_tgt)
            hold_l = huber_loss(outputs['hold_time'], hold_tgt)

            reg_weight = (1.0 - gate_w) * (4.0 / 3.0)
            total_loss = (gate_w * 4.0) * gate_l + reg_weight * (mfe_l + mae_l + 0.6 * hold_l)

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        model.eval()
        all_gate_probs = []
        with torch.no_grad():
            for batch in val_loader:
                feats = batch['features'].to(DEVICE)
                outputs = model(feats)
                all_gate_probs.append(outputs['gate'].cpu().numpy())
        gate_probs = np.concatenate(all_gate_probs)

        best_thresh_sharpe = float('-inf')
        for thresh in GATE_THRESHOLDS:
            result = evaluate_gated_trading(gate_probs, data['val_preds'], data['val_labels'], threshold=thresh)
            if result['n_trades'] >= 10 and result.get('sharpe', float('-inf')) > best_thresh_sharpe:
                best_thresh_sharpe = result['sharpe']

        val_sharpe = best_thresh_sharpe if best_thresh_sharpe > float('-inf') else 0.0

        if val_sharpe > best_val_sharpe:
            best_val_sharpe = val_sharpe
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= PATIENCE:
            break

    # Restore best and run full eval
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    all_gate_probs = []
    all_mfe_preds = []
    all_mae_preds = []
    all_hold_preds = []

    with torch.no_grad():
        val_tensor = torch.from_numpy(data['val_norm']).to(DEVICE)
        chunk_size = 10000
        for start in range(0, len(val_tensor), chunk_size):
            end = min(start + chunk_size, len(val_tensor))
            out = model(val_tensor[start:end])
            all_gate_probs.append(out['gate'].cpu().numpy())
            all_mfe_preds.append(out['mfe'].cpu().numpy())
            all_mae_preds.append(out['mae'].cpu().numpy())
            all_hold_preds.append(out['hold_time'].cpu().numpy())

    gate_probs = np.concatenate(all_gate_probs)
    mfe_preds = np.concatenate(all_mfe_preds)
    mae_preds = np.concatenate(all_mae_preds)
    hold_preds = np.concatenate(all_hold_preds)

    # Comprehensive threshold evaluation — COMBINED + PER-SIDE
    logger.info(f"\n{'='*100}")
    logger.info(f"RETRAINED VALIDATION RESULTS (MIDPOINT-BASED, cost={COST_TICKS_RT:.3f} ticks RT)")
    logger.info(f"{'='*100}")

    # Combined results table
    logger.info(f"\n--- COMBINED (Both Sides) ---")
    logger.info(f"{'Threshold':>10s} | {'N_trades':>8s} | {'Coverage':>8s} | {'WinRate':>7s} | "
                f"{'PF':>6s} | {'Sharpe':>8s} | {'Sortino':>8s} | {'AvgPnL':>8s}")
    logger.info("-" * 85)

    overall_best = None
    all_threshold_results = {}
    for thresh in GATE_THRESHOLDS:
        per_side = evaluate_per_side(gate_probs, data['val_preds'], data['val_labels'], threshold=thresh)
        all_threshold_results[thresh] = per_side
        r = per_side['combined']
        if r['n_trades'] < 5:
            continue
        logger.info(
            f"{thresh:>10.2f} | {r['n_trades']:>8d} | {r.get('coverage', 0):>7.1%} | "
            f"{r['win_rate']:>6.1%} | {r.get('pf', 0):>5.2f} | "
            f"{r.get('sharpe', 0):>8.2f} | {r.get('sortino', 0):>8.2f} | ${r.get('avg_pnl', 0):>7.2f}"
        )
        if overall_best is None or r.get('sharpe', -999) > overall_best.get('sharpe', -999):
            overall_best = r
            overall_best['threshold'] = thresh

    # Long-side results table
    logger.info(f"\n--- LONG SIDE ONLY ---")
    logger.info(f"{'Threshold':>10s} | {'N_trades':>8s} | {'WinRate':>7s} | "
                f"{'PF':>6s} | {'Sharpe':>8s} | {'Sortino':>8s} | {'AvgPnL':>8s}")
    logger.info("-" * 75)

    for thresh in GATE_THRESHOLDS:
        r = all_threshold_results[thresh]['long']
        if r['n_trades'] < 5:
            continue
        logger.info(
            f"{thresh:>10.2f} | {r['n_trades']:>8d} | "
            f"{r['win_rate']:>6.1%} | {r.get('pf', 0):>5.2f} | "
            f"{r.get('sharpe', 0):>8.2f} | {r.get('sortino', 0):>8.2f} | ${r.get('avg_pnl', 0):>7.2f}"
        )

    # Short-side results table
    logger.info(f"\n--- SHORT SIDE ONLY ---")
    logger.info(f"{'Threshold':>10s} | {'N_trades':>8s} | {'WinRate':>7s} | "
                f"{'PF':>6s} | {'Sharpe':>8s} | {'Sortino':>8s} | {'AvgPnL':>8s}")
    logger.info("-" * 75)

    for thresh in GATE_THRESHOLDS:
        r = all_threshold_results[thresh]['short']
        if r['n_trades'] < 5:
            continue
        logger.info(
            f"{thresh:>10.2f} | {r['n_trades']:>8d} | "
            f"{r['win_rate']:>6.1%} | {r.get('pf', 0):>5.2f} | "
            f"{r.get('sharpe', 0):>8.2f} | {r.get('sortino', 0):>8.2f} | ${r.get('avg_pnl', 0):>7.2f}"
        )

    # Save model
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = OUTPUT_DIR / "optuna_best_exec_mlp_v2.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'norm_mean': data['norm_mean'],
        'norm_std': data['norm_std'],
        'feature_names': data['feature_names'],
        'input_dim': input_dim,
        'hidden_dim': p['hidden_dim'],
        'n_layers': p['n_layers'],
        'dropout': p['dropout'],
        'best_threshold': overall_best['threshold'] if overall_best else 0.5,
        'cost_ticks_rt': COST_TICKS_RT,
        'signal_flip_threshold': p['signal_flip_threshold'],
        'cancel_decay_window': p['cancel_decay_window'],
        'optuna_params': dict(best_trial.params),
        'optuna_trial_number': best_trial.number,
        'optuna_best_sharpe': best_trial.value,
        'retrained_best_sharpe': best_val_sharpe,
        'retrained_best_epoch': best_epoch,
        'version': 'v2_both_sides',
    }, str(save_path))
    logger.info(f"\nBest model saved to {save_path}")

    # Save predictions
    pred_path = OUTPUT_DIR / "optuna_best_val_predictions.npz"
    np.savez(
        str(pred_path),
        gate_probs=gate_probs,
        mfe_preds=mfe_preds,
        mae_preds=mae_preds,
        hold_preds=hold_preds,
        val_preds=data['val_preds'],
        val_labels=data['val_labels'],
    )
    logger.info(f"Predictions saved to {pred_path}")

    # Save study results as JSON
    results_path = OUTPUT_DIR / "optuna_study_results.json"

    # Build per-threshold results for JSON (including per-side)
    threshold_details = {}
    for thresh, per_side in all_threshold_results.items():
        threshold_details[str(thresh)] = {
            'combined': {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                        for k, v in per_side['combined'].items()},
            'long': {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                    for k, v in per_side['long'].items()},
            'short': {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                     for k, v in per_side['short'].items()},
        }

    study_results = {
        'best_trial': {
            'number': best_trial.number,
            'value': best_trial.value,
            'params': dict(best_trial.params),
        },
        'retrained_sharpe': best_val_sharpe,
        'retrained_epoch': best_epoch,
        'best_threshold_result': {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                                  for k, v in (overall_best or {}).items()},
        'per_threshold_per_side': threshold_details,
        'n_trials_completed': len(study.trials),
        'n_trials_pruned': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        'cost_ticks_rt': COST_TICKS_RT,
        'execution_basis': 'MIDPOINT-BASED (no FIFO fill sim, HC #57)',
        'signal_sides': 'both (long + short)',
        'v2_features': [
            'signal_flip_strength', 'signal_flip_occurred', 'signal_decay_rate',
            'time_since_signal_peak', 'signal_zscore',
            'spread_regime', 'signal_spread_interaction', 'tight_spread_indicator',
        ],
        'all_trials': [
            {
                'number': t.number,
                'value': t.value if t.value is not None else None,
                'state': t.state.name,
                'params': dict(t.params),
            }
            for t in study.trials
        ],
    }
    with open(str(results_path), 'w') as f:
        json.dump(study_results, f, indent=2, default=str)
    logger.info(f"Study results saved to {results_path}")

    if MLFLOW_AVAILABLE:
        try:
            mlflow.log_artifact(str(save_path))
            mlflow.log_artifact(str(results_path))
        except Exception:
            pass


# ============================================================
# Main
# ============================================================

def _update_globals(max_epochs: int, n_trials: int):
    global MAX_EPOCHS, N_TRIALS
    MAX_EPOCHS = max_epochs
    N_TRIALS = n_trials


def main():
    parser = argparse.ArgumentParser(description="Optuna HP sweep for Execution MLP v2 (Both Sides)")
    parser.add_argument('--n-trials', type=int, default=100, help='Number of Optuna trials (default: 100)')
    parser.add_argument('--max-epochs', type=int, default=60, help='Max epochs per trial')
    parser.add_argument('--study-name', type=str, default='exec_mlp_optuna_v2_both_sides',
                        help='Optuna study name')
    parser.add_argument('--storage', type=str, default=None,
                        help='Optuna storage URL (e.g. sqlite:///optuna.db). Default: in-memory.')
    args = parser.parse_args()

    # Update module-level settings from CLI args
    _update_globals(args.max_epochs, args.n_trials)

    logger.info("=" * 80)
    logger.info("Optuna Hyperparameter Optimization — Execution MLP v2 (BOTH SIDES)")
    logger.info("=" * 80)
    logger.info(f"  N trials:      {N_TRIALS}")
    logger.info(f"  Max epochs:    {MAX_EPOCHS}")
    logger.info(f"  Pruner:        MedianPruner (startup={PRUNER_STARTUP})")
    logger.info(f"  Cost:          {COST_TICKS_RT:.3f} ticks RT (${COST_TICKS_RT * TICK_VAL:.2f}, HC #52)")
    logger.info(f"  Results:       MIDPOINT-BASED (no FIFO fill sim, HC #57)")
    logger.info(f"  Metrics:       Sharpe AND Sortino, per-side (long/short)")
    logger.info(f"  Signal sides:  BOTH (long + short)")
    logger.info(f"  Gate range:    {GATE_THRESHOLDS}")
    logger.info(f"  Device:        {DEVICE}")
    logger.info(f"  Output:        {OUTPUT_DIR}")
    logger.info(f"  v2 features:   signal_flip, signal_decay, time_since_peak,")
    logger.info(f"                 signal_zscore, spread_regime, signal_spread_interaction,")
    logger.info(f"                 tight_spread_indicator")
    logger.info("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Pre-load data with default feature params (will be reloaded per trial if params differ)
    load_and_cache_data(signal_flip_threshold=1.5, cancel_decay_window=20)

    # MLflow parent run
    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_tracking_uri(MLFLOW_URI)
            mlflow.set_experiment(EXPERIMENT_NAME)
            mlflow.start_run(run_name=f"optuna_v2_sweep_{time.strftime('%Y%m%d_%H%M%S')}")
            mlflow.log_params({
                'sweep_type': 'optuna_tpe_v2',
                'n_trials': N_TRIALS,
                'max_epochs': MAX_EPOCHS,
                'pruner': 'MedianPruner',
                'pruner_startup': PRUNER_STARTUP,
                'cost_ticks_rt': COST_TICKS_RT,
                'execution_basis': 'MIDPOINT-BASED (no FIFO fill sim, HC #57)',
                'objective': 'val_sharpe_at_optimal_gate',
                'signal_sides': 'both (long + short)',
                'gate_thresholds': str(GATE_THRESHOLDS),
                'v2_new_features': 'signal_flip, decay_rate, time_since_peak, spread_regime, signal_spread_interaction',
                'v2_new_optuna_params': 'signal_flip_threshold, cancel_decay_window',
            })
        except Exception as e:
            logger.warning(f"MLflow setup failed: {e}")

    # Create Optuna study
    storage = args.storage
    if storage:
        study = optuna.create_study(
            study_name=args.study_name,
            storage=storage,
            direction="maximize",
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=PRUNER_STARTUP,
                interval_steps=1,
            ),
            load_if_exists=True,
        )
    else:
        study = optuna.create_study(
            study_name=args.study_name,
            direction="maximize",
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=PRUNER_STARTUP,
                interval_steps=1,
            ),
        )

    # Run optimization
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("OPTUNA v2 SWEEP COMPLETE")
    logger.info("=" * 80)
    logger.info(f"  Trials completed: {len(study.trials)}")
    pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    logger.info(f"  Completed: {completed}, Pruned: {pruned}")
    logger.info(f"\n  Best trial: #{study.best_trial.number}")
    logger.info(f"  Best Sharpe: {study.best_trial.value:.2f}")
    logger.info(f"  Best params:")
    for k, v in study.best_trial.params.items():
        logger.info(f"    {k}: {v}")

    # Log top 5 trials
    logger.info(f"\n  Top 5 trials:")
    sorted_trials = sorted(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE],
        key=lambda t: t.value if t.value is not None else float('-inf'),
        reverse=True,
    )
    for i, t in enumerate(sorted_trials[:5]):
        logger.info(f"    #{t.number}: Sharpe={t.value:.2f} | lr={t.params['lr']:.2e}, "
                     f"hidden={t.params['hidden_dim']}, layers={t.params['n_layers']}, "
                     f"drop={t.params['dropout']:.2f}, "
                     f"flip_t={t.params['signal_flip_threshold']:.2f}, "
                     f"decay_w={t.params['cancel_decay_window']}")

    # Save best trial's model and run full per-side evaluation
    save_best_trial(study)

    # Log final metrics to MLflow parent run
    if MLFLOW_AVAILABLE:
        try:
            mlflow.log_metrics({
                'best_trial_number': study.best_trial.number,
                'best_trial_sharpe': study.best_trial.value,
                'n_completed': completed,
                'n_pruned': pruned,
            })
            mlflow.end_run()
        except Exception:
            pass

    logger.info(f"\nAll outputs saved to {OUTPUT_DIR}")
    logger.info(f"Cost basis: {COST_TICKS_RT:.3f} ticks RT (${COST_TICKS_RT * TICK_VAL:.2f}, HC #52)")
    logger.info(f"All results MIDPOINT-BASED (no FIFO fill sim, HC #57)")
    logger.info(f"Metrics: Sharpe AND Sortino, per-side (long/short) reported")
    logger.info(f"Signal sides: BOTH long and short trained and evaluated")


if __name__ == "__main__":
    main()
