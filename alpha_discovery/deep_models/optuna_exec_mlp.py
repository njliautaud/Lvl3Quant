#!/usr/bin/env python3
"""
optuna_exec_mlp.py — Optuna Hyperparameter Optimization for Execution MLP
==========================================================================
Runs on Neptune RTX 3090 (24GB VRAM, Ubuntu).

Sweeps hyperparameters for the multi-task ExecMLP (gate, MFE, MAE, hold_time)
using Optuna with MedianPruner. Objective: maximize validation Sharpe at
optimal gate threshold.

Imports core data loading, feature engineering, model architecture, and
evaluation functions from train_exec_mlp_gpu.py.

Cost basis: 0.376 ticks RT ($4.70 AMP/Rithmic, HC #52).
All results are MIDPOINT-BASED (no FIFO fill sim, HC #70).
Reports Sharpe AND Sortino (HC #57).
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
OUTPUT_DIR = Path("/home/nick/Lvl3Quant/output/optuna_exec_mlp")
MLFLOW_URI = "http://localhost:5000"
EXPERIMENT_NAME = "optuna_exec_mlp"

# Optuna config
N_TRIALS = 50
MAX_EPOCHS = 60
PATIENCE = 15           # early stopping patience per trial
PRUNER_STARTUP = 15     # MedianPruner starts after this many epochs

# Gate thresholds to evaluate for finding optimal Sharpe
GATE_THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]


# ============================================================
# Data cache (loaded once, reused across trials)
# ============================================================
_DATA_CACHE = {}


def load_and_cache_data() -> dict:
    """Load all fold data, build features and targets. Cache globally."""
    global _DATA_CACHE
    if _DATA_CACHE:
        return _DATA_CACHE

    logger.info("=" * 80)
    logger.info("Loading and caching fold data (one-time)")
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

    # Build features for all folds
    logger.info("\nBuilding features...")
    fold_features = {}
    fold_targets = {}
    feature_names = None

    for fold_idx, data in all_data.items():
        logger.info(f"  Fold {fold_idx:02d} ({data['date']}):")
        feats, names = build_all_features(data)
        tgts = build_targets(data)
        fold_features[fold_idx] = feats
        fold_targets[fold_idx] = tgts
        if feature_names is None:
            feature_names = names

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

    logger.info(f"\nTrain samples: {len(train_X):,}")
    logger.info(f"Val samples:   {len(val_X):,}")
    logger.info(f"Features:      {train_X.shape[1]}")

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
    }
    return _DATA_CACHE


# ============================================================
# Optuna Objective
# ============================================================

def objective(trial: optuna.Trial) -> float:
    """
    Single Optuna trial: train ExecMLP with sampled hyperparameters,
    return best validation Sharpe across gate thresholds.
    """
    data = load_and_cache_data()

    # --- Sample hyperparameters ---
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    hidden_dim = trial.suggest_int("hidden_dim", 128, 512, step=64)
    n_layers = trial.suggest_int("n_layers", 2, 6)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    batch_size = trial.suggest_int("batch_size", 512, 4096, step=256)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    gate_loss_weight = trial.suggest_float("gate_loss_weight", 0.3, 0.7)
    label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2)

    trial_name = f"trial_{trial.number:03d}"
    logger.info(f"\n{'='*70}")
    logger.info(f"Trial {trial.number}: lr={lr:.2e}, hidden={hidden_dim}, layers={n_layers}, "
                f"drop={dropout:.2f}, bs={batch_size}, wd={weight_decay:.2e}, "
                f"gate_w={gate_loss_weight:.2f}, ls={label_smoothing:.2f}")
    logger.info(f"{'='*70}")

    # --- MLflow logging for this trial ---
    mlflow_run = None
    if MLFLOW_AVAILABLE:
        try:
            mlflow_run = mlflow.start_run(
                run_name=f"optuna_{trial_name}_{time.strftime('%H%M%S')}",
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
                'max_epochs': MAX_EPOCHS,
                'cost_ticks_rt': COST_TICKS_RT,
                'execution_basis': 'midpoint (no FIFO fill sim)',
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

    # Create datasets
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
    logger.info(f"  Model params: {n_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    # Loss functions
    huber_loss = nn.SmoothL1Loss()

    # Gate class weight
    gate_positive_ratio = data['gate_positive_ratio']
    gate_class_weight = max(0.5, min(2.0, (1 - gate_positive_ratio) / max(gate_positive_ratio, 0.01)))

    # Apply label smoothing to gate targets
    # smooth_target = target * (1 - smoothing) + 0.5 * smoothing
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
            # Scale so gate_loss_weight=0.5 roughly matches original (2.0 * gate + 0.5*mfe + 0.5*mae + 0.3*hold)
            # gate gets weight: gate_loss_weight * 4.0, rest share (1 - gate_loss_weight) * (4.0/3)
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

        # Find best Sharpe across gate thresholds
        best_thresh_sharpe = float('-inf')
        best_thresh = 0.5
        best_thresh_result = {}

        for thresh in GATE_THRESHOLDS:
            result = evaluate_gated_trading(
                gate_probs, data['val_preds'], data['val_labels'], threshold=thresh,
            )
            if result['n_trades'] >= 10 and result.get('sharpe', float('-inf')) > best_thresh_sharpe:
                best_thresh_sharpe = result['sharpe']
                best_thresh = thresh
                best_thresh_result = result

        val_sharpe = best_thresh_sharpe if best_thresh_sharpe > float('-inf') else 0.0
        val_sortino = best_thresh_result.get('sortino', 0.0)

        # Log every 5 epochs or at end
        if epoch % 5 == 0 or epoch == MAX_EPOCHS - 1:
            logger.info(
                f"  Epoch {epoch:>3d}/{MAX_EPOCHS} | "
                f"Loss: {avg_train_loss:.4f} | "
                f"Val Sharpe: {val_sharpe:>7.2f} (t={best_thresh:.2f}) | "
                f"Sortino: {val_sortino:>7.2f} | "
                f"Trades: {best_thresh_result.get('n_trades', 0)} | "
                f"WR: {best_thresh_result.get('win_rate', 0):.1%}"
            )

        # MLflow per-epoch metrics
        if MLFLOW_AVAILABLE:
            try:
                mlflow.log_metrics({
                    'train_loss': avg_train_loss,
                    'val_sharpe': val_sharpe,
                    'val_sortino': val_sortino,
                    'val_best_threshold': best_thresh,
                    'val_n_trades': best_thresh_result.get('n_trades', 0),
                    'val_win_rate': best_thresh_result.get('win_rate', 0),
                    'val_pf': min(best_thresh_result.get('pf', 0), 99.99),
                    'val_avg_pnl': best_thresh_result.get('avg_pnl', 0),
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

    # Save best model state for this trial (will be overwritten; final best saved separately)
    if best_state is not None:
        trial.set_user_attr('best_epoch', best_epoch)
        trial.set_user_attr('best_state_dict', best_state)

    return best_val_sharpe


# ============================================================
# Post-sweep: save best trial's model and run full evaluation
# ============================================================

def save_best_trial(study: optuna.Study):
    """Save the best trial's model weights and run comprehensive evaluation."""
    data = load_and_cache_data()
    best_trial = study.best_trial

    logger.info("\n" + "=" * 80)
    logger.info("BEST TRIAL RESULTS")
    logger.info("=" * 80)
    logger.info(f"  Trial number: {best_trial.number}")
    logger.info(f"  Best Sharpe:  {best_trial.value:.2f}")
    logger.info(f"  Params:")
    for k, v in best_trial.params.items():
        logger.info(f"    {k}: {v}")

    # Retrain the best model (since we can't reliably pass state_dict through Optuna attrs
    # for large models — retrain is safer and also validates reproducibility)
    logger.info("\nRetraining best trial for final model save...")

    p = best_trial.params
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

    # Comprehensive threshold evaluation
    logger.info(f"\nRETRAINED VALIDATION RESULTS (MIDPOINT-BASED, folds 09-10)")
    logger.info(f"{'Threshold':>10s} | {'N_trades':>8s} | {'Coverage':>8s} | {'WinRate':>7s} | "
                f"{'PF':>6s} | {'Sharpe':>8s} | {'Sortino':>8s} | {'AvgPnL':>8s}")
    logger.info("-" * 85)

    overall_best = None
    for thresh in GATE_THRESHOLDS:
        r = evaluate_gated_trading(gate_probs, data['val_preds'], data['val_labels'], threshold=thresh)
        if r['n_trades'] < 10:
            continue
        logger.info(
            f"{thresh:>10.2f} | {r['n_trades']:>8d} | {r['coverage']:>7.1%} | "
            f"{r['win_rate']:>6.1%} | {r['pf']:>5.2f} | "
            f"{r['sharpe']:>8.2f} | {r['sortino']:>8.2f} | ${r['avg_pnl']:>7.2f}"
        )
        if overall_best is None or r['sharpe'] > overall_best.get('sharpe', -999):
            overall_best = r
            overall_best['threshold'] = thresh

    # Save model
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = OUTPUT_DIR / "optuna_best_exec_mlp.pt"
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
        'optuna_params': dict(best_trial.params),
        'optuna_trial_number': best_trial.number,
        'optuna_best_sharpe': best_trial.value,
        'retrained_best_sharpe': best_val_sharpe,
        'retrained_best_epoch': best_epoch,
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
    study_results = {
        'best_trial': {
            'number': best_trial.number,
            'value': best_trial.value,
            'params': dict(best_trial.params),
        },
        'retrained_sharpe': best_val_sharpe,
        'retrained_epoch': best_epoch,
        'best_threshold_result': overall_best,
        'n_trials_completed': len(study.trials),
        'n_trials_pruned': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        'cost_ticks_rt': COST_TICKS_RT,
        'execution_basis': 'midpoint (no FIFO fill sim)',
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
    parser = argparse.ArgumentParser(description="Optuna HP sweep for Execution MLP")
    parser.add_argument('--n-trials', type=int, default=50, help='Number of Optuna trials')
    parser.add_argument('--max-epochs', type=int, default=60, help='Max epochs per trial')
    parser.add_argument('--study-name', type=str, default='exec_mlp_optuna_v1', help='Optuna study name')
    parser.add_argument('--storage', type=str, default=None,
                        help='Optuna storage URL (e.g. sqlite:///optuna.db). Default: in-memory.')
    args = parser.parse_args()

    # Update module-level settings from CLI args
    _update_globals(args.max_epochs, args.n_trials)

    logger.info("=" * 80)
    logger.info("Optuna Hyperparameter Optimization — Execution MLP")
    logger.info(f"  N trials:   {N_TRIALS}")
    logger.info(f"  Max epochs: {MAX_EPOCHS}")
    logger.info(f"  Pruner:     MedianPruner (startup={PRUNER_STARTUP})")
    logger.info(f"  Cost:       {COST_TICKS_RT:.3f} ticks RT (${COST_TICKS_RT * TICK_VAL:.2f}, HC #52)")
    logger.info(f"  Results:    MIDPOINT-BASED (no FIFO fill sim, HC #70)")
    logger.info(f"  Metrics:    Sharpe AND Sortino (HC #57)")
    logger.info(f"  Device:     {DEVICE}")
    logger.info(f"  Output:     {OUTPUT_DIR}")
    logger.info("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Pre-load data before starting trials
    load_and_cache_data()

    # MLflow parent run
    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_tracking_uri(MLFLOW_URI)
            mlflow.set_experiment(EXPERIMENT_NAME)
            mlflow.start_run(run_name=f"optuna_sweep_{time.strftime('%Y%m%d_%H%M%S')}")
            mlflow.log_params({
                'sweep_type': 'optuna_tpe',
                'n_trials': N_TRIALS,
                'max_epochs': MAX_EPOCHS,
                'pruner': 'MedianPruner',
                'pruner_startup': PRUNER_STARTUP,
                'cost_ticks_rt': COST_TICKS_RT,
                'execution_basis': 'midpoint (no FIFO fill sim)',
                'objective': 'val_sharpe_at_optimal_gate',
            })
        except Exception as e:
            logger.warning(f"MLflow setup failed: {e}")

    # Create Optuna study
    storage = args.storage
    if storage:
        # Persistent storage for resumability
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
    logger.info("OPTUNA SWEEP COMPLETE")
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
                     f"drop={t.params['dropout']:.2f}")

    # Save best trial's model
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
    logger.info(f"Cost basis: {COST_TICKS_RT:.3f} ticks RT (${COST_TICKS_RT * TICK_VAL:.2f})")
    logger.info(f"All results MIDPOINT-BASED (no FIFO fill sim)")
    logger.info(f"Metrics reported: Sharpe AND Sortino (HC #57)")


if __name__ == "__main__":
    main()
