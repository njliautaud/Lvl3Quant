#!/usr/bin/env python3
"""
train_retrace_timing_v1.py — Retrace Timing Predictor (MLP Regression)
======================================================================
Predicts log(time_to_retrace_1tick) for SHORT signals from CNN-Mamba v2.

CONTEXT: We have a confirmed short-side edge (+0.32 ticks/trade) but
profitability depends on passive exit fills. If price retraces 1 tick
within 30s, passive fill rate is 83-85%. Predicting retrace SPEED lets
us skip slow-retrace events and improve realized P&L.

Architecture: 25 smart_v3 features → 128 → 64 → 32 → 1 (regression)
  - BatchNorm + ReLU + Dropout(0.2) between layers
  - Huber loss (robust to outliers in retrace timing)

Target construction (MFE proxy):
  - MFE_1s >= 0.25pts → retrace_time ~ 0.5s
  - MFE_5s >= 0.25pts (but not 1s) → retrace_time ~ 3.0s
  - MFE_10s >= 0.25pts (but not 5s) → retrace_time ~ 7.5s
  - MFE_10s < 0.25pts → use MAE heuristic or cap at 30s/60s

Signal filter: Only train/eval on top-5% short signals from CNN-Mamba v2.

Walk-forward: 10-date sliding train, 1-date OOT test, sliding window.

Usage (on Razer):
  C:\\Users\\claude\\Lvl3Quant\\.venv_research\\Scripts\\python.exe ^
    scripts/train_retrace_timing_v1.py

Author: Claude (autonomous research)
"""

import argparse
import gc
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

# ============================================================
# Configuration
# ============================================================

N_FEATURES = 25
SHORT_PERCENTILE = 5  # top 5% short signals (lowest return_1s predictions)

# MFE label indices in the [N, 9] label array
IDX_MFE_1S = 3
IDX_MFE_5S = 4
IDX_MFE_10S = 5
IDX_MAE_1S = 6
IDX_MAE_5S = 7
IDX_MAE_10S = 8

# Retrace time buckets (seconds) mapped to log(seconds)
RETRACE_FAST = 0.5       # MFE_1s >= 1 tick
RETRACE_MEDIUM = 3.0     # MFE_5s >= 1 tick (but not 1s)
RETRACE_SLOW = 7.5       # MFE_10s >= 1 tick (but not 5s)
RETRACE_VERY_SLOW = 30.0 # beyond 10s horizon, still possible
RETRACE_NEVER = 60.0     # never retraces within observation window

ONE_TICK_PTS = 0.25  # 1 tick = 0.25 ES points

# Training
TRAIN_WINDOW = 10   # dates
TEST_WINDOW = 1     # date
BATCH_SIZE = 4096
MAX_EPOCHS = 30
PATIENCE = 5
LR = 1e-3
WEIGHT_DECAY = 1e-4

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ============================================================
# Model
# ============================================================

class RetraceTimingMLP(nn.Module):
    """MLP regression: 25 features → log(retrace_time_seconds)."""

    def __init__(self, n_features: int = 25, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ============================================================
# Data loading
# ============================================================

def discover_dates(features_dir: Path, predictions_dir: Path) -> List[str]:
    """Find dates that have both features and predictions, sorted chronologically."""
    feat_files = {f.name.split("_mbo_events")[0]: f for f in features_dir.glob("*_mbo_events.npz")}
    pred_files = {f.name.split("_predictions")[0]: f for f in predictions_dir.glob("*_predictions.npz")}

    common_dates = sorted(set(feat_files.keys()) & set(pred_files.keys()))
    log.info(f"Found {len(common_dates)} dates with both features and predictions")

    if len(common_dates) < TRAIN_WINDOW + TEST_WINDOW:
        raise ValueError(
            f"Need at least {TRAIN_WINDOW + TEST_WINDOW} dates, found {len(common_dates)}"
        )

    return common_dates, feat_files, pred_files


def load_date_data(
    date_key: str,
    feat_files: Dict[str, Path],
    pred_files: Dict[str, Path],
    short_threshold: Optional[float] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Load features and construct retrace timing target for a single date.

    Actual data format:
      - Features npz: key 'events' [N_events, 25], 'labels_1s' [N,], 'labels_5s' [N,], etc.
      - Predictions npz: key 'predictions' [N_preds, 3], 'labels' [N_preds, 3], plus 'stride', 'window_size'
      - N_preds << N_events because predictions are strided windows

    Returns:
        (features [N_filtered, 25], targets [N_filtered]) or None if no valid events
    """
    feat_path = feat_files[date_key]
    pred_path = pred_files[date_key]

    feat_data = np.load(str(feat_path), allow_pickle=True)
    pred_data = np.load(str(pred_path), allow_pickle=True)

    events = feat_data["events"]          # [N_events, 25]
    labels_1s = feat_data["labels_1s"]    # [N_events]
    labels_5s = feat_data["labels_5s"]    # [N_events]
    labels_10s = feat_data["labels_10s"]  # [N_events]

    predictions = pred_data["predictions"]  # [N_preds, 3] — horizons 1s/5s/10s
    pred_labels = pred_data["labels"]       # [N_preds, 3]

    # Get stride and window_size for alignment
    stride = int(pred_data["stride"]) if "stride" in pred_data else STRIDE
    window_size = int(pred_data["window_size"]) if "window_size" in pred_data else WINDOW_SIZE

    n_preds = predictions.shape[0]
    n_events = events.shape[0]

    if n_preds == 0:
        return None

    # Align: prediction[i] corresponds to events at index (i * stride + window_size - 1)
    # That's the last event in the window
    pred_event_indices = np.arange(n_preds) * stride + window_size - 1
    valid = pred_event_indices < n_events
    pred_event_indices = pred_event_indices[valid]
    predictions = predictions[valid]
    pred_labels = pred_labels[valid]

    # Extract aligned features (last event in each window)
    features = events[pred_event_indices]  # [N_valid, 25]

    # Extract aligned labels at the prediction points
    ret_1s = labels_1s[pred_event_indices]
    ret_5s = labels_5s[pred_event_indices]
    ret_10s = labels_10s[pred_event_indices]

    # Verify feature count
    if features.shape[1] != N_FEATURES:
        log.warning(f"Expected {N_FEATURES} features, got {features.shape[1]}")
        return None

    # CNN-Mamba short signal: prediction col 0 = return_1s, negative = short
    pred_return_1s = predictions[:, 0]

    # Filter to short signals only
    if short_threshold is not None:
        short_mask = pred_return_1s <= short_threshold
    else:
        short_mask = np.ones(len(predictions), dtype=bool)

    if short_mask.sum() == 0:
        return None

    features = features[short_mask]
    ret_1s = ret_1s[short_mask]
    ret_5s = ret_5s[short_mask]
    ret_10s = ret_10s[short_mask]

    # Construct retrace timing target from RETURNS (no MFE available)
    # For SHORT trades, favorable = price goes DOWN = negative return
    # If return_1s < -0.25 (dropped 1 tick in 1s), fast retrace
    n = features.shape[0]
    retrace_time = np.full(n, RETRACE_NEVER, dtype=np.float32)

    # Return_10s < -1 tick → retrace in 5-10s
    mask_10s = (ret_10s < -ONE_TICK_PTS) & (ret_5s >= -ONE_TICK_PTS)
    retrace_time[mask_10s] = RETRACE_SLOW

    # Return_5s < -1 tick but return_1s >= -1 tick → retrace in 1-5s
    mask_5s = (ret_5s < -ONE_TICK_PTS) & (ret_1s >= -ONE_TICK_PTS)
    retrace_time[mask_5s] = RETRACE_MEDIUM

    # Return_1s < -1 tick → fast retrace < 1s
    mask_1s = ret_1s < -ONE_TICK_PTS
    retrace_time[mask_1s] = RETRACE_FAST

    # Partial retrace: return dropped at least half a tick by 10s
    mask_partial = (ret_10s >= -ONE_TICK_PTS) & (ret_10s < -ONE_TICK_PTS * 0.5)
    retrace_time[mask_partial] = RETRACE_VERY_SLOW
    # Rest stays at RETRACE_NEVER (60s)

    # Log transform
    targets = np.log(retrace_time).astype(np.float32)

    return features.astype(np.float32), targets


def compute_short_threshold(
    dates: List[str],
    pred_files: Dict[str, Path],
    percentile: float = SHORT_PERCENTILE,
) -> float:
    """Compute the global top-N% short signal threshold across all dates."""
    all_preds = []
    for date_key in dates:
        pred_path = pred_files[date_key]
        pred_data = np.load(str(pred_path))
        pred_return_1s = pred_data["predictions"][:, 0]
        all_preds.append(pred_return_1s)

    all_preds = np.concatenate(all_preds)
    # Top 5% short = bottom 5th percentile of return predictions
    threshold = np.percentile(all_preds, percentile)
    log.info(
        f"Short threshold (p{percentile}): {threshold:.6f} "
        f"({(all_preds <= threshold).sum():,} / {len(all_preds):,} events)"
    )
    return threshold


# ============================================================
# Training loop
# ============================================================

def train_one_fold(
    model: RetraceTimingMLP,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    val_features: np.ndarray,
    val_targets: np.ndarray,
    device: torch.device,
    fold_idx: int,
) -> Tuple[RetraceTimingMLP, Dict[str, float], np.ndarray]:
    """Train one walk-forward fold with early stopping."""

    # Prepare datasets
    X_train = torch.from_numpy(train_features).to(device)
    y_train = torch.from_numpy(train_targets).to(device)
    X_val = torch.from_numpy(val_features).to(device)
    y_val = torch.from_numpy(val_targets).to(device)

    # Standardize features using train stats
    train_mean = X_train.mean(dim=0)
    train_std = X_train.std(dim=0).clamp(min=1e-8)
    X_train = (X_train - train_mean) / train_std
    X_val = (X_val - train_mean) / train_std

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,  # already on device
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )
    criterion = nn.HuberLoss(delta=1.0)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(MAX_EPOCHS):
        # --- Train ---
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_loss = epoch_loss / max(n_batches, 1)

        # --- Validate ---
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()

            # Correlation
            vp = val_pred.cpu().numpy()
            vt = y_val.cpu().numpy()
            if vp.std() > 1e-8 and vt.std() > 1e-8:
                val_corr = np.corrcoef(vp, vt)[0, 1]
            else:
                val_corr = 0.0

        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 5 == 0 or epochs_no_improve >= PATIENCE:
            log.info(
                f"  Fold {fold_idx} epoch {epoch:2d}: "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"val_corr={val_corr:.4f}  lr={optimizer.param_groups[0]['lr']:.2e}"
            )

        if epochs_no_improve >= PATIENCE:
            log.info(f"  Early stopping at epoch {epoch}")
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final validation predictions
    model.eval()
    with torch.no_grad():
        final_pred = model(X_val).cpu().numpy()
        final_loss = criterion(model(X_val), y_val).item()
        vt = y_val.cpu().numpy()
        if final_pred.std() > 1e-8 and vt.std() > 1e-8:
            final_corr = np.corrcoef(final_pred, vt)[0, 1]
        else:
            final_corr = 0.0

    metrics = {
        "val_loss": final_loss,
        "val_corr": final_corr,
        "best_epoch": MAX_EPOCHS - epochs_no_improve if best_state else MAX_EPOCHS,
        "n_train": len(train_features),
        "n_val": len(val_features),
        "train_mean": train_mean.cpu().numpy(),
        "train_std": train_std.cpu().numpy(),
    }

    return model, metrics, final_pred


def main():
    parser = argparse.ArgumentParser(description="Train retrace timing predictor (MLP regression)")
    parser.add_argument(
        "--features-dir",
        type=str,
        default=r"C:\Users\claude\Lvl3Quant\data\processed\mbo_events_smart_v3",
        help="Directory with *_mbo_events.npz feature files",
    )
    parser.add_argument(
        "--predictions-dir",
        type=str,
        default=r"C:\Users\claude\Lvl3Quant\output\cnn_mamba_v2_bulk_oot_v2",
        help="Directory with *_oot_predictions.npz from CNN-Mamba v2",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=r"C:\Users\claude\Lvl3Quant\output\retrace_timing_v1",
        help="Directory for fold predictions and model weights",
    )
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default="http://jupiter:5000",
        help="MLflow tracking URI",
    )
    parser.add_argument(
        "--short-percentile",
        type=float,
        default=SHORT_PERCENTILE,
        help="Top N%% short signals to train on (default: 5)",
    )
    args = parser.parse_args()

    features_dir = Path(args.features_dir)
    predictions_dir = Path(args.predictions_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate directories
    if not features_dir.exists():
        log.error(f"Features directory not found: {features_dir}")
        sys.exit(1)
    if not predictions_dir.exists():
        log.error(f"Predictions directory not found: {predictions_dir}")
        sys.exit(1)

    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info(f"Using GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        device = torch.device("cpu")
        log.warning("CUDA not available, using CPU (will be slow)")

    # Discover dates
    dates, feat_files, pred_files = discover_dates(features_dir, predictions_dir)
    log.info(f"Date range: {dates[0]} to {dates[-1]}")

    # Compute global short signal threshold
    short_threshold = compute_short_threshold(dates, pred_files, args.short_percentile)

    # Setup MLflow
    mlflow_run = None
    if HAS_MLFLOW:
        try:
            mlflow.set_tracking_uri(args.mlflow_uri)
            mlflow.set_experiment("retrace_timing_v1")
            mlflow_run = mlflow.start_run(run_name=f"retrace_timing_v1_{time.strftime('%Y%m%d_%H%M%S')}")
            mlflow.log_params({
                "model": "MLP_128_64_32",
                "loss": "huber",
                "train_window": TRAIN_WINDOW,
                "test_window": TEST_WINDOW,
                "batch_size": BATCH_SIZE,
                "max_epochs": MAX_EPOCHS,
                "patience": PATIENCE,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "short_percentile": args.short_percentile,
                "short_threshold": float(short_threshold),
                "n_dates": len(dates),
                "dropout": 0.2,
            })
            log.info(f"MLflow run started: {mlflow_run.info.run_id}")
        except Exception as e:
            log.warning(f"MLflow setup failed: {e}. Continuing without tracking.")
            HAS_MLFLOW_ACTIVE = False
    else:
        log.warning("MLflow not installed. Continuing without tracking.")

    # ============================================================
    # Walk-forward loop
    # ============================================================

    n_folds = len(dates) - TRAIN_WINDOW
    all_fold_metrics = []
    all_oot_predictions = []
    all_oot_targets = []
    all_oot_dates = []

    log.info(f"Starting walk-forward: {n_folds} folds")
    log.info(f"=" * 70)

    for fold_idx in range(n_folds):
        fold_start = time.time()
        train_dates = dates[fold_idx : fold_idx + TRAIN_WINDOW]
        test_dates = dates[fold_idx + TRAIN_WINDOW : fold_idx + TRAIN_WINDOW + TEST_WINDOW]
        test_date = test_dates[0]

        log.info(
            f"Fold {fold_idx + 1}/{n_folds}: "
            f"train={train_dates[0]}..{train_dates[-1]}, test={test_date}"
        )

        # Load training data
        train_features_list = []
        train_targets_list = []
        for d in train_dates:
            result = load_date_data(d, feat_files, pred_files, short_threshold)
            if result is not None:
                train_features_list.append(result[0])
                train_targets_list.append(result[1])

        if not train_features_list:
            log.warning(f"  No training data for fold {fold_idx + 1}, skipping")
            continue

        train_features = np.concatenate(train_features_list, axis=0)
        train_targets = np.concatenate(train_targets_list, axis=0)

        # Load test data
        test_result = load_date_data(test_date, feat_files, pred_files, short_threshold)
        if test_result is None:
            log.warning(f"  No test data for {test_date}, skipping")
            continue

        val_features, val_targets = test_result

        # Handle NaN/Inf in features
        train_nan_mask = np.isfinite(train_features).all(axis=1) & np.isfinite(train_targets)
        val_nan_mask = np.isfinite(val_features).all(axis=1) & np.isfinite(val_targets)

        if train_nan_mask.sum() < 100:
            log.warning(f"  Too few valid training samples ({train_nan_mask.sum()}), skipping")
            continue

        train_features = train_features[train_nan_mask]
        train_targets = train_targets[train_nan_mask]
        val_features = val_features[val_nan_mask]
        val_targets = val_targets[val_nan_mask]

        if len(val_features) == 0:
            log.warning(f"  No valid val samples for {test_date}, skipping")
            continue

        # Target distribution
        unique_targets, counts = np.unique(np.round(train_targets, 2), return_counts=True)
        target_dist = ", ".join(f"{np.exp(t):.1f}s:{c}" for t, c in zip(unique_targets, counts))
        log.info(f"  Train: {len(train_features):,} events, Val: {len(val_features):,} events")
        log.info(f"  Target dist (exp): {target_dist}")

        # Create and train model
        model = RetraceTimingMLP(n_features=N_FEATURES, dropout=0.2).to(device)

        model, metrics, fold_predictions = train_one_fold(
            model, train_features, train_targets,
            val_features, val_targets, device, fold_idx + 1
        )

        fold_elapsed = time.time() - fold_start
        log.info(
            f"  RESULT fold {fold_idx + 1}: "
            f"val_loss={metrics['val_loss']:.4f}  "
            f"val_corr={metrics['val_corr']:.4f}  "
            f"best_epoch={metrics['best_epoch']}  "
            f"time={fold_elapsed:.1f}s"
        )

        # Save fold predictions
        fold_output = {
            "predictions": fold_predictions,
            "targets": val_targets,
            "date": test_date,
            "train_mean": metrics["train_mean"],
            "train_std": metrics["train_std"],
        }
        fold_path = output_dir / f"fold_{fold_idx + 1:03d}_{test_date}_retrace_timing.npz"
        np.savez_compressed(str(fold_path), **fold_output)

        # Save model weights
        model_path = output_dir / f"fold_{fold_idx + 1:03d}_{test_date}_model.pt"
        torch.save(model.state_dict(), str(model_path))

        # Accumulate for aggregate metrics
        all_fold_metrics.append({
            "fold": fold_idx + 1,
            "date": test_date,
            **{k: v for k, v in metrics.items() if k not in ("train_mean", "train_std")},
        })
        all_oot_predictions.append(fold_predictions)
        all_oot_targets.append(val_targets)
        all_oot_dates.append(test_date)

        # Log to MLflow
        if HAS_MLFLOW and mlflow_run:
            try:
                mlflow.log_metrics(
                    {
                        f"fold_val_loss": metrics["val_loss"],
                        f"fold_val_corr": metrics["val_corr"],
                    },
                    step=fold_idx + 1,
                )
            except Exception:
                pass

        # GPU memory cleanup
        del model, train_features, train_targets, val_features, val_targets
        torch.cuda.empty_cache()
        gc.collect()

    # ============================================================
    # Aggregate results
    # ============================================================

    log.info("=" * 70)
    log.info("AGGREGATE RESULTS")
    log.info("=" * 70)

    if not all_fold_metrics:
        log.error("No folds completed successfully!")
        if HAS_MLFLOW and mlflow_run:
            mlflow.end_run(status="FAILED")
        sys.exit(1)

    # Concat all OOT predictions
    concat_preds = np.concatenate(all_oot_predictions)
    concat_targets = np.concatenate(all_oot_targets)

    # Overall correlation
    if concat_preds.std() > 1e-8 and concat_targets.std() > 1e-8:
        concat_corr = np.corrcoef(concat_preds, concat_targets)[0, 1]
    else:
        concat_corr = 0.0

    # Per-fold stats
    val_losses = [m["val_loss"] for m in all_fold_metrics]
    val_corrs = [m["val_corr"] for m in all_fold_metrics]

    log.info(f"Total OOT events: {len(concat_preds):,}")
    log.info(f"Concat correlation: {concat_corr:.4f}")
    log.info(f"Mean fold val_loss: {np.mean(val_losses):.4f} ± {np.std(val_losses):.4f}")
    log.info(f"Mean fold val_corr: {np.mean(val_corrs):.4f} ± {np.std(val_corrs):.4f}")
    log.info(f"Median fold val_corr: {np.median(val_corrs):.4f}")
    log.info(f"Folds with positive corr: {sum(1 for c in val_corrs if c > 0)}/{len(val_corrs)}")

    # Analyze prediction quality by retrace bucket
    log.info("\nPrediction quality by actual retrace bucket:")
    for bucket_time, bucket_name in [
        (RETRACE_FAST, "fast (<1s)"),
        (RETRACE_MEDIUM, "medium (1-5s)"),
        (RETRACE_SLOW, "slow (5-10s)"),
        (RETRACE_VERY_SLOW, "very_slow (10-30s)"),
        (RETRACE_NEVER, "never (60s)"),
    ]:
        bucket_log = np.log(bucket_time)
        mask = np.abs(concat_targets - bucket_log) < 0.01
        if mask.sum() > 0:
            pred_mean = np.mean(concat_preds[mask])
            pred_std = np.std(concat_preds[mask])
            log.info(
                f"  {bucket_name:20s}: n={mask.sum():6,}  "
                f"pred_mean={pred_mean:.3f} (exp={np.exp(pred_mean):.1f}s)  "
                f"pred_std={pred_std:.3f}"
            )

    # Save concat predictions
    concat_path = output_dir / "concat_oot_retrace_timing.npz"
    np.savez_compressed(
        str(concat_path),
        predictions=concat_preds,
        targets=concat_targets,
        dates=np.array(all_oot_dates),
    )
    log.info(f"\nConcat predictions saved to: {concat_path}")

    # Log aggregate to MLflow
    if HAS_MLFLOW and mlflow_run:
        try:
            mlflow.log_metrics({
                "concat_corr": concat_corr,
                "mean_val_loss": float(np.mean(val_losses)),
                "mean_val_corr": float(np.mean(val_corrs)),
                "median_val_corr": float(np.median(val_corrs)),
                "std_val_corr": float(np.std(val_corrs)),
                "n_folds_completed": len(all_fold_metrics),
                "n_oot_events": len(concat_preds),
                "pct_positive_corr": sum(1 for c in val_corrs if c > 0) / len(val_corrs),
            })
            mlflow.end_run(status="FINISHED")
            log.info(f"MLflow run completed: {mlflow_run.info.run_id}")
        except Exception as e:
            log.warning(f"MLflow final logging failed: {e}")

    # Print per-fold summary table
    log.info("\nPer-fold summary:")
    log.info(f"{'Fold':>4}  {'Date':>12}  {'Val Loss':>10}  {'Val Corr':>10}  {'N_train':>8}  {'N_val':>8}")
    log.info("-" * 60)
    for m in all_fold_metrics:
        log.info(
            f"{m['fold']:4d}  {m['date']:>12}  {m['val_loss']:10.4f}  "
            f"{m['val_corr']:10.4f}  {m['n_train']:8,}  {m['n_val']:8,}"
        )

    log.info(f"\nDone. Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
