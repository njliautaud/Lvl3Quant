#!/usr/bin/env python3
"""
train_exit_fill_predictor_v1.py — Exit Fill Probability Predictor
=================================================================
Trains a small MLP to predict: after a passive short fill (sell at ask),
will mid price retrace DOWN by at least T ticks within H seconds?

If yes → passive exit (buy at bid) is likely to fill → take the trade.
If no  → skip the trade, passive exit won't fill in time.

Architecture: 25 smart_v3 features → 128 → 64 → 32 → 8 outputs
8 targets = 2 thresholds (0.5, 1.0 ticks) × 4 horizons (5s, 10s, 30s, 60s)

Target computation:
  For each event i, chain labels_1s forward at 1s resolution to build
  a mid price path over the next H seconds. Compute MFE (minimum cumulative
  change = maximum down move). If MFE <= -T → label = 1.

Walk-forward: 10-day train, 1-day OOT, sliding window.

Usage (on Razer):
  C:\\Users\\claude\\Lvl3Quant\\.venv_research\\Scripts\\python.exe ^
    scripts/train_exit_fill_predictor_v1.py ^
    --output-dir output/exit_fill_predictor_v1 ^
    --mlflow-uri http://jupiter:5000

Author: Claude (autonomous research)
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Optional imports with fallbacks
try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

# ============================================================
# Configuration
# ============================================================

FEATURE_NAMES = [
    "time_delta_log", "event_type_id", "side_id",
    "price_rel_ticks", "qty_log", "spread_ticks",
    "cancel_side_asym_50", "rolling_ofi_500", "event_density_20",
    "price_mom_10", "qty_price_mom_50", "price_sign_momentum_200",
    "event_type_entropy_200", "fill_add_restoration_100", "spread_velocity_50",
    "queue_replenishment", "mom_divergence", "ofi_x_spread",
    "vol_weighted_pmom", "buy_sell_intensity_ratio", "realized_volatility",
    "sweep_intensity", "ofi_short_100", "ofi_long_2000", "ofi_acceleration",
]

N_FEATURES = 25
TICK_THRESHOLDS = [0.5, 1.0]                    # T in ticks
HORIZON_SECONDS = [5, 10, 30, 60]               # H in seconds
N_TARGETS = len(TICK_THRESHOLDS) * len(HORIZON_SECONDS)  # 8

TARGET_NAMES = [
    f"retrace_{t}t_{h}s"
    for t in TICK_THRESHOLDS
    for h in HORIZON_SECONDS
]

# Training hyperparams
BATCH_SIZE = 2048
LR = 1e-3
EPOCHS = 5
DROPOUT = 0.2

# Walk-forward
TRAIN_DAYS = 10
OOT_DAYS = 1

# Data
EVENT_DIR_DEFAULT = "data/processed/mbo_events_smart_v3"
SUBSAMPLE_STRIDE = 1000  # Take every Nth event (~250ms pred stride). 19M→19K/day.


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("exit_fill")


# ============================================================
# Model
# ============================================================

class ExitFillMLP(nn.Module):
    """Small MLP for multi-task binary classification."""

    def __init__(self, n_in: int = N_FEATURES, n_out: int = N_TARGETS, dropout: float = DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_out),
        )

    def forward(self, x):
        return self.net(x)  # raw logits, apply sigmoid for probabilities


# ============================================================
# Target computation
# ============================================================

def compute_targets_for_day(
    labels_1s: np.ndarray,
    timestamps: np.ndarray,
    tick_thresholds: List[float],
    horizon_seconds: List[int],
) -> np.ndarray:
    """Compute binary retrace targets by chaining labels_1s forward.

    For each event i, build a 1s-resolution mid price path by chaining
    labels_1s forward. Track running minimum (MFE for shorts).
    If running_min <= -T at any point within H seconds → label = 1.

    Args:
        labels_1s: (N,) mid price change over next 1s in ticks
        timestamps: (N,) nanosecond timestamps
        tick_thresholds: list of T values
        horizon_seconds: list of H values

    Returns:
        targets: (N, n_targets) float32 binary labels (0/1), NaN for invalid
    """
    N = len(labels_1s)
    n_targets = len(tick_thresholds) * len(horizon_seconds)
    max_horizon = max(horizon_seconds)
    targets = np.full((N, n_targets), np.nan, dtype=np.float32)

    # Precompute: for each event i, find the event index ~1s later
    # Using vectorized searchsorted for speed
    target_times = timestamps + 1_000_000_000  # +1 second in nanoseconds
    next_event_idx = np.searchsorted(timestamps, target_times)
    # Clamp to valid range
    next_event_idx = np.clip(next_event_idx, 0, N - 1)

    # For efficiency, process in chunks
    # Chain labels_1s forward up to max_horizon steps
    # Build MFE array: mfe_at_step[i, s] = min cumulative mid change from event i through step s

    log.info(f"  Computing retrace targets for {N:,} events, max horizon {max_horizon}s...")
    t0 = time.time()

    # Strategy: for each event, chain forward max_horizon steps of 1s each
    # Track cumulative change and running minimum
    # This is O(N * max_horizon) but max_horizon is small (60)

    # Precompute jump table: event_at_step[i] = event index that is ~1s after event i
    # Then event_at_step[event_at_step[i]] = event index ~2s after event i, etc.

    # Build the jump table
    jump = next_event_idx.copy()  # jump[i] = event at t_i + 1s

    # Now chain forward and compute MFE
    # For memory efficiency, process in blocks
    BLOCK = 500_000
    for block_start in range(0, N, BLOCK):
        block_end = min(block_start + BLOCK, N)
        block_size = block_end - block_start

        # Track current position and cumulative change for this block
        current_idx = np.arange(block_start, block_end, dtype=np.int64)
        cum_change = np.zeros(block_size, dtype=np.float64)
        running_min = np.zeros(block_size, dtype=np.float64)
        valid = np.ones(block_size, dtype=bool)

        # Which horizon checkpoints have we passed?
        horizon_set = sorted(set(horizon_seconds))
        horizon_reached = {h: False for h in horizon_set}

        for step in range(1, max_horizon + 1):
            # Get labels_1s at current position
            label_vals = labels_1s[current_idx]

            # Mark invalid where label is NaN or we've hit the end
            step_invalid = np.isnan(label_vals) | (current_idx >= N - 1)
            valid &= ~step_invalid

            # Accumulate for valid events
            cum_change[valid] += label_vals[valid]
            running_min[valid] = np.minimum(running_min[valid], cum_change[valid])

            # Check if this step completes any horizon
            if step in horizon_set:
                for t_idx, T in enumerate(tick_thresholds):
                    target_col = t_idx * len(horizon_seconds) + horizon_set.index(step)
                    # Label = 1 if running min reached -T at any point up to this horizon
                    block_targets = np.where(valid, (running_min <= -T).astype(np.float32), np.nan)
                    targets[block_start:block_end, target_col] = block_targets

            # Advance: jump to next 1s event
            next_idx = jump[current_idx]
            # Check for events that didn't advance (stuck at end)
            stuck = (next_idx == current_idx)
            valid &= ~stuck
            current_idx = next_idx

        if block_start % (BLOCK * 5) == 0 and block_start > 0:
            elapsed = time.time() - t0
            pct = block_end / N * 100
            log.info(f"    {pct:.0f}% done ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    log.info(f"  Target computation done in {elapsed:.1f}s")

    return targets


def log_positive_rates(targets: np.ndarray, target_names: List[str], prefix: str = ""):
    """Log positive rate for each target. Warn if too imbalanced."""
    for i, name in enumerate(target_names):
        col = targets[:, i]
        valid = ~np.isnan(col)
        if valid.sum() == 0:
            log.warning(f"  {prefix}{name}: ALL NaN")
            continue
        pos_rate = col[valid].mean()
        n_valid = valid.sum()
        flag = ""
        if pos_rate < 0.05:
            flag = " ⚠ VERY LOW (<5%)"
        elif pos_rate > 0.95:
            flag = " ⚠ VERY HIGH (>95%)"
        elif pos_rate < 0.10:
            flag = " (low)"
        elif pos_rate > 0.90:
            flag = " (high)"
        log.info(f"  {prefix}{name}: pos_rate={pos_rate:.4f} ({pos_rate*100:.1f}%), "
                 f"n_valid={n_valid:,}{flag}")


# ============================================================
# Data loading
# ============================================================

def get_available_dates(event_dir: Path) -> List[str]:
    """Get sorted list of available date strings."""
    dates = []
    for f in sorted(event_dir.glob("*_mbo_events.npz")):
        date_str = f.name[:8]
        # Skip weekends and obvious non-trading days
        dates.append(date_str)
    return sorted(dates)


def load_day(event_dir: Path, date_str: str) -> Optional[Dict]:
    """Load features, labels, timestamps for one day. Subsamples to every SUBSAMPLE_STRIDE events."""
    path = event_dir / f"{date_str}_mbo_events.npz"
    if not path.exists():
        return None
    try:
        data = np.load(path, mmap_mode="r")
        # Subsample: take every Nth event to reduce 19M→~19K per day
        n_raw = data["events"].shape[0]
        idx = np.arange(0, n_raw, SUBSAMPLE_STRIDE)
        features = np.array(data["events"][idx])  # (N/stride, 25)
        labels_1s = np.array(data["labels_1s"][idx])
        timestamps = np.array(data["timestamps"][idx])

        if features.shape[1] != N_FEATURES:
            log.warning(f"  {date_str}: unexpected feature count {features.shape[1]}, expected {N_FEATURES}")
            return None

        return {
            "features": features,
            "labels_1s": labels_1s,
            "timestamps": timestamps,
            "n_events": len(features),
            "n_raw": n_raw,
        }
    except Exception as e:
        log.error(f"  Failed to load {date_str}: {e}")
        return None


def load_and_compute_targets(
    event_dir: Path,
    dates: List[str],
    tick_thresholds: List[float],
    horizon_seconds: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Load multiple days and compute targets. Returns (features, targets)."""
    all_features = []
    all_targets = []

    for date_str in dates:
        day_data = load_day(event_dir, date_str)
        if day_data is None:
            log.warning(f"  Skipping {date_str} — load failed")
            continue

        log.info(f"  {date_str}: {day_data['n_events']:,} events")

        # Compute targets
        targets = compute_targets_for_day(
            day_data["labels_1s"],
            day_data["timestamps"],
            tick_thresholds,
            horizon_seconds,
        )

        all_features.append(day_data["features"])
        all_targets.append(targets)

    if not all_features:
        return np.empty((0, N_FEATURES)), np.empty((0, N_TARGETS))

    return np.concatenate(all_features), np.concatenate(all_targets)


# ============================================================
# Training
# ============================================================

def train_one_fold(
    train_X: np.ndarray,
    train_Y: np.ndarray,
    val_X: np.ndarray,
    val_Y: np.ndarray,
    device: torch.device,
    fold_idx: int,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
) -> Tuple[ExitFillMLP, Dict]:
    """Train MLP for one fold. Returns (model, metrics_dict)."""

    # Z-score normalization (per-fold)
    train_mean = np.nanmean(train_X, axis=0)
    train_std = np.nanstd(train_X, axis=0)
    train_std[train_std < 1e-8] = 1.0  # avoid div by zero

    train_X_norm = (train_X - train_mean) / train_std
    val_X_norm = (val_X - train_mean) / train_std

    # Replace NaN/Inf in features
    train_X_norm = np.nan_to_num(train_X_norm, nan=0.0, posinf=5.0, neginf=-5.0)
    val_X_norm = np.nan_to_num(val_X_norm, nan=0.0, posinf=5.0, neginf=-5.0)

    # Create valid mask (events with non-NaN targets)
    # We need at least ONE valid target per event for training
    train_valid = ~np.isnan(train_Y).all(axis=1)
    val_valid = ~np.isnan(val_Y).all(axis=1)

    train_X_t = torch.from_numpy(train_X_norm[train_valid]).float().to(device)
    train_Y_t = torch.from_numpy(np.nan_to_num(train_Y[train_valid], nan=0.0)).float().to(device)
    train_mask = torch.from_numpy((~np.isnan(train_Y[train_valid])).astype(np.float32)).to(device)

    val_X_t = torch.from_numpy(val_X_norm[val_valid]).float().to(device)
    val_Y_t = torch.from_numpy(np.nan_to_num(val_Y[val_valid], nan=0.0)).float().to(device)
    val_mask = torch.from_numpy((~np.isnan(val_Y[val_valid])).astype(np.float32)).to(device)

    log.info(f"  Fold {fold_idx:02d}: train={train_X_t.shape[0]:,} val={val_X_t.shape[0]:,}")

    # DataLoader
    train_ds = TensorDataset(train_X_t, train_Y_t, train_mask)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=0, pin_memory=False, drop_last=False)

    # Model
    model = ExitFillMLP(n_in=N_FEATURES, n_out=N_TARGETS, dropout=DROPOUT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Training loop
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch_idx, (bx, by, bmask) in enumerate(train_dl):
            logits = model(bx)
            # Masked BCE loss: only compute loss where target is valid
            loss_per_target = F.binary_cross_entropy_with_logits(logits, by, reduction="none")
            loss = (loss_per_target * bmask).sum() / bmask.sum().clamp(min=1.0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            if (batch_idx + 1) % 200 == 0:
                log.info(f"    Epoch {epoch+1}/{epochs} batch {batch_idx+1}/{len(train_dl)} "
                         f"loss={loss.item():.4f}")

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(val_X_t)
            val_loss_per = F.binary_cross_entropy_with_logits(val_logits, val_Y_t, reduction="none")
            val_loss = (val_loss_per * val_mask).sum() / val_mask.sum().clamp(min=1.0)
            val_loss_val = val_loss.item()

        if val_loss_val < best_val_loss:
            best_val_loss = val_loss_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        log.info(f"    Epoch {epoch+1}/{epochs}: train_loss={avg_train_loss:.4f} "
                 f"val_loss={val_loss_val:.4f} {'*' if val_loss_val == best_val_loss else ''}")

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # Compute OOT predictions and metrics
    with torch.no_grad():
        val_logits = model(val_X_t)
        val_probs = torch.sigmoid(val_logits).cpu().numpy()

    val_labels = val_Y_t.cpu().numpy()
    val_mask_np = val_mask.cpu().numpy()

    metrics = {}
    for i, name in enumerate(TARGET_NAMES):
        col_mask = val_mask_np[:, i] > 0.5
        if col_mask.sum() < 100:
            continue
        y_true = val_labels[col_mask, i]
        y_prob = val_probs[col_mask, i]

        # AUC and AP
        n_pos = y_true.sum()
        n_neg = col_mask.sum() - n_pos
        if n_pos < 10 or n_neg < 10:
            log.info(f"    {name}: skipped (too few pos={n_pos:.0f} or neg={n_neg:.0f})")
            continue

        if HAS_SKLEARN:
            auc = roc_auc_score(y_true, y_prob)
            ap = average_precision_score(y_true, y_prob)
        else:
            auc = manual_auc(y_true, y_prob)
            ap = manual_ap(y_true, y_prob)

        metrics[f"{name}_auc"] = auc
        metrics[f"{name}_ap"] = ap
        metrics[f"{name}_pos_rate"] = float(y_true.mean())

        log.info(f"    {name}: AUC={auc:.4f} AP={ap:.4f} pos_rate={y_true.mean():.3f}")

    return model, metrics, {
        "val_probs": val_probs,
        "val_labels": val_labels,
        "val_mask": val_mask_np,
        "train_mean": train_mean,
        "train_std": train_std,
        "best_val_loss": best_val_loss,
    }


# ============================================================
# Manual AUC/AP (fallback if sklearn unavailable)
# ============================================================

def manual_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUC-ROC without sklearn."""
    desc_idx = np.argsort(-y_score)
    y_sorted = y_true[desc_idx]
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tp = 0
    fp = 0
    auc = 0.0
    prev_fp = 0
    prev_tp = 0
    for i in range(len(y_sorted)):
        if y_sorted[i] > 0.5:
            tp += 1
        else:
            fp += 1
            auc += tp  # rectangle area
    return auc / (n_pos * n_neg)


def manual_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute average precision without sklearn."""
    desc_idx = np.argsort(-y_score)
    y_sorted = y_true[desc_idx]
    n_pos = y_true.sum()
    if n_pos == 0:
        return 0.0
    tp = 0
    ap = 0.0
    for i in range(len(y_sorted)):
        if y_sorted[i] > 0.5:
            tp += 1
            ap += tp / (i + 1)
    return ap / n_pos


# ============================================================
# Calibration analysis
# ============================================================

def compute_calibration(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Dict:
    """Compute calibration: predicted probability vs actual frequency."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_freqs = []
    bin_counts = []

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        bin_centers.append((lo + hi) / 2)
        bin_freqs.append(y_true[mask].mean())
        bin_counts.append(mask.sum())

    return {
        "bin_centers": bin_centers,
        "bin_actual_freq": bin_freqs,
        "bin_counts": bin_counts,
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train Exit Fill Predictor v1")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for models and predictions")
    parser.add_argument("--event-dir", type=str, default=None,
                        help="MBO events directory (default: auto-detect)")
    parser.add_argument("--mlflow-uri", type=str, default="http://jupiter:5000",
                        help="MLflow tracking URI")
    parser.add_argument("--n-folds", type=int, default=23,
                        help="Number of walk-forward folds")
    parser.add_argument("--train-days", type=int, default=TRAIN_DAYS,
                        help="Training window size in days")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--device", type=str, default=None,
                        help="Device (auto-detect if not specified)")
    args = parser.parse_args()

    # Auto-detect paths
    # Try multiple roots for cross-platform compatibility
    roots_to_try = [
        Path("C:/Users/claude/Lvl3Quant"),   # Razer Windows
        Path("/home/jupiter/Lvl3Quant"),      # Jupiter
        Path("/home/nick/Lvl3Quant"),         # Neptune
    ]

    lvl3_root = None
    for root in roots_to_try:
        if root.exists():
            lvl3_root = root
            break

    if lvl3_root is None:
        log.error("Could not find Lvl3Quant root directory")
        sys.exit(1)

    event_dir = Path(args.event_dir) if args.event_dir else lvl3_root / EVENT_DIR_DEFAULT
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = lvl3_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Lvl3Quant root: {lvl3_root}")
    log.info(f"Event dir: {event_dir}")
    log.info(f"Output dir: {output_dir}")

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    log.info(f"Device: {device}")

    # Get available dates
    all_dates = get_available_dates(event_dir)
    log.info(f"Available dates: {len(all_dates)}")

    if len(all_dates) < args.train_days + args.n_folds:
        log.error(f"Not enough dates: {len(all_dates)} < {args.train_days} + {args.n_folds}")
        sys.exit(1)

    # Use the LAST (train_days + n_folds) dates for walk-forward
    # This ensures we use the most recent data
    total_needed = args.train_days + args.n_folds
    dates_subset = all_dates[-total_needed:]
    log.info(f"Using dates {dates_subset[0]} to {dates_subset[-1]} "
             f"({len(dates_subset)} days, {args.n_folds} folds)")

    # MLflow setup
    mlflow_run = None
    if HAS_MLFLOW:
        try:
            mlflow.set_tracking_uri(args.mlflow_uri)
            mlflow.set_experiment("exit_fill_predictor_v1")
            mlflow_run = mlflow.start_run(run_name=f"exit_fill_v1_{time.strftime('%Y%m%d_%H%M%S')}")
            mlflow.log_params({
                "n_features": N_FEATURES,
                "n_targets": N_TARGETS,
                "tick_thresholds": str(TICK_THRESHOLDS),
                "horizon_seconds": str(HORIZON_SECONDS),
                "train_days": args.train_days,
                "n_folds": args.n_folds,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "dropout": DROPOUT,
                "architecture": "MLP_128_64_32",
                "device": str(device),
                "dates_range": f"{dates_subset[0]}-{dates_subset[-1]}",
            })
            log.info(f"MLflow run started: {mlflow_run.info.run_id}")
        except Exception as e:
            log.warning(f"MLflow setup failed: {e} — continuing without tracking")
            HAS_MLFLOW_ACTIVE = False
    else:
        HAS_MLFLOW_ACTIVE = False
        log.info("MLflow not available — skipping tracking")

    # Walk-forward training
    all_oot_probs = []
    all_oot_labels = []
    all_oot_masks = []
    fold_metrics_list = []

    for fold_idx in range(args.n_folds):
        fold_start = fold_idx
        train_dates = dates_subset[fold_start:fold_start + args.train_days]
        oot_date = dates_subset[fold_start + args.train_days]

        log.info(f"\n{'='*60}")
        log.info(f"FOLD {fold_idx:02d}: train={train_dates[0]}..{train_dates[-1]}, OOT={oot_date}")
        log.info(f"{'='*60}")

        # Check for resume
        pred_path = output_dir / f"fold_{fold_idx:02d}_preds.npz"
        model_path = output_dir / f"fold_{fold_idx:02d}_model.pt"
        if pred_path.exists() and model_path.exists():
            log.info(f"  Fold {fold_idx:02d} already complete — loading cached predictions")
            cached = np.load(pred_path, allow_pickle=True)
            all_oot_probs.append(cached["probs"])
            all_oot_labels.append(cached["labels"])
            all_oot_masks.append(cached["masks"])

            # Reconstruct metrics from cached predictions
            fold_metrics = {}
            for i, name in enumerate(TARGET_NAMES):
                col_mask = cached["masks"][:, i] > 0.5
                if col_mask.sum() < 100:
                    continue
                y_true = cached["labels"][col_mask, i]
                y_prob = cached["probs"][col_mask, i]
                n_pos = y_true.sum()
                n_neg = col_mask.sum() - n_pos
                if n_pos < 10 or n_neg < 10:
                    continue
                if HAS_SKLEARN:
                    fold_metrics[f"{name}_auc"] = roc_auc_score(y_true, y_prob)
                    fold_metrics[f"{name}_ap"] = average_precision_score(y_true, y_prob)
                else:
                    fold_metrics[f"{name}_auc"] = manual_auc(y_true, y_prob)
                    fold_metrics[f"{name}_ap"] = manual_ap(y_true, y_prob)
            fold_metrics_list.append(fold_metrics)
            continue

        # Load train data
        log.info(f"  Loading training data ({len(train_dates)} days)...")
        train_X, train_Y = load_and_compute_targets(
            event_dir, train_dates, TICK_THRESHOLDS, HORIZON_SECONDS
        )

        if len(train_X) == 0:
            log.warning(f"  Fold {fold_idx:02d}: no training data — skipping")
            continue

        log.info(f"  Training data: {train_X.shape[0]:,} events")
        log_positive_rates(train_Y, TARGET_NAMES, prefix="TRAIN ")

        # Load OOT data
        log.info(f"  Loading OOT data ({oot_date})...")
        oot_X, oot_Y = load_and_compute_targets(
            event_dir, [oot_date], TICK_THRESHOLDS, HORIZON_SECONDS
        )

        if len(oot_X) == 0:
            log.warning(f"  Fold {fold_idx:02d}: no OOT data — skipping")
            del train_X, train_Y
            continue

        log.info(f"  OOT data: {oot_X.shape[0]:,} events")
        log_positive_rates(oot_Y, TARGET_NAMES, prefix="OOT ")

        # Train
        model, fold_metrics, fold_extras = train_one_fold(
            train_X, train_Y, oot_X, oot_Y,
            device, fold_idx,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        )

        # Store OOT predictions
        all_oot_probs.append(fold_extras["val_probs"])
        all_oot_labels.append(fold_extras["val_labels"])
        all_oot_masks.append(fold_extras["val_mask"])
        fold_metrics_list.append(fold_metrics)

        # Save fold outputs
        np.savez_compressed(
            str(pred_path),
            probs=fold_extras["val_probs"],
            labels=fold_extras["val_labels"],
            masks=fold_extras["val_mask"],
            date=oot_date,
            target_names=TARGET_NAMES,
        )

        torch.save({
            "model_state_dict": model.state_dict(),
            "train_mean": fold_extras["train_mean"],
            "train_std": fold_extras["train_std"],
            "n_features": N_FEATURES,
            "n_targets": N_TARGETS,
            "target_names": TARGET_NAMES,
            "fold_idx": fold_idx,
            "train_dates": train_dates,
            "oot_date": oot_date,
        }, str(model_path))

        log.info(f"  Saved: {pred_path.name}, {model_path.name}")

        # Log to MLflow
        if HAS_MLFLOW and mlflow_run:
            try:
                for k, v in fold_metrics.items():
                    mlflow.log_metric(f"fold_{fold_idx:02d}/{k}", v, step=fold_idx)
                mlflow.log_metric(f"fold_{fold_idx:02d}/val_loss", fold_extras["best_val_loss"], step=fold_idx)
            except Exception:
                pass

        # Free memory
        del train_X, train_Y, oot_X, oot_Y, model, fold_extras
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ============================================================
    # Concat OOT metrics
    # ============================================================
    log.info(f"\n{'='*60}")
    log.info("CONCAT OOT METRICS (all folds combined)")
    log.info(f"{'='*60}")

    if not all_oot_probs:
        log.error("No completed folds — cannot compute concat metrics")
        if HAS_MLFLOW and mlflow_run:
            mlflow.end_run()
        return

    concat_probs = np.concatenate(all_oot_probs, axis=0)
    concat_labels = np.concatenate(all_oot_labels, axis=0)
    concat_masks = np.concatenate(all_oot_masks, axis=0)

    log.info(f"Total OOT events: {concat_probs.shape[0]:,}")

    concat_metrics = {}
    best_target_name = None
    best_target_auc = 0.0
    best_target_idx = 0

    for i, name in enumerate(TARGET_NAMES):
        col_mask = concat_masks[:, i] > 0.5
        if col_mask.sum() < 100:
            log.info(f"  {name}: insufficient valid samples ({col_mask.sum()})")
            continue
        y_true = concat_labels[col_mask, i]
        y_prob = concat_probs[col_mask, i]
        n_pos = y_true.sum()
        n_neg = col_mask.sum() - n_pos

        if n_pos < 10 or n_neg < 10:
            log.info(f"  {name}: too few positives ({n_pos:.0f}) or negatives ({n_neg:.0f})")
            continue

        if HAS_SKLEARN:
            auc = roc_auc_score(y_true, y_prob)
            ap = average_precision_score(y_true, y_prob)
        else:
            auc = manual_auc(y_true, y_prob)
            ap = manual_ap(y_true, y_prob)

        pos_rate = float(y_true.mean())
        concat_metrics[f"concat_{name}_auc"] = auc
        concat_metrics[f"concat_{name}_ap"] = ap
        concat_metrics[f"concat_{name}_pos_rate"] = pos_rate

        log.info(f"  {name}: AUC={auc:.4f} AP={ap:.4f} pos_rate={pos_rate:.3f} "
                 f"(n={col_mask.sum():,}, pos={n_pos:.0f})")

        if auc > best_target_auc:
            best_target_auc = auc
            best_target_name = name
            best_target_idx = i

    # Per-fold AUC summary
    log.info(f"\nPer-fold AUC summary:")
    for i, name in enumerate(TARGET_NAMES):
        key = f"{name}_auc"
        aucs = [fm.get(key, float("nan")) for fm in fold_metrics_list]
        valid_aucs = [a for a in aucs if not np.isnan(a)]
        if valid_aucs:
            log.info(f"  {name}: mean_AUC={np.mean(valid_aucs):.4f} "
                     f"std={np.std(valid_aucs):.4f} "
                     f"min={min(valid_aucs):.4f} max={max(valid_aucs):.4f} "
                     f"({len(valid_aucs)}/{len(aucs)} folds)")

    # ============================================================
    # Calibration for best target
    # ============================================================
    if best_target_name:
        log.info(f"\nCalibration analysis for best target: {best_target_name} (AUC={best_target_auc:.4f})")
        col_mask = concat_masks[:, best_target_idx] > 0.5
        y_true = concat_labels[col_mask, best_target_idx]
        y_prob = concat_probs[col_mask, best_target_idx]
        cal = compute_calibration(y_true, y_prob, n_bins=10)

        log.info(f"  {'Bin Center':>10} {'Actual Freq':>12} {'Count':>10}")
        for center, freq, count in zip(cal["bin_centers"], cal["bin_actual_freq"], cal["bin_counts"]):
            log.info(f"  {center:10.2f} {freq:12.4f} {count:10,}")

    # ============================================================
    # Save concat predictions
    # ============================================================
    concat_path = output_dir / "concat_oot_predictions.npz"
    np.savez_compressed(
        str(concat_path),
        probs=concat_probs,
        labels=concat_labels,
        masks=concat_masks,
        target_names=TARGET_NAMES,
        feature_names=FEATURE_NAMES,
        tick_thresholds=TICK_THRESHOLDS,
        horizon_seconds=HORIZON_SECONDS,
    )
    log.info(f"\nSaved concat predictions: {concat_path}")

    # ============================================================
    # MLflow final metrics
    # ============================================================
    if HAS_MLFLOW and mlflow_run:
        try:
            for k, v in concat_metrics.items():
                mlflow.log_metric(k, v)
            if best_target_name:
                mlflow.log_metric("best_target_auc", best_target_auc)
                mlflow.log_param("best_target", best_target_name)
            mlflow.log_metric("total_oot_events", concat_probs.shape[0])
            mlflow.log_metric("n_completed_folds", len(fold_metrics_list))
            mlflow.end_run()
            log.info("MLflow run completed")
        except Exception as e:
            log.warning(f"MLflow finalization failed: {e}")
            try:
                mlflow.end_run()
            except Exception:
                pass

    # ============================================================
    # Summary
    # ============================================================
    log.info(f"\n{'='*60}")
    log.info("SUMMARY")
    log.info(f"{'='*60}")
    log.info(f"Completed {len(fold_metrics_list)} folds over {len(dates_subset)} dates")
    log.info(f"Total OOT events: {concat_probs.shape[0]:,}")
    log.info(f"Best target: {best_target_name} (AUC={best_target_auc:.4f})")
    log.info(f"Output: {output_dir}")
    log.info(f"\nTarget AUCs (concat):")
    for name in TARGET_NAMES:
        auc_key = f"concat_{name}_auc"
        ap_key = f"concat_{name}_ap"
        if auc_key in concat_metrics:
            log.info(f"  {name}: AUC={concat_metrics[auc_key]:.4f} AP={concat_metrics[ap_key]:.4f}")

    # Key question: is book state predictive of retrace probability?
    predictive_targets = [name for name in TARGET_NAMES
                          if concat_metrics.get(f"concat_{name}_auc", 0.5) > 0.55]
    if predictive_targets:
        log.info(f"\nPREDICTIVE targets (AUC > 0.55): {', '.join(predictive_targets)}")
        log.info("Book microstructure IS predictive of exit fill probability!")
    else:
        log.info("\nNo targets exceeded AUC 0.55 — book state may not predict exit fills well.")
        log.info("Consider: (a) adding more features, (b) longer training, (c) different thresholds")


if __name__ == "__main__":
    main()
