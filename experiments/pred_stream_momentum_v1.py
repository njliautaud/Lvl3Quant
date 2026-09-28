#!/usr/bin/env python3
"""
Prediction Stream Momentum v1
==============================
MLP that learns temporal patterns in CNN-Mamba v2 prediction streams.

Hypothesis: A rolling window of recent predictions (last 20 = 5 seconds at 250ms stride)
contains momentum/mean-reversion patterns that predict trade profitability beyond
what a single-point prediction captures.

Features per window:
  - Raw predictions at each of 3 horizons (1s, 5s, 10s) x 20 steps = 60
  - Signed change from previous prediction (3 x 19 = 57, first step has no delta)
  - Summary stats: mean, std, slope, max, min, sign-consistency per horizon (6 x 3 = 18)
  Total: 60 + 57 + 18 = 135 features

Architecture: MLP 256->128->64 with BatchNorm+GELU+Dropout(0.2)
Target: 1s forward return (labels[:,0] from CNN-Mamba)
Walk-forward: sliding 10d train / 3d eval
"""

import os
import sys
import glob
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import spearmanr, linregress
from pathlib import Path
from datetime import datetime

# ── Config ──────────────────────────────────────────────────────────────────
WINDOW_SIZE = 20           # 20 predictions = 5 seconds at 250ms stride
TRAIN_DAYS = 10
EVAL_DAYS = 3
BATCH_SIZE = 2048
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.2
HIDDEN_DIMS = [256, 128, 64]
PATIENCE = 5               # early stopping patience
COMMISSION_TICKS = 0.376   # passive-passive AMP/Rithmic
TICK_VALUE = 12.50

DATA_DIR = "/home/nick/Lvl3Quant/output/cnn_mamba_v2_bulk_oot"
META_V7_PATH = "/home/nick/Lvl3Quant/output/meta_v7_prod/concat_oot_predictions.npz"
OUTPUT_DIR = "/home/nick/Lvl3Quant/output/pred_stream_momentum_v1"
MLFLOW_URI = "http://localhost:5000"
EXPERIMENT_NAME = "pred_stream_momentum_v1"

N_HORIZONS = 3  # 1s, 5s, 10s
# Features: raw preds (WINDOW*3) + deltas ((WINDOW-1)*3) + summary stats (6*3)
N_RAW = WINDOW_SIZE * N_HORIZONS         # 60
N_DELTA = (WINDOW_SIZE - 1) * N_HORIZONS  # 57
N_SUMMARY = 6 * N_HORIZONS               # 18
N_FEATURES = N_RAW + N_DELTA + N_SUMMARY  # 135

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Data Loading ────────────────────────────────────────────────────────────

def load_date_predictions(date_str: str) -> dict:
    """Load raw CNN-Mamba predictions for a single date."""
    path = os.path.join(DATA_DIR, f"{date_str}_predictions.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    return {
        "predictions": d["predictions"],  # (N, 3) - 1s, 5s, 10s
        "labels": d["labels"],            # (N, 3) - 1s, 5s, 10s forward returns
        "date": date_str,
    }


def get_available_dates() -> list:
    """Get sorted list of dates with available predictions."""
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*_predictions.npz")))
    dates = [os.path.basename(f).replace("_predictions.npz", "") for f in files]
    return dates


def build_window_features(predictions: np.ndarray) -> tuple:
    """
    Build rolling window features from time-ordered predictions (VECTORIZED).

    Args:
        predictions: (N, 3) array of predictions at 3 horizons

    Returns:
        features: (N - WINDOW_SIZE + 1, N_FEATURES) array
        valid_indices: indices into original array for each windowed sample
    """
    N = predictions.shape[0]
    if N < WINDOW_SIZE:
        return np.empty((0, N_FEATURES), dtype=np.float32), np.empty(0, dtype=np.int64)

    n_samples = N - WINDOW_SIZE + 1

    # Use stride_tricks to create rolling windows: (n_samples, WINDOW_SIZE, 3)
    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(predictions, WINDOW_SIZE, axis=0)  # (n_samples, 3, WINDOW_SIZE)
    windows = windows.transpose(0, 2, 1)  # (n_samples, WINDOW_SIZE, 3)
    # Force contiguous copy for fast downstream ops
    windows = np.ascontiguousarray(windows, dtype=np.float32)

    # 1. Raw predictions flattened: (n_samples, WINDOW_SIZE * 3)
    raw = windows.reshape(n_samples, -1)

    # 2. Signed deltas: (n_samples, (WINDOW_SIZE-1) * 3)
    deltas = np.diff(windows, axis=1).reshape(n_samples, -1)

    # 3. Summary stats per horizon (vectorized): 6 stats x 3 horizons = 18
    # windows shape: (n_samples, WINDOW_SIZE, 3)
    w_mean = windows.mean(axis=1)          # (n_samples, 3)
    w_std = windows.std(axis=1)            # (n_samples, 3)
    w_max = windows.max(axis=1)            # (n_samples, 3)
    w_min = windows.min(axis=1)            # (n_samples, 3)

    # Sign consistency: fraction matching sign of last prediction in window
    last_sign = np.sign(windows[:, -1:, :])  # (n_samples, 1, 3)
    all_signs = np.sign(windows)              # (n_samples, WINDOW_SIZE, 3)
    sign_match = (all_signs == last_sign).mean(axis=1)  # (n_samples, 3)
    # Where last_sign == 0, set to 0.5
    zero_mask = (last_sign.squeeze(1) == 0)
    sign_match[zero_mask] = 0.5

    # Slope via vectorized linear regression: slope = cov(x, y) / var(x)
    x = np.arange(WINDOW_SIZE, dtype=np.float32)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()
    # windows: (n_samples, WINDOW_SIZE, 3), x: (WINDOW_SIZE,)
    x_centered = x - x_mean  # (WINDOW_SIZE,)
    # cov = sum((x - x_mean) * (y - y_mean)) for each sample and horizon
    y_centered = windows - w_mean[:, np.newaxis, :]  # (n_samples, WINDOW_SIZE, 3)
    cov_xy = (x_centered[np.newaxis, :, np.newaxis] * y_centered).sum(axis=1)  # (n_samples, 3)
    slope = cov_xy / x_var  # (n_samples, 3)
    # Zero slope where std is tiny
    slope[w_std < 1e-8] = 0.0

    # Interleave summary stats: [mean_h0, std_h0, slope_h0, max_h0, min_h0, sign_h0, mean_h1, ...]
    summary = np.empty((n_samples, N_SUMMARY), dtype=np.float32)
    for h in range(N_HORIZONS):
        base = h * 6
        summary[:, base + 0] = w_mean[:, h]
        summary[:, base + 1] = w_std[:, h]
        summary[:, base + 2] = slope[:, h]
        summary[:, base + 3] = w_max[:, h]
        summary[:, base + 4] = w_min[:, h]
        summary[:, base + 5] = sign_match[:, h]

    features = np.concatenate([raw, deltas, summary], axis=1).astype(np.float32)
    valid_indices = np.arange(WINDOW_SIZE - 1, N)
    return features, valid_indices


def build_dataset_for_dates(dates: list) -> tuple:
    """Build features and labels for a list of dates."""
    all_features = []
    all_labels = []
    all_dates = []

    for date_str in dates:
        data = load_date_predictions(date_str)
        if data is None:
            print(f"  WARNING: No data for {date_str}, skipping")
            continue

        preds = data["predictions"]  # (N, 3)
        labels_3h = data["labels"]   # (N, 3)

        features, valid_idx = build_window_features(preds)
        if len(features) == 0:
            print(f"  WARNING: Too few samples for {date_str}, skipping")
            continue

        # Target: 1s forward return (column 0)
        target = labels_3h[valid_idx, 0]

        # Filter out NaN/inf in labels AND features
        valid_mask = np.isfinite(target) & np.all(np.isfinite(features), axis=1)
        if valid_mask.sum() == 0:
            print(f"  WARNING: All NaN labels for {date_str}, skipping")
            continue
        if valid_mask.sum() < len(target):
            n_dropped = len(target) - valid_mask.sum()
            # Only warn if significant number dropped
            if n_dropped > 100:
                print(f"  WARNING: Dropped {n_dropped} NaN samples from {date_str}")

        features = features[valid_mask]
        target = target[valid_mask]

        all_features.append(features)
        all_labels.append(target)
        all_dates.extend([date_str] * len(features))

    if not all_features:
        return np.empty((0, N_FEATURES)), np.empty(0), np.array([])

    return (
        np.concatenate(all_features, axis=0),
        np.concatenate(all_labels, axis=0),
        np.array(all_dates),
    )


# ── Dataset & Model ────────────────────────────────────────────────────────

class StreamDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).float()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class PredStreamMLP(nn.Module):
    def __init__(self, input_dim: int = N_FEATURES, hidden_dims: list = None, dropout: float = DROPOUT):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = HIDDEN_DIMS

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ── Training ────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for features, labels in loader:
        features, labels = features.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        preds = model(features)
        loss = criterion(preds, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    n_batches = 0
    criterion = nn.HuberLoss(delta=1.0)

    for features, labels in loader:
        features, labels = features.to(DEVICE), labels.to(DEVICE)
        preds = model(features)
        total_loss += criterion(preds, labels).item()
        n_batches += 1
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    loss = total_loss / max(n_batches, 1)

    # Spearman correlation
    sp_corr = spearmanr(preds, labels).correlation if len(preds) > 10 else 0.0
    if np.isnan(sp_corr):
        sp_corr = 0.0

    return loss, sp_corr, preds, labels


def compute_pnl_metrics(preds: np.ndarray, labels: np.ndarray, thresholds: list = [5, 10, 20]):
    """Compute P&L metrics at various prediction confidence thresholds."""
    results = {}
    n = len(preds)

    for pct in thresholds:
        k = max(1, int(n * pct / 100))

        # Top predictions (most positive = long signal)
        top_idx = np.argsort(preds)[-k:]
        # Bottom predictions (most negative = short signal)
        bot_idx = np.argsort(preds)[:k]

        # For long trades: profit = label - cost
        long_pnl = labels[top_idx] - COMMISSION_TICKS
        # For short trades: profit = -label - cost
        short_pnl = -labels[bot_idx] - COMMISSION_TICKS

        combined_pnl = np.concatenate([long_pnl, short_pnl])

        # Metrics
        wins = (combined_pnl > 0).sum()
        total = len(combined_pnl)
        wr = wins / total if total > 0 else 0

        gross_profit = combined_pnl[combined_pnl > 0].sum()
        gross_loss = abs(combined_pnl[combined_pnl < 0].sum())
        pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        mean_pnl = float(np.nanmean(combined_pnl))
        std_pnl = float(np.nanstd(combined_pnl))

        # Per-trade Sharpe (not annualized since these are individual trades)
        sharpe = mean_pnl / std_pnl if std_pnl > 1e-8 else 0.0

        # Sortino (downside deviation only)
        downside = combined_pnl[combined_pnl < 0]
        downside_std = float(downside.std()) if len(downside) > 1 else std_pnl
        sortino = mean_pnl / downside_std if downside_std > 1e-8 else 0.0

        results[f"top{pct}"] = {
            "n_trades": total,
            "wr": wr,
            "pf": pf,
            "sharpe": sharpe,
            "sortino": sortino,
            "mean_pnl_ticks": mean_pnl,
            "total_pnl_ticks": combined_pnl.sum(),
            "total_pnl_usd": combined_pnl.sum() * TICK_VALUE,
        }

    return results


# ── Walk-Forward Engine ─────────────────────────────────────────────────────

def run_walk_forward():
    """Run sliding window walk-forward training."""
    import mlflow

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # MLflow setup
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    all_dates = get_available_dates()
    print(f"Available dates ({len(all_dates)}): {all_dates}")

    # Filter to dates that overlap with meta_v7 OOT dates for consistency
    meta = np.load(META_V7_PATH, allow_pickle=True)
    meta_dates = set(np.unique(meta["dates"]))
    # Use ALL available dates from CNN-Mamba (superset)
    # but note which are also in meta_v7
    print(f"Meta v7 OOT dates: {sorted(meta_dates)}")
    print(f"CNN-Mamba dates not in meta v7: {sorted(set(all_dates) - meta_dates)}")

    min_required = TRAIN_DAYS + EVAL_DAYS
    if len(all_dates) < min_required:
        print(f"ERROR: Need at least {min_required} dates, have {len(all_dates)}")
        sys.exit(1)

    # Walk-forward folds
    n_folds = len(all_dates) - TRAIN_DAYS - EVAL_DAYS + 1
    print(f"\nWalk-forward: {n_folds} folds, {TRAIN_DAYS}d train / {EVAL_DAYS}d eval")
    print(f"Features: {N_FEATURES} (raw={N_RAW}, deltas={N_DELTA}, summary={N_SUMMARY})")
    print(f"Window size: {WINDOW_SIZE} predictions = {WINDOW_SIZE * 0.25:.1f}s")
    print(f"Device: {DEVICE}")
    print("=" * 80)

    all_oot_preds = []
    all_oot_labels = []
    all_oot_dates = []
    fold_metrics = []

    with mlflow.start_run(run_name=f"pred_stream_momentum_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log params
        mlflow.log_params({
            "window_size": WINDOW_SIZE,
            "train_days": TRAIN_DAYS,
            "eval_days": EVAL_DAYS,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "dropout": DROPOUT,
            "hidden_dims": str(HIDDEN_DIMS),
            "n_features": N_FEATURES,
            "n_folds": n_folds,
            "patience": PATIENCE,
            "commission_ticks": COMMISSION_TICKS,
            "target": "1s_forward_return",
        })

        for fold_idx in range(n_folds):
            fold_start = time.time()

            train_dates = all_dates[fold_idx:fold_idx + TRAIN_DAYS]
            eval_dates = all_dates[fold_idx + TRAIN_DAYS:fold_idx + TRAIN_DAYS + EVAL_DAYS]

            print(f"\n{'─' * 80}")
            print(f"Fold {fold_idx+1}/{n_folds}")
            print(f"  Train: {train_dates[0]} → {train_dates[-1]} ({len(train_dates)} days)")
            print(f"  Eval:  {eval_dates[0]} → {eval_dates[-1]} ({len(eval_dates)} days)")

            # Build datasets
            print("  Building train features...", end=" ", flush=True)
            t0 = time.time()
            train_X, train_y, train_d = build_dataset_for_dates(train_dates)
            print(f"{len(train_X)} samples in {time.time()-t0:.1f}s")

            print("  Building eval features...", end=" ", flush=True)
            t0 = time.time()
            eval_X, eval_y, eval_d = build_dataset_for_dates(eval_dates)
            print(f"{len(eval_X)} samples in {time.time()-t0:.1f}s")

            if len(train_X) == 0 or len(eval_X) == 0:
                print("  SKIPPING: insufficient data")
                continue

            # Clean NaN/inf in features
            train_X = np.nan_to_num(train_X, nan=0.0, posinf=0.0, neginf=0.0)
            eval_X = np.nan_to_num(eval_X, nan=0.0, posinf=0.0, neginf=0.0)

            # Normalize features using train stats (per-feature z-score)
            train_mean = train_X.mean(axis=0)
            train_std = train_X.std(axis=0)
            train_std[train_std < 1e-8] = 1.0  # avoid div by zero

            train_X_norm = (train_X - train_mean) / train_std
            eval_X_norm = (eval_X - train_mean) / train_std

            # Clip extreme feature values
            train_X_norm = np.clip(train_X_norm, -10, 10)
            eval_X_norm = np.clip(eval_X_norm, -10, 10)

            # Normalize labels for training (z-score), keep raw for P&L
            label_mean = train_y.mean()
            label_std = train_y.std()
            if label_std < 1e-8:
                label_std = 1.0
            train_y_norm = (train_y - label_mean) / label_std
            eval_y_norm = (eval_y - label_mean) / label_std

            # DataLoaders (use normalized labels for training)
            train_ds = StreamDataset(train_X_norm, train_y_norm)
            eval_ds = StreamDataset(eval_X_norm, eval_y_norm)
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
            eval_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

            # Also build a raw-label eval loader for P&L computation
            eval_ds_raw = StreamDataset(eval_X_norm, eval_y)
            eval_loader_raw = DataLoader(eval_ds_raw, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

            print(f"  Label stats: mean={label_mean:.4f} std={label_std:.4f} (train)")

            # Model
            model = PredStreamMLP().to(DEVICE)
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
            criterion = nn.HuberLoss(delta=1.0)

            best_eval_loss = float("inf")
            best_epoch = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            for epoch in range(EPOCHS):
                train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
                eval_loss, sp_corr, _, _ = evaluate(model, eval_loader)
                scheduler.step()

                # Check for NaN and skip
                if np.isnan(train_loss) or np.isnan(eval_loss):
                    print(f"    Epoch {epoch+1:3d}/{EPOCHS}: NaN detected, stopping fold")
                    break

                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    best_epoch = epoch
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

                if epoch % 5 == 0 or epoch == EPOCHS - 1:
                    print(f"    Epoch {epoch+1:3d}/{EPOCHS}: train_loss={train_loss:.6f} eval_loss={eval_loss:.6f} spearman={sp_corr:.4f}")

                # Early stopping
                if epoch - best_epoch >= PATIENCE:
                    print(f"    Early stopping at epoch {epoch+1} (best={best_epoch+1})")
                    break

            # Load best model and get final eval predictions (raw labels for P&L)
            model.load_state_dict(best_state)
            # Evaluate with normalized labels for Spearman (rank-invariant, doesn't matter)
            # but use raw labels for P&L
            eval_loss, sp_corr, eval_preds_norm, _ = evaluate(model, eval_loader)
            _, _, eval_preds, eval_labels = evaluate(model, eval_loader_raw)

            # P&L metrics
            pnl = compute_pnl_metrics(eval_preds, eval_labels)

            fold_time = time.time() - fold_start
            print(f"  RESULT: spearman={sp_corr:.4f}, best_epoch={best_epoch+1}, time={fold_time:.1f}s")
            for pct_key, m in pnl.items():
                print(f"    {pct_key}: WR={m['wr']:.3f} PF={m['pf']:.2f} Sharpe={m['sharpe']:.2f} Sortino={m['sortino']:.2f} PnL={m['total_pnl_ticks']:.1f}t")

            # Log to MLflow
            mlflow.log_metrics({
                f"fold_{fold_idx+1}_spearman": sp_corr,
                f"fold_{fold_idx+1}_eval_loss": eval_loss,
                f"fold_{fold_idx+1}_best_epoch": best_epoch + 1,
            }, step=fold_idx)

            for pct_key, m in pnl.items():
                mlflow.log_metrics({
                    f"fold_{fold_idx+1}_{pct_key}_wr": m["wr"],
                    f"fold_{fold_idx+1}_{pct_key}_pf": min(m["pf"], 99.0),
                    f"fold_{fold_idx+1}_{pct_key}_sharpe": m["sharpe"],
                    f"fold_{fold_idx+1}_{pct_key}_sortino": m["sortino"],
                }, step=fold_idx)

            # Save fold weights
            fold_dir = os.path.join(OUTPUT_DIR, f"fold_{fold_idx+1:03d}")
            os.makedirs(fold_dir, exist_ok=True)
            torch.save({
                "model_state_dict": best_state,
                "train_mean": train_mean,
                "train_std": train_std,
                "label_mean": label_mean,
                "label_std": label_std,
                "train_dates": train_dates,
                "eval_dates": eval_dates,
                "best_epoch": best_epoch,
                "spearman": sp_corr,
            }, os.path.join(fold_dir, "checkpoint.pt"))

            # Accumulate OOT predictions
            all_oot_preds.append(eval_preds)
            all_oot_labels.append(eval_labels)
            all_oot_dates.append(eval_d)

            fold_metrics.append({
                "fold": fold_idx + 1,
                "train_dates": train_dates,
                "eval_dates": eval_dates,
                "spearman": float(sp_corr),
                "eval_loss": float(eval_loss),
                "best_epoch": best_epoch + 1,
                "n_train": len(train_X),
                "n_eval": len(eval_X),
                "pnl": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in pnl.items()},
            })

        # ── Concat OOT predictions ──────────────────────────────────────────
        if all_oot_preds:
            concat_preds = np.concatenate(all_oot_preds)
            concat_labels = np.concatenate(all_oot_labels)
            concat_dates = np.concatenate(all_oot_dates)

            print(f"\n{'=' * 80}")
            print(f"CONCAT OOT RESULTS ({len(concat_preds)} samples, {len(np.unique(concat_dates))} dates)")

            # Overall Spearman
            overall_sp = spearmanr(concat_preds, concat_labels).correlation
            print(f"  Overall Spearman: {overall_sp:.4f}")

            # Per-date Spearman
            unique_eval_dates = sorted(np.unique(concat_dates))
            print(f"\n  Per-date breakdown:")
            date_sharpes = []
            for dt in unique_eval_dates:
                mask = concat_dates == dt
                dt_preds = concat_preds[mask]
                dt_labels = concat_labels[mask]
                dt_sp = spearmanr(dt_preds, dt_labels).correlation if len(dt_preds) > 10 else 0.0
                if np.isnan(dt_sp):
                    dt_sp = 0.0

                # Quick P&L for this date (top 10%)
                k = max(1, int(len(dt_preds) * 0.10))
                top_idx = np.argsort(dt_preds)[-k:]
                bot_idx = np.argsort(dt_preds)[:k]
                long_pnl = dt_labels[top_idx] - COMMISSION_TICKS
                short_pnl = -dt_labels[bot_idx] - COMMISSION_TICKS
                day_pnl = np.concatenate([long_pnl, short_pnl])
                day_mean = day_pnl.mean()
                day_std = day_pnl.std() if len(day_pnl) > 1 else 1.0
                day_sharpe = day_mean / day_std if day_std > 0 else 0.0
                date_sharpes.append(day_sharpe)

                print(f"    {dt}: n={mask.sum():6d} spearman={dt_sp:.4f} top10%_pnl={day_pnl.sum():.1f}t day_sharpe={day_sharpe:.3f}")

            # Concat P&L metrics
            concat_pnl = compute_pnl_metrics(concat_preds, concat_labels)
            print(f"\n  Concat P&L metrics:")
            for pct_key, m in concat_pnl.items():
                print(f"    {pct_key}: n={m['n_trades']} WR={m['wr']:.3f} PF={m['pf']:.2f} Sharpe={m['sharpe']:.2f} Sortino={m['sortino']:.2f} PnL={m['total_pnl_ticks']:.1f}t (${m['total_pnl_usd']:.0f})")

            # Log concat metrics to MLflow
            mlflow.log_metrics({
                "concat_spearman": overall_sp,
                "concat_n_samples": len(concat_preds),
                "concat_n_dates": len(unique_eval_dates),
            })
            for pct_key, m in concat_pnl.items():
                mlflow.log_metrics({
                    f"concat_{pct_key}_wr": m["wr"],
                    f"concat_{pct_key}_pf": min(m["pf"], 99.0),
                    f"concat_{pct_key}_sharpe": m["sharpe"],
                    f"concat_{pct_key}_sortino": m["sortino"],
                    f"concat_{pct_key}_total_pnl_ticks": m["total_pnl_ticks"],
                })

            # Regime analysis: classify dates by label mean (proxy for green/red)
            green_sharpes = []
            red_sharpes = []
            for i, dt in enumerate(unique_eval_dates):
                mask = concat_dates == dt
                dt_mean_label = concat_labels[mask].mean()
                if dt_mean_label > 0:
                    green_sharpes.append(date_sharpes[i])
                else:
                    red_sharpes.append(date_sharpes[i])

            if green_sharpes and red_sharpes:
                green_mean = np.mean(green_sharpes)
                red_mean = np.mean(red_sharpes)
                regime_gap = abs(green_mean - red_mean) / max(abs(green_mean), abs(red_mean), 1e-8)
                print(f"\n  Regime analysis:")
                print(f"    Green days mean Sharpe: {green_mean:.3f} ({len(green_sharpes)} days)")
                print(f"    Red days mean Sharpe:   {red_mean:.3f} ({len(red_sharpes)} days)")
                print(f"    Regime gap ratio:       {regime_gap:.3f} (REJECT if > 0.50)")

                mlflow.log_metrics({
                    "regime_green_sharpe": green_mean,
                    "regime_red_sharpe": red_mean,
                    "regime_gap_ratio": regime_gap,
                })

            # Save concat OOT predictions
            np.savez_compressed(
                os.path.join(OUTPUT_DIR, "concat_oot_predictions.npz"),
                predictions=concat_preds,
                labels=concat_labels,
                dates=concat_dates,
            )
            print(f"\n  Saved concat OOT predictions to {OUTPUT_DIR}/concat_oot_predictions.npz")

            # Save fold metrics
            with open(os.path.join(OUTPUT_DIR, "fold_metrics.json"), "w") as f:
                json.dump(fold_metrics, f, indent=2)

            # Log artifacts
            mlflow.log_artifact(os.path.join(OUTPUT_DIR, "concat_oot_predictions.npz"))
            mlflow.log_artifact(os.path.join(OUTPUT_DIR, "fold_metrics.json"))

        print(f"\n{'=' * 80}")
        print("DONE. All results saved and logged to MLflow.")


if __name__ == "__main__":
    print(f"Prediction Stream Momentum v1")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config: window={WINDOW_SIZE}, features={N_FEATURES}, hidden={HIDDEN_DIMS}")
    print(f"Walk-forward: {TRAIN_DAYS}d train / {EVAL_DAYS}d eval, sliding")
    print(f"Device: {DEVICE}")
    print()

    run_walk_forward()
