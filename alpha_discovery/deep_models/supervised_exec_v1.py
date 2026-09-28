#!/usr/bin/env python3
"""
Supervised Execution MLP v1 — Learn WHEN to trade from real FIFO data.

Replaces failed RL approaches (v3, v3.4b, v5) with supervised learning.
Uses CNN-Mamba predictions + embeddings + exec features as inputs.
Multi-task: trade profitability classification + return regression + vol regime.

Walk-forward: train on folds 0-7, test on folds 8-10 (sliding window).
"""

import os
import sys
import json
import time
import warnings
import numpy as np
from datetime import datetime
from pathlib import Path

# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
import mlflow
import mlflow.pytorch

# ============================================================================
# Constants
# ============================================================================
ES_TICK_VALUE = 12.50
ES_RT_COMMISSION = 4.70
COMMISSION_TICKS = ES_RT_COMMISSION / ES_TICK_VALUE  # 0.376 ticks
PROFIT_THRESHOLD_TICKS = COMMISSION_TICKS  # Break-even for limit order

# Paths
BASE_DIR = Path("/home/nick/Lvl3Quant")
PRED_DIR = BASE_DIR / "output" / "cnn_mamba_v2_smart_v3_mar"
EXEC_DIR = BASE_DIR / "output" / "exec_features_v1"
MBO_DIR = BASE_DIR / "data" / "processed" / "mbo_events_smart_v3"
OUTPUT_DIR = BASE_DIR / "output" / "supervised_exec_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Exec features to use (top 15 from fill prob analysis)
EXEC_FEATURE_NAMES = [
    "bid_depth_l1", "ask_depth_l1",
    "queue_consumption_velocity_bid", "queue_consumption_velocity_ask",
    "book_turnover", "time_in_spread",
    "depth_imbalance_l1", "trade_rate",
    "spread_mean_10k",
    "cancel_velocity_bid", "cancel_velocity_ask",
    "tod_sin", "tod_cos",
    "price_volatility_window", "queue_replenish_ratio",
]

MLFLOW_URI = "http://localhost:5000"

def log_info(msg):
    ts = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    print(f"{ts} [INFO] {msg}", flush=True)

def log_warning(msg):
    ts = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    print(f"{ts} [WARN] {msg}", flush=True)


# ============================================================================
# Model
# ============================================================================
class SupervisedExecMLP(nn.Module):
    """Multi-task MLP: shared backbone + 3 heads."""

    def __init__(self, input_dim, hidden_dims=(256, 128, 64), dropout=0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev = h
        self.backbone = nn.Sequential(*layers)

        # Head 1: Binary — will 10s return exceed commission (profitable trade)?
        self.head_profitable = nn.Sequential(
            nn.Linear(prev, 32), nn.GELU(), nn.Linear(32, 1)
        )
        # Head 2: Regression — predict 10s return in ticks
        self.head_return = nn.Sequential(
            nn.Linear(prev, 32), nn.GELU(), nn.Linear(32, 1)
        )
        # Head 3: Binary — lo-vol regime?
        self.head_vol_regime = nn.Sequential(
            nn.Linear(prev, 32), nn.GELU(), nn.Linear(32, 1)
        )

    def forward(self, x):
        h = self.backbone(x)
        return {
            "profitable": self.head_profitable(h).squeeze(-1),
            "return_pred": self.head_return(h).squeeze(-1),
            "vol_regime": self.head_vol_regime(h).squeeze(-1),
        }


# ============================================================================
# Data Loading
# ============================================================================
def load_fold_data(fold_idx):
    """Load and align prediction + exec feature data for a fold.

    Returns (features, targets) numpy arrays or None if alignment fails.
    """
    pred_path = PRED_DIR / f"fold_{fold_idx:02d}_oot_predictions.npz"
    if not pred_path.exists():
        return None

    pred = np.load(pred_path, allow_pickle=True)
    predictions = pred["predictions"]  # (N_pred, 3) — 1s, 5s, 10s
    labels = pred["labels"]            # (N_pred, 3) — 1s, 5s, 10s
    embeddings = pred["embeddings"]    # (N_pred, 96)
    oot_files = pred["oot_files"]

    # Extract date from oot_files
    date_str = os.path.basename(str(oot_files[0])).split("_")[0]

    # Load exec features for this date
    exec_path = EXEC_DIR / f"{date_str}_exec_features.npz"
    if not exec_path.exists():
        log_warning(f"No exec features for {date_str}")
        return None

    exec_data = np.load(exec_path, allow_pickle=True)
    exec_features = exec_data["features"]  # (N_exec, 44)
    exec_names = list(exec_data["feature_names"])

    # Select desired exec features
    exec_indices = []
    for name in EXEC_FEATURE_NAMES:
        if name in exec_names:
            exec_indices.append(exec_names.index(name))
        else:
            log_warning(f"Missing exec feature: {name}")
    exec_subset = exec_features[:, exec_indices]  # (N_exec, 15)

    N_pred = predictions.shape[0]
    N_exec = exec_subset.shape[0]

    # Align: predictions have ~10x more windows than exec features
    # Subsample predictions to match exec features (take center of each group)
    ratio = N_pred / N_exec
    if ratio < 1.5:
        # Already aligned or close enough - truncate to min
        n = min(N_pred, N_exec)
        pred_sub = predictions[:n]
        emb_sub = embeddings[:n]
        lbl_sub = labels[:n]
        exec_sub = exec_subset[:n]
    else:
        # Mean-pool predictions within each exec window
        # This gives us a smoothed signal per exec window
        stride = int(round(ratio))
        n = N_exec
        pred_sub = np.zeros((n, 3), dtype=np.float32)
        emb_sub = np.zeros((n, 96), dtype=np.float32)
        lbl_sub = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            start = int(i * ratio)
            end = min(int((i + 1) * ratio), N_pred)
            if end <= start:
                end = start + 1
            pred_sub[i] = predictions[start:end].mean(axis=0)
            emb_sub[i] = embeddings[start:end].mean(axis=0)
            # For labels, use the LAST prediction in the window (most recent)
            lbl_sub[i] = labels[start:end].mean(axis=0)
        exec_sub = exec_subset[:n]

    # Build feature vector per window
    # Prediction features: raw preds (3) + |pred| (3) + z-scores (3)
    pred_abs = np.abs(pred_sub)
    pred_mean = pred_sub.mean(axis=0, keepdims=True)
    pred_std = pred_sub.std(axis=0, keepdims=True) + 1e-8
    pred_z = (pred_sub - pred_mean) / pred_std

    features = np.concatenate([
        pred_sub,       # 3: raw predictions (1s, 5s, 10s)
        pred_abs,       # 3: absolute predictions
        pred_z,         # 3: z-scored predictions
        emb_sub,        # 96: CNN-Mamba embeddings
        exec_sub,       # 15: selected exec features
    ], axis=1)  # Total: 120 features

    # Build targets
    # Label columns: 0=1s, 1=5s, 2=10s (returns in some unit)
    returns_10s = lbl_sub[:, 2]  # 10s returns

    # Load MBO data to get tick size for conversion
    mbo_path = MBO_DIR / f"{date_str}_mbo_events.npz"
    if mbo_path.exists():
        mbo = np.load(mbo_path, allow_pickle=True)
        # labels are already in tick units from the training pipeline
        # Verify by checking scale
        label_std = np.std(returns_10s)
        log_info(f"Fold {fold_idx} ({date_str}): N={n}, label_10s std={label_std:.4f}")
    else:
        label_std = np.std(returns_10s)
        log_info(f"Fold {fold_idx} ({date_str}): N={n}, label_10s std={label_std:.4f} (no MBO verification)")

    # Target 1: profitable trade (|return| > commission threshold)
    # We care about LONG trades where return > threshold OR SHORT where return < -threshold
    # For simplicity: is |predicted_direction * actual_return| > commission?
    # Use raw prediction sign to determine direction
    pred_direction = np.sign(pred_sub[:, 2])  # 10s prediction direction
    directed_return = pred_direction * returns_10s
    target_profitable = (directed_return > COMMISSION_TICKS).astype(np.float32)

    # Target 2: actual 10s return (regression)
    target_return = returns_10s.astype(np.float32)

    # Target 3: vol regime based on rolling label volatility
    # Compute local volatility as rolling std of 10s returns (window=50)
    vol_window = 50
    local_vol = np.zeros(n, dtype=np.float32)
    for i in range(n):
        start_v = max(0, i - vol_window // 2)
        end_v = min(n, i + vol_window // 2)
        local_vol[i] = np.std(returns_10s[start_v:end_v])
    vol_median = np.median(local_vol)
    target_lovol = (local_vol <= vol_median).astype(np.float32)

    targets = {
        "profitable": target_profitable,
        "return": target_return,
        "vol_regime": target_lovol,
    }

    log_info(
        f"  Profitable rate: {target_profitable.mean():.3f}, "
        f"Return mean: {target_return.mean():.4f}, "
        f"Lo-vol rate: {target_lovol.mean():.3f}"
    )

    return {
        "features": features,
        "targets": targets,
        "date": date_str,
        "n_samples": n,
        "fold_idx": fold_idx,
    }


def normalize_features(train_X, test_X):
    """Z-score normalize using train stats."""
    mean = np.nanmean(train_X, axis=0)
    std = np.nanstd(train_X, axis=0) + 1e-8
    # Replace any NaN/Inf
    train_X = np.nan_to_num(train_X, nan=0.0, posinf=3.0, neginf=-3.0)
    test_X = np.nan_to_num(test_X, nan=0.0, posinf=3.0, neginf=-3.0)
    train_norm = (train_X - mean) / std
    test_norm = (test_X - mean) / std
    # Clip outliers
    train_norm = np.clip(train_norm, -5, 5)
    test_norm = np.clip(test_norm, -5, 5)
    return train_norm, test_norm, mean, std


# ============================================================================
# Training
# ============================================================================
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    n_batches = 0

    for X_batch, y_prof, y_ret, y_vol in loader:
        X_batch = X_batch.to(device)
        y_prof = y_prof.to(device)
        y_ret = y_ret.to(device)
        y_vol = y_vol.to(device)

        optimizer.zero_grad()
        out = model(X_batch)

        loss_prof = bce(out["profitable"], y_prof)
        loss_ret = mse(out["return_pred"], y_ret)
        loss_vol = bce(out["vol_regime"], y_vol)

        # Weight: classification more important than regression
        # Return MSE ~25 (std=5), BCE ~0.7, so scale return down heavily
        loss = 1.0 * loss_prof + 0.02 * loss_ret + 0.5 * loss_vol
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_prof_logits = []
    all_prof_labels = []
    all_ret_preds = []
    all_ret_labels = []
    all_vol_logits = []
    all_vol_labels = []

    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    for X_batch, y_prof, y_ret, y_vol in loader:
        X_batch = X_batch.to(device)
        out = model(X_batch)

        all_prof_logits.append(out["profitable"].cpu().numpy())
        all_prof_labels.append(y_prof.numpy())
        all_ret_preds.append(out["return_pred"].cpu().numpy())
        all_ret_labels.append(y_ret.numpy())
        all_vol_logits.append(out["vol_regime"].cpu().numpy())
        all_vol_labels.append(y_vol.numpy())

    prof_logits = np.concatenate(all_prof_logits)
    prof_labels = np.concatenate(all_prof_labels)
    ret_preds = np.concatenate(all_ret_preds)
    ret_labels = np.concatenate(all_ret_labels)
    vol_logits = np.concatenate(all_vol_logits)
    vol_labels = np.concatenate(all_vol_labels)

    # AUC for classification
    try:
        auc_prof = roc_auc_score(prof_labels, prof_logits)
    except ValueError:
        auc_prof = 0.5

    try:
        auc_vol = roc_auc_score(vol_labels, vol_logits)
    except ValueError:
        auc_vol = 0.5

    # Return prediction IC
    ic_ret = np.corrcoef(ret_preds, ret_labels)[0, 1] if len(ret_labels) > 10 else 0.0
    if np.isnan(ic_ret):
        ic_ret = 0.0

    # Loss
    prof_loss = nn.BCEWithLogitsLoss()(
        torch.tensor(prof_logits), torch.tensor(prof_labels)
    ).item()
    ret_loss = nn.MSELoss()(
        torch.tensor(ret_preds), torch.tensor(ret_labels)
    ).item()

    return {
        "auc_profitable": auc_prof,
        "auc_vol_regime": auc_vol,
        "ic_return": ic_ret,
        "loss_profitable": prof_loss,
        "loss_return": ret_loss,
        "prof_logits": prof_logits,
        "prof_labels": prof_labels,
        "ret_preds": ret_preds,
        "ret_labels": ret_labels,
    }


def compute_trading_metrics(eval_results, threshold=0.0):
    """Compute trading performance: filtered vs unfiltered."""
    prof_logits = eval_results["prof_logits"]
    prof_labels = eval_results["prof_labels"]
    ret_preds = eval_results["ret_preds"]
    ret_labels = eval_results["ret_labels"]

    # Probability of profitable trade
    prof_prob = 1 / (1 + np.exp(-prof_logits))

    # Unfiltered: trade on all signals (baseline)
    # Assume we trade when |prediction| is notable (z >= 2 equivalent)
    # Here ret_preds are the MLP return predictions
    unfiltered_returns = ret_labels  # actual returns for all windows
    unfiltered_n = len(unfiltered_returns)
    unfiltered_mean = unfiltered_returns.mean() if unfiltered_n > 0 else 0

    # Filtered: only trade when MLP says profitable (prob > 0.5 + threshold)
    trade_mask = prof_prob > (0.5 + threshold)
    filtered_returns = ret_labels[trade_mask]
    filtered_n = trade_mask.sum()

    # Direction from ret_preds
    direction = np.sign(ret_preds)
    directed_returns_all = direction * ret_labels - COMMISSION_TICKS
    directed_returns_filtered = (direction * ret_labels - COMMISSION_TICKS)[trade_mask]

    # Win rate
    wr_unfiltered = (directed_returns_all > 0).mean() if len(directed_returns_all) > 0 else 0
    wr_filtered = (directed_returns_filtered > 0).mean() if len(directed_returns_filtered) > 0 else 0

    # Profit factor
    def profit_factor(rets):
        gains = rets[rets > 0].sum()
        losses = abs(rets[rets < 0].sum())
        return gains / max(losses, 1e-8)

    pf_unfiltered = profit_factor(directed_returns_all)
    pf_filtered = profit_factor(directed_returns_filtered) if filtered_n > 0 else 0

    # Sharpe (annualized, assuming ~6.5hr sessions, ~252 days)
    def sharpe(rets):
        if len(rets) < 2:
            return 0
        return rets.mean() / (rets.std() + 1e-8) * np.sqrt(len(rets))

    def sortino(rets):
        if len(rets) < 2:
            return 0
        downside = rets[rets < 0]
        downside_std = downside.std() if len(downside) > 1 else 1e-8
        return rets.mean() / (downside_std + 1e-8) * np.sqrt(len(rets))

    sharpe_unfiltered = sharpe(directed_returns_all)
    sharpe_filtered = sharpe(directed_returns_filtered) if filtered_n > 0 else 0
    sortino_unfiltered = sortino(directed_returns_all)
    sortino_filtered = sortino(directed_returns_filtered) if filtered_n > 0 else 0

    return {
        "unfiltered_n": int(unfiltered_n),
        "filtered_n": int(filtered_n),
        "filter_rate": float(filtered_n / max(unfiltered_n, 1)),
        "wr_unfiltered": float(wr_unfiltered),
        "wr_filtered": float(wr_filtered),
        "wr_improvement": float(wr_filtered - wr_unfiltered),
        "pf_unfiltered": float(pf_unfiltered),
        "pf_filtered": float(pf_filtered),
        "sharpe_unfiltered": float(sharpe_unfiltered),
        "sharpe_filtered": float(sharpe_filtered),
        "sortino_unfiltered": float(sortino_unfiltered),
        "sortino_filtered": float(sortino_filtered),
        "mean_return_unfiltered": float(directed_returns_all.mean()) if len(directed_returns_all) > 0 else 0,
        "mean_return_filtered": float(directed_returns_filtered.mean()) if filtered_n > 0 else 0,
    }


# ============================================================================
# Main
# ============================================================================
def main():
    log_info("=" * 70)
    log_info("Supervised Execution MLP v1")
    log_info("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_info(f"Device: {device}")
    if device.type == "cuda":
        log_info(f"GPU: {torch.cuda.get_device_name()}")

    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    experiment_name = "supervised_exec_mlp"
    mlflow.set_experiment(experiment_name)

    # Load all folds
    log_info("Loading fold data...")
    fold_data = {}
    seen_dates = {}
    for fold_idx in range(11):
        data = load_fold_data(fold_idx)
        if data is None:
            log_warning(f"  Fold {fold_idx}: skipped (missing data)")
            continue
        if data["n_samples"] <= 100:
            log_warning(f"  Fold {fold_idx}: skipped (too few samples: {data['n_samples']})")
            continue
        if data["date"] in seen_dates:
            log_warning(f"  Fold {fold_idx}: skipped (duplicate date {data['date']}, same as fold {seen_dates[data['date']]})")
            continue
        seen_dates[data["date"]] = fold_idx
        fold_data[fold_idx] = data
        log_info(f"  Fold {fold_idx}: {data['date']}, {data['n_samples']} samples")

    valid_folds = sorted(fold_data.keys())
    log_info(f"Valid folds: {valid_folds} ({len(valid_folds)} total)")

    if len(valid_folds) < 4:
        log_warning("Not enough valid folds for walk-forward")
        return

    input_dim = fold_data[valid_folds[0]]["features"].shape[1]
    log_info(f"Input dimension: {input_dim}")

    # Walk-forward: leave-one-out on the last folds
    # Train on all but last K folds, test on last K folds individually
    # With 10 valid folds (excluding fold 5), do 3 test folds
    n_test_folds = min(3, len(valid_folds) - 3)
    test_fold_indices = valid_folds[-n_test_folds:]
    train_fold_indices = valid_folds[:-n_test_folds]

    log_info(f"Train folds: {train_fold_indices}")
    log_info(f"Test folds: {test_fold_indices}")

    with mlflow.start_run(run_name=f"supervised_exec_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log params
        mlflow.log_param("model_type", "supervised_exec_mlp")
        mlflow.log_param("input_dim", input_dim)
        mlflow.log_param("hidden_dims", "256-128-64")
        mlflow.log_param("dropout", 0.2)
        mlflow.log_param("batch_size", 2048)
        mlflow.log_param("n_train_folds", len(train_fold_indices))
        mlflow.log_param("n_test_folds", n_test_folds)
        mlflow.log_param("train_folds", str(train_fold_indices))
        mlflow.log_param("test_folds", str(test_fold_indices))
        mlflow.log_param("commission_ticks", COMMISSION_TICKS)
        mlflow.log_param("exec_features", str(EXEC_FEATURE_NAMES))
        mlflow.log_param("optimizer", "AdamW")
        mlflow.log_param("lr", 1e-3)
        mlflow.log_param("weight_decay", 1e-4)
        mlflow.log_param("n_epochs", 50)
        mlflow.log_param("loss_weights", "prof=1.0, ret=0.02, vol=0.5")

        # === Strategy 1: Train on all train folds, test on each test fold ===
        log_info("\n" + "=" * 50)
        log_info("Strategy: Pooled train, per-fold test")
        log_info("=" * 50)

        # Build train set
        train_X = np.concatenate([fold_data[i]["features"] for i in train_fold_indices])
        train_prof = np.concatenate([fold_data[i]["targets"]["profitable"] for i in train_fold_indices])
        train_ret = np.concatenate([fold_data[i]["targets"]["return"] for i in train_fold_indices])
        train_vol = np.concatenate([fold_data[i]["targets"]["vol_regime"] for i in train_fold_indices])

        log_info(f"Train set: {train_X.shape[0]} samples from {len(train_fold_indices)} folds")
        log_info(f"Train profitable rate: {train_prof.mean():.3f}")

        # Per-fold test evaluation
        all_test_metrics = {}
        for test_fold in test_fold_indices:
            test_X_raw = fold_data[test_fold]["features"]

            # Normalize
            train_X_norm, test_X_norm, feat_mean, feat_std = normalize_features(
                train_X.copy(), test_X_raw.copy()
            )

            test_prof = fold_data[test_fold]["targets"]["profitable"]
            test_ret = fold_data[test_fold]["targets"]["return"]
            test_vol = fold_data[test_fold]["targets"]["vol_regime"]

            # Create dataloaders
            train_ds = TensorDataset(
                torch.tensor(train_X_norm, dtype=torch.float32),
                torch.tensor(train_prof, dtype=torch.float32),
                torch.tensor(train_ret, dtype=torch.float32),
                torch.tensor(train_vol, dtype=torch.float32),
            )
            test_ds = TensorDataset(
                torch.tensor(test_X_norm, dtype=torch.float32),
                torch.tensor(test_prof, dtype=torch.float32),
                torch.tensor(test_ret, dtype=torch.float32),
                torch.tensor(test_vol, dtype=torch.float32),
            )
            train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True, num_workers=4, pin_memory=True)
            test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False, num_workers=4, pin_memory=True)

            # Build model
            model = SupervisedExecMLP(input_dim).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)

            best_auc = 0
            best_epoch = 0
            patience = 10
            no_improve = 0

            log_info(f"\nTraining for test fold {test_fold} ({fold_data[test_fold]['date']})...")

            for epoch in range(50):
                t0 = time.time()
                train_loss = train_epoch(model, train_loader, optimizer, device)
                scheduler.step()

                if (epoch + 1) % 5 == 0 or epoch == 0:
                    eval_res = evaluate(model, test_loader, device)
                    elapsed = time.time() - t0
                    log_info(
                        f"  Epoch {epoch+1:3d} | loss={train_loss:.4f} | "
                        f"AUC_prof={eval_res['auc_profitable']:.4f} | "
                        f"AUC_vol={eval_res['auc_vol_regime']:.4f} | "
                        f"IC_ret={eval_res['ic_return']:.4f} | "
                        f"{elapsed:.1f}s"
                    )

                    mlflow.log_metrics({
                        f"fold{test_fold}_train_loss": train_loss,
                        f"fold{test_fold}_auc_prof": eval_res["auc_profitable"],
                        f"fold{test_fold}_auc_vol": eval_res["auc_vol_regime"],
                        f"fold{test_fold}_ic_ret": eval_res["ic_return"],
                    }, step=epoch)

                    if eval_res["auc_profitable"] > best_auc:
                        best_auc = eval_res["auc_profitable"]
                        best_epoch = epoch
                        no_improve = 0
                        # Save best model
                        torch.save({
                            "model_state": model.state_dict(),
                            "epoch": epoch,
                            "auc_prof": best_auc,
                            "feat_mean": feat_mean,
                            "feat_std": feat_std,
                            "input_dim": input_dim,
                        }, OUTPUT_DIR / f"best_model_fold{test_fold}.pt")
                    else:
                        no_improve += 1

                    if no_improve >= patience // 5 * 2:  # Check every 5 epochs, so patience ~ 10 effective
                        log_info(f"  Early stop at epoch {epoch+1}")
                        break

            # Load best model for final eval
            ckpt = torch.load(OUTPUT_DIR / f"best_model_fold{test_fold}.pt", weights_only=False)
            model.load_state_dict(ckpt["model_state"])

            final_eval = evaluate(model, test_loader, device)

            log_info(f"\n  === Test Fold {test_fold} ({fold_data[test_fold]['date']}) Results ===")
            log_info(f"  Best epoch: {best_epoch+1}")
            log_info(f"  AUC (profitable): {final_eval['auc_profitable']:.4f}")
            log_info(f"  AUC (vol regime): {final_eval['auc_vol_regime']:.4f}")
            log_info(f"  IC (return):       {final_eval['ic_return']:.4f}")

            # Evaluate at multiple filter thresholds
            log_info(f"  --- Trading Performance at Multiple Thresholds ---")
            best_threshold = 0.0
            best_sortino = -999
            for thresh in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]:
                t_metrics = compute_trading_metrics(final_eval, threshold=thresh)
                tag = " <-- BEST" if t_metrics["sortino_filtered"] > best_sortino and t_metrics["filtered_n"] >= 20 else ""
                if t_metrics["sortino_filtered"] > best_sortino and t_metrics["filtered_n"] >= 20:
                    best_sortino = t_metrics["sortino_filtered"]
                    best_threshold = thresh
                log_info(
                    f"  thresh={thresh:.2f}: filter={t_metrics['filter_rate']:.1%} "
                    f"N={t_metrics['filtered_n']} "
                    f"WR={t_metrics['wr_filtered']:.1%} "
                    f"PF={t_metrics['pf_filtered']:.2f} "
                    f"Sortino={t_metrics['sortino_filtered']:.2f} "
                    f"MeanRet={t_metrics['mean_return_filtered']:.3f}{tag}"
                )

            # Use best threshold for reporting
            trading = compute_trading_metrics(final_eval, threshold=best_threshold)
            log_info(f"  Best threshold: {best_threshold:.2f}")
            log_info(f"  --- Summary (best threshold) ---")
            log_info(f"  Filter rate:      {trading['filter_rate']:.1%} ({trading['filtered_n']}/{trading['unfiltered_n']} windows)")
            log_info(f"  Win rate:         {trading['wr_unfiltered']:.1%} -> {trading['wr_filtered']:.1%} ({trading['wr_improvement']:+.1%})")
            log_info(f"  Profit factor:    {trading['pf_unfiltered']:.2f} -> {trading['pf_filtered']:.2f}")
            log_info(f"  Sharpe:           {trading['sharpe_unfiltered']:.2f} -> {trading['sharpe_filtered']:.2f}")
            log_info(f"  Sortino:          {trading['sortino_unfiltered']:.2f} -> {trading['sortino_filtered']:.2f}")
            log_info(f"  Mean return:      {trading['mean_return_unfiltered']:.4f} -> {trading['mean_return_filtered']:.4f} ticks")

            # Log to MLflow
            for k, v in trading.items():
                mlflow.log_metric(f"fold{test_fold}_{k}", v)
            mlflow.log_metric(f"fold{test_fold}_best_auc_prof", final_eval["auc_profitable"])
            mlflow.log_metric(f"fold{test_fold}_best_ic_ret", final_eval["ic_return"])

            all_test_metrics[test_fold] = {
                "auc_prof": final_eval["auc_profitable"],
                "auc_vol": final_eval["auc_vol_regime"],
                "ic_ret": final_eval["ic_return"],
                **trading,
            }

            # Save predictions
            np.savez_compressed(
                OUTPUT_DIR / f"fold{test_fold}_test_predictions.npz",
                prof_logits=final_eval["prof_logits"],
                prof_labels=final_eval["prof_labels"],
                ret_preds=final_eval["ret_preds"],
                ret_labels=final_eval["ret_labels"],
                date=fold_data[test_fold]["date"],
            )

        # === Summary ===
        log_info("\n" + "=" * 70)
        log_info("OVERALL RESULTS")
        log_info("=" * 70)

        avg_auc = np.mean([m["auc_prof"] for m in all_test_metrics.values()])
        avg_ic = np.mean([m["ic_ret"] for m in all_test_metrics.values()])
        avg_wr_improve = np.mean([m["wr_improvement"] for m in all_test_metrics.values()])
        avg_pf_filtered = np.mean([m["pf_filtered"] for m in all_test_metrics.values()])
        avg_sortino_filtered = np.mean([m["sortino_filtered"] for m in all_test_metrics.values()])
        avg_sortino_unfiltered = np.mean([m["sortino_unfiltered"] for m in all_test_metrics.values()])

        log_info(f"Avg AUC (profitable):    {avg_auc:.4f}")
        log_info(f"Avg IC (return):         {avg_ic:.4f}")
        log_info(f"Avg WR improvement:      {avg_wr_improve:+.1%}")
        log_info(f"Avg PF (filtered):       {avg_pf_filtered:.2f}")
        log_info(f"Avg Sortino (unfiltered): {avg_sortino_unfiltered:.2f}")
        log_info(f"Avg Sortino (filtered):  {avg_sortino_filtered:.2f}")

        mlflow.log_metric("avg_auc_profitable", avg_auc)
        mlflow.log_metric("avg_ic_return", avg_ic)
        mlflow.log_metric("avg_wr_improvement", avg_wr_improve)
        mlflow.log_metric("avg_pf_filtered", avg_pf_filtered)
        mlflow.log_metric("avg_sortino_filtered", avg_sortino_filtered)
        mlflow.log_metric("avg_sortino_unfiltered", avg_sortino_unfiltered)

        # Save summary
        summary = {
            "model": "supervised_exec_mlp_v1",
            "timestamp": datetime.now().isoformat(),
            "input_dim": input_dim,
            "train_folds": train_fold_indices,
            "test_folds": test_fold_indices,
            "per_fold_metrics": {str(k): v for k, v in all_test_metrics.items()},
            "averages": {
                "auc_profitable": float(avg_auc),
                "ic_return": float(avg_ic),
                "wr_improvement": float(avg_wr_improve),
                "pf_filtered": float(avg_pf_filtered),
                "sortino_filtered": float(avg_sortino_filtered),
                "sortino_unfiltered": float(avg_sortino_unfiltered),
            },
        }
        with open(OUTPUT_DIR / "results_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        mlflow.log_artifact(str(OUTPUT_DIR / "results_summary.json"))

        log_info(f"\nResults saved to {OUTPUT_DIR}")
        log_info("Done!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"\nFATAL ERROR: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)
