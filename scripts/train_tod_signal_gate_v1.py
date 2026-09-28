#!/usr/bin/env python3
"""
Time-of-Day Signal Gate v1 — MLP classifier that learns when short signals are profitable.

CONTEXT: CNN-Mamba short signal IC varies ~2x by time of day (10:30 AM IC=0.294 vs noon
IC=0.167). This model learns which time windows amplify signal quality so we can
concentrate trades in high-edge windows.

Architecture: MLP classifier
  Input: 25 smart_v3 microstructure features + 4 time features = 29
  Output: binary — 1 if short trade profitable (labels_1s < 0), 0 otherwise
  Hidden: 64 → 32 → 1, ReLU, BatchNorm, Dropout(0.15)
  Loss: BCE with auto-balanced class weights

Walk-forward: sliding 10-date train, 1-date OOT
Signal filter: only events where CNN-Mamba prediction[:,0] in bottom 5th percentile (shorts)
"""

import os
import sys
import json
import gc
import warnings
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings("ignore")

# ── Paths ──
if sys.platform == "win32":
    DATA_ROOT = Path(r"C:\Users\claude\Lvl3Quant")
else:
    DATA_ROOT = Path(os.environ.get("LVL3_ROOT", "/home/jupiter/Lvl3Quant"))

FEATURES_DIR = DATA_ROOT / "data" / "processed" / "mbo_events_smart_v3"
PREDS_DIR = DATA_ROOT / "output" / "cnn_mamba_v2_bulk_oot_v2"
OUT_DIR = DATA_ROOT / "output" / "tod_signal_gate_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ──
TRAIN_WINDOW = 10       # sliding window dates
HIDDEN_DIMS = [64, 32]
LR = 1e-3
EPOCHS = 30
BATCH_SIZE = 4096
WEIGHT_DECAY = 1e-4
DROPOUT = 0.15
PATIENCE = 5            # early stopping
SHORT_PERCENTILE = 5    # bottom 5% of predictions = short signals
ET_OFFSET_HOURS = 4     # EDT = UTC - 4 (summer 2026)

# ── MLflow ──
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "tod_signal_gate_v1"

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

# ── Model ──
class TODSignalGateMLP(nn.Module):
    def __init__(self, in_dim=29, hidden_dims=[64, 32], dropout=0.15):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def discover_dates():
    """Find dates that have both features and predictions files."""
    feat_dates = set()
    for f in FEATURES_DIR.glob("*_mbo_events.npz"):
        date_str = f.stem.split("_")[0]
        if len(date_str) == 8 and date_str.isdigit():
            feat_dates.add(date_str)

    pred_dates = set()
    for f in PREDS_DIR.glob("*_predictions.npz"):
        date_str = f.stem.split("_")[0]
        if len(date_str) == 8 and date_str.isdigit():
            pred_dates.add(date_str)

    common = sorted(feat_dates & pred_dates)
    print(f"[DATES] Features: {len(feat_dates)}, Predictions: {len(pred_dates)}, Common: {len(common)}")
    return common


def make_time_features(timestamps):
    """Construct 4 time features from Unix timestamps.

    Returns: [sin_hour, cos_hour, minutes_since_open, session_half]
    """
    # Handle nanosecond timestamps
    if len(timestamps) > 0 and timestamps[0] > 1e12:
        timestamps = timestamps / 1e9

    # UTC hour fractional
    hour_utc = (timestamps % 86400) / 3600.0
    # Convert to ET (EDT = UTC - 4)
    hour_et = hour_utc - ET_OFFSET_HOURS
    # Wrap negatives
    hour_et = np.where(hour_et < 0, hour_et + 24, hour_et)

    # hour_frac for cyclical encoding (use ET hour)
    hour_frac = hour_et / 24.0
    sin_hour = np.sin(hour_frac * 2 * np.pi)
    cos_hour = np.cos(hour_frac * 2 * np.pi)

    # Minutes since 9:30 AM ET
    minutes_since_open = np.clip((hour_et - 9.5) * 60.0, 0, 390)

    # Session half: 0 = AM (before noon ET), 1 = PM
    session_half = (hour_et >= 12.0).astype(np.float32)

    return np.column_stack([
        sin_hour.astype(np.float32),
        cos_hour.astype(np.float32),
        minutes_since_open.astype(np.float32),
        session_half,
    ])


def load_date(date_str):
    """Load features, timestamps, labels, and CNN-Mamba predictions for a date.

    Returns aligned (features_29d, labels_1s, hour_et) for short-signal-filtered events,
    or None if loading fails.
    """
    feat_path = FEATURES_DIR / f"{date_str}_mbo_events.npz"
    pred_path = PREDS_DIR / f"{date_str}_predictions.npz"

    if not feat_path.exists() or not pred_path.exists():
        return None

    try:
        feat_data = np.load(feat_path)
        pred_data = np.load(pred_path)
    except Exception as e:
        print(f"  [WARN] Failed to load {date_str}: {e}")
        return None

    events = feat_data["events"]           # (N_events, 25)
    timestamps = feat_data["timestamps"]   # (N_events,)
    labels_1s = feat_data["labels_1s"]     # (N_events,)

    predictions = pred_data["predictions"]  # (N_pred, 3)
    stride = int(pred_data["stride"])
    window_size = int(pred_data["window_size"])

    # Align predictions to events
    # prediction[i] corresponds to events[i * stride + window_size - 1]
    n_pred = predictions.shape[0]
    event_indices = np.arange(n_pred) * stride + (window_size - 1)

    # Filter valid indices
    valid_mask = event_indices < len(events)
    event_indices = event_indices[valid_mask]
    preds_aligned = predictions[valid_mask]

    if len(event_indices) == 0:
        return None

    # Extract aligned data
    aligned_events = events[event_indices]       # (M, 25)
    aligned_ts = timestamps[event_indices]       # (M,)
    aligned_labels = labels_1s[event_indices]    # (M,)

    # Filter to short signals: bottom 5th percentile of prediction[:,0]
    short_threshold = np.percentile(preds_aligned[:, 0], SHORT_PERCENTILE)
    short_mask = preds_aligned[:, 0] <= short_threshold

    if short_mask.sum() < 10:
        return None

    # Apply filter
    filt_events = aligned_events[short_mask]      # (K, 25)
    filt_ts = aligned_ts[short_mask]              # (K,)
    filt_labels = aligned_labels[short_mask]      # (K,)

    # Build time features
    time_feats = make_time_features(filt_ts)      # (K, 4)

    # Combine: 25 micro features + 4 time features = 29
    features = np.hstack([filt_events.astype(np.float32), time_feats])  # (K, 29)

    # Target: 1 if labels_1s < 0 (price went down = profitable short)
    targets = (filt_labels < 0).astype(np.float32)

    # Compute hour_et for time-of-day analysis
    ts_sec = filt_ts.copy()
    if len(ts_sec) > 0 and ts_sec[0] > 1e12:
        ts_sec = ts_sec / 1e9
    hour_et = ((ts_sec % 86400) / 3600.0) - ET_OFFSET_HOURS
    hour_et = np.where(hour_et < 0, hour_et + 24, hour_et)

    return features, targets, hour_et


def compute_class_weights(targets):
    """Compute balanced class weights for BCE."""
    n_pos = targets.sum()
    n_neg = len(targets) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    w_pos = n_neg / n_pos
    return w_pos


def train_fold(model, train_features, train_targets, device, class_weight_pos):
    """Train model for one fold with early stopping."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    X = torch.tensor(train_features, dtype=torch.float32)
    y = torch.tensor(train_targets, dtype=torch.float32).unsqueeze(1)

    # Normalize features (fit on train)
    feat_mean = X.mean(dim=0)
    feat_std = X.std(dim=0).clamp(min=1e-8)
    X = (X - feat_mean) / feat_std

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=0, pin_memory=True)

    pos_weight = torch.tensor([class_weight_pos], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # Early stopping on training loss (no val split to keep it simple)
        if avg_loss < best_loss - 1e-5:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    return feat_mean, feat_std, best_loss


def evaluate_fold(model, test_features, test_targets, test_hours, feat_mean, feat_std, device):
    """Evaluate model on OOT fold. Returns metrics dict."""
    model.eval()
    X = torch.tensor(test_features, dtype=torch.float32)
    X = (X - feat_mean) / feat_std

    with torch.no_grad():
        logits = model(X.to(device))
        probs = torch.sigmoid(logits).cpu().numpy().flatten()

    targets = test_targets
    preds_binary = (probs >= 0.5).astype(int)

    # Overall metrics
    try:
        auc = roc_auc_score(targets, probs)
    except ValueError:
        auc = 0.5
    acc = accuracy_score(targets, preds_binary)
    loss_val = nn.BCELoss()(torch.tensor(probs), torch.tensor(targets)).item()

    # Per-window analysis (30-min bins from 9:30 to 16:00 ET)
    window_stats = {}
    for window_start in np.arange(9.5, 16.0, 0.5):
        window_end = window_start + 0.5
        mask = (test_hours >= window_start) & (test_hours < window_end)
        n_in_window = mask.sum()
        if n_in_window < 5:
            continue

        w_targets = targets[mask]
        w_probs = probs[mask]
        w_preds = preds_binary[mask]

        w_acc = accuracy_score(w_targets, w_preds)
        try:
            w_auc = roc_auc_score(w_targets, w_probs)
        except ValueError:
            w_auc = 0.5

        # Format window label
        h = int(window_start)
        m = int((window_start % 1) * 60)
        label = f"{h:02d}:{m:02d}"
        window_stats[label] = {
            "count": int(n_in_window),
            "accuracy": float(w_acc),
            "auc": float(w_auc),
            "pos_rate": float(w_targets.mean()),
        }

    return {
        "auc": float(auc),
        "accuracy": float(acc),
        "loss": float(loss_val),
        "n_samples": int(len(targets)),
        "pos_rate": float(targets.mean()),
        "window_stats": window_stats,
        "probs": probs,
        "targets": targets,
    }


def main():
    print("=" * 70)
    print("TOD Signal Gate v1 — Time-of-Day MLP Classifier")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    # MLflow setup
    mlflow_available = False
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        mlflow_available = True
        print(f"[MLFLOW] Connected to {MLFLOW_URI}, experiment={EXPERIMENT_NAME}")
    except Exception as e:
        print(f"[MLFLOW] Not available: {e}")

    # Discover dates
    dates = discover_dates()
    if len(dates) < TRAIN_WINDOW + 1:
        print(f"[ERROR] Need at least {TRAIN_WINDOW + 1} dates, found {len(dates)}")
        sys.exit(1)

    print(f"[DATES] {len(dates)} dates: {dates[0]} → {dates[-1]}")
    print(f"[WF] Sliding window: {TRAIN_WINDOW} train, 1 OOT")
    print(f"[FOLDS] {len(dates) - TRAIN_WINDOW} folds")
    print()

    # Preload all dates
    print("[LOAD] Preloading all dates...")
    date_cache = {}
    for d in dates:
        result = load_date(d)
        if result is not None:
            date_cache[d] = result
            n_samp = result[0].shape[0]
            pos_rate = result[1].mean()
            print(f"  {d}: {n_samp:>6} short signals, pos_rate={pos_rate:.3f}")
        else:
            print(f"  {d}: SKIP (no data or too few short signals)")

    available_dates = [d for d in dates if d in date_cache]
    print(f"\n[LOAD] {len(available_dates)} dates loaded successfully")

    if len(available_dates) < TRAIN_WINDOW + 1:
        print(f"[ERROR] Need at least {TRAIN_WINDOW + 1} available dates")
        sys.exit(1)

    # Walk-forward
    all_oot_probs = []
    all_oot_targets = []
    all_oot_hours = []
    fold_results = []

    if mlflow_available:
        mlflow.start_run(run_name=f"tod_gate_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        mlflow.log_params({
            "train_window": TRAIN_WINDOW,
            "hidden_dims": str(HIDDEN_DIMS),
            "lr": LR,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "dropout": DROPOUT,
            "patience": PATIENCE,
            "short_percentile": SHORT_PERCENTILE,
            "n_dates": len(available_dates),
            "n_folds": len(available_dates) - TRAIN_WINDOW,
        })

    for fold_idx in range(TRAIN_WINDOW, len(available_dates)):
        test_date = available_dates[fold_idx]
        train_dates = available_dates[fold_idx - TRAIN_WINDOW: fold_idx]

        # Gather training data
        train_feats_list = []
        train_targets_list = []
        for td in train_dates:
            feats, targets, _ = date_cache[td]
            train_feats_list.append(feats)
            train_targets_list.append(targets)

        train_features = np.concatenate(train_feats_list, axis=0)
        train_targets = np.concatenate(train_targets_list, axis=0)

        # Test data
        test_features, test_targets, test_hours = date_cache[test_date]

        # Class weights
        cw_pos = compute_class_weights(train_targets)
        if cw_pos is None:
            print(f"  Fold {fold_idx - TRAIN_WINDOW + 1}: {test_date} — SKIP (degenerate labels)")
            continue

        # Build model
        model = TODSignalGateMLP(
            in_dim=29, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT
        ).to(device)

        # Train
        feat_mean, feat_std, train_loss = train_fold(
            model, train_features, train_targets, device, cw_pos
        )

        # Evaluate
        metrics = evaluate_fold(
            model, test_features, test_targets, test_hours,
            feat_mean, feat_std, device
        )

        fold_num = fold_idx - TRAIN_WINDOW + 1
        print(f"  Fold {fold_num:>3}: {test_date} | "
              f"AUC={metrics['auc']:.4f} | Acc={metrics['accuracy']:.4f} | "
              f"Loss={metrics['loss']:.4f} | N={metrics['n_samples']} | "
              f"PosRate={metrics['pos_rate']:.3f}")

        # Log per-fold to MLflow
        if mlflow_available:
            mlflow.log_metrics({
                f"fold_auc": metrics["auc"],
                f"fold_accuracy": metrics["accuracy"],
                f"fold_loss": metrics["loss"],
                f"fold_n_samples": metrics["n_samples"],
            }, step=fold_num)

        # Accumulate OOT
        all_oot_probs.append(metrics["probs"])
        all_oot_targets.append(metrics["targets"])
        all_oot_hours.append(test_hours)

        fold_results.append({
            "fold": fold_num,
            "test_date": test_date,
            "auc": metrics["auc"],
            "accuracy": metrics["accuracy"],
            "loss": metrics["loss"],
            "n_samples": metrics["n_samples"],
            "pos_rate": metrics["pos_rate"],
            "window_stats": metrics["window_stats"],
        })

        # Save fold predictions
        fold_out = OUT_DIR / f"{test_date}_tod_gate_preds.npz"
        np.savez_compressed(fold_out,
                            probs=metrics["probs"],
                            targets=metrics["targets"],
                            hours=test_hours)

        # GPU cleanup
        del model
        torch.cuda.empty_cache()
        gc.collect()

    # ── Concat OOT Analysis ──
    print("\n" + "=" * 70)
    print("CONCAT OOT ANALYSIS")
    print("=" * 70)

    if len(all_oot_probs) == 0:
        print("[ERROR] No folds completed!")
        sys.exit(1)

    concat_probs = np.concatenate(all_oot_probs)
    concat_targets = np.concatenate(all_oot_targets)
    concat_hours = np.concatenate(all_oot_hours)

    concat_auc = roc_auc_score(concat_targets, concat_probs)
    concat_acc = accuracy_score(concat_targets, (concat_probs >= 0.5).astype(int))
    concat_pos_rate = concat_targets.mean()

    print(f"\n  Overall OOT AUC:      {concat_auc:.4f}")
    print(f"  Overall OOT Accuracy: {concat_acc:.4f}")
    print(f"  Total samples:        {len(concat_targets)}")
    print(f"  Positive rate:        {concat_pos_rate:.3f}")

    # Per-window analysis (30-min bins)
    print(f"\n  {'Window':<10} {'Count':>7} {'Accuracy':>10} {'AUC':>8} {'PosRate':>9} {'PredHigh':>10}")
    print("  " + "-" * 58)

    window_summary = {}
    for window_start in np.arange(9.5, 16.0, 0.5):
        window_end = window_start + 0.5
        mask = (concat_hours >= window_start) & (concat_hours < window_end)
        n = mask.sum()
        if n < 10:
            continue

        w_targets = concat_targets[mask]
        w_probs = concat_probs[mask]
        w_preds = (w_probs >= 0.5).astype(int)

        w_acc = accuracy_score(w_targets, w_preds)
        try:
            w_auc = roc_auc_score(w_targets, w_probs)
        except ValueError:
            w_auc = 0.5
        w_pos_rate = w_targets.mean()
        w_pred_high = (w_probs >= 0.5).mean()

        h = int(window_start)
        m = int((window_start % 1) * 60)
        label = f"{h:02d}:{m:02d}"

        print(f"  {label:<10} {n:>7} {w_acc:>10.4f} {w_auc:>8.4f} {w_pos_rate:>9.3f} {w_pred_high:>10.3f}")

        window_summary[label] = {
            "count": int(n),
            "accuracy": float(w_acc),
            "auc": float(w_auc),
            "pos_rate": float(w_pos_rate),
            "pred_high_rate": float(w_pred_high),
        }

    # Per-fold AUC summary
    aucs = [f["auc"] for f in fold_results]
    print(f"\n  Per-fold AUC: mean={np.mean(aucs):.4f}, std={np.std(aucs):.4f}, "
          f"min={np.min(aucs):.4f}, max={np.max(aucs):.4f}")

    # Save results
    results = {
        "concat_auc": float(concat_auc),
        "concat_accuracy": float(concat_acc),
        "concat_pos_rate": float(concat_pos_rate),
        "total_samples": int(len(concat_targets)),
        "n_folds": len(fold_results),
        "per_fold_auc_mean": float(np.mean(aucs)),
        "per_fold_auc_std": float(np.std(aucs)),
        "window_summary": window_summary,
        "fold_results": fold_results,
        "hyperparams": {
            "train_window": TRAIN_WINDOW,
            "hidden_dims": HIDDEN_DIMS,
            "lr": LR,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "dropout": DROPOUT,
            "patience": PATIENCE,
            "short_percentile": SHORT_PERCENTILE,
        },
        "timestamp": datetime.now().isoformat(),
    }

    results_path = OUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVE] Results → {results_path}")

    # Save concat predictions
    concat_path = OUT_DIR / "concat_oot_predictions.npz"
    np.savez_compressed(concat_path,
                        probs=concat_probs,
                        targets=concat_targets,
                        hours=concat_hours)
    print(f"[SAVE] Concat predictions → {concat_path}")

    # MLflow final logging
    if mlflow_available:
        mlflow.log_metrics({
            "concat_auc": concat_auc,
            "concat_accuracy": concat_acc,
            "concat_pos_rate": concat_pos_rate,
            "total_oot_samples": len(concat_targets),
            "per_fold_auc_mean": float(np.mean(aucs)),
            "per_fold_auc_std": float(np.std(aucs)),
        })
        # Log window AUCs as individual metrics
        for label, ws in window_summary.items():
            safe_label = label.replace(":", "")
            mlflow.log_metric(f"window_auc_{safe_label}", ws["auc"])
            mlflow.log_metric(f"window_count_{safe_label}", ws["count"])
            mlflow.log_metric(f"window_acc_{safe_label}", ws["accuracy"])

        mlflow.log_artifact(str(results_path))
        mlflow.end_run()
        print("[MLFLOW] Run logged and closed")

    print(f"\n{'=' * 70}")
    print(f"DONE — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
