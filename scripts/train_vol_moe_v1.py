#!/usr/bin/env python3
"""
Volatility MoE v1 — Mixture-of-Experts gated by volatility regime.

CONTEXT: CNN-Mamba signal has regime imbalance (Sharpe varies by market condition).
This MoE dispatches to 3 expert networks based on volatility, learned end-to-end
by a gate network. If concat AUC > 0.60 (vs ToD gate baseline 0.588), it's useful.

Architecture: MoE with 3 experts
  Gate network: 25 → 16 → 3 (softmax) — learns regime assignment from raw features
  Expert 1 (low vol):  25 → 64 → 32 → 1 (sigmoid, BCE)
  Expert 2 (mid vol):  25 → 64 → 32 → 1
  Expert 3 (high vol): 25 → 64 → 32 → 1
  Output: weighted sum of expert outputs using gate weights
  Total params: ~15K
  Loss: BCE (predicting profitable short) + 0.1 * load_balancing_loss

Walk-forward: sliding 10-date train, 1-date OOT
Signal filter: only events where CNN-Mamba prediction[:,0] in bottom 5th percentile
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
OUT_DIR = DATA_ROOT / "output" / "vol_moe_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ──
TRAIN_WINDOW = 10       # sliding window dates
NUM_EXPERTS = 3
GATE_HIDDEN = 16
EXPERT_HIDDEN = [64, 32]
LR = 1e-3
EPOCHS = 30
BATCH_SIZE = 4096
WEIGHT_DECAY = 1e-4
DROPOUT = 0.15
PATIENCE = 5            # early stopping
SHORT_PERCENTILE = 5    # bottom 5% of predictions = short signals
LOAD_BALANCE_COEFF = 0.1  # weight for load balancing loss
INPUT_DIM = 25

# ── MLflow ──
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "vol_moe_v1"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score


# ── Model ──

class ExpertNetwork(nn.Module):
    """Single expert MLP: input_dim → 64 → 32 → 1."""

    def __init__(self, in_dim=25, hidden_dims=[64, 32], dropout=0.15):
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
        return self.net(x)  # (B, 1) — raw logit


class GateNetwork(nn.Module):
    """Gate that learns regime assignment: input_dim → 16 → num_experts (softmax)."""

    def __init__(self, in_dim=25, hidden_dim=16, num_experts=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, x):
        return F.softmax(self.net(x), dim=-1)  # (B, num_experts)


class VolatilityMoE(nn.Module):
    """Mixture-of-Experts model gated by learned volatility regime."""

    def __init__(self, in_dim=25, num_experts=3, gate_hidden=16,
                 expert_hidden=[64, 32], dropout=0.15):
        super().__init__()
        self.num_experts = num_experts
        self.gate = GateNetwork(in_dim, gate_hidden, num_experts)
        self.experts = nn.ModuleList([
            ExpertNetwork(in_dim, expert_hidden, dropout)
            for _ in range(num_experts)
        ])

    def forward(self, x):
        """
        Returns:
            logits: (B, 1) — weighted sum of expert logits
            gate_weights: (B, num_experts) — for load balancing loss
        """
        gate_weights = self.gate(x)            # (B, num_experts)
        expert_outputs = torch.stack(
            [expert(x) for expert in self.experts], dim=-1
        )                                       # (B, 1, num_experts)
        # Weighted sum: (B, 1, num_experts) * (B, 1, num_experts) → sum → (B, 1)
        gate_expanded = gate_weights.unsqueeze(1)  # (B, 1, num_experts)
        logits = (expert_outputs * gate_expanded).sum(dim=-1)  # (B, 1)
        return logits, gate_weights

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def load_balancing_loss(gate_weights):
    """Encourage even expert usage. Penalize concentration.

    Uses the standard Switch Transformer load balancing: importance * fraction.
    Lower when all experts get equal traffic.
    """
    # gate_weights: (B, num_experts)
    # Fraction of tokens routed to each expert
    fractions = gate_weights.mean(dim=0)  # (num_experts,)
    # Importance: sum of gate weights per expert
    # For soft MoE, importance = fractions * B, but we normalize
    # Loss = num_experts * sum(f_i^2) — minimized when all f_i = 1/num_experts
    num_experts = gate_weights.shape[1]
    loss = num_experts * (fractions ** 2).sum()
    return loss


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


def load_date(date_str):
    """Load features, labels, and CNN-Mamba predictions for a date.

    Returns (features_25d, targets, cnn_mamba_preds_aligned) for short-signal-filtered
    events, or None if loading fails.
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
    aligned_labels = labels_1s[event_indices]    # (M,)

    # Filter to short signals: bottom 5th percentile of prediction[:,0]
    short_threshold = np.percentile(preds_aligned[:, 0], SHORT_PERCENTILE)
    short_mask = preds_aligned[:, 0] <= short_threshold

    if short_mask.sum() < 10:
        return None

    # Apply filter
    filt_events = aligned_events[short_mask].astype(np.float32)  # (K, 25)
    filt_labels = aligned_labels[short_mask]                      # (K,)

    # Target: 1 if labels_1s < 0 (price went down = profitable short)
    targets = (filt_labels < 0).astype(np.float32)

    return filt_events, targets


def compute_class_weights(targets):
    """Compute balanced class weights for BCE."""
    n_pos = targets.sum()
    n_neg = len(targets) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    w_pos = n_neg / n_pos
    return w_pos


def train_fold(model, train_features, train_targets, device, class_weight_pos):
    """Train MoE model for one fold with early stopping.

    Returns (feat_mean, feat_std, best_loss, expert_utilization_train).
    """
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
    best_state = None

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        epoch_lb_loss = 0.0
        n_batches = 0

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()

            logits, gate_weights = model(xb)
            bce_loss = criterion(logits, yb)
            lb_loss = load_balancing_loss(gate_weights)
            loss = bce_loss + LOAD_BALANCE_COEFF * lb_loss

            loss.backward()
            optimizer.step()

            epoch_loss += bce_loss.item()
            epoch_lb_loss += lb_loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_lb = epoch_lb_loss / max(n_batches, 1)

        # Early stopping on BCE loss (not total loss)
        if avg_loss < best_loss - 1e-5:
            best_loss = avg_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return feat_mean, feat_std, best_loss


def evaluate_fold(model, test_features, test_targets, feat_mean, feat_std, device):
    """Evaluate MoE on OOT fold. Returns metrics dict with expert analysis."""
    model.eval()
    X = torch.tensor(test_features, dtype=torch.float32)
    X = (X - feat_mean) / feat_std

    with torch.no_grad():
        logits, gate_weights = model(X.to(device))
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        gate_np = gate_weights.cpu().numpy()  # (N, num_experts)

    targets = test_targets
    preds_binary = (probs >= 0.5).astype(int)

    # Overall metrics
    try:
        auc = roc_auc_score(targets, probs)
    except ValueError:
        auc = 0.5
    acc = accuracy_score(targets, preds_binary)

    # ── Expert Analysis ──
    # Which expert has highest gate weight per sample
    dominant_expert = gate_np.argmax(axis=1)  # (N,)

    expert_stats = {}
    for e in range(NUM_EXPERTS):
        mask = dominant_expert == e
        n_assigned = mask.sum()
        pct = n_assigned / len(targets) * 100 if len(targets) > 0 else 0

        if n_assigned >= 5:
            e_targets = targets[mask]
            e_probs = probs[mask]
            e_preds = preds_binary[mask]
            try:
                e_auc = roc_auc_score(e_targets, e_probs)
            except ValueError:
                e_auc = 0.5
            e_acc = accuracy_score(e_targets, e_preds)
            e_pos_rate = float(e_targets.mean())
        else:
            e_auc = 0.5
            e_acc = 0.0
            e_pos_rate = 0.0

        expert_stats[f"expert_{e}"] = {
            "n_assigned": int(n_assigned),
            "pct_assigned": float(pct),
            "auc": float(e_auc),
            "accuracy": float(e_acc),
            "pos_rate": float(e_pos_rate),
            "avg_gate_weight": float(gate_np[:, e].mean()),
        }

    # Mean gate weights (soft utilization)
    mean_gate = gate_np.mean(axis=0)  # (num_experts,)

    return {
        "auc": float(auc),
        "accuracy": float(acc),
        "n_samples": int(len(targets)),
        "pos_rate": float(targets.mean()),
        "expert_stats": expert_stats,
        "mean_gate_weights": [float(g) for g in mean_gate],
        "probs": probs,
        "targets": targets,
        "gate_weights": gate_np,
        "dominant_expert": dominant_expert,
    }


def main():
    print("=" * 70)
    print("Volatility MoE v1 — Mixture-of-Experts by Regime")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    # Model param count
    test_model = VolatilityMoE(INPUT_DIM, NUM_EXPERTS, GATE_HIDDEN, EXPERT_HIDDEN, DROPOUT)
    print(f"[MODEL] VolatilityMoE — {test_model.count_parameters():,} params, {NUM_EXPERTS} experts")
    del test_model

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

    print(f"[DATES] {len(dates)} dates: {dates[0]} -> {dates[-1]}")
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
    all_oot_gates = []
    all_oot_dominant = []
    fold_results = []

    if mlflow_available:
        mlflow.start_run(run_name=f"vol_moe_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        mlflow.log_params({
            "train_window": TRAIN_WINDOW,
            "num_experts": NUM_EXPERTS,
            "gate_hidden": GATE_HIDDEN,
            "expert_hidden": str(EXPERT_HIDDEN),
            "lr": LR,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "dropout": DROPOUT,
            "patience": PATIENCE,
            "short_percentile": SHORT_PERCENTILE,
            "load_balance_coeff": LOAD_BALANCE_COEFF,
            "input_dim": INPUT_DIM,
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
            feats, targets = date_cache[td]
            train_feats_list.append(feats)
            train_targets_list.append(targets)

        train_features = np.concatenate(train_feats_list, axis=0)
        train_targets = np.concatenate(train_targets_list, axis=0)

        # Test data
        test_features, test_targets = date_cache[test_date]

        # Class weights
        cw_pos = compute_class_weights(train_targets)
        if cw_pos is None:
            print(f"  Fold {fold_idx - TRAIN_WINDOW + 1}: {test_date} — SKIP (degenerate labels)")
            continue

        # Build model
        model = VolatilityMoE(
            in_dim=INPUT_DIM,
            num_experts=NUM_EXPERTS,
            gate_hidden=GATE_HIDDEN,
            expert_hidden=EXPERT_HIDDEN,
            dropout=DROPOUT,
        ).to(device)

        # Train
        feat_mean, feat_std, train_loss = train_fold(
            model, train_features, train_targets, device, cw_pos
        )

        # Evaluate
        metrics = evaluate_fold(
            model, test_features, test_targets, feat_mean, feat_std, device
        )

        fold_num = fold_idx - TRAIN_WINDOW + 1

        # Expert utilization summary
        expert_pcts = [metrics["expert_stats"][f"expert_{e}"]["pct_assigned"] for e in range(NUM_EXPERTS)]
        expert_aucs = [metrics["expert_stats"][f"expert_{e}"]["auc"] for e in range(NUM_EXPERTS)]

        print(f"  Fold {fold_num:>3}: {test_date} | "
              f"AUC={metrics['auc']:.4f} | Acc={metrics['accuracy']:.4f} | "
              f"N={metrics['n_samples']} | "
              f"Experts: [{expert_pcts[0]:.0f}%/{expert_pcts[1]:.0f}%/{expert_pcts[2]:.0f}%] "
              f"AUC:[{expert_aucs[0]:.3f}/{expert_aucs[1]:.3f}/{expert_aucs[2]:.3f}]")

        # Log per-fold to MLflow
        if mlflow_available:
            log_dict = {
                "fold_auc": metrics["auc"],
                "fold_accuracy": metrics["accuracy"],
                "fold_n_samples": metrics["n_samples"],
                "fold_train_loss": train_loss,
            }
            for e in range(NUM_EXPERTS):
                log_dict[f"fold_expert{e}_pct"] = expert_pcts[e]
                log_dict[f"fold_expert{e}_auc"] = expert_aucs[e]
            mlflow.log_metrics(log_dict, step=fold_num)

        # Accumulate OOT
        all_oot_probs.append(metrics["probs"])
        all_oot_targets.append(metrics["targets"])
        all_oot_gates.append(metrics["gate_weights"])
        all_oot_dominant.append(metrics["dominant_expert"])

        fold_results.append({
            "fold": fold_num,
            "test_date": test_date,
            "auc": metrics["auc"],
            "accuracy": metrics["accuracy"],
            "n_samples": metrics["n_samples"],
            "pos_rate": metrics["pos_rate"],
            "expert_stats": metrics["expert_stats"],
            "mean_gate_weights": metrics["mean_gate_weights"],
        })

        # Save fold predictions
        fold_out = OUT_DIR / f"{test_date}_vol_moe_preds.npz"
        np.savez_compressed(
            fold_out,
            probs=metrics["probs"],
            targets=metrics["targets"],
            gate_weights=metrics["gate_weights"],
            dominant_expert=metrics["dominant_expert"],
        )

        # Save model weights for this fold
        model_out = OUT_DIR / f"{test_date}_vol_moe_model.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "feat_mean": feat_mean,
            "feat_std": feat_std,
            "fold_num": fold_num,
            "test_date": test_date,
        }, model_out)

        # GPU cleanup
        del model, feat_mean, feat_std
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
    concat_gates = np.concatenate(all_oot_gates)
    concat_dominant = np.concatenate(all_oot_dominant)

    concat_auc = roc_auc_score(concat_targets, concat_probs)
    concat_acc = accuracy_score(concat_targets, (concat_probs >= 0.5).astype(int))
    concat_pos_rate = concat_targets.mean()

    print(f"\n  Overall OOT AUC:      {concat_auc:.4f}  (baseline ToD gate: 0.588)")
    print(f"  Overall OOT Accuracy: {concat_acc:.4f}")
    print(f"  Total samples:        {len(concat_targets)}")
    print(f"  Positive rate:        {concat_pos_rate:.3f}")

    # ── Expert Utilization Analysis ──
    print(f"\n  {'Expert':<10} {'Assigned':>10} {'Pct':>8} {'AUC':>8} {'Accuracy':>10} {'PosRate':>9} {'AvgGate':>9}")
    print("  " + "-" * 68)

    expert_summary = {}
    for e in range(NUM_EXPERTS):
        mask = concat_dominant == e
        n_assigned = mask.sum()
        pct = n_assigned / len(concat_targets) * 100

        if n_assigned >= 10:
            e_targets = concat_targets[mask]
            e_probs = concat_probs[mask]
            e_preds = (e_probs >= 0.5).astype(int)
            try:
                e_auc = roc_auc_score(e_targets, e_probs)
            except ValueError:
                e_auc = 0.5
            e_acc = accuracy_score(e_targets, e_preds)
            e_pos_rate = float(e_targets.mean())
        else:
            e_auc = 0.5
            e_acc = 0.0
            e_pos_rate = 0.0

        avg_gate = float(concat_gates[:, e].mean())

        print(f"  Expert {e:<4} {n_assigned:>10} {pct:>7.1f}% {e_auc:>8.4f} {e_acc:>10.4f} "
              f"{e_pos_rate:>9.3f} {avg_gate:>9.3f}")

        expert_summary[f"expert_{e}"] = {
            "n_assigned": int(n_assigned),
            "pct_assigned": float(pct),
            "auc": float(e_auc),
            "accuracy": float(e_acc),
            "pos_rate": float(e_pos_rate),
            "avg_gate_weight": avg_gate,
        }

    # ── Per-Fold AUC Summary ──
    aucs = [f["auc"] for f in fold_results]
    print(f"\n  Per-fold AUC: mean={np.mean(aucs):.4f}, std={np.std(aucs):.4f}, "
          f"min={np.min(aucs):.4f}, max={np.max(aucs):.4f}")

    # ── Expert Specialization Analysis ──
    # Check if experts learned different regimes by looking at feature variance
    # within each expert's assigned samples
    print("\n  Expert Specialization (feature std within each expert's samples):")
    for e in range(NUM_EXPERTS):
        mask = concat_dominant == e
        if mask.sum() < 10:
            continue
        # We don't have raw features in concat, but we can check gate weight entropy
        e_gates = concat_gates[mask]  # (n_assigned, num_experts)
        entropy = -(e_gates * np.log(e_gates + 1e-10)).sum(axis=1).mean()
        print(f"    Expert {e}: avg gate entropy={entropy:.3f} "
              f"(lower = more confident routing)")

    # ── Verdict ──
    print("\n" + "-" * 70)
    if concat_auc > 0.60:
        verdict = f"PASS — MoE AUC {concat_auc:.4f} > 0.60 baseline. Worth pursuing."
    elif concat_auc > 0.588:
        verdict = f"MARGINAL — MoE AUC {concat_auc:.4f} > ToD gate (0.588) but below 0.60."
    else:
        verdict = f"FAIL — MoE AUC {concat_auc:.4f} <= ToD gate baseline (0.588). Not useful."
    print(f"  VERDICT: {verdict}")
    print("-" * 70)

    # Save results
    results = {
        "concat_auc": float(concat_auc),
        "concat_accuracy": float(concat_acc),
        "concat_pos_rate": float(concat_pos_rate),
        "total_samples": int(len(concat_targets)),
        "n_folds": len(fold_results),
        "per_fold_auc_mean": float(np.mean(aucs)),
        "per_fold_auc_std": float(np.std(aucs)),
        "expert_summary": expert_summary,
        "fold_results": fold_results,
        "verdict": verdict,
        "baseline_tod_gate_auc": 0.588,
        "hyperparams": {
            "train_window": TRAIN_WINDOW,
            "num_experts": NUM_EXPERTS,
            "gate_hidden": GATE_HIDDEN,
            "expert_hidden": EXPERT_HIDDEN,
            "lr": LR,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "dropout": DROPOUT,
            "patience": PATIENCE,
            "short_percentile": SHORT_PERCENTILE,
            "load_balance_coeff": LOAD_BALANCE_COEFF,
            "input_dim": INPUT_DIM,
        },
        "timestamp": datetime.now().isoformat(),
    }

    results_path = OUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVE] Results saved")

    # Save concat predictions
    concat_path = OUT_DIR / "concat_oot_predictions.npz"
    np.savez_compressed(
        concat_path,
        probs=concat_probs,
        targets=concat_targets,
        gate_weights=concat_gates,
        dominant_expert=concat_dominant,
    )
    print(f"[SAVE] Concat predictions saved")

    # MLflow final logging
    if mlflow_available:
        mlflow.log_metrics({
            "concat_auc": concat_auc,
            "concat_accuracy": concat_acc,
            "concat_pos_rate": concat_pos_rate,
            "total_oot_samples": len(concat_targets),
            "per_fold_auc_mean": float(np.mean(aucs)),
            "per_fold_auc_std": float(np.std(aucs)),
            "beats_baseline": float(concat_auc > 0.588),
        })
        for e in range(NUM_EXPERTS):
            es = expert_summary[f"expert_{e}"]
            mlflow.log_metric(f"expert{e}_pct_assigned", es["pct_assigned"])
            mlflow.log_metric(f"expert{e}_auc", es["auc"])
            mlflow.log_metric(f"expert{e}_accuracy", es["accuracy"])
            mlflow.log_metric(f"expert{e}_avg_gate_weight", es["avg_gate_weight"])

        mlflow.log_artifact(str(results_path))
        mlflow.end_run()
        print("[MLFLOW] Run logged and closed")

    print(f"\n{'=' * 70}")
    print(f"DONE — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
