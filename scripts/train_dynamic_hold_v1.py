#!/usr/bin/env python3
"""
train_dynamic_hold_v1.py — Dynamic Hold Duration Predictor

Trains a lightweight MLP to predict the OPTIMAL HOLD DURATION per trade.
Context: 1s hold has best Sharpe (0.192), 30s has best per-trade ticks (1.14).
A dynamic selector could capture the best of both.

Model: 4-class classifier (1s, 5s, 10s, 30s hold) using MBO event features.
Target: argmax of realized MFE across horizons after commission cost.
Walk-forward: 60-day sliding train, 1-day OOT.

Designed for Razer (Windows, RTX 3070 8GB) but auto-detects OS.
"""

import os
import sys
import json
import time
import glob
import platform
import warnings
from datetime import datetime
from pathlib import Path
from collections import Counter

import numpy as np

warnings.filterwarnings("ignore")

# ── Path Setup (Windows vs Linux) ──────────────────────────────────────────
IS_WINDOWS = platform.system() == "Windows"

if IS_WINDOWS:
    ROOT = Path(r"C:\Users\claude\Lvl3Quant")
else:
    ROOT = Path("/home/jupiter/Lvl3Quant") if Path("/home/jupiter/Lvl3Quant").exists() \
        else Path("/home/nick/Lvl3Quant")

MBO_DIR = ROOT / "data" / "processed" / "mbo_events_smart_v3"
PRED_DIR_V1 = ROOT / "output" / "cnn_mamba_v2_bulk_oot"
PRED_DIR_V2 = ROOT / "output" / "cnn_mamba_v2_bulk_oot_v2"
OUTPUT_DIR = ROOT / "output" / "dynamic_hold_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────
COST_TICKS = 0.376          # passive limit round-trip commission
HORIZONS = ["1s", "5s", "10s", "30s"]
N_CLASSES = len(HORIZONS)
HORIZON_LABELS = {h: i for i, h in enumerate(HORIZONS)}

# MLP architecture
HIDDEN_DIMS = [128, 64, 32]
BATCH_SIZE = 256
EPOCHS = 5
LR = 1e-3
WEIGHT_DECAY = 1e-4

# Walk-forward config
TRAIN_DAYS = 15  # reduced from 60 — only 33 MBO dates available across two chunks
# CNN-Mamba prediction alignment
CNN_WINDOW = 3000
CNN_STRIDE = 250

# ── Torch imports (deferred for cleaner error) ────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    HAS_TORCH = True
except ImportError:
    print("ERROR: PyTorch not found. Install with: pip install torch")
    sys.exit(1)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ── MLP Model ─────────────────────────────────────────────────────────────
class HoldDurationMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=HIDDEN_DIMS, n_classes=N_CLASSES, dropout=0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ── Data Loading Utilities ─────────────────────────────────────────────────
def get_available_dates():
    """Get sorted list of dates with both MBO events and CNN predictions."""
    mbo_dates = set()
    for f in MBO_DIR.glob("*_mbo_events.npz"):
        date_str = f.stem.split("_")[0]
        if date_str.startswith("2026"):
            mbo_dates.add(date_str)

    # Check both prediction directories
    pred_dates = set()
    for pred_dir in [PRED_DIR_V1, PRED_DIR_V2]:
        if pred_dir.exists():
            for f in pred_dir.glob("*_predictions.npz"):
                date_str = f.stem.split("_")[0]
                pred_dates.add(date_str)

    common = sorted(mbo_dates & pred_dates)
    print(f"Found {len(mbo_dates)} MBO dates, {len(pred_dates)} prediction dates, {len(common)} overlap")
    return common


def load_pred_file(date_str):
    """Load CNN-Mamba predictions for a date, checking both directories."""
    for pred_dir in [PRED_DIR_V1, PRED_DIR_V2]:
        path = pred_dir / f"{date_str}_predictions.npz"
        if path.exists():
            return np.load(path, allow_pickle=True)
    return None


def inspect_mbo_structure():
    """Print MBO structure and exit if format is unexpected."""
    sample_files = sorted(MBO_DIR.glob("*_mbo_events.npz"))
    if not sample_files:
        print(f"ERROR: No MBO event files found in {MBO_DIR}")
        sys.exit(1)

    d = np.load(sample_files[0], allow_pickle=True)
    keys = sorted(d.keys())
    print(f"\nMBO file structure ({sample_files[0].name}):")
    for k in keys:
        arr = d[k]
        print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}")

    required = {"events", "labels_1s", "labels_5s", "labels_10s", "labels_30s", "timestamps"}
    if not required.issubset(set(keys)):
        print(f"\nERROR: Missing required keys. Have: {keys}, Need: {required}")
        sys.exit(1)

    print(f"\n  Feature dim: {d['events'].shape[1]}")
    print(f"  Events per file: ~{d['events'].shape[0]:,}")
    return d["events"].shape[1]  # feature dimension


def load_day_features(date_str, subsample_stride=None):
    """
    Load features and multi-horizon labels for one day.

    Returns features at CNN-Mamba prediction points (every 250 events, offset by window=3000).
    This aligns with where we'd actually make trading decisions.

    Features: 25 MBO features + CNN prediction (3 horizons) + derived features = 31 dims
    Labels: MFE at each of 4 horizons (1s, 5s, 10s, 30s) in ticks
    """
    mbo_path = MBO_DIR / f"{date_str}_mbo_events.npz"
    if not mbo_path.exists():
        return None, None

    mbo = np.load(mbo_path, allow_pickle=True)
    events = mbo["events"]     # (N, 25)
    n_events = events.shape[0]

    # Labels at each horizon (directional move in ticks from entry)
    labels = {}
    for h in HORIZONS:
        labels[h] = mbo[f"labels_{h}"]  # (N,)

    # CNN-Mamba prediction indices: prediction i -> event index CNN_WINDOW + i * CNN_STRIDE
    pred_data = load_pred_file(date_str)
    if pred_data is not None:
        cnn_preds = pred_data["predictions"]  # (n_pred, 3) for 1s/5s/10s
        n_pred = cnn_preds.shape[0]
    else:
        # No predictions available — use zeros
        n_pred = max(0, (n_events - CNN_WINDOW) // CNN_STRIDE + 1)
        cnn_preds = np.zeros((n_pred, 3), dtype=np.float32)

    if n_pred == 0:
        return None, None

    # Build feature matrix at each prediction point
    pred_indices = CNN_WINDOW + np.arange(n_pred) * CNN_STRIDE  # MBO event indices
    # Clip to valid range
    valid_mask = pred_indices < n_events
    pred_indices = pred_indices[valid_mask]
    cnn_preds = cnn_preds[valid_mask[:len(cnn_preds)]]
    n_valid = len(pred_indices)

    if n_valid == 0:
        return None, None

    # Base MBO features at each prediction point
    base_features = events[pred_indices]  # (n_valid, 25)

    # CNN predictions as features (confidence signal)
    cnn_features = cnn_preds[:n_valid]  # (n_valid, 3) — 1s/5s/10s predictions

    # Derived features:
    # 1. CNN prediction magnitude (overall confidence)
    cnn_mag = np.abs(cnn_features).mean(axis=1, keepdims=True)  # (n_valid, 1)
    # 2. CNN prediction agreement (do all horizons agree on direction?)
    cnn_sign_agree = (np.sign(cnn_features).std(axis=1, keepdims=True))  # (n_valid, 1)
    # 3. Recent volatility proxy: std of feature 0 over a local window
    #    Use rolling std over last 50 prediction points
    feat0 = base_features[:, 0]
    vol_proxy = np.zeros((n_valid, 1), dtype=np.float32)
    for i in range(n_valid):
        start = max(0, i - 50)
        vol_proxy[i, 0] = feat0[start:i+1].std() if i > 0 else 0.0

    # Stack all features: 25 + 3 + 1 + 1 + 1 = 31
    X = np.hstack([base_features, cnn_features, cnn_mag, cnn_sign_agree, vol_proxy]).astype(np.float32)

    # Labels: realized directional move at each horizon (already in ticks)
    Y = np.zeros((n_valid, N_CLASSES), dtype=np.float32)
    for i, h in enumerate(HORIZONS):
        h_labels = labels[h][pred_indices]
        Y[:, i] = h_labels

    # Remove rows with any NaN
    valid = ~(np.isnan(X).any(axis=1) | np.isnan(Y).any(axis=1))
    X = X[valid]
    Y = Y[valid]

    if len(X) == 0:
        return None, None

    # Optional subsampling for training efficiency
    if subsample_stride and subsample_stride > 1:
        idx = np.arange(0, len(X), subsample_stride)
        X = X[idx]
        Y = Y[idx]

    return X, Y


def compute_optimal_hold_labels(Y_raw, direction="short"):
    """
    Given raw directional moves (ticks) at each horizon, compute the optimal hold class.

    Y_raw: (N, 4) — directional moves at 1s, 5s, 10s, 30s
    direction: "short" or "long" — determines profit sign

    For shorts: profit = -move (price going down = profit)
    For longs:  profit = +move (price going up = profit)

    Returns: (N,) class labels 0-3 = argmax of (profit - cost)
    """
    if direction == "short":
        profits = -Y_raw - COST_TICKS  # short profit at each horizon
    else:
        profits = Y_raw - COST_TICKS   # long profit at each horizon

    # The optimal hold is whichever horizon gives the best profit
    # If all negative, still pick the least-bad (model learns to avoid these via meta-model)
    labels = np.argmax(profits, axis=1)
    return labels, profits


# ── Training & Evaluation ──────────────────────────────────────────────────
def train_fold(X_train, y_train, X_val, y_val, input_dim, fold_id=""):
    """Train MLP for one fold. Returns model and predictions on val set."""
    model = HoldDurationMLP(input_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    # Class weights for imbalanced classes
    class_counts = Counter(y_train.tolist())
    total = len(y_train)
    weights = torch.tensor([total / (N_CLASSES * class_counts.get(i, 1)) for i in range(N_CLASSES)],
                           dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # DataLoaders
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)

    # Train
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

    # Predict on validation
    model.eval()
    with torch.no_grad():
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
        # Process in chunks to avoid OOM
        preds = []
        probs_list = []
        chunk_size = 4096
        for i in range(0, len(X_val_t), chunk_size):
            chunk = X_val_t[i:i+chunk_size]
            logits = model(chunk)
            prob = torch.softmax(logits, dim=1)
            preds.append(logits.argmax(dim=1).cpu().numpy())
            probs_list.append(prob.cpu().numpy())
        preds = np.concatenate(preds)
        probs = np.concatenate(probs_list)

    return model, preds, probs


def evaluate_dynamic_vs_fixed(preds, y_true, profits_raw):
    """
    Compare dynamic hold selection vs fixed hold strategies.

    preds: (N,) predicted hold class 0-3
    y_true: (N,) optimal hold class 0-3
    profits_raw: (N, 4) profit at each horizon in ticks

    Returns dict of metrics.
    """
    n = len(preds)
    if n == 0:
        return {}

    # Dynamic strategy: use predicted hold
    dynamic_pnl = np.array([profits_raw[i, preds[i]] for i in range(n)])

    # Fixed strategies
    fixed_pnl = {}
    for c, h in enumerate(HORIZONS):
        fixed_pnl[h] = profits_raw[:, c]

    # Oracle (always picks best)
    oracle_pnl = np.array([profits_raw[i, y_true[i]] for i in range(n)])

    # Accuracy
    accuracy = (preds == y_true).mean()

    # Per-class F1
    from collections import defaultdict
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    for p, t in zip(preds, y_true):
        if p == t:
            tp[p] += 1
        else:
            fp[p] += 1
            fn[t] += 1

    f1_per_class = {}
    for c in range(N_CLASSES):
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0
        rec = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        f1_per_class[HORIZONS[c]] = round(f1, 4)

    def sharpe(arr):
        if len(arr) == 0 or arr.std() == 0:
            return 0.0
        return float(arr.mean() / arr.std())

    results = {
        "n_trades": n,
        "accuracy": round(float(accuracy), 4),
        "f1_per_class": f1_per_class,
        "dynamic": {
            "total_ticks": round(float(dynamic_pnl.sum()), 2),
            "mean_ticks": round(float(dynamic_pnl.mean()), 4),
            "win_rate": round(float((dynamic_pnl > 0).mean()), 4),
            "sharpe": round(sharpe(dynamic_pnl), 4),
        },
        "oracle": {
            "total_ticks": round(float(oracle_pnl.sum()), 2),
            "mean_ticks": round(float(oracle_pnl.mean()), 4),
        },
    }

    for h in HORIZONS:
        c = HORIZON_LABELS[h]
        arr = fixed_pnl[h]
        results[f"fixed_{h}"] = {
            "total_ticks": round(float(arr.sum()), 2),
            "mean_ticks": round(float(arr.mean()), 4),
            "win_rate": round(float((arr > 0).mean()), 4),
            "sharpe": round(sharpe(arr), 4),
        }

    return results


# ── Main Walk-Forward Loop ─────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("Dynamic Hold Duration Predictor v1")
    print(f"Root: {ROOT}")
    print(f"MBO dir: {MBO_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 70)

    # Verify data structure
    feature_dim_raw = inspect_mbo_structure()
    input_dim = feature_dim_raw + 3 + 1 + 1 + 1  # 25 + 3 CNN + mag + agree + vol = 31

    # Get dates with both MBO and CNN predictions (for OOT evaluation)
    oot_candidate_dates = get_available_dates()

    # All MBO dates (for training — don't need CNN predictions for train data)
    all_mbo_dates = sorted([
        f.stem.split("_")[0] for f in MBO_DIR.glob("*_mbo_events.npz")
        if f.stem.split("_")[0] >= "20250714"  # start from first valid date
    ])
    print(f"Total MBO dates: {len(all_mbo_dates)}")

    # OOT dates = dates with both MBO + CNN, that have 60 prior MBO days for training
    oot_candidates = []
    for d in oot_candidate_dates:
        prior = [m for m in all_mbo_dates if m < d]
        if len(prior) >= TRAIN_DAYS:
            oot_candidates.append(d)

    if not oot_candidates:
        print(f"ERROR: No valid OOT dates with {TRAIN_DAYS} prior training days")
        sys.exit(1)

    # Limit to ~15 OOT folds for speed (spread across available range)
    if len(oot_candidates) > 20:
        step = max(1, len(oot_candidates) // 15)
        oot_dates = oot_candidates[::step][:15]
    else:
        oot_dates = oot_candidates[:15]

    print(f"\nOOT dates ({len(oot_dates)}): {oot_dates[0]} ... {oot_dates[-1]}")
    print(f"Training window: {TRAIN_DAYS} days sliding")
    print(f"Input dim: {input_dim}, Classes: {N_CLASSES} ({HORIZONS})")
    print(f"Architecture: MLP {input_dim}->{'->'.join(map(str, HIDDEN_DIMS))}->{N_CLASSES}")
    print()

    # Walk-forward loop
    all_preds = []
    all_true = []
    all_profits = []
    all_oot_dates = []
    fold_results = []
    best_model = None
    best_accuracy = 0.0

    for fold_i, oot_date in enumerate(oot_dates):
        t0 = time.time()
        print(f"Fold {fold_i+1}/{len(oot_dates)}: OOT={oot_date}", end=" ... ", flush=True)

        # Find training dates: 60 MBO dates prior to OOT
        prior_mbo = [d for d in all_mbo_dates if d < oot_date]
        if len(prior_mbo) < TRAIN_DAYS:
            print(f"SKIP (only {len(prior_mbo)} prior dates)")
            continue
        train_dates = prior_mbo[-TRAIN_DAYS:]

        # Load training data (subsample every 4th point for speed)
        X_trains = []
        Y_trains = []
        for td in train_dates:
            X_day, Y_day = load_day_features(td, subsample_stride=4)
            if X_day is not None:
                X_trains.append(X_day)
                Y_trains.append(Y_day)

        if not X_trains:
            print("SKIP (no valid training data)")
            continue

        X_train = np.concatenate(X_trains)
        Y_train_raw = np.concatenate(Y_trains)

        # Compute optimal hold labels for shorts (primary edge)
        y_train, train_profits = compute_optimal_hold_labels(Y_train_raw, direction="short")

        # Load OOT data (no subsampling)
        X_oot, Y_oot_raw = load_day_features(oot_date, subsample_stride=None)
        if X_oot is None:
            print("SKIP (no OOT data)")
            continue

        y_oot, oot_profits = compute_optimal_hold_labels(Y_oot_raw, direction="short")

        # Normalize features (fit on train, apply to both)
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std[std == 0] = 1.0
        X_train_n = (X_train - mean) / std
        X_oot_n = (X_oot - mean) / std

        # Replace any residual NaN/inf
        X_train_n = np.nan_to_num(X_train_n, nan=0.0, posinf=0.0, neginf=0.0)
        X_oot_n = np.nan_to_num(X_oot_n, nan=0.0, posinf=0.0, neginf=0.0)

        # Train
        model, preds, probs = train_fold(X_train_n, y_train, X_oot_n, y_oot,
                                          input_dim=input_dim, fold_id=oot_date)

        # Evaluate
        fold_eval = evaluate_dynamic_vs_fixed(preds, y_oot, oot_profits)
        fold_eval["date"] = oot_date
        fold_eval["train_size"] = len(X_train)
        fold_eval["oot_size"] = len(X_oot)
        fold_results.append(fold_eval)

        # Collect for concat evaluation
        all_preds.append(preds)
        all_true.append(y_oot)
        all_profits.append(oot_profits)
        all_oot_dates.extend([oot_date] * len(preds))

        elapsed = time.time() - t0

        # Track best model
        acc = fold_eval.get("accuracy", 0)
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_fold_date = oot_date

        # Summary line
        dyn_ticks = fold_eval.get("dynamic", {}).get("mean_ticks", 0)
        f1s_ticks = fold_eval.get("fixed_1s", {}).get("mean_ticks", 0)
        f30s_ticks = fold_eval.get("fixed_30s", {}).get("mean_ticks", 0)
        class_dist = Counter(preds.tolist())
        dist_str = " ".join(f"{HORIZONS[k]}:{v}" for k, v in sorted(class_dist.items()))
        print(f"acc={acc:.3f} | dyn={dyn_ticks:+.3f}t | fix1s={f1s_ticks:+.3f}t | "
              f"fix30s={f30s_ticks:+.3f}t | dist=[{dist_str}] | {elapsed:.1f}s")

        # Save per-fold predictions
        np.savez_compressed(
            OUTPUT_DIR / f"{oot_date}_hold_preds.npz",
            predictions=preds,
            probabilities=probs,
            true_labels=y_oot,
            profits=oot_profits,
            date=oot_date,
        )

    # ── Concat Evaluation ──────────────────────────────────────────────────
    if not all_preds:
        print("\nERROR: No successful folds. Check data availability.")
        sys.exit(1)

    all_preds_cat = np.concatenate(all_preds)
    all_true_cat = np.concatenate(all_true)
    all_profits_cat = np.concatenate(all_profits)

    print("\n" + "=" * 70)
    print("CONCAT RESULTS (all OOT folds combined)")
    print("=" * 70)

    concat_eval = evaluate_dynamic_vs_fixed(all_preds_cat, all_true_cat, all_profits_cat)

    print(f"\nTotal OOT trades: {concat_eval['n_trades']:,}")
    print(f"Accuracy: {concat_eval['accuracy']:.4f}")
    print(f"Per-class F1: {concat_eval['f1_per_class']}")

    # Class distribution
    pred_dist = Counter(all_preds_cat.tolist())
    true_dist = Counter(all_true_cat.tolist())
    print(f"\nPredicted distribution:")
    for c in range(N_CLASSES):
        print(f"  {HORIZONS[c]}: {pred_dist.get(c, 0):,} ({pred_dist.get(c, 0)/len(all_preds_cat)*100:.1f}%)")
    print(f"True optimal distribution:")
    for c in range(N_CLASSES):
        print(f"  {HORIZONS[c]}: {true_dist.get(c, 0):,} ({true_dist.get(c, 0)/len(all_true_cat)*100:.1f}%)")

    print(f"\n{'Strategy':<15} {'Total Ticks':>12} {'Mean/Trade':>12} {'Win Rate':>10} {'Sharpe':>8}")
    print("-" * 60)
    dyn = concat_eval["dynamic"]
    print(f"{'Dynamic':<15} {dyn['total_ticks']:>12.1f} {dyn['mean_ticks']:>12.4f} {dyn['win_rate']:>10.4f} {dyn['sharpe']:>8.4f}")
    orc = concat_eval["oracle"]
    print(f"{'Oracle':<15} {orc['total_ticks']:>12.1f} {orc['mean_ticks']:>12.4f} {'—':>10} {'—':>8}")
    for h in HORIZONS:
        fx = concat_eval[f"fixed_{h}"]
        print(f"{'Fixed '+h:<15} {fx['total_ticks']:>12.1f} {fx['mean_ticks']:>12.4f} {fx['win_rate']:>10.4f} {fx['sharpe']:>8.4f}")

    # Tick value conversion
    TICK_VALUE = 12.50
    print(f"\nDynamic total P&L: ${dyn['total_ticks'] * TICK_VALUE:,.2f} "
          f"(vs Fixed 1s: ${concat_eval['fixed_1s']['total_ticks'] * TICK_VALUE:,.2f}, "
          f"Fixed 30s: ${concat_eval['fixed_30s']['total_ticks'] * TICK_VALUE:,.2f})")

    # ── Save Results ───────────────────────────────────────────────────────
    results = {
        "experiment": "dynamic_hold_v1",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "train_days": TRAIN_DAYS,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "hidden_dims": HIDDEN_DIMS,
            "input_dim": input_dim,
            "n_classes": N_CLASSES,
            "horizons": HORIZONS,
            "cost_ticks": COST_TICKS,
            "direction": "short",
            "train_subsample_stride": 4,
        },
        "oot_dates": [d for d in oot_dates if any(fr["date"] == d for fr in fold_results)],
        "concat": concat_eval,
        "per_fold": fold_results,
    }

    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Save best model weights
    if best_model is not None:
        weights_path = OUTPUT_DIR / "best_model.pt"
        torch.save({
            "model_state_dict": best_model.state_dict(),
            "input_dim": input_dim,
            "hidden_dims": HIDDEN_DIMS,
            "n_classes": N_CLASSES,
            "best_fold_date": best_fold_date,
            "best_accuracy": best_accuracy,
        }, weights_path)
        print(f"Best model weights saved to {weights_path} (fold {best_fold_date}, acc={best_accuracy:.4f})")

    # Save concat predictions
    np.savez_compressed(
        OUTPUT_DIR / "concat_predictions.npz",
        predictions=all_preds_cat,
        true_labels=all_true_cat,
        profits=all_profits_cat,
        dates=np.array(all_oot_dates),
    )
    print(f"Concat predictions saved to {OUTPUT_DIR / 'concat_predictions.npz'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
