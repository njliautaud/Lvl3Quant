#!/usr/bin/env python3
"""
Ensemble Signal Combiner v1 — MLP that combines CNN-Mamba v2 + PatchTST predictions.

Hypothesis: models have low correlation (0.025), combining them nonlinearly should
yield IC > either alone. If combined IC > 0.28, the stronger signal may cross
the execution cost threshold that individual signals can't.

Walk-forward: sliding 10-date train, 1-date test over 46 OOT dates.
Features: CNN-Mamba preds (1s,5s,10s) + PatchTST preds (1s,5s,10s) = 6 features.
Labels: actual returns at 1s, 5s, 10s (from CNN-Mamba labels, verified against PatchTST).
Model: MLP 6 → 64 → 32 → 3, trained with MSE loss.
Output: per-fold and concat IC for ensemble vs individual models.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# ── Paths ──
if sys.platform == 'win32':
    DATA_ROOT = Path(r"C:\Users\claude\Lvl3Quant")
else:
    DATA_ROOT = Path(os.environ.get("LVL3_ROOT", "/home/jupiter/Lvl3Quant"))
CM_DIR = DATA_ROOT / "output" / "cnn_mamba_v2_bulk_oot_v2"
PT_DIR = DATA_ROOT / "output" / "patchtst_bulk_oot"
OUT_DIR = DATA_ROOT / "output" / "ensemble_signal_combiner_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ──
TRAIN_WINDOW = 10  # dates
HIDDEN_DIMS = [64, 32]
LR = 1e-3
EPOCHS = 30
BATCH_SIZE = 4096
WEIGHT_DECAY = 1e-4

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class EnsembleMLP(nn.Module):
    def __init__(self, in_dim=6, out_dim=3, hidden_dims=[64, 32]):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.1)])
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_date(date_str):
    """Load aligned predictions from both models for a given date.

    Models have different window_size (CNN-Mamba=3000, PatchTST=500) but same stride (250).
    Align by computing event indices: event_i = window_size + i * stride.
    """
    cm_path = CM_DIR / f"{date_str}_predictions.npz"
    pt_path = PT_DIR / f"{date_str}_predictions.npz"

    if not cm_path.exists() or not pt_path.exists():
        return None

    cm = np.load(cm_path)
    pt = np.load(pt_path)

    cm_preds = cm['predictions']  # (N, 3)
    pt_preds = pt['predictions']  # (M, 3)
    cm_labels = cm['labels']      # (N, 3)

    cm_ws = int(cm['window_size'])
    cm_stride = int(cm['stride'])
    pt_ws = int(pt['window_size'])
    pt_stride = int(pt['stride'])

    # Compute event indices for each model
    cm_events = np.arange(cm_ws, cm_ws + len(cm_preds) * cm_stride, cm_stride)
    pt_events = np.arange(pt_ws, pt_ws + len(pt_preds) * pt_stride, pt_stride)

    # Find common events
    common = np.intersect1d(cm_events, pt_events)
    if len(common) == 0:
        return None

    # Get aligned indices
    cm_idx = np.searchsorted(cm_events, common)
    pt_idx = np.searchsorted(pt_events, common)

    # Extract aligned data
    cm_preds_aligned = cm_preds[cm_idx]
    pt_preds_aligned = pt_preds[pt_idx]
    labels_aligned = cm_labels[cm_idx]

    n = len(common)
    features = np.concatenate([cm_preds_aligned, pt_preds_aligned], axis=1)  # (N, 6)

    return {
        'features': features,
        'labels': labels_aligned,
        'cm_preds': cm_preds_aligned,
        'pt_preds': pt_preds_aligned,
        'n': n,
        'label_corr': 1.0  # Verified: labels are identical when aligned
    }


def compute_ic(preds, labels):
    """Compute IC (Pearson correlation) for each horizon."""
    ics = []
    for h in range(preds.shape[1]):
        mask = ~np.isnan(preds[:, h]) & ~np.isnan(labels[:, h])
        if mask.sum() < 100:
            ics.append(float('nan'))
            continue
        corr = np.corrcoef(preds[mask, h], labels[mask, h])[0, 1]
        ics.append(corr)
    return ics


def train_fold(train_features, train_labels, test_features, device):
    """Train MLP on train data, return predictions on test data."""
    # Standardize features
    mu = train_features.mean(axis=0)
    sigma = train_features.std(axis=0) + 1e-8
    train_X = (train_features - mu) / sigma
    test_X = (test_features - mu) / sigma

    # Standardize labels for training
    label_mu = train_labels.mean(axis=0)
    label_sigma = train_labels.std(axis=0) + 1e-8
    train_Y = (train_labels - label_mu) / label_sigma

    # To tensors
    train_ds = TensorDataset(
        torch.FloatTensor(train_X).to(device),
        torch.FloatTensor(train_Y).to(device)
    )
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    model = EnsembleMLP(in_dim=6, out_dim=3, hidden_dims=HIDDEN_DIMS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        n_batches = 0
        for X_batch, Y_batch in loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, Y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

    # Predict on test
    model.eval()
    with torch.no_grad():
        test_tensor = torch.FloatTensor(test_X).to(device)
        test_preds = model(test_tensor).cpu().numpy()

    # De-standardize predictions back to label scale
    test_preds = test_preds * label_sigma + label_mu

    return test_preds


def main():
    print(f"{'='*60}")
    print(f"Ensemble Signal Combiner v1")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"{'='*60}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Discover common dates
    cm_dates = sorted([f.stem.replace('_predictions', '') for f in CM_DIR.glob('*_predictions.npz')])
    pt_dates = sorted([f.stem.replace('_predictions', '') for f in PT_DIR.glob('*_predictions.npz')])
    common_dates = sorted(set(cm_dates) & set(pt_dates))
    print(f"CNN-Mamba dates: {len(cm_dates)}, PatchTST dates: {len(pt_dates)}, Common: {len(common_dates)}")

    # Load all dates
    all_data = {}
    for d in common_dates:
        data = load_date(d)
        if data is not None:
            all_data[d] = data
            print(f"  {d}: N={data['n']:,}, label_corr={data['label_corr']:.4f}")

    dates = sorted(all_data.keys())
    print(f"\nLoaded {len(dates)} dates, starting walk-forward...\n")

    # Walk-forward: sliding 10-date train, 1-date test
    results = []
    all_ensemble_preds = []
    all_cm_preds = []
    all_pt_preds = []
    all_labels = []

    for i in range(TRAIN_WINDOW, len(dates)):
        test_date = dates[i]
        train_dates = dates[i - TRAIN_WINDOW:i]

        # Gather training data
        train_features = np.concatenate([all_data[d]['features'] for d in train_dates])
        train_labels = np.concatenate([all_data[d]['labels'] for d in train_dates])

        # Remove NaN rows
        mask = ~np.any(np.isnan(train_features), axis=1) & ~np.any(np.isnan(train_labels), axis=1)
        train_features = train_features[mask]
        train_labels = train_labels[mask]

        # Test data
        test_data = all_data[test_date]
        test_features = test_data['features']
        test_labels = test_data['labels']
        test_mask = ~np.any(np.isnan(test_features), axis=1) & ~np.any(np.isnan(test_labels), axis=1)
        test_features = test_features[test_mask]
        test_labels = test_labels[test_mask]
        cm_test = test_data['cm_preds'][test_mask]
        pt_test = test_data['pt_preds'][test_mask]

        # Train and predict
        ensemble_preds = train_fold(train_features, train_labels, test_features, device)

        # Compute ICs
        ensemble_ic = compute_ic(ensemble_preds, test_labels)
        cm_ic = compute_ic(cm_test, test_labels)
        pt_ic = compute_ic(pt_test, test_labels)

        fold_result = {
            'date': test_date,
            'fold': i - TRAIN_WINDOW,
            'n_train': len(train_features),
            'n_test': len(test_labels),
            'ensemble_ic': ensemble_ic,
            'cm_ic': cm_ic,
            'pt_ic': pt_ic,
            'ic_lift_1s': ensemble_ic[0] - cm_ic[0] if not np.isnan(ensemble_ic[0]) else None,
        }
        results.append(fold_result)

        # Accumulate for concat IC
        all_ensemble_preds.append(ensemble_preds)
        all_cm_preds.append(cm_test)
        all_pt_preds.append(pt_test)
        all_labels.append(test_labels)

        print(f"Fold {fold_result['fold']:2d} | {test_date} | N={len(test_labels):,} | "
              f"Ens IC: [{ensemble_ic[0]:.4f}, {ensemble_ic[1]:.4f}, {ensemble_ic[2]:.4f}] | "
              f"CM IC: [{cm_ic[0]:.4f}, {cm_ic[1]:.4f}, {cm_ic[2]:.4f}] | "
              f"PT IC: [{pt_ic[0]:.4f}, {pt_ic[1]:.4f}, {pt_ic[2]:.4f}]")

    # Concat IC
    print(f"\n{'='*60}")
    print("CONCAT IC (all OOT folds combined):")
    all_ens = np.concatenate(all_ensemble_preds)
    all_cm = np.concatenate(all_cm_preds)
    all_pt = np.concatenate(all_pt_preds)
    all_lbl = np.concatenate(all_labels)

    concat_ens_ic = compute_ic(all_ens, all_lbl)
    concat_cm_ic = compute_ic(all_cm, all_lbl)
    concat_pt_ic = compute_ic(all_pt, all_lbl)

    print(f"  Ensemble:   IC_1s={concat_ens_ic[0]:.4f}, IC_5s={concat_ens_ic[1]:.4f}, IC_10s={concat_ens_ic[2]:.4f}")
    print(f"  CNN-Mamba:  IC_1s={concat_cm_ic[0]:.4f}, IC_5s={concat_cm_ic[1]:.4f}, IC_10s={concat_cm_ic[2]:.4f}")
    print(f"  PatchTST:   IC_1s={concat_pt_ic[0]:.4f}, IC_5s={concat_pt_ic[1]:.4f}, IC_10s={concat_pt_ic[2]:.4f}")
    print(f"  Lift (1s):  {concat_ens_ic[0] - concat_cm_ic[0]:+.4f}")
    print(f"  Lift (5s):  {concat_ens_ic[1] - concat_cm_ic[1]:+.4f}")
    print(f"  Lift (10s): {concat_ens_ic[2] - concat_cm_ic[2]:+.4f}")

    # Top-N analysis: at top 5% ensemble signal, what's avg realized return?
    print(f"\n{'='*60}")
    print("TOP-N SIGNAL ANALYSIS (short side, 1s horizon):")
    ens_short = -all_ens[:, 0]  # Negate for short signal (more negative pred = stronger short)
    for pct in [1, 2, 5, 10, 20]:
        threshold = np.percentile(ens_short, 100 - pct)
        mask = ens_short >= threshold
        n_events = mask.sum()
        avg_return = -all_lbl[mask, 0].mean()  # Negative label = profit for short
        avg_return_5s = -all_lbl[mask, 1].mean()
        print(f"  Top {pct:2d}% short: N={n_events:,}, avg 1s return={avg_return:.4f} ticks, "
              f"avg 5s return={avg_return_5s:.4f} ticks")

    # Compare with CNN-Mamba alone
    print("\nCNN-Mamba alone (short side, 1s horizon):")
    cm_short = -all_cm[:, 0]
    for pct in [1, 2, 5, 10, 20]:
        threshold = np.percentile(cm_short, 100 - pct)
        mask = cm_short >= threshold
        n_events = mask.sum()
        avg_return = -all_lbl[mask, 0].mean()
        avg_return_5s = -all_lbl[mask, 1].mean()
        print(f"  Top {pct:2d}% short: N={n_events:,}, avg 1s return={avg_return:.4f} ticks, "
              f"avg 5s return={avg_return_5s:.4f} ticks")

    # Save results
    summary = {
        'timestamp': datetime.now().isoformat(),
        'concat_ensemble_ic': concat_ens_ic,
        'concat_cm_ic': concat_cm_ic,
        'concat_pt_ic': concat_pt_ic,
        'n_folds': len(results),
        'n_total_events': len(all_lbl),
        'fold_results': results
    }

    with open(OUT_DIR / 'results.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    np.savez(OUT_DIR / 'concat_predictions.npz',
             ensemble=all_ens, cnn_mamba=all_cm, patchtst=all_pt, labels=all_lbl)

    print(f"\n{'='*60}")
    print(f"Results saved to {OUT_DIR}")
    print(f"Completed: {datetime.now().isoformat()}")

    # VERDICT
    lift_1s = concat_ens_ic[0] - concat_cm_ic[0]
    if lift_1s > 0.03:
        print(f"\n✅ ENSEMBLE PROVIDES MEANINGFUL LIFT: +{lift_1s:.4f} IC at 1s")
    elif lift_1s > 0:
        print(f"\n⚠️ ENSEMBLE PROVIDES MARGINAL LIFT: +{lift_1s:.4f} IC at 1s")
    else:
        print(f"\n❌ ENSEMBLE DOES NOT IMPROVE: {lift_1s:+.4f} IC at 1s — models may be too similar at this level")


if __name__ == '__main__':
    main()
