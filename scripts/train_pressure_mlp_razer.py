#!/usr/bin/env python3
"""
Train MLP on Razer GPU to predict pressure_score from smart_v3 MBO features.
Per HC #504 R2d: MLP heads whitelisted for Razer.
Per HC #509: Smooth pressure prediction is the target direction.

Architecture: MLP 256→128→64 (per HC #469 R5 e1 spec).
Walk-forward sliding: 20-day train, 1-day OOT.
"""

import os
import sys
import json
import time
import logging
import argparse
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
log = logging.getLogger(__name__)


class PressureMLP(nn.Module):
    """MLP 256→128→64 for pressure prediction."""
    def __init__(self, input_dim=25, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_day(event_dir: Path, pressure_dir: Path, date_str: str, target_key: str = 'pressure_score'):
    """Load features and pressure labels for a single day."""
    event_file = event_dir / f"{date_str}_mbo_events.npz"
    pressure_file = pressure_dir / f"{date_str}_pressure.npz"

    if not event_file.exists() or not pressure_file.exists():
        return None

    try:
        ev = np.load(event_file)
        pr = np.load(pressure_file)
    except Exception as e:
        log.warning(f"Failed to load {date_str}: {e}")
        return None

    features = ev['events']  # (N, 25)

    if target_key not in pr:
        log.warning(f"No {target_key} in pressure file for {date_str}")
        return None

    labels = pr[target_key]

    if features.shape[0] != labels.shape[0]:
        log.warning(f"Shape mismatch {date_str}: {features.shape[0]} vs {labels.shape[0]}")
        return None

    valid = np.isfinite(labels) & np.all(np.isfinite(features), axis=1)
    if valid.sum() < 1000:
        return None

    # Subsample to limit memory (500K per day max)
    max_per_day = 500_000
    valid_idx = np.where(valid)[0]
    if len(valid_idx) > max_per_day:
        rng = np.random.RandomState(42)
        valid_idx = rng.choice(valid_idx, max_per_day, replace=False)
        valid_idx.sort()

    result = {
        'date': date_str,
        'features': features[valid_idx].astype(np.float32).copy(),
        'labels': labels[valid_idx].astype(np.float32).copy(),
        'n_samples': len(valid_idx),
    }
    del features, labels
    return result


def train_fold(train_days, test_day, device, epochs=15, batch_size=4096, lr=1e-3):
    """Train one WF fold."""
    X_train = np.concatenate([d['features'] for d in train_days])
    y_train = np.concatenate([d['labels'] for d in train_days])

    X_test = test_day['features']
    y_test = test_day['labels']

    # Normalize features (per-fold to avoid leakage)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    # Normalize labels
    y_mean = y_train.mean()
    y_std = y_train.std() + 1e-8
    y_train_norm = (y_train - y_mean) / y_std

    # Subsample if too large
    if X_train.shape[0] > 3_000_000:
        idx = np.random.choice(X_train.shape[0], 3_000_000, replace=False)
        X_train = X_train[idx]
        y_train_norm = y_train_norm[idx]

    train_ds = TensorDataset(
        torch.from_numpy(X_train).to(device),
        torch.from_numpy(y_train_norm).to(device),
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    model = PressureMLP(input_dim=X_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    best_loss = float('inf')
    patience = 3
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0
        for xb, yb in train_dl:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    # Predict on test
    model.eval()
    with torch.no_grad():
        X_test_t = torch.from_numpy(X_test).to(device)
        # Predict in chunks to avoid OOM
        preds_list = []
        for i in range(0, len(X_test_t), batch_size * 4):
            chunk = X_test_t[i:i + batch_size * 4]
            p = model(chunk).cpu().numpy()
            preds_list.append(p)
        preds_norm = np.concatenate(preds_list)

    # Denormalize predictions
    preds = preds_norm * y_std + y_mean

    ic, _ = spearmanr(preds, y_test)
    dir_acc = float(np.mean(np.sign(preds) == np.sign(y_test)))

    return {
        'fold_date': test_day['date'],
        'n_train': X_train.shape[0],
        'n_test': X_test.shape[0],
        'ic': float(ic) if not np.isnan(ic) else 0.0,
        'dir_acc': dir_acc,
        'train_loss': best_loss,
        'epochs_run': epoch + 1,
        'predictions': preds,
        'actuals': y_test,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--pressure-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--target', type=str, default='pressure_score')
    parser.add_argument('--train-window', type=int, default=20)
    parser.add_argument('--year-min', type=int, default=2026)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    event_dir = Path(args.data_dir)
    pressure_dir = Path(args.pressure_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Device: {device}")

    # Find dates
    event_dates = {f.stem.split('_')[0] for f in event_dir.glob('*_mbo_events.npz')}
    pressure_dates = {f.stem.split('_')[0] for f in pressure_dir.glob('*_pressure.npz')}
    common_dates = sorted(event_dates & pressure_dates)

    if args.year_min:
        common_dates = [d for d in common_dates if int(d[:4]) >= args.year_min]

    log.info(f"Found {len(common_dates)} common dates")

    # Load data
    log.info("Loading data...")
    all_days = []
    for date_str in common_dates:
        day = load_day(event_dir, pressure_dir, date_str, args.target)
        if day is not None:
            all_days.append(day)

    total_samples = sum(d['n_samples'] for d in all_days)
    log.info(f"Loaded {len(all_days)} days, {total_samples:,} samples")

    if len(all_days) < args.train_window + 1:
        log.error(f"Not enough days ({len(all_days)})")
        return

    # Walk-forward
    tw = args.train_window
    n_folds = len(all_days) - tw
    log.info(f"Running {n_folds} WF folds (window={tw})")

    fold_results = []
    all_preds = []
    all_actuals = []

    for i in range(n_folds):
        train_days = all_days[i:i + tw]
        test_day = all_days[i + tw]

        t0 = time.time()
        result = train_fold(train_days, test_day, device, args.epochs, args.batch_size, args.lr)
        elapsed = time.time() - t0

        fold_results.append({k: v for k, v in result.items() if k not in ('predictions', 'actuals')})
        all_preds.append(result['predictions'])
        all_actuals.append(result['actuals'])

        log.info(f"Fold {i+1}/{n_folds} [{result['fold_date']}]: "
                 f"IC={result['ic']:.4f}, DirAcc={result['dir_acc']:.3f}, "
                 f"loss={result['train_loss']:.6f}, {elapsed:.1f}s")

    # Concat metrics
    concat_preds = np.concatenate(all_preds)
    concat_actuals = np.concatenate(all_actuals)
    concat_ic, _ = spearmanr(concat_preds, concat_actuals)
    concat_dir_acc = float(np.mean(np.sign(concat_preds) == np.sign(concat_actuals)))
    per_fold_ics = [r['ic'] for r in fold_results]

    summary = {
        'target': args.target,
        'model': 'MLP_256_128_64',
        'n_folds': n_folds,
        'n_oos_samples': len(concat_preds),
        'concat_spearman_ic': float(concat_ic),
        'concat_dir_acc': concat_dir_acc,
        'per_fold_mean_ic': float(np.mean(per_fold_ics)),
        'per_fold_median_ic': float(np.median(per_fold_ics)),
        'per_fold_min_ic': float(np.min(per_fold_ics)),
        'per_fold_max_ic': float(np.max(per_fold_ics)),
        'per_fold_std_ic': float(np.std(per_fold_ics)),
        'device': str(device),
        'fold_details': fold_results,
    }

    log.info(f"\n{'='*60}")
    log.info(f"RESULTS: {args.target} MLP 256→128→64")
    log.info(f"  Concat Spearman IC: {concat_ic:.4f}")
    log.info(f"  Concat Dir Acc: {concat_dir_acc:.3f}")
    log.info(f"  Per-fold mean IC: {np.mean(per_fold_ics):.4f} (±{np.std(per_fold_ics):.4f})")
    log.info(f"  Per-fold range: [{np.min(per_fold_ics):.4f}, {np.max(per_fold_ics):.4f}]")
    log.info(f"  OOS samples: {len(concat_preds):,}")
    log.info(f"{'='*60}")

    # Save
    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    preds_path = output_dir / "predictions.npz"
    np.savez_compressed(preds_path, predictions=concat_preds, actuals=concat_actuals)

    log.info(f"Saved to {output_dir}")
    log.info("Done.")


if __name__ == "__main__":
    main()
