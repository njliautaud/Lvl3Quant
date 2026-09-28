#!/usr/bin/env python3
"""
Confluence-Aware Meta-Model v2
===============================
Trains MLP (256→128→64→1) to predict realized FIFO-proxy net P&L
from microstructure features at signal time.

TARGET: For top 3% short signals (5s hold, 2-tick stop):
  - If stop hit (label_1s >= 2.0): target = -(STOP_TICKS + SLIPPAGE) - COMMISSION = -3.376
  - If passive-passive: target = (-label_5s) - COMMISSION
  This gives us realized P&L under passive-passive assumption.
  Meta-model learns WHICH trades actually work.

FEATURES (29-dim):
  - 25 microstructure features from MBO events (at signal time)
  - pred_1s, pred_5s, pred_10s (model confidence across horizons)
  - signal_rank (percentile within day, normalized 0-1)

Walk-forward: 10-date train, 1-date OOT test (sliding window).
GPU: RTX 3070 8GB compatible (small model, batch processing).
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# === CONSTANTS ===
COMMISSION_RT_TICKS = 0.376
STOP_TICKS = 2.0
STOP_SLIPPAGE = 1.0
STOP_LOSS_TOTAL = STOP_TICKS + STOP_SLIPPAGE  # 3 ticks
SHORT_PERCENTILE = 3  # top 3%

# Auto-detect platform for paths
if sys.platform == 'win32':
    _BASE = Path(r'C:\Users\claude\Lvl3Quant')
else:
    _BASE = Path('/home/jupiter/Lvl3Quant')

PRED_DIR = _BASE / 'output' / 'cnn_mamba_v2_bulk_oot_v2'
MBO_DIR = _BASE / 'data' / 'processed' / 'mbo_events_smart_v3'
OUT_DIR = _BASE / 'output' / 'confluence_meta_v2'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Training params
TRAIN_WINDOW = 10  # dates
BATCH_SIZE = 4096
LR = 1e-3
EPOCHS = 30
HIDDEN_DIMS = [256, 128, 64]
DROPOUT = 0.2
WEIGHT_DECAY = 1e-4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MetaMLP(nn.Module):
    """MLP predicting realized net P&L from microstructure features."""
    def __init__(self, input_dim, hidden_dims, dropout=0.2):
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
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_day(pred_file):
    """Load one OOT day: predictions aligned with MBO features + labels."""
    pred_data = np.load(pred_file, allow_pickle=True)
    date_str = str(pred_data['date'])
    preds = pred_data['predictions']  # (n_windows, 3) -> [1s, 5s, 10s]
    n_windows = int(pred_data['n_windows'])
    window_size = int(pred_data['window_size'])
    stride = int(pred_data['stride'])

    mbo_file = MBO_DIR / f'{date_str}_mbo_events.npz'
    if not mbo_file.exists():
        return None

    mbo = np.load(mbo_file, allow_pickle=True)
    events = mbo['events']  # (N, 25)
    l1s = mbo['labels_1s']
    l5s = mbo['labels_5s']

    indices = np.arange(n_windows) * stride + (window_size - 1)
    max_idx = min(len(events), len(l1s), len(l5s)) - 1
    valid = indices <= max_idx
    indices = indices[valid]
    preds = preds[:len(indices)]

    # Extract aligned features and labels
    feat = events[indices]  # (n, 25)
    label_1s = l1s[indices]
    label_5s = l5s[indices]

    # Remove NaN
    valid_mask = ~(np.isnan(label_1s) | np.isnan(label_5s) | np.any(np.isnan(feat), axis=1))
    feat = feat[valid_mask]
    label_1s = label_1s[valid_mask]
    label_5s = label_5s[valid_mask]
    preds = preds[valid_mask]

    if len(feat) == 0:
        return None

    # Compute target: realized P&L for short trades (passive-passive)
    # Short P&L = -label_5s (positive when price drops)
    short_pnl_gross = -label_5s

    # Stop logic: if label_1s >= 2.0, stopped out
    stop_hit = label_1s >= STOP_TICKS
    target = np.where(
        stop_hit,
        -(STOP_LOSS_TOTAL + COMMISSION_RT_TICKS),  # -3.376
        short_pnl_gross - COMMISSION_RT_TICKS       # realized - commission
    )

    # Build feature matrix: 25 MBO + 3 prediction + 1 rank = 29
    pred_1s = preds[:, 0]
    pred_5s = preds[:, 1]
    pred_10s = preds[:, 2]

    # Signal rank (percentile of 5s prediction within day, 0=most negative/strongest short)
    ranks = np.argsort(np.argsort(pred_5s)).astype(np.float32) / len(pred_5s)

    features = np.column_stack([
        feat,           # 25 MBO features
        pred_1s,        # 1s prediction
        pred_5s,        # 5s prediction
        pred_10s,       # 10s prediction
        ranks,          # signal rank within day
    ])  # (n, 29)

    # Select top 3% shorts only (most negative pred_5s)
    threshold = np.percentile(pred_5s, SHORT_PERCENTILE)
    short_mask = pred_5s <= threshold

    return {
        'date': date_str,
        'features': features[short_mask].astype(np.float32),
        'target': target[short_mask].astype(np.float32),
        'features_all': features.astype(np.float32),
        'target_all': target.astype(np.float32),
        'pred_5s': pred_5s,
        'short_mask': short_mask,
    }


def normalize_features(train_feats, test_feats):
    """Z-score normalize using train stats."""
    mean = train_feats.mean(axis=0)
    std = train_feats.std(axis=0) + 1e-8
    return (train_feats - mean) / std, (test_feats - mean) / std, mean, std


def train_fold(train_days, test_day, fold_idx):
    """Train one WF fold, return OOT predictions."""
    # Combine training data
    train_X = np.concatenate([d['features'] for d in train_days])
    train_y = np.concatenate([d['target'] for d in train_days])

    test_X = test_day['features']
    test_y = test_day['target']

    if len(train_X) < 100 or len(test_X) < 10:
        return None

    # Normalize
    train_X_norm, test_X_norm, _, _ = normalize_features(train_X, test_X)

    # Create datasets
    train_ds = TensorDataset(
        torch.from_numpy(train_X_norm),
        torch.from_numpy(train_y),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)

    # Model
    input_dim = train_X_norm.shape[1]
    model = MetaMLP(input_dim, HIDDEN_DIMS, DROPOUT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.MSELoss()

    # Train
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        n_batches = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()

    # Evaluate on OOT
    model.eval()
    with torch.no_grad():
        test_tensor = torch.from_numpy(test_X_norm).to(device)
        oot_preds = model(test_tensor).cpu().numpy()

    # Correlation between predicted P&L and actual P&L
    if len(oot_preds) > 1 and np.std(oot_preds) > 1e-8:
        corr = np.corrcoef(oot_preds, test_y)[0, 1]
    else:
        corr = 0.0

    # Test as filter: top 50% of predicted P&L vs bottom 50%
    if len(oot_preds) >= 20:
        median_pred = np.median(oot_preds)
        top_half = test_y[oot_preds >= median_pred]
        bot_half = test_y[oot_preds < median_pred]
        top_mean = top_half.mean() if len(top_half) > 0 else 0
        bot_mean = bot_half.mean() if len(bot_half) > 0 else 0
        separation = top_mean - bot_mean
    else:
        top_mean = bot_mean = separation = 0

    return {
        'fold': fold_idx,
        'date': test_day['date'],
        'n_train': len(train_X),
        'n_test': len(test_X),
        'corr': float(corr),
        'top_half_mean_pnl': float(top_mean),
        'bot_half_mean_pnl': float(bot_mean),
        'separation': float(separation),
        'oot_preds': oot_preds,
        'oot_actuals': test_y,
        'train_loss': total_loss / max(n_batches, 1),
    }


def main():
    print(f"Device: {device}", flush=True)
    print(f"Predictions: {PRED_DIR}", flush=True)
    print(f"MBO data: {MBO_DIR}", flush=True)
    print(f"Output: {OUT_DIR}", flush=True)

    # Load all prediction files
    pred_files = sorted([
        f for f in PRED_DIR.glob('*_predictions.npz')
        if '_stale_' not in str(f)
    ])
    print(f"\nFound {len(pred_files)} prediction files", flush=True)

    # Load all days
    all_days = []
    for i, pf in enumerate(pred_files):
        if i % 10 == 0:
            print(f"  Loading day {i+1}/{len(pred_files)}...", flush=True)
        day = load_day(pf)
        if day is not None:
            all_days.append(day)
    print(f"Loaded {len(all_days)} days with valid data", flush=True)
    print(f"Total top-3% short trades: {sum(len(d['features']) for d in all_days)}", flush=True)

    # Walk-forward training
    results = []
    all_oot_preds = []
    all_oot_actuals = []

    n_folds = len(all_days) - TRAIN_WINDOW
    print(f"\n{'='*80}")
    print(f"WALK-FORWARD: {n_folds} folds (train={TRAIN_WINDOW} days, test=1 day)")
    print(f"{'='*80}")
    print(f"{'Fold':>5} {'Date':>10} {'N_test':>7} {'Corr':>7} {'Top50%':>8} {'Bot50%':>8} {'Sep':>7}", flush=True)
    print(f"{'-'*60}")

    for fold_idx in range(n_folds):
        train_days = all_days[fold_idx:fold_idx + TRAIN_WINDOW]
        test_day = all_days[fold_idx + TRAIN_WINDOW]

        result = train_fold(train_days, test_day, fold_idx)
        if result is None:
            continue

        results.append(result)
        all_oot_preds.append(result['oot_preds'])
        all_oot_actuals.append(result['oot_actuals'])

        print(f"{fold_idx:>5} {result['date']:>10} {result['n_test']:>7} "
              f"{result['corr']:>+6.3f} {result['top_half_mean_pnl']:>+7.3f} "
              f"{result['bot_half_mean_pnl']:>+7.3f} {result['separation']:>+6.3f}", flush=True)

    if not results:
        print("No valid folds!")
        return

    # Concat metrics
    concat_preds = np.concatenate(all_oot_preds)
    concat_actuals = np.concatenate(all_oot_actuals)
    concat_corr = np.corrcoef(concat_preds, concat_actuals)[0, 1] if len(concat_preds) > 1 else 0

    per_fold_corrs = [r['corr'] for r in results]
    per_fold_seps = [r['separation'] for r in results]

    print(f"\n{'='*80}")
    print(f"AGGREGATE RESULTS")
    print(f"{'='*80}")
    print(f"Concat correlation: {concat_corr:+.4f}")
    print(f"Mean per-fold corr: {np.mean(per_fold_corrs):+.4f} ± {np.std(per_fold_corrs):.4f}")
    print(f"Mean separation:    {np.mean(per_fold_seps):+.4f} ticks")
    print(f"Positive-corr folds: {sum(1 for c in per_fold_corrs if c > 0)}/{len(per_fold_corrs)}")
    print(f"Total OOT trades:   {len(concat_preds)}")

    # Evaluate as trade filter: use meta-model to filter top 50% / top 30% / top 20%
    print(f"\n{'='*80}")
    print(f"FILTER ANALYSIS: Use meta-model score to select trades")
    print(f"{'='*80}")
    print(f"{'Filter':>12} {'N_trades':>9} {'MeanPnL':>8} {'WR%':>6} {'PF':>6} {'Improvement':>12}")
    print(f"{'-'*60}")

    baseline_pnl = concat_actuals.mean()
    baseline_wr = (concat_actuals > 0).mean() * 100

    for pct_label, pct in [('All (base)', 100), ('Top 70%', 70), ('Top 50%', 50),
                            ('Top 30%', 30), ('Top 20%', 20), ('Top 10%', 10)]:
        if pct == 100:
            sel = concat_actuals
        else:
            thresh = np.percentile(concat_preds, 100 - pct)
            sel = concat_actuals[concat_preds >= thresh]

        if len(sel) == 0:
            continue

        mean_pnl = sel.mean()
        wr = (sel > 0).mean() * 100
        wins = sel[sel > 0].sum() if (sel > 0).any() else 0
        losses = abs(sel[sel < 0].sum()) if (sel < 0).any() else 1
        pf = wins / losses if losses > 0 else float('inf')
        improvement = mean_pnl - baseline_pnl

        print(f"{pct_label:>12} {len(sel):>9} {mean_pnl:>+7.3f} {wr:>5.1f} {pf:>5.2f} "
              f"{improvement:>+11.3f}")

    # Save results
    summary = {
        'concat_corr': float(concat_corr),
        'mean_per_fold_corr': float(np.mean(per_fold_corrs)),
        'std_per_fold_corr': float(np.std(per_fold_corrs)),
        'mean_separation': float(np.mean(per_fold_seps)),
        'positive_corr_folds': int(sum(1 for c in per_fold_corrs if c > 0)),
        'total_folds': len(results),
        'total_oot_trades': int(len(concat_preds)),
        'baseline_mean_pnl': float(baseline_pnl),
        'model_arch': f'MLP {HIDDEN_DIMS}',
        'features': '25_mbo + pred_1s + pred_5s + pred_10s + rank = 29',
        'target': 'realized_net_pnl_passive_passive',
        'per_fold': [{k: v for k, v in r.items() if k not in ('oot_preds', 'oot_actuals')} for r in results],
    }
    with open(OUT_DIR / 'results.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Save predictions for downstream use
    np.savez_compressed(
        OUT_DIR / 'oot_predictions.npz',
        predictions=concat_preds,
        actuals=concat_actuals,
    )

    print(f"\nResults saved to {OUT_DIR}")
    print(f"\nVERDICT: {'PASS' if concat_corr > 0.05 else 'WEAK' if concat_corr > 0 else 'FAIL'} "
          f"(concat_corr={concat_corr:+.4f}, threshold=0.05)")


if __name__ == '__main__':
    main()
