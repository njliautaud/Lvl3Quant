#!/usr/bin/env python3
"""
Exit Fill Meta-Predictor v1
============================
Predicts passive EXIT fill probability using ENRICHED features:
  - 25 MBO microstructure features
  - pred_1s, pred_5s, pred_10s (model confidence)
  - signal_rank (day percentile)
  - meta_pnl_score (from meta-model v2 — predicted realized P&L)

The key insight: the meta-model's P&L prediction captures information about
trade quality that should correlate with fill probability. High-quality trades
(where the model is more certain about the short direction) are more likely
to see the 1-tick retrace needed for passive exit fill.

TARGET: Binary — did price drop ≥1 tick within 5s? (passive exit would fill)
  Short fill = min(labels_1s, labels_5s) <= -1.0

ARCHITECTURE: MLP 128→64→32 with BN+GELU+Dropout
Walk-forward: 10-date train, 1-date OOT

This is a TWO-STAGE model:
  Stage 1: Meta-model predicts P&L (already trained, we use its predictions)
  Stage 2: Fill predictor uses meta-model score + raw features to predict fills
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

if sys.platform == 'win32':
    _BASE = Path(r'C:\Users\claude\Lvl3Quant')
else:
    _BASE = Path('/home/jupiter/Lvl3Quant')

PRED_DIR = _BASE / 'output' / 'cnn_mamba_v2_bulk_oot_v2'
MBO_DIR = _BASE / 'data' / 'processed' / 'mbo_events_smart_v3'
OUT_DIR = _BASE / 'output' / 'exit_fill_meta_v1'
OUT_DIR.mkdir(parents=True, exist_ok=True)

COMMISSION_RT = 0.376
STOP_TICKS = 2.0
STOP_SLIPPAGE = 1.0
SHORT_PERCENTILE = 3
TRAIN_WINDOW = 10
BATCH_SIZE = 4096
LR = 1e-3
EPOCHS = 25
WEIGHT_DECAY = 1e-4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FillMetaMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout=0.2):
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


class PnLMetaModel(nn.Module):
    """Frozen meta-model for generating P&L score feature."""
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.2):
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
    """Load one day with all needed data."""
    pred_data = np.load(pred_file, allow_pickle=True)
    date_str = str(pred_data['date'])
    preds = pred_data['predictions']
    n_windows = int(pred_data['n_windows'])
    window_size = int(pred_data['window_size'])
    stride = int(pred_data['stride'])

    mbo_file = MBO_DIR / f'{date_str}_mbo_events.npz'
    if not mbo_file.exists():
        return None

    mbo = np.load(mbo_file, allow_pickle=True)
    events = mbo['events']
    l1s = mbo['labels_1s']
    l5s = mbo['labels_5s']

    indices = np.arange(n_windows) * stride + (window_size - 1)
    max_idx = min(len(events), len(l1s), len(l5s)) - 1
    valid = indices <= max_idx
    indices = indices[valid]
    preds = preds[:len(indices)]

    feat = events[indices]
    label_1s = l1s[indices]
    label_5s = l5s[indices]

    valid_mask = ~(np.isnan(label_1s) | np.isnan(label_5s) | np.any(np.isnan(feat), axis=1))
    feat = feat[valid_mask]
    label_1s = label_1s[valid_mask]
    label_5s = label_5s[valid_mask]
    preds = preds[valid_mask]

    if len(feat) == 0:
        return None

    pred_1s, pred_5s, pred_10s = preds[:, 0], preds[:, 1], preds[:, 2]
    ranks = np.argsort(np.argsort(pred_5s)).astype(np.float32) / len(pred_5s)

    # Base features (29 dim) for both meta-model and fill predictor
    base_features = np.column_stack([feat, pred_1s, pred_5s, pred_10s, ranks]).astype(np.float32)

    # P&L target (for training meta-model within each fold)
    short_pnl = -label_5s
    stop_hit = label_1s >= STOP_TICKS
    pnl_target = np.where(
        stop_hit,
        -(STOP_TICKS + STOP_SLIPPAGE + COMMISSION_RT),
        short_pnl - COMMISSION_RT,
    ).astype(np.float32)

    # Fill target
    min_label = np.minimum(label_1s, label_5s)
    fill_target = (min_label <= -1.0).astype(np.float32)

    # Short selection
    threshold = np.percentile(pred_5s, SHORT_PERCENTILE)
    short_mask = pred_5s <= threshold

    return {
        'date': date_str,
        'base_features': base_features[short_mask],
        'pnl_target': pnl_target[short_mask],
        'fill_target': fill_target[short_mask],
        'actual_pnl': pnl_target[short_mask],
    }


def train_fold(train_days, test_day, fold_idx):
    """Two-stage training: meta-model → fill predictor with meta-score feature."""
    train_X = np.concatenate([d['base_features'] for d in train_days])
    train_pnl = np.concatenate([d['pnl_target'] for d in train_days])
    train_fill = np.concatenate([d['fill_target'] for d in train_days])

    test_X = test_day['base_features']
    test_fill = test_day['fill_target']
    test_pnl = test_day['actual_pnl']

    if len(train_X) < 100 or len(test_X) < 10:
        return None

    # Normalize base features
    mean = train_X.mean(axis=0)
    std = train_X.std(axis=0) + 1e-8
    train_X_n = (train_X - mean) / std
    test_X_n = (test_X - mean) / std

    input_dim = train_X_n.shape[1]

    # === STAGE 1: Train meta-model (P&L prediction) ===
    meta_model = PnLMetaModel(input_dim).to(device)
    optimizer = torch.optim.AdamW(meta_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    mse = nn.MSELoss()

    train_ds = TensorDataset(torch.from_numpy(train_X_n), torch.from_numpy(train_pnl))
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    meta_model.train()
    for epoch in range(15):
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = mse(meta_model(X_b), y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(meta_model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

    # Generate meta-scores for train and test
    meta_model.eval()
    with torch.no_grad():
        train_meta_score = meta_model(torch.from_numpy(train_X_n).to(device)).cpu().numpy()
        test_meta_score = meta_model(torch.from_numpy(test_X_n).to(device)).cpu().numpy()

    # === STAGE 2: Train fill predictor with meta-score as additional feature ===
    # Extended features: base_features (29) + meta_score (1) = 30
    train_X_ext = np.column_stack([train_X_n, train_meta_score]).astype(np.float32)
    test_X_ext = np.column_stack([test_X_n, test_meta_score]).astype(np.float32)

    pos_rate = train_fill.mean()
    if pos_rate == 0 or pos_rate == 1:
        return None
    pos_weight = torch.tensor([(1 - pos_rate) / pos_rate]).to(device)

    fill_model = FillMetaMLP(train_X_ext.shape[1]).to(device)
    optimizer2 = torch.optim.AdamW(fill_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=EPOCHS)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    fill_ds = TensorDataset(torch.from_numpy(train_X_ext), torch.from_numpy(train_fill))
    fill_loader = DataLoader(fill_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    fill_model.train()
    for epoch in range(EPOCHS):
        for X_b, y_b in fill_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer2.zero_grad()
            loss = bce(fill_model(X_b), y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fill_model.parameters(), 1.0)
            optimizer2.step()
        scheduler2.step()

    # Evaluate
    fill_model.eval()
    with torch.no_grad():
        logits = fill_model(torch.from_numpy(test_X_ext).to(device)).cpu().numpy()
        probs = 1 / (1 + np.exp(-logits))

    try:
        auc = roc_auc_score(test_fill, probs)
    except ValueError:
        auc = 0.5

    # Also evaluate baseline fill predictor (without meta-score, 29 features)
    fill_model_base = FillMetaMLP(input_dim).to(device)
    opt_base = torch.optim.AdamW(fill_model_base.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched_base = torch.optim.lr_scheduler.CosineAnnealingLR(opt_base, T_max=EPOCHS)

    base_ds = TensorDataset(torch.from_numpy(train_X_n), torch.from_numpy(train_fill))
    base_loader = DataLoader(base_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    fill_model_base.train()
    for epoch in range(EPOCHS):
        for X_b, y_b in base_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            opt_base.zero_grad()
            loss = bce(fill_model_base(X_b), y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fill_model_base.parameters(), 1.0)
            opt_base.step()
        sched_base.step()

    fill_model_base.eval()
    with torch.no_grad():
        base_logits = fill_model_base(torch.from_numpy(test_X_n).to(device)).cpu().numpy()
        base_probs = 1 / (1 + np.exp(-base_logits))

    try:
        base_auc = roc_auc_score(test_fill, base_probs)
    except ValueError:
        base_auc = 0.5

    # Fill rate separation
    high_conf = probs >= np.percentile(probs, 70)
    low_conf = probs < np.percentile(probs, 30)
    high_fr = test_fill[high_conf].mean() if high_conf.sum() > 0 else 0
    low_fr = test_fill[low_conf].mean() if low_conf.sum() > 0 else 0

    # Combined filter: meta-score top 50% AND fill-probability top 50%
    meta_top50 = test_meta_score >= np.median(test_meta_score)
    fill_top50 = probs >= np.median(probs)
    combined = meta_top50 & fill_top50
    combined_pnl = test_pnl[combined].mean() if combined.sum() > 0 else 0
    all_pnl = test_pnl.mean()

    return {
        'fold': fold_idx,
        'date': test_day['date'],
        'n_test': len(test_X),
        'auc_with_meta': float(auc),
        'auc_without_meta': float(base_auc),
        'auc_lift': float(auc - base_auc),
        'high_conf_fill_rate': float(high_fr),
        'low_conf_fill_rate': float(low_fr),
        'fill_separation': float(high_fr - low_fr),
        'base_fill_rate': float(test_fill.mean()),
        'combined_filter_pnl': float(combined_pnl),
        'all_pnl': float(all_pnl),
        'combined_lift': float(combined_pnl - all_pnl),
        'n_combined': int(combined.sum()),
    }


def main():
    print(f"Device: {device}", flush=True)

    pred_files = sorted([f for f in PRED_DIR.glob('*_predictions.npz') if '_stale_' not in str(f)])
    print(f"Found {len(pred_files)} prediction files", flush=True)

    all_days = []
    for i, pf in enumerate(pred_files):
        if i % 10 == 0:
            print(f"  Loading day {i+1}/{len(pred_files)}...", flush=True)
        day = load_day(pf)
        if day is not None:
            all_days.append(day)
    print(f"Loaded {len(all_days)} days", flush=True)

    n_folds = len(all_days) - TRAIN_WINDOW
    print(f"\nWalk-forward: {n_folds} folds (2-stage: meta-model + fill predictor)")
    print(f"{'Fold':>5} {'Date':>10} {'N':>6} {'AUC+meta':>9} {'AUC-meta':>9} {'Lift':>6} "
          f"{'HiFR':>6} {'LoFR':>6} {'CombPnL':>8}", flush=True)
    print(f"{'-'*75}", flush=True)

    results = []
    for fold_idx in range(n_folds):
        train_days = all_days[fold_idx:fold_idx + TRAIN_WINDOW]
        test_day = all_days[fold_idx + TRAIN_WINDOW]
        result = train_fold(train_days, test_day, fold_idx)
        if result is None:
            continue
        results.append(result)
        r = result
        print(f"{r['fold']:>5} {r['date']:>10} {r['n_test']:>6} {r['auc_with_meta']:>8.3f} "
              f"{r['auc_without_meta']:>8.3f} {r['auc_lift']:>+5.3f} "
              f"{r['high_conf_fill_rate']:>5.1%} {r['low_conf_fill_rate']:>5.1%} "
              f"{r['combined_filter_pnl']:>+7.3f}", flush=True)

    if not results:
        print("No valid folds!")
        return

    # Aggregate
    print(f"\n{'='*75}")
    print(f"AGGREGATE RESULTS ({len(results)} folds)")
    print(f"{'='*75}")
    print(f"Mean AUC with meta-score:    {np.mean([r['auc_with_meta'] for r in results]):.4f}")
    print(f"Mean AUC without meta-score: {np.mean([r['auc_without_meta'] for r in results]):.4f}")
    print(f"Mean AUC lift from meta:     {np.mean([r['auc_lift'] for r in results]):+.4f}")
    print(f"Mean fill separation:        {np.mean([r['fill_separation'] for r in results]):.1%}")
    print(f"Mean combined filter P&L:    {np.mean([r['combined_filter_pnl'] for r in results]):+.3f}")
    print(f"Mean all P&L:                {np.mean([r['all_pnl'] for r in results]):+.3f}")
    print(f"Combined filter lift:        {np.mean([r['combined_lift'] for r in results]):+.3f}")

    # Save
    summary = {
        'mean_auc_with_meta': float(np.mean([r['auc_with_meta'] for r in results])),
        'mean_auc_without_meta': float(np.mean([r['auc_without_meta'] for r in results])),
        'mean_auc_lift': float(np.mean([r['auc_lift'] for r in results])),
        'mean_fill_separation': float(np.mean([r['fill_separation'] for r in results])),
        'mean_combined_pnl': float(np.mean([r['combined_filter_pnl'] for r in results])),
        'n_folds': len(results),
        'per_fold': results,
    }
    with open(OUT_DIR / 'results.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {OUT_DIR}", flush=True)
    auc_lift = np.mean([r['auc_lift'] for r in results])
    print(f"\nVERDICT: {'PASS' if auc_lift > 0.01 else 'MARGINAL' if auc_lift > 0 else 'FAIL'} "
          f"(AUC lift from meta = {auc_lift:+.4f})")


if __name__ == '__main__':
    main()
