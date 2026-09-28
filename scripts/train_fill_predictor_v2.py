#!/usr/bin/env python3
"""
Fill Predictor v2
==================
Trains 1D-CNN to predict whether a passive limit order will get filled
within the hold window (proxy: does price retrace ≥1 tick within 5s?).

This addresses the CORE execution problem: the strategy only works with
passive fills on both sides. If we can predict fill probability, we can
filter to trades where we're confident of getting filled.

TARGET (binary classification):
  For short entries (top 3% signals):
  - label_5s <= -1.0 tick → price dropped 1+ tick → passive EXIT would fill (1)
  - label_5s > -1.0 tick → price didn't drop enough → no fill (0)

  For entry fills, we approximate:
  - label_1s <= 0 → price went in our direction within 1s → entry likely filled (1)

FEATURES (29-dim): Same as confluence meta v2
  - 25 MBO microstructure features
  - pred_1s, pred_5s, pred_10s (confidence)
  - signal_rank (percentile)

Walk-forward: 10-date sliding window.
GPU: RTX 3070 8GB compatible.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

# === CONSTANTS ===
SHORT_PERCENTILE = 3
RETRACE_THRESHOLD = 1.0  # 1 tick retrace = passive fill

if sys.platform == 'win32':
    _BASE = Path(r'C:\Users\claude\Lvl3Quant')
else:
    _BASE = Path('/home/jupiter/Lvl3Quant')

PRED_DIR = _BASE / 'output' / 'cnn_mamba_v2_bulk_oot_v2'
MBO_DIR = _BASE / 'data' / 'processed' / 'mbo_events_smart_v3'
OUT_DIR = _BASE / 'output' / 'fill_predictor_v2'
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_WINDOW = 10
BATCH_SIZE = 4096
LR = 1e-3
EPOCHS = 30
WEIGHT_DECAY = 1e-4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FillPredictor1DCNN(nn.Module):
    """1D CNN for fill prediction from microstructure features."""
    def __init__(self, input_dim, hidden=64):
        super().__init__()
        # Treat features as 1D sequence of length input_dim, 1 channel
        self.conv1 = nn.Conv1d(1, hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (batch, features) -> (batch, 1, features)
        x = x.unsqueeze(1)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)  # (batch, hidden)
        return self.fc(x).squeeze(-1)


class FillPredictorMLP(nn.Module):
    """MLP baseline for fill prediction."""
    def __init__(self, input_dim, hidden_dims=[128, 64, 32]):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(0.2),
            ])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_day(pred_file):
    """Load one day with fill labels."""
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
    l10s = mbo['labels_10s']

    indices = np.arange(n_windows) * stride + (window_size - 1)
    max_idx = min(len(events), len(l1s), len(l5s), len(l10s)) - 1
    valid = indices <= max_idx
    indices = indices[valid]
    preds = preds[:len(indices)]

    feat = events[indices]
    label_1s = l1s[indices]
    label_5s = l5s[indices]
    label_10s = l10s[indices]

    valid_mask = ~(np.isnan(label_1s) | np.isnan(label_5s) | np.isnan(label_10s) | np.any(np.isnan(feat), axis=1))
    feat = feat[valid_mask]
    label_1s = label_1s[valid_mask]
    label_5s = label_5s[valid_mask]
    label_10s = label_10s[valid_mask]
    preds = preds[valid_mask]

    if len(feat) == 0:
        return None

    # Build features
    pred_1s = preds[:, 0]
    pred_5s = preds[:, 1]
    pred_10s = preds[:, 2]
    ranks = np.argsort(np.argsort(pred_5s)).astype(np.float32) / len(pred_5s)

    features = np.column_stack([feat, pred_1s, pred_5s, pred_10s, ranks]).astype(np.float32)

    # Select top 3% shorts
    threshold = np.percentile(pred_5s, SHORT_PERCENTILE)
    short_mask = pred_5s <= threshold

    # Fill target: does price retrace ≥1 tick within 5s? (for short: label_5s <= -1.0)
    # Actually for a short, favorable = price drops = negative label
    # "Will passive exit fill?" = did price move favorably enough?
    # We check: within the 5s hold, did price drop at least 1 tick at ANY point?
    # Approximation: min(label_1s, label_5s) <= -1.0 (price dropped 1+ tick at either checkpoint)
    min_label = np.minimum(label_1s, label_5s)
    fill_target = (min_label <= -RETRACE_THRESHOLD).astype(np.float32)

    # Also compute a "strong fill" target (2-tick retrace)
    strong_fill = (min_label <= -2.0).astype(np.float32)

    return {
        'date': date_str,
        'features': features[short_mask],
        'fill_target': fill_target[short_mask],
        'strong_fill': strong_fill[short_mask],
        'label_5s': label_5s[short_mask],
        'features_all': features,
        'fill_target_all': fill_target,
    }


def train_fold(train_days, test_day, fold_idx, model_type='mlp'):
    """Train one WF fold."""
    train_X = np.concatenate([d['features'] for d in train_days])
    train_y = np.concatenate([d['fill_target'] for d in train_days])
    test_X = test_day['features']
    test_y = test_day['fill_target']

    if len(train_X) < 100 or len(test_X) < 10:
        return None

    # Normalize
    mean = train_X.mean(axis=0)
    std = train_X.std(axis=0) + 1e-8
    train_X_n = (train_X - mean) / std
    test_X_n = (test_X - mean) / std

    # Class weights for imbalanced data
    pos_rate = train_y.mean()
    if pos_rate == 0 or pos_rate == 1:
        return None
    pos_weight = torch.tensor([(1 - pos_rate) / pos_rate]).to(device)

    train_ds = TensorDataset(torch.from_numpy(train_X_n), torch.from_numpy(train_y))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    input_dim = train_X_n.shape[1]
    if model_type == 'cnn':
        model = FillPredictor1DCNN(input_dim, hidden=64).to(device)
    else:
        model = FillPredictorMLP(input_dim).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.train()
    for epoch in range(EPOCHS):
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(test_X_n).to(device)).cpu().numpy()
        probs = 1 / (1 + np.exp(-logits))

    try:
        auc = roc_auc_score(test_y, probs)
    except ValueError:
        auc = 0.5

    # Calibration: predicted fill rate vs actual
    high_conf = probs >= np.percentile(probs, 70)
    low_conf = probs < np.percentile(probs, 30)
    high_actual = test_y[high_conf].mean() if high_conf.sum() > 0 else 0
    low_actual = test_y[low_conf].mean() if low_conf.sum() > 0 else 0

    return {
        'fold': fold_idx,
        'date': test_day['date'],
        'n_test': len(test_X),
        'auc': float(auc),
        'base_fill_rate': float(test_y.mean()),
        'high_conf_fill_rate': float(high_actual),
        'low_conf_fill_rate': float(low_actual),
        'separation': float(high_actual - low_actual),
        'probs': probs,
        'actuals': test_y,
    }


def main():
    print(f"Device: {device}", flush=True)
    print(f"Output: {OUT_DIR}", flush=True)

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

    base_fill_rate = np.concatenate([d['fill_target'] for d in all_days]).mean()
    print(f"Base fill rate (1-tick retrace in 5s): {base_fill_rate:.1%}", flush=True)

    for model_type in ['mlp', 'cnn']:
        print(f"\n{'='*80}")
        print(f"MODEL: {model_type.upper()}")
        print(f"{'='*80}")
        print(f"{'Fold':>5} {'Date':>10} {'N':>6} {'AUC':>6} {'BaseFR':>7} {'HiCFR':>7} {'LoCFR':>7} {'Sep':>6}")
        print(f"{'-'*65}", flush=True)

        results = []
        all_probs = []
        all_actuals = []
        n_folds = len(all_days) - TRAIN_WINDOW

        for fold_idx in range(n_folds):
            train_days = all_days[fold_idx:fold_idx + TRAIN_WINDOW]
            test_day = all_days[fold_idx + TRAIN_WINDOW]
            result = train_fold(train_days, test_day, fold_idx, model_type)
            if result is None:
                continue

            results.append(result)
            all_probs.append(result['probs'])
            all_actuals.append(result['actuals'])

            print(f"{fold_idx:>5} {result['date']:>10} {result['n_test']:>6} "
                  f"{result['auc']:>5.3f} {result['base_fill_rate']:>6.1%} "
                  f"{result['high_conf_fill_rate']:>6.1%} {result['low_conf_fill_rate']:>6.1%} "
                  f"{result['separation']:>+5.1%}", flush=True)

        if not results:
            continue

        concat_probs = np.concatenate(all_probs)
        concat_actuals = np.concatenate(all_actuals)
        concat_auc = roc_auc_score(concat_actuals, concat_probs)

        print(f"\n{'='*80}")
        print(f"AGGREGATE — {model_type.upper()}")
        print(f"{'='*80}")
        print(f"Concat AUC: {concat_auc:.4f}")
        print(f"Mean fold AUC: {np.mean([r['auc'] for r in results]):.4f}")
        print(f"Mean separation: {np.mean([r['separation'] for r in results]):.1%}")

        # Filter analysis
        print(f"\nFILTER: Take only trades where fill predictor says high confidence")
        print(f"{'Threshold':>12} {'N_trades':>9} {'ActualFR':>9} {'Lift':>6}")
        print(f"{'-'*40}")
        for pct in [100, 80, 60, 50, 40, 30, 20]:
            if pct == 100:
                sel = concat_actuals
            else:
                thresh = np.percentile(concat_probs, 100 - pct)
                sel = concat_actuals[concat_probs >= thresh]
            actual_fr = sel.mean() if len(sel) > 0 else 0
            lift = actual_fr - concat_actuals.mean()
            print(f"{'All' if pct==100 else f'Top {pct}%':>12} {len(sel):>9} {actual_fr:>8.1%} {lift:>+5.1%}")

        summary = {
            'model_type': model_type,
            'concat_auc': float(concat_auc),
            'mean_fold_auc': float(np.mean([r['auc'] for r in results])),
            'base_fill_rate': float(concat_actuals.mean()),
            'n_folds': len(results),
            'per_fold': [{k: v for k, v in r.items() if k not in ('probs', 'actuals')} for r in results],
        }
        with open(OUT_DIR / f'results_{model_type}.json', 'w') as f:
            json.dump(summary, f, indent=2)

    print(f"\nResults saved to {OUT_DIR}", flush=True)


if __name__ == '__main__':
    main()
