#!/usr/bin/env python3
"""
Production Meta-Model for LONGS v1
=====================================
Same as production shorts model but for long signals.
Saves per-fold weights for deployment alongside short-side weights.
Uses confirmed best config: deeper MLP 256→128→64→32, 1s horizon, top 3% longs.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

if sys.platform == 'win32':
    _BASE = Path(r'C:\Users\claude\Lvl3Quant')
else:
    _BASE = Path('/home/jupiter/Lvl3Quant')

PRED_DIR = _BASE / 'output' / 'cnn_mamba_v2_bulk_oot_v2'
MBO_DIR = _BASE / 'data' / 'processed' / 'mbo_events_smart_v3'
OUT_DIR = _BASE / 'output' / 'meta_production_longs_v1'
WEIGHTS_DIR = OUT_DIR / 'weights'
OUT_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

COMMISSION_RT = 0.376
STOP_TICKS = 2.0
STOP_SLIPPAGE = 1.0
LONG_PERCENTILE = 97
TRAIN_WINDOW = 10
BATCH_SIZE = 4096
LR = 1e-3
EPOCHS = 30
WEIGHT_DECAY = 1e-4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ProductionMetaMLP(nn.Module):
    def __init__(self, input_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_day(pred_file):
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

    indices = np.arange(n_windows) * stride + (window_size - 1)
    max_idx = min(len(events), len(l1s)) - 1
    valid = indices <= max_idx
    indices = indices[valid]
    preds = preds[:len(indices)]

    feat = events[indices]
    label_1s = l1s[indices]

    valid_mask = ~(np.isnan(label_1s) | np.any(np.isnan(feat), axis=1))
    feat = feat[valid_mask]
    label_1s = label_1s[valid_mask]
    preds = preds[valid_mask]

    if len(feat) == 0:
        return None

    pred_1s = preds[:, 0]
    threshold = np.percentile(pred_1s, LONG_PERCENTILE)
    long_mask = pred_1s >= threshold

    feat_l = feat[long_mask]
    preds_l = preds[long_mask]
    label_1s_l = label_1s[long_mask]

    long_pnl = label_1s_l
    stop_hit = label_1s_l <= -STOP_TICKS
    pnl_target = np.where(
        stop_hit, -(STOP_TICKS + STOP_SLIPPAGE + COMMISSION_RT), long_pnl - COMMISSION_RT
    ).astype(np.float32)

    ranks = np.argsort(np.argsort(-preds_l[:, 0])).astype(np.float32) / max(len(preds_l), 1)
    features = np.column_stack([feat_l, preds_l[:, 0], preds_l[:, 1], preds_l[:, 2], ranks]).astype(np.float32)

    return {'date': date_str, 'features': features, 'pnl_target': pnl_target}


def main():
    print(f"{'='*70}", flush=True)
    print(f"PRODUCTION META-MODEL (LONGS) v1 TRAINING", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"{'='*70}\n", flush=True)

    pred_files = sorted([f for f in PRED_DIR.glob('*_predictions.npz') if '_stale_' not in str(f)])
    print(f"Found {len(pred_files)} prediction files", flush=True)

    all_days = []
    for i, pf in enumerate(pred_files):
        if i % 10 == 0:
            print(f"  Loading day {i+1}/{len(pred_files)}...", flush=True)
        day = load_day(pf)
        if day is not None:
            all_days.append(day)
    print(f"Loaded {len(all_days)} days\n", flush=True)

    n_folds = len(all_days) - TRAIN_WINDOW
    all_preds_list, all_actuals_list, fold_results = [], [], []

    for fold_idx in range(n_folds):
        train_days = all_days[fold_idx:fold_idx + TRAIN_WINDOW]
        test_day = all_days[fold_idx + TRAIN_WINDOW]

        train_X = np.concatenate([d['features'] for d in train_days])
        train_y = np.concatenate([d['pnl_target'] for d in train_days])
        test_X, test_y = test_day['features'], test_day['pnl_target']

        if len(train_X) < 100 or len(test_X) < 10:
            continue

        mean, std = train_X.mean(axis=0), train_X.std(axis=0) + 1e-8
        train_X_n, test_X_n = (train_X - mean) / std, (test_X - mean) / std

        model = ProductionMetaMLP(train_X_n.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        criterion = nn.MSELoss()

        ds = TensorDataset(torch.from_numpy(train_X_n), torch.from_numpy(train_y))
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

        best_loss = float('inf')
        model.train()
        for epoch in range(EPOCHS):
            epoch_loss, n_b = 0, 0
            for X_b, y_b in loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                loss = criterion(model(X_b), y_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item(); n_b += 1
            scheduler.step()
            if epoch_loss / n_b < best_loss:
                best_loss = epoch_loss / n_b

        fold_date = test_day['date']
        torch.save({
            'model_state': model.state_dict(), 'norm_mean': mean, 'norm_std': std,
            'fold_idx': fold_idx, 'test_date': fold_date,
            'train_dates': [d['date'] for d in train_days],
            'input_dim': train_X_n.shape[1], 'best_loss': best_loss,
            'side': 'long',
        }, WEIGHTS_DIR / f'fold_{fold_idx:03d}_{fold_date}.pt')

        model.eval()
        with torch.no_grad():
            oot_preds = model(torch.from_numpy(test_X_n).to(device)).cpu().numpy()

        corr = float(np.corrcoef(oot_preds, test_y)[0, 1]) if np.std(oot_preds) > 1e-8 else 0.0
        top30_t = np.percentile(oot_preds, 70)
        top30_pnl = float(test_y[oot_preds >= top30_t].mean()) if (oot_preds >= top30_t).sum() > 0 else 0

        fold_results.append({
            'fold': fold_idx, 'date': fold_date, 'n_test': len(test_X),
            'corr': corr, 'mean_pnl': float(test_y.mean()), 'top30_pnl': top30_pnl,
        })
        all_preds_list.append(oot_preds)
        all_actuals_list.append(test_y)

        status = "✓" if corr > 0 else "✗"
        print(f"  Fold {fold_idx:2d} [{fold_date}]: corr={corr:+.4f} pnl={test_y.mean():+.3f} top30={top30_pnl:+.3f} {status}", flush=True)

    concat_preds = np.concatenate(all_preds_list)
    concat_actuals = np.concatenate(all_actuals_list)
    concat_corr = float(np.corrcoef(concat_preds, concat_actuals)[0, 1])
    pos_folds = sum(1 for f in fold_results if f['corr'] > 0)

    print(f"\n{'='*70}", flush=True)
    print(f"RESULTS: corr={concat_corr:+.4f}, {pos_folds}/{len(fold_results)} positive folds", flush=True)

    filter_results = {}
    print(f"\n  {'Pct':>6} {'N':>7} {'MeanPnL':>8} {'Gross':>7} {'WR%':>6} {'PF':>6}", flush=True)
    for pct in [100, 50, 30, 20, 10]:
        sel = concat_actuals if pct == 100 else concat_actuals[concat_preds >= np.percentile(concat_preds, 100 - pct)]
        if len(sel) == 0: continue
        mpnl = float(sel.mean())
        wr = float((sel > 0).mean() * 100)
        wins = sel[sel > 0].sum() if (sel > 0).any() else 0
        losses = abs(sel[sel < 0].sum()) if (sel < 0).any() else 1
        pf = float(wins / losses) if losses > 0 else 999
        filter_results[str(pct)] = {'n': len(sel), 'mean_pnl': mpnl, 'gross': mpnl + COMMISSION_RT, 'wr': wr, 'pf': pf}
        print(f"  {pct:>5}% {len(sel):>7} {mpnl:>+7.3f} {mpnl+COMMISSION_RT:>+6.3f} {wr:>5.1f} {pf:>5.2f}", flush=True)

    results = {
        'side': 'long', 'concat_corr': concat_corr, 'pos_folds': pos_folds,
        'total_folds': len(fold_results), 'total_trades': len(concat_actuals),
        'filter_results': filter_results, 'per_fold': fold_results,
        'weights_saved': True, 'timestamp': datetime.now().isoformat(),
    }
    with open(OUT_DIR / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{len(fold_results)} fold weights saved to {WEIGHTS_DIR}", flush=True)
    print(f"Results saved to {OUT_DIR / 'results.json'}", flush=True)


if __name__ == '__main__':
    main()
