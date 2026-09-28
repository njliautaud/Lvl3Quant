#!/usr/bin/env python3
"""
Stacked Meta + Fill Filter v1
===============================
Combines two proven signals:
  1. Meta-model (predicts realized P&L) — concat corr +0.082
  2. Fill predictor (predicts passive fill probability) — AUC 0.558

Two approaches:
  A) Sequential filter: meta-model top N% → then fill predictor top M%
  B) Joint model: train single MLP with both targets as auxiliary losses
  C) Score fusion: weighted combination of meta-score and fill-score

Walk-forward: 10-date train, 1-date OOT.
"""

import json
import sys
from pathlib import Path

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
OUT_DIR = _BASE / 'output' / 'stacked_filter_v1'
OUT_DIR.mkdir(parents=True, exist_ok=True)

COMMISSION_RT = 0.376
STOP_TICKS = 2.0
STOP_SLIPPAGE = 1.0
SHORT_PERCENTILE = 3
TRAIN_WINDOW = 10
BATCH_SIZE = 4096
LR = 1e-3
EPOCHS = 30
WEIGHT_DECAY = 1e-4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DualHeadMLP(nn.Module):
    """MLP with shared backbone, dual heads: P&L regression + fill classification."""
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
        self.backbone = nn.Sequential(*layers)
        self.pnl_head = nn.Linear(prev, 1)   # regression: realized P&L
        self.fill_head = nn.Linear(prev, 1)   # classification: will fill?

    def forward(self, x):
        feat = self.backbone(x)
        pnl = self.pnl_head(feat).squeeze(-1)
        fill = self.fill_head(feat).squeeze(-1)
        return pnl, fill


def load_day(pred_file):
    """Load one day with both targets."""
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

    # P&L target (passive-passive)
    short_pnl = -label_5s
    stop_hit = label_1s >= STOP_TICKS
    pnl_target = np.where(
        stop_hit,
        -(STOP_TICKS + STOP_SLIPPAGE + COMMISSION_RT),
        short_pnl - COMMISSION_RT,
    ).astype(np.float32)

    # Fill target (1-tick retrace within 5s)
    min_label = np.minimum(label_1s, label_5s)
    fill_target = (min_label <= -1.0).astype(np.float32)

    # Features
    pred_1s, pred_5s, pred_10s = preds[:, 0], preds[:, 1], preds[:, 2]
    ranks = np.argsort(np.argsort(pred_5s)).astype(np.float32) / len(pred_5s)
    features = np.column_stack([feat, pred_1s, pred_5s, pred_10s, ranks]).astype(np.float32)

    # Short selection
    threshold = np.percentile(pred_5s, SHORT_PERCENTILE)
    short_mask = pred_5s <= threshold

    return {
        'date': date_str,
        'features': features[short_mask],
        'pnl_target': pnl_target[short_mask],
        'fill_target': fill_target[short_mask],
        'label_5s': label_5s[short_mask],
    }


def train_fold(train_days, test_day, fold_idx):
    """Train dual-head model on one WF fold."""
    train_X = np.concatenate([d['features'] for d in train_days])
    train_pnl = np.concatenate([d['pnl_target'] for d in train_days])
    train_fill = np.concatenate([d['fill_target'] for d in train_days])

    test_X = test_day['features']
    test_pnl = test_day['pnl_target']
    test_fill = test_day['fill_target']

    if len(train_X) < 100 or len(test_X) < 10:
        return None

    # Normalize
    mean = train_X.mean(axis=0)
    std = train_X.std(axis=0) + 1e-8
    train_X_n = (train_X - mean) / std
    test_X_n = (test_X - mean) / std

    # Class weight for fill head
    pos_rate = train_fill.mean()
    if pos_rate == 0 or pos_rate == 1:
        return None
    pos_weight = torch.tensor([(1 - pos_rate) / pos_rate]).to(device)

    train_ds = TensorDataset(
        torch.from_numpy(train_X_n),
        torch.from_numpy(train_pnl),
        torch.from_numpy(train_fill),
    )
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    model = DualHeadMLP(train_X_n.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.train()
    for epoch in range(EPOCHS):
        for X_b, pnl_b, fill_b in loader:
            X_b = X_b.to(device)
            pnl_b = pnl_b.to(device)
            fill_b = fill_b.to(device)
            optimizer.zero_grad()
            pnl_pred, fill_pred = model(X_b)
            loss = mse(pnl_pred, pnl_b) + 0.5 * bce(fill_pred, fill_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        test_tensor = torch.from_numpy(test_X_n).to(device)
        pnl_pred, fill_pred = model(test_tensor)
        pnl_scores = pnl_pred.cpu().numpy()
        fill_probs = torch.sigmoid(fill_pred).cpu().numpy()

    # Combined score: normalize both to [0,1] then average
    pnl_rank = np.argsort(np.argsort(pnl_scores)).astype(np.float32) / len(pnl_scores)
    fill_rank = np.argsort(np.argsort(fill_probs)).astype(np.float32) / len(fill_probs)

    # Multiple fusion strategies
    results = {}
    for name, combined in [
        ('pnl_only', pnl_rank),
        ('fill_only', fill_rank),
        ('equal_avg', 0.5 * pnl_rank + 0.5 * fill_rank),
        ('pnl_heavy', 0.7 * pnl_rank + 0.3 * fill_rank),
        ('fill_heavy', 0.3 * pnl_rank + 0.7 * fill_rank),
    ]:
        # Evaluate as filter
        filter_results = {}
        for pct in [100, 70, 50, 30, 20, 10]:
            if pct == 100:
                sel_pnl = test_pnl
            else:
                thresh = np.percentile(combined, 100 - pct)
                sel_mask = combined >= thresh
                sel_pnl = test_pnl[sel_mask]

            if len(sel_pnl) == 0:
                continue

            mean_pnl = float(sel_pnl.mean())
            wr = float((sel_pnl > 0).mean() * 100)
            wins = sel_pnl[sel_pnl > 0].sum() if (sel_pnl > 0).any() else 0
            losses = abs(sel_pnl[sel_pnl < 0].sum()) if (sel_pnl < 0).any() else 1
            pf = float(wins / losses) if losses > 0 else 999

            filter_results[str(pct)] = {
                'n': len(sel_pnl), 'mean_pnl': mean_pnl,
                'wr': wr, 'pf': pf,
            }

        results[name] = filter_results

    # Correlation
    pnl_corr = float(np.corrcoef(pnl_scores, test_pnl)[0, 1]) if np.std(pnl_scores) > 1e-8 else 0

    return {
        'fold': fold_idx,
        'date': test_day['date'],
        'n_test': len(test_X),
        'pnl_corr': pnl_corr,
        'base_fill_rate': float(test_fill.mean()),
        'filter_results': results,
        'pnl_scores': pnl_scores,
        'test_pnl': test_pnl,
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
    print(f"\nWalk-forward: {n_folds} folds", flush=True)
    print(f"{'Fold':>5} {'Date':>10} {'N':>6} {'PnLCorr':>8}", flush=True)
    print(f"{'-'*35}", flush=True)

    results = []
    all_pnl_scores = []
    all_test_pnl = []

    for fold_idx in range(n_folds):
        train_days = all_days[fold_idx:fold_idx + TRAIN_WINDOW]
        test_day = all_days[fold_idx + TRAIN_WINDOW]
        result = train_fold(train_days, test_day, fold_idx)
        if result is None:
            continue
        results.append(result)
        all_pnl_scores.append(result['pnl_scores'])
        all_test_pnl.append(result['test_pnl'])
        print(f"{fold_idx:>5} {result['date']:>10} {result['n_test']:>6} {result['pnl_corr']:>+7.3f}", flush=True)

    if not results:
        print("No valid folds!")
        return

    # Aggregate
    concat_scores = np.concatenate(all_pnl_scores)
    concat_pnl = np.concatenate(all_test_pnl)
    concat_corr = float(np.corrcoef(concat_scores, concat_pnl)[0, 1])

    print(f"\n{'='*80}")
    print(f"AGGREGATE RESULTS")
    print(f"{'='*80}")
    print(f"Concat P&L correlation: {concat_corr:+.4f}")
    print(f"Positive-corr folds: {sum(1 for r in results if r['pnl_corr'] > 0)}/{len(results)}")

    # Aggregate filter analysis per fusion strategy
    print(f"\n{'='*80}")
    print(f"FILTER COMPARISON (concat across all folds)")
    print(f"{'='*80}")

    fusion_names = ['pnl_only', 'fill_only', 'equal_avg', 'pnl_heavy', 'fill_heavy']
    for fname in fusion_names:
        print(f"\n--- {fname} ---")
        print(f"{'Pct':>6} {'MeanPnL':>8} {'WR%':>6} {'PF':>6}")
        # Average across folds
        for pct_str in ['100', '70', '50', '30', '20', '10']:
            pnls = [r['filter_results'][fname][pct_str]['mean_pnl']
                    for r in results if pct_str in r['filter_results'].get(fname, {})]
            wrs = [r['filter_results'][fname][pct_str]['wr']
                   for r in results if pct_str in r['filter_results'].get(fname, {})]
            pfs = [r['filter_results'][fname][pct_str]['pf']
                   for r in results if pct_str in r['filter_results'].get(fname, {})]
            if pnls:
                print(f"{pct_str + '%':>6} {np.mean(pnls):>+7.3f} {np.mean(wrs):>5.1f} {np.mean(pfs):>5.2f}")

    # Save
    summary = {
        'concat_pnl_corr': concat_corr,
        'n_folds': len(results),
        'per_fold': [{k: v for k, v in r.items() if k not in ('pnl_scores', 'test_pnl')} for r in results],
    }
    with open(OUT_DIR / 'results.json', 'w') as f:
        json.dump(summary, f, indent=2)

    np.savez_compressed(OUT_DIR / 'concat_predictions.npz',
                        pnl_scores=concat_scores, actual_pnl=concat_pnl)

    print(f"\nResults saved to {OUT_DIR}", flush=True)
    print(f"\nVERDICT: {'PASS' if concat_corr > 0.05 else 'WEAK' if concat_corr > 0 else 'FAIL'} "
          f"(concat_corr={concat_corr:+.4f})")


if __name__ == '__main__':
    main()
