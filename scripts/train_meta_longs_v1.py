#!/usr/bin/env python3
"""
Meta-Model for LONG Signals v1
==================================
Tree-branch: does the meta-model filter work for LONG signals?
All prior work tested shorts only (top 3% most negative pred_1s).
This tests top 3% LONGS (most positive pred_1s).

If longs work, we can double trade capacity by trading both directions.
Uses confirmed best config: deeper MLP 256→128→64→32, 1s horizon.
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
OUT_DIR = _BASE / 'output' / 'meta_longs_v1'
OUT_DIR.mkdir(parents=True, exist_ok=True)

COMMISSION_RT = 0.376
STOP_TICKS = 2.0
STOP_SLIPPAGE = 1.0
LONG_PERCENTILE = 97  # top 3% longs = pred_1s >= p97
TRAIN_WINDOW = 10
BATCH_SIZE = 4096
LR = 1e-3
EPOCHS = 30
WEIGHT_DECAY = 1e-4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MetaMLP(nn.Module):
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

    # Select top 3% LONGS (most positive pred_1s)
    pred_1s = preds[:, 0]
    threshold = np.percentile(pred_1s, LONG_PERCENTILE)
    long_mask = pred_1s >= threshold

    feat_l = feat[long_mask]
    preds_l = preds[long_mask]
    label_1s_l = label_1s[long_mask]

    # P&L for LONGS: profit when price goes UP (positive label_1s)
    long_pnl = label_1s_l  # long profits from positive moves
    stop_hit = label_1s_l <= -STOP_TICKS  # stop hit when price drops
    pnl_target = np.where(
        stop_hit,
        -(STOP_TICKS + STOP_SLIPPAGE + COMMISSION_RT),
        long_pnl - COMMISSION_RT,
    ).astype(np.float32)

    # Features: 25 MBO + 3 preds + rank = 29
    # For longs, rank should be descending (highest pred = best)
    ranks = np.argsort(np.argsort(-preds_l[:, 0])).astype(np.float32) / max(len(preds_l), 1)
    features = np.column_stack([feat_l, preds_l[:, 0], preds_l[:, 1], preds_l[:, 2], ranks]).astype(np.float32)

    return {
        'date': date_str,
        'features': features,
        'pnl_target': pnl_target,
        'label_1s': label_1s_l,
        'n_total': int(valid_mask.sum()),
        'n_longs': int(long_mask.sum()),
    }


def main():
    print(f"{'='*70}", flush=True)
    print(f"META-MODEL FOR LONGS v1", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Architecture: 256→128→64→32", flush=True)
    print(f"Horizon: 1s, Top {100 - LONG_PERCENTILE}% LONGS", flush=True)
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
    print(f"Walk-forward folds: {n_folds}\n", flush=True)

    # First: baseline analysis (no meta filter, just raw long signals)
    all_raw_pnl = np.concatenate([d['pnl_target'] for d in all_days[TRAIN_WINDOW:]])
    baseline_pnl = float(all_raw_pnl.mean())
    baseline_wr = float((all_raw_pnl > 0).mean() * 100)
    baseline_gross = baseline_pnl + COMMISSION_RT
    print(f"BASELINE (top 3% longs, no meta filter):", flush=True)
    print(f"  N trades: {len(all_raw_pnl)}", flush=True)
    print(f"  Mean P&L: {baseline_pnl:+.3f} ticks", flush=True)
    print(f"  Gross: {baseline_gross:+.3f} ticks", flush=True)
    print(f"  WR: {baseline_wr:.1f}%", flush=True)

    # Compare to shorts baseline (from prior results)
    print(f"\n  [Context: shorts baseline was +0.366 ticks, WR 60.0%]", flush=True)
    print(f"  Long vs Short baseline: {baseline_pnl - 0.366:+.3f} ticks difference\n", flush=True)

    # Walk-forward meta-model training
    all_preds_list = []
    all_actuals_list = []
    fold_results = []

    for fold_idx in range(n_folds):
        train_days = all_days[fold_idx:fold_idx + TRAIN_WINDOW]
        test_day = all_days[fold_idx + TRAIN_WINDOW]

        train_X = np.concatenate([d['features'] for d in train_days])
        train_y = np.concatenate([d['pnl_target'] for d in train_days])
        test_X = test_day['features']
        test_y = test_day['pnl_target']

        if len(train_X) < 100 or len(test_X) < 10:
            print(f"  Fold {fold_idx}: SKIP (train={len(train_X)}, test={len(test_X)})", flush=True)
            continue

        mean = train_X.mean(axis=0)
        std = train_X.std(axis=0) + 1e-8
        train_X_n = (train_X - mean) / std
        test_X_n = (test_X - mean) / std

        model = MetaMLP(train_X_n.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        criterion = nn.MSELoss()

        ds = TensorDataset(torch.from_numpy(train_X_n), torch.from_numpy(train_y))
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

        model.train()
        for epoch in range(EPOCHS):
            for X_b, y_b in loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                loss = criterion(model(X_b), y_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

        model.eval()
        with torch.no_grad():
            oot_preds = model(torch.from_numpy(test_X_n).to(device)).cpu().numpy()

        corr = float(np.corrcoef(oot_preds, test_y)[0, 1]) if np.std(oot_preds) > 1e-8 else 0.0

        # Filter analysis
        top30_thresh = np.percentile(oot_preds, 70)
        top30_pnl = float(test_y[oot_preds >= top30_thresh].mean()) if (oot_preds >= top30_thresh).sum() > 0 else 0
        top10_thresh = np.percentile(oot_preds, 90)
        top10_pnl = float(test_y[oot_preds >= top10_thresh].mean()) if (oot_preds >= top10_thresh).sum() > 0 else 0

        fold_results.append({
            'fold': fold_idx, 'date': test_day['date'], 'n_test': len(test_X),
            'corr': corr, 'mean_pnl': float(test_y.mean()),
            'top30_pnl': top30_pnl, 'top10_pnl': top10_pnl,
        })
        all_preds_list.append(oot_preds)
        all_actuals_list.append(test_y)

        status = "✓" if corr > 0 else "✗"
        print(f"  Fold {fold_idx:2d} [{test_day['date']}]: corr={corr:+.4f} pnl={test_y.mean():+.3f} "
              f"top30={top30_pnl:+.3f} top10={top10_pnl:+.3f} {status}", flush=True)

    # Concat analysis
    concat_preds = np.concatenate(all_preds_list)
    concat_actuals = np.concatenate(all_actuals_list)
    concat_corr = float(np.corrcoef(concat_preds, concat_actuals)[0, 1])
    pos_folds = sum(1 for f in fold_results if f['corr'] > 0)

    print(f"\n{'='*70}", flush=True)
    print(f"CONCAT RESULTS ({len(concat_preds)} long trades, {len(fold_results)} folds)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"Concat correlation: {concat_corr:+.4f}", flush=True)
    print(f"Positive folds: {pos_folds}/{len(fold_results)}", flush=True)

    # Filter table
    print(f"\n  Filter performance:", flush=True)
    print(f"  {'Pct':>6} {'N':>7} {'MeanPnL':>8} {'Gross':>7} {'WR%':>6} {'PF':>6} {'Lift':>7}", flush=True)
    filter_results = {}
    for pct in [100, 50, 30, 20, 10]:
        if pct == 100:
            sel = concat_actuals
        else:
            thresh = np.percentile(concat_preds, 100 - pct)
            sel = concat_actuals[concat_preds >= thresh]
        if len(sel) == 0:
            continue
        mpnl = float(sel.mean())
        wr = float((sel > 0).mean() * 100)
        wins = sel[sel > 0].sum() if (sel > 0).any() else 0
        losses = abs(sel[sel < 0].sum()) if (sel < 0).any() else 1
        pf = float(wins / losses) if losses > 0 else 999
        gross = mpnl + COMMISSION_RT
        lift = float(mpnl - concat_actuals.mean())
        filter_results[str(pct)] = {
            'n': len(sel), 'mean_pnl': mpnl, 'gross': gross, 'wr': wr, 'pf': pf, 'lift': lift,
        }
        print(f"  {pct:>5}% {len(sel):>7} {mpnl:>+7.3f} {gross:>+6.3f} {wr:>5.1f} {pf:>5.2f} {lift:>+6.3f}", flush=True)

    # Comparison with shorts
    print(f"\n{'='*70}", flush=True)
    print(f"LONGS vs SHORTS COMPARISON", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'Metric':>20} {'Shorts':>10} {'Longs':>10} {'Diff':>10}", flush=True)
    short_corr = 0.138
    short_base = 0.366
    short_t30 = 1.019
    short_t10 = 1.065
    l_t30 = filter_results.get('30', {}).get('gross', 0)
    l_t10 = filter_results.get('10', {}).get('gross', 0)
    print(f"{'Concat corr':>20} {short_corr:>+9.4f} {concat_corr:>+9.4f} {concat_corr-short_corr:>+9.4f}", flush=True)
    print(f"{'Baseline gross':>20} {short_base+0.376:>+9.3f} {baseline_gross:>+9.3f} {baseline_gross-(short_base+0.376):>+9.3f}", flush=True)
    print(f"{'Top 30% gross':>20} {short_t30:>+9.3f} {l_t30:>+9.3f} {l_t30-short_t30:>+9.3f}", flush=True)
    print(f"{'Top 10% gross':>20} {short_t10:>+9.3f} {l_t10:>+9.3f} {l_t10-short_t10:>+9.3f}", flush=True)
    print(f"{'Pos folds':>20} {'14/15':>10} {f'{pos_folds}/{len(fold_results)}':>10}", flush=True)

    # Verdict
    print(f"\n{'='*70}", flush=True)
    if concat_corr > 0.05 and pos_folds >= len(fold_results) * 0.7:
        print(f"VERDICT: PASS — Long meta-model has predictive power. Can trade both sides.", flush=True)
    elif concat_corr > 0.02:
        print(f"VERDICT: WEAK PASS — Some signal but significantly weaker than shorts.", flush=True)
    else:
        print(f"VERDICT: REJECT — Long meta-model does not have sufficient predictive power.", flush=True)
    print(f"{'='*70}", flush=True)

    # Save
    results = {
        'concat_corr': concat_corr,
        'pos_folds': pos_folds,
        'total_folds': len(fold_results),
        'total_trades': len(concat_actuals),
        'baseline_pnl': baseline_pnl,
        'baseline_gross': baseline_gross,
        'filter_results': filter_results,
        'per_fold': fold_results,
    }
    with open(OUT_DIR / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT_DIR}", flush=True)


if __name__ == '__main__':
    main()
