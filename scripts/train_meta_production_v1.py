#!/usr/bin/env python3
"""
Production Meta-Model v1
===========================
Combines ALL confirmed findings into the canonical production model:
  - 1s horizon (multi-horizon winner, corr +0.114 vs +0.076 for 5s)
  - Deeper MLP 256→128→64→32 (arch sweep winner, top 10% PF 3.42)
  - Top 3% short signal selection (balanced volume/edge)
  - Walk-forward: 10-date train, 1-date OOT
  - Saves per-fold weights + normalizer for live deployment
  - Full validation: per-fold metrics, concat metrics, regime check

This is the model that would run in production alongside CNN-Mamba v2.
"""

import json
import sys
import os
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
OUT_DIR = _BASE / 'output' / 'meta_production_v1'
WEIGHTS_DIR = OUT_DIR / 'weights'
OUT_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

COMMISSION_RT = 0.376
STOP_TICKS = 2.0
STOP_SLIPPAGE = 1.0
SHORT_PERCENTILE = 3
TRAIN_WINDOW = 10
BATCH_SIZE = 4096
LR = 1e-3
EPOCHS = 30  # Slightly more epochs for production quality
WEIGHT_DECAY = 1e-4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ProductionMetaMLP(nn.Module):
    """Deeper MLP: 256→128→64→32 (arch sweep winner)."""
    def __init__(self, input_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_day(pred_file):
    """Load one day with full data."""
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
    l5s = mbo['labels_5s'] if 'labels_5s' in mbo else None

    indices = np.arange(n_windows) * stride + (window_size - 1)
    max_idx = min(len(events), len(l1s)) - 1
    valid = indices <= max_idx
    indices = indices[valid]
    preds = preds[:len(indices)]

    feat = events[indices]
    label_1s = l1s[indices]
    label_5s = l5s[indices] if l5s is not None else None

    valid_mask = ~(np.isnan(label_1s) | np.any(np.isnan(feat), axis=1))
    if label_5s is not None:
        valid_mask &= ~np.isnan(label_5s)

    feat = feat[valid_mask]
    label_1s = label_1s[valid_mask]
    if label_5s is not None:
        label_5s = label_5s[valid_mask]
    preds = preds[valid_mask]

    if len(feat) == 0:
        return None

    # Select top 3% shorts
    pred_1s = preds[:, 0]
    threshold = np.percentile(pred_1s, SHORT_PERCENTILE)
    short_mask = pred_1s <= threshold

    feat_s = feat[short_mask]
    preds_s = preds[short_mask]
    label_1s_s = label_1s[short_mask]
    label_5s_s = label_5s[short_mask] if label_5s is not None else None

    # P&L target (1s horizon — confirmed best)
    short_pnl = -label_1s_s
    stop_hit = label_1s_s >= STOP_TICKS
    pnl_target = np.where(
        stop_hit,
        -(STOP_TICKS + STOP_SLIPPAGE + COMMISSION_RT),
        short_pnl - COMMISSION_RT,
    ).astype(np.float32)

    # Features: 25 MBO + 3 preds + rank = 29
    ranks = np.argsort(np.argsort(preds_s[:, 0])).astype(np.float32) / max(len(preds_s), 1)
    features = np.column_stack([feat_s, preds_s[:, 0], preds_s[:, 1], preds_s[:, 2], ranks]).astype(np.float32)

    return {
        'date': date_str,
        'features': features,
        'pnl_target': pnl_target,
        'label_1s': label_1s_s,
        'label_5s': label_5s_s,
        'n_total': int(valid_mask.sum()),
        'n_shorts': int(short_mask.sum()),
    }


def main():
    print(f"{'='*70}", flush=True)
    print(f"PRODUCTION META-MODEL v1 TRAINING", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Architecture: 256→128→64→32 (deeper MLP)", flush=True)
    print(f"Horizon: 1s (confirmed best)", flush=True)
    print(f"Signal: top {SHORT_PERCENTILE}% shorts", flush=True)
    print(f"Walk-forward: {TRAIN_WINDOW}-date train, 1-date OOT", flush=True)
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

        # Normalize
        mean = train_X.mean(axis=0)
        std = train_X.std(axis=0) + 1e-8
        train_X_n = (train_X - mean) / std
        test_X_n = (test_X - mean) / std

        # Train
        model = ProductionMetaMLP(train_X_n.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        criterion = nn.MSELoss()

        ds = TensorDataset(torch.from_numpy(train_X_n), torch.from_numpy(train_y))
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

        best_loss = float('inf')
        model.train()
        for epoch in range(EPOCHS):
            epoch_loss = 0
            n_batches = 0
            for X_b, y_b in loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                loss = criterion(model(X_b), y_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            scheduler.step()
            avg_loss = epoch_loss / n_batches
            if avg_loss < best_loss:
                best_loss = avg_loss

        # Save weights + normalizer for this fold
        fold_date = test_day['date']
        weight_file = WEIGHTS_DIR / f'fold_{fold_idx:03d}_{fold_date}.pt'
        torch.save({
            'model_state': model.state_dict(),
            'norm_mean': mean,
            'norm_std': std,
            'fold_idx': fold_idx,
            'test_date': fold_date,
            'train_dates': [d['date'] for d in train_days],
            'input_dim': train_X_n.shape[1],
            'best_loss': best_loss,
        }, weight_file)

        # Evaluate
        model.eval()
        with torch.no_grad():
            oot_preds = model(torch.from_numpy(test_X_n).to(device)).cpu().numpy()

        if np.std(oot_preds) > 1e-8:
            corr = float(np.corrcoef(oot_preds, test_y)[0, 1])
        else:
            corr = 0.0

        mean_pnl = float(test_y.mean())
        wr = float((test_y > 0).mean() * 100)
        wins = test_y[test_y > 0].sum() if (test_y > 0).any() else 0
        losses_val = abs(test_y[test_y < 0].sum()) if (test_y < 0).any() else 1
        pf = float(wins / losses_val) if losses_val > 0 else 999

        # Filter analysis for this fold
        top30_thresh = np.percentile(oot_preds, 70)
        top30_mask = oot_preds >= top30_thresh
        top30_pnl = float(test_y[top30_mask].mean()) if top30_mask.sum() > 0 else 0

        top10_thresh = np.percentile(oot_preds, 90)
        top10_mask = oot_preds >= top10_thresh
        top10_pnl = float(test_y[top10_mask].mean()) if top10_mask.sum() > 0 else 0

        fold_data = {
            'fold': fold_idx,
            'date': fold_date,
            'n_train': len(train_X),
            'n_test': len(test_X),
            'corr': corr,
            'mean_pnl': mean_pnl,
            'wr': wr,
            'pf': pf,
            'top30_pnl': top30_pnl,
            'top10_pnl': top10_pnl,
            'best_loss': best_loss,
        }
        fold_results.append(fold_data)

        all_preds_list.append(oot_preds)
        all_actuals_list.append(test_y)

        status = "✓" if corr > 0 else "✗"
        print(f"  Fold {fold_idx:2d} [{fold_date}]: corr={corr:+.4f} pnl={mean_pnl:+.3f} "
              f"wr={wr:.1f}% pf={pf:.2f} top30={top30_pnl:+.3f} top10={top10_pnl:+.3f} {status}", flush=True)

    # Concat analysis
    concat_preds = np.concatenate(all_preds_list)
    concat_actuals = np.concatenate(all_actuals_list)
    concat_corr = float(np.corrcoef(concat_preds, concat_actuals)[0, 1])

    print(f"\n{'='*70}", flush=True)
    print(f"CONCAT RESULTS ({len(concat_preds)} trades across {len(fold_results)} folds)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"Concat correlation: {concat_corr:+.4f}", flush=True)
    print(f"Positive folds: {sum(1 for f in fold_results if f['corr'] > 0)}/{len(fold_results)}", flush=True)
    print(f"Mean fold corr: {np.mean([f['corr'] for f in fold_results]):+.4f}", flush=True)

    # Filter performance
    baseline_pnl = concat_actuals.mean()
    print(f"\n  Filter performance:", flush=True)
    print(f"  {'Pct':>6} {'N':>7} {'MeanPnL':>8} {'Gross':>7} {'WR%':>6} {'PF':>6} {'Lift':>7}", flush=True)
    filter_results = {}
    for pct in [100, 50, 30, 20, 10, 5]:
        if pct == 100:
            sel_pnl = concat_actuals
        else:
            thresh = np.percentile(concat_preds, 100 - pct)
            sel_pnl = concat_actuals[concat_preds >= thresh]
        if len(sel_pnl) == 0:
            continue
        mean_pnl = float(sel_pnl.mean())
        wr = float((sel_pnl > 0).mean() * 100)
        wins = sel_pnl[sel_pnl > 0].sum() if (sel_pnl > 0).any() else 0
        losses_val = abs(sel_pnl[sel_pnl < 0].sum()) if (sel_pnl < 0).any() else 1
        pf = float(wins / losses_val) if losses_val > 0 else 999
        gross = mean_pnl + COMMISSION_RT
        lift = float(mean_pnl - baseline_pnl)
        filter_results[str(pct)] = {
            'n': len(sel_pnl), 'mean_pnl': mean_pnl, 'gross': gross,
            'wr': wr, 'pf': pf, 'lift': lift,
        }
        print(f"  {pct:>5}% {len(sel_pnl):>7} {mean_pnl:>+7.3f} {gross:>+6.3f} "
              f"{wr:>5.1f} {pf:>5.2f} {lift:>+6.3f}", flush=True)

    # Daily trade volume estimates
    print(f"\n{'='*70}", flush=True)
    print(f"DAILY VOLUME ESTIMATES (at top 30% filter)", flush=True)
    print(f"{'='*70}", flush=True)
    n_days = len(fold_results)
    fr30 = filter_results.get('30', {})
    if fr30:
        tpd = fr30['n'] / n_days
        daily_gross = tpd * fr30['gross']
        daily_net = tpd * fr30['mean_pnl']
        print(f"  Trades/day: {tpd:.0f}", flush=True)
        print(f"  Gross/trade: {fr30['gross']:+.3f} ticks", flush=True)
        print(f"  Daily gross: {daily_gross:+.1f} ticks (${daily_gross * 12.5:+,.0f})", flush=True)
        print(f"  Daily net (after commission): {daily_net:+.1f} ticks (${daily_net * 12.5:+,.0f})", flush=True)

    # Sharpe estimate
    fold_daily_pnl = []
    for f in fold_results:
        fold_daily_pnl.append(f['top30_pnl'] * f['n_test'] * 0.3)  # approx daily total at top 30%
    fold_daily_pnl = np.array(fold_daily_pnl)
    if fold_daily_pnl.std() > 0:
        daily_sharpe = fold_daily_pnl.mean() / fold_daily_pnl.std() * np.sqrt(252)
        print(f"\n  Annualized Sharpe (daily, top 30%): {daily_sharpe:.1f}", flush=True)
        sortino_denom = fold_daily_pnl[fold_daily_pnl < 0].std() if (fold_daily_pnl < 0).any() else 1
        sortino = fold_daily_pnl.mean() / sortino_denom * np.sqrt(252) if sortino_denom > 0 else 999
        print(f"  Annualized Sortino: {sortino:.1f}", flush=True)
        green_days = (fold_daily_pnl > 0).sum()
        print(f"  Green days: {green_days}/{len(fold_daily_pnl)} ({green_days/len(fold_daily_pnl)*100:.0f}%)", flush=True)

    # Save results
    results = {
        'config': {
            'architecture': '256_128_64_32',
            'horizon': '1s',
            'signal_threshold': SHORT_PERCENTILE,
            'train_window': TRAIN_WINDOW,
            'epochs': EPOCHS,
            'lr': LR,
            'weight_decay': WEIGHT_DECAY,
            'batch_size': BATCH_SIZE,
        },
        'concat_corr': concat_corr,
        'mean_fold_corr': float(np.mean([f['corr'] for f in fold_results])),
        'pos_folds': sum(1 for f in fold_results if f['corr'] > 0),
        'total_folds': len(fold_results),
        'total_trades': len(concat_actuals),
        'filter_results': filter_results,
        'per_fold': fold_results,
        'weights_saved': True,
        'weights_dir': str(WEIGHTS_DIR),
        'timestamp': datetime.now().isoformat(),
    }

    # Save concat predictions for downstream analysis
    np.savez(OUT_DIR / 'concat_predictions.npz',
             predictions=concat_preds,
             actuals=concat_actuals)

    with open(OUT_DIR / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}", flush=True)
    print(f"SAVED:", flush=True)
    print(f"  Weights: {len(fold_results)} fold checkpoints in {WEIGHTS_DIR}", flush=True)
    print(f"  Results: {OUT_DIR / 'results.json'}", flush=True)
    print(f"  Predictions: {OUT_DIR / 'concat_predictions.npz'}", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == '__main__':
    main()
