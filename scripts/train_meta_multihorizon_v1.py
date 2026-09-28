#!/usr/bin/env python3
"""
Multi-Horizon Meta-Model v1
============================
Tree-branch from confirmed meta-model v2 win.
Tests: does the meta-model filter work at 1s and 10s horizons, not just 5s?

For each horizon h in {1s, 5s, 10s}:
  - Select top 3% shorts by pred_h (most negative prediction at horizon h)
  - Train MLP 256→128→64 predicting realized P&L at horizon h
  - Evaluate filter performance (top 50%, 30%, 10%)

If 1s or 10s horizons show better meta-model lift than 5s, that's a new branch to explore.

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
OUT_DIR = _BASE / 'output' / 'meta_multihorizon_v1'
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


class MetaMLP(nn.Module):
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
    """Load one day with labels at all horizons."""
    pred_data = np.load(pred_file, allow_pickle=True)
    date_str = str(pred_data['date'])
    preds = pred_data['predictions']  # (n, 3) = [1s, 5s, 10s]
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

    return {
        'date': date_str,
        'features': feat.astype(np.float32),
        'preds': preds.astype(np.float32),
        'label_1s': label_1s.astype(np.float32),
        'label_5s': label_5s.astype(np.float32),
        'label_10s': label_10s.astype(np.float32),
    }


def build_horizon_data(day, horizon_idx, label_key):
    """Select top 3% shorts for a specific horizon and build features+target."""
    pred_h = day['preds'][:, horizon_idx]
    threshold = np.percentile(pred_h, SHORT_PERCENTILE)
    short_mask = pred_h <= threshold

    feat = day['features'][short_mask]
    preds = day['preds'][short_mask]
    label_h = day[label_key][short_mask]
    label_1s = day['label_1s'][short_mask]

    # P&L target
    short_pnl = -label_h
    stop_hit = label_1s >= STOP_TICKS
    pnl_target = np.where(
        stop_hit,
        -(STOP_TICKS + STOP_SLIPPAGE + COMMISSION_RT),
        short_pnl - COMMISSION_RT,
    ).astype(np.float32)

    # Build extended features (25 MBO + 3 preds + rank = 29)
    ranks = np.argsort(np.argsort(pred_h[short_mask] if False else preds[:, horizon_idx])).astype(np.float32) / max(len(preds), 1)
    features = np.column_stack([feat, preds[:, 0], preds[:, 1], preds[:, 2], ranks]).astype(np.float32)

    return features, pnl_target


def train_horizon(all_days, horizon_name, horizon_idx, label_key):
    """Run full walk-forward for one horizon."""
    n_folds = len(all_days) - TRAIN_WINDOW

    all_preds_list = []
    all_actuals_list = []
    fold_corrs = []

    for fold_idx in range(n_folds):
        train_days = all_days[fold_idx:fold_idx + TRAIN_WINDOW]
        test_day = all_days[fold_idx + TRAIN_WINDOW]

        # Build data
        train_parts = [build_horizon_data(d, horizon_idx, label_key) for d in train_days]
        train_X = np.concatenate([p[0] for p in train_parts])
        train_y = np.concatenate([p[1] for p in train_parts])

        test_X, test_y = build_horizon_data(test_day, horizon_idx, label_key)

        if len(train_X) < 100 or len(test_X) < 10:
            continue

        # Normalize
        mean = train_X.mean(axis=0)
        std = train_X.std(axis=0) + 1e-8
        train_X_n = (train_X - mean) / std
        test_X_n = (test_X - mean) / std

        # Train
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

        if np.std(oot_preds) > 1e-8:
            corr = np.corrcoef(oot_preds, test_y)[0, 1]
        else:
            corr = 0.0

        fold_corrs.append(float(corr))
        all_preds_list.append(oot_preds)
        all_actuals_list.append(test_y)

    if not all_preds_list:
        return None

    concat_preds = np.concatenate(all_preds_list)
    concat_actuals = np.concatenate(all_actuals_list)
    concat_corr = float(np.corrcoef(concat_preds, concat_actuals)[0, 1]) if len(concat_preds) > 1 else 0

    # Filter analysis
    baseline_pnl = concat_actuals.mean()
    filter_results = {}
    for pct in [100, 50, 30, 20, 10]:
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
        pf = wins / losses if losses > 0 else 999
        gross = mean_pnl + COMMISSION_RT
        filter_results[str(pct)] = {
            'n': len(sel), 'mean_pnl': float(mean_pnl), 'gross': float(gross),
            'wr': float(wr), 'pf': float(pf), 'lift': float(mean_pnl - baseline_pnl),
        }

    return {
        'horizon': horizon_name,
        'concat_corr': concat_corr,
        'mean_fold_corr': float(np.mean(fold_corrs)),
        'pos_folds': int(sum(1 for c in fold_corrs if c > 0)),
        'total_folds': len(fold_corrs),
        'total_trades': len(concat_actuals),
        'baseline_pnl': float(baseline_pnl),
        'filter_results': filter_results,
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
    print(f"Loaded {len(all_days)} days\n", flush=True)

    horizons = [
        ('1s', 0, 'label_1s'),
        ('5s', 1, 'label_5s'),
        ('10s', 2, 'label_10s'),
    ]

    results = {}
    for h_name, h_idx, h_label in horizons:
        print(f"{'='*70}")
        print(f"HORIZON: {h_name}")
        print(f"{'='*70}", flush=True)

        r = train_horizon(all_days, h_name, h_idx, h_label)
        if r is None:
            print(f"  SKIPPED — no valid folds")
            continue

        results[h_name] = r
        print(f"  Concat corr: {r['concat_corr']:+.4f}")
        print(f"  Pos folds: {r['pos_folds']}/{r['total_folds']}")
        print(f"  Total trades: {r['total_trades']}")
        print(f"  Baseline P&L: {r['baseline_pnl']:+.3f}")
        print(f"\n  Filter performance:")
        print(f"  {'Pct':>6} {'N':>7} {'MeanPnL':>8} {'Gross':>7} {'WR%':>6} {'PF':>6} {'Lift':>7}")
        for pct_str in ['100', '50', '30', '20', '10']:
            if pct_str in r['filter_results']:
                fr = r['filter_results'][pct_str]
                print(f"  {pct_str+'%':>6} {fr['n']:>7} {fr['mean_pnl']:>+7.3f} {fr['gross']:>+6.3f} "
                      f"{fr['wr']:>5.1f} {fr['pf']:>5.2f} {fr['lift']:>+6.3f}")
        print(flush=True)

    # Comparison
    print(f"\n{'='*70}")
    print(f"CROSS-HORIZON COMPARISON")
    print(f"{'='*70}")
    print(f"{'Horizon':>8} {'ConcCorr':>9} {'PosFolds':>9} {'Base':>7} {'Top10%':>8} {'Lift10%':>8} {'Gross10%':>9}")
    for h in ['1s', '5s', '10s']:
        if h in results:
            r = results[h]
            t10 = r['filter_results'].get('10', {})
            print(f"{h:>8} {r['concat_corr']:>+8.4f} {r['pos_folds']}/{r['total_folds']:>2} "
                  f"{r['baseline_pnl']:>+6.3f} {t10.get('mean_pnl',0):>+7.3f} "
                  f"{t10.get('lift',0):>+7.3f} {t10.get('gross',0):>+8.3f}")

    # Save
    with open(OUT_DIR / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT_DIR}", flush=True)


if __name__ == '__main__':
    main()
