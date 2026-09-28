#!/usr/bin/env python3
"""
Meta-Model Architecture Sweep v1
===================================
Tree-branch from confirmed meta-model wins.
Tests whether different MLP architectures improve the meta-model filter.

Uses 1s horizon (confirmed best) with top 3% shorts (balanced volume/edge).

Architectures tested:
  A) Baseline: 256→128→64 (current best, concat corr +0.111)
  B) Wider: 512→256→128
  C) Deeper: 256→128→64→32
  D) Shallow: 128→64
  E) XL: 512→256→128→64
  F) Residual: 256→128→64 with skip connections

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
OUT_DIR = _BASE / 'output' / 'meta_arch_sweep_v1'
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


class ResidualMLP(nn.Module):
    """MLP with residual connections between blocks."""
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])
        self.blocks = nn.ModuleList()
        for i in range(len(hidden_dims)):
            block = nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i]),
                nn.BatchNorm1d(hidden_dims[i]),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.blocks.append(block)
        # Projections for dimension changes
        self.projs = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.projs.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
        self.head = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x):
        x = self.input_proj(x)
        for i, block in enumerate(self.blocks):
            residual = x
            x = block(x)
            x = x + residual  # skip connection
            if i < len(self.projs):
                x = self.projs[i](x)
        return self.head(x).squeeze(-1)


ARCHITECTURES = {
    'baseline_256_128_64': {'cls': MetaMLP, 'dims': [256, 128, 64]},
    'wider_512_256_128': {'cls': MetaMLP, 'dims': [512, 256, 128]},
    'deeper_256_128_64_32': {'cls': MetaMLP, 'dims': [256, 128, 64, 32]},
    'shallow_128_64': {'cls': MetaMLP, 'dims': [128, 64]},
    'xl_512_256_128_64': {'cls': MetaMLP, 'dims': [512, 256, 128, 64]},
    'residual_256_128_64': {'cls': ResidualMLP, 'dims': [256, 128, 64]},
}


def load_day(pred_file):
    """Load one day."""
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

    # Select top 3% shorts by pred_1s
    pred_1s = preds[:, 0]
    threshold = np.percentile(pred_1s, SHORT_PERCENTILE)
    short_mask = pred_1s <= threshold

    feat_s = feat[short_mask]
    preds_s = preds[short_mask]
    label_1s_s = label_1s[short_mask]

    # P&L target (1s horizon)
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
    }


def train_arch(all_days, arch_name, arch_config):
    """Run full walk-forward for one architecture."""
    n_folds = len(all_days) - TRAIN_WINDOW
    all_preds_list = []
    all_actuals_list = []
    fold_corrs = []

    for fold_idx in range(n_folds):
        train_days = all_days[fold_idx:fold_idx + TRAIN_WINDOW]
        test_day = all_days[fold_idx + TRAIN_WINDOW]

        train_X = np.concatenate([d['features'] for d in train_days])
        train_y = np.concatenate([d['pnl_target'] for d in train_days])
        test_X = test_day['features']
        test_y = test_day['pnl_target']

        if len(train_X) < 100 or len(test_X) < 10:
            continue

        mean = train_X.mean(axis=0)
        std = train_X.std(axis=0) + 1e-8
        train_X_n = (train_X - mean) / std
        test_X_n = (test_X - mean) / std

        model_cls = arch_config['cls']
        model = model_cls(train_X_n.shape[1], arch_config['dims']).to(device)
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

    n_params = sum(p.numel() for p in model_cls(29, arch_config['dims']).parameters())

    return {
        'arch': arch_name,
        'n_params': n_params,
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

    results = {}
    for arch_name, arch_config in ARCHITECTURES.items():
        print(f"{'='*70}", flush=True)
        print(f"ARCHITECTURE: {arch_name}", flush=True)
        print(f"{'='*70}", flush=True)

        r = train_arch(all_days, arch_name, arch_config)
        if r is None:
            print(f"  SKIPPED", flush=True)
            continue

        results[arch_name] = r
        print(f"  Params: {r['n_params']:,}", flush=True)
        print(f"  Concat corr: {r['concat_corr']:+.4f}", flush=True)
        print(f"  Pos folds: {r['pos_folds']}/{r['total_folds']}", flush=True)
        print(f"\n  Filter performance:", flush=True)
        print(f"  {'Pct':>6} {'N':>7} {'MeanPnL':>8} {'Gross':>7} {'WR%':>6} {'PF':>6} {'Lift':>7}", flush=True)
        for pct_str in ['100', '50', '30', '20', '10']:
            if pct_str in r['filter_results']:
                fr = r['filter_results'][pct_str]
                print(f"  {pct_str+'%':>6} {fr['n']:>7} {fr['mean_pnl']:>+7.3f} {fr['gross']:>+6.3f} "
                      f"{fr['wr']:>5.1f} {fr['pf']:>5.2f} {fr['lift']:>+6.3f}", flush=True)
        print(flush=True)

    # Comparison
    print(f"\n{'='*70}", flush=True)
    print(f"ARCHITECTURE COMPARISON", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'Arch':>25} {'Params':>8} {'ConcCorr':>9} {'PosFolds':>9} {'Top30%':>8} {'Top10%':>8} {'Gross10':>8}", flush=True)
    for name in ARCHITECTURES:
        if name in results:
            r = results[name]
            t30 = r['filter_results'].get('30', {})
            t10 = r['filter_results'].get('10', {})
            print(f"{name:>25} {r['n_params']:>7,} {r['concat_corr']:>+8.4f} "
                  f"{r['pos_folds']}/{r['total_folds']:>2} {t30.get('mean_pnl',0):>+7.3f} "
                  f"{t10.get('mean_pnl',0):>+7.3f} {t10.get('gross',0):>+7.3f}", flush=True)

    with open(OUT_DIR / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT_DIR}", flush=True)


if __name__ == '__main__':
    main()
