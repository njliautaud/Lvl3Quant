#!/usr/bin/env python3
"""
Meta-Model Ensemble v1
========================
Train N=5 independent meta-models with different random seeds.
Average their predictions. Test if ensemble is more stable/profitable.

Uses confirmed best config: deeper 256→128→64→32, 1s horizon, top 3% shorts.
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
OUT_DIR = _BASE / 'output' / 'meta_ensemble_v1'
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
N_ENSEMBLE = 5
SEEDS = [42, 123, 456, 789, 2026]

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

    pred_1s = preds[:, 0]
    threshold = np.percentile(pred_1s, SHORT_PERCENTILE)
    short_mask = pred_1s <= threshold

    feat_s = feat[short_mask]
    preds_s = preds[short_mask]
    label_1s_s = label_1s[short_mask]

    short_pnl = -label_1s_s
    stop_hit = label_1s_s >= STOP_TICKS
    pnl_target = np.where(
        stop_hit, -(STOP_TICKS + STOP_SLIPPAGE + COMMISSION_RT), short_pnl - COMMISSION_RT
    ).astype(np.float32)

    ranks = np.argsort(np.argsort(preds_s[:, 0])).astype(np.float32) / max(len(preds_s), 1)
    features = np.column_stack([feat_s, preds_s[:, 0], preds_s[:, 1], preds_s[:, 2], ranks]).astype(np.float32)

    return {'date': date_str, 'features': features, 'pnl_target': pnl_target}


def train_single_model(train_X_n, train_y, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

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

    return model


def evaluate_preds(preds, actuals):
    if np.std(preds) < 1e-8:
        return {'corr': 0, 'filters': {}}

    corr = float(np.corrcoef(preds, actuals)[0, 1])
    baseline = actuals.mean()
    filters = {}
    for pct in [100, 50, 30, 20, 10]:
        if pct == 100:
            sel = actuals
        else:
            thresh = np.percentile(preds, 100 - pct)
            sel = actuals[preds >= thresh]
        if len(sel) == 0:
            continue
        mpnl = float(sel.mean())
        wr = float((sel > 0).mean() * 100)
        wins = sel[sel > 0].sum() if (sel > 0).any() else 0
        losses = abs(sel[sel < 0].sum()) if (sel < 0).any() else 1
        pf = float(wins / losses) if losses > 0 else 999
        filters[str(pct)] = {
            'n': len(sel), 'mean_pnl': mpnl, 'gross': mpnl + COMMISSION_RT,
            'wr': wr, 'pf': pf, 'lift': float(mpnl - baseline),
        }
    return {'corr': corr, 'filters': filters}


def main():
    print(f"Device: {device}", flush=True)
    print(f"Ensemble size: {N_ENSEMBLE}, seeds: {SEEDS}\n", flush=True)

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

    # Collect predictions from each ensemble member and from the ensemble average
    ensemble_preds_all = {i: [] for i in range(N_ENSEMBLE)}
    ensemble_avg_preds_all = []
    actuals_all = []

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

        fold_preds = []
        for i, seed in enumerate(SEEDS):
            model = train_single_model(train_X_n, train_y, seed)
            model.eval()
            with torch.no_grad():
                p = model(torch.from_numpy(test_X_n).to(device)).cpu().numpy()
            fold_preds.append(p)
            ensemble_preds_all[i].append(p)

        # Ensemble average
        avg_pred = np.mean(fold_preds, axis=0)
        ensemble_avg_preds_all.append(avg_pred)
        actuals_all.append(test_y)

        # Per-fold comparison
        single_corrs = [float(np.corrcoef(fp, test_y)[0, 1]) if np.std(fp) > 1e-8 else 0 for fp in fold_preds]
        avg_corr = float(np.corrcoef(avg_pred, test_y)[0, 1]) if np.std(avg_pred) > 1e-8 else 0
        best_single = max(single_corrs)

        print(f"Fold {fold_idx:2d} [{test_day['date']}]: "
              f"singles=[{', '.join(f'{c:+.3f}' for c in single_corrs)}] "
              f"avg={avg_corr:+.3f} {'✓ BETTER' if avg_corr > best_single else ''}", flush=True)

    # Concat analysis
    concat_actuals = np.concatenate(actuals_all)
    concat_avg = np.concatenate(ensemble_avg_preds_all)

    print(f"\n{'='*70}", flush=True)
    print(f"CONCAT RESULTS", flush=True)
    print(f"{'='*70}", flush=True)

    # Single model concat results
    results = {'singles': [], 'ensemble': None}
    for i in range(N_ENSEMBLE):
        concat_single = np.concatenate(ensemble_preds_all[i])
        ev = evaluate_preds(concat_single, concat_actuals)
        results['singles'].append({
            'seed': SEEDS[i], 'corr': ev['corr'], 'filters': ev['filters']
        })
        t30 = ev['filters'].get('30', {})
        t10 = ev['filters'].get('10', {})
        print(f"  Single (seed={SEEDS[i]:4d}): corr={ev['corr']:+.4f} "
              f"top30={t30.get('gross',0):+.3f} top10={t10.get('gross',0):+.3f}", flush=True)

    ev_avg = evaluate_preds(concat_avg, concat_actuals)
    results['ensemble'] = {'corr': ev_avg['corr'], 'filters': ev_avg['filters']}
    t30 = ev_avg['filters'].get('30', {})
    t10 = ev_avg['filters'].get('10', {})
    print(f"\n  ENSEMBLE AVG:       corr={ev_avg['corr']:+.4f} "
          f"top30={t30.get('gross',0):+.3f} top10={t10.get('gross',0):+.3f}", flush=True)

    # Detailed ensemble filter
    print(f"\n  Ensemble filter performance:", flush=True)
    print(f"  {'Pct':>6} {'N':>7} {'MeanPnL':>8} {'Gross':>7} {'WR%':>6} {'PF':>6} {'Lift':>7}", flush=True)
    for pct_str in ['100', '50', '30', '20', '10']:
        if pct_str in ev_avg['filters']:
            fr = ev_avg['filters'][pct_str]
            print(f"  {pct_str+'%':>6} {fr['n']:>7} {fr['mean_pnl']:>+7.3f} {fr['gross']:>+6.3f} "
                  f"{fr['wr']:>5.1f} {fr['pf']:>5.2f} {fr['lift']:>+6.3f}", flush=True)

    # Improvement over best single
    best_single_corr = max(r['corr'] for r in results['singles'])
    ensemble_improvement = ev_avg['corr'] - best_single_corr
    print(f"\n  Ensemble vs best single: {ensemble_improvement:+.4f} correlation improvement", flush=True)

    results['ensemble_improvement'] = ensemble_improvement
    results['n_folds'] = n_folds
    results['total_trades'] = len(concat_actuals)

    with open(OUT_DIR / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT_DIR}", flush=True)


if __name__ == '__main__':
    main()
