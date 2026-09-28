#!/usr/bin/env python3
"""
Meta-Model Threshold Sweep v1
================================
Tree-branch from multi-horizon finding: 1s horizon meta-model is best
(concat corr +0.114 vs +0.076 for 5s).

Tests: does signal selection threshold matter?
  - Top 1%, 2%, 3%, 5%, 7%, 10% shorts by pred_1s
  - For each: train MLP meta-model predicting realized P&L
  - Compare filter lift, trade count, and gross P&L

If tighter selection (1-2%) gives better meta-model lift per trade,
we trade fewer but higher-quality. If looser (5-10%) works, we get
more volume with acceptable edge.

Walk-forward: 10-date train, 1-date OOT.
Uses 1s horizon for both signal selection AND P&L target (confirmed best).
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
OUT_DIR = _BASE / 'output' / 'meta_threshold_sweep_v1'
OUT_DIR.mkdir(parents=True, exist_ok=True)

COMMISSION_RT = 0.376
STOP_TICKS = 2.0
STOP_SLIPPAGE = 1.0
TRAIN_WINDOW = 10
BATCH_SIZE = 4096
LR = 1e-3
EPOCHS = 25
WEIGHT_DECAY = 1e-4

THRESHOLDS = [1, 2, 3, 5, 7, 10]  # percentile for short selection

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
    """Load one day with all needed data."""
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

    return {
        'date': date_str,
        'features': feat.astype(np.float32),
        'preds': preds.astype(np.float32),
        'label_1s': label_1s.astype(np.float32),
        'label_5s': label_5s.astype(np.float32),
    }


def build_threshold_data(day, percentile):
    """Select top N% shorts by pred_1s (most negative) and build features+target."""
    pred_1s = day['preds'][:, 0]  # 1s horizon prediction
    threshold = np.percentile(pred_1s, percentile)
    short_mask = pred_1s <= threshold

    feat = day['features'][short_mask]
    preds = day['preds'][short_mask]
    label_1s = day['label_1s'][short_mask]

    # P&L target using 1s horizon (confirmed best)
    short_pnl = -label_1s
    stop_hit = label_1s >= STOP_TICKS
    pnl_target = np.where(
        stop_hit,
        -(STOP_TICKS + STOP_SLIPPAGE + COMMISSION_RT),
        short_pnl - COMMISSION_RT,
    ).astype(np.float32)

    # Extended features: 25 MBO + 3 preds + rank = 29
    ranks = np.argsort(np.argsort(preds[:, 0])).astype(np.float32) / max(len(preds), 1)
    features = np.column_stack([feat, preds[:, 0], preds[:, 1], preds[:, 2], ranks]).astype(np.float32)

    return features, pnl_target


def train_threshold(all_days, percentile):
    """Run full walk-forward for one threshold setting."""
    n_folds = len(all_days) - TRAIN_WINDOW

    all_preds_list = []
    all_actuals_list = []
    fold_corrs = []

    for fold_idx in range(n_folds):
        train_days = all_days[fold_idx:fold_idx + TRAIN_WINDOW]
        test_day = all_days[fold_idx + TRAIN_WINDOW]

        train_parts = [build_threshold_data(d, percentile) for d in train_days]
        train_X = np.concatenate([p[0] for p in train_parts])
        train_y = np.concatenate([p[1] for p in train_parts])

        test_X, test_y = build_threshold_data(test_day, percentile)

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
        'percentile': percentile,
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
    for pctl in THRESHOLDS:
        print(f"{'='*70}", flush=True)
        print(f"THRESHOLD: top {pctl}% shorts (pred_1s ≤ p{pctl})", flush=True)
        print(f"{'='*70}", flush=True)

        r = train_threshold(all_days, pctl)
        if r is None:
            print(f"  SKIPPED — no valid folds", flush=True)
            continue

        results[str(pctl)] = r
        print(f"  Concat corr: {r['concat_corr']:+.4f}", flush=True)
        print(f"  Pos folds: {r['pos_folds']}/{r['total_folds']}", flush=True)
        print(f"  Total trades: {r['total_trades']}", flush=True)
        print(f"  Baseline P&L: {r['baseline_pnl']:+.3f}", flush=True)
        print(f"\n  Filter performance:", flush=True)
        print(f"  {'Pct':>6} {'N':>7} {'MeanPnL':>8} {'Gross':>7} {'WR%':>6} {'PF':>6} {'Lift':>7}", flush=True)
        for pct_str in ['100', '50', '30', '20', '10']:
            if pct_str in r['filter_results']:
                fr = r['filter_results'][pct_str]
                print(f"  {pct_str+'%':>6} {fr['n']:>7} {fr['mean_pnl']:>+7.3f} {fr['gross']:>+6.3f} "
                      f"{fr['wr']:>5.1f} {fr['pf']:>5.2f} {fr['lift']:>+6.3f}", flush=True)
        print(flush=True)

    # Cross-threshold comparison
    print(f"\n{'='*70}", flush=True)
    print(f"CROSS-THRESHOLD COMPARISON", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'Pctl':>6} {'ConcCorr':>9} {'PosFolds':>9} {'Trades':>7} {'Base':>7} "
          f"{'Top30%':>8} {'Top10%':>8} {'Gross30':>8} {'Gross10':>8}", flush=True)
    for pctl_str in [str(p) for p in THRESHOLDS]:
        if pctl_str in results:
            r = results[pctl_str]
            t30 = r['filter_results'].get('30', {})
            t10 = r['filter_results'].get('10', {})
            print(f"{pctl_str+'%':>6} {r['concat_corr']:>+8.4f} {r['pos_folds']}/{r['total_folds']:>2} "
                  f"{r['total_trades']:>7} {r['baseline_pnl']:>+6.3f} "
                  f"{t30.get('mean_pnl',0):>+7.3f} {t10.get('mean_pnl',0):>+7.3f} "
                  f"{t30.get('gross',0):>+7.3f} {t10.get('gross',0):>+7.3f}", flush=True)

    # Trade volume × edge comparison (key metric)
    print(f"\n{'='*70}", flush=True)
    print(f"VOLUME × EDGE (daily estimate at top 30% filter)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'Pctl':>6} {'Trades/day':>10} {'Gross/trade':>11} {'DailyGross':>11}", flush=True)
    for pctl_str in [str(p) for p in THRESHOLDS]:
        if pctl_str in results:
            r = results[pctl_str]
            t30 = r['filter_results'].get('30', {})
            if t30:
                trades_per_day = t30['n'] / r['total_folds']
                gross = t30.get('gross', 0)
                daily_gross = trades_per_day * gross
                print(f"{pctl_str+'%':>6} {trades_per_day:>9.0f} {gross:>+10.3f} {daily_gross:>+10.1f} ticks", flush=True)

    # Save
    with open(OUT_DIR / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT_DIR}", flush=True)


if __name__ == '__main__':
    main()
