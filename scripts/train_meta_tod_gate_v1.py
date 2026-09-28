#!/usr/bin/env python3
"""
Meta-Model Time-of-Day Gating v1
====================================
Tree-branch: does the meta-model filter work differently at different times of day?

For each 30-min window (9:30-10:00, 10:00-10:30, ..., 15:30-16:00):
  - Analyze meta-model filter lift
  - Check if certain windows have significantly better or worse edge

If some windows are dead (no edge), we can gate them out.
If some windows are golden (extra edge), we can concentrate there.

Uses the full 48-date dataset on CPU.
Walk-forward: 10-date train, 1-date OOT.
Uses 1s horizon (confirmed best).
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
OUT_DIR = _BASE / 'output' / 'meta_tod_gate_v1'
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

# 30-min windows from 9:30 AM to 4:00 PM ET
# In nanoseconds-since-midnight: 9:30 = 34200e9, 16:00 = 57600e9
WINDOW_MINUTES = 30
RTH_START_NS = int(9.5 * 3600 * 1e9)   # 9:30 AM
RTH_END_NS = int(16 * 3600 * 1e9)      # 4:00 PM

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


def load_day_with_time(pred_file):
    """Load one day with timestamps for ToD analysis."""
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

    # Try to get timestamps
    if 'timestamps' in mbo:
        timestamps = mbo['timestamps']
    elif 'ts_event' in mbo:
        timestamps = mbo['ts_event']
    else:
        # Estimate from index position — assume events span RTH evenly
        n_events = len(events)
        timestamps = np.linspace(RTH_START_NS, RTH_END_NS, n_events)

    indices = np.arange(n_windows) * stride + (window_size - 1)
    max_idx = min(len(events), len(l1s), len(timestamps)) - 1
    valid = indices <= max_idx
    indices = indices[valid]
    preds = preds[:len(indices)]

    feat = events[indices]
    label_1s = l1s[indices]
    ts = timestamps[indices]

    valid_mask = ~(np.isnan(label_1s) | np.any(np.isnan(feat), axis=1))
    feat = feat[valid_mask]
    label_1s = label_1s[valid_mask]
    preds = preds[valid_mask]
    ts = ts[valid_mask]

    if len(feat) == 0:
        return None

    # Select top 3% shorts by pred_1s
    pred_1s = preds[:, 0]
    threshold = np.percentile(pred_1s, SHORT_PERCENTILE)
    short_mask = pred_1s <= threshold

    feat_s = feat[short_mask]
    preds_s = preds[short_mask]
    label_1s_s = label_1s[short_mask]
    ts_s = ts[short_mask]

    # P&L target
    short_pnl = -label_1s_s
    stop_hit = label_1s_s >= STOP_TICKS
    pnl_target = np.where(
        stop_hit,
        -(STOP_TICKS + STOP_SLIPPAGE + COMMISSION_RT),
        short_pnl - COMMISSION_RT,
    ).astype(np.float32)

    # Features
    ranks = np.argsort(np.argsort(preds_s[:, 0])).astype(np.float32) / max(len(preds_s), 1)
    features = np.column_stack([feat_s, preds_s[:, 0], preds_s[:, 1], preds_s[:, 2], ranks]).astype(np.float32)

    # Compute time-of-day bin for each trade
    # Convert timestamp to seconds-since-midnight
    if ts_s.max() > 1e15:  # nanoseconds
        ts_seconds = ts_s / 1e9
    elif ts_s.max() > 1e12:  # microseconds
        ts_seconds = ts_s / 1e6
    elif ts_s.max() > 1e9:  # milliseconds
        ts_seconds = ts_s / 1e3
    else:
        ts_seconds = ts_s  # already seconds

    # Modulo to get time within day (seconds since midnight)
    seconds_in_day = ts_seconds % 86400

    # Assign to 30-min bins
    tod_bin = ((seconds_in_day - 34200) / (WINDOW_MINUTES * 60)).astype(int)
    tod_bin = np.clip(tod_bin, 0, 12)  # 13 bins: 9:30-10:00 ... 15:30-16:00

    return {
        'date': date_str,
        'features': features,
        'pnl_target': pnl_target,
        'tod_bin': tod_bin,
    }


def main():
    print(f"Device: {device}", flush=True)

    pred_files = sorted([f for f in PRED_DIR.glob('*_predictions.npz') if '_stale_' not in str(f)])
    print(f"Found {len(pred_files)} prediction files", flush=True)

    all_days = []
    for i, pf in enumerate(pred_files):
        if i % 10 == 0:
            print(f"  Loading day {i+1}/{len(pred_files)}...", flush=True)
        day = load_day_with_time(pf)
        if day is not None:
            all_days.append(day)
    print(f"Loaded {len(all_days)} days\n", flush=True)

    # First: train meta-model normally (all times) and get OOT predictions
    n_folds = len(all_days) - TRAIN_WINDOW
    print(f"Walk-forward: {n_folds} folds", flush=True)

    all_meta_preds = []
    all_actual_pnl = []
    all_tod_bins = []

    for fold_idx in range(n_folds):
        if fold_idx % 5 == 0:
            print(f"  Fold {fold_idx+1}/{n_folds}...", flush=True)
        train_days = all_days[fold_idx:fold_idx + TRAIN_WINDOW]
        test_day = all_days[fold_idx + TRAIN_WINDOW]

        train_X = np.concatenate([d['features'] for d in train_days])
        train_y = np.concatenate([d['pnl_target'] for d in train_days])
        test_X = test_day['features']
        test_y = test_day['pnl_target']
        test_tod = test_day['tod_bin']

        if len(train_X) < 100 or len(test_X) < 10:
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

        all_meta_preds.append(oot_preds)
        all_actual_pnl.append(test_y)
        all_tod_bins.append(test_tod)

    concat_preds = np.concatenate(all_meta_preds)
    concat_pnl = np.concatenate(all_actual_pnl)
    concat_tod = np.concatenate(all_tod_bins)

    # Analyze by time-of-day bin
    bin_labels = [
        '09:30-10:00', '10:00-10:30', '10:30-11:00', '11:00-11:30',
        '11:30-12:00', '12:00-12:30', '12:30-13:00', '13:00-13:30',
        '13:30-14:00', '14:00-14:30', '14:30-15:00', '15:00-15:30',
        '15:30-16:00',
    ]

    print(f"\n{'='*90}", flush=True)
    print(f"TIME-OF-DAY ANALYSIS (all trades, no meta filter)", flush=True)
    print(f"{'='*90}", flush=True)
    print(f"{'Window':>14} {'N':>7} {'MeanPnL':>8} {'Gross':>7} {'WR%':>6} {'PF':>6} {'MetaCorr':>9}", flush=True)

    tod_results = {}
    for bin_idx in range(13):
        mask = concat_tod == bin_idx
        if mask.sum() < 20:
            continue
        pnl_bin = concat_pnl[mask]
        pred_bin = concat_preds[mask]

        mean_pnl = float(pnl_bin.mean())
        gross = mean_pnl + COMMISSION_RT
        wr = float((pnl_bin > 0).mean() * 100)
        wins = pnl_bin[pnl_bin > 0].sum() if (pnl_bin > 0).any() else 0
        losses = abs(pnl_bin[pnl_bin < 0].sum()) if (pnl_bin < 0).any() else 1
        pf = float(wins / losses) if losses > 0 else 999

        if np.std(pred_bin) > 1e-8:
            corr = float(np.corrcoef(pred_bin, pnl_bin)[0, 1])
        else:
            corr = 0.0

        label = bin_labels[bin_idx] if bin_idx < len(bin_labels) else f'bin_{bin_idx}'
        print(f"{label:>14} {mask.sum():>7} {mean_pnl:>+7.3f} {gross:>+6.3f} {wr:>5.1f} {pf:>5.2f} {corr:>+8.4f}", flush=True)

        tod_results[label] = {
            'n': int(mask.sum()), 'mean_pnl': mean_pnl, 'gross': gross,
            'wr': wr, 'pf': pf, 'meta_corr': corr,
        }

    # Analyze with meta filter by ToD
    print(f"\n{'='*90}", flush=True)
    print(f"TIME-OF-DAY ANALYSIS (meta top 30% filter)", flush=True)
    print(f"{'='*90}", flush=True)
    print(f"{'Window':>14} {'N':>7} {'MeanPnL':>8} {'Gross':>7} {'WR%':>6} {'PF':>6} {'Lift':>7}", flush=True)

    # Global top 30% threshold
    meta_thresh_30 = np.percentile(concat_preds, 70)
    meta_mask_30 = concat_preds >= meta_thresh_30

    tod_filtered = {}
    for bin_idx in range(13):
        mask = (concat_tod == bin_idx) & meta_mask_30
        mask_all = concat_tod == bin_idx
        if mask.sum() < 10 or mask_all.sum() < 10:
            continue
        pnl_bin = concat_pnl[mask]
        pnl_all = concat_pnl[mask_all]

        mean_pnl = float(pnl_bin.mean())
        gross = mean_pnl + COMMISSION_RT
        wr = float((pnl_bin > 0).mean() * 100)
        wins = pnl_bin[pnl_bin > 0].sum() if (pnl_bin > 0).any() else 0
        losses = abs(pnl_bin[pnl_bin < 0].sum()) if (pnl_bin < 0).any() else 1
        pf = float(wins / losses) if losses > 0 else 999
        lift = float(mean_pnl - pnl_all.mean())

        label = bin_labels[bin_idx] if bin_idx < len(bin_labels) else f'bin_{bin_idx}'
        print(f"{label:>14} {mask.sum():>7} {mean_pnl:>+7.3f} {gross:>+6.3f} {wr:>5.1f} {pf:>5.2f} {lift:>+6.3f}", flush=True)

        tod_filtered[label] = {
            'n': int(mask.sum()), 'mean_pnl': mean_pnl, 'gross': gross,
            'wr': wr, 'pf': pf, 'lift': lift,
        }

    # Summary
    print(f"\n{'='*90}", flush=True)
    print(f"GATING RECOMMENDATION", flush=True)
    print(f"{'='*90}", flush=True)

    # Find windows where filtered P&L is negative (should be gated out)
    gate_out = []
    gate_in = []
    for label, data in tod_filtered.items():
        if data['gross'] < 0.5:  # Less than 0.5 ticks gross — marginal
            gate_out.append(label)
        elif data['gross'] > 1.0:  # More than 1.0 ticks gross — golden
            gate_in.append(label)

    if gate_out:
        print(f"GATE OUT (gross < 0.5): {', '.join(gate_out)}", flush=True)
    else:
        print(f"No windows should be gated out (all gross > 0.5)", flush=True)

    if gate_in:
        print(f"GOLDEN WINDOWS (gross > 1.0): {', '.join(gate_in)}", flush=True)

    # Save
    results = {
        'unfiltered_by_tod': tod_results,
        'filtered_top30_by_tod': tod_filtered,
        'gate_out': gate_out,
        'gate_in': gate_in,
        'total_trades': len(concat_pnl),
        'n_folds': n_folds,
    }
    with open(OUT_DIR / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT_DIR}", flush=True)


if __name__ == '__main__':
    main()
