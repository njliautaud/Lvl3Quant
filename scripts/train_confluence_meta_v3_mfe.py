#!/usr/bin/env python3
"""
Confluence-Aware Meta-Model v3 — MFE Target
=============================================
Same MLP 256→128→64 as v2, but TARGET = mfe_5s (max favorable excursion
within 5 seconds). This is a richer target than raw P&L:
  - mfe_5s > 0 means price dropped (favorable for short) within 5s
  - High MFE = high-quality short opportunity regardless of exit timing
  - A trade with good MFE but negative P&L just needed a better exit

MFE/MAE labels come from pre-computed npz files (aligned 1:1 with predictions).
These are NOT available at inference time — used only as a training target.

FEATURES (29-dim): same as v2
  - 25 microstructure features from MBO events
  - pred_1s, pred_5s, pred_10s (CNN-Mamba v2 predictions)
  - signal_rank (percentile of pred_5s within day, 0-1)

Walk-forward: 10-date train, 1-date OOT (sliding window).
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# === CONSTANTS ===
SHORT_PERCENTILE = 3   # top 3% shorts by pred_5s
TRAIN_WINDOW = 10      # days

# Auto-detect platform
if sys.platform == 'win32':
    _BASE = Path(r'C:\Users\claude\Lvl3Quant')
else:
    _BASE = Path('/home/jupiter/Lvl3Quant')

PRED_DIR  = _BASE / 'output' / 'cnn_mamba_v2_bulk_oot_v2'
MBO_DIR   = _BASE / 'data' / 'processed' / 'mbo_events_smart_v3'
MFE_DIR   = _BASE / 'output' / 'mfe_mae_labels_v1'
OUT_DIR   = _BASE / 'output' / 'confluence_meta_v3_mfe'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Training hyper-params
BATCH_SIZE   = 4096
LR           = 1e-3
EPOCHS       = 30
HIDDEN_DIMS  = [256, 128, 64]
DROPOUT      = 0.2
WEIGHT_DECAY = 1e-4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MetaMLP(nn.Module):
    """MLP predicting MFE_5s from microstructure features."""
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


def load_day(pred_file):
    """Load one day: align MBO features + MFE labels + predictions."""
    pred_data = np.load(pred_file, allow_pickle=True)
    date_str  = str(pred_data['date'])
    preds     = pred_data['predictions']   # (n_windows, 3) → [1s, 5s, 10s]
    n_windows = int(pred_data['n_windows'])
    window_size = int(pred_data['window_size'])
    stride    = int(pred_data['stride'])

    mbo_file = MBO_DIR / f'{date_str}_mbo_events.npz'
    mfe_file = MFE_DIR / f'{date_str}_mfe_mae.npz'

    if not mbo_file.exists():
        print(f"  [SKIP] No MBO file for {date_str}", flush=True)
        return None
    if not mfe_file.exists():
        print(f"  [SKIP] No MFE file for {date_str}", flush=True)
        return None

    mbo = np.load(mbo_file, allow_pickle=True)
    events = mbo['events']           # (N, 25)

    mfe_data = np.load(mfe_file, allow_pickle=True)
    mfe_5s   = mfe_data['mfe_5s']   # (n_valid,) — already aligned with predictions
    n_valid  = int(mfe_data['n_valid'])

    # Build indices: same logic as prediction generation
    indices = np.arange(n_windows) * stride + (window_size - 1)
    max_idx = len(events) - 1
    valid_mask = indices <= max_idx
    indices = indices[valid_mask]
    preds   = preds[:len(indices)]

    # MFE labels are indexed [0..n_valid-1] aligned with predictions
    # Trim to common length
    n_common = min(len(indices), len(preds), len(mfe_5s))
    if n_common == 0:
        return None

    indices = indices[:n_common]
    preds   = preds[:n_common]
    mfe_5s  = mfe_5s[:n_common]

    feat = events[indices]   # (n_common, 25)

    # Remove NaN rows
    nan_mask = (
        np.isnan(mfe_5s) |
        np.any(np.isnan(feat), axis=1) |
        np.any(np.isnan(preds), axis=1)
    )
    keep = ~nan_mask
    feat   = feat[keep]
    preds  = preds[keep]
    mfe_5s = mfe_5s[keep]

    if len(feat) == 0:
        return None

    # Build feature matrix: 25 MBO + pred_1s + pred_5s + pred_10s + rank = 29
    pred_1s  = preds[:, 0]
    pred_5s  = preds[:, 1]
    pred_10s = preds[:, 2]

    # Signal rank: percentile of pred_5s within day (0=strongest short)
    ranks = np.argsort(np.argsort(pred_5s)).astype(np.float32) / max(len(pred_5s) - 1, 1)

    features = np.column_stack([
        feat,     # 25 MBO
        pred_1s,
        pred_5s,
        pred_10s,
        ranks,
    ]).astype(np.float32)  # (n, 29)

    # Target: mfe_5s (positive = favorable move for short within 5s)
    # For shorts: mfe_5s is positive when price drops (good for us)
    target = mfe_5s.astype(np.float32)

    # Select top 3% shorts
    threshold   = np.percentile(pred_5s, SHORT_PERCENTILE)
    short_mask  = pred_5s <= threshold

    return {
        'date':         date_str,
        'features':     features[short_mask],
        'target':       target[short_mask],
        'pred_5s':      pred_5s,
        'short_mask':   short_mask,
        'n_total':      len(feat),
    }


def normalize_features(train_feats, test_feats):
    """Z-score normalize using train statistics."""
    mean = train_feats.mean(axis=0)
    std  = train_feats.std(axis=0) + 1e-8
    return (train_feats - mean) / std, (test_feats - mean) / std, mean, std


def train_fold(train_days, test_day, fold_idx):
    """Train one WF fold; return OOT predictions vs actuals (MFE_5s)."""
    train_X = np.concatenate([d['features'] for d in train_days])
    train_y = np.concatenate([d['target']   for d in train_days])
    test_X  = test_day['features']
    test_y  = test_day['target']

    if len(train_X) < 100 or len(test_X) < 10:
        return None

    train_X_norm, test_X_norm, _, _ = normalize_features(train_X, test_X)

    train_ds = TensorDataset(
        torch.from_numpy(train_X_norm),
        torch.from_numpy(train_y),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=(device.type == 'cuda'))

    input_dim = train_X_norm.shape[1]
    model     = MetaMLP(input_dim, HIDDEN_DIMS, DROPOUT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.MSELoss()

    final_loss = 0.0
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        n_batches  = 0
        for X_b, y_b in train_loader:
            X_b = X_b.to(device)
            y_b = y_b.to(device)
            optimizer.zero_grad()
            pred = model(X_b)
            loss = criterion(pred, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1
        scheduler.step()
        final_loss = total_loss / max(n_batches, 1)

    # OOT evaluation
    model.eval()
    with torch.no_grad():
        oot_preds = model(torch.from_numpy(test_X_norm).to(device)).cpu().numpy()

    corr = float(np.corrcoef(oot_preds, test_y)[0, 1]) \
           if len(oot_preds) > 1 and np.std(oot_preds) > 1e-8 else 0.0

    # Top-50% filter: does high predicted MFE → high actual MFE?
    top_mean = bot_mean = separation = 0.0
    if len(oot_preds) >= 20:
        median_pred = np.median(oot_preds)
        top_half    = test_y[oot_preds >= median_pred]
        bot_half    = test_y[oot_preds <  median_pred]
        top_mean    = float(top_half.mean()) if len(top_half) > 0 else 0.0
        bot_mean    = float(bot_half.mean()) if len(bot_half) > 0 else 0.0
        separation  = top_mean - bot_mean

    return {
        'fold':             fold_idx,
        'date':             test_day['date'],
        'n_train':          len(train_X),
        'n_test':           len(test_X),
        'corr':             corr,
        'top_half_mean_mfe': top_mean,
        'bot_half_mean_mfe': bot_mean,
        'separation':       separation,
        'train_loss':       final_loss,
        'oot_preds':        oot_preds,
        'oot_actuals':      test_y,
    }


def main():
    print(f"Confluence Meta v3 — MFE Target", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"MFE labels dir: {MFE_DIR}", flush=True)
    print(f"Predictions:    {PRED_DIR}", flush=True)
    print(f"Output:         {OUT_DIR}", flush=True)

    pred_files = sorted([
        f for f in PRED_DIR.glob('*_predictions.npz')
        if '_stale_' not in str(f)
    ])
    print(f"\nFound {len(pred_files)} prediction files", flush=True)

    all_days = []
    for i, pf in enumerate(pred_files):
        if i % 10 == 0:
            print(f"  Loading day {i+1}/{len(pred_files)}...", flush=True)
        day = load_day(pf)
        if day is not None:
            all_days.append(day)

    print(f"\nLoaded {len(all_days)} days with valid MFE data", flush=True)
    total_shorts = sum(len(d['features']) for d in all_days)
    print(f"Total top-3% short trades: {total_shorts}", flush=True)

    if len(all_days) < TRAIN_WINDOW + 1:
        print(f"ERROR: Need ≥{TRAIN_WINDOW + 1} days, have {len(all_days)}", flush=True)
        return

    results        = []
    all_oot_preds  = []
    all_oot_actuals = []

    n_folds = len(all_days) - TRAIN_WINDOW
    print(f"\n{'='*80}", flush=True)
    print(f"WALK-FORWARD: {n_folds} folds (train={TRAIN_WINDOW} days, OOT=1 day)", flush=True)
    print(f"TARGET: mfe_5s (ticks favorable for short within 5s)", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"{'Fold':>5} {'Date':>10} {'N_test':>7} {'Corr':>7} {'Top50%MFE':>10} {'Bot50%MFE':>10} {'Sep':>7}", flush=True)
    print(f"{'-'*65}", flush=True)

    for fold_idx in range(n_folds):
        train_days = all_days[fold_idx : fold_idx + TRAIN_WINDOW]
        test_day   = all_days[fold_idx + TRAIN_WINDOW]

        result = train_fold(train_days, test_day, fold_idx)
        if result is None:
            continue

        results.append(result)
        all_oot_preds.append(result['oot_preds'])
        all_oot_actuals.append(result['oot_actuals'])

        print(f"{fold_idx:>5} {result['date']:>10} {result['n_test']:>7} "
              f"{result['corr']:>+7.3f} {result['top_half_mean_mfe']:>+9.3f} "
              f"{result['bot_half_mean_mfe']:>+9.3f} {result['separation']:>+6.3f}", flush=True)

    if not results:
        print("No valid folds — check data alignment.", flush=True)
        return

    # === AGGREGATE METRICS ===
    concat_preds   = np.concatenate(all_oot_preds)
    concat_actuals = np.concatenate(all_oot_actuals)
    concat_corr    = float(np.corrcoef(concat_preds, concat_actuals)[0, 1]) \
                     if len(concat_preds) > 1 else 0.0

    per_fold_corrs = [r['corr'] for r in results]
    per_fold_seps  = [r['separation'] for r in results]

    print(f"\n{'='*80}", flush=True)
    print(f"AGGREGATE RESULTS", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"Concat correlation (pred_MFE5s vs actual_MFE5s): {concat_corr:+.4f}", flush=True)
    print(f"Mean per-fold corr: {np.mean(per_fold_corrs):+.4f} ± {np.std(per_fold_corrs):.4f}", flush=True)
    print(f"Mean separation:    {np.mean(per_fold_seps):+.4f} ticks MFE", flush=True)
    print(f"Positive-corr folds: {sum(1 for c in per_fold_corrs if c > 0)}/{len(per_fold_corrs)}", flush=True)
    print(f"Total OOT trades:   {len(concat_preds)}", flush=True)

    # === FILTER ANALYSIS ===
    print(f"\n{'='*80}", flush=True)
    print(f"FILTER ANALYSIS: high predicted MFE_5s → select trades", flush=True)
    print(f"Baseline = mean MFE_5s across all top-3% shorts", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"{'Filter':>12} {'N':>7} {'MeanMFE5s':>10} {'MFE>0%':>8} {'Improvement':>12}", flush=True)
    print(f"{'-'*55}", flush=True)

    baseline_mfe = concat_actuals.mean()
    baseline_pos = (concat_actuals > 0).mean() * 100

    for label, pct in [('All (base)', 100), ('Top 70%', 70), ('Top 50%', 50),
                       ('Top 30%', 30), ('Top 20%', 20), ('Top 10%', 10)]:
        if pct == 100:
            sel = concat_actuals
        else:
            thresh = np.percentile(concat_preds, 100 - pct)
            sel    = concat_actuals[concat_preds >= thresh]

        if len(sel) == 0:
            continue

        mean_mfe   = sel.mean()
        pos_rate   = (sel > 0).mean() * 100
        improvement = mean_mfe - baseline_mfe

        print(f"{label:>12} {len(sel):>7} {mean_mfe:>+9.3f} {pos_rate:>7.1f}% {improvement:>+11.3f}", flush=True)

    # === SAVE ===
    summary = {
        'concat_corr':          concat_corr,
        'mean_per_fold_corr':   float(np.mean(per_fold_corrs)),
        'std_per_fold_corr':    float(np.std(per_fold_corrs)),
        'mean_separation_ticks': float(np.mean(per_fold_seps)),
        'positive_corr_folds':  int(sum(1 for c in per_fold_corrs if c > 0)),
        'total_folds':          len(results),
        'total_oot_trades':     int(len(concat_preds)),
        'baseline_mean_mfe_5s': float(baseline_mfe),
        'model_arch':           f'MLP {HIDDEN_DIMS}',
        'features':             '25_mbo + pred_1s + pred_5s + pred_10s + rank = 29',
        'target':               'mfe_5s_ticks',
        'verdict':              'PASS' if concat_corr > 0.05 else ('WEAK' if concat_corr > 0 else 'FAIL'),
        'per_fold': [
            {k: v for k, v in r.items() if k not in ('oot_preds', 'oot_actuals')}
            for r in results
        ],
    }

    with open(OUT_DIR / 'results.json', 'w') as f:
        json.dump(summary, f, indent=2)

    np.savez_compressed(
        OUT_DIR / 'oot_predictions.npz',
        predictions=concat_preds,
        actuals=concat_actuals,
    )

    print(f"\nResults saved to {OUT_DIR}", flush=True)
    verdict = summary['verdict']
    print(f"\nVERDICT: {verdict}  (concat_corr={concat_corr:+.4f}, threshold=0.05)", flush=True)
    print(f"Done.", flush=True)


if __name__ == '__main__':
    main()
