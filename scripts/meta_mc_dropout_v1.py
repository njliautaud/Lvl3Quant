#!/usr/bin/env python3
"""
MC Dropout Uncertainty Estimation — Meta-Model v1
====================================================
TREE BRANCH experiment off production meta-model.

Uses MC Dropout at INFERENCE time to get uncertainty estimates from the
existing production model weights. No retraining needed.

Method:
  - Load each fold's saved weights (model_state, norm_mean, norm_std, input_dim)
  - For the OOT test day, run N=30 forward passes with dropout enabled
  - Compute per-trade: mean prediction, std (uncertainty), CV
  - Test whether filtering by LOW uncertainty improves meta-filter quality
  - Compare to baseline production top 30% gross = +1.02 ticks

Key insight: model.train() keeps Dropout active during forward pass,
giving stochastic outputs. The variance across passes = model uncertainty.
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

if sys.platform == 'win32':
    _BASE = Path(r'C:\Users\claude\Lvl3Quant')
else:
    _BASE = Path('/home/jupiter/Lvl3Quant')

PRED_DIR = _BASE / 'output' / 'cnn_mamba_v2_bulk_oot_v2'
MBO_DIR = _BASE / 'data' / 'processed' / 'mbo_events_smart_v3'
WEIGHTS_DIR = _BASE / 'output' / 'meta_production_v1' / 'weights'
OUT_DIR = _BASE / 'output' / 'meta_mc_dropout_v1'
OUT_DIR.mkdir(parents=True, exist_ok=True)

COMMISSION_RT = 0.376
STOP_TICKS = 2.0
STOP_SLIPPAGE = 1.0
SHORT_PERCENTILE = 3

N_MC_PASSES = 30

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ProductionMetaMLP(nn.Module):
    """Deeper MLP: 256->128->64->32 (arch sweep winner)."""
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
    """Load one day: select top 3% shorts, compute 1s P&L target, build 29-dim features."""
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


def compute_metrics(pnl_arr):
    """Compute mean P&L, WR, PF, gross ticks for an array of P&L values."""
    if len(pnl_arr) == 0:
        return {'n': 0, 'mean_pnl': 0, 'wr': 0, 'pf': 0, 'gross': 0}
    mean_pnl = float(pnl_arr.mean())
    wr = float((pnl_arr > 0).mean() * 100)
    wins = pnl_arr[pnl_arr > 0].sum() if (pnl_arr > 0).any() else 0
    losses_val = abs(pnl_arr[pnl_arr < 0].sum()) if (pnl_arr < 0).any() else 1e-8
    pf = float(wins / losses_val) if losses_val > 0 else 999
    gross = mean_pnl + COMMISSION_RT
    return {'n': len(pnl_arr), 'mean_pnl': mean_pnl, 'wr': wr, 'pf': pf, 'gross': gross}


def mc_dropout_forward(model, X_tensor, n_passes):
    """Run N forward passes with dropout enabled, return (n_samples, n_passes) predictions."""
    model.train()  # Enable dropout
    all_passes = []
    with torch.no_grad():
        for _ in range(n_passes):
            preds = model(X_tensor).cpu().numpy()
            all_passes.append(preds)
    return np.stack(all_passes, axis=1)  # (n_samples, n_passes)


def main():
    print(f"{'='*70}", flush=True)
    print(f"MC DROPOUT UNCERTAINTY ESTIMATION — Meta-Model v1", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"MC passes: {N_MC_PASSES}", flush=True)
    print(f"Baseline target: top 30% gross > +1.02 ticks", flush=True)
    print(f"{'='*70}\n", flush=True)

    # Discover saved fold weights
    weight_files = sorted(WEIGHTS_DIR.glob('fold_*.pt'))
    print(f"Found {len(weight_files)} fold weight files", flush=True)
    if len(weight_files) == 0:
        print("ERROR: No weight files found. Run train_meta_production_v1.py first.", flush=True)
        return

    # Load all prediction files (needed to reconstruct test data for each fold)
    pred_files = sorted([f for f in PRED_DIR.glob('*_predictions.npz') if '_stale_' not in str(f)])
    print(f"Found {len(pred_files)} prediction files", flush=True)

    all_days = []
    for i, pf in enumerate(pred_files):
        if i % 10 == 0:
            print(f"  Loading day {i+1}/{len(pred_files)}...", flush=True)
        day = load_day(pf)
        if day is not None:
            all_days.append(day)
    print(f"Loaded {len(all_days)} valid days\n", flush=True)

    # Build date-to-index lookup
    date_to_idx = {d['date']: i for i, d in enumerate(all_days)}

    # Process each fold
    TRAIN_WINDOW = 10  # Same as production

    # Collect concat-level arrays
    concat_mc_mean = []
    concat_mc_std = []
    concat_actuals = []
    concat_det_preds = []  # Deterministic (eval mode) predictions for baseline comparison

    fold_results = []

    for wf in weight_files:
        # Load checkpoint
        ckpt = torch.load(wf, map_location=device, weights_only=False)
        fold_idx = ckpt['fold_idx']
        test_date = str(ckpt['test_date'])
        input_dim = ckpt['input_dim']
        norm_mean = ckpt['norm_mean']
        norm_std = ckpt['norm_std']

        # Find test day data
        if test_date not in date_to_idx:
            print(f"  Fold {fold_idx:2d} [{test_date}]: SKIP (test date not in loaded days)", flush=True)
            continue

        test_day = all_days[date_to_idx[test_date]]
        test_X = test_day['features']
        test_y = test_day['pnl_target']

        if len(test_X) < 10:
            print(f"  Fold {fold_idx:2d} [{test_date}]: SKIP (only {len(test_X)} trades)", flush=True)
            continue

        # Normalize using saved normalizer
        test_X_n = ((test_X - norm_mean) / norm_std).astype(np.float32)

        # Build model, load weights
        model = ProductionMetaMLP(input_dim).to(device)
        model.load_state_dict(ckpt['model_state'])

        X_tensor = torch.from_numpy(test_X_n).to(device)

        # Deterministic prediction (eval mode, no dropout)
        model.eval()
        with torch.no_grad():
            det_preds = model(X_tensor).cpu().numpy()

        # MC Dropout forward passes
        mc_preds = mc_dropout_forward(model, X_tensor, N_MC_PASSES)  # (n_samples, N_MC_PASSES)

        mc_mean = mc_preds.mean(axis=1)
        mc_std = mc_preds.std(axis=1)
        mc_cv = np.where(np.abs(mc_mean) > 1e-8, mc_std / np.abs(mc_mean), 0.0)

        # Per-fold analysis
        n_trades = len(test_y)
        baseline_metrics = compute_metrics(test_y)

        # Deterministic top 30% filter (baseline)
        det_top30_thresh = np.percentile(det_preds, 70)
        det_top30_mask = det_preds >= det_top30_thresh
        det_top30_metrics = compute_metrics(test_y[det_top30_mask])

        # Uncertainty filters: keep low-std trades (high confidence)
        uncertainty_results = {}
        for pctl in [10, 20, 30, 50]:
            std_thresh = np.percentile(mc_std, pctl)
            low_unc_mask = mc_std <= std_thresh
            unc_metrics = compute_metrics(test_y[low_unc_mask])
            uncertainty_results[f'unc_p{pctl}'] = unc_metrics

            # Combined: meta score top 30% AND low uncertainty
            combined_mask = det_top30_mask & low_unc_mask
            combined_metrics = compute_metrics(test_y[combined_mask])
            uncertainty_results[f'combined_top30_unc_p{pctl}'] = combined_metrics

        fold_data = {
            'fold': fold_idx,
            'date': test_date,
            'n_trades': n_trades,
            'baseline': baseline_metrics,
            'det_top30': det_top30_metrics,
            'mc_std_mean': float(mc_std.mean()),
            'mc_std_median': float(np.median(mc_std)),
            'mc_cv_mean': float(mc_cv.mean()),
            'uncertainty': uncertainty_results,
        }
        fold_results.append(fold_data)

        # Collect for concat
        concat_mc_mean.append(mc_mean)
        concat_mc_std.append(mc_std)
        concat_actuals.append(test_y)
        concat_det_preds.append(det_preds)

        # Print fold summary
        best_unc_key = None
        best_unc_gross = -999
        for k, v in uncertainty_results.items():
            if k.startswith('combined_') and v['n'] > 0 and v['gross'] > best_unc_gross:
                best_unc_gross = v['gross']
                best_unc_key = k

        best_tag = f" best_combined={best_unc_gross:+.3f}" if best_unc_key else ""
        print(f"  Fold {fold_idx:2d} [{test_date}]: n={n_trades:4d}  "
              f"det_top30={det_top30_metrics['gross']:+.3f}  "
              f"mc_std_avg={mc_std.mean():.4f}{best_tag}", flush=True)

    if len(concat_actuals) == 0:
        print("\nERROR: No folds processed successfully.", flush=True)
        return

    # ==========================================================================
    # CONCAT ANALYSIS
    # ==========================================================================
    concat_mc_mean = np.concatenate(concat_mc_mean)
    concat_mc_std = np.concatenate(concat_mc_std)
    concat_actuals = np.concatenate(concat_actuals)
    concat_det_preds = np.concatenate(concat_det_preds)
    n_total = len(concat_actuals)

    print(f"\n{'='*70}", flush=True)
    print(f"CONCAT RESULTS ({n_total} trades across {len(fold_results)} folds)", flush=True)
    print(f"{'='*70}", flush=True)

    # Baseline metrics
    baseline_all = compute_metrics(concat_actuals)
    print(f"\nBaseline (all trades): mean={baseline_all['mean_pnl']:+.3f}  "
          f"gross={baseline_all['gross']:+.3f}  wr={baseline_all['wr']:.1f}%  pf={baseline_all['pf']:.2f}", flush=True)

    # Deterministic top 30% (production baseline to beat)
    det_top30_thresh = np.percentile(concat_det_preds, 70)
    det_top30_mask = concat_det_preds >= det_top30_thresh
    det_top30 = compute_metrics(concat_actuals[det_top30_mask])
    print(f"Det top 30%:           mean={det_top30['mean_pnl']:+.3f}  "
          f"gross={det_top30['gross']:+.3f}  wr={det_top30['wr']:.1f}%  pf={det_top30['pf']:.2f}  n={det_top30['n']}", flush=True)

    # MC Dropout uncertainty statistics
    print(f"\n--- MC Dropout Uncertainty Distribution ---", flush=True)
    for pctl in [10, 25, 50, 75, 90]:
        print(f"  MC std p{pctl}: {np.percentile(concat_mc_std, pctl):.5f}", flush=True)
    print(f"  MC std mean: {concat_mc_std.mean():.5f}", flush=True)

    # Uncertainty-only filters
    print(f"\n--- Uncertainty-Only Filters (keep low-uncertainty trades) ---", flush=True)
    print(f"  {'Filter':>12} {'N':>7} {'MeanPnL':>8} {'Gross':>7} {'WR%':>6} {'PF':>6} {'vs_base':>8}", flush=True)
    unc_only_results = {}
    for pctl in [10, 20, 30, 50, 75, 100]:
        if pctl == 100:
            mask = np.ones(n_total, dtype=bool)
            label = 'all'
        else:
            std_thresh = np.percentile(concat_mc_std, pctl)
            mask = concat_mc_std <= std_thresh
            label = f'unc_p{pctl}'

        m = compute_metrics(concat_actuals[mask])
        delta = m['gross'] - baseline_all['gross']
        unc_only_results[label] = m
        print(f"  {label:>12} {m['n']:>7} {m['mean_pnl']:>+7.3f} {m['gross']:>+6.3f} "
              f"{m['wr']:>5.1f} {m['pf']:>5.2f} {delta:>+7.3f}", flush=True)

    # Det top 30% + uncertainty filters (the key test)
    print(f"\n--- Combined: Det Top 30% + Low Uncertainty ---", flush=True)
    print(f"  {'Filter':>20} {'N':>7} {'MeanPnL':>8} {'Gross':>7} {'WR%':>6} {'PF':>6} {'vs_det30':>8}", flush=True)
    combined_results = {}
    for pctl in [10, 20, 30, 50, 75]:
        std_thresh = np.percentile(concat_mc_std, pctl)
        combined_mask = det_top30_mask & (concat_mc_std <= std_thresh)
        label = f'top30+unc_p{pctl}'
        m = compute_metrics(concat_actuals[combined_mask])
        delta = m['gross'] - det_top30['gross']
        combined_results[label] = m
        print(f"  {label:>20} {m['n']:>7} {m['mean_pnl']:>+7.3f} {m['gross']:>+6.3f} "
              f"{m['wr']:>5.1f} {m['pf']:>5.2f} {delta:>+7.3f}", flush=True)

    # Also test: MC mean top 30% (using MC mean instead of deterministic score)
    print(f"\n--- MC Mean Score Filters (replace det score with MC mean) ---", flush=True)
    print(f"  {'Filter':>20} {'N':>7} {'MeanPnL':>8} {'Gross':>7} {'WR%':>6} {'PF':>6} {'vs_det30':>8}", flush=True)
    for pct in [30, 20, 10]:
        mc_thresh = np.percentile(concat_mc_mean, 100 - pct)
        mc_mask = concat_mc_mean >= mc_thresh
        label = f'mc_mean_top{pct}%'
        m = compute_metrics(concat_actuals[mc_mask])
        delta = m['gross'] - det_top30['gross']
        print(f"  {label:>20} {m['n']:>7} {m['mean_pnl']:>+7.3f} {m['gross']:>+6.3f} "
              f"{m['wr']:>5.1f} {m['pf']:>5.2f} {delta:>+7.3f}", flush=True)

    # MC mean top 30% + low uncertainty
    print(f"\n--- MC Mean Top 30% + Low Uncertainty ---", flush=True)
    print(f"  {'Filter':>25} {'N':>7} {'MeanPnL':>8} {'Gross':>7} {'WR%':>6} {'PF':>6} {'vs_det30':>8}", flush=True)
    mc_top30_thresh = np.percentile(concat_mc_mean, 70)
    mc_top30_mask = concat_mc_mean >= mc_top30_thresh
    mc_combined_results = {}
    for pctl in [10, 20, 30, 50]:
        std_thresh = np.percentile(concat_mc_std, pctl)
        combined_mask = mc_top30_mask & (concat_mc_std <= std_thresh)
        label = f'mc_top30+unc_p{pctl}'
        m = compute_metrics(concat_actuals[combined_mask])
        delta = m['gross'] - det_top30['gross']
        mc_combined_results[label] = m
        print(f"  {label:>25} {m['n']:>7} {m['mean_pnl']:>+7.3f} {m['gross']:>+6.3f} "
              f"{m['wr']:>5.1f} {m['pf']:>5.2f} {delta:>+7.3f}", flush=True)

    # ==========================================================================
    # VERDICT
    # ==========================================================================
    print(f"\n{'='*70}", flush=True)
    print(f"VERDICT", flush=True)
    print(f"{'='*70}", flush=True)

    baseline_gross = 1.02  # Production baseline: top 30% gross
    det_top30_gross = det_top30['gross']

    # Check if ANY combined filter beats baseline
    best_filter = None
    best_gross = -999
    best_label = ""

    # Check det combined
    for label, m in combined_results.items():
        if m['n'] >= 20 and m['gross'] > best_gross:  # Minimum 20 trades for significance
            best_gross = m['gross']
            best_label = label
            best_filter = m

    # Check mc combined
    for label, m in mc_combined_results.items():
        if m['n'] >= 20 and m['gross'] > best_gross:
            best_gross = m['gross']
            best_label = label
            best_filter = m

    passed = best_gross > baseline_gross and best_gross > det_top30_gross

    print(f"\n  Production baseline (top 30% gross): +{baseline_gross:.2f} ticks", flush=True)
    print(f"  This run det top 30% gross:          {det_top30_gross:+.3f} ticks", flush=True)
    if best_filter:
        print(f"  Best combined filter:                {best_label}", flush=True)
        print(f"    Gross: {best_gross:+.3f}  N={best_filter['n']}  WR={best_filter['wr']:.1f}%  PF={best_filter['pf']:.2f}", flush=True)
        print(f"    Improvement over det top 30%:      {best_gross - det_top30_gross:+.3f} ticks", flush=True)
    else:
        print(f"  No combined filter with >= 20 trades found.", flush=True)

    if passed:
        print(f"\n  >>> PASS: MC Dropout uncertainty filtering improves meta-filter quality <<<", flush=True)
        print(f"  >>> Best filter {best_label}: gross {best_gross:+.3f} > baseline {baseline_gross:+.2f} <<<", flush=True)
    else:
        print(f"\n  >>> FAIL: MC Dropout uncertainty filtering does NOT beat baseline <<<", flush=True)
        if best_filter:
            print(f"  >>> Best was {best_label}: gross {best_gross:+.3f} vs baseline {baseline_gross:+.2f} <<<", flush=True)

    # ==========================================================================
    # SAVE RESULTS
    # ==========================================================================
    results = {
        'config': {
            'n_mc_passes': N_MC_PASSES,
            'model': 'ProductionMetaMLP 256->128->64->32',
            'dropout': 0.2,
            'baseline_gross_target': baseline_gross,
            'device': str(device),
        },
        'concat': {
            'n_total': n_total,
            'n_folds': len(fold_results),
            'baseline_all': baseline_all,
            'det_top30': det_top30,
            'mc_std_stats': {
                'mean': float(concat_mc_std.mean()),
                'median': float(np.median(concat_mc_std)),
                'p10': float(np.percentile(concat_mc_std, 10)),
                'p90': float(np.percentile(concat_mc_std, 90)),
            },
        },
        'uncertainty_only': {k: v for k, v in unc_only_results.items()},
        'combined_det': {k: v for k, v in combined_results.items()},
        'combined_mc': {k: v for k, v in mc_combined_results.items()},
        'verdict': 'PASS' if passed else 'FAIL',
        'best_filter': best_label if best_filter else None,
        'best_gross': best_gross if best_filter else None,
        'per_fold': fold_results,
        'timestamp': datetime.now().isoformat(),
    }

    np.savez(OUT_DIR / 'mc_dropout_predictions.npz',
             mc_mean=concat_mc_mean,
             mc_std=concat_mc_std,
             det_preds=concat_det_preds,
             actuals=concat_actuals)

    with open(OUT_DIR / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nSaved results to {OUT_DIR}", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == '__main__':
    main()
