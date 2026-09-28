#!/usr/bin/env python3
"""
Meta Hold Optimization v1
=========================
For meta-filtered trades, compute MFE profiles at multiple hold horizons
(1s, 5s, 10s, 30s) to find optimal hold duration. Current strategy holds 30s,
but meta-filtered trades may realize edge faster.

Compares meta-filtered vs unfiltered shorts and longs at each horizon.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

if sys.platform == 'win32':
    _BASE = Path(r'C:\Users\claude\Lvl3Quant')
else:
    _BASE = Path('/home/jupiter/Lvl3Quant')

# Try v2 first (what training used), fall back to v1
PRED_DIR = _BASE / 'output' / 'cnn_mamba_v2_bulk_oot_v2'
if not PRED_DIR.exists():
    PRED_DIR = _BASE / 'output' / 'cnn_mamba_v2_bulk_oot'
MBO_DIR = _BASE / 'data' / 'processed' / 'mbo_events_smart_v3'
SHORT_WEIGHTS_DIR = _BASE / 'output' / 'meta_production_v1' / 'weights'
LONG_WEIGHTS_DIR = _BASE / 'output' / 'meta_production_longs_v1' / 'weights'
OUT_DIR = _BASE / 'output' / 'meta_hold_opt_v1'
OUT_DIR.mkdir(parents=True, exist_ok=True)

SHORT_PERCENTILE = 3
LONG_PERCENTILE = 97
STOP_TICKS = 8.0
HORIZONS = ['1s', '5s', '10s', '30s']
LABEL_KEYS = {'1s': 'labels_1s', '5s': 'labels_5s', '10s': 'labels_10s', '30s': 'labels_30s'}
COMMISSION_PASSIVE = 0.376   # ticks, passive limit
COMMISSION_MARKET = 1.376    # ticks, market order

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
p = lambda *a, **k: (print(*a, **k), sys.stdout.flush())


class MetaMLP(nn.Module):
    """Must match training architecture exactly: GELU, dropout=0.2."""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_fold_models(weights_dir):
    """Load all fold weights from a directory."""
    files = sorted(weights_dir.glob('fold_*.pt'))
    models = []
    for f in files:
        ckpt = torch.load(f, map_location=device, weights_only=False)
        dim = ckpt['input_dim']
        model = MetaMLP(dim).to(device)
        model.load_state_dict(ckpt['model_state'])
        model.eval()
        models.append({'model': model, 'mean': ckpt['norm_mean'], 'std': ckpt['norm_std']})
    return models


def ensemble_predict(models, features):
    """Average predictions across all fold models."""
    preds_all = []
    X = features.astype(np.float32)
    for m in models:
        X_n = (X - m['mean']) / (m['std'] + 1e-8)
        with torch.no_grad():
            t = torch.from_numpy(X_n).to(device)
            pred = m['model'](t).cpu().numpy()
        preds_all.append(pred)
    return np.mean(preds_all, axis=0)


def compute_horizon_stats(pnl_arr, label):
    """Compute stats for a P&L array at a given horizon."""
    if len(pnl_arr) == 0:
        return None
    mean_pnl = float(np.mean(pnl_arr))
    std_pnl = float(np.std(pnl_arr)) if len(pnl_arr) > 1 else 1e-8
    wr = float(np.mean(pnl_arr > 0) * 100)
    wins = pnl_arr[pnl_arr > 0].sum() if (pnl_arr > 0).any() else 0
    losses = abs(pnl_arr[pnl_arr < 0].sum()) if (pnl_arr < 0).any() else 1e-8
    pf = float(wins / losses) if losses > 0 else 999.0
    sharpe = float(mean_pnl / std_pnl) if std_pnl > 1e-8 else 0.0
    pcts = np.percentile(pnl_arr, [25, 50, 75, 90])
    return {
        'horizon': label, 'n': len(pnl_arr),
        'mean_pnl': round(mean_pnl, 4), 'std': round(std_pnl, 4),
        'wr': round(wr, 2), 'pf': round(pf, 3), 'sharpe': round(sharpe, 4),
        'p25': round(float(pcts[0]), 4), 'p50': round(float(pcts[1]), 4),
        'p75': round(float(pcts[2]), 4), 'p90': round(float(pcts[3]), 4),
    }


def main():
    p(f"{'='*70}")
    p(f"META HOLD OPTIMIZATION v1 — {datetime.now():%Y-%m-%d %H:%M}")
    p(f"Device: {device}  |  Pred dir: {PRED_DIR.name}")
    p(f"Stop: {STOP_TICKS} ticks  |  Short/Long percentile: {SHORT_PERCENTILE}%/{100-LONG_PERCENTILE}%")
    p(f"{'='*70}\n")

    # Load meta-models
    p("Loading meta-model weights...")
    short_models = load_fold_models(SHORT_WEIGHTS_DIR)
    long_models = load_fold_models(LONG_WEIGHTS_DIR)
    p(f"  Short folds: {len(short_models)}, Long folds: {len(long_models)}\n")

    # Collect per-horizon P&L arrays
    keys = ['short_meta', 'short_unfilt', 'long_meta', 'long_unfilt']
    results = {k: {h: [] for h in HORIZONS} for k in keys}

    pred_files = sorted([f for f in PRED_DIR.glob('*_predictions.npz') if '_stale_' not in str(f)])
    p(f"Found {len(pred_files)} prediction files\n")

    n_loaded = 0
    for pf in pred_files:
        pd = np.load(pf, allow_pickle=True)
        date_str = str(pd['date'])
        preds = pd['predictions']
        n_win = int(pd['n_windows'])
        win_sz = int(pd['window_size'])
        stride = int(pd['stride'])

        mbo_file = MBO_DIR / f'{date_str}_mbo_events.npz'
        if not mbo_file.exists():
            continue

        mbo = np.load(mbo_file, allow_pickle=True)
        events = mbo['events']

        # Load all label horizons
        labels = {}
        skip = False
        for h, lk in LABEL_KEYS.items():
            if lk not in mbo:
                skip = True
                break
            labels[h] = mbo[lk]
        if skip:
            continue

        # Align predictions to events
        indices = np.arange(n_win) * stride + (win_sz - 1)
        max_idx = min(len(events), min(len(l) for l in labels.values())) - 1
        valid = indices <= max_idx
        indices = indices[valid]
        preds = preds[:len(indices)]

        feat = events[indices]
        labs = {h: labels[h][indices] for h in HORIZONS}

        # Valid mask (no NaN in features or any label)
        vmask = ~np.any(np.isnan(feat), axis=1)
        for h in HORIZONS:
            vmask &= ~np.isnan(labs[h])
        if vmask.sum() < 100:
            continue

        feat = feat[vmask]
        preds = preds[vmask]
        labs = {h: labs[h][vmask] for h in HORIZONS}
        pred_1s = preds[:, 0]

        # --- SHORTS ---
        short_thresh = np.percentile(pred_1s, SHORT_PERCENTILE)
        smask = pred_1s <= short_thresh
        if smask.sum() > 10:
            s_feat = feat[smask]
            s_preds = preds[smask]
            s_labs = {h: labs[h][smask] for h in HORIZONS}

            # Build meta features: 25 MBO + 3 preds + rank = 29
            ranks = np.argsort(np.argsort(s_preds[:, 0])).astype(np.float32) / max(len(s_preds), 1)
            meta_feat = np.column_stack([s_feat, s_preds[:, 0], s_preds[:, 1], s_preds[:, 2], ranks]).astype(np.float32)

            meta_scores = ensemble_predict(short_models, meta_feat)
            meta_pass = meta_scores > 0

            for h in HORIZONS:
                raw_pnl = -s_labs[h]  # short P&L = negative label
                np.clip(raw_pnl, -STOP_TICKS, None, out=raw_pnl)
                results['short_unfilt'][h].append(raw_pnl)
                if meta_pass.sum() > 0:
                    results['short_meta'][h].append(raw_pnl[meta_pass])

        # --- LONGS ---
        long_thresh = np.percentile(pred_1s, LONG_PERCENTILE)
        lmask = pred_1s >= long_thresh
        if lmask.sum() > 10:
            l_feat = feat[lmask]
            l_preds = preds[lmask]
            l_labs = {h: labs[h][lmask] for h in HORIZONS}

            ranks = np.argsort(np.argsort(-l_preds[:, 0])).astype(np.float32) / max(len(l_preds), 1)
            meta_feat = np.column_stack([l_feat, l_preds[:, 0], l_preds[:, 1], l_preds[:, 2], ranks]).astype(np.float32)

            meta_scores = ensemble_predict(long_models, meta_feat)
            meta_pass = meta_scores > 0

            for h in HORIZONS:
                raw_pnl = l_labs[h].copy()  # long P&L = label
                np.clip(raw_pnl, -STOP_TICKS, None, out=raw_pnl)
                results['long_unfilt'][h].append(raw_pnl)
                if meta_pass.sum() > 0:
                    results['long_meta'][h].append(raw_pnl[meta_pass])

        n_loaded += 1
        if n_loaded % 10 == 0:
            p(f"  Processed {n_loaded} dates...")

    p(f"\nProcessed {n_loaded} dates total\n")

    # Aggregate and report
    summary = {}
    for cost_label, cost in [('passive', COMMISSION_PASSIVE), ('market', COMMISSION_MARKET)]:
        p(f"\n{'='*70}")
        p(f"  RESULTS — {cost_label.upper()} COST = {cost:.3f} ticks RT")
        p(f"{'='*70}")
        cost_summary = {}

        for side_key, side_label in [('short_meta', 'SHORT META-FILTERED'),
                                      ('short_unfilt', 'SHORT UNFILTERED'),
                                      ('long_meta', 'LONG META-FILTERED'),
                                      ('long_unfilt', 'LONG UNFILTERED')]:
            p(f"\n  --- {side_label} ---")
            p(f"  {'Horizon':>8s} {'N':>7s} {'MeanPnL':>8s} {'WR%':>6s} {'PF':>6s} {'Sharpe':>7s} {'p25':>7s} {'p50':>7s} {'p75':>7s} {'p90':>7s}")

            side_stats = {}
            for h in HORIZONS:
                arrs = results[side_key][h]
                if not arrs:
                    p(f"  {h:>8s}  -- no data --")
                    continue
                combined = np.concatenate(arrs) - cost
                stats = compute_horizon_stats(combined, h)
                if stats:
                    side_stats[h] = stats
                    p(f"  {h:>8s} {stats['n']:>7d} {stats['mean_pnl']:>+8.4f} {stats['wr']:>6.1f} "
                      f"{stats['pf']:>6.2f} {stats['sharpe']:>+7.4f} {stats['p25']:>+7.3f} "
                      f"{stats['p50']:>+7.3f} {stats['p75']:>+7.3f} {stats['p90']:>+7.3f}")

            cost_summary[side_key] = side_stats

            # Best horizon
            if side_stats:
                best_h = max(side_stats, key=lambda h: side_stats[h]['sharpe'])
                p(f"  >>> BEST HORIZON: {best_h} (Sharpe={side_stats[best_h]['sharpe']:+.4f})")

        summary[cost_label] = cost_summary

    # Meta filter rate
    p(f"\n{'='*70}")
    p(f"  META FILTER PASS RATES")
    p(f"{'='*70}")
    for side in ['short', 'long']:
        meta_n = sum(len(a) for a in results[f'{side}_meta']['1s']) if results[f'{side}_meta']['1s'] else 0
        unfilt_n = sum(len(a) for a in results[f'{side}_unfilt']['1s']) if results[f'{side}_unfilt']['1s'] else 0
        rate = meta_n / unfilt_n * 100 if unfilt_n > 0 else 0
        p(f"  {side.upper()}: {meta_n}/{unfilt_n} = {rate:.1f}% pass rate")

    # Save JSON
    out_file = OUT_DIR / 'results.json'
    with open(out_file, 'w') as f:
        json.dump({
            'generated': datetime.now().isoformat(),
            'n_dates': n_loaded,
            'pred_dir': str(PRED_DIR.name),
            'stop_ticks': STOP_TICKS,
            'short_pct': SHORT_PERCENTILE,
            'long_pct': LONG_PERCENTILE,
            'results': summary,
        }, f, indent=2)
    p(f"\nResults saved to {out_file}")
    p(f"\n{'='*70}")
    p("DONE")


if __name__ == '__main__':
    main()
