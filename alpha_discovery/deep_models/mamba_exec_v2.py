#!/usr/bin/env python3
"""
Mamba + PatchTST Execution Strategy Engine v2
==============================================
FIXED PnL: Uses labels correctly as forward returns from each event.

For each potential trade entry, PnL = labels_Xs[entry] * direction - costs
where X is the hold horizon (1s, 5s, or 10s).

Strategies select WHICH events to enter. Multi-horizon labels show
PnL at different hold durations for the same entry.

Tests:
1. Entry filters: confidence thresholds, multi-horizon agreement, coherence
2. Hold horizon comparison: which horizon maximizes risk-adjusted returns?
3. MFE/MAE profile: do winning trades move favorably before reaching target?
4. Embedding-based MLP: can a small model learn which entries are profitable?
5. Confidence-tier analysis: Top50/25/10/5/1/0.5% performance
"""

import os
import sys
import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple
from collections import defaultdict

LVL3_ROOT = Path(__file__).resolve().parent.parent.parent
MAMBA_PRED_DIR = LVL3_ROOT / 'output' / 'mamba_v7_tiny_smart_v3_mar_apr'
RESULTS_DIR = LVL3_ROOT / 'alpha_discovery' / 'deep_models' / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TICK_VALUE = 12.50
COMMISSION_TICKS = 0.376  # $4.70 RT

_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
log = logging.getLogger('exec_v2')
log.setLevel(logging.INFO)
for h in [logging.FileHandler(str(RESULTS_DIR / f'mamba_exec_v2_{_ts}.log'), 'w', 'utf-8'),
          logging.StreamHandler(sys.stdout)]:
    h.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
    log.addHandler(h)


def load_per_fold():
    """Load per-fold Mamba v7 predictions with date info."""
    import re
    folds = []
    for f in sorted(MAMBA_PRED_DIR.glob('fold_*_oot_predictions.npz')):
        d = np.load(str(f), allow_pickle=True)
        fold_idx = int(f.stem.split('_')[1])
        oot = str(d['oot_files'][0]) if 'oot_files' in d else ''
        m = re.search(r'(\d{8})_mbo', oot)
        date = m.group(1) if m else ''
        folds.append({
            'fold': fold_idx, 'date': date,
            'preds': d['predictions'],    # (N, 3) = 1s, 5s, 10s
            'labels': d['labels'],        # (N, 3) = actual forward returns
            'embeddings': d['embeddings'],
            'n': d['predictions'].shape[0],
        })
    return folds


def analyze_entry_filter(preds, labels, mask, cost_ticks, name):
    """
    Given a boolean mask of selected entries, compute PnL metrics at each horizon.
    preds: (N, 3), labels: (N, 3), mask: (N,) boolean
    labels[i, h] = actual price move (ticks) at horizon h from event i
    PnL = labels[i, h] * sign(preds[i, h]) - cost_ticks
    """
    results = {}
    horizon_names = ['1s', '5s', '10s']

    for h, hname in enumerate(horizon_names):
        sel_preds = preds[mask, h]
        sel_labels = labels[mask, h]
        n = mask.sum()

        if n == 0:
            results[hname] = {'n': 0}
            continue

        direction = np.sign(sel_preds)
        # PnL = actual move * direction - costs
        pnl = sel_labels * direction - cost_ticks
        winners = pnl > 0

        # Direction accuracy
        correct = np.sign(sel_labels) == direction
        da = correct.mean()

        # Magnitude-direction correlation
        mag_corr = np.corrcoef(np.abs(sel_preds), sel_labels * direction)[0, 1] if n > 2 else 0

        results[hname] = {
            'n': int(n),
            'da': float(da),
            'mag_corr': float(mag_corr) if not np.isnan(mag_corr) else 0,
            'win_rate': float(winners.mean()),
            'total_pnl_ticks': float(pnl.sum()),
            'total_pnl_dollars': float(pnl.sum() * TICK_VALUE),
            'avg_pnl_ticks': float(pnl.mean()),
            'avg_pnl_dollars': float(pnl.mean() * TICK_VALUE),
            'avg_winner_ticks': float(pnl[winners].mean()) if winners.any() else 0,
            'avg_loser_ticks': float(pnl[~winners].mean()) if (~winners).any() else 0,
            'profit_factor': float(pnl[winners].sum() / max(abs(pnl[~winners].sum()), 1e-8)) if winners.any() else 0,
            'sortino': _sortino(pnl),
            'avg_magnitude': float(np.abs(sel_labels).mean()),
            'pnl_per_dollar_cost': float(pnl.sum() / max(n * cost_ticks * TICK_VALUE, 1)),
            # Long/short breakdown
            'long_n': int((direction > 0).sum()),
            'short_n': int((direction < 0).sum()),
            'long_da': float(correct[direction > 0].mean()) if (direction > 0).any() else 0,
            'short_da': float(correct[direction < 0].mean()) if (direction < 0).any() else 0,
            'long_pnl': float(pnl[direction > 0].sum() * TICK_VALUE) if (direction > 0).any() else 0,
            'short_pnl': float(pnl[direction < 0].sum() * TICK_VALUE) if (direction < 0).any() else 0,
        }

    return results


def _sortino(pnl_arr):
    if len(pnl_arr) < 2:
        return 0
    neg = pnl_arr[pnl_arr < 0]
    ds = np.std(neg) if len(neg) > 1 else 1e-8
    return float(pnl_arr.mean() / ds) if ds > 0 else 0


def confidence_tier_analysis(preds, labels, cost_ticks):
    """Analyze performance at confidence tiers: All, 50%, 25%, 10%, 5%, 1%, 0.5%."""
    confidence = np.mean(np.abs(preds), axis=1)  # avg |pred| across horizons
    n = len(preds)
    tiers = {
        'All': np.ones(n, dtype=bool),
        'Top50%': confidence >= np.percentile(confidence, 50),
        'Top25%': confidence >= np.percentile(confidence, 75),
        'Top10%': confidence >= np.percentile(confidence, 90),
        'Top5%': confidence >= np.percentile(confidence, 95),
        'Top1%': confidence >= np.percentile(confidence, 99),
        'Top0.5%': confidence >= np.percentile(confidence, 99.5),
    }

    results = {}
    for tier_name, mask in tiers.items():
        results[tier_name] = analyze_entry_filter(preds, labels, mask, cost_ticks, tier_name)
    return results


def multi_horizon_agreement_analysis(preds, labels, cost_ticks):
    """Analyze entries where all 3 horizons agree on direction."""
    p1, p5, p10 = preds[:, 0], preds[:, 1], preds[:, 2]
    signs = np.sign(preds)
    all_agree = np.all(signs == signs[:, :1], axis=1) & (signs[:, 0] != 0)

    confidence = np.mean(np.abs(preds), axis=1)
    n = len(preds)

    filters = {
        'agree_all': all_agree,
        'agree_top50': all_agree & (confidence >= np.percentile(confidence[all_agree], 50)) if all_agree.any() else np.zeros(n, dtype=bool),
        'agree_top25': all_agree & (confidence >= np.percentile(confidence[all_agree], 75)) if all_agree.any() else np.zeros(n, dtype=bool),
        'agree_top10': all_agree & (confidence >= np.percentile(confidence[all_agree], 90)) if all_agree.any() else np.zeros(n, dtype=bool),
        'agree_top5': all_agree & (confidence >= np.percentile(confidence[all_agree], 95)) if all_agree.any() else np.zeros(n, dtype=bool),
        'agree_top1': all_agree & (confidence >= np.percentile(confidence[all_agree], 99)) if all_agree.any() else np.zeros(n, dtype=bool),
    }

    results = {}
    for name, mask in filters.items():
        results[name] = analyze_entry_filter(preds, labels, mask, cost_ticks, name)
    return results


def coherence_analysis(preds, labels, cost_ticks):
    """Signal coherence: when horizons are highly aligned, do we make money?"""
    p1, p5, p10 = preds[:, 0], preds[:, 1], preds[:, 2]

    # Coherence metric: all same sign AND magnitudes scale properly (1s > 5s > 10s is unlikely noise)
    signs = np.sign(preds)
    same_sign = np.all(signs == signs[:, :1], axis=1) & (signs[:, 0] != 0)

    # Magnitude coherence: are predictions proportional? (normalized std)
    abs_preds = np.abs(preds)
    mag_std = np.std(abs_preds, axis=1) / (np.mean(abs_preds, axis=1) + 1e-8)

    # Low mag_std = coherent magnitudes, high = divergent
    coherent = same_sign & (mag_std < np.median(mag_std[same_sign])) if same_sign.any() else np.zeros(len(preds), dtype=bool)
    very_coherent = same_sign & (mag_std < np.percentile(mag_std[same_sign], 25)) if same_sign.any() else np.zeros(len(preds), dtype=bool)

    filters = {
        'coherent_agree': coherent,
        'very_coherent': very_coherent,
    }

    results = {}
    for name, mask in filters.items():
        results[name] = analyze_entry_filter(preds, labels, mask, cost_ticks, name)
    return results


def momentum_burst_analysis(preds, labels, cost_ticks):
    """Detect sudden signal spikes — momentum bursts."""
    p1 = preds[:, 0]
    abs_p1 = np.abs(p1)

    # Rolling signal change
    delta = np.zeros_like(abs_p1)
    delta[1:] = abs_p1[1:] - abs_p1[:-1]

    # Burst = large positive delta (signal strengthening) + direction agreement
    signs = np.sign(preds)
    agree = np.all(signs == signs[:, :1], axis=1) & (signs[:, 0] != 0)

    for pct_name, pct in [('burst_top5', 95), ('burst_top1', 99)]:
        threshold = np.percentile(delta[delta > 0], pct) if (delta > 0).any() else 999
        mask = (delta > threshold) & agree
        yield pct_name, analyze_entry_filter(preds, labels, mask, cost_ticks, pct_name)


def embedding_mlp_analysis(preds, labels, embeddings, cost_ticks):
    """Train small MLP on embeddings to predict profitable entries."""
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
    except ImportError:
        log.warning("PyTorch unavailable — skipping MLP")
        return {}

    n = len(preds)
    train_n = n // 2  # temporal split

    # Target: is 10s hold profitable after costs?
    direction = np.sign(preds[:, 2])  # 10s direction
    profit_10s = labels[:, 2] * direction - cost_ticks
    target = (profit_10s > 0).astype(np.float32)

    # Features: embeddings + predictions + derived
    confidence = np.mean(np.abs(preds), axis=1, keepdims=True)
    features = np.hstack([embeddings, preds, np.abs(preds), confidence])

    X_tr = torch.FloatTensor(features[:train_n])
    y_tr = torch.FloatTensor(target[:train_n])
    X_te = torch.FloatTensor(features[train_n:])

    model = nn.Sequential(
        nn.Linear(features.shape[1], 64), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(32, 1), nn.Sigmoid(),
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.BCELoss()

    # Train
    ds = TensorDataset(X_tr, y_tr)
    loader = DataLoader(ds, batch_size=1024, shuffle=True)
    model.train()
    for epoch in range(15):
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(model(xb).squeeze(), yb)
            loss.backward()
            opt.step()

    # Predict
    model.eval()
    with torch.no_grad():
        probs = model(X_te).squeeze().numpy()

    # Evaluate MLP-selected entries at different thresholds
    test_preds = preds[train_n:]
    test_labels = labels[train_n:]
    results = {}

    for thresh_name, thresh in [('mlp_p60', 0.6), ('mlp_p70', 0.7), ('mlp_p80', 0.8), ('mlp_p90', 0.9)]:
        mask = probs > thresh
        if mask.sum() == 0:
            results[thresh_name] = {h: {'n': 0} for h in ['1s', '5s', '10s']}
            continue
        results[thresh_name] = analyze_entry_filter(test_preds, test_labels, mask, cost_ticks, thresh_name)
        log.info(f"  {thresh_name}: n={mask.sum()}, "
                f"10s WR={results[thresh_name]['10s'].get('win_rate', 0):.1%}, "
                f"10s DA={results[thresh_name]['10s'].get('da', 0):.1%}, "
                f"10s PnL=${results[thresh_name]['10s'].get('total_pnl_dollars', 0):,.0f}")

    return results


def print_summary(all_results, n_days):
    """Print comprehensive summary table."""
    log.info("\n" + "=" * 140)
    log.info("MAMBA v7 EXECUTION STRATEGY COMPARISON — March OOT (mid-price cost: 0.376 ticks)")
    log.info("=" * 140)

    header = f"{'Strategy':<28} {'N':>8} | {'DA':>6} {'WR':>6} {'PnL($)':>11} {'$/Trade':>9} {'PF':>6} {'Sort':>6} | {'DA':>6} {'WR':>6} {'PnL($)':>11} {'$/Trade':>9} {'PF':>6} {'Sort':>6} | {'DA':>6} {'WR':>6} {'PnL($)':>11}"
    log.info(f"{'':28} {'':>8} | {'--- 1s Hold ---':^52} | {'--- 5s Hold ---':^52} | {'--- 10s Hold ---':^30}")
    log.info(header)
    log.info("-" * 140)

    for name, data in sorted(all_results.items(), key=lambda x: x[1].get('10s', {}).get('total_pnl_dollars', -1e9), reverse=True):
        r1 = data.get('1s', {})
        r5 = data.get('5s', {})
        r10 = data.get('10s', {})
        n = r10.get('n', r1.get('n', 0))
        if n == 0:
            continue
        line = (f"{name:<28} {n:>8} | "
                f"{r1.get('da', 0):>5.1%} {r1.get('win_rate', 0):>5.1%} ${r1.get('total_pnl_dollars', 0):>10,.0f} ${r1.get('avg_pnl_dollars', 0):>8.2f} {r1.get('profit_factor', 0):>5.2f} {r1.get('sortino', 0):>5.3f} | "
                f"{r5.get('da', 0):>5.1%} {r5.get('win_rate', 0):>5.1%} ${r5.get('total_pnl_dollars', 0):>10,.0f} ${r5.get('avg_pnl_dollars', 0):>8.2f} {r5.get('profit_factor', 0):>5.2f} {r5.get('sortino', 0):>5.3f} | "
                f"{r10.get('da', 0):>5.1%} {r10.get('win_rate', 0):>5.1%} ${r10.get('total_pnl_dollars', 0):>10,.0f}")
        log.info(line)

    log.info("=" * 140)


def main():
    log.info("MAMBA v7 CREATIVE EXECUTION STRATEGIES v2 — Correct PnL")
    log.info(f"Cost: {COMMISSION_TICKS:.3f} ticks ($4.70 RT)")
    log.info("PnL = labels[entry, horizon] * direction - cost\n")

    folds = load_per_fold()
    if not folds:
        log.error("No predictions found!")
        return

    log.info(f"Loaded {len(folds)} folds:")
    for f in folds:
        log.info(f"  F{f['fold']:02d} ({f['date']}): {f['n']:,} events")

    # Concatenate all folds for aggregate analysis
    all_preds = np.vstack([f['preds'] for f in folds])
    all_labels = np.vstack([f['labels'] for f in folds])
    all_embeds = np.vstack([f['embeddings'] for f in folds])
    log.info(f"\nTotal: {len(all_preds):,} events across {len(folds)} March days")

    cost = COMMISSION_TICKS
    all_results = {}

    # 1. Confidence tier analysis
    log.info("\n" + "=" * 60)
    log.info("1. CONFIDENCE TIER ANALYSIS")
    log.info("=" * 60)
    tier_results = confidence_tier_analysis(all_preds, all_labels, cost)
    for tier, data in tier_results.items():
        r10 = data.get('10s', {})
        r1 = data.get('1s', {})
        if r10.get('n', 0) > 0:
            log.info(f"  {tier:>8}: n={r10['n']:>7,}  1s[DA={r1.get('da',0):.1%} WR={r1.get('win_rate',0):.1%} PnL=${r1.get('total_pnl_dollars',0):>10,.0f}]  "
                    f"10s[DA={r10['da']:.1%} WR={r10['win_rate']:.1%} PnL=${r10['total_pnl_dollars']:>10,.0f}  PF={r10['profit_factor']:.2f}]")
    all_results.update(tier_results)

    # 2. Multi-horizon agreement
    log.info("\n" + "=" * 60)
    log.info("2. MULTI-HORIZON AGREEMENT (all 3 horizons same direction)")
    log.info("=" * 60)
    agree_results = multi_horizon_agreement_analysis(all_preds, all_labels, cost)
    for name, data in agree_results.items():
        r10 = data.get('10s', {})
        r1 = data.get('1s', {})
        if r10.get('n', 0) > 0:
            log.info(f"  {name:>15}: n={r10['n']:>6,}  1s[DA={r1.get('da',0):.1%} PnL=${r1.get('total_pnl_dollars',0):>10,.0f}]  "
                    f"10s[DA={r10['da']:.1%} WR={r10['win_rate']:.1%} PnL=${r10['total_pnl_dollars']:>10,.0f}  PF={r10['profit_factor']:.2f}  Long/Short: ${r10.get('long_pnl',0):,.0f}/${r10.get('short_pnl',0):,.0f}]")
    all_results.update(agree_results)

    # 3. Coherence analysis
    log.info("\n" + "=" * 60)
    log.info("3. SIGNAL COHERENCE (same direction + proportional magnitudes)")
    log.info("=" * 60)
    coh_results = coherence_analysis(all_preds, all_labels, cost)
    for name, data in coh_results.items():
        r10 = data.get('10s', {})
        if r10.get('n', 0) > 0:
            log.info(f"  {name:>18}: n={r10['n']:>6,}  10s[DA={r10['da']:.1%} WR={r10['win_rate']:.1%} PnL=${r10['total_pnl_dollars']:>10,.0f}  Sortino={r10['sortino']:.3f}]")
    all_results.update(coh_results)

    # 4. Momentum burst
    log.info("\n" + "=" * 60)
    log.info("4. MOMENTUM BURST (sudden signal spike + agreement)")
    log.info("=" * 60)
    for name, data in momentum_burst_analysis(all_preds, all_labels, cost):
        r10 = data.get('10s', {})
        if r10.get('n', 0) > 0:
            log.info(f"  {name:>15}: n={r10['n']:>6,}  10s[DA={r10['da']:.1%} WR={r10['win_rate']:.1%} PnL=${r10['total_pnl_dollars']:>10,.0f}  Sortino={r10['sortino']:.3f}]")
        all_results[name] = data

    # 5. Embedding MLP
    log.info("\n" + "=" * 60)
    log.info("5. EMBEDDING MLP (learn which entries are profitable)")
    log.info("=" * 60)
    mlp_results = embedding_mlp_analysis(all_preds, all_labels, all_embeds, cost)
    all_results.update(mlp_results)

    # 6. Per-fold breakdown for best strategies
    log.info("\n" + "=" * 60)
    log.info("6. PER-FOLD BREAKDOWN (Top1% confidence, 10s hold)")
    log.info("=" * 60)
    for f in folds:
        conf = np.mean(np.abs(f['preds']), axis=1)
        mask = conf >= np.percentile(conf, 99)
        res = analyze_entry_filter(f['preds'], f['labels'], mask, cost, f'F{f["fold"]:02d}')
        r10 = res.get('10s', {})
        r1 = res.get('1s', {})
        if r10.get('n', 0) > 0:
            log.info(f"  F{f['fold']:02d} ({f['date']}): n={r10['n']:>4}  "
                    f"1s[DA={r1.get('da',0):.1%} PnL=${r1.get('total_pnl_dollars',0):>7,.0f}]  "
                    f"10s[DA={r10['da']:.1%} WR={r10['win_rate']:.1%} PnL=${r10['total_pnl_dollars']:>7,.0f}  "
                    f"Long=${r10.get('long_pnl',0):>7,.0f} Short=${r10.get('short_pnl',0):>7,.0f}]")

    # Summary table
    print_summary(all_results, len(folds))

    # Save
    save_path = RESULTS_DIR / f'mamba_exec_v2_{_ts}.json'
    with open(save_path, 'w') as fp:
        json.dump({
            'timestamp': _ts,
            'n_events': len(all_preds),
            'n_folds': len(folds),
            'fold_dates': [f['date'] for f in folds],
            'cost_ticks': cost,
            'results': {k: v for k, v in all_results.items()},
        }, fp, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else str(x))

    log.info(f"\nSaved: {save_path}")


if __name__ == '__main__':
    main()
