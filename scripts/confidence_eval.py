#!/usr/bin/env python3
"""
Confidence-conditional model evaluation per DIRECTIVES HC #13.

We ONLY trade confidence — full-OOT IC is forbidden as a sole decision metric.
Every model fold MUST be evaluated with this script before any keep/kill decision.

Usage:
    python3 confidence_eval.py <fold_oot_predictions.npz> [--out fold_confidence_eval.json]

Input NPZ schema (current convention):
    predictions: (N, 4..6) float — first 4 cols = dz_{1s,5s,10s,30s} (z-direction signals)
    labels:      (N, 4)    float — realized {1s,5s,10s,30s} returns
    horizons:    (4,)      str  — ['1s','5s','10s','30s']

Outputs (per horizon):
    - full_ic                : Spearman(pred, actual) over all events
    - top{10,5,1,0.5}_ic     : Spearman within top-X% of |pred|  (CONFIDENCE-CONDITIONAL IC)
    - mag_ic                 : Spearman(|pred|, |actual|) — does big pred mean big move?
    - hit_top{10,1}          : directional accuracy at confidence thresholds
    - sortino_g_top{10,1}    : gross Sortino (no fill costs) at threshold
    - avg_signed_top{10,1}   : mean of sign(pred)*actual at threshold
    - n_trades_top{10,1}     : count at each threshold (coverage)
"""
import argparse, json, os, sys
import numpy as np
from scipy.stats import spearmanr

THRESH = [(0.10,'top10'), (0.05,'top5'), (0.01,'top1'), (0.005,'top05')]

def confidence_metrics(npz_path: str) -> dict:
    d = np.load(npz_path, allow_pickle=False)
    pred = d['predictions']; y = d['labels']
    horizons = [str(h) for h in d['horizons']] if 'horizons' in d.files else ['1s','5s','10s','30s']
    n_h = min(pred.shape[1], y.shape[1], len(horizons))
    horizons = horizons[:n_h]

    out = {
        'N_total': int(len(y)),
        'horizons': horizons,
        'full_ic': {}, 'mag_ic': {},
        'top10_ic': {}, 'top5_ic': {}, 'top1_ic': {}, 'top05_ic': {},
        'hit_top10': {}, 'hit_top1': {},
        'sortino_g_top10': {}, 'sortino_g_top1': {},
        'avg_signed_top10': {}, 'avg_signed_top1': {},
        'n_trades_top10': {}, 'n_trades_top1': {},
    }
    for i, h in enumerate(horizons):
        p = pred[:, i].astype(np.float64); t = y[:, i].astype(np.float64)
        m = np.isfinite(p) & np.isfinite(t)
        p, t = p[m], t[m]
        if len(p) < 100:
            continue
        absp = np.abs(p)
        out['full_ic'][h] = float(spearmanr(p, t).correlation)
        out['mag_ic'][h]  = float(spearmanr(absp, np.abs(t)).correlation)
        for frac, key in THRESH:
            k = max(20, int(len(p) * frac))
            idx = np.argpartition(-absp, k - 1)[:k]
            ic = spearmanr(p[idx], t[idx]).correlation
            out[f'{key}_ic'][h] = float(ic) if np.isfinite(ic) else None
        for frac, key in [(0.10,'top10'), (0.01,'top1')]:
            k = max(20, int(len(p) * frac))
            idx = np.argpartition(-absp, k - 1)[:k]
            pp, tt = p[idx], t[idx]
            mask = tt != 0
            hit = float((np.sign(pp[mask]) == np.sign(tt[mask])).mean()) if mask.any() else float('nan')
            signed = np.sign(pp) * tt
            downside = signed[signed < 0]
            dstd = float(downside.std()) if len(downside) > 5 else 0.0
            sortino = float(signed.mean() / dstd * np.sqrt(k)) if dstd > 0 else float('nan')
            out[f'hit_{key}'][h] = hit
            out[f'sortino_g_{key}'][h] = sortino
            out[f'avg_signed_{key}'][h] = float(signed.mean())
            out[f'n_trades_{key}'][h] = int(k)
    return out

def fmt_table(m: dict) -> str:
    rows = ['horizon  full_IC  top10_IC  top5_IC  top1_IC  mag_IC  hit_top1  sortino_top1  N_top1']
    for h in m['horizons']:
        rows.append(
            f"{h:<7} {m['full_ic'].get(h,float('nan')):>7.4f}  "
            f"{m['top10_ic'].get(h,float('nan')):>7.4f}  "
            f"{m['top5_ic'].get(h,float('nan')):>7.4f}  "
            f"{m['top1_ic'].get(h,float('nan')):>7.4f}  "
            f"{m['mag_ic'].get(h,float('nan')):>6.4f}  "
            f"{m['hit_top1'].get(h,float('nan')):>7.3f}  "
            f"{m['sortino_g_top1'].get(h,float('nan')):>11.2f}  "
            f"{m['n_trades_top1'].get(h,0):>6d}"
        )
    return '\n'.join(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('npz', help='Path to fold_NN_oot_predictions.npz')
    ap.add_argument('--out', default=None, help='Output JSON path (default: <npz_dir>/<basename>_confidence_eval.json)')
    args = ap.parse_args()

    if not os.path.exists(args.npz):
        print(f'ERROR: {args.npz} not found', file=sys.stderr); sys.exit(2)

    m = confidence_metrics(args.npz)
    out_path = args.out or args.npz.replace('_oot_predictions.npz', '_confidence_eval.json')
    with open(out_path, 'w') as f:
        json.dump(m, f, indent=2)
    print(fmt_table(m))
    print(f'\nWrote {out_path}')

if __name__ == '__main__':
    main()
