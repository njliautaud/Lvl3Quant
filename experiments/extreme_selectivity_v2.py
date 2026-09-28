#!/usr/bin/env python3
"""
EXTREME SELECTIVITY V2 — proper signed-label version.

Uses validated smart_v3 signed labels_10s (price move in ticks over 10s) aligned to
CNN-Mamba v2 OOT predictions via the same stride/window-end indexing as v1.

Tests entry thresholds at 1.5σ, 2.0σ, 2.5σ, 3.0σ, 3.5σ, 4.0σ on the 10s prediction head.
Cost = 0.376t commission (HC #512). Regime stratification per HC #428 R1.
"""
from __future__ import annotations
import os, json
from pathlib import Path
from datetime import datetime
import numpy as np

PRED_DIR = Path('/home/jupiter/Lvl3Quant/output/cnn_mamba_v2_all_oot')
MBO_DIR  = Path('/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3')
OUT_DIR  = Path('/home/jupiter/Lvl3Quant/output/extreme_selectivity_v2')
OUT_DIR.mkdir(parents=True, exist_ok=True)

COMMISSION_TICKS = 0.376
PRED_COL = 2  # 10s head
SIGMA_LEVELS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

def load_aligned(pred_file: Path) -> tuple[np.ndarray, np.ndarray] | None:
    date = pred_file.stem.replace('_predictions', '')
    mbo  = MBO_DIR / f'{date}_mbo_events.npz'
    if not mbo.exists(): return None
    pd = np.load(pred_file, allow_pickle=True)
    md = np.load(mbo, allow_pickle=True)
    preds = pd['predictions']         # (n_windows, 3)
    n_windows  = int(pd['n_windows'])
    window_size = int(pd['window_size'])
    stride     = int(pd['stride'])
    labels_10s = md['labels_10s']
    N = len(labels_10s)
    idx = np.arange(n_windows) * stride + window_size - 1
    mask = idx < N
    return preds[mask][:, PRED_COL].astype(np.float64), labels_10s[idx[mask]].astype(np.float64)

def daily_sharpe(daily: np.ndarray) -> float:
    if len(daily) < 3: return float('nan')
    mu, sd = float(daily.mean()), float(daily.std(ddof=1))
    if sd < 1e-9: return float('nan')
    return mu / sd * np.sqrt(252)

def main():
    pred_files = sorted(PRED_DIR.glob('*_predictions.npz'))
    print(f'Found {len(pred_files)} prediction files')

    # Pass 1: collect all predictions to estimate sigma
    all_preds, all_labels, per_date = [], [], []
    for pf in pred_files:
        out = load_aligned(pf)
        if out is None: continue
        p, l = out
        if len(p) < 100: continue
        # NaN-safe
        mask = np.isfinite(p) & np.isfinite(l)
        p, l = p[mask], l[mask]
        if len(p) < 100: continue
        all_preds.append(p); all_labels.append(l)
        per_date.append({'date': pf.stem.replace('_predictions',''), 'p': p, 'l': l})
    if not per_date:
        raise SystemExit('No aligned data found')

    P_all = np.concatenate(all_preds)
    L_all = np.concatenate(all_labels)
    sigma = float(P_all.std())
    print(f'σ = {sigma:.4f}, mean pred = {P_all.mean():.4f}, n_total = {len(P_all):,}, dates = {len(per_date)}')

    # Concat IC across all data
    concat_ic = float(np.corrcoef(P_all, L_all)[0,1])
    print(f'Concat IC (10s signed): {concat_ic:.4f}')

    results = []
    for sk in SIGMA_LEVELS:
        thr = sk * sigma
        daily_nets, daily_grosses, all_T = [], [], []
        n_dates_used = 0
        for d in per_date:
            mask = np.abs(d['p']) >= thr
            if mask.sum() < 5: continue
            signed = np.sign(d['p'][mask])
            gross = signed * d['l'][mask]
            net = gross - COMMISSION_TICKS
            all_T.append(net)
            daily_nets.append(float(net.mean()))
            daily_grosses.append(float(gross.mean()))
            n_dates_used += 1
        if not all_T: continue
        T = np.concatenate(all_T)
        daily = np.array(daily_nets)
        # Regime stratification by daily mean signed return on the day's labels
        regimes = {'green':[], 'red':[], 'flat':[]}
        for d, dn in zip(per_date, daily_nets):
            mu = float(d['l'].mean())
            r = 'green' if mu>0.5 else 'red' if mu<-0.5 else 'flat'
            if not np.isfinite(dn): continue
            # only include if this date was used for this threshold
            mask = np.abs(d['p']) >= thr
            if mask.sum() < 5: continue
            regimes[r].append(dn)
        srm = {k: daily_sharpe(np.array(v)) for k,v in regimes.items()}
        days = {k: len(v) for k,v in regimes.items()}
        sg = srm.get('green') or 0; sr = srm.get('red') or 0
        rgap = abs(sg-sr)/max(abs(sg),abs(sr),1e-9)
        # Day concentration on positive-contribution side
        positives = daily[daily > 0]
        day_conc = float(positives.max()/positives.sum()) if positives.sum()>0 else 0.0
        results.append({
            'sigma_k': sk, 'threshold': thr, 'n_dates_used': n_dates_used,
            'trades_per_day_avg': float(np.mean([np.abs(d['p'])[np.abs(d['p'])>=thr].size for d in per_date])),
            'net_mean_ticks': float(T.mean()),
            'gross_mean_ticks': float(T.mean() + COMMISSION_TICKS),
            'wr': float(((T + COMMISSION_TICKS) > 0).mean()),
            'daily_sharpe': daily_sharpe(daily),
            'pct_profitable_days': float((daily > 0).mean()),
            'sharpe_by_regime': srm,
            'days_by_regime': days,
            'regime_gap': float(rgap),
            'regime_agnostic_pass': rgap <= 0.50,
            'day_concentration': day_conc,
            'day_conc_pass': day_conc <= 0.70,
            'overall_pass': bool(float(T.mean()) > 0 and rgap <= 0.50 and day_conc <= 0.70 and float((daily>0).mean()) >= 0.55),
        })

    print('\n=== EXTREME SELECTIVITY V2 (proper signed 10s labels) ===')
    print(f'{"σ":>4} | {"thr":>6} | {"trades/d":>8} | {"days":>4} | {"gross":>7} | {"net":>7} | {"WR":>6} | {"Sharpe":>7} | {"profD":>6} | {"gap":>5} | PASS?')
    print('-' * 110)
    for r in results:
        print(f'{r["sigma_k"]:4.1f} | {r["threshold"]:6.3f} | {r["trades_per_day_avg"]:8.1f} | '
              f'{r["n_dates_used"]:4d} | {r["gross_mean_ticks"]:+7.3f} | {r["net_mean_ticks"]:+7.3f} | '
              f'{r["wr"]:6.1%} | {r["daily_sharpe"]:7.2f} | {r["pct_profitable_days"]:6.1%} | '
              f'{r["regime_gap"]:5.2f} | {"YES" if r["overall_pass"] else "no"}')

    out_path = OUT_DIR / f'extreme_selectivity_v2_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(out_path, 'w') as fh:
        json.dump({'sigma': sigma, 'concat_ic': concat_ic, 'results': results}, fh, indent=2)
    print(f'\nWrote {out_path}')

    winners = [r for r in results if r['overall_pass']]
    print(f'\n=== VERDICT: {len(winners)} extreme-selectivity configs PASS all gates ===')
    for w in winners:
        print(f"  σ={w['sigma_k']} → net +{w['net_mean_ticks']:.3f}t, {w['trades_per_day_avg']:.0f} trades/day, Sharpe {w['daily_sharpe']:.2f}")

if __name__ == '__main__':
    main()
