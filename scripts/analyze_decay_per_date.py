"""
Analyze per-date IC/DA/MagCorr from existing v4_comprehensive predictions.
Find exact decay onset date.
"""
import numpy as np
import os
from pathlib import Path
from scipy.stats import spearmanr

base = Path('/home/nick/Lvl3Quant/output/decay_v4_comprehensive/CNN-Mamba_v2')
data_dir = Path('/home/nick/Lvl3Quant/data/processed/mbo_events_smart_v3')

dates = sorted(os.listdir(base))
print(f'Found {len(dates)} dates with predictions')
print()
print(f'{"Date":<10} {"vol_1s":>7} {"IC_1s":>7} {"IC_5s":>7} {"IC_10s":>7} {"DA_all":>7} {"DA_T25":>7} {"DA_T10":>7} {"MagC":>7} {"n":>6}  note')
print('-' * 95)

results = []
for d in dates:
    pred_file = base / d / 'predictions.npz'
    if not pred_file.exists():
        continue

    pdata = np.load(pred_file)
    keys = list(pdata.keys())
    preds = pdata['preds'] if 'preds' in pdata else (pdata['predictions'] if 'predictions' in pdata else pdata[keys[0]])
    # Labels may be separate arrays
    if 'labels_1s' in pdata:
        labels = np.stack([pdata['labels_1s'], pdata['labels_5s'], pdata['labels_10s']], axis=1)
    elif 'labels' in pdata:
        labels = pdata['labels']
    else:
        labels = pdata[keys[1]]

    # Get vol from data
    data_file = data_dir / f'{d}_mbo_events.npz'
    vol_1s = 0
    if data_file.exists():
        dd = np.load(data_file)
        l1_raw = dd['labels_1s']
        valid_raw = l1_raw[~np.isnan(l1_raw)]
        vol_1s = float(np.std(valid_raw)) if len(valid_raw) > 0 else 0
        del dd

    # Handle different prediction shapes
    if preds.ndim == 1:
        p1 = preds
        l1 = labels if labels.ndim == 1 else labels[:, 0]
        p5, p10, l5, l10 = None, None, None, None
    else:
        p1, l1 = preds[:, 0], labels[:, 0]
        p5, l5 = (preds[:, 1], labels[:, 1]) if preds.shape[1] > 1 else (None, None)
        p10, l10 = (preds[:, 2], labels[:, 2]) if preds.shape[1] > 2 else (None, None)

    valid = ~np.isnan(l1) & ~np.isnan(p1) & np.isfinite(l1) & np.isfinite(p1)
    p1v, l1v = p1[valid], l1[valid]
    n = len(p1v)

    if n < 20:
        print(f'{d:<10} {vol_1s:>7.2f} {"N/A":>7} (n={n})')
        continue

    ic_1s = spearmanr(p1v, l1v)[0]
    da_all = np.mean(np.sign(p1v) == np.sign(l1v))
    mag_corr = float(np.corrcoef(np.abs(p1v), np.abs(l1v))[0, 1])

    # Confidence bands
    conf = np.abs(p1v)
    t25_mask = conf >= np.percentile(conf, 75)
    t10_mask = conf >= np.percentile(conf, 90)
    da_t25 = np.mean(np.sign(p1v[t25_mask]) == np.sign(l1v[t25_mask]))
    da_t10 = np.mean(np.sign(p1v[t10_mask]) == np.sign(l1v[t10_mask]))

    # IC for 5s and 10s
    ic_5s, ic_10s = 0.0, 0.0
    if p5 is not None:
        v5 = ~np.isnan(l5) & ~np.isnan(p5) & np.isfinite(l5) & np.isfinite(p5)
        if v5.sum() > 20:
            ic_5s = spearmanr(p5[v5], l5[v5])[0]
    if p10 is not None:
        v10 = ~np.isnan(l10) & ~np.isnan(p10) & np.isfinite(l10) & np.isfinite(p10)
        if v10.sum() > 20:
            ic_10s = spearmanr(p10[v10], l10[v10])[0]

    note = ''
    if vol_1s > 8: note = '*** EXTREME VOL ***'
    elif vol_1s > 5.5: note = '** HIGH VOL **'
    elif ic_1s < 0.05: note = '!! LOW IC !!'
    elif ic_1s > 0.2: note = '++ STRONG ++'

    print(f'{d:<10} {vol_1s:>7.2f} {ic_1s:>7.4f} {ic_5s:>7.4f} {ic_10s:>7.4f} {da_all:>7.3f} {da_t25:>7.3f} {da_t10:>7.3f} {mag_corr:>7.4f} {n:>6}  {note}')

    results.append({
        'date': d, 'vol_1s': vol_1s, 'ic_1s': float(ic_1s), 'ic_5s': float(ic_5s),
        'ic_10s': float(ic_10s), 'da_all': float(da_all), 'da_t25': float(da_t25),
        'da_t10': float(da_t10), 'mag_corr': mag_corr, 'n': n
    })

# Summary statistics
print('\n\n=== DECAY ANALYSIS SUMMARY ===')
if results:
    ics = [r['ic_1s'] for r in results]
    vols = [r['vol_1s'] for r in results]

    # Find first date where IC drops below 0.10 consistently
    print('\nRolling 3-day IC average:')
    for i in range(len(results)):
        if i >= 2:
            avg3 = np.mean([results[j]['ic_1s'] for j in range(i-2, i+1)])
            print(f"  {results[i]['date']}: 3d_avg_IC={avg3:.4f} vol={results[i]['vol_1s']:.2f}")

    # Correlation analysis
    corr_vol_ic = np.corrcoef(vols, ics)[0, 1]
    print(f'\nCorrelation(vol_1s, IC_1s) = {corr_vol_ic:.4f}')

    # Split by vol regime
    low_vol = [r for r in results if r['vol_1s'] < 4.5]
    mid_vol = [r for r in results if 4.5 <= r['vol_1s'] < 7.0]
    high_vol = [r for r in results if r['vol_1s'] >= 7.0]

    if low_vol:
        print(f'\nLow vol (<4.5): n={len(low_vol)}, avg IC_1s={np.mean([r["ic_1s"] for r in low_vol]):.4f}')
    if mid_vol:
        print(f'Mid vol (4.5-7): n={len(mid_vol)}, avg IC_1s={np.mean([r["ic_1s"] for r in mid_vol]):.4f}')
    if high_vol:
        print(f'High vol (>7):  n={len(high_vol)}, avg IC_1s={np.mean([r["ic_1s"] for r in high_vol]):.4f}')

import json
with open('/home/nick/Lvl3Quant/output/decay_per_date_analysis.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\nSaved to output/decay_per_date_analysis.json')
