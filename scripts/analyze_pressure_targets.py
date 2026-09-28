"""
Comprehensive analysis of smooth pressure targets.
Run after build_smooth_pressure_targets.py completes all files.

Computes:
- Per-target, per-horizon IC (Spearman) across all OOT days
- Cross-target correlations
- Autocorrelation structure
- IC stability across time
- Recommended head weights for v4 multi-head training
"""

import numpy as np
import os
import glob
import warnings
import json
from datetime import datetime
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

DATA_DIR = '/home/jupiter/Lvl3Quant/data/processed/smooth_pressure_targets'
OUTPUT = '/home/jupiter/Lvl3Quant/output/pressure_target_analysis.json'


def analyze():
    files = sorted(glob.glob(f'{DATA_DIR}/*.npz'))
    print(f"Found {len(files)} pressure target files")

    targets = ['ntps', 'eofi', 'pdi', 'tia']
    horizons = ['1s', '5s', '10s', '30s']

    # ── 1. Per-target, per-horizon IC ──
    print("\n" + "=" * 70)
    print("1. IC: current_pressure_feature vs future_pressure_label")
    print("=" * 70)

    ic_matrix = {}
    for t in targets:
        ic_matrix[t] = {}
        for h in horizons:
            ics = []
            for f in files:
                d = np.load(f)
                feat_key = t
                label_key = f'{t}_label_{h}'
                if feat_key not in d or label_key not in d:
                    continue
                feat = d[feat_key]
                lab = d[label_key]
                mask = ~(np.isnan(feat) | np.isnan(lab))
                n = mask.sum()
                if n > 5000:
                    idx = np.where(mask)[0]
                    # Sample to keep runtime reasonable
                    if len(idx) > 50000:
                        idx = idx[::len(idx)//50000][:50000]
                    ic, _ = spearmanr(feat[idx], lab[idx])
                    if not np.isnan(ic):
                        ics.append(ic)

            ic_matrix[t][h] = {
                'mean': float(np.mean(ics)) if ics else 0,
                'std': float(np.std(ics)) if ics else 0,
                'min': float(np.min(ics)) if ics else 0,
                'max': float(np.max(ics)) if ics else 0,
                'n_days': len(ics),
                'values': [float(x) for x in ics]
            }

    # Print IC table
    header = '%8s' % 'Target'
    for h in horizons:
        header += ' | %12s' % h
    print(header)
    print('-' * 70)
    for t in targets:
        row = '%8s' % t.upper()
        for h in horizons:
            m = ic_matrix[t][h]
            row += ' | %5.4f±%.4f' % (m['mean'], m['std'])
        print(row)

    # ── 2. Cross-target correlations ──
    print("\n" + "=" * 70)
    print("2. Cross-target correlation (using label_10s)")
    print("=" * 70)

    cross_corr = {}
    # Use middle file for cross-correlation
    mid_f = files[len(files) // 2]
    d = np.load(mid_f)
    date = os.path.basename(mid_f).replace('_pressure.npz', '')
    print(f"Sample date: {date}")

    for i, t1 in enumerate(targets):
        for j, t2 in enumerate(targets):
            if j <= i:
                continue
            k1 = f'{t1}_label_10s'
            k2 = f'{t2}_label_10s'
            if k1 in d and k2 in d:
                v1 = d[k1]
                v2 = d[k2]
                mask = ~(np.isnan(v1) | np.isnan(v2))
                if mask.sum() > 5000:
                    idx = np.where(mask)[0][:50000]
                    ic, _ = spearmanr(v1[idx], v2[idx])
                    cross_corr[f'{t1}_vs_{t2}'] = float(ic)
                    print(f"  {t1.upper()} vs {t2.upper()}: {ic:.4f}")

    # ── 3. Autocorrelation structure ──
    print("\n" + "=" * 70)
    print("3. Autocorrelation (AC1) of labels by target and horizon")
    print("=" * 70)

    ac_matrix = {}
    for t in targets:
        ac_matrix[t] = {}
        for h in horizons:
            acs = []
            for f in files[::5]:  # sample every 5th
                d = np.load(f)
                k = f'{t}_label_{h}'
                if k in d:
                    v = d[k]
                    v = v[~np.isnan(v)][:200000]
                    if len(v) > 1000:
                        ac = np.corrcoef(v[:-1], v[1:])[0, 1]
                        if not np.isnan(ac):
                            acs.append(ac)
            ac_matrix[t][h] = float(np.mean(acs)) if acs else 0

    header = '%8s' % 'Target'
    for h in horizons:
        header += ' | %8s' % h
    print(header)
    print('-' * 55)
    for t in targets:
        row = '%8s' % t.upper()
        for h in horizons:
            row += ' |   %.4f' % ac_matrix[t][h]
        print(row)

    # ── 4. IC stability over time ──
    print("\n" + "=" * 70)
    print("4. IC stability: rolling 10-day mean IC for EOFI_label_10s (best target)")
    print("=" * 70)

    eofi_ics = ic_matrix['eofi']['10s']['values']
    if len(eofi_ics) >= 10:
        for i in range(0, len(eofi_ics) - 9, 10):
            chunk = eofi_ics[i:i+10]
            print(f"  Days {i+1:3d}-{i+10:3d}: mean IC={np.mean(chunk):.4f}, std={np.std(chunk):.4f}")

    # ── 5. Recommended head weights ──
    print("\n" + "=" * 70)
    print("5. RECOMMENDED V4 HEAD WEIGHTS (based on IC strength)")
    print("=" * 70)

    # Weight proportional to IC at 10s horizon
    ic_10s = {t: ic_matrix[t]['10s']['mean'] for t in targets}
    total_ic = sum(max(0, v) for v in ic_10s.values())

    if total_ic > 0:
        weights = {t: max(0.5, round(max(0, ic_10s[t]) / total_ic * 5, 1)) for t in targets}
    else:
        weights = {t: 1.0 for t in targets}

    # Directional head weight = 1.0 (baseline)
    print(f"  Head A (directional): weight = 1.0 (baseline)")
    for t in targets:
        letter = {'ntps': 'B', 'eofi': 'C', 'pdi': 'D', 'tia': 'E'}[t]
        print(f"  Head {letter} ({t.upper()}): weight = {weights[t]:.1f}  (IC_10s = {ic_10s[t]:.4f})")

    # ── Save results ──
    results = {
        'timestamp': datetime.now().isoformat(),
        'n_files': len(files),
        'ic_matrix': ic_matrix,
        'cross_correlations': cross_corr,
        'autocorrelation': ac_matrix,
        'recommended_weights': weights,
        'ic_10s_summary': ic_10s
    }

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT}")

    return results


if __name__ == '__main__':
    analyze()
