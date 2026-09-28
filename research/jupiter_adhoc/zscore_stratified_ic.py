import numpy as np
import json
from pathlib import Path
from scipy import stats as scipy_stats

BOOK_DIR = Path('/home/jupiter/Lvl3Quant/data/processed/dl_book_cache_oot')
OF_DIR = Path('/home/jupiter/Lvl3Quant/data/processed/orderflow_features')
OF4_DIR = Path('/home/jupiter/Lvl3Quant/data/processed/of4_dom_depth')
OOT_NPZ = Path('/home/jupiter/Lvl3Quant/data/processed/wider_cnn_preds/oos_predictions_wider_cnn_20260318_080909.npz')
TICK = 0.25

# Horizons in bar lags (100ms per bar)
HORIZONS = {'1s': 10, '5s': 50, '10s': 100, '30s': 300, '60s': 600}

# Z-score buckets (absolute value thresholds)
Z_BUCKETS = [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 999.0)]
Z_NAMES = ['0-0.5', '0.5-1', '1-1.5', '1.5-2', '2-2.5', '2.5+']

oot = np.load(str(OOT_NPZ), allow_pickle=True)
dates = sorted(set(k[:-6] for k in oot.files if k.endswith('_preds')))

# Accumulators
# z_ic[bucket][horizon] = list of (pred, target) pairs across all dates
z_acc = {zn: {h: {'preds': [], 'targs': []} for h in HORIZONS} for zn in Z_NAMES}
z_counts = {zn: 0 for zn in Z_NAMES}

# Orthogonality: partial IC of depth_imbalance_5 after regressing out CNN pred
# acc_orth[horizon] = list of per-date partial correlations
orth_acc = {h: [] for h in HORIZONS}
# Also raw IC for comparison
di5_raw_acc = {h: [] for h in HORIZONS}
cnn_raw_acc = {h: [] for h in HORIZONS}

n_dates = 0

for date in dates:
    date_key = date
    date_nodash = date.replace('-', '')

    bc_path = BOOK_DIR / f'{date_key}_book_tensors.npz'
    of12_path = OF_DIR / f'{date_nodash}_orderflow.npz'
    of4_path = OF4_DIR / f'{date_nodash}_of4.npz'

    if not bc_path.exists() or not of12_path.exists():
        continue

    bc = np.load(str(bc_path), allow_pickle=True)
    mid = bc['mid_prices'].astype(np.float32)
    of12 = np.load(str(of12_path), allow_pickle=True)

    has_of4 = of4_path.exists()
    if has_of4:
        of4 = np.load(str(of4_path), allow_pickle=True)

    cnn = oot[date_key + '_preds'].astype(np.float64)
    targ = oot[date_key + '_targets'].astype(np.float64)  # roll_delta targets

    n_cnn = len(cnn)
    n_bars = len(mid)
    bar_offset = n_bars - n_cnn

    # Standardize CNN preds per day to get z-scores
    cnn_mean = np.mean(cnn)
    cnn_std = np.std(cnn)
    if cnn_std < 1e-9:
        continue
    z = (cnn - cnn_mean) / cnn_std
    abs_z = np.abs(z)

    # --- Part 1: Z-score stratified IC vs roll_delta targets ---
    # Targets are aligned 1:1 with preds (same length)
    for zn, (zlo, zhi) in zip(Z_NAMES, Z_BUCKETS):
        mask = (abs_z >= zlo) & (abs_z < zhi)
        if mask.sum() < 10:
            continue
        z_counts[zn] += int(mask.sum())
        # For multi-horizon: use stored targets for 10s
        # For other horizons, use bar-derived price returns
        # Store pred subset for aggregation
        z_acc[zn]['10s']['preds'].extend(cnn[mask].tolist())
        z_acc[zn]['10s']['targs'].extend(targ[mask].tolist())

    # For non-10s horizons in z-score analysis, use bar price returns
    for hor_name, lag in HORIZONS.items():
        if hor_name == '10s':
            continue  # handled above with proper targets
        i_start = bar_offset
        i_end = n_bars - lag
        if i_end <= i_start:
            continue
        fwd = (mid[i_start + lag:i_end + lag] - mid[i_start:i_end]) / TICK
        cnn_sl = cnn[:i_end - i_start]
        targ_sl = targ[:i_end - i_start]
        z_sl = z[:i_end - i_start]
        abs_z_sl = abs_z[:i_end - i_start]

        for zn, (zlo, zhi) in zip(Z_NAMES, Z_BUCKETS):
            mask = (abs_z_sl >= zlo) & (abs_z_sl < zhi)
            if mask.sum() < 10:
                continue
            z_acc[zn][hor_name]['preds'].extend(cnn_sl[mask].tolist())
            z_acc[zn][hor_name]['targs'].extend(fwd[mask].tolist())

    # --- Part 2: Orthogonality of depth_imbalance_5 vs CNN ---
    if not has_of4 or 'depth_imbalance_5' not in of4:
        continue

    di5 = of4['depth_imbalance_5'].astype(np.float64)

    for hor_name, lag in HORIZONS.items():
        i_start = bar_offset
        i_end = n_bars - lag
        if i_end <= i_start:
            continue
        fwd = (mid[i_start + lag:i_end + lag] - mid[i_start:i_end]) / TICK
        cnn_sl = cnn[:i_end - i_start]
        di5_sl = di5[i_start:i_end]
        n = len(fwd)

        if n < 50 or np.std(fwd) < 1e-9 or np.std(cnn_sl) < 1e-9 or np.std(di5_sl) < 1e-9:
            continue

        # Raw ICs
        ic_cnn = float(np.corrcoef(cnn_sl, fwd)[0, 1])
        ic_di5 = float(np.corrcoef(di5_sl, fwd)[0, 1])

        # Partial correlation of di5 with fwd after regressing out cnn
        # residual_fwd = fwd - proj(fwd onto cnn)
        # residual_di5 = di5 - proj(di5 onto cnn)
        cnn_norm = cnn_sl / (np.std(cnn_sl) + 1e-9)
        res_fwd = fwd - np.dot(fwd, cnn_norm) / n * cnn_norm
        res_di5 = di5_sl - np.dot(di5_sl, cnn_norm) / n * cnn_norm

        if np.std(res_fwd) < 1e-9 or np.std(res_di5) < 1e-9:
            continue

        partial_ic = float(np.corrcoef(res_di5, res_fwd)[0, 1])
        if not np.isnan(partial_ic) and not np.isnan(ic_cnn) and not np.isnan(ic_di5):
            orth_acc[hor_name].append(partial_ic)
            di5_raw_acc[hor_name].append(ic_di5)
            cnn_raw_acc[hor_name].append(ic_cnn)

    n_dates += 1

# Compute z-score stratified IC
z_results = {}
for zn in Z_NAMES:
    z_results[zn] = {'count': z_counts[zn]}
    for hor_name in HORIZONS:
        ps = z_acc[zn][hor_name]['preds']
        ts = z_acc[zn][hor_name]['targs']
        if len(ps) > 10:
            ic = float(np.corrcoef(ps, ts)[0, 1])
            z_results[zn][hor_name] = round(ic, 4)
        else:
            z_results[zn][hor_name] = None

# Compute orthogonality results
orth_results = {}
for hor_name in HORIZONS:
    orth_results[hor_name] = {
        'partial_ic_di5': round(float(np.mean(orth_acc[hor_name])), 4) if orth_acc[hor_name] else None,
        'raw_ic_di5': round(float(np.mean(di5_raw_acc[hor_name])), 4) if di5_raw_acc[hor_name] else None,
        'raw_ic_cnn': round(float(np.mean(cnn_raw_acc[hor_name])), 4) if cnn_raw_acc[hor_name] else None,
    }

result = {'n_dates': n_dates, 'z_stratified_ic': z_results, 'di5_orthogonality': orth_results}

with open('/home/jupiter/zscore_stratified_results.json', 'w') as f:
    json.dump(result, f, indent=2)

raise RuntimeError('DONE n_dates=' + str(n_dates) + ' z_results_sample=' + str({zn: z_results[zn].get('10s') for zn in Z_NAMES}) + ' orth_1s=' + str(orth_results.get('1s')))
