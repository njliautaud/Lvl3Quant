import numpy as np
import json
from pathlib import Path

BOOK_DIR = Path('/home/jupiter/Lvl3Quant/data/processed/dl_book_cache_oot')
OF_DIR = Path('/home/jupiter/Lvl3Quant/data/processed/orderflow_features')
OF4_DIR = Path('/home/jupiter/Lvl3Quant/data/processed/of4_dom_depth')
OOT_NPZ = Path('/home/jupiter/Lvl3Quant/data/processed/wider_cnn_preds/oos_predictions_wider_cnn_20260318_080909.npz')
OUT = Path('/home/jupiter/multihorizon_ic_results.json')

TICK = 0.25
# Bar lags for each horizon (100ms per bar)
HORIZONS = {
    '100ms': 1,
    '500ms': 5,
    '1s': 10,
    '5s': 50,
    '10s': 100,
    '30s': 300,
    '60s': 600,
}

OF12_FEATURES = ['book_imbalance', 'cum_delta', 'roll_delta_10s', 'large_order_mask', 'large_order_count', 'total_bid_size', 'total_ask_size']
OF4_FEATURES = ['depth_imbalance_5', 'depth_imbalance_10', 'bid_depth_5', 'ask_depth_5', 'wall_imbalance', 'poc_dist_ticks']
ALL_FEATURES = OF12_FEATURES + OF4_FEATURES + ['cnn_pred']

oot = np.load(str(OOT_NPZ), allow_pickle=True)
dates = sorted(set(k[:-6] for k in oot.files if k.endswith('_preds')))

# acc[feature][horizon] = list of per-date ICs
acc = {f: {h: [] for h in HORIZONS} for f in ALL_FEATURES}
n_dates_used = 0

for date in dates:
    date_key = date  # '2025-07-22'
    date_nodash = date.replace('-', '')  # '20250722'

    # Load book cache (timestamps + mid_prices)
    bc_path = BOOK_DIR / f'{date_key}_book_tensors.npz'
    if not bc_path.exists():
        continue
    bc = np.load(str(bc_path), allow_pickle=True)
    mid = bc['mid_prices'].astype(np.float32)  # (234000,)

    # Load OF1/2 features
    of12_path = OF_DIR / f'{date_nodash}_orderflow.npz'
    if not of12_path.exists():
        continue
    of12 = np.load(str(of12_path), allow_pickle=True)

    # Load OF4 features
    of4_path = OF4_DIR / f'{date_nodash}_of4.npz'
    has_of4 = of4_path.exists()
    if has_of4:
        of4 = np.load(str(of4_path), allow_pickle=True)

    # CNN predictions
    cnn = oot[date_key + '_preds'].astype(np.float32)  # (233880,) or similar

    # Determine common length (CNN preds are shorter by warmup window)
    n_cnn = len(cnn)
    n_bars = len(mid)
    # CNN preds align to the END of the bar array (last n_cnn bars)
    bar_offset = n_bars - n_cnn  # typically 120

    for hor_name, lag in HORIZONS.items():
        # Forward returns in ticks: mid[i+lag] - mid[i]
        # For bar index i in [0, n_bars-lag), forward_ret[i] = (mid[i+lag] - mid[i]) / TICK
        # CNN preds are at bar indices [bar_offset, n_bars)
        # We need forward returns for those same bars, minus the tail (need i+lag < n_bars)
        # Valid range for CNN: bar_offset <= i < n_bars - lag
        i_start = bar_offset
        i_end = n_bars - lag
        if i_end <= i_start:
            continue

        fwd = (mid[i_start + lag:i_end + lag] - mid[i_start:i_end]) / TICK
        cnn_slice = cnn[:i_end - i_start]

        # CNN IC
        if len(fwd) > 10 and np.std(fwd) > 0 and np.std(cnn_slice) > 0:
            ic = float(np.corrcoef(cnn_slice, fwd)[0, 1])
            if not np.isnan(ic):
                acc['cnn_pred'][hor_name].append(ic)

        # OF features — full bar range [0, n_bars-lag)
        fwd_full = (mid[lag:] - mid[:-lag]) / TICK  # (n_bars-lag,)
        full_end = n_bars - lag

        for feat in OF12_FEATURES:
            if feat not in of12:
                continue
            arr = of12[feat].astype(np.float32)[:full_end]
            fwd_f = fwd_full[:len(arr)]
            if len(arr) < 10 or np.std(arr) < 1e-9 or np.std(fwd_f) < 1e-9:
                continue
            ic = float(np.corrcoef(arr, fwd_f)[0, 1])
            if not np.isnan(ic):
                acc[feat][hor_name].append(ic)

        if has_of4:
            for feat in OF4_FEATURES:
                if feat not in of4:
                    continue
                arr = of4[feat].astype(np.float32)[:full_end]
                fwd_f = fwd_full[:len(arr)]
                if len(arr) < 10 or np.std(arr) < 1e-9 or np.std(fwd_f) < 1e-9:
                    continue
                ic = float(np.corrcoef(arr, fwd_f)[0, 1])
                if not np.isnan(ic):
                    acc[feat][hor_name].append(ic)

    n_dates_used += 1

# Aggregate: mean IC per feature × horizon
result = {'n_dates': n_dates_used, 'horizons': list(HORIZONS.keys()), 'features': ALL_FEATURES, 'matrix': {}}
for feat in ALL_FEATURES:
    result['matrix'][feat] = {}
    for hor_name in HORIZONS:
        vals = acc[feat][hor_name]
        result['matrix'][feat][hor_name] = round(float(np.mean(vals)), 4) if vals else None

with open(str(OUT), 'w') as f:
    json.dump(result, f, indent=2)

raise RuntimeError('DONE n_dates=' + str(n_dates_used) + ' matrix_sample=' + str({f: result['matrix'][f] for f in ['cnn_pred', 'roll_delta_10s', 'book_imbalance']}))
