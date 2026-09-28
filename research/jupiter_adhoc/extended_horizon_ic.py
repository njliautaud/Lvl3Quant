import numpy as np
import json
from pathlib import Path

BOOK_DIR = Path('/home/jupiter/Lvl3Quant/data/processed/dl_book_cache_oot')
OF4_DIR = Path('/home/jupiter/Lvl3Quant/data/processed/of4_dom_depth')
OOT_NPZ = Path('/home/jupiter/Lvl3Quant/data/processed/wider_cnn_preds/oos_predictions_wider_cnn_20260318_080909.npz')
TICK = 0.25

# Extended horizons: 2min, 3min, 5min (in 100ms bars)
HORIZONS_EXT = {"1s": 10, "5s": 50, "10s": 100, "30s": 300, "60s": 600, "2min": 1200, "3min": 1800, "5min": 3000}

# Also test other OF4 features at extended horizons
FEATURES_OF4 = ["wall_imbalance", "depth_imbalance_5", "depth_imbalance_10", "poc_dist_ticks", "bid_wall", "ask_wall"]

oot = np.load(str(OOT_NPZ), allow_pickle=True)
dates = sorted(set(k[:-6] for k in oot.files if k.endswith("_preds")))

# acc[feature][horizon] = list of per-date ICs
acc = {f: {h: [] for h in HORIZONS_EXT} for f in FEATURES_OF4}
n_dates = 0

for date in dates:
    date_key = date
    date_nodash = date.replace("-", "")

    bc_path = BOOK_DIR / f"{date_key}_book_tensors.npz"
    of4_path = OF4_DIR / f"{date_nodash}_of4.npz"

    if not bc_path.exists() or not of4_path.exists():
        continue

    bc = np.load(str(bc_path), allow_pickle=True)
    mid = bc["mid_prices"].astype(np.float32)
    of4 = np.load(str(of4_path), allow_pickle=True)

    cnn = oot[date_key + "_preds"].astype(np.float64)
    n_cnn = len(cnn)
    n_bars = len(mid)
    bar_offset = n_bars - n_cnn

    for hor_name, lag in HORIZONS_EXT.items():
        i_start = bar_offset
        i_end = n_bars - lag
        if i_end <= i_start:
            continue
        fwd = (mid[i_start + lag:i_end + lag] - mid[i_start:i_end]) / TICK
        if np.std(fwd) < 1e-9:
            continue

        for feat in FEATURES_OF4:
            if feat not in of4:
                continue
            arr = of4[feat].astype(np.float64)
            if len(arr) < n_bars:
                continue
            feat_sl = arr[i_start:i_end]
            if np.std(feat_sl) < 1e-9 or len(feat_sl) != len(fwd):
                continue
            ic = float(np.corrcoef(feat_sl, fwd)[0, 1])
            if not np.isnan(ic):
                acc[feat][hor_name].append(ic)

    n_dates += 1

results = {}
for feat in FEATURES_OF4:
    results[feat] = {}
    for hor_name in HORIZONS_EXT:
        vals = acc[feat][hor_name]
        if vals:
            results[feat][hor_name] = round(float(np.mean(vals)), 4)
        else:
            results[feat][hor_name] = None

with open("/home/jupiter/extended_horizon_ic.json", "w") as f:
    json.dump({"n_dates": n_dates, "results": results}, f, indent=2)

wall_row = {h: results["wall_imbalance"].get(h) for h in HORIZONS_EXT}
di5_row = {h: results["depth_imbalance_5"].get(h) for h in HORIZONS_EXT}
raise RuntimeError("DONE n_dates=" + str(n_dates) + " wall_imbalance=" + str(wall_row) + " depth_imbalance_5=" + str(di5_row))
