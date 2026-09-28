import numpy as np
import json
from pathlib import Path

BOOK_DIR = Path('/home/jupiter/Lvl3Quant/data/processed/dl_book_cache_oot')
OOT_NPZ = Path('/home/jupiter/Lvl3Quant/data/processed/wider_cnn_preds/oos_predictions_wider_cnn_20260318_080909.npz')
TICK = 0.25
BARS_PER_SESSION = 234000  # 6.5h RTH at 100ms per bar

# Horizons
LAG_10s = 100  # bars

# Z buckets
Z_BUCKETS = [(0.0,0.5),(0.5,1.0),(1.0,1.5),(1.5,2.0),(2.0,2.5),(2.5,9999.)]
Z_NAMES = ["0-0.5","0.5-1","1-1.5","1.5-2","2-2.5","2.5+"]

# ToD thirds: open=first 20%, midday=40-60%, close=last 20%
# We use bar index within session (0..233999)
TOD_OPEN_END = int(BARS_PER_SESSION * 0.20)
TOD_MID_START = int(BARS_PER_SESSION * 0.40)
TOD_MID_END = int(BARS_PER_SESSION * 0.60)
TOD_CLOSE_START = int(BARS_PER_SESSION * 0.80)

oot = np.load(str(OOT_NPZ), allow_pickle=True)
dates = sorted(set(k[:-6] for k in oot.files if k.endswith("_preds")))

# acc[zn][tod] = {n_correct, n_total, ic_sum, n_ic}
tods = ["open","midday","close","all"]
acc = {zn: {tod: {"correct": 0, "total": 0, "preds": [], "targs": []} for tod in tods} for zn in Z_NAMES}

n_dates = 0
for date in dates:
    bc_path = BOOK_DIR / f"{date}_book_tensors.npz"
    if not bc_path.exists():
        continue

    bc = np.load(str(bc_path), allow_pickle=True)
    mid = bc["mid_prices"].astype(np.float32)
    targ_stored = oot[date + "_targets"].astype(np.float64)
    cnn = oot[date + "_preds"].astype(np.float64)

    n_cnn = len(cnn)
    n_bars = len(mid)
    bar_offset = n_bars - n_cnn  # warmup offset

    # Standardize preds per day
    cnn_std = np.std(cnn)
    if cnn_std < 1e-9:
        continue
    z = (cnn - np.mean(cnn)) / cnn_std

    # Forward price return at 10s
    i_start = bar_offset
    i_end = n_bars - LAG_10s
    if i_end <= i_start:
        continue
    fwd = (mid[i_start + LAG_10s:i_end + LAG_10s] - mid[i_start:i_end]) / TICK

    cnn_sl = cnn[:i_end - i_start]
    z_sl = z[:i_end - i_start]

    # Bar indices within session (absolute bar index in the day)
    bar_indices = np.arange(i_start, i_end)  # absolute bar positions in the 234000-bar session

    for zn, (zlo, zhi) in zip(Z_NAMES, Z_BUCKETS):
        mask_z = (np.abs(z_sl) >= zlo) & (np.abs(z_sl) < zhi)

        for tod, (tod_start, tod_end) in zip(
            ["open","midday","close","all"],
            [(0, TOD_OPEN_END), (TOD_MID_START, TOD_MID_END), (TOD_CLOSE_START, BARS_PER_SESSION), (0, BARS_PER_SESSION)]
        ):
            mask_tod = (bar_indices >= tod_start) & (bar_indices < tod_end)
            mask = mask_z & mask_tod

            if mask.sum() < 5:
                continue

            preds_sub = cnn_sl[mask]
            fwd_sub = fwd[mask]

            # Directional accuracy: sign(pred) == sign(fwd) — exclude zero fwd
            nonzero = fwd_sub != 0
            if nonzero.sum() < 5:
                continue
            correct = int(np.sum(np.sign(preds_sub[nonzero]) == np.sign(fwd_sub[nonzero])))
            total = int(nonzero.sum())

            acc[zn][tod]["correct"] += correct
            acc[zn][tod]["total"] += total
            acc[zn][tod]["preds"].extend(preds_sub.tolist())
            acc[zn][tod]["targs"].extend(fwd_sub.tolist())

    n_dates += 1

# Compile results
results = {}
for zn in Z_NAMES:
    results[zn] = {}
    for tod in tods:
        a = acc[zn][tod]
        hit = round(a["correct"] / a["total"], 4) if a["total"] > 0 else None
        ic = round(float(np.corrcoef(a["preds"], a["targs"])[0,1]), 4) if len(a["preds"]) > 10 else None
        results[zn][tod] = {"hit_rate": hit, "ic": ic, "n": a["total"]}

with open("/home/jupiter/directional_accuracy_matrix.json", "w") as f:
    json.dump({"n_dates": n_dates, "results": results}, f, indent=2)

# Print key slice: midday IC and hit_rate per z-bucket
mid_summary = {zn: results[zn]["midday"] for zn in Z_NAMES}
all_summary = {zn: results[zn]["all"] for zn in Z_NAMES}
raise RuntimeError("DONE n_dates=" + str(n_dates) + " midday=" + str(mid_summary) + " all=" + str(all_summary))
