import numpy as np
import json
from pathlib import Path

BOOK_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/dl_book_cache_oot")
OOT_NPZ = Path("/home/jupiter/Lvl3Quant/data/processed/wider_cnn_preds/oos_predictions_wider_cnn_20260318_080909.npz")
TICK = 0.25

# After a fill, how does price move at various lags?
# Simulate: entry at bar i when |z| > threshold, measure price at i+lag vs i+0
# Adverse selection = price moving AGAINST us before the 10s target

ENTRY_LAGS = [1, 2, 5, 10, 20, 50, 100]  # bars after fill = 100ms, 200ms, 500ms, 1s, 2s, 5s, 10s
Z_THRESH = [(1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 9999)]
Z_NAMES = ["1-1.5", "1.5-2", "2-2.5", "2.5+"]

oot = np.load(str(OOT_NPZ), allow_pickle=True)
dates = sorted(set(k[:-6] for k in oot.files if k.endswith("_preds")))

# acc[zn][lag] = list of signed returns (positive = in direction of signal)
acc = {zn: {lag: [] for lag in ENTRY_LAGS} for zn in Z_NAMES}
n_dates = 0

for date in dates:
    bc_path = BOOK_DIR / f"{date}_book_tensors.npz"
    if not bc_path.exists():
        continue

    bc = np.load(str(bc_path), allow_pickle=True)
    mid = bc["mid_prices"].astype(np.float64)
    cnn = oot[date + "_preds"].astype(np.float64)

    n_cnn = len(cnn)
    n_bars = len(mid)
    bar_offset = n_bars - n_cnn

    cnn_std = np.std(cnn)
    if cnn_std < 1e-9:
        continue
    z = (cnn - np.mean(cnn)) / cnn_std

    # For each z-bucket, find signal bars and compute price path
    max_lag = max(ENTRY_LAGS)
    i_start = bar_offset
    i_end = n_bars - max_lag - 1

    if i_end <= i_start:
        continue

    cnn_sl = cnn[:i_end - i_start]
    z_sl = z[:i_end - i_start]

    for zn, (zlo, zhi) in zip(Z_NAMES, Z_THRESH):
        mask = (np.abs(z_sl) >= zlo) & (np.abs(z_sl) < zhi)
        signal_indices = np.where(mask)[0]  # indices into cnn_sl

        # Sample max 2000 signals per date per bucket to keep memory reasonable
        if len(signal_indices) > 2000:
            rng = np.random.default_rng(42)
            signal_indices = rng.choice(signal_indices, 2000, replace=False)

        for idx in signal_indices:
            bar_idx = i_start + idx  # absolute bar index
            signal_dir = np.sign(cnn_sl[idx])  # +1 long, -1 short
            entry_price = mid[bar_idx]

            for lag in ENTRY_LAGS:
                if bar_idx + lag >= n_bars:
                    continue
                future_price = mid[bar_idx + lag]
                # Signed return in direction of signal
                signed_ret = signal_dir * (future_price - entry_price) / TICK
                acc[zn][lag].append(signed_ret)

    n_dates += 1

# Compute mean signed return at each lag per z-bucket
results = {}
for zn in Z_NAMES:
    results[zn] = {}
    for lag in ENTRY_LAGS:
        vals = acc[zn][lag]
        if vals:
            arr = np.array(vals)
            results[zn][lag] = {
                "mean_ticks": round(float(np.mean(arr)), 4),
                "pct_positive": round(float(np.mean(arr > 0)), 4),
                "n": len(arr)
            }

with open("/home/jupiter/adverse_selection.json", "w") as f:
    json.dump({"n_dates": n_dates, "results": results}, f, indent=2)

# Key summary: mean signed return at 10s for each z-bucket
summary = {zn: results[zn].get(100, {}).get("mean_ticks") for zn in Z_NAMES}
pct_pos = {zn: results[zn].get(100, {}).get("pct_positive") for zn in Z_NAMES}
early = {zn: results[zn].get(5, {}).get("mean_ticks") for zn in Z_NAMES}  # 500ms
raise RuntimeError(f"DONE n_dates={n_dates} mean_10s={summary} pct_pos={pct_pos} mean_500ms={early}")
