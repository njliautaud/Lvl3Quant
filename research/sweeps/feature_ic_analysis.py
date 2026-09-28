"""
Feature IC analysis — subsampled, fast. ~5 min total on CPU.
Stride=250 subsample per file (matches training config), then Spearman IC.
"""
import numpy as np
from scipy.stats import spearmanr
import os, time

FEAT_DIR = '/home/jupiter/Lvl3Quant/data/processed/mbo_events_feat/'
OUT_FILE = '/home/jupiter/Lvl3Quant/feature_ic_results.txt'
STRIDE   = 250   # match CNN training stride — ~60K samples per large file

FEATURE_NAMES = [
    'time_delta_log', 'event_type_id', 'side_id', 'price_rel_ticks',
    'qty_log', 'spread_ticks', 'cancel_side_asym_50', 'rolling_ofi_500',
    'event_density_20', 'price_mom_10', 'qty_price_mom_50',
    'price_sign_mom_200', 'event_type_entropy_200',
    'fill_add_restore_100', 'spread_velocity_50',
]
N_FEAT   = 15
HORIZONS = ['labels_1s', 'labels_5s', 'labels_10s']

files = sorted([f for f in os.listdir(FEAT_DIR) if f.endswith('.npz')])
print(f'Files: {len(files)} | stride={STRIDE} | {files[0]} -> {files[-1]}', flush=True)

# Accumulate strided samples across all files then compute IC once
feat_bufs = [[] for _ in range(N_FEAT)]
lbl_bufs  = {h: [] for h in HORIZONS}

t0 = time.time()
for idx, fname in enumerate(files):
    d  = np.load(FEAT_DIR + fname, allow_pickle=True)
    ev = d['events']
    N  = ev.shape[0]
    if N < 500 or ev.shape[1] < N_FEAT:
        continue
    idx_s = np.arange(0, N, STRIDE)
    for i in range(N_FEAT):
        feat_bufs[i].append(ev[idx_s, i].astype(np.float32))
    for h in HORIZONS:
        lbl_bufs[h].append(d[h][idx_s].astype(np.float32))
    if (idx + 1) % 50 == 0:
        print(f'  [{idx+1}/{len(files)}] {time.time()-t0:.0f}s', flush=True)

print(f'Loaded in {time.time()-t0:.0f}s. Concatenating & computing IC...', flush=True)
feat_arrs = [np.concatenate(feat_bufs[i]) for i in range(N_FEAT)]
lbl_arrs  = {h: np.concatenate(lbl_bufs[h]) for h in HORIZONS}
N_total   = len(lbl_arrs['labels_10s'])
print(f'Total samples: {N_total:,}', flush=True)

rows = []
for i in range(N_FEAT):
    row = [FEATURE_NAMES[i]]
    for h in HORIZONS:
        feat = feat_arrs[i]
        lbl  = lbl_arrs[h]
        mask = np.isfinite(feat) & np.isfinite(lbl)
        ic   = spearmanr(feat[mask], lbl[mask]).correlation if mask.sum() > 200 else float('nan')
        row.append(round(float(ic), 4))
    rows.append(row)

rows.sort(key=lambda r: abs(r[3]) if not np.isnan(r[3]) else 0, reverse=True)
header = f"{'Feature':<28} {'IC_1s':>8} {'IC_5s':>8} {'IC_10s':>8}"
lines  = [header, '-'*56]
for r in rows:
    lines.append(f"{r[0]:<28} {r[1]:>8.4f} {r[2]:>8.4f} {r[3]:>8.4f}")

out = '\n'.join(lines)
print('\n' + out, flush=True)
with open(OUT_FILE, 'w') as f:
    f.write(out + '\n')
print(f'\nDone. Saved -> {OUT_FILE}', flush=True)
