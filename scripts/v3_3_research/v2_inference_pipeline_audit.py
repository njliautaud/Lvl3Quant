#!/usr/bin/env python3
"""
v2_inference_pipeline_audit.py — HC #355 anomaly diagnostic.

Investigates whether the CNN-Mamba v2 IC collapse on Mar 6+ is:
  (A) A pipeline bug in cnn_mamba_v2_bulk_oot/ generation, OR
  (B) Real signal decay after the training fold cutoff (Mar 5).

Method:
  1. Compare prediction-distribution statistics (mean/std/quantiles) between
     fold-source NPZs (Feb 23..Mar 5, smart_v3_mar/fold_*_oot_predictions.npz)
     and per-day NPZs (Mar 6..Apr 24, cnn_mamba_v2_bulk_oot/*.npz).
  2. Re-run inference from scratch on a Mar 6+ date using the SAME checkpoint
     (fold_10_best.pt) + SAME feature_stats + SAME smart_v3 event file
     + SAME stride=250 (matches bulk_oot metadata) and compare IC and
     prediction stats to bulk_oot's stored predictions for that date.

Outputs to: output/v2_inference_pipeline_audit_20260514/

Constraints: read-only audit. No edits to trainer or paper-trader.
"""
from __future__ import annotations
import os, sys, json, time, hashlib
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

LVL3 = Path('/home/jupiter/Lvl3Quant')
OUT = LVL3 / 'output' / 'v2_inference_pipeline_audit_20260514'
OUT.mkdir(parents=True, exist_ok=True)

# Force CPU-friendly env per HC #307D / Jupiter CPU
os.environ.setdefault('MAMBA_FEATURE_SET', 'smart_v3')
os.environ.setdefault('SKIP_NORMALIZE', '1')
os.environ.setdefault('EVENT_WINDOW_SIZE', '3000')
os.environ.setdefault('EVENT_STRIDE', '250')

sys.path.insert(0, '/tmp')  # train_cnn_mamba_v2.py lives here
sys.path.insert(0, str(LVL3 / 'live_trading_linux'))

import torch

# ----------------------------------------------------------------------
# Step 1: enumerate prediction-distribution stats across all sources
# ----------------------------------------------------------------------
def npz_stats(path: Path) -> dict:
    d = np.load(path, allow_pickle=True)
    pred = d['predictions']
    lab = d['labels']
    out = {
        'path': str(path.name),
        'parent': path.parent.name,
        'n_rows': int(len(pred)),
        'oot_file': str(d['oot_files'][0]) if 'oot_files' in d else None,
    }
    for h_idx, h in enumerate(['1s','5s','10s']):
        if pred.shape[1] <= h_idx: continue
        p = pred[:, h_idx]; l = lab[:, h_idx]
        out[f'pred_{h}_mean'] = float(p.mean())
        out[f'pred_{h}_std'] = float(p.std())
        out[f'pred_{h}_q05'] = float(np.percentile(p, 5))
        out[f'pred_{h}_q95'] = float(np.percentile(p, 95))
        mask = np.isfinite(l) & (np.abs(l) > 1e-9)
        if mask.sum() > 100:
            ic = spearmanr(p[mask], l[mask]).correlation
            out[f'ic_{h}'] = float(ic if not np.isnan(ic) else 0.0)
            out[f'lab_{h}_mean'] = float(l[mask].mean())
            out[f'lab_{h}_std'] = float(l[mask].std())
        out[f'n_valid_lab_{h}'] = int(mask.sum())
    return out

def collect_all_stats():
    rows = []
    # fold-source (training-time OOT)
    fold_dir = LVL3 / 'output' / 'cnn_mamba_v2_smart_v3_mar'
    for f in sorted(fold_dir.glob('fold_*_oot_predictions.npz')):
        rows.append({**npz_stats(f), 'source': 'fold_oot'})
    # bulk_oot (per-day inference)
    bulk_dir = LVL3 / 'output' / 'cnn_mamba_v2_bulk_oot'
    for f in sorted(bulk_dir.glob('2026*_predictions.npz')):
        rows.append({**npz_stats(f), 'source': 'bulk_oot'})
    return rows

print('[audit] Step 1: collect prediction-distribution stats from BOTH sources')
rows = collect_all_stats()
import csv
csv_path = OUT / 'diff_table.csv'
keys = sorted({k for r in rows for k in r.keys()})
with open(csv_path, 'w', newline='') as fp:
    w = csv.DictWriter(fp, fieldnames=keys)
    w.writeheader()
    for r in rows:
        w.writerow(r)
print(f'  wrote {csv_path} ({len(rows)} rows)')

# ----------------------------------------------------------------------
# Step 2: hash checkpoint + stats
# ----------------------------------------------------------------------
print('[audit] Step 2: hash checkpoint + feature_stats')
def sha256(p):
    h = hashlib.sha256()
    with open(p,'rb') as fp:
        for chunk in iter(lambda: fp.read(1<<20), b''):
            h.update(chunk)
    return h.hexdigest()
ckpt_path = LVL3 / 'output' / 'cnn_mamba_v2_smart_v3_mar' / 'fold_10_best.pt'
stats_path = LVL3 / 'output' / 'cnn_mamba_v2_smart_v3_mar' / 'fold_09_feature_stats.npz'
print(f'  fold_10_best.pt sha256={sha256(ckpt_path)}')
print(f'  fold_09_feature_stats.npz sha256={sha256(stats_path)}')
stats = np.load(stats_path)
print(f'  feature_stats.mean[:5]={stats["mean"][:5]}, std[:5]={stats["std"][:5]}')
print(f'  feature_stats: n_features={len(stats["mean"])}, mean_is_zero={np.allclose(stats["mean"],0)}, std_is_one={np.allclose(stats["std"],1)}')

# ----------------------------------------------------------------------
# Step 3: re-run inference on Mar 6 from scratch using SAME ckpt
# ----------------------------------------------------------------------
print('[audit] Step 3: re-run inference on 20260306 from scratch')

import train_cnn_mamba_v2 as T

# Load mar 6 raw events
mbo = LVL3 / 'data' / 'processed' / 'mbo_events_smart_v3' / '20260306_mbo_events.npz'
d = np.load(mbo, allow_pickle=True)
events = d['events'].astype(np.float32)  # (N, base_features)
lab1 = d['labels_1s'].astype(np.float32)
lab5 = d['labels_5s'].astype(np.float32)
lab10 = d['labels_10s'].astype(np.float32)
N_events = events.shape[0]
print(f'  Mar 6 events: shape={events.shape}, n_total={N_events}')

# Build model exactly like inference engine
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
state = ckpt['model_state']
arch = ckpt.get('arch', {})
print(f'  ckpt arch: {arch}')
print(f'  ckpt epoch={ckpt.get("epoch")}, fold={ckpt.get("fold")}, val_ic_10s={ckpt.get("val_ic_10s")}')

kwargs = {}
for k in ('d_model','d_state','n_layers','dt_rank','d_conv','dropout','n_targets',
          'feature_mlp_hidden','feature_mlp_out','cnn_channels_per_scale'):
    if k in arch:
        kwargs[k] = arch[k]
d_state = kwargs.get('d_state', 32)
key = 'blocks.0.ssm.x_proj.weight'
if key in state:
    kwargs['dt_rank'] = state[key].shape[0] - 2 * d_state
ws = int(arch.get('window_size', 3000))
print(f'  Building CNNMambaV2 kwargs={kwargs}, window_size={ws}')
model = T.CNNMambaV2(**kwargs)
missing, unexpected = model.load_state_dict(state, strict=False)
print(f'  missing={missing}, unexpected={unexpected}')
model.eval()

# Generate windows at stride=250, window=3000 (matches bulk_oot metadata + training)
stride = 250
W = ws
n_windows_full = max(0, (N_events - W)//stride + 1)
# CPU-bound: sample a representative subset (uniform). Plenty for IC & dist test.
N_SAMPLE = min(int(os.environ.get('AUDIT_NSAMPLE', '4000')), n_windows_full)
sample_idx = np.linspace(0, n_windows_full-1, N_SAMPLE, dtype=int)
n_windows = N_SAMPLE
print(f'  Total windows={n_windows_full} stride={stride} W={W}; sampling {N_SAMPLE} uniformly')

# Reproduce training feature pipeline: USE_DERIVED_FEATURES?
print(f'  USE_DERIVED_FEATURES={T.USE_DERIVED_FEATURES}')
print(f'  N_TOTAL_FEATURES={T.N_TOTAL_FEATURES}')
print(f'  SKIP_NORMALIZE={T.SKIP_NORMALIZE}')

# Inference in batches
import torch
batch = 64
preds = np.empty((n_windows, 3), dtype=np.float32)
labs_aligned = np.empty((n_windows, 3), dtype=np.float32)
t0 = time.time()
with torch.no_grad():
    for bi in range(0, n_windows, batch):
        be = min(bi+batch, n_windows)
        chunks = []
        for local_idx in range(bi, be):
            w_idx = int(sample_idx[local_idx])
            s = w_idx * stride
            e = s + W
            ev = events[s:e]
            if T.USE_DERIVED_FEATURES:
                derived = T.compute_derived_features(ev)
                ev = np.concatenate([ev, derived], axis=1)
            chunks.append(ev)
            # label at last position of window
            li = e - 1
            labs_aligned[local_idx, 0] = lab1[li]
            labs_aligned[local_idx, 1] = lab5[li]
            labs_aligned[local_idx, 2] = lab10[li]
        x = torch.from_numpy(np.stack(chunks))  # (B, W, F)
        if not T.SKIP_NORMALIZE:
            # would normalize here but smart_v3 skips
            pass
        out = model(x)
        if isinstance(out, tuple): out = out[0]
        if out.dim() == 3:
            out = out[:, -1, :]  # last token
        preds[bi:be] = out.numpy()
        if bi % (batch*20) == 0:
            elapsed = time.time() - t0
            rate = (bi+batch)/max(elapsed,1e-3)
            print(f'    {bi+batch}/{n_windows} ({rate:.1f} win/s)')

elapsed = time.time() - t0
print(f'  Inference done in {elapsed:.1f}s')

# Save the re-run predictions
rerun_path = OUT / '20260306_rerun_predictions.npz'
np.savez(rerun_path, predictions=preds, labels=labs_aligned, oot_files=[str(mbo)],
         horizons=np.array(['1s','5s','10s']), n_windows=n_windows, window_size=W, stride=stride)
print(f'  saved {rerun_path}')

# Compare with bulk_oot stored — align on same window indices
stored = np.load(LVL3 / 'output' / 'cnn_mamba_v2_bulk_oot' / '20260306_predictions.npz', allow_pickle=True)
stored_pred_full = stored['predictions']
stored_lab_full = stored['labels']
# bulk_oot stored should also be stride=250 from start; align by sample_idx
stored_pred = stored_pred_full[sample_idx]
stored_lab = stored_lab_full[sample_idx]

# Compute IC and prediction stats
def ic_of(pred, lab, h_idx):
    mask = np.isfinite(lab[:,h_idx]) & (np.abs(lab[:,h_idx]) > 1e-9)
    if mask.sum() < 100: return None
    return float(spearmanr(pred[mask, h_idx], lab[mask, h_idx]).correlation)

print()
print('=== Comparison: re-run vs stored bulk_oot for 20260306 (aligned on same window indices) ===')
for h_idx, h in enumerate(['1s','5s','10s']):
    print(f'  Re-run     [{h}]: mean={preds[:,h_idx].mean():+.4f} std={preds[:,h_idx].std():.4f} IC={ic_of(preds, labs_aligned, h_idx)}')
    print(f'  Stored-sub [{h}]: mean={stored_pred[:,h_idx].mean():+.4f} std={stored_pred[:,h_idx].std():.4f} IC={ic_of(stored_pred, stored_lab, h_idx)}')
    print(f'  Stored-full[{h}]: mean={stored_pred_full[:,h_idx].mean():+.4f} std={stored_pred_full[:,h_idx].std():.4f} IC={ic_of(stored_pred_full, stored_lab_full, h_idx)}')
    print(f'  Pair-corr (n={len(preds)}): pearson={float(np.corrcoef(preds[:,h_idx], stored_pred[:,h_idx])[0,1]):+.4f}')

# Save final summary
summary = {
    'mar6_rerun_pred_1s_mean': float(preds[:,0].mean()),
    'mar6_rerun_pred_1s_std': float(preds[:,0].std()),
    'mar6_rerun_ic_1s': ic_of(preds, labs_aligned, 0),
    'mar6_rerun_ic_5s': ic_of(preds, labs_aligned, 1),
    'mar6_rerun_ic_10s': ic_of(preds, labs_aligned, 2),
    'mar6_stored_pred_1s_mean_full': float(stored_pred_full[:,0].mean()),
    'mar6_stored_pred_1s_std_full': float(stored_pred_full[:,0].std()),
    'mar6_stored_ic_1s_full': ic_of(stored_pred_full, stored_lab_full, 0),
    'mar6_stored_ic_5s_full': ic_of(stored_pred_full, stored_lab_full, 1),
    'mar6_stored_ic_10s_full': ic_of(stored_pred_full, stored_lab_full, 2),
    'mar6_stored_ic_1s_aligned': ic_of(stored_pred, stored_lab, 0),
    'mar6_pair_pearson_1s': float(np.corrcoef(preds[:,0], stored_pred[:,0])[0,1]),
    'mar6_pair_pearson_5s': float(np.corrcoef(preds[:,1], stored_pred[:,1])[0,1]),
    'mar6_pair_pearson_10s': float(np.corrcoef(preds[:,2], stored_pred[:,2])[0,1]),
    'ckpt_sha256': sha256(ckpt_path),
    'stats_sha256': sha256(stats_path),
    'n_windows_rerun': int(n_windows),
    'n_windows_stored_full': int(len(stored_pred_full)),
    'n_sampled': int(N_SAMPLE),
}
with open(OUT / 'summary.json', 'w') as fp:
    json.dump(summary, fp, indent=2)
print(f'\nWrote {OUT/"summary.json"}')
print(json.dumps(summary, indent=2))
