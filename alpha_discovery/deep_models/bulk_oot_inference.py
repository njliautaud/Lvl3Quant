#!/usr/bin/env python3
"""
Bulk OOT Inference Script — CNN-Mamba v2 + PatchTST
====================================================
Generates per-date predictions for Mar 6 - Apr 29, 2026.

CNN-Mamba v2:
  - Mar 6 - Apr 15: fold_10 weights (cnn_mamba_v2_smart_v3_mar)
  - Apr 16 - Apr 29: warmstart fold_00 weights (cnn_mamba_v2_warmstart_apr)
  - Output: /home/nick/Lvl3Quant/output/cnn_mamba_v2_bulk_oot/YYYYMMDD_predictions.npz

PatchTST:
  - All dates: fold_12 weights (patchtst_razer_weights)
  - Output: /home/nick/Lvl3Quant/output/patchtst_bulk_oot/YYYYMMDD_predictions.npz

Each output file contains:
  predictions (N, 3), labels (N, 3), oot_files (list), horizons, date, window_size, stride
"""
import os, sys, gc, time, logging, warnings
os.environ.setdefault('MAMBA_FEATURE_SET', 'smart_v3')
os.environ.setdefault('SKIP_NORMALIZE', '1')
os.environ.setdefault('EVENT_WINDOW_SIZE', '3000')

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import scipy.stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger('bulk_oot')

# ── imports ──────────────────────────────────────────────────────────────────
import train_cnn_mamba_v2 as CNN_T
import train_event_patchtst as TST_T

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_DIR          = Path('/home/nick/Lvl3Quant/data/processed/mbo_events_smart_v3')
CNN_FOLD10_DIR    = Path('/home/nick/Lvl3Quant/output/cnn_mamba_v2_smart_v3_mar')
CNN_WARMSTART_DIR = Path('/home/nick/Lvl3Quant/output/cnn_mamba_v2_warmstart_apr')
TST_WEIGHTS_DIR   = Path('/home/nick/Lvl3Quant/output/patchtst_razer_weights')
CNN_OUT_DIR       = Path('/home/nick/Lvl3Quant/output/cnn_mamba_v2_bulk_oot')
TST_OUT_DIR       = Path('/home/nick/Lvl3Quant/output/patchtst_bulk_oot')

CNN_OUT_DIR.mkdir(parents=True, exist_ok=True)
TST_OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── date ranges ───────────────────────────────────────────────────────────────
# Mar 6 - Apr 15: fold_10; Apr 16 - Apr 29: warmstart fold_00
CNN_WARMSTART_CUTOFF = '20260416'   # first date to use warmstart
DATE_START = '20260306'
DATE_END   = '20260429'

# ── inference params ──────────────────────────────────────────────────────────
CNN_WINDOW = 3000
CNN_STRIDE = 250
TST_WINDOW = 500
TST_STRIDE = 250   # default from training script
BATCH      = 256
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'

HORIZONS = ['1s', '5s', '10s']


# ═══════════════════════════════════════════════════════════════════════════════
# Model loaders
# ═══════════════════════════════════════════════════════════════════════════════

def load_cnn_model(weights_path: Path) -> nn.Module:
    ckpt = torch.load(weights_path, map_location=DEVICE, weights_only=False)
    arch = ckpt.get('arch') or {}
    # Infer from model_state if arch missing
    if not arch:
        state = ckpt['model_state']
        d_model = state['cnn_to_model.0.weight'].shape[0]
        d_state = state['blocks.0.ssm.A_log'].shape[1]
        n_layers = sum(1 for k in state if k.startswith('blocks.') and k.endswith('.norm.weight'))
        arch = dict(d_model=d_model, d_state=d_state, n_layers=n_layers,
                    dt_rank=16, d_conv=4, dropout=0.1)
        log.info(f'  Inferred arch from weights: {arch}')
    model = CNN_T.CNNMambaV2(
        d_model   = arch['d_model'],
        d_state   = arch['d_state'],
        n_layers  = arch['n_layers'],
        dt_rank   = arch.get('dt_rank', 16),
        d_conv    = arch.get('d_conv', 4),
        dropout   = arch.get('dropout', 0.1),
    )
    model.load_state_dict(ckpt['model_state'], strict=False)
    model.to(DEVICE).eval()
    n = sum(p.numel() for p in model.parameters())
    log.info(f'  CNN-Mamba loaded: {n:,} params | val_ic_10s={ckpt.get("val_ic_10s","?")}')
    return model


def load_patchtst_model(weights_path: Path) -> nn.Module:
    ckpt = torch.load(weights_path, map_location=DEVICE, weights_only=False)
    arch = ckpt['arch']
    model = TST_T.PatchTST(
        n_features  = arch['n_features'],
        patch_size  = arch['patch_size'],
        d_model     = arch['d_model'],
        n_heads     = arch['n_heads'],
        head_dim    = arch['head_dim'],
        n_layers    = arch['n_layers'],
        ffn_dim     = arch['ffn_dim'],
        dropout     = arch.get('dropout', 0.1),
        window_size = arch['window_size'],
    )
    model.load_state_dict(ckpt['model_state'], strict=False)
    model.to(DEVICE).eval()
    n = sum(p.numel() for p in model.parameters())
    log.info(f'  PatchTST loaded: {n:,} params | val_ic_10s={ckpt.get("val_ic_10s","?")}')
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Core inference
# ═══════════════════════════════════════════════════════════════════════════════

def infer_date(model, npz_path: Path, window: int, stride: int, date_str: str):
    """Run inference on one date file. Returns dict or None."""
    data = np.load(npz_path)
    events = data['events'].astype(np.float32)   # (N, 25)
    n_events, n_feat = events.shape

    if n_events < window:
        log.warning(f'  {date_str}: only {n_events} events < window {window}, skipping')
        return None

    n_win = (n_events - window) // stride + 1

    # Collect labels at last event of each window
    labels_out = np.full((n_win, 3), np.nan, dtype=np.float32)
    for j, hz in enumerate(['labels_1s', 'labels_5s', 'labels_10s']):
        if hz in data:
            lbl = data[hz]
            for w in range(n_win):
                idx = w * stride + window - 1
                if idx < len(lbl):
                    labels_out[w, j] = lbl[idx]

    # Batch inference
    all_preds = []
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=(DEVICE == 'cuda')):
        for b_start in range(0, n_win, BATCH):
            b_end = min(b_start + BATCH, n_win)
            bs = b_end - b_start
            batch = np.zeros((bs, window, n_feat), dtype=np.float32)
            for i in range(bs):
                s = (b_start + i) * stride
                batch[i] = events[s:s+window]
            x = torch.from_numpy(batch).to(DEVICE, non_blocking=True)
            out = model(x)
            # Handle (preds, emb) tuple from CNN model with return_embedding=False default
            if isinstance(out, tuple):
                out = out[0]
            all_preds.append(out.float().cpu().numpy())
            del x, batch

    preds = np.concatenate(all_preds, axis=0)   # (n_win, 3)

    # Compute IC per horizon
    for h_idx, h in enumerate(HORIZONS):
        valid = ~np.isnan(labels_out[:, h_idx])
        if valid.sum() >= 20:
            ic = scipy.stats.spearmanr(preds[valid, h_idx], labels_out[valid, h_idx]).correlation
            log.info(f'    IC_{h}={ic:.4f}')

    return dict(
        predictions = preds.astype(np.float32),
        labels      = labels_out.astype(np.float32),
        oot_files   = np.array([str(npz_path)]),
        horizons    = np.array(HORIZONS),
        date        = date_str,
        n_windows   = n_win,
        window_size = window,
        stride      = stride,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def run_cnn_mamba():
    log.info('=' * 70)
    log.info('CNN-MAMBA v2 BULK OOT INFERENCE')
    log.info(f'Window={CNN_WINDOW}, Stride={CNN_STRIDE}, Batch={BATCH}, Device={DEVICE}')

    # Load both models upfront
    log.info(f'Loading fold_10 weights...')
    model_fold10     = load_cnn_model(CNN_FOLD10_DIR / 'fold_10_best.pt')
    log.info(f'Loading warmstart fold_00 weights...')
    model_warmstart  = load_cnn_model(CNN_WARMSTART_DIR / 'fold_00_best.pt')

    # Find target files
    all_files = sorted(DATA_DIR.glob('*_mbo_events.npz'))
    target = [f for f in all_files
              if DATE_START <= f.stem.split('_')[0] <= DATE_END]
    log.info(f'Target dates: {len(target)} files ({DATE_START} to {DATE_END})')

    done, skipped, failed = 0, 0, 0
    t0 = time.time()

    for i, npz_path in enumerate(target):
        date_str = npz_path.stem.split('_')[0]
        out_path = CNN_OUT_DIR / f'{date_str}_predictions.npz'

        if out_path.exists():
            log.info(f'[{i+1}/{len(target)}] {date_str} — already exists, skipping')
            skipped += 1
            continue

        # Choose weights
        model = model_warmstart if date_str >= CNN_WARMSTART_CUTOFF else model_fold10
        weights_label = 'warmstart_fold00' if date_str >= CNN_WARMSTART_CUTOFF else 'fold_10'
        log.info(f'[{i+1}/{len(target)}] {date_str} [{weights_label}]')

        try:
            result = infer_date(model, npz_path, CNN_WINDOW, CNN_STRIDE, date_str)
            if result is None:
                failed += 1
                continue
            np.savez_compressed(out_path, **result)
            done += 1
            elapsed = time.time() - t0
            rate = done / max(elapsed, 1) * 60
            log.info(f'  Saved {result["n_windows"]} windows | {done} done ({rate:.1f}/min)')
            gc.collect()
        except Exception as e:
            log.error(f'  FAILED: {e}')
            import traceback; traceback.print_exc()
            failed += 1
            gc.collect()
            torch.cuda.empty_cache()

    elapsed = time.time() - t0
    log.info(f'CNN-Mamba COMPLETE: {done} done, {skipped} skipped, {failed} failed in {elapsed:.0f}s')
    return done, failed


def run_patchtst():
    log.info('=' * 70)
    log.info('PATCHTST BULK OOT INFERENCE')
    log.info(f'Window={TST_WINDOW}, Stride={TST_STRIDE}, Batch={BATCH}, Device={DEVICE}')

    log.info('Loading PatchTST fold_12 weights...')
    model = load_patchtst_model(TST_WEIGHTS_DIR / 'fold_12_best.pt')

    # Find target files
    all_files = sorted(DATA_DIR.glob('*_mbo_events.npz'))
    target = [f for f in all_files
              if DATE_START <= f.stem.split('_')[0] <= DATE_END]
    log.info(f'Target dates: {len(target)} files ({DATE_START} to {DATE_END})')

    done, skipped, failed = 0, 0, 0
    t0 = time.time()

    for i, npz_path in enumerate(target):
        date_str = npz_path.stem.split('_')[0]
        out_path = TST_OUT_DIR / f'{date_str}_predictions.npz'

        if out_path.exists():
            log.info(f'[{i+1}/{len(target)}] {date_str} — already exists, skipping')
            skipped += 1
            continue

        log.info(f'[{i+1}/{len(target)}] {date_str} [patchtst_fold12]')
        try:
            result = infer_date(model, npz_path, TST_WINDOW, TST_STRIDE, date_str)
            if result is None:
                failed += 1
                continue
            np.savez_compressed(out_path, **result)
            done += 1
            elapsed = time.time() - t0
            rate = done / max(elapsed, 1) * 60
            log.info(f'  Saved {result["n_windows"]} windows | {done} done ({rate:.1f}/min)')
            gc.collect()
        except Exception as e:
            log.error(f'  FAILED: {e}')
            import traceback; traceback.print_exc()
            failed += 1
            gc.collect()
            torch.cuda.empty_cache()

    elapsed = time.time() - t0
    log.info(f'PatchTST COMPLETE: {done} done, {skipped} skipped, {failed} failed in {elapsed:.0f}s')
    return done, failed


if __name__ == '__main__':
    log.info(f'Device: {DEVICE}')
    if DEVICE == 'cuda':
        log.info(f'GPU: {torch.cuda.get_device_name(0)}')
        log.info(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')

    t_start = time.time()

    cnn_done, cnn_fail = run_cnn_mamba()
    tst_done, tst_fail = run_patchtst()

    total = time.time() - t_start
    log.info('=' * 70)
    log.info(f'ALL DONE in {total:.0f}s ({total/60:.1f}min)')
    log.info(f'CNN-Mamba: {cnn_done} dates processed, {cnn_fail} failed')
    log.info(f'PatchTST:  {tst_done} dates processed, {tst_fail} failed')
    log.info(f'CNN output: {CNN_OUT_DIR}')
    log.info(f'TST output: {TST_OUT_DIR}')
