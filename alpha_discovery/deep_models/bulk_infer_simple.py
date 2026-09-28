#!/usr/bin/env python3
"""Efficient bulk inference using strided views — no copy per window."""
import os, sys, gc, time, logging, warnings
os.environ.setdefault('MAMBA_FEATURE_SET', 'smart_v3')
os.environ.setdefault('SKIP_NORMALIZE', '1')
os.environ.setdefault('EVENT_WINDOW_SIZE', '3000')
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger('bulk_infer')

import train_cnn_mamba_v2 as T

DATA_DIR = Path('/home/nick/Lvl3Quant/data/processed/mbo_events_smart_v3')
WEIGHTS_DIR = Path('/home/nick/Lvl3Quant/output/cnn_mamba_v2_smart_v3_mar')
OUT_DIR = Path('/home/nick/Lvl3Quant/output/cnn_mamba_v2_bulk_inference')
OUT_DIR.mkdir(parents=True, exist_ok=True)

FOLDS = [8, 9, 10]
DEVICE = 'cuda'
BATCH = 256
WINDOW = 3000
STRIDE = 250

def load_fold_model(fold_idx):
    path = WEIGHTS_DIR / f'fold_{fold_idx:02d}_best.pt'
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    arch = ckpt['arch']
    model = T.CNNMambaV2(
        d_model=arch['d_model'], d_state=arch['d_state'],
        n_layers=arch['n_layers'],
        dt_rank=arch.get('dt_rank', 16), d_conv=arch.get('d_conv', 4),
        dropout=arch.get('dropout', 0.1)
    )
    model.load_state_dict(ckpt['model_state'], strict=False)
    model.to(DEVICE).eval()
    log.info(f'  Loaded fold {fold_idx}: {sum(p.numel() for p in model.parameters()):,} params')
    return model

def infer_date(models, npz_path, date_str):
    data = np.load(npz_path)
    events = data['events']  # (N, 25) float32
    n_events = events.shape[0]
    n_feat = events.shape[1]
    
    if n_events < WINDOW:
        return None
    
    n_win = (n_events - WINDOW) // STRIDE + 1
    
    # Collect labels at end of each window
    labels_out = np.zeros((n_win, 3), dtype=np.float32)
    for j, hz in enumerate(['labels_1s', 'labels_5s', 'labels_10s']):
        if hz in data:
            lbl = data[hz]
            for w in range(n_win):
                idx = w * STRIDE + WINDOW - 1
                labels_out[w, j] = lbl[idx] if idx < len(lbl) else np.nan
    
    # Process in batches — create only batch-sized windows at a time
    all_preds = []
    for b_start in range(0, n_win, BATCH):
        b_end = min(b_start + BATCH, n_win)
        bs = b_end - b_start
        
        # Build batch tensor directly
        batch = np.zeros((bs, WINDOW, n_feat), dtype=np.float32)
        for i in range(bs):
            s = (b_start + i) * STRIDE
            batch[i] = events[s:s+WINDOW]
        
        x = torch.from_numpy(batch).to(DEVICE)
        
        batch_preds = []
        with torch.no_grad():
            for model in models:
                out = model(x)
                batch_preds.append(out.cpu().numpy())
        
        avg_pred = np.mean(batch_preds, axis=0)
        all_preds.append(avg_pred)
        del x, batch
    
    preds = np.concatenate(all_preds, axis=0)
    return {'predictions': preds, 'labels': labels_out,
            'horizons': np.array(['1s', '5s', '10s']),
            'date': date_str, 'n_windows': n_win,
            'window_size': WINDOW, 'stride': STRIDE,
            'fold_weights': np.array(FOLDS)}

def main():
    log.info('CNN-Mamba v2 Bulk Inference')
    log.info(f'  Window: {WINDOW}, Stride: {STRIDE}, Batch: {BATCH}')
    
    models = [load_fold_model(f) for f in FOLDS]
    npz_files = sorted(DATA_DIR.glob('*_mbo_events.npz'))
    log.info(f'Found {len(npz_files)} dates')
    
    done, skipped, failed = 0, 0, 0
    t0 = time.time()
    
    for i, npz_path in enumerate(npz_files):
        date_str = npz_path.stem.split('_')[0]
        out_path = OUT_DIR / f'{date_str}_predictions.npz'
        if out_path.exists():
            skipped += 1
            continue
        try:
            result = infer_date(models, npz_path, date_str)
            if result is None:
                failed += 1
                continue
            np.savez_compressed(out_path, **result)
            done += 1
            elapsed = time.time() - t0
            rate = done / elapsed * 60
            log.info(f'[{i+1}/{len(npz_files)}] {date_str}: {result["n_windows"]} windows | '
                    f'{done} done ({rate:.1f}/min), {skipped} skip, {failed} fail')
            gc.collect()
        except Exception as e:
            log.error(f'[{i+1}/{len(npz_files)}] {date_str} FAILED: {e}')
            import traceback; traceback.print_exc()
            failed += 1
            gc.collect()
            torch.cuda.empty_cache()
    
    log.info(f'COMPLETE: {done} processed, {skipped} skipped, {failed} failed in {time.time()-t0:.0f}s')

if __name__ == '__main__':
    main()
