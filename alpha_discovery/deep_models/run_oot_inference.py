#!/usr/bin/env python3
"""Inference-only on OOT data. Loads ONE day at a time. RAM-safe."""
import sys, os, numpy as np, torch, gc, time, psutil
from pathlib import Path
from datetime import datetime

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from book_spatial_cnn import BookSpatialCNN

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BookSpatialCNN(window_size=20, num_levels=20, num_features=4,
        spatial_channels=(32, 64, 128, 256), temporal_channels=256,
        dropout=0.1, num_classes=1).to(device)
    ckpt = ROOT / 'alpha_discovery' / 'deep_models' / 'results' / 'best_cnn_oot_20260311_015004.pt'
    state = torch.load(str(ckpt), map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    ram = psutil.virtual_memory().available / 1e9
    print(f'Model loaded on {device}. RAM: {ram:.1f}GB free', flush=True)

    oot_dir = ROOT / 'data' / 'processed' / 'dl_book_cache_oot'
    files = sorted(oot_dir.glob('*_book_tensors.npz'))
    print(f'{len(files)} OOT days', flush=True)

    ws = 20
    bs = 512
    chunk = 2000  # process 2000 bars at a time on GPU
    pred_data = {}
    t0 = time.time()

    for fi, f in enumerate(files):
        # RAM safety
        ram = psutil.virtual_memory().available / 1e9
        if ram < 3.0:
            print(f'ABORT: Only {ram:.1f}GB RAM free', flush=True)
            break

        date = f.name.replace('_book_tensors.npz', '')
        td = time.time()

        npz = np.load(str(f))
        bt = npz['book_tensors'].astype(np.float32)
        mid = npz['mid_prices'].copy()
        npz.close()
        np.log1p(bt[:, :, 1], out=bt[:, :, 1])
        np.log1p(bt[:, :, 2], out=bt[:, :, 2])
        np.log1p(bt[:, :, 3], out=bt[:, :, 3])
        n = len(bt)
        preds = np.zeros(n, dtype=np.float32)

        with torch.no_grad():
            for cstart in range(0, n, chunk):
                cend = min(cstart + chunk, n)
                slice_start = max(0, cstart - ws + 1)
                bt_chunk = bt[slice_start:cend].copy()
                bt_gpu = torch.from_numpy(bt_chunk).to(device)

                if len(bt_gpu) < ws:
                    del bt_gpu
                    continue

                windowed = bt_gpu.unfold(0, ws, 1).permute(0, 3, 1, 2).contiguous()
                pred_offset = slice_start + ws - 1

                for bstart in range(0, len(windowed), bs):
                    bend = min(bstart + bs, len(windowed))
                    if device.type == 'cuda':
                        with torch.amp.autocast('cuda'):
                            out = model(windowed[bstart:bend]).squeeze(-1)
                    else:
                        out = model(windowed[bstart:bend]).squeeze(-1)
                    g_start = pred_offset + bstart
                    g_end = pred_offset + bend
                    preds[g_start:g_end] = out.cpu().float().numpy()

                del bt_gpu, windowed
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

        pred_data[f'{date}_preds'] = preds.astype(np.float64)
        pred_data[f'{date}_mid'] = mid

        day_time = time.time() - td
        total = time.time() - t0
        ram = psutil.virtual_memory().available / 1e9
        print(f'[{fi+1}/{len(files)}] {date}: {n:,} bars {day_time:.1f}s '
              f'[{preds[ws:].min():.3f},{preds[ws:].max():.3f}] '
              f'RAM:{ram:.0f}GB ({total:.0f}s total)', flush=True)

        del bt, mid, preds
        gc.collect()

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = ROOT / 'alpha_discovery' / 'deep_models' / 'results' / f'oos_predictions_book_oot_{ts}.npz'
    np.savez_compressed(str(out_path), **pred_data)
    total = time.time() - t0
    print(f'DONE: {len(pred_data)//2} days in {total:.0f}s -> {out_path.name}', flush=True)

if __name__ == '__main__':
    main()
