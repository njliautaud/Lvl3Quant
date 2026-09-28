#!/usr/bin/env python3
"""
Fold 74 Decay Test — Run clean fold_74 checkpoint on OOT data (Nov 2025 - Mar 2026).
Measures whether the model's edge decays over 5 months out-of-sample.

Uses the WIDER architecture: spatial=(64,128,256,512), temporal=512.
Checkpoint: fold_74_2025-11-03.pt (last clean fold before contamination).
"""
import sys, os, numpy as np, torch, gc, time, psutil, json
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from book_spatial_cnn import BookSpatialCNN


def compute_mfe_net_single_day(mid_prices, horizon_bars=100, tick_size=0.25):
    """Compute mfe_net = mfe_long - mfe_short for a single day. No day boundary issues."""
    N = len(mid_prices)
    H = horizon_bars
    mfe_net = np.full(N, np.nan, dtype=np.float32)

    for i in range(N - H):
        window = mid_prices[i+1:i+1+H]
        entry = mid_prices[i]
        mfe_long = (np.max(window) - entry) / tick_size
        mfe_short = (entry - np.min(window)) / tick_size
        mfe_net[i] = mfe_long - mfe_short

    return mfe_net


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # WIDER architecture
    model = BookSpatialCNN(
        window_size=20, num_levels=20, num_features=4,
        spatial_channels=(64, 128, 256, 512),
        temporal_channels=512,
        dropout=0.15, num_classes=1
    ).to(device)

    ckpt_path = ROOT / 'alpha_discovery' / 'deep_models' / 'results' / 'wider_cnn' / 'checkpoints' / 'fold_74_2025-11-03.pt'
    state = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    ram = psutil.virtual_memory().available / 1e9
    print(f'Wider CNN loaded on {device}. RAM: {ram:.1f}GB free', flush=True)
    print(f'Checkpoint: {ckpt_path.name}', flush=True)

    # Load all OOT days (post fold 74 test date of 2025-11-03)
    data_dir = ROOT / 'data' / 'processed' / 'dl_book_cache'
    all_files = sorted(data_dir.glob('*_book_tensors.npz'))
    oot_files = [f for f in all_files if f.name[:10] > '2025-11-03']
    print(f'{len(oot_files)} OOT days to process', flush=True)

    ws = 20      # window size
    bs = 512     # batch size
    chunk = 2000 # GPU chunk size
    horizon = 100  # 10s at 100ms bars

    results = []
    t0 = time.time()

    for fi, f in enumerate(oot_files):
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

        # Apply same transforms as training
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

        # Compute targets (mfe_net)
        targets = compute_mfe_net_single_day(mid, horizon_bars=horizon)

        # Compute IC on valid bars only (have both prediction and target)
        valid = (np.arange(n) >= ws) & (~np.isnan(targets))
        if valid.sum() < 100:
            print(f'[{fi+1}/{len(oot_files)}] {date}: Too few valid bars ({valid.sum()}), skipping', flush=True)
            del bt, mid, preds, targets
            gc.collect()
            continue

        ic, pval = spearmanr(preds[valid], targets[valid])

        day_time = time.time() - td
        month = date[:7]
        results.append({
            'date': date,
            'month': month,
            'ic': float(ic),
            'pval': float(pval),
            'n_valid': int(valid.sum()),
            'pred_mean': float(preds[valid].mean()),
            'pred_std': float(preds[valid].std()),
        })

        print(f'[{fi+1}/{len(oot_files)}] {date}: IC={ic:+.4f} (p={pval:.4f}) '
              f'n={valid.sum():,} {day_time:.1f}s', flush=True)

        del bt, mid, preds, targets
        gc.collect()

    # Summary
    total_time = time.time() - t0
    print(f'\n{"="*60}', flush=True)
    print(f'FOLD 74 DECAY TEST — RESULTS', flush=True)
    print(f'{"="*60}', flush=True)
    print(f'Total: {len(results)} days in {total_time:.0f}s', flush=True)

    if results:
        ics = [r['ic'] for r in results]
        print(f'\nOverall: Mean IC = {np.mean(ics):+.4f}, Median = {np.median(ics):+.4f}', flush=True)
        print(f'Positive IC days: {sum(1 for x in ics if x > 0)}/{len(ics)} ({100*sum(1 for x in ics if x > 0)/len(ics):.0f}%)', flush=True)

        # By month
        months = sorted(set(r['month'] for r in results))
        print(f'\nMonthly breakdown:', flush=True)
        for m in months:
            m_ics = [r['ic'] for r in results if r['month'] == m]
            pos = sum(1 for x in m_ics if x > 0)
            print(f'  {m}: Mean IC = {np.mean(m_ics):+.4f}, Pos days = {pos}/{len(m_ics)} ({100*pos/len(m_ics):.0f}%)', flush=True)

        # Decay trend
        print(f'\nDecay analysis:', flush=True)
        print(f'  Clean WF baseline (folds 37-75): ~0.145', flush=True)
        overall_ic = np.mean(ics)
        if overall_ic > 0.10:
            print(f'  VERDICT: MODEL DURABLE — edge holds ({overall_ic:.3f} > 0.10)', flush=True)
        elif overall_ic > 0.05:
            print(f'  VERDICT: MILD DECAY — consider re-training every ~2 months ({overall_ic:.3f})', flush=True)
        else:
            print(f'  VERDICT: SIGNIFICANT DECAY — must continue WF training ({overall_ic:.3f})', flush=True)

    # Save results
    out_path = ROOT / 'alpha_discovery' / 'deep_models' / 'results' / 'wider_cnn' / 'fold74_decay_test_results.json'
    with open(str(out_path), 'w') as f:
        json.dump({
            'checkpoint': 'fold_74_2025-11-03.pt',
            'architecture': 'wider_cnn_64_128_256_512',
            'horizon_bars': horizon,
            'n_oot_days': len(results),
            'overall_mean_ic': float(np.mean(ics)) if results else None,
            'overall_median_ic': float(np.median(ics)) if results else None,
            'pct_positive': float(sum(1 for x in ics if x > 0) / len(ics) * 100) if results else None,
            'per_day': results,
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2)
    print(f'\nResults saved: {out_path.name}', flush=True)


if __name__ == '__main__':
    main()
