#!/usr/bin/env python3
"""
regenerate_v2_bulk_oot.py - HC #360 fix-up.

Regenerates per-day CNN-Mamba v2 OOT predictions with the CORRECT
window_size pulled from the checkpoint's `arch` dict (NOT hardcoded).

Background: output/cnn_mamba_v2_bulk_oot/ was generated May 3 with
EVENT_WINDOW_SIZE=3000 (trainer default) even though fold_10_best.pt's
arch records window_size=1000. All 46 per-day NPZs are corrupted
(IC_1s ~0.01 vs canonical 0.22).

This script:
  - Reads arch.window_size from fold_10_best.pt (= 1000) -- the fix.
  - Uses smart_v3 feature set => SKIP_NORMALIZE=True, 25 features.
  - stride=250 to match the broken bulk_oot metadata schema.
  - Saves per-day NPZ with same schema as the broken bulk_oot.
  - Parallelises across dates with multiprocessing.

Output: output/cnn_mamba_v2_bulk_oot_v2/
"""
from __future__ import annotations
import os, sys, json, time, hashlib, argparse, traceback
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

LVL3 = Path('/home/jupiter/Lvl3Quant')
OUT = LVL3 / 'output' / 'cnn_mamba_v2_bulk_oot_v2'
OUT.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUT / 'regen.log'

CKPT = LVL3 / 'output' / 'cnn_mamba_v2_smart_v3_mar' / 'fold_10_best.pt'
STATS = LVL3 / 'output' / 'cnn_mamba_v2_smart_v3_mar' / 'fold_09_feature_stats.npz'
MBO_DIR = LVL3 / 'data' / 'processed' / 'mbo_events_smart_v3'
SRC_BULK = LVL3 / 'output' / 'cnn_mamba_v2_bulk_oot'


def log(msg: str):
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG_PATH, 'a') as fp:
        fp.write(line + '\n')


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _setup_torch_env(n_threads: int):
    os.environ['MAMBA_FEATURE_SET'] = 'smart_v3'
    os.environ['SKIP_NORMALIZE'] = '1'
    os.environ['EVENT_STRIDE'] = '250'
    os.environ['OMP_NUM_THREADS'] = str(n_threads)
    os.environ['MKL_NUM_THREADS'] = str(n_threads)


def _worker(date_strs, batch, stride, torch_threads, worker_id):
    """Run inference on a list of dates in a single child process."""
    _setup_torch_env(torch_threads)
    sys.path.insert(0, '/tmp')
    sys.path.insert(0, str(LVL3 / 'live_trading_linux'))
    import torch
    torch.set_num_threads(torch_threads)
    import train_cnn_mamba_v2 as T

    ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
    state = ckpt['model_state']
    arch = ckpt.get('arch', {})
    kwargs = {}
    for k in ('d_model', 'd_state', 'n_layers', 'dt_rank', 'd_conv', 'dropout',
              'n_targets', 'feature_mlp_hidden', 'feature_mlp_out',
              'cnn_channels_per_scale'):
        if k in arch:
            kwargs[k] = arch[k]
    d_state = kwargs.get('d_state', 32)
    key = 'blocks.0.ssm.x_proj.weight'
    if key in state:
        kwargs['dt_rank'] = state[key].shape[0] - 2 * d_state
    ws = int(arch['window_size'])
    model = T.CNNMambaV2(**kwargs)
    model.load_state_dict(state, strict=False)
    model.eval()

    results = []
    for ds in date_strs:
        mbo = MBO_DIR / f'{ds}_mbo_events.npz'
        if not mbo.exists():
            log(f'[w{worker_id}] {ds}: SKIP (no MBO file)')
            results.append({'date': ds, 'skipped': True})
            continue
        try:
            d = np.load(mbo, allow_pickle=True)
            # Avoid redundant copy: dtype is already float32 on disk.
            events = d['events']
            if events.dtype != np.float32:
                events = events.astype(np.float32)
            lab1 = d['labels_1s']
            lab5 = d['labels_5s']
            lab10 = d['labels_10s']
            N_events = events.shape[0]
            n_windows = max(0, (N_events - ws) // stride + 1)
            if n_windows <= 0:
                log(f'[w{worker_id}] {ds}: SKIP (events<window)')
                continue

            preds = np.empty((n_windows, 3), dtype=np.float32)
            labs = np.empty((n_windows, 3), dtype=np.float32)
            t0 = time.time()
            with torch.no_grad():
                for bi in range(0, n_windows, batch):
                    be = min(bi + batch, n_windows)
                    bs = be - bi
                    chunks = np.empty((bs, ws, T.N_TOTAL_FEATURES), dtype=np.float32)
                    for li in range(bs):
                        w_idx = bi + li
                        s = w_idx * stride
                        e = s + ws
                        chunks[li] = events[s:e]
                        labs[w_idx, 0] = lab1[e - 1]
                        labs[w_idx, 1] = lab5[e - 1]
                        labs[w_idx, 2] = lab10[e - 1]
                    x = torch.from_numpy(chunks)
                    out = model(x)
                    if isinstance(out, tuple):
                        out = out[0]
                    if out.dim() == 3:
                        out = out[:, -1, :]
                    preds[bi:be] = out.numpy()
            elapsed = time.time() - t0

            def ic(p, l):
                m = np.isfinite(l) & (np.abs(l) > 1e-9)
                if m.sum() < 100: return float('nan')
                v = spearmanr(p[m], l[m]).correlation
                return float(v if not np.isnan(v) else 0.0)

            ic_1s = ic(preds[:, 0], labs[:, 0])
            ic_5s = ic(preds[:, 1], labs[:, 1])
            ic_10s = ic(preds[:, 2], labs[:, 2])

            out_path = OUT / f'{ds}_predictions.npz'
            np.savez(
                out_path,
                predictions=preds,
                labels=labs,
                oot_files=np.array([str(mbo)]),
                horizons=np.array(['1s', '5s', '10s']),
                date=np.array(ds),
                n_windows=np.int64(n_windows),
                window_size=np.int64(ws),
                stride=np.int64(stride),
            )

            log(f'[w{worker_id}] {ds}: n={n_windows:>6d} '
                f'pred_1s mean={preds[:,0].mean():+.4f} std={preds[:,0].std():.4f} '
                f'IC_1s={ic_1s:+.4f} IC_5s={ic_5s:+.4f} IC_10s={ic_10s:+.4f}  '
                f'infer={elapsed:.0f}s  rate={n_windows/max(elapsed,1e-3):.0f} win/s')

            results.append({
                'date': ds,
                'n_windows': int(n_windows),
                'pred_1s_mean': float(preds[:, 0].mean()),
                'pred_1s_std': float(preds[:, 0].std()),
                'ic_1s': ic_1s, 'ic_5s': ic_5s, 'ic_10s': ic_10s,
                'elapsed_s': float(elapsed),
            })
            # Persist worker partial after every date.
            with open(OUT / f'partial_w{worker_id}.json', 'w') as fp:
                json.dump(results, fp, indent=2)
        except Exception as e:
            log(f'[w{worker_id}] {ds}: ERROR {type(e).__name__}: {e}')
            log(traceback.format_exc())
            results.append({'date': ds, 'error': str(e)})

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dates', nargs='*', default=None)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--stride', type=int, default=250)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--torch-threads', type=int, default=4)
    args = ap.parse_args()

    log('=' * 70)
    log('regenerate_v2_bulk_oot.py START')
    log(f'CKPT  sha256 = {sha256(CKPT)}')
    log(f'STATS sha256 = {sha256(STATS)}')

    if args.dates:
        date_strs = list(args.dates)
    else:
        date_strs = sorted(
            p.name.split('_')[0]
            for p in SRC_BULK.glob('2026*_predictions.npz')
        )
    if args.limit:
        date_strs = date_strs[:args.limit]
    log(f'Dates to process: {len(date_strs)} ({date_strs[0]} -> {date_strs[-1]})')
    log(f'Workers={args.workers}  TorchThreads={args.torch_threads}  '
        f'Batch={args.batch}  Stride={args.stride}')

    buckets = [date_strs[i::args.workers] for i in range(args.workers)]
    for i, b in enumerate(buckets):
        if b:
            log(f'  worker {i}: {len(b)} dates ({b[0]} ... {b[-1]})')

    if args.workers == 1:
        results_all = _worker(date_strs, args.batch, args.stride,
                              args.torch_threads, 0)
    else:
        import multiprocessing as mp
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
        procs = []
        for i, bucket in enumerate(buckets):
            if not bucket:
                continue
            p = mp.Process(target=_worker,
                           args=(bucket, args.batch, args.stride,
                                 args.torch_threads, i))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
        results_all = []
        for i in range(args.workers):
            f = OUT / f'partial_w{i}.json'
            if f.exists():
                results_all.extend(json.loads(f.read_text()))

    with open(OUT / 'per_date_summary.json', 'w') as fp:
        json.dump(sorted(results_all, key=lambda r: r.get('date', '')),
                  fp, indent=2)
    log('=' * 70)
    log(f'DONE. {len(results_all)} dates written to {OUT}')


if __name__ == '__main__':
    main()
