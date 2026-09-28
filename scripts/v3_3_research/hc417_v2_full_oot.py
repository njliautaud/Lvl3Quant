#!/usr/bin/env python3
"""
hc417_v2_full_oot.py — HC #417 deliverable.

Produces ``output/hc417_v2_full_oot_56d.npz`` containing CNN-Mamba v2
predictions on the full 56-day OOT window (2026-02-23 → 2026-04-29).

Strategy:
  * 46 of 56 days already exist in output/cnn_mamba_v2_bulk_oot_v2/
    (window_size=1000, stride=250, ckpt=fold_10_best.pt, HC #360 correction).
  * Infer the missing 10 days (2026-02-23 → 2026-03-05) at the SAME
    settings using the exact pattern from regenerate_v2_bulk_oot.py.
  * Concatenate all 56 days into a single NPZ with v2's natural schema:
      pred_log_ret_1s/5s/10s (concatenated across days)
      target_log_ret_1s/5s/10s (concatenated)
      mask_log_ret_1s/5s/10s   (finite & |label|>1e-9)
      day_index               (which date each row belongs to, int)
      oot_dates               (list of 56 'YYYYMMDD' strings)
      per_day_n_windows       (n predictions per day)
      window_size, stride, ckpt_sha256

Run:
  python3 scripts/v3_3_research/hc417_v2_full_oot.py
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

LVL3 = Path('/home/jupiter/Lvl3Quant')
CKPT = LVL3 / 'output' / 'cnn_mamba_v2_smart_v3_mar' / 'fold_10_best.pt'
STATS = LVL3 / 'output' / 'cnn_mamba_v2_smart_v3_mar' / 'fold_09_feature_stats.npz'
MBO_DIR = LVL3 / 'data' / 'processed' / 'mbo_events_smart_v3'
SRC_BULK_V2 = LVL3 / 'output' / 'cnn_mamba_v2_bulk_oot_v2'   # canonical 46d
MISSING_OUT = LVL3 / 'output' / 'cnn_mamba_v2_bulk_oot_v2'   # write missing days here too
FINAL_NPZ = LVL3 / 'output' / 'hc417_v2_full_oot_56d.npz'
RUN_LOG = LVL3 / 'output' / 'hc417_v2_full_oot_56d_run.log'

OOT_DATES_ALL = [
    '20260223', '20260224', '20260225', '20260226', '20260227',
    '20260301', '20260302', '20260303', '20260304', '20260305',
    '20260306', '20260309', '20260310', '20260311', '20260312', '20260313',
    '20260315', '20260316', '20260317', '20260318', '20260319', '20260320',
    '20260322', '20260323', '20260324', '20260325', '20260326', '20260327',
    '20260329', '20260330', '20260331',
    '20260401', '20260402', '20260403', '20260405', '20260406', '20260407',
    '20260408', '20260409', '20260410', '20260412', '20260413', '20260414',
    '20260415', '20260416', '20260417', '20260419', '20260420', '20260421',
    '20260422', '20260423', '20260424', '20260426', '20260427', '20260428',
    '20260429',
]


def log(msg: str):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(RUN_LOG, 'a') as fp:
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
    os.environ['OMP_NUM_THREADS'] = str(n_threads)
    os.environ['MKL_NUM_THREADS'] = str(n_threads)


def _load_model(torch_threads: int):
    _setup_torch_env(torch_threads)
    sys.path.insert(0, '/tmp')
    sys.path.insert(0, str(LVL3 / 'live_trading_linux'))
    import torch
    torch.set_num_threads(torch_threads)
    import train_cnn_mamba_v2 as T  # noqa: N814

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
    proj_key = 'blocks.0.ssm.x_proj.weight'
    if proj_key in state:
        kwargs['dt_rank'] = state[proj_key].shape[0] - 2 * d_state
    ws = int(arch['window_size'])
    model = T.CNNMambaV2(**kwargs)
    model.load_state_dict(state, strict=False)
    model.eval()
    return torch, T, model, ws


def _worker(date_strs, batch, stride, torch_threads, worker_id):
    """Inference on a list of dates in a child process."""
    torch, T, model, ws = _load_model(torch_threads)
    results = []
    for ds in date_strs:
        mbo = MBO_DIR / f'{ds}_mbo_events.npz'
        out_path = MISSING_OUT / f'{ds}_predictions.npz'
        if out_path.exists():
            log(f'[w{worker_id}] {ds}: SKIP (already exists)')
            results.append({'date': ds, 'skipped': 'exists'})
            continue
        if not mbo.exists():
            log(f'[w{worker_id}] {ds}: SKIP (no MBO file)')
            results.append({'date': ds, 'skipped': 'no_mbo'})
            continue
        try:
            d = np.load(mbo, allow_pickle=True)
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
                results.append({'date': ds, 'skipped': 'too_few_events'})
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

            def ic(p, lab):
                m = np.isfinite(lab) & (np.abs(lab) > 1e-9)
                if m.sum() < 100:
                    return float('nan')
                v = spearmanr(p[m], lab[m]).correlation
                return float(v if not np.isnan(v) else 0.0)

            ic_1s = ic(preds[:, 0], labs[:, 0])
            ic_5s = ic(preds[:, 1], labs[:, 1])
            ic_10s = ic(preds[:, 2], labs[:, 2])

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
        except Exception as e:
            log(f'[w{worker_id}] {ds}: ERROR {type(e).__name__}: {e}')
            log(traceback.format_exc())
            results.append({'date': ds, 'error': str(e)})
    # Persist partial
    with open(MISSING_OUT / f'hc417_partial_w{worker_id}.json', 'w') as fp:
        json.dump(results, fp, indent=2)
    return results


def infer_missing(workers: int, torch_threads: int, batch: int, stride: int):
    """Run inference on the 10 missing dates (and any other absent from bulk_oot_v2)."""
    missing = [d for d in OOT_DATES_ALL
               if not (MISSING_OUT / f'{d}_predictions.npz').exists()]
    log(f'Missing days to infer: {len(missing)}: {missing}')
    if not missing:
        return
    buckets = [missing[i::workers] for i in range(workers)]
    for i, b in enumerate(buckets):
        if b:
            log(f'  worker {i}: {len(b)} dates ({b[0]} ... {b[-1]})')
    if workers == 1:
        _worker(missing, batch, stride, torch_threads, 0)
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
                           args=(bucket, batch, stride, torch_threads, i))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()


def assemble_full_npz(stride: int):
    """Concatenate all 56 per-day predictions into the deliverable NPZ."""
    log('=' * 60)
    log('Assembling full 56-day NPZ from per-day files')
    pred1, pred5, pred10 = [], [], []
    tgt1, tgt5, tgt10 = [], [], []
    day_index = []
    per_day_n = []
    dates_present = []
    failures = []
    ws_seen = set()
    stride_seen = set()
    for di, ds in enumerate(OOT_DATES_ALL):
        p = MISSING_OUT / f'{ds}_predictions.npz'
        if not p.exists():
            failures.append({'date': ds, 'reason': 'file_missing'})
            log(f'  {ds}: MISSING file -- will be excluded')
            per_day_n.append(0)
            continue
        d = np.load(p, allow_pickle=True)
        ws_seen.add(int(d['window_size']))
        stride_seen.add(int(d['stride']))
        preds = d['predictions']  # (N, 3)
        labels = d['labels']      # (N, 3)
        n = preds.shape[0]
        pred1.append(preds[:, 0]); pred5.append(preds[:, 1]); pred10.append(preds[:, 2])
        tgt1.append(labels[:, 0]); tgt5.append(labels[:, 1]); tgt10.append(labels[:, 2])
        day_index.append(np.full(n, di, dtype=np.int16))
        per_day_n.append(n)
        dates_present.append(ds)
        log(f'  {ds}: n={n}')

    pred1 = np.concatenate(pred1).astype(np.float32)
    pred5 = np.concatenate(pred5).astype(np.float32)
    pred10 = np.concatenate(pred10).astype(np.float32)
    tgt1 = np.concatenate(tgt1).astype(np.float32)
    tgt5 = np.concatenate(tgt5).astype(np.float32)
    tgt10 = np.concatenate(tgt10).astype(np.float32)
    day_index = np.concatenate(day_index)

    def mask(lab):
        return (np.isfinite(lab) & (np.abs(lab) > 1e-9)).astype(np.float32)

    mask1 = mask(tgt1)
    mask5 = mask(tgt5)
    mask10 = mask(tgt10)

    # Global ICs (sanity)
    def ic(p, lab, m):
        mm = m.astype(bool)
        if mm.sum() < 100:
            return float('nan')
        v = spearmanr(p[mm], lab[mm]).correlation
        return float(v if not np.isnan(v) else 0.0)

    ic_1s = ic(pred1, tgt1, mask1)
    ic_5s = ic(pred5, tgt5, mask5)
    ic_10s = ic(pred10, tgt10, mask10)

    # Per-day pass-rate metric (HC #415 rule 2 prerequisite — % of days with IC_10s > 0)
    per_day_ic_10s = []
    cursor = 0
    for di, n in enumerate(per_day_n):
        if n == 0:
            per_day_ic_10s.append(float('nan'))
            continue
        sl_p = pred10[cursor:cursor + n]
        sl_t = tgt10[cursor:cursor + n]
        sl_m = mask10[cursor:cursor + n].astype(bool)
        per_day_ic_10s.append(ic(sl_p, sl_t, sl_m))
        cursor += n
    per_day_ic_10s = np.array(per_day_ic_10s, dtype=np.float32)
    pass_rate_ic10s_pos = float(np.mean(per_day_ic_10s[~np.isnan(per_day_ic_10s)] > 0))

    log(f'GLOBAL IC_1s={ic_1s:+.4f}  IC_5s={ic_5s:+.4f}  IC_10s={ic_10s:+.4f}')
    log(f'days_with_data={len(dates_present)}/56  total_samples={pred1.shape[0]}')
    log(f'per_day_pass_rate(IC_10s>0)={pass_rate_ic10s_pos*100:.1f}%')
    log(f'window_size set: {ws_seen}  stride set: {stride_seen}')

    np.savez_compressed(
        FINAL_NPZ,
        pred_log_ret_1s=pred1,
        pred_log_ret_5s=pred5,
        pred_log_ret_10s=pred10,
        target_log_ret_1s=tgt1,
        target_log_ret_5s=tgt5,
        target_log_ret_10s=tgt10,
        mask_log_ret_1s=mask1,
        mask_log_ret_5s=mask5,
        mask_log_ret_10s=mask10,
        day_index=day_index,
        oot_dates=np.array(OOT_DATES_ALL),
        dates_present=np.array(dates_present),
        per_day_n_windows=np.array(per_day_n, dtype=np.int64),
        per_day_ic_10s=per_day_ic_10s,
        metric_ic_log_ret_1s=np.float64(ic_1s),
        metric_ic_log_ret_5s=np.float64(ic_5s),
        metric_ic_log_ret_10s=np.float64(ic_10s),
        metric_per_day_pass_rate_ic_10s_pos=np.float64(pass_rate_ic10s_pos),
        window_size=np.int64(next(iter(ws_seen)) if len(ws_seen) == 1 else -1),
        stride=np.int64(next(iter(stride_seen)) if len(stride_seen) == 1 else -1),
        ckpt_path=np.array(str(CKPT)),
        ckpt_sha256=np.array(sha256(CKPT)),
        feature_set=np.array('smart_v3'),
        skip_normalize=np.bool_(True),
        n_features=np.int64(25),
    )
    size_mb = FINAL_NPZ.stat().st_size / 1e6
    log(f'WROTE {FINAL_NPZ}  size={size_mb:.1f} MB')
    return {
        'final_npz': str(FINAL_NPZ),
        'size_mb': size_mb,
        'n_days_present': len(dates_present),
        'n_total_samples': int(pred1.shape[0]),
        'ic_1s': ic_1s, 'ic_5s': ic_5s, 'ic_10s': ic_10s,
        'per_day_pass_rate_ic_10s_pos': pass_rate_ic10s_pos,
        'window_size': sorted(ws_seen),
        'stride': sorted(stride_seen),
        'failures': failures,
        'dates_present': dates_present,
        'per_day_ic_10s': per_day_ic_10s.tolist(),
        'per_day_n_windows': per_day_n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=5)
    ap.add_argument('--torch-threads', type=int, default=2)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--stride', type=int, default=250)
    ap.add_argument('--skip-infer', action='store_true', help='Only assemble (assume per-day NPZ all exist).')
    ap.add_argument('--assemble-only', action='store_true', help='Alias for --skip-infer.')
    args = ap.parse_args()

    log('=' * 70)
    log('hc417_v2_full_oot.py START')
    log(f'CKPT  sha256 = {sha256(CKPT)}')
    log(f'STATS sha256 = {sha256(STATS)}')
    log(f'Workers={args.workers} TorchThreads={args.torch_threads} '
        f'Batch={args.batch} Stride={args.stride}')

    if not (args.skip_infer or args.assemble_only):
        infer_missing(args.workers, args.torch_threads, args.batch, args.stride)
    else:
        log('Skipping inference (assemble-only mode)')

    summary = assemble_full_npz(args.stride)
    summary_path = LVL3 / 'output' / 'hc417_v2_full_oot_56d_summary.json'
    with open(summary_path, 'w') as fp:
        json.dump(summary, fp, indent=2, default=str)
    log(f'Wrote summary -> {summary_path}')
    log('hc417_v2_full_oot.py DONE')


if __name__ == '__main__':
    main()
