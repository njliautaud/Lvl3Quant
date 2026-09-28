"""
TRACK 3: Temporal CNN 300-Frame Diagnostic
==========================================
10-fold quick diagnostic for all three temporal architectures:
  1. CausalAttentionCNN      (~14M params)
  2. ThreeDimensionalCNN     (~13M params)
  3. CausalTransformerCNN    (~15M params)

All use window_size=300 (30 seconds of context) to predict 10s ahead.

Runs ONE architecture at a time — pass --arch argument:
  python run_temporal_300f_diagnostic.py --arch causal_attn
  python run_temporal_300f_diagnostic.py --arch 3d_cnn
  python run_temporal_300f_diagnostic.py --arch causal_transformer

Results compared against baseline CNN (window=20, IC=0.145).
Hypothesis: temporal patterns building over 30s give IC > 0.15.

Saves: .pt weights + .npz predictions per fold. Full MLflow logging.
"""

import argparse
import sys
import os
import json
import time
import gc
from pathlib import Path
from datetime import datetime

os.environ.setdefault('OMP_NUM_THREADS', '16')
os.environ.setdefault('MKL_NUM_THREADS', '16')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '16')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from temporal_cnn_300f_models import build_temporal_model

# ── Config ────────────────────────────────────────────────────────────────────
BOOK_CACHE_DIR = str(PROJECT_ROOT / 'data' / 'processed' / 'dl_book_cache')
N_FOLDS        = 10
MIN_TRAIN_DAYS = 10
PURGE_DAYS     = 1
EPOCHS         = 2
BATCH_SIZE     = 64    # smaller than 2s-window CNN due to 300 frames × 20 × 4
SUBSAMPLE_TRAIN = 20   # subsample more aggressively: 300-frame windows are bigger
WINDOW_SIZE    = 300   # 30 seconds of 100ms bars
HORIZON_BARS   = 100   # 10s
TICK_SIZE      = 0.25
NUM_WORKERS    = 8
PIN_MEMORY     = True

ARCH_CONFIGS = {
    'causal_attn':        {'epochs': EPOCHS, 'batch_size': BATCH_SIZE},
    '3d_cnn':             {'epochs': EPOCHS, 'batch_size': 32},   # 3D convs are heavier
    'causal_transformer': {'epochs': EPOCHS, 'batch_size': BATCH_SIZE},
}


# ── Dataset ───────────────────────────────────────────────────────────────────

def compute_mfe_net(mid_prices, day_boundaries, horizon_bars=100, tick_size=0.25):
    from numpy.lib.stride_tricks import sliding_window_view
    N  = len(mid_prices)
    H  = horizon_bars
    mfe_long  = np.full(N, np.nan, dtype=np.float32)
    mfe_short = np.full(N, np.nan, dtype=np.float32)
    valid_len = N - H - 1
    if valid_len > 0:
        windows = sliding_window_view(mid_prices[1:], H)[:valid_len]
        mfe_long[:valid_len]  = np.maximum(0.0, (windows.max(1) - mid_prices[:valid_len]) / tick_size)
        mfe_short[:valid_len] = np.maximum(0.0, (mid_prices[:valid_len] - windows.min(1)) / tick_size)
    for d in range(len(day_boundaries) - 2):
        day_end = day_boundaries[d + 1]
        nan_s   = max(day_boundaries[d], day_end - H)
        mfe_long[nan_s:day_end]  = np.nan
        mfe_short[nan_s:day_end] = np.nan
    return (mfe_long - mfe_short).astype(np.float32)


class TemporalBookDataset(Dataset):
    """
    300-frame window dataset for temporal CNN experiments.
    Same structure as BookWindowDataset but window_size=300.
    Uses float16 storage to manage RAM.
    """
    def __init__(self, day_data_list, target, day_boundaries,
                 window_size=300, subsample=1):
        self.window_size = window_size

        all_tensors = []
        for d in day_data_list:
            t = d['book_tensors'].astype(np.float32)
            t[:, :, 1] = np.log1p(t[:, :, 1])
            t[:, :, 2] = np.log1p(t[:, :, 2])
            t[:, :, 3] = np.log1p(t[:, :, 3])
            all_tensors.append(t.astype(np.float16))
        self.tensors = np.concatenate(all_tensors, axis=0)
        self.target  = target

        valid = []
        for di in range(len(day_boundaries) - 1):
            s = day_boundaries[di]
            e = day_boundaries[di + 1]
            for i in range(s + window_size - 1, e - HORIZON_BARS):
                if np.isfinite(target[i]):
                    valid.append(i)
        if subsample > 1:
            valid = valid[::subsample]
        self.valid = valid

    def __len__(self): return len(self.valid)

    def __getitem__(self, idx):
        i  = self.valid[idx]
        w  = self.tensors[i - self.window_size + 1 : i + 1].astype(np.float32)
        t  = float(self.target[i])
        return torch.from_numpy(w), torch.tensor(t, dtype=torch.float32)


def load_book_days(cache_dir, dates):
    cache = Path(cache_dir)
    data_list, boundaries, all_mids = [], [0], []
    for date in dates:
        f = cache / f'{date}_book_tensors.npz'
        if not f.exists():
            continue
        npz = np.load(f)
        data_list.append(dict(npz))
        all_mids.append(npz['mid_prices'])
        boundaries.append(boundaries[-1] + len(npz['mid_prices']))
    mid_concat = np.concatenate(all_mids) if all_mids else np.array([])
    return data_list, mid_concat, boundaries


def get_dates(cache_dir):
    return sorted([f.name.replace('_book_tensors.npz', '')
                   for f in Path(cache_dir).glob('*_book_tensors.npz')])


# ── Training ──────────────────────────────────────────────────────────────────

def train_fold(model, train_loader, val_loader, device, n_epochs, arch):
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-4,
        steps_per_epoch=max(len(train_loader), 1),
        epochs=n_epochs,
    )
    scaler    = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    criterion = nn.HuberLoss(delta=1.0)

    for epoch in range(n_epochs):
        model.train()
        n_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            if scaler:
                with torch.amp.autocast('cuda'):
                    pred = model(xb)
                    loss = criterion(pred, yb)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            n_batches += 1
        print(f"    Epoch {epoch+1}/{n_epochs}: {n_batches} batches")

    # Evaluate
    model.eval()
    preds_all, tgts_all = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            if scaler:
                with torch.amp.autocast('cuda'):
                    p = model(xb).cpu().float().numpy()
            else:
                p = model(xb).cpu().numpy()
            preds_all.append(p)
            tgts_all.append(yb.numpy())

    preds   = np.concatenate(preds_all)
    targets = np.concatenate(tgts_all)
    mask    = np.isfinite(preds) & np.isfinite(targets)
    if mask.sum() < 10:
        return 0.0, preds, targets
    ic, _ = spearmanr(preds[mask], targets[mask])
    return float(ic) if np.isfinite(ic) else 0.0, preds, targets


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Temporal CNN 300-frame diagnostic')
    parser.add_argument('--arch', required=True,
                        choices=['causal_attn', '3d_cnn', 'causal_transformer'],
                        help='Architecture to run')
    parser.add_argument('--n-folds', type=int, default=N_FOLDS)
    parser.add_argument('--epochs',  type=int, default=EPOCHS)
    args = parser.parse_args()

    arch   = args.arch
    epochs = args.epochs
    n_folds = args.n_folds
    cfg    = ARCH_CONFIGS[arch]
    bs     = cfg['batch_size']

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir   = SCRIPT_DIR / 'results' / f'temporal_300f_{arch}'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"TRACK 3: Temporal CNN — {arch}")
    print(f"  Window: {WINDOW_SIZE} frames = 30 seconds of context")
    print(f"  Predict: 10s ahead (100 bars)")
    print(f"  Folds: {n_folds}, Epochs: {epochs}, Batch: {bs}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # MLflow
    mlrun = None
    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_tracking_uri('http://localhost:5000')
            mlflow.set_experiment('CNN_Temporal300')
            mlrun = mlflow.start_run(run_name=f'temporal_{arch}_{timestamp}')
            mlflow.log_params({
                'arch': arch,
                'window_size': WINDOW_SIZE,
                'horizon_bars': HORIZON_BARS,
                'n_folds': n_folds,
                'epochs': epochs,
                'batch_size': bs,
                'subsample_train': SUBSAMPLE_TRAIN,
                'track': 'TRACK3_TEMPORAL_CNN',
            })
            print(f"MLflow run: {mlrun.info.run_id}")
        except Exception as e:
            print(f"MLflow init failed (non-fatal): {e}")
            mlrun = None

    dates = get_dates(BOOK_CACHE_DIR)
    print(f"Available dates: {len(dates)} ({dates[0]} to {dates[-1]})")

    needed = n_folds + MIN_TRAIN_DAYS + PURGE_DAYS
    if len(dates) < needed:
        raise ValueError(f"Need {needed} dates, have {len(dates)}")
    use_dates = dates[-needed:]

    fold_ics, all_preds, all_tgts = [], [], []

    for fi in range(n_folds):
        test_idx    = MIN_TRAIN_DAYS + PURGE_DAYS + fi
        train_dates = use_dates[:MIN_TRAIN_DAYS + fi]
        test_dates  = [use_dates[test_idx]]

        print(f"\n--- Fold {fi+1}/{n_folds} | Train: {train_dates[0]}..{train_dates[-1]} ({len(train_dates)}d) | Test: {test_dates[0]}")
        t0 = time.time()

        tr_data, tr_mids, tr_bounds = load_book_days(BOOK_CACHE_DIR, train_dates)
        if not tr_data:
            print("  No train data, skipping")
            continue

        tr_target  = compute_mfe_net(tr_mids, tr_bounds, HORIZON_BARS, TICK_SIZE)
        fin_mask   = np.isfinite(tr_target)
        if fin_mask.sum() < 500:
            print(f"  Too few samples ({fin_mask.sum()}), skipping")
            continue

        tgt_mean = float(tr_target[fin_mask].mean())
        tgt_std  = float(tr_target[fin_mask].std()) or 1.0
        tr_target = (tr_target - tgt_mean) / tgt_std

        train_ds = TemporalBookDataset(tr_data, tr_target, tr_bounds,
                                       window_size=WINDOW_SIZE, subsample=SUBSAMPLE_TRAIN)
        del tr_data, tr_mids, tr_target, tr_bounds
        gc.collect()

        te_data, te_mids, te_bounds = load_book_days(BOOK_CACHE_DIR, test_dates)
        if not te_data:
            print("  No test data, skipping")
            continue

        te_target = compute_mfe_net(te_mids, te_bounds, HORIZON_BARS, TICK_SIZE)
        te_target = (te_target - tgt_mean) / tgt_std
        test_ds   = TemporalBookDataset(te_data, te_target, te_bounds,
                                        window_size=WINDOW_SIZE, subsample=1)
        del te_data, te_mids, te_bounds
        gc.collect()

        if len(train_ds) < 50 or len(test_ds) < 5:
            print(f"  Dataset too small, skipping")
            continue

        print(f"  Train samples: {len(train_ds):,} | Test samples: {len(test_ds):,}")

        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                                  drop_last=True)
        test_loader  = DataLoader(test_ds, batch_size=bs, shuffle=False,
                                  num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

        model = build_temporal_model(arch, window_size=WINDOW_SIZE).to(device)

        ic, preds, tgts = train_fold(model, train_loader, test_loader, device, epochs, arch)
        fold_ics.append(ic)
        all_preds.append(preds)
        all_tgts.append(tgts)

        elapsed = time.time() - t0
        print(f"  Fold {fi+1} IC={ic:.4f} | {elapsed:.0f}s")

        # Save artifacts
        torch.save(model.state_dict(),
                   str(out_dir / f'{arch}_fold_{fi+1:03d}_{timestamp}.pt'))
        np.savez(str(out_dir / f'{arch}_fold_{fi+1:03d}_{timestamp}_preds.npz'),
                 predictions=preds, targets=tgts, fold=fi+1, ic=ic,
                 test_date=test_dates[0], arch=arch)

        if MLFLOW_AVAILABLE and mlrun:
            try:
                mlflow.log_metrics({'fold_ic': ic, 'elapsed_s': elapsed}, step=fi+1)
            except Exception:
                pass

        del model, train_ds, test_ds, train_loader, test_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary
    if fold_ics:
        agg_ic      = float(np.mean(fold_ics))
        ic_std      = float(np.std(fold_ics))
        icir        = agg_ic / ic_std if ic_std > 1e-8 else 0.0
        all_p_cat   = np.concatenate(all_preds)
        all_t_cat   = np.concatenate(all_tgts)
        mask        = np.isfinite(all_p_cat) & np.isfinite(all_t_cat)
        concat_ic   = float(spearmanr(all_p_cat[mask], all_t_cat[mask])[0])

        print(f"\n{'='*60}")
        print(f"TEMPORAL CNN {arch.upper()} — DIAGNOSTIC RESULTS")
        print(f"  Folds:        {len(fold_ics)}/{n_folds}")
        print(f"  Mean fold IC: {agg_ic:.4f} ± {ic_std:.4f}")
        print(f"  IC-IR:        {icir:.4f}")
        print(f"  Concat IC:    {concat_ic:.4f}  ← primary metric (inflation-free)")
        print(f"  Baseline:     0.145 (wider CNN window=20)")
        print(f"  Lift:         {concat_ic - 0.145:+.4f}")
        print(f"  Leakage audit: PENDING")
        print(f"{'='*60}")

        summary = {
            'experiment': f'temporal_300f_{arch}_diagnostic',
            'track': 'TRACK3_TEMPORAL_CNN',
            'arch': arch,
            'window_size': WINDOW_SIZE,
            'n_folds': len(fold_ics),
            'fold_ics': fold_ics,
            'mean_fold_ic': agg_ic,
            'ic_std': ic_std,
            'icir': icir,
            'concat_ic': concat_ic,
            'baseline_ic': 0.145,
            'lift_vs_baseline': concat_ic - 0.145,
            'timestamp': timestamp,
            'leakage_audit': 'PENDING',
        }
        sp = out_dir / f'summary_{timestamp}.json'
        with open(sp, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary: {sp}")

        if MLFLOW_AVAILABLE and mlrun:
            try:
                mlflow.log_metrics({'agg_ic': agg_ic, 'ic_std': ic_std,
                                    'icir': icir, 'concat_ic': concat_ic,
                                    'lift_vs_baseline': concat_ic - 0.145})
                mlflow.log_artifact(str(sp))
                mlflow.end_run()
            except Exception:
                pass

    return fold_ics


if __name__ == '__main__':
    main()
