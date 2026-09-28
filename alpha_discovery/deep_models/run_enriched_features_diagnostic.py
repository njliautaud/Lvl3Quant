"""
TRACK 1: Enriched Features CNN Diagnostic
==========================================
Tests BookSpatialCNN with feature_engineering.py derived features (4→16 features
per level) vs baseline 4-feature CNN on 10-fold quick diagnostic.

Architecture: Same WiderBookSpatialCNN (12.6M params) but with num_features=16
instead of 4. Feature engineering applied on GPU in the forward pass via the
existing BookFeatureEngineer module (all PyTorch ops, fully differentiable).

Top LGBM features that overlap with book signal space:
  - count_imb_top1, imb_top1 → order_flow_imbalance, book_asymmetry
  - depth_ratio_L1           → cumulative_depth_ratio
  - wmid_dev                 → pressure_gradient

10-fold diagnostic: 10 most recent dates used as folds for speed.
Saves: .pt weights + .npz predictions per fold. Full MLflow logging.
"""

import sys
import os
import json
import time
import gc
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

# Thread limits — BELOW_NORMAL priority for Neptune
os.environ.setdefault('OMP_NUM_THREADS', '16')
os.environ.setdefault('MKL_NUM_THREADS', '16')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '16')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from book_spatial_cnn import BookSpatialCNN
from feature_engineering import BookFeatureEngineer

# ── Config ────────────────────────────────────────────────────────────────────
BOOK_CACHE_DIR  = str(PROJECT_ROOT / 'data' / 'processed' / 'dl_book_cache')
OUTPUT_DIR      = SCRIPT_DIR / 'results' / 'enriched_features_diag'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_FOLDS          = 10      # quick diagnostic
MIN_TRAIN_DAYS   = 10
PURGE_DAYS       = 1
EPOCHS           = 2
BATCH_SIZE       = 256
SUBSAMPLE_TRAIN  = 10      # use every 10th bar during training for speed
WINDOW_SIZE      = 20
HORIZON_BARS     = 100     # 10s at 100ms bars
TICK_SIZE        = 0.25
NUM_WORKERS      = 8
PIN_MEMORY       = True

# Enriched: 4 raw + 12 from feature_engineering = 16 total features per level
# We use all 12 BookFeatureEngineer features
ENRICHED_FEATURES = {
    'order_flow_imbalance':    True,
    'pressure_gradient':       True,
    'spread':                  True,
    'queue_age_momentum':      True,
    'depth_change_velocity':   True,
    'book_asymmetry':          True,
    'cumulative_depth_ratio':  True,
    'queue_decay_rate':        True,
    'phantom_liquidity':       True,
    'book_renewal_asymmetry':  True,
    'cross_level_pressure':    True,
    'book_elasticity':         True,
}


# ── Wider CNN + enriched features ────────────────────────────────────────────

class WiderCNNEnriched(nn.Module):
    """
    WiderBookSpatialCNN (64,128,256,512) with enriched 16-feature input.

    Feature engineering is applied in forward() via BookFeatureEngineer.
    This keeps the model self-contained — no changes to the data pipeline.

    Input:  (B, T, 20, 4)   — raw 4-feature book window
    Output: (B, 1)           — mfe_net regression prediction
    """

    def __init__(
        self,
        window_size:       int   = 20,
        dropout:           float = 0.15,
        enriched_features: dict  = None,
    ):
        super().__init__()
        self.engineer = BookFeatureEngineer(
            features=enriched_features or ENRICHED_FEATURES
        )
        # Count how many features we'll have after engineering
        dummy = torch.zeros(1, window_size, 20, 4)
        with torch.no_grad():
            enriched_dummy = self.engineer(dummy)
        n_features = enriched_dummy.shape[-1]

        self.cnn = BookSpatialCNN(
            window_size=window_size,
            num_levels=20,
            num_features=n_features,
            spatial_channels=(64, 128, 256, 512),
            temporal_channels=512,
            dropout=dropout,
            num_classes=1,   # regression output
        )
        n_params = sum(p.numel() for p in self.parameters())
        print(f"WiderCNNEnriched: n_features={n_features}, params={n_params:,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 20, 4) — raw book window
        x_enr = self.engineer(x)   # (B, T, 20, n_features)
        return self.cnn(x_enr).squeeze(-1)   # (B,)


# ── Dataset ───────────────────────────────────────────────────────────────────

def compute_mfe_net(mid_prices: np.ndarray, day_boundaries: list,
                    horizon_bars: int = 100, tick_size: float = 0.25) -> np.ndarray:
    from numpy.lib.stride_tricks import sliding_window_view
    N  = len(mid_prices)
    H  = horizon_bars
    mfe_long  = np.full(N, np.nan, dtype=np.float32)
    mfe_short = np.full(N, np.nan, dtype=np.float32)
    valid_len = N - H - 1
    if valid_len > 0:
        mid_shifted = mid_prices[1:]
        windows  = sliding_window_view(mid_shifted, H)[:valid_len]
        mfe_long[:valid_len]  = np.maximum(0.0, (windows.max(1) - mid_prices[:valid_len]) / tick_size)
        mfe_short[:valid_len] = np.maximum(0.0, (mid_prices[:valid_len] - windows.min(1)) / tick_size)
    for d in range(len(day_boundaries) - 2):
        day_end   = day_boundaries[d + 1]
        nan_start = max(day_boundaries[d], day_end - H)
        mfe_long[nan_start:day_end]  = np.nan
        mfe_short[nan_start:day_end] = np.nan
    return (mfe_long - mfe_short).astype(np.float32)


class BookWindowDataset(Dataset):
    def __init__(self, day_data_list, target, day_boundaries,
                 window_size=20, subsample=1):
        self.window_size = window_size
        all_tensors = []
        for d in day_data_list:
            all_tensors.append(d['book_tensors'].astype(np.float16))  # float16 for RAM
        self.tensors = np.concatenate(all_tensors, axis=0)
        self.target  = target

        # Apply log-transforms (same as BookTensorDataset)
        t32 = self.tensors.astype(np.float32)
        t32[:, :, 1] = np.log1p(t32[:, :, 1])
        t32[:, :, 2] = np.log1p(t32[:, :, 2])
        t32[:, :, 3] = np.log1p(t32[:, :, 3])
        self.tensors = t32.astype(np.float16)

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
        i = self.valid[idx]
        w = self.tensors[i - self.window_size + 1 : i + 1].astype(np.float32)
        t = float(self.target[i])
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


# ── Training Loop ─────────────────────────────────────────────────────────────

def train_one_fold(model, train_loader, val_loader, device, n_epochs):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3,
        steps_per_epoch=len(train_loader),
        epochs=n_epochs,
    )
    scaler    = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    criterion = nn.HuberLoss(delta=1.0)

    for epoch in range(n_epochs):
        model.train()
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

    # Evaluate IC on val set
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            with torch.amp.autocast('cuda') if scaler else torch.no_grad():
                p = model(xb).cpu().float().numpy()
            preds.append(p)
            targets.append(yb.numpy())
    preds   = np.concatenate(preds)
    targets = np.concatenate(targets)
    mask    = np.isfinite(preds) & np.isfinite(targets)
    if mask.sum() < 10:
        return 0.0, preds, targets
    ic, _ = spearmanr(preds[mask], targets[mask])
    return float(ic) if np.isfinite(ic) else 0.0, preds, targets


# ── Main Walk-Forward ─────────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # MLflow setup
    mlrun = None
    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_tracking_uri('http://localhost:5000')
            mlflow.set_experiment('CNN_EnrichedFeatures')
            mlrun = mlflow.start_run(run_name=f'enriched_16feat_diag_{timestamp}')
            mlflow.log_params({
                'model':            'WiderCNNEnriched',
                'n_features':       16,
                'spatial_channels': '64,128,256,512',
                'temporal_channels': 512,
                'n_folds':          N_FOLDS,
                'epochs':           EPOCHS,
                'batch_size':       BATCH_SIZE,
                'subsample_train':  SUBSAMPLE_TRAIN,
                'window_size':      WINDOW_SIZE,
                'track':            'TRACK1_ENRICHED_FEATURES',
            })
            print(f"MLflow run: {mlrun.info.run_id}")
        except Exception as e:
            print(f"MLflow init failed (non-fatal): {e}")
            mlrun = None

    dates = get_dates(BOOK_CACHE_DIR)
    print(f"Available dates: {len(dates)} ({dates[0]} to {dates[-1]})")

    # Use last N_FOLDS + MIN_TRAIN_DAYS + PURGE_DAYS dates for quick diagnostic
    needed = N_FOLDS + MIN_TRAIN_DAYS + PURGE_DAYS
    if len(dates) < needed:
        raise ValueError(f"Need {needed} dates, only have {len(dates)}")
    use_dates = dates[-(needed):]
    print(f"Diagnostic using dates: {use_dates[0]} to {use_dates[-1]} ({len(use_dates)} total)")

    fold_ics   = []
    all_preds  = []
    all_tgts   = []

    for fold_i in range(N_FOLDS):
        test_idx       = MIN_TRAIN_DAYS + PURGE_DAYS + fold_i
        train_dates    = use_dates[:MIN_TRAIN_DAYS + fold_i]
        test_dates     = [use_dates[test_idx]]

        print(f"\n--- Fold {fold_i+1}/{N_FOLDS} | Train: {train_dates[0]}..{train_dates[-1]} | Test: {test_dates[0]}")
        t0 = time.time()

        # Load data
        tr_data, tr_mids, tr_bounds = load_book_days(BOOK_CACHE_DIR, train_dates)
        if not tr_data:
            print(f"  No train data, skipping")
            continue

        tr_target = compute_mfe_net(tr_mids, tr_bounds, HORIZON_BARS, TICK_SIZE)
        fin_mask  = np.isfinite(tr_target)
        if fin_mask.sum() < 1000:
            print(f"  Too few train samples ({fin_mask.sum()}), skipping")
            continue

        # Normalize target
        tgt_mean = float(tr_target[fin_mask].mean())
        tgt_std  = float(tr_target[fin_mask].std()) or 1.0
        tr_target = (tr_target - tgt_mean) / tgt_std

        train_ds = BookWindowDataset(tr_data, tr_target, tr_bounds,
                                     window_size=WINDOW_SIZE, subsample=SUBSAMPLE_TRAIN)
        del tr_data, tr_mids, tr_target, tr_bounds
        gc.collect()

        te_data, te_mids, te_bounds = load_book_days(BOOK_CACHE_DIR, test_dates)
        if not te_data:
            print(f"  No test data, skipping")
            continue

        te_target = compute_mfe_net(te_mids, te_bounds, HORIZON_BARS, TICK_SIZE)
        te_target = (te_target - tgt_mean) / tgt_std
        test_ds   = BookWindowDataset(te_data, te_target, te_bounds,
                                      window_size=WINDOW_SIZE, subsample=1)
        del te_data, te_mids, te_bounds
        gc.collect()

        if len(train_ds) < 100 or len(test_ds) < 10:
            print(f"  Dataset too small (train={len(train_ds)}, test={len(test_ds)}), skipping")
            continue

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                                  drop_last=True)
        test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

        # Build fresh model per fold (no warm-start for diagnostic — clean comparison)
        model = WiderCNNEnriched(window_size=WINDOW_SIZE).to(device)

        ic, preds, tgts = train_one_fold(model, train_loader, test_loader, device, EPOCHS)
        fold_ics.append(ic)
        all_preds.append(preds)
        all_tgts.append(tgts)

        elapsed = time.time() - t0
        print(f"  Fold {fold_i+1} IC={ic:.4f} | {elapsed:.0f}s | train={len(train_ds):,} test={len(test_ds):,}")

        # Save weights + predictions
        fold_model_path = OUTPUT_DIR / f'enriched_fold_{fold_i+1:03d}_{timestamp}.pt'
        torch.save(model.state_dict(), str(fold_model_path))
        fold_pred_path = OUTPUT_DIR / f'enriched_fold_{fold_i+1:03d}_{timestamp}_preds.npz'
        np.savez(str(fold_pred_path), predictions=preds, targets=tgts,
                 fold=fold_i+1, ic=ic, test_date=test_dates[0])

        # MLflow per-fold metrics
        if MLFLOW_AVAILABLE and mlrun:
            try:
                mlflow.log_metrics({'fold_ic': ic, 'fold_elapsed_s': elapsed}, step=fold_i+1)
            except Exception:
                pass

        # Free model memory
        del model, train_ds, test_ds, train_loader, test_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Aggregate results
    if fold_ics:
        agg_ic  = float(np.mean(fold_ics))
        ic_std  = float(np.std(fold_ics))
        icir    = agg_ic / ic_std if ic_std > 1e-8 else 0.0
        concat_preds  = np.concatenate(all_preds)
        concat_tgts   = np.concatenate(all_tgts)
        mask          = np.isfinite(concat_preds) & np.isfinite(concat_tgts)
        concat_ic, _  = spearmanr(concat_preds[mask], concat_tgts[mask])

        print(f"\n{'='*60}")
        print(f"ENRICHED FEATURES DIAGNOSTIC RESULTS")
        print(f"  Folds completed: {len(fold_ics)}/{N_FOLDS}")
        print(f"  Mean per-fold IC: {agg_ic:.4f} ± {ic_std:.4f}")
        print(f"  IC-IR:            {icir:.4f}")
        print(f"  Concat IC:        {float(concat_ic):.4f}  ← primary metric")
        print(f"  Baseline:         ~0.145 (wider CNN, folds 37-75)")
        print(f"  Lift vs baseline: {float(concat_ic) - 0.145:+.4f}")
        print(f"  Leakage audit:    PENDING")
        print(f"{'='*60}")

        # Save aggregate results
        summary = {
            'experiment': 'enriched_features_4to16_diagnostic',
            'track': 'TRACK1_ENRICHED_FEATURES',
            'n_folds': len(fold_ics),
            'fold_ics': fold_ics,
            'mean_fold_ic': agg_ic,
            'ic_std': ic_std,
            'icir': icir,
            'concat_ic': float(concat_ic),
            'baseline_ic': 0.145,
            'lift_vs_baseline': float(concat_ic) - 0.145,
            'n_features': 16,
            'timestamp': timestamp,
            'leakage_audit': 'PENDING',
        }
        summary_path = OUTPUT_DIR / f'summary_{timestamp}.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved: {summary_path}")

        if MLFLOW_AVAILABLE and mlrun:
            try:
                mlflow.log_metrics({
                    'agg_ic': agg_ic, 'ic_std': ic_std, 'icir': icir,
                    'concat_ic': float(concat_ic),
                    'lift_vs_baseline': float(concat_ic) - 0.145,
                })
                mlflow.log_artifact(str(summary_path))
                mlflow.end_run()
            except Exception:
                pass
    else:
        print("No folds completed!")

    return fold_ics


if __name__ == '__main__':
    main()
