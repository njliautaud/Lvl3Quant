"""
Direction Classification CNN — predicts P(price goes UP) instead of mfe_net magnitude.

Uses BookSpatialCNN with binary cross-entropy loss.
Target: 1 if mfe_net > 0 (bullish), 0 if mfe_net <= 0 (bearish).
Uses balanced class weights to avoid learning base rate bias.

Output: sigmoid probability [0, 1] = P(bullish)
"""
import sys
import os
import gc
import time
import json
import logging
import numpy as np
from pathlib import Path

os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'alpha_discovery' / 'deep_models'))

import torch
import torch.nn as nn
from scipy.stats import spearmanr

import train_walkforward as twf
from book_spatial_cnn import BookSpatialCNN

RESULTS_DIR = PROJECT_ROOT / 'alpha_discovery' / 'deep_models' / 'results' / 'direction_cnn'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def run_direction_holdout(
    train_days=60, oot_days=20, epochs=5, batch_size=512,
    lr=3e-4, subsample=3, horizon=100, window_size=20, device_str='cuda'
):
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')

    cache_dir = Path(twf.DEFAULT_BOOK_DIR)
    day_files = sorted(cache_dir.glob('*_book_tensors.npz'))
    dates = [f.name.split('_book_tensors')[0] for f in day_files]

    train_dates = dates[:train_days]
    oot_dates = dates[train_days:train_days + oot_days]

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_path = RESULTS_DIR / f'direction_cnn_{timestamp}.log'
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s', datefmt='%H:%M:%S',
                        handlers=[logging.FileHandler(str(log_path)), logging.StreamHandler()])
    logger = logging.getLogger('direction_cnn')

    logger.info('=' * 60)
    logger.info('DIRECTION CLASSIFICATION CNN — HOLDOUT')
    logger.info(f'  Train: {train_dates[0]}..{train_dates[-1]} ({len(train_dates)}d)')
    logger.info(f'  OOT: {oot_dates[0]}..{oot_dates[-1]} ({len(oot_dates)}d)')
    logger.info(f'  Horizon: {horizon} bars | Window: {window_size} bars')
    logger.info(f'  Epochs: {epochs} | Batch: {batch_size} | LR: {lr}')
    logger.info('=' * 60)

    # Load data
    train_data, train_mids, train_bounds = twf.load_day_files(cache_dir, 'book', train_dates)
    oot_data, oot_mids, oot_bounds = twf.load_day_files(cache_dir, 'book', oot_dates)

    # Compute MFE target then convert to direction
    train_mfe = twf.compute_mfe_net(train_mids, train_bounds, horizon_bars=horizon)
    oot_mfe = twf.compute_mfe_net(oot_mids, oot_bounds, horizon_bars=horizon)

    # Direction: 1 = bullish (mfe_net > 0), 0 = bearish
    train_dir = np.where(np.isfinite(train_mfe), (train_mfe > 0).astype(np.float32), np.nan)
    oot_dir = np.where(np.isfinite(oot_mfe), (oot_mfe > 0).astype(np.float32), np.nan)

    # Compute class balance for weighted loss
    train_mask = np.isfinite(train_dir)
    pos_rate = train_dir[train_mask].mean()
    neg_rate = 1 - pos_rate
    # pos_weight = neg_rate / pos_rate (upweight minority class)
    pos_weight = torch.tensor([neg_rate / pos_rate]).to(device)
    logger.info(f'  Base rate: {pos_rate:.3f} bullish, {neg_rate:.3f} bearish')
    logger.info(f'  pos_weight: {pos_weight.item():.3f} (balances classes)')

    # Build datasets (use direction as target instead of mfe_net)
    train_ds = twf.BarDataset(train_data, train_dir, train_bounds,
                              model_type='book', window_size=window_size, subsample=subsample)
    oot_ds = twf.BarDataset(oot_data, oot_dir, oot_bounds,
                            model_type='book', window_size=window_size, subsample=1)

    logger.info(f'  Dataset: {len(train_ds):,} train | {len(oot_ds):,} OOT')

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                                num_workers=0, collate_fn=twf.collate_book)
    oot_loader = torch.utils.data.DataLoader(oot_ds, batch_size=batch_size, shuffle=False,
                                              num_workers=0, collate_fn=twf.collate_book)

    # Build model — same CNN but output is sigmoid probability
    model = BookSpatialCNN(window_size=window_size, dropout=0.15).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f'  Model: {n_params:,} params')

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Train
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        total_loss = 0
        total_n = 0
        for batch in train_loader:
            windows, targets = batch
            windows = windows.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            logits = model(windows).squeeze(-1)
            mask = torch.isfinite(targets)
            if mask.sum() == 0:
                continue
            loss = criterion(logits[mask], targets[mask])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * mask.sum().item()
            total_n += mask.sum().item()

        train_loss = total_loss / max(total_n, 1)

        # Evaluate
        model.eval()
        all_probs, all_targets, all_mfe = [], [], []
        with torch.no_grad():
            for batch in oot_loader:
                windows, targets = batch
                windows = windows.to(device)
                logits = model(windows).squeeze(-1)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)
                all_targets.append(targets.numpy())

        probs = np.concatenate(all_probs)
        targets = np.concatenate(all_targets)
        mask = np.isfinite(targets)
        probs = probs[mask]
        targets_clean = targets[mask]

        # Metrics
        from sklearn.metrics import accuracy_score, roc_auc_score
        pred_binary = (probs > 0.5).astype(int)
        acc = accuracy_score(targets_clean, pred_binary)
        auc = roc_auc_score(targets_clean, probs)
        pred_up_rate = pred_binary.mean()
        actual_up_rate = targets_clean.mean()
        # Balanced accuracy
        up_mask = targets_clean == 1
        down_mask = targets_clean == 0
        acc_up = pred_binary[up_mask].mean() if up_mask.sum() > 0 else 0
        acc_down = (1 - pred_binary[down_mask]).mean() if down_mask.sum() > 0 else 0
        balanced_acc = (acc_up + acc_down) / 2

        # Also compute IC on the probability (correlation with mfe direction)
        ic = spearmanr(probs, targets_clean)[0]

        elapsed = time.time() - t0
        logger.info(f'  Epoch {epoch+1}/{epochs} ({elapsed:.0f}s): loss={train_loss:.4f} | '
                    f'AUC={auc:.4f} Acc={acc:.3f} BalAcc={balanced_acc:.3f} IC={ic:.4f} | '
                    f'PredUP={pred_up_rate:.3f} ActualUP={actual_up_rate:.3f} | '
                    f'AccWhenUP={acc_up:.3f} AccWhenDOWN={acc_down:.3f}')

    # Save
    result = {
        'mode': 'direction_holdout',
        'horizon': horizon, 'window_size': window_size,
        'auc': float(auc), 'accuracy': float(acc), 'balanced_accuracy': float(balanced_acc),
        'ic': float(ic), 'pred_up_rate': float(pred_up_rate), 'actual_up_rate': float(actual_up_rate),
        'acc_when_up': float(acc_up), 'acc_when_down': float(acc_down),
        'params': n_params, 'epochs': epochs,
    }
    with open(RESULTS_DIR / f'direction_result_{timestamp}.json', 'w') as f:
        json.dump(result, f, indent=2)
    logger.info(f'\nFINAL: AUC={auc:.4f} BalAcc={balanced_acc:.3f} IC={ic:.4f}')
    return result

if __name__ == '__main__':
    run_direction_holdout()
