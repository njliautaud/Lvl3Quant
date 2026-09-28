"""
Fast Window Size Sweep — tests multiple window sizes with a small model.

Window sizes: [20, 50, 100, 200, 300] bars = [2s, 5s, 10s, 20s, 30s]
Model: spatial_channels=(32, 64, 128) — ~2-3M params (vs 12.6M for full)
Protocol: 10 folds, 2 epochs/fold, subsample=20x
Batch size: 512 for w<=50, 256 for w=100, 128 for w=200, 64 for w=300

Runs ALL window sizes sequentially and prints a final summary table.
Output: alpha_discovery/deep_models/results/window_sweep/
"""

import gc
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).resolve().parent
ROOT_DIR    = SCRIPT_DIR.parent.parent  # Lvl3Quant root (deep_models -> alpha_discovery -> Lvl3Quant)
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Import book model + walk-forward infrastructure
from book_spatial_cnn import BookSpatialCNN, SpatialResBlock, TemporalResBlock
from train_walkforward import (
    BarDataset,
    collate_book,
    compute_mfe_net,
    get_available_dates,
    load_day_files,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WINDOW_SIZES  = [20, 50, 100, 200, 300]
BATCH_SIZE_MAP = {20: 512, 50: 512, 100: 256, 200: 128, 300: 64}
SPATIAL_CHANNELS = (32, 64, 128)   # ~2-3M params
TEMPORAL_CHANNELS = 128
N_FOLDS_PER_WINDOW = 10            # folds to test per window size
MIN_TRAIN_DAYS = 5
PURGE_DAYS = 1
EPOCHS_PER_FOLD = 2
SUBSAMPLE = 20                     # use every 20th bar
HORIZON_BARS = 100                 # 10s at 100ms
LR = 3e-4

DEFAULT_BOOK_DIR = str(ROOT_DIR / 'data' / 'processed' / 'dl_book_cache')
OUTPUT_DIR = str(SCRIPT_DIR / 'results' / 'window_sweep')

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format='%(asctime)s [window_sweep] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger('window_sweep')


def setup_file_logger(output_dir: str, timestamp: str):
    log_path = Path(output_dir) / f'window_sweep_{timestamp}.log'
    fh = logging.FileHandler(str(log_path), mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s [window_sweep] %(levelname)s: %(message)s', datefmt='%H:%M:%S'
    ))
    logger.addHandler(fh)
    return log_path


# ---------------------------------------------------------------------------
# Small BookSpatialCNN with custom channel config
# ---------------------------------------------------------------------------

def build_small_model(window_size: int, device: torch.device) -> nn.Module:
    """Build BookSpatialCNN with small spatial_channels=(32,64,128)."""
    model = BookSpatialCNN(
        window_size=window_size,
        num_levels=20,
        num_features=4,
        spatial_channels=SPATIAL_CHANNELS,
        temporal_channels=TEMPORAL_CHANNELS,
        dropout=0.1,
        num_classes=1,
    )
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f'  Model: BookSpatialCNN(win={window_size}, ch={SPATIAL_CHANNELS}) — {n_params:,} params')
    return model.to(device)


# ---------------------------------------------------------------------------
# Train one epoch
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, device, scaler):
    model.train()
    criterion = nn.HuberLoss(delta=1.0)
    total_loss = 0.0
    total_n = 0

    for batch in loader:
        windows, targets = batch
        windows = windows.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                preds = model(windows).squeeze(-1)
                loss = criterion(preds, targets.float())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(windows).squeeze(-1)
            loss = criterion(preds, targets.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * len(targets)
        total_n += len(targets)

    return total_loss / max(total_n, 1)


# ---------------------------------------------------------------------------
# Evaluate one epoch
# ---------------------------------------------------------------------------

def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_tgts = []
    criterion = nn.HuberLoss(delta=1.0)
    total_loss = 0.0
    total_n = 0

    ctx = torch.amp.autocast('cuda') if device.type == 'cuda' else torch.no_grad()
    with ctx if device.type == 'cuda' else torch.no_grad():
        for batch in loader:
            windows, targets = batch
            windows = windows.to(device)
            with torch.no_grad():
                preds = model(windows).squeeze(-1).cpu().float().numpy()
            val_loss = criterion(torch.tensor(preds), targets.float()).item()
            all_preds.append(preds)
            all_tgts.append(targets.numpy())
            total_loss += val_loss * len(targets)
            total_n += len(targets)

    preds_arr = np.concatenate(all_preds)
    tgts_arr  = np.concatenate(all_tgts)
    mask = np.isfinite(preds_arr) & np.isfinite(tgts_arr)
    if mask.sum() < 10:
        return 0.0, total_loss / max(total_n, 1), preds_arr, tgts_arr
    ic, _ = spearmanr(preds_arr[mask], tgts_arr[mask])
    return float(ic) if np.isfinite(ic) else 0.0, total_loss / max(total_n, 1), preds_arr, tgts_arr


# ---------------------------------------------------------------------------
# Run one window size — returns result dict
# ---------------------------------------------------------------------------

def run_window_size(window_size: int, book_dir: str, device: torch.device,
                    all_dates: List[str], output_dir: str, timestamp: str,
                    mlflow_run_id: Optional[str] = None) -> Dict:
    batch_size = BATCH_SIZE_MAP[window_size]
    logger.info('')
    logger.info('=' * 70)
    logger.info(f'WINDOW SIZE: {window_size} bars ({window_size/10:.0f}s)')
    logger.info(f'  batch_size={batch_size}, subsample={SUBSAMPLE}x, epochs={EPOCHS_PER_FOLD}, folds={N_FOLDS_PER_WINDOW}')
    logger.info('=' * 70)

    t_window_start = time.time()

    # Use last N_FOLDS_PER_WINDOW + MIN_TRAIN_DAYS + PURGE_DAYS dates so all windows
    # test on the SAME dates (most recent folds) for fair comparison.
    total_dates_needed = N_FOLDS_PER_WINDOW + MIN_TRAIN_DAYS + PURGE_DAYS
    if len(all_dates) >= total_dates_needed:
        dates = all_dates[-total_dates_needed:]
    else:
        dates = all_dates

    n_total = len(dates)
    n_folds = n_total - MIN_TRAIN_DAYS - PURGE_DAYS
    logger.info(f'  Using {n_total} days: {dates[0]} .. {dates[-1]}')
    logger.info(f'  Total folds: {n_folds}')

    fold_ics = []
    fold_details = []
    all_oos_preds = []
    all_oos_tgts = []

    for fold_idx in range(n_folds):
        t_fold_start = time.time()

        test_day_idx = MIN_TRAIN_DAYS + PURGE_DAYS + fold_idx
        train_end    = test_day_idx - PURGE_DAYS
        train_dates  = dates[0:train_end]
        test_dates   = [dates[test_day_idx]]

        logger.info(f'\n--- Fold {fold_idx + 1}/{n_folds} | test={test_dates[0]} | train={len(train_dates)} days ---')

        # Load train data
        train_data_list, train_mids, train_bounds = load_day_files(book_dir, 'book', train_dates)
        if not train_data_list:
            logger.warning(f'  No train data, skipping')
            continue

        train_target = compute_mfe_net(train_mids, train_bounds, HORIZON_BARS)
        n_valid = int(np.isfinite(train_target).sum())
        if n_valid < 100:
            logger.warning(f'  Too few valid train samples ({n_valid}), skipping')
            del train_data_list, train_mids, train_target, train_bounds
            continue

        tgt_mean = float(np.nanmean(train_target))
        tgt_std  = float(np.nanstd(train_target))
        if tgt_std < 1e-8:
            tgt_std = 1.0
        train_target = (train_target - tgt_mean) / tgt_std

        train_dataset = BarDataset(
            train_data_list, train_target, train_bounds,
            model_type='book', window_size=window_size,
            subsample=SUBSAMPLE,
        )
        del train_data_list, train_mids, train_target, train_bounds
        gc.collect()

        # Load test data
        test_data_list, test_mids, test_bounds = load_day_files(book_dir, 'book', test_dates)
        if not test_data_list:
            logger.warning(f'  No test data, skipping')
            del train_dataset
            continue

        test_target = compute_mfe_net(test_mids, test_bounds, HORIZON_BARS)
        n_test_valid = int(np.isfinite(test_target).sum())
        if n_test_valid < 100:
            logger.warning(f'  Too few test samples ({n_test_valid}), skipping')
            del train_dataset, test_data_list, test_mids, test_target, test_bounds
            continue

        test_target = (test_target - tgt_mean) / tgt_std

        test_dataset = BarDataset(
            test_data_list, test_target, test_bounds,
            model_type='book', window_size=window_size,
            subsample=1,
        )
        del test_data_list, test_mids, test_target, test_bounds
        gc.collect()

        logger.info(f'  Dataset: {len(train_dataset):,} train | {len(test_dataset):,} test')

        if len(train_dataset) < 50:
            logger.warning(f'  Too few train items ({len(train_dataset)}), skipping')
            del train_dataset, test_dataset
            continue

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=collate_book, num_workers=0, pin_memory=False, drop_last=True,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size * 2, shuffle=False,
            collate_fn=collate_book, num_workers=0, pin_memory=False,
        )

        # Build fresh model each fold (no warm start)
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        model = build_small_model(window_size, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=LR,
            steps_per_epoch=max(len(train_loader), 1),
            epochs=EPOCHS_PER_FOLD,
        )

        best_ic = -999.0
        best_preds = None
        best_tgts = None
        epoch_stats = []

        for epoch in range(EPOCHS_PER_FOLD):
            t_ep = time.time()
            train_loss = train_epoch(model, train_loader, optimizer, device, scaler)
            scheduler.step()
            ic, val_loss, preds, tgts = evaluate(model, test_loader, device)

            epoch_stats.append({'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss, 'ic': ic})
            logger.info(f'    Epoch {epoch + 1}/{EPOCHS_PER_FOLD}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} IC={ic:+.4f}  [{time.time()-t_ep:.1f}s]')

            if ic > best_ic:
                best_ic = ic
                best_preds = preds.copy()
                best_tgts = tgts.copy()

        fold_ics.append(best_ic)
        if best_preds is not None:
            all_oos_preds.append(best_preds)
            all_oos_tgts.append(best_tgts)

        fold_details.append({
            'fold': fold_idx + 1,
            'test_date': test_dates[0],
            'train_days': len(train_dates),
            'ic': best_ic,
            'epoch_stats': epoch_stats,
        })

        logger.info(f'  Fold {fold_idx + 1} best IC: {best_ic:+.4f} [{time.time()-t_fold_start:.1f}s]')

        # Cleanup VRAM
        if device.type == 'cuda':
            model.cpu()
        del model, optimizer, scheduler, scaler, train_dataset, test_dataset, train_loader, test_loader
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # ----- Aggregate -----
    if not fold_ics:
        logger.warning(f'  No folds completed for window_size={window_size}')
        return {'window_size': window_size, 'error': 'No folds completed', 'fold_ics': []}

    ic_arr    = np.array(fold_ics)
    agg_ic    = float(ic_arr.mean())
    agg_std   = float(ic_arr.std())
    agg_icir  = float(agg_ic / agg_std) if agg_std > 0 else 0.0

    # Concat IC across all test folds
    if all_oos_preds:
        all_p = np.concatenate(all_oos_preds)
        all_t = np.concatenate(all_oos_tgts)
        mask  = np.isfinite(all_p) & np.isfinite(all_t)
        concat_ic = float(spearmanr(all_p[mask], all_t[mask])[0]) if mask.sum() > 10 else 0.0
    else:
        concat_ic = 0.0

    elapsed = time.time() - t_window_start
    logger.info('')
    logger.info(f'WINDOW {window_size} RESULTS:')
    logger.info(f'  Mean fold IC:  {agg_ic:+.4f}  (std={agg_std:.4f}  ICIR={agg_icir:+.3f})')
    logger.info(f'  Concat IC:     {concat_ic:+.4f}')
    logger.info(f'  Folds done:    {len(fold_ics)}/{N_FOLDS_PER_WINDOW}')
    logger.info(f'  Total time:    {elapsed/60:.1f}m')

    result = {
        'window_size':    window_size,
        'window_seconds': window_size / 10.0,
        'n_folds':        len(fold_ics),
        'fold_ics':       [float(x) for x in fold_ics],
        'agg_ic':         agg_ic,
        'agg_ic_std':     agg_std,
        'agg_icir':       agg_icir,
        'concat_ic':      concat_ic,
        'elapsed_sec':    elapsed,
        'fold_details':   fold_details,
        'config': {
            'batch_size':       BATCH_SIZE_MAP[window_size],
            'epochs_per_fold':  EPOCHS_PER_FOLD,
            'subsample':        SUBSAMPLE,
            'spatial_channels': list(SPATIAL_CHANNELS),
            'temporal_channels': TEMPORAL_CHANNELS,
        },
    }

    # Save per-window JSON
    out_path = Path(output_dir) / f'window_{window_size}bars_{timestamp}.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    logger.info(f'  Saved: {out_path.name}')

    # Log to MLflow if available
    if MLFLOW_AVAILABLE and mlflow_run_id:
        try:
            with mlflow.start_run(run_id=mlflow_run_id):
                mlflow.log_metrics({
                    f'win{window_size}_mean_ic': agg_ic,
                    f'win{window_size}_concat_ic': concat_ic,
                    f'win{window_size}_icir': agg_icir,
                })
        except Exception as e:
            logger.warning(f'  MLflow logging failed: {e}')

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = setup_file_logger(OUTPUT_DIR, timestamp)
    logger.info(f'Fast Window Sweep — log: {log_path}')
    logger.info(f'Output dir: {OUTPUT_DIR}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')
    if device.type == 'cuda':
        logger.info(f'GPU: {torch.cuda.get_device_name(0)}  VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB')

    # Get all available dates once
    all_dates = get_available_dates(DEFAULT_BOOK_DIR, 'book')
    logger.info(f'Available dates: {len(all_dates)} ({all_dates[0]} .. {all_dates[-1]})')

    if len(all_dates) < MIN_TRAIN_DAYS + PURGE_DAYS + 1:
        logger.error(f'Not enough dates ({len(all_dates)}) to run sweep')
        sys.exit(1)

    # MLflow experiment
    mlflow_run_id = None
    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_experiment('WindowSizeSweep')
            run = mlflow.start_run(run_name=f'window_sweep_{timestamp}')
            mlflow_run_id = run.info.run_id
            mlflow.log_params({
                'window_sizes': str(WINDOW_SIZES),
                'spatial_channels': str(SPATIAL_CHANNELS),
                'n_folds': N_FOLDS_PER_WINDOW,
                'epochs_per_fold': EPOCHS_PER_FOLD,
                'subsample': SUBSAMPLE,
            })
            mlflow.end_run()
            logger.info(f'MLflow run_id: {mlflow_run_id}')
        except Exception as e:
            logger.warning(f'MLflow init failed: {e}')

    # Run all window sizes
    all_results = []
    t_total = time.time()

    for window_size in WINDOW_SIZES:
        result = run_window_size(
            window_size=window_size,
            book_dir=DEFAULT_BOOK_DIR,
            device=device,
            all_dates=all_dates,
            output_dir=OUTPUT_DIR,
            timestamp=timestamp,
            mlflow_run_id=mlflow_run_id,
        )
        all_results.append(result)
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    total_elapsed = time.time() - t_total

    # ----- Final Summary Table -----
    logger.info('')
    logger.info('=' * 70)
    logger.info('FINAL SUMMARY — Window Size Sweep')
    logger.info(f'Model: BookSpatialCNN channels={SPATIAL_CHANNELS}, {N_FOLDS_PER_WINDOW} folds, {EPOCHS_PER_FOLD} epochs, subsample={SUBSAMPLE}x')
    logger.info('=' * 70)
    logger.info(f'{"Window":>8}  {"Secs":>6}  {"Folds":>6}  {"MeanIC":>8}  {"ConcatIC":>10}  {"ICIR":>7}  {"Time(m)":>8}')
    logger.info('-' * 70)

    best_mean_ic = -999.0
    best_window = None

    for r in all_results:
        if 'error' in r:
            logger.info(f'{r["window_size"]:>8}  {"?":>6}  {"ERR":>6}  {"?":>8}  {"?":>10}  {"?":>7}  {"?":>8}')
            continue
        secs   = r['window_seconds']
        folds  = r['n_folds']
        mic    = r['agg_ic']
        cic    = r['concat_ic']
        icir   = r['agg_icir']
        mins   = r['elapsed_sec'] / 60.0
        star   = ' <-- BEST' if mic > best_mean_ic else ''
        logger.info(f'{r["window_size"]:>8}  {secs:>6.0f}  {folds:>6}  {mic:>+8.4f}  {cic:>+10.4f}  {icir:>+7.3f}  {mins:>8.1f}{star}')
        if mic > best_mean_ic:
            best_mean_ic = mic
            best_window = r['window_size']

    logger.info('=' * 70)
    logger.info(f'Best window size: {best_window} bars ({best_window/10 if best_window else "?"}s) — mean IC={best_mean_ic:+.4f}')
    logger.info(f'Total sweep time: {total_elapsed/3600:.2f}h ({total_elapsed/60:.1f}m)')
    logger.info('')

    # Save combined results JSON
    summary = {
        'timestamp': timestamp,
        'total_elapsed_sec': total_elapsed,
        'best_window_size': best_window,
        'best_mean_ic': float(best_mean_ic),
        'window_sizes_tested': WINDOW_SIZES,
        'config': {
            'spatial_channels': list(SPATIAL_CHANNELS),
            'temporal_channels': TEMPORAL_CHANNELS,
            'n_folds_per_window': N_FOLDS_PER_WINDOW,
            'epochs_per_fold': EPOCHS_PER_FOLD,
            'subsample': SUBSAMPLE,
            'horizon_bars': HORIZON_BARS,
            'lr': LR,
        },
        'results': all_results,
    }
    summary_path = Path(OUTPUT_DIR) / f'window_sweep_summary_{timestamp}.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f'Summary saved: {summary_path}')

    # Print the table to stdout one more time for easy reading
    print('\n' + '=' * 70)
    print('WINDOW SWEEP RESULTS:')
    print(f'{"Window":>8}  {"Secs":>6}  {"MeanIC":>8}  {"ConcatIC":>10}  {"ICIR":>7}')
    print('-' * 70)
    for r in all_results:
        if 'error' in r:
            print(f'{r["window_size"]:>8}  {"?":>6}  {"ERROR":>8}  {"?":>10}  {"?":>7}')
        else:
            mark = ' *' if r['window_size'] == best_window else ''
            print(f'{r["window_size"]:>8}  {r["window_seconds"]:>6.0f}  {r["agg_ic"]:>+8.4f}  {r["concat_ic"]:>+10.4f}  {r["agg_icir"]:>+7.3f}{mark}')
    print('=' * 70)
    print(f'Best: {best_window} bars  Mean IC={best_mean_ic:+.4f}')


if __name__ == '__main__':
    main()
