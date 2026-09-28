#!/usr/bin/env python3
"""
Inference-only: Load existing CNN checkpoint, predict on OOT data.
No training. Minimal GPU/RAM usage.

Usage:
    python alpha_discovery/deep_models/infer_oot_only.py
    python alpha_discovery/deep_models/infer_oot_only.py --checkpoint path/to/model.pt --device cpu
"""

import argparse
import gc
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(MODELS_DIR))

from book_spatial_cnn import BookSpatialCNN

logging.basicConfig(
    format='%(asctime)s [infer] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger('infer')

OOT_BOOK_DIR = ROOT_DIR / 'data' / 'processed' / 'dl_book_cache_oot'
OUTPUT_DIR = MODELS_DIR / 'results'
DEFAULT_CHECKPOINT = OUTPUT_DIR / 'best_cnn_oot_20260311_015004.pt'


class BookInferenceDataset(torch.utils.data.Dataset):
    def __init__(self, tensors, day_boundaries, window_size=20):
        self.window_size = window_size
        self.tensors = tensors.astype(np.float32)
        self.tensors[:, :, 1] = np.log1p(self.tensors[:, :, 1])
        self.tensors[:, :, 2] = np.log1p(self.tensors[:, :, 2])
        self.tensors[:, :, 3] = np.log1p(self.tensors[:, :, 3])
        valid = []
        for day_idx in range(len(day_boundaries) - 1):
            start = day_boundaries[day_idx]
            end = day_boundaries[day_idx + 1]
            for i in range(start + window_size - 1, end):
                valid.append(i)
        self.valid_indices = np.array(valid, dtype=np.int64)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        i = int(self.valid_indices[idx])
        window = self.tensors[i - self.window_size + 1 : i + 1]
        window = torch.from_numpy(window.copy())
        return window, i


def collate_inference(batch):
    windows = torch.stack([b[0] for b in batch])
    indices = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return windows, indices


@torch.no_grad()
def predict_day(model, day_data, day_n_bars, device, window_size=20, batch_size=512):
    model.eval()
    predictions = np.zeros(day_n_bars, dtype=np.float64)
    boundaries = [0, day_n_bars]
    dataset = BookInferenceDataset(day_data['book_tensors'], boundaries, window_size=window_size)
    if len(dataset) == 0:
        return predictions
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                       num_workers=0, collate_fn=collate_inference)
    ctx = torch.amp.autocast('cuda') if device.type == 'cuda' else torch.no_grad()
    for windows, indices in loader:
        windows = windows.to(device)
        with ctx:
            preds = model(windows).squeeze(-1)
        preds = preds.cpu().float().numpy()
        indices = indices.numpy()
        predictions[indices] = preds
    return predictions


def list_book_days(cache_dir):
    """List available book tensor files without loading them."""
    cache_path = Path(cache_dir)
    files = sorted(cache_path.glob('*_book_tensors.npz'))
    dates = []
    paths = []
    for f in files:
        date = f.name.replace('_book_tensors.npz', '')
        dates.append(date)
        paths.append(f)
    return dates, paths


def load_single_day(filepath):
    """Load a single day's book tensors."""
    npz = np.load(str(filepath))
    return dict(npz)


def main():
    parser = argparse.ArgumentParser(description='CNN OOT inference only (no training)')
    parser.add_argument('--checkpoint', type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument('--oot-dir', type=str, default=str(OOT_BOOK_DIR))
    parser.add_argument('--output-dir', type=str, default=str(OUTPUT_DIR))
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--window-size', type=int, default=20)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    logger.info('=' * 70)
    logger.info('CNN OOT Inference Only (no training)')
    logger.info(f'  Checkpoint: {args.checkpoint}')
    logger.info(f'  OOT data:   {args.oot_dir}')
    logger.info(f'  Device:     {device}')
    logger.info(f'  Batch size: {args.batch_size}')
    logger.info('=' * 70)

    # Load model
    logger.info('Loading model...')
    model = BookSpatialCNN(
        window_size=args.window_size, num_levels=20, num_features=4,
        spatial_channels=(32, 64, 128, 256), temporal_channels=256,
        dropout=0.1, num_classes=1,
    ).to(device)

    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f'  Model loaded: {n_params:,} params')

    # List OOT files (don't load all into memory)
    oot_dates, oot_paths = list_book_days(args.oot_dir)
    logger.info(f'  OOT: {len(oot_dates)} days ({oot_dates[0]} to {oot_dates[-1]})')

    # Incremental save file — resume from partial runs
    incremental_file = Path(args.output_dir) / 'oot_predictions_incremental.npz'
    pred_data = {}
    done_dates = set()
    if incremental_file.exists():
        existing = np.load(str(incremental_file), allow_pickle=True)
        for k in existing.files:
            pred_data[k] = existing[k]
            if k.endswith('_preds'):
                done_dates.add(k.replace('_preds', ''))
        existing.close()
        logger.info(f'  Resuming: {len(done_dates)} days already done')

    # Predict each day ONE AT A TIME to save RAM
    logger.info('\n--- Generating OOT predictions ---')
    t0 = time.time()
    for i, (date, fpath) in enumerate(zip(oot_dates, oot_paths)):
        if date in done_dates:
            logger.info(f'  [{i+1}/{len(oot_dates)}] {date}: SKIPPED (already done)')
            continue

        # Load single day
        day_data = load_single_day(fpath)
        n_bars = len(day_data['mid_prices'])

        preds = predict_day(model, day_data, n_bars, device,
                           window_size=args.window_size, batch_size=args.batch_size)
        pred_data[f'{date}_preds'] = preds
        pred_data[f'{date}_mid'] = day_data['mid_prices']
        done_dates.add(date)

        elapsed = time.time() - t0
        logger.info(f'  [{i+1}/{len(oot_dates)}] {date}: {n_bars:,} bars, '
                   f'pred range [{preds.min():.3f}, {preds.max():.3f}] ({elapsed:.0f}s)')

        # Save after EVERY day (crash-resilient)
        np.savez_compressed(str(incremental_file), **pred_data)
        logger.info(f'    Saved ({len(done_dates)}/{len(oot_dates)} days)')

        # Free memory after each day
        del day_data
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # Final save with timestamp
    pred_file = Path(args.output_dir) / f'oos_predictions_book_oot_{timestamp}.npz'
    np.savez_compressed(str(pred_file), **pred_data)
    total_time = time.time() - t0
    logger.info(f'\nOOT predictions saved: {pred_file}')
    logger.info(f'  {len(done_dates)} days in {total_time:.0f}s')

    logger.info('\n' + '=' * 70)
    logger.info('DONE — Ready for cnn_rust_sim_validation.py')
    logger.info('=' * 70)


if __name__ == '__main__':
    main()
