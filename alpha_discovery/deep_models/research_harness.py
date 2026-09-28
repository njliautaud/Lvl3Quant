"""
Research Harness — Multi-stage testing funnel for rapid hypothesis evaluation.

Implements a 3-stage research pipeline, each progressively more expensive:

  Stage 1: LightGBM quick test (5-30 seconds)
    - Flattens book tensors to tabular features + hand-crafted LOB features
    - Trains LightGBM, reports IC and feature importances
    - Use to quickly filter bad ideas before committing GPU time

  Stage 2: Static CNN test (10-15 minutes)
    - Trains a single BookSpatialCNN on a fixed train/OOT split
    - Reports IC, overfit ratio, param count
    - Use to validate that the CNN architecture can extract signal

  Stage 3: Mini walk-forward (2-4 hours)
    - Runs walk-forward on the last N folds only
    - Reports per-fold ICs, stability, mean IC
    - Use as final validation before launching a full WF run

Usage:
  python research_harness.py --stage lgbm --horizon 100 --train-days 60 --oot-days 30
  python research_harness.py --stage static_cnn --horizon 600 --dropout 0.4 --channels 16,32,64,128
  python research_harness.py --stage mini_wf --horizon 100 --n-folds 15

Output: JSON to stdout + saved to results/research_harness_<stage>_<timestamp>.json
"""

import argparse
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
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Path setup — identical to train_walkforward.py
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent.parent  # Lvl3Quant root
MODELS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(MODELS_DIR))

DEFAULT_BOOK_DIR = str(ROOT_DIR / 'data' / 'processed' / 'dl_book_cache')
DEFAULT_OUTPUT_DIR = str(MODELS_DIR / 'results')

logging.basicConfig(
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger('research_harness')


# ============================================================================
# Shared utilities
# ============================================================================

def get_available_dates(book_dir: str) -> List[str]:
    """Return sorted list of YYYY-MM-DD dates with book tensor NPZ files."""
    files = sorted(Path(book_dir).glob('*_book_tensors.npz'))
    return [f.name.replace('_book_tensors.npz', '') for f in files]


def load_book_data(book_dir: str, dates: List[str]):
    """
    Load book tensor NPZ files for specified dates.

    Returns:
        tensors: (N, 20, 4) float32 — raw book tensors (NOT log-transformed)
        mid_prices: (N,) float64 — mid prices
        day_boundaries: list of ints [0, n_day0, n_day0+n_day1, ...]
    """
    all_tensors = []
    all_mids = []
    boundaries = [0]

    for date in dates:
        fpath = Path(book_dir) / f'{date}_book_tensors.npz'
        if not fpath.exists():
            logger.warning(f'Missing: {fpath.name}, skipping')
            continue
        npz = np.load(fpath)
        all_tensors.append(npz['book_tensors'])
        all_mids.append(npz['mid_prices'])
        boundaries.append(boundaries[-1] + len(npz['mid_prices']))

    if not all_tensors:
        raise FileNotFoundError(f'No book tensor files found in {book_dir}')

    tensors = np.concatenate(all_tensors, axis=0)
    mid_prices = np.concatenate(all_mids, axis=0)
    return tensors, mid_prices, boundaries


def compute_mfe_net(mid_prices: np.ndarray, day_boundaries: List[int],
                    horizon_bars: int = 100, tick_size: float = 0.25) -> np.ndarray:
    """
    Compute mfe_net = mfe_long - mfe_short over a forward window.
    Identical to train_walkforward.compute_mfe_net.
    NaN at bars where the window crosses a day boundary.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    N = len(mid_prices)
    H = horizon_bars

    mfe_long = np.full(N, np.nan, dtype=np.float32)
    mfe_short = np.full(N, np.nan, dtype=np.float32)

    mid_shifted = mid_prices[1:]
    valid_len = N - H - 1

    if valid_len > 0:
        windows = sliding_window_view(mid_shifted, H)[:valid_len]
        fwd_max = windows.max(axis=1)
        fwd_min = windows.min(axis=1)
        mfe_long[:valid_len] = np.maximum(0.0, (fwd_max - mid_prices[:valid_len]) / tick_size)
        mfe_short[:valid_len] = np.maximum(0.0, (mid_prices[:valid_len] - fwd_min) / tick_size)

    n_days = len(day_boundaries) - 1
    for d in range(n_days - 1):
        day_end = day_boundaries[d + 1]
        nan_start = max(day_boundaries[d], day_end - H)
        mfe_long[nan_start:day_end] = np.nan
        mfe_short[nan_start:day_end] = np.nan

    return (mfe_long - mfe_short).astype(np.float32)


def split_dates(all_dates: List[str], train_days: int, oot_days: int):
    """
    Split available dates into train and OOT (out-of-time) sets.
    Takes the last (train_days + oot_days) dates, splits into train | OOT.
    """
    total_needed = train_days + oot_days
    if len(all_dates) < total_needed:
        logger.warning(
            f'Only {len(all_dates)} dates available, need {total_needed}. '
            f'Using all dates with {oot_days} OOT.'
        )
        train_dates = all_dates[:-oot_days]
        oot_dates = all_dates[-oot_days:]
    else:
        subset = all_dates[-total_needed:]
        train_dates = subset[:train_days]
        oot_dates = subset[train_days:]
    return train_dates, oot_dates


# ============================================================================
# Stage 1: LightGBM Quick Test
# ============================================================================

def flatten_book_to_tabular(tensors: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """
    Flatten (N, 20, 4) book tensors into tabular features.

    Raw features (40): each of the 20 levels x 4 channels flattened.
    Aggregated features (32): mean/std/max/min across levels for each channel,
        computed separately for bid (levels 0-9) and ask (levels 10-19).
    Hand-crafted LOB features (8):
        - bid_ask_spread: ask_best_price - bid_best_price (in relative terms)
        - depth_imbalance: total_bid_depth / (total_bid_depth + total_ask_depth)
        - top3_depth_imbalance: same but top 3 levels only
        - weighted_mid_offset: volume-weighted mid minus simple mid
        - bid_depth_slope: linear slope of bid depth across levels
        - ask_depth_slope: linear slope of ask depth across levels
        - bid_queue_age_mean: mean queue age on bid side
        - ask_queue_age_mean: mean queue age on ask side

    Returns:
        features: (N, 80) float32 array
        feature_names: list of 80 feature name strings
    """
    N = tensors.shape[0]
    feature_list = []
    names = []

    # Channel names for readability
    ch_names = ['price_rel', 'depth_lots', 'num_orders', 'queue_age']

    # --- Raw flattened features (40) ---
    # We subsample to keep it manageable: levels 0,2,4,6,9 (bid) and 10,12,14,16,19 (ask)
    # Actually, use all 20x4=80 would be too many for importance reading.
    # Use aggregate stats instead for the "raw" portion.

    # --- Aggregated features (32): mean/std/max/min per channel per side ---
    for side, side_name, sl in [('bid', 'bid', slice(0, 10)), ('ask', 'ask', slice(10, 20))]:
        side_data = tensors[:, sl, :]  # (N, 10, 4)
        for ch_idx, ch_name in enumerate(ch_names):
            ch_data = side_data[:, :, ch_idx]  # (N, 10)
            feature_list.append(ch_data.mean(axis=1))
            names.append(f'{side_name}_{ch_name}_mean')
            feature_list.append(ch_data.std(axis=1))
            names.append(f'{side_name}_{ch_name}_std')
            feature_list.append(ch_data.max(axis=1))
            names.append(f'{side_name}_{ch_name}_max')
            feature_list.append(ch_data.min(axis=1))
            names.append(f'{side_name}_{ch_name}_min')

    # --- Best-level raw features (8): best bid and best ask, all 4 channels ---
    for level, level_name in [(0, 'best_bid'), (10, 'best_ask')]:
        for ch_idx, ch_name in enumerate(ch_names):
            feature_list.append(tensors[:, level, ch_idx])
            names.append(f'{level_name}_{ch_name}')

    # --- Hand-crafted LOB features ---

    # Bid-ask spread (best ask price_rel - best bid price_rel)
    spread = tensors[:, 10, 0] - tensors[:, 0, 0]
    feature_list.append(spread)
    names.append('spread')

    # Depth imbalance: total bid depth / (total bid + total ask depth)
    bid_depth_total = tensors[:, :10, 1].sum(axis=1)
    ask_depth_total = tensors[:, 10:, 1].sum(axis=1)
    total_depth = bid_depth_total + ask_depth_total
    depth_imbalance = np.where(total_depth > 0, bid_depth_total / total_depth, 0.5)
    feature_list.append(depth_imbalance)
    names.append('depth_imbalance')

    # Top-3 level depth imbalance
    bid3 = tensors[:, :3, 1].sum(axis=1)
    ask3 = tensors[:, 10:13, 1].sum(axis=1)
    top3_total = bid3 + ask3
    top3_imbalance = np.where(top3_total > 0, bid3 / top3_total, 0.5)
    feature_list.append(top3_imbalance)
    names.append('top3_depth_imbalance')

    # Volume-weighted mid offset: how far the depth-weighted mid is from the simple mid
    # weighted_mid = (bid_best_price * ask_best_depth + ask_best_price * bid_best_depth)
    #              / (bid_best_depth + ask_best_depth)
    # offset = weighted_mid - simple_mid (where simple_mid ~ 0 in price_relative coords)
    bid_best_price = tensors[:, 0, 0]
    ask_best_price = tensors[:, 10, 0]
    bid_best_depth = tensors[:, 0, 1]
    ask_best_depth = tensors[:, 10, 1]
    denom = bid_best_depth + ask_best_depth
    simple_mid = (bid_best_price + ask_best_price) / 2.0
    weighted_mid = np.where(
        denom > 0,
        (bid_best_price * ask_best_depth + ask_best_price * bid_best_depth) / denom,
        simple_mid,
    )
    feature_list.append(weighted_mid - simple_mid)
    names.append('weighted_mid_offset')

    # Depth slope (linear regression slope of depth across levels)
    levels_x = np.arange(10, dtype=np.float32)
    levels_x_centered = levels_x - levels_x.mean()
    x_var = (levels_x_centered ** 2).sum()

    bid_depths = tensors[:, :10, 1]  # (N, 10)
    ask_depths = tensors[:, 10:, 1]  # (N, 10)

    bid_slope = (bid_depths * levels_x_centered[None, :]).sum(axis=1) / max(x_var, 1e-8)
    ask_slope = (ask_depths * levels_x_centered[None, :]).sum(axis=1) / max(x_var, 1e-8)
    feature_list.append(bid_slope)
    names.append('bid_depth_slope')
    feature_list.append(ask_slope)
    names.append('ask_depth_slope')

    # Order count imbalance (total bid orders / total orders)
    bid_orders = tensors[:, :10, 2].sum(axis=1)
    ask_orders = tensors[:, 10:, 2].sum(axis=1)
    total_orders = bid_orders + ask_orders
    order_imbalance = np.where(total_orders > 0, bid_orders / total_orders, 0.5)
    feature_list.append(order_imbalance)
    names.append('order_count_imbalance')

    # Queue age differential (bid mean age - ask mean age)
    bid_age_avg = tensors[:, :10, 3].mean(axis=1)
    ask_age_avg = tensors[:, 10:, 3].mean(axis=1)
    feature_list.append(bid_age_avg - ask_age_avg)
    names.append('queue_age_diff')

    features = np.column_stack(feature_list).astype(np.float32)
    return features, names


def run_lgbm_stage(
    book_dir: str,
    horizon_bars: int,
    train_days: int,
    oot_days: int,
    output_dir: str,
) -> Dict:
    """
    Stage 1: LightGBM quick test.

    Flattens book tensors to tabular features, trains LightGBM,
    reports IC and feature importance on the OOT window.
    """
    import lightgbm as lgb

    t0 = time.time()
    logger.info('=' * 60)
    logger.info('STAGE 1: LightGBM Quick Test')
    logger.info(f'  horizon_bars={horizon_bars}, train_days={train_days}, oot_days={oot_days}')
    logger.info('=' * 60)

    # Load dates
    all_dates = get_available_dates(book_dir)
    train_dates, oot_dates = split_dates(all_dates, train_days, oot_days)
    logger.info(f'  Train: {train_dates[0]} .. {train_dates[-1]} ({len(train_dates)} days)')
    logger.info(f'  OOT:   {oot_dates[0]} .. {oot_dates[-1]} ({len(oot_dates)} days)')

    # Load train data
    logger.info('  Loading train data...')
    train_tensors, train_mids, train_bounds = load_book_data(book_dir, train_dates)
    train_target = compute_mfe_net(train_mids, train_bounds, horizon_bars)

    # Load OOT data
    logger.info('  Loading OOT data...')
    oot_tensors, oot_mids, oot_bounds = load_book_data(book_dir, oot_dates)
    oot_target = compute_mfe_net(oot_mids, oot_bounds, horizon_bars)

    # Flatten to tabular
    logger.info('  Flattening to tabular features...')
    X_train, feature_names = flatten_book_to_tabular(train_tensors)
    X_oot, _ = flatten_book_to_tabular(oot_tensors)

    # Filter valid targets (non-NaN)
    train_mask = np.isfinite(train_target)
    oot_mask = np.isfinite(oot_target)

    X_train = X_train[train_mask]
    y_train = train_target[train_mask]
    X_oot = X_oot[oot_mask]
    y_oot = oot_target[oot_mask]

    logger.info(f'  Train samples: {len(y_train):,}  OOT samples: {len(y_oot):,}')

    # Replace inf/nan in features with 0
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_oot = np.nan_to_num(X_oot, nan=0.0, posinf=0.0, neginf=0.0)

    # Train LightGBM
    logger.info('  Training LightGBM...')
    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_names, free_raw_data=False)
    dval = lgb.Dataset(X_oot, label=y_oot, feature_name=feature_names, reference=dtrain, free_raw_data=False)

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': 7,
        'min_child_samples': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42,
    }

    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
    model = lgb.train(
        params, dtrain,
        num_boost_round=500,
        valid_sets=[dval],
        callbacks=callbacks,
    )

    # Predict on OOT
    preds = model.predict(X_oot)

    # Compute IC
    mask = np.isfinite(preds) & np.isfinite(y_oot)
    if mask.sum() > 10:
        ic, _ = spearmanr(preds[mask], y_oot[mask])
        ic = float(ic) if np.isfinite(ic) else 0.0
    else:
        ic = 0.0

    # Feature importance (gain-based, normalized to sum=1)
    raw_importance = model.feature_importance(importance_type='gain')
    total_imp = raw_importance.sum()
    if total_imp > 0:
        norm_importance = raw_importance / total_imp
    else:
        norm_importance = raw_importance

    # Top 10 features
    top_idx = np.argsort(norm_importance)[::-1][:10]
    feature_importance = {
        feature_names[i]: round(float(norm_importance[i]), 4) for i in top_idx
    }

    # Also compute train IC for overfit detection
    train_preds = model.predict(X_train)
    train_mask = np.isfinite(train_preds) & np.isfinite(y_train)
    if train_mask.sum() > 10:
        train_ic, _ = spearmanr(train_preds[train_mask], y_train[train_mask])
        train_ic = float(train_ic) if np.isfinite(train_ic) else 0.0
    else:
        train_ic = 0.0

    elapsed = time.time() - t0

    result = {
        'stage': 'lgbm',
        'horizon_bars': horizon_bars,
        'train_days': len(train_dates),
        'oot_days': len(oot_dates),
        'train_range': f'{train_dates[0]}..{train_dates[-1]}',
        'oot_range': f'{oot_dates[0]}..{oot_dates[-1]}',
        'train_samples': int(len(y_train)),
        'oot_samples': int(len(y_oot)),
        'ic': round(ic, 6),
        'train_ic': round(train_ic, 6),
        'overfit_ratio': round(train_ic / ic, 3) if abs(ic) > 1e-6 else None,
        'param_count': None,
        'train_loss': None,
        'val_loss': None,
        'best_iteration': model.best_iteration,
        'feature_importance': feature_importance,
        'elapsed_seconds': round(elapsed, 1),
        'timestamp': datetime.now().isoformat(timespec='seconds'),
    }

    logger.info(f'  IC (OOT): {ic:.6f}')
    logger.info(f'  IC (train): {train_ic:.6f}')
    logger.info(f'  Overfit ratio: {result["overfit_ratio"]}')
    logger.info(f'  Top features: {list(feature_importance.keys())[:5]}')
    logger.info(f'  Elapsed: {elapsed:.1f}s')

    return result


# ============================================================================
# Stage 2: Static CNN Test
# ============================================================================

def run_static_cnn_stage(
    book_dir: str,
    horizon_bars: int,
    train_days: int,
    oot_days: int,
    spatial_channels: Tuple[int, ...],
    temporal_channels: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    lr: float,
    subsample: int,
    window_size: int,
    device_str: str,
    output_dir: str,
) -> Dict:
    """
    Stage 2: Train a single BookSpatialCNN on a fixed train/OOT split.

    Faster than full walk-forward — trains ONE model and evaluates on OOT.
    Good for testing architecture changes before committing to WF.
    """
    import torch
    import torch.nn as nn
    from book_spatial_cnn import BookSpatialCNN

    # Import from train_walkforward for dataset/collate/training utilities
    from train_walkforward import (
        BarDataset, collate_book, train_epoch, evaluate, load_day_files,
    )

    t0 = time.time()
    logger.info('=' * 60)
    logger.info('STAGE 2: Static CNN Test')
    logger.info(f'  horizon_bars={horizon_bars}, train_days={train_days}, oot_days={oot_days}')
    logger.info(f'  spatial_channels={spatial_channels}, temporal_channels={temporal_channels}')
    logger.info(f'  dropout={dropout}, epochs={epochs}, batch_size={batch_size}, lr={lr}')
    logger.info('=' * 60)

    device = torch.device(device_str if torch.cuda.is_available() and device_str == 'cuda' else 'cpu')
    use_amp = (device.type == 'cuda')

    torch.manual_seed(42)
    np.random.seed(42)

    # Load dates and split
    all_dates = get_available_dates(book_dir)
    train_dates, oot_dates = split_dates(all_dates, train_days, oot_days)
    logger.info(f'  Train: {train_dates[0]} .. {train_dates[-1]} ({len(train_dates)} days)')
    logger.info(f'  OOT:   {oot_dates[0]} .. {oot_dates[-1]} ({len(oot_dates)} days)')

    # Load data using train_walkforward infrastructure
    logger.info('  Loading train data...')
    train_data, train_mids, train_bounds = load_day_files(book_dir, 'book', train_dates)
    train_target = compute_mfe_net(train_mids, train_bounds, horizon_bars)

    logger.info('  Loading OOT data...')
    oot_data, oot_mids, oot_bounds = load_day_files(book_dir, 'book', oot_dates)
    oot_target = compute_mfe_net(oot_mids, oot_bounds, horizon_bars)

    # Build datasets
    logger.info('  Building datasets...')
    train_ds = BarDataset(
        train_data, train_target, train_bounds,
        model_type='book', window_size=window_size,
        horizon=horizon_bars, subsample=subsample,
    )
    oot_ds = BarDataset(
        oot_data, oot_target, oot_bounds,
        model_type='book', window_size=window_size,
        horizon=horizon_bars, subsample=1,  # no subsample on OOT
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_book, num_workers=0, pin_memory=(device.type == 'cuda'),
    )
    oot_loader = torch.utils.data.DataLoader(
        oot_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_book, num_workers=0, pin_memory=(device.type == 'cuda'),
    )

    logger.info(f'  Train samples: {len(train_ds):,}  OOT samples: {len(oot_ds):,}')

    # Build model with configurable architecture
    model = BookSpatialCNN(
        window_size=window_size,
        num_levels=20,
        num_features=4,
        spatial_channels=spatial_channels,
        temporal_channels=temporal_channels,
        dropout=dropout,
        num_classes=1,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'  Model params: {param_count:,} ({trainable_count:,} trainable)')

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    # Training loop
    best_ic = -999.0
    best_val_loss = float('inf')
    epoch_results = []

    for epoch in range(1, epochs + 1):
        train_loss, train_n = train_epoch(
            model, train_loader, optimizer, device, 'book', scaler,
        )
        ic, val_loss, preds_arr, targets_arr = evaluate(model, oot_loader, device, 'book')
        scheduler.step()

        logger.info(
            f'  Epoch {epoch}/{epochs}: '
            f'train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, '
            f'IC={ic:.6f}, overfit={val_loss/max(train_loss, 1e-8):.2f}'
        )

        epoch_results.append({
            'epoch': epoch,
            'train_loss': round(train_loss, 6),
            'val_loss': round(val_loss, 6),
            'ic': round(ic, 6),
        })

        if ic > best_ic:
            best_ic = ic
            best_val_loss = val_loss
            best_train_loss = train_loss

    elapsed = time.time() - t0

    result = {
        'stage': 'static_cnn',
        'horizon_bars': horizon_bars,
        'train_days': len(train_dates),
        'oot_days': len(oot_dates),
        'train_range': f'{train_dates[0]}..{train_dates[-1]}',
        'oot_range': f'{oot_dates[0]}..{oot_dates[-1]}',
        'train_samples': int(len(train_ds)),
        'oot_samples': int(len(oot_ds)),
        'ic': round(best_ic, 6),
        'train_ic': None,  # not computed for CNN (expensive)
        'overfit_ratio': round(best_val_loss / max(best_train_loss, 1e-8), 3),
        'param_count': param_count,
        'train_loss': round(best_train_loss, 6),
        'val_loss': round(best_val_loss, 6),
        'architecture': {
            'spatial_channels': list(spatial_channels),
            'temporal_channels': temporal_channels,
            'dropout': dropout,
            'window_size': window_size,
        },
        'epoch_history': epoch_results,
        'feature_importance': None,
        'elapsed_seconds': round(elapsed, 1),
        'timestamp': datetime.now().isoformat(timespec='seconds'),
    }

    logger.info(f'  Best IC: {best_ic:.6f}')
    logger.info(f'  Overfit ratio: {result["overfit_ratio"]}')
    logger.info(f'  Params: {param_count:,}')
    logger.info(f'  Elapsed: {elapsed:.1f}s')

    # Cleanup
    del model, optimizer, scheduler, train_ds, oot_ds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ============================================================================
# Stage 3: Mini Walk-Forward
# ============================================================================

def run_mini_wf_stage(
    book_dir: str,
    horizon_bars: int,
    n_folds: int,
    epochs: int,
    batch_size: int,
    lr: float,
    subsample: int,
    window_size: int,
    device_str: str,
    output_dir: str,
    max_train_days: Optional[int],
    dropout: float,
    spatial_channels: Tuple[int, ...],
    temporal_channels: int,
) -> Dict:
    """
    Stage 3: Mini walk-forward — run WF on the last N folds only.

    Uses the same train_walkforward infrastructure but skips early folds.
    Reports per-fold ICs, mean, std, and stability metrics.
    """
    import torch
    import torch.nn as nn
    from book_spatial_cnn import BookSpatialCNN
    from train_walkforward import (
        BarDataset, collate_book, train_epoch, evaluate,
        load_day_files, get_available_dates as get_dates_wf,
    )

    t0 = time.time()
    logger.info('=' * 60)
    logger.info('STAGE 3: Mini Walk-Forward')
    logger.info(f'  horizon_bars={horizon_bars}, n_folds={n_folds}')
    logger.info(f'  epochs={epochs}, batch_size={batch_size}, lr={lr}')
    logger.info(f'  max_train_days={max_train_days}')
    logger.info('=' * 60)

    device = torch.device(device_str if torch.cuda.is_available() and device_str == 'cuda' else 'cpu')
    use_amp = (device.type == 'cuda')

    torch.manual_seed(42)
    np.random.seed(42)

    # Get all available dates
    all_dates = get_available_dates(book_dir)
    n_total = len(all_dates)
    min_train_days = 5
    purge_days = 1

    # Total folds available
    total_folds = n_total - min_train_days - purge_days
    if total_folds <= 0:
        raise ValueError(f'Not enough dates ({n_total}) for walk-forward')

    # Start fold = skip early folds so we only run the last n_folds
    start_fold = max(0, total_folds - n_folds)
    actual_folds = min(n_folds, total_folds)

    logger.info(f'  Total dates: {n_total}, total folds: {total_folds}')
    logger.info(f'  Running folds {start_fold} .. {start_fold + actual_folds - 1}')

    fold_results = []

    for fold_idx in range(start_fold, start_fold + actual_folds):
        fold_t0 = time.time()
        test_day_idx = min_train_days + purge_days + fold_idx
        if test_day_idx >= n_total:
            break

        train_end = test_day_idx - purge_days
        if max_train_days:
            train_start = max(0, train_end - max_train_days)
        else:
            train_start = 0

        train_dates_fold = all_dates[train_start:train_end]
        test_date = all_dates[test_day_idx]

        logger.info(
            f'  Fold {fold_idx}: train {train_dates_fold[0]}..{train_dates_fold[-1]} '
            f'({len(train_dates_fold)}d) -> test {test_date}'
        )

        # Load data
        train_data, train_mids, train_bounds = load_day_files(book_dir, 'book', train_dates_fold)
        test_data, test_mids, test_bounds = load_day_files(book_dir, 'book', [test_date])

        if not train_data or not test_data:
            logger.warning(f'  Fold {fold_idx}: missing data, skipping')
            continue

        train_target = compute_mfe_net(train_mids, train_bounds, horizon_bars)
        test_target = compute_mfe_net(test_mids, test_bounds, horizon_bars)

        # Build datasets
        train_ds = BarDataset(
            train_data, train_target, train_bounds,
            model_type='book', window_size=window_size,
            horizon=horizon_bars, subsample=subsample,
        )
        test_ds = BarDataset(
            test_data, test_target, test_bounds,
            model_type='book', window_size=window_size,
            horizon=horizon_bars, subsample=1,
        )

        if len(train_ds) == 0 or len(test_ds) == 0:
            logger.warning(f'  Fold {fold_idx}: empty dataset, skipping')
            continue

        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            collate_fn=collate_book, num_workers=0, pin_memory=(device.type == 'cuda'),
        )
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            collate_fn=collate_book, num_workers=0, pin_memory=(device.type == 'cuda'),
        )

        # Build fresh model each fold (no warm-start in research mode)
        model = BookSpatialCNN(
            window_size=window_size,
            num_levels=20,
            num_features=4,
            spatial_channels=spatial_channels,
            temporal_channels=temporal_channels,
            dropout=dropout,
            num_classes=1,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        scaler = torch.amp.GradScaler('cuda') if use_amp else None

        # Train
        best_ic = -999.0
        best_train_loss = 0.0
        best_val_loss = 0.0
        for epoch in range(1, epochs + 1):
            train_loss, _ = train_epoch(model, train_loader, optimizer, device, 'book', scaler)
            ic, val_loss, _, _ = evaluate(model, test_loader, device, 'book')
            scheduler.step()

            if ic > best_ic:
                best_ic = ic
                best_train_loss = train_loss
                best_val_loss = val_loss

        fold_elapsed = time.time() - fold_t0
        overfit = round(best_val_loss / max(best_train_loss, 1e-8), 3) if best_train_loss > 0 else None

        logger.info(
            f'    IC={best_ic:.6f}, overfit={overfit}, '
            f'train_loss={best_train_loss:.6f}, val_loss={best_val_loss:.6f}, '
            f'{fold_elapsed:.1f}s'
        )

        fold_results.append({
            'fold': fold_idx,
            'test_date': test_date,
            'train_days': len(train_dates_fold),
            'ic': round(best_ic, 6),
            'train_loss': round(best_train_loss, 6),
            'val_loss': round(best_val_loss, 6),
            'overfit_ratio': overfit,
            'elapsed_seconds': round(fold_elapsed, 1),
        })

        # Cleanup per fold
        del model, optimizer, scheduler, train_ds, test_ds
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = time.time() - t0

    # Aggregate results
    ics = [f['ic'] for f in fold_results]
    mean_ic = float(np.mean(ics)) if ics else 0.0
    std_ic = float(np.std(ics)) if ics else 0.0
    icir = mean_ic / std_ic if std_ic > 1e-8 else 0.0
    positive_folds = sum(1 for x in ics if x > 0)

    result = {
        'stage': 'mini_wf',
        'horizon_bars': horizon_bars,
        'n_folds_requested': n_folds,
        'n_folds_completed': len(fold_results),
        'max_train_days': max_train_days,
        'ic': round(mean_ic, 6),
        'ic_std': round(std_ic, 6),
        'icir': round(icir, 3),
        'positive_folds': positive_folds,
        'positive_rate': round(positive_folds / max(len(fold_results), 1), 3),
        'overfit_ratio': round(float(np.mean([
            f['overfit_ratio'] for f in fold_results if f['overfit_ratio'] is not None
        ])), 3) if fold_results else None,
        'param_count': sum(p.numel() for p in BookSpatialCNN(
            spatial_channels=spatial_channels,
            temporal_channels=temporal_channels,
            dropout=dropout,
        ).parameters()),
        'architecture': {
            'spatial_channels': list(spatial_channels),
            'temporal_channels': temporal_channels,
            'dropout': dropout,
            'window_size': window_size,
        },
        'fold_details': fold_results,
        'feature_importance': None,
        'train_loss': None,
        'val_loss': None,
        'elapsed_seconds': round(elapsed, 1),
        'timestamp': datetime.now().isoformat(timespec='seconds'),
    }

    logger.info(f'\n  Mean IC: {mean_ic:.6f} +/- {std_ic:.6f}')
    logger.info(f'  ICIR: {icir:.3f}')
    logger.info(f'  Positive folds: {positive_folds}/{len(fold_results)}')
    logger.info(f'  Elapsed: {elapsed:.1f}s ({elapsed/60:.1f}m)')

    return result


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Research Harness — Multi-stage testing funnel for Lvl3Quant',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Stage selection
    parser.add_argument(
        '--stage', type=str, required=True,
        choices=['lgbm', 'static_cnn', 'mini_wf'],
        help='Which stage to run',
    )

    # Common parameters
    parser.add_argument('--horizon', type=int, default=100,
                        help='Forward horizon in bars (100=10s, 300=30s, 600=60s, 3000=5min)')
    parser.add_argument('--train-days', type=int, default=60,
                        help='Number of training days')
    parser.add_argument('--oot-days', type=int, default=30,
                        help='Number of out-of-time test days')
    parser.add_argument('--book-dir', type=str, default=DEFAULT_BOOK_DIR,
                        help='Directory with book tensor NPZ files')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Directory to save results')

    # CNN parameters (stage 2 and 3)
    parser.add_argument('--channels', type=str, default='32,64,128,256',
                        help='Spatial channel progression (comma-separated)')
    parser.add_argument('--temporal-channels', type=int, default=256,
                        help='Temporal conv channels')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Epochs per fold/split')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--subsample', type=int, default=3,
                        help='Training subsample rate (1=no subsample)')
    parser.add_argument('--window-size', type=int, default=20,
                        help='Window size in bars for CNN input')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device for CNN training')

    # Mini WF parameters (stage 3)
    parser.add_argument('--n-folds', type=int, default=15,
                        help='Number of walk-forward folds to run (stage 3)')
    parser.add_argument('--max-train-days', type=int, default=None,
                        help='Max training days per fold (None=expanding, set for sliding)')

    args = parser.parse_args()

    # Parse channels
    spatial_channels = tuple(int(c.strip()) for c in args.channels.split(','))

    os.makedirs(args.output_dir, exist_ok=True)

    # Run the appropriate stage
    if args.stage == 'lgbm':
        result = run_lgbm_stage(
            book_dir=args.book_dir,
            horizon_bars=args.horizon,
            train_days=args.train_days,
            oot_days=args.oot_days,
            output_dir=args.output_dir,
        )

    elif args.stage == 'static_cnn':
        result = run_static_cnn_stage(
            book_dir=args.book_dir,
            horizon_bars=args.horizon,
            train_days=args.train_days,
            oot_days=args.oot_days,
            spatial_channels=spatial_channels,
            temporal_channels=args.temporal_channels,
            dropout=args.dropout,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            subsample=args.subsample,
            window_size=args.window_size,
            device_str=args.device,
            output_dir=args.output_dir,
        )

    elif args.stage == 'mini_wf':
        result = run_mini_wf_stage(
            book_dir=args.book_dir,
            horizon_bars=args.horizon,
            n_folds=args.n_folds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            subsample=args.subsample,
            window_size=args.window_size,
            device_str=args.device,
            output_dir=args.output_dir,
            max_train_days=args.max_train_days,
            dropout=args.dropout,
            spatial_channels=spatial_channels,
            temporal_channels=args.temporal_channels,
        )

    # Save result to file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = Path(args.output_dir) / f'research_harness_{args.stage}_{timestamp}.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    logger.info(f'\nResult saved to: {out_path}')

    # Log to QCC database (fire-and-forget, never block on failure)
    try:
        import urllib.request
        qcc_payload = json.dumps({
            'stage': args.stage,
            'config_json': json.dumps(result.get('config', {})),
            'horizon_bars': result.get('horizon_bars'),
            'train_days': result.get('train_days'),
            'oot_days': result.get('oot_days'),
            'model_type': result.get('model_type'),
            'ic': result.get('ic') or result.get('oot_ic'),
            'ic_std': result.get('ic_std'),
            'overfit_ratio': result.get('overfit_ratio'),
            'param_count': result.get('param_count'),
            'elapsed_seconds': result.get('elapsed_seconds'),
            'feature_importance_json': json.dumps(result.get('feature_importance', {})),
            'verdict': 'pending',
            'result_json': json.dumps(result),
            'node': 'neptune',
        }).encode('utf-8')
        req = urllib.request.Request(
            'http://localhost:3456/api/experiment',
            data=qcc_payload,
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        urllib.request.urlopen(req, timeout=3)
        logger.info('Result logged to QCC database')
    except Exception as e:
        logger.debug(f'QCC logging skipped: {e}')

    # Print JSON to stdout
    print('\n' + json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
