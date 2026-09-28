#!/usr/bin/env python3
"""
Deep Learning Benchmark — Temporal and Spatial models for MBO alpha.

Tests architectures that capture sequential and spatial structure in order book data:
  1. TemporalCNN   — 1D causal convolutions on feature sequences
  2. LSTM          — LSTM with temporal attention
  3. Transformer   — Self-attention on feature sequences
  4. SpatialCNN    — 2D spatial (book levels) + temporal CNN on RAW snapshots

All models use rolling windows of order book data (not just flat features).
Walk-forward evaluation with same methodology as arch_benchmark.py.

Usage:
  python deep_benchmark.py --n-days 20 --arch all --window 20
  python deep_benchmark.py --n-days 70 --arch transformer --window 20
  python deep_benchmark.py --arch spatial_cnn --n-days 20  # Uses raw snapshots
"""

import argparse
import gc
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('deep_benchmark')


# ============================================================
# Lazy Window Dataset — creates rolling windows on-the-fly
# ============================================================
class LazyWindowDataset(torch.utils.data.Dataset):
    """Creates rolling windows from flat feature array without pre-allocating all windows."""

    def __init__(self, features, targets, window_size, feature_indices=None, subsample=1):
        """
        Args:
            features: (N, F) float32 array
            targets: (N,) float32 array
            window_size: number of bars per window
            feature_indices: which features to use (for memory efficiency)
            subsample: use every Nth sample (1 = all, 5 = every 5th)
        """
        self.features = features
        self.targets = targets
        self.window_size = window_size
        self.feature_indices = feature_indices
        self.subsample = subsample

        # Valid indices: target must be finite and window must be available
        valid = np.isfinite(targets) & (np.arange(len(targets)) >= window_size - 1)
        self.valid_indices = np.where(valid)[0]

        if subsample > 1:
            self.valid_indices = self.valid_indices[::subsample]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        i = self.valid_indices[idx]
        start = i - self.window_size + 1

        if self.feature_indices is not None:
            window = self.features[start:i+1, :][:, self.feature_indices]
        else:
            window = self.features[start:i+1]

        # Clean NaN/inf
        window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)

        return (
            torch.tensor(window, dtype=torch.float32),
            torch.tensor(self.targets[i], dtype=torch.float32),
        )


class SpatialWindowDataset(torch.utils.data.Dataset):
    """Creates rolling windows from raw snapshot data (node_features + global_features)."""

    def __init__(self, node_features, global_features, targets, window_size, subsample=1):
        """
        Args:
            node_features: (N, 20, 9) float32
            global_features: (N, 45) float32
            targets: (N,) float32
            window_size: bars per window
            subsample: use every Nth sample
        """
        self.node_features = node_features
        self.global_features = global_features
        self.targets = targets
        self.window_size = window_size

        valid = np.isfinite(targets) & (np.arange(len(targets)) >= window_size - 1)
        self.valid_indices = np.where(valid)[0]
        if subsample > 1:
            self.valid_indices = self.valid_indices[::subsample]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        i = self.valid_indices[idx]
        start = i - self.window_size + 1

        nodes = self.node_features[start:i+1]  # (W, 20, 9)
        globs = self.global_features[start:i+1]  # (W, 45)

        nodes = np.nan_to_num(nodes, nan=0.0, posinf=0.0, neginf=0.0)
        globs = np.nan_to_num(globs, nan=0.0, posinf=0.0, neginf=0.0)

        return (
            torch.tensor(nodes, dtype=torch.float32),
            torch.tensor(globs, dtype=torch.float32),
            torch.tensor(self.targets[i], dtype=torch.float32),
        )


# ============================================================
# Training Loop (shared)
# ============================================================
def train_epoch(model, loader, optimizer, device, is_spatial=False):
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in loader:
        if is_spatial:
            nodes, globs, targets = batch
            nodes = nodes.to(device)
            globs = globs.to(device)
            targets = targets.to(device).unsqueeze(1)
            preds = model(nodes, globs)
        else:
            features, targets = batch
            features = features.to(device)
            targets = targets.to(device).unsqueeze(1)
            preds = model(features)

        loss = torch.nn.functional.mse_loss(preds, targets)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device, is_spatial=False):
    model.eval()
    all_preds = []
    all_targets = []

    for batch in loader:
        if is_spatial:
            nodes, globs, targets = batch
            nodes = nodes.to(device)
            globs = globs.to(device)
            preds = model(nodes, globs).cpu().numpy().flatten()
        else:
            features, targets = batch
            features = features.to(device)
            preds = model(features).cpu().numpy().flatten()
            targets = targets

        all_preds.append(preds)
        all_targets.append(targets.numpy() if isinstance(targets, torch.Tensor) else targets)

    return np.concatenate(all_preds), np.concatenate(all_targets)


# ============================================================
# Feature Selection (use LightGBM importance)
# ============================================================
def select_top_features(features, targets, top_k=80):
    """Quick LGB to select top-K features by importance."""
    import lightgbm as lgb

    n_sample = min(200000, len(features))
    valid = np.isfinite(targets[:n_sample])
    X = features[:n_sample][valid]
    y = targets[:n_sample][valid]

    # Clean for LGB
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    model = lgb.LGBMRegressor(
        n_estimators=50, max_depth=4, learning_rate=0.1,
        verbose=-1, n_jobs=-1,
    )
    model.fit(X, y)
    importances = model.feature_importances_
    top_indices = np.argsort(importances)[-top_k:]

    logger.info(f"Selected top {top_k} features by LGB importance")
    del model
    gc.collect()
    return top_indices


# ============================================================
# Walk-Forward Deep Learning Benchmark
# ============================================================
def walk_forward_deep(
    arch_name: str,
    features: np.ndarray,
    target: np.ndarray,
    day_boundaries: List[int],
    device: str = 'cuda',
    window_size: int = 20,
    top_k_features: int = 80,
    min_train_days: int = 5,
    max_epochs: int = 15,
    batch_size: int = 4096,
    subsample_train: int = 3,
    node_features: Optional[np.ndarray] = None,
    global_features: Optional[np.ndarray] = None,
) -> dict:
    """Walk-forward evaluation for deep learning models."""
    from temporal_models import create_model

    n_days = len(day_boundaries) - 1
    if n_days < min_train_days + 1:
        return {'error': f'Need {min_train_days+1} days, have {n_days}'}

    is_spatial = (arch_name == 'spatial_cnn')
    torch_device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Select features for non-spatial models
    feature_indices = None
    n_feat = features.shape[1] if features is not None else 0
    if not is_spatial and n_feat > top_k_features:
        feature_indices = select_top_features(features, target, top_k_features)
        n_feat = top_k_features

    all_preds = []
    all_actuals = []
    fold_ics = []
    total_train_time = 0.0
    n_folds = 0

    for test_day in range(min_train_days, n_days):
        train_end_day = test_day - 1
        train_start = day_boundaries[0]
        train_end = day_boundaries[train_end_day + 1]
        test_start = day_boundaries[test_day]
        test_end = day_boundaries[test_day + 1]

        t0 = time.time()

        try:
            if is_spatial:
                # Create spatial datasets
                train_ds = SpatialWindowDataset(
                    node_features[train_start:train_end],
                    global_features[train_start:train_end],
                    target[train_start:train_end],
                    window_size, subsample=subsample_train,
                )
                test_ds = SpatialWindowDataset(
                    node_features[test_start:test_end],
                    global_features[test_start:test_end],
                    target[test_start:test_end],
                    window_size, subsample=1,
                )
            else:
                # Create flat feature datasets
                train_ds = LazyWindowDataset(
                    features[train_start:train_end],
                    target[train_start:train_end],
                    window_size, feature_indices, subsample=subsample_train,
                )
                test_ds = LazyWindowDataset(
                    features[test_start:test_end],
                    target[test_start:test_end],
                    window_size, feature_indices, subsample=1,
                )

            if len(train_ds) < 500 or len(test_ds) < 100:
                continue

            train_loader = torch.utils.data.DataLoader(
                train_ds, batch_size=batch_size, shuffle=True,
                num_workers=0, pin_memory=True,
            )
            test_loader = torch.utils.data.DataLoader(
                test_ds, batch_size=batch_size * 2, shuffle=False,
                num_workers=0, pin_memory=True,
            )

            # Create fresh model each fold
            if is_spatial:
                model = create_model('spatial_cnn', n_outputs=1).to(torch_device)
            else:
                model = create_model(
                    arch_name, n_features=n_feat,
                    n_outputs=1, model_size='medium',
                ).to(torch_device)

            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=2,
            )

            # Quick training
            best_loss = float('inf')
            patience_counter = 0
            for epoch in range(max_epochs):
                loss = train_epoch(model, train_loader, optimizer, torch_device, is_spatial)
                scheduler.step(loss)

                if loss < best_loss:
                    best_loss = loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= 3:
                        break

            # Evaluate
            preds, actuals = evaluate(model, test_loader, torch_device, is_spatial)

            valid = np.isfinite(preds) & np.isfinite(actuals)
            if valid.sum() < 100:
                continue

            preds = preds[valid]
            actuals = actuals[valid]

            all_preds.append(preds)
            all_actuals.append(actuals)

            fold_ic = float(spearmanr(preds, actuals)[0])
            if np.isfinite(fold_ic):
                fold_ics.append(fold_ic)

            train_time = time.time() - t0
            total_train_time += train_time
            n_folds += 1

            if n_folds <= 3 or n_folds % 5 == 0:
                logger.info(
                    f"  Fold {n_folds} (day {test_day}): IC={fold_ic:.4f}, "
                    f"train={train_time:.0f}s, loss={best_loss:.6f}"
                )

            # Cleanup
            del model, optimizer, scheduler, train_loader, test_loader
            torch.cuda.empty_cache()

        except Exception as e:
            logger.warning(f"  Fold day {test_day} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not all_preds:
        return {'error': 'No valid predictions'}

    predictions = np.concatenate(all_preds)
    actuals = np.concatenate(all_actuals)

    valid = np.isfinite(predictions) & np.isfinite(actuals)
    p, a = predictions[valid], actuals[valid]

    if len(p) < 50:
        return {'error': f'Too few predictions: {len(p)}'}

    ic = float(spearmanr(p, a)[0])
    hr = float((np.sign(p) == np.sign(a)).mean())
    winners = np.abs(a[np.sign(p) == np.sign(a)]).sum()
    losers = np.abs(a[np.sign(p) != np.sign(a)]).sum()
    pf = float(winners / losers) if losers > 0 else 0.0

    if len(fold_ics) > 2:
        ic_mean = float(np.mean(fold_ics))
        ic_std = float(np.std(fold_ics))
        icir = ic_mean / ic_std if ic_std > 0 else 0.0
        tstat = ic_mean / ic_std * np.sqrt(len(fold_ics)) if ic_std > 0 else 0.0
    else:
        icir, tstat = 0.0, 0.0

    return {
        'ic': ic, 'icir': icir, 'tstat': tstat,
        'hit_rate': hr, 'profit_factor': pf,
        'n_predictions': len(p), 'n_folds': n_folds,
        'total_train_time': total_train_time,
        'avg_fold_time': total_train_time / n_folds if n_folds else 0,
        'fold_ics': fold_ics,
    }


# ============================================================
# Data Loaders
# ============================================================
def load_flat_data(feature_cache, snapshot_cache, n_days, horizon):
    """Load pre-computed flat features (same as arch_benchmark)."""
    sys.path.insert(0, str(Path(__file__).parent))
    from mbo_alpha_scan import MBOAlphaScanner
    from run_mfe_scan import compute_mfe_targets

    scanner = MBOAlphaScanner(sample_interval_ms=100)
    scanner.load_precomputed_features(
        feature_cache_dir=feature_cache,
        snapshot_cache_dir=snapshot_cache,
        n_days=n_days,
    )

    hz_sec = int(horizon.replace('ret_', '').replace('s', ''))
    mfe = compute_mfe_targets(
        scanner.mid_prices, scanner.day_boundaries,
        sample_interval_ms=100, horizons_sec={f'{hz_sec}s': hz_sec},
    )

    return {
        'features': scanner.features,
        'target': mfe[f'mfe_net_{hz_sec}s'],
        'day_boundaries': scanner.day_boundaries,
    }


def load_raw_snapshots(snapshot_cache, n_days, horizon):
    """Load raw snapshot data for SpatialTemporalCNN."""
    sys.path.insert(0, str(Path(__file__).parent))
    from run_mfe_scan import compute_mfe_targets

    snap_dir = Path(snapshot_cache)
    files = sorted(snap_dir.glob('*_snapshots.npz'))[:n_days]

    all_nodes = []
    all_globals = []
    all_mids = []
    day_boundaries = [0]
    offset = 0

    for i, f in enumerate(files):
        data = np.load(str(f), allow_pickle=True)
        nodes = data['node_features']    # (N, 20, 9)
        globs = data['global_features']  # (N, 45)
        mids = data['mid_prices']        # (N,)
        data.close()

        n_rows = len(mids)
        all_nodes.append(nodes)
        all_globals.append(globs)
        all_mids.append(mids)
        offset += n_rows
        day_boundaries.append(offset)

        if i < 3 or i % 10 == 0:
            logger.info(f"  [{i+1}/{len(files)}] {f.name[:10]}: {n_rows:,} bars, "
                       f"nodes={nodes.shape}, globals={globs.shape}")

    node_features = np.concatenate(all_nodes)    # (N_total, 20, 9)
    global_features = np.concatenate(all_globals) # (N_total, 45)
    mid_prices = np.concatenate(all_mids)         # (N_total,)

    hz_sec = int(horizon.replace('ret_', '').replace('s', ''))
    mfe = compute_mfe_targets(
        mid_prices, day_boundaries,
        sample_interval_ms=100, horizons_sec={f'{hz_sec}s': hz_sec},
    )

    logger.info(f"Loaded {len(files)} days, {len(mid_prices):,} bars, "
               f"nodes={node_features.shape}, globals={global_features.shape}")

    return {
        'node_features': node_features,
        'global_features': global_features,
        'target': mfe[f'mfe_net_{hz_sec}s'],
        'day_boundaries': day_boundaries,
        'features': None,  # Not used for spatial
    }


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Deep Learning Benchmark')
    parser.add_argument('--arch', type=str, default='all',
                       help='cnn,lstm,transformer,spatial_cnn or "all"')
    parser.add_argument('--n-days', type=int, default=20)
    parser.add_argument('--horizon', type=str, default='ret_10s')
    parser.add_argument('--window', type=int, default=20,
                       help='Rolling window size in bars (1 bar = 100ms)')
    parser.add_argument('--top-k', type=int, default=80,
                       help='Top K features for flat models')
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--max-epochs', type=int, default=15)
    parser.add_argument('--subsample-train', type=int, default=3,
                       help='Use every Nth training sample (saves memory)')
    parser.add_argument('--feature-cache', type=str, default=None)
    parser.add_argument('--snapshot-cache', type=str, default=None)
    parser.add_argument('--min-train-days', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = results_dir / f'deep_benchmark_{timestamp}.log'

    fh = logging.FileHandler(str(log_file))
    fh.setFormatter(logging.Formatter('%(asctime)s [%(name)s] %(message)s', '%H:%M:%S'))
    logger.addHandler(fh)

    base = Path(__file__).parent.parent
    if args.feature_cache is None:
        args.feature_cache = str(base / 'data' / 'processed' / 'mbo_features_cache')
    if args.snapshot_cache is None:
        args.snapshot_cache = str(base / 'data' / 'processed' / 'medium_snapshots_cache')

    if args.arch == 'all':
        arch_list = ['cnn', 'lstm', 'transformer', 'spatial_cnn']
    else:
        arch_list = [a.strip() for a in args.arch.split(',')]

    # Determine what data to load
    need_flat = any(a in arch_list for a in ['cnn', 'lstm', 'transformer'])
    need_spatial = 'spatial_cnn' in arch_list

    flat_data = None
    spatial_data = None

    if need_flat:
        logger.info("Loading pre-computed features for flat models...")
        flat_data = load_flat_data(args.feature_cache, args.snapshot_cache,
                                   args.n_days, args.horizon)

    if need_spatial:
        logger.info("Loading raw snapshots for spatial model...")
        spatial_data = load_raw_snapshots(args.snapshot_cache, args.n_days, args.horizon)

    logger.info("")
    logger.info("=" * 80)
    logger.info("DEEP LEARNING BENCHMARK")
    logger.info(f"Architectures: {arch_list}")
    logger.info(f"Days: {args.n_days}, Horizon: {args.horizon}, Window: {args.window}")
    logger.info(f"Top-K features: {args.top_k}, Subsample: {args.subsample_train}")
    logger.info("=" * 80)

    results = {}

    for arch_name in arch_list:
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"TESTING: {arch_name.upper()}")
        logger.info("=" * 60)

        t0 = time.time()
        try:
            if arch_name == 'spatial_cnn':
                data = spatial_data
                result = walk_forward_deep(
                    arch_name, None, data['target'], data['day_boundaries'],
                    device=args.device, window_size=args.window,
                    min_train_days=args.min_train_days,
                    max_epochs=args.max_epochs, batch_size=args.batch_size,
                    subsample_train=args.subsample_train,
                    node_features=data['node_features'],
                    global_features=data['global_features'],
                )
            else:
                data = flat_data
                result = walk_forward_deep(
                    arch_name, data['features'], data['target'],
                    data['day_boundaries'],
                    device=args.device, window_size=args.window,
                    top_k_features=args.top_k,
                    min_train_days=args.min_train_days,
                    max_epochs=args.max_epochs, batch_size=args.batch_size,
                    subsample_train=args.subsample_train,
                )

            result['wall_time'] = time.time() - t0
            result['architecture'] = arch_name
            results[arch_name] = result

            if 'error' not in result:
                logger.info(f"\n  {arch_name.upper()} RESULTS:")
                logger.info(f"  IC={result['ic']:.4f}  ICIR={result['icir']:.2f}  "
                          f"t={result['tstat']:.2f}")
                logger.info(f"  HR={result['hit_rate']:.1%}  PF={result['profit_factor']:.2f}")
                logger.info(f"  Time: {result['wall_time']:.0f}s total")
            else:
                logger.error(f"  FAILED: {result['error']}")

        except Exception as e:
            logger.error(f"  CRASHED: {e}")
            import traceback
            logger.error(traceback.format_exc())
            results[arch_name] = {'error': str(e), 'architecture': arch_name}

        gc.collect()
        torch.cuda.empty_cache()

    # Print comparison table
    logger.info("")
    logger.info("=" * 80)
    logger.info("DEEP LEARNING COMPARISON TABLE")
    logger.info("=" * 80)
    logger.info(f"{'Architecture':<16} {'IC':>8} {'ICIR':>8} {'t-stat':>8} "
                f"{'HR':>8} {'PF':>8} {'Time(s)':>10}")
    logger.info("-" * 80)

    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1].get('ic', -999),
        reverse=True,
    )

    for arch_name, result in sorted_results:
        if 'error' in result:
            logger.info(f"{arch_name:<16} {'FAILED':>60}")
        else:
            logger.info(
                f"{arch_name:<16} {result['ic']:>8.4f} {result['icir']:>8.2f} "
                f"{result['tstat']:>8.2f} {result['hit_rate']:>7.1%} "
                f"{result['profit_factor']:>8.2f} {result['wall_time']:>10.0f}"
            )

    logger.info("=" * 80)

    # Save
    json_file = results_dir / f'deep_benchmark_{timestamp}.json'
    save_results = {}
    for k, v in results.items():
        save_results[k] = {kk: (float(vv) if isinstance(vv, np.floating) else vv)
                           for kk, vv in v.items() if kk != 'fold_ics'}
        if 'fold_ics' in v:
            save_results[k]['fold_ics'] = [float(x) for x in v['fold_ics']]

    with open(json_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'config': {
                'n_days': args.n_days, 'horizon': args.horizon,
                'window': args.window, 'top_k': args.top_k,
                'subsample_train': args.subsample_train,
                'max_epochs': args.max_epochs,
            },
            'results': save_results,
        }, f, indent=2)

    logger.info(f"\nResults: {json_file}")
    logger.info(f"Log: {log_file}")


if __name__ == '__main__':
    main()
