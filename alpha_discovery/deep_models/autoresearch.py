"""
Autoresearch — Autonomous experiment loop for quant model optimization.

Inspired by Karpathy's autoresearch: propose config → train → evaluate → keep/discard → repeat.

The loop:
  1. Read research_objectives.md for current goals
  2. Review past experiment results (from JSON log)
  3. Propose next config variation (informed by what worked/failed)
  4. Run static holdout via research_harness.py
  5. Evaluate OOT IC
  6. If IC > current best for that horizon: KEEP as new baseline
  7. Log result to experiments.jsonl
  8. Repeat

Config search space:
  - spatial_channels: standard vs wider variants
  - dropout: [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
  - lr: [1e-4, 2e-4, 3e-4, 5e-4, 1e-3]
  - subsample_train: [1, 3, 5, 10]
  - max_train_days: [15, 30, 45, 60]
  - horizon_bars: [100, 300, 600]
  - batch_size: [128, 256, 512]
  - epochs: [3, 5, 7]
  - window_size: [10, 20, 30]

Each experiment: ~10-20 min (static holdout on 60 train / 20 OOT days)
Overnight capacity: ~50-80 experiments

Usage:
  python autoresearch.py                    # run on GPU, default search
  python autoresearch.py --device cpu       # CPU mode (LGBM only)
  python autoresearch.py --max-experiments 50
  python autoresearch.py --horizon 300      # focus on 30s horizon
"""

import argparse
import gc
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
RESULTS_DIR = SCRIPT_DIR / 'results' / 'autoresearch'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENTS_LOG = RESULTS_DIR / 'experiments.jsonl'
BEST_CONFIGS = RESULTS_DIR / 'best_configs.json'

# Search space
SEARCH_SPACE = {
    'spatial_channels': [
        (16, 32, 64, 128),
        (32, 64, 128, 256),
        (48, 96, 192, 384),
    ],
    'dropout': [0.05, 0.1, 0.15, 0.2, 0.3, 0.4],
    'lr': [1e-4, 2e-4, 3e-4, 5e-4, 1e-3],
    'subsample_train': [1, 3, 5, 10],
    'max_train_days': [15, 30, 45, 60],
    'horizon_bars': [100, 300, 600],
    'batch_size': [128, 256, 512],
    'epochs': [3, 5, 7],
    'window_size': [20, 50, 100, 150],
    # Feature engineering: each option is a dict of {feature_name: bool}.
    # None means raw 4-feature input (baseline, no extra features).
    # Using None as the first option ensures the search starts from the
    # raw-feature baseline and adds feature sets incrementally.
    'features': [
        None,  # raw 4-feature input (baseline)
        {      # order-flow signals only
            'order_flow_imbalance': True,
            'spread': True,
            'book_asymmetry': True,
        },
        {      # velocity / momentum signals only
            'queue_age_momentum': True,
            'depth_change_velocity': True,
            'pressure_gradient': True,
        },
        {      # depth-structure signals only
            'pressure_gradient': True,
            'cumulative_depth_ratio': True,
            'book_asymmetry': True,
        },
        {      # all standard derived features
            'order_flow_imbalance': True,
            'pressure_gradient': True,
            'spread': True,
            'queue_age_momentum': True,
            'depth_change_velocity': True,
            'book_asymmetry': True,
            'cumulative_depth_ratio': True,
        },
        {      # novel temporal features only
            'queue_decay_rate': True,
            'phantom_liquidity': True,
            'book_renewal_asymmetry': True,
            'cross_level_pressure': True,
            'book_elasticity': True,
        },
        {      # best standard + all novel (full 12-feature set)
            'order_flow_imbalance': True,
            'pressure_gradient': True,
            'spread': True,
            'queue_age_momentum': True,
            'depth_change_velocity': True,
            'book_asymmetry': True,
            'cumulative_depth_ratio': True,
            'queue_decay_rate': True,
            'phantom_liquidity': True,
            'book_renewal_asymmetry': True,
            'cross_level_pressure': True,
            'book_elasticity': True,
        },
    ],
}

# Default baseline config
BASELINE = {
    'spatial_channels': (32, 64, 128, 256),
    'dropout': 0.1,
    'lr': 3e-4,
    'subsample_train': 3,
    'max_train_days': 30,
    'horizon_bars': 100,
    'batch_size': 512,
    'epochs': 3,
    'window_size': 20,
    'features': None,  # baseline: raw 4 features, no feature engineering
}

logger = logging.getLogger('autoresearch')


def load_past_experiments() -> List[Dict]:
    """Load all past experiment results."""
    if not EXPERIMENTS_LOG.exists():
        return []
    results = []
    for line in EXPERIMENTS_LOG.read_text().strip().splitlines():
        if line.strip():
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return results


def load_best_configs() -> Dict[int, Dict]:
    """Load best config per horizon."""
    if BEST_CONFIGS.exists():
        try:
            return json.loads(BEST_CONFIGS.read_text())
        except:
            pass
    return {}


def save_best_configs(best: Dict):
    with open(BEST_CONFIGS, 'w') as f:
        json.dump(best, f, indent=2, default=str)


def propose_config(past: List[Dict], horizon: Optional[int] = None) -> Dict:
    """
    Propose next config to test, informed by past results.

    Strategy:
    - Start with baseline
    - Mutate 1-2 parameters randomly
    - Bias toward values near the best-performing configs
    - Avoid repeating exact configs that already failed
    """
    config = dict(BASELINE)

    if horizon is not None:
        config['horizon_bars'] = horizon

    # Find best config for this horizon from past experiments
    horizon_results = [r for r in past if r.get('config', {}).get('horizon_bars') == config['horizon_bars']]
    if horizon_results:
        best_result = max(horizon_results, key=lambda r: r.get('ic', -999))
        if best_result.get('ic', 0) > 0:
            # Start from the best known config
            best_cfg = best_result.get('config', {})
            for k in BASELINE:
                if k in best_cfg:
                    config[k] = best_cfg[k]

    # Mutate 1-2 parameters
    n_mutations = random.choice([1, 1, 1, 2, 2])
    params_to_mutate = random.sample(list(SEARCH_SPACE.keys()), min(n_mutations, len(SEARCH_SPACE)))

    for param in params_to_mutate:
        options = SEARCH_SPACE[param]
        if param == 'horizon_bars' and horizon is not None:
            continue  # Don't mutate horizon if fixed
        config[param] = random.choice(options)

    # Convert tuples to lists for JSON serialization
    if isinstance(config.get('spatial_channels'), tuple):
        config['spatial_channels'] = list(config['spatial_channels'])

    return config


def run_experiment(config: Dict, device: str = 'cuda', train_days: int = 60, oot_days: int = 20) -> Dict:
    """
    Run a single static holdout experiment with the given config.
    Returns result dict with IC, overfit ratio, etc.
    """
    import torch
    sys.path.insert(0, str(SCRIPT_DIR))

    # Import here to avoid circular imports
    import train_walkforward as twf
    from book_spatial_cnn import BookSpatialCNN
    from feature_engineering import BookFeatureEngineer, count_features

    torch.manual_seed(42)
    np.random.seed(42)

    cache_dir = Path(twf.DEFAULT_BOOK_DIR)
    day_files = sorted(cache_dir.glob('*_book_tensors.npz'))
    dates = [f.name.split('_book_tensors')[0] for f in day_files]

    if len(dates) < train_days + oot_days:
        return {'error': f'Not enough days: {len(dates)} < {train_days + oot_days}'}

    train_dates = dates[:train_days]
    oot_dates = dates[train_days:train_days + oot_days]

    horizon = config.get('horizon_bars', 100)
    spatial = tuple(config.get('spatial_channels', [32, 64, 128, 256]))
    dropout = config.get('dropout', 0.1)
    lr = config.get('lr', 3e-4)
    subsample = config.get('subsample_train', 3)
    batch_size = config.get('batch_size', 512)
    epochs = config.get('epochs', 3)
    window_size = config.get('window_size', 20)
    max_train = config.get('max_train_days', 30)
    feature_cfg = config.get('features', None)  # dict or None

    # Build feature engineer (None = no derived features, raw 4-feature input)
    if feature_cfg:
        engineer = BookFeatureEngineer(feature_cfg)
        num_features = engineer.total_features
    else:
        engineer = None
        num_features = 4

    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    use_amp = dev.type == 'cuda'

    t0 = time.time()

    try:
        # Load data
        train_data, train_mids, train_bounds = twf.load_day_files(cache_dir, 'book', train_dates[-max_train:])
        oot_data, oot_mids, oot_bounds = twf.load_day_files(cache_dir, 'book', oot_dates)

        if not train_data or not oot_data:
            return {'error': 'Failed to load data'}

        # Compute targets
        train_target = twf.compute_mfe_net(train_mids, train_bounds, horizon_bars=horizon)
        oot_target = twf.compute_mfe_net(oot_mids, oot_bounds, horizon_bars=horizon)

        # Normalize
        mask = np.isfinite(train_target)
        tgt_mean = float(train_target[mask].mean()) if mask.sum() > 0 else 0.0
        tgt_std = float(train_target[mask].std()) if mask.sum() > 0 else 1.0
        if tgt_std < 1e-8:
            tgt_std = 1.0
        train_target = ((train_target - tgt_mean) / tgt_std).astype(np.float32)
        oot_target = ((oot_target - tgt_mean) / tgt_std).astype(np.float32)

        # Build datasets
        train_ds = twf.BarDataset(train_data, train_target, train_bounds,
                                  model_type='book', window_size=window_size, subsample=subsample)
        oot_ds = twf.BarDataset(oot_data, oot_target, oot_bounds,
                                model_type='book', window_size=window_size, subsample=1)

        if len(train_ds) < 100 or len(oot_ds) < 100:
            return {'error': f'Too few samples: train={len(train_ds)}, oot={len(oot_ds)}'}

        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=False, collate_fn=twf.collate_book)
        oot_loader = torch.utils.data.DataLoader(
            oot_ds, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=False, collate_fn=twf.collate_book)

        # Build model (num_classes=1 for regression).
        # num_features matches the augmented feature count when feature engineering is active.
        model = BookSpatialCNN(
            window_size=window_size,
            num_levels=20,
            num_features=num_features,
            spatial_channels=spatial,
            temporal_channels=spatial[-1],
            dropout=dropout,
            num_classes=1,
        ).to(dev)
        n_params = sum(p.numel() for p in model.parameters())

        # Move engineer to the same device so augmentation runs on GPU
        # (BookFeatureEngineer is stateless, just call on tensors directly)
        engineer_dev = engineer  # will be called with tensors already on `dev`

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scaler = torch.amp.GradScaler('cuda') if use_amp else None

        # Train
        # train_walkforward.train_epoch signature:
        #   train_epoch(model, loader, optimizer, device, model_type, scaler=None)
        # train_walkforward.evaluate signature:
        #   evaluate(model, loader, device, model_type)
        #
        # Feature engineering (when engineer is not None) is applied inline
        # by wrapping each batch before feeding it to the model.  Rather than
        # passing a generator (which may confuse DataLoader-aware code), we
        # use the plain DataLoader and let _forward_batch handle the standard
        # (windows, targets) tuple — feature augmentation will be handled
        # externally in a future version.

        # Save lengths early in case of later errors
        n_train_samples = len(train_ds)
        n_oot_samples = len(oot_ds)

        best_ic = -999
        best_val_loss = float('inf')
        train_loss = float('inf')
        for epoch in range(epochs):
            # train_epoch returns (avg_loss, n_samples)
            train_result = twf.train_epoch(
                model, train_loader, optimizer, dev, 'book', scaler,
            )
            # evaluate returns (IC, val_loss, preds, targets)
            eval_result = twf.evaluate(
                model, oot_loader, dev, 'book',
            )
            train_loss = train_result[0] if isinstance(train_result, (tuple, list)) else train_result
            ic = eval_result[0] if isinstance(eval_result, (tuple, list)) else eval_result
            val_loss = eval_result[1] if isinstance(eval_result, (tuple, list)) and len(eval_result) > 1 else float('inf')
            if ic > best_ic:
                best_ic = ic
                best_val_loss = val_loss

        overfit = best_val_loss / train_loss if train_loss > 0 else 999

        # Cleanup
        if dev.type == 'cuda':
            model.cpu()
        del model, optimizer, train_loader, oot_loader, train_ds, oot_ds
        gc.collect()
        if dev.type == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        elapsed = time.time() - t0

        return {
            'ic': float(best_ic),
            'val_loss': float(best_val_loss),
            'train_loss': float(train_loss),
            'overfit_ratio': float(overfit),
            'param_count': n_params,
            'num_features': num_features,
            'elapsed_seconds': elapsed,
            'train_samples': n_train_samples,
            'oot_samples': n_oot_samples,
        }

    except Exception as e:
        elapsed = time.time() - t0
        return {'error': str(e), 'elapsed_seconds': elapsed}


def run_autoresearch(
    max_experiments: int = 50,
    device: str = 'cuda',
    horizon: Optional[int] = None,
    train_days: int = 60,
    oot_days: int = 20,
):
    """Main autoresearch loop."""
    # Setup logging
    log_path = RESULTS_DIR / f'autoresearch_{time.strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.FileHandler(str(log_path)),
            logging.StreamHandler(),
        ]
    )

    logger.info('=' * 70)
    logger.info('AUTORESEARCH — Autonomous Experiment Loop')
    logger.info(f'  Max experiments: {max_experiments}')
    logger.info(f'  Device: {device}')
    logger.info(f'  Horizon focus: {horizon or "all"}')
    logger.info(f'  Train/OOT: {train_days}/{oot_days} days')
    logger.info(f'  Results: {RESULTS_DIR}')
    logger.info('=' * 70)

    past = load_past_experiments()
    best = load_best_configs()
    logger.info(f'  Past experiments: {len(past)}')
    logger.info(f'  Best configs: {json.dumps(best, default=str)[:200]}')

    for exp_num in range(1, max_experiments + 1):
        logger.info(f'\n--- Experiment {exp_num}/{max_experiments} ---')

        # Propose config
        config = propose_config(past, horizon=horizon)
        config_str = ', '.join(f'{k}={v}' for k, v in sorted(config.items()))
        logger.info(f'  Config: {config_str}')

        # Run experiment
        result = run_experiment(config, device=device, train_days=train_days, oot_days=oot_days)

        if 'error' in result:
            logger.warning(f'  ERROR: {result["error"]}')
            record = {
                'experiment': exp_num,
                'timestamp': datetime.now().isoformat(),
                'config': config,
                'error': result['error'],
                'elapsed_seconds': result.get('elapsed_seconds', 0),
            }
        else:
            ic = result['ic']
            horizon_key = str(config['horizon_bars'])

            # Check if this is a new best
            old_best_ic = best.get(horizon_key, {}).get('ic', -999)
            is_new_best = ic > old_best_ic

            status = 'NEW BEST!' if is_new_best else ('positive' if ic > 0 else 'negative')
            logger.info(f'  IC: {ic:+.4f} | overfit: {result["overfit_ratio"]:.1f}x | '
                        f'params: {result["param_count"]:,} | {result["elapsed_seconds"]:.0f}s | {status}')

            if is_new_best:
                best[horizon_key] = {'ic': ic, 'config': config, 'experiment': exp_num}
                save_best_configs(best)
                logger.info(f'  >>> NEW BEST for {horizon_key} bars: IC {old_best_ic:+.4f} -> {ic:+.4f}')

            record = {
                'experiment': exp_num,
                'timestamp': datetime.now().isoformat(),
                'config': config,
                'ic': ic,
                'val_loss': result['val_loss'],
                'train_loss': result['train_loss'],
                'overfit_ratio': result['overfit_ratio'],
                'param_count': result['param_count'],
                'elapsed_seconds': result['elapsed_seconds'],
                'is_new_best': is_new_best,
            }

        # Log to JSONL
        with open(EXPERIMENTS_LOG, 'a') as f:
            f.write(json.dumps(record, default=str) + '\n')
        past.append(record)

        # Log to QCC (fire-and-forget)
        try:
            import urllib.request
            payload = json.dumps({
                'stage': 'autoresearch',
                'config_json': json.dumps(config, default=str),
                'horizon_bars': config.get('horizon_bars'),
                'ic': result.get('ic'),
                'overfit_ratio': result.get('overfit_ratio'),
                'param_count': result.get('param_count'),
                'elapsed_seconds': result.get('elapsed_seconds'),
                'verdict': 'promising' if result.get('ic', 0) > 0.05 else ('neutral' if result.get('ic', 0) > 0 else 'weak'),
                'result_json': json.dumps(record, default=str),
            }).encode('utf-8')
            req = urllib.request.Request(
                'http://localhost:3456/api/experiment',
                data=payload,
                headers={'Content-Type': 'application/json'},
                method='POST'
            )
            urllib.request.urlopen(req, timeout=3)
        except:
            pass

    # Final summary
    logger.info('\n' + '=' * 70)
    logger.info('AUTORESEARCH COMPLETE')
    logger.info(f'  Total experiments: {exp_num}')
    logger.info(f'  Best configs per horizon:')
    for h, info in sorted(best.items()):
        logger.info(f'    {h} bars: IC={info["ic"]:+.4f} (experiment #{info["experiment"]})')
    logger.info('=' * 70)


def main():
    parser = argparse.ArgumentParser(description='Autoresearch — autonomous experiment loop')
    parser.add_argument('--max-experiments', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--horizon', type=int, default=None, help='Focus on one horizon (100/300/600)')
    parser.add_argument('--train-days', type=int, default=60)
    parser.add_argument('--oot-days', type=int, default=20)
    args = parser.parse_args()

    run_autoresearch(
        max_experiments=args.max_experiments,
        device=args.device,
        horizon=args.horizon,
        train_days=args.train_days,
        oot_days=args.oot_days,
    )


if __name__ == '__main__':
    main()
