"""
Multi-Alpha Orchestration — Run the full multi-channel alpha discovery pipeline.

8 Phases:
1. Load data + compute features (existing 149 + 14 event features)
2. Event detection pipeline
3. L1 residual extraction (per walk-forward fold)
4. Independent channel training (5 channels + residual channel)
5. Conditional model (two-stage)
6. Smart ensemble with decorrelation weighting
7. Comparison: single model vs multi-channel vs conditional vs residual
8. Execution backtest on best configuration

Usage:
    python alpha_discovery/run_multi_alpha.py [--n-days N] [--horizon HORIZON] [--target-type TYPE] [--no-execution]

    --n-days N         : Number of days to load (default: all)
    --horizon HORIZON  : Target horizon (default: ret_3s)
    --target-type TYPE : return | mfe_net | mfe_long | mfe_short (default: return)
    --no-execution     : Skip execution backtest phase
"""

import gc
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy.stats import spearmanr

# Add project root
LVL3_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(LVL3_ROOT))

from alpha_discovery.mbo_alpha_scan import MBOAlphaScanner
from alpha_discovery.mbo_features import get_feature_names, TOTAL_FEATURES
from alpha_discovery.event_detector import (
    EventDetectionPipeline,
    compute_event_features,
    get_event_feature_names,
    N_EVENT_FEATURES,
)
from alpha_discovery.multi_channel_alpha import (
    MultiChannelAlphaScanner,
    SmartEnsemble,
    StackingMetaModel,
    extract_l1_residual,
    validate_channel_disjointness,
    format_multi_channel_report,
    CHANNEL_DEFINITIONS,
    CH1_L1_IMBALANCE,
)
from alpha_discovery.conditional_model import (
    ConditionalModel,
    format_conditional_report,
)
from alpha_discovery.run_mfe_scan import compute_mfe_targets

# Results directory
RESULTS_DIR = LVL3_ROOT / "alpha_discovery" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging — explicit handlers (basicConfig is unreliable when modules pre-configure root)
_log_file = RESULTS_DIR / f"multi_alpha_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
_root = logging.getLogger()
_root.setLevel(logging.INFO)
for _h in _root.handlers[:]:
    _root.removeHandler(_h)
_fmt = logging.Formatter('%(asctime)s [%(name)s] %(message)s', datefmt='%H:%M:%S')
_fh = logging.FileHandler(str(_log_file), mode='w')
_fh.setFormatter(_fmt)
_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(_fmt)
_root.addHandler(_fh)
_root.addHandler(_sh)
sys.stdout.flush()
logger = logging.getLogger("multi_alpha")

# Horizon definitions (bars at 100ms interval)
HORIZONS = {
    'ret_3s': 30,
    'ret_5s': 50,
    'ret_10s': 100,
    'ret_30s': 300,
    'ret_1m': 600,
    'ret_3m': 1800,
    'ret_5m': 3000,
}


def compute_target(mid_prices, horizon_bars, day_boundaries):
    """Compute forward return target with day-boundary NaN masking."""
    N = len(mid_prices)
    n_days = len(day_boundaries) - 1

    future_mid = np.empty(N, dtype=np.float32)
    future_mid[:N - horizon_bars] = mid_prices[horizon_bars:]
    future_mid[N - horizon_bars:] = np.nan

    # NaN-fill targets crossing day boundaries
    if n_days > 1:
        for d in range(n_days - 1):
            day_end = day_boundaries[d + 1]
            nan_start = max(day_boundaries[d], day_end - horizon_bars)
            future_mid[nan_start:day_end] = np.nan

    ret = (future_mid - mid_prices) / np.maximum(mid_prices, 1.0)
    return ret


def run_pipeline(args):
    """Run the full multi-alpha pipeline."""
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIR / f"multi_alpha_{timestamp}.json"

    logger.info("=" * 80)
    logger.info("MULTI-ALPHA ARCHITECTURE — Pipeline Start")
    logger.info("=" * 80)

    all_results = {
        'timestamp': timestamp,
        'horizon': args.horizon,
        'phases': {},
    }

    # ================================================================
    # PHASE 1: Load data + compute features
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1: Loading data and computing features")
    logger.info("=" * 60)

    t0 = time.time()
    scanner = MBOAlphaScanner()

    use_precomputed = bool(args.feature_cache)
    if use_precomputed:
        # Load pre-computed 290 features + pre-allocate event columns (zero-copy augmentation)
        logger.info(f"Using pre-computed feature cache: {args.feature_cache}")
        load_info = scanner.load_precomputed_features(
            feature_cache_dir=args.feature_cache,
            n_days=args.n_days,
            extra_cols=N_EVENT_FEATURES,  # Pre-allocate space for event features
        )
    else:
        load_info = scanner.load_from_cache(n_days=args.n_days)

    logger.info(f"Loaded {load_info['n_days']} days, "
                f"{load_info['n_snapshots']:,} snapshots, "
                f"{load_info['n_features']} features"
                + (" (pre-computed)" if load_info.get('precomputed') else ""))

    all_results['phases']['load'] = {
        'n_days': load_info['n_days'],
        'n_snapshots': load_info['n_snapshots'],
        'n_base_features': load_info['n_features'],
        'time_sec': time.time() - t0,
    }

    # ================================================================
    # PHASE 2: Event detection pipeline
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2: Event detection pipeline")
    logger.info("=" * 60)

    t0 = time.time()
    n_base = load_info['n_features']

    pipeline = EventDetectionPipeline()
    event_features = pipeline.detect_all(
        scanner.features[:, :n_base],  # Only pass base features (not extra cols)
        scanner.feature_names,
        scanner.day_boundaries,
    )

    # Get event stats
    event_stats = pipeline.get_event_stats(event_features)
    for name, stats in event_stats.items():
        if stats.get('fire_rate', 0) > 0:
            logger.info(f"  {name}: fire_rate={stats['fire_rate']:.4f} "
                        f"({stats['n_fires']:,} fires)")

    augmented_names = list(scanner.feature_names) + get_event_feature_names()
    n_event = event_features.shape[1]

    if use_precomputed and load_info.get('extra_cols', 0) >= n_event:
        # In-place: event features go into the pre-allocated extra columns (ZERO extra memory)
        scanner.features[:, n_base:n_base + n_event] = event_features
        del event_features
        gc.collect()
        augmented_features = scanner.features  # No copy — same array
        logger.info(f"In-place augmentation: {n_base} + {n_event} = {n_base + n_event} features "
                    f"(no extra memory)")
    else:
        # Fallback: allocate new array (original path for non-cached runs)
        n_rows = scanner.features.shape[0]
        logger.info(f"Memory-safe augmentation: pre-allocate {n_rows:,} x {n_base + n_event} ...")
        augmented_features = np.empty((n_rows, n_base + n_event), dtype=np.float32)
        augmented_features[:, :n_base] = scanner.features
        del scanner.features
        gc.collect()
        augmented_features[:, n_base:] = event_features
        del event_features
        gc.collect()

    logger.info(f"Augmented features shape: {augmented_features.shape}")

    all_results['phases']['events'] = {
        'n_event_features': N_EVENT_FEATURES,
        'total_features': augmented_features.shape[1],
        'event_stats': {k: {kk: vv for kk, vv in v.items()
                            if isinstance(vv, (int, float))}
                        for k, v in event_stats.items()},
        'time_sec': time.time() - t0,
    }

    # Validate channel disjointness
    issues = validate_channel_disjointness(augmented_names)
    overlaps = {k: v for k, v in issues.items() if 'missing' not in k and k != 'unassigned'}
    if overlaps:
        logger.warning(f"CHANNEL OVERLAP DETECTED: {overlaps}")
    missing_issues = {k: v for k, v in issues.items() if 'missing' in k}
    for ch, missing in missing_issues.items():
        logger.info(f"  {ch}: {len(missing)} features not in matrix (OK if event features)")

    # ================================================================
    # PHASE 3: Compute target + L1 residual extraction
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3: Target computation + L1 residual extraction")
    logger.info("=" * 60)

    t0 = time.time()
    horizon_bars = HORIZONS.get(args.horizon, 30)
    target_type = getattr(args, 'target_type', 'return')

    # Always compute return target (needed for L1 residual + comparison)
    return_target = compute_target(
        scanner.mid_prices, horizon_bars, scanner.day_boundaries
    )

    if target_type == 'return':
        target = return_target
        target_label = 'return'
    else:
        # Compute MFE targets for this horizon
        hz_name = args.horizon.replace('ret_', '')  # 'ret_5s' -> '5s'
        hz_sec_map = {
            '3s': 3, '5s': 5, '10s': 10, '30s': 30, '1m': 60, '3m': 180, '5m': 300,
        }
        hz_sec = hz_sec_map.get(hz_name, 5)
        logger.info(f"Computing MFE targets for {hz_name} ({hz_sec}s) ...")
        mfe_targets = compute_mfe_targets(
            mid_prices=scanner.mid_prices,
            day_boundaries=scanner.day_boundaries,
            sample_interval_ms=100,
            horizons_sec={hz_name: hz_sec},
            tick_size=0.25,
        )

        if target_type == 'mfe_net':
            target = mfe_targets[f'mfe_net_{hz_name}']
            target_label = f'mfe_net_{hz_name}'
        elif target_type == 'mfe_long':
            target = mfe_targets[f'mfe_long_{hz_name}']
            target_label = f'mfe_long_{hz_name}'
        elif target_type == 'mfe_short':
            target = mfe_targets[f'mfe_short_{hz_name}']
            target_label = f'mfe_short_{hz_name}'
        else:
            logger.warning(f"Unknown target_type '{target_type}', falling back to return")
            target = return_target
            target_label = 'return'

        logger.info(f"Using target: {target_label}")
        all_results['target_type'] = target_type
        all_results['target_label'] = target_label

        # Free MFE dict (we only need the selected target)
        del mfe_targets
        gc.collect()

    n_valid_target = np.isfinite(target).sum()
    logger.info(f"Target: {args.horizon} ({horizon_bars} bars), type={target_type}, "
                f"{n_valid_target:,} valid values")

    # Extract L1 residual (always against return target for consistency)
    residual_target = extract_l1_residual(
        augmented_features,
        return_target,
        augmented_names,
        scanner.day_boundaries,
        min_train_days=args.min_train_days,
    )

    n_valid_residual = np.isfinite(residual_target).sum()
    residual_corr = 0.0
    if n_valid_residual > 100:
        mask = np.isfinite(target) & np.isfinite(residual_target)
        if mask.sum() > 100:
            residual_corr = float(np.corrcoef(target[mask], residual_target[mask])[0, 1])
    logger.info(f"Residual: {n_valid_residual:,} valid, "
                f"corr with primary target: {residual_corr:.4f}")

    all_results['phases']['residual'] = {
        'horizon': args.horizon,
        'horizon_bars': horizon_bars,
        'n_valid_target': int(n_valid_target),
        'n_valid_residual': int(n_valid_residual),
        'residual_corr_with_original': residual_corr,
        'time_sec': time.time() - t0,
    }

    # ================================================================
    # PHASE 4: Independent channel training
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 4: Independent channel training")
    logger.info("=" * 60)

    t0 = time.time()
    multi_scanner = MultiChannelAlphaScanner(
        features=augmented_features,
        feature_names=augmented_names,
        mid_prices=scanner.mid_prices,
        day_boundaries=scanner.day_boundaries,
        min_train_days=args.min_train_days,
    )

    channel_results = multi_scanner.run_all_channels(
        target=target,
        target_name=target_type if target_type != 'return' else 'return',
        horizon_name=args.horizon,
        residual_target=residual_target,
    )

    all_results['phases']['channels'] = {
        'n_channels': len(channel_results),
        'per_channel': {},
        'time_sec': time.time() - t0,
    }
    for ch_name, result in channel_results.items():
        all_results['phases']['channels']['per_channel'][ch_name] = {
            k: v for k, v in result.items()
            if isinstance(v, (int, float, str, bool, list))
        }

    # Checkpoint: save intermediate results after expensive channel training
    checkpoint_file = RESULTS_DIR / f"checkpoint_channels_{timestamp}.json"
    with open(str(checkpoint_file), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Checkpoint saved: {checkpoint_file.name}")
    gc.collect()

    # ================================================================
    # PHASE 5: Conditional model (two-stage)
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 5: Conditional model (two-stage)")
    logger.info("=" * 60)

    t0 = time.time()
    cond_model = ConditionalModel(
        prob_threshold=0.5,
        interestingness_percentile=70.0,
    )
    cond_result = cond_model.run(
        augmented_features,
        target,
        scanner.day_boundaries,
        min_train_days=args.min_train_days,
    )

    all_results['phases']['conditional'] = {
        k: v for k, v in cond_result.items()
        if isinstance(v, (int, float, str, bool, list))
    }
    all_results['phases']['conditional']['time_sec'] = time.time() - t0

    if 'error' not in cond_result:
        logger.info(f"Conditional model: IC={cond_result['ic']:.4f} "
                    f"t={cond_result['tstat']:.2f} "
                    f"on {cond_result['pct_interesting']:.1%} of bars")

    # ================================================================
    # PHASE 6: Smart ensemble with decorrelation weighting
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 6: Smart ensemble")
    logger.info("=" * 60)

    t0 = time.time()
    ensemble = SmartEnsemble(min_ic=0.0, decorr_weight=0.5)
    ens_preds, ens_actuals, weights = ensemble.combine(
        channel_results, multi_scanner.channels
    )

    if len(ens_preds) > 50:
        ens_metrics = SmartEnsemble.compute_ensemble_metrics(
            ens_preds, ens_actuals, 'smart_ensemble'
        )
        logger.info(f"Ensemble: IC={ens_metrics.get('ic', 0):.4f} "
                    f"HR={ens_metrics.get('hit_rate', 0):.1%} "
                    f"PF={ens_metrics.get('profit_factor', 0):.2f}")
    else:
        ens_metrics = {'error': 'insufficient predictions'}
        logger.info("Ensemble: insufficient predictions")

    all_results['phases']['ensemble'] = {
        'metrics': ens_metrics,
        'weights': {k: float(v) for k, v in weights.items()},
        'time_sec': time.time() - t0,
    }

    # ================================================================
    # PHASE 6b: Stacking meta-model ensemble
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 6b: Stacking meta-model ensemble")
    logger.info("=" * 60)

    t0 = time.time()
    try:
        stacking = StackingMetaModel(magnitude_threshold=1.5)
        stack_preds, stack_actuals, stack_info = stacking.combine(
            channel_results, multi_scanner.channels
        )

        if len(stack_preds) > 50:
            stack_metrics = SmartEnsemble.compute_ensemble_metrics(
                stack_preds, stack_actuals, 'stacking_ensemble'
            )
            logger.info(f"Stacking: IC={stack_metrics.get('ic', 0):.4f} "
                        f"HR={stack_metrics.get('hit_rate', 0):.1%} "
                        f"PF={stack_metrics.get('profit_factor', 0):.2f} "
                        f"(method={stack_info.get('method', '?')}, "
                        f"folds={stack_info.get('n_folds_used', '?')})")
        else:
            stack_metrics = {'error': 'insufficient predictions'}
            logger.info(f"Stacking: insufficient predictions ({stack_info})")
    except Exception as e:
        logger.warning(f"Stacking failed: {e}")
        stack_metrics = {'error': str(e)}
        stack_info = {}

    all_results['phases']['stacking_ensemble'] = {
        'metrics': stack_metrics,
        'info': {k: v for k, v in stack_info.items() if isinstance(v, (int, float, str, bool, list))},
        'time_sec': time.time() - t0,
    }

    # ================================================================
    # PHASE 6c: Magnitude-gated selective trading
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 6c: Magnitude-gated selective trading")
    logger.info("=" * 60)

    t0 = time.time()
    tick_size = 0.25
    try:
        # Compute magnitude target (abs price change in ticks)
        future_mid = np.empty(len(scanner.mid_prices), dtype=np.float32)
        future_mid[:len(scanner.mid_prices) - horizon_bars] = scanner.mid_prices[horizon_bars:]
        future_mid[len(scanner.mid_prices) - horizon_bars:] = np.nan
        # NaN-fill day boundary crossings
        if scanner.day_boundaries and len(scanner.day_boundaries) > 1:
            for d in range(len(scanner.day_boundaries) - 2):
                day_end = scanner.day_boundaries[d + 1]
                nan_start = max(scanner.day_boundaries[d], day_end - horizon_bars)
                future_mid[nan_start:day_end] = np.nan
        mag_actual = np.abs(future_mid - scanner.mid_prices) / tick_size

        # Walk-forward magnitude prediction using scanner
        scanner.features = augmented_features
        scanner.feature_names = list(augmented_names)
        # Reconstruct hour_of_day and time_since_rth from the feature matrix
        # (load_precomputed_features does not set these; they live in the feature columns)
        if scanner.hour_of_day is None and 'hour_norm' in augmented_names:
            hour_idx = list(augmented_names).index('hour_norm')
            scanner.hour_of_day = augmented_features[:, hour_idx] * 24.0
        if scanner.time_since_rth is None and 'time_since_rth' in augmented_names:
            rth_idx = list(augmented_names).index('time_since_rth')
            scanner.time_since_rth = augmented_features[:, rth_idx]
        mag_result = scanner.walk_forward_evaluate(
            target=mag_actual,
            target_name='magnitude',
            horizon_name=args.horizon,
            min_train_days=args.min_train_days,
        )

        if 'error' not in mag_result:
            logger.info(f"Magnitude model: IC={mag_result['ic']:.4f} "
                        f"t={mag_result['tstat']:.2f}")

            # Apply magnitude gate: only evaluate direction on high-magnitude bars
            # Must align per-fold: fold predictions correspond to NaN-filtered test windows
            valid_channels = {
                name: ch for name, ch in multi_scanner.channels.items()
                if name in channel_results
                and 'error' not in channel_results[name]
                and len(ch.fold_predictions) > 0
            }

            n_days = len(scanner.day_boundaries) - 1

            for mag_thresh in [1.0, 1.5, 2.0, 3.0]:
                finite_mag = np.isfinite(mag_actual)
                pct_gated = float((mag_actual[finite_mag] > mag_thresh).mean()) if finite_mag.any() else 0.0

                if not valid_channels:
                    continue

                n_folds = min(len(ch.fold_predictions) for ch in valid_channels.values())
                gated_preds_all = []
                gated_actuals_all = []
                gated_mag_all = []

                for fold_idx in range(n_folds):
                    test_day = fold_idx + args.min_train_days
                    if test_day >= n_days:
                        continue

                    test_start = scanner.day_boundaries[test_day]
                    test_end = scanner.day_boundaries[test_day + 1]

                    # Replicate the NaN filter that train_fold() applied
                    test_target = target[test_start:test_end]
                    test_valid = np.isfinite(test_target)
                    if test_valid.sum() == 0:
                        continue

                    # Magnitude for valid (non-NaN target) rows in this test window
                    fold_mag = mag_actual[test_start:test_end][test_valid]
                    gate = fold_mag > mag_thresh
                    if gate.sum() < 5:
                        continue

                    # Reconstruct ensemble prediction for this fold
                    fold_lens = [
                        len(valid_channels[ch_name].fold_predictions[fold_idx])
                        for ch_name in valid_channels
                    ]
                    min_len = min(fold_lens)
                    if min_len == 0:
                        continue

                    fold_pred = np.zeros(min_len, dtype=np.float32)
                    for ch_name, ch in valid_channels.items():
                        w = weights.get(ch_name, 0.0)
                        fold_pred += w * ch.fold_predictions[fold_idx][:min_len]

                    first_ch = next(iter(valid_channels.values()))
                    fold_actual = first_ch.fold_actuals[fold_idx][:min_len]

                    # Apply gate (trim to prediction length)
                    gate_trim = gate[:min_len]
                    if gate_trim.sum() > 0:
                        gated_preds_all.append(fold_pred[gate_trim])
                        gated_actuals_all.append(fold_actual[gate_trim])
                        gated_mag_all.append(fold_mag[:min_len][gate_trim])

                if gated_preds_all:
                    all_gp = np.concatenate(gated_preds_all)
                    all_ga = np.concatenate(gated_actuals_all)
                    all_gm = np.concatenate(gated_mag_all)

                    if len(all_gp) > 50:
                        gated_ic = float(spearmanr(all_gp, all_ga)[0])
                        gated_hr = float((np.sign(all_gp) == np.sign(all_ga)).mean())
                        avg_move = float(all_gm.mean())  # already in ticks
                        logger.info(
                            f"  Gate >{mag_thresh:.0f}t: IC={gated_ic:.4f} HR={gated_hr:.1%} "
                            f"avg_move={avg_move:.1f}t "
                            f"({pct_gated:.1%} of bars, {len(all_gp)} predictions)")
        else:
            logger.info(f"Magnitude model failed: {mag_result.get('error', 'unknown')}")
    except Exception as e:
        logger.warning(f"Magnitude gate failed: {e}")
        mag_result = {'error': str(e)}

    all_results['phases']['magnitude_gate'] = {
        k: v for k, v in mag_result.items()
        if isinstance(v, (int, float, str, bool, list))
    }
    all_results['phases']['magnitude_gate']['time_sec'] = time.time() - t0

    # ================================================================
    # PHASE 6d: Save predictions for offline simulation (optional)
    # ================================================================
    if getattr(args, 'save_predictions', False):
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 6d: Saving full-length prediction arrays")
        logger.info("=" * 60)

        t0 = time.time()
        N = len(scanner.mid_prices)
        n_days = len(scanner.day_boundaries) - 1

        # Reconstruct full-length ensemble predictions from channel folds
        valid_channels = {
            name: ch for name, ch in multi_scanner.channels.items()
            if name in channel_results
            and 'error' not in channel_results[name]
            and len(ch.fold_predictions) > 0
        }

        full_direction_preds = np.full(N, np.nan, dtype=np.float32)
        if valid_channels:
            n_folds = min(len(ch.fold_predictions) for ch in valid_channels.values())
            for fold_idx in range(n_folds):
                test_day = fold_idx + args.min_train_days
                if test_day >= n_days:
                    continue
                test_start = scanner.day_boundaries[test_day]
                test_end = scanner.day_boundaries[test_day + 1]
                test_target = target[test_start:test_end]
                test_valid = np.isfinite(test_target)
                if test_valid.sum() == 0:
                    continue

                # Weighted ensemble prediction for this fold
                fold_lens = [len(valid_channels[ch].fold_predictions[fold_idx])
                             for ch in valid_channels]
                min_len = min(fold_lens)
                if min_len == 0:
                    continue

                fold_pred = np.zeros(min_len, dtype=np.float32)
                for ch_name, ch in valid_channels.items():
                    w = weights.get(ch_name, 0.0)
                    fold_pred += w * ch.fold_predictions[fold_idx][:min_len]

                # Map back to full array
                valid_positions = np.arange(test_start, test_end)[test_valid]
                n = min(len(valid_positions), min_len)
                full_direction_preds[valid_positions[:n]] = fold_pred[:n]

        n_dir_valid = np.isfinite(full_direction_preds).sum()
        logger.info(f"  Direction predictions: {n_dir_valid:,} valid out of {N:,}")

        # Magnitude predictions: reconstruct from magnitude walk-forward
        # (mag_actual is ground truth; we save both model preds and ground truth)
        full_magnitude_preds = np.full(N, np.nan, dtype=np.float32)
        # The magnitude model was trained via scanner.walk_forward_evaluate()
        # which stores concatenated preds but NOT per-fold mapping.
        # Re-train magnitude with our train_walk_forward to get full-length array:
        try:
            from alpha_discovery.magnitude_gated_sim import train_walk_forward as _twf
            logger.info("  Training magnitude model for prediction saving...")
            scanner.features = augmented_features
            scanner.feature_names = list(augmented_names)
            full_magnitude_preds, _ = _twf(
                augmented_features, mag_actual, scanner.day_boundaries,
                min_train_days=args.min_train_days,
                target_name='magnitude_save',
            )
        except Exception as e:
            logger.warning(f"  Could not save magnitude predictions: {e}")
            full_magnitude_preds = mag_actual  # fallback to ground truth

        n_mag_valid = np.isfinite(full_magnitude_preds).sum()
        logger.info(f"  Magnitude predictions: {n_mag_valid:,} valid")

        pred_file = RESULTS_DIR / f"predictions_{args.horizon}_{timestamp}.npz"
        np.savez_compressed(
            str(pred_file),
            mid_prices=scanner.mid_prices,
            direction_preds=full_direction_preds,
            magnitude_preds=full_magnitude_preds,
            direction_target=target,
            magnitude_target=mag_actual,
            day_boundaries=np.array(scanner.day_boundaries),
        )
        logger.info(f"  Predictions saved: {pred_file.name} ({time.time()-t0:.0f}s)")
        all_results['phases']['save_predictions'] = {
            'file': str(pred_file.name),
            'n_direction_valid': int(n_dir_valid),
            'n_magnitude_valid': int(n_mag_valid),
            'time_sec': time.time() - t0,
        }

    # ================================================================
    # PHASE 7: Single model baseline + comparison
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 7: Single model baseline + comparison")
    logger.info("=" * 60)

    t0 = time.time()

    try:
        # Re-assign augmented features to scanner (originals were freed for memory)
        scanner.features = augmented_features
        scanner.feature_names = list(augmented_names)
        # Reconstruct hour_of_day and time_since_rth from the feature matrix
        # (load_precomputed_features does not set these; they live in the feature columns)
        if scanner.hour_of_day is None and 'hour_norm' in augmented_names:
            hour_idx = list(augmented_names).index('hour_norm')
            scanner.hour_of_day = augmented_features[:, hour_idx] * 24.0
        if scanner.time_since_rth is None and 'time_since_rth' in augmented_names:
            rth_idx = list(augmented_names).index('time_since_rth')
            scanner.time_since_rth = augmented_features[:, rth_idx]
        logger.info(f"  Scanner features: {scanner.features.shape}, names: {len(scanner.feature_names)}")

        # Run single model (all features, original target)
        single_result_rebuild = scanner.walk_forward_evaluate(
            target=target,
            target_name='return',
            horizon_name=args.horizon,
        )
    except Exception as e:
        logger.warning(f"Phase 7 single model failed: {e}")
        # Use overnight results as fallback reference
        single_result_rebuild = {
            'error': str(e),
            'note': 'See overnight Phase 1 results for single-model baseline (IC=0.1156 on ret_5s)',
        }

    all_results['phases']['single_model'] = {
        k: v for k, v in single_result_rebuild.items()
        if isinstance(v, (int, float, str, bool, list))
    }
    all_results['phases']['single_model']['time_sec'] = time.time() - t0

    if 'error' not in single_result_rebuild:
        logger.info(f"Single model: IC={single_result_rebuild['ic']:.4f} "
                    f"t={single_result_rebuild['tstat']:.2f} "
                    f"HR={single_result_rebuild['hit_rate']:.1%}")
    else:
        logger.info(f"Single model: skipped ({single_result_rebuild.get('error', 'unknown')})")

    # ================================================================
    # COMPARISON TABLE
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON TABLE")
    logger.info("=" * 60)

    comparison = []

    # Single model
    if 'error' not in single_result_rebuild:
        comparison.append({
            'method': 'Single LightGBM (all features)',
            'ic': single_result_rebuild['ic'],
            'icir': single_result_rebuild.get('icir', 0),
            'tstat': single_result_rebuild.get('tstat', 0),
            'hr': single_result_rebuild['hit_rate'],
            'pf': single_result_rebuild['profit_factor'],
            'n': single_result_rebuild['n_predictions'],
        })

    # Individual channels
    for ch_name, result in channel_results.items():
        if 'error' not in result:
            comparison.append({
                'method': f'Channel: {ch_name}',
                'ic': result['ic'],
                'icir': result.get('icir', 0),
                'tstat': result.get('tstat', 0),
                'hr': result['hit_rate'],
                'pf': result['profit_factor'],
                'n': result['n_predictions'],
            })

    # Ensemble (IC-weighted)
    if 'error' not in ens_metrics:
        comparison.append({
            'method': 'Smart Ensemble (IC+decorr)',
            'ic': ens_metrics['ic'],
            'icir': 0,
            'tstat': 0,
            'hr': ens_metrics['hit_rate'],
            'pf': ens_metrics['profit_factor'],
            'n': ens_metrics['n_predictions'],
        })

    # Stacking meta-model
    if 'error' not in stack_metrics:
        comparison.append({
            'method': 'Stacking Meta-Model',
            'ic': stack_metrics['ic'],
            'icir': 0,
            'tstat': 0,
            'hr': stack_metrics['hit_rate'],
            'pf': stack_metrics['profit_factor'],
            'n': stack_metrics['n_predictions'],
        })

    # Conditional
    if 'error' not in cond_result:
        comparison.append({
            'method': f'Conditional ({cond_result["pct_interesting"]:.0%} bars)',
            'ic': cond_result['ic'],
            'icir': cond_result.get('icir', 0),
            'tstat': cond_result.get('tstat', 0),
            'hr': cond_result['hit_rate'],
            'pf': cond_result['profit_factor'],
            'n': cond_result['n_predictions'],
        })

    # Format comparison table
    comparison.sort(key=lambda x: abs(x['ic']), reverse=True)
    logger.info(f"\n{'Method':<40s} {'IC':>7s} {'ICIR':>6s} {'t':>6s} "
                f"{'HR':>6s} {'PF':>6s} {'N':>8s}")
    logger.info("-" * 85)
    for row in comparison:
        logger.info(
            f"{row['method']:<40s} {row['ic']:>7.4f} {row['icir']:>6.2f} "
            f"{row['tstat']:>6.2f} {row['hr']:>6.1%} {row['pf']:>6.2f} "
            f"{row['n']:>8d}"
        )
    logger.info("-" * 85)

    all_results['comparison'] = comparison

    # ================================================================
    # PHASE 8: Execution backtest (optional)
    # ================================================================
    if not args.no_execution:
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 8: Execution backtest")
        logger.info("=" * 60)

        t0 = time.time()
        try:
            from alpha_discovery.execution_backtest import run_execution_backtest

            # Use ensemble predictions if available, else single model
            if len(ens_preds) > 50:
                exec_preds = ens_preds
                exec_actuals = ens_actuals
                exec_label = 'ensemble'
            elif 'error' not in single_result_rebuild:
                # Rebuild single model predictions
                exec_preds = np.concatenate(
                    [scanner.walk_forward_evaluate(
                        target=target, target_name='return',
                        horizon_name=args.horizon,
                    ).get('fold_predictions', [np.array([])])]
                ) if False else np.array([])
                exec_label = 'single'
            else:
                exec_preds = np.array([])
                exec_label = 'none'

            if len(exec_preds) > 100:
                # Build day_boundaries that fit within exec_preds length.
                # ens_preds is a compact NaN-filtered array; its length does not
                # equal the full dataset length, so we must clip the boundaries.
                n_exec = len(exec_preds)
                exec_boundaries = np.array(
                    [b for b in scanner.day_boundaries if b <= n_exec],
                    dtype=np.int64,
                )
                # Ensure the last boundary equals exactly n_exec so the final
                # day is fully covered and nothing runs past the array end.
                if len(exec_boundaries) == 0 or exec_boundaries[-1] < n_exec:
                    exec_boundaries = np.append(exec_boundaries, n_exec)
                exec_result = run_execution_backtest(
                    predictions=exec_preds,
                    actuals=exec_actuals,
                    mid_prices=scanner.mid_prices[:n_exec],
                    features=augmented_features[:n_exec],
                    feature_names=augmented_names,
                    day_boundaries=exec_boundaries,
                    horizon=args.horizon,
                )
                logger.info(f"Execution backtest ({exec_label}): "
                            f"completed in {time.time() - t0:.0f}s")
                all_results['phases']['execution'] = {
                    'source': exec_label,
                    'time_sec': time.time() - t0,
                }
            else:
                logger.info("Skipping execution backtest — insufficient predictions")
                all_results['phases']['execution'] = {'skipped': True}
        except Exception as e:
            logger.warning(f"Execution backtest failed: {e}")
            all_results['phases']['execution'] = {'error': str(e)}
    else:
        logger.info("\nPhase 8 skipped (--no-execution)")

    # ================================================================
    # GENERATE REPORTS
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("GENERATING REPORTS")
    logger.info("=" * 60)

    # Multi-channel report
    report = format_multi_channel_report(
        channel_results=channel_results,
        ensemble_result=ens_metrics,
        single_model_result=single_result_rebuild if 'error' not in single_result_rebuild else None,
        weights=weights,
    )
    logger.info(report)

    # Conditional report
    cond_report = format_conditional_report(
        cond_result=cond_result,
        unconditional_result=single_result_rebuild if 'error' not in single_result_rebuild else None,
    )
    logger.info(cond_report)

    # Save full results
    total_time = time.time() - start_time
    all_results['total_time_sec'] = total_time

    with open(str(results_file), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nResults saved to: {results_file}")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info(f"PIPELINE COMPLETE — Total time: {total_time:.0f}s ({total_time/60:.1f}m)")
    logger.info("=" * 80)

    # Key findings
    best_method = comparison[0] if comparison else None
    if best_method:
        logger.info(f"Best method: {best_method['method']} (IC={best_method['ic']:.4f})")

    # Check for multi-alpha evidence
    positive_channels = [
        name for name, r in channel_results.items()
        if 'error' not in r and r.get('ic', 0) > 0.005
    ]
    if len(positive_channels) > 1:
        logger.info(f"MULTI-ALPHA EVIDENCE: {len(positive_channels)} channels "
                    f"with independent positive IC: {positive_channels}")
    else:
        logger.info("Single alpha source — L1 imbalance remains dominant")

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Multi-Alpha Pipeline')
    parser.add_argument('--n-days', type=int, default=None,
                        help='Number of days to load (default: all)')
    parser.add_argument('--horizon', type=str, default='ret_3s',
                        choices=list(HORIZONS.keys()),
                        help='Target horizon (default: ret_3s)')
    parser.add_argument('--min-train-days', type=int, default=5,
                        help='Minimum training days (default: 5)')
    parser.add_argument('--no-execution', action='store_true',
                        help='Skip execution backtest')
    parser.add_argument('--save-predictions', action='store_true',
                        help='Save full-length prediction arrays as NPZ for offline simulation')
    parser.add_argument('--target-type', type=str, default='return',
                        choices=['return', 'mfe_net', 'mfe_long', 'mfe_short'],
                        help='Target type: return (endpoint), mfe_net (directional asymmetry), '
                             'mfe_long (max favorable long), mfe_short (max favorable short)')
    parser.add_argument('--feature-cache', type=str, default=None,
                        help='Path to pre-computed feature cache dir (skips 3h feature computation)')
    args = parser.parse_args()

    try:
        results = run_pipeline(args)
        return results
    except Exception as e:
        logger.error(f"PIPELINE FAILED: {e}", exc_info=True)
        sys.stdout.flush()
        raise


if __name__ == '__main__':
    main()
