"""
Genetic Algorithm Formula Discovery — Use gplearn symbolic regression to evolve
new signal formulas from MBO features.

Evolves mathematical expressions that combine raw features into directional signals.
Target: maximize IC vs 10s forward return. Walk-forward to prevent overfitting.

Requirements: pip install gplearn

Usage:
    python alpha_discovery/ga_formula_discovery.py
    python alpha_discovery/ga_formula_discovery.py --n-days 50 --population 2000
"""

import sys
import time
import json
import logging
import argparse
import platform
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

import numpy as np
from scipy.stats import pearsonr

sys.stdout.reconfigure(line_buffering=True)

LVL3_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(LVL3_ROOT))
from alpha_discovery.mbo_features import get_feature_names

# Logging
_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
RESULTS_DIR = LVL3_ROOT / "alpha_discovery" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
_log_file = RESULTS_DIR / f"ga_discovery_{_ts}.log"
_root = logging.getLogger()
_root.setLevel(logging.INFO)
for _h in _root.handlers[:]:
    _root.removeHandler(_h)
_fmt = logging.Formatter('%(asctime)s %(message)s', datefmt='%H:%M:%S')
_fh = logging.FileHandler(str(_log_file), mode='w')
_fh.setFormatter(_fmt)
_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(_fmt)
_root.addHandler(_fh)
_root.addHandler(_sh)
logger = logging.getLogger("ga")

TICK_SIZE = 0.25
BARS_PER_SEC = 10

if platform.system() == 'Windows':
    FEATURE_CACHE = LVL3_ROOT / "data" / "processed" / "mbo_features_cache"
else:
    FEATURE_CACHE = Path.home() / "lvl3quant" / "data" / "processed" / "mbo_features_cache"

OUTPUT_DIR = LVL3_ROOT / "data" / "processed" / "signal_predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_NAMES = get_feature_names()

# Select most promising features for GA (avoid noise from all 340)
# Use features from groups that showed promise + all from groups already in composite
SELECTED_FEATURES = [
    # Group A (already in composite via meta signals): key ones
    'microprice', 'vol_imbalance', 'pressure_imbalance', 'spread',
    # Group B: rolling OFI
    'ofi_5', 'ofi_20', 'trade_imb_5', 'trade_imb_20',
    # Group F: book shape
    'depth_ratio_l1', 'weighted_book_imb', 'spread_change',
    # Group I: MBO enhanced
    'cancel_asym_5', 'cancel_asym_20', 'modify_rate_5', 'aggr_imb_5', 'aggr_imb_20',
    # Group J: spatial
    'bid_L1_orders', 'ask_L1_orders',
    # Group M: rolling book shape
    'cog_asym_5', 'cog_asym_20', 'depth_accel', 'l1_ratio_shift_5',
    # Group N: quote dynamics
    'depletion_momentum_5', 'depletion_dir_20', 'sell_pressure_5', 'institutional_flow_5',
    # Group P: magnitude
    'stacked_depth_asym', 'flow_acceleration', 'cross_flow_agreement', 'institutional_signal',
    # Group Q: flow sequences
    'run_direction_20', 'flow_persistence_20', 'momentum_exhaustion', 'flow_regime_zscore',
    # Group R: deep OFI
    'dwfi_5', 'dwfi_20', 'ofi_l3_5', 'ofi_deep_5', 'ofi_deep_vs_l1',
    # Group S: interactions
    'imb_spread_interaction', 'toxicity_imb_combo', 'multi_signal_strength',
    # Group T: queue dynamics
    'queue_depletion_asymmetry_5', 'queue_depletion_asymmetry_20',
    # Group U: herding
    'burst_same_dir_5', 'trade_persistence_5', 'herding_intensity_5',
    # Group V: momentum
    'momentum_alignment_short', 'momentum_divergence', 'momentum_reversal_50',
    # Group X: size
    'size_imbalance_5', 'institutional_flow_proxy_5', 'order_size_divergence',
]

# Build index map for selected features
NAME_TO_IDX = {name: idx for idx, name in enumerate(ALL_NAMES)}
SELECTED_INDICES = []
SELECTED_NAMES = []
for name in SELECTED_FEATURES:
    if name in NAME_TO_IDX:
        SELECTED_INDICES.append(NAME_TO_IDX[name])
        SELECTED_NAMES.append(name)
    else:
        logger.warning(f"Feature {name} not found in feature names, skipping")


def discover_days() -> List[Tuple[str, Path]]:
    days = []
    for f in sorted(FEATURE_CACHE.glob("*_mbo_features.npz")):
        date_str = f.stem.replace("_mbo_features", "")
        days.append((date_str, f))
    return days


def load_day_subset(fpath: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load one day, returning only selected features + mid prices."""
    data = np.load(str(fpath))
    features = data['mbo_features']
    mid = features[:, 0].copy()
    subset = features[:, SELECTED_INDICES].copy()
    return subset, mid


def subsample_day(features: np.ndarray, mid: np.ndarray, target: np.ndarray,
                  max_bars: int = 50000) -> Tuple[np.ndarray, np.ndarray]:
    """Subsample to max_bars for faster GA evolution."""
    n = len(features)
    if n <= max_bars:
        valid = np.isfinite(target) & np.all(np.isfinite(features), axis=1)
        return features[valid], target[valid]

    step = n // max_bars
    indices = np.arange(0, n, step)
    valid = np.isfinite(target[indices]) & np.all(np.isfinite(features[indices]), axis=1)
    return features[indices][valid], target[indices][valid]


def rolling_zscore_matrix(features: np.ndarray, window: int = 5000) -> np.ndarray:
    """Apply rolling z-score to each column."""
    n, p = features.shape
    result = np.zeros_like(features)
    for col in range(p):
        raw = features[:, col]
        cs = np.cumsum(np.where(np.isfinite(raw), raw, 0.0))
        cs2 = np.cumsum(np.where(np.isfinite(raw), raw ** 2, 0.0))
        for i in range(window, n):
            start = max(0, i - window)
            s = cs[i] - (cs[start] if start > 0 else 0)
            s2 = cs2[i] - (cs2[start] if start > 0 else 0)
            count = window
            mu = s / count
            var = s2 / count - mu ** 2
            if var > 1e-20:
                result[i, col] = (raw[i] - mu) / np.sqrt(var)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-days', type=int, default=0, help='0 = all')
    parser.add_argument('--horizon', type=int, default=100, help='Forward return horizon bars')
    parser.add_argument('--population', type=int, default=1000, help='GA population size')
    parser.add_argument('--generations', type=int, default=20, help='GA generations')
    parser.add_argument('--train-days', type=int, default=30, help='Training window (days)')
    parser.add_argument('--test-days', type=int, default=10, help='Test window (days)')
    parser.add_argument('--max-bars-per-day', type=int, default=30000, help='Max bars per day for speed')
    parser.add_argument('--n-jobs', type=int, default=-1, help='Parallel jobs (-1=all cores)')
    args = parser.parse_args()

    try:
        from gplearn.genetic import SymbolicRegressor
        from gplearn.functions import make_function
    except ImportError:
        logger.error("gplearn not installed! Run: pip install gplearn")
        logger.info("Attempting install...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gplearn"])
        from gplearn.genetic import SymbolicRegressor

    days = discover_days()
    if args.n_days > 0:
        days = days[:args.n_days]

    logger.info(f"GA Formula Discovery")
    logger.info(f"  Features: {len(SELECTED_NAMES)} selected from 340")
    logger.info(f"  Days: {len(days)}")
    logger.info(f"  Population: {args.population}, Generations: {args.generations}")
    logger.info(f"  Walk-forward: train={args.train_days}d, test={args.test_days}d")
    logger.info(f"  Max bars/day: {args.max_bars_per_day}")
    logger.info(f"  Feature names: {SELECTED_NAMES[:10]}...")

    # Walk-forward GA evolution
    all_formulas = []
    fold_results = []
    fold_idx = 0

    for test_start in range(args.train_days, len(days) - args.test_days + 1, args.test_days):
        train_end = test_start
        train_start = max(0, test_start - args.train_days)
        test_end = min(test_start + args.test_days, len(days))

        train_days = days[train_start:train_end]
        test_days_list = days[test_start:test_end]
        fold_idx += 1

        logger.info(f"\n{'='*70}")
        logger.info(f"Fold {fold_idx}: Train [{days[train_start][0]}..{days[train_end-1][0]}] "
                    f"({len(train_days)}d) -> Test [{days[test_start][0]}..{days[test_end-1][0]}] ({len(test_days_list)}d)")

        # Load and concatenate training data
        logger.info("  Loading training data...")
        X_train_parts = []
        y_train_parts = []
        for date_str, fpath in train_days:
            try:
                feats, mid = load_day_subset(fpath)
                # Z-score normalize
                feats_z = rolling_zscore_matrix(feats)
                # Forward return
                fwd_ret = np.full(len(mid), np.nan)
                if len(mid) > args.horizon:
                    fwd_ret[:-args.horizon] = (mid[args.horizon:] - mid[:-args.horizon]) / TICK_SIZE
                # Subsample
                X_sub, y_sub = subsample_day(feats_z, mid, fwd_ret, args.max_bars_per_day)
                X_train_parts.append(X_sub)
                y_train_parts.append(y_sub)
                del feats, mid, feats_z, fwd_ret
            except Exception as e:
                logger.warning(f"    Skip {date_str}: {e}")

        if not X_train_parts:
            logger.warning("  No training data, skip fold")
            continue

        X_train = np.vstack(X_train_parts)
        y_train = np.concatenate(y_train_parts)
        del X_train_parts, y_train_parts

        logger.info(f"  Train size: {X_train.shape[0]:,} bars x {X_train.shape[1]} features")

        # Replace any remaining NaN/inf
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=3.0, neginf=-3.0)
        y_train = np.nan_to_num(y_train, nan=0.0, posinf=10.0, neginf=-10.0)

        # Evolve formulas
        logger.info(f"  Evolving GP formulas (pop={args.population}, gen={args.generations})...")
        t0 = time.time()

        gp = SymbolicRegressor(
            population_size=args.population,
            generations=args.generations,
            tournament_size=20,
            stopping_criteria=0.0,  # don't stop early
            const_range=(-2.0, 2.0),
            init_depth=(2, 6),
            init_method='half and half',
            function_set=['add', 'sub', 'mul', 'div', 'abs', 'neg', 'max', 'min'],
            metric='pearson',  # maximize correlation
            parsimony_coefficient=0.001,  # penalize complexity slightly
            p_crossover=0.7,
            p_subtree_mutation=0.1,
            p_hoist_mutation=0.05,
            p_point_mutation=0.1,
            max_samples=min(0.9, 200000 / max(len(X_train), 1)),
            verbose=0,
            n_jobs=args.n_jobs,
            warm_start=False,
            random_state=fold_idx,
            feature_names=SELECTED_NAMES,
        )

        gp.fit(X_train, y_train)
        train_time = time.time() - t0

        # Get top programs
        best_program = gp._program
        train_pred = gp.predict(X_train)
        train_ic = float(np.corrcoef(train_pred, y_train)[0, 1]) if len(train_pred) > 100 else 0.0

        logger.info(f"  Evolution complete ({train_time:.0f}s)")
        logger.info(f"  Best formula: {best_program}")
        logger.info(f"  Train IC: {train_ic:+.4f}")
        logger.info(f"  Complexity: {best_program.length_} nodes")

        del X_train, y_train

        # Test on holdout
        logger.info("  Testing on holdout...")
        test_ics = []
        for date_str, fpath in test_days_list:
            try:
                feats, mid = load_day_subset(fpath)
                feats_z = rolling_zscore_matrix(feats)
                fwd_ret = np.full(len(mid), np.nan)
                if len(mid) > args.horizon:
                    fwd_ret[:-args.horizon] = (mid[args.horizon:] - mid[:-args.horizon]) / TICK_SIZE

                X_test = np.nan_to_num(feats_z, nan=0.0, posinf=3.0, neginf=-3.0)
                y_test = np.nan_to_num(fwd_ret, nan=0.0, posinf=10.0, neginf=-10.0)

                valid = (y_test != 0.0) & np.isfinite(y_test)
                if valid.sum() > 100:
                    pred = gp.predict(X_test[valid])
                    ic, _ = pearsonr(pred, y_test[valid])
                    test_ics.append(float(ic) if np.isfinite(ic) else 0.0)
                else:
                    test_ics.append(0.0)

                # Save predictions for promising formulas
                if len(test_ics) > 0 and abs(np.mean(test_ics)) > 0.02:
                    full_pred = gp.predict(X_test)
                    out_path = OUTPUT_DIR / f"ga_fold{fold_idx}_{date_str}.npz"
                    np.savez_compressed(str(out_path), predictions=full_pred.astype(np.float32))

                del feats, mid, feats_z, fwd_ret, X_test, y_test
            except Exception as e:
                logger.warning(f"    Test error {date_str}: {e}")
                test_ics.append(0.0)

        mean_test_ic = np.mean(test_ics) if test_ics else 0.0
        logger.info(f"  Test ICs: {[f'{ic:+.3f}' for ic in test_ics]}")
        logger.info(f"  Mean test IC: {mean_test_ic:+.4f}")

        fold_result = {
            'fold': fold_idx,
            'train_dates': [d[0] for d in train_days],
            'test_dates': [d[0] for d in test_days_list],
            'formula': str(best_program),
            'complexity': best_program.length_,
            'train_ic': float(train_ic),
            'test_ics': test_ics,
            'mean_test_ic': float(mean_test_ic),
            'train_time_s': train_time,
        }
        fold_results.append(fold_result)
        all_formulas.append({
            'formula': str(best_program),
            'fold': fold_idx,
            'train_ic': train_ic,
            'test_ic': mean_test_ic,
        })

    # Final summary
    logger.info(f"\n{'='*80}")
    logger.info("GA FORMULA DISCOVERY SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"{'Fold':>5s} {'Train IC':>10s} {'Test IC':>10s} {'Complexity':>11s} {'Formula'}")
    logger.info("-" * 80)

    for r in fold_results:
        logger.info(f"{r['fold']:5d} {r['train_ic']:+10.4f} {r['mean_test_ic']:+10.4f} "
                   f"{r['complexity']:11d} {r['formula'][:60]}")

    # Overall stats
    if fold_results:
        all_test = [r['mean_test_ic'] for r in fold_results]
        logger.info(f"\nOverall: mean_test_IC={np.mean(all_test):+.4f}, "
                   f"std={np.std(all_test):.4f}, "
                   f"pct+={100*np.mean(np.array(all_test)>0):.0f}%")

        # Best formula
        best = max(fold_results, key=lambda x: x['mean_test_ic'])
        logger.info(f"\nBest OOS formula (fold {best['fold']}): {best['formula']}")
        logger.info(f"  Train IC: {best['train_ic']:+.4f}, Test IC: {best['mean_test_ic']:+.4f}")

    # Save
    out_file = RESULTS_DIR / f"ga_discovery_{_ts}.json"
    with open(str(out_file), 'w') as f:
        json.dump({
            'timestamp': _ts,
            'n_features': len(SELECTED_NAMES),
            'feature_names': SELECTED_NAMES,
            'population': args.population,
            'generations': args.generations,
            'horizon_bars': args.horizon,
            'folds': fold_results,
            'all_formulas': all_formulas,
        }, f, indent=2)
    logger.info(f"\nResults saved: {out_file}")


if __name__ == '__main__':
    main()
