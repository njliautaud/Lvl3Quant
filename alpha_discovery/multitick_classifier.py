"""
Multi-Tick Move Classifier — Predict profitable trades, not continuous returns.

THE KEY INSIGHT: IC=0.131 on continuous returns is not profitable because:
- Average move at 5s = 1.46 ticks, but cost = 1.24 ticks RT
- We need to predict WHICH bars will have 3+ tick moves (profitable after costs)
- Only ~15-20% of bars have 3+ tick moves — these are the ones worth trading

This classifier predicts 3 classes:
  0 = FLAT (|move| < threshold ticks — DO NOT TRADE)
  1 = LONG (move > +threshold ticks — BUY)
  2 = SHORT (move < -threshold ticks — SELL)

Walk-forward protocol identical to mbo_alpha_scan.py:
  - Expanding window, 1-day purge gap
  - LightGBM with GPU
  - Per-fold metrics: precision, recall, F1 per class

Success criteria:
  - Precision on LONG/SHORT > 35% (better than base rate ~20%)
  - Capture 3+ tick moves with enough signal to overcome execution costs
  - Combined with magnitude model: trade only when classifier says LONG/SHORT
    AND magnitude model predicts large move

Usage:
    python alpha_discovery/multitick_classifier.py --horizon 5s --threshold 3
    python alpha_discovery/multitick_classifier.py --horizon 10s --threshold 3 --quick
"""

import gc
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from alpha_discovery.mbo_features import (
    compute_mbo_features, get_feature_names, TOTAL_FEATURES, N_GLOBAL_FEATURES,
)


def load_from_cache(cache_dir: str):
    """
    Load per-date snapshot caches. Standalone version for the classifier.
    Returns: (mid_prices, global_features, node_features, timestamps, day_boundaries)
    """
    from pathlib import Path
    cache_path = Path(cache_dir)
    date_files = sorted(cache_path.glob("????-??-??_snapshots.npz"))
    if not date_files:
        raise ValueError(f"No cache files in {cache_dir}. Run rebuild_snapshot_caches.py first.")

    logger.info(f"Loading {len(date_files)} cache files from {cache_dir}")

    all_mid, all_global, all_node, all_ts = [], [], [], []
    day_boundaries = [0]
    total_bars = 0

    for f in date_files:
        data = np.load(str(f))
        mid = data['mid_prices']
        gf = data['global_features']
        nf = data['node_features']
        ts = data.get('timestamps', np.zeros(len(mid), dtype=np.int64))

        n = len(mid)
        all_mid.append(mid)
        all_global.append(gf)
        all_node.append(nf)
        all_ts.append(ts)
        total_bars += n
        day_boundaries.append(total_bars)

    mid_prices = np.concatenate(all_mid)
    global_features = np.concatenate(all_global)
    node_features = np.concatenate(all_node)
    timestamps = np.concatenate(all_ts)

    logger.info(f"Loaded {len(date_files)} days, {total_bars:,} bars, "
                f"global_features: {global_features.shape}")

    return mid_prices, global_features, node_features, timestamps, day_boundaries

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("multitick_classifier")


# ============================================================================
# TARGET CONSTRUCTION
# ============================================================================

def compute_multitick_targets(
    mid_prices: np.ndarray,
    horizon_bars: int,
    threshold_ticks: float = 3.0,
    tick_size: float = 0.25,
) -> np.ndarray:
    """
    Compute 3-class target: FLAT(0), LONG(1), SHORT(2).

    Args:
        mid_prices: (N,) mid prices
        horizon_bars: Forward look (e.g., 50 = 5s at 100ms)
        threshold_ticks: Minimum ticks for LONG/SHORT classification
        tick_size: ES tick size (0.25)

    Returns:
        (N,) int array with 0=FLAT, 1=LONG, 2=SHORT. NaN at end.
    """
    N = len(mid_prices)
    threshold_pts = threshold_ticks * tick_size

    # Forward return
    future_mid = np.empty(N, dtype=np.float64)
    future_mid[:N - horizon_bars] = mid_prices[horizon_bars:]
    future_mid[N - horizon_bars:] = np.nan

    move = future_mid - mid_prices

    targets = np.zeros(N, dtype=np.float32)
    targets[move > threshold_pts] = 1.0    # LONG
    targets[move < -threshold_pts] = 2.0   # SHORT
    targets[np.isnan(move)] = np.nan

    return targets


def compute_magnitude_target(
    mid_prices: np.ndarray,
    horizon_bars: int,
    tick_size: float = 0.25,
) -> np.ndarray:
    """
    Compute magnitude target: absolute move in ticks.
    This is the MAGNITUDE prediction — how big will the move be?
    """
    N = len(mid_prices)
    future_mid = np.empty(N, dtype=np.float64)
    future_mid[:N - horizon_bars] = mid_prices[horizon_bars:]
    future_mid[N - horizon_bars:] = np.nan

    abs_move_ticks = np.abs(future_mid - mid_prices) / tick_size
    return abs_move_ticks


# ============================================================================
# WALK-FORWARD CLASSIFIER
# ============================================================================

class MultiTickWalkForward:
    """Walk-forward LightGBM classifier for multi-tick moves."""

    def __init__(
        self,
        horizon_bars: int = 50,
        threshold_ticks: float = 3.0,
        tick_size: float = 0.25,
        min_train_days: int = 20,
        purge_days: int = 1,
        lgb_params: Optional[dict] = None,
    ):
        self.horizon_bars = horizon_bars
        self.threshold_ticks = threshold_ticks
        self.tick_size = tick_size
        self.min_train_days = min_train_days
        self.purge_days = purge_days

        self.lgb_params = lgb_params or {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'n_estimators': 300,
            'max_depth': 5,
            'learning_rate': 0.05,
            'min_child_samples': 500,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'verbose': -1,
            'n_jobs': -1,
        }

        # Try GPU
        try:
            import lightgbm as lgb
            test_params = {**self.lgb_params, 'device': 'gpu', 'n_estimators': 2}
            d = lgb.Dataset(np.random.randn(100, 5), label=np.random.randint(0, 3, 100))
            lgb.train(test_params, d, num_boost_round=2)
            self.lgb_params['device'] = 'gpu'
            logger.info("GPU acceleration enabled")
        except Exception:
            logger.info("Using CPU (GPU not available)")

    def run(
        self,
        features: np.ndarray,
        mid_prices: np.ndarray,
        day_boundaries: List[int],
        feature_names: List[str],
        magnitude_target: bool = True,
    ) -> Dict:
        """
        Run walk-forward classification + optional magnitude regression.

        Returns dict with per-fold metrics and aggregate results.
        """
        import lightgbm as lgb

        N = features.shape[0]
        n_days = len(day_boundaries) - 1

        # Compute targets
        class_targets = compute_multitick_targets(
            mid_prices, self.horizon_bars, self.threshold_ticks, self.tick_size
        )
        mag_targets = compute_magnitude_target(
            mid_prices, self.horizon_bars, self.tick_size
        ) if magnitude_target else None

        # Class distribution
        valid = ~np.isnan(class_targets)
        n_flat = np.sum(class_targets[valid] == 0)
        n_long = np.sum(class_targets[valid] == 1)
        n_short = np.sum(class_targets[valid] == 2)
        n_total = np.sum(valid)
        logger.info(f"Class distribution: FLAT={n_flat/n_total:.1%}, "
                     f"LONG={n_long/n_total:.1%}, SHORT={n_short/n_total:.1%}")
        logger.info(f"Threshold: {self.threshold_ticks} ticks = "
                     f"{self.threshold_ticks * self.tick_size} pts")

        fold_results = []
        all_class_preds = np.full(N, np.nan)
        all_class_probs = np.full((N, 3), np.nan)
        all_mag_preds = np.full(N, np.nan) if magnitude_target else None

        for test_day in range(self.min_train_days + self.purge_days, n_days):
            test_start = day_boundaries[test_day]
            test_end = day_boundaries[test_day + 1] if test_day + 1 < len(day_boundaries) else N

            # Training: all days before purge gap
            train_end_day = test_day - self.purge_days
            train_end = day_boundaries[train_end_day]
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)

            # Filter valid samples
            train_valid = train_idx[~np.isnan(class_targets[train_idx])]
            test_valid = test_idx[~np.isnan(class_targets[test_idx])]

            if len(train_valid) < 1000 or len(test_valid) < 100:
                continue

            X_train = features[train_valid]
            y_train = class_targets[train_valid].astype(int)
            X_test = features[test_valid]
            y_test = class_targets[test_valid].astype(int)

            # Train classifier
            train_data = lgb.Dataset(X_train, label=y_train)
            model = lgb.train(
                self.lgb_params,
                train_data,
                num_boost_round=self.lgb_params.get('n_estimators', 300),
            )

            # Predict class probabilities
            probs = model.predict(X_test)  # (N_test, 3)
            preds = np.argmax(probs, axis=1)

            all_class_preds[test_valid] = preds
            all_class_probs[test_valid] = probs

            # === Magnitude regression (separate model) ===
            if magnitude_target and mag_targets is not None:
                mag_valid_train = train_idx[~np.isnan(mag_targets[train_idx])]
                mag_valid_test = test_idx[~np.isnan(mag_targets[test_idx])]

                if len(mag_valid_train) > 1000:
                    mag_params = {
                        'objective': 'regression',
                        'metric': 'mae',
                        'n_estimators': 200,
                        'max_depth': 5,
                        'learning_rate': 0.05,
                        'min_child_samples': 500,
                        'subsample': 0.8,
                        'verbose': -1,
                        'n_jobs': -1,
                    }
                    if 'device' in self.lgb_params:
                        mag_params['device'] = self.lgb_params['device']

                    mag_train_data = lgb.Dataset(
                        features[mag_valid_train],
                        label=mag_targets[mag_valid_train]
                    )
                    mag_model = lgb.train(
                        mag_params, mag_train_data,
                        num_boost_round=mag_params['n_estimators'],
                    )
                    mag_preds = mag_model.predict(features[mag_valid_test])
                    all_mag_preds[mag_valid_test] = mag_preds

            # Per-fold metrics
            fold_metrics = self._compute_fold_metrics(y_test, preds, probs)
            fold_metrics['fold'] = test_day
            fold_metrics['n_train'] = len(train_valid)
            fold_metrics['n_test'] = len(test_valid)

            if test_day % 10 == 0 or test_day == self.min_train_days + self.purge_days:
                logger.info(
                    f"  Fold {test_day}: accuracy={fold_metrics['accuracy']:.3f}, "
                    f"precision_long={fold_metrics.get('precision_1', 0):.3f}, "
                    f"precision_short={fold_metrics.get('precision_2', 0):.3f}, "
                    f"recall_long={fold_metrics.get('recall_1', 0):.3f}"
                )

            fold_results.append(fold_metrics)
            del model, train_data
            gc.collect()

        # Aggregate results
        results = self._aggregate_results(
            fold_results, class_targets, all_class_preds, all_class_probs,
            mag_targets, all_mag_preds, mid_prices, feature_names, features
        )

        return results

    def _compute_fold_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
    ) -> Dict:
        """Compute per-fold classification metrics."""
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
        }

        for cls in [0, 1, 2]:
            cls_mask = y_true == cls
            pred_mask = y_pred == cls

            tp = np.sum(cls_mask & pred_mask)
            fp = np.sum(~cls_mask & pred_mask)
            fn = np.sum(cls_mask & ~pred_mask)

            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            f1 = 2 * precision * recall / max(1e-8, precision + recall)

            metrics[f'precision_{cls}'] = float(precision)
            metrics[f'recall_{cls}'] = float(recall)
            metrics[f'f1_{cls}'] = float(f1)
            metrics[f'n_pred_{cls}'] = int(np.sum(pred_mask))
            metrics[f'n_true_{cls}'] = int(np.sum(cls_mask))

        # Expected profit metric: for LONG/SHORT predictions, what's the average
        # direction-aligned move in ticks?
        # This is the actual money metric.
        return metrics

    def _aggregate_results(
        self,
        fold_results: List[Dict],
        class_targets: np.ndarray,
        class_preds: np.ndarray,
        class_probs: np.ndarray,
        mag_targets: Optional[np.ndarray],
        mag_preds: Optional[np.ndarray],
        mid_prices: np.ndarray,
        feature_names: List[str],
        features: np.ndarray,
    ) -> Dict:
        """Aggregate all fold results into final report."""
        if not fold_results:
            return {'error': 'No folds completed'}

        # Average metrics across folds
        avg = {}
        for key in fold_results[0]:
            if isinstance(fold_results[0][key], (int, float)):
                vals = [f[key] for f in fold_results]
                avg[f'mean_{key}'] = float(np.mean(vals))
                avg[f'std_{key}'] = float(np.std(vals))

        # === THE MONEY METRIC ===
        # For each bar where we predict LONG or SHORT:
        # What is the ACTUAL forward move in ticks?
        valid = ~np.isnan(class_preds) & ~np.isnan(class_targets)
        future_mid = np.empty_like(mid_prices)
        future_mid[:len(mid_prices) - self.horizon_bars] = mid_prices[self.horizon_bars:]
        future_mid[len(mid_prices) - self.horizon_bars:] = np.nan
        actual_move_ticks = (future_mid - mid_prices) / self.tick_size

        long_mask = valid & (class_preds == 1)
        short_mask = valid & (class_preds == 2)
        flat_mask = valid & (class_preds == 0)

        # Signed P&L in ticks per trade
        long_pnl = actual_move_ticks[long_mask]  # Buy → profit if move > 0
        short_pnl = -actual_move_ticks[short_mask]  # Sell → profit if move < 0
        all_trade_pnl = np.concatenate([long_pnl, short_pnl]) if len(long_pnl) + len(short_pnl) > 0 else np.array([])

        cost_ticks = 0.376  # RT commission only ($4.70 RT, HC #52)
        net_pnl = all_trade_pnl - cost_ticks

        money = {
            'n_long_predictions': int(np.sum(long_mask)),
            'n_short_predictions': int(np.sum(short_mask)),
            'n_flat_predictions': int(np.sum(flat_mask)),
            'trade_rate': float((np.sum(long_mask) + np.sum(short_mask)) / max(1, np.sum(valid))),
        }

        if len(all_trade_pnl) > 0:
            money['mean_gross_pnl_ticks'] = float(np.mean(all_trade_pnl))
            money['mean_net_pnl_ticks'] = float(np.mean(net_pnl))
            money['median_net_pnl_ticks'] = float(np.median(net_pnl))
            money['win_rate'] = float(np.mean(net_pnl > 0))
            money['profit_factor'] = float(
                np.sum(net_pnl[net_pnl > 0]) / max(0.01, -np.sum(net_pnl[net_pnl < 0]))
            )
            money['total_net_ticks'] = float(np.sum(net_pnl))
            money['n_trades'] = len(all_trade_pnl)
            money['sharpe'] = float(
                np.mean(net_pnl) / max(1e-8, np.std(net_pnl)) * np.sqrt(252 * 23400)
            )

        if len(long_pnl) > 0:
            money['long_mean_gross'] = float(np.mean(long_pnl))
            money['long_win_rate'] = float(np.mean(long_pnl > cost_ticks))
        if len(short_pnl) > 0:
            money['short_mean_gross'] = float(np.mean(short_pnl))
            money['short_win_rate'] = float(np.mean(short_pnl > cost_ticks))

        # === MAGNITUDE MODEL METRICS ===
        mag_metrics = {}
        if mag_targets is not None and mag_preds is not None:
            mag_valid = ~np.isnan(mag_preds) & ~np.isnan(mag_targets)
            if np.sum(mag_valid) > 100:
                ic, _ = spearmanr(mag_preds[mag_valid], mag_targets[mag_valid])
                mag_metrics['magnitude_ic'] = float(ic)
                mag_metrics['magnitude_mae'] = float(
                    np.mean(np.abs(mag_preds[mag_valid] - mag_targets[mag_valid]))
                )

                # Combined strategy: trade only when classifier says LONG/SHORT
                # AND magnitude model predicts large move (top 30%)
                trade_mask = valid & ((class_preds == 1) | (class_preds == 2))
                mag_threshold = np.percentile(mag_preds[~np.isnan(mag_preds)], 70)

                high_mag_mask = trade_mask & (~np.isnan(mag_preds)) & (mag_preds > mag_threshold)
                if np.sum(high_mag_mask) > 0:
                    high_mag_long = high_mag_mask & (class_preds == 1)
                    high_mag_short = high_mag_mask & (class_preds == 2)
                    hm_long_pnl = actual_move_ticks[high_mag_long]
                    hm_short_pnl = -actual_move_ticks[high_mag_short]
                    hm_all = np.concatenate([hm_long_pnl, hm_short_pnl])
                    hm_net = hm_all - cost_ticks

                    mag_metrics['combined_n_trades'] = len(hm_all)
                    mag_metrics['combined_mean_gross'] = float(np.mean(hm_all))
                    mag_metrics['combined_mean_net'] = float(np.mean(hm_net))
                    mag_metrics['combined_win_rate'] = float(np.mean(hm_net > 0))
                    mag_metrics['combined_trade_rate'] = float(
                        np.sum(high_mag_mask) / max(1, np.sum(valid))
                    )

        # Feature importance (from last fold's model — approximate)
        return {
            'config': {
                'horizon_bars': self.horizon_bars,
                'threshold_ticks': self.threshold_ticks,
                'tick_size': self.tick_size,
                'n_features': features.shape[1] if features is not None else 0,
                'n_folds': len(fold_results),
            },
            'class_distribution': {
                'flat_rate': float(np.mean(class_targets[~np.isnan(class_targets)] == 0)),
                'long_rate': float(np.mean(class_targets[~np.isnan(class_targets)] == 1)),
                'short_rate': float(np.mean(class_targets[~np.isnan(class_targets)] == 2)),
            },
            'classification_metrics': avg,
            'money_metrics': money,
            'magnitude_metrics': mag_metrics,
            'fold_results': fold_results,
        }


# ============================================================================
# MAIN
# ============================================================================

HORIZON_MAP = {
    '3s': 30, '5s': 50, '10s': 100, '30s': 300, '1m': 600,
}


def main():
    parser = argparse.ArgumentParser(description="Multi-Tick Move Classifier")
    parser.add_argument('--horizon', default='5s', choices=HORIZON_MAP.keys())
    parser.add_argument('--threshold', type=float, default=3.0,
                        help='Min ticks for LONG/SHORT (default: 3)')
    parser.add_argument('--quick', action='store_true',
                        help='Use only first 30 days for quick test')
    parser.add_argument('--no-magnitude', action='store_true',
                        help='Skip magnitude regression model')
    parser.add_argument('--thresholds', type=str, default=None,
                        help='Comma-separated thresholds to sweep (e.g., "2,3,4,5")')
    args = parser.parse_args()

    horizon_bars = HORIZON_MAP[args.horizon]

    logger.info("=" * 70)
    logger.info("MULTI-TICK MOVE CLASSIFIER")
    logger.info(f"  Horizon: {args.horizon} ({horizon_bars} bars)")
    logger.info(f"  Threshold: {args.threshold} ticks")
    logger.info(f"  Features: {TOTAL_FEATURES}")
    logger.info("=" * 70)

    # Load data
    logger.info("Loading cached data...")
    cache_dir = ROOT / "data" / "processed" / "medium_snapshots_cache"
    mid_prices, global_features, node_features, timestamps, day_boundaries = load_from_cache(
        str(cache_dir)
    )

    n_days = len(day_boundaries) - 1
    if args.quick and n_days > 30:
        cutoff = day_boundaries[30]
        mid_prices = mid_prices[:cutoff]
        global_features = global_features[:cutoff]
        node_features = node_features[:cutoff]
        timestamps = timestamps[:cutoff]
        day_boundaries = day_boundaries[:31]
        n_days = 30
        logger.info(f"Quick mode: using first 30 days ({cutoff:,} bars)")

    logger.info(f"Data: {len(mid_prices):,} bars, {n_days} days")

    # Compute features
    logger.info("Computing features...")
    t0 = time.time()
    features = compute_mbo_features(
        mid_prices, global_features, node_features,
        day_boundaries=day_boundaries,
    )
    logger.info(f"Features computed in {time.time()-t0:.1f}s: {features.shape}")

    feature_names = get_feature_names()

    # Threshold sweep or single run
    thresholds = [args.threshold]
    if args.thresholds:
        thresholds = [float(t) for t in args.thresholds.split(',')]

    all_results = {}
    for threshold in thresholds:
        logger.info(f"\n{'='*70}")
        logger.info(f"THRESHOLD: {threshold} ticks = {threshold * 0.25} pts")
        logger.info(f"{'='*70}")

        classifier = MultiTickWalkForward(
            horizon_bars=horizon_bars,
            threshold_ticks=threshold,
            min_train_days=20,
            purge_days=1,
        )

        results = classifier.run(
            features, mid_prices, day_boundaries, feature_names,
            magnitude_target=not args.no_magnitude,
        )

        all_results[f'threshold_{threshold}'] = results

        # Print summary
        money = results.get('money_metrics', {})
        mag = results.get('magnitude_metrics', {})
        cls_dist = results.get('class_distribution', {})

        logger.info(f"\n--- RESULTS ({threshold} tick threshold) ---")
        logger.info(f"Class dist: FLAT={cls_dist.get('flat_rate',0):.1%}, "
                     f"LONG={cls_dist.get('long_rate',0):.1%}, "
                     f"SHORT={cls_dist.get('short_rate',0):.1%}")
        logger.info(f"Trade rate: {money.get('trade_rate',0):.1%}")
        logger.info(f"Mean gross P&L: {money.get('mean_gross_pnl_ticks',0):.3f} ticks")
        logger.info(f"Mean net P&L:   {money.get('mean_net_pnl_ticks',0):.3f} ticks")
        logger.info(f"Win rate:       {money.get('win_rate',0):.1%}")
        logger.info(f"Profit factor:  {money.get('profit_factor',0):.2f}")
        logger.info(f"Total net:      {money.get('total_net_ticks',0):.0f} ticks")

        if mag:
            logger.info(f"\nMagnitude IC: {mag.get('magnitude_ic',0):.4f}")
            logger.info(f"Combined (direction+magnitude) strategy:")
            logger.info(f"  Trades: {mag.get('combined_n_trades',0)}")
            logger.info(f"  Mean net: {mag.get('combined_mean_net',0):.3f} ticks")
            logger.info(f"  Win rate: {mag.get('combined_win_rate',0):.1%}")
            logger.info(f"  Trade rate: {mag.get('combined_trade_rate',0):.1%}")

    # Save results
    out_dir = ROOT / "alpha_discovery" / "results"
    out_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"multitick_classifier_{args.horizon}_{ts}.json"

    # Make JSON-serializable
    def make_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    with open(out_file, 'w') as f:
        json.dump(make_serializable(all_results), f, indent=2)
    logger.info(f"\nResults saved to: {out_file}")


if __name__ == "__main__":
    main()
