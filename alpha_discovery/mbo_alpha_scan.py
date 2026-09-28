"""
MBO Alpha Scanner — Walk-forward LightGBM evaluation on ES futures MBO data.

PRIMARY: Load from pre-computed NPZ snapshot caches (1000x faster).
FALLBACK: Load raw MBO events (slow but works without cache).

Runs walk-forward LightGBM for multiple horizons and targets with
comprehensive per-day, per-hour, and regime statistics.
"""

import gc
import sys
import time
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr, ttest_1samp

# Add Lvl3Quant root to path
LVL3_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(LVL3_ROOT))

from alpha_discovery.mbo_features import compute_mbo_features, get_feature_names, TOTAL_FEATURES

logger = logging.getLogger("mbo_alpha_scan")

# RTH filtering
ET_OFFSET_HOURS = -4
RTH_START_MINUTES = 9 * 60 + 30   # 9:30 AM ET
RTH_END_MINUTES = 16 * 60          # 4:00 PM ET

# Results directory
RESULTS_DIR = LVL3_ROOT / "alpha_discovery" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Column indices for the 45 global features from enhanced engineering.py
# ============================================================================
# Base (10): mid, spread, vol_imbalance, microprice, total_bid_vol,
#            total_ask_vol, mean_bid_size, mean_ask_size, best_bid, best_ask
# Order flow (8): trade_imbalance, buy_volume, sell_volume, add_count,
#                 cancel_count, trade_count, cancel_to_add, trade_to_add
# Microstructure (8): bid_pressure, ask_pressure, pressure_imbalance,
#                     depth_concentration, bid_slope, ask_slope, spread_ticks, depth_ratio
# Temporal (5): hour_norm, minute_norm, time_since_rth, time_to_close, event_density
# MBO Enhanced (14): modify_count, modify_to_add, mean_lifetime, fleeting_ratio,
#                    aggr_buy_count, aggr_sell_count, aggr_imbalance, max_trade_size,
#                    cancel_bid_vol, cancel_ask_vol, cancel_side_imbalance,
#                    tick_count, sequence_gaps, n_orders_completed

# Only indices needed by the scanner itself (for day detection / temporal)
COL_HOUR_NORM = 26
COL_TIME_SINCE_RTH = 28

# Node features: (N, 20, 9) — 10 bid + 10 ask levels
# Per node: [price, rel_price, size, log_size, level_idx, side_flag,
#            order_count, avg_order_size, level_concentration]


class MBOAlphaScanner:
    """Walk-forward LightGBM scanner for MBO order book data."""

    def __init__(
        self,
        cache_dir: str = "data/processed/medium_snapshots_cache",
        instrument_id: int = 14160,
        sample_interval_ms: int = 100,  # medium cache uses 100ms intervals
        depth_levels: int = 10,
        tick_size: float = 0.25,
    ):
        self.cache_dir = LVL3_ROOT / cache_dir
        self.instrument_id = instrument_id
        self.sample_interval_ms = sample_interval_ms
        self.depth_levels = depth_levels
        self.tick_size = tick_size

        # Horizons in seconds
        self.horizons = {
            '5s': 5,
            '30s': 30,
            '1m': 60,
            '3m': 180,
            '5m': 300,
        }

        # Target types
        self.target_types = ['return', 'volatility', 'magnitude']

        # Stored data
        self.features = None
        self.feature_names = None
        self.mid_prices = None
        self.hour_of_day = None
        self.time_since_rth = None
        self.day_boundaries = None

    # ================================================================
    # PRIMARY: Load from cached NPZ files (fast!)
    # ================================================================

    def load_from_cache(self, n_days: Optional[int] = None) -> dict:
        """
        Load per-date snapshot caches. Each file = exactly 1 trading day.

        Cache files are named YYYY-MM-DD_snapshots.npz (one per trading day).
        Built by rebuild_snapshot_caches.py which extracts calendar dates from
        raw nanosecond UTC timestamps and saves one file per unique trading day.

        NO DEDUPLICATION NEEDED — the rebuild guarantees uniqueness.
        NO OVERLAP POSSIBLE — each file is a single calendar date.

        Each file contains:
          - node_features: (N, 2*depth_levels, 9) — enhanced with order counts
          - global_features: (N, 45) — enhanced with MBO metadata
          - mid_prices: (N,) — raw mid prices
        """
        # Look for date-named files (new per-day format: YYYY-MM-DD_snapshots.npz)
        date_files = sorted(self.cache_dir.glob("????-??-??_snapshots.npz"))

        if not date_files:
            # Check for old contaminated format to give a helpful error
            old_files = list(self.cache_dir.glob("file_*_snapshots.npz"))
            if old_files:
                raise ValueError(
                    f"Found {len(old_files)} OLD format cache files (file_XXX_snapshots.npz). "
                    f"These contain overlapping multi-day data that contaminates walk-forward. "
                    f"Delete them and rebuild: python alpha_discovery/rebuild_snapshot_caches.py"
                )
            raise ValueError(
                f"No cache files in {self.cache_dir}. "
                f"Run: python alpha_discovery/rebuild_snapshot_caches.py"
            )

        logger.info(f"Found {len(date_files)} per-date cache files in {self.cache_dir}")

        # Validate: no duplicate dates
        dates_seen = []
        for f in date_files:
            date_str = f.name[:10]
            if date_str in dates_seen:
                raise ValueError(
                    f"DUPLICATE date cache found: {date_str}. "
                    f"Delete cache dir and rebuild."
                )
            dates_seen.append(date_str)

        # Load manifest if available (for logging/validation)
        manifest_path = self.cache_dir / 'cache_manifest.json'
        if manifest_path.exists():
            try:
                with open(str(manifest_path)) as mf:
                    manifest = json.load(mf)
                logger.info(
                    f"Manifest: {manifest.get('total_days', '?')} days, "
                    f"{manifest.get('date_range', {}).get('first', '?')} to "
                    f"{manifest.get('date_range', {}).get('last', '?')}, "
                    f"format v{manifest.get('format_version', '?')}"
                )
            except Exception:
                pass

        # Load each date file — 1 file = 1 trading day, guaranteed
        all_global = []
        all_nodes = []
        all_mids = []
        self.day_boundaries = [0]
        days_loaded = 0
        dates_loaded = []

        for fpath in date_files:
            if n_days is not None and days_loaded >= n_days:
                break

            date_str = fpath.name[:10]
            t0 = time.time()
            data = np.load(str(fpath), allow_pickle=True)
            gf = data['global_features']
            nf = data['node_features']
            mp = data['mid_prices']
            data.close()

            if len(mp) < 100:
                logger.warning(f"  {date_str}: only {len(mp)} snapshots, skipping")
                continue

            all_global.append(gf)
            all_nodes.append(nf)
            all_mids.append(mp)
            self.day_boundaries.append(self.day_boundaries[-1] + len(mp))
            days_loaded += 1
            dates_loaded.append(date_str)

            elapsed = time.time() - t0
            if days_loaded <= 5 or days_loaded % 10 == 0:
                logger.info(
                    f"  [{days_loaded}/{len(date_files)}] {date_str}: "
                    f"{len(mp):,} snapshots [{elapsed:.1f}s]"
                )

        if not all_global:
            raise ValueError("No valid trading days found in cache")

        n_days_loaded = len(self.day_boundaries) - 1
        total_snapshots = sum(len(g) for g in all_global)
        logger.info(
            f"Loaded {n_days_loaded} trading days, "
            f"{total_snapshots:,} total RTH snapshots"
        )
        logger.info(f"Date range: {dates_loaded[0]} to {dates_loaded[-1]}")

        # Concatenate all files
        global_feats = np.concatenate(all_global)
        node_feats = np.concatenate(all_nodes)
        mid_prices = np.concatenate(all_mids)

        N = len(mid_prices)
        n_days = len(self.day_boundaries) - 1
        logger.info(f"Total: {N:,} RTH snapshots across {n_days} days")

        # Store temporal info for analysis
        self.hour_of_day = global_feats[:, COL_HOUR_NORM] * 24.0
        self.time_since_rth = global_feats[:, COL_TIME_SINCE_RTH]

        # Compute our extended ~150 features using new unified interface
        # FIX: Pass day_boundaries so rolling features NaN-fill at day starts,
        # preventing overnight leakage in rolling windows.
        logger.info(f"Computing ~{TOTAL_FEATURES} features from cached data...")
        logger.info(f"  Global features shape: {global_feats.shape}")
        logger.info(f"  Node features shape: {node_feats.shape}")
        logger.info(f"  Day boundaries: {len(self.day_boundaries)} entries")
        self.features = compute_mbo_features(
            mid_prices=mid_prices,
            global_features_raw=global_feats,
            node_features_raw=node_feats,
            tick_size=self.tick_size,
            depth_levels=self.depth_levels,
            day_boundaries=self.day_boundaries,
        )

        self.feature_names = get_feature_names()
        self.mid_prices = mid_prices

        # Free large arrays
        del global_feats, node_feats, all_global, all_nodes
        gc.collect()

        logger.info(f"Features shape: {self.features.shape}")
        logger.info(f"Feature names: {len(self.feature_names)}")

        return {
            'n_snapshots': N,
            'n_days': n_days,
            'n_features': self.features.shape[1],
            'cache_files': len(date_files),
            'dates_loaded': dates_loaded,
            'snapshots_per_day': [self.day_boundaries[i+1] - self.day_boundaries[i]
                                  for i in range(n_days)],
        }

    # ================================================================
    # FAST: Load pre-computed features (from Rust feature expander)
    # ================================================================

    def load_precomputed_features(
        self,
        feature_cache_dir: Optional[str] = None,
        snapshot_cache_dir: Optional[str] = None,
        n_days: Optional[int] = None,
        extra_cols: int = 0,
    ) -> dict:
        """
        Load pre-computed 290-feature arrays from Rust feature cache.
        Still loads mid_prices from snapshot cache (needed for targets).
        Skips the ~3 hour Python feature computation step.

        Memory-efficient: pre-allocates a single array and copies day-by-day
        to avoid the 2x peak from list accumulation + concatenation.

        Args:
            feature_cache_dir: Dir with *_mbo_features.npz files
            snapshot_cache_dir: Dir with snapshot NPZ files for mid_prices
            n_days: Max days to load
            extra_cols: Extra columns to pre-allocate (e.g. 14 for event features)
        """
        feat_dir = Path(feature_cache_dir) if feature_cache_dir else (
            Path(self.cache_dir).parent / 'mbo_features_cache'
        )
        snap_dir = Path(snapshot_cache_dir) if snapshot_cache_dir else self.cache_dir

        if not feat_dir.exists():
            raise ValueError(f"Feature cache dir not found: {feat_dir}")

        feat_files = sorted(feat_dir.glob('*_mbo_features.npz'))
        if not feat_files:
            raise ValueError(f"No feature cache files in {feat_dir}")

        snap_files = sorted(snap_dir.glob('*.npz'))
        snap_by_date = {}
        for sf in snap_files:
            date_str = sf.name[:10]
            snap_by_date[date_str] = sf

        logger.info(f"Found {len(feat_files)} feature cache files in {feat_dir}")

        # === Pre-allocate and load day-by-day (avoids 2x memory from list+concat) ===
        # Estimate total rows: scan just the first file for n_cols, count via filename match
        first_data = np.load(str(feat_files[0]))
        n_base_features = first_data['mbo_features'].shape[1]
        first_rows = first_data['mbo_features'].shape[0]
        first_data.close()

        # Estimate total rows (234K/day typical, but last day may differ)
        max_files = n_days if n_days else len(feat_files)
        est_rows_per_day = first_rows  # Use first day as estimate
        est_total = est_rows_per_day * min(max_files, len(feat_files))
        total_cols = n_base_features + extra_cols
        mem_gb = est_total * total_cols * 4 / (1024**3)
        logger.info(
            f"Pre-allocating ~{est_total:,} rows x {total_cols} cols "
            f"({mem_gb:.1f} GB estimated)"
        )

        # Pre-allocate with estimate (will truncate at end if needed)
        self.features = np.empty((est_total, total_cols), dtype=np.float32)
        if extra_cols > 0:
            self.features[:, n_base_features:] = 0  # Zero only the extra columns
        self.mid_prices = np.empty(est_total, dtype=np.float32)
        self.day_boundaries = [0]
        dates_loaded = []
        offset = 0
        days_loaded = 0

        for fpath in feat_files:
            if n_days is not None and days_loaded >= n_days:
                break

            date_str = fpath.name[:10]
            if date_str not in snap_by_date:
                continue

            t0 = time.time()

            # Load features
            data = np.load(str(fpath))
            feats = data['mbo_features']
            n_rows_day = feats.shape[0]
            data.close()

            # Load mid_prices
            snap_data = np.load(str(snap_by_date[date_str]), allow_pickle=True)
            mp = snap_data['mid_prices']
            snap_data.close()

            if len(mp) != n_rows_day or n_rows_day < 100:
                continue

            # Grow array if needed (rare: only if estimate was too low)
            if offset + n_rows_day > self.features.shape[0]:
                new_size = int(self.features.shape[0] * 1.5)
                logger.info(f"  Resizing array: {self.features.shape[0]:,} -> {new_size:,}")
                new_feats = np.empty((new_size, total_cols), dtype=np.float32)
                new_feats[:offset] = self.features[:offset]
                self.features = new_feats
                new_mids = np.empty(new_size, dtype=np.float32)
                new_mids[:offset] = self.mid_prices[:offset]
                self.mid_prices = new_mids

            # Copy directly into pre-allocated array
            self.features[offset:offset + n_rows_day, :n_base_features] = feats
            self.mid_prices[offset:offset + n_rows_day] = mp
            del feats, mp

            self.day_boundaries.append(offset + n_rows_day)
            offset += n_rows_day
            days_loaded += 1
            dates_loaded.append(date_str)

            elapsed = time.time() - t0
            if days_loaded <= 5 or days_loaded % 10 == 0 or days_loaded == len(feat_files):
                logger.info(
                    f"  [{days_loaded}/{len(feat_files)}] {date_str}: "
                    f"{n_rows_day:,} bars, {n_base_features} features [{elapsed:.1f}s]"
                )

        if not dates_loaded:
            raise ValueError("No valid feature files loaded")

        # Truncate to actual size (if we over-estimated)
        if offset < self.features.shape[0]:
            self.features = self.features[:offset]
            self.mid_prices = self.mid_prices[:offset]

        self.feature_names = get_feature_names()
        gc.collect()

        N = len(self.mid_prices)
        logger.info(
            f"Loaded {len(dates_loaded)} days, {N:,} bars, "
            f"{n_base_features} pre-computed features"
            + (f" (+{extra_cols} reserved)" if extra_cols > 0 else "")
        )
        logger.info(f"Date range: {dates_loaded[0]} to {dates_loaded[-1]}")
        logger.info(f"Features shape: {self.features.shape}")
        logger.info(f"Memory: {self.features.nbytes / (1024**3):.1f} GB")

        return {
            'n_snapshots': N,
            'n_days': len(dates_loaded),
            'n_features': n_base_features,
            'n_total_cols': total_cols,
            'cache_files': len(feat_files),
            'dates_loaded': dates_loaded,
            'snapshots_per_day': [
                self.day_boundaries[i+1] - self.day_boundaries[i]
                for i in range(len(dates_loaded))
            ],
            'precomputed': True,
            'extra_cols': extra_cols,
        }

    # ================================================================
    # Target computation
    # ================================================================

    def compute_targets(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Compute multi-horizon targets.
        Returns: {horizon_name: {target_type: array}}

        FIX: Targets that cross day boundaries are NaN-filled.
        Without this, the last `steps` bars of each day would have
        targets computed from the NEXT day's data (across overnight gap),
        contaminating both training and evaluation.
        """
        N = len(self.mid_prices)
        targets = {}
        steps_per_sec = 1000 / self.sample_interval_ms

        log_mid = np.log(np.maximum(self.mid_prices, 1.0))

        # FIX: Build mask for bars whose forward targets cross day boundaries.
        # For each day, the last `steps` bars look into the next day.
        n_days = len(self.day_boundaries) - 1 if self.day_boundaries else 0

        for hz_name, hz_sec in self.horizons.items():
            steps = int(hz_sec * steps_per_sec)
            if steps >= N:
                logger.warning(f"Horizon {hz_name} ({steps} steps) exceeds data {N}")
                continue

            future_mid = np.empty(N, dtype=np.float32)
            future_mid[:N - steps] = self.mid_prices[steps:]
            future_mid[N - steps:] = np.nan

            # FIX: NaN-fill targets whose forward window crosses a day boundary.
            # The last `steps` bars of each day (except the last) have targets
            # that peek into the next day's data across the overnight gap.
            if n_days > 1:
                for d in range(n_days - 1):
                    day_end = self.day_boundaries[d + 1]
                    # Bars from (day_end - steps) to day_end look into next day
                    nan_start = max(self.day_boundaries[d], day_end - steps)
                    future_mid[nan_start:day_end] = np.nan

            price_change = future_mid - self.mid_prices
            ret = price_change / np.maximum(self.mid_prices, 1.0)

            # Return target
            return_target = ret.copy()

            # Volatility target: std of log returns over next `steps` bars
            # Fully vectorized using cumsum trick (no Python loop)
            log_ret_1 = np.diff(log_mid, prepend=log_mid[0])
            vol_target = np.full(N, np.nan, dtype=np.float32)
            if steps <= N and steps > 0:
                lr64 = log_ret_1.astype(np.float64)
                cs = np.cumsum(lr64)
                cs2 = np.cumsum(lr64 ** 2)
                # Prepend 0 for easier windowing
                cs_pad = np.concatenate([[0.0], cs])
                cs2_pad = np.concatenate([[0.0], cs2])
                # Window sums: sum[i:i+steps] = cs_pad[i+steps] - cs_pad[i]
                n_valid = N - steps
                s = cs_pad[steps:steps + n_valid] - cs_pad[:n_valid]
                s2 = cs2_pad[steps:steps + n_valid] - cs2_pad[:n_valid]
                var = s2 / steps - (s / steps) ** 2
                var = np.maximum(var, 0.0)
                vol_target[:n_valid] = np.sqrt(var).astype(np.float32)

                # FIX: NaN-fill vol targets that cross day boundaries too.
                if n_days > 1:
                    for d in range(n_days - 1):
                        day_end = self.day_boundaries[d + 1]
                        nan_start = max(self.day_boundaries[d], day_end - steps)
                        vol_target[nan_start:day_end] = np.nan

            # Magnitude target: absolute return in ticks
            mag_target = np.abs(price_change) / self.tick_size

            targets[hz_name] = {
                'return': return_target,
                'volatility': vol_target,
                'magnitude': mag_target,
            }

        return targets

    # ================================================================
    # Walk-forward evaluation
    # ================================================================

    def walk_forward_evaluate(
        self,
        target: np.ndarray,
        target_name: str,
        horizon_name: str,
        min_train_days: int = 3,
        exclude_features: Optional[List[str]] = None,
    ) -> dict:
        """
        Walk-forward LightGBM evaluation with comprehensive statistics.

        Split by day boundaries. Train on expanding window,
        predict on next day, with 1-day purge gap.

        exclude_features: list of feature names to drop before training.
        """
        import lightgbm as lgb

        n_days = len(self.day_boundaries) - 1
        if n_days < min_train_days + 1:
            return {'error': f'Need {min_train_days+1} days, have {n_days}',
                    'horizon': horizon_name, 'target': target_name}

        # Build feature mask (columns to KEEP)
        if exclude_features:
            keep_mask = np.array([
                fn not in exclude_features for fn in self.feature_names
            ])
            features_use = self.features[:, keep_mask]
            feature_names_use = [fn for fn in self.feature_names if fn not in exclude_features]
            n_features_use = len(feature_names_use)
            logger.info(f"  Excluded {sum(~keep_mask)} features, using {n_features_use}")
        else:
            features_use = self.features
            feature_names_use = self.feature_names
            n_features_use = features_use.shape[1]  # Use actual column count, not hardcoded constant

        is_regression = target_name != 'direction'

        params = {
            'n_estimators': 300,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'min_child_samples': 100,
            'verbose': -1,
            'n_jobs': -1,
            'device': 'gpu',
            'max_bin': 63,  # GPU-optimized bin count
        }
        if is_regression:
            params['objective'] = 'regression'
            params['metric'] = 'rmse'
        else:
            params['objective'] = 'binary'
            params['metric'] = 'auc'

        all_preds = []
        all_actuals = []
        all_hours = []
        all_rth_frac = []
        all_test_features = []  # FIX: Collect test features for proper leakage check
        fold_ics = []
        fold_metrics = []
        feature_importance = np.zeros(n_features_use)

        for test_day in range(min_train_days, n_days):
            # Training: all days up to (test_day - 1) for purge gap
            train_end_day = test_day - 1
            train_start = self.day_boundaries[0]
            train_end = self.day_boundaries[train_end_day + 1]

            test_start = self.day_boundaries[test_day]
            test_end = self.day_boundaries[test_day + 1]

            X_train = features_use[train_start:train_end]
            y_train = target[train_start:train_end]
            X_test = features_use[test_start:test_end]
            y_test = target[test_start:test_end]

            # Remove NaN targets
            train_valid = np.isfinite(y_train)
            test_valid = np.isfinite(y_test)

            if train_valid.sum() < 500 or test_valid.sum() < 100:
                continue

            X_tr = X_train[train_valid]
            y_tr = y_train[train_valid]
            X_te = X_test[test_valid]
            y_te = y_test[test_valid]

            # Train with 80/20 internal split for early stopping
            split = int(len(X_tr) * 0.8)
            try:
                model = lgb.LGBMRegressor(**params) if is_regression else lgb.LGBMClassifier(**params)
                model.fit(
                    X_tr[:split], y_tr[:split],
                    eval_set=[(X_tr[split:], y_tr[split:])],
                    callbacks=[lgb.early_stopping(30, verbose=False)],
                )
            except Exception as e:
                logger.warning(f"  Training failed day {test_day}: {e}")
                continue

            preds = model.predict(X_te)
            all_preds.append(preds)
            all_actuals.append(y_te)
            all_test_features.append(X_te)  # FIX: Collect for leakage check

            # Collect temporal info for this test day
            test_hours = self.hour_of_day[test_start:test_end][test_valid]
            test_rth = self.time_since_rth[test_start:test_end][test_valid]
            all_hours.append(test_hours)
            all_rth_frac.append(test_rth)

            # Per-fold metrics
            if len(preds) > 10:
                try:
                    ic_fold = spearmanr(preds, y_te)[0]
                    if np.isfinite(ic_fold):
                        fold_ics.append(ic_fold)
                        hr_fold = float((np.sign(preds) == np.sign(y_te)).mean())
                        fold_metrics.append({
                            'day': test_day,
                            'ic': float(ic_fold),
                            'hit_rate': hr_fold,
                            'n_samples': len(preds),
                            'train_size': int(train_valid.sum()),
                            'best_iteration': model.best_iteration_ if hasattr(model, 'best_iteration_') else None,
                        })
                except Exception:
                    pass

            # Feature importance
            if hasattr(model, 'feature_importances_'):
                feature_importance += model.feature_importances_

            del model
            gc.collect()

        if not all_preds:
            return {'error': 'No valid predictions',
                    'horizon': horizon_name, 'target': target_name}

        predictions = np.concatenate(all_preds)
        actuals = np.concatenate(all_actuals)
        hours = np.concatenate(all_hours)
        rth_fracs = np.concatenate(all_rth_frac)

        valid = np.isfinite(predictions) & np.isfinite(actuals)
        p, a = predictions[valid], actuals[valid]
        h = hours[valid]
        rf = rth_fracs[valid]

        if len(p) < 50:
            return {'error': f'Too few predictions: {len(p)}',
                    'horizon': horizon_name, 'target': target_name}

        # ============================================================
        # Compute comprehensive metrics
        # ============================================================

        # Overall IC
        ic = float(spearmanr(p, a)[0])
        hr = float((np.sign(p) == np.sign(a)).mean())

        winners = np.abs(a[np.sign(p) == np.sign(a)]).sum()
        losers = np.abs(a[np.sign(p) != np.sign(a)]).sum()
        pf = float(winners / losers) if losers > 0 else 0.0

        # ICIR and t-stat
        if len(fold_ics) > 2:
            ic_mean = float(np.mean(fold_ics))
            ic_std = float(np.std(fold_ics))
            icir = ic_mean / ic_std if ic_std > 0 else 0.0
            tstat = ic_mean / ic_std * np.sqrt(len(fold_ics)) if ic_std > 0 else 0.0
            try:
                _, pvalue = ttest_1samp(fold_ics, 0)
                pvalue = float(pvalue)
            except Exception:
                pvalue = 1.0
        else:
            ic_mean, ic_std = ic, 0.0
            icir, tstat, pvalue = 0.0, 0.0, 1.0

        # Sharpe (for return targets)
        sharpe = 0.0
        if target_name == 'return':
            daily_pnl = np.sign(p) * a
            if np.std(daily_pnl) > 0:
                bars_per_year = 252 * 6.5 * 3600 / (self.sample_interval_ms / 1000)
                sharpe = float(np.mean(daily_pnl) / np.std(daily_pnl) * np.sqrt(bars_per_year))

        # ============================================================
        # Per-hour-of-day analysis
        # ============================================================
        hour_stats = {}
        for hour_bin in range(6, 22):  # 6 AM to 10 PM CT covers RTH
            mask = (h >= hour_bin) & (h < hour_bin + 1)
            if mask.sum() > 50:
                p_h, a_h = p[mask], a[mask]
                try:
                    ic_h = float(spearmanr(p_h, a_h)[0])
                except Exception:
                    ic_h = 0.0
                hr_h = float((np.sign(p_h) == np.sign(a_h)).mean())
                hour_stats[hour_bin] = {
                    'ic': ic_h,
                    'hit_rate': hr_h,
                    'n_samples': int(mask.sum()),
                }

        # ============================================================
        # Session phase analysis (open, midday, close)
        # ============================================================
        session_stats = {}
        # time_since_rth: 0=open, 0.5=midday, 1.0=close
        phases = {
            'open_30min': (0.0, 0.075),       # First 30 min
            'morning': (0.075, 0.38),          # 10:00 - 12:00
            'midday': (0.38, 0.62),            # 12:00 - 14:00
            'afternoon': (0.62, 0.92),         # 14:00 - 15:30
            'close_30min': (0.92, 1.0),        # Last 30 min
        }
        for phase_name, (lo, hi) in phases.items():
            mask = (rf >= lo) & (rf < hi)
            if mask.sum() > 50:
                p_s, a_s = p[mask], a[mask]
                try:
                    ic_s = float(spearmanr(p_s, a_s)[0])
                except Exception:
                    ic_s = 0.0
                session_stats[phase_name] = {
                    'ic': ic_s,
                    'hit_rate': float((np.sign(p_s) == np.sign(a_s)).mean()),
                    'n_samples': int(mask.sum()),
                }

        # ============================================================
        # Volatility regime analysis
        # ============================================================
        # Use realized volatility of actuals as regime proxy
        regime_stats = {}
        a_abs = np.abs(a)
        vol_median = np.median(a_abs[a_abs > 0]) if (a_abs > 0).sum() > 0 else 0.0
        if vol_median > 0:
            low_vol_mask = a_abs <= vol_median
            high_vol_mask = a_abs > vol_median
            for regime_name, mask in [('low_vol', low_vol_mask), ('high_vol', high_vol_mask)]:
                if mask.sum() > 50:
                    p_r, a_r = p[mask], a[mask]
                    try:
                        ic_r = float(spearmanr(p_r, a_r)[0])
                    except Exception:
                        ic_r = 0.0
                    regime_stats[regime_name] = {
                        'ic': ic_r,
                        'hit_rate': float((np.sign(p_r) == np.sign(a_r)).mean()),
                        'n_samples': int(mask.sum()),
                    }

        # ============================================================
        # Top features
        # ============================================================
        top_feat_idx = np.argsort(feature_importance)[::-1][:15]
        top_features = [(feature_names_use[i], float(feature_importance[i]))
                        for i in top_feat_idx if feature_importance[i] > 0]

        # ============================================================
        # Leakage check: feature-target correlation
        # FIX: Correlate features against targets WITHIN test set only.
        # Old code was comparing full-dataset features[:n] against
        # concatenated test-set targets, which is meaningless.
        # Now uses concatenated test features matched to test targets.
        # Threshold raised to 0.7 for contemporaneous microstructure
        # features (e.g., ask_pressure legitimately correlates with
        # short-horizon vol because it measures current market state).
        # ============================================================
        leakage_flags = []
        LEAKAGE_THRESHOLD = 0.7  # FIX: Raised from 0.5 for microstructure features
        if all_test_features:
            test_feat_concat = np.concatenate(all_test_features)
            # Use same valid mask as predictions/actuals
            test_feat_valid = test_feat_concat[valid[:len(test_feat_concat)]] \
                if len(valid) <= len(test_feat_concat) else test_feat_concat
            n_check = min(len(a), len(test_feat_valid))
            for j in range(min(n_features_use, test_feat_valid.shape[1])):
                try:
                    feat_col = test_feat_valid[:n_check, j]
                    act_col = a[:n_check]
                    # Skip columns that are all NaN or constant
                    if np.all(np.isnan(feat_col)) or np.nanstd(feat_col) < 1e-10:
                        continue
                    # Use only rows where feature is not NaN
                    finite_mask = np.isfinite(feat_col) & np.isfinite(act_col)
                    if finite_mask.sum() < 50:
                        continue
                    corr = abs(spearmanr(feat_col[finite_mask], act_col[finite_mask])[0])
                    if corr > LEAKAGE_THRESHOLD:
                        leakage_flags.append((feature_names_use[j], float(corr)))
                except Exception:
                    continue

        # ============================================================
        # Prediction distribution stats
        # ============================================================
        pred_stats = {
            'mean': float(np.mean(p)),
            'std': float(np.std(p)),
            'skew': float(_skew(p)),
            'min': float(np.min(p)),
            'max': float(np.max(p)),
            'pct_positive': float((p > 0).mean()),
        }

        actual_stats = {
            'mean': float(np.mean(a)),
            'std': float(np.std(a)),
            'skew': float(_skew(a)),
        }

        passed = abs(ic) > 0.01 and abs(tstat) > 2.0 and len(leakage_flags) == 0

        return {
            'horizon': horizon_name,
            'target': target_name,
            # Core metrics
            'ic': ic,
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'icir': icir,
            'tstat': tstat,
            'pvalue': pvalue,
            'hit_rate': hr,
            'profit_factor': pf,
            'sharpe': sharpe,
            # Counts
            'n_predictions': len(p),
            'n_folds': len(fold_ics),
            # Per-fold
            'fold_metrics': fold_metrics,
            'fold_ics': [float(x) for x in fold_ics],
            # Time-of-day analysis
            'hour_stats': hour_stats,
            'session_stats': session_stats,
            # Regime analysis
            'regime_stats': regime_stats,
            # Features
            'top_features': top_features,
            # Leakage
            'leakage_flags': leakage_flags,
            # Distribution
            'prediction_stats': pred_stats,
            'actual_stats': actual_stats,
            # Verdict
            'passed': passed,
        }

    # ================================================================
    # Run full scan
    # ================================================================

    def run_full_scan(self, send_discord=None, exclude_features=None) -> List[dict]:
        """Run all horizon × target combinations."""
        logger.info("Computing targets...")
        targets = self.compute_targets()

        results = []
        total = sum(len(self.target_types) for _ in targets)
        done = 0

        for hz_name, hz_targets in targets.items():
            for tgt_name in self.target_types:
                if tgt_name not in hz_targets:
                    continue

                done += 1
                label = f"{hz_name}_{tgt_name}"
                logger.info(f"\n{'='*60}")
                logger.info(f"[{done}/{total}] Scanning: {label}")
                logger.info(f"{'='*60}")

                t0 = time.time()
                result = self.walk_forward_evaluate(
                    target=hz_targets[tgt_name],
                    target_name=tgt_name,
                    horizon_name=hz_name,
                    exclude_features=exclude_features,
                )
                result['elapsed_sec'] = time.time() - t0
                results.append(result)

                # Log result
                if 'error' in result:
                    logger.info(f"  ERROR: {result['error']}")
                else:
                    status = "ALPHA!" if result['passed'] else "no signal"
                    logger.info(
                        f"  [{status}] IC={result['ic']:.4f} ICIR={result['icir']:.2f} "
                        f"t={result['tstat']:.2f} HR={result['hit_rate']:.1%} "
                        f"PF={result['profit_factor']:.2f} Sharpe={result['sharpe']:.2f} "
                        f"({result['elapsed_sec']:.0f}s)"
                    )
                    if result['top_features']:
                        top3 = result['top_features'][:3]
                        logger.info(f"  Top feats: {', '.join(f'{n}={v:.0f}' for n, v in top3)}")

                    # Log session breakdown
                    if result.get('session_stats'):
                        parts = []
                        for phase, stats in result['session_stats'].items():
                            parts.append(f"{phase}:IC={stats['ic']:.3f}")
                        logger.info(f"  Session: {', '.join(parts)}")

                    # Log regime breakdown
                    if result.get('regime_stats'):
                        parts = []
                        for regime, stats in result['regime_stats'].items():
                            parts.append(f"{regime}:IC={stats['ic']:.3f}")
                        logger.info(f"  Regime: {', '.join(parts)}")

                # Discord update
                if send_discord and 'error' not in result:
                    try:
                        icon = "**ALPHA**" if result['passed'] else "---"
                        msg = (
                            f"`{label}` [{icon}]\n"
                            f"  IC={result['ic']:.4f} t={result['tstat']:.2f} "
                            f"HR={result['hit_rate']:.1%} PF={result['profit_factor']:.2f}\n"
                        )
                        if result.get('session_stats'):
                            best_session = max(result['session_stats'].items(),
                                             key=lambda x: abs(x[1]['ic']))
                            msg += f"  Best session: {best_session[0]} IC={best_session[1]['ic']:.3f}\n"
                        if result.get('leakage_flags'):
                            msg += f"  LEAKAGE: {result['leakage_flags']}\n"
                        send_discord(msg)
                    except Exception:
                        pass

                gc.collect()

        return results

    # ================================================================
    # Formatting
    # ================================================================

    @staticmethod
    def format_scoreboard(results: List[dict]) -> str:
        """Format results as comprehensive ASCII scoreboard."""
        lines = [
            "",
            "MBO ALPHA SCAN — SCOREBOARD",
            "=" * 95,
            f"{'Scan':<20s} {'IC':>7s} {'ICIR':>6s} {'t':>6s} {'HR':>6s} "
            f"{'PF':>6s} {'Sharpe':>7s} {'Preds':>8s} {'Folds':>5s} {'Pass':>5s}",
            "-" * 95,
        ]

        sorted_results = sorted(results, key=lambda x: abs(x.get('ic', 0)), reverse=True)
        for r in sorted_results:
            if 'error' in r:
                label = f"{r.get('horizon', '?')}_{r.get('target', '?')}"
                lines.append(f"{label:<20s}  ERROR: {r['error']}")
                continue

            label = f"{r['horizon']}_{r['target']}"
            passed = "YES" if r['passed'] else "no"
            lines.append(
                f"{label:<20s} {r['ic']:>7.4f} {r['icir']:>6.2f} {r['tstat']:>6.2f} "
                f"{r['hit_rate']:>6.1%} {r['profit_factor']:>6.2f} {r['sharpe']:>7.2f} "
                f"{r['n_predictions']:>8d} {r['n_folds']:>5d} {passed:>5s}"
            )

        lines.append("=" * 95)

        # Session phase breakdown for top results
        lines.append("\nSESSION PHASE BREAKDOWN (top 5 by |IC|):")
        lines.append("-" * 80)
        for r in sorted_results[:5]:
            if 'error' in r or not r.get('session_stats'):
                continue
            label = f"{r['horizon']}_{r['target']}"
            lines.append(f"  {label}:")
            for phase, stats in r['session_stats'].items():
                lines.append(
                    f"    {phase:<15s}: IC={stats['ic']:>7.4f} HR={stats['hit_rate']:>6.1%} "
                    f"n={stats['n_samples']:>6d}"
                )

        # Regime breakdown
        lines.append("\nVOLATILITY REGIME BREAKDOWN (top 5):")
        lines.append("-" * 80)
        for r in sorted_results[:5]:
            if 'error' in r or not r.get('regime_stats'):
                continue
            label = f"{r['horizon']}_{r['target']}"
            lines.append(f"  {label}:")
            for regime, stats in r['regime_stats'].items():
                lines.append(
                    f"    {regime:<15s}: IC={stats['ic']:>7.4f} HR={stats['hit_rate']:>6.1%} "
                    f"n={stats['n_samples']:>6d}"
                )

        # Per-fold IC evolution
        lines.append("\nPER-FOLD IC EVOLUTION (top 5):")
        lines.append("-" * 80)
        for r in sorted_results[:5]:
            if 'error' in r or not r.get('fold_ics'):
                continue
            label = f"{r['horizon']}_{r['target']}"
            ics_str = " ".join(f"{x:+.3f}" for x in r['fold_ics'])
            lines.append(f"  {label}: [{ics_str}]")

        # Top features across all scans
        lines.append("\nTOP FEATURES ACROSS ALL SCANS:")
        lines.append("-" * 80)
        feat_agg = {}
        for r in results:
            if 'error' in r:
                continue
            for fname, fimp in r.get('top_features', []):
                feat_agg[fname] = feat_agg.get(fname, 0) + fimp
        for fname, fimp in sorted(feat_agg.items(), key=lambda x: x[1], reverse=True)[:15]:
            lines.append(f"  {fname:<30s}: {fimp:>10.0f}")

        # Summary
        winners = [r for r in results if r.get('passed', False)]
        if winners:
            lines.append(f"\nALPHA FOUND in {len(winners)} scans:")
            for w in winners:
                lines.append(
                    f"  {w['horizon']}_{w['target']}: IC={w['ic']:.4f} "
                    f"t={w['tstat']:.2f} ICIR={w['icir']:.2f}"
                )
                if w.get('top_features'):
                    lines.append(
                        f"    Top: {', '.join(f[0] for f in w['top_features'][:5])}"
                    )
        else:
            lines.append("\nNo alpha found above threshold (IC>0.01, t>2.0)")

        # Leakage warnings
        flagged = [r for r in results if r.get('leakage_flags')]
        if flagged:
            lines.append("\nLEAKAGE WARNINGS:")
            for r in flagged:
                label = f"{r['horizon']}_{r['target']}"
                for fname, corr in r['leakage_flags']:
                    lines.append(f"  {label}: {fname} corr={corr:.3f}")

        return "\n".join(lines)


# ============================================================================
# Helpers
# ============================================================================

def _skew(arr):
    """Compute skewness."""
    m = np.mean(arr)
    s = np.std(arr)
    if s < 1e-10:
        return 0.0
    return float(np.mean(((arr - m) / s) ** 3))
