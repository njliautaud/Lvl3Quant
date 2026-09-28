"""
MFE Path Analysis — Adaptive Exit Strategy Research

The core question: instead of exiting at a fixed time horizon, what happens if we
hold a position until an exit CONDITION is met? This script analyzes the FULL price
path for high-conviction signal bars to answer:

  1. What does the MFE PATH look like (not just endpoint)?
  2. How fast does the trade reach its peak? When should we exit?
  3. How much adverse excursion happens BEFORE the favorable excursion?
     (determines limit order fill probability)
  4. Which exit strategy works best?
     - Fixed target (1t, 2t, 3t, 5t)
     - Trailing stop (trail after profit threshold)
     - Signal reversal (exit when direction flips)
     - MFE-adaptive (exit at X% of predicted magnitude)
     - Combined (MFE-adaptive with trailing stop backup)

KEY INSIGHT:
  Fixed time horizons + market order exits = spread costs eat edge.
  Limit entry + adaptive exit = keep the edge, control the risk.

Usage:
    python alpha_discovery/mfe_path_analysis.py \\
        --n-days 20 \\
        --feature-cache "C:\\Users\\Footb\\Documents\\Github\\Lvl3Quant\\data\\processed\\mbo_features_cache"

    # Load saved predictions (skip training):
    python alpha_discovery/mfe_path_analysis.py \\
        --load-predictions results/predictions_ret_10s_TIMESTAMP.npz \\
        --n-days 20

    # Quick sweep (fewer params):
    python alpha_discovery/mfe_path_analysis.py --n-days 20 --quick \\
        --feature-cache "C:\\path\\to\\cache"
"""

import gc
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any

import numpy as np
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
LVL3_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(LVL3_ROOT))

RESULTS_DIR = LVL3_ROOT / "alpha_discovery" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_log_file = RESULTS_DIR / f"mfe_path_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
logger = logging.getLogger("mfe_path")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TICK_SIZE = 0.25
TICK_VALUE = 12.50        # ES micro: $12.50 per tick
COMMISSION_RT = 4.70  # HC #52: $4.70 RT (AMP)      # $3.00 round-trip (both sides)
COMMISSION_TICKS = COMMISSION_RT / TICK_VALUE  # 0.24 ticks
HALF_TICK = TICK_SIZE / 2  # 0.125 — bid = mid - 0.125, ask = mid + 0.125
BARS_PER_SEC = 10          # 100ms bars
MAX_HOLD_BARS = 3000       # 5 minutes = 3000 bars (safety valve)
PATH_BARS = 3000           # Forward path length to analyze


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PathRecord:
    """Full forward price path data for a single signal bar."""
    day: int
    signal_bar: int                  # bar index in full array
    direction: int                   # +1 long, -1 short
    signal_strength: float           # |direction_pred|
    magnitude_pred: float            # predicted ticks of move
    entry_mid: float                 # mid price at signal bar

    # Price path (in ticks, signed by direction)
    # path_ticks[k] = (mid[signal_bar + k + 1] - entry_mid) / TICK_SIZE * direction
    # Positive = favorable, Negative = adverse
    path_ticks: np.ndarray = field(default_factory=lambda: np.array([]))
    path_len: int = 0                # actual bars available (may be < PATH_BARS at day end)

    # Path statistics (computed from path_ticks)
    mfe_ticks: float = 0.0           # max favorable excursion
    mae_ticks: float = 0.0           # max adverse excursion (before MFE)
    time_to_mfe_bars: int = 0        # bars until MFE reached
    mae_before_mfe: float = 0.0      # max adverse excursion BEFORE the MFE bar
    first_adverse_bar: int = -1      # first bar where path goes adverse
    time_to_first_adverse: int = -1  # bars until first adverse tick

    # Limit order fill analysis
    # Can a limit order 1 tick BETTER than mid get filled?
    # Long: limit at mid - HALF_TICK, fills when path_ticks <= -0.5 (mid drops to bid)
    # Short: limit at mid + HALF_TICK, fills when path_ticks >= +0.5 in adverse direction
    # But actually: for a LONG limit at entry_mid - HALF_TICK,
    #   fill condition = future_mid <= entry_mid - HALF_TICK
    #   i.e., raw path (unsigned) = future_mid - entry_mid <= -HALF_TICK
    #   i.e., path_ticks (adverse = negative for long) <= -0.5
    # We track: does price pull back at least 0.5 ticks before moving favorably?
    fill_probability: float = 0.0    # 1 if fill would have occurred within fill_horizon
    fill_bar: int = -1               # bar when filled (-1 if not)
    fill_adverse_ticks: float = 0.0  # adverse excursion needed to get filled


@dataclass
class TradeResult:
    """Result of simulating one exit strategy on one PathRecord."""
    path_idx: int                    # index into PathRecord list
    exit_strategy: str
    exit_bar: int                    # bars after fill (relative to fill bar)
    dir_pnl_ticks: float             # directional PnL from fill to exit (vs entry_price)
    exit_edge_ticks: float           # +0.5 limit exit, -0.5 market exit
    net_ticks: float                 # after commission
    net_dollars: float
    exit_type: str                   # 'target', 'stop', 'reversal', 'timeout', 'mfe_adaptive'
    filled: bool = True


# ---------------------------------------------------------------------------
# Phase 1: Data Loading (same as magnitude_gated_sim.py)
# ---------------------------------------------------------------------------

def load_data(feature_cache_dir: str, n_days: Optional[int] = None):
    """Load pre-computed features from cache. Identical to magnitude_gated_sim approach."""
    from alpha_discovery.mbo_alpha_scan import MBOAlphaScanner

    scanner = MBOAlphaScanner()
    load_info = scanner.load_precomputed_features(
        feature_cache_dir=feature_cache_dir,
        n_days=n_days,
        extra_cols=0,
    )
    logger.info(f"Loaded {load_info['n_days']} days, {load_info['n_snapshots']:,} bars, "
                f"{load_info['n_features']} features")

    feature_names = list(scanner.feature_names)

    # Downcast to float16 to halve memory
    logger.info("  Downcasting features to float16...")
    np.clip(scanner.features, -60000, 60000, out=scanner.features)
    scanner.features = scanner.features.astype(np.float16)
    gc.collect()
    logger.info(f"  Features memory: {scanner.features.nbytes / 1e9:.1f} GB")

    return scanner, feature_names, load_info


# ---------------------------------------------------------------------------
# Phase 2: Walk-Forward Training (reuse from magnitude_gated_sim.py)
# ---------------------------------------------------------------------------

def train_walk_forward(
    features: np.ndarray,
    target: np.ndarray,
    day_boundaries: list,
    min_train_days: int = 5,
    max_train_days: int = 30,
    target_name: str = 'direction',
) -> Tuple[np.ndarray, list]:
    """
    Walk-forward LightGBM with rolling window.
    Returns full-length prediction array (NaN where no prediction).
    Copied from magnitude_gated_sim.py for self-containment.
    """
    import lightgbm as lgb

    N = len(target)
    n_days = len(day_boundaries) - 1
    full_preds = np.full(N, np.nan, dtype=np.float32)
    fold_ics = []

    params = {
        'n_estimators': 300,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.3,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'min_child_samples': 100,
        'verbose': -1,
        'n_jobs': 2,
        'device': 'cpu',
        'max_bin': 63,
        'force_row_wise': True,
        'objective': 'regression',
        'metric': 'rmse',
    }

    n_folds = 0
    for test_day in range(min_train_days, n_days):
        train_end_day = test_day - 1
        train_start_day = max(0, train_end_day - max_train_days + 1)
        train_start = day_boundaries[train_start_day]
        train_end = day_boundaries[train_end_day + 1]
        test_start = day_boundaries[test_day]
        test_end = day_boundaries[test_day + 1]

        X_train = features[train_start:train_end]
        y_train = target[train_start:train_end]
        X_test = features[test_start:test_end]
        y_test = target[test_start:test_end]

        train_valid = np.isfinite(y_train)
        test_valid = np.isfinite(y_test)

        if train_valid.sum() < 500 or test_valid.sum() < 100:
            continue

        MAX_TRAIN_SAMPLES = 500_000
        valid_indices = np.where(train_valid)[0]
        if len(valid_indices) > MAX_TRAIN_SAMPLES:
            rng = np.random.default_rng(seed=test_day)
            sampled = rng.choice(valid_indices, MAX_TRAIN_SAMPLES, replace=False)
            sampled.sort()
            X_tr = X_train[sampled].astype(np.float32)
            y_tr = y_train[sampled]
        else:
            X_tr = X_train[train_valid].astype(np.float32)
            y_tr = y_train[train_valid]
        X_te = X_test[test_valid].astype(np.float32)
        y_te = y_test[test_valid]

        split = int(len(X_tr) * 0.8)
        try:
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_tr[:split], y_tr[:split],
                eval_set=[(X_tr[split:], y_tr[split:])],
                callbacks=[lgb.early_stopping(30, verbose=False)],
            )
            preds = model.predict(X_te)
        except Exception as e:
            logger.warning(f"  Training failed day {test_day}: {e}")
            continue

        del X_tr, y_tr
        gc.collect()

        valid_positions = np.arange(test_start, test_end)[test_valid]
        n = min(len(valid_positions), len(preds))
        full_preds[valid_positions[:n]] = preds[:n].astype(np.float32)

        if len(preds) > 10:
            try:
                ic = float(spearmanr(preds, y_te)[0])
                if np.isfinite(ic):
                    fold_ics.append(ic)
            except Exception:
                pass

        n_folds += 1
        if n_folds % 5 == 0:
            ic_so_far = np.mean(fold_ics) if fold_ics else 0
            logger.info(f"  [{target_name}] Fold {n_folds} (day {test_day}): IC={ic_so_far:.4f}")
            sys.stdout.flush()

        del model
        gc.collect()

    n_valid = np.isfinite(full_preds).sum()
    overall_ic = 0.0
    if n_valid > 50:
        mask = np.isfinite(full_preds) & np.isfinite(target)
        if mask.sum() > 50:
            overall_ic = float(spearmanr(full_preds[mask], target[mask])[0])

    logger.info(f"  [{target_name}] Complete: {n_folds} folds, {n_valid:,} preds, IC={overall_ic:.4f}")
    return full_preds, fold_ics


# ---------------------------------------------------------------------------
# Phase 3: Compute Targets (same as magnitude_gated_sim.py)
# ---------------------------------------------------------------------------

def compute_targets(mid_prices, horizon_bars, day_boundaries):
    """
    Compute direction target (mfe_net_5s) and magnitude target (abs move in ticks).
    We use a 5s MFE net target for the DIRECTION model — it captures directional
    asymmetry better than a fixed-horizon return.
    The MAGNITUDE model predicts absolute move in ticks at the same 5s horizon.
    """
    from alpha_discovery.run_mfe_scan import compute_mfe_targets

    N = len(mid_prices)
    n_days = len(day_boundaries) - 1

    # Future mid at horizon
    future_mid = np.empty(N, dtype=np.float32)
    future_mid[:N - horizon_bars] = mid_prices[horizon_bars:]
    future_mid[N - horizon_bars:] = np.nan
    for d in range(n_days - 1):
        day_end = day_boundaries[d + 1]
        nan_start = max(day_boundaries[d], day_end - horizon_bars)
        future_mid[nan_start:day_end] = np.nan

    # Direction target: mfe_net_5s (directional asymmetry)
    hz_name = {30: '3s', 50: '5s', 100: '10s', 300: '30s'}.get(horizon_bars, '5s')
    hz_sec = {'3s': 3, '5s': 5, '10s': 10, '30s': 30}.get(hz_name, 5)
    mfe_targets = compute_mfe_targets(
        mid_prices=mid_prices,
        day_boundaries=day_boundaries,
        sample_interval_ms=100,
        horizons_sec={hz_name: hz_sec},
        tick_size=TICK_SIZE,
    )
    direction_target = mfe_targets[f'mfe_net_{hz_name}']
    del mfe_targets
    gc.collect()

    # Magnitude target: abs move in ticks
    magnitude_target = np.abs(future_mid - mid_prices) / TICK_SIZE

    return direction_target, magnitude_target


# ---------------------------------------------------------------------------
# Phase 4: Compute Full Price Paths for Signal Bars
# ---------------------------------------------------------------------------

def compute_price_paths(
    mid_prices: np.ndarray,
    direction_preds: np.ndarray,
    magnitude_preds: np.ndarray,
    day_boundaries: list,
    mag_gate_threshold: float = 2.0,
    signal_quantile: float = 0.80,
    latency_bars: int = 1,
    fill_horizon_bars: int = 50,       # max bars to wait for limit fill
    min_bars_between_signals: int = 50,
    max_paths: int = 0,                 # 0 = no cap (was 5000, biased results)
) -> List[PathRecord]:
    """
    For each bar passing the magnitude gate + direction confidence filter:
    1. Post a limit order 1 tick better than mid
    2. Compute the full forward price path (PATH_BARS bars)
    3. Track fill probability, MFE trajectory, MAE before MFE

    Returns a list of PathRecord objects, one per signal bar (filled or not).
    """
    N = len(mid_prices)
    n_days = len(day_boundaries) - 1

    has_pred = np.isfinite(direction_preds) & np.isfinite(magnitude_preds)
    valid_dir = direction_preds[has_pred]

    if len(valid_dir) < 100:
        logger.warning("Insufficient predictions for path analysis")
        return []

    abs_dir = np.abs(valid_dir)
    signal_threshold = np.percentile(abs_dir, signal_quantile * 100)
    logger.info(f"  Signal threshold (Q={signal_quantile:.0%}): {signal_threshold:.4f}")

    paths: List[PathRecord] = []
    next_allowed_bar = 0
    n_gated = 0
    n_weak = 0
    n_posted = 0

    for day_idx in range(n_days):
        day_start = day_boundaries[day_idx]
        day_end = day_boundaries[day_idx + 1]

        for i in range(day_start, day_end):
            if i < next_allowed_bar:
                continue
            if not has_pred[i]:
                continue

            # Gate checks
            if magnitude_preds[i] < mag_gate_threshold:
                n_gated += 1
                continue
            if abs(direction_preds[i]) < signal_threshold:
                n_weak += 1
                continue

            direction = 1 if direction_preds[i] > 0 else -1
            post_bar = i + latency_bars
            if post_bar >= day_end:
                continue

            post_mid = mid_prices[post_bar]
            n_posted += 1

            # Limit entry price: 1 tick BETTER than mid
            # Long: post limit BUY at bid (mid - half_tick)
            # Short: post limit SELL at ask (mid + half_tick)
            if direction == 1:
                entry_price = post_mid - HALF_TICK
            else:
                entry_price = post_mid + HALF_TICK

            # Build forward price path from post_bar
            path_end = min(post_bar + PATH_BARS, day_end)
            path_len = path_end - post_bar - 1  # exclude the post_bar itself

            if path_len <= 0:
                continue

            # Raw path prices
            raw_path = mid_prices[post_bar + 1 : post_bar + 1 + path_len]

            # Convert to signed ticks relative to entry_price (from entry_price perspective)
            # For LONG: positive = favorable (price went up from entry)
            # For SHORT: positive = favorable (price went down from entry)
            path_ticks = (raw_path - entry_price) / TICK_SIZE * direction

            # --- Fill analysis ---
            # Long fills when future_mid <= entry_price (raw_path drops to or below bid)
            # Short fills when future_mid >= entry_price (raw_path rises to or above ask)
            fill_bar = -1
            fill_window = min(fill_horizon_bars, path_len)
            if direction == 1:
                fill_cond = raw_path[:fill_window] <= entry_price
            else:
                fill_cond = raw_path[:fill_window] >= entry_price

            fill_indices = np.where(fill_cond)[0]
            if len(fill_indices) > 0:
                fill_bar = int(fill_indices[0])  # bars after post_bar

            fill_probability = 1.0 if fill_bar >= 0 else 0.0

            # --- Path statistics ---
            # Only meaningful if we got filled
            mfe_ticks = float(np.max(path_ticks)) if len(path_ticks) > 0 else 0.0
            mae_ticks = float(np.max(-path_ticks)) if len(path_ticks) > 0 else 0.0

            # Time to peak (MFE)
            if len(path_ticks) > 0:
                time_to_mfe_bars = int(np.argmax(path_ticks))
            else:
                time_to_mfe_bars = 0

            # Max adverse excursion BEFORE the MFE bar
            if time_to_mfe_bars > 0:
                pre_mfe_path = path_ticks[:time_to_mfe_bars]
                mae_before_mfe = float(np.max(-pre_mfe_path)) if len(pre_mfe_path) > 0 else 0.0
            else:
                mae_before_mfe = 0.0

            # First adverse bar
            adverse_mask = path_ticks < 0
            adverse_indices = np.where(adverse_mask)[0]
            first_adverse_bar = int(adverse_indices[0]) if len(adverse_indices) > 0 else -1

            rec = PathRecord(
                day=day_idx,
                signal_bar=i,
                direction=direction,
                signal_strength=float(abs(direction_preds[i])),
                magnitude_pred=float(magnitude_preds[i]),
                entry_mid=float(post_mid),
                path_ticks=path_ticks.copy(),
                path_len=path_len,
                mfe_ticks=mfe_ticks,
                mae_ticks=mae_ticks,
                time_to_mfe_bars=time_to_mfe_bars,
                mae_before_mfe=mae_before_mfe,
                first_adverse_bar=first_adverse_bar,
                time_to_first_adverse=first_adverse_bar,
                fill_probability=fill_probability,
                fill_bar=fill_bar,
                fill_adverse_ticks=float(-path_ticks[fill_bar]) if fill_bar >= 0 and fill_bar < len(path_ticks) else 0.0,
            )
            paths.append(rec)
            next_allowed_bar = i + min_bars_between_signals

            if max_paths > 0 and len(paths) >= max_paths:
                logger.info(f"  Path cap ({max_paths}) reached at day {day_idx}")
                break

        if max_paths > 0 and len(paths) >= max_paths:
            break

    filled_paths = [p for p in paths if p.fill_bar >= 0]
    logger.info(f"  Path analysis: {n_posted} posted, {len(paths)} recorded, "
                f"{len(filled_paths)} filled ({len(filled_paths)/max(len(paths),1):.1%})")
    logger.info(f"  Gated out: {n_gated}, signal weak: {n_weak}")

    return paths


# ---------------------------------------------------------------------------
# Phase 5: MFE Path Statistics
# ---------------------------------------------------------------------------

def analyze_path_statistics(paths: List[PathRecord]) -> Dict:
    """
    Aggregate statistics across all path records.
    Answers the KEY QUESTIONS:
    1. What does the MFE path look like?
    2. How quickly does MFE get reached?
    3. How much adverse excursion precedes it?
    """
    filled = [p for p in paths if p.fill_bar >= 0]
    if not filled:
        return {'error': 'No filled paths'}

    n = len(filled)

    # --- MFE distribution ---
    mfe_arr = np.array([p.mfe_ticks for p in filled])
    mae_arr = np.array([p.mae_ticks for p in filled])
    mae_before_arr = np.array([p.mae_before_mfe for p in filled])
    ttm_arr = np.array([p.time_to_mfe_bars for p in filled])
    mag_arr = np.array([p.magnitude_pred for p in filled])
    fill_adv_arr = np.array([p.fill_adverse_ticks for p in filled])

    # --- Average path trajectory (aggregate over all filled trades) ---
    # Build a matrix of path_ticks, padded/trimmed to 600 bars (60s)
    horizon_bars = min(600, min(p.path_len for p in filled))
    path_matrix = np.zeros((n, horizon_bars), dtype=np.float32)
    for k, p in enumerate(filled):
        end = min(p.path_len, horizon_bars)
        path_matrix[k, :end] = p.path_ticks[:end]
        if end < horizon_bars:
            path_matrix[k, end:] = p.path_ticks[end - 1] if end > 0 else 0.0

    avg_path = path_matrix.mean(axis=0)    # mean path in ticks
    pct_positive = (path_matrix > 0).mean(axis=0)  # fraction of trades positive at each bar
    pct_hit_1t = (path_matrix >= 1.0).mean(axis=0)  # fraction that hit 1t at each bar
    pct_hit_2t = (path_matrix >= 2.0).mean(axis=0)
    pct_hit_3t = (path_matrix >= 3.0).mean(axis=0)

    # Time to hit each target (across all filled trades)
    def time_to_target(target_ticks: float) -> np.ndarray:
        """Returns array of bars-to-target for each path (-1 if never reached)."""
        result = np.full(n, -1, dtype=np.int32)
        for k, p in enumerate(filled):
            hits = np.where(p.path_ticks >= target_ticks)[0]
            if len(hits) > 0:
                result[k] = hits[0]
        return result

    ttm_1t = time_to_target(1.0)
    ttm_2t = time_to_target(2.0)
    ttm_3t = time_to_target(3.0)

    # --- Adverse excursion before favorable ---
    # What fraction of trades experience adverse before favorable?
    n_adverse_first = sum(1 for p in filled if p.first_adverse_bar == 0 or (
        p.first_adverse_bar >= 0 and p.first_adverse_bar < p.time_to_mfe_bars))

    stats = {
        'n_filled': n,
        'n_paths_total': len(paths),
        'fill_rate': n / max(len(paths), 1),

        # MFE distribution
        'mfe_mean': float(mfe_arr.mean()),
        'mfe_median': float(np.median(mfe_arr)),
        'mfe_p25': float(np.percentile(mfe_arr, 25)),
        'mfe_p75': float(np.percentile(mfe_arr, 75)),
        'mfe_p90': float(np.percentile(mfe_arr, 90)),
        'mfe_min': float(mfe_arr.min()),
        'mfe_max': float(mfe_arr.max()),

        # MAE distribution
        'mae_mean': float(mae_arr.mean()),
        'mae_median': float(np.median(mae_arr)),
        'mae_before_mfe_mean': float(mae_before_arr.mean()),
        'mae_before_mfe_median': float(np.median(mae_before_arr)),

        # Fill adverse (how much adverse to get filled)
        'fill_adverse_mean': float(fill_adv_arr.mean()),
        'fill_adverse_median': float(np.median(fill_adv_arr)),

        # Time to MFE
        'time_to_mfe_mean_bars': float(ttm_arr.mean()),
        'time_to_mfe_median_bars': float(np.median(ttm_arr)),
        'time_to_mfe_mean_sec': float(ttm_arr.mean()) / BARS_PER_SEC,
        'time_to_mfe_median_sec': float(np.median(ttm_arr)) / BARS_PER_SEC,
        'time_to_mfe_p90_bars': float(np.percentile(ttm_arr, 90)),
        'time_to_mfe_p90_sec': float(np.percentile(ttm_arr, 90)) / BARS_PER_SEC,

        # Time to each target level
        'time_to_1t_mean_sec': float(ttm_1t[ttm_1t >= 0].mean()) / BARS_PER_SEC if (ttm_1t >= 0).any() else -1,
        'time_to_1t_hit_rate': float((ttm_1t >= 0).mean()),
        'time_to_2t_mean_sec': float(ttm_2t[ttm_2t >= 0].mean()) / BARS_PER_SEC if (ttm_2t >= 0).any() else -1,
        'time_to_2t_hit_rate': float((ttm_2t >= 0).mean()),
        'time_to_3t_mean_sec': float(ttm_3t[ttm_3t >= 0].mean()) / BARS_PER_SEC if (ttm_3t >= 0).any() else -1,
        'time_to_3t_hit_rate': float((ttm_3t >= 0).mean()),

        # Adverse profile
        'pct_adverse_before_favorable': float(n_adverse_first / n),
        'pct_mfe_ge_1t': float((mfe_arr >= 1.0).mean()),
        'pct_mfe_ge_2t': float((mfe_arr >= 2.0).mean()),
        'pct_mfe_ge_3t': float((mfe_arr >= 3.0).mean()),

        # Magnitude prediction quality (for filled trades)
        'magnitude_pred_mean': float(mag_arr.mean()),
        'mfe_vs_pred_ratio': float(mfe_arr.mean() / max(mag_arr.mean(), 0.01)),

        # Path trajectory (downsampled to 60 bars = 6s)
        'avg_path_ticks_60bars': [float(x) for x in avg_path[:60]],
        'pct_positive_60bars': [float(x) for x in pct_positive[:60]],
        'pct_hit_1t_60bars': [float(x) for x in pct_hit_1t[:60]],
        'pct_hit_2t_60bars': [float(x) for x in pct_hit_2t[:60]],
        'pct_hit_3t_60bars': [float(x) for x in pct_hit_3t[:60]],
    }

    return stats


# ---------------------------------------------------------------------------
# Phase 6: Simulate Exit Strategies
# ---------------------------------------------------------------------------

def simulate_exit_strategy(
    paths: List[PathRecord],
    strategy: str,
    # Strategy parameters
    target_ticks: float = 2.0,           # for 'fixed_target'
    stop_ticks: float = 2.0,             # for 'fixed_target', 'trailing'
    trail_trigger_ticks: float = 1.0,    # for 'trailing': start trailing after this profit
    trail_stop_ticks: float = 1.0,       # for 'trailing': trail by this many ticks
    mfe_fraction: float = 0.8,           # for 'mfe_adaptive': exit at X% of mag_pred
    max_hold_bars: int = MAX_HOLD_BARS,  # safety valve for all strategies
    # Direction predictions for 'signal_reversal'
    direction_preds: Optional[np.ndarray] = None,
    signal_bar_offset: int = 0,          # for indexing direction_preds by signal_bar
    reversal_threshold: float = 0.0,     # sign flip in direction_pred triggers exit
) -> Dict:
    """
    Simulate a single exit strategy across all filled path records.

    Strategies:
      'fixed_target' : exit when PnL >= target_ticks OR PnL <= -stop_ticks OR timeout
      'trailing'     : once profit >= trail_trigger, set stop at peak - trail_stop_ticks
      'mfe_adaptive' : exit when PnL >= mfe_fraction * magnitude_pred OR timeout
      'signal_reversal': exit when direction_pred flips sign (requires direction_preds array)
      'combined'     : mfe_adaptive + trailing stop backup

    All strategies:
      - Use limit EXIT when possible (+0.5t edge)
      - Fall back to market EXIT at timeout (-0.5t edge)
      - $3.00 RT commission (0.24 ticks)
    """
    filled = [p for p in paths if p.fill_bar >= 0]
    if not filled:
        return {'error': 'No filled paths', 'strategy': strategy}

    results = []

    for idx, path in enumerate(filled):
        if path.path_len == 0:
            continue

        # Path from fill_bar onward (fill_bar is relative to post_bar)
        # path.path_ticks is relative to entry_price, so after fill the PnL
        # at path bar k is: path_ticks[fill_bar + k] (already vs entry_price)
        # fill_bar is relative to post_bar (offset from where path starts)
        fill_offset = path.fill_bar  # index into path_ticks where fill occurred
        if fill_offset >= path.path_len:
            continue

        post_fill_ticks = path.path_ticks[fill_offset:]  # PnL from fill onward
        n_post = len(post_fill_ticks)
        if n_post == 0:
            continue

        # Clip to max_hold_bars
        hold_end = min(n_post, max_hold_bars)
        post_fill_ticks = post_fill_ticks[:hold_end]
        n_post = len(post_fill_ticks)

        exit_bar = n_post - 1     # default: timeout at max_hold
        exit_type = 'timeout'
        use_limit_exit = False

        if strategy == 'fixed_target':
            for k in range(n_post):
                pnl = post_fill_ticks[k]
                if pnl >= target_ticks:
                    exit_bar = k
                    exit_type = 'target'
                    use_limit_exit = True  # can use limit order at target
                    break
                elif stop_ticks > 0 and pnl <= -stop_ticks:
                    exit_bar = k
                    exit_type = 'stop'
                    use_limit_exit = False  # market exit on stop
                    break

        elif strategy == 'trailing':
            peak_pnl = 0.0
            trailing_active = False
            for k in range(n_post):
                pnl = post_fill_ticks[k]
                if pnl > peak_pnl:
                    peak_pnl = pnl
                if pnl >= trail_trigger_ticks:
                    trailing_active = True
                if trailing_active:
                    trail_stop = peak_pnl - trail_stop_ticks
                    if pnl <= trail_stop:
                        exit_bar = k
                        exit_type = 'trailing_stop'
                        use_limit_exit = False  # market on trail stop hit
                        break
            # If no stop hit and profit achieved, can also add a max target
            # but let's keep it simple: trail until timeout

        elif strategy == 'mfe_adaptive':
            # Exit at X% of predicted magnitude
            adaptive_target = mfe_fraction * path.magnitude_pred
            adaptive_target = max(adaptive_target, 1.0)  # floor at 1 tick
            for k in range(n_post):
                pnl = post_fill_ticks[k]
                if pnl >= adaptive_target:
                    exit_bar = k
                    exit_type = 'mfe_adaptive'
                    use_limit_exit = True
                    break

        elif strategy == 'signal_reversal':
            # Exit when direction prediction changes sign
            if direction_preds is None:
                exit_bar = n_post - 1
                exit_type = 'timeout'
            else:
                entry_dir = path.direction
                # path.signal_bar is the original signal bar index
                for k in range(n_post):
                    # global bar index = path.signal_bar + path.fill_bar (fill relative to post_bar)
                    # + 1 (post_bar offset) + k (bars after fill)
                    bar_idx = path.signal_bar + 1 + path.fill_bar + k
                    if bar_idx >= len(direction_preds):
                        break
                    dpred = direction_preds[bar_idx]
                    if not np.isfinite(dpred):
                        continue
                    new_dir = 1 if dpred > reversal_threshold else -1
                    if new_dir != entry_dir:
                        exit_bar = k
                        exit_type = 'signal_reversal'
                        use_limit_exit = False  # market exit on reversal
                        break

        elif strategy == 'combined':
            # MFE-adaptive target + trailing stop as backup
            adaptive_target = mfe_fraction * path.magnitude_pred
            adaptive_target = max(adaptive_target, 1.0)
            peak_pnl = 0.0
            trailing_active = False
            for k in range(n_post):
                pnl = post_fill_ticks[k]
                if pnl > peak_pnl:
                    peak_pnl = pnl
                # Activate trailing once some profit achieved
                if pnl >= trail_trigger_ticks:
                    trailing_active = True
                # Check adaptive target
                if pnl >= adaptive_target:
                    exit_bar = k
                    exit_type = 'mfe_adaptive'
                    use_limit_exit = True
                    break
                # Check trailing stop
                if trailing_active:
                    trail_stop = peak_pnl - trail_stop_ticks
                    if pnl <= trail_stop:
                        exit_bar = k
                        exit_type = 'trailing_stop'
                        use_limit_exit = False
                        break

        # --- Compute PnL ---
        dir_pnl_ticks = float(post_fill_ticks[exit_bar])
        exit_edge_ticks = 0.5 if use_limit_exit else -0.5
        net_ticks = dir_pnl_ticks + exit_edge_ticks - COMMISSION_TICKS
        net_dollars = net_ticks * TICK_VALUE

        results.append(TradeResult(
            path_idx=idx,
            exit_strategy=strategy,
            exit_bar=exit_bar,
            dir_pnl_ticks=dir_pnl_ticks,
            exit_edge_ticks=exit_edge_ticks,
            net_ticks=net_ticks,
            net_dollars=net_dollars,
            exit_type=exit_type,
            filled=True,
        ))

    if not results:
        return {'error': 'No completed trades', 'strategy': strategy}

    net_ticks_arr = np.array([r.net_ticks for r in results])
    net_dollars_arr = np.array([r.net_dollars for r in results])
    exit_bars_arr = np.array([r.exit_bar for r in results])
    dir_pnl_arr = np.array([r.dir_pnl_ticks for r in results])

    # Exit type breakdown
    exit_types: Dict[str, Dict] = {}
    for r in results:
        if r.exit_type not in exit_types:
            exit_types[r.exit_type] = {'count': 0, 'pnl': [], 'wins': 0}
        exit_types[r.exit_type]['count'] += 1
        exit_types[r.exit_type]['pnl'].append(r.net_ticks)
        if r.net_ticks > 0:
            exit_types[r.exit_type]['wins'] += 1

    exit_breakdown = {}
    for et, d in exit_types.items():
        pnl_arr = np.array(d['pnl'])
        exit_breakdown[et] = {
            'count': d['count'],
            'pct': d['count'] / len(results),
            'mean_pnl_ticks': float(pnl_arr.mean()),
            'total_pnl_ticks': float(pnl_arr.sum()),
            'win_rate': d['wins'] / d['count'] if d['count'] > 0 else 0,
        }

    # Day-by-day PnL for Sharpe
    day_pnl: Dict[int, float] = {}
    for r in results:
        day = filled[r.path_idx].day
        day_pnl[day] = day_pnl.get(day, 0.0) + r.net_dollars
    day_pnl_arr = np.array(list(day_pnl.values()))

    # Sharpe
    if len(day_pnl_arr) > 2 and day_pnl_arr.std() > 0:
        sharpe = float(day_pnl_arr.mean() / day_pnl_arr.std() * np.sqrt(252))
    else:
        sharpe = 0.0

    # Profit factor
    gross_profit = float(net_ticks_arr[net_ticks_arr > 0].sum()) if (net_ticks_arr > 0).any() else 0.0
    gross_loss = float(abs(net_ticks_arr[net_ticks_arr < 0].sum())) if (net_ticks_arr < 0).any() else 0.001
    profit_factor = gross_profit / gross_loss

    # Max drawdown
    cum_pnl = np.cumsum(net_dollars_arr)
    peak = np.maximum.accumulate(cum_pnl)
    max_dd = float((peak - cum_pnl).max())

    win_rate = float((net_ticks_arr > 0).mean())
    trades_per_day = len(results) / max(len(day_pnl), 1)

    return {
        'strategy': strategy,
        'n_trades': len(results),
        'trades_per_day': trades_per_day,
        # PnL
        'total_pnl_ticks': float(net_ticks_arr.sum()),
        'total_pnl_dollars': float(net_dollars_arr.sum()),
        'mean_pnl_ticks': float(net_ticks_arr.mean()),
        'mean_pnl_dollars': float(net_dollars_arr.mean()),
        'median_pnl_ticks': float(np.median(net_ticks_arr)),
        'mean_dir_pnl_ticks': float(dir_pnl_arr.mean()),
        # Quality
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'sharpe_annualized': sharpe,
        'max_drawdown_dollars': max_dd,
        # Timing
        'mean_hold_bars': float(exit_bars_arr.mean()),
        'mean_hold_sec': float(exit_bars_arr.mean()) / BARS_PER_SEC,
        'median_hold_sec': float(np.median(exit_bars_arr)) / BARS_PER_SEC,
        # Day stats
        'n_positive_days': int((day_pnl_arr > 0).sum()),
        'n_negative_days': int((day_pnl_arr < 0).sum()),
        'day_pnl_mean': float(day_pnl_arr.mean()) if len(day_pnl_arr) > 0 else 0,
        'day_pnl_std': float(day_pnl_arr.std()) if len(day_pnl_arr) > 1 else 0,
        # Breakdown
        'exit_type_breakdown': exit_breakdown,
    }


# ---------------------------------------------------------------------------
# Phase 7: Strategy Sweep
# ---------------------------------------------------------------------------

def run_strategy_sweep(
    paths: List[PathRecord],
    direction_preds: np.ndarray,
    quick: bool = False,
) -> List[Dict]:
    """
    Sweep all exit strategies and parameter combinations.
    Returns list of result dicts sorted by Sharpe.
    """
    results = []
    filled = [p for p in paths if p.fill_bar >= 0]
    if not filled:
        logger.warning("No filled paths for strategy sweep!")
        return results

    logger.info(f"\n{'='*60}")
    logger.info(f"STRATEGY SWEEP — {len(filled)} filled trades")
    logger.info(f"{'='*60}")

    # --- Fixed target strategies ---
    logger.info("\n--- Fixed Target + Stop Loss ---")
    if quick:
        targets = [1.0, 2.0, 3.0]
        stops = [None, 2.0, 3.0]
    else:
        targets = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
        stops = [None, 1.0, 1.5, 2.0, 3.0, 5.0]

    for tgt in targets:
        for stop in stops:
            r = simulate_exit_strategy(
                paths, 'fixed_target',
                target_ticks=tgt,
                stop_ticks=stop if stop is not None else 9999,
                max_hold_bars=MAX_HOLD_BARS,
            )
            if 'error' not in r:
                stop_str = f"SL={stop:.1f}" if stop else "NoSL"
                logger.info(
                    f"  TP={tgt:.1f}t {stop_str}: "
                    f"N={r['n_trades']} WR={r['win_rate']:.1%} "
                    f"PF={r['profit_factor']:.2f} "
                    f"Sharpe={r['sharpe_annualized']:.2f} "
                    f"$/t={r['mean_pnl_dollars']:+.2f} "
                    f"Hold={r['mean_hold_sec']:.1f}s"
                )
                r['params'] = {'target': tgt, 'stop': stop, 'type': 'fixed_target'}
                results.append(r)

    # --- Trailing stop strategies ---
    logger.info("\n--- Trailing Stop ---")
    if quick:
        triggers = [1.0, 1.5]
        trails = [0.5, 1.0]
    else:
        triggers = [0.5, 1.0, 1.5, 2.0, 3.0]
        trails = [0.5, 1.0, 1.5, 2.0]

    for trigger in triggers:
        for trail in trails:
            r = simulate_exit_strategy(
                paths, 'trailing',
                trail_trigger_ticks=trigger,
                trail_stop_ticks=trail,
                max_hold_bars=MAX_HOLD_BARS,
            )
            if 'error' not in r:
                logger.info(
                    f"  Trail(trig={trigger:.1f}t, trail={trail:.1f}t): "
                    f"N={r['n_trades']} WR={r['win_rate']:.1%} "
                    f"PF={r['profit_factor']:.2f} "
                    f"Sharpe={r['sharpe_annualized']:.2f} "
                    f"$/t={r['mean_pnl_dollars']:+.2f} "
                    f"Hold={r['mean_hold_sec']:.1f}s"
                )
                r['params'] = {'trigger': trigger, 'trail': trail, 'type': 'trailing'}
                results.append(r)

    # --- MFE-adaptive strategies ---
    logger.info("\n--- MFE-Adaptive (exit at % of predicted magnitude) ---")
    if quick:
        fractions = [0.5, 0.7, 0.9]
    else:
        fractions = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2]

    for frac in fractions:
        r = simulate_exit_strategy(
            paths, 'mfe_adaptive',
            mfe_fraction=frac,
            max_hold_bars=MAX_HOLD_BARS,
        )
        if 'error' not in r:
            logger.info(
                f"  MFE-adaptive(frac={frac:.1f}): "
                f"N={r['n_trades']} WR={r['win_rate']:.1%} "
                f"PF={r['profit_factor']:.2f} "
                f"Sharpe={r['sharpe_annualized']:.2f} "
                f"$/t={r['mean_pnl_dollars']:+.2f} "
                f"Hold={r['mean_hold_sec']:.1f}s"
            )
            r['params'] = {'fraction': frac, 'type': 'mfe_adaptive'}
            results.append(r)

    # --- Signal reversal ---
    logger.info("\n--- Signal Reversal ---")
    r = simulate_exit_strategy(
        paths, 'signal_reversal',
        direction_preds=direction_preds,
        max_hold_bars=MAX_HOLD_BARS,
    )
    if 'error' not in r:
        logger.info(
            f"  Signal reversal: "
            f"N={r['n_trades']} WR={r['win_rate']:.1%} "
            f"PF={r['profit_factor']:.2f} "
            f"Sharpe={r['sharpe_annualized']:.2f} "
            f"$/t={r['mean_pnl_dollars']:+.2f} "
            f"Hold={r['mean_hold_sec']:.1f}s"
        )
        r['params'] = {'type': 'signal_reversal'}
        results.append(r)

    # --- Combined: MFE-adaptive + trailing stop ---
    logger.info("\n--- Combined (MFE-adaptive + Trailing Stop) ---")
    if quick:
        combo_params = [(0.7, 1.0, 0.5), (0.8, 1.5, 1.0)]
    else:
        combo_params = [
            (0.5, 0.5, 0.5), (0.6, 1.0, 0.5), (0.7, 1.0, 0.5), (0.7, 1.0, 1.0),
            (0.8, 1.0, 0.5), (0.8, 1.5, 1.0), (0.9, 1.5, 1.0), (0.9, 2.0, 1.5),
        ]

    for frac, trigger, trail in combo_params:
        r = simulate_exit_strategy(
            paths, 'combined',
            mfe_fraction=frac,
            trail_trigger_ticks=trigger,
            trail_stop_ticks=trail,
            max_hold_bars=MAX_HOLD_BARS,
        )
        if 'error' not in r:
            logger.info(
                f"  Combined(frac={frac:.1f}, trig={trigger:.1f}t, trail={trail:.1f}t): "
                f"N={r['n_trades']} WR={r['win_rate']:.1%} "
                f"PF={r['profit_factor']:.2f} "
                f"Sharpe={r['sharpe_annualized']:.2f} "
                f"$/t={r['mean_pnl_dollars']:+.2f} "
                f"Hold={r['mean_hold_sec']:.1f}s"
            )
            r['params'] = {'fraction': frac, 'trigger': trigger, 'trail': trail, 'type': 'combined'}
            results.append(r)

    # Sort by Sharpe
    valid = [r for r in results if 'error' not in r]
    valid.sort(key=lambda r: r['sharpe_annualized'], reverse=True)
    return valid


# ---------------------------------------------------------------------------
# Phase 8: Gate Threshold Sweep
# ---------------------------------------------------------------------------

def run_gate_sweep(
    mid_prices: np.ndarray,
    direction_preds: np.ndarray,
    magnitude_preds: np.ndarray,
    day_boundaries: list,
    best_strategy_params: Dict,
    quick: bool = False,
) -> List[Dict]:
    """
    Sweep magnitude gate thresholds and direction confidence quantiles
    using the best exit strategy found above.
    """
    logger.info(f"\n{'='*60}")
    logger.info("GATE THRESHOLD SWEEP")
    logger.info(f"Best strategy: {best_strategy_params}")
    logger.info(f"{'='*60}")

    if quick:
        gate_thresholds = [1.0, 2.0, 3.0]
        quantiles = [0.70, 0.85]
    else:
        gate_thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
        quantiles = [0.60, 0.70, 0.80, 0.90]

    results = []
    for gate in gate_thresholds:
        for q in quantiles:
            paths = compute_price_paths(
                mid_prices, direction_preds, magnitude_preds, day_boundaries,
                mag_gate_threshold=gate,
                signal_quantile=q,
                latency_bars=1,
                fill_horizon_bars=50,
                min_bars_between_signals=50,
            )

            if not any(p.fill_bar >= 0 for p in paths):
                continue

            # Use best strategy from sweep
            strategy_type = best_strategy_params.get('type', 'fixed_target')
            r = simulate_exit_strategy(
                paths, strategy_type,
                target_ticks=best_strategy_params.get('target', 2.0),
                stop_ticks=best_strategy_params.get('stop', 9999) or 9999,
                mfe_fraction=best_strategy_params.get('fraction', 0.8),
                trail_trigger_ticks=best_strategy_params.get('trigger', 1.0),
                trail_stop_ticks=best_strategy_params.get('trail', 0.5),
                max_hold_bars=MAX_HOLD_BARS,
                direction_preds=direction_preds,
            )

            if 'error' not in r:
                filled_count = sum(1 for p in paths if p.fill_bar >= 0)
                fill_rate = filled_count / max(len(paths), 1)
                logger.info(
                    f"  Gate>{gate:.1f}t Q={q:.0%}: "
                    f"Posted={len(paths):,} FillR={fill_rate:.1%} "
                    f"Trades={r['n_trades']} "
                    f"WR={r['win_rate']:.1%} "
                    f"PF={r['profit_factor']:.2f} "
                    f"Sharpe={r['sharpe_annualized']:.2f} "
                    f"$/t={r['mean_pnl_dollars']:+.2f}"
                )
                r['gate'] = gate
                r['quantile'] = q
                r['fill_rate'] = fill_rate
                results.append(r)

    results.sort(key=lambda r: r.get('sharpe_annualized', 0), reverse=True)
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_path_stats_report(stats: Dict) -> str:
    """Format path statistics into a readable report."""
    if 'error' in stats:
        return f"PATH STATS ERROR: {stats['error']}"

    lines = [
        "",
        "=" * 70,
        "MFE PATH ANALYSIS — KEY FINDINGS",
        "=" * 70,
        "",
        f"Total paths analyzed:    {stats['n_paths_total']:,}",
        f"Filled (limit entry):    {stats['n_filled']:,} ({stats['fill_rate']:.1%})",
        "",
        "MFE DISTRIBUTION (max favorable excursion):",
        f"  Mean:     {stats['mfe_mean']:.2f} ticks",
        f"  Median:   {stats['mfe_median']:.2f} ticks",
        f"  P25/P75:  {stats['mfe_p25']:.2f} / {stats['mfe_p75']:.2f} ticks",
        f"  P90:      {stats['mfe_p90']:.2f} ticks",
        f"  Hit 1t:   {stats['pct_mfe_ge_1t']:.1%}",
        f"  Hit 2t:   {stats['pct_mfe_ge_2t']:.1%}",
        f"  Hit 3t:   {stats['pct_mfe_ge_3t']:.1%}",
        "",
        "ADVERSE EXCURSION PROFILE:",
        f"  MAE mean:              {stats['mae_mean']:.2f} ticks",
        f"  MAE before MFE mean:   {stats['mae_before_mfe_mean']:.2f} ticks",
        f"  % with adverse first:  {stats['pct_adverse_before_favorable']:.1%}",
        f"  Fill adverse mean:     {stats['fill_adverse_mean']:.3f} ticks",
        f"  (adverse needed to trigger limit fill)",
        "",
        "TIME TO TARGET:",
        f"  Mean time to MFE:   {stats['time_to_mfe_mean_sec']:.1f}s ({stats['time_to_mfe_mean_bars']:.0f} bars)",
        f"  Median time to MFE: {stats['time_to_mfe_median_sec']:.1f}s",
        f"  P90 time to MFE:    {stats['time_to_mfe_p90_sec']:.1f}s",
        f"  Time to 1t:         {stats['time_to_1t_mean_sec']:.1f}s (hit rate: {stats['time_to_1t_hit_rate']:.1%})",
        f"  Time to 2t:         {stats['time_to_2t_mean_sec']:.1f}s (hit rate: {stats['time_to_2t_hit_rate']:.1%})",
        f"  Time to 3t:         {stats['time_to_3t_mean_sec']:.1f}s (hit rate: {stats['time_to_3t_hit_rate']:.1%})",
        "",
        "MAGNITUDE PREDICTION QUALITY (gated bars):",
        f"  Mean magnitude pred:  {stats['magnitude_pred_mean']:.2f} ticks",
        f"  Mean actual MFE:      {stats['mfe_mean']:.2f} ticks",
        f"  Actual/Predicted:     {stats['mfe_vs_pred_ratio']:.2f}x",
        "",
        "PATH TRAJECTORY (avg signed PnL, 10 bar samples up to 60 bars):",
    ]
    # Guard against short paths (horizon_bars < 60)
    n_traj = len(stats.get('avg_path_ticks_60bars', []))
    if n_traj > 0:
        sample_bars = [i * 10 for i in range(7) if i * 10 < n_traj]
        lines += [
            "  Bar:  " + "  ".join(f"{b:3d}" for b in sample_bars),
            "  Tick: " + "  ".join(f"{stats['avg_path_ticks_60bars'][b]:+.2f}" for b in sample_bars),
            "  %+:   " + "  ".join(f"{stats['pct_positive_60bars'][b]:.0%}" for b in sample_bars),
        ]
    lines += [
        "=" * 70,
    ]
    return "\n".join(lines)


def format_strategy_results(results: List[Dict]) -> str:
    """Format top strategy sweep results."""
    if not results:
        return "No valid strategy results."

    lines = [
        "",
        "=" * 100,
        "EXIT STRATEGY COMPARISON — TOP 15 BY SHARPE",
        "=" * 100,
        f"{'Strategy':<30s} {'N':>6s} {'WR':>6s} {'PF':>6s} {'Sharpe':>7s} "
        f"{'Total$':>9s} {'$/trade':>8s} {'Hold(s)':>7s}",
        "-" * 100,
    ]

    for r in results[:15]:
        params = r.get('params', {})
        strategy_type = params.get('type', r.get('strategy', '?'))

        # Build short name from params
        if strategy_type == 'fixed_target':
            name = f"FixedTP={params.get('target',0):.1f}t SL={params.get('stop','off')}"
        elif strategy_type == 'trailing':
            name = f"Trail(trig={params.get('trigger',0):.1f} trail={params.get('trail',0):.1f}t)"
        elif strategy_type == 'mfe_adaptive':
            name = f"MFE-adaptive({params.get('fraction',0):.1f}x mag)"
        elif strategy_type == 'signal_reversal':
            name = "Signal reversal"
        elif strategy_type == 'combined':
            name = (f"Combined(frac={params.get('fraction',0):.1f},"
                    f"trig={params.get('trigger',0):.1f},"
                    f"trail={params.get('trail',0):.1f}t)")
        else:
            name = strategy_type

        lines.append(
            f"{name:<30s} {r['n_trades']:>6,d} {r['win_rate']:>6.1%} "
            f"{r['profit_factor']:>6.2f} {r['sharpe_annualized']:>7.2f} "
            f"${r['total_pnl_dollars']:>+8,.0f} ${r['mean_pnl_dollars']:>+7.2f} "
            f"{r['mean_hold_sec']:>7.1f}s"
        )

    lines.append("=" * 100)

    # Best result details
    best = results[0]
    params = best.get('params', {})
    lines += [
        "",
        "BEST STRATEGY DETAILS:",
        f"  Strategy: {params}",
        f"  Trades: {best['n_trades']:,}  ({best['trades_per_day']:.1f}/day)",
        f"  Win Rate: {best['win_rate']:.1%}",
        f"  Profit Factor: {best['profit_factor']:.2f}",
        f"  Sharpe (annualized): {best['sharpe_annualized']:.2f}",
        f"  Total PnL: ${best['total_pnl_dollars']:+,.2f}",
        f"  Mean PnL/trade: ${best['mean_pnl_dollars']:+.2f} ({best['mean_pnl_ticks']:+.3f}t)",
        f"  Max Drawdown: ${best['max_drawdown_dollars']:,.2f}",
        f"  Mean Hold Time: {best['mean_hold_sec']:.1f}s",
        f"  Days: {best['n_positive_days']} win / {best['n_negative_days']} lose",
        f"  Daily PnL: ${best['day_pnl_mean']:+,.2f} mean, ${best['day_pnl_std']:,.2f} std",
        "",
        "  Exit type breakdown:",
    ]
    for et, d in best.get('exit_type_breakdown', {}).items():
        lines.append(
            f"    {et}: {d['count']:,} ({d['pct']:.1%}) "
            f"mean={d['mean_pnl_ticks']:+.2f}t "
            f"WR={d['win_rate']:.1%}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args):
    """Full MFE path analysis pipeline."""
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("=" * 80)
    logger.info("MFE PATH ANALYSIS — ADAPTIVE EXIT STRATEGY RESEARCH")
    logger.info("=" * 80)
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"N-days: {args.n_days}")
    logger.info(f"Quick mode: {getattr(args, 'quick', False)}")

    HORIZON_BARS = 50  # 5s at 100ms = 50 bars (direction/magnitude target horizon)

    # ================================================================
    # Load or train predictions
    # ================================================================
    if getattr(args, 'load_predictions', None):
        logger.info(f"\nLoading saved predictions: {args.load_predictions}")
        data = np.load(args.load_predictions)
        mid_prices = data['mid_prices']
        direction_preds = data['direction_preds']
        magnitude_preds = data['magnitude_preds']
        day_boundaries = data['day_boundaries'].tolist()
        logger.info(f"  Loaded {len(mid_prices):,} bars, "
                    f"{np.isfinite(direction_preds).sum():,} direction preds")

        # Optionally limit to n_days
        if args.n_days and len(day_boundaries) - 1 > args.n_days:
            cut = day_boundaries[args.n_days]
            mid_prices = mid_prices[:cut]
            direction_preds = direction_preds[:cut]
            magnitude_preds = magnitude_preds[:cut]
            day_boundaries = day_boundaries[:args.n_days + 1]
            logger.info(f"  Trimmed to {args.n_days} days ({len(mid_prices):,} bars)")

    else:
        feature_cache = args.feature_cache or str(
            LVL3_ROOT / "data" / "processed" / "mbo_features_cache"
        )
        logger.info(f"\nLoading data from: {feature_cache}")
        scanner, feature_names, load_info = load_data(feature_cache, args.n_days)

        logger.info(f"\nComputing targets (5s MFE net + magnitude)")
        direction_target, magnitude_target = compute_targets(
            scanner.mid_prices, HORIZON_BARS, scanner.day_boundaries
        )
        logger.info(f"  Direction target valid: {np.isfinite(direction_target).sum():,}")
        logger.info(f"  Magnitude target valid: {np.isfinite(magnitude_target).sum():,}")

        features = scanner.features
        mid_prices = scanner.mid_prices
        day_boundaries = scanner.day_boundaries

        # Train models
        logger.info(f"\n{'='*60}")
        logger.info("TRAINING DIRECTION MODEL")
        logger.info(f"{'='*60}")
        t0 = time.time()
        direction_preds, dir_ics = train_walk_forward(
            features, direction_target, day_boundaries,
            min_train_days=args.min_train_days,
            target_name='direction',
        )
        logger.info(f"  Direction model: {time.time()-t0:.0f}s | IC={np.mean(dir_ics):.4f}")
        gc.collect()

        logger.info(f"\n{'='*60}")
        logger.info("TRAINING MAGNITUDE MODEL")
        logger.info(f"{'='*60}")
        t0 = time.time()
        magnitude_preds, mag_ics = train_walk_forward(
            features, magnitude_target, day_boundaries,
            min_train_days=args.min_train_days,
            target_name='magnitude',
        )
        logger.info(f"  Magnitude model: {time.time()-t0:.0f}s | IC={np.mean(mag_ics):.4f}")
        gc.collect()

        # Save predictions for reuse
        pred_file = RESULTS_DIR / f"predictions_mfe_path_{timestamp}.npz"
        np.savez_compressed(
            str(pred_file),
            mid_prices=mid_prices,
            direction_preds=direction_preds,
            magnitude_preds=magnitude_preds,
            direction_target=direction_target,
            magnitude_target=magnitude_target,
            day_boundaries=np.array(day_boundaries),
        )
        logger.info(f"\nPredictions saved: {pred_file.name}")
        logger.info("(Reuse with --load-predictions to skip training)")

        del features, scanner
        gc.collect()

    # ================================================================
    # OOS split setup
    # ================================================================
    oos_split_day = getattr(args, 'oos_split_day', None)
    n_pred_days = len(day_boundaries) - 1
    if oos_split_day:
        if oos_split_day >= n_pred_days:
            logger.warning(f"--oos-split-day {oos_split_day} >= n_pred_days {n_pred_days}, "
                           "running full IS mode instead")
            oos_split_day = None
        else:
            logger.info(f"\n{'='*60}")
            logger.info(f"OOS SPLIT ENABLED: IS=days 1-{oos_split_day}, "
                        f"OOS=days {oos_split_day+1}-{n_pred_days}")
            logger.info(f"{'='*60}")

    # Helper to slice data by day range
    def _slice_data(start_day, end_day):
        s = day_boundaries[start_day]
        e = day_boundaries[end_day]
        return (mid_prices[s:e], direction_preds[s:e], magnitude_preds[s:e],
                [b - s for b in day_boundaries[start_day:end_day+1]])

    # ================================================================
    # Compute price paths for the main gate config
    # ================================================================
    if oos_split_day:
        is_mid, is_dir, is_mag, is_bounds = _slice_data(0, oos_split_day)
        oos_mid, oos_dir, oos_mag, oos_bounds = _slice_data(oos_split_day, n_pred_days)
        logger.info(f"IS data: {len(is_mid):,} bars, {len(is_bounds)-1} days")
        logger.info(f"OOS data: {len(oos_mid):,} bars, {len(oos_bounds)-1} days")
    else:
        is_mid, is_dir, is_mag, is_bounds = mid_prices, direction_preds, magnitude_preds, day_boundaries

    logger.info(f"\n{'='*60}")
    logger.info("COMPUTING PRICE PATHS — IN-SAMPLE (gate>2t, Q=0.80)")
    logger.info(f"{'='*60}")

    paths = compute_price_paths(
        is_mid, is_dir, is_mag, is_bounds,
        mag_gate_threshold=2.0,
        signal_quantile=0.80,
        latency_bars=1,
        fill_horizon_bars=50,
        min_bars_between_signals=50,
    )

    if not paths:
        logger.error("No paths computed! Check data and predictions.")
        return None

    # ================================================================
    # Path statistics (IS)
    # ================================================================
    logger.info(f"\n{'='*60}")
    logger.info("ANALYZING PATH STATISTICS (IS)")
    logger.info(f"{'='*60}")

    path_stats = analyze_path_statistics(paths)
    report = format_path_stats_report(path_stats)
    logger.info(report)

    # ================================================================
    # Strategy sweep (IS only — parameter selection)
    # ================================================================
    logger.info(f"\n{'='*60}")
    logger.info("RUNNING EXIT STRATEGY SWEEP (IS — param selection)")
    logger.info(f"{'='*60}")

    strategy_results = run_strategy_sweep(
        paths, is_dir,
        quick=getattr(args, 'quick', False),
    )

    strategy_report = format_strategy_results(strategy_results)
    logger.info(strategy_report)

    # ================================================================
    # Gate sweep with best strategy (IS only — param selection)
    # ================================================================
    if strategy_results:
        best_params = strategy_results[0].get('params', {'type': 'fixed_target', 'target': 2.0})
        logger.info(f"\n{'='*60}")
        logger.info(f"GATE SWEEP (IS) with best strategy: {best_params}")
        logger.info(f"{'='*60}")

        gate_results = run_gate_sweep(
            is_mid, is_dir, is_mag, is_bounds,
            best_strategy_params=best_params,
            quick=getattr(args, 'quick', False),
        )

        if gate_results:
            logger.info(f"\n{'='*60}")
            logger.info("GATE SWEEP TOP RESULTS (IS)")
            logger.info(f"{'='*60}")
            logger.info(f"{'Gate':>6s} {'Q':>5s} {'FillR':>6s} {'N':>6s} "
                        f"{'WR':>6s} {'PF':>6s} {'Sharpe':>7s} {'Total$':>9s} {'$/t':>7s}")
            logger.info("-" * 72)
            for r in gate_results[:10]:
                logger.info(
                    f"{r['gate']:>6.1f} {r['quantile']:>5.0%} {r['fill_rate']:>6.1%} "
                    f"{r['n_trades']:>6,d} {r['win_rate']:>6.1%} "
                    f"{r['profit_factor']:>6.2f} {r['sharpe_annualized']:>7.2f} "
                    f"${r['total_pnl_dollars']:>+8,.0f} ${r['mean_pnl_dollars']:>+6.2f}"
                )
    else:
        gate_results = []

    # ================================================================
    # OOS EVALUATION — frozen params from IS
    # ================================================================
    oos_results = {}
    if oos_split_day and strategy_results:
        logger.info(f"\n{'='*80}")
        logger.info("OUT-OF-SAMPLE EVALUATION — FROZEN PARAMS FROM IS")
        logger.info(f"{'='*80}")

        # Freeze best strategy and best gate config from IS
        frozen_strategy = strategy_results[0].get('params', {'type': 'fixed_target', 'target': 2.0})
        frozen_gate = gate_results[0] if gate_results else {'gate': 2.0, 'quantile': 0.80}
        frozen_gate_val = frozen_gate['gate']
        frozen_quantile = frozen_gate['quantile']

        logger.info(f"Frozen strategy: {frozen_strategy}")
        logger.info(f"Frozen gate: >{frozen_gate_val:.1f}t, Q={frozen_quantile:.0%}")

        # Compute OOS paths with frozen gate
        oos_paths = compute_price_paths(
            oos_mid, oos_dir, oos_mag, oos_bounds,
            mag_gate_threshold=frozen_gate_val,
            signal_quantile=frozen_quantile,
            latency_bars=1,
            fill_horizon_bars=50,
            min_bars_between_signals=50,
        )

        if oos_paths:
            oos_path_stats = analyze_path_statistics(oos_paths)
            oos_report = format_path_stats_report(oos_path_stats)
            logger.info(f"\n--- OOS Path Statistics ---")
            logger.info(oos_report)

            # Evaluate frozen strategy on OOS
            strategy_type = frozen_strategy.get('type', 'fixed_target')
            oos_strat_result = simulate_exit_strategy(
                oos_paths, strategy_type,
                target_ticks=frozen_strategy.get('target', 2.0),
                stop_ticks=frozen_strategy.get('stop', 9999) or 9999,
                mfe_fraction=frozen_strategy.get('fraction', 0.8),
                trail_trigger_ticks=frozen_strategy.get('trigger', 1.0),
                trail_stop_ticks=frozen_strategy.get('trail', 0.5),
                max_hold_bars=MAX_HOLD_BARS,
                direction_preds=oos_dir,
            )

            logger.info(f"\n--- OOS RESULTS (FROZEN PARAMS) ---")
            if 'error' not in oos_strat_result:
                logger.info(f"  Trades:     {oos_strat_result['n_trades']:,}")
                logger.info(f"  Win Rate:   {oos_strat_result['win_rate']:.1%}")
                logger.info(f"  PF:         {oos_strat_result['profit_factor']:.2f}")
                logger.info(f"  Sharpe:     {oos_strat_result['sharpe_annualized']:.2f}")
                logger.info(f"  Total PnL:  ${oos_strat_result['total_pnl_dollars']:+,.2f}")
                logger.info(f"  $/trade:    ${oos_strat_result['mean_pnl_dollars']:+.2f}")
                logger.info(f"  Days:       {oos_strat_result.get('n_positive_days', '?')} pos / "
                            f"{oos_strat_result.get('n_negative_days', '?')} neg")

                # Also run all gate configs on OOS with frozen strategy (for comparison)
                logger.info(f"\n--- OOS Gate Sensitivity (frozen strategy, varying gate/Q) ---")
                oos_gate_results = run_gate_sweep(
                    oos_mid, oos_dir, oos_mag, oos_bounds,
                    best_strategy_params=frozen_strategy,
                    quick=getattr(args, 'quick', False),
                )

                oos_results = {
                    'frozen_strategy': frozen_strategy,
                    'frozen_gate': frozen_gate_val,
                    'frozen_quantile': frozen_quantile,
                    'n_oos_days': len(oos_bounds) - 1,
                    'strategy_result': oos_strat_result,
                    'path_stats': {k: v for k, v in oos_path_stats.items()
                                   if not isinstance(v, np.ndarray)},
                    'gate_sensitivity': oos_gate_results[:10] if oos_gate_results else [],
                }
            else:
                logger.warning(f"  OOS strategy evaluation failed: {oos_strat_result['error']}")
        else:
            logger.warning("No OOS paths computed!")

    # ================================================================
    # Save results
    # ================================================================
    results_file = RESULTS_DIR / f"mfe_path_analysis_{timestamp}.json"

    serializable_path_stats = {
        k: v for k, v in path_stats.items()
        if not isinstance(v, np.ndarray)
    }

    save_data = {
        'timestamp': timestamp,
        'n_days': len(day_boundaries) - 1,
        'oos_split_day': oos_split_day,
        'n_paths': len(paths),
        'n_filled': sum(1 for p in paths if p.fill_bar >= 0),
        'path_statistics': serializable_path_stats,
        'strategy_results_top10': strategy_results[:10] if strategy_results else [],
        'gate_results_top10': gate_results[:10] if gate_results else [],
        'oos_results': oos_results if oos_results else None,
        'total_time_sec': time.time() - start_time,
    }

    with open(str(results_file), 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    logger.info(f"\nResults saved: {results_file.name}")
    logger.info(f"Log: {_log_file.name}")

    total_time = time.time() - start_time
    logger.info(f"\n{'='*80}")
    logger.info(f"PIPELINE COMPLETE — {total_time:.0f}s ({total_time/60:.1f}m)")
    logger.info(f"{'='*80}")

    # Final summary for Discord
    _print_discord_summary(path_stats, strategy_results, gate_results, total_time, oos_results)

    return save_data


def _print_discord_summary(path_stats: Dict, strategy_results: List, gate_results: List, elapsed: float, oos_results: Dict = None):
    """Print a compact Discord-ready summary."""
    lines = [
        "",
        "--- DISCORD SUMMARY ---",
        "**MFE PATH ANALYSIS COMPLETE**",
        f"Elapsed: {elapsed/60:.1f} min",
        "",
    ]

    if 'error' not in path_stats:
        lines += [
            "**PATH STATISTICS (filled trades):**",
            f"```",
            f"Fill rate:           {path_stats['fill_rate']:.1%} ({path_stats['n_filled']:,}/{path_stats['n_paths_total']:,})",
            f"Mean MFE:            {path_stats['mfe_mean']:.2f}t",
            f"MFE hit 1t/2t/3t:   {path_stats['pct_mfe_ge_1t']:.0%} / {path_stats['pct_mfe_ge_2t']:.0%} / {path_stats['pct_mfe_ge_3t']:.0%}",
            f"Time to MFE (mean):  {path_stats['time_to_mfe_mean_sec']:.1f}s",
            f"MAE before MFE:      {path_stats['mae_before_mfe_mean']:.2f}t",
            f"% adverse-first:     {path_stats['pct_adverse_before_favorable']:.0%}",
            f"Mag pred vs actual:  {path_stats['mfe_vs_pred_ratio']:.2f}x (actual/pred)",
            f"```",
            "",
        ]

    if strategy_results:
        best = strategy_results[0]
        params = best.get('params', {})
        lines += [
            "**BEST EXIT STRATEGY:**",
            f"```",
            f"Type:       {params.get('type', '?')}",
            f"Params:     {params}",
            f"Trades:     {best['n_trades']:,} ({best['trades_per_day']:.1f}/day)",
            f"Win Rate:   {best['win_rate']:.1%}",
            f"Prof Factor:{best['profit_factor']:.2f}",
            f"Sharpe:     {best['sharpe_annualized']:.2f}",
            f"Total PnL:  ${best['total_pnl_dollars']:+,.0f}",
            f"$/trade:    ${best['mean_pnl_dollars']:+.2f}",
            f"Hold time:  {best['mean_hold_sec']:.1f}s",
            f"```",
            "",
        ]

    if gate_results:
        best_gate = gate_results[0]
        lines += [
            "**BEST GATE CONFIG (IS):**",
            f"  Gate>{best_gate['gate']:.1f}t Q={best_gate['quantile']:.0%} -> "
            f"Sharpe={best_gate['sharpe_annualized']:.2f} "
            f"$/t=${best_gate['mean_pnl_dollars']:+.2f}",
        ]

    if oos_results and 'strategy_result' in oos_results:
        oos_r = oos_results['strategy_result']
        lines += [
            "",
            "**=== OUT-OF-SAMPLE RESULTS (FROZEN PARAMS) ===**",
            f"```",
            f"OOS Days:    {oos_results['n_oos_days']}",
            f"Frozen gate: >{oos_results['frozen_gate']:.1f}t Q={oos_results['frozen_quantile']:.0%}",
            f"Strategy:    {oos_results['frozen_strategy']}",
            f"Trades:      {oos_r['n_trades']:,} ({oos_r['trades_per_day']:.1f}/day)",
            f"Win Rate:    {oos_r['win_rate']:.1%}",
            f"Prof Factor: {oos_r['profit_factor']:.2f}",
            f"Sharpe:      {oos_r['sharpe_annualized']:.2f}",
            f"Total PnL:   ${oos_r['total_pnl_dollars']:+,.0f}",
            f"$/trade:     ${oos_r['mean_pnl_dollars']:+.2f}",
            f"Days:        {oos_r.get('n_positive_days', '?')} pos / {oos_r.get('n_negative_days', '?')} neg",
            f"```",
        ]

    summary = "\n".join(lines)
    print(summary)
    logger.info(summary)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='MFE Path Analysis — Adaptive Exit Strategy Research for ES Futures',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--n-days', type=int, default=None,
        help='Number of days to load (default: all available)',
    )
    parser.add_argument(
        '--feature-cache', type=str, default=None,
        help='Path to pre-computed feature cache directory',
    )
    parser.add_argument(
        '--load-predictions', type=str, default=None,
        help='Load saved predictions NPZ file (skip training)',
    )
    parser.add_argument(
        '--min-train-days', type=int, default=5,
        help='Minimum training days for walk-forward (default: 5)',
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Quick mode: fewer parameter combinations in sweep',
    )
    parser.add_argument(
        '--mag-gate', type=float, default=2.0,
        help='Magnitude gate threshold in ticks for path collection (default: 2.0)',
    )
    parser.add_argument(
        '--oos-split-day', type=int, default=None,
        help='OOS split: select params on days 1..N (IS), evaluate on days N+1..end (OOS). '
             'Example: --oos-split-day 50 uses first 50 prediction days for param selection, '
             'remaining days as true held-out OOS.',
    )
    args = parser.parse_args()

    try:
        results = run_pipeline(args)
        return results
    except Exception as e:
        logger.error(f"PIPELINE FAILED: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
