"""
Multi-Strategy Combination Tester for ES Futures MBO Alpha

Tests ALL promising strategy combinations: direction + magnitude gating +
time filters + vol filters + horizon stacking + different entry/exit methods.

APPROACH:
  Step 1: Load 340 features, 70 days from mbo_features_cache
  Step 2: Train walk-forward LightGBM models at 3s, 10s, 30s horizons
           (direction AND magnitude at each horizon)
  Step 3: Smart combinatorial testing — test each component alone first,
           then combine top-ranked components
  Step 4: Cost-adjusted backtest with realistic fill simulation
  Step 5: Rank all combos by OOS Sharpe, report top 10

COSTS:
  Limit entry: commission only = 0.24t
  Market entry: spread + commission = 1.24t (0.5t spread + 0.24t comm + 0.5t impact)
  Exit (any): commission portion only (in dir_pnl calc), always 0.5t spread for market

Usage:
    python alpha_discovery/strategy_combinator.py --n-days 70
    python alpha_discovery/strategy_combinator.py --n-days 70 --quick
    python alpha_discovery/strategy_combinator.py --load-predictions results/combinator_preds.npz
"""

import gc
import sys
import time
import json
import logging
import argparse
import platform
from pathlib import Path
from datetime import datetime
from itertools import combinations
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import spearmanr

LVL3_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(LVL3_ROOT))

# Cross-platform feature cache default
if platform.system() == 'Windows':
    DEFAULT_FEATURE_CACHE = str(LVL3_ROOT / "data" / "processed" / "mbo_features_cache")
else:
    DEFAULT_FEATURE_CACHE = str(Path.home() / "lvl3quant" / "data" / "processed" / "mbo_features_cache")

RESULTS_DIR = LVL3_ROOT / "alpha_discovery" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging
_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
_log_file = RESULTS_DIR / f"strategy_combinator_{_ts}.log"
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
logger = logging.getLogger("combinator")

# ============================================================================
# Constants — ES MES futures
# ============================================================================
TICK_SIZE = 0.25
TICK_VALUE = 12.50          # $12.50/tick for MES (or $62.50/tick for ES full)
COMMISSION_RT = 4.70  # HC #52: $4.70 RT (AMP)        # $3.00 round-trip
COMMISSION_TICKS = COMMISSION_RT / TICK_VALUE   # 0.24t
HALF_TICK = TICK_SIZE / 2   # 0.125
BARS_PER_SEC = 10           # 100ms bars

# Entry costs in ticks (round-trip, entry side only; exit costs in sim)
LIMIT_ENTRY_COST = COMMISSION_TICKS   # 0.24t (passive fill = 0 spread)
MARKET_ENTRY_COST = HALF_TICK + COMMISSION_TICKS  # 0.74t (0.5t spread + 0.24t)

# Exit costs (market exit always loses half spread)
MARKET_EXIT_SPREAD = HALF_TICK  # 0.5t

# RTH times in minutes from midnight ET
RTH_OPEN_MIN = 9 * 60 + 30   # 9:30 AM
RTH_CLOSE_MIN = 16 * 60       # 4:00 PM
OPEN_END_MIN = 10 * 60 + 30   # 10:30 AM (end of volatile open period)
PRIME_TIME_START_MIN = 10 * 60 + 30  # 10:30 AM
PRIME_TIME_END_MIN = 14 * 60 + 30    # 2:30 PM

HORIZONS_BARS = {
    '3s': 30,
    '10s': 100,
    '30s': 300,
}

# Early-stopping threshold for combos: if first 20 OOS days show Sharpe < -1.0, skip
EARLY_STOP_THRESHOLD = -1.0
EARLY_STOP_FOLDS = 20


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class Trade:
    day: int
    bar: int
    direction: int           # +1 long, -1 short
    entry_type: str          # 'limit' or 'market'
    entry_price: float
    exit_price: float
    exit_type: str           # 'fixed_hold', 'signal_flip', 'take_profit', 'trailing', 'eod'
    bars_held: int
    pnl_ticks: float
    pnl_dollars: float
    combo_id: str = ''


@dataclass
class ComboResult:
    combo_id: str
    components: List[str]
    entry_method: str
    exit_method: str
    horizon: str
    n_trades: int
    trades_per_day: float
    total_pnl_dollars: float
    avg_pnl_per_trade: float
    avg_pnl_ticks: float
    win_rate: float
    profit_factor: float
    sharpe: float
    avg_winner_ticks: float
    avg_loser_ticks: float
    win_loss_ratio: float
    positive_days: int
    total_oos_days: int
    max_daily_loss: float
    max_daily_win: float
    sharpe_first20: float    # early-stop signal: Sharpe on first 20 OOS days
    skipped: bool = False
    skip_reason: str = ''


# ============================================================================
# Step 1: Data Loading
# ============================================================================

def load_data(feature_cache_dir: str, n_days: Optional[int] = None):
    """
    Load pre-computed features and mid prices from cache.

    Memory-efficient: loads day-by-day directly into float16 arrays,
    avoiding the 20 GB float32 pre-allocation from load_precomputed_features.
    70 days x 234k bars x 340 features x 2 bytes = ~11 GB float16.
    """
    from pathlib import Path

    feat_dir = Path(feature_cache_dir)
    if not feat_dir.exists():
        raise ValueError(f"Feature cache dir not found: {feat_dir}")

    feat_files = sorted(feat_dir.glob('*_mbo_features.npz'))
    if not feat_files:
        raise ValueError(f"No feature cache files in {feat_dir}")

    # Find snapshot dir for mid_prices (sibling directory)
    snap_dir = feat_dir.parent / 'medium_snapshots_cache'
    if not snap_dir.exists():
        # Try same dir
        snap_dir = feat_dir

    snap_files = sorted(snap_dir.glob('*.npz'))
    snap_by_date = {sf.name[:10]: sf for sf in snap_files}

    logger.info(f"Found {len(feat_files)} feature cache files in {feat_dir}")
    logger.info(f"Found {len(snap_by_date)} snapshot files in {snap_dir}")

    # First pass: count total rows and get n_features
    first_data = np.load(str(feat_files[0]))
    n_base_features = first_data['mbo_features'].shape[1]
    first_rows = first_data['mbo_features'].shape[0]
    first_data.close()

    max_files = n_days if n_days else len(feat_files)
    est_total = first_rows * min(max_files, len(feat_files))

    mem_f16_gb = est_total * n_base_features * 2 / (1024**3)
    logger.info(f"Pre-allocating ~{est_total:,} rows x {n_base_features} cols "
                f"({mem_f16_gb:.1f} GB float16 estimated)")

    # If estimate > 3 GB, use a smaller initial allocation and grow as needed
    # This avoids OOM when system RAM is limited.
    if mem_f16_gb > 3.0:
        # Use a single-day estimate and grow dynamically
        init_alloc = first_rows * 5  # start with 5 days, grow as needed
        logger.info(f"  Large estimate — using dynamic allocation (start: {init_alloc:,} rows)")
        est_total = init_alloc

    # Pre-allocate as float16 (half the memory of float32)
    features = np.empty((est_total, n_base_features), dtype=np.float16)
    mid_prices = np.empty(est_total, dtype=np.float32)
    day_boundaries = [0]
    dates_loaded = []
    offset = 0
    days_loaded = 0

    for fpath in feat_files:
        if n_days is not None and days_loaded >= n_days:
            break

        date_str = fpath.name[:10]
        if date_str not in snap_by_date:
            # Try without snapshot requirement — skip mid prices for this day
            continue

        t0 = time.time()

        # Load features for this day
        data = np.load(str(fpath))
        feats = data['mbo_features']  # float32 from disk
        n_rows_day = feats.shape[0]
        data.close()

        # Load mid_prices
        snap_data = np.load(str(snap_by_date[date_str]), allow_pickle=True)
        mp = snap_data['mid_prices']
        snap_data.close()

        if len(mp) != n_rows_day or n_rows_day < 100:
            del feats, mp
            continue

        # Grow arrays if needed
        if offset + n_rows_day > features.shape[0]:
            new_size = int(features.shape[0] * 1.5)
            logger.info(f"  Resizing: {features.shape[0]:,} -> {new_size:,}")
            new_feats = np.empty((new_size, n_base_features), dtype=np.float16)
            new_feats[:offset] = features[:offset]
            features = new_feats
            new_mids = np.empty(new_size, dtype=np.float32)
            new_mids[:offset] = mid_prices[:offset]
            mid_prices = new_mids

        # Downcast to float16 on copy (clip first to avoid inf)
        np.clip(feats, -60000, 60000, out=feats)
        features[offset:offset + n_rows_day] = feats.astype(np.float16)
        mid_prices[offset:offset + n_rows_day] = mp

        del feats, mp

        day_boundaries.append(offset + n_rows_day)
        offset += n_rows_day
        days_loaded += 1
        dates_loaded.append(date_str)

        elapsed = time.time() - t0
        if days_loaded <= 5 or days_loaded % 10 == 0:
            logger.info(f"  [{days_loaded}] {date_str}: {n_rows_day:,} bars [{elapsed:.1f}s]")

    if not dates_loaded:
        raise ValueError("No valid feature files loaded")

    # Truncate to actual size
    if offset < features.shape[0]:
        features = features[:offset]
        mid_prices = mid_prices[:offset]

    gc.collect()

    logger.info(f"Loaded {len(dates_loaded)} days, {offset:,} bars, {n_base_features} features")
    logger.info(f"Features memory: {features.nbytes / 1e9:.1f} GB (float16)")
    logger.info(f"Date range: {dates_loaded[0]} to {dates_loaded[-1]}")

    # Return a simple namespace to match original scanner interface
    class DataHolder:
        pass

    holder = DataHolder()
    holder.features = features
    holder.mid_prices = mid_prices
    holder.day_boundaries = day_boundaries
    holder.feature_names = [f'feat_{i}' for i in range(n_base_features)]

    load_info = {
        'n_days': len(dates_loaded),
        'n_snapshots': offset,
        'n_features': n_base_features,
        'dates_loaded': dates_loaded,
    }

    return holder, load_info


# ============================================================================
# Step 2: Walk-Forward Training
# ============================================================================

def train_walk_forward_lgbm(
    features: np.ndarray,
    target: np.ndarray,
    day_boundaries: list,
    min_train_days: int = 5,
    max_train_days: int = 30,
    target_name: str = 'signal',
    objective: str = 'regression',
) -> Tuple[np.ndarray, list]:
    """
    Walk-forward LightGBM with rolling window.
    Returns full-length prediction array + per-fold ICs.

    NO in-sample optimization of thresholds — thresholds always computed
    from IS data only, applied to OOS.
    """
    import lightgbm as lgb

    N = len(target)
    n_days = len(day_boundaries) - 1
    full_preds = np.full(N, np.nan, dtype=np.float32)
    fold_ics = []

    is_binary = objective == 'binary'
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
        'objective': objective,
        'metric': 'auc' if is_binary else 'rmse',
    }

    MAX_TRAIN_SAMPLES = 500_000
    n_folds = 0
    t0 = time.time()

    for test_day in range(min_train_days, n_days):
        train_end_day = test_day - 1
        train_start_day = max(0, train_end_day - max_train_days + 1)
        ts = day_boundaries[train_start_day]
        te = day_boundaries[train_end_day + 1]
        vs = day_boundaries[test_day]
        ve = day_boundaries[test_day + 1]

        y_tr = target[ts:te]
        y_te = target[vs:ve]
        tr_valid = np.isfinite(y_tr)
        te_valid = np.isfinite(y_te)

        if tr_valid.sum() < 500 or te_valid.sum() < 100:
            continue

        valid_idx = np.where(tr_valid)[0]
        if len(valid_idx) > MAX_TRAIN_SAMPLES:
            rng = np.random.default_rng(seed=test_day)
            valid_idx = np.sort(rng.choice(valid_idx, MAX_TRAIN_SAMPLES, replace=False))

        X_tr = features[ts:te][valid_idx].astype(np.float32)
        y_tr_s = y_tr[valid_idx]
        if is_binary:
            y_tr_s = y_tr_s.astype(int)

        X_te = features[vs:ve][te_valid].astype(np.float32)
        y_te_s = y_te[te_valid]

        split = int(len(X_tr) * 0.8)

        try:
            if is_binary:
                model = lgb.LGBMClassifier(**params)
            else:
                model = lgb.LGBMRegressor(**params)

            model.fit(
                X_tr[:split], y_tr_s[:split],
                eval_set=[(X_tr[split:], y_tr_s[split:])],
                callbacks=[lgb.early_stopping(30, verbose=False)],
            )

            if is_binary:
                preds = model.predict_proba(X_te)[:, 1]
            else:
                preds = model.predict(X_te)

        except Exception as e:
            logger.warning(f"  [{target_name}] Fold {test_day} failed: {e}")
            continue
        finally:
            del X_tr, y_tr_s

        valid_pos = np.arange(vs, ve)[te_valid]
        n = min(len(valid_pos), len(preds))
        full_preds[valid_pos[:n]] = preds[:n].astype(np.float32)

        if len(preds) > 10:
            try:
                ic = float(spearmanr(preds, y_te_s)[0])
                if np.isfinite(ic):
                    fold_ics.append(ic)
            except Exception:
                pass

        n_folds += 1
        if n_folds % 10 == 0:
            logger.info(f"  [{target_name}] Fold {n_folds} (day {test_day}): "
                        f"IC={np.mean(fold_ics):.4f} [{time.time()-t0:.0f}s]")

        del X_te, preds
        gc.collect()

    mask = np.isfinite(full_preds) & np.isfinite(target)
    overall_ic = float(spearmanr(full_preds[mask], target[mask])[0]) if mask.sum() > 50 else 0
    logger.info(f"  [{target_name}] DONE: {n_folds} folds, IC={overall_ic:.4f} "
                f"[{time.time()-t0:.0f}s]")

    return full_preds, fold_ics


def compute_forward_return(mid_prices, horizon_bars, day_boundaries):
    """Forward return in ticks, NaN at day boundaries."""
    N = len(mid_prices)
    n_days = len(day_boundaries) - 1
    ret = np.full(N, np.nan, dtype=np.float32)

    for d in range(n_days):
        s, e = day_boundaries[d], day_boundaries[d + 1]
        dl = e - s
        if dl <= horizon_bars:
            continue
        ret[s:s + dl - horizon_bars] = (
            mid_prices[s + horizon_bars:e] - mid_prices[s:s + dl - horizon_bars]
        ) / TICK_SIZE

    return ret


def compute_magnitude_target(mid_prices, horizon_bars, day_boundaries):
    """Absolute move in ticks within horizon."""
    N = len(mid_prices)
    n_days = len(day_boundaries) - 1
    mag = np.full(N, np.nan, dtype=np.float32)

    future_mid = np.empty(N, dtype=np.float32)
    future_mid[:N - horizon_bars] = mid_prices[horizon_bars:]
    future_mid[N - horizon_bars:] = np.nan

    # NaN-fill day boundary crossings
    for d in range(n_days - 1):
        day_end = day_boundaries[d + 1]
        nan_start = max(day_boundaries[d], day_end - horizon_bars)
        future_mid[nan_start:day_end] = np.nan

    valid = np.isfinite(future_mid)
    mag[valid] = np.abs(future_mid[valid] - mid_prices[valid]) / TICK_SIZE
    return mag


def compute_realized_vol(mid_prices, day_boundaries, window_bars=300):
    """Rolling realized vol (std of returns) in ticks."""
    N = len(mid_prices)
    n_days = len(day_boundaries) - 1
    vol = np.full(N, np.nan, dtype=np.float32)

    for d in range(n_days):
        s, e = day_boundaries[d], day_boundaries[d + 1]
        day_prices = mid_prices[s:e]
        day_rets = np.diff(day_prices) / TICK_SIZE
        if len(day_rets) < window_bars:
            continue
        cum = np.cumsum(day_rets)
        cum2 = np.cumsum(day_rets ** 2)
        for i in range(window_bars, len(day_rets)):
            s_sum = cum[i] - cum[i - window_bars]
            s_sq = cum2[i] - cum2[i - window_bars]
            variance = (s_sq - s_sum ** 2 / window_bars) / window_bars
            vol[s + i + 1] = np.sqrt(max(variance, 0))

    return vol


def compute_time_of_day_minutes(features: np.ndarray) -> np.ndarray:
    """
    Extract time-of-day in minutes from midnight ET from feature array.
    Feature col 26 = hour_norm (0-1 mapped to 0-24 hours).
    Feature col 27 = minute_norm.
    We reconstruct: col26 * 24 * 60 + col27 * 60.

    Falls back to col 26 * 24 * 60 if col 27 not available.
    """
    hour_col = 26
    minute_col = 27

    if features.shape[1] > minute_col:
        hour_norm = features[:, hour_col].astype(np.float32)
        minute_norm = features[:, minute_col].astype(np.float32)
        # hour_norm is 0..1 spanning midnight to 23:59
        # Reconstruct: hours = hour_norm * 24, minutes = hour_norm*24*60 + minute_norm*60
        tod_minutes = hour_norm * 24.0 * 60.0 + minute_norm * 60.0
    else:
        hour_norm = features[:, hour_col].astype(np.float32)
        tod_minutes = hour_norm * 24.0 * 60.0

    return tod_minutes


def train_all_models(
    features: np.ndarray,
    mid_prices: np.ndarray,
    day_boundaries: list,
    min_train_days: int = 5,
) -> Dict[str, np.ndarray]:
    """
    Train walk-forward models for all 3 horizons x 2 targets (direction + magnitude).

    Returns dict with keys like 'dir_3s', 'mag_3s', 'dir_10s', 'mag_10s', 'dir_30s', 'mag_30s'.
    """
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING ALL HORIZON MODELS (walk-forward)")
    logger.info("=" * 70)

    predictions = {}

    for hz_name, hz_bars in HORIZONS_BARS.items():
        logger.info(f"\n--- Horizon: {hz_name} ({hz_bars} bars) ---")

        # Direction target = forward return in ticks
        logger.info(f"  Computing direction target ({hz_name})...")
        t0 = time.time()
        dir_target = compute_forward_return(mid_prices, hz_bars, day_boundaries)
        valid_dir = np.isfinite(dir_target).sum()
        logger.info(f"  Direction target: {valid_dir:,} valid [{time.time()-t0:.1f}s]")

        # Magnitude target = absolute move in ticks
        logger.info(f"  Computing magnitude target ({hz_name})...")
        mag_target = compute_magnitude_target(mid_prices, hz_bars, day_boundaries)
        valid_mag = np.isfinite(mag_target).sum()
        logger.info(f"  Magnitude target: {valid_mag:,} valid")

        # Train direction model
        logger.info(f"  Training direction model ({hz_name})...")
        dir_preds, dir_ics = train_walk_forward_lgbm(
            features, dir_target, day_boundaries,
            min_train_days=min_train_days,
            target_name=f'dir_{hz_name}',
            objective='regression',
        )
        predictions[f'dir_{hz_name}'] = dir_preds
        gc.collect()

        # Train magnitude model
        logger.info(f"  Training magnitude model ({hz_name})...")
        mag_preds, mag_ics = train_walk_forward_lgbm(
            features, mag_target, day_boundaries,
            min_train_days=min_train_days,
            target_name=f'mag_{hz_name}',
            objective='regression',
        )
        predictions[f'mag_{hz_name}'] = mag_preds
        gc.collect()

        del dir_target, mag_target
        gc.collect()

        logger.info(f"  [{hz_name}] dir IC={np.mean(dir_ics):.4f}, "
                    f"mag IC={np.mean(mag_ics):.4f}")

    return predictions


# ============================================================================
# Step 3: Strategy Components
# ============================================================================

def build_direction_signal(
    dir_preds: np.ndarray,
    quantile: float,
    day_boundaries: list,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert direction predictions to trade signals.

    Threshold computed from IS data only — no forward-looking bias.
    Returns: (signal_mask, direction_array) where signal_mask = bool array,
             direction = +1 (long) or -1 (short) or 0 (no signal).

    quantile: top/bottom X% of absolute direction predictions get signals.
    """
    N = len(dir_preds)
    n_days = len(day_boundaries) - 1
    signal_mask = np.zeros(N, dtype=bool)
    direction = np.zeros(N, dtype=np.int8)

    # Walk-forward: threshold from IS data
    for test_day in range(1, n_days):
        vs = day_boundaries[test_day]
        ve = day_boundaries[test_day + 1]

        # IS mask: all predictions before this test day
        is_mask = np.zeros(N, dtype=bool)
        is_mask[:vs] = True
        is_valid = is_mask & np.isfinite(dir_preds)

        if is_valid.sum() < 100:
            continue

        # Threshold from IS absolute values
        abs_preds_is = np.abs(dir_preds[is_valid])
        threshold = np.percentile(abs_preds_is, quantile * 100)

        # Apply to OOS
        oos_valid = np.isfinite(dir_preds[vs:ve])
        oos_abs = np.abs(dir_preds[vs:ve])

        for bi in range(ve - vs):
            gi = vs + bi
            if not np.isfinite(dir_preds[gi]):
                continue
            if oos_abs[bi] >= threshold:
                signal_mask[gi] = True
                direction[gi] = 1 if dir_preds[gi] > 0 else -1

    return signal_mask, direction


def build_magnitude_gate(
    mag_preds: np.ndarray,
    threshold_ticks: float,
    day_boundaries: list,
) -> np.ndarray:
    """
    Gate: only allow signals where predicted magnitude > threshold_ticks.
    Returns bool mask (True = allowed).
    """
    N = len(mag_preds)
    gate = np.zeros(N, dtype=bool)
    valid = np.isfinite(mag_preds)
    gate[valid] = mag_preds[valid] >= threshold_ticks
    return gate


def build_mtf_agreement(
    dir_preds_dict: Dict[str, np.ndarray],
    horizons: List[str],
    min_agree: int = 2,
    day_boundaries: list = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Multi-timeframe agreement signal.
    Requires min_agree horizons to agree on direction.

    Returns: (signal_mask, direction) — only fires when horizons agree.
    """
    N = len(list(dir_preds_dict.values())[0])

    # For each horizon, get sign of prediction (+1, -1, 0 if no pred)
    signs = []
    for hz in horizons:
        if hz not in dir_preds_dict:
            continue
        preds = dir_preds_dict[hz]
        sign = np.zeros(N, dtype=np.int8)
        valid = np.isfinite(preds)
        sign[valid & (preds > 0)] = 1
        sign[valid & (preds < 0)] = -1
        signs.append(sign)

    if not signs:
        return np.zeros(N, dtype=bool), np.zeros(N, dtype=np.int8)

    signs_arr = np.stack(signs, axis=1)  # (N, n_horizons)
    n_horizons = len(signs)

    # Count agreements per direction
    n_long = (signs_arr == 1).sum(axis=1)
    n_short = (signs_arr == -1).sum(axis=1)

    signal_mask = np.zeros(N, dtype=bool)
    direction = np.zeros(N, dtype=np.int8)

    long_agree = n_long >= min_agree
    short_agree = n_short >= min_agree

    signal_mask[long_agree] = True
    direction[long_agree] = 1
    signal_mask[short_agree] = True
    direction[short_agree] = -1

    return signal_mask, direction


def build_time_filter(
    features: np.ndarray,
    filter_type: str,
) -> np.ndarray:
    """
    Time-of-day filter masks.

    filter_type options:
      'skip_open'    — skip first hour 9:30-10:30
      'prime_only'   — only 10:30-14:30 (prime time)
      'all_rth'      — no filter (all RTH)
    """
    tod_minutes = compute_time_of_day_minutes(features)
    N = len(features)
    mask = np.ones(N, dtype=bool)

    if filter_type == 'skip_open':
        # Skip 9:30-10:30 (first 60 minutes)
        in_open = (tod_minutes >= RTH_OPEN_MIN) & (tod_minutes < OPEN_END_MIN)
        mask[in_open] = False

    elif filter_type == 'prime_only':
        # Only allow 10:30-14:30
        in_prime = (tod_minutes >= PRIME_TIME_START_MIN) & (tod_minutes < PRIME_TIME_END_MIN)
        mask = in_prime

    elif filter_type == 'all_rth':
        # No filter — all RTH bars allowed
        mask = np.ones(N, dtype=bool)

    return mask


def build_vol_filter(
    mid_prices: np.ndarray,
    day_boundaries: list,
    filter_type: str,
) -> np.ndarray:
    """
    Volatility regime filter.

    filter_type options:
      'skip_high_vol'  — skip Q5 (top 20% vol)
      'low_vol_only'   — only Q1-Q2 (bottom 40% vol)
      'all_vol'        — no filter
    """
    N = len(mid_prices)
    mask = np.ones(N, dtype=bool)

    if filter_type == 'all_vol':
        return mask

    # Compute rolling vol
    vol = compute_realized_vol(mid_prices, day_boundaries, window_bars=300)

    valid_vol = vol[np.isfinite(vol)]
    if len(valid_vol) < 100:
        return mask

    if filter_type == 'skip_high_vol':
        # Skip top 20% vol
        p80 = np.percentile(valid_vol, 80)
        high_vol = np.isfinite(vol) & (vol >= p80)
        mask[high_vol] = False

    elif filter_type == 'low_vol_only':
        # Only allow bottom 40% vol
        p40 = np.percentile(valid_vol, 40)
        low_vol = np.isfinite(vol) & (vol <= p40)
        mask = low_vol

    return mask


# ============================================================================
# Step 4: Cost-Adjusted Backtest Simulator
# ============================================================================

def simulate_strategy(
    mid_prices: np.ndarray,
    day_boundaries: list,
    signal_mask: np.ndarray,
    direction: np.ndarray,
    entry_method: str,         # 'limit' or 'market'
    exit_method: str,          # 'fixed_10s', 'fixed_20s', 'fixed_30s', 'fixed_60s',
                               # 'signal_flip', 'take_profit_1t', 'take_profit_2t',
                               # 'take_profit_3t', 'trailing_1t', 'trailing_2t'
    dir_preds_for_flip: Optional[np.ndarray] = None,  # needed for signal_flip
    min_bars_between: int = 50,     # cooldown bars
    fill_horizon_bars: int = 100,   # max wait for limit fill
    oos_start_day: int = 5,
    combo_id: str = '',
) -> Optional[ComboResult]:
    """
    Full cost-adjusted backtest.

    Entry methods:
      limit:  Post at bid (long) or ask (short). Cost = 0.24t commission only.
              Fill: wait up to fill_horizon_bars for mid to cross entry price.
      market: Immediate fill at mid +/- 0.5t spread. Cost = 0.74t total.

    Exit methods:
      fixed_Xs:      Hold X seconds then market exit.
      signal_flip:   Exit when direction prediction flips.
      take_profit_Nt: Take profit at N ticks (market exit).
      trailing_Nt:   Trailing stop N ticks from peak (market exit).

    Returns ComboResult or None if insufficient trades.
    """
    N = len(mid_prices)
    n_days = len(day_boundaries) - 1

    # Parse exit params
    tp_ticks = None
    trail_ticks = None
    fixed_hold_bars = None
    use_signal_flip = False

    if exit_method.startswith('fixed_'):
        secs = int(exit_method.split('_')[1].replace('s', ''))
        fixed_hold_bars = secs * BARS_PER_SEC
    elif exit_method == 'signal_flip':
        use_signal_flip = True
        fixed_hold_bars = 300  # max hold = 30s fallback
    elif exit_method.startswith('take_profit_'):
        tp_ticks = float(exit_method.split('_')[2].replace('t', ''))
        fixed_hold_bars = 300  # max hold fallback
    elif exit_method.startswith('trailing_'):
        trail_ticks = float(exit_method.split('_')[1].replace('t', ''))
        fixed_hold_bars = 300  # max hold fallback

    all_trades: List[Trade] = []
    daily_pnls: Dict[int, float] = {d: 0.0 for d in range(oos_start_day, n_days)}

    for day_idx in range(oos_start_day, n_days):
        day_start = day_boundaries[day_idx]
        day_end = day_boundaries[day_idx + 1]

        in_position = False
        pos_dir = 0
        pos_entry_price = 0.0
        pos_entry_bar = 0
        pos_entry_type = ''
        pos_peak_unrealized = 0.0  # for trailing stop
        next_allowed_bar = day_start

        bar = day_start
        while bar < day_end:
            mid = mid_prices[bar]

            # Manage open position
            if in_position:
                unrealized = pos_dir * (mid - pos_entry_price) / TICK_SIZE

                if trail_ticks is not None:
                    pos_peak_unrealized = max(pos_peak_unrealized, unrealized)

                should_exit = False
                exit_reason = ''

                # Check take profit
                if tp_ticks is not None and unrealized >= tp_ticks:
                    should_exit = True
                    exit_reason = 'take_profit'

                # Check trailing stop
                elif trail_ticks is not None and pos_peak_unrealized > 0:
                    if unrealized <= pos_peak_unrealized - trail_ticks:
                        should_exit = True
                        exit_reason = 'trailing'

                # Check fixed hold
                elif fixed_hold_bars is not None and (bar - pos_entry_bar) >= fixed_hold_bars:
                    should_exit = True
                    exit_reason = 'fixed_hold'

                # Check signal flip
                elif use_signal_flip and bar < day_end:
                    if (np.isfinite(dir_preds_for_flip[bar]) and
                            np.sign(dir_preds_for_flip[bar]) == -pos_dir and
                            abs(dir_preds_for_flip[bar]) > 0):
                        should_exit = True
                        exit_reason = 'signal_flip'

                # EOD close check
                if bar == day_end - 1:
                    should_exit = True
                    exit_reason = 'eod'

                if should_exit:
                    # Market exit: cost = spread (0.5t) on exit side
                    if pos_dir > 0:
                        exit_price = mid - HALF_TICK  # sell at bid
                    else:
                        exit_price = mid + HALF_TICK  # buy at ask

                    # PnL: from entry_price (which already accounts for entry method costs)
                    # to exit_price (which accounts for exit market spread)
                    dir_pnl = pos_dir * (exit_price - pos_entry_price) / TICK_SIZE
                    net_ticks = dir_pnl - COMMISSION_TICKS  # subtract commission RT

                    t = Trade(
                        day=day_idx,
                        bar=bar,
                        direction=pos_dir,
                        entry_type=pos_entry_type,
                        entry_price=pos_entry_price,
                        exit_price=exit_price,
                        exit_type=exit_reason,
                        bars_held=bar - pos_entry_bar,
                        pnl_ticks=net_ticks,
                        pnl_dollars=net_ticks * TICK_VALUE,
                        combo_id=combo_id,
                    )
                    all_trades.append(t)
                    daily_pnls[day_idx] += t.pnl_dollars

                    in_position = False
                    pos_dir = 0
                    next_allowed_bar = bar + min_bars_between

            # New signal entry (only when not in position, cooldown expired)
            if (not in_position and bar >= next_allowed_bar and
                    bar < day_end and signal_mask[bar] and direction[bar] != 0):

                sig_dir = int(direction[bar])

                if entry_method == 'limit':
                    # Post limit at bid (long) or ask (short)
                    if sig_dir > 0:
                        limit_price = mid - HALF_TICK   # buy at bid
                    else:
                        limit_price = mid + HALF_TICK   # sell at ask

                    # Try to fill within fill_horizon_bars
                    fill_end = min(bar + fill_horizon_bars, day_end - 1)
                    filled = False
                    fill_bar = bar

                    for fb in range(bar + 1, fill_end + 1):
                        future_mid = mid_prices[fb]
                        # BUG FIX: removed fill condition offset (audited 2026-02-25)
                        # WRONG was: future_mid <= limit_price + HALF_TICK (simplified to mid <= mid)
                        if sig_dir > 0 and future_mid <= limit_price:
                            filled = True
                            fill_bar = fb
                            break
                        elif sig_dir < 0 and future_mid >= limit_price:
                            filled = True
                            fill_bar = fb
                            break

                    if filled:
                        in_position = True
                        pos_dir = sig_dir
                        pos_entry_price = limit_price   # entry at bid/ask (not mid)
                        pos_entry_bar = fill_bar
                        pos_entry_type = 'limit'
                        pos_peak_unrealized = 0.0
                        bar = fill_bar  # advance to fill bar

                    # If not filled, no position opened, bar advances normally

                elif entry_method == 'market':
                    # Immediate fill, cross spread: cost embedded in entry_price
                    if sig_dir > 0:
                        entry_price = mid + HALF_TICK   # buy at ask
                    else:
                        entry_price = mid - HALF_TICK   # sell at bid

                    in_position = True
                    pos_dir = sig_dir
                    pos_entry_price = entry_price
                    pos_entry_bar = bar
                    pos_entry_type = 'market'
                    pos_peak_unrealized = 0.0

            bar += 1

        # Force close any open position at EOD if not already
        if in_position:
            mid = mid_prices[day_end - 1]
            if pos_dir > 0:
                exit_price = mid - HALF_TICK
            else:
                exit_price = mid + HALF_TICK
            dir_pnl = pos_dir * (exit_price - pos_entry_price) / TICK_SIZE
            net_ticks = dir_pnl - COMMISSION_TICKS
            t = Trade(
                day=day_idx, bar=day_end - 1, direction=pos_dir,
                entry_type=pos_entry_type, entry_price=pos_entry_price,
                exit_price=exit_price, exit_type='eod',
                bars_held=day_end - 1 - pos_entry_bar,
                pnl_ticks=net_ticks, pnl_dollars=net_ticks * TICK_VALUE,
                combo_id=combo_id,
            )
            all_trades.append(t)
            daily_pnls[day_idx] += t.pnl_dollars

    if len(all_trades) < 5:
        return None

    pnls = np.array([t.pnl_ticks for t in all_trades])
    dollars = np.array([t.pnl_dollars for t in all_trades])
    n_oos_days = n_days - oos_start_day
    daily_arr = np.array([daily_pnls[d] for d in range(oos_start_day, n_days)])

    wins = pnls > 0
    losses = pnls < 0
    win_rate = float(wins.mean())
    total_pnl = float(dollars.sum())
    avg_pnl = float(dollars.mean())

    gross_profit = float(pnls[wins].sum()) if wins.any() else 0
    gross_loss = abs(float(pnls[losses].sum())) if losses.any() else 0.001
    pf = gross_profit / gross_loss

    if n_oos_days > 1 and daily_arr.std() > 0:
        sharpe = float(daily_arr.mean() / daily_arr.std() * np.sqrt(252))
    else:
        sharpe = 0.0

    avg_winner = float(pnls[wins].mean()) if wins.any() else 0
    avg_loser = float(pnls[losses].mean()) if losses.any() else 0
    wl_ratio = abs(avg_winner / avg_loser) if avg_loser != 0 else 0

    # Early-stop signal: Sharpe on first 20 OOS days
    first20_days = min(20, n_oos_days)
    first20_arr = daily_arr[:first20_days]
    if first20_days > 1 and first20_arr.std() > 0:
        sharpe_first20 = float(first20_arr.mean() / first20_arr.std() * np.sqrt(252))
    else:
        sharpe_first20 = 0.0

    return ComboResult(
        combo_id=combo_id,
        components=[],
        entry_method=entry_method,
        exit_method=exit_method,
        horizon='',
        n_trades=len(all_trades),
        trades_per_day=len(all_trades) / max(n_oos_days, 1),
        total_pnl_dollars=total_pnl,
        avg_pnl_per_trade=avg_pnl,
        avg_pnl_ticks=float(pnls.mean()),
        win_rate=win_rate,
        profit_factor=pf,
        sharpe=sharpe,
        avg_winner_ticks=avg_winner,
        avg_loser_ticks=avg_loser,
        win_loss_ratio=wl_ratio,
        positive_days=int((daily_arr > 0).sum()),
        total_oos_days=n_oos_days,
        max_daily_loss=float(daily_arr.min()),
        max_daily_win=float(daily_arr.max()),
        sharpe_first20=sharpe_first20,
    )


# ============================================================================
# Step 5: Smart Combinatorial Testing
# ============================================================================

def run_component_solo_tests(
    mid_prices: np.ndarray,
    features: np.ndarray,
    day_boundaries: list,
    predictions: Dict[str, np.ndarray],
    oos_start_day: int = 5,
) -> List[ComboResult]:
    """
    Phase 1: Test each component individually at 30s horizon (our best).
    Ranks components by OOS Sharpe improvement over bare direction signal.
    """
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1: Individual Component Tests (30s horizon baseline)")
    logger.info("=" * 70)

    results = []
    primary_hz = '30s'
    dir_preds_30s = predictions[f'dir_{primary_hz}']
    mag_preds_30s = predictions[f'mag_{primary_hz}']

    # ---- A) Baseline: direction signal only, top 10% conviction ----
    logger.info("\nA) BASELINE: Direction top 10% — no filters, limit entry, 30s hold")
    signal_mask, direction = build_direction_signal(dir_preds_30s, quantile=0.90, day_boundaries=day_boundaries)
    result = simulate_strategy(
        mid_prices, day_boundaries, signal_mask, direction,
        entry_method='limit', exit_method='fixed_30s',
        oos_start_day=oos_start_day, combo_id='baseline_dir90_30s',
    )
    if result:
        result.components = ['dir_top10pct']
        result.horizon = primary_hz
        result.combo_id = 'baseline_dir90_30s'
        results.append(result)
        logger.info(f"  Sharpe={result.sharpe:.2f}, trades/day={result.trades_per_day:.1f}, "
                    f"WR={result.win_rate:.1%}, PF={result.profit_factor:.2f}, "
                    f"$={result.total_pnl_dollars:+,.0f}")
    else:
        logger.info("  No trades — insufficient signals")

    # ---- B) Direction thresholds: top 5%, 10%, 20% ----
    logger.info("\nB) Direction threshold sweep (top 5%, 10%, 20%)")
    for quantile, label in [(0.95, 'top5pct'), (0.90, 'top10pct'), (0.80, 'top20pct')]:
        signal_mask, direction = build_direction_signal(dir_preds_30s, quantile=quantile, day_boundaries=day_boundaries)
        result = simulate_strategy(
            mid_prices, day_boundaries, signal_mask, direction,
            entry_method='limit', exit_method='fixed_30s',
            oos_start_day=oos_start_day, combo_id=f'dir_{label}_30s',
        )
        if result:
            result.components = [f'dir_{label}']
            result.horizon = primary_hz
            result.combo_id = f'dir_{label}_30s'
            results.append(result)
            logger.info(f"  dir_{label}: Sharpe={result.sharpe:.2f}, "
                        f"trades/day={result.trades_per_day:.1f}, "
                        f"WR={result.win_rate:.1%}, $={result.total_pnl_dollars:+,.0f}")

    # ---- C) Magnitude gate: 2t, 3t, 4t ----
    logger.info("\nC) Magnitude gate sweep (2t, 3t, 4t) — combined with dir_top10pct")
    signal_mask_base, direction_base = build_direction_signal(
        dir_preds_30s, quantile=0.90, day_boundaries=day_boundaries)

    for mag_thresh in [2.0, 3.0, 4.0]:
        mag_gate = build_magnitude_gate(mag_preds_30s, mag_thresh, day_boundaries)
        combined_mask = signal_mask_base & mag_gate
        result = simulate_strategy(
            mid_prices, day_boundaries, combined_mask, direction_base,
            entry_method='limit', exit_method='fixed_30s',
            oos_start_day=oos_start_day, combo_id=f'mag_gate_{mag_thresh:.0f}t',
        )
        if result:
            result.components = ['dir_top10pct', f'mag_gate_{mag_thresh:.0f}t']
            result.horizon = primary_hz
            result.combo_id = f'mag_gate_{mag_thresh:.0f}t'
            results.append(result)
            logger.info(f"  mag_gate_{mag_thresh:.0f}t: Sharpe={result.sharpe:.2f}, "
                        f"trades/day={result.trades_per_day:.1f}, "
                        f"WR={result.win_rate:.1%}, $={result.total_pnl_dollars:+,.0f}")

    # ---- D) Multi-timeframe agreement ----
    logger.info("\nD) Multi-timeframe agreement (2/3, 3/3 horizons agree)")
    dir_preds_dict = {
        '3s': predictions['dir_3s'],
        '10s': predictions['dir_10s'],
        '30s': predictions['dir_30s'],
    }

    for min_agree in [2, 3]:
        mtf_mask, mtf_direction = build_mtf_agreement(
            dir_preds_dict, ['3s', '10s', '30s'], min_agree=min_agree)
        result = simulate_strategy(
            mid_prices, day_boundaries, mtf_mask, mtf_direction,
            entry_method='limit', exit_method='fixed_30s',
            oos_start_day=oos_start_day, combo_id=f'mtf_{min_agree}of3',
        )
        if result:
            result.components = [f'mtf_{min_agree}of3']
            result.horizon = 'multi'
            result.combo_id = f'mtf_{min_agree}of3'
            results.append(result)
            logger.info(f"  mtf_{min_agree}of3: Sharpe={result.sharpe:.2f}, "
                        f"trades/day={result.trades_per_day:.1f}, "
                        f"WR={result.win_rate:.1%}, $={result.total_pnl_dollars:+,.0f}")

    # ---- E) Time-of-day filters ----
    logger.info("\nE) Time-of-day filters (skip_open, prime_only)")
    for time_filter in ['skip_open', 'prime_only']:
        time_mask = build_time_filter(features, time_filter)
        combined_mask = signal_mask_base & time_mask
        result = simulate_strategy(
            mid_prices, day_boundaries, combined_mask, direction_base,
            entry_method='limit', exit_method='fixed_30s',
            oos_start_day=oos_start_day, combo_id=f'time_{time_filter}',
        )
        if result:
            result.components = ['dir_top10pct', f'time_{time_filter}']
            result.horizon = primary_hz
            result.combo_id = f'time_{time_filter}'
            results.append(result)
            logger.info(f"  time_{time_filter}: Sharpe={result.sharpe:.2f}, "
                        f"trades/day={result.trades_per_day:.1f}, "
                        f"WR={result.win_rate:.1%}, $={result.total_pnl_dollars:+,.0f}")

    # ---- F) Volatility filters ----
    logger.info("\nF) Volatility filters (skip_high_vol, low_vol_only)")
    for vol_filter in ['skip_high_vol', 'low_vol_only']:
        vol_mask = build_vol_filter(mid_prices, day_boundaries, vol_filter)
        combined_mask = signal_mask_base & vol_mask
        result = simulate_strategy(
            mid_prices, day_boundaries, combined_mask, direction_base,
            entry_method='limit', exit_method='fixed_30s',
            oos_start_day=oos_start_day, combo_id=f'vol_{vol_filter}',
        )
        if result:
            result.components = ['dir_top10pct', f'vol_{vol_filter}']
            result.horizon = primary_hz
            result.combo_id = f'vol_{vol_filter}'
            results.append(result)
            logger.info(f"  vol_{vol_filter}: Sharpe={result.sharpe:.2f}, "
                        f"trades/day={result.trades_per_day:.1f}, "
                        f"WR={result.win_rate:.1%}, $={result.total_pnl_dollars:+,.0f}")

    # ---- G) Entry methods ----
    logger.info("\nG) Entry methods (limit vs market)")
    for entry_method in ['limit', 'market']:
        result = simulate_strategy(
            mid_prices, day_boundaries, signal_mask_base, direction_base,
            entry_method=entry_method, exit_method='fixed_30s',
            oos_start_day=oos_start_day, combo_id=f'entry_{entry_method}',
        )
        if result:
            result.components = ['dir_top10pct', f'entry_{entry_method}']
            result.horizon = primary_hz
            result.combo_id = f'entry_{entry_method}'
            results.append(result)
            logger.info(f"  entry_{entry_method}: Sharpe={result.sharpe:.2f}, "
                        f"trades/day={result.trades_per_day:.1f}, "
                        f"WR={result.win_rate:.1%}, $={result.total_pnl_dollars:+,.0f}")

    # ---- H) Exit methods ----
    logger.info("\nH) Exit methods (fixed holds, signal_flip, take_profit, trailing)")
    exit_methods = [
        'fixed_10s', 'fixed_20s', 'fixed_30s', 'fixed_60s',
        'signal_flip',
        'take_profit_1t', 'take_profit_2t', 'take_profit_3t',
        'trailing_1t', 'trailing_2t',
    ]

    for exit_method in exit_methods:
        flip_preds = dir_preds_30s if exit_method == 'signal_flip' else None
        result = simulate_strategy(
            mid_prices, day_boundaries, signal_mask_base, direction_base,
            entry_method='limit', exit_method=exit_method,
            dir_preds_for_flip=flip_preds,
            oos_start_day=oos_start_day, combo_id=f'exit_{exit_method}',
        )
        if result:
            result.components = ['dir_top10pct', f'exit_{exit_method}']
            result.horizon = primary_hz
            result.combo_id = f'exit_{exit_method}'
            results.append(result)
            logger.info(f"  exit_{exit_method}: Sharpe={result.sharpe:.2f}, "
                        f"trades/day={result.trades_per_day:.1f}, "
                        f"WR={result.win_rate:.1%}, $={result.total_pnl_dollars:+,.0f}")

    return results


def rank_components(solo_results: List[ComboResult]) -> List[str]:
    """
    Rank component improvements by Sharpe lift over baseline.
    Returns list of component names sorted by Sharpe (descending).
    """
    baseline_sharpe = 0.0
    for r in solo_results:
        if r.combo_id == 'baseline_dir90_30s':
            baseline_sharpe = r.sharpe
            break

    logger.info(f"\nBaseline Sharpe: {baseline_sharpe:.2f}")
    logger.info("\nComponent Sharpe Lift:")

    component_sharpes = {}
    for r in solo_results:
        if r.combo_id == 'baseline_dir90_30s':
            continue
        lift = r.sharpe - baseline_sharpe
        component_sharpes[r.combo_id] = (r.sharpe, lift)
        logger.info(f"  {r.combo_id}: Sharpe={r.sharpe:.2f} (lift={lift:+.2f})")

    # Sort by absolute Sharpe (not lift — we want best absolute performers)
    ranked = sorted(component_sharpes.items(), key=lambda x: x[1][0], reverse=True)
    ranked_ids = [k for k, v in ranked]

    logger.info(f"\nTop components by Sharpe: {ranked_ids[:5]}")
    return ranked_ids


def run_combination_tests(
    mid_prices: np.ndarray,
    features: np.ndarray,
    day_boundaries: list,
    predictions: Dict[str, np.ndarray],
    solo_results: List[ComboResult],
    ranked_components: List[str],
    oos_start_day: int = 5,
    quick: bool = False,
) -> List[ComboResult]:
    """
    Phase 2: Combine top-ranked components.
    Test top 3-5 components in all pairwise and triple combinations.

    Early stopping: skip combo if first 20 OOS folds show Sharpe < -1.0
    """
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2: Combination Tests (top components)")
    logger.info("=" * 70)

    results = []

    dir_preds_30s = predictions['dir_30s']
    mag_preds_30s = predictions['mag_30s']
    dir_preds_dict = {hz: predictions[f'dir_{hz}'] for hz in HORIZONS_BARS}

    # Get top N components
    n_top = 3 if quick else 5
    top_components = ranked_components[:n_top]
    logger.info(f"Top {n_top} components to combine: {top_components}")

    # Helper: build signal mask from component list
    def build_signal_from_components(
        comp_list: List[str],
        dir_thresh_q: float = 0.90,
        entry_method: str = 'limit',
        exit_method: str = 'fixed_30s',
    ) -> Tuple[np.ndarray, np.ndarray, str, str]:
        """Construct combined signal mask from component IDs."""

        # Start with direction signal
        signal_mask, direction = build_direction_signal(
            dir_preds_30s, quantile=dir_thresh_q, day_boundaries=day_boundaries)

        # Apply each component
        for comp in comp_list:
            if comp == 'baseline_dir90_30s' or comp.startswith('baseline'):
                continue

            if comp.startswith('mag_gate_'):
                thresh = float(comp.split('_')[2].replace('t', ''))
                gate = build_magnitude_gate(mag_preds_30s, thresh, day_boundaries)
                signal_mask = signal_mask & gate

            elif comp.startswith('mtf_'):
                # e.g. 'mtf_2of3'
                parts = comp.split('_')[1]   # '2of3'
                min_agree = int(parts.split('of')[0])
                mtf_mask, mtf_dir = build_mtf_agreement(
                    dir_preds_dict, list(HORIZONS_BARS.keys()), min_agree=min_agree)
                signal_mask = signal_mask & mtf_mask
                # Use MTF direction when available, else keep dir_preds direction
                valid_mtf = mtf_mask & (mtf_dir != 0)
                direction[valid_mtf] = mtf_dir[valid_mtf]

            elif comp.startswith('time_'):
                ft = '_'.join(comp.split('_')[1:])
                t_mask = build_time_filter(features, ft)
                signal_mask = signal_mask & t_mask

            elif comp.startswith('vol_'):
                ft = '_'.join(comp.split('_')[1:])
                v_mask = build_vol_filter(mid_prices, day_boundaries, ft)
                signal_mask = signal_mask & v_mask

            elif comp.startswith('entry_'):
                entry_method = comp.split('_')[1]

            elif comp.startswith('exit_'):
                exit_method = '_'.join(comp.split('_')[1:])

            elif comp.startswith('dir_'):
                # Direction threshold override
                q_str = comp.split('_')[1]
                q_map = {'top5pct': 0.95, 'top10pct': 0.90, 'top20pct': 0.80}
                new_q = q_map.get(q_str, dir_thresh_q)
                signal_mask, direction = build_direction_signal(
                    dir_preds_30s, quantile=new_q, day_boundaries=day_boundaries)

        return signal_mask, direction, entry_method, exit_method

    # ---- Pairwise combinations of top components ----
    tested_combos = set()

    # Build candidate component lists (filter entry/exit for structured combos)
    filter_comps = [c for c in top_components if not c.startswith('entry_')
                    and not c.startswith('exit_') and not c.startswith('baseline')]
    entry_comps = [c for c in top_components if c.startswith('entry_')] or ['entry_limit']
    exit_comps = [c for c in top_components if c.startswith('exit_')] or ['exit_fixed_30s']

    # Best entry/exit from solo tests
    best_entry = 'limit'
    best_exit = 'fixed_30s'
    best_entry_sharpe = -999
    best_exit_sharpe = -999
    for r in solo_results:
        if r.combo_id.startswith('entry_') and r.sharpe > best_entry_sharpe:
            best_entry_sharpe = r.sharpe
            best_entry = r.entry_method
        if r.combo_id.startswith('exit_') and r.sharpe > best_exit_sharpe:
            best_exit_sharpe = r.sharpe
            best_exit = r.exit_method

    logger.info(f"\nBest entry from solo: {best_entry} (Sharpe={best_entry_sharpe:.2f})")
    logger.info(f"Best exit from solo: {best_exit} (Sharpe={best_exit_sharpe:.2f})")

    # Pairwise combinations
    logger.info("\n--- Pairwise component combinations ---")
    n_tested = 0
    for c1, c2 in combinations(filter_comps[:5], 2):
        combo_key = frozenset([c1, c2, best_entry, best_exit])
        if combo_key in tested_combos:
            continue
        tested_combos.add(combo_key)

        combo_name = f"{c1}_x_{c2}"
        logger.info(f"\n  Testing: {combo_name} + entry={best_entry} + exit={best_exit}")

        signal_mask, direction, entry_m, exit_m = build_signal_from_components(
            [c1, c2], entry_method=best_entry, exit_method=best_exit)
        flip_preds = dir_preds_30s if exit_m == 'signal_flip' else None

        result = simulate_strategy(
            mid_prices, day_boundaries, signal_mask, direction,
            entry_method=entry_m, exit_method=exit_m,
            dir_preds_for_flip=flip_preds,
            oos_start_day=oos_start_day, combo_id=combo_name,
        )

        if result:
            result.components = [c1, c2]
            result.entry_method = entry_m
            result.exit_method = exit_m
            result.horizon = '30s'
            result.combo_id = combo_name

            # Early stopping: skip if first20 Sharpe terrible
            if result.sharpe_first20 < EARLY_STOP_THRESHOLD and n_tested > EARLY_STOP_FOLDS:
                logger.info(f"    EARLY STOP: first20 Sharpe={result.sharpe_first20:.2f} < {EARLY_STOP_THRESHOLD}")
                result.skipped = True
                result.skip_reason = f'early_stop_sharpe_{result.sharpe_first20:.2f}'

            results.append(result)
            n_tested += 1
            logger.info(f"    Sharpe={result.sharpe:.2f}, trades/day={result.trades_per_day:.1f}, "
                        f"WR={result.win_rate:.1%}, PF={result.profit_factor:.2f}, "
                        f"$={result.total_pnl_dollars:+,.0f}, "
                        f"first20_Sharpe={result.sharpe_first20:.2f}")

    # Triple combinations (top 3 only, to limit compute)
    if not quick and len(filter_comps) >= 3:
        logger.info("\n--- Triple component combinations (top 3) ---")
        for c1, c2, c3 in combinations(filter_comps[:4], 3):
            combo_key = frozenset([c1, c2, c3, best_entry, best_exit])
            if combo_key in tested_combos:
                continue
            tested_combos.add(combo_key)

            combo_name = f"{c1}_x_{c2}_x_{c3}"
            logger.info(f"\n  Testing: {combo_name}")

            signal_mask, direction, entry_m, exit_m = build_signal_from_components(
                [c1, c2, c3], entry_method=best_entry, exit_method=best_exit)
            flip_preds = dir_preds_30s if exit_m == 'signal_flip' else None

            result = simulate_strategy(
                mid_prices, day_boundaries, signal_mask, direction,
                entry_method=entry_m, exit_method=exit_m,
                dir_preds_for_flip=flip_preds,
                oos_start_day=oos_start_day, combo_id=combo_name,
            )

            if result:
                result.components = [c1, c2, c3]
                result.entry_method = entry_m
                result.exit_method = exit_m
                result.horizon = '30s'
                result.combo_id = combo_name
                results.append(result)
                n_tested += 1
                logger.info(f"    Sharpe={result.sharpe:.2f}, trades/day={result.trades_per_day:.1f}, "
                            f"WR={result.win_rate:.1%}, $={result.total_pnl_dollars:+,.0f}")

    # ---- Cross-horizon: best filter combo at 10s and 3s horizons ----
    logger.info("\n--- Cross-horizon: apply best combo at 3s and 10s ---")
    if filter_comps:
        best_filter = filter_comps[0]  # top filter component

        for hz in ['10s', '3s']:
            dir_preds_hz = predictions[f'dir_{hz}']
            mag_preds_hz = predictions[f'mag_{hz}']

            hz_sig_mask, hz_direction = build_direction_signal(
                dir_preds_hz, quantile=0.90, day_boundaries=day_boundaries)

            # Apply best filter to hz signal
            if best_filter.startswith('mag_gate_'):
                thresh = float(best_filter.split('_')[2].replace('t', ''))
                gate = build_magnitude_gate(mag_preds_hz, thresh, day_boundaries)
                hz_sig_mask = hz_sig_mask & gate
            elif best_filter.startswith('time_'):
                ft = '_'.join(best_filter.split('_')[1:])
                t_mask = build_time_filter(features, ft)
                hz_sig_mask = hz_sig_mask & t_mask
            elif best_filter.startswith('vol_'):
                ft = '_'.join(best_filter.split('_')[1:])
                v_mask = build_vol_filter(mid_prices, day_boundaries, ft)
                hz_sig_mask = hz_sig_mask & v_mask

            hold_sec = int(hz.replace('s', ''))
            combo_name = f"hz_{hz}_best_filter_{hold_sec}s_hold"
            flip_preds = dir_preds_hz if best_exit == 'signal_flip' else None

            result = simulate_strategy(
                mid_prices, day_boundaries, hz_sig_mask, hz_direction,
                entry_method=best_entry, exit_method=f'fixed_{hold_sec}s',
                oos_start_day=oos_start_day, combo_id=combo_name,
            )
            if result:
                result.components = [f'dir_{hz}', best_filter]
                result.horizon = hz
                result.combo_id = combo_name
                results.append(result)
                logger.info(f"  {combo_name}: Sharpe={result.sharpe:.2f}, "
                            f"trades/day={result.trades_per_day:.1f}, "
                            f"WR={result.win_rate:.1%}, $={result.total_pnl_dollars:+,.0f}")

    # ---- Best entry + best exit + best filter combo ----
    logger.info("\n--- Full best combo: best filter + best entry + best exit ---")
    for filter_comp in filter_comps[:3]:
        for entry_m in ['limit', 'market']:
            for exit_m in ['fixed_30s', best_exit, 'take_profit_2t', 'trailing_1t']:
                combo_key = frozenset([filter_comp, f'entry_{entry_m}', f'exit_{exit_m}'])
                if combo_key in tested_combos:
                    continue
                tested_combos.add(combo_key)

                combo_name = f"{filter_comp}_entry_{entry_m}_exit_{exit_m}"
                signal_mask, direction, _, _ = build_signal_from_components(
                    [filter_comp], entry_method=entry_m, exit_method=exit_m)
                flip_preds = dir_preds_30s if exit_m == 'signal_flip' else None

                result = simulate_strategy(
                    mid_prices, day_boundaries, signal_mask, direction,
                    entry_method=entry_m, exit_method=exit_m,
                    dir_preds_for_flip=flip_preds,
                    oos_start_day=oos_start_day, combo_id=combo_name,
                )
                if result:
                    result.components = [filter_comp, f'entry_{entry_m}', f'exit_{exit_m}']
                    result.horizon = '30s'
                    result.combo_id = combo_name
                    results.append(result)
                    logger.info(f"  {combo_name}: Sharpe={result.sharpe:.2f}, "
                                f"trades/day={result.trades_per_day:.1f}, "
                                f"$={result.total_pnl_dollars:+,.0f}")

    return results


def print_final_report(all_results: List[ComboResult], log_path: Optional[str] = None):
    """Print comprehensive ranked report of all tested combos."""
    logger.info("\n" + "=" * 80)
    logger.info("FINAL REPORT — ALL COMBINATIONS RANKED BY OOS SHARPE")
    logger.info("=" * 80)

    valid = [r for r in all_results if not r.skipped and r.n_trades >= 5]

    if not valid:
        logger.warning("No valid results to report!")
        return

    valid.sort(key=lambda r: r.sharpe, reverse=True)

    # Header
    hdr = (f"{'Rank':>4} {'Combo ID':<45} {'Hz':>4} {'Entry':>6} {'Exit':<15} "
           f"{'Sharpe':>7} {'PF':>5} {'WR':>6} "
           f"{'Trades/d':>9} {'$/trade':>8} {'Total$':>10} "
           f"{'AvgW':>5} {'AvgL':>5} {'W/L':>5} {'POS%':>5}")
    logger.info(hdr)
    logger.info("-" * 160)

    for rank, r in enumerate(valid[:20], 1):
        logger.info(
            f"{rank:>4} {r.combo_id:<45} {r.horizon:>4} {r.entry_method:>6} "
            f"{r.exit_method:<15} "
            f"{r.sharpe:>7.2f} {r.profit_factor:>5.2f} {r.win_rate:>6.1%} "
            f"{r.trades_per_day:>9.1f} {r.avg_pnl_per_trade:>+8.2f} "
            f"{r.total_pnl_dollars:>+10,.0f} "
            f"{r.avg_winner_ticks:>+5.2f} {r.avg_loser_ticks:>+5.2f} "
            f"{r.win_loss_ratio:>5.2f} "
            f"{r.positive_days/max(r.total_oos_days,1):>5.1%}"
        )

    # Top 10 detailed breakdown
    logger.info(f"\n{'='*80}")
    logger.info("TOP 10 DETAILED BREAKDOWN")
    logger.info(f"{'='*80}")

    for rank, r in enumerate(valid[:10], 1):
        logger.info(f"\n#{rank}: {r.combo_id}")
        logger.info(f"  Components: {r.components}")
        logger.info(f"  Horizon: {r.horizon} | Entry: {r.entry_method} | Exit: {r.exit_method}")
        logger.info(f"  Sharpe: {r.sharpe:.2f} (first20 days: {r.sharpe_first20:.2f})")
        logger.info(f"  Profit Factor: {r.profit_factor:.2f}")
        logger.info(f"  Win Rate: {r.win_rate:.1%}")
        logger.info(f"  Trades/Day: {r.trades_per_day:.1f} ({r.n_trades} total)")
        logger.info(f"  Avg Winner: {r.avg_winner_ticks:+.2f}t | Avg Loser: {r.avg_loser_ticks:+.2f}t | W/L Ratio: {r.win_loss_ratio:.2f}x")
        logger.info(f"  Total PnL: ${r.total_pnl_dollars:+,.0f} (${r.avg_pnl_per_trade:+.2f}/trade)")
        logger.info(f"  Days: {r.positive_days}/{r.total_oos_days} positive "
                    f"({r.positive_days/max(r.total_oos_days,1):.1%})")
        logger.info(f"  Max Daily Loss: ${r.max_daily_loss:+,.0f} | Max Daily Win: ${r.max_daily_win:+,.0f}")

    # Component value-add analysis
    logger.info(f"\n{'='*80}")
    logger.info("COMPONENT VALUE-ADD ANALYSIS")
    logger.info("(Which components appear in top 10 vs bottom 10)")
    logger.info(f"{'='*80}")

    comp_in_top = {}
    comp_in_bottom = {}

    top10 = valid[:10]
    bottom10 = [r for r in valid if r.sharpe < 0][:10]

    for r in top10:
        for c in r.components:
            comp_in_top[c] = comp_in_top.get(c, 0) + 1

    for r in bottom10:
        for c in r.components:
            comp_in_bottom[c] = comp_in_bottom.get(c, 0) + 1

    all_comps = set(list(comp_in_top.keys()) + list(comp_in_bottom.keys()))
    logger.info(f"\n{'Component':<40} {'Top10 freq':>10} {'Neg freq':>10} {'Signal':>10}")
    for comp in sorted(all_comps, key=lambda c: comp_in_top.get(c, 0), reverse=True):
        top_freq = comp_in_top.get(comp, 0)
        bot_freq = comp_in_bottom.get(comp, 0)
        signal = "ADDS VALUE" if top_freq > bot_freq else ("NOISE" if top_freq == bot_freq else "HURTS")
        logger.info(f"  {comp:<40} {top_freq:>10} {bot_freq:>10} {signal:>10}")

    logger.info(f"\n{'='*80}")
    logger.info(f"Total combos tested: {len(all_results)}")
    logger.info(f"Viable combos (>=5 trades): {len(valid)}")
    logger.info(f"Profitable combos (Sharpe>0): {sum(1 for r in valid if r.sharpe > 0)}")
    logger.info(f"Best Sharpe: {valid[0].sharpe:.2f} ({valid[0].combo_id})")
    logger.info(f"Best $/day: ${max(r.total_pnl_dollars/max(r.total_oos_days,1) for r in valid):+,.0f}")

    return valid


# ============================================================================
# Main Pipeline
# ============================================================================

def send_discord_update(msg: str):
    """Send progress update to Discord."""
    try:
        import subprocess
        logger.info(f"[DISCORD] {msg}")
    except Exception:
        pass


def run_pipeline(args):
    """Full multi-strategy combination test pipeline."""
    total_start = time.time()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    logger.info("=" * 80)
    logger.info("MULTI-STRATEGY COMBINATION TESTER")
    logger.info(f"  n_days:     {args.n_days}")
    logger.info(f"  quick:      {args.quick}")
    logger.info(f"  feature_cache: {args.feature_cache}")
    logger.info("=" * 80)

    # ================================================================
    # Step 1: Load data
    # ================================================================
    logger.info("\n[STEP 1] Loading data...")

    if args.load_predictions:
        logger.info(f"Loading pre-computed predictions from {args.load_predictions}")
        data = np.load(args.load_predictions, allow_pickle=True)
        mid_prices = data['mid_prices']
        day_boundaries = list(data['day_boundaries'])
        features = data['features'] if 'features' in data else None
        predictions = {k: data[k] for k in data.files
                       if k not in ('mid_prices', 'day_boundaries', 'features')}
        logger.info(f"  Loaded {len(mid_prices):,} bars, {len(day_boundaries)-1} days")
        logger.info(f"  Predictions: {list(predictions.keys())}")

        if features is None:
            logger.warning("Features not in predictions file — time/vol filters unavailable")
            features = np.zeros((len(mid_prices), 50), dtype=np.float16)

    else:
        scanner, load_info = load_data(args.feature_cache, args.n_days)
        mid_prices = scanner.mid_prices
        day_boundaries = scanner.day_boundaries
        features = scanner.features  # float16

        n_days = load_info['n_days']
        elapsed = time.time() - total_start
        logger.info(f"  Data loaded: {n_days} days, {load_info['n_snapshots']:,} bars "
                    f"[{elapsed:.0f}s]")

        # ================================================================
        # Step 2: Train all models
        # ================================================================
        logger.info("\n[STEP 2] Training walk-forward models...")
        logger.info("  3 horizons (3s, 10s, 30s) x 2 targets (dir, mag) = 6 models")

        predictions = train_all_models(
            features, mid_prices, day_boundaries,
            min_train_days=args.min_train_days,
        )

        # Save predictions for re-use
        pred_file = RESULTS_DIR / f"combinator_preds_{timestamp}.npz"
        save_dict = {'mid_prices': mid_prices, 'day_boundaries': np.array(day_boundaries)}
        save_dict.update(predictions)
        # Don't save full features (too large), but save minimal feature subset for time/vol filters
        # Save the first 30 columns (including time features at col 26-27)
        feat_cols = min(30, features.shape[1])
        save_dict['features'] = features[:, :feat_cols].astype(np.float16)
        np.savez_compressed(str(pred_file), **save_dict)
        logger.info(f"  Predictions saved: {pred_file.name}")

        del scanner
        gc.collect()

    elapsed = time.time() - total_start
    logger.info(f"\n[STEP 2 DONE] Training complete [{elapsed:.0f}s / {elapsed/60:.1f}m]")

    # ================================================================
    # Step 3: Individual component tests
    # ================================================================
    logger.info("\n[STEP 3] Individual component tests...")

    oos_start_day = args.min_train_days  # OOS starts after min training days

    solo_results = run_component_solo_tests(
        mid_prices, features, day_boundaries, predictions,
        oos_start_day=oos_start_day,
    )

    elapsed = time.time() - total_start
    logger.info(f"\n[STEP 3 DONE] Solo tests complete: {len(solo_results)} results "
                f"[{elapsed:.0f}s / {elapsed/60:.1f}m]")

    # ================================================================
    # Step 4: Rank components and test combinations
    # ================================================================
    logger.info("\n[STEP 4] Ranking components and testing combinations...")

    ranked = rank_components(solo_results)

    combo_results = run_combination_tests(
        mid_prices, features, day_boundaries, predictions, solo_results, ranked,
        oos_start_day=oos_start_day,
        quick=args.quick,
    )

    elapsed = time.time() - total_start
    logger.info(f"\n[STEP 4 DONE] Combo tests complete: {len(combo_results)} results "
                f"[{elapsed:.0f}s / {elapsed/60:.1f}m]")

    # ================================================================
    # Step 5: Final report
    # ================================================================
    all_results = solo_results + combo_results

    valid = print_final_report(all_results)

    # Save JSON results
    results_file = RESULTS_DIR / f"strategy_combinator_{timestamp}.json"

    def result_to_dict(r: ComboResult) -> dict:
        return {
            'combo_id': r.combo_id,
            'components': r.components,
            'entry_method': r.entry_method,
            'exit_method': r.exit_method,
            'horizon': r.horizon,
            'n_trades': r.n_trades,
            'trades_per_day': round(r.trades_per_day, 2),
            'total_pnl_dollars': round(r.total_pnl_dollars, 2),
            'avg_pnl_per_trade': round(r.avg_pnl_per_trade, 2),
            'avg_pnl_ticks': round(r.avg_pnl_ticks, 4),
            'win_rate': round(r.win_rate, 4),
            'profit_factor': round(r.profit_factor, 3),
            'sharpe': round(r.sharpe, 3),
            'avg_winner_ticks': round(r.avg_winner_ticks, 4),
            'avg_loser_ticks': round(r.avg_loser_ticks, 4),
            'win_loss_ratio': round(r.win_loss_ratio, 3),
            'positive_days': r.positive_days,
            'total_oos_days': r.total_oos_days,
            'max_daily_loss': round(r.max_daily_loss, 2),
            'max_daily_win': round(r.max_daily_win, 2),
            'sharpe_first20': round(r.sharpe_first20, 3),
            'skipped': r.skipped,
            'skip_reason': r.skip_reason,
        }

    all_results_sorted = sorted(all_results, key=lambda r: r.sharpe, reverse=True)
    save_data = {
        'timestamp': timestamp,
        'n_days': args.n_days,
        'oos_start_day': oos_start_day,
        'total_combos_tested': len(all_results),
        'viable_combos': len([r for r in all_results if not r.skipped and r.n_trades >= 5]),
        'profitable_combos': len([r for r in all_results if r.sharpe > 0]),
        'total_time_sec': time.time() - total_start,
        'all_results': [result_to_dict(r) for r in all_results_sorted],
        'top10': [result_to_dict(r) for r in (valid or [])[:10]],
    }

    with open(str(results_file), 'w') as f:
        json.dump(save_data, f, indent=2)
    logger.info(f"\nResults saved: {results_file.name}")
    logger.info(f"Log: {_log_file.name}")

    total_elapsed = time.time() - total_start
    logger.info(f"\n{'='*80}")
    logger.info(f"PIPELINE COMPLETE — {total_elapsed:.0f}s ({total_elapsed/60:.1f}m)")
    logger.info(f"{'='*80}")

    return save_data


def main():
    parser = argparse.ArgumentParser(description='Multi-Strategy Combination Tester')
    parser.add_argument('--n-days', type=int, default=70,
                        help='Number of days to load (default: 70)')
    parser.add_argument('--feature-cache', type=str, default=DEFAULT_FEATURE_CACHE,
                        help='Feature cache directory')
    parser.add_argument('--min-train-days', type=int, default=5,
                        help='Minimum training days before OOS (default: 5)')
    parser.add_argument('--load-predictions', type=str, default=None,
                        help='Load pre-computed predictions NPZ (skip training)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: fewer combinations, faster sweep')
    args = parser.parse_args()

    if args.quick:
        args.n_days = min(args.n_days, 30)
        logger.info("[QUICK MODE] Limited to 30 days and reduced sweep")

    try:
        results = run_pipeline(args)
        return results
    except Exception as e:
        logger.error(f"PIPELINE FAILED: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
