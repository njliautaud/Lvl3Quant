"""
Continuation / Reversal Event Detector — Lvl3Quant

Concept: Detect large moves in progress (sweeps, institutional flow bursts,
momentum events) and classify whether they will CONTINUE or REVERSE.
Then ride continuations longer and fade reversals.

Pipeline:
  Step 1: Identify "event bars" — bars where something just happened
           * |price_change| > 2 ticks in last 10 bars (1 second)
           * OR aggressive_volume spike > 3 std above rolling mean
           * OR OFI > 90th percentile
  Step 2: Label each event bar: CONTINUATION(0) / REVERSAL(1) / NEUTRAL(2)
           * CONTINUATION: price moves 2+ ticks in SAME direction within 30s
           * REVERSAL: price moves 2+ ticks in OPPOSITE direction within 30s
           * NEUTRAL: price stays within 2 ticks of event bar
  Step 3: Train LightGBM multiclass classifier with walk-forward / purge gap
  Step 4: Backtest with fixed 30s hold + cost model ($3 RT)
  Step 5: Adaptive hold — ride continuations until momentum dies

Usage:
    python alpha_discovery/continuation_reversal.py --n-days 70
    python alpha_discovery/continuation_reversal.py --n-days 20 --quick
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
from typing import Optional, Dict, List, Tuple

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

_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
_log_file = RESULTS_DIR / f"cont_rev_{_ts}.log"
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
logger = logging.getLogger("cont_rev")

# ES futures constants
TICK_SIZE = 0.25
TICK_VALUE = 12.50
COMMISSION_RT = 4.70  # HC #52: $4.70 RT (AMP)
COMMISSION_TICKS = COMMISSION_RT / TICK_VALUE   # 0.24t
BARS_PER_SEC = 10
BARS_PER_MIN = 600

# Event detection thresholds
PRICE_MOVE_TICKS = 2.0          # |move| >= 2 ticks to qualify as "large"
PRICE_LOOKBACK_BARS = 10        # 1 second lookback for price change
VOL_SPIKE_STD = 3.0             # aggressive_volume > mean + 3*std
VOL_LOOKBACK_BARS = 300         # 30s rolling window for vol baseline
OFI_PERCENTILE = 90             # OFI spike threshold (per day)

# Labeling
LABEL_HORIZON_BARS = 300        # 30 seconds forward window
CONT_THRESH_TICKS = 2.0         # 2+ ticks to count as CONTINUATION/REVERSAL
LABEL_CONTINUATION = 0
LABEL_REVERSAL = 1
LABEL_NEUTRAL = 2

# Backtest
HOLD_BARS_FIXED = 300           # 30s fixed hold
MAX_FILL_WAIT_BARS = 100        # 10s max for limit order fill
COOLDOWN_BARS = 150             # 15s between trades


# ============================================================================
# STEP 1: EVENT DETECTION
# ============================================================================

def detect_event_bars(
    mid_prices: np.ndarray,
    features: np.ndarray,
    feature_names: List[str],
    day_boundaries: List[int],
) -> np.ndarray:
    """
    Identify bars where a significant move just happened.

    Returns boolean mask over all bars — True = event bar.

    Three conditions (OR):
      A. |price_change over last 10 bars| > 2 ticks
      B. aggressive_volume > rolling mean + 3*std (30s window)
      C. OFI > 90th percentile (computed per day)
    """
    N = len(mid_prices)
    n_days = len(day_boundaries) - 1
    is_event = np.zeros(N, dtype=bool)

    # Identify feature column indices we need
    fn = {name: idx for idx, name in enumerate(feature_names)}

    # aggressive_volume: sum of aggressive_buy_count + aggressive_sell_count
    agg_buy_col = fn.get('aggressive_buy_count', -1)
    agg_sell_col = fn.get('aggressive_sell_count', -1)
    # OFI: use trade_imbalance as OFI proxy (col 10 in standard feature set)
    # Prefer explicit OFI-like features if available
    ofi_col = fn.get('trade_imbalance', fn.get('aggressive_imbalance', 10))

    logger.info(f"  Event detection columns: agg_buy={agg_buy_col}, "
                f"agg_sell={agg_sell_col}, ofi={ofi_col} ({feature_names[ofi_col] if ofi_col < len(feature_names) else 'unknown'})")

    n_event_a = 0
    n_event_b = 0
    n_event_c = 0

    for d in range(n_days):
        s, e = day_boundaries[d], day_boundaries[d + 1]
        dl = e - s
        if dl < PRICE_LOOKBACK_BARS + LABEL_HORIZON_BARS:
            continue

        day_mid = mid_prices[s:e]

        # --- Condition A: Price move > 2 ticks in last 10 bars ---
        # |mid[i] - mid[i - 10]| > 2 * TICK_SIZE
        a_mask = np.zeros(dl, dtype=bool)
        if dl > PRICE_LOOKBACK_BARS:
            price_change = np.abs(
                day_mid[PRICE_LOOKBACK_BARS:] - day_mid[:-PRICE_LOOKBACK_BARS]
            ) / TICK_SIZE
            a_mask[PRICE_LOOKBACK_BARS:] = price_change >= PRICE_MOVE_TICKS
        n_event_a += a_mask.sum()

        # --- Condition B: Aggressive volume spike > 3 std ---
        b_mask = np.zeros(dl, dtype=bool)
        if agg_buy_col >= 0 and agg_sell_col >= 0 and features.shape[1] > max(agg_buy_col, agg_sell_col):
            agg_vol = (features[s:e, agg_buy_col].astype(np.float32) +
                       features[s:e, agg_sell_col].astype(np.float32))
            # Vectorized rolling mean and std via cumsum
            lb = VOL_LOOKBACK_BARS
            if dl > lb:
                cum = np.cumsum(agg_vol)
                cum2 = np.cumsum(agg_vol ** 2)
                # Rolling stats from position lb onward
                n = float(lb)
                s_sum = cum[lb:] - cum[:dl - lb]    # shape: (dl - lb,)
                s_sq  = cum2[lb:] - cum2[:dl - lb]
                mu    = s_sum / n
                var   = np.maximum((s_sq - s_sum ** 2 / n) / n, 0.0)
                sigma = np.sqrt(var)
                # Spike: current value > mean + 3*std, and sigma > 0
                spike = (agg_vol[lb:] > mu + VOL_SPIKE_STD * sigma) & (sigma > 0)
                b_mask[lb:] = spike
        n_event_b += b_mask.sum()

        # --- Condition C: OFI > 90th percentile (computed per day) ---
        c_mask = np.zeros(dl, dtype=bool)
        if ofi_col >= 0 and ofi_col < features.shape[1]:
            day_ofi = features[s:e, ofi_col].astype(np.float32)
            valid_ofi = day_ofi[np.isfinite(day_ofi)]
            if len(valid_ofi) > 100:
                threshold = np.percentile(np.abs(valid_ofi), OFI_PERCENTILE)
                if threshold > 0:
                    c_mask = np.abs(day_ofi) >= threshold

        n_event_c += c_mask.sum()

        # Combine: OR of all three conditions
        day_event = a_mask | b_mask | c_mask
        # Exclude edge bars (can't label or can't compute lookback)
        day_event[:PRICE_LOOKBACK_BARS] = False
        day_event[dl - LABEL_HORIZON_BARS:] = False

        is_event[s:e] = day_event

    n_events = is_event.sum()
    logger.info(f"  Events: {n_events:,} total")
    logger.info(f"    Condition A (price move): {n_event_a:,}")
    logger.info(f"    Condition B (vol spike):  {n_event_b:,}")
    logger.info(f"    Condition C (OFI spike):  {n_event_c:,}")
    logger.info(f"  Event rate: {n_events / N:.2%} of all bars")
    return is_event


# ============================================================================
# STEP 2: LABEL OUTCOMES
# ============================================================================

def label_event_outcomes(
    mid_prices: np.ndarray,
    is_event: np.ndarray,
    day_boundaries: List[int],
    horizon_bars: int = LABEL_HORIZON_BARS,
    thresh_ticks: float = CONT_THRESH_TICKS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each event bar, determine the move direction at event time
    and classify the forward outcome as CONTINUATION / REVERSAL / NEUTRAL.

    Returns:
        labels: int array (CONT=0, REV=1, NEUTRAL=2), NaN for non-events
        move_direction: float array (+1=up, -1=down) at event time
        move_size_ticks: float array, size of the triggering move in ticks
    """
    N = len(mid_prices)
    n_days = len(day_boundaries) - 1

    labels = np.full(N, np.nan, dtype=np.float32)
    move_direction = np.full(N, np.nan, dtype=np.float32)
    move_size_ticks = np.full(N, np.nan, dtype=np.float32)

    event_indices = np.where(is_event)[0]
    n_cont = 0
    n_rev = 0
    n_neut = 0

    # Build fast day_end lookup: for each bar index, store its day's end bar
    # Use searchsorted on day_boundaries to find which day each event belongs to
    db_arr = np.array(day_boundaries)

    for gi in event_indices:
        # Find day index using binary search on day_boundaries
        d = int(np.searchsorted(db_arr[1:], gi, side='right'))
        if d >= n_days:
            continue
        day_end = day_boundaries[d + 1]

        # Move direction: based on price change over last PRICE_LOOKBACK_BARS bars
        lookback_start = max(0, gi - PRICE_LOOKBACK_BARS)
        raw_move = (mid_prices[gi] - mid_prices[lookback_start]) / TICK_SIZE
        if abs(raw_move) < 0.01:
            # No clear directional move — neutral
            labels[gi] = LABEL_NEUTRAL
            move_direction[gi] = 0.0
            move_size_ticks[gi] = 0.0
            n_neut += 1
            continue

        direction = 1.0 if raw_move > 0 else -1.0
        move_size_ticks[gi] = abs(raw_move)
        move_direction[gi] = direction

        # Forward window — respecting day boundary
        fwd_end = min(gi + horizon_bars + 1, day_end)
        fwd_prices = mid_prices[gi + 1:fwd_end]

        if len(fwd_prices) < 5:
            labels[gi] = LABEL_NEUTRAL
            n_neut += 1
            continue

        # Max favorable excursion in the SAME direction (continuation)
        if direction > 0:
            cont_move = (np.max(fwd_prices) - mid_prices[gi]) / TICK_SIZE
            rev_move = (mid_prices[gi] - np.min(fwd_prices)) / TICK_SIZE
        else:
            cont_move = (mid_prices[gi] - np.min(fwd_prices)) / TICK_SIZE
            rev_move = (np.max(fwd_prices) - mid_prices[gi]) / TICK_SIZE

        # Check which happens first: 2-tick continuation vs 2-tick reversal
        # Walk bar by bar for first-hit semantics
        first_cont_bar = -1
        first_rev_bar = -1

        for j, fp in enumerate(fwd_prices):
            fwd_tick = (fp - mid_prices[gi]) / TICK_SIZE * direction
            if first_cont_bar < 0 and fwd_tick >= thresh_ticks:
                first_cont_bar = j
            if first_rev_bar < 0 and fwd_tick <= -thresh_ticks:
                first_rev_bar = j
            if first_cont_bar >= 0 and first_rev_bar >= 0:
                break

        if first_cont_bar >= 0 and (first_rev_bar < 0 or first_cont_bar < first_rev_bar):
            labels[gi] = LABEL_CONTINUATION
            n_cont += 1
        elif first_rev_bar >= 0 and (first_cont_bar < 0 or first_rev_bar < first_cont_bar):
            labels[gi] = LABEL_REVERSAL
            n_rev += 1
        else:
            labels[gi] = LABEL_NEUTRAL
            n_neut += 1

    n_labeled = n_cont + n_rev + n_neut
    logger.info(f"  Labels: {n_labeled:,} total events")
    if n_labeled > 0:
        logger.info(f"    CONTINUATION: {n_cont:,} ({n_cont/n_labeled:.1%})")
        logger.info(f"    REVERSAL:     {n_rev:,} ({n_rev/n_labeled:.1%})")
        logger.info(f"    NEUTRAL:      {n_neut:,} ({n_neut/n_labeled:.1%})")

    return labels, move_direction, move_size_ticks


# ============================================================================
# STEP 3: FEATURE ENGINEERING FOR EVENT BARS
# ============================================================================

def build_event_features(
    features: np.ndarray,
    feature_names: List[str],
    mid_prices: np.ndarray,
    is_event: np.ndarray,
    move_size_ticks: np.ndarray,
    move_direction: np.ndarray,
    day_boundaries: List[int],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build augmented feature matrix for event bars only.

    Adds event-specific context features:
      - move_size: magnitude of triggering move
      - move_speed: ticks per bar in the lookback window
      - volume_in_move: aggregate volume during the move
      - book_pressure_after: pressure imbalance at event bar
      - rel_bid_vol, rel_ask_vol: book thinning context
      - time_since_last_event: bars since last event (sparsity signal)
    """
    fn = {name: idx for idx, name in enumerate(feature_names)}

    event_indices = np.where(is_event)[0]
    n_events = len(event_indices)
    n_base = features.shape[1]

    # Extra context features
    extra_names = [
        'move_size_ticks',
        'move_speed_ticks_per_bar',
        'move_direction',
        'abs_move_size',
        'book_pressure_after',
        'time_since_last_event_bars',
        'vol_during_move',
        'bid_ask_imbalance_post',
    ]
    n_extra = len(extra_names)
    n_total = n_base + n_extra

    X = np.zeros((n_events, n_total), dtype=np.float32)
    y_indices = event_indices.copy()

    # Pressure imbalance col
    pressure_col = fn.get('pressure_imbalance', fn.get('vol_imbalance', 2))
    bid_vol_col = fn.get('total_bid_vol', 4)
    ask_vol_col = fn.get('total_ask_vol', 5)
    agg_buy_col = fn.get('aggressive_buy_count', -1)
    agg_sell_col = fn.get('aggressive_sell_count', -1)

    last_event_bar = -COOLDOWN_BARS
    for i, gi in enumerate(event_indices):
        # Base features at event bar
        X[i, :n_base] = features[gi]

        # Extra: move context
        ms = move_size_ticks[gi] if np.isfinite(move_size_ticks[gi]) else 0.0
        md = move_direction[gi] if np.isfinite(move_direction[gi]) else 0.0
        X[i, n_base + 0] = ms
        X[i, n_base + 1] = ms / max(PRICE_LOOKBACK_BARS, 1)  # speed
        X[i, n_base + 2] = md
        X[i, n_base + 3] = abs(ms)

        # Book pressure at event bar
        if pressure_col < features.shape[1]:
            X[i, n_base + 4] = features[gi, pressure_col]

        # Time since last event
        X[i, n_base + 5] = float(gi - last_event_bar)
        last_event_bar = gi

        # Volume during move (sum of agg vol over lookback)
        lb_start = max(0, gi - PRICE_LOOKBACK_BARS)
        if agg_buy_col >= 0 and agg_sell_col >= 0 and features.shape[1] > max(agg_buy_col, agg_sell_col):
            vol_sum = (features[lb_start:gi + 1, agg_buy_col].sum() +
                       features[lb_start:gi + 1, agg_sell_col].sum())
            X[i, n_base + 6] = float(vol_sum)

        # Bid/ask imbalance from book depth
        if bid_vol_col < features.shape[1] and ask_vol_col < features.shape[1]:
            bv = features[gi, bid_vol_col]
            av = features[gi, ask_vol_col]
            denom = bv + av
            if denom > 0:
                X[i, n_base + 7] = float((bv - av) / denom)

    all_names = list(feature_names) + extra_names
    logger.info(f"  Event feature matrix: {X.shape} ({n_extra} extra context features added)")
    return X, y_indices, all_names


# ============================================================================
# STEP 3: TRAIN CLASSIFIER (WALK-FORWARD)
# ============================================================================

def train_cont_rev_classifier(
    X_all: np.ndarray,
    y_all: np.ndarray,
    event_indices: np.ndarray,
    day_boundaries: List[int],
    min_train_days: int = 5,
    max_train_days: int = 30,
    purge_gap_bars: int = 600,  # 60s purge gap to avoid leakage
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
    """
    Walk-forward LightGBM multiclass (CONT=0 / REV=1 / NEUTRAL=2).

    Uses purge gap: training data ends purge_gap_bars before test day start
    to prevent forward-looking contamination from labels computed on overlapping
    30s windows.

    Returns:
        pred_probs: (n_events, 3) probability array — filled for OOS events only
        pred_classes: (n_events,) predicted class (NaN for IS events)
        fold_results: per-fold accuracy metrics
    """
    import lightgbm as lgb

    n_events = len(event_indices)
    n_days = len(day_boundaries) - 1

    pred_probs = np.full((n_events, 3), np.nan, dtype=np.float32)
    pred_classes = np.full(n_events, np.nan, dtype=np.float32)

    fold_results = []
    MAX_TRAIN_EVENTS = 200_000

    # Map each event to its day index
    event_day = np.searchsorted(day_boundaries[1:], event_indices, side='right')

    params = {
        'n_estimators': 400,
        'max_depth': 6,
        'learning_rate': 0.03,
        'subsample': 0.8,
        'colsample_bytree': 0.3,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'min_child_samples': 50,
        'verbose': -1,
        'n_jobs': 4,
        'device': 'cpu',
        'max_bin': 63,
        'force_row_wise': True,
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'class_weight': 'balanced',
    }

    t0 = time.time()
    n_folds = 0

    logger.info(f"  Walk-forward: {n_days} days, min_train={min_train_days}, "
                f"purge={purge_gap_bars} bars")

    for test_day in range(min_train_days, n_days):
        # Training: all events in [train_start_day, test_day-1] with purge
        train_end_day = test_day - 1
        train_start_day = max(0, train_end_day - max_train_days + 1)

        # Purge: exclude events within purge_gap_bars of test day start
        test_day_start_bar = day_boundaries[test_day]
        purge_cutoff_bar = test_day_start_bar - purge_gap_bars

        train_mask = (
            (event_day >= train_start_day) &
            (event_day <= train_end_day) &
            (event_indices < purge_cutoff_bar)
        )
        test_mask = event_day == test_day

        y_tr = y_all[train_mask]
        y_te = y_all[test_mask]

        tr_valid = np.isfinite(y_tr) & (y_tr >= 0)
        te_valid = np.isfinite(y_te) & (y_te >= 0)

        if tr_valid.sum() < 100 or te_valid.sum() < 10:
            continue

        idx_tr = np.where(train_mask)[0][tr_valid]
        idx_te = np.where(test_mask)[0][te_valid]

        # Subsample if needed
        if len(idx_tr) > MAX_TRAIN_EVENTS:
            rng = np.random.default_rng(seed=test_day)
            idx_tr = np.sort(rng.choice(idx_tr, MAX_TRAIN_EVENTS, replace=False))

        X_tr = X_all[idx_tr].astype(np.float32)
        y_tr_s = y_all[idx_tr].astype(int)
        X_te = X_all[idx_te].astype(np.float32)
        y_te_s = y_all[idx_te].astype(int)

        # Class balance: print distribution
        class_counts_tr = np.bincount(y_tr_s, minlength=3)

        split = int(len(X_tr) * 0.8)
        try:
            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_tr[:split], y_tr_s[:split],
                eval_set=[(X_tr[split:], y_tr_s[split:])],
                callbacks=[lgb.early_stopping(30, verbose=False)],
            )
            probs = model.predict_proba(X_te)  # (n_test, 3)
            pred_cls = np.argmax(probs, axis=1)

        except Exception as ex:
            logger.warning(f"  [fold {test_day}] Failed: {ex}")
            continue

        # Store predictions for OOS events
        pred_probs[idx_te] = probs.astype(np.float32)
        pred_classes[idx_te] = pred_cls.astype(np.float32)

        # Per-class accuracy
        per_class_acc = {}
        for cls in [LABEL_CONTINUATION, LABEL_REVERSAL, LABEL_NEUTRAL]:
            mask_cls = y_te_s == cls
            if mask_cls.sum() > 0:
                acc_cls = float((pred_cls[mask_cls] == cls).mean())
                per_class_acc[cls] = acc_cls

        overall_acc = float((pred_cls == y_te_s).mean())

        fold_info = {
            'day': test_day,
            'n_train': len(idx_tr),
            'n_test': len(idx_te),
            'overall_acc': overall_acc,
            'cont_acc': per_class_acc.get(LABEL_CONTINUATION, np.nan),
            'rev_acc': per_class_acc.get(LABEL_REVERSAL, np.nan),
            'neut_acc': per_class_acc.get(LABEL_NEUTRAL, np.nan),
            'train_dist': class_counts_tr.tolist(),
            'test_dist': np.bincount(y_te_s, minlength=3).tolist(),
        }
        fold_results.append(fold_info)
        n_folds += 1

        if n_folds % 10 == 0 or n_folds <= 3:
            cont_a = per_class_acc.get(LABEL_CONTINUATION, 0)
            rev_a = per_class_acc.get(LABEL_REVERSAL, 0)
            logger.info(f"  [fold {test_day}/{n_days}] "
                        f"n_tr={len(idx_tr):,} n_te={len(idx_te):,} "
                        f"OvAcc={overall_acc:.3f} "
                        f"CONT={cont_a:.3f} REV={rev_a:.3f} "
                        f"[{time.time()-t0:.0f}s]")

        del model, X_tr, X_te
        gc.collect()

    # Aggregate fold stats
    if fold_results:
        mean_acc = float(np.mean([f['overall_acc'] for f in fold_results]))
        mean_cont = float(np.nanmean([f['cont_acc'] for f in fold_results]))
        mean_rev = float(np.nanmean([f['rev_acc'] for f in fold_results]))
        mean_neut = float(np.nanmean([f['neut_acc'] for f in fold_results]))
        logger.info(f"\n  CLASSIFIER RESULTS ({n_folds} folds):")
        logger.info(f"    Overall accuracy:      {mean_acc:.3f}")
        logger.info(f"    CONTINUATION accuracy: {mean_cont:.3f}")
        logger.info(f"    REVERSAL accuracy:     {mean_rev:.3f}")
        logger.info(f"    NEUTRAL accuracy:      {mean_neut:.3f}")

    return pred_probs, pred_classes, fold_results


# ============================================================================
# STEP 4: BACKTEST — FIXED 30S HOLD
# ============================================================================

def backtest_fixed_hold(
    mid_prices: np.ndarray,
    event_indices: np.ndarray,
    pred_classes: np.ndarray,
    pred_probs: np.ndarray,
    labels: np.ndarray,
    move_direction: np.ndarray,
    day_boundaries: List[int],
    oos_start_day: int,
    hold_bars: int = HOLD_BARS_FIXED,
    min_confidence: float = 0.45,
    label: str = 'fixed_30s',
) -> Optional[Dict]:
    """
    Backtest with fixed hold period.

    Trading rules:
      CONTINUATION prediction: enter IN direction of move, limit entry, hold hold_bars
      REVERSAL prediction: enter AGAINST direction of move, limit entry, hold hold_bars
      NEUTRAL prediction: do nothing

    Cost model: $3 RT, limit entry (earns half spread), market exit.

    Only trades on OOS data. Min confidence filter applied.
    """
    N = len(mid_prices)
    n_days = len(day_boundaries) - 1

    # Build a full-bar lookup for event predictions
    # Map: bar_index -> (pred_class, pred_prob_cont, pred_prob_rev, move_dir)
    bar_event_info = {}
    for i, gi in enumerate(event_indices):
        if np.isnan(pred_classes[i]):
            continue
        if np.any(np.isnan(pred_probs[i])):
            continue
        bar_event_info[gi] = {
            'pred_class': int(pred_classes[i]),
            'p_cont': float(pred_probs[i, LABEL_CONTINUATION]),
            'p_rev': float(pred_probs[i, LABEL_REVERSAL]),
            'p_neut': float(pred_probs[i, LABEL_NEUTRAL]),
            'move_dir': float(move_direction[i]) if np.isfinite(move_direction[i]) else 0.0,
        }

    all_trades = []
    n_signals = 0
    n_skipped_confidence = 0

    for d in range(oos_start_day, n_days):
        s, e = day_boundaries[d], day_boundaries[d + 1]
        dl = e - s
        day_prices = mid_prices[s:e]

        in_position = False
        pos_dir = 0
        pos_fill_price = 0.0
        pos_fill_bar = -1
        pending = False
        pending_limit = 0.0
        pending_dir = 0
        pending_bar = -1
        cooldown = 0

        for bar in range(dl):
            gi = s + bar
            mid = day_prices[bar]
            bid = mid - TICK_SIZE / 2
            ask = mid + TICK_SIZE / 2

            if cooldown > 0:
                cooldown -= 1

            # Check pending limit fill
            if pending and not in_position:
                if bar - pending_bar > MAX_FILL_WAIT_BARS:
                    pending = False  # Limit order expired
                else:
                    # BUG FIX: removed fill condition offset (audited 2026-02-25)
                    # WRONG was: mid <= pending_limit + TICK_SIZE/2 (simplified to mid <= mid)
                    if pending_dir > 0 and mid <= pending_limit:
                        in_position = True
                        pos_dir = 1
                        pos_fill_price = pending_limit
                        pos_fill_bar = bar
                        pending = False
                    elif pending_dir < 0 and mid >= pending_limit:
                        in_position = True
                        pos_dir = -1
                        pos_fill_price = pending_limit
                        pos_fill_bar = bar
                        pending = False

            # Check exit
            if in_position:
                if bar - pos_fill_bar >= hold_bars:
                    # Market exit
                    exit_price = (bid if pos_dir > 0 else ask)
                    pnl_ticks = (pos_dir * (exit_price - pos_fill_price) / TICK_SIZE
                                 - COMMISSION_TICKS)
                    all_trades.append({
                        'day': d, 'bar': bar, 'direction': pos_dir,
                        'pnl_ticks': pnl_ticks,
                        'pnl_dollars': pnl_ticks * TICK_VALUE,
                        'bars_held': bar - pos_fill_bar,
                        'entry_type': 'hold_expired',
                    })
                    in_position = False
                    cooldown = COOLDOWN_BARS
                    continue

            # Check for new signal at this bar
            if not in_position and not pending and cooldown == 0 and gi in bar_event_info:
                info = bar_event_info[gi]
                pred_cls = info['pred_class']
                move_dir = info['move_dir']

                if move_dir == 0.0:
                    continue

                n_signals += 1

                # Confidence filter
                if pred_cls == LABEL_CONTINUATION:
                    confidence = info['p_cont']
                elif pred_cls == LABEL_REVERSAL:
                    confidence = info['p_rev']
                else:
                    confidence = 0.0  # NEUTRAL — skip

                if confidence < min_confidence:
                    n_skipped_confidence += 1
                    continue

                if pred_cls == LABEL_CONTINUATION:
                    trade_dir = int(np.sign(move_dir))
                elif pred_cls == LABEL_REVERSAL:
                    trade_dir = -int(np.sign(move_dir))
                else:
                    continue  # NEUTRAL

                # Place limit order
                pending = True
                pending_dir = trade_dir
                pending_limit = bid if trade_dir > 0 else ask
                pending_bar = bar

        # EOD forced exit
        if in_position:
            mid = day_prices[-1]
            eod_bid = mid - TICK_SIZE / 2
            eod_ask = mid + TICK_SIZE / 2
            exit_price = eod_bid if pos_dir > 0 else eod_ask
            pnl_ticks = pos_dir * (exit_price - pos_fill_price) / TICK_SIZE - COMMISSION_TICKS
            all_trades.append({
                'day': d, 'bar': dl - 1, 'direction': pos_dir,
                'pnl_ticks': pnl_ticks,
                'pnl_dollars': pnl_ticks * TICK_VALUE,
                'bars_held': dl - 1 - pos_fill_bar,
                'entry_type': 'eod_forced',
            })

    logger.info(f"\n  [{label}] Signals seen: {n_signals:,}, "
                f"skipped (low confidence): {n_skipped_confidence:,}")

    return _compute_sim_stats(all_trades, n_days, oos_start_day, day_boundaries, label)


# ============================================================================
# STEP 5: ADAPTIVE HOLD — RIDE THE CONTINUATION
# ============================================================================

def backtest_adaptive_hold(
    mid_prices: np.ndarray,
    event_indices: np.ndarray,
    pred_classes: np.ndarray,
    pred_probs: np.ndarray,
    labels: np.ndarray,
    move_direction: np.ndarray,
    day_boundaries: List[int],
    oos_start_day: int,
    min_confidence: float = 0.45,
    max_hold_bars: int = 3000,    # 5 min max
    stop_loss_ticks: float = 3.0,  # Stop at 3 ticks adverse
    momentum_window: int = 30,    # 3s for momentum check
    label: str = 'adaptive',
) -> Optional[Dict]:
    """
    Adaptive hold: enter on CONTINUATION signal, hold until momentum exhausts.

    Exit conditions (whichever comes first):
      1. Stop loss: 3 ticks adverse from fill
      2. Momentum reversal: short-term price change flips sign for 3s (30 bars)
         AND we're still net positive (protect profit)
      3. Max hold: 5 minutes
      4. EOD forced exit

    On REVERSAL signal: always use fixed hold (reversals tend to be shorter).
    On CONTINUATION: use adaptive hold.
    """
    bar_event_info = {}
    for i, gi in enumerate(event_indices):
        if np.isnan(pred_classes[i]):
            continue
        if np.any(np.isnan(pred_probs[i])):
            continue
        bar_event_info[gi] = {
            'pred_class': int(pred_classes[i]),
            'p_cont': float(pred_probs[i, LABEL_CONTINUATION]),
            'p_rev': float(pred_probs[i, LABEL_REVERSAL]),
            'p_neut': float(pred_probs[i, LABEL_NEUTRAL]),
            'move_dir': float(move_direction[i]) if np.isfinite(move_direction[i]) else 0.0,
        }

    all_trades = []
    n_signals = 0

    n_days = len(day_boundaries) - 1

    for d in range(oos_start_day, n_days):
        s, e = day_boundaries[d], day_boundaries[d + 1]
        dl = e - s
        day_prices = mid_prices[s:e]

        in_position = False
        pos_dir = 0
        pos_fill_price = 0.0
        pos_fill_bar = -1
        pos_is_adaptive = False  # True = adaptive hold, False = fixed 30s
        pending = False
        pending_limit = 0.0
        pending_dir = 0
        pending_bar = -1
        cooldown = 0

        for bar in range(dl):
            gi = s + bar
            mid = day_prices[bar]
            bid = mid - TICK_SIZE / 2
            ask = mid + TICK_SIZE / 2

            if cooldown > 0:
                cooldown -= 1

            # Check pending fill
            if pending and not in_position:
                if bar - pending_bar > MAX_FILL_WAIT_BARS:
                    pending = False
                else:
                    # BUG FIX: removed fill condition offset (audited 2026-02-25)
                    # WRONG was: mid <= pending_limit + TICK_SIZE/2 (simplified to mid <= mid)
                    if pending_dir > 0 and mid <= pending_limit:
                        in_position = True
                        pos_dir = 1
                        pos_fill_price = pending_limit
                        pos_fill_bar = bar
                        pending = False
                    elif pending_dir < 0 and mid >= pending_limit:
                        in_position = True
                        pos_dir = -1
                        pos_fill_price = pending_limit
                        pos_fill_bar = bar
                        pending = False

            # Check exit
            if in_position:
                bars_held = bar - pos_fill_bar
                current_pnl_ticks = pos_dir * (mid - pos_fill_price) / TICK_SIZE

                exit_now = False
                exit_reason = ''

                if not pos_is_adaptive:
                    # Fixed hold for reversals
                    if bars_held >= HOLD_BARS_FIXED:
                        exit_now = True
                        exit_reason = 'fixed_hold'
                else:
                    # Adaptive hold for continuations
                    # 1. Stop loss
                    if current_pnl_ticks <= -stop_loss_ticks:
                        exit_now = True
                        exit_reason = 'stop_loss'

                    # 2. Max hold
                    elif bars_held >= max_hold_bars:
                        exit_now = True
                        exit_reason = 'max_hold'

                    # 3. Momentum exhaustion: price flipped direction for >30 bars
                    # and we're in profit (protect gains)
                    elif bars_held >= momentum_window * 2 and current_pnl_ticks > COMMISSION_TICKS:
                        # Check if recent momentum is against position
                        lookback_bar = max(0, bar - momentum_window)
                        momentum_tick = pos_dir * (mid - day_prices[lookback_bar]) / TICK_SIZE
                        if momentum_tick < -1.0:
                            exit_now = True
                            exit_reason = 'momentum_exhausted'

                if exit_now:
                    exit_price = bid if pos_dir > 0 else ask
                    pnl_ticks = pos_dir * (exit_price - pos_fill_price) / TICK_SIZE - COMMISSION_TICKS
                    all_trades.append({
                        'day': d, 'bar': bar, 'direction': pos_dir,
                        'pnl_ticks': pnl_ticks,
                        'pnl_dollars': pnl_ticks * TICK_VALUE,
                        'bars_held': bars_held,
                        'entry_type': 'adaptive' if pos_is_adaptive else 'fixed',
                        'exit_reason': exit_reason,
                    })
                    in_position = False
                    cooldown = COOLDOWN_BARS
                    continue

            # New signal
            if not in_position and not pending and cooldown == 0 and gi in bar_event_info:
                info = bar_event_info[gi]
                pred_cls = info['pred_class']
                move_dir = info['move_dir']

                if move_dir == 0.0:
                    continue

                n_signals += 1

                if pred_cls == LABEL_CONTINUATION:
                    confidence = info['p_cont']
                    use_adaptive = True
                    trade_dir = int(np.sign(move_dir))
                elif pred_cls == LABEL_REVERSAL:
                    confidence = info['p_rev']
                    use_adaptive = False
                    trade_dir = -int(np.sign(move_dir))
                else:
                    continue

                if confidence < min_confidence:
                    continue

                pending = True
                pending_dir = trade_dir
                pending_limit = bid if trade_dir > 0 else ask
                pending_bar = bar
                pos_is_adaptive = use_adaptive  # Store for when position opens

        # EOD exit
        if in_position:
            mid = day_prices[-1]
            eod_bid = mid - TICK_SIZE / 2
            eod_ask = mid + TICK_SIZE / 2
            exit_price = eod_bid if pos_dir > 0 else eod_ask
            pnl_ticks = pos_dir * (exit_price - pos_fill_price) / TICK_SIZE - COMMISSION_TICKS
            all_trades.append({
                'day': d, 'bar': dl - 1, 'direction': pos_dir,
                'pnl_ticks': pnl_ticks,
                'pnl_dollars': pnl_ticks * TICK_VALUE,
                'bars_held': dl - 1 - pos_fill_bar,
                'entry_type': 'adaptive' if pos_is_adaptive else 'fixed',
                'exit_reason': 'eod_forced',
            })

    logger.info(f"\n  [{label}] Signals seen: {n_signals:,}")
    if all_trades:
        by_reason = {}
        for t in all_trades:
            r = t.get('exit_reason', 'unknown')
            by_reason[r] = by_reason.get(r, 0) + 1
        logger.info(f"    Exit reasons: {by_reason}")

    return _compute_sim_stats(all_trades, n_days, oos_start_day, day_boundaries, label)


# ============================================================================
# HELPER: COMPUTE SIM STATS
# ============================================================================

def _compute_sim_stats(
    all_trades: List[Dict],
    n_days: int,
    oos_start_day: int,
    day_boundaries: List[int],
    label: str,
) -> Optional[Dict]:
    """Compute and log backtest statistics."""
    if not all_trades:
        logger.warning(f"  [{label}] No trades executed")
        return None

    pnls = np.array([t['pnl_ticks'] for t in all_trades])
    dollars = np.array([t['pnl_dollars'] for t in all_trades])
    n_trades = len(all_trades)
    n_oos = n_days - oos_start_day

    wins = pnls > 0
    losses = pnls < 0

    total_pnl = float(dollars.sum())
    avg_pnl = float(dollars.mean())
    win_rate = float(wins.mean())
    pf = (abs(pnls[wins].sum() / pnls[losses].sum())
          if losses.sum() != 0 else float('inf'))

    daily_pnls = []
    for d in range(oos_start_day, n_days):
        day_trades = [t for t in all_trades if t['day'] == d]
        daily_pnls.append(sum(t['pnl_dollars'] for t in day_trades))
    daily_pnls = np.array(daily_pnls)

    sharpe = float(np.mean(daily_pnls) / max(np.std(daily_pnls), 1e-8) * np.sqrt(252))
    avg_hold = float(np.mean([t['bars_held'] for t in all_trades]))

    result = {
        'label': label,
        'n_trades': n_trades,
        'trades_per_day': round(n_trades / max(n_oos, 1), 1),
        'total_pnl_dollars': round(total_pnl, 2),
        'avg_pnl_per_trade': round(avg_pnl, 2),
        'avg_pnl_ticks': round(float(pnls.mean()), 3),
        'win_rate': round(win_rate, 4),
        'profit_factor': round(pf, 3),
        'sharpe': round(sharpe, 2),
        'avg_daily_pnl': round(float(daily_pnls.mean()), 2),
        'positive_days': int((daily_pnls > 0).sum()),
        'total_oos_days': n_oos,
        'avg_hold_bars': round(avg_hold, 1),
        'avg_hold_sec': round(avg_hold / BARS_PER_SEC, 1),
        'max_daily_loss': round(float(daily_pnls.min()), 2),
        'max_daily_win': round(float(daily_pnls.max()), 2),
    }

    logger.info(f"\n  [{label}] BACKTEST RESULTS:")
    logger.info(f"    Trades:      {n_trades} ({n_trades/max(n_oos,1):.1f}/day)")
    logger.info(f"    PnL:         ${total_pnl:+,.0f} total, ${avg_pnl:+.2f}/trade "
                f"({pnls.mean():+.3f} ticks)")
    logger.info(f"    Win Rate:    {win_rate:.1%}")
    logger.info(f"    Profit Fac:  {pf:.2f}")
    logger.info(f"    Sharpe:      {sharpe:.2f}")
    logger.info(f"    Daily avg:   ${daily_pnls.mean():+,.0f}/day  "
                f"Pos days: {(daily_pnls>0).sum()}/{n_oos}")
    logger.info(f"    Avg hold:    {avg_hold/BARS_PER_SEC:.1f}s")

    return result


# ============================================================================
# ORACLE ANALYSIS — UPPER BOUND
# ============================================================================

def oracle_analysis(
    labels: np.ndarray,
    event_indices: np.ndarray,
    move_direction: np.ndarray,
    mid_prices: np.ndarray,
    day_boundaries: List[int],
    oos_start_day: int,
) -> Dict:
    """
    Upper-bound analysis: how much profit is theoretically available
    if we had perfect CONT/REV predictions?

    This tells us the maximum edge available from this signal.
    """
    n_days = len(day_boundaries) - 1
    n_oos = n_days - oos_start_day

    cont_pnls = []
    rev_pnls = []

    for i, gi in enumerate(event_indices):
        d = np.searchsorted(day_boundaries[1:], gi, side='right')
        if d < oos_start_day:
            continue

        s, e = day_boundaries[d], day_boundaries[d + 1]
        lbl = labels[gi]
        md = move_direction[gi]

        if not np.isfinite(lbl) or not np.isfinite(md) or md == 0:
            continue

        fwd_end = min(gi + LABEL_HORIZON_BARS + 1, e)
        fwd_prices = mid_prices[gi + 1:fwd_end]
        if len(fwd_prices) < 5:
            continue

        if lbl == LABEL_CONTINUATION:
            # Enter in direction, exit at max favorable within window
            trade_dir = int(np.sign(md))
            if trade_dir > 0:
                best_exit = np.max(fwd_prices)
                pnl_ticks = (best_exit - mid_prices[gi]) / TICK_SIZE - COMMISSION_TICKS
            else:
                best_exit = np.min(fwd_prices)
                pnl_ticks = (mid_prices[gi] - best_exit) / TICK_SIZE - COMMISSION_TICKS
            cont_pnls.append(max(0, pnl_ticks))

        elif lbl == LABEL_REVERSAL:
            # Enter against direction, exit at max favorable within window
            trade_dir = -int(np.sign(md))
            if trade_dir > 0:
                best_exit = np.max(fwd_prices)
                pnl_ticks = (best_exit - mid_prices[gi]) / TICK_SIZE - COMMISSION_TICKS
            else:
                best_exit = np.min(fwd_prices)
                pnl_ticks = (mid_prices[gi] - best_exit) / TICK_SIZE - COMMISSION_TICKS
            rev_pnls.append(max(0, pnl_ticks))

    oracle_result = {
        'n_cont_events': len(cont_pnls),
        'n_rev_events': len(rev_pnls),
        'cont_avg_pnl_ticks': round(float(np.mean(cont_pnls)) if cont_pnls else 0, 3),
        'rev_avg_pnl_ticks': round(float(np.mean(rev_pnls)) if rev_pnls else 0, 3),
        'cont_total_pnl_dollars': round(float(np.sum(cont_pnls) * TICK_VALUE) if cont_pnls else 0, 2),
        'rev_total_pnl_dollars': round(float(np.sum(rev_pnls) * TICK_VALUE) if rev_pnls else 0, 2),
        'n_oos_days': n_oos,
    }

    logger.info(f"\n  ORACLE UPPER BOUND (OOS, perfect classification):")
    logger.info(f"    CONT events: {len(cont_pnls):,}, "
                f"avg={oracle_result['cont_avg_pnl_ticks']:.3f}t, "
                f"total=${oracle_result['cont_total_pnl_dollars']:+,.0f}")
    logger.info(f"    REV events:  {len(rev_pnls):,}, "
                f"avg={oracle_result['rev_avg_pnl_ticks']:.3f}t, "
                f"total=${oracle_result['rev_total_pnl_dollars']:+,.0f}")

    return oracle_result


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Continuation/Reversal Event Detector')
    parser.add_argument('--n-days', type=int, default=70)
    parser.add_argument('--feature-cache', type=str, default=DEFAULT_FEATURE_CACHE)
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: 20 days')
    parser.add_argument('--min-confidence', type=float, default=0.45,
                        help='Minimum classifier confidence to trade')
    args = parser.parse_args()

    if args.quick:
        args.n_days = min(args.n_days, 20)

    logger.info("=" * 70)
    logger.info("CONTINUATION / REVERSAL EVENT DETECTOR")
    logger.info(f"  n_days:         {args.n_days}")
    logger.info(f"  feature_cache:  {args.feature_cache}")
    logger.info(f"  min_confidence: {args.min_confidence}")
    logger.info(f"  quick:          {args.quick}")
    logger.info(f"  log:            {_log_file}")
    logger.info("=" * 70)

    t_total = time.time()

    # ------------------------------------------------------------------ #
    # LOAD DATA — memory-efficient float16 loader
    # ------------------------------------------------------------------ #
    logger.info("\n[STEP 0] Loading feature cache (memory-efficient float16 loader)...")
    from alpha_discovery.mbo_features import get_feature_names
    import psutil

    avail_gb = psutil.virtual_memory().available / 1e9
    logger.info(f"  Available RAM: {avail_gb:.1f} GB")

    feat_dir = Path(args.feature_cache)
    if not feat_dir.exists():
        raise ValueError(f"Feature cache dir not found: {feat_dir}")

    feat_files = sorted(feat_dir.glob('*_mbo_features.npz'))
    if not feat_files:
        raise ValueError(f"No feature cache files in {feat_dir}")

    # Snapshot dir: same parent, different folder
    snap_dir = feat_dir.parent / 'medium_snapshots_cache_v1_backup'
    if not snap_dir.exists():
        # Try alternate: standard snapshot cache name
        snap_dir = feat_dir.parent / 'medium_snapshots_cache'
    snap_by_date = {}
    if snap_dir.exists():
        for sf in sorted(snap_dir.glob('*.npz')):
            snap_by_date[sf.name[:10]] = sf

    # Limit to n_days
    feat_files = feat_files[:args.n_days]
    logger.info(f"  Found {len(feat_files)} feature files, loading up to {args.n_days}")

    # Pre-scan first file for shape
    first_data = np.load(str(feat_files[0]))
    n_base_features = first_data['mbo_features'].shape[1]
    first_rows = first_data['mbo_features'].shape[0]
    first_data.close()

    # Memory estimate for float16
    est_total = first_rows * len(feat_files)
    mem_gb_f16 = est_total * n_base_features * 2 / 1e9
    logger.info(f"  Estimated: {est_total:,} rows x {n_base_features} cols = {mem_gb_f16:.1f} GB (float16)")

    # Two-pass load: first pass counts rows per day, second pass copies into pre-allocated array
    # This avoids the 2x memory peak of list-accumulate + concatenate.
    logger.info(f"  Pass 1: counting rows...")
    day_row_counts = []
    valid_feat_files = []
    valid_snap_dates = []
    for fpath in feat_files:
        date_str = fpath.name[:10]
        data = np.load(str(fpath))
        n_rows_day = data['mbo_features'].shape[0]
        data.close()
        # Check snapshot exists
        if n_rows_day < 100:
            continue
        day_row_counts.append(n_rows_day)
        valid_feat_files.append(fpath)
        valid_snap_dates.append(date_str)

    total_rows = sum(day_row_counts)
    mem_gb_f16 = total_rows * n_base_features * 2 / 1e9
    mem_gb_mid = total_rows * 4 / 1e9
    logger.info(f"  Pass 1 done: {len(valid_feat_files)} valid days, {total_rows:,} rows")
    logger.info(f"  Pre-allocating: {mem_gb_f16:.1f} GB (float16 features) + {mem_gb_mid:.2f} GB (mid prices)")

    features = np.empty((total_rows, n_base_features), dtype=np.float16)
    mid_prices = np.empty(total_rows, dtype=np.float32)
    day_boundaries_list = [0]
    days_loaded = 0
    offset = 0

    logger.info(f"  Pass 2: loading data...")
    for fpath, date_str, n_rows_day in zip(valid_feat_files, valid_snap_dates, day_row_counts):
        t0_d = time.time()

        # Load features and copy directly into pre-allocated array
        data = np.load(str(fpath))
        raw_feats = data['mbo_features']
        np.clip(raw_feats, -60000, 60000, out=raw_feats)
        features[offset:offset + n_rows_day] = raw_feats.astype(np.float16)
        data.close()
        del raw_feats

        # Load mid prices from snapshot cache
        if date_str in snap_by_date:
            snap_data = np.load(str(snap_by_date[date_str]), allow_pickle=True)
            mp = snap_data['mid_prices'].astype(np.float32)
            snap_data.close()
            if len(mp) == n_rows_day:
                mid_prices[offset:offset + n_rows_day] = mp
            else:
                # Fallback: use col 0 (mid price stored in features)
                mid_prices[offset:offset + n_rows_day] = features[offset:offset + n_rows_day, 0].astype(np.float32)
            del mp
        else:
            mid_prices[offset:offset + n_rows_day] = features[offset:offset + n_rows_day, 0].astype(np.float32)

        offset += n_rows_day
        day_boundaries_list.append(offset)
        days_loaded += 1

        if days_loaded <= 5 or days_loaded % 10 == 0:
            avail = psutil.virtual_memory().available / 1e9
            logger.info(f"  [{days_loaded}/{len(valid_feat_files)}] {date_str}: "
                        f"{n_rows_day:,} bars [{time.time()-t0_d:.1f}s] "
                        f"RAM avail: {avail:.1f}GB")

    gc.collect()

    day_boundaries = day_boundaries_list
    n_days = len(day_boundaries) - 1
    feature_names = get_feature_names()

    # Trim/extend feature_names to match actual number of features loaded
    n_feat = features.shape[1]
    if len(feature_names) > n_feat:
        feature_names = feature_names[:n_feat]
    elif len(feature_names) < n_feat:
        feature_names = feature_names + [f'feat_{i}' for i in range(len(feature_names), n_feat)]

    oos_start_day = max(5, int(n_days * 0.7))
    n_oos = n_days - oos_start_day

    logger.info(f"Loaded {n_days} days, {len(mid_prices):,} bars, {n_feat} features (float16)")
    logger.info(f"OOS start: day {oos_start_day} ({n_oos} OOS days)")
    gc.collect()

    # ------------------------------------------------------------------ #
    # STEP 1: DETECT EVENTS
    # ------------------------------------------------------------------ #
    logger.info("\n[STEP 1] Detecting event bars...")
    t1 = time.time()

    is_event = detect_event_bars(mid_prices, features, feature_names, day_boundaries)

    n_events = is_event.sum()
    bars_per_day = len(mid_prices) / max(n_days, 1)
    events_per_day = n_events / max(n_days, 1)
    logger.info(f"  Total events: {n_events:,} ({events_per_day:.0f}/day)")
    logger.info(f"  Bars per day: {bars_per_day:.0f}")
    logger.info(f"  Event rate: {n_events/len(mid_prices):.2%}")
    logger.info(f"  [Step 1 done in {time.time()-t1:.1f}s]")

    # ------------------------------------------------------------------ #
    # STEP 2: LABEL OUTCOMES
    # ------------------------------------------------------------------ #
    logger.info("\n[STEP 2] Labeling event outcomes (CONT/REV/NEUTRAL)...")
    t2 = time.time()

    labels, move_direction, move_size_ticks = label_event_outcomes(
        mid_prices, is_event, day_boundaries,
        horizon_bars=LABEL_HORIZON_BARS,
        thresh_ticks=CONT_THRESH_TICKS,
    )
    logger.info(f"  [Step 2 done in {time.time()-t2:.1f}s]")

    # ------------------------------------------------------------------ #
    # ORACLE UPPER BOUND
    # ------------------------------------------------------------------ #
    logger.info("\n[ORACLE] Computing theoretical upper bound...")
    event_indices_all = np.where(is_event)[0]
    oracle_result = oracle_analysis(
        labels, event_indices_all, move_direction,
        mid_prices, day_boundaries, oos_start_day,
    )

    # ------------------------------------------------------------------ #
    # STEP 3: BUILD FEATURES + TRAIN CLASSIFIER
    # ------------------------------------------------------------------ #
    logger.info("\n[STEP 3] Building event-bar feature matrix...")
    t3 = time.time()

    X_events, event_indices, all_feature_names = build_event_features(
        features, feature_names, mid_prices,
        is_event, move_size_ticks, move_direction, day_boundaries,
    )
    y_events = labels[event_indices]

    logger.info(f"  Event feature matrix: {X_events.shape}")
    logger.info(f"  Labels: CONT={( y_events==0).sum():,}, "
                f"REV={( y_events==1).sum():,}, "
                f"NEUT={( y_events==2).sum():,}")

    # Replace NaNs/Infs
    X_events = np.nan_to_num(X_events, nan=0.0, posinf=60000.0, neginf=-60000.0)
    X_events = X_events.astype(np.float32)
    gc.collect()

    logger.info(f"\n[STEP 3b] Training classifier (walk-forward)...")
    pred_probs, pred_classes, fold_results = train_cont_rev_classifier(
        X_events, y_events, event_indices, day_boundaries,
        min_train_days=5, max_train_days=30, purge_gap_bars=LABEL_HORIZON_BARS,
    )
    logger.info(f"  [Step 3 done in {time.time()-t3:.1f}s]")

    # ------------------------------------------------------------------ #
    # STEP 4: BACKTEST — FIXED 30S HOLD
    # ------------------------------------------------------------------ #
    logger.info("\n[STEP 4] Backtesting with fixed 30s hold...")
    t4 = time.time()

    result_fixed = backtest_fixed_hold(
        mid_prices, event_indices, pred_classes, pred_probs,
        labels, move_direction[event_indices],
        day_boundaries, oos_start_day,
        hold_bars=HOLD_BARS_FIXED,
        min_confidence=args.min_confidence,
        label='fixed_30s',
    )
    logger.info(f"  [Step 4 done in {time.time()-t4:.1f}s]")

    # ------------------------------------------------------------------ #
    # STEP 5: ADAPTIVE HOLD
    # ------------------------------------------------------------------ #
    logger.info("\n[STEP 5] Backtesting with adaptive hold (ride continuation)...")
    t5 = time.time()

    result_adaptive = backtest_adaptive_hold(
        mid_prices, event_indices, pred_classes, pred_probs,
        labels, move_direction[event_indices],
        day_boundaries, oos_start_day,
        min_confidence=args.min_confidence,
        max_hold_bars=3000,
        stop_loss_ticks=3.0,
        momentum_window=30,
        label='adaptive_hold',
    )
    logger.info(f"  [Step 5 done in {time.time()-t5:.1f}s]")

    # ------------------------------------------------------------------ #
    # COMPARISON TABLE
    # ------------------------------------------------------------------ #
    logger.info(f"\n{'='*80}")
    logger.info("CONTINUATION/REVERSAL DETECTOR — FINAL RESULTS")
    logger.info(f"{'='*80}")

    logger.info(f"\n  DATASET:")
    logger.info(f"    Total days:   {n_days}")
    logger.info(f"    OOS days:     {n_oos}")
    logger.info(f"    Total bars:   {len(mid_prices):,}")
    logger.info(f"    Event bars:   {n_events:,} ({n_events/len(mid_prices):.2%})")
    logger.info(f"    Events/day:   {events_per_day:.0f}")

    if fold_results:
        mean_overall = float(np.mean([f['overall_acc'] for f in fold_results]))
        mean_cont = float(np.nanmean([f['cont_acc'] for f in fold_results]))
        mean_rev = float(np.nanmean([f['rev_acc'] for f in fold_results]))
        mean_neut = float(np.nanmean([f['neut_acc'] for f in fold_results]))
        logger.info(f"\n  CLASSIFIER ACCURACY (OOS walk-forward):")
        logger.info(f"    Overall:       {mean_overall:.3f}")
        logger.info(f"    CONTINUATION:  {mean_cont:.3f}")
        logger.info(f"    REVERSAL:      {mean_rev:.3f}")
        logger.info(f"    NEUTRAL:       {mean_neut:.3f}")
        logger.info(f"    (Baseline random 3-class: 0.333)")

    logger.info(f"\n  ORACLE UPPER BOUND (perfect classification):")
    logger.info(f"    CONT: {oracle_result['n_cont_events']:,} events, "
                f"avg={oracle_result['cont_avg_pnl_ticks']:.3f}t, "
                f"total=${oracle_result['cont_total_pnl_dollars']:+,.0f}")
    logger.info(f"    REV:  {oracle_result['n_rev_events']:,} events, "
                f"avg={oracle_result['rev_avg_pnl_ticks']:.3f}t, "
                f"total=${oracle_result['rev_total_pnl_dollars']:+,.0f}")

    results_list = [r for r in [result_fixed, result_adaptive] if r is not None]

    if results_list:
        header = (f"  {'Strategy':>20s}  {'Trades':>7s}  {'Tr/Day':>6s}  "
                  f"{'Total$':>10s}  {'$/tr':>7s}  {'t/tr':>6s}  "
                  f"{'WR':>6s}  {'PF':>5s}  {'Sharpe':>7s}  {'Hold':>6s}")
        logger.info(f"\n  BACKTEST COMPARISON:")
        logger.info(header)
        logger.info("  " + "-" * (len(header) - 2))
        for r in results_list:
            logger.info(
                f"  {r['label']:>20s}  "
                f"{r['n_trades']:>7d}  "
                f"{r['trades_per_day']:>6.1f}  "
                f"${r['total_pnl_dollars']:>+9,.0f}  "
                f"${r['avg_pnl_per_trade']:>+6.2f}  "
                f"{r['avg_pnl_ticks']:>+5.3f}t  "
                f"{r['win_rate']:>5.1%}  "
                f"{r['profit_factor']:>5.2f}  "
                f"{r['sharpe']:>7.2f}  "
                f"{r.get('avg_hold_sec', 0):>5.1f}s"
            )

    # ------------------------------------------------------------------ #
    # SAVE RESULTS
    # ------------------------------------------------------------------ #
    elapsed = time.time() - t_total
    output = {
        'config': {
            'n_days': n_days,
            'oos_days': n_oos,
            'oos_start_day': oos_start_day,
            'n_events': int(n_events),
            'events_per_day': round(events_per_day, 1),
            'min_confidence': args.min_confidence,
            'elapsed_sec': round(elapsed, 1),
        },
        'classifier': {
            'mean_overall_acc': round(mean_overall, 4) if fold_results else None,
            'mean_cont_acc': round(mean_cont, 4) if fold_results else None,
            'mean_rev_acc': round(mean_rev, 4) if fold_results else None,
            'mean_neut_acc': round(mean_neut, 4) if fold_results else None,
            'n_folds': len(fold_results),
        },
        'oracle': oracle_result,
        'backtest_fixed_30s': result_fixed,
        'backtest_adaptive': result_adaptive,
    }

    json_path = RESULTS_DIR / f"cont_rev_{_ts}.json"
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"\n{'='*70}")
    logger.info(f"COMPLETE in {elapsed:.0f}s ({elapsed/60:.1f}m)")
    logger.info(f"Results JSON: {json_path}")
    logger.info(f"Log: {_log_file}")
    logger.info(f"{'='*70}")


if __name__ == '__main__':
    main()
