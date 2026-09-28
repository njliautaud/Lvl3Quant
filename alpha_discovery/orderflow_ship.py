"""
Orderflow Ship Detector — Lvl3Quant

Concept: Detect when order flow signatures indicate a LARGE directional move
is about to happen (institutional flow, news reaction, sweep events) BEFORE
the price fully moves. Jump on the "ship" and ride the momentum.

Key difference from continuation_reversal.py:
  - cont_rev detects AFTER a large move and asks "will it continue?"
  - THIS detector looks at PRE-MOVE orderflow patterns to predict the move

What we're looking for (precursor signals):
  1. Book thinning — one side of book gets eaten, depth ratio shifts
  2. Flow acceleration — OFI ramps up over 2-5s, not just a single spike
  3. Aggressive volume building — sustained aggressor imbalance (not random)
  4. Deep book disagreement — L5 depth vs L1 diverges (iceberg detection)
  5. Spread regime change — spread widening or micro-tightening
  6. Queue depletion asymmetry — one side draining fast, other refilling
  7. Volatility compression → expansion transition
  8. Toxicity ramp — kyle_lambda, adverse_sel climbing before the move

Pipeline:
  Step 1: Label bars where a large move STARTS within the next N seconds
          * Forward window: scan next 5s, 10s, 30s for >3 tick moves
          * Direction: UP (+1) or DOWN (-1)
          * Magnitude: actual tick move achieved
  Step 2: Build precursor features from the 1-5 seconds BEFORE the label bar
          * Use existing 340 features + engineered "acceleration" features
  Step 3: Train a 3-class classifier: UP_SHIP / DOWN_SHIP / NO_SHIP
          * Walk-forward with purge gap, no leakage
  Step 4: Trade on high-confidence predictions (P(ship) > threshold)
          * Enter immediately at market (we need speed for this strategy)
          * Hold for adaptive duration based on momentum decay
          * Exit when: (a) target reached, (b) momentum dies, (c) max hold

Usage:
    python alpha_discovery/orderflow_ship.py --n-days 70
    python alpha_discovery/orderflow_ship.py --n-days 20 --quick
    python alpha_discovery/orderflow_ship.py --n-days 70 --horizon 10s
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
_log_file = RESULTS_DIR / f"orderflow_ship_{_ts}.log"
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
logger = logging.getLogger("ship")

# ===========================================================================
# Constants
# ===========================================================================
TICK_SIZE = 0.25
TICK_VALUE = 12.50
COMMISSION_RT = 4.70  # HC #52: $4.70 RT (AMP)           # round-trip commission
COMMISSION_TICKS = COMMISSION_RT / TICK_VALUE   # 0.24 ticks
BARS_PER_SEC = 10              # 100ms bars

# Ship detection thresholds
LARGE_MOVE_TICKS = 3.0         # minimum move to qualify as "ship" (3 ticks = $37.50)
SHIP_WINDOWS_SEC = [5, 10, 30] # look forward 5s, 10s, 30s for large moves
SHIP_WINDOWS_BARS = [w * BARS_PER_SEC for w in SHIP_WINDOWS_SEC]

# Precursor lookback
PRECURSOR_BARS = 50            # look at last 5s of features before event

# Labels
LABEL_NO_SHIP = 0
LABEL_UP_SHIP = 1
LABEL_DOWN_SHIP = 2

# Backtest
MIN_CONFIDENCE = 0.60          # minimum P(ship) to enter
MARKET_ENTRY_COST_TICKS = 1.24 # market entry: 1 tick spread + 0.24 commission
MARKET_EXIT_COST_TICKS = 1.24  # market exit: same
TOTAL_COST_TICKS = MARKET_ENTRY_COST_TICKS + MARKET_EXIT_COST_TICKS  # 2.48t
# For large moves (3+ ticks), this 2.48t cost is reasonable if we capture 60%+

# Hold strategies
HOLD_BARS_FIXED = {5: 50, 10: 100, 30: 300}  # per ship_window
MAX_HOLD_BARS = 300            # 30s absolute max
TRAIL_STOP_TICKS = 2.0         # trailing stop at 2 ticks from peak
MOMENTUM_DECAY_BARS = 30       # exit if no progress in 3s

COOLDOWN_BARS = 100            # 10s cooldown between trades


# ===========================================================================
# STEP 1: LABEL GENERATION — find bars where a large move starts
# ===========================================================================

def label_ship_events(
    mid_prices: np.ndarray,
    day_start: int,
    day_end: int,
    ship_window_bars: int,
    move_threshold_ticks: float = LARGE_MOVE_TICKS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each bar in [day_start, day_end), look forward ship_window_bars
    and find the MAX directional move achieved.

    Returns:
        labels: (N,) array — 0=NO_SHIP, 1=UP_SHIP, 2=DOWN_SHIP
        magnitudes: (N,) array — max move in ticks (signed)
        first_hit_bars: (N,) array — bars until the threshold was first hit
    """
    N = day_end - day_start
    labels = np.full(N, LABEL_NO_SHIP, dtype=np.int32)
    magnitudes = np.zeros(N, dtype=np.float32)
    first_hit = np.full(N, -1, dtype=np.int32)

    day_mid = mid_prices[day_start:day_end]
    threshold = move_threshold_ticks * TICK_SIZE

    for i in range(N - ship_window_bars):
        ref_price = day_mid[i]
        future_prices = day_mid[i + 1: i + 1 + ship_window_bars]
        future_moves = future_prices - ref_price

        # Max favorable excursion in each direction
        max_up = np.max(future_moves) if len(future_moves) > 0 else 0.0
        max_down = np.min(future_moves) if len(future_moves) > 0 else 0.0

        # First-hit: which direction reaches threshold first?
        up_hits = np.where(future_moves >= threshold)[0]
        down_hits = np.where(future_moves <= -threshold)[0]

        first_up = up_hits[0] if len(up_hits) > 0 else ship_window_bars + 1
        first_down = down_hits[0] if len(down_hits) > 0 else ship_window_bars + 1

        if first_up < first_down and first_up <= ship_window_bars:
            labels[i] = LABEL_UP_SHIP
            magnitudes[i] = max_up / TICK_SIZE
            first_hit[i] = first_up
        elif first_down < first_up and first_down <= ship_window_bars:
            labels[i] = LABEL_DOWN_SHIP
            magnitudes[i] = max_down / TICK_SIZE
            first_hit[i] = first_down
        else:
            # No large move — record the max absolute move for reference
            if abs(max_up) > abs(max_down):
                magnitudes[i] = max_up / TICK_SIZE
            else:
                magnitudes[i] = max_down / TICK_SIZE

    return labels, magnitudes, first_hit


def vectorized_label_ship_events(
    mid_prices: np.ndarray,
    day_start: int,
    day_end: int,
    ship_window_bars: int,
    move_threshold_ticks: float = LARGE_MOVE_TICKS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized labeling — much faster than bar-by-bar loop."""
    N = day_end - day_start
    labels = np.full(N, LABEL_NO_SHIP, dtype=np.int32)
    magnitudes = np.zeros(N, dtype=np.float32)

    day_mid = mid_prices[day_start:day_end].astype(np.float64)
    threshold = move_threshold_ticks * TICK_SIZE
    usable = N - ship_window_bars

    if usable <= 0:
        return labels, magnitudes

    # Build forward return matrix: (usable, ship_window_bars)
    # future_returns[i, j] = mid[i+j+1] - mid[i]
    # Memory-efficient: process in chunks
    chunk_size = min(usable, 50000)
    for cs in range(0, usable, chunk_size):
        ce = min(cs + chunk_size, usable)
        cl = ce - cs

        # For each bar i in [cs, ce), compute max up and max down
        # in the next ship_window_bars bars
        best_up = np.zeros(cl, dtype=np.float64)
        best_down = np.zeros(cl, dtype=np.float64)
        first_up = np.full(cl, ship_window_bars + 1, dtype=np.int32)
        first_down = np.full(cl, ship_window_bars + 1, dtype=np.int32)

        for j in range(1, ship_window_bars + 1):
            moves = day_mid[cs + j: ce + j] - day_mid[cs:ce]
            # Track best
            better_up = moves > best_up
            best_up = np.where(better_up, moves, best_up)
            better_down = moves < best_down
            best_down = np.where(better_down, moves, best_down)
            # First hit
            up_hit = (moves >= threshold) & (first_up > ship_window_bars)
            first_up = np.where(up_hit, j, first_up)
            down_hit = (moves <= -threshold) & (first_down > ship_window_bars)
            first_down = np.where(down_hit, j, first_down)

        # Classify
        up_ship = (first_up <= ship_window_bars) & (first_up < first_down)
        down_ship = (first_down <= ship_window_bars) & (first_down < first_up)

        labels[cs:ce] = np.where(
            up_ship, LABEL_UP_SHIP,
            np.where(down_ship, LABEL_DOWN_SHIP, LABEL_NO_SHIP)
        )
        magnitudes[cs:ce] = np.where(
            up_ship, best_up / TICK_SIZE,
            np.where(down_ship, best_down / TICK_SIZE, 0.0)
        ).astype(np.float32)

    return labels, magnitudes


# ===========================================================================
# STEP 2: PRECURSOR FEATURE ENGINEERING
# ===========================================================================

def compute_precursor_features(
    features: np.ndarray,
    feature_names: List[str],
    mid_prices: np.ndarray,
    day_start: int,
    day_end: int,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build "acceleration" / "ramp" features from existing features.

    For key flow metrics, compute:
    - Rate of change over last 1s, 3s, 5s
    - Acceleration (change of change)
    - Z-score vs recent history
    - Asymmetry ratios (bid vs ask changes)

    Returns augmented feature matrix + names for this day segment.
    """
    N = day_end - day_start
    fn = {name: idx for idx, name in enumerate(feature_names)}
    extra_feats = []
    extra_names = []

    # Key columns for ship detection
    KEY_COLS = {
        'ofi_5': fn.get('ofi_5', -1),
        'ofi_20': fn.get('ofi_20', -1),
        'ofi_50': fn.get('ofi_50', -1),
        'depth_ratio_l1': fn.get('depth_ratio_l1', -1),
        'depth_ratio_l3': fn.get('depth_ratio_l3', -1),
        'depth_ratio_l5': fn.get('depth_ratio_l5', -1),
        'aggressive_imbalance': fn.get('aggressive_imbalance', fn.get('trade_imbalance', -1)),
        'aggressive_buy_count': fn.get('aggressive_buy_count', -1),
        'aggressive_sell_count': fn.get('aggressive_sell_count', -1),
        'spread': fn.get('spread', -1),
        'kyle_lambda_20': fn.get('kyle_lambda_20', -1),
        'toxicity_score': fn.get('toxicity_score', -1),
        'vpin_20': fn.get('vpin_20', -1),
        'vpin_50': fn.get('vpin_50', -1),
        'book_thinning': fn.get('book_thinning', -1),
        'flow_acceleration': fn.get('flow_acceleration', -1),
        'volatility_compression': fn.get('volatility_compression', -1),
        'queue_depletion_rate_bid_5': fn.get('queue_depletion_rate_bid_5', -1),
        'queue_depletion_rate_ask_5': fn.get('queue_depletion_rate_ask_5', -1),
        'bid_depth_skew': fn.get('bid_depth_skew', -1),
        'ask_depth_skew': fn.get('ask_depth_skew', -1),
    }

    day_feats = features[day_start:day_end]

    # For each key column, compute acceleration features
    roc_windows = [10, 30, 50]  # 1s, 3s, 5s rate of change
    for col_name, col_idx in KEY_COLS.items():
        if col_idx < 0 or col_idx >= features.shape[1]:
            continue
        vals = day_feats[:, col_idx].astype(np.float32)

        for w in roc_windows:
            if N <= w:
                continue
            # Rate of change: (val[i] - val[i-w]) / w
            roc = np.zeros(N, dtype=np.float32)
            roc[w:] = (vals[w:] - vals[:-w]) / w
            extra_feats.append(roc)
            extra_names.append(f"{col_name}_roc_{w}")

            # Acceleration: change of roc
            if w == roc_windows[0] and len(roc_windows) > 1:
                accel = np.zeros(N, dtype=np.float32)
                accel[w:] = roc[w:] - roc[:-w]
                extra_feats.append(accel)
                extra_names.append(f"{col_name}_accel_{w}")

    # Depth asymmetry change: how fast is one side draining vs other
    bid_dep = KEY_COLS.get('queue_depletion_rate_bid_5', -1)
    ask_dep = KEY_COLS.get('queue_depletion_rate_ask_5', -1)
    if bid_dep >= 0 and ask_dep >= 0:
        bid_drain = day_feats[:, bid_dep].astype(np.float32)
        ask_drain = day_feats[:, ask_dep].astype(np.float32)
        drain_asym = bid_drain - ask_drain  # positive = bid draining faster (bearish)
        extra_feats.append(drain_asym)
        extra_names.append('drain_asymmetry')

        # Change in drain asymmetry
        for w in [10, 30]:
            roc = np.zeros(N, dtype=np.float32)
            roc[w:] = drain_asym[w:] - drain_asym[:-w]
            extra_feats.append(roc)
            extra_names.append(f'drain_asym_roc_{w}')

    # OFI momentum: is OFI accelerating in one direction?
    ofi5 = KEY_COLS.get('ofi_5', -1)
    ofi20 = KEY_COLS.get('ofi_20', -1)
    ofi50 = KEY_COLS.get('ofi_50', -1)
    if ofi5 >= 0 and ofi20 >= 0:
        # Short vs medium OFI divergence
        ofi_diverge = day_feats[:, ofi5] - day_feats[:, ofi20]
        extra_feats.append(ofi_diverge.astype(np.float32))
        extra_names.append('ofi_short_vs_med')
    if ofi5 >= 0 and ofi50 >= 0:
        ofi_diverge2 = day_feats[:, ofi5] - day_feats[:, ofi50]
        extra_feats.append(ofi_diverge2.astype(np.float32))
        extra_names.append('ofi_short_vs_long')

    # Depth ratio divergence: L1 vs L5 (iceberg detection)
    dr1 = KEY_COLS.get('depth_ratio_l1', -1)
    dr5 = KEY_COLS.get('depth_ratio_l5', -1)
    if dr1 >= 0 and dr5 >= 0:
        depth_diverge = day_feats[:, dr1] - day_feats[:, dr5]
        extra_feats.append(depth_diverge.astype(np.float32))
        extra_names.append('depth_l1_vs_l5')

    # Price micro-momentum: change over last 0.5s, 1s, 3s
    day_mid = mid_prices[day_start:day_end]
    for w in [5, 10, 30]:
        if N <= w:
            continue
        pmom = np.zeros(N, dtype=np.float32)
        pmom[w:] = (day_mid[w:] - day_mid[:-w]) / TICK_SIZE
        extra_feats.append(pmom)
        extra_names.append(f'price_momentum_{w}b')

    # Aggressive volume imbalance ramp
    ab = KEY_COLS.get('aggressive_buy_count', -1)
    as_ = KEY_COLS.get('aggressive_sell_count', -1)
    if ab >= 0 and as_ >= 0:
        buy_vol = day_feats[:, ab].astype(np.float32)
        sell_vol = day_feats[:, as_].astype(np.float32)
        total_vol = buy_vol + sell_vol + 1e-8
        imb_ratio = (buy_vol - sell_vol) / total_vol
        # Cumulative imbalance over last 3s
        cumimb = np.zeros(N, dtype=np.float32)
        w = 30
        if N > w:
            cs = np.cumsum(imb_ratio)
            cumimb[w:] = (cs[w:] - cs[:-w]) / w
        extra_feats.append(cumimb)
        extra_names.append('cumulative_imbalance_3s')

        # Volume acceleration
        vol_roc = np.zeros(N, dtype=np.float32)
        if N > 10:
            vol_roc[10:] = total_vol[10:] - total_vol[:-10]
        extra_feats.append(vol_roc)
        extra_names.append('volume_acceleration')

    # Stack all extra features
    if extra_feats:
        extra_matrix = np.column_stack(extra_feats)
        augmented = np.hstack([day_feats, extra_matrix])
        all_names = list(feature_names) + extra_names
    else:
        augmented = day_feats
        all_names = list(feature_names)

    # Store as float16 to halve memory usage; caller converts to float32 before model training
    return augmented.astype(np.float16), all_names


# ===========================================================================
# STEP 3: TRAIN / PREDICT with walk-forward
# ===========================================================================

def train_ship_detector(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    model_type: str = 'multiclass',
) -> Tuple[np.ndarray, object, Dict]:
    """
    Train LightGBM classifier for ship detection.

    Returns:
        predictions: (N_test, 3) class probabilities [NO_SHIP, UP_SHIP, DOWN_SHIP]
        model: trained model
        metrics: dict with training metrics
    """
    try:
        import lightgbm as lgb
    except ImportError:
        logger.error("LightGBM not available")
        return np.zeros((len(X_test), 3)), None, {}

    # Clean data
    valid_train = np.isfinite(X_train).all(axis=1) & (y_train >= 0)
    X_tr = X_train[valid_train]
    y_tr = y_train[valid_train]

    if len(X_tr) < 100:
        logger.warning(f"  Too few training samples: {len(X_tr)}")
        return np.zeros((len(X_test), 3)), None, {}

    # Class distribution
    n_no = (y_tr == LABEL_NO_SHIP).sum()
    n_up = (y_tr == LABEL_UP_SHIP).sum()
    n_down = (y_tr == LABEL_DOWN_SHIP).sum()
    logger.info(f"  Train: NO_SHIP={n_no:,} UP={n_up:,} DOWN={n_down:,}")

    if n_up < 50 or n_down < 50:
        logger.warning("  Too few ship events for training")
        return np.zeros((len(X_test), 3)), None, {}

    # Handle class imbalance: ships are rare
    # Use sample weights: upweight ship events
    weights = np.ones(len(y_tr), dtype=np.float32)
    if n_no > 0:
        ship_weight = n_no / max(n_up + n_down, 1)
        ship_weight = min(ship_weight, 20.0)  # cap at 20x
        weights[y_tr == LABEL_UP_SHIP] = ship_weight
        weights[y_tr == LABEL_DOWN_SHIP] = ship_weight
        logger.info(f"  Ship event weight: {ship_weight:.1f}x")

    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': 7,
        'min_child_samples': 50,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42,
    }

    dtrain = lgb.Dataset(X_tr, y_tr, weight=weights)
    model = lgb.train(
        params, dtrain,
        num_boost_round=500,
        valid_sets=[dtrain],
        callbacks=[lgb.log_evaluation(0)],
    )

    # Predict on test
    valid_test = np.isfinite(X_test).all(axis=1)
    preds = np.zeros((len(X_test), 3), dtype=np.float32)
    preds[:, 0] = 1.0  # default: no ship
    if valid_test.sum() > 0:
        preds[valid_test] = model.predict(X_test[valid_test])

    # Metrics
    metrics = {
        'n_train': int(len(X_tr)),
        'n_ship_up': int(n_up),
        'n_ship_down': int(n_down),
        'n_test': int(len(X_test)),
    }

    return preds, model, metrics


# ===========================================================================
# STEP 4: BACKTEST — ride the ship
# ===========================================================================

def backtest_ship_riding(
    mid_prices: np.ndarray,
    predictions: np.ndarray,   # (N, 3) class probs
    day_start: int,
    day_end: int,
    labels: np.ndarray,
    min_confidence: float = MIN_CONFIDENCE,
    hold_mode: str = 'trailing',  # 'fixed', 'trailing', 'momentum'
    ship_window_bars: int = 100,
    best_bid: Optional[np.ndarray] = None,   # real bid from MBO snapshots
    best_ask: Optional[np.ndarray] = None,   # real ask from MBO snapshots
    queue_depth_bid: Optional[np.ndarray] = None,  # queue depth at best bid
    queue_depth_ask: Optional[np.ndarray] = None,  # queue depth at best ask
    rng: Optional[np.random.RandomState] = None,
) -> Dict:
    """
    Backtest: enter on high-confidence ship predictions, ride the move.

    Uses REAL bid/ask prices from MBO snapshot data when available.
    For ship-riding, we use MARKET entry (cross the spread immediately).
    Fill is deterministic for market orders (you always get filled at the ask/bid).

    Cost model:
      - Market entry: pay full spread (real ask - real bid) + commission
      - Market exit: same
      - If real bid/ask not available, falls back to mid ± half_tick
    """
    if rng is None:
        rng = np.random.RandomState(42)

    N = day_end - day_start
    day_mid = mid_prices[day_start:day_end]

    # Use real bid/ask if available, else synthetic from mid
    if best_bid is not None and best_ask is not None:
        day_bid = best_bid[day_start:day_end]
        day_ask = best_ask[day_start:day_end]
        using_real_ba = True
    else:
        day_bid = day_mid - TICK_SIZE / 2
        day_ask = day_mid + TICK_SIZE / 2
        using_real_ba = False

    trades = []
    i = 0

    while i < N - 10:
        # Check prediction confidence
        p_up = predictions[i, LABEL_UP_SHIP]
        p_down = predictions[i, LABEL_DOWN_SHIP]
        p_ship = max(p_up, p_down)

        if p_ship < min_confidence:
            i += 1
            continue

        # Determine direction
        direction = 1 if p_up > p_down else -1
        confidence = p_ship

        # MARKET ENTRY: cross the spread immediately
        # Buy: pay the ask price. Sell: receive the bid price.
        if direction == 1:
            entry_price = day_ask[i]  # buy at ask
        else:
            entry_price = day_bid[i]  # sell at bid

        # Real spread at entry time
        entry_spread_ticks = (day_ask[i] - day_bid[i]) / TICK_SIZE
        entry_commission_ticks = COMMISSION_TICKS  # 0.24t

        # Hold and exit
        exit_bar = i
        exit_price = entry_price
        peak_favorable = 0.0
        exit_reason = 'end_of_day'

        if hold_mode == 'fixed':
            max_hold = min(ship_window_bars, N - i - 1)
            exit_bar = i + max_hold
            # Market exit: sell at bid (if long) or buy at ask (if short)
            if direction == 1:
                exit_price = day_bid[exit_bar]
            else:
                exit_price = day_ask[exit_bar]
            exit_reason = 'fixed_hold'

        elif hold_mode == 'trailing':
            trail_ticks = TRAIL_STOP_TICKS
            for j in range(1, min(MAX_HOLD_BARS, N - i)):
                # Track PnL using mid (for trailing logic) but exit at bid/ask
                current_mid = day_mid[i + j]
                pnl_mid_ticks = direction * (current_mid - day_mid[i]) / TICK_SIZE
                if pnl_mid_ticks > peak_favorable:
                    peak_favorable = pnl_mid_ticks
                # Trail stop
                if peak_favorable > 1.0 and (peak_favorable - pnl_mid_ticks) >= trail_ticks:
                    exit_bar = i + j
                    exit_price = day_bid[exit_bar] if direction == 1 else day_ask[exit_bar]
                    exit_reason = 'trailing_stop'
                    break
                # Hard stop: 4 ticks adverse
                if pnl_mid_ticks < -4.0:
                    exit_bar = i + j
                    exit_price = day_bid[exit_bar] if direction == 1 else day_ask[exit_bar]
                    exit_reason = 'hard_stop'
                    break
            else:
                exit_bar = i + min(MAX_HOLD_BARS, N - i - 1)
                exit_price = day_bid[exit_bar] if direction == 1 else day_ask[exit_bar]
                exit_reason = 'max_hold'

        elif hold_mode == 'momentum':
            last_progress = i
            for j in range(1, min(MAX_HOLD_BARS, N - i)):
                current_mid = day_mid[i + j]
                pnl_mid_ticks = direction * (current_mid - day_mid[i]) / TICK_SIZE
                if pnl_mid_ticks > peak_favorable:
                    peak_favorable = pnl_mid_ticks
                    last_progress = i + j
                if (i + j - last_progress) >= MOMENTUM_DECAY_BARS and peak_favorable > 0:
                    exit_bar = i + j
                    exit_price = day_bid[exit_bar] if direction == 1 else day_ask[exit_bar]
                    exit_reason = 'momentum_decay'
                    break
                if pnl_mid_ticks < -4.0:
                    exit_bar = i + j
                    exit_price = day_bid[exit_bar] if direction == 1 else day_ask[exit_bar]
                    exit_reason = 'hard_stop'
                    break
            else:
                exit_bar = i + min(MAX_HOLD_BARS, N - i - 1)
                exit_price = day_bid[exit_bar] if direction == 1 else day_ask[exit_bar]
                exit_reason = 'max_hold'

        # Compute PnL — entry and exit already include spread via real bid/ask
        # Long: bought at ask, sold at bid. Short: sold at bid, bought at ask.
        gross_ticks = direction * (exit_price - entry_price) / TICK_SIZE
        # Only add commission (spread is already in the prices)
        net_ticks = gross_ticks - 2 * COMMISSION_TICKS  # commission on entry + exit
        net_dollars = net_ticks * TICK_VALUE
        hold_bars = exit_bar - i

        # Was there actually a ship?
        actual_label = labels[i] if i < len(labels) else LABEL_NO_SHIP
        correct_direction = (
            (direction == 1 and actual_label == LABEL_UP_SHIP) or
            (direction == -1 and actual_label == LABEL_DOWN_SHIP)
        )

        trades.append({
            'bar': i + day_start,
            'direction': direction,
            'confidence': float(confidence),
            'entry_price': float(entry_price),
            'exit_price': float(exit_price),
            'entry_spread_ticks': float(entry_spread_ticks),
            'gross_ticks': float(gross_ticks),
            'net_ticks': float(net_ticks),
            'net_dollars': float(net_dollars),
            'hold_bars': int(hold_bars),
            'peak_favorable_ticks': float(peak_favorable),
            'exit_reason': exit_reason,
            'actual_label': int(actual_label),
            'correct_direction': correct_direction,
            'real_ba': using_real_ba,
        })

        # Skip cooldown
        i = exit_bar + COOLDOWN_BARS
        continue

    return {'trades': trades, 'n_trades': len(trades), 'used_real_ba': using_real_ba}


# ===========================================================================
# STEP 5: MAIN PIPELINE
# ===========================================================================

def run_pipeline(
    n_days: int = 70,
    ship_window_sec: int = 10,
    feature_cache: str = DEFAULT_FEATURE_CACHE,
    quick: bool = False,
    move_threshold: float = LARGE_MOVE_TICKS,
    min_conf: float = MIN_CONFIDENCE,
):
    """Full pipeline: label → features → walk-forward train → backtest."""

    ship_window_bars = ship_window_sec * BARS_PER_SEC
    logger.info("=" * 70)
    logger.info("ORDERFLOW SHIP DETECTOR")
    logger.info(f"  n_days:         {n_days}")
    logger.info(f"  ship_window:    {ship_window_sec}s ({ship_window_bars} bars)")
    logger.info(f"  move_threshold: {move_threshold} ticks (${move_threshold * TICK_VALUE:.2f})")
    logger.info(f"  min_confidence: {min_conf:.0%}")
    logger.info(f"  feature_cache:  {feature_cache}")
    logger.info(f"  quick:          {quick}")
    logger.info("=" * 70)

    cache_path = Path(feature_cache)
    if not cache_path.exists():
        logger.error(f"Feature cache not found: {cache_path}")
        return

    # Discover available days — match pattern from continuation_reversal.py
    npz_files = sorted(cache_path.glob("*_mbo_features.npz"))
    if not npz_files:
        npz_files = sorted(cache_path.glob("*.npz"))
    total_available = len(npz_files)
    logger.info(f"  Available cache files: {total_available}")

    if total_available == 0:
        logger.error("No cache files found")
        return

    use_days = min(n_days, total_available)
    if quick:
        use_days = min(use_days, 20)
    logger.info(f"  Using: {use_days} days")

    # Get feature names from mbo_features module
    try:
        from alpha_discovery.mbo_features import get_feature_names
        base_feature_names = get_feature_names()
    except ImportError:
        base_feature_names = [f"f_{i}" for i in range(340)]
    logger.info(f"  Base features: {len(base_feature_names)}")

    # Also check for snapshot cache (for mid prices)
    snap_cache = LVL3_ROOT / "data" / "processed" / "medium_snapshots_cache"
    snap_by_date = {}
    if snap_cache.exists():
        for sf in snap_cache.glob("*.npz"):
            date_str = sf.name[:10]  # e.g. "2025-07-14"
            snap_by_date[date_str] = sf
        logger.info(f"  Snapshot cache files: {len(snap_by_date)}")

    # Walk-forward: train on first K days, test on next day
    min_train_days = 5
    purge_gap_bars = BARS_PER_SEC * 60  # 1 minute purge gap

    all_trades = []
    all_metrics = []
    fold_results = []

    # First pass: load all features + mid_prices and labels for each day
    day_info = []
    logger.info("\nPhase 1: Loading data and computing labels...")
    t0 = time.time()

    for d in range(use_days):
        fpath = npz_files[d]
        date_str = fpath.name[:10]

        try:
            data = np.load(str(fpath), allow_pickle=True)
        except Exception as e:
            logger.warning(f"  Day {d} ({date_str}): load error: {e}")
            continue

        # Features are stored as 'mbo_features'
        if 'mbo_features' in data:
            feats = data['mbo_features'].astype(np.float32)
        elif 'features' in data:
            feats = data['features'].astype(np.float32)
        else:
            logger.warning(f"  Day {d} ({date_str}): no features key, skipping")
            data.close()
            continue

        N = feats.shape[0]
        data.close()

        if N < ship_window_bars + 200:
            logger.warning(f"  Day {d} ({date_str}): too few bars ({N}), skipping")
            continue

        # Load mid_prices + real bid/ask from snapshot cache
        mid = None
        real_bid = None
        real_ask = None
        if date_str in snap_by_date:
            try:
                snap_data = np.load(str(snap_by_date[date_str]), allow_pickle=True)
                if 'mid_prices' in snap_data:
                    mp = snap_data['mid_prices'].astype(np.float64)
                    if len(mp) == N:
                        mid = mp
                # Real bid/ask from global_features columns 8 and 9
                if 'global_features' in snap_data:
                    gf = snap_data['global_features']
                    if gf.shape[0] == N and gf.shape[1] > 9:
                        real_bid = gf[:, 8].astype(np.float64)
                        real_ask = gf[:, 9].astype(np.float64)
                        # Sanity check: bid < ask, both > 0
                        valid_ba = (real_bid > 0) & (real_ask > 0) & (real_ask > real_bid)
                        if valid_ba.sum() < N * 0.5:
                            real_bid = None
                            real_ask = None
                snap_data.close()
            except Exception:
                pass
        if mid is None:
            # Fallback: column 0 is typically mid_price in the feature cache
            mid = feats[:, 0].astype(np.float64)

        # Trim feature names to match
        fn = base_feature_names[:feats.shape[1]]
        if len(fn) < feats.shape[1]:
            fn = fn + [f'feat_{i}' for i in range(len(fn), feats.shape[1])]

        # Label ship events
        labels, magnitudes = vectorized_label_ship_events(
            mid, 0, N, ship_window_bars, move_threshold
        )

        n_up = (labels == LABEL_UP_SHIP).sum()
        n_down = (labels == LABEL_DOWN_SHIP).sum()
        n_events = n_up + n_down

        # Compute precursor features
        aug_feats, aug_names = compute_precursor_features(
            feats, fn, mid, 0, N
        )

        day_info.append({
            'day_idx': d,
            'date': date_str,
            'mid': mid,
            'best_bid': real_bid,   # real bid from MBO snapshot (or None)
            'best_ask': real_ask,   # real ask from MBO snapshot (or None)
            'features': aug_feats.astype(np.float16),  # float16 to halve peak memory
            'feature_names': aug_names,
            'labels': labels,
            'magnitudes': magnitudes,
            'N': N,
            'n_up': n_up,
            'n_down': n_down,
        })

        del feats
        gc.collect()

        if d % 10 == 0 or d == use_days - 1:
            logger.info(f"  Day {d}/{use_days} ({date_str}): N={N:,}, ships={n_events:,} "
                        f"(UP={n_up}, DOWN={n_down})")

    elapsed_load = time.time() - t0
    logger.info(f"  Loaded {len(day_info)} days in {elapsed_load:.0f}s")

    total_ships = sum(d['n_up'] + d['n_down'] for d in day_info)
    logger.info(f"  Total ship events: {total_ships:,}")

    if len(day_info) < min_train_days + 1:
        logger.error("Not enough days for walk-forward")
        return

    # Walk-forward training
    logger.info("\nPhase 2: Walk-forward training...")
    t1 = time.time()

    n_folds = len(day_info) - min_train_days
    aug_feature_names = day_info[0]['feature_names']
    n_features = len(aug_feature_names)

    ROLLING_WINDOW = 20  # only keep the most recent 20 training days

    for fold in range(n_folds):
        # Rolling window: use at most ROLLING_WINDOW of the most recent training days
        all_train_days = day_info[:min_train_days + fold]
        train_days = all_train_days[-ROLLING_WINDOW:]
        test_day = day_info[min_train_days + fold]

        # Build training set: subsample NO_SHIP PER DAY before stacking
        # to avoid allocating the full concatenated array before subsampling.
        rng = np.random.RandomState(fold)
        X_train_parts = []
        y_train_parts = []
        for td in train_days:
            # Use consistent feature count (still float16 here)
            f = td['features']
            if f.shape[1] != n_features:
                if f.shape[1] < n_features:
                    pad = np.zeros((f.shape[0], n_features - f.shape[1]), dtype=np.float16)
                    f = np.hstack([f, pad])
                else:
                    f = f[:, :n_features]

            # Apply purge: skip last purge_gap_bars of each training day
            usable = max(0, td['N'] - purge_gap_bars)
            f = f[:usable]
            lbl = td['labels'][:usable]

            # Subsample NO_SHIP before stacking to keep peak memory low
            ship_mask_d = lbl != LABEL_NO_SHIP
            no_ship_mask_d = lbl == LABEL_NO_SHIP
            n_ship_d = ship_mask_d.sum()
            n_no_ship_d = no_ship_mask_d.sum()
            if n_no_ship_d > 3 * n_ship_d and n_ship_d > 0:
                no_ship_idx_d = np.where(no_ship_mask_d)[0]
                keep_idx_d = rng.choice(no_ship_idx_d, size=min(3 * n_ship_d, n_no_ship_d), replace=False)
                ship_idx_d = np.where(ship_mask_d)[0]
                sel = np.sort(np.concatenate([ship_idx_d, keep_idx_d]))
                f = f[sel]
                lbl = lbl[sel]

            X_train_parts.append(f)
            y_train_parts.append(lbl)

        # Stack subsampled parts and convert to float32 only now for LightGBM
        X_train = np.vstack(X_train_parts).astype(np.float32)
        y_train = np.concatenate(y_train_parts)

        # Test set: convert float16 -> float32 for prediction
        X_test = test_day['features'].astype(np.float32)
        if X_test.shape[1] != n_features:
            if X_test.shape[1] < n_features:
                pad = np.zeros((X_test.shape[0], n_features - X_test.shape[1]), dtype=np.float32)
                X_test = np.hstack([X_test, pad])
            else:
                X_test = X_test[:, :n_features]
        y_test = test_day['labels']

        # Train
        preds, model, metrics = train_ship_detector(X_train, y_train, X_test)

        if model is None:
            continue

        # Backtest on test day — try all hold modes
        # Pass real bid/ask from MBO snapshots for realistic fill simulation
        for hold_mode in ['trailing', 'momentum', 'fixed']:
            bt = backtest_ship_riding(
                test_day['mid'], preds, 0, test_day['N'],
                y_test, min_confidence=min_conf,
                hold_mode=hold_mode,
                ship_window_bars=ship_window_bars,
                best_bid=test_day.get('best_bid'),
                best_ask=test_day.get('best_ask'),
            )

            day_trades = bt['trades']
            day_pnl = sum(t['net_dollars'] for t in day_trades)
            n_trades = len(day_trades)
            n_correct = sum(1 for t in day_trades if t['correct_direction'])
            n_winners = sum(1 for t in day_trades if t['net_dollars'] > 0)

            fold_results.append({
                'fold': fold,
                'day_idx': test_day['day_idx'],
                'hold_mode': hold_mode,
                'n_trades': n_trades,
                'n_correct': n_correct,
                'n_winners': n_winners,
                'day_pnl': day_pnl,
                'avg_pnl': day_pnl / max(n_trades, 1),
                'peak_avg': np.mean([t['peak_favorable_ticks'] for t in day_trades]) if day_trades else 0,
            })

            if hold_mode == 'trailing':
                all_trades.extend(day_trades)

        if fold % 5 == 0 or fold == n_folds - 1:
            # Summary for trailing mode
            trailing_folds = [r for r in fold_results if r['hold_mode'] == 'trailing']
            if trailing_folds:
                cum_pnl = sum(r['day_pnl'] for r in trailing_folds)
                cum_trades = sum(r['n_trades'] for r in trailing_folds)
                logger.info(f"  Fold {fold}/{n_folds}: "
                            f"trades={cum_trades}, cumPnL=${cum_pnl:+,.0f}")

        # Memory cleanup
        del X_train, y_train, X_test, preds, model
        gc.collect()

    elapsed_train = time.time() - t1
    logger.info(f"  Training complete in {elapsed_train:.0f}s ({elapsed_train/60:.1f}m)")

    # ===========================================================================
    # ANALYSIS & RESULTS
    # ===========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS")
    logger.info("=" * 70)

    for hold_mode in ['trailing', 'momentum', 'fixed']:
        mode_folds = [r for r in fold_results if r['hold_mode'] == hold_mode]
        if not mode_folds:
            continue

        total_trades = sum(r['n_trades'] for r in mode_folds)
        total_pnl = sum(r['day_pnl'] for r in mode_folds)
        total_correct = sum(r['n_correct'] for r in mode_folds)
        total_winners = sum(r['n_winners'] for r in mode_folds)
        n_test_days = len(mode_folds)
        daily_pnls = [r['day_pnl'] for r in mode_folds]
        positive_days = sum(1 for p in daily_pnls if p > 0)

        avg_pnl_per_trade = total_pnl / max(total_trades, 1)
        trades_per_day = total_trades / max(n_test_days, 1)
        win_rate = total_winners / max(total_trades, 1)
        direction_accuracy = total_correct / max(total_trades, 1)

        # Sharpe (daily)
        if len(daily_pnls) > 1:
            mean_daily = np.mean(daily_pnls)
            std_daily = np.std(daily_pnls, ddof=1)
            sharpe = mean_daily / std_daily * np.sqrt(252) if std_daily > 0 else 0
        else:
            sharpe = 0

        # Profit factor
        winners_total = sum(r['day_pnl'] for r in mode_folds if r['day_pnl'] > 0)
        losers_total = abs(sum(r['day_pnl'] for r in mode_folds if r['day_pnl'] < 0))
        pf = winners_total / max(losers_total, 1)

        logger.info(f"\n  --- {hold_mode.upper()} EXIT ---")
        logger.info(f"    Trades: {total_trades} ({trades_per_day:.0f}/day)")
        logger.info(f"    Total PnL: ${total_pnl:+,.0f}")
        logger.info(f"    Avg PnL/trade: ${avg_pnl_per_trade:+,.2f}")
        logger.info(f"    Win rate: {win_rate:.1%}")
        logger.info(f"    Direction accuracy: {direction_accuracy:.1%}")
        logger.info(f"    Sharpe: {sharpe:+.2f}")
        logger.info(f"    Profit factor: {pf:.3f}")
        logger.info(f"    Positive days: {positive_days}/{n_test_days}")
        logger.info(f"    Max daily win: ${max(daily_pnls):+,.0f}")
        logger.info(f"    Max daily loss: ${min(daily_pnls):+,.0f}")

    # Trade analysis for trailing mode
    if all_trades:
        logger.info("\n  --- TRADE ANALYSIS (trailing) ---")
        net_ticks = [t['net_ticks'] for t in all_trades]
        peaks = [t['peak_favorable_ticks'] for t in all_trades]
        holds = [t['hold_bars'] for t in all_trades]
        reasons = {}
        for t in all_trades:
            r = t['exit_reason']
            reasons[r] = reasons.get(r, 0) + 1

        logger.info(f"    Avg net ticks: {np.mean(net_ticks):+.3f}")
        logger.info(f"    Avg peak favorable: {np.mean(peaks):+.2f} ticks")
        logger.info(f"    Avg hold: {np.mean(holds):.0f} bars ({np.mean(holds)/BARS_PER_SEC:.1f}s)")
        logger.info(f"    Exit reasons: {reasons}")

        # By confidence bucket
        conf_buckets = [(0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
        for lo, hi in conf_buckets:
            bucket = [t for t in all_trades if lo <= t['confidence'] < hi]
            if bucket:
                b_pnl = sum(t['net_dollars'] for t in bucket)
                b_wr = sum(1 for t in bucket if t['net_dollars'] > 0) / len(bucket)
                logger.info(f"    Conf [{lo:.0%}-{hi:.0%}]: "
                            f"n={len(bucket)}, PnL=${b_pnl:+,.0f}, WR={b_wr:.1%}")

    # Save results
    result = {
        'timestamp': _ts,
        'config': {
            'n_days': n_days,
            'ship_window_sec': ship_window_sec,
            'move_threshold_ticks': move_threshold,
            'min_confidence': min_conf,
            'quick': quick,
            'total_cost_ticks': TOTAL_COST_TICKS,
            'n_base_features': len(base_feature_names),
            'n_augmented_features': n_features,
        },
        'total_ships': total_ships,
        'fold_results': fold_results,
        'summary': {},
    }

    # Build summary per hold mode
    for hold_mode in ['trailing', 'momentum', 'fixed']:
        mode_folds = [r for r in fold_results if r['hold_mode'] == hold_mode]
        if not mode_folds:
            continue
        total_trades = sum(r['n_trades'] for r in mode_folds)
        total_pnl = sum(r['day_pnl'] for r in mode_folds)
        daily_pnls = [r['day_pnl'] for r in mode_folds]
        mean_daily = np.mean(daily_pnls) if daily_pnls else 0
        std_daily = np.std(daily_pnls, ddof=1) if len(daily_pnls) > 1 else 1
        sharpe = mean_daily / std_daily * np.sqrt(252) if std_daily > 0 else 0

        result['summary'][hold_mode] = {
            'n_trades': int(total_trades),
            'total_pnl': float(total_pnl),
            'sharpe': float(sharpe),
            'win_rate': float(sum(r['n_winners'] for r in mode_folds) / max(total_trades, 1)),
            'trades_per_day': float(total_trades / max(len(mode_folds), 1)),
            'positive_days': int(sum(1 for p in daily_pnls if p > 0)),
            'total_days': int(len(mode_folds)),
        }

    result_path = RESULTS_DIR / f"orderflow_ship_{_ts}.json"
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    logger.info(f"\nResults saved: {result_path}")
    logger.info(f"Log: {_log_file}")

    # Verdict
    best_mode = max(result['summary'].items(),
                    key=lambda x: x[1].get('sharpe', -999)) if result['summary'] else ('none', {})
    best_sharpe = best_mode[1].get('sharpe', 0)
    best_pnl = best_mode[1].get('total_pnl', 0)
    if best_sharpe > 1.0 and best_pnl > 0:
        verdict = "PROMISING"
    elif best_sharpe > 0 and best_pnl > 0:
        verdict = "MARGINAL"
    elif best_pnl > 0:
        verdict = "WEAK"
    else:
        verdict = "NOT VIABLE"
    logger.info(f"\nVERDICT: {verdict} (best: {best_mode[0]}, Sharpe={best_sharpe:.2f})")

    # Notify completion
    try:
        from alpha_discovery.compute_notifier import notify_complete
        notify_complete(
            task_name="orderflow_ship_detector",
            status="completed",
            result_summary=(
                f"Ship detector: {verdict}, best={best_mode[0]} "
                f"Sharpe={best_sharpe:.2f}, PnL=${best_pnl:+,.0f}, "
                f"threshold={move_threshold}t, window={ship_window_sec}s"
            ),
            result_file=str(result_path),
        )
    except Exception:
        pass

    return result


def main():
    parser = argparse.ArgumentParser(description='Orderflow Ship Detector')
    parser.add_argument('--n-days', type=int, default=70)
    parser.add_argument('--ship-window', type=int, default=10,
                        choices=[5, 10, 30], help='Forward window in seconds')
    parser.add_argument('--feature-cache', type=str, default=DEFAULT_FEATURE_CACHE)
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode (20 days max)')
    parser.add_argument('--move-threshold', type=float, default=LARGE_MOVE_TICKS,
                        help='Min ticks for ship event')
    parser.add_argument('--min-confidence', type=float, default=MIN_CONFIDENCE,
                        help='Min P(ship) to enter')
    args = parser.parse_args()

    run_pipeline(
        n_days=args.n_days,
        ship_window_sec=args.ship_window,
        feature_cache=args.feature_cache,
        quick=args.quick,
        move_threshold=args.move_threshold,
        min_conf=args.min_confidence,
    )


if __name__ == '__main__':
    main()
