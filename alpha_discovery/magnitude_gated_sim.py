"""
Magnitude-Gated Limit Order Fill Simulator

Self-contained pipeline that:
1. Loads pre-computed features from cache
2. Trains direction + magnitude models (walk-forward LightGBM)
3. Saves all predictions as NPZ for reuse
4. Simulates limit order fills using magnitude gate
5. Sweeps parameters and reports comprehensive results

Usage:
    # Full training + simulation
    python alpha_discovery/magnitude_gated_sim.py --horizon ret_10s --target-type mfe_net

    # Load saved predictions, skip training
    python alpha_discovery/magnitude_gated_sim.py --load-predictions results/predictions_ret_10s.npz

    # Quick mode (fewer parameter sweeps)
    python alpha_discovery/magnitude_gated_sim.py --horizon ret_10s --target-type mfe_net --quick
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
from typing import Optional, List, Dict, Tuple

import numpy as np
from scipy.stats import spearmanr

# Add project root
LVL3_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(LVL3_ROOT))

RESULTS_DIR = LVL3_ROOT / "alpha_discovery" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging
_log_file = RESULTS_DIR / f"mag_gated_sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
logger = logging.getLogger("mag_gated_sim")

# ============================================================================
# Constants
# ============================================================================
TICK_SIZE = 0.25
TICK_VALUE = 12.50       # ES micro: $12.50 per tick
ES_POINT_VALUE = 50      # $50 per point (ES micro = $5, but we use $50/4 = $12.50/tick)
COMMISSION_RT = 4.70  # HC #52: $4.70 RT (AMP)     # $3.00 round-trip (AMP+Rithmic+CME)
COMMISSION_TICKS = COMMISSION_RT / TICK_VALUE  # 0.24 ticks
HALF_TICK = TICK_SIZE / 2  # 0.125 — bid = mid - 0.125, ask = mid + 0.125
BARS_PER_SEC = 10        # 100ms bars

HORIZONS = {
    'ret_3s': 30, 'ret_5s': 50, 'ret_10s': 100,
    'ret_30s': 300, 'ret_1m': 600,
    'ret_3m': 1800, 'ret_5m': 3000,
}


# ============================================================================
# Trade Record
# ============================================================================
@dataclass
class LimitTrade:
    day: int
    direction: int            # +1 long, -1 short
    signal_strength: float    # direction prediction value
    magnitude_pred: float     # predicted move size (ticks)
    signal_bar: int           # bar where signal fired
    post_bar: int             # bar where limit order posted
    fill_bar: int = -1        # bar where filled (-1 = not filled)
    exit_bar: int = -1
    fill_offset_bars: int = 0 # how long to get filled
    bars_held: int = 0
    entry_price: float = 0.0  # limit entry price
    fill_mid: float = 0.0     # mid at fill time
    exit_mid: float = 0.0
    entry_edge_ticks: float = 0.5  # passive fill = +0.5t
    dir_pnl_ticks: float = 0.0
    exit_edge_ticks: float = 0.0   # +0.5 if limit exit, -0.5 if market
    net_ticks: float = 0.0
    net_dollars: float = 0.0
    exit_type: str = ''       # 'limit_exit', 'take_profit', 'stop_loss', 'timeout'
    filled: bool = False


# ============================================================================
# Phase 1: Data Loading
# ============================================================================
def load_data(feature_cache_dir: str, n_days: Optional[int] = None):
    """Load pre-computed features and mid_prices from cache.

    Skips event features (Ch2 IC near 0) to save ~2 GB memory
    and avoid OOM during event detection on large datasets.
    """
    from alpha_discovery.mbo_alpha_scan import MBOAlphaScanner

    scanner = MBOAlphaScanner()
    load_info = scanner.load_precomputed_features(
        feature_cache_dir=feature_cache_dir,
        n_days=n_days,
        extra_cols=0,  # No event features — saves memory
    )
    logger.info(f"Loaded {load_info['n_days']} days, {load_info['n_snapshots']:,} bars, "
                f"{load_info['n_features']} features ({scanner.features.shape[1]} cols)")

    feature_names = list(scanner.feature_names)

    # Downcast features to float16 to halve memory (12.6 GB -> 6.3 GB for 50 days)
    # LightGBM bins features anyway, so float16 precision is sufficient
    logger.info(f"  Clipping features and downcasting to float16...")
    # Clip to float16 range to avoid overflow/inf
    np.clip(scanner.features, -60000, 60000, out=scanner.features)
    scanner.features = scanner.features.astype(np.float16)
    gc.collect()
    logger.info(f"  Features memory after downcast: "
                f"{scanner.features.nbytes / 1e9:.1f} GB")

    return scanner, feature_names, load_info


# ============================================================================
# Phase 2: Walk-Forward Training (Direction + Magnitude)
# ============================================================================
def train_walk_forward(
    features: np.ndarray,
    target: np.ndarray,
    day_boundaries: list,
    min_train_days: int = 5,
    max_train_days: int = 30,
    target_name: str = 'direction',
) -> Tuple[np.ndarray, list]:
    """
    Walk-forward LightGBM training with ROLLING window (capped at max_train_days).
    Returns FULL-LENGTH prediction array (NaN where no prediction) and per-fold ICs.

    Rolling window prevents OOM: max ~30 days × 234k bars × 290 features = ~7.6 GB copy.
    Also better statistically — markets change, old data becomes stale.
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
        'colsample_bytree': 0.3,  # Fewer features per tree: memory + regularization + decorrelation
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'min_child_samples': 100,
        'verbose': -1,
        'n_jobs': 2,             # Limit threads: n_jobs=-1 multiplies histogram memory by num_cores
        'device': 'cpu',         # Explicit CPU mode
        'max_bin': 63,           # Match pipeline GPU params: 63 bins = 4x less histogram memory
        'force_row_wise': True,  # Avoid column-parallel doubling memory
        'objective': 'regression',
        'metric': 'rmse',
    }

    n_folds = 0
    for test_day in range(min_train_days, n_days):
        train_end_day = test_day - 1  # 1-day purge gap
        # Rolling window: cap training at max_train_days to prevent OOM
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

        # Subsample training rows to prevent OOM
        # 1M samples is plenty for LightGBM with 300 trees
        MAX_TRAIN_SAMPLES = 500_000
        valid_indices = np.where(train_valid)[0]
        if len(valid_indices) > MAX_TRAIN_SAMPLES:
            rng = np.random.default_rng(seed=test_day)
            sampled = rng.choice(valid_indices, MAX_TRAIN_SAMPLES, replace=False)
            sampled.sort()
            X_tr = X_train[sampled].astype(np.float32)  # upcast from float16
            y_tr = y_train[sampled]
        else:
            X_tr = X_train[train_valid].astype(np.float32)
            y_tr = y_train[train_valid]
        X_te = X_test[test_valid].astype(np.float32)
        y_te = y_test[test_valid]

        # Train with 80/20 internal split
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
            sys.stdout.flush()
            for h in logging.getLogger().handlers:
                h.flush()
            continue

        # Free training data immediately
        del X_tr, y_tr
        gc.collect()

        # Map predictions back to full-length array
        valid_positions = np.arange(test_start, test_end)[test_valid]
        n = min(len(valid_positions), len(preds))
        full_preds[valid_positions[:n]] = preds[:n].astype(np.float32)

        # Per-fold IC
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
            logger.info(f"  [{target_name}] Fold {n_folds} (day {test_day}): "
                        f"running IC={ic_so_far:.4f}")
            sys.stdout.flush()
            for h in logging.getLogger().handlers:
                h.flush()

        del model
        gc.collect()

    n_valid = np.isfinite(full_preds).sum()
    overall_ic = 0.0
    if n_valid > 50:
        mask = np.isfinite(full_preds) & np.isfinite(target)
        if mask.sum() > 50:
            overall_ic = float(spearmanr(full_preds[mask], target[mask])[0])

    logger.info(f"  [{target_name}] Complete: {n_folds} folds, {n_valid:,} predictions, "
                f"IC={overall_ic:.4f}")

    return full_preds, fold_ics


def compute_targets(mid_prices, horizon_bars, day_boundaries, target_type='mfe_net'):
    """Compute direction and magnitude targets."""
    N = len(mid_prices)
    n_days = len(day_boundaries) - 1

    # Future mid prices
    future_mid = np.empty(N, dtype=np.float32)
    future_mid[:N - horizon_bars] = mid_prices[horizon_bars:]
    future_mid[N - horizon_bars:] = np.nan

    # NaN-fill day boundary crossings
    for d in range(n_days - 1):
        day_end = day_boundaries[d + 1]
        nan_start = max(day_boundaries[d], day_end - horizon_bars)
        future_mid[nan_start:day_end] = np.nan

    # Direction target
    if target_type == 'return':
        direction_target = (future_mid - mid_prices) / np.maximum(mid_prices, 1.0)
    elif target_type.startswith('mfe'):
        from alpha_discovery.run_mfe_scan import compute_mfe_targets
        hz_name = {30: '3s', 50: '5s', 100: '10s', 300: '30s', 600: '1m', 1800: '3m', 3000: '5m'}.get(horizon_bars, '5s')
        hz_sec_map = {'3s': 3, '5s': 5, '10s': 10, '30s': 30, '1m': 60, '3m': 180, '5m': 300}
        hz_sec = hz_sec_map.get(hz_name, 5)
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
    else:
        direction_target = (future_mid - mid_prices) / np.maximum(mid_prices, 1.0)

    # Magnitude target (absolute move in ticks, always from raw prices)
    magnitude_target = np.abs(future_mid - mid_prices) / TICK_SIZE

    return direction_target, magnitude_target


# ============================================================================
# Phase 3: Limit Order Fill Simulation
# ============================================================================
def simulate_limit_orders(
    mid_prices: np.ndarray,
    direction_preds: np.ndarray,
    magnitude_preds: np.ndarray,
    day_boundaries: list,
    # Gate parameters
    mag_gate_threshold: float = 2.0,    # minimum predicted magnitude (ticks)
    signal_quantile: float = 0.80,      # minimum signal strength percentile
    # Timing parameters
    latency_bars: int = 1,              # bars delay before posting (100ms each)
    fill_horizon_bars: int = 50,        # max bars to wait for fill (5s)
    hold_horizon_bars: int = 100,       # max bars to hold after fill (10s)
    # Exit parameters
    stop_loss_ticks: Optional[float] = None,    # stop loss in ticks
    take_profit_ticks: Optional[float] = None,  # take profit in ticks
    use_limit_exit: bool = True,        # try passive exit first
    # Overlap control
    min_bars_between_signals: int = 50, # minimum spacing between signals
) -> Dict:
    """
    Simulate limit order entries and exits with magnitude gate.

    Fill logic:
        - Long: post limit BUY at best_bid = mid - HALF_TICK
          Fill when future mid drops to or below our limit price
        - Short: post limit SELL at best_ask = mid + HALF_TICK
          Fill when future mid rises to or above our limit price

    Exit logic (in priority order):
        1. Limit exit: post at opposite side, fill when mid crosses
        2. Take profit: if unrealized PnL >= take_profit_ticks
        3. Stop loss: if unrealized PnL <= -stop_loss_ticks
        4. Timeout: after hold_horizon_bars bars
    """
    N = len(mid_prices)
    n_days = len(day_boundaries) - 1

    # Filter to bars with valid predictions
    has_pred = np.isfinite(direction_preds) & np.isfinite(magnitude_preds)

    # Compute signal thresholds from valid predictions
    valid_dir = direction_preds[has_pred]
    if len(valid_dir) < 100:
        return {'error': 'Insufficient predictions', 'n_valid': int(has_pred.sum())}

    abs_dir = np.abs(valid_dir)
    signal_threshold = np.percentile(abs_dir, signal_quantile * 100)

    trades: List[LimitTrade] = []
    n_posted = 0
    n_filled = 0
    n_not_filled = 0
    n_gated_out = 0
    n_signal_weak = 0
    next_allowed_bar = 0

    for day_idx in range(n_days):
        day_start = day_boundaries[day_idx]
        day_end = day_boundaries[day_idx + 1]

        for i in range(day_start, day_end):
            if i < next_allowed_bar:
                continue
            if not has_pred[i]:
                continue

            # Check magnitude gate
            if magnitude_preds[i] < mag_gate_threshold:
                n_gated_out += 1
                continue

            # Check signal strength
            if abs(direction_preds[i]) < signal_threshold:
                n_signal_weak += 1
                continue

            direction = 1 if direction_preds[i] > 0 else -1

            # Post limit order after latency
            post_bar = i + latency_bars
            if post_bar >= day_end:
                continue

            # Limit price
            post_mid = mid_prices[post_bar]
            if direction == 1:
                # Long: post at best_bid
                entry_limit = post_mid - HALF_TICK
            else:
                # Short: post at best_ask
                entry_limit = post_mid + HALF_TICK

            trade = LimitTrade(
                day=day_idx,
                direction=direction,
                signal_strength=float(direction_preds[i]),
                magnitude_pred=float(magnitude_preds[i]),
                signal_bar=i,
                post_bar=post_bar,
                entry_price=entry_limit,
            )
            n_posted += 1

            # Try to fill within fill_horizon
            fill_end = min(post_bar + fill_horizon_bars, day_end)
            filled = False

            for j in range(post_bar + 1, fill_end):
                future_mid = mid_prices[j]
                if direction == 1 and future_mid <= entry_limit:
                    # Long filled: mid dropped to our bid
                    trade.fill_bar = j
                    trade.fill_mid = future_mid
                    trade.fill_offset_bars = j - post_bar
                    trade.filled = True
                    filled = True
                    break
                elif direction == -1 and future_mid >= entry_limit:
                    # Short filled: mid rose to our ask
                    trade.fill_bar = j
                    trade.fill_mid = future_mid
                    trade.fill_offset_bars = j - post_bar
                    trade.filled = True
                    filled = True
                    break

            if not filled:
                n_not_filled += 1
                trade.exit_type = 'not_filled'
                trades.append(trade)
                next_allowed_bar = i + min_bars_between_signals
                continue

            n_filled += 1

            # --- Exit simulation ---
            # Entry edge: +0.5 ticks (passive fill advantage)
            trade.entry_edge_ticks = 0.5

            # Place limit exit order at opposite side of fill mid
            if use_limit_exit:
                if direction == 1:
                    exit_limit = trade.fill_mid + HALF_TICK  # sell at ask
                else:
                    exit_limit = trade.fill_mid - HALF_TICK  # buy at bid
            else:
                exit_limit = None

            exit_end = min(trade.fill_bar + hold_horizon_bars, day_end)
            exited = False

            for j in range(trade.fill_bar + 1, exit_end):
                exit_mid = mid_prices[j]
                bars_held = j - trade.fill_bar

                # Unrealized PnL in ticks
                unrealized = (exit_mid - trade.entry_price) / TICK_SIZE * direction

                # Check limit exit
                if use_limit_exit and exit_limit is not None:
                    if direction == 1 and exit_mid >= exit_limit:
                        trade.exit_bar = j
                        trade.exit_mid = exit_mid
                        trade.bars_held = bars_held
                        # Actual exit price = exit_limit, not mid
                        trade.exit_edge_ticks = (exit_limit - exit_mid) / TICK_SIZE * direction
                        trade.exit_type = 'limit_exit'
                        exited = True
                        break
                    elif direction == -1 and exit_mid <= exit_limit:
                        trade.exit_bar = j
                        trade.exit_mid = exit_mid
                        trade.bars_held = bars_held
                        trade.exit_edge_ticks = (exit_limit - exit_mid) / TICK_SIZE * direction
                        trade.exit_type = 'limit_exit'
                        exited = True
                        break

                # Check take profit
                if take_profit_ticks is not None and unrealized >= take_profit_ticks:
                    trade.exit_bar = j
                    trade.exit_mid = exit_mid
                    trade.bars_held = bars_held
                    trade.exit_edge_ticks = -0.5  # market exit
                    trade.exit_type = 'take_profit'
                    exited = True
                    break

                # Check stop loss
                if stop_loss_ticks is not None and unrealized <= -stop_loss_ticks:
                    trade.exit_bar = j
                    trade.exit_mid = exit_mid
                    trade.bars_held = bars_held
                    trade.exit_edge_ticks = -0.5  # market exit
                    trade.exit_type = 'stop_loss'
                    exited = True
                    break

            if not exited:
                # Timeout: market exit
                trade.exit_bar = exit_end - 1
                trade.exit_mid = mid_prices[trade.exit_bar]
                trade.bars_held = trade.exit_bar - trade.fill_bar
                trade.exit_edge_ticks = -0.5  # market exit
                trade.exit_type = 'timeout'

            # Compute PnL — CORRECTED FORMULA (2026-02-22)
            # dir_pnl is measured from entry_price (bid/ask), NOT mid,
            # so it already includes the passive entry edge.
            # Adding entry_edge again would double-count it.
            trade.dir_pnl_ticks = (trade.exit_mid - trade.entry_price) / TICK_SIZE * direction
            trade.net_ticks = (trade.dir_pnl_ticks +
                              trade.exit_edge_ticks - COMMISSION_TICKS)
            trade.net_dollars = trade.net_ticks * TICK_VALUE

            trades.append(trade)
            next_allowed_bar = i + min_bars_between_signals

    # Compute statistics
    filled_trades = [t for t in trades if t.filled]
    if not filled_trades:
        return {
            'error': 'No filled trades',
            'n_posted': n_posted,
            'n_gated_out': n_gated_out,
            'n_signal_weak': n_signal_weak,
        }

    net_ticks = np.array([t.net_ticks for t in filled_trades])
    net_dollars = np.array([t.net_dollars for t in filled_trades])
    dir_pnl = np.array([t.dir_pnl_ticks for t in filled_trades])

    # Exit type breakdown
    exit_types = {}
    for t in filled_trades:
        if t.exit_type not in exit_types:
            exit_types[t.exit_type] = {'count': 0, 'pnl': [], 'win': 0}
        exit_types[t.exit_type]['count'] += 1
        exit_types[t.exit_type]['pnl'].append(t.net_ticks)
        if t.net_ticks > 0:
            exit_types[t.exit_type]['win'] += 1

    exit_breakdown = {}
    for et, data in exit_types.items():
        pnl_arr = np.array(data['pnl'])
        exit_breakdown[et] = {
            'count': data['count'],
            'pct': data['count'] / len(filled_trades),
            'mean_pnl_ticks': float(pnl_arr.mean()),
            'total_pnl_ticks': float(pnl_arr.sum()),
            'win_rate': data['win'] / data['count'] if data['count'] > 0 else 0,
        }

    # Day-by-day PnL
    day_pnl = {}
    for t in filled_trades:
        if t.day not in day_pnl:
            day_pnl[t.day] = 0.0
        day_pnl[t.day] += t.net_dollars
    day_pnl_arr = np.array(list(day_pnl.values()))

    # Max drawdown
    cum_pnl = np.cumsum(net_dollars)
    peak = np.maximum.accumulate(cum_pnl)
    drawdown = peak - cum_pnl
    max_dd = float(drawdown.max()) if len(drawdown) > 0 else 0

    # Sharpe (annualized from daily)
    if len(day_pnl_arr) > 2 and day_pnl_arr.std() > 0:
        sharpe = float(day_pnl_arr.mean() / day_pnl_arr.std() * np.sqrt(252))
    else:
        sharpe = 0.0

    # Profit factor
    gross_profit = float(net_ticks[net_ticks > 0].sum()) if (net_ticks > 0).any() else 0
    gross_loss = float(abs(net_ticks[net_ticks < 0].sum())) if (net_ticks < 0).any() else 0.001
    profit_factor = gross_profit / gross_loss

    # Win rate
    win_rate = float((net_ticks > 0).mean())

    result = {
        'config': {
            'mag_gate_threshold': mag_gate_threshold,
            'signal_quantile': signal_quantile,
            'latency_bars': latency_bars,
            'fill_horizon_bars': fill_horizon_bars,
            'hold_horizon_bars': hold_horizon_bars,
            'stop_loss_ticks': stop_loss_ticks,
            'take_profit_ticks': take_profit_ticks,
            'use_limit_exit': use_limit_exit,
            'min_bars_between_signals': min_bars_between_signals,
        },
        # Counts
        'n_posted': n_posted,
        'n_filled': n_filled,
        'n_not_filled': n_not_filled,
        'n_gated_out': n_gated_out,
        'n_signal_weak': n_signal_weak,
        'fill_rate': n_filled / n_posted if n_posted > 0 else 0,
        # PnL
        'total_pnl_ticks': float(net_ticks.sum()),
        'total_pnl_dollars': float(net_dollars.sum()),
        'mean_pnl_ticks': float(net_ticks.mean()),
        'mean_pnl_dollars': float(net_dollars.mean()),
        'std_pnl_ticks': float(net_ticks.std()),
        'median_pnl_ticks': float(np.median(net_ticks)),
        # Directional
        'mean_dir_pnl_ticks': float(dir_pnl.mean()),
        'mean_entry_edge': float(np.mean([t.entry_edge_ticks for t in filled_trades])),
        'mean_exit_edge': float(np.mean([t.exit_edge_ticks for t in filled_trades])),
        # Quality
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'sharpe_annualized': sharpe,
        'max_drawdown_dollars': max_dd,
        # Timing
        'mean_fill_offset_sec': float(np.mean([t.fill_offset_bars for t in filled_trades])) / BARS_PER_SEC,
        'mean_bars_held': float(np.mean([t.bars_held for t in filled_trades])),
        'trades_per_day': len(filled_trades) / max(len(day_pnl), 1),
        # Breakdowns
        'exit_type_breakdown': exit_breakdown,
        'n_positive_days': int((day_pnl_arr > 0).sum()),
        'n_negative_days': int((day_pnl_arr < 0).sum()),
        'day_pnl_mean': float(day_pnl_arr.mean()) if len(day_pnl_arr) > 0 else 0,
        'day_pnl_std': float(day_pnl_arr.std()) if len(day_pnl_arr) > 1 else 0,
        # Magnitude stats
        'mean_mag_pred': float(np.mean([t.magnitude_pred for t in filled_trades])),
        'mean_signal_strength': float(np.mean([abs(t.signal_strength) for t in filled_trades])),
    }

    return result


# ============================================================================
# Phase 4: Parameter Sweep
# ============================================================================
def run_parameter_sweep(
    mid_prices: np.ndarray,
    direction_preds: np.ndarray,
    magnitude_preds: np.ndarray,
    day_boundaries: list,
    quick: bool = False,
) -> List[Dict]:
    """Sweep magnitude gate thresholds and timing parameters."""

    if quick:
        gate_thresholds = [1.0, 2.0, 3.0]
        quantiles = [0.70, 0.85]
        hold_horizons = [50, 100]
        stop_losses = [None, 3.0]
        take_profits = [None, 3.0]
        limit_exit_opts = [True]
    else:
        gate_thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
        quantiles = [0.60, 0.70, 0.80, 0.90]
        hold_horizons = [30, 50, 100, 200]
        stop_losses = [None, 2.0, 3.0, 5.0]
        take_profits = [None, 2.0, 3.0, 5.0]
        limit_exit_opts = [True, False]

    results = []

    # Phase 1: Gate sweep (fixed other params)
    logger.info("\n--- PHASE 1: Gate Threshold Sweep ---")
    for gate in gate_thresholds:
        for q in quantiles:
            r = simulate_limit_orders(
                mid_prices, direction_preds, magnitude_preds, day_boundaries,
                mag_gate_threshold=gate,
                signal_quantile=q,
                latency_bars=1,
                fill_horizon_bars=50,
                hold_horizon_bars=100,
                use_limit_exit=True,
            )
            if 'error' not in r:
                logger.info(
                    f"  Gate>{gate:.1f}t Q={q:.0%}: "
                    f"Filled={r['n_filled']:,} ({r['fill_rate']:.1%}) "
                    f"PnL=${r['total_pnl_dollars']:+,.0f} "
                    f"WR={r['win_rate']:.1%} "
                    f"PF={r['profit_factor']:.2f} "
                    f"Sharpe={r['sharpe_annualized']:.2f} "
                    f"$/trade={r['mean_pnl_dollars']:+.2f}"
                )
            else:
                logger.info(f"  Gate>{gate:.1f}t Q={q:.0%}: {r.get('error', 'failed')}")
            results.append(r)

    # Phase 2: Best gate + timing sweep
    # Find best gate config by Sharpe
    valid_results = [r for r in results if 'error' not in r and r['sharpe_annualized'] > 0]
    if not valid_results:
        logger.warning("No profitable configs found in Phase 1!")
        return results

    best = max(valid_results, key=lambda r: r['sharpe_annualized'])
    best_gate = best['config']['mag_gate_threshold']
    best_q = best['config']['signal_quantile']
    logger.info(f"\nBest Phase 1: Gate>{best_gate:.1f}t Q={best_q:.0%} "
                f"Sharpe={best['sharpe_annualized']:.2f}")

    logger.info("\n--- PHASE 2: Timing + Exit Sweep (best gate) ---")
    for hold in hold_horizons:
        for sl in stop_losses:
            for tp in take_profits:
                for limit_exit in limit_exit_opts:
                    r = simulate_limit_orders(
                        mid_prices, direction_preds, magnitude_preds, day_boundaries,
                        mag_gate_threshold=best_gate,
                        signal_quantile=best_q,
                        latency_bars=1,
                        fill_horizon_bars=50,
                        hold_horizon_bars=hold,
                        stop_loss_ticks=sl,
                        take_profit_ticks=tp,
                        use_limit_exit=limit_exit,
                    )
                    if 'error' not in r:
                        sl_str = f"SL={sl:.0f}" if sl else "SL=off"
                        tp_str = f"TP={tp:.0f}" if tp else "TP=off"
                        le_str = "LimExit" if limit_exit else "MktExit"
                        logger.info(
                            f"  Hold={hold} {sl_str} {tp_str} {le_str}: "
                            f"PnL=${r['total_pnl_dollars']:+,.0f} "
                            f"WR={r['win_rate']:.1%} "
                            f"Sharpe={r['sharpe_annualized']:.2f} "
                            f"$/trade={r['mean_pnl_dollars']:+.2f}"
                        )
                    results.append(r)

    return results


# ============================================================================
# Main Pipeline
# ============================================================================
def run_pipeline(args):
    """Full magnitude-gated limit order simulation pipeline."""
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("=" * 80)
    logger.info("MAGNITUDE-GATED LIMIT ORDER FILL SIMULATOR")
    logger.info("=" * 80)

    horizon_bars = HORIZONS.get(args.horizon, 100)

    # ================================================================
    # Load or train predictions
    # ================================================================
    if args.load_predictions:
        logger.info(f"\nLoading predictions from: {args.load_predictions}")
        data = np.load(args.load_predictions)
        mid_prices = data['mid_prices']
        direction_preds = data['direction_preds']
        magnitude_preds = data['magnitude_preds']
        day_boundaries = data['day_boundaries'].tolist()
        direction_target = data.get('direction_target', None)
        magnitude_target = data.get('magnitude_target', None)

        logger.info(f"  Loaded {len(mid_prices):,} bars, "
                    f"{np.isfinite(direction_preds).sum():,} direction predictions, "
                    f"{np.isfinite(magnitude_preds).sum():,} magnitude predictions")

        # Compute overall IC if targets available
        if direction_target is not None:
            mask = np.isfinite(direction_preds) & np.isfinite(direction_target)
            if mask.sum() > 50:
                ic = float(spearmanr(direction_preds[mask], direction_target[mask])[0])
                logger.info(f"  Direction model IC: {ic:.4f}")
        if magnitude_target is not None:
            mask = np.isfinite(magnitude_preds) & np.isfinite(magnitude_target)
            if mask.sum() > 50:
                ic = float(spearmanr(magnitude_preds[mask], magnitude_target[mask])[0])
                logger.info(f"  Magnitude model IC: {ic:.4f}")

    else:
        # Load data
        feature_cache = args.feature_cache or str(
            LVL3_ROOT / "data" / "processed" / "mbo_features_cache"
        )
        logger.info(f"\nLoading data from: {feature_cache}")
        scanner, feature_names, load_info = load_data(feature_cache, args.n_days)

        # Compute targets
        logger.info(f"\nComputing targets: {args.target_type} @ {args.horizon}")
        direction_target, magnitude_target = compute_targets(
            scanner.mid_prices, horizon_bars,
            scanner.day_boundaries, args.target_type,
        )
        logger.info(f"  Direction target: {np.isfinite(direction_target).sum():,} valid")
        logger.info(f"  Magnitude target: {np.isfinite(magnitude_target).sum():,} valid")

        features = scanner.features
        mid_prices = scanner.mid_prices
        day_boundaries = scanner.day_boundaries

        # Train direction model
        logger.info(f"\n{'='*60}")
        logger.info("TRAINING DIRECTION MODEL (walk-forward)")
        logger.info(f"{'='*60}")
        t0 = time.time()
        direction_preds, dir_fold_ics = train_walk_forward(
            features, direction_target, day_boundaries,
            min_train_days=args.min_train_days,
            target_name='direction',
        )
        logger.info(f"  Direction training: {time.time()-t0:.0f}s")

        # Free any lingering training memory before next model
        gc.collect()

        # Train magnitude model
        logger.info(f"\n{'='*60}")
        logger.info("TRAINING MAGNITUDE MODEL (walk-forward)")
        logger.info(f"{'='*60}")
        t0 = time.time()
        magnitude_preds, mag_fold_ics = train_walk_forward(
            features, magnitude_target, day_boundaries,
            min_train_days=args.min_train_days,
            target_name='magnitude',
        )
        logger.info(f"  Magnitude training: {time.time()-t0:.0f}s")

        # Save predictions
        pred_file = RESULTS_DIR / f"predictions_{args.horizon}_{timestamp}.npz"
        np.savez_compressed(
            str(pred_file),
            mid_prices=mid_prices,
            direction_preds=direction_preds,
            magnitude_preds=magnitude_preds,
            direction_target=direction_target,
            magnitude_target=magnitude_target,
            day_boundaries=np.array(day_boundaries),
            dir_fold_ics=np.array(dir_fold_ics),
            mag_fold_ics=np.array(mag_fold_ics),
        )
        logger.info(f"\nPredictions saved: {pred_file.name}")

        # Free feature memory
        del features, scanner
        gc.collect()

    # ================================================================
    # Run simulations
    # ================================================================
    logger.info(f"\n{'='*60}")
    logger.info("LIMIT ORDER FILL SIMULATION")
    logger.info(f"{'='*60}")

    # Quick sanity check: oracle magnitude gate (ground truth)
    if magnitude_target is not None:
        logger.info("\n--- Oracle Magnitude Gate (ground truth, upper bound) ---")
        for gate in [1.0, 2.0, 3.0]:
            oracle_r = simulate_limit_orders(
                mid_prices, direction_preds, magnitude_target, day_boundaries,
                mag_gate_threshold=gate,
                signal_quantile=0.70,
                latency_bars=1,
                fill_horizon_bars=50,
                hold_horizon_bars=100,
                use_limit_exit=True,
            )
            if 'error' not in oracle_r:
                logger.info(
                    f"  ORACLE Gate>{gate:.0f}t: "
                    f"Filled={oracle_r['n_filled']:,} ({oracle_r['fill_rate']:.1%}) "
                    f"PnL=${oracle_r['total_pnl_dollars']:+,.0f} "
                    f"WR={oracle_r['win_rate']:.1%} "
                    f"Sharpe={oracle_r['sharpe_annualized']:.2f}"
                )

    # Main simulation: model-based magnitude gate
    logger.info("\n--- Model-Based Magnitude Gate ---")
    sweep_results = run_parameter_sweep(
        mid_prices, direction_preds, magnitude_preds, day_boundaries,
        quick=args.quick,
    )

    # ================================================================
    # Report
    # ================================================================
    logger.info(f"\n{'='*80}")
    logger.info("RESULTS SUMMARY")
    logger.info(f"{'='*80}")

    valid_results = [r for r in sweep_results if 'error' not in r and r.get('n_filled', 0) > 0]
    if valid_results:
        # Sort by Sharpe
        valid_results.sort(key=lambda r: r['sharpe_annualized'], reverse=True)

        logger.info(f"\nTop 5 Configurations by Sharpe:")
        logger.info(f"{'Gate':>6s} {'Q':>5s} {'Hold':>5s} {'SL':>4s} {'TP':>4s} "
                    f"{'Fills':>7s} {'FillR':>6s} {'WR':>6s} {'PF':>6s} "
                    f"{'Sharpe':>7s} {'Total$':>10s} {'$/trade':>8s}")
        logger.info("-" * 90)

        for r in valid_results[:10]:
            cfg = r['config']
            sl_str = f"{cfg['stop_loss_ticks']:.0f}" if cfg['stop_loss_ticks'] else "off"
            tp_str = f"{cfg['take_profit_ticks']:.0f}" if cfg['take_profit_ticks'] else "off"
            logger.info(
                f"{cfg['mag_gate_threshold']:>6.1f} {cfg['signal_quantile']:>5.0%} "
                f"{cfg['hold_horizon_bars']:>5d} {sl_str:>4s} {tp_str:>4s} "
                f"{r['n_filled']:>7,d} {r['fill_rate']:>6.1%} "
                f"{r['win_rate']:>6.1%} {r['profit_factor']:>6.2f} "
                f"{r['sharpe_annualized']:>7.2f} "
                f"${r['total_pnl_dollars']:>+9,.0f} "
                f"${r['mean_pnl_dollars']:>+7.2f}"
            )

        # Best result details
        best = valid_results[0]
        logger.info(f"\n{'='*60}")
        logger.info(f"BEST CONFIG DETAILS")
        logger.info(f"{'='*60}")
        logger.info(f"Config: {json.dumps(best['config'], indent=2)}")
        logger.info(f"Trades: {best['n_filled']:,} filled / {best['n_posted']:,} posted "
                    f"({best['fill_rate']:.1%} fill rate)")
        logger.info(f"PnL: ${best['total_pnl_dollars']:+,.2f} total "
                    f"(${best['mean_pnl_dollars']:+.2f}/trade)")
        logger.info(f"Win Rate: {best['win_rate']:.1%}")
        logger.info(f"Profit Factor: {best['profit_factor']:.2f}")
        logger.info(f"Sharpe: {best['sharpe_annualized']:.2f}")
        logger.info(f"Max Drawdown: ${best['max_drawdown_dollars']:,.2f}")
        logger.info(f"Trades/Day: {best['trades_per_day']:.1f}")
        logger.info(f"Mean Fill Time: {best['mean_fill_offset_sec']:.2f}s")
        logger.info(f"Mean Hold: {best['mean_bars_held']:.0f} bars "
                    f"({best['mean_bars_held']/BARS_PER_SEC:.1f}s)")
        logger.info(f"Mean Direction PnL: {best['mean_dir_pnl_ticks']:+.3f}t")
        logger.info(f"Mean Entry Edge: {best['mean_entry_edge']:+.3f}t")
        logger.info(f"Mean Exit Edge: {best['mean_exit_edge']:+.3f}t")
        logger.info(f"Gated out: {best['n_gated_out']:,} bars (magnitude too low)")
        logger.info(f"Signal weak: {best['n_signal_weak']:,} bars")

        if best.get('exit_type_breakdown'):
            logger.info(f"\nExit Type Breakdown:")
            for et, data in best['exit_type_breakdown'].items():
                logger.info(f"  {et}: {data['count']:,} ({data['pct']:.1%}) "
                            f"mean_pnl={data['mean_pnl_ticks']:+.2f}t "
                            f"WR={data['win_rate']:.1%}")

        logger.info(f"\nDay Stats: {best['n_positive_days']} winning / "
                    f"{best['n_negative_days']} losing days")
        logger.info(f"Daily PnL: ${best['day_pnl_mean']:+,.2f} mean, "
                    f"${best['day_pnl_std']:,.2f} std")

    else:
        logger.warning("NO PROFITABLE CONFIGURATIONS FOUND!")
        logger.info("This may indicate:")
        logger.info("  1. Alpha insufficient for this horizon")
        logger.info("  2. Fill rates too low (try wider fill horizon)")
        logger.info("  3. Magnitude model predictions not discriminative enough")

    # Save all results
    results_file = RESULTS_DIR / f"mag_gated_sim_{timestamp}.json"
    save_results = {
        'timestamp': timestamp,
        'horizon': args.horizon,
        'target_type': args.target_type,
        'total_configs_tested': len(sweep_results),
        'profitable_configs': len([r for r in valid_results if r.get('total_pnl_dollars', 0) > 0]),
        'top_results': valid_results[:10] if valid_results else [],
        'total_time_sec': time.time() - start_time,
    }
    with open(str(results_file), 'w') as f:
        json.dump(save_results, f, indent=2, default=str)
    logger.info(f"\nResults saved: {results_file.name}")
    logger.info(f"Log: {_log_file.name}")

    total_time = time.time() - start_time
    logger.info(f"\n{'='*80}")
    logger.info(f"PIPELINE COMPLETE — {total_time:.0f}s ({total_time/60:.1f}m)")
    logger.info(f"{'='*80}")

    return save_results


def main():
    parser = argparse.ArgumentParser(description='Magnitude-Gated Limit Order Simulator')
    parser.add_argument('--horizon', type=str, default='ret_10s',
                        choices=list(HORIZONS.keys()),
                        help='Target horizon (default: ret_10s)')
    parser.add_argument('--target-type', type=str, default='mfe_net',
                        choices=['return', 'mfe_net', 'mfe_long', 'mfe_short'],
                        help='Target type (default: mfe_net)')
    parser.add_argument('--n-days', type=int, default=None,
                        help='Number of days to load (default: all)')
    parser.add_argument('--min-train-days', type=int, default=5,
                        help='Minimum training days (default: 5)')
    parser.add_argument('--feature-cache', type=str, default=None,
                        help='Path to feature cache dir')
    parser.add_argument('--load-predictions', type=str, default=None,
                        help='Load predictions from NPZ file (skip training)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: fewer parameter sweeps')
    args = parser.parse_args()

    try:
        results = run_pipeline(args)
        return results
    except Exception as e:
        logger.error(f"PIPELINE FAILED: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
