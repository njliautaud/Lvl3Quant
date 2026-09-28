"""
Novel Training Targets — Beyond Direction + Magnitude

Tests fundamentally different training objectives through the production sim.
Instead of predicting "which way" and "how far," these targets ask:

1. COST_ADJUSTED_MFE: MFE_net minus execution costs
   → Model learns to avoid bars where edge < costs

2. TRADEABLE_RETURN: Return × I(|return| > cost_threshold)
   → Model learns to focus on bars where moves overcome costs

3. OPTIMAL_ACTION_VALUE: max(buy_pnl, sell_pnl, 0) with sign
   → Model learns the VALUE of the best action including doing nothing

4. FILL_WEIGHTED_PNL: expected_PnL × fill_probability
   → Model learns which bars would actually get filled profitably

5. RISK_ADJUSTED_RETURN: return / realized_vol
   → Model learns to pick low-risk directional trades

6. TIME_TO_MOVE: I(|move_next_30s| > 3 ticks)
   → Binary: is a big move coming? (then separately predict direction)

All targets evaluated through the PRODUCTION SIM — no isolated IC metrics.

Usage:
    python alpha_discovery/novel_targets.py --n-days 70
    python alpha_discovery/novel_targets.py --n-days 20 --quick
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
from typing import Optional, Dict, Tuple

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
_log_file = RESULTS_DIR / f"novel_targets_{_ts}.log"
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
logger = logging.getLogger("novel_targets")

TICK_SIZE = 0.25
TICK_VALUE = 12.50
COMMISSION_TICKS = 3.00 / TICK_VALUE   # 0.24t round-trip
HALF_SPREAD_TICKS = 0.5                # ES spread is 1 tick, half = 0.5t
LIMIT_ENTRY_EDGE = 0.5                 # Limit entry earns half spread
LIMIT_COST = COMMISSION_TICKS          # 0.24t (limit in + limit out)
MARKET_COST = 1.0 + COMMISSION_TICKS   # 1.24t (spread + commission)
BARS_PER_SEC = 10


# ============================================================================
# Target Computation Functions
# ============================================================================

def compute_forward_return_ticks(mid_prices, horizon_bars, day_boundaries):
    """Simple forward return in ticks."""
    N = len(mid_prices)
    n_days = len(day_boundaries) - 1
    ret = np.full(N, np.nan, dtype=np.float32)
    for d in range(n_days):
        s, e = day_boundaries[d], day_boundaries[d + 1]
        day_len = e - s
        for i in range(day_len - horizon_bars):
            ret[s + i] = (mid_prices[s + i + horizon_bars] - mid_prices[s + i]) / TICK_SIZE
    return ret


def compute_realized_vol(mid_prices, window_bars, day_boundaries):
    """Realized volatility (std of returns) in ticks over window."""
    N = len(mid_prices)
    n_days = len(day_boundaries) - 1
    vol = np.full(N, np.nan, dtype=np.float32)
    for d in range(n_days):
        s, e = day_boundaries[d], day_boundaries[d + 1]
        day_prices = mid_prices[s:e]
        rets = np.diff(day_prices) / TICK_SIZE
        # Rolling std
        for i in range(window_bars, len(day_prices)):
            vol[s + i] = np.std(rets[i - window_bars:i])
    return vol


def target_standard_return(mid_prices, horizon_bars, day_boundaries):
    """Standard: simple forward return in ticks."""
    return compute_forward_return_ticks(mid_prices, horizon_bars, day_boundaries)


def target_mfe_net(mid_prices, horizon_bars, day_boundaries):
    """MFE-net: signed max favorable excursion."""
    from alpha_discovery.run_mfe_scan import compute_mfe_targets
    hz_map = {30: '3s', 50: '5s', 100: '10s', 300: '30s', 600: '1m', 1800: '3m', 3000: '5m'}
    hz_name = hz_map.get(horizon_bars, '10s')
    hz_sec_map = {'3s': 3, '5s': 5, '10s': 10, '30s': 30, '1m': 60, '3m': 180, '5m': 300}
    mfe = compute_mfe_targets(
        mid_prices=mid_prices,
        day_boundaries=day_boundaries,
        sample_interval_ms=100,
        horizons_sec={hz_name: hz_sec_map.get(hz_name, 10)},
        tick_size=TICK_SIZE,
    )
    return mfe[f'mfe_net_{hz_name}']


def target_cost_adjusted_mfe(mid_prices, horizon_bars, day_boundaries):
    """Target 1: MFE_net minus limit order costs.

    Positive only when the best achievable P&L exceeds execution costs.
    Model learns: "is there enough edge HERE to justify a trade?"
    """
    mfe = target_mfe_net(mid_prices, horizon_bars, day_boundaries)
    # Subtract cost: limit entry (earn 0.5t) + limit exit (earn 0.5t) - commission
    # Net cost per trade = commission only = 0.24t
    # But MFE already measures from mid, so the actual available edge is:
    # mfe_net + limit_entry_edge - commission = mfe_net + 0.5 - 0.24 = mfe_net + 0.26
    # We want the target to be 0 when trading isn't profitable
    cost_adjusted = mfe.copy()
    # For positive MFE (long opportunities): need mfe > cost
    # For negative MFE (short opportunities): need |mfe| > cost
    cost = LIMIT_COST  # 0.24 ticks
    # Zero out bars where |MFE| < cost (not worth trading)
    small_edge = np.abs(cost_adjusted) < cost
    cost_adjusted[small_edge] = 0.0
    # Reduce magnitude by cost (so target reflects NET profit)
    pos = cost_adjusted > 0
    neg = cost_adjusted < 0
    cost_adjusted[pos] -= cost
    cost_adjusted[neg] += cost
    return cost_adjusted


def target_tradeable_return(mid_prices, horizon_bars, day_boundaries):
    """Target 2: Forward return, zeroed out when |move| < cost threshold.

    Model learns to predict returns ONLY on bars where the move is
    large enough to overcome transaction costs.
    """
    ret = compute_forward_return_ticks(mid_prices, horizon_bars, day_boundaries)
    # Zero out small moves that can't overcome limit order costs
    threshold = LIMIT_COST + 0.5  # Need at least 0.74t move to be profitable
    small = np.abs(ret) < threshold
    ret_filtered = ret.copy()
    ret_filtered[small] = 0.0
    return ret_filtered


def target_optimal_action_value(mid_prices, horizon_bars, day_boundaries):
    """Target 3: Value of optimal action (buy, sell, or nothing).

    For each bar:
    - buy_value = forward_return + limit_entry_edge - commission
    - sell_value = -forward_return + limit_entry_edge - commission
    - nothing_value = 0
    - target = sign(best_action) * max(buy_value, sell_value, 0)

    This teaches the model to output SIGNED expected profit.
    """
    ret = compute_forward_return_ticks(mid_prices, horizon_bars, day_boundaries)
    net_edge = LIMIT_ENTRY_EDGE - COMMISSION_TICKS  # 0.5 - 0.24 = 0.26

    buy_value = ret + net_edge
    sell_value = -ret + net_edge

    target = np.zeros_like(ret)
    valid = np.isfinite(ret)

    # Best action at each bar
    buy_best = valid & (buy_value > sell_value) & (buy_value > 0)
    sell_best = valid & (sell_value > buy_value) & (sell_value > 0)

    target[buy_best] = buy_value[buy_best]
    target[sell_best] = -sell_value[sell_best]  # Negative = short signal

    return target


def target_risk_adjusted_return(mid_prices, horizon_bars, day_boundaries):
    """Target 5: Return / realized_vol.

    Normalizes returns by recent volatility. Model learns to pick
    trades with good risk/reward ratio, not just big moves.
    """
    ret = compute_forward_return_ticks(mid_prices, horizon_bars, day_boundaries)
    vol = compute_realized_vol(mid_prices, horizon_bars, day_boundaries)

    # Avoid division by zero
    vol_safe = np.maximum(vol, 0.1)
    risk_adj = ret / vol_safe

    # Clip extremes
    risk_adj = np.clip(risk_adj, -10, 10)
    return risk_adj


def target_time_to_move(mid_prices, horizon_bars, day_boundaries, threshold_ticks=3.0):
    """Target 6: Binary - is a big move coming?

    target = 1 if max(|price_path|) > threshold within horizon
           = 0 otherwise

    This predicts WHEN to trade, not which direction.
    Combine with direction model for full signal.
    """
    N = len(mid_prices)
    n_days = len(day_boundaries) - 1
    target = np.full(N, np.nan, dtype=np.float32)

    for d in range(n_days):
        s, e = day_boundaries[d], day_boundaries[d + 1]
        day_prices = mid_prices[s:e]
        day_len = e - s

        for i in range(day_len - horizon_bars):
            future_window = day_prices[i:i + horizon_bars + 1]
            max_move = np.max(np.abs(future_window - day_prices[i])) / TICK_SIZE
            target[s + i] = 1.0 if max_move >= threshold_ticks else 0.0

    return target


# ============================================================================
# Walk-Forward Training
# ============================================================================
def train_walk_forward(features, target, day_boundaries,
                       min_train_days=5, max_train_days=30, label=''):
    """Walk-forward LightGBM. Returns predictions and fold ICs."""
    import lightgbm as lgb
    N = len(target)
    n_days = len(day_boundaries) - 1
    preds = np.full(N, np.nan, dtype=np.float32)
    fold_ics = []
    params = {
        'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.05,
        'subsample': 0.8, 'colsample_bytree': 0.3,
        'reg_alpha': 0.1, 'reg_lambda': 1.0, 'min_child_samples': 100,
        'verbose': -1, 'n_jobs': 4, 'device': 'cpu',
        'max_bin': 63, 'force_row_wise': True,
        'objective': 'regression', 'metric': 'rmse',
    }
    MAX_TRAIN = 500_000
    n_folds = 0
    t0 = time.time()

    for test_day in range(min_train_days, n_days):
        train_end_day = test_day - 1
        train_start_day = max(0, train_end_day - max_train_days + 1)
        ts, te = day_boundaries[train_start_day], day_boundaries[train_end_day + 1]
        vs, ve = day_boundaries[test_day], day_boundaries[test_day + 1]
        y_tr_raw = target[ts:te]
        y_te_raw = target[vs:ve]
        tr_valid = np.isfinite(y_tr_raw)
        te_valid = np.isfinite(y_te_raw)
        if tr_valid.sum() < 500 or te_valid.sum() < 100:
            continue
        idx = np.where(tr_valid)[0]
        if len(idx) > MAX_TRAIN:
            rng = np.random.default_rng(seed=test_day)
            idx = np.sort(rng.choice(idx, MAX_TRAIN, replace=False))
        X_tr = features[ts:te][idx].astype(np.float32)
        y_tr = y_tr_raw[idx]
        X_te = features[vs:ve][te_valid].astype(np.float32)
        y_te = y_te_raw[te_valid]
        split = int(len(X_tr) * 0.8)
        try:
            model = lgb.LGBMRegressor(**params)
            model.fit(X_tr[:split], y_tr[:split],
                      eval_set=[(X_tr[split:], y_tr[split:])],
                      callbacks=[lgb.early_stopping(30, verbose=False)])
            p = model.predict(X_te)
        except Exception as e:
            logger.warning(f"[{label}] Fold failed: {e}")
            continue
        del X_tr, y_tr
        gc.collect()
        valid_pos = np.arange(vs, ve)[te_valid]
        n = min(len(valid_pos), len(p))
        preds[valid_pos[:n]] = p[:n].astype(np.float32)
        if len(p) > 10:
            try:
                ic = float(spearmanr(p, y_te)[0])
                if np.isfinite(ic):
                    fold_ics.append(ic)
            except Exception:
                pass
        n_folds += 1
        if n_folds % 10 == 0:
            logger.info(f"  [{label}] Fold {n_folds} (day {test_day}): "
                        f"IC={np.mean(fold_ics):.4f} [{time.time()-t0:.0f}s]")
        del model
        gc.collect()

    mask = np.isfinite(preds) & np.isfinite(target)
    overall_ic = float(spearmanr(preds[mask], target[mask])[0]) if mask.sum() > 50 else 0
    logger.info(f"  [{label}] DONE: {n_folds} folds, IC={overall_ic:.4f} [{time.time()-t0:.0f}s]")
    return preds, fold_ics


# ============================================================================
# Production Sim Evaluation
# ============================================================================
def evaluate_through_sim(preds, mid_prices, day_boundaries, oos_start_day,
                         hold_bars=100, label=''):
    """
    Evaluate predictions through a realistic trading simulation.

    This is THE definitive evaluation — no isolated IC metrics.

    Rules:
    - Trade when |prediction| is in top 10% (conviction filter)
    - Direction = sign(prediction)
    - Entry: limit order at bid/ask (passive, earn 0.5t)
    - Fill: assume fill when mid crosses entry price within 10s
    - Hold: hold_bars after fill
    - Exit: limit exit at mid (assume fill, 0t cost)
    - Costs: commission 0.24t per RT
    - Risk: no position if already in one
    """
    N = len(mid_prices)
    n_days = len(day_boundaries) - 1
    oos_start = day_boundaries[oos_start_day]

    # Only evaluate OOS
    oos_mask = np.zeros(N, dtype=bool)
    oos_mask[oos_start:] = True
    valid = oos_mask & np.isfinite(preds)

    if valid.sum() < 1000:
        logger.warning(f"  [{label}] Too few valid predictions: {valid.sum()}")
        return None

    # Conviction filter: top 10% absolute prediction strength
    abs_pred = np.abs(preds)
    threshold = np.nanpercentile(abs_pred[valid], 90)
    trade_mask = valid & (abs_pred >= threshold)

    # Simulate trades day by day
    all_trades = []
    for d in range(oos_start_day, n_days):
        s, e = day_boundaries[d], day_boundaries[d + 1]
        day_len = e - s
        day_prices = mid_prices[s:e]

        in_position = False
        position_entry_bar = -1
        position_direction = 0
        position_fill_price = 0.0
        position_fill_bar = -1
        pending_entry = False
        pending_limit = 0.0
        pending_dir = 0
        pending_bar = -1
        cooldown = 0

        for bar in range(day_len):
            global_bar = s + bar
            mid = day_prices[bar]
            bid = mid - TICK_SIZE / 2
            ask = mid + TICK_SIZE / 2

            # Decrement cooldown
            if cooldown > 0:
                cooldown -= 1

            # Check pending entry fill
            if pending_entry and not in_position:
                bars_waiting = bar - pending_bar
                if bars_waiting > 100:  # 10s max wait
                    pending_entry = False
                else:
                    # BUG FIX: removed fill condition offset (audited 2026-02-25)
                    # WRONG was: mid <= pending_limit + TICK_SIZE/2 (simplified to mid <= mid)
                    # Fill check: mid crosses our limit
                    if pending_dir > 0 and mid <= pending_limit:
                        # Long fill
                        in_position = True
                        position_direction = 1
                        position_fill_price = pending_limit
                        position_fill_bar = bar
                        pending_entry = False
                    elif pending_dir < 0 and mid >= pending_limit:
                        # Short fill
                        in_position = True
                        position_direction = -1
                        position_fill_price = pending_limit
                        position_fill_bar = bar
                        pending_entry = False

            # Check position exit
            if in_position:
                bars_held = bar - position_fill_bar
                if bars_held >= hold_bars:
                    # BUG FIX: removed fill condition offset (audited 2026-02-25)
                    # Exit at bid (long) or ask (short), not raw mid
                    exit_price = bid if position_direction > 0 else ask
                    pnl_ticks = position_direction * (exit_price - position_fill_price) / TICK_SIZE
                    net_pnl = pnl_ticks - COMMISSION_TICKS
                    all_trades.append({
                        'day': d, 'bar': bar,
                        'direction': position_direction,
                        'fill_price': position_fill_price,
                        'exit_price': exit_price,
                        'pnl_ticks': net_pnl,
                        'pnl_dollars': net_pnl * TICK_VALUE,
                        'bars_held': bars_held,
                    })
                    in_position = False
                    cooldown = 50  # 5s cooldown
                    continue

            # Generate new signal
            if not in_position and not pending_entry and cooldown == 0:
                if trade_mask[global_bar]:
                    direction = int(np.sign(preds[global_bar]))
                    if direction != 0:
                        # Post limit order
                        pending_entry = True
                        pending_dir = direction
                        pending_limit = bid if direction > 0 else ask
                        pending_bar = bar

        # EOD: force close
        if in_position:
            mid = day_prices[-1]
            pnl_ticks = position_direction * (mid - position_fill_price) / TICK_SIZE
            net_pnl = pnl_ticks - COMMISSION_TICKS - 0.5  # Market exit penalty
            all_trades.append({
                'day': d, 'bar': day_len - 1,
                'direction': position_direction,
                'fill_price': position_fill_price,
                'exit_price': mid,
                'pnl_ticks': net_pnl,
                'pnl_dollars': net_pnl * TICK_VALUE,
                'bars_held': day_len - 1 - position_fill_bar,
                'eod': True,
            })

    if not all_trades:
        logger.warning(f"  [{label}] No trades executed")
        return None

    # Aggregate results
    pnls = np.array([t['pnl_ticks'] for t in all_trades])
    dollars = np.array([t['pnl_dollars'] for t in all_trades])
    n_trades = len(all_trades)
    n_oos_days = n_days - oos_start_day
    wins = pnls > 0
    losses = pnls < 0

    total_pnl = float(dollars.sum())
    avg_pnl = float(dollars.mean())
    win_rate = float(wins.mean())
    pf = abs(pnls[wins].sum() / pnls[losses].sum()) if losses.sum() != 0 else 0

    # Daily P&L for Sharpe
    daily_pnls = []
    for d in range(oos_start_day, n_days):
        day_trades = [t for t in all_trades if t['day'] == d]
        daily_pnls.append(sum(t['pnl_dollars'] for t in day_trades))
    daily_pnls = np.array(daily_pnls)
    sharpe = float(np.mean(daily_pnls) / max(np.std(daily_pnls), 1) * np.sqrt(252))

    # Fill rate
    n_signals = trade_mask.sum()
    fill_rate = n_trades / max(n_signals / n_oos_days, 1) if n_signals > 0 else 0

    result = {
        'label': label,
        'n_trades': n_trades,
        'trades_per_day': n_trades / max(n_oos_days, 1),
        'total_pnl_dollars': total_pnl,
        'avg_pnl_per_trade': avg_pnl,
        'avg_pnl_ticks': float(pnls.mean()),
        'win_rate': win_rate,
        'profit_factor': pf,
        'sharpe': sharpe,
        'avg_daily_pnl': float(daily_pnls.mean()),
        'positive_days': int((daily_pnls > 0).sum()),
        'total_days': n_oos_days,
        'max_daily_loss': float(daily_pnls.min()),
        'max_daily_win': float(daily_pnls.max()),
    }

    logger.info(f"\n  [{label}] PRODUCTION SIM RESULTS:")
    logger.info(f"    Trades: {n_trades} ({n_trades/max(n_oos_days,1):.1f}/day)")
    logger.info(f"    PnL: ${total_pnl:+,.0f} total, ${avg_pnl:+.2f}/trade")
    logger.info(f"    WR: {win_rate:.1%}  PF: {pf:.2f}  Sharpe: {sharpe:.2f}")
    logger.info(f"    Daily: ${daily_pnls.mean():+,.0f}/day  "
                f"Positive: {(daily_pnls>0).sum()}/{n_oos_days}")

    return result


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Novel Training Targets')
    parser.add_argument('--n-days', type=int, default=70)
    parser.add_argument('--feature-cache', type=str,
                        default=DEFAULT_FEATURE_CACHE)
    parser.add_argument('--horizon', type=str, default='10s',
                        help='Evaluation horizon (3s, 10s, 30s, 1m, 3m)')
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()

    if args.quick:
        args.n_days = min(args.n_days, 20)

    hz_bars_map = {'3s': 30, '10s': 100, '30s': 300, '1m': 600, '3m': 1800, '5m': 3000}
    hz_bars = hz_bars_map.get(args.horizon, 100)

    logger.info("=" * 70)
    logger.info("NOVEL TRAINING TARGETS — HEAD-TO-HEAD COMPARISON")
    logger.info(f"  n_days:   {args.n_days}")
    logger.info(f"  horizon:  {args.horizon} ({hz_bars} bars)")
    logger.info(f"  log:      {_log_file}")
    logger.info("=" * 70)

    t_total = time.time()

    # Load data
    from alpha_discovery.mbo_alpha_scan import MBOAlphaScanner
    scanner = MBOAlphaScanner()
    load_info = scanner.load_precomputed_features(
        feature_cache_dir=args.feature_cache, n_days=args.n_days, extra_cols=0)
    features = scanner.features
    mid_prices = scanner.mid_prices
    day_boundaries = scanner.day_boundaries
    n_days = len(day_boundaries) - 1
    oos_start_day = max(5, int(n_days * 0.7))

    logger.info(f"Loaded {n_days} days, {len(mid_prices):,} bars")
    np.clip(features, -60000, 60000, out=features)
    features = features.astype(np.float16)
    gc.collect()

    # Define all targets
    target_funcs = {
        'standard_return': lambda: target_standard_return(mid_prices, hz_bars, day_boundaries),
        'mfe_net': lambda: target_mfe_net(mid_prices, hz_bars, day_boundaries),
        'cost_adjusted_mfe': lambda: target_cost_adjusted_mfe(mid_prices, hz_bars, day_boundaries),
        'tradeable_return': lambda: target_tradeable_return(mid_prices, hz_bars, day_boundaries),
        'optimal_action_value': lambda: target_optimal_action_value(mid_prices, hz_bars, day_boundaries),
        'risk_adjusted_return': lambda: target_risk_adjusted_return(mid_prices, hz_bars, day_boundaries),
    }

    if args.quick:
        # Skip expensive targets in quick mode
        target_funcs = {k: v for k, v in target_funcs.items()
                        if k in ('standard_return', 'cost_adjusted_mfe', 'optimal_action_value')}

    # Train and evaluate each target
    all_results = []
    for target_name, target_func in target_funcs.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"TARGET: {target_name}")
        logger.info(f"{'='*60}")

        t0 = time.time()
        target = target_func()
        n_valid = np.isfinite(target).sum()
        n_nonzero = (np.isfinite(target) & (target != 0)).sum()
        logger.info(f"  Valid: {n_valid:,}  Nonzero: {n_nonzero:,} "
                    f"({n_nonzero/max(n_valid,1):.1%})  [{time.time()-t0:.1f}s]")

        # Target statistics
        t_valid = target[np.isfinite(target)]
        logger.info(f"  Stats: mean={t_valid.mean():.4f}  std={t_valid.std():.4f}  "
                    f"P10={np.percentile(t_valid, 10):.3f}  "
                    f"P90={np.percentile(t_valid, 90):.3f}")

        # Train model
        preds, fold_ics = train_walk_forward(
            features, target, day_boundaries,
            min_train_days=5, max_train_days=30, label=target_name)

        # Evaluate through production sim
        # Use multiple hold periods to find optimal
        for hold_sec in [10, 30]:
            hold_bars = hold_sec * BARS_PER_SEC
            result = evaluate_through_sim(
                preds, mid_prices, day_boundaries, oos_start_day,
                hold_bars=hold_bars,
                label=f"{target_name}_hold{hold_sec}s")
            if result:
                result['target_name'] = target_name
                result['hold_sec'] = hold_sec
                result['fold_ic_mean'] = float(np.mean(fold_ics)) if fold_ics else 0
                all_results.append(result)

        del target, preds
        gc.collect()

    # ================================================================
    # COMPARISON TABLE
    # ================================================================
    logger.info(f"\n\n{'='*70}")
    logger.info("HEAD-TO-HEAD COMPARISON — ALL TARGETS THROUGH PRODUCTION SIM")
    logger.info(f"{'='*70}")

    if all_results:
        sorted_results = sorted(all_results, key=lambda x: x.get('total_pnl_dollars', 0),
                                reverse=True)
        header = f"{'Target':>30s}  {'Hold':>5s}  {'Trades':>7s}  " \
                 f"{'Total$':>10s}  {'$/trade':>8s}  {'WR':>6s}  " \
                 f"{'PF':>5s}  {'Sharpe':>7s}  {'IC':>7s}"
        logger.info(header)
        logger.info("-" * len(header))

        for r in sorted_results:
            logger.info(
                f"{r['label']:>30s}  {r.get('hold_sec','?'):>5s}s  "
                f"{r['n_trades']:>7d}  "
                f"${r['total_pnl_dollars']:>+9,.0f}  "
                f"${r['avg_pnl_per_trade']:>+7.2f}  "
                f"{r['win_rate']:>6.1%}  "
                f"{r['profit_factor']:>5.2f}  "
                f"{r['sharpe']:>7.2f}  "
                f"{r.get('fold_ic_mean', 0):>+7.4f}")

    # Save results
    elapsed = time.time() - t_total
    output = {
        'config': {
            'n_days': args.n_days,
            'horizon': args.horizon,
            'horizon_bars': hz_bars,
            'elapsed': elapsed,
        },
        'results': all_results,
    }
    json_path = RESULTS_DIR / f"novel_targets_{_ts}.json"
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"\n{'='*70}")
    logger.info(f"COMPLETE in {elapsed:.0f}s ({elapsed/60:.1f}m)")
    logger.info(f"Results: {json_path}")
    logger.info(f"Log: {_log_file}")
    logger.info(f"{'='*70}")


if __name__ == '__main__':
    main()
