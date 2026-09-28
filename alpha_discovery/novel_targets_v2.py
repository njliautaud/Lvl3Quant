"""
Novel Training Targets V2 — Proper Model Types for Each Objective

Each training objective gets the RIGHT model architecture and best practices:

1. DIRECT_PNL: Huber regression on realized limit-order PnL
   - Target: actual net PnL in ticks after costs
   - Model: LightGBM Huber loss (robust to PnL outliers/fat tails)
   - Insight: model directly learns what matters — net profit

2. OPTIMAL_ACTION: 3-class classification (BUY/SELL/HOLD)
   - Target: class labels based on whether buying, selling, or doing nothing is best
   - Model: LightGBM multiclass with profit-weighted classes
   - Insight: learns WHEN to act, not just direction

3. FILL_PROBABILITY: Binary classification predicting profitable fills
   - Target: I(limit order fills AND produces positive PnL)
   - Model: LightGBM binary with AUC metric + calibrated probabilities
   - Insight: prioritize signals that actually get executed profitably

4. TIME_TO_MOVE: Binary classification for big-move detection
   - Target: I(|max_move| > threshold within horizon)
   - Model: LightGBM binary + separate direction model → combined 2-stage signal
   - Insight: separate WHEN from WHICH WAY for cleaner learning

5. RISK_ADJUSTED: Vol-normalized return prediction
   - Target: forward_return / realized_vol (Sharpe-style)
   - Model: LightGBM regression, quantile regression for tails
   - Insight: learns to avoid high-vol choppy periods

6. RL_REWARD: Q-value estimation for action selection
   - Target: expected cumulative reward per action (buy/sell/hold)
   - Model: 3 separate LightGBM models (one per action), select max Q
   - Insight: accounts for sequential position management + costs

All evaluated through PRODUCTION SIM — the only metric that matters.

Usage:
    python alpha_discovery/novel_targets_v2.py --n-days 70
    python alpha_discovery/novel_targets_v2.py --n-days 70 --test direct_pnl
    python alpha_discovery/novel_targets_v2.py --n-days 20 --quick
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
_log_file = RESULTS_DIR / f"novel_v2_{_ts}.log"
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
logger = logging.getLogger("novel_v2")

# ES futures constants
TICK_SIZE = 0.25
TICK_VALUE = 12.50
COMMISSION_RT = 4.70  # HC #52: $4.70 RT (AMP)           # $3.00 round-trip commission
COMMISSION_TICKS = COMMISSION_RT / TICK_VALUE  # 0.24t
HALF_SPREAD = 0.5              # ES spread = 1 tick, half = 0.5t
LIMIT_ENTRY_EDGE = 0.5        # Limit entry earns half spread
LIMIT_COST = COMMISSION_TICKS  # 0.24t per RT (limit in + limit out)
BARS_PER_SEC = 10


# ============================================================================
# SHARED UTILITIES
# ============================================================================

def compute_forward_return(mid_prices, horizon_bars, day_boundaries):
    """Forward return in ticks, respecting day boundaries."""
    N = len(mid_prices)
    n_days = len(day_boundaries) - 1
    ret = np.full(N, np.nan, dtype=np.float32)
    for d in range(n_days):
        s, e = day_boundaries[d], day_boundaries[d + 1]
        dl = e - s
        if dl <= horizon_bars:
            continue
        # Vectorized within each day
        ret[s:s + dl - horizon_bars] = (
            mid_prices[s + horizon_bars:e] - mid_prices[s:s + dl - horizon_bars]
        ) / TICK_SIZE
    return ret


def compute_mfe_net(mid_prices, horizon_bars, day_boundaries):
    """Signed max favorable excursion."""
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


def compute_max_move(mid_prices, horizon_bars, day_boundaries):
    """Max |move| within horizon, per bar."""
    N = len(mid_prices)
    n_days = len(day_boundaries) - 1
    max_move = np.full(N, np.nan, dtype=np.float32)
    for d in range(n_days):
        s, e = day_boundaries[d], day_boundaries[d + 1]
        dl = e - s
        for i in range(dl - horizon_bars):
            window = mid_prices[s + i:s + i + horizon_bars + 1]
            max_move[s + i] = np.max(np.abs(window - mid_prices[s + i])) / TICK_SIZE
    return max_move


def compute_realized_vol(mid_prices, window_bars, day_boundaries):
    """Realized vol (std of returns) in ticks, vectorized."""
    N = len(mid_prices)
    n_days = len(day_boundaries) - 1
    vol = np.full(N, np.nan, dtype=np.float32)
    for d in range(n_days):
        s, e = day_boundaries[d], day_boundaries[d + 1]
        day_rets = np.diff(mid_prices[s:e]) / TICK_SIZE
        if len(day_rets) < window_bars:
            continue
        # Cumsum trick for rolling std
        cum = np.cumsum(day_rets)
        cum2 = np.cumsum(day_rets ** 2)
        for i in range(window_bars, len(day_rets)):
            n = window_bars
            s_sum = cum[i] - cum[i - n]
            s_sq = cum2[i] - cum2[i - n]
            variance = (s_sq - s_sum ** 2 / n) / n
            vol[s + i + 1] = np.sqrt(max(variance, 0))
    return vol


def compute_limit_fill_pnl(mid_prices, horizon_bars, day_boundaries):
    """For each bar, simulate a limit order entry and compute net PnL.

    Returns:
        fill_happened: bool array (did the limit order fill?)
        fill_pnl_long: float array (net PnL if we went long, NaN if no fill)
        fill_pnl_short: float array (net PnL if we went short, NaN if no fill)
    """
    N = len(mid_prices)
    n_days = len(day_boundaries) - 1
    fill_happened_long = np.zeros(N, dtype=bool)
    fill_happened_short = np.zeros(N, dtype=bool)
    fill_pnl_long = np.full(N, np.nan, dtype=np.float32)
    fill_pnl_short = np.full(N, np.nan, dtype=np.float32)
    max_wait = min(100, horizon_bars)  # 10s max wait for fill

    for d in range(n_days):
        s, e = day_boundaries[d], day_boundaries[d + 1]
        dl = e - s
        for i in range(dl - horizon_bars):
            gi = s + i
            mid = mid_prices[gi]
            bid = mid - TICK_SIZE / 2
            ask = mid + TICK_SIZE / 2

            # Long: place limit buy at bid
            for w in range(1, max_wait + 1):
                if mid_prices[gi + w] <= bid:
                    # Filled at bid
                    fill_happened_long[gi] = True
                    # Hold for remaining horizon bars
                    exit_bar = min(gi + horizon_bars, e - 1)
                    exit_mid = mid_prices[exit_bar]
                    pnl = (exit_mid - bid) / TICK_SIZE - COMMISSION_TICKS
                    fill_pnl_long[gi] = pnl
                    break

            # Short: place limit sell at ask
            for w in range(1, max_wait + 1):
                if mid_prices[gi + w] >= ask:
                    # Filled at ask
                    fill_happened_short[gi] = True
                    exit_bar = min(gi + horizon_bars, e - 1)
                    exit_mid = mid_prices[exit_bar]
                    pnl = (ask - exit_mid) / TICK_SIZE - COMMISSION_TICKS
                    fill_pnl_short[gi] = pnl
                    break

    return fill_happened_long, fill_happened_short, fill_pnl_long, fill_pnl_short


# ============================================================================
# TARGET 1: DIRECT PNL PREDICTION
# ============================================================================

def build_direct_pnl_target(mid_prices, horizon_bars, day_boundaries):
    """Target = actual limit-order PnL including costs.

    For each bar, compute: max(long_pnl, short_pnl) with sign.
    If neither direction fills or both lose, target = 0.
    Model directly learns what matters: net profit after execution.
    """
    logger.info("  Computing Direct PnL target (limit fill simulation)...")
    t0 = time.time()

    N = len(mid_prices)
    n_days = len(day_boundaries) - 1
    target = np.full(N, np.nan, dtype=np.float32)

    max_wait = min(100, horizon_bars)  # 10s max wait

    for d in range(n_days):
        s, e = day_boundaries[d], day_boundaries[d + 1]
        dl = e - s
        for i in range(dl - horizon_bars):
            gi = s + i
            mid = mid_prices[gi]
            bid = mid - TICK_SIZE / 2
            ask = mid + TICK_SIZE / 2

            # Try long entry at bid
            long_pnl = 0.0
            for w in range(1, max_wait + 1):
                if gi + w >= e:
                    break
                if mid_prices[gi + w] <= bid:
                    exit_bar = min(gi + horizon_bars, e - 1)
                    long_pnl = (mid_prices[exit_bar] - bid) / TICK_SIZE - COMMISSION_TICKS
                    break

            # Try short entry at ask
            short_pnl = 0.0
            for w in range(1, max_wait + 1):
                if gi + w >= e:
                    break
                if mid_prices[gi + w] >= ask:
                    exit_bar = min(gi + horizon_bars, e - 1)
                    short_pnl = (ask - mid_prices[exit_bar]) / TICK_SIZE - COMMISSION_TICKS
                    break

            # Target = best action's PnL with sign
            if long_pnl > short_pnl and long_pnl > 0:
                target[gi] = long_pnl
            elif short_pnl > long_pnl and short_pnl > 0:
                target[gi] = -short_pnl  # Negative = short opportunity
            else:
                target[gi] = 0.0

    elapsed = time.time() - t0
    valid = np.isfinite(target)
    nonzero = valid & (target != 0)
    logger.info(f"  Direct PnL: {valid.sum():,} valid, {nonzero.sum():,} tradeable "
                f"({nonzero.sum()/max(valid.sum(),1):.1%}) [{elapsed:.0f}s]")
    return target


def train_direct_pnl(features, target, day_boundaries, min_train_days=5,
                     max_train_days=30, label='direct_pnl'):
    """Train with Huber loss — robust to PnL fat tails."""
    import lightgbm as lgb

    N = len(target)
    n_days = len(day_boundaries) - 1
    preds = np.full(N, np.nan, dtype=np.float32)
    fold_ics = []
    MAX_TRAIN = 500_000

    params = {
        'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.05,
        'subsample': 0.8, 'colsample_bytree': 0.3,
        'reg_alpha': 0.1, 'reg_lambda': 1.0, 'min_child_samples': 100,
        'verbose': -1, 'n_jobs': 4, 'device': 'cpu',
        'max_bin': 63, 'force_row_wise': True,
        'objective': 'huber',  # Robust to PnL outliers
        'huber_delta': 2.0,    # Transition point from L2 to L1
        'metric': 'huber',
    }

    t0 = time.time()
    n_folds = 0

    for test_day in range(min_train_days, n_days):
        train_end = test_day - 1
        train_start = max(0, train_end - max_train_days + 1)
        ts, te = day_boundaries[train_start], day_boundaries[train_end + 1]
        vs, ve = day_boundaries[test_day], day_boundaries[test_day + 1]

        y_tr = target[ts:te]
        y_te = target[vs:ve]
        tr_valid = np.isfinite(y_tr)
        te_valid = np.isfinite(y_te)
        if tr_valid.sum() < 500 or te_valid.sum() < 100:
            continue

        idx = np.where(tr_valid)[0]
        if len(idx) > MAX_TRAIN:
            rng = np.random.default_rng(seed=test_day)
            idx = np.sort(rng.choice(idx, MAX_TRAIN, replace=False))

        X_tr = features[ts:te][idx].astype(np.float32)
        y_tr_s = y_tr[idx]
        X_te = features[vs:ve][te_valid].astype(np.float32)
        y_te_s = y_te[te_valid]

        split = int(len(X_tr) * 0.8)
        try:
            model = lgb.LGBMRegressor(**params)
            model.fit(X_tr[:split], y_tr_s[:split],
                      eval_set=[(X_tr[split:], y_tr_s[split:])],
                      callbacks=[lgb.early_stopping(30, verbose=False)])
            p = model.predict(X_te)
        except Exception as e:
            logger.warning(f"[{label}] Fold {n_folds} failed: {e}")
            continue

        valid_pos = np.arange(vs, ve)[te_valid]
        n = min(len(valid_pos), len(p))
        preds[valid_pos[:n]] = p[:n].astype(np.float32)

        if len(p) > 10:
            try:
                ic = float(spearmanr(p, y_te_s)[0])
                if np.isfinite(ic):
                    fold_ics.append(ic)
            except Exception:
                pass

        n_folds += 1
        if n_folds % 10 == 0:
            logger.info(f"  [{label}] Fold {n_folds} (day {test_day}): "
                        f"IC={np.mean(fold_ics):.4f} [{time.time()-t0:.0f}s]")

        del model, X_tr, y_tr_s, X_te
        gc.collect()

    mask = np.isfinite(preds) & np.isfinite(target)
    overall_ic = float(spearmanr(preds[mask], target[mask])[0]) if mask.sum() > 50 else 0
    logger.info(f"  [{label}] DONE: {n_folds} folds, IC={overall_ic:.4f} [{time.time()-t0:.0f}s]")
    return preds, fold_ics


# ============================================================================
# TARGET 2: OPTIMAL ACTION CLASSIFICATION
# ============================================================================

def build_optimal_action_target(mid_prices, horizon_bars, day_boundaries):
    """3-class target: BUY(2), HOLD(1), SELL(0).

    Class assignment based on whether the best action is profitable:
    - BUY if long_return > cost_threshold (would profit going long)
    - SELL if -long_return > cost_threshold (would profit going short)
    - HOLD otherwise (move too small to overcome costs)
    """
    logger.info("  Computing Optimal Action target (3-class)...")
    t0 = time.time()

    ret = compute_forward_return(mid_prices, horizon_bars, day_boundaries)
    cost_threshold = LIMIT_COST + 0.5  # 0.74t — need to clear this to profit

    target = np.full(len(ret), np.nan, dtype=np.float32)
    valid = np.isfinite(ret)

    # 3 classes: 0=SELL, 1=HOLD, 2=BUY
    target[valid & (ret > cost_threshold)] = 2.0   # BUY
    target[valid & (ret < -cost_threshold)] = 0.0   # SELL
    target[valid & (np.abs(ret) <= cost_threshold)] = 1.0  # HOLD

    elapsed = time.time() - t0
    n_valid = np.isfinite(target).sum()
    n_buy = (target == 2).sum()
    n_sell = (target == 0).sum()
    n_hold = (target == 1).sum()
    logger.info(f"  Optimal Action: BUY={n_buy:,} ({n_buy/max(n_valid,1):.1%}), "
                f"SELL={n_sell:,} ({n_sell/max(n_valid,1):.1%}), "
                f"HOLD={n_hold:,} ({n_hold/max(n_valid,1):.1%}) [{elapsed:.1f}s]")
    return target


def train_optimal_action(features, target, day_boundaries, min_train_days=5,
                         max_train_days=30, label='optimal_action'):
    """Train 3-class classifier. Output = P(BUY) - P(SELL) as signed signal."""
    import lightgbm as lgb

    N = len(target)
    n_days = len(day_boundaries) - 1
    preds = np.full(N, np.nan, dtype=np.float32)
    fold_metrics = []
    MAX_TRAIN = 500_000

    params = {
        'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.05,
        'subsample': 0.8, 'colsample_bytree': 0.3,
        'reg_alpha': 0.1, 'reg_lambda': 1.0, 'min_child_samples': 100,
        'verbose': -1, 'n_jobs': 4, 'device': 'cpu',
        'max_bin': 63, 'force_row_wise': True,
        'objective': 'multiclass', 'num_class': 3,
        'metric': 'multi_logloss',
    }

    t0 = time.time()
    n_folds = 0

    for test_day in range(min_train_days, n_days):
        train_end = test_day - 1
        train_start = max(0, train_end - max_train_days + 1)
        ts, te = day_boundaries[train_start], day_boundaries[train_end + 1]
        vs, ve = day_boundaries[test_day], day_boundaries[test_day + 1]

        y_tr = target[ts:te]
        y_te = target[vs:ve]
        tr_valid = np.isfinite(y_tr)
        te_valid = np.isfinite(y_te)
        if tr_valid.sum() < 500 or te_valid.sum() < 100:
            continue

        idx = np.where(tr_valid)[0]
        if len(idx) > MAX_TRAIN:
            rng = np.random.default_rng(seed=test_day)
            idx = np.sort(rng.choice(idx, MAX_TRAIN, replace=False))

        X_tr = features[ts:te][idx].astype(np.float32)
        y_tr_s = y_tr[idx].astype(int)
        X_te = features[vs:ve][te_valid].astype(np.float32)
        y_te_s = y_te[te_valid].astype(int)

        # Class weights proportional to inverse frequency (balance classes)
        class_counts = np.bincount(y_tr_s, minlength=3).astype(float)
        class_counts = np.maximum(class_counts, 1)
        # Upweight BUY and SELL relative to HOLD (we care more about getting trades right)
        sample_weights = np.ones(len(y_tr_s), dtype=np.float32)
        for c in [0, 2]:  # SELL, BUY
            mask_c = y_tr_s == c
            sample_weights[mask_c] = len(y_tr_s) / (3 * class_counts[c])
        # HOLD gets lower weight — we care less about predicting it
        mask_hold = y_tr_s == 1
        sample_weights[mask_hold] = 0.5 * len(y_tr_s) / (3 * class_counts[1])

        split = int(len(X_tr) * 0.8)
        try:
            model = lgb.LGBMClassifier(**params)
            model.fit(X_tr[:split], y_tr_s[:split],
                      sample_weight=sample_weights[:split],
                      eval_set=[(X_tr[split:], y_tr_s[split:])],
                      callbacks=[lgb.early_stopping(30, verbose=False)])
            # Get class probabilities
            probs = model.predict_proba(X_te)  # (N, 3): P(SELL), P(HOLD), P(BUY)
        except Exception as e:
            logger.warning(f"[{label}] Fold {n_folds} failed: {e}")
            continue

        # Signal = P(BUY) - P(SELL) — ranges from -1 to +1
        signal = probs[:, 2] - probs[:, 0]

        valid_pos = np.arange(vs, ve)[te_valid]
        n = min(len(valid_pos), len(signal))
        preds[valid_pos[:n]] = signal[:n].astype(np.float32)

        # Metric: accuracy on BUY/SELL classes only
        pred_classes = np.argmax(probs, axis=1)
        action_mask = (y_te_s == 0) | (y_te_s == 2)
        if action_mask.sum() > 0:
            action_acc = float((pred_classes[action_mask] == y_te_s[action_mask]).mean())
            fold_metrics.append(action_acc)

        n_folds += 1
        if n_folds % 10 == 0:
            logger.info(f"  [{label}] Fold {n_folds} (day {test_day}): "
                        f"Action Acc={np.mean(fold_metrics):.4f} [{time.time()-t0:.0f}s]")

        del model, X_tr, probs
        gc.collect()

    logger.info(f"  [{label}] DONE: {n_folds} folds, "
                f"Avg Action Acc={np.mean(fold_metrics):.4f} [{time.time()-t0:.0f}s]")
    return preds, fold_metrics


# ============================================================================
# TARGET 3: FILL PROBABILITY PREDICTION
# ============================================================================

def build_fill_prob_target(mid_prices, horizon_bars, day_boundaries):
    """Binary target: I(limit order fills AND produces positive PnL).

    For each bar, simulate placing a limit BUY at bid:
    - target = 1 if fill happens within 10s AND exit PnL > 0
    - target = 0 otherwise

    We also track short side separately and return the combined.
    """
    logger.info("  Computing Fill Probability target (binary, limit+profitable)...")
    t0 = time.time()

    N = len(mid_prices)
    n_days = len(day_boundaries) - 1
    target = np.full(N, np.nan, dtype=np.float32)
    max_wait = min(100, horizon_bars)

    for d in range(n_days):
        s, e = day_boundaries[d], day_boundaries[d + 1]
        dl = e - s
        for i in range(dl - horizon_bars):
            gi = s + i
            mid = mid_prices[gi]
            bid = mid - TICK_SIZE / 2
            ask = mid + TICK_SIZE / 2

            profitable_fill = False

            # Check long fill
            for w in range(1, max_wait + 1):
                if gi + w >= e:
                    break
                if mid_prices[gi + w] <= bid:
                    exit_bar = min(gi + horizon_bars, e - 1)
                    pnl = (mid_prices[exit_bar] - bid) / TICK_SIZE - COMMISSION_TICKS
                    if pnl > 0:
                        profitable_fill = True
                    break

            # Check short fill
            if not profitable_fill:
                for w in range(1, max_wait + 1):
                    if gi + w >= e:
                        break
                    if mid_prices[gi + w] >= ask:
                        exit_bar = min(gi + horizon_bars, e - 1)
                        pnl = (ask - mid_prices[exit_bar]) / TICK_SIZE - COMMISSION_TICKS
                        if pnl > 0:
                            profitable_fill = True
                        break

            target[gi] = 1.0 if profitable_fill else 0.0

    elapsed = time.time() - t0
    valid = np.isfinite(target)
    pos = (target == 1).sum()
    logger.info(f"  Fill Prob: {valid.sum():,} valid, {pos:,} positive "
                f"({pos/max(valid.sum(),1):.1%} fill+profit rate) [{elapsed:.0f}s]")
    return target


def train_fill_probability(features, target, day_boundaries, min_train_days=5,
                           max_train_days=30, label='fill_prob'):
    """Train binary classifier with AUC metric. Calibrated probabilities."""
    import lightgbm as lgb

    N = len(target)
    n_days = len(day_boundaries) - 1
    preds = np.full(N, np.nan, dtype=np.float32)
    fold_aucs = []
    MAX_TRAIN = 500_000

    # Compute class balance for scale_pos_weight
    valid_target = target[np.isfinite(target)]
    n_pos = (valid_target == 1).sum()
    n_neg = (valid_target == 0).sum()
    spw = n_neg / max(n_pos, 1)

    params = {
        'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.05,
        'subsample': 0.8, 'colsample_bytree': 0.3,
        'reg_alpha': 0.1, 'reg_lambda': 1.0, 'min_child_samples': 100,
        'verbose': -1, 'n_jobs': 4, 'device': 'cpu',
        'max_bin': 63, 'force_row_wise': True,
        'objective': 'binary',
        'metric': 'auc',
        'scale_pos_weight': min(spw, 10.0),  # Cap at 10x
        'is_unbalance': False,
    }

    t0 = time.time()
    n_folds = 0

    for test_day in range(min_train_days, n_days):
        train_end = test_day - 1
        train_start = max(0, train_end - max_train_days + 1)
        ts, te = day_boundaries[train_start], day_boundaries[train_end + 1]
        vs, ve = day_boundaries[test_day], day_boundaries[test_day + 1]

        y_tr = target[ts:te]
        y_te = target[vs:ve]
        tr_valid = np.isfinite(y_tr)
        te_valid = np.isfinite(y_te)
        if tr_valid.sum() < 500 or te_valid.sum() < 100:
            continue

        idx = np.where(tr_valid)[0]
        if len(idx) > MAX_TRAIN:
            rng = np.random.default_rng(seed=test_day)
            idx = np.sort(rng.choice(idx, MAX_TRAIN, replace=False))

        X_tr = features[ts:te][idx].astype(np.float32)
        y_tr_s = y_tr[idx].astype(int)
        X_te = features[vs:ve][te_valid].astype(np.float32)
        y_te_s = y_te[te_valid].astype(int)

        split = int(len(X_tr) * 0.8)
        try:
            model = lgb.LGBMClassifier(**params)
            model.fit(X_tr[:split], y_tr_s[:split],
                      eval_set=[(X_tr[split:], y_tr_s[split:])],
                      callbacks=[lgb.early_stopping(30, verbose=False)])
            p = model.predict_proba(X_te)[:, 1]  # P(profitable fill)
        except Exception as e:
            logger.warning(f"[{label}] Fold {n_folds} failed: {e}")
            continue

        valid_pos = np.arange(vs, ve)[te_valid]
        n = min(len(valid_pos), len(p))
        preds[valid_pos[:n]] = p[:n].astype(np.float32)

        # AUC
        if len(np.unique(y_te_s)) > 1 and len(p) > 10:
            from sklearn.metrics import roc_auc_score
            try:
                auc = float(roc_auc_score(y_te_s, p))
                fold_aucs.append(auc)
            except Exception:
                pass

        n_folds += 1
        if n_folds % 10 == 0:
            logger.info(f"  [{label}] Fold {n_folds} (day {test_day}): "
                        f"AUC={np.mean(fold_aucs):.4f} [{time.time()-t0:.0f}s]")

        del model, X_tr
        gc.collect()

    logger.info(f"  [{label}] DONE: {n_folds} folds, "
                f"AUC={np.mean(fold_aucs):.4f} [{time.time()-t0:.0f}s]")
    return preds, fold_aucs


# ============================================================================
# TARGET 4: TIME-TO-MOVE PREDICTION (2-stage)
# ============================================================================

def build_time_to_move_target(mid_prices, horizon_bars, day_boundaries,
                              threshold_ticks=3.0):
    """Binary: is a big move coming within horizon?

    target = 1 if max(|price_path|) > threshold
    target = 0 otherwise
    """
    logger.info(f"  Computing Time-to-Move target (threshold={threshold_ticks}t)...")
    t0 = time.time()
    max_move = compute_max_move(mid_prices, horizon_bars, day_boundaries)
    target = np.full(len(max_move), np.nan, dtype=np.float32)
    valid = np.isfinite(max_move)
    target[valid] = (max_move[valid] >= threshold_ticks).astype(np.float32)
    elapsed = time.time() - t0
    pos = (target == 1).sum()
    logger.info(f"  Time-to-Move: {valid.sum():,} valid, {pos:,} big moves "
                f"({pos/max(valid.sum(),1):.1%}) [{elapsed:.0f}s]")
    return target


def train_time_to_move(features, target, direction_target, day_boundaries,
                       min_train_days=5, max_train_days=30, label='ttm'):
    """2-stage model: Stage 1 predicts timing (binary), Stage 2 predicts direction.

    Final signal = P(big_move) × sign(direction_prediction) × |direction_prediction|
    """
    import lightgbm as lgb

    N = len(target)
    n_days = len(day_boundaries) - 1
    preds = np.full(N, np.nan, dtype=np.float32)
    fold_metrics = []
    MAX_TRAIN = 500_000

    # Stage 1 params: binary classification for timing
    timing_params = {
        'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.05,
        'subsample': 0.8, 'colsample_bytree': 0.3,
        'reg_alpha': 0.1, 'reg_lambda': 1.0, 'min_child_samples': 100,
        'verbose': -1, 'n_jobs': 4, 'device': 'cpu',
        'max_bin': 63, 'force_row_wise': True,
        'objective': 'binary', 'metric': 'auc',
    }

    # Stage 2 params: regression for direction (only on bars with big moves)
    dir_params = {
        'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.05,
        'subsample': 0.8, 'colsample_bytree': 0.3,
        'reg_alpha': 0.1, 'reg_lambda': 1.0, 'min_child_samples': 100,
        'verbose': -1, 'n_jobs': 4, 'device': 'cpu',
        'max_bin': 63, 'force_row_wise': True,
        'objective': 'regression', 'metric': 'rmse',
    }

    t0 = time.time()
    n_folds = 0

    for test_day in range(min_train_days, n_days):
        train_end = test_day - 1
        train_start = max(0, train_end - max_train_days + 1)
        ts, te = day_boundaries[train_start], day_boundaries[train_end + 1]
        vs, ve = day_boundaries[test_day], day_boundaries[test_day + 1]

        # Timing target
        yt_tr = target[ts:te]
        yt_te = target[vs:ve]
        # Direction target
        yd_tr = direction_target[ts:te]
        yd_te = direction_target[vs:ve]

        tr_valid = np.isfinite(yt_tr) & np.isfinite(yd_tr)
        te_valid = np.isfinite(yt_te) & np.isfinite(yd_te)
        if tr_valid.sum() < 500 or te_valid.sum() < 100:
            continue

        idx = np.where(tr_valid)[0]
        if len(idx) > MAX_TRAIN:
            rng = np.random.default_rng(seed=test_day)
            idx = np.sort(rng.choice(idx, MAX_TRAIN, replace=False))

        X_tr = features[ts:te][idx].astype(np.float32)
        X_te = features[vs:ve][te_valid].astype(np.float32)

        try:
            # Stage 1: Timing model (when to trade)
            model_timing = lgb.LGBMClassifier(**timing_params)
            yt_s = yt_tr[idx].astype(int)
            split = int(len(X_tr) * 0.8)
            model_timing.fit(X_tr[:split], yt_s[:split],
                             eval_set=[(X_tr[split:], yt_s[split:])],
                             callbacks=[lgb.early_stopping(30, verbose=False)])
            timing_prob = model_timing.predict_proba(X_te)[:, 1]

            # Stage 2: Direction model (which way)
            # Train only on bars where big moves happen
            big_move_mask = yt_tr[idx] == 1.0
            if big_move_mask.sum() > 100:
                X_tr_big = X_tr[big_move_mask]
                y_dir_big = yd_tr[idx][big_move_mask]
                split_d = int(len(X_tr_big) * 0.8)
                model_dir = lgb.LGBMRegressor(**dir_params)
                model_dir.fit(X_tr_big[:split_d], y_dir_big[:split_d],
                              eval_set=[(X_tr_big[split_d:], y_dir_big[split_d:])],
                              callbacks=[lgb.early_stopping(30, verbose=False)])
                dir_pred = model_dir.predict(X_te)
            else:
                dir_pred = np.zeros(len(X_te))

            # Combined signal: timing × direction
            # Scale timing_prob to emphasize high-probability bars
            signal = timing_prob * dir_pred

        except Exception as e:
            logger.warning(f"[{label}] Fold {n_folds} failed: {e}")
            continue

        valid_pos = np.arange(vs, ve)[te_valid]
        n = min(len(valid_pos), len(signal))
        preds[valid_pos[:n]] = signal[:n].astype(np.float32)

        # Metric: IC of combined signal vs actual return
        if len(signal) > 10:
            try:
                ic = float(spearmanr(signal, yd_te[te_valid])[0])
                if np.isfinite(ic):
                    fold_metrics.append(ic)
            except Exception:
                pass

        n_folds += 1
        if n_folds % 10 == 0:
            logger.info(f"  [{label}] Fold {n_folds} (day {test_day}): "
                        f"IC={np.mean(fold_metrics):.4f} [{time.time()-t0:.0f}s]")

        del model_timing, X_tr, X_te
        gc.collect()

    logger.info(f"  [{label}] DONE: {n_folds} folds, "
                f"IC={np.mean(fold_metrics):.4f} [{time.time()-t0:.0f}s]")
    return preds, fold_metrics


# ============================================================================
# TARGET 5: RISK-ADJUSTED RETURN
# ============================================================================

def build_risk_adjusted_target(mid_prices, horizon_bars, day_boundaries):
    """Return / realized_vol — Sharpe-like target.

    In low-vol environments, even a 1-tick move is significant.
    In high-vol environments, a 3-tick move is noise.
    Model learns to distinguish signal from noise.
    """
    logger.info("  Computing Risk-Adjusted Return target...")
    t0 = time.time()
    ret = compute_forward_return(mid_prices, horizon_bars, day_boundaries)
    vol = compute_realized_vol(mid_prices, horizon_bars, day_boundaries)

    vol_safe = np.maximum(vol, 0.1)  # Floor at 0.1 ticks
    target = ret / vol_safe
    target = np.clip(target, -10, 10)  # Clip extremes

    elapsed = time.time() - t0
    valid = np.isfinite(target)
    logger.info(f"  Risk-Adj: {valid.sum():,} valid, "
                f"mean={target[valid].mean():.4f}, std={target[valid].std():.4f} [{elapsed:.1f}s]")
    return target


def train_risk_adjusted(features, target, day_boundaries, min_train_days=5,
                        max_train_days=30, label='risk_adj'):
    """Standard LightGBM regression on vol-normalized target.

    Also trains a quantile model to capture tail risk.
    """
    import lightgbm as lgb

    N = len(target)
    n_days = len(day_boundaries) - 1
    preds = np.full(N, np.nan, dtype=np.float32)
    fold_ics = []
    MAX_TRAIN = 500_000

    params = {
        'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.05,
        'subsample': 0.8, 'colsample_bytree': 0.3,
        'reg_alpha': 0.1, 'reg_lambda': 1.0, 'min_child_samples': 100,
        'verbose': -1, 'n_jobs': 4, 'device': 'cpu',
        'max_bin': 63, 'force_row_wise': True,
        'objective': 'regression', 'metric': 'rmse',
    }

    t0 = time.time()
    n_folds = 0

    for test_day in range(min_train_days, n_days):
        train_end = test_day - 1
        train_start = max(0, train_end - max_train_days + 1)
        ts, te = day_boundaries[train_start], day_boundaries[train_end + 1]
        vs, ve = day_boundaries[test_day], day_boundaries[test_day + 1]

        y_tr = target[ts:te]
        y_te = target[vs:ve]
        tr_valid = np.isfinite(y_tr)
        te_valid = np.isfinite(y_te)
        if tr_valid.sum() < 500 or te_valid.sum() < 100:
            continue

        idx = np.where(tr_valid)[0]
        if len(idx) > MAX_TRAIN:
            rng = np.random.default_rng(seed=test_day)
            idx = np.sort(rng.choice(idx, MAX_TRAIN, replace=False))

        X_tr = features[ts:te][idx].astype(np.float32)
        y_tr_s = y_tr[idx]
        X_te = features[vs:ve][te_valid].astype(np.float32)
        y_te_s = y_te[te_valid]

        split = int(len(X_tr) * 0.8)
        try:
            model = lgb.LGBMRegressor(**params)
            model.fit(X_tr[:split], y_tr_s[:split],
                      eval_set=[(X_tr[split:], y_tr_s[split:])],
                      callbacks=[lgb.early_stopping(30, verbose=False)])
            p = model.predict(X_te)
        except Exception as e:
            logger.warning(f"[{label}] Fold {n_folds} failed: {e}")
            continue

        valid_pos = np.arange(vs, ve)[te_valid]
        n = min(len(valid_pos), len(p))
        preds[valid_pos[:n]] = p[:n].astype(np.float32)

        if len(p) > 10:
            try:
                ic = float(spearmanr(p, y_te_s)[0])
                if np.isfinite(ic):
                    fold_ics.append(ic)
            except Exception:
                pass

        n_folds += 1
        if n_folds % 10 == 0:
            logger.info(f"  [{label}] Fold {n_folds} (day {test_day}): "
                        f"IC={np.mean(fold_ics):.4f} [{time.time()-t0:.0f}s]")

        del model, X_tr
        gc.collect()

    mask = np.isfinite(preds) & np.isfinite(target)
    overall_ic = float(spearmanr(preds[mask], target[mask])[0]) if mask.sum() > 50 else 0
    logger.info(f"  [{label}] DONE: {n_folds} folds, IC={overall_ic:.4f} [{time.time()-t0:.0f}s]")
    return preds, fold_ics


# ============================================================================
# TARGET 6: RL REWARD (Q-VALUE ESTIMATION)
# ============================================================================

def build_rl_reward_targets(mid_prices, horizon_bars, day_boundaries):
    """Q-value targets for 3 actions: BUY, SELL, HOLD.

    For each bar, compute the expected cumulative reward for each action:
    - Q(buy)  = limit_buy_pnl if filled, else 0 (missed opportunity cost)
    - Q(sell) = limit_sell_pnl if filled, else 0
    - Q(hold) = 0 (no risk, no reward)

    The model learns to predict Q-values for each action.
    Trading policy: take action with max Q-value.
    """
    logger.info("  Computing RL Q-value targets (3 actions)...")
    t0 = time.time()

    N = len(mid_prices)
    n_days = len(day_boundaries) - 1
    q_buy = np.full(N, np.nan, dtype=np.float32)
    q_sell = np.full(N, np.nan, dtype=np.float32)
    max_wait = min(100, horizon_bars)

    for d in range(n_days):
        s, e = day_boundaries[d], day_boundaries[d + 1]
        dl = e - s
        for i in range(dl - horizon_bars):
            gi = s + i
            mid = mid_prices[gi]
            bid = mid - TICK_SIZE / 2
            ask = mid + TICK_SIZE / 2

            # Q(buy): limit buy at bid
            buy_reward = 0.0
            for w in range(1, max_wait + 1):
                if gi + w >= e:
                    break
                if mid_prices[gi + w] <= bid:
                    exit_bar = min(gi + horizon_bars, e - 1)
                    buy_reward = (mid_prices[exit_bar] - bid) / TICK_SIZE - COMMISSION_TICKS
                    break
            q_buy[gi] = buy_reward

            # Q(sell): limit sell at ask
            sell_reward = 0.0
            for w in range(1, max_wait + 1):
                if gi + w >= e:
                    break
                if mid_prices[gi + w] >= ask:
                    exit_bar = min(gi + horizon_bars, e - 1)
                    sell_reward = (ask - mid_prices[exit_bar]) / TICK_SIZE - COMMISSION_TICKS
                    break
            q_sell[gi] = sell_reward

    elapsed = time.time() - t0
    valid = np.isfinite(q_buy)
    logger.info(f"  RL Q-values: {valid.sum():,} valid")
    logger.info(f"    Q(buy)  mean={q_buy[valid].mean():.3f}  P90={np.percentile(q_buy[valid], 90):.3f}")
    logger.info(f"    Q(sell) mean={q_sell[valid].mean():.3f}  P90={np.percentile(q_sell[valid], 90):.3f}")
    logger.info(f"    [{elapsed:.0f}s]")
    return q_buy, q_sell


def train_rl_reward(features, q_buy, q_sell, day_boundaries,
                    min_train_days=5, max_train_days=30, label='rl'):
    """Train 2 Q-value models (buy and sell). Policy: max(Q_buy, Q_sell, 0).

    Signal = Q_buy - Q_sell (positive = buy, negative = sell).
    Only trade when max(Q_buy, Q_sell) > 0 (expected positive reward).
    """
    import lightgbm as lgb

    N = len(q_buy)
    n_days = len(day_boundaries) - 1
    preds = np.full(N, np.nan, dtype=np.float32)
    fold_rewards = []
    MAX_TRAIN = 500_000

    params = {
        'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.05,
        'subsample': 0.8, 'colsample_bytree': 0.3,
        'reg_alpha': 0.1, 'reg_lambda': 1.0, 'min_child_samples': 100,
        'verbose': -1, 'n_jobs': 4, 'device': 'cpu',
        'max_bin': 63, 'force_row_wise': True,
        'objective': 'huber', 'huber_delta': 2.0,
        'metric': 'huber',
    }

    t0 = time.time()
    n_folds = 0

    for test_day in range(min_train_days, n_days):
        train_end = test_day - 1
        train_start = max(0, train_end - max_train_days + 1)
        ts, te = day_boundaries[train_start], day_boundaries[train_end + 1]
        vs, ve = day_boundaries[test_day], day_boundaries[test_day + 1]

        yb_tr, ys_tr = q_buy[ts:te], q_sell[ts:te]
        yb_te, ys_te = q_buy[vs:ve], q_sell[vs:ve]
        tr_valid = np.isfinite(yb_tr) & np.isfinite(ys_tr)
        te_valid = np.isfinite(yb_te) & np.isfinite(ys_te)
        if tr_valid.sum() < 500 or te_valid.sum() < 100:
            continue

        idx = np.where(tr_valid)[0]
        if len(idx) > MAX_TRAIN:
            rng = np.random.default_rng(seed=test_day)
            idx = np.sort(rng.choice(idx, MAX_TRAIN, replace=False))

        X_tr = features[ts:te][idx].astype(np.float32)
        X_te = features[vs:ve][te_valid].astype(np.float32)
        split = int(len(X_tr) * 0.8)

        try:
            # Train Q(buy) model
            model_buy = lgb.LGBMRegressor(**params)
            model_buy.fit(X_tr[:split], yb_tr[idx][:split],
                          eval_set=[(X_tr[split:], yb_tr[idx][split:])],
                          callbacks=[lgb.early_stopping(30, verbose=False)])
            qb_pred = model_buy.predict(X_te)

            # Train Q(sell) model
            model_sell = lgb.LGBMRegressor(**params)
            model_sell.fit(X_tr[:split], ys_tr[idx][:split],
                           eval_set=[(X_tr[split:], ys_tr[idx][split:])],
                           callbacks=[lgb.early_stopping(30, verbose=False)])
            qs_pred = model_sell.predict(X_te)
        except Exception as e:
            logger.warning(f"[{label}] Fold {n_folds} failed: {e}")
            continue

        # Policy: signal = Q(buy) - Q(sell)
        # Only trade when max(Q_buy, Q_sell) > 0
        signal = qb_pred - qs_pred
        # Zero out when neither action is profitable
        no_trade = (qb_pred <= 0) & (qs_pred <= 0)
        signal[no_trade] = 0.0

        valid_pos = np.arange(vs, ve)[te_valid]
        n = min(len(valid_pos), len(signal))
        preds[valid_pos[:n]] = signal[:n].astype(np.float32)

        # Metric: average predicted reward of chosen actions
        chosen_q = np.maximum(qb_pred, qs_pred)
        actual_q = np.where(qb_pred > qs_pred, yb_te[te_valid], ys_te[te_valid])
        if len(actual_q) > 10:
            try:
                ic = float(spearmanr(chosen_q, actual_q)[0])
                if np.isfinite(ic):
                    fold_rewards.append(ic)
            except Exception:
                pass

        n_folds += 1
        if n_folds % 10 == 0:
            logger.info(f"  [{label}] Fold {n_folds} (day {test_day}): "
                        f"Q-IC={np.mean(fold_rewards):.4f} [{time.time()-t0:.0f}s]")

        del model_buy, model_sell, X_tr, X_te
        gc.collect()

    logger.info(f"  [{label}] DONE: {n_folds} folds, "
                f"Q-IC={np.mean(fold_rewards):.4f} [{time.time()-t0:.0f}s]")
    return preds, fold_rewards


# ============================================================================
# PRODUCTION SIM EVALUATION (shared across all targets)
# ============================================================================

def evaluate_through_sim(preds, mid_prices, day_boundaries, oos_start_day,
                         hold_bars=100, label=''):
    """Full production simulation with limit orders.

    Rules:
    - Trade when |prediction| in top 10% (conviction filter)
    - Direction = sign(prediction)
    - Entry: limit order at bid/ask
    - Fill: mid crosses entry price within 10s
    - Hold: hold_bars after fill
    - Exit: limit at mid (assume fill)
    - Costs: 0.24t RT commission
    - Constraints: max 1 position, 50-bar cooldown between trades
    """
    N = len(mid_prices)
    n_days = len(day_boundaries) - 1
    oos_start = day_boundaries[oos_start_day]

    oos_mask = np.zeros(N, dtype=bool)
    oos_mask[oos_start:] = True
    valid = oos_mask & np.isfinite(preds)

    if valid.sum() < 1000:
        logger.warning(f"  [{label}] Too few valid predictions: {valid.sum()}")
        return None

    # Top 10% conviction — threshold MUST use only IS data (no forward-looking bias).
    # The fallback NEVER uses OOS predictions; when IS data is sparse we use whatever
    # IS predictions exist (even a handful calibrate magnitude), and if there are truly
    # zero IS predictions we conservatively skip all trading for this target.
    abs_pred = np.abs(preds)
    is_mask = (~oos_mask) & np.isfinite(preds)
    if is_mask.sum() > 100:
        threshold = np.nanpercentile(abs_pred[is_mask], 90)
    elif is_mask.sum() > 0:
        # Too few IS predictions to compute a reliable 90th percentile — use the IS
        # median as a conservative threshold (trades only on above-average conviction).
        threshold = np.nanpercentile(abs_pred[is_mask], 50)
        logger.warning(
            f"  [{label}] Only {is_mask.sum()} IS predictions available; "
            f"using IS median ({threshold:.4f}) as conviction threshold instead of P90."
        )
    else:
        # No IS predictions at all — we have no basis for a forward-safe threshold.
        # Trade nothing rather than leak OOS distribution into the filter.
        logger.warning(
            f"  [{label}] Zero IS predictions — skipping sim (no leakage-safe threshold)."
        )
        return None
    trade_mask = valid & (abs_pred >= threshold)

    all_trades = []
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

            # Check pending fill
            if pending and not in_position:
                if bar - pending_bar > 100:
                    pending = False
                else:
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
                    # Market exit: sell at bid (longs) or buy at ask (shorts)
                    exit_price = bid if pos_dir > 0 else ask
                    pnl = pos_dir * (exit_price - pos_fill_price) / TICK_SIZE - COMMISSION_TICKS
                    all_trades.append({
                        'day': d, 'bar': bar, 'direction': pos_dir,
                        'pnl_ticks': pnl, 'pnl_dollars': pnl * TICK_VALUE,
                        'bars_held': bar - pos_fill_bar,
                    })
                    in_position = False
                    cooldown = 50
                    continue

            # New signal
            if not in_position and not pending and cooldown == 0 and trade_mask[gi]:
                direction = int(np.sign(preds[gi]))
                if direction != 0:
                    pending = True
                    pending_dir = direction
                    pending_limit = bid if direction > 0 else ask
                    pending_bar = bar

        # EOD close — forced market exit at bid/ask
        if in_position:
            mid = day_prices[-1]
            eod_bid = mid - TICK_SIZE / 2
            eod_ask = mid + TICK_SIZE / 2
            exit_price = eod_bid if pos_dir > 0 else eod_ask
            pnl = pos_dir * (exit_price - pos_fill_price) / TICK_SIZE - COMMISSION_TICKS
            all_trades.append({
                'day': d, 'bar': dl - 1, 'direction': pos_dir,
                'pnl_ticks': pnl, 'pnl_dollars': pnl * TICK_VALUE,
                'bars_held': dl - 1 - pos_fill_bar, 'eod': True,
            })

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
    pf = abs(pnls[wins].sum() / pnls[losses].sum()) if losses.sum() != 0 else 0

    daily_pnls = []
    for d in range(oos_start_day, n_days):
        day_trades = [t for t in all_trades if t['day'] == d]
        daily_pnls.append(sum(t['pnl_dollars'] for t in day_trades))
    daily_pnls = np.array(daily_pnls)
    sharpe = float(np.mean(daily_pnls) / max(np.std(daily_pnls), 1) * np.sqrt(252))

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
        'total_days': n_oos,
        'max_daily_loss': round(float(daily_pnls.min()), 2),
        'max_daily_win': round(float(daily_pnls.max()), 2),
    }

    logger.info(f"\n  [{label}] PRODUCTION SIM:")
    logger.info(f"    Trades: {n_trades} ({n_trades/max(n_oos,1):.1f}/day)")
    logger.info(f"    PnL: ${total_pnl:+,.0f} total, ${avg_pnl:+.2f}/trade ({pnls.mean():+.3f}t)")
    logger.info(f"    WR: {win_rate:.1%}  PF: {pf:.2f}  Sharpe: {sharpe:.2f}")
    logger.info(f"    Daily: ${daily_pnls.mean():+,.0f}/day  "
                f"Positive: {(daily_pnls>0).sum()}/{n_oos}")

    return result


# ============================================================================
# TEST RUNNER
# ============================================================================

def run_test(test_name, features, mid_prices, day_boundaries, hz_bars,
             oos_start_day, hold_secs=(10, 30)):
    """Run a single training objective test."""
    logger.info(f"\n{'='*70}")
    logger.info(f"TEST: {test_name.upper()}")
    logger.info(f"{'='*70}")

    t0 = time.time()
    results = []

    if test_name == 'direct_pnl':
        target = build_direct_pnl_target(mid_prices, hz_bars, day_boundaries)
        preds, fold_metrics = train_direct_pnl(features, target, day_boundaries)
        metric_name = 'IC'
        metric_val = np.mean(fold_metrics) if fold_metrics else 0

    elif test_name == 'optimal_action':
        target = build_optimal_action_target(mid_prices, hz_bars, day_boundaries)
        preds, fold_metrics = train_optimal_action(features, target, day_boundaries)
        metric_name = 'Action_Acc'
        metric_val = np.mean(fold_metrics) if fold_metrics else 0

    elif test_name == 'fill_probability':
        target = build_fill_prob_target(mid_prices, hz_bars, day_boundaries)
        preds, fold_metrics = train_fill_probability(features, target, day_boundaries)
        # Combine fill prob with direction: need direction model too
        logger.info("  Training direction model for fill_prob combination...")
        ret = compute_forward_return(mid_prices, hz_bars, day_boundaries)
        dir_preds, _ = train_risk_adjusted(features, ret, day_boundaries, label='fill_dir')
        # Combined: fill_prob × sign(direction) × |direction|
        valid = np.isfinite(preds) & np.isfinite(dir_preds)
        combined = np.full(len(preds), np.nan, dtype=np.float32)
        combined[valid] = preds[valid] * dir_preds[valid]
        preds = combined
        metric_name = 'AUC'
        metric_val = np.mean(fold_metrics) if fold_metrics else 0

    elif test_name == 'time_to_move':
        timing_target = build_time_to_move_target(mid_prices, hz_bars, day_boundaries)
        direction_target = compute_forward_return(mid_prices, hz_bars, day_boundaries)
        preds, fold_metrics = train_time_to_move(
            features, timing_target, direction_target, day_boundaries)
        metric_name = 'IC'
        metric_val = np.mean(fold_metrics) if fold_metrics else 0

    elif test_name == 'risk_adjusted':
        target = build_risk_adjusted_target(mid_prices, hz_bars, day_boundaries)
        preds, fold_metrics = train_risk_adjusted(features, target, day_boundaries)
        metric_name = 'IC'
        metric_val = np.mean(fold_metrics) if fold_metrics else 0

    elif test_name == 'rl_reward':
        q_buy, q_sell = build_rl_reward_targets(mid_prices, hz_bars, day_boundaries)
        preds, fold_metrics = train_rl_reward(features, q_buy, q_sell, day_boundaries)
        metric_name = 'Q-IC'
        metric_val = np.mean(fold_metrics) if fold_metrics else 0
        del q_buy, q_sell

    else:
        logger.error(f"Unknown test: {test_name}")
        return []

    # Evaluate through sim at multiple hold periods
    for hold_sec in hold_secs:
        hold_bars = hold_sec * BARS_PER_SEC
        result = evaluate_through_sim(
            preds, mid_prices, day_boundaries, oos_start_day,
            hold_bars=hold_bars, label=f"{test_name}_hold{hold_sec}s")
        if result:
            result['test_name'] = test_name
            result['hold_sec'] = hold_sec
            result[f'fold_{metric_name.lower()}'] = round(metric_val, 4)
            result['train_time_sec'] = round(time.time() - t0, 0)
            results.append(result)

    del preds
    gc.collect()

    elapsed = time.time() - t0
    logger.info(f"  [{test_name}] Total time: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    return results


# ============================================================================
# MAIN
# ============================================================================

ALL_TESTS = ['direct_pnl', 'optimal_action', 'fill_probability',
             'time_to_move', 'risk_adjusted', 'rl_reward']

# Compute-optimized ordering (fastest first for quick feedback)
FAST_ORDER = ['risk_adjusted', 'optimal_action', 'time_to_move',
              'direct_pnl', 'fill_probability', 'rl_reward']


def main():
    parser = argparse.ArgumentParser(description='Novel Training Targets V2')
    parser.add_argument('--n-days', type=int, default=70)
    parser.add_argument('--feature-cache', type=str,
                        default=DEFAULT_FEATURE_CACHE)
    parser.add_argument('--horizon', type=str, default='10s',
                        help='Horizon (3s, 10s, 30s, 1m, 3m)')
    parser.add_argument('--test', type=str, default=None,
                        help='Run single test (direct_pnl, optimal_action, fill_probability, '
                             'time_to_move, risk_adjusted, rl_reward)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: 20 days, fewer hold periods')
    args = parser.parse_args()

    if args.quick:
        args.n_days = min(args.n_days, 20)

    hz_map = {'3s': 30, '10s': 100, '30s': 300, '1m': 600, '3m': 1800, '5m': 3000}
    hz_bars = hz_map.get(args.horizon, 100)

    logger.info("=" * 70)
    logger.info("NOVEL TRAINING TARGETS V2 — PROPER MODEL TYPES")
    logger.info(f"  n_days:    {args.n_days}")
    logger.info(f"  horizon:   {args.horizon} ({hz_bars} bars)")
    logger.info(f"  tests:     {args.test or 'ALL'}")
    logger.info(f"  quick:     {args.quick}")
    logger.info(f"  log:       {_log_file}")
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
    logger.info(f"OOS start: day {oos_start_day} ({n_days - oos_start_day} OOS days)")

    np.clip(features, -60000, 60000, out=features)
    features = features.astype(np.float16)
    gc.collect()

    # Select tests
    if args.test:
        tests_to_run = [args.test]
    else:
        tests_to_run = FAST_ORDER

    hold_secs = (10,) if args.quick else (10, 30, 60, 120)

    # Run tests
    all_results = []
    for test_name in tests_to_run:
        try:
            results = run_test(
                test_name, features, mid_prices, day_boundaries,
                hz_bars, oos_start_day, hold_secs)
            all_results.extend(results)
        except Exception as e:
            logger.error(f"TEST {test_name} FAILED: {e}")
            import traceback
            traceback.print_exc()

    # ================================================================
    # COMPARISON TABLE
    # ================================================================
    logger.info(f"\n\n{'='*80}")
    logger.info("HEAD-TO-HEAD COMPARISON — ALL TARGETS THROUGH PRODUCTION SIM")
    logger.info(f"{'='*80}")

    if all_results:
        sorted_results = sorted(all_results, key=lambda x: x.get('total_pnl_dollars', 0),
                                reverse=True)

        header = (f"{'Test':>20s} {'Hold':>5s}  {'Trades':>6s}  "
                  f"{'Total$':>10s}  {'$/tr':>7s}  {'t/tr':>6s}  "
                  f"{'WR':>6s}  {'PF':>5s}  {'Sharpe':>7s}  {'Time':>5s}")
        logger.info(header)
        logger.info("-" * len(header))

        for r in sorted_results:
            logger.info(
                f"{r['test_name']:>20s} {r.get('hold_sec','?'):>4}s  "
                f"{r['n_trades']:>6d}  "
                f"${r['total_pnl_dollars']:>+9,.0f}  "
                f"${r['avg_pnl_per_trade']:>+6.2f}  "
                f"{r['avg_pnl_ticks']:>+5.3f}  "
                f"{r['win_rate']:>5.1%}  "
                f"{r['profit_factor']:>5.2f}  "
                f"{r['sharpe']:>7.2f}  "
                f"{r.get('train_time_sec', 0):>4.0f}s")

        # Winner
        best = sorted_results[0]
        logger.info(f"\nWINNER: {best['test_name']} (hold={best.get('hold_sec')}s) "
                    f"— ${best['total_pnl_dollars']:+,.0f} total, "
                    f"Sharpe={best['sharpe']:.2f}")

    # Save results
    elapsed = time.time() - t_total
    output = {
        'config': {
            'n_days': args.n_days,
            'horizon': args.horizon,
            'horizon_bars': hz_bars,
            'oos_start_day': oos_start_day,
            'elapsed': elapsed,
        },
        'results': all_results,
    }
    json_path = RESULTS_DIR / f"novel_v2_{_ts}.json"
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"\n{'='*70}")
    logger.info(f"COMPLETE in {elapsed:.0f}s ({elapsed/60:.1f}m)")
    logger.info(f"Results: {json_path}")
    logger.info(f"Log: {_log_file}")
    logger.info(f"{'='*70}")


if __name__ == '__main__':
    main()
