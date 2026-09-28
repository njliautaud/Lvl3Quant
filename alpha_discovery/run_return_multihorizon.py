"""
Multi-Horizon Return Prediction — Short-Horizon Alpha Discovery

Tests whether GNN+Transformer spatial orderbook understanding captures
directional signal across multiple short horizons.

KEY DESIGN PRINCIPLES:
1. EXCLUDE all vol proxies (rvol, vov, tick_count) — prevents model from just
   learning vol persistence and calling it direction
2. EXCLUDE price/time features — absolute price = day ID, time = intraday bias
3. INCLUDE genuine microstructure: book shape, OFI, aggressive flow, VPIN, toxicity
4. Walk-forward evaluation with 1-day purge gap
5. NaN-fill targets that cross day boundaries (no overnight leakage)

Targets:
- Multi-horizon returns: 3s, 5s, 10s, 15s, 30s, 60s
- Signed tick: sign of next price change (simplest directional target)
- Tick magnitude: how many ticks does it move?
- Aggressive flow: will buys or sells dominate next 5s?

Usage:
    python alpha_discovery/run_return_multihorizon.py
    python alpha_discovery/run_return_multihorizon.py --fast  (fewer folds)
    python alpha_discovery/run_return_multihorizon.py --horizons 3s 5s 10s
"""

import sys
import gc
import json
import time
import logging
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr, ttest_1samp
from typing import Dict, List, Optional, Tuple

# Setup path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from alpha_discovery.mbo_alpha_scan import MBOAlphaScanner, RESULTS_DIR
from alpha_discovery.mbo_features import get_feature_names, TOTAL_FEATURES

# ============================================================================
# LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(RESULTS_DIR / 'return_multihorizon.log', mode='a'),
    ]
)
logger = logging.getLogger("run_return_multihorizon")

# ============================================================================
# FEATURE CACHE
# ============================================================================
FEATURE_CACHE = RESULTS_DIR / "feature_cache"


def _feature_names_hash() -> str:
    import hashlib
    names_str = ",".join(get_feature_names())
    return hashlib.md5(names_str.encode()).hexdigest()[:12]


def load_feature_cache(scanner: MBOAlphaScanner) -> Optional[dict]:
    """Load pre-computed features from disk cache."""
    path = FEATURE_CACHE / "features_alldays.npz"
    if not path.exists():
        return None

    logger.info(f"Loading cached features from {path}...")
    t0 = time.time()
    data = np.load(str(path), allow_pickle=True)

    cached_features = data['features']
    if cached_features.shape[1] != TOTAL_FEATURES:
        logger.warning(
            f"Feature cache STALE: {cached_features.shape[1]} vs {TOTAL_FEATURES} expected."
        )
        data.close()
        return None

    stats_path = str(path).replace('.npz', '_stats.json')
    if Path(stats_path).exists():
        with open(stats_path) as f:
            stats = json.load(f)
        cached_hash = stats.get('feature_names_hash', '')
        current_hash = _feature_names_hash()
        if cached_hash and cached_hash != current_hash:
            logger.warning(f"Feature cache STALE: feature names changed. Will recompute.")
            data.close()
            return None
    else:
        stats = {
            'n_snapshots': len(data['mid_prices']),
            'n_days': len(data['day_boundaries']) - 1,
            'n_features': cached_features.shape[1],
        }

    scanner.features = cached_features
    scanner.mid_prices = data['mid_prices']
    scanner.hour_of_day = data['hour_of_day']
    scanner.time_since_rth = data['time_since_rth']
    scanner.day_boundaries = list(data['day_boundaries'])
    scanner.feature_names = get_feature_names()

    logger.info(f"Cache loaded in {time.time() - t0:.1f}s: {scanner.features.shape}")
    return stats


# ============================================================================
# FEATURES TO EXCLUDE
# ============================================================================

# These are the features we EXCLUDE from direction prediction.
# We want the model to learn from order flow and book structure ONLY.
EXCLUDE_FEATURES_DIRECTION = [
    # --- Absolute price level (acts as day identifier) ---
    'mid', 'best_bid', 'best_ask', 'microprice',

    # --- Time-of-day (known intraday seasonality, not alpha) ---
    'hour_norm', 'minute_norm', 'time_since_rth', 'time_to_close',

    # --- Vol proxies (prevent learning "vol persists" as direction) ---
    'rvol_10', 'rvol_20', 'rvol_50',      # Realized vol (direct vol measure)
    'vov_10', 'vov_20', 'vov_50',         # Vol-of-vol (vol regime)
    'tick_count',                           # Raw event density (vol proxy)

    # --- Event intensity features (pure vol proxies) ---
    'event_int_5', 'event_int_20', 'event_int_50',

    # --- Per-level event densities (rolling, same issue) ---
    'tick_density_5', 'tick_density_20',
]

# The features we KEEP (all genuine microstructure):
# - Book shape: depth_ratio*, weighted_book_imb, bid/ask slope, spread*
# - Order flow: OFI, trade_imb, cancel_trade, net_flow, VPIN
# - Momentum: ret_N, ret_vel_N (price momentum)
# - Dynamics: spread_change, spread_zscore, mid_ret, book_refresh, mid_accel
# - Toxicity: kyle_lambda, price_impact, adverse_sel, toxicity_score
# - Microstructure strategies: iceberg_score, absorption*, spoof_score, etc.
# - MBO-enhanced: modify_rate, fleeting_ratio, aggr_imb, cancel_asym, lifetime
# - Spatial: bid/ask L1-L5 order counts and concentrations
# - Vol-direction interactions: vol_regime, flow_during_vol, aggr_momentum
KEEP_MESSAGE = (
    "KEEPING: book shape, OFI, VPIN, momentum, toxicity, absorption, "
    "aggressive flow, spatial levels, vol-direction interactions"
)


# ============================================================================
# MULTI-HORIZON TARGET COMPUTATION
# ============================================================================

def compute_return_targets(
    mid_prices: np.ndarray,
    day_boundaries: list,
    sample_interval_ms: int = 100,
    horizons_sec: Dict[str, int] = None,
    include_flow_target: bool = True,
    global_features_raw: np.ndarray = None,
) -> Dict[str, np.ndarray]:
    """
    Compute all return prediction targets.

    Returns dict of:
    - 'ret_{horizon}': log return over next N bars
    - 'signed_tick': sign of next 100ms price change (+1/-1, 0s excluded from eval)
    - 'tick_mag': absolute move in ticks over next 10 bars (1s)
    - 'aggr_flow_5s': sign(buy_vol - sell_vol) over next 50 bars (5s)
    """
    if horizons_sec is None:
        horizons_sec = {
            '3s':  3,
            '5s':  5,
            '10s': 10,
            '15s': 15,
            '30s': 30,
            '60s': 60,
        }

    N = len(mid_prices)
    steps_per_sec = 1000.0 / sample_interval_ms
    tick_size = 0.25  # ES futures
    targets = {}

    n_days = len(day_boundaries) - 1
    log_mid = np.log(np.maximum(mid_prices, 1.0))

    # Helper: NaN-fill bars that cross day boundaries
    def nan_fill_boundary_crossings(arr, steps):
        if n_days > 1:
            for d in range(n_days - 1):
                day_end = day_boundaries[d + 1]
                nan_start = max(day_boundaries[d], day_end - steps)
                arr[nan_start:day_end] = np.nan
        return arr

    # --- Multi-horizon log returns ---
    for hz_name, hz_sec in horizons_sec.items():
        steps = int(hz_sec * steps_per_sec)
        if steps >= N:
            logger.warning(f"Horizon {hz_name} ({steps} steps) exceeds N={N}, skipping")
            continue

        future_log = np.empty(N, dtype=np.float32)
        future_log[:N - steps] = log_mid[steps:]
        future_log[N - steps:] = np.nan
        ret = (future_log - log_mid).astype(np.float32)

        # NaN-fill day-boundary crossings
        if n_days > 1:
            for d in range(n_days - 1):
                day_end = day_boundaries[d + 1]
                nan_start = max(day_boundaries[d], day_end - steps)
                ret[nan_start:day_end] = np.nan

        targets[f'ret_{hz_name}'] = ret
        logger.info(f"  ret_{hz_name}: steps={steps}, valid={np.isfinite(ret).sum():,}")

    # --- Signed tick target: sign of next price change ---
    # Most granular: did price tick up or down in next 100ms?
    # NaN where no move (price unchanged — no directional information)
    next_mid = np.empty(N, dtype=np.float32)
    next_mid[:N - 1] = mid_prices[1:]
    next_mid[N - 1] = np.nan
    price_change_1bar = next_mid - mid_prices

    # NaN where no change (model can't predict unchanged)
    signed_tick = np.where(
        np.abs(price_change_1bar) > tick_size * 0.1,
        np.sign(price_change_1bar),
        np.nan
    ).astype(np.float32)
    # NaN fill last bar and day boundaries
    signed_tick = nan_fill_boundary_crossings(signed_tick, 1)
    targets['signed_tick'] = signed_tick
    valid_ticks = np.isfinite(signed_tick)
    pct_nonzero = valid_ticks.sum() / N
    logger.info(f"  signed_tick: {valid_ticks.sum():,} valid ({pct_nonzero:.1%} of bars have moves)")

    # --- Tick magnitude: abs move in ticks over 1s (10 bars) ---
    steps_1s = int(1.0 * steps_per_sec)
    future_mid_1s = np.empty(N, dtype=np.float32)
    future_mid_1s[:N - steps_1s] = mid_prices[steps_1s:]
    future_mid_1s[N - steps_1s:] = np.nan
    tick_mag = np.abs(future_mid_1s - mid_prices) / tick_size
    tick_mag = nan_fill_boundary_crossings(tick_mag, steps_1s)
    targets['tick_mag_1s'] = tick_mag.astype(np.float32)
    logger.info(f"  tick_mag_1s: valid={np.isfinite(tick_mag).sum():,}, "
                f"mean={np.nanmean(tick_mag):.3f} ticks")

    # --- Aggressive flow dominance (next 5s) ---
    # sign(sum(buy_vol) - sum(sell_vol)) over next 50 bars
    # Tests: can we predict which side will be more aggressive?
    if include_flow_target and global_features_raw is not None:
        from alpha_discovery.mbo_features import COL_BUY_VOL, COL_SELL_VOL
        buy_vol = global_features_raw[:, COL_BUY_VOL]
        sell_vol = global_features_raw[:, COL_SELL_VOL]
        steps_5s = int(5.0 * steps_per_sec)

        # Future cumulative flow imbalance
        signed_flow = buy_vol - sell_vol

        # Rolling sum forward (using cumsum trick, reversed)
        flow_fwd = np.full(N, np.nan, dtype=np.float32)
        # Use sliding window: sum of signed_flow[i:i+steps_5s]
        cs = np.cumsum(signed_flow.astype(np.float64))
        cs_pad = np.concatenate([[0.0], cs])
        valid_n = N - steps_5s
        flow_fwd[:valid_n] = (cs_pad[steps_5s:steps_5s + valid_n] - cs_pad[:valid_n]).astype(np.float32)

        # Normalize: only sign matters
        aggr_flow_sign = np.where(
            np.abs(flow_fwd) > 0.5,
            np.sign(flow_fwd),
            np.nan
        ).astype(np.float32)
        aggr_flow_sign = nan_fill_boundary_crossings(aggr_flow_sign, steps_5s)
        targets['aggr_flow_5s'] = aggr_flow_sign
        valid_flow = np.isfinite(aggr_flow_sign)
        logger.info(f"  aggr_flow_5s: {valid_flow.sum():,} valid "
                    f"({valid_flow.mean():.1%} of bars)")
    else:
        logger.info("  aggr_flow_5s: skipped (no raw global features)")

    return targets


# ============================================================================
# WALK-FORWARD EVALUATION (adapted for return targets)
# ============================================================================

def walk_forward_evaluate_return(
    scanner: MBOAlphaScanner,
    target: np.ndarray,
    target_name: str,
    exclude_features: List[str],
    min_train_days: int = 3,
    is_classification: bool = False,
) -> dict:
    """
    Walk-forward LightGBM evaluation focused on return prediction.

    Key differences from base scanner:
    - Spearman IC for continuous targets (IC = directional ranking quality)
    - For binary/sign targets: also reports AUC and accuracy
    - Consistent fold structure with purge gap
    """
    import lightgbm as lgb

    n_days = len(scanner.day_boundaries) - 1
    if n_days < min_train_days + 1:
        return {
            'error': f'Need {min_train_days + 1} days, have {n_days}',
            'horizon': target_name,
            'target': target_name,
        }

    # Build feature mask
    keep_mask = np.array([fn not in exclude_features for fn in scanner.feature_names])
    features_use = scanner.features[:, keep_mask]
    feature_names_use = [fn for fn in scanner.feature_names if fn not in exclude_features]
    n_features_use = len(feature_names_use)
    logger.info(f"  Using {n_features_use} features (excluded {keep_mask.sum() - n_features_use} ... wait, excluded {(~keep_mask).sum()})")

    # LightGBM params
    params = {
        'n_estimators': 500,
        'max_depth': 6,
        'learning_rate': 0.03,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'min_child_samples': 100,
        'verbose': -1,
        'n_jobs': -1,
    }

    if is_classification:
        params['objective'] = 'binary'
        params['metric'] = 'auc'
    else:
        params['objective'] = 'regression'
        params['metric'] = 'rmse'

    all_preds = []
    all_actuals = []
    all_hours = []
    all_rth_frac = []
    all_test_features = []
    fold_ics = []
    fold_metrics = []
    feature_importance = np.zeros(n_features_use)

    for test_day in range(min_train_days, n_days):
        train_end_day = test_day - 1
        train_start = scanner.day_boundaries[0]
        train_end = scanner.day_boundaries[train_end_day + 1]

        test_start = scanner.day_boundaries[test_day]
        test_end = scanner.day_boundaries[test_day + 1]

        X_train = features_use[train_start:train_end]
        y_train = target[train_start:train_end]
        X_test = features_use[test_start:test_end]
        y_test = target[test_start:test_end]

        # Remove NaN targets
        train_valid = np.isfinite(y_train)
        test_valid = np.isfinite(y_test)

        if train_valid.sum() < 500 or test_valid.sum() < 50:
            logger.warning(f"  Day {test_day}: train={train_valid.sum()}, test={test_valid.sum()} — skipping")
            continue

        X_tr = X_train[train_valid]
        y_tr = y_train[train_valid]
        X_te = X_test[test_valid]
        y_te = y_test[test_valid]

        # For classification, convert to 0/1
        if is_classification:
            y_tr_fit = (y_tr > 0).astype(int)
            y_te_eval = (y_te > 0).astype(int)
        else:
            y_tr_fit = y_tr
            y_te_eval = y_te

        split = int(len(X_tr) * 0.8)
        try:
            if is_classification:
                model = lgb.LGBMClassifier(**params)
            else:
                model = lgb.LGBMRegressor(**params)
            model.fit(
                X_tr[:split], y_tr_fit[:split],
                eval_set=[(X_tr[split:], y_tr_fit[split:])],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
        except Exception as e:
            logger.warning(f"  Training failed day {test_day}: {e}")
            continue

        if is_classification:
            preds = model.predict_proba(X_te)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_te)
        else:
            preds = model.predict(X_te)

        all_preds.append(preds)
        all_actuals.append(y_te)  # raw (not binarized) for IC calculation
        all_test_features.append(X_te)

        test_hours = scanner.hour_of_day[test_start:test_end][test_valid]
        test_rth = scanner.time_since_rth[test_start:test_end][test_valid]
        all_hours.append(test_hours)
        all_rth_frac.append(test_rth)

        # Per-fold metrics
        if len(preds) > 10:
            try:
                ic_fold = spearmanr(preds, y_te)[0]
                if np.isfinite(ic_fold):
                    fold_ics.append(float(ic_fold))
                    hr_fold = float((np.sign(preds - np.median(preds)) == np.sign(y_te)).mean())
                    fold_metrics.append({
                        'day': test_day,
                        'ic': float(ic_fold),
                        'hit_rate': hr_fold,
                        'n_samples': int(len(preds)),
                        'train_size': int(train_valid.sum()),
                    })
            except Exception:
                pass

        if hasattr(model, 'feature_importances_'):
            feature_importance += model.feature_importances_

        del model
        gc.collect()

    if not all_preds:
        return {'error': 'No valid predictions', 'target': target_name}

    predictions = np.concatenate(all_preds)
    actuals = np.concatenate(all_actuals)
    hours = np.concatenate(all_hours)
    rth_fracs = np.concatenate(all_rth_frac)

    valid = np.isfinite(predictions) & np.isfinite(actuals)
    p, a = predictions[valid], actuals[valid]
    h = hours[valid]
    rf = rth_fracs[valid]

    if len(p) < 50:
        return {'error': f'Too few predictions: {len(p)}', 'target': target_name}

    # ============================================================
    # Metrics
    # ============================================================
    ic = float(spearmanr(p, a)[0])
    hr = float((np.sign(p - np.median(p)) == np.sign(a)).mean())

    # Profit factor: total profit of correct direction vs total loss of wrong direction
    winners = np.abs(a[np.sign(p - np.median(p)) == np.sign(a)]).sum()
    losers = np.abs(a[np.sign(p - np.median(p)) != np.sign(a)]).sum()
    pf = float(winners / losers) if losers > 0 else 0.0

    # ICIR and t-stat across folds
    if len(fold_ics) > 2:
        ic_mean = float(np.mean(fold_ics))
        ic_std = float(np.std(fold_ics))
        icir = ic_mean / ic_std if ic_std > 0 else 0.0
        tstat = ic_mean / ic_std * np.sqrt(len(fold_ics)) if ic_std > 0 else 0.0
        try:
            from scipy.stats import ttest_1samp
            _, pvalue = ttest_1samp(fold_ics, 0)
            pvalue = float(pvalue)
        except Exception:
            pvalue = 1.0
    else:
        ic_mean, ic_std = ic, 0.0
        icir, tstat, pvalue = 0.0, 0.0, 1.0

    # Sharpe (for return targets): sign(pred) * actual_return, annualized
    sharpe = 0.0
    bar_pnl = np.sign(p - np.median(p)) * a
    if np.std(bar_pnl) > 0:
        # 100ms bars * 6.5h * 252 days = 5,913,000 bars/year
        bars_per_year = 252 * 6.5 * 3600 * (1000.0 / 100)
        sharpe = float(np.mean(bar_pnl) / np.std(bar_pnl) * np.sqrt(bars_per_year))

    # ============================================================
    # Session phase breakdown
    # ============================================================
    session_stats = {}
    phases = {
        'open_30min': (0.0, 0.075),
        'morning':    (0.075, 0.38),
        'midday':     (0.38, 0.62),
        'afternoon':  (0.62, 0.92),
        'close_30min': (0.92, 1.0),
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
                'hit_rate': float((np.sign(p_s - np.median(p)) == np.sign(a_s)).mean()),
                'n_samples': int(mask.sum()),
            }

    # ============================================================
    # Volatility regime analysis
    # ============================================================
    regime_stats = {}
    a_abs = np.abs(a)
    vol_p25 = np.percentile(a_abs[a_abs > 0], 25) if (a_abs > 0).sum() > 0 else 0.0
    vol_p75 = np.percentile(a_abs[a_abs > 0], 75) if (a_abs > 0).sum() > 0 else 0.0
    regimes = {
        'low_vol':  a_abs <= vol_p25,
        'mid_vol':  (a_abs > vol_p25) & (a_abs <= vol_p75),
        'high_vol': a_abs > vol_p75,
    }
    for regime_name, mask in regimes.items():
        if mask.sum() > 50:
            p_r, a_r = p[mask], a[mask]
            try:
                ic_r = float(spearmanr(p_r, a_r)[0])
            except Exception:
                ic_r = 0.0
            regime_stats[regime_name] = {
                'ic': ic_r,
                'hit_rate': float((np.sign(p_r - np.median(p)) == np.sign(a_r)).mean()),
                'n_samples': int(mask.sum()),
            }

    # ============================================================
    # Top features (by cumulative importance across folds)
    # ============================================================
    top_feat_idx = np.argsort(feature_importance)[::-1][:20]
    top_features = [
        (feature_names_use[i], float(feature_importance[i]))
        for i in top_feat_idx
        if feature_importance[i] > 0
    ]

    # ============================================================
    # Fold consistency check
    # ============================================================
    n_positive_folds = sum(1 for ic in fold_ics if ic > 0)
    fold_consistency = n_positive_folds / len(fold_ics) if fold_ics else 0.0

    # Pass criteria:
    # - IC > 0.02 (meaningful directional signal)
    # - t-stat > 2.0 (statistically significant)
    # - fold consistency > 60% (not one-fold wonder)
    passed = (
        abs(ic) > 0.02 and
        abs(tstat) > 2.0 and
        fold_consistency > 0.60
    )

    return {
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
        # Fold consistency
        'fold_consistency': fold_consistency,
        'n_positive_folds': n_positive_folds,
        'n_folds': len(fold_ics),
        'fold_ics': [float(x) for x in fold_ics],
        'fold_metrics': fold_metrics,
        # Time-phase breakdown
        'session_stats': session_stats,
        # Regime breakdown
        'regime_stats': regime_stats,
        # Top features
        'top_features': top_features,
        # Sample counts
        'n_predictions': len(p),
        # Pass verdict
        'passed': passed,
    }


# ============================================================================
# SCOREBOARD FORMATTING
# ============================================================================

def format_multihorizon_scoreboard(results: List[dict]) -> str:
    """Format multi-horizon results as comprehensive ASCII table."""
    lines = [
        "",
        "MULTI-HORIZON RETURN PREDICTION — SCOREBOARD",
        "=" * 110,
        f"{'Target':<20s} {'IC':>7s} {'ICIR':>6s} {'t':>6s} {'HR':>6s} "
        f"{'PF':>6s} {'Sharpe':>8s} {'FoldC%':>7s} {'Folds':>5s} {'Preds':>8s} {'Pass':>5s}",
        "-" * 110,
    ]

    sorted_results = sorted(results, key=lambda x: abs(x.get('ic', 0)), reverse=True)
    for r in sorted_results:
        if 'error' in r:
            lines.append(f"{r.get('target', '?'):<20s}  ERROR: {r['error']}")
            continue

        passed = "YES" if r['passed'] else "no"
        fc_pct = f"{r['fold_consistency']:.0%}"
        lines.append(
            f"{r['target']:<20s} {r['ic']:>7.4f} {r['icir']:>6.2f} {r['tstat']:>6.2f} "
            f"{r['hit_rate']:>6.1%} {r['profit_factor']:>6.2f} {r['sharpe']:>8.2f} "
            f"{fc_pct:>7s} {r['n_folds']:>5d} {r['n_predictions']:>8d} {passed:>5s}"
        )

    lines.append("=" * 110)

    # ---- Per-fold IC evolution for top 5 ----
    lines.append("\nPER-FOLD IC EVOLUTION (top 5 by |IC|):")
    lines.append("-" * 90)
    for r in sorted_results[:5]:
        if 'error' in r or not r.get('fold_ics'):
            continue
        ics_str = " ".join(f"{x:+.3f}" for x in r['fold_ics'])
        lines.append(f"  {r['target']:<20s}: [{ics_str}]")

    # ---- Session phase breakdown ----
    lines.append("\nSESSION PHASE BREAKDOWN (top 5 by |IC|):")
    lines.append("-" * 90)
    for r in sorted_results[:5]:
        if 'error' in r or not r.get('session_stats'):
            continue
        lines.append(f"  {r['target']}:")
        for phase, stats in r['session_stats'].items():
            lines.append(
                f"    {phase:<15s}: IC={stats['ic']:>7.4f} HR={stats['hit_rate']:>6.1%} "
                f"n={stats['n_samples']:>7,}"
            )

    # ---- Regime breakdown ----
    lines.append("\nVOLATILITY REGIME BREAKDOWN (top 5 by |IC|):")
    lines.append("-" * 90)
    for r in sorted_results[:5]:
        if 'error' in r or not r.get('regime_stats'):
            continue
        lines.append(f"  {r['target']}:")
        for regime, stats in r['regime_stats'].items():
            lines.append(
                f"    {regime:<15s}: IC={stats['ic']:>7.4f} HR={stats['hit_rate']:>6.1%} "
                f"n={stats['n_samples']:>7,}"
            )

    # ---- Top features across all scans ----
    lines.append("\nTOP FEATURES ACROSS ALL SCANS (aggregated importance):")
    lines.append("-" * 90)
    feat_agg = {}
    for r in results:
        if 'error' in r:
            continue
        for fname, fimp in r.get('top_features', []):
            feat_agg[fname] = feat_agg.get(fname, 0) + fimp
    for fname, fimp in sorted(feat_agg.items(), key=lambda x: x[1], reverse=True)[:20]:
        lines.append(f"  {fname:<35s}: {fimp:>12.0f}")

    # ---- Winners ----
    winners = [r for r in results if r.get('passed', False)]
    if winners:
        lines.append(f"\nALPHA FOUND in {len(winners)}/{len(results)} targets:")
        for w in winners:
            lines.append(
                f"  {w['target']}: IC={w['ic']:.4f} ICIR={w['icir']:.2f} "
                f"t={w['tstat']:.2f} FoldC={w['fold_consistency']:.0%} "
                f"Sharpe={w['sharpe']:.2f}"
            )
            if w.get('top_features'):
                top5 = [f[0] for f in w['top_features'][:5]]
                lines.append(f"    Top features: {', '.join(top5)}")
            if w.get('session_stats'):
                best_session = max(
                    w['session_stats'].items(),
                    key=lambda x: abs(x[1]['ic'])
                )
                lines.append(
                    f"    Best session: {best_session[0]} "
                    f"IC={best_session[1]['ic']:.4f}"
                )
    else:
        lines.append(
            f"\nNo alpha found above threshold "
            f"(IC>0.02, t>2.0, fold_consistency>60%)"
        )
        # Show best result anyway
        if sorted_results and 'error' not in sorted_results[0]:
            best = sorted_results[0]
            lines.append(
                f"  Best: {best['target']} IC={best['ic']:.4f} "
                f"t={best['tstat']:.2f} ICIR={best['icir']:.2f}"
            )

    return "\n".join(lines)


def format_discord_summary(results: List[dict], elapsed_sec: float) -> str:
    """Concise Discord message with key findings."""
    sorted_results = sorted(results, key=lambda x: abs(x.get('ic', 0)), reverse=True)
    winners = [r for r in results if r.get('passed', False)]

    lines = [
        "**RETURN PREDICTION SCAN COMPLETE**",
        f"Elapsed: {elapsed_sec/60:.1f} min | Targets scanned: {len(results)}",
        "",
        "**SCOREBOARD** (sorted by |IC|):",
        "```",
        f"{'Target':<20s} {'IC':>7s} {'t':>6s} {'FoldC':>6s} {'Pass':>5s}",
        "-" * 50,
    ]

    for r in sorted_results[:12]:
        if 'error' in r:
            lines.append(f"{r.get('target', '?'):<20s}  ERROR")
            continue
        fc = f"{r['fold_consistency']:.0%}"
        passed = "YES" if r['passed'] else "---"
        lines.append(
            f"{r['target']:<20s} {r['ic']:>7.4f} {r['tstat']:>6.2f} {fc:>6s} {passed:>5s}"
        )

    lines.append("```")

    if winners:
        lines.append(f"\n**ALPHA FOUND in {len(winners)} targets!**")
        for w in winners:
            lines.append(f"- `{w['target']}`: IC={w['ic']:.4f}, t={w['tstat']:.2f}, Sharpe={w['sharpe']:.2f}")
            if w.get('top_features'):
                top3 = [f[0] for f in w['top_features'][:3]]
                lines.append(f"  Top: {', '.join(top3)}")
    else:
        if sorted_results and 'error' not in sorted_results[0]:
            best = sorted_results[0]
            lines.append(
                f"\n**No clear alpha.** Best: `{best['target']}` IC={best['ic']:.4f}, "
                f"t={best['tstat']:.2f}"
            )
        else:
            lines.append("\n**No alpha found above threshold.**")

    return "\n".join(lines)


# ============================================================================
# HONEST ASSESSMENT
# ============================================================================

def generate_honest_assessment(results: List[dict]) -> str:
    """Generate an honest assessment of what the results mean."""
    sorted_results = sorted(results, key=lambda x: abs(x.get('ic', 0)), reverse=True)
    winners = [r for r in results if r.get('passed', False)]
    best = sorted_results[0] if sorted_results and 'error' not in sorted_results[0] else None

    lines = ["", "HONEST ASSESSMENT", "=" * 70]

    if winners:
        lines.append(f"\nSTATUS: ALPHA FOUND ({len(winners)} targets pass all criteria)")
        lines.append("")
        for w in winners:
            lines.append(f"Target: {w['target']}")
            lines.append(f"  IC={w['ic']:.4f}, t={w['tstat']:.2f}, ICIR={w['icir']:.2f}")
            lines.append(f"  Fold consistency: {w['n_positive_folds']}/{w['n_folds']} folds positive")
            lines.append(f"  Annualized Sharpe: {w['sharpe']:.2f}")

            # Feature assessment
            top_feats = [f[0] for f in w.get('top_features', [])[:5]]
            genuine_feats = [f for f in top_feats if f not in [
                'rvol_10', 'rvol_20', 'rvol_50', 'vov_10', 'vov_20', 'vov_50',
                'event_int_5', 'event_int_20', 'event_int_50', 'tick_count',
                'tick_density_5', 'tick_density_20',
            ]]
            lines.append(f"  Top features: {', '.join(top_feats)}")
            if len(genuine_feats) >= 3:
                lines.append(f"  Assessment: Features look like genuine microstructure (book flow, order dynamics)")
            else:
                lines.append(f"  WARNING: Some top features may be vol proxies despite exclusion")

            # Session context
            if w.get('session_stats'):
                best_s = max(w['session_stats'].items(), key=lambda x: abs(x[1]['ic']))
                worst_s = min(w['session_stats'].items(), key=lambda x: abs(x[1]['ic']))
                lines.append(f"  Strongest session: {best_s[0]} (IC={best_s[1]['ic']:.4f})")
                lines.append(f"  Weakest session: {worst_s[0]} (IC={worst_s[1]['ic']:.4f})")

            # Tradability
            if w['ic'] > 0.05:
                lines.append(f"  TRADABILITY: Strong signal. Consider actual trade sizing.")
            elif w['ic'] > 0.03:
                lines.append(f"  TRADABILITY: Moderate signal. Real-world transaction costs may eat it.")
            else:
                lines.append(f"  TRADABILITY: Marginal signal. Transaction costs (1-2 ticks) likely dominate.")

    elif best:
        lines.append(f"\nSTATUS: WEAK/NO ALPHA")
        lines.append(f"Best result: {best['target']} IC={best['ic']:.4f}, t={best['tstat']:.2f}")
        lines.append(f"Fold consistency: {best.get('fold_consistency', 0):.0%}")
        lines.append("")

        if abs(best['ic']) < 0.01:
            lines.append("INTERPRETATION: The orderbook features have essentially zero")
            lines.append("directional predictive power at these horizons. This could mean:")
            lines.append("  1. The market is genuinely efficient at 100ms-60s horizons")
            lines.append("  2. The features don't capture the right microstructure dynamics")
            lines.append("  3. More data needed (16 days may be too few for short-horizon signals)")
            lines.append("  4. LightGBM may not capture the spatial relationships the GNN sees")
        elif abs(best['ic']) < 0.02:
            lines.append("INTERPRETATION: Very weak signal. Likely noise or")
            lines.append("not statistically reliable across folds.")
    else:
        lines.append("STATUS: No results available")

    lines.append("")
    lines.append("ES FUTURES CONTEXT:")
    lines.append("  1 tick = $12.50/contract. Round-trip cost = ~1-2 ticks ($12.50-$25)")
    lines.append("  For IC=0.02 to be profitable: need hit rate > 52-53%")
    lines.append("  For IC=0.05: hit rate likely 53-55%, meaningful edge after costs")
    lines.append("  For IC>0.10: substantial edge, consistent profitability likely")

    return "\n".join(lines)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Multi-horizon return prediction scan')
    parser.add_argument('--fast', action='store_true',
                        help='Faster run: fewer horizons, fewer folds')
    parser.add_argument('--horizons', nargs='+',
                        default=['1s', '3s', '5s', '10s', '15s', '30s', '60s'],
                        help='Which return horizons to test')
    parser.add_argument('--min-train-days', type=int, default=3,
                        help='Minimum training days before first test')
    parser.add_argument('--no-flow-target', action='store_true',
                        help='Skip aggressive flow prediction target')
    args = parser.parse_args()

    logger.info("=" * 75)
    logger.info("MULTI-HORIZON RETURN PREDICTION SCAN")
    logger.info(f"  Horizons: {args.horizons}")
    logger.info(f"  Excluded features: {len(EXCLUDE_FEATURES_DIRECTION)}")
    logger.info(f"  {KEEP_MESSAGE}")
    logger.info("=" * 75)

    t_start = time.time()

    # Load scanner and feature cache
    scanner = MBOAlphaScanner(sample_interval_ms=100)
    stats = load_feature_cache(scanner)
    if stats is None:
        logger.info("No feature cache — computing from scratch (slow)...")
        stats = scanner.load_from_cache()
    else:
        logger.info(f"Cached: {stats}")

    logger.info(f"Data: {stats.get('n_snapshots', 'N/A'):,} snapshots, "
                f"{stats.get('n_days', 'N/A')} days, "
                f"{len(scanner.feature_names)} features")
    logger.info(f"Excluding {len(EXCLUDE_FEATURES_DIRECTION)} features (vol proxies + price + time)")

    # Need raw global features for aggressive flow target
    raw_global = None
    if not args.no_flow_target:
        # Try to extract from scanner features (columns 0-44 are static snapshot)
        # Column indices: buy_vol=11, sell_vol=12
        from alpha_discovery.mbo_features import COL_BUY_VOL, COL_SELL_VOL
        raw_global = scanner.features[:, :45]  # First 45 cols = static snapshot

    # Compute targets
    logger.info("\nComputing multi-horizon return targets...")
    horizons_sec = {hz: int(hz.replace('s', '').replace('m', '')) *
                   (60 if hz.endswith('m') else 1)
                   for hz in args.horizons}

    targets = compute_return_targets(
        mid_prices=scanner.mid_prices,
        day_boundaries=scanner.day_boundaries,
        sample_interval_ms=100,
        horizons_sec=horizons_sec,
        include_flow_target=not args.no_flow_target,
        global_features_raw=raw_global,
    )

    logger.info(f"Computed {len(targets)} targets: {list(targets.keys())}")

    # Run walk-forward evaluation for each target
    results = []
    total = len(targets)

    for idx, (tgt_name, tgt_array) in enumerate(targets.items()):
        logger.info(f"\n{'='*65}")
        logger.info(f"[{idx+1}/{total}] Evaluating: {tgt_name}")
        logger.info(f"{'='*65}")

        # Determine if this is a classification target
        is_classification = tgt_name in ['signed_tick', 'aggr_flow_5s']

        t0 = time.time()
        result = walk_forward_evaluate_return(
            scanner=scanner,
            target=tgt_array,
            target_name=tgt_name,
            exclude_features=EXCLUDE_FEATURES_DIRECTION,
            min_train_days=args.min_train_days,
            is_classification=is_classification,
        )
        elapsed = time.time() - t0
        result['elapsed_sec'] = elapsed
        results.append(result)

        # Log result
        if 'error' in result:
            logger.info(f"  ERROR: {result['error']}")
        else:
            status = "ALPHA" if result['passed'] else "---"
            logger.info(
                f"  [{status}] IC={result['ic']:.4f} ICIR={result['icir']:.2f} "
                f"t={result['tstat']:.2f} HR={result['hit_rate']:.1%} "
                f"FoldC={result['fold_consistency']:.0%} "
                f"Sharpe={result['sharpe']:.2f} ({elapsed:.0f}s)"
            )
            if result.get('top_features'):
                top3 = ", ".join(f"{n}={v:.0f}" for n, v in result['top_features'][:3])
                logger.info(f"  Top: {top3}")
            if result.get('fold_ics'):
                ics_str = " ".join(f"{x:+.3f}" for x in result['fold_ics'])
                logger.info(f"  Fold ICs: [{ics_str}]")

        gc.collect()

    # Format scoreboards
    scoreboard = format_multihorizon_scoreboard(results)
    assessment = generate_honest_assessment(results)
    elapsed_total = time.time() - t_start

    # Print full scoreboard
    logger.info("\n" + scoreboard)
    logger.info("\n" + assessment)

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = RESULTS_DIR / f"return_multihorizon_{timestamp}.json"

    serializable = []
    for r in results:
        sr = dict(r)
        sr['top_features'] = [(n, float(v)) for n, v in r.get('top_features', [])]
        serializable.append(sr)

    with open(result_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'horizons_tested': args.horizons,
            'excluded_features': EXCLUDE_FEATURES_DIRECTION,
            'stats': stats,
            'results': serializable,
            'scoreboard': scoreboard,
            'assessment': assessment,
            'elapsed_sec': elapsed_total,
        }, f, indent=2, default=str)

    logger.info(f"\nResults saved to: {result_file}")
    logger.info(f"Total elapsed: {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")

    print("\n" + "=" * 75)
    print(scoreboard)
    print(assessment)
    print("=" * 75)

    # Discord summary
    discord_msg = format_discord_summary(results, elapsed_total)
    discord_full = discord_msg + f"\n\nResults: `{result_file.name}`"
    print("\n--- DISCORD SUMMARY ---")
    print(discord_full)

    return results, scoreboard


if __name__ == '__main__':
    main()
