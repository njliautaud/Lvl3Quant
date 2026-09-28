"""
Longer Horizon Alpha Explorer

Tests whether microstructure alpha persists at 1-5 minute horizons where
transaction cost economics are fundamentally better:
- At 10s: avg move ~3.5t, costs 0.25-1.25t (7-36% of move)
- At 1m:  avg move ~8-12t, costs 0.25-1.25t (2-16% of move)
- At 3m:  avg move ~15-25t, costs 0.25-1.25t (1-8% of move)
- At 5m:  avg move ~20-35t, costs 0.25-1.25t (<4% of move)

If IC > 0.03 at 3-5 min with magnitude gating, the system is clearly profitable.

Also tests the "train fast, trade slow" insight: does training on 3s MFE target
but HOLDING for 1-5 minutes outperform models trained directly at those horizons?

Usage:
    python alpha_discovery/longer_horizon_explorer.py --n-days 70
    python alpha_discovery/longer_horizon_explorer.py --n-days 70 --quick
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
_log_file = RESULTS_DIR / f"longer_horizon_{_ts}.log"
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
logger = logging.getLogger("long_hz")

TICK_SIZE = 0.25
TICK_VALUE = 12.50
COMMISSION_TICKS = 3.00 / TICK_VALUE  # 0.24t RT
BARS_PER_SEC = 10

# Extended horizons
ALL_HORIZONS = {
    '3s': 30, '10s': 100, '30s': 300,
    '1m': 600, '3m': 1800, '5m': 3000,
}


def load_data(feature_cache_dir: str, n_days: Optional[int] = None):
    from alpha_discovery.mbo_alpha_scan import MBOAlphaScanner
    scanner = MBOAlphaScanner()
    load_info = scanner.load_precomputed_features(
        feature_cache_dir=feature_cache_dir, n_days=n_days, extra_cols=0,
    )
    logger.info(f"Loaded {load_info['n_days']} days, {load_info['n_snapshots']:,} bars")
    np.clip(scanner.features, -60000, 60000, out=scanner.features)
    scanner.features = scanner.features.astype(np.float16)
    gc.collect()
    logger.info(f"  Features: {scanner.features.nbytes / 1e9:.1f} GB")
    return scanner, load_info


def compute_forward_return_ticks(mid_prices, horizon_bars, day_boundaries):
    """Simple forward return in ticks."""
    N = len(mid_prices)
    n_days = len(day_boundaries) - 1
    future_mid = np.empty(N, dtype=np.float32)
    future_mid[:] = np.nan

    for d in range(n_days):
        s, e = day_boundaries[d], day_boundaries[d + 1]
        day_prices = mid_prices[s:e]
        day_len = e - s
        for i in range(day_len):
            if i + horizon_bars < day_len:
                future_mid[s + i] = day_prices[i + horizon_bars]

    return (future_mid - mid_prices) / TICK_SIZE


def compute_mfe_net_target(mid_prices, horizon_bars, day_boundaries):
    """MFE-net target: signed max favorable excursion."""
    from alpha_discovery.run_mfe_scan import compute_mfe_targets
    hz_map = {30: '3s', 50: '5s', 100: '10s', 300: '30s', 600: '1m', 1800: '3m', 3000: '5m'}
    hz_name = hz_map.get(horizon_bars, '10s')
    hz_sec_map = {'3s': 3, '5s': 5, '10s': 10, '30s': 30, '1m': 60, '3m': 180, '5m': 300}
    hz_sec = hz_sec_map.get(hz_name, 10)
    mfe_targets = compute_mfe_targets(
        mid_prices=mid_prices,
        day_boundaries=day_boundaries,
        sample_interval_ms=100,
        horizons_sec={hz_name: hz_sec},
        tick_size=TICK_SIZE,
    )
    target = mfe_targets[f'mfe_net_{hz_name}']
    del mfe_targets
    gc.collect()
    return target


def train_walk_forward(features, target, day_boundaries,
                       min_train_days=5, max_train_days=30, label=''):
    """Walk-forward LightGBM. Returns prediction array and fold ICs."""
    import lightgbm as lgb
    N = len(target)
    n_days = len(day_boundaries) - 1
    full_preds = np.full(N, np.nan, dtype=np.float32)
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
        train_start = day_boundaries[train_start_day]
        train_end = day_boundaries[train_end_day + 1]
        test_start = day_boundaries[test_day]
        test_end = day_boundaries[test_day + 1]
        y_train = target[train_start:train_end]
        y_test = target[test_start:test_end]
        train_valid = np.isfinite(y_train)
        test_valid = np.isfinite(y_test)
        if train_valid.sum() < 500 or test_valid.sum() < 100:
            continue
        valid_idx = np.where(train_valid)[0]
        if len(valid_idx) > MAX_TRAIN:
            rng = np.random.default_rng(seed=test_day)
            sampled = rng.choice(valid_idx, MAX_TRAIN, replace=False)
            sampled.sort()
            X_tr = features[train_start:train_end][sampled].astype(np.float32)
            y_tr = y_train[sampled]
        else:
            X_tr = features[train_start:train_end][train_valid].astype(np.float32)
            y_tr = y_train[train_valid]
        X_te = features[test_start:test_end][test_valid].astype(np.float32)
        y_te = y_test[test_valid]
        split = int(len(X_tr) * 0.8)
        try:
            model = lgb.LGBMRegressor(**params)
            model.fit(X_tr[:split], y_tr[:split],
                      eval_set=[(X_tr[split:], y_tr[split:])],
                      callbacks=[lgb.early_stopping(30, verbose=False)])
            preds = model.predict(X_te)
        except Exception as e:
            logger.warning(f"  [{label}] Fold failed day {test_day}: {e}")
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
        if n_folds % 10 == 0:
            ic_mean = np.mean(fold_ics) if fold_ics else 0
            logger.info(f"  [{label}] Fold {n_folds} (day {test_day}): "
                        f"IC={ic_mean:.4f} [{time.time()-t0:.0f}s]")
        del model
        gc.collect()

    n_valid = np.isfinite(full_preds).sum()
    mask = np.isfinite(full_preds) & np.isfinite(target)
    overall_ic = float(spearmanr(full_preds[mask], target[mask])[0]) if mask.sum() > 50 else 0
    logger.info(f"  [{label}] DONE: {n_folds} folds, {n_valid:,} preds, "
                f"IC={overall_ic:.4f} [{time.time()-t0:.0f}s]")
    return full_preds, fold_ics


def analyze_horizon(preds, target, ret_ticks, day_boundaries,
                    oos_start_day, hz_name, label):
    """Comprehensive analysis for one prediction set against one evaluation horizon."""
    N = len(preds)
    oos_start = day_boundaries[oos_start_day]
    oos = np.zeros(N, dtype=bool)
    oos[oos_start:] = True
    valid = oos & np.isfinite(preds) & np.isfinite(target) & np.isfinite(ret_ticks)

    if valid.sum() < 100:
        return None

    ic_target = float(spearmanr(preds[valid], target[valid])[0])
    ic_ret = float(spearmanr(preds[valid], ret_ticks[valid])[0])

    # Direction accuracy
    sig_dir = np.sign(preds[valid])
    actual_dir = np.sign(ret_ticks[valid])
    hr = (sig_dir == actual_dir).mean()

    # PnL analysis
    signed_ret = sig_dir * ret_ticks[valid]
    mean_t = float(np.mean(signed_ret))
    median_t = float(np.median(signed_ret))
    wins = signed_ret[signed_ret > 0]
    losses = signed_ret[signed_ret < 0]
    pf = abs(wins.sum() / losses.sum()) if len(losses) > 0 and losses.sum() != 0 else 0
    wr = len(wins) / len(signed_ret) if len(signed_ret) > 0 else 0

    # Move size statistics
    abs_ret = np.abs(ret_ticks[valid])
    avg_move = float(np.mean(abs_ret))
    p90_move = float(np.percentile(abs_ret, 90))

    # Cost analysis
    limit_cost = COMMISSION_TICKS  # 0.248t
    market_cost = 1.0 + COMMISSION_TICKS  # 1.248t (spread + commission)
    net_limit = mean_t - limit_cost
    net_market = mean_t - market_cost

    n_oos_days = len(day_boundaries) - 1 - oos_start_day
    per_day = valid.sum() / max(n_oos_days, 1)

    # Magnitude gating (top 10% signal strength)
    # CRITICAL: Compute threshold ONLY from IS predictions (no forward-looking bias).
    # The threshold determines which high-conviction trades to take, so it must not
    # leak the OOS prediction distribution into the selection filter.
    abs_pred_all = np.abs(preds)
    is_mask = (~oos) & np.isfinite(preds)

    if is_mask.sum() > 100:
        # Sufficient IS data: compute 90th percentile on IS predictions
        p90_thresh = np.nanpercentile(abs_pred_all[is_mask], 90)
    elif is_mask.sum() > 0:
        # Sparse IS data: use IS median as conservative fallback
        p90_thresh = np.nanpercentile(abs_pred_all[is_mask], 50)
    else:
        # No IS predictions at all: cannot compute leakage-safe threshold
        p90_thresh = None

    if p90_thresh is not None:
        top10 = abs_pred_all[valid] > p90_thresh
        if top10.sum() > 50:
            top10_signed = np.sign(preds[valid][top10]) * ret_ticks[valid][top10]
            top10_mean = float(np.mean(top10_signed))
            top10_wr = float((top10_signed > 0).mean())
            top10_net_limit = top10_mean - limit_cost
            top10_daily = top10.sum() / max(n_oos_days, 1)
        else:
            top10_mean = top10_wr = top10_net_limit = top10_daily = 0
    else:
        top10_mean = top10_wr = top10_net_limit = top10_daily = 0

    # Daily IC
    daily_ics = []
    for d in range(oos_start_day, len(day_boundaries) - 1):
        s, e = day_boundaries[d], day_boundaries[d + 1]
        m = valid[s:e]
        if m.sum() > 50:
            ic = float(spearmanr(preds[s:e][m], ret_ticks[s:e][m])[0])
            if np.isfinite(ic):
                daily_ics.append(ic)
    icir = np.mean(daily_ics) / max(np.std(daily_ics), 1e-6) if daily_ics else 0
    pct_positive = sum(1 for x in daily_ics if x > 0) / max(len(daily_ics), 1)

    result = {
        'label': label,
        'hz': hz_name,
        'ic_target': ic_target,
        'ic_ret': ic_ret,
        'icir': icir,
        'hr': hr,
        'wr': wr,
        'pf': pf,
        'mean_t': mean_t,
        'median_t': median_t,
        'avg_move': avg_move,
        'p90_move': p90_move,
        'net_limit': net_limit,
        'net_market': net_market,
        'per_day': per_day,
        'n_bars': int(valid.sum()),
        'top10_mean': top10_mean,
        'top10_wr': top10_wr,
        'top10_net_limit': top10_net_limit,
        'top10_daily': top10_daily,
        'daily_ic_mean': float(np.mean(daily_ics)) if daily_ics else 0,
        'pct_positive_days': pct_positive,
    }

    return result


def main():
    parser = argparse.ArgumentParser(description='Longer Horizon Alpha Explorer')
    parser.add_argument('--n-days', type=int, default=70)
    parser.add_argument('--feature-cache', type=str,
                        default=DEFAULT_FEATURE_CACHE)
    parser.add_argument('--min-train-days', type=int, default=5)
    parser.add_argument('--max-train-days', type=int, default=30)
    parser.add_argument('--quick', action='store_true',
                        help='Skip MFE targets, use simple returns only')
    parser.add_argument('--horizons', type=str, default='3s,10s,30s,1m,3m,5m',
                        help='Comma-separated horizons to test')
    args = parser.parse_args()

    horizons_to_test = {k: ALL_HORIZONS[k] for k in args.horizons.split(',') if k in ALL_HORIZONS}

    logger.info("=" * 70)
    logger.info("LONGER HORIZON ALPHA EXPLORER")
    logger.info(f"  n_days:    {args.n_days}")
    logger.info(f"  horizons:  {list(horizons_to_test.keys())}")
    logger.info(f"  mode:      {'return only (quick)' if args.quick else 'MFE + return'}")
    logger.info(f"  log:       {_log_file}")
    logger.info("=" * 70)

    t_total = time.time()

    # Load data
    scanner, load_info = load_data(args.feature_cache, args.n_days)
    features = scanner.features
    mid_prices = scanner.mid_prices
    day_boundaries = scanner.day_boundaries
    n_days = len(day_boundaries) - 1
    oos_start_day = max(args.min_train_days, int(n_days * 0.7))

    logger.info(f"IS/OOS split: days 0-{oos_start_day-1} / {oos_start_day}-{n_days-1}")

    # ================================================================
    # PHASE 1: Train models at each horizon
    # ================================================================
    all_preds = {}     # (horizon, target_type) -> prediction array
    all_targets = {}   # (horizon, target_type) -> target array
    all_ret_ticks = {} # horizon -> forward return in ticks

    for hz_name, hz_bars in horizons_to_test.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"HORIZON: {hz_name} ({hz_bars} bars = {hz_bars/BARS_PER_SEC:.0f}s)")
        logger.info(f"{'='*60}")

        # Compute forward returns (always needed for evaluation)
        logger.info(f"  Computing forward returns...")
        ret_ticks = compute_forward_return_ticks(mid_prices, hz_bars, day_boundaries)
        n_valid_ret = np.isfinite(ret_ticks).sum()
        avg_move = np.nanmean(np.abs(ret_ticks))
        p90_move = np.nanpercentile(np.abs(ret_ticks), 90)
        logger.info(f"  Forward returns: {n_valid_ret:,} valid, "
                    f"avg |move|={avg_move:.2f}t, P90={p90_move:.2f}t")
        all_ret_ticks[hz_name] = ret_ticks

        # Train on simple return target
        logger.info(f"\n  --- Training on RETURN target ---")
        preds_ret, ics_ret = train_walk_forward(
            features, ret_ticks, day_boundaries,
            args.min_train_days, args.max_train_days, f'{hz_name}_ret')
        all_preds[(hz_name, 'return')] = preds_ret
        all_targets[(hz_name, 'return')] = ret_ticks

        # Train on MFE target (unless --quick)
        if not args.quick:
            logger.info(f"\n  --- Training on MFE target ---")
            t0 = time.time()
            mfe_target = compute_mfe_net_target(mid_prices, hz_bars, day_boundaries)
            logger.info(f"  MFE target computed in {time.time()-t0:.0f}s")
            preds_mfe, ics_mfe = train_walk_forward(
                features, mfe_target, day_boundaries,
                args.min_train_days, args.max_train_days, f'{hz_name}_mfe')
            all_preds[(hz_name, 'mfe')] = preds_mfe
            all_targets[(hz_name, 'mfe')] = mfe_target
            del mfe_target
            gc.collect()

        gc.collect()

    # ================================================================
    # PHASE 2: Evaluate all models against all horizons
    # ================================================================
    logger.info(f"\n\n{'='*70}")
    logger.info("COMPREHENSIVE CROSS-EVALUATION")
    logger.info(f"{'='*70}")

    all_results = []

    # Each model evaluated against its own horizon
    logger.info(f"\n--- SELF-EVALUATION (model trained/evaluated at same horizon) ---")
    header = f"{'Model':>25s}  {'IC(ret)':>8s}  {'ICIR':>6s}  {'HR':>6s}  " \
             f"{'Mean(t)':>8s}  {'AvgMove':>8s}  {'NetLMT':>8s}  {'NetMKT':>8s}  " \
             f"{'Top10%':>8s}  {'$/day':>10s}"
    logger.info(header)
    logger.info("-" * len(header))

    for hz_name in horizons_to_test:
        ret_ticks = all_ret_ticks[hz_name]
        for tgt_type in ['return', 'mfe']:
            key = (hz_name, tgt_type)
            if key not in all_preds:
                continue
            preds = all_preds[key]
            target = all_targets[key]
            label = f"{hz_name}_{tgt_type}"
            r = analyze_horizon(preds, target, ret_ticks, day_boundaries,
                                oos_start_day, hz_name, label)
            if r:
                all_results.append(r)
                daily_dollar = r['net_limit'] * TICK_VALUE * r['per_day']
                top10_dollar = r['top10_net_limit'] * TICK_VALUE * r['top10_daily']
                logger.info(
                    f"{label:>25s}  {r['ic_ret']:>+8.4f}  {r['icir']:>6.2f}  "
                    f"{r['hr']:>6.1%}  {r['mean_t']:>+8.3f}  {r['avg_move']:>8.2f}  "
                    f"{r['net_limit']:>+8.3f}  {r['net_market']:>+8.3f}  "
                    f"{r['top10_net_limit']:>+8.3f}  ${top10_dollar:>+9.0f}")

    # Cross-horizon evaluation: "train fast, trade slow"
    logger.info(f"\n--- CROSS-EVALUATION (train at horizon X, evaluate at horizon Y) ---")
    logger.info(f"  'Train fast, trade slow' — using 3s model for longer-horizon trading")

    cross_results = []
    # For each training horizon model, evaluate at ALL holding horizons
    for train_hz in horizons_to_test:
        for tgt_type in ['return', 'mfe']:
            key = (train_hz, tgt_type)
            if key not in all_preds:
                continue
            preds = all_preds[key]

            for eval_hz in horizons_to_test:
                if eval_hz == train_hz:
                    continue  # Already done in self-evaluation
                ret_ticks = all_ret_ticks[eval_hz]
                label = f"train_{train_hz}_{tgt_type}→eval_{eval_hz}"
                r = analyze_horizon(preds, all_targets.get((eval_hz, 'return'), ret_ticks),
                                    ret_ticks, day_boundaries, oos_start_day, eval_hz, label)
                if r:
                    cross_results.append(r)

    if cross_results:
        logger.info(f"\n{'Key cross-horizon results (sorted by net LMT PnL):':}")
        sorted_cross = sorted(cross_results, key=lambda x: x['net_limit'], reverse=True)
        for r in sorted_cross[:15]:
            daily_dollar = r['net_limit'] * TICK_VALUE * r['per_day']
            logger.info(
                f"  {r['label']:>40s}: IC={r['ic_ret']:+.4f}  "
                f"Mean={r['mean_t']:+.3f}t  NetLMT={r['net_limit']:+.3f}t  "
                f"AvgMove={r['avg_move']:.1f}t  Top10%={r['top10_net_limit']:+.3f}t  "
                f"${daily_dollar:+.0f}/day")

    # ================================================================
    # PHASE 3: Best Strategy Selection
    # ================================================================
    logger.info(f"\n\n{'='*70}")
    logger.info("BEST STRATEGY SELECTION")
    logger.info(f"{'='*70}")

    # Combine all results
    all_combined = all_results + cross_results
    if all_combined:
        # Best by net limit PnL
        best_lmt = max(all_combined, key=lambda x: x['net_limit'])
        logger.info(f"\n  Best by net limit PnL:")
        logger.info(f"    {best_lmt['label']}: {best_lmt['net_limit']:+.3f}t/trade, "
                    f"IC={best_lmt['ic_ret']:+.4f}, HR={best_lmt['hr']:.1%}")
        daily_dollar = best_lmt['net_limit'] * TICK_VALUE * best_lmt['per_day']
        logger.info(f"    ~${daily_dollar:+.0f}/day ({best_lmt['per_day']:.0f} signals/day)")

        # Best by top-10% net limit PnL
        best_top10 = max(all_combined, key=lambda x: x['top10_net_limit'])
        logger.info(f"\n  Best by top-10% conviction limit PnL:")
        logger.info(f"    {best_top10['label']}: {best_top10['top10_net_limit']:+.3f}t/trade, "
                    f"WR={best_top10['top10_wr']:.1%}")
        top10_dollar = best_top10['top10_net_limit'] * TICK_VALUE * best_top10['top10_daily']
        logger.info(f"    ~${top10_dollar:+.0f}/day ({best_top10['top10_daily']:.0f} top-10% signals/day)")

        # Best IC
        best_ic = max(all_combined, key=lambda x: x['ic_ret'])
        logger.info(f"\n  Best IC vs endpoint return:")
        logger.info(f"    {best_ic['label']}: IC={best_ic['ic_ret']:+.4f}, "
                    f"ICIR={best_ic['icir']:.2f}")

    # ================================================================
    # PHASE 4: Economic Viability Summary
    # ================================================================
    logger.info(f"\n\n{'='*70}")
    logger.info("ECONOMIC VIABILITY SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"\n  Cost thresholds:")
    logger.info(f"    Limit in + Limit out: {COMMISSION_TICKS:.3f}t")
    logger.info(f"    Limit in + Market out: {0.5 + COMMISSION_TICKS:.3f}t")
    logger.info(f"    Market in + Market out: {1.0 + COMMISSION_TICKS:.3f}t")

    for r in sorted(all_combined, key=lambda x: x['net_limit'], reverse=True):
        if r['net_limit'] > 0:
            daily_dollar = r['net_limit'] * TICK_VALUE * r['per_day']
            logger.info(f"\n  PROFITABLE (limit orders): {r['label']}")
            logger.info(f"    Net: {r['net_limit']:+.3f}t/trade, "
                        f"${daily_dollar:+.0f}/day, "
                        f"IC={r['ic_ret']:+.4f}, HR={r['hr']:.1%}, "
                        f"AvgMove={r['avg_move']:.1f}t, "
                        f"{r['pct_positive_days']:.0%} positive days")

    # Save results
    elapsed = time.time() - t_total
    output = {
        'config': {
            'n_days': args.n_days,
            'horizons': list(horizons_to_test.keys()),
            'oos_start_day': oos_start_day,
            'quick': args.quick,
            'elapsed_sec': elapsed,
        },
        'self_eval': [r for r in all_results],
        'cross_eval': [r for r in cross_results],
    }

    json_path = RESULTS_DIR / f"longer_horizon_{_ts}.json"
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Save predictions for reuse
    save_data = {'mid_prices': mid_prices, 'day_boundaries': np.array(day_boundaries)}
    for key, preds in all_preds.items():
        hz, tgt = key
        save_data[f'preds_{hz}_{tgt}'] = preds
    for hz, ret in all_ret_ticks.items():
        save_data[f'ret_{hz}'] = ret
    npz_path = RESULTS_DIR / f"longer_horizon_preds_{_ts}.npz"
    np.savez_compressed(str(npz_path), **save_data)

    logger.info(f"\n{'='*70}")
    logger.info(f"COMPLETE in {elapsed:.0f}s ({elapsed/60:.1f}m)")
    logger.info(f"Results: {json_path}")
    logger.info(f"Predictions: {npz_path}")
    logger.info(f"Log: {_log_file}")
    logger.info(f"{'='*70}")


if __name__ == '__main__':
    main()
