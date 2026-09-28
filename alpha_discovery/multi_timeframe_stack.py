"""
Multi-Timeframe Stacking Analysis

Trains separate LightGBM direction models at 3s, 10s, and 30s horizons,
then tests whether AGREEMENT between timeframes produces a stronger,
more tradable signal than any single model.

HYPOTHESIS: When a fast model (3s) and slow model (30s) both agree on
direction, the signal is more robust — it's not just noise or a quick blip,
but a genuine directional move with staying power.

Architecture variants tested:
1. Individual models at each horizon
2. Vote counting (2/3, 3/3 agreement)
3. Agreement-filtered IC (only predict when models agree)
4. Weighted consensus (IC-weighted average of predictions)
5. Stacking meta-model (predictions as features → final prediction)
6. Cross-horizon IC (does 3s agreement improve 10s prediction?)

Usage:
    # Quick test (20 days)
    python alpha_discovery/multi_timeframe_stack.py --n-days 20 --quick

    # Full analysis (70 days)
    python alpha_discovery/multi_timeframe_stack.py --n-days 70

    # Load saved predictions
    python alpha_discovery/multi_timeframe_stack.py --load-predictions results/mt_preds_*.npz
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
from typing import Optional, List, Dict, Tuple

import numpy as np
from scipy.stats import spearmanr

# Add project root
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
_log_file = RESULTS_DIR / f"multi_timeframe_{_ts}.log"
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
logger = logging.getLogger("mt_stack")

# Constants
TICK_SIZE = 0.25
TICK_VALUE = 12.50
COMMISSION_TICKS = 3.00 / TICK_VALUE  # 0.24t RT
BARS_PER_SEC = 10  # 100ms bars

# Horizons to test
HORIZONS = {
    '3s': 30,
    '10s': 100,
    '30s': 300,
}


# ============================================================================
# Data Loading
# ============================================================================
def load_data(feature_cache_dir: str, n_days: Optional[int] = None):
    """Load pre-computed features from cache."""
    from alpha_discovery.mbo_alpha_scan import MBOAlphaScanner

    scanner = MBOAlphaScanner()
    load_info = scanner.load_precomputed_features(
        feature_cache_dir=feature_cache_dir,
        n_days=n_days,
        extra_cols=0,
    )
    logger.info(f"Loaded {load_info['n_days']} days, {load_info['n_snapshots']:,} bars, "
                f"{load_info['n_features']} features")

    # Downcast to float16
    np.clip(scanner.features, -60000, 60000, out=scanner.features)
    scanner.features = scanner.features.astype(np.float16)
    gc.collect()
    logger.info(f"  Features memory: {scanner.features.nbytes / 1e9:.1f} GB")

    return scanner, load_info


# ============================================================================
# Target Computation
# ============================================================================
def compute_direction_target(mid_prices, horizon_bars, day_boundaries, use_mfe=True):
    """Compute direction target for a given horizon.

    Returns target in ticks (signed MFE-net or simple return).
    """
    N = len(mid_prices)
    n_days = len(day_boundaries) - 1

    if use_mfe:
        from alpha_discovery.run_mfe_scan import compute_mfe_targets
        hz_map = {30: '3s', 50: '5s', 100: '10s', 300: '30s', 600: '1m'}
        hz_name = hz_map.get(horizon_bars, '10s')
        hz_sec_map = {'3s': 3, '5s': 5, '10s': 10, '30s': 30, '1m': 60}
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
    else:
        # Simple return in ticks
        future_mid = np.empty(N, dtype=np.float32)
        future_mid[:N - horizon_bars] = mid_prices[horizon_bars:]
        future_mid[N - horizon_bars:] = np.nan
        # NaN at day boundaries
        for d in range(n_days - 1):
            day_end = day_boundaries[d + 1]
            nan_start = max(day_boundaries[d], day_end - horizon_bars)
            future_mid[nan_start:day_end] = np.nan
        return (future_mid - mid_prices) / TICK_SIZE


# ============================================================================
# Walk-Forward Training
# ============================================================================
def train_walk_forward(
    features: np.ndarray,
    target: np.ndarray,
    day_boundaries: list,
    min_train_days: int = 5,
    max_train_days: int = 30,
    horizon_name: str = '10s',
) -> Tuple[np.ndarray, list]:
    """Walk-forward LightGBM with rolling window. Returns prediction array and fold ICs."""
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

    MAX_TRAIN_SAMPLES = 500_000
    n_folds = 0
    t0 = time.time()

    for test_day in range(min_train_days, n_days):
        train_end_day = test_day - 1  # 1-day purge gap
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
            logger.warning(f"  [{horizon_name}] Train failed day {test_day}: {e}")
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
            ic_so_far = np.mean(fold_ics) if fold_ics else 0
            elapsed = time.time() - t0
            logger.info(f"  [{horizon_name}] Fold {n_folds} (day {test_day}): "
                        f"IC={ic_so_far:.4f}  [{elapsed:.0f}s]")

        del model
        gc.collect()

    n_valid = np.isfinite(full_preds).sum()
    overall_ic = 0.0
    mask = np.isfinite(full_preds) & np.isfinite(target)
    if mask.sum() > 50:
        overall_ic = float(spearmanr(full_preds[mask], target[mask])[0])

    elapsed = time.time() - t0
    logger.info(f"  [{horizon_name}] DONE: {n_folds} folds, {n_valid:,} preds, "
                f"IC={overall_ic:.4f}  [{elapsed:.0f}s total]")

    return full_preds, fold_ics


# ============================================================================
# Analysis Functions
# ============================================================================
def compute_ic(preds, target, mask=None):
    """Spearman IC with optional mask."""
    if mask is None:
        mask = np.isfinite(preds) & np.isfinite(target)
    else:
        mask = mask & np.isfinite(preds) & np.isfinite(target)
    if mask.sum() < 50:
        return 0.0, 0
    return float(spearmanr(preds[mask], target[mask])[0]), int(mask.sum())


def compute_daily_ics(preds, target, day_boundaries, mask=None):
    """Per-day IC series."""
    n_days = len(day_boundaries) - 1
    ics = []
    for d in range(n_days):
        s, e = day_boundaries[d], day_boundaries[d + 1]
        p = preds[s:e]
        t = target[s:e]
        if mask is not None:
            m = mask[s:e] & np.isfinite(p) & np.isfinite(t)
        else:
            m = np.isfinite(p) & np.isfinite(t)
        if m.sum() > 50:
            ic = float(spearmanr(p[m], t[m])[0])
            if np.isfinite(ic):
                ics.append(ic)
    return ics


def compute_forward_return_ticks(mid_prices, horizon_bars, day_boundaries):
    """Compute simple forward return in ticks (for PnL evaluation)."""
    N = len(mid_prices)
    n_days = len(day_boundaries) - 1
    future_mid = np.empty(N, dtype=np.float32)
    future_mid[:N - horizon_bars] = mid_prices[horizon_bars:]
    future_mid[N - horizon_bars:] = np.nan
    for d in range(n_days - 1):
        day_end = day_boundaries[d + 1]
        nan_start = max(day_boundaries[d], day_end - horizon_bars)
        future_mid[nan_start:day_end] = np.nan
    return (future_mid - mid_prices) / TICK_SIZE


# ============================================================================
# Agreement Analysis
# ============================================================================
def analyze_agreement(
    preds_dict: Dict[str, np.ndarray],  # horizon_name -> OOS prediction array
    targets_dict: Dict[str, np.ndarray],  # horizon_name -> target array
    mid_prices: np.ndarray,
    day_boundaries: list,
    oos_start_day: int,
):
    """Comprehensive multi-timeframe agreement analysis."""
    results = {}
    horizons = sorted(preds_dict.keys())
    logger.info(f"\n{'='*70}")
    logger.info(f"MULTI-TIMEFRAME AGREEMENT ANALYSIS")
    logger.info(f"{'='*70}")

    # OOS mask
    oos_start = day_boundaries[oos_start_day]
    N = len(mid_prices)
    oos_mask = np.zeros(N, dtype=bool)
    oos_mask[oos_start:] = True

    # Common valid mask: all horizons have predictions
    valid = oos_mask.copy()
    for hz in horizons:
        valid &= np.isfinite(preds_dict[hz])
    for hz in horizons:
        valid &= np.isfinite(targets_dict[hz])
    logger.info(f"OOS bars with all predictions valid: {valid.sum():,}")

    # Direction signs for each horizon
    dirs = {}
    for hz in horizons:
        dirs[hz] = np.sign(preds_dict[hz])

    # ---- Section 1: Individual Model Performance ----
    logger.info(f"\n--- 1. INDIVIDUAL MODEL PERFORMANCE (OOS) ---")
    individual_results = {}
    for hz in horizons:
        ic, n = compute_ic(preds_dict[hz], targets_dict[hz], valid)
        daily_ics = compute_daily_ics(preds_dict[hz], targets_dict[hz], day_boundaries, valid)
        icir = np.mean(daily_ics) / max(np.std(daily_ics), 1e-6) if daily_ics else 0

        # Hit rate (sign match with own target)
        t = targets_dict[hz]
        p = preds_dict[hz]
        sign_match = (np.sign(p[valid]) == np.sign(t[valid]))
        hr = sign_match.mean() if len(sign_match) > 0 else 0

        individual_results[hz] = {'ic': ic, 'icir': icir, 'hr': hr, 'n': n}
        logger.info(f"  {hz:>4s}: IC={ic:+.4f}  ICIR={icir:.2f}  HR={hr:.1%}  N={n:,}")

    results['individual'] = individual_results

    # ---- Section 2: Cross-Horizon Correlations ----
    logger.info(f"\n--- 2. PREDICTION CROSS-CORRELATIONS (OOS) ---")
    corr_results = {}
    for i, hz1 in enumerate(horizons):
        for hz2 in horizons[i+1:]:
            p1 = preds_dict[hz1][valid]
            p2 = preds_dict[hz2][valid]
            corr = float(spearmanr(p1, p2)[0])
            corr_results[f"{hz1}_vs_{hz2}"] = corr
            # Direction agreement rate
            agree = (dirs[hz1][valid] == dirs[hz2][valid]).mean()
            logger.info(f"  {hz1} vs {hz2}: corr={corr:+.3f}  direction_agree={agree:.1%}")

    results['cross_correlations'] = corr_results

    # ---- Section 3: Agreement Filtering ----
    logger.info(f"\n--- 3. AGREEMENT-FILTERED PERFORMANCE ---")
    agreement_results = {}

    # All 3 agree
    all_agree = valid.copy()
    for hz1 in horizons:
        for hz2 in horizons:
            if hz1 != hz2:
                all_agree &= (dirs[hz1] == dirs[hz2])

    agree_rate = all_agree.sum() / max(valid.sum(), 1)
    logger.info(f"\n  ALL AGREE (3/3): {all_agree.sum():,} bars ({agree_rate:.1%} of valid)")

    for eval_hz in horizons:
        ic, n = compute_ic(preds_dict[eval_hz], targets_dict[eval_hz], all_agree)
        # Also evaluate against 10s simple return for tradability
        ret_10s = compute_forward_return_ticks(mid_prices, 100, day_boundaries)
        ic_vs_ret10, _ = compute_ic(preds_dict[eval_hz], ret_10s, all_agree)
        logger.info(f"    {eval_hz} model IC vs own target: {ic:+.4f}  "
                    f"IC vs ret_10s: {ic_vs_ret10:+.4f}  N={n:,}")

    agreement_results['all_agree'] = {
        'n_bars': int(all_agree.sum()),
        'rate': float(agree_rate),
    }

    # 2/3 agree (majority vote)
    for pair in [(horizons[0], horizons[1]),
                 (horizons[0], horizons[2]),
                 (horizons[1], horizons[2])]:
        pair_name = f"{pair[0]}+{pair[1]}"
        pair_agree = valid & (dirs[pair[0]] == dirs[pair[1]])
        rate = pair_agree.sum() / max(valid.sum(), 1)
        logger.info(f"\n  PAIR AGREE ({pair_name}): {pair_agree.sum():,} bars ({rate:.1%})")

        for eval_hz in ['10s']:
            ic, n = compute_ic(preds_dict[eval_hz], targets_dict[eval_hz], pair_agree)
            logger.info(f"    10s model IC vs own target: {ic:+.4f}  N={n:,}")

        agreement_results[pair_name] = {
            'n_bars': int(pair_agree.sum()),
            'rate': float(rate),
        }

    results['agreement'] = agreement_results

    # ---- Section 4: Consensus Signal Construction ----
    logger.info(f"\n--- 4. CONSENSUS SIGNAL CONSTRUCTION ---")

    # 4a: Simple vote (sign of majority)
    vote_sum = np.zeros(N, dtype=np.float32)
    for hz in horizons:
        vote_sum += dirs[hz]
    consensus_dir = np.sign(vote_sum)

    # 4b: IC-weighted average of predictions (standardized)
    weighted_pred = np.zeros(N, dtype=np.float32)
    total_weight = 0
    for hz in horizons:
        ic = individual_results[hz]['ic']
        w = max(ic, 0)  # Only use positive-IC models
        # Standardize predictions before combining
        p = preds_dict[hz].copy()
        p_valid = p[valid]
        if len(p_valid) > 0:
            p_std = np.nanstd(p_valid)
            if p_std > 1e-10:
                p = p / p_std
        weighted_pred += w * p
        total_weight += w
    if total_weight > 0:
        weighted_pred /= total_weight

    # 4c: Stacking meta-model (use predictions as features for a new model)
    # Build a stacking feature matrix from OOS predictions
    stack_features = np.column_stack([preds_dict[hz] for hz in horizons])
    # Add interaction terms
    for i, hz1 in enumerate(horizons):
        for hz2 in horizons[i+1:]:
            # Agreement indicator
            stack_features = np.column_stack([
                stack_features,
                (dirs[hz1] == dirs[hz2]).astype(np.float32),
            ])
            # Product (captures strength × agreement)
            stack_features = np.column_stack([
                stack_features,
                preds_dict[hz1] * preds_dict[hz2],
            ])

    consensus_results = {}

    # Evaluate each consensus method against the 10s target
    eval_target = targets_dict['10s']
    ret_10s = compute_forward_return_ticks(mid_prices, 100, day_boundaries)

    # Simple vote
    vote_pred = consensus_dir.astype(np.float32)  # -1, 0, or +1
    ic_vote, n = compute_ic(vote_pred, eval_target, valid)
    ic_vote_ret, _ = compute_ic(vote_pred, ret_10s, valid)
    consensus_results['majority_vote'] = {'ic_vs_mfe10s': ic_vote, 'ic_vs_ret10s': ic_vote_ret}
    logger.info(f"  Majority vote:       IC vs MFE_10s={ic_vote:+.4f}  IC vs ret_10s={ic_vote_ret:+.4f}")

    # IC-weighted
    ic_wt, _ = compute_ic(weighted_pred, eval_target, valid)
    ic_wt_ret, _ = compute_ic(weighted_pred, ret_10s, valid)
    consensus_results['ic_weighted'] = {'ic_vs_mfe10s': ic_wt, 'ic_vs_ret10s': ic_wt_ret}
    logger.info(f"  IC-weighted avg:     IC vs MFE_10s={ic_wt:+.4f}  IC vs ret_10s={ic_wt_ret:+.4f}")

    # Stacking meta-model (walk-forward on OOS predictions)
    # Split OOS into meta-train / meta-test
    oos_days_total = len(day_boundaries) - 1 - oos_start_day
    meta_split_day = oos_start_day + oos_days_total * 7 // 10
    meta_train_mask = oos_mask & np.zeros(N, dtype=bool)
    meta_test_mask = oos_mask & np.zeros(N, dtype=bool)
    for d in range(oos_start_day, len(day_boundaries) - 1):
        s, e = day_boundaries[d], day_boundaries[d + 1]
        if d < meta_split_day:
            meta_train_mask[s:e] = True
        else:
            meta_test_mask[s:e] = True
    meta_train_mask &= valid
    meta_test_mask &= valid

    logger.info(f"\n  Stacking meta-model: train={meta_train_mask.sum():,} bars, "
                f"test={meta_test_mask.sum():,} bars")

    if meta_train_mask.sum() > 1000 and meta_test_mask.sum() > 500:
        import lightgbm as lgb
        X_meta_tr = stack_features[meta_train_mask].astype(np.float32)
        y_meta_tr = eval_target[meta_train_mask]
        X_meta_te = stack_features[meta_test_mask].astype(np.float32)
        y_meta_te = eval_target[meta_test_mask]

        meta_model = lgb.LGBMRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            subsample=0.8, min_child_samples=200, verbose=-1, n_jobs=2,
        )
        split = int(len(X_meta_tr) * 0.8)
        meta_model.fit(
            X_meta_tr[:split], y_meta_tr[:split],
            eval_set=[(X_meta_tr[split:], y_meta_tr[split:])],
            callbacks=[lgb.early_stopping(20, verbose=False)],
        )
        meta_preds = meta_model.predict(X_meta_te)
        ic_stack = float(spearmanr(meta_preds, y_meta_te)[0])
        ic_stack_ret = float(spearmanr(
            meta_preds,
            ret_10s[meta_test_mask]
        )[0]) if np.isfinite(ret_10s[meta_test_mask]).sum() > 50 else 0
        consensus_results['stacking'] = {'ic_vs_mfe10s': ic_stack, 'ic_vs_ret10s': ic_stack_ret}
        logger.info(f"  Stacking meta:       IC vs MFE_10s={ic_stack:+.4f}  IC vs ret_10s={ic_stack_ret:+.4f}")

        # Feature importances
        feat_names = [f'pred_{hz}' for hz in horizons]
        for i, hz1 in enumerate(horizons):
            for hz2 in horizons[i+1:]:
                feat_names.append(f'agree_{hz1}_{hz2}')
                feat_names.append(f'product_{hz1}_{hz2}')
        importances = meta_model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        logger.info(f"  Meta-model feature importances:")
        for idx in sorted_idx[:8]:
            logger.info(f"    {feat_names[idx]:25s}: {importances[idx]:.0f}")

        del meta_model, X_meta_tr, X_meta_te
        gc.collect()

    results['consensus'] = consensus_results

    # ---- Section 5: Tradability Analysis ----
    logger.info(f"\n--- 5. TRADABILITY ANALYSIS ---")

    # For each signal variant, compute PnL metrics
    ret_10s_valid = ret_10s[valid]
    mid_valid = mid_prices[valid]

    for signal_name, signal_pred in [
            ('10s_model_alone', preds_dict['10s'][valid]),
            ('majority_vote', vote_pred[valid]),
            ('ic_weighted', weighted_pred[valid]),
            ('3s+10s_agree_10s_pred', None),
            ('all_agree_10s_pred', None),
    ]:
        if signal_name == '3s+10s_agree_10s_pred':
            pair_agree = (dirs['3s'][valid] == dirs['10s'][valid])
            signal_pred = preds_dict['10s'][valid].copy()
            signal_pred[~pair_agree] = 0  # zero out disagreement bars
        elif signal_name == 'all_agree_10s_pred':
            all_ag = np.ones(valid.sum(), dtype=bool)
            for hz1 in horizons:
                for hz2 in horizons:
                    if hz1 != hz2:
                        all_ag &= (dirs[hz1][valid] == dirs[hz2][valid])
            signal_pred = preds_dict['10s'][valid].copy()
            signal_pred[~all_ag] = 0

        if signal_pred is None:
            continue

        # Direction from signal
        sig_dir = np.sign(signal_pred)
        active = sig_dir != 0
        if active.sum() < 100:
            logger.info(f"  {signal_name}: too few active bars ({active.sum()})")
            continue

        # Signed PnL per bar (in ticks)
        signed_ret = sig_dir[active] * ret_10s_valid[active]
        valid_pnl = np.isfinite(signed_ret)
        signed_ret = signed_ret[valid_pnl]

        if len(signed_ret) < 100:
            continue

        mean_t = float(np.mean(signed_ret))
        median_t = float(np.median(signed_ret))
        n_active = int(active.sum())
        n_days_oos = len(day_boundaries) - 1 - oos_start_day
        per_day = n_active / max(n_days_oos, 1)

        # Profit factor
        wins = signed_ret[signed_ret > 0]
        losses = signed_ret[signed_ret < 0]
        pf = abs(wins.sum() / losses.sum()) if len(losses) > 0 and losses.sum() != 0 else 0
        wr = len(wins) / len(signed_ret)

        # After limit order costs (0.248t RT)
        net_t = mean_t - COMMISSION_TICKS
        daily_dollar = net_t * TICK_VALUE * per_day

        logger.info(f"  {signal_name}:")
        logger.info(f"    Active: {n_active:,} ({per_day:.0f}/day)  "
                    f"Mean: {mean_t:+.3f}t  Median: {median_t:+.3f}t  "
                    f"WR: {wr:.1%}  PF: {pf:.2f}")
        logger.info(f"    After LMT costs: {net_t:+.3f}t/trade  "
                    f"${daily_dollar:+.0f}/day")

    results['tradability'] = {}  # Populated above via logging

    # ---- Section 6: Conviction-Based Analysis ----
    logger.info(f"\n--- 6. CONVICTION-BASED ANALYSIS ---")
    logger.info(f"  Testing: do high-conviction consensus signals work better?")

    # Absolute prediction strength
    abs_10s = np.abs(preds_dict['10s'][valid])
    abs_wt = np.abs(weighted_pred[valid])

    for signal_name, abs_sig, pred_sig in [
            ('10s model', abs_10s, preds_dict['10s'][valid]),
            ('IC-weighted', abs_wt, weighted_pred[valid]),
    ]:
        for qtile in [50, 75, 90]:
            threshold = np.nanpercentile(abs_sig[abs_sig > 0], qtile)
            strong_mask = abs_sig > threshold
            n_strong = strong_mask.sum()
            if n_strong < 100:
                continue

            sig_dir = np.sign(pred_sig[strong_mask])
            signed_ret = sig_dir * ret_10s_valid[strong_mask]
            signed_ret = signed_ret[np.isfinite(signed_ret)]
            if len(signed_ret) < 50:
                continue

            mean_t = np.mean(signed_ret)
            wr = (signed_ret > 0).mean()
            pf = abs(signed_ret[signed_ret > 0].sum() / signed_ret[signed_ret < 0].sum()) \
                if (signed_ret < 0).sum() > 0 else 0

            logger.info(f"  {signal_name} top {100-qtile}% ({n_strong:,} bars): "
                        f"mean={mean_t:+.3f}t  WR={wr:.1%}  PF={pf:.2f}")

    # ---- Section 7: Cross-Horizon Prediction ----
    logger.info(f"\n--- 7. CROSS-HORIZON PREDICTIVE POWER ---")
    logger.info(f"  Does a 3s model predict 10s returns? Does a 30s model predict 3s returns?")

    for pred_hz in horizons:
        for target_hz in horizons:
            ic, n = compute_ic(preds_dict[pred_hz], targets_dict[target_hz], valid)
            logger.info(f"  pred_{pred_hz} vs target_{target_hz}: IC={ic:+.4f}  (N={n:,})")

    # ---- Section 8: Agreement Stability ----
    logger.info(f"\n--- 8. AGREEMENT STABILITY ACROSS DAYS ---")
    n_days = len(day_boundaries) - 1
    daily_agree_rates = []
    daily_agree_ics = []
    for d in range(oos_start_day, n_days):
        s, e = day_boundaries[d], day_boundaries[d + 1]
        day_valid = valid[s:e]
        if day_valid.sum() < 100:
            continue

        # Agreement rate this day
        d3 = dirs['3s'][s:e][day_valid]
        d10 = dirs['10s'][s:e][day_valid]
        d30 = dirs['30s'][s:e][day_valid]
        all_ag = (d3 == d10) & (d10 == d30)
        daily_agree_rates.append(float(all_ag.mean()))

        # IC on agreement bars this day
        if all_ag.sum() > 50:
            p = preds_dict['10s'][s:e][day_valid][all_ag]
            t = targets_dict['10s'][s:e][day_valid][all_ag]
            ic = float(spearmanr(p, t)[0]) if len(p) > 10 else 0
            daily_agree_ics.append(ic)

    if daily_agree_rates:
        logger.info(f"  Daily 3/3 agreement rate: mean={np.mean(daily_agree_rates):.1%}  "
                    f"std={np.std(daily_agree_rates):.1%}  "
                    f"min={np.min(daily_agree_rates):.1%}  max={np.max(daily_agree_rates):.1%}")
    if daily_agree_ics:
        logger.info(f"  Daily 10s IC on agreement bars: mean={np.mean(daily_agree_ics):+.4f}  "
                    f"std={np.std(daily_agree_ics):.4f}  "
                    f"positive {sum(1 for x in daily_agree_ics if x > 0)}/{len(daily_agree_ics)} days")

    results['stability'] = {
        'daily_agree_rate_mean': float(np.mean(daily_agree_rates)) if daily_agree_rates else 0,
        'daily_agree_ic_mean': float(np.mean(daily_agree_ics)) if daily_agree_ics else 0,
        'n_positive_days': sum(1 for x in daily_agree_ics if x > 0) if daily_agree_ics else 0,
        'n_total_days': len(daily_agree_ics) if daily_agree_ics else 0,
    }

    return results


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Multi-Timeframe Stacking Analysis')
    parser.add_argument('--n-days', type=int, default=70,
                        help='Number of days to load (default: 70)')
    parser.add_argument('--feature-cache', type=str,
                        default=DEFAULT_FEATURE_CACHE,
                        help='Feature cache directory')
    parser.add_argument('--min-train-days', type=int, default=5,
                        help='Min training days before first prediction')
    parser.add_argument('--max-train-days', type=int, default=30,
                        help='Max training days (rolling window)')
    parser.add_argument('--use-return', action='store_true',
                        help='Use simple return target instead of MFE')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode (20 days, fewer analysis)')
    parser.add_argument('--save-predictions', action='store_true', default=True,
                        help='Save prediction arrays as NPZ')
    parser.add_argument('--load-predictions', type=str, default=None,
                        help='Load saved predictions NPZ instead of training')
    args = parser.parse_args()

    if args.quick:
        args.n_days = min(args.n_days, 20)

    logger.info("=" * 70)
    logger.info("MULTI-TIMEFRAME STACKING ANALYSIS")
    logger.info(f"  n_days:      {args.n_days}")
    logger.info(f"  horizons:    {list(HORIZONS.keys())}")
    logger.info(f"  target:      {'return' if args.use_return else 'MFE'}")
    logger.info(f"  train_days:  {args.min_train_days}-{args.max_train_days}")
    logger.info(f"  log:         {_log_file}")
    logger.info("=" * 70)

    t_start = time.time()

    # Load data
    scanner, load_info = load_data(args.feature_cache, args.n_days)
    features = scanner.features
    mid_prices = scanner.mid_prices
    day_boundaries = scanner.day_boundaries

    if args.load_predictions:
        # Load saved predictions
        logger.info(f"Loading predictions from {args.load_predictions}")
        data = np.load(args.load_predictions)
        preds_dict = {}
        targets_dict = {}
        for hz in HORIZONS:
            preds_dict[hz] = data[f'preds_{hz}']
            targets_dict[hz] = data[f'target_{hz}']
        logger.info(f"Loaded predictions for {list(preds_dict.keys())}")
    else:
        # Train models for each horizon
        preds_dict = {}
        targets_dict = {}
        fold_ics_dict = {}

        for hz_name, hz_bars in HORIZONS.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"TRAINING {hz_name} MODEL (horizon={hz_bars} bars)")
            logger.info(f"{'='*50}")

            # Compute target
            logger.info(f"  Computing {hz_name} target...")
            t0 = time.time()
            target = compute_direction_target(
                mid_prices, hz_bars, day_boundaries,
                use_mfe=not args.use_return,
            )
            logger.info(f"  Target computed in {time.time()-t0:.1f}s  "
                        f"({np.isfinite(target).sum():,} valid)")
            targets_dict[hz_name] = target

            # Train walk-forward
            preds, fold_ics = train_walk_forward(
                features=features,
                target=target,
                day_boundaries=day_boundaries,
                min_train_days=args.min_train_days,
                max_train_days=args.max_train_days,
                horizon_name=hz_name,
            )
            preds_dict[hz_name] = preds
            fold_ics_dict[hz_name] = fold_ics

            gc.collect()

        # Save predictions
        if args.save_predictions:
            save_path = RESULTS_DIR / f"mt_preds_{_ts}.npz"
            save_data = {'mid_prices': mid_prices, 'day_boundaries': np.array(day_boundaries)}
            for hz in HORIZONS:
                save_data[f'preds_{hz}'] = preds_dict[hz]
                save_data[f'target_{hz}'] = targets_dict[hz]
            np.savez_compressed(str(save_path), **save_data)
            logger.info(f"\nPredictions saved to {save_path}")

    # Determine OOS split (70/30)
    n_days = len(day_boundaries) - 1
    oos_start_day = max(args.min_train_days, int(n_days * 0.7))
    logger.info(f"\nIS/OOS split: IS days 0-{oos_start_day-1}, OOS days {oos_start_day}-{n_days-1}")

    # Run agreement analysis
    results = analyze_agreement(
        preds_dict=preds_dict,
        targets_dict=targets_dict,
        mid_prices=mid_prices,
        day_boundaries=day_boundaries,
        oos_start_day=oos_start_day,
    )

    # Save results
    elapsed = time.time() - t_start
    results['config'] = {
        'n_days': args.n_days,
        'horizons': list(HORIZONS.keys()),
        'target_type': 'return' if args.use_return else 'mfe',
        'min_train_days': args.min_train_days,
        'max_train_days': args.max_train_days,
        'oos_start_day': oos_start_day,
        'elapsed_sec': elapsed,
    }

    json_path = RESULTS_DIR / f"multi_timeframe_{_ts}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\n{'='*70}")
    logger.info(f"COMPLETE in {elapsed:.0f}s ({elapsed/60:.1f}m)")
    logger.info(f"Results: {json_path}")
    logger.info(f"Log:     {_log_file}")
    logger.info(f"{'='*70}")


if __name__ == '__main__':
    main()
