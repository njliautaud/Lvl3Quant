"""
Vol-Regime Gate — Only trade on "active" days where signal has edge.

Prior research showed:
- ACTIVE days (vol >= 0.06 bps): IC = +0.184, 100% positive, 16/35 days
- DEAD days (vol < 0.06 bps): IC ~ 0, pure noise, 19/35 days
- Active days contribute 99% of total edge

This module provides:
1. Real-time vol regime detection (is today active or dead?)
2. Walk-forward regime-gated evaluation (only score on active days)
3. Adaptive threshold selection from historical data
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


# Default threshold from prior research (bps of mid price)
DEFAULT_VOL_THRESHOLD_BPS = 0.06


def compute_daily_vol(mid_prices: np.ndarray, day_boundaries: np.ndarray) -> np.ndarray:
    """Compute realized volatility per day in basis points of mid price.

    Returns array of shape (n_days,) with vol in bps.
    """
    n_days = len(day_boundaries) - 1
    daily_vol = np.zeros(n_days, dtype=np.float32)

    for d in range(n_days):
        ds = day_boundaries[d]
        de = day_boundaries[d + 1]
        day_mid = mid_prices[ds:de]

        if len(day_mid) < 100:
            daily_vol[d] = 0.0
            continue

        # Realized vol: std of 1-bar log returns, annualized to bps
        log_ret = np.diff(np.log(np.maximum(day_mid, 1.0)))
        daily_vol[d] = float(np.std(log_ret)) * 1e4  # Convert to bps

    return daily_vol


def classify_regime(daily_vol: np.ndarray,
                     threshold_bps: float = DEFAULT_VOL_THRESHOLD_BPS) -> np.ndarray:
    """Classify each day as ACTIVE (1) or DEAD (0).

    Returns boolean array of shape (n_days,).
    """
    return daily_vol >= threshold_bps


def compute_intraday_vol_signal(mid_prices: np.ndarray,
                                 window: int = 500) -> np.ndarray:
    """Real-time rolling vol signal (for live gating).

    Computes rolling std of 1-bar returns over `window` bars.
    Returns array same length as mid_prices, in bps.
    First `window` bars are NaN.
    """
    N = len(mid_prices)
    vol_signal = np.full(N, np.nan, dtype=np.float32)

    log_ret = np.diff(np.log(np.maximum(mid_prices, 1.0)))

    # Rolling std using cumsum trick
    ret_sq = log_ret ** 2
    cumsum = np.cumsum(ret_sq)
    cumsum_mean = np.cumsum(log_ret)

    for i in range(window, len(log_ret)):
        sum_sq = cumsum[i] - cumsum[i - window]
        sum_mean = cumsum_mean[i] - cumsum_mean[i - window]
        var = sum_sq / window - (sum_mean / window) ** 2
        vol_signal[i + 1] = float(np.sqrt(max(var, 0))) * 1e4

    return vol_signal


def find_optimal_threshold(daily_vol: np.ndarray,
                            fold_ics: np.ndarray,
                            n_candidates: int = 20) -> Dict:
    """Find optimal vol threshold that maximizes gated IC.

    Tests multiple thresholds and picks the one that maximizes
    mean IC on active days while maintaining enough trading days.

    Args:
        daily_vol: Per-day volatility (n_days,)
        fold_ics: Per-fold IC values aligned with days (n_folds,)
        n_candidates: Number of threshold candidates to test

    Returns:
        Dict with optimal threshold and stats
    """
    # Both arrays must be same length (one per test day)
    assert len(daily_vol) == len(fold_ics), \
        f"Length mismatch: daily_vol={len(daily_vol)}, fold_ics={len(fold_ics)}"

    vol_sorted = np.sort(daily_vol)
    # Test thresholds from 10th to 90th percentile
    candidates = np.linspace(
        np.percentile(vol_sorted, 10),
        np.percentile(vol_sorted, 90),
        n_candidates
    )

    best = {'threshold': DEFAULT_VOL_THRESHOLD_BPS, 'score': -1}
    results = []

    for thresh in candidates:
        active = daily_vol >= thresh
        n_active = active.sum()

        if n_active < 5:  # Need at least 5 active days
            continue

        active_ics = fold_ics[active]
        dead_ics = fold_ics[~active]

        mean_active_ic = float(np.mean(active_ics))
        pct_active = n_active / len(daily_vol)

        # Score: IC * sqrt(n_active) — balances signal strength with sample size
        score = mean_active_ic * np.sqrt(n_active)

        result = {
            'threshold': float(thresh),
            'n_active': int(n_active),
            'n_dead': int(len(daily_vol) - n_active),
            'pct_active': float(pct_active),
            'mean_active_ic': mean_active_ic,
            'mean_dead_ic': float(np.mean(dead_ics)) if len(dead_ics) > 0 else 0,
            'score': float(score),
        }
        results.append(result)

        if score > best['score']:
            best = result

    return {
        'optimal': best,
        'all_candidates': sorted(results, key=lambda x: x['score'], reverse=True),
    }


def gated_walk_forward(features: np.ndarray,
                        target: np.ndarray,
                        mid_prices: np.ndarray,
                        day_boundaries: np.ndarray,
                        threshold_bps: float = DEFAULT_VOL_THRESHOLD_BPS,
                        min_train_days: int = 5) -> Dict:
    """Walk-forward evaluation with vol-regime gating.

    Only evaluates predictions on ACTIVE days. Training uses all days.
    Returns separate metrics for active/dead/all days.
    """
    import lightgbm as lgb
    from scipy.stats import spearmanr

    n_days = len(day_boundaries) - 1
    daily_vol = compute_daily_vol(mid_prices, day_boundaries)
    regime = classify_regime(daily_vol, threshold_bps)

    params = {
        'n_estimators': 300, 'max_depth': 5, 'learning_rate': 0.05,
        'subsample': 0.8, 'colsample_bytree': 0.7,
        'min_child_samples': 500, 'verbose': -1, 'n_jobs': -1,
        'device': 'gpu',
        'objective': 'regression', 'metric': 'rmse',
    }

    active_ics = []
    dead_ics = []
    all_ics = []

    for test_day in range(min_train_days, n_days):
        train_end = day_boundaries[test_day - 1 + 1]
        test_start = day_boundaries[test_day]
        test_end = day_boundaries[test_day + 1]

        X_tr, y_tr = features[:train_end], target[:train_end]
        X_te, y_te = features[test_start:test_end], target[test_start:test_end]

        tr_valid = np.isfinite(y_tr)
        te_valid = np.isfinite(y_te)

        if tr_valid.sum() < 1000 or te_valid.sum() < 100:
            continue

        split = int(tr_valid.sum() * 0.8)
        try:
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_tr[tr_valid][:split], y_tr[tr_valid][:split],
                eval_set=[(X_tr[tr_valid][split:], y_tr[tr_valid][split:])],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            preds = model.predict(X_te[te_valid])
            ic = float(spearmanr(preds, y_te[te_valid])[0])
        except Exception:
            continue

        if not np.isfinite(ic):
            continue

        all_ics.append(ic)
        if regime[test_day]:
            active_ics.append(ic)
        else:
            dead_ics.append(ic)

        del model

    def _stats(ics):
        if not ics:
            return {'ic': 0, 'icir': 0, 'tstat': 0, 'n': 0, 'pct_positive': 0}
        arr = np.array(ics)
        m, s = arr.mean(), arr.std()
        return {
            'ic': float(m),
            'icir': float(m / s) if s > 0 else 0,
            'tstat': float(m / s * np.sqrt(len(arr))) if s > 0 else 0,
            'n': len(arr),
            'pct_positive': float((arr > 0).mean()),
        }

    return {
        'all': _stats(all_ics),
        'active': _stats(active_ics),
        'dead': _stats(dead_ics),
        'threshold_bps': threshold_bps,
        'regime_summary': {
            'n_active_days': int(regime.sum()),
            'n_dead_days': int((~regime).sum()),
            'pct_active': float(regime.mean()),
        },
    }
