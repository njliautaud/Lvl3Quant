"""
MFE (Max Favorable Excursion) Alpha Scan.

The user's insight: fixed-horizon returns miss scalping alpha.
If price goes +2 ticks at bar 20 then returns to 0 at bar 30, the 3s
return is zero but a scalper captured +2 ticks.

MFE targets ask: "within the next N bars, how far can price run in
each direction?"  The NET MFE tests whether the model can predict
which direction has MORE room to run, even when the horizon return is flat.

Targets:
  mfe_long_3s   — max(mid[t+1:t+30]  - mid[t]) / tick_size   (best upward move in 3s)
  mfe_short_3s  — max(mid[t]  - mid[t+1:t+30]) / tick_size   (best downward move in 3s)
  mfe_net_3s    — mfe_long_3s - mfe_short_3s  (directional asymmetry)
  ... same for 5s (50 bars) and 10s (100 bars) ...

Key design decisions:
1. Reuses the existing feature cache — no re-computation.
2. Excludes the SAME 20 vol/price/time features as run_return_multihorizon.py.
3. Day-boundary NaN-fill: the MFE window must not cross midnight.
4. Walk-forward LightGBM, identical hyperparams to the return scan.
5. Reports per-fold IC, fold consistency, t-stat, top features.
6. Compares results to the known 3s return IC=0.079.

Usage:
    python alpha_discovery/run_mfe_scan.py
    python alpha_discovery/run_mfe_scan.py --fast
    python alpha_discovery/run_mfe_scan.py --horizons 3s 5s
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

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from alpha_discovery.mbo_alpha_scan import MBOAlphaScanner, RESULTS_DIR
from alpha_discovery.mbo_features import get_feature_names, TOTAL_FEATURES

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(RESULTS_DIR / 'mfe_scan.log', mode='a'),
    ]
)
logger = logging.getLogger("run_mfe_scan")

# ---------------------------------------------------------------------------
# Feature cache — same location as the return scan
# ---------------------------------------------------------------------------
FEATURE_CACHE = RESULTS_DIR / "feature_cache"


def _feature_names_hash() -> str:
    import hashlib
    names_str = ",".join(get_feature_names())
    return hashlib.md5(names_str.encode()).hexdigest()[:12]


def load_feature_cache(scanner: MBOAlphaScanner) -> Optional[dict]:
    """Load pre-computed features from disk cache (identical to return scan)."""
    path = FEATURE_CACHE / "features_alldays.npz"
    if not path.exists():
        return None

    logger.info(f"Loading cached features from {path} ...")
    t0 = time.time()
    data = np.load(str(path), allow_pickle=True)

    cached_features = data['features']
    if cached_features.shape[1] != TOTAL_FEATURES:
        logger.warning(
            f"Feature cache STALE: {cached_features.shape[1]} vs "
            f"{TOTAL_FEATURES} expected — will recompute."
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
            logger.warning("Feature cache STALE: feature names changed — will recompute.")
            data.close()
            return None
    else:
        stats = {
            'n_snapshots': int(len(data['mid_prices'])),
            'n_days': int(len(data['day_boundaries']) - 1),
            'n_features': int(cached_features.shape[1]),
        }

    scanner.features      = cached_features
    scanner.mid_prices    = data['mid_prices']
    scanner.hour_of_day   = data['hour_of_day']
    scanner.time_since_rth = data['time_since_rth']
    scanner.day_boundaries = list(data['day_boundaries'])
    scanner.feature_names  = get_feature_names()

    logger.info(f"Cache loaded in {time.time()-t0:.1f}s: {scanner.features.shape}")
    return stats


# ---------------------------------------------------------------------------
# Features to exclude (SAME list as run_return_multihorizon.py)
# ---------------------------------------------------------------------------
EXCLUDE_FEATURES = [
    # Absolute price level
    'mid', 'best_bid', 'best_ask', 'microprice',
    # Time-of-day
    'hour_norm', 'minute_norm', 'time_since_rth', 'time_to_close',
    # Vol proxies
    'rvol_10', 'rvol_20', 'rvol_50',
    'vov_10',  'vov_20',  'vov_50',
    'tick_count',
    # Event intensity
    'event_int_5', 'event_int_20', 'event_int_50',
    # Per-level event density
    'tick_density_5', 'tick_density_20',
]


# ---------------------------------------------------------------------------
# MFE Target Computation
# ---------------------------------------------------------------------------

def compute_mfe_targets(
    mid_prices: np.ndarray,
    day_boundaries: List[int],
    sample_interval_ms: int = 100,
    horizons_sec: Optional[Dict[str, int]] = None,
    tick_size: float = 0.25,
) -> Dict[str, np.ndarray]:
    """
    Compute Max Favorable Excursion targets.

    For each bar t and horizon H (in bars):
      mfe_long[t]  = max(mid[t+1 : t+H+1] - mid[t]) / tick_size
                   = best upward excursion in H bars (in ticks)
      mfe_short[t] = max(mid[t]  - mid[t+1 : t+H+1]) / tick_size
                   = best downward excursion in H bars (in ticks)
      mfe_net[t]   = mfe_long[t] - mfe_short[t]
                   = directional asymmetry (positive → more upside room)

    All three are NaN when the window crosses a day boundary (strict causal rule).
    mfe_long and mfe_short are always >= 0.
    mfe_net can be negative (more downside than upside).

    Key property:
      Even when the fixed-horizon return is 0 at t+H, mfe_net can be large if
      price ran +3 ticks then returned.  The model is asked to predict the
      DIRECTIONAL ASYMMETRY — not just end-of-horizon direction.

    Returns dict of arrays, all shape (N,) float32.
    """
    if horizons_sec is None:
        horizons_sec = {'3s': 3, '5s': 5, '10s': 10}

    N = len(mid_prices)
    steps_per_sec = 1000.0 / sample_interval_ms
    n_days = len(day_boundaries) - 1

    targets = {}

    # Build a day-index array: day_idx[t] = which day bar t belongs to
    # Used to efficiently mask boundary crossings.
    day_idx = np.full(N, -1, dtype=np.int32)
    for d in range(n_days):
        s = day_boundaries[d]
        e = day_boundaries[d + 1]
        day_idx[s:e] = d

    for hz_name, hz_sec in horizons_sec.items():
        H = int(hz_sec * steps_per_sec)  # number of forward bars
        if H >= N:
            logger.warning(f"Horizon {hz_name} ({H} bars) >= N={N}, skipping")
            continue

        logger.info(f"  Computing MFE for {hz_name} ({H} bars) ...")
        t0 = time.time()

        mfe_long  = np.full(N, np.nan, dtype=np.float32)
        mfe_short = np.full(N, np.nan, dtype=np.float32)

        # ------------------------------------------------------------------
        # Vectorised rolling max using cumulative max from the right.
        #
        # mfe_long[t]  = max(mid[t+1:t+H+1] - mid[t]) / tick
        # mfe_short[t] = max(mid[t] - mid[t+1:t+H+1]) / tick
        #
        # We compute these as:
        #   fwd_max[t] = max(mid[t+1:t+H+1])
        #   fwd_min[t] = min(mid[t+1:t+H+1])
        # then:
        #   mfe_long[t]  = max(0, fwd_max[t] - mid[t]) / tick
        #   mfe_short[t] = max(0, mid[t] - fwd_min[t]) / tick
        #
        # Rolling max/min over a forward window of size H can be computed
        # efficiently with sliding_window_view on the reversed array, but
        # for simplicity we use a segment-tree approach: for each bar t,
        # the future window is mid[t+1 : t+H+1].
        #
        # Vectorised approach:
        #   rolling_max[t] = max of mid[t+1:t+H+1]
        # Use stride tricks for efficiency (no Python loops).
        # ------------------------------------------------------------------
        from numpy.lib.stride_tricks import sliding_window_view

        # We need fwd_max[t] = max(mid[t+1:t+H+1]) for t in [0, N-H-2].
        # valid_len = N - H - 1 (number of bars t where the window is fully within array)
        # mid_shifted[t:t+H] = mid[t+1:t+H+1], so:
        #   sliding_window_view(mid_shifted, H) gives (N-H, H) windows
        #   Take first valid_len = N-H-1 of them (= all windows where t+H <= N-1, i.e., t <= N-H-1)
        # Note: N-H windows are available (indices t in [0, N-H-1]) but we stop at N-H-2
        # because t=N-H-1 would give window mid[N-H:N] — fully valid, so actually valid_len = N-H-1.
        mid_shifted = mid_prices[1:]  # length N-1; mid_shifted[i] = mid[i+1]
        valid_len = N - H - 1         # bars t = 0 .. N-H-2 (window mid[t+1:t+H+1] fits in array)

        if valid_len <= 0:
            logger.warning(f"  {hz_name}: valid_len={valid_len}, skipping")
            continue

        # sliding_window_view(mid_shifted, H) has shape (N-1-H+1, H) = (N-H, H)
        # window[t] = mid_shifted[t:t+H] = mid[t+1:t+H+1]
        # We need t in [0, valid_len-1] = [0, N-H-2], so take first valid_len windows.
        all_windows = sliding_window_view(mid_shifted, H)  # (N-H, H)
        windows = all_windows[:valid_len]                   # (valid_len, H)
        assert windows.shape == (valid_len, H), \
            f"windows shape {windows.shape} != ({valid_len}, {H})"

        fwd_max = windows.max(axis=1)  # (valid_len,)
        fwd_min = windows.min(axis=1)  # (valid_len,)

        # MFE in ticks (always >= 0)
        mfe_long_vals  = np.maximum(0.0, (fwd_max - mid_prices[:valid_len]) / tick_size)
        mfe_short_vals = np.maximum(0.0, (mid_prices[:valid_len] - fwd_min) / tick_size)

        mfe_long[:valid_len]  = mfe_long_vals.astype(np.float32)
        mfe_short[:valid_len] = mfe_short_vals.astype(np.float32)

        # ------------------------------------------------------------------
        # NaN-fill bars whose windows cross a day boundary.
        # Any bar t where day_idx[t] != day_idx[t+H] must be NaN.
        # (The window [t+1:t+H+1] spans day_idx[t] and at least one later day.)
        # ------------------------------------------------------------------
        if n_days > 1:
            for d in range(n_days - 1):
                day_end = day_boundaries[d + 1]
                # Bars from max(day_start, day_end - H) to day_end - 1 cross boundary
                nan_start = max(day_boundaries[d], day_end - H)
                nan_end   = day_end  # exclusive
                mfe_long[nan_start:nan_end]  = np.nan
                mfe_short[nan_start:nan_end] = np.nan

        # Net MFE (directional asymmetry)
        mfe_net = mfe_long - mfe_short  # NaN where either is NaN

        targets[f'mfe_long_{hz_name}']  = mfe_long
        targets[f'mfe_short_{hz_name}'] = mfe_short
        targets[f'mfe_net_{hz_name}']   = mfe_net

        n_valid = int(np.isfinite(mfe_net).sum())
        mean_l  = float(np.nanmean(mfe_long))
        mean_s  = float(np.nanmean(mfe_short))
        mean_n  = float(np.nanmean(mfe_net))
        logger.info(
            f"  {hz_name}: valid={n_valid:,} | "
            f"long_mean={mean_l:.3f}t  short_mean={mean_s:.3f}t  "
            f"net_mean={mean_n:+.4f}t  ({time.time()-t0:.1f}s)"
        )

    return targets


# ---------------------------------------------------------------------------
# Walk-Forward Evaluation (re-uses the same structure as return scan)
# ---------------------------------------------------------------------------

def walk_forward_evaluate_mfe(
    scanner: MBOAlphaScanner,
    target: np.ndarray,
    target_name: str,
    exclude_features: List[str],
    min_train_days: int = 3,
) -> dict:
    """
    Walk-forward LightGBM evaluation for MFE targets (regression).

    Identical to the return scan's walk_forward_evaluate_return — kept as a
    standalone function to avoid coupling and to allow future specialisation.
    """
    import lightgbm as lgb

    n_days = len(scanner.day_boundaries) - 1
    if n_days < min_train_days + 1:
        return {
            'error': f'Need {min_train_days + 1} days, have {n_days}',
            'target': target_name,
        }

    # Feature mask
    keep_mask = np.array([fn not in exclude_features for fn in scanner.feature_names])
    features_use       = scanner.features[:, keep_mask]
    feature_names_use  = [fn for fn in scanner.feature_names if fn not in exclude_features]
    n_features_use     = len(feature_names_use)
    logger.info(f"  Using {n_features_use} features (excluded {(~keep_mask).sum()})")

    params = {
        'n_estimators':      500,
        'max_depth':         6,
        'learning_rate':     0.03,
        'subsample':         0.8,
        'colsample_bytree':  0.7,
        'reg_alpha':         0.1,
        'reg_lambda':        1.0,
        'min_child_samples': 100,
        'objective':         'regression',
        'metric':            'rmse',
        'verbose':           -1,
        'n_jobs':            -1,
    }

    all_preds        = []
    all_actuals      = []
    all_rth_frac     = []
    fold_ics         = []
    fold_metrics     = []
    feature_importance = np.zeros(n_features_use)

    for test_day in range(min_train_days, n_days):
        train_end_day = test_day - 1
        train_start   = scanner.day_boundaries[0]
        train_end     = scanner.day_boundaries[train_end_day + 1]
        test_start    = scanner.day_boundaries[test_day]
        test_end      = scanner.day_boundaries[test_day + 1]

        X_train = features_use[train_start:train_end]
        y_train = target[train_start:train_end]
        X_test  = features_use[test_start:test_end]
        y_test  = target[test_start:test_end]

        train_valid = np.isfinite(y_train)
        test_valid  = np.isfinite(y_test)

        if train_valid.sum() < 500 or test_valid.sum() < 50:
            logger.warning(f"  Day {test_day}: train={train_valid.sum()}, "
                           f"test={test_valid.sum()} — skipping")
            continue

        X_tr = X_train[train_valid]
        y_tr = y_train[train_valid]
        X_te = X_test[test_valid]
        y_te = y_test[test_valid]

        split = int(len(X_tr) * 0.8)
        try:
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_tr[:split], y_tr[:split],
                eval_set=[(X_tr[split:], y_tr[split:])],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
        except Exception as e:
            logger.warning(f"  Training failed day {test_day}: {e}")
            continue

        preds = model.predict(X_te)
        all_preds.append(preds)
        all_actuals.append(y_te)

        rth = scanner.time_since_rth[test_start:test_end][test_valid]
        all_rth_frac.append(rth)

        if len(preds) > 10:
            try:
                ic_fold = float(spearmanr(preds, y_te)[0])
                if np.isfinite(ic_fold):
                    fold_ics.append(ic_fold)
                    fold_metrics.append({
                        'day': test_day,
                        'ic': ic_fold,
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
    actuals     = np.concatenate(all_actuals)
    rth_fracs   = np.concatenate(all_rth_frac)

    valid = np.isfinite(predictions) & np.isfinite(actuals)
    p, a  = predictions[valid], actuals[valid]
    rf    = rth_fracs[valid]

    if len(p) < 50:
        return {'error': f'Too few predictions: {len(p)}', 'target': target_name}

    # ---- Core metrics ----
    ic   = float(spearmanr(p, a)[0])
    hr   = float((np.sign(p - np.median(p)) == np.sign(a)).mean())

    winners = np.abs(a[np.sign(p - np.median(p)) == np.sign(a)]).sum()
    losers  = np.abs(a[np.sign(p - np.median(p)) != np.sign(a)]).sum()
    pf      = float(winners / losers) if losers > 0 else 0.0

    # ---- ICIR / t-stat ----
    if len(fold_ics) > 2:
        ic_mean  = float(np.mean(fold_ics))
        ic_std   = float(np.std(fold_ics))
        icir     = ic_mean / ic_std if ic_std > 0 else 0.0
        tstat    = ic_mean / ic_std * np.sqrt(len(fold_ics)) if ic_std > 0 else 0.0
        _, pvalue = ttest_1samp(fold_ics, 0)
        pvalue   = float(pvalue)
    else:
        ic_mean, ic_std = ic, 0.0
        icir, tstat, pvalue = 0.0, 0.0, 1.0

    # ---- Session phase breakdown ----
    session_stats = {}
    phases = {
        'open_30min':  (0.0,  0.075),
        'morning':     (0.075, 0.38),
        'midday':      (0.38, 0.62),
        'afternoon':   (0.62, 0.92),
        'close_30min': (0.92, 1.0),
    }
    for phase_name, (lo, hi) in phases.items():
        mask = (rf >= lo) & (rf < hi)
        if mask.sum() > 50:
            try:
                ic_s = float(spearmanr(p[mask], a[mask])[0])
            except Exception:
                ic_s = 0.0
            session_stats[phase_name] = {
                'ic': ic_s,
                'hit_rate': float((np.sign(p[mask] - np.median(p)) == np.sign(a[mask])).mean()),
                'n_samples': int(mask.sum()),
            }

    # ---- Top features ----
    top_feat_idx = np.argsort(feature_importance)[::-1][:20]
    top_features = [
        (feature_names_use[i], float(feature_importance[i]))
        for i in top_feat_idx if feature_importance[i] > 0
    ]

    # ---- Fold consistency ----
    n_pos    = sum(1 for ic_ in fold_ics if ic_ > 0)
    fold_con = n_pos / len(fold_ics) if fold_ics else 0.0

    passed = abs(ic) > 0.02 and abs(tstat) > 2.0 and fold_con > 0.60

    return {
        'target':           target_name,
        'ic':               ic,
        'ic_mean':          ic_mean,
        'ic_std':           ic_std,
        'icir':             icir,
        'tstat':            tstat,
        'pvalue':           pvalue,
        'hit_rate':         hr,
        'profit_factor':    pf,
        'fold_consistency': fold_con,
        'n_positive_folds': n_pos,
        'n_folds':          len(fold_ics),
        'fold_ics':         [float(x) for x in fold_ics],
        'fold_metrics':     fold_metrics,
        'session_stats':    session_stats,
        'top_features':     top_features,
        'n_predictions':    len(p),
        'passed':           passed,
    }


# ---------------------------------------------------------------------------
# Comparison against fixed-horizon returns
# ---------------------------------------------------------------------------

RETURN_SCAN_REFERENCE = {
    # From run_return_multihorizon.py results (20260217_070243)
    'ret_3s':  {'ic': 0.079,  'tstat': 2.68, 'fold_consistency': 0.833,
                'top_features': ['ask_L1_conc', 'bid_L1_conc', 'depth_ratio_l1']},
    'ret_5s':  {'ic': 0.000,  'tstat': 0.00, 'fold_consistency': 0.0},
    'ret_10s': {'ic': 0.000,  'tstat': 0.00, 'fold_consistency': 0.0},
    'tick_mag_1s': {'ic': 0.186, 'tstat': 2.56, 'fold_consistency': 1.0,
                    'note': '100% folds — strongest so far'},
}


def compare_to_returns(mfe_results: List[dict]) -> str:
    """Format comparison table: MFE vs fixed-horizon returns."""
    lines = [
        "",
        "MFE vs FIXED-HORIZON RETURN COMPARISON",
        "=" * 90,
        f"{'Target':<22s} {'IC':>7s} {'t':>6s} {'FoldC':>6s} {'Pass':>5s}  NOTE",
        "-" * 90,
        "--- Fixed-horizon (reference) ---",
    ]
    ref_order = ['tick_mag_1s', 'ret_3s', 'ret_5s', 'ret_10s']
    for k in ref_order:
        r = RETURN_SCAN_REFERENCE.get(k, {})
        ic_s = f"{r.get('ic', 0):>7.4f}"
        t_s  = f"{r.get('tstat', 0):>6.2f}"
        fc   = f"{r.get('fold_consistency', 0):>6.1%}"
        note = r.get('note', '')
        lines.append(f"  {k:<20s} {ic_s} {t_s} {fc} {'YES' if r.get('ic',0) > 0.02 else 'no':>5s}  {note}")

    lines.append("")
    lines.append("--- MFE targets (new) ---")
    # Sort by |IC|
    sorted_mfe = sorted(mfe_results, key=lambda x: abs(x.get('ic', 0)), reverse=True)
    for r in sorted_mfe:
        if 'error' in r:
            lines.append(f"  {r.get('target','?'):<20s}  ERROR: {r['error']}")
            continue
        passed = "YES" if r['passed'] else "no"
        fc     = f"{r['fold_consistency']:>6.1%}"
        lines.append(
            f"  {r['target']:<20s} {r['ic']:>7.4f} {r['tstat']:>6.2f} "
            f"{fc} {passed:>5s}"
        )

    lines.append("=" * 90)
    lines.append("")
    lines.append("INTERPRETATION:")
    lines.append("  If mfe_net IC > ret_3s IC (0.079) → MFE captures more directional signal")
    lines.append("  If mfe_net IC ~ ret_3s IC       → MFE is equivalent; fixed-horizon is fine")
    lines.append("  If mfe_net IC < ret_3s IC       → MFE doesn't help; user intuition wrong")
    lines.append("  If mfe_long/short IC high but net IC low → directional magnitude not predictable,")
    lines.append("    only vol/uncertainty is (matches our existing vol alpha)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Scoreboard
# ---------------------------------------------------------------------------

def format_scoreboard(results: List[dict]) -> str:
    lines = [
        "",
        "MFE SCAN SCOREBOARD",
        "=" * 100,
        f"{'Target':<22s} {'IC':>7s} {'ICIR':>6s} {'t':>6s} {'HR':>6s} "
        f"{'PF':>6s} {'FoldC%':>7s} {'Folds':>5s} {'Preds':>8s} {'Pass':>5s}",
        "-" * 100,
    ]

    sorted_r = sorted(results, key=lambda x: abs(x.get('ic', 0)), reverse=True)
    for r in sorted_r:
        if 'error' in r:
            lines.append(f"{r.get('target','?'):<22s}  ERROR: {r['error']}")
            continue
        passed = "YES" if r['passed'] else "no"
        fc     = f"{r['fold_consistency']:.0%}"
        lines.append(
            f"{r['target']:<22s} {r['ic']:>7.4f} {r['icir']:>6.2f} {r['tstat']:>6.2f} "
            f"{r['hit_rate']:>6.1%} {r['profit_factor']:>6.2f} {fc:>7s} "
            f"{r['n_folds']:>5d} {r['n_predictions']:>8d} {passed:>5s}"
        )

    lines.append("=" * 100)

    lines.append("\nPER-FOLD IC EVOLUTION (top 6):")
    for r in sorted_r[:6]:
        if 'error' in r or not r.get('fold_ics'):
            continue
        ics = " ".join(f"{x:+.3f}" for x in r['fold_ics'])
        lines.append(f"  {r['target']:<22s}: [{ics}]")

    lines.append("\nTOP FEATURES (top 5 by |IC|):")
    for r in sorted_r[:5]:
        if 'error' in r or not r.get('top_features'):
            continue
        top = ", ".join(f[0] for f in r['top_features'][:5])
        lines.append(f"  {r['target']:<22s}: {top}")

    winners = [r for r in results if r.get('passed', False)]
    if winners:
        lines.append(f"\nALPHA FOUND in {len(winners)}/{len(results)} MFE targets:")
        for w in winners:
            lines.append(
                f"  {w['target']}: IC={w['ic']:.4f} t={w['tstat']:.2f} "
                f"FoldC={w['fold_consistency']:.0%}"
            )
    else:
        if sorted_r and 'error' not in sorted_r[0]:
            b = sorted_r[0]
            lines.append(
                f"\nNo MFE target beats threshold. "
                f"Best: {b['target']} IC={b['ic']:.4f} t={b['tstat']:.2f}"
            )

    return "\n".join(lines)


def format_discord_summary(results: List[dict], comparison: str, elapsed_sec: float) -> str:
    sorted_r = sorted(results, key=lambda x: abs(x.get('ic', 0)), reverse=True)
    winners  = [r for r in results if r.get('passed', False)]

    lines = [
        "**MFE SCAN COMPLETE**",
        f"Elapsed: {elapsed_sec/60:.1f} min | Targets: {len(results)}",
        "",
        "**SCOREBOARD** (sorted by |IC|):",
        "```",
        f"{'Target':<22s} {'IC':>7s} {'t':>6s} {'FoldC':>6s} {'Pass':>5s}",
        "-" * 52,
    ]
    for r in sorted_r[:12]:
        if 'error' in r:
            lines.append(f"{r.get('target','?'):<22s}  ERROR")
            continue
        fc     = f"{r['fold_consistency']:.0%}"
        passed = "YES" if r['passed'] else "---"
        lines.append(f"{r['target']:<22s} {r['ic']:>7.4f} {r['tstat']:>6.2f} {fc:>6s} {passed:>5s}")
    lines.append("```")

    lines.append("\n**vs REFERENCE (ret_3s IC=0.079, t=2.68):**")
    if winners:
        lines.append(f"**ALPHA FOUND in {len(winners)} MFE targets!**")
        for w in winners:
            lines.append(f"- `{w['target']}`: IC={w['ic']:.4f}, t={w['tstat']:.2f}")
            if w.get('top_features'):
                top3 = [f[0] for f in w['top_features'][:3]]
                lines.append(f"  Top: {', '.join(top3)}")
    else:
        if sorted_r and 'error' not in sorted_r[0]:
            b = sorted_r[0]
            lines.append(
                f"No MFE alpha beats threshold. Best: `{b['target']}` "
                f"IC={b['ic']:.4f}, t={b['tstat']:.2f}"
            )

    # Honest verdict on user's intuition
    net_results = [r for r in results if 'net' in r.get('target', '') and 'error' not in r]
    if net_results:
        best_net = max(net_results, key=lambda x: abs(x.get('ic', 0)))
        ref_ic   = 0.079
        if best_net['ic'] > ref_ic * 1.1:
            verdict = f"**CONFIRMED: MFE net IC={best_net['ic']:.4f} > ret_3s IC={ref_ic:.4f}. Scalping alpha EXISTS.**"
        elif best_net['ic'] > ref_ic * 0.9:
            verdict = f"**NEUTRAL: MFE net IC={best_net['ic']:.4f} ~ ret_3s IC={ref_ic:.4f}. MFE doesn't add much.**"
        else:
            verdict = f"**NOT CONFIRMED: MFE net IC={best_net['ic']:.4f} < ret_3s IC={ref_ic:.4f}. Fixed-horizon is better.**"
        lines.append(f"\n{verdict}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='MFE scan')
    parser.add_argument('--fast', action='store_true',
                        help='Fast mode: skip long/short, only run net targets')
    parser.add_argument('--horizons', nargs='+', default=['3s', '5s', '10s'],
                        help='Horizons to scan (default: 3s 5s 10s)')
    parser.add_argument('--min-train-days', type=int, default=3)
    args = parser.parse_args()

    logger.info("=" * 75)
    logger.info("MFE SCAN — MAX FAVORABLE EXCURSION ALPHA DISCOVERY")
    logger.info(f"  Horizons: {args.horizons}")
    logger.info(f"  Fast mode: {args.fast}")
    logger.info(f"  Excluded features: {len(EXCLUDE_FEATURES)}")
    logger.info("=" * 75)

    t_start = time.time()

    # Load feature cache
    scanner = MBOAlphaScanner(sample_interval_ms=100)
    stats   = load_feature_cache(scanner)
    if stats is None:
        logger.info("No feature cache — computing from scratch (slow)...")
        stats = scanner.load_from_cache()
    else:
        logger.info(f"Cached: {stats}")

    logger.info(
        f"Data: {stats.get('n_snapshots','?'):,} snapshots, "
        f"{stats.get('n_days','?')} days, "
        f"{len(scanner.feature_names)} features"
    )

    # Parse horizon dict
    horizons_sec = {}
    for hz in args.horizons:
        if hz.endswith('s'):
            horizons_sec[hz] = int(hz[:-1])
        elif hz.endswith('m'):
            horizons_sec[hz] = int(hz[:-1]) * 60
        else:
            horizons_sec[hz] = int(hz)

    # Compute MFE targets
    logger.info("\nComputing MFE targets ...")
    mfe_targets = compute_mfe_targets(
        mid_prices=scanner.mid_prices,
        day_boundaries=scanner.day_boundaries,
        sample_interval_ms=100,
        horizons_sec=horizons_sec,
    )
    logger.info(f"Computed {len(mfe_targets)} MFE targets: {list(mfe_targets.keys())}")

    # In fast mode: only run net targets
    if args.fast:
        targets_to_run = {k: v for k, v in mfe_targets.items() if 'net' in k}
        logger.info(f"Fast mode: running {len(targets_to_run)} net targets only")
    else:
        targets_to_run = mfe_targets

    # Walk-forward evaluation
    results = []
    total   = len(targets_to_run)
    for idx, (tgt_name, tgt_array) in enumerate(targets_to_run.items()):
        logger.info(f"\n{'='*65}")
        logger.info(f"[{idx+1}/{total}] Evaluating: {tgt_name}")
        logger.info(f"{'='*65}")

        t0 = time.time()
        result = walk_forward_evaluate_mfe(
            scanner=scanner,
            target=tgt_array,
            target_name=tgt_name,
            exclude_features=EXCLUDE_FEATURES,
            min_train_days=args.min_train_days,
        )
        elapsed = time.time() - t0
        result['elapsed_sec'] = elapsed
        results.append(result)

        if 'error' in result:
            logger.info(f"  ERROR: {result['error']}")
        else:
            status = "ALPHA" if result['passed'] else "---"
            logger.info(
                f"  [{status}] IC={result['ic']:.4f} ICIR={result['icir']:.2f} "
                f"t={result['tstat']:.2f} HR={result['hit_rate']:.1%} "
                f"FoldC={result['fold_consistency']:.0%} ({elapsed:.0f}s)"
            )
            if result.get('top_features'):
                top3 = ", ".join(f"{n}" for n, _ in result['top_features'][:3])
                logger.info(f"  Top: {top3}")
            if result.get('fold_ics'):
                ics_str = " ".join(f"{x:+.3f}" for x in result['fold_ics'])
                logger.info(f"  Fold ICs: [{ics_str}]")

        gc.collect()

    # Format and print results
    elapsed_total = time.time() - t_start
    scoreboard    = format_scoreboard(results)
    comparison    = compare_to_returns(results)

    logger.info("\n" + scoreboard)
    logger.info("\n" + comparison)

    # Save results
    timestamp   = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = RESULTS_DIR / f"mfe_scan_{timestamp}.json"

    serializable = []
    for r in results:
        sr = dict(r)
        sr['top_features'] = [(n, float(v)) for n, v in r.get('top_features', [])]
        serializable.append(sr)

    with open(result_file, 'w') as f:
        json.dump({
            'timestamp':           timestamp,
            'horizons_tested':     args.horizons,
            'excluded_features':   EXCLUDE_FEATURES,
            'stats':               stats,
            'results':             serializable,
            'scoreboard':          scoreboard,
            'comparison':          comparison,
            'elapsed_sec':         elapsed_total,
            'return_reference':    RETURN_SCAN_REFERENCE,
        }, f, indent=2, default=str)

    logger.info(f"\nResults saved to: {result_file}")
    logger.info(f"Total elapsed: {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")

    print("\n" + "=" * 75)
    print(scoreboard)
    print(comparison)
    print("=" * 75)

    # Discord summary (printed for the bridge to pick up)
    discord_msg = format_discord_summary(results, comparison, elapsed_total)
    discord_full = discord_msg + f"\n\nResults: `{result_file.name}`"
    print("\n--- DISCORD SUMMARY ---")
    print(discord_full)

    return results, scoreboard


if __name__ == '__main__':
    main()
