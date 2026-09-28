"""
XGBoost Walk-Forward Prediction
================================
Alternative to Ridge regression and LightGBM — captures non-linear feature
interactions that linear models miss.

Four model variants:
  A) XGBoost Standard        — regression on raw forward return
  B) XGBoost Asymmetric      — custom loss penalises wrong-direction predictions 2x
  C) XGBoost Quantile        — 3 models (p25/p50/p75); only trades when IQR > cost
  D) LightGBM Comparison     — same walk-forward for direct head-to-head

Walk-forward protocol:
  - Sort days chronologically
  - For each test day: train on all prior days (min 10), skip 1-day purge gap
  - Downsample training to 500K rows if larger
  - Predict all bars on the test day
  - Save as data/processed/signal_predictions/xgb_{horizon}s_YYYY-MM-DD.npz

Usage:
    python alpha_discovery/xgboost_walkforward.py
    python alpha_discovery/xgboost_walkforward.py --horizon 10 --model xgb
    python alpha_discovery/xgboost_walkforward.py --horizon all --model all --n-days 30
    python alpha_discovery/xgboost_walkforward.py --quick --model xgb --horizon 10
"""

import sys
import json
import time
import argparse
import warnings
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Platform-aware MBO features cache directory
if sys.platform == 'win32':
    MBO_FEAT_DIR = Path(r'C:\Users\Footb\Documents\Github\Lvl3Quant\data\processed\mbo_features_cache')
else:
    MBO_FEAT_DIR = Path.home() / 'lvl3quant' / 'data' / 'processed' / 'mbo_features_cache'

SNAP_DIR = ROOT / 'data' / 'processed' / 'medium_snapshots_cache'
SIGNAL_DIR = ROOT / 'data' / 'processed' / 'signal_predictions'
RESULTS_DIR = ROOT / 'alpha_discovery' / 'results'
SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# ES constants
# ---------------------------------------------------------------------------
TICK_SIZE = 0.25
TICK_VALUE = 12.50
BARS_PER_SEC = 10           # 100ms bars

# ---------------------------------------------------------------------------
# Horizon definitions (in 100ms bars)
# ---------------------------------------------------------------------------
HORIZONS = {
    '3s':  30,
    '10s': 100,
    '30s': 300,
}

# ---------------------------------------------------------------------------
# Leakage-excluded feature indices
# (mid, spread, best_bid, best_ask, microprice, hour_norm, minute_norm,
#  time_since_rth, time_to_close)
# ---------------------------------------------------------------------------
LEAKAGE_INDICES = [0, 1, 8, 9, 3, 26, 27, 28, 29]

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
XGB_AVAILABLE = False
LGB_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
    print(f"XGBoost {xgb.__version__} available")
except ImportError:
    print("WARNING: xgboost not installed. Run: pip install xgboost")
    print("         XGBoost models will be skipped; LightGBM will still run.")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
    print(f"LightGBM {lgb.__version__} available")
except ImportError:
    print("WARNING: lightgbm not installed. Run: pip install lightgbm")


# ---------------------------------------------------------------------------
# Feature names
# ---------------------------------------------------------------------------
def get_feature_names_safe():
    """Load feature names, fall back to generic names on error."""
    try:
        from alpha_discovery.mbo_features import get_feature_names
        return get_feature_names()
    except Exception as e:
        print(f"  Warning: could not load feature names ({e}). Using generic names.")
        return [f'feat_{i}' for i in range(340)]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_mbo_day(date_str: str):
    """
    Load MBO features (N, 340) and mid prices (N,) for one day.
    Returns (features, mid_prices) or (None, None) if missing.
    """
    f = MBO_FEAT_DIR / f'{date_str}_mbo_features.npz'
    if not f.exists():
        # Try alternate naming conventions
        for pattern in [f'{date_str}_features.npz', f'{date_str}.npz']:
            alt = MBO_FEAT_DIR / pattern
            if alt.exists():
                f = alt
                break
        else:
            candidates = list(MBO_FEAT_DIR.glob(f'*{date_str}*.npz'))
            if not candidates:
                return None, None
            f = candidates[0]

    try:
        data = np.load(str(f), allow_pickle=False)
        feats = data.get('mbo_features', data.get('features'))
        if feats is None:
            return None, None
        feats = feats.astype(np.float32)
    except Exception as e:
        print(f"  Warning: failed to load {f}: {e}")
        return None, None

    # Load mid prices — prefer snapshot cache (more reliable)
    mid = None
    sf = SNAP_DIR / f'{date_str}_snapshots.npz'
    if sf.exists():
        try:
            mid = np.load(str(sf), allow_pickle=False)['mid_prices'].astype(np.float64)
        except Exception:
            pass

    if mid is None:
        # Fall back to col 0 of MBO features (mid price column)
        try:
            mid = feats[:, 0].astype(np.float64)
        except Exception:
            return None, None

    # Align lengths
    n = min(len(feats), len(mid))
    return feats[:n], mid[:n]


def compute_forward_returns(mid_prices: np.ndarray, horizon_bars: int) -> np.ndarray:
    """Forward price change in index points at given bar horizon."""
    n = len(mid_prices)
    ret = np.full(n, np.nan)
    ret[:n - horizon_bars] = mid_prices[horizon_bars:] - mid_prices[:n - horizon_bars]
    return ret


def build_feature_matrix(feats: np.ndarray) -> np.ndarray:
    """Remove leakage columns, return clean feature matrix."""
    all_cols = np.arange(feats.shape[1])
    keep_cols = np.array([c for c in all_cols if c not in LEAKAGE_INDICES])
    return feats[:, keep_cols]


# ---------------------------------------------------------------------------
# IC helpers
# ---------------------------------------------------------------------------
def pearson_ic(preds: np.ndarray, actuals: np.ndarray) -> float:
    mask = np.isfinite(preds) & np.isfinite(actuals)
    if mask.sum() < 50:
        return 0.0
    p, r = preds[mask], actuals[mask]
    p -= p.mean(); r -= r.mean()
    denom = np.sqrt((p**2).sum() * (r**2).sum())
    return float((p * r).sum() / max(denom, 1e-12))


def rank_ic(preds: np.ndarray, actuals: np.ndarray) -> float:
    mask = np.isfinite(preds) & np.isfinite(actuals)
    if mask.sum() < 50:
        return 0.0
    corr, _ = spearmanr(preds[mask], actuals[mask])
    return float(corr) if np.isfinite(corr) else 0.0


def ic_summary(ics: list) -> dict:
    arr = np.array(ics, dtype=float)
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    tstat = mean / max(std / np.sqrt(n), 1e-9) if n > 1 else 0.0
    return {
        'n_days': n,
        'mean_ic': round(mean, 6),
        'std_ic': round(std, 6),
        't_stat': round(tstat, 3),
        'positive_pct': round(float((arr > 0).mean()), 4),
        'min_ic': round(float(arr.min()), 6) if n > 0 else 0.0,
        'max_ic': round(float(arr.max()), 6) if n > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Custom XGBoost objective: asymmetric loss (wrong-direction penalty = 2x)
# ---------------------------------------------------------------------------
def asymmetric_obj(y_pred: np.ndarray, dtrain) -> tuple:
    """
    Custom gradient/hessian for XGBoost.
    When sign(pred) != sign(actual), multiply squared error by 2.
    """
    y_true = dtrain.get_label()
    residual = y_pred - y_true
    wrong_dir = (np.sign(y_pred) != np.sign(y_true)).astype(np.float32)
    weight = 1.0 + wrong_dir  # 1.0 if correct direction, 2.0 if wrong
    grad = weight * residual
    hess = weight * np.ones_like(residual)
    return grad, hess


# ---------------------------------------------------------------------------
# Model A: XGBoost Standard
# ---------------------------------------------------------------------------
def build_xgb_standard(quick: bool = False) -> 'xgb.XGBRegressor':
    n_est = 100 if quick else 500
    return xgb.XGBRegressor(
        n_estimators=n_est,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.6,
        min_child_weight=100,
        reg_alpha=0.1,
        reg_lambda=1.0,
        tree_method='hist',
        n_jobs=-1,
        verbosity=0,
    )


# ---------------------------------------------------------------------------
# Model B: XGBoost Asymmetric (custom objective via native API)
# ---------------------------------------------------------------------------
def train_xgb_asymmetric(X_train: np.ndarray, y_train: np.ndarray,
                          quick: bool = False) -> 'xgb.Booster':
    n_est = 100 if quick else 500
    dtrain = xgb.DMatrix(X_train, label=y_train)
    params = {
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.6,
        'min_child_weight': 100,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'tree_method': 'hist',
        'nthread': -1,
        'verbosity': 0,
    }
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=n_est,
        obj=asymmetric_obj,
        verbose_eval=False,
    )
    return booster


# ---------------------------------------------------------------------------
# Model C: XGBoost Quantile Regression
# ---------------------------------------------------------------------------
def build_xgb_quantile(alpha: float, quick: bool = False) -> 'xgb.XGBRegressor':
    """Build an XGBoost quantile regressor for a given alpha (0-1)."""
    n_est = 100 if quick else 500
    return xgb.XGBRegressor(
        n_estimators=n_est,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.6,
        min_child_weight=100,
        reg_alpha=0.1,
        reg_lambda=1.0,
        tree_method='hist',
        n_jobs=-1,
        verbosity=0,
        objective='reg:quantileerror',
        quantile_alpha=alpha,
    )


# ---------------------------------------------------------------------------
# Model D: LightGBM
# ---------------------------------------------------------------------------
def build_lgb_model(quick: bool = False) -> 'lgb.LGBMRegressor':
    n_est = 100 if quick else 500
    return lgb.LGBMRegressor(
        n_estimators=n_est,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.6,
        min_child_samples=100,
        n_jobs=-1,
        verbose=-1,
    )


# ---------------------------------------------------------------------------
# Core walk-forward runner
# ---------------------------------------------------------------------------
def run_walkforward(
    dates: list,
    horizon_name: str,
    model_name: str,
    feat_names: list,
    min_train_days: int = 10,
    quick: bool = False,
    max_train_rows: int = 500_000,
) -> dict:
    """
    Walk-forward training and evaluation for one (model, horizon) pair.

    Returns a result dict with per-day ICs, feature importances, and
    paths to saved prediction NPZ files.
    """
    horizon_bars = HORIZONS[horizon_name]
    horizon_sec = int(horizon_name.replace('s', ''))
    print(f"\n{'='*60}")
    print(f"MODEL: {model_name.upper()}  |  HORIZON: {horizon_name}  |  "
          f"{'QUICK' if quick else 'FULL'}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 1. Load all days
    # ------------------------------------------------------------------
    print(f"Loading {len(dates)} days of MBO features...")
    all_features = []   # list of (N_i, F) float32
    all_returns = []    # list of (N_i,) float64
    all_mids = []       # list of (N_i,) float64
    valid_dates = []

    for d in dates:
        feats, mid = load_mbo_day(d)
        if feats is None or mid is None:
            continue
        # Remove leakage features
        X = build_feature_matrix(feats)
        # Replace inf/nan with 0
        X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
        fwd_ret = compute_forward_returns(mid, horizon_bars)
        all_features.append(X)
        all_returns.append(fwd_ret)
        all_mids.append(mid)
        valid_dates.append(d)

    n_days = len(valid_dates)
    if n_days == 0:
        print("  ERROR: No valid days found.")
        return {'error': 'no_data'}

    n_features = all_features[0].shape[1]
    print(f"Loaded {n_days} days | features: {n_features} (excl. {len(LEAKAGE_INDICES)} leakage cols)")

    if n_days <= min_train_days:
        print(f"  ERROR: Need >{min_train_days} days, only have {n_days}")
        return {'error': 'insufficient_days'}

    # Build clean feature name list (remove leakage columns)
    all_col_idx = np.arange(len(feat_names) if len(feat_names) <= 340 else 340)
    keep_col_idx = [c for c in all_col_idx if c not in LEAKAGE_INDICES]
    clean_feat_names = [feat_names[c] if c < len(feat_names) else f'feat_{c}'
                        for c in keep_col_idx]

    # ------------------------------------------------------------------
    # 2. Walk-forward loop
    # ------------------------------------------------------------------
    ics_pearson = {}
    ics_spearman = {}
    pred_files = []
    feat_importance_accum = np.zeros(n_features, dtype=np.float64)
    feat_importance_count = 0
    all_test_preds = {}   # date -> predictions (for cross-model correlation)

    for test_idx in range(min_train_days, n_days):
        test_date = valid_dates[test_idx]
        # Purge gap: skip day immediately before test (test_idx - 1 is purge)
        train_end = test_idx - 1   # exclusive (1-day purge)
        train_start = 0

        if train_end - train_start < min_train_days:
            continue

        # Build training set
        X_parts, y_parts = [], []
        for i in range(train_start, train_end):
            X_i = all_features[i]
            y_i = all_returns[i]
            mask = np.isfinite(y_i)
            X_parts.append(X_i[mask])
            y_parts.append(y_i[mask].astype(np.float32))

        X_train = np.vstack(X_parts)
        y_train = np.concatenate(y_parts)

        # Downsample
        if len(y_train) > max_train_rows:
            rng = np.random.RandomState(42 + test_idx)
            idx = rng.choice(len(y_train), max_train_rows, replace=False)
            X_train = X_train[idx]
            y_train = y_train[idx]

        # Test data
        X_test = all_features[test_idx]
        y_test = all_returns[test_idx]

        # ------------------------------------------------------------------
        # Train + predict based on model variant
        # ------------------------------------------------------------------
        preds = None
        importance = None

        try:
            if model_name == 'xgb':
                model = build_xgb_standard(quick=quick)
                model.fit(X_train, y_train)
                preds = model.predict(X_test).astype(np.float64)
                importance = model.feature_importances_

            elif model_name == 'xgb_asymmetric':
                booster = train_xgb_asymmetric(X_train, y_train, quick=quick)
                dtest = xgb.DMatrix(X_test)
                preds = booster.predict(dtest).astype(np.float64)
                # Feature importance from booster
                scores = booster.get_score(importance_type='gain')
                importance = np.zeros(n_features)
                for k, v in scores.items():
                    idx_str = k.replace('f', '')
                    if idx_str.isdigit():
                        importance[int(idx_str)] = v

            elif model_name == 'xgb_quantile':
                # Train three quantile models
                cost_threshold = 1.24 * TICK_SIZE  # ~$1.55 in index points

                m_p25 = build_xgb_quantile(0.25, quick=quick)
                m_p50 = build_xgb_quantile(0.50, quick=quick)
                m_p75 = build_xgb_quantile(0.75, quick=quick)

                m_p25.fit(X_train, y_train)
                m_p50.fit(X_train, y_train)
                m_p75.fit(X_train, y_train)

                pred_p25 = m_p25.predict(X_test)
                pred_p50 = m_p50.predict(X_test)
                pred_p75 = m_p75.predict(X_test)

                iqr = pred_p75 - pred_p25
                # Only predict where IQR > cost AND median is directional
                confident = (iqr > cost_threshold) & (np.abs(pred_p50) > cost_threshold / 2)
                preds = np.where(confident, pred_p50, 0.0).astype(np.float64)

                importance = m_p50.feature_importances_

            elif model_name == 'lgb':
                model = build_lgb_model(quick=quick)
                model.fit(X_train, y_train)
                preds = model.predict(X_test).astype(np.float64)
                importance = model.feature_importances_

        except Exception as e:
            print(f"  WARN [{test_date}]: training failed — {e}")
            continue

        if preds is None:
            continue

        # ------------------------------------------------------------------
        # IC computation
        # ------------------------------------------------------------------
        ic_p = pearson_ic(preds, y_test)
        ic_r = rank_ic(preds, y_test)
        ics_pearson[test_date] = ic_p
        ics_spearman[test_date] = ic_r
        all_test_preds[test_date] = preds

        # Accumulate feature importances
        if importance is not None and len(importance) == n_features:
            feat_importance_accum += importance.astype(np.float64)
            feat_importance_count += 1

        # ------------------------------------------------------------------
        # Save prediction NPZ
        # ------------------------------------------------------------------
        out_name = SIGNAL_DIR / f'xgb_{model_name}_{horizon_sec}s_{test_date}.npz'
        np.savez_compressed(
            str(out_name),
            predictions=preds,
            mid_prices=all_mids[test_idx],
        )
        pred_files.append(str(out_name))

        # Progress log every 5 days
        if (test_idx - min_train_days) % 5 == 0:
            recent = [ics_pearson[valid_dates[j]]
                      for j in range(max(min_train_days, test_idx - 4), test_idx + 1)
                      if valid_dates[j] in ics_pearson]
            print(f"  [{test_idx - min_train_days + 1:3d}/{n_days - min_train_days}] "
                  f"{test_date}  IC(P)={ic_p:+.4f}  IC(R)={ic_r:+.4f}  "
                  f"(5d avg: {np.mean(recent):+.4f})")

    # ------------------------------------------------------------------
    # 3. Summary
    # ------------------------------------------------------------------
    test_dates = sorted(ics_pearson.keys())
    ics_p_list = [ics_pearson[d] for d in test_dates]
    ics_r_list = [ics_spearman[d] for d in test_dates]

    summary_p = ic_summary(ics_p_list)
    summary_r = ic_summary(ics_r_list)

    print(f"\n  Pearson IC  : mean={summary_p['mean_ic']:+.4f}  std={summary_p['std_ic']:.4f}  "
          f"t={summary_p['t_stat']:.2f}  pos={summary_p['positive_pct']:.0%}")
    print(f"  Rank IC     : mean={summary_r['mean_ic']:+.4f}  std={summary_r['std_ic']:.4f}  "
          f"t={summary_r['t_stat']:.2f}  pos={summary_r['positive_pct']:.0%}")

    # Feature importances
    top_features = []
    if feat_importance_count > 0:
        avg_importance = feat_importance_accum / feat_importance_count
        top_idx = np.argsort(avg_importance)[::-1][:20]
        top_features = [
            {'rank': int(i + 1),
             'name': clean_feat_names[idx] if idx < len(clean_feat_names) else f'feat_{idx}',
             'importance': round(float(avg_importance[idx]), 4)}
            for i, idx in enumerate(top_idx)
        ]
        print(f"\n  Top 10 features by gain:")
        for ft in top_features[:10]:
            print(f"    {ft['rank']:2d}. {ft['name']:<40s}  {ft['importance']:.4f}")

    return {
        'model': model_name,
        'horizon': horizon_name,
        'horizon_sec': horizon_sec,
        'n_test_days': len(test_dates),
        'test_dates': test_dates,
        'ics_pearson': {d: round(v, 6) for d, v in ics_pearson.items()},
        'ics_spearman': {d: round(v, 6) for d, v in ics_spearman.items()},
        'pearson_summary': summary_p,
        'rank_ic_summary': summary_r,
        'top_features': top_features,
        'prediction_files': pred_files,
        '_test_preds': all_test_preds,   # kept in memory for cross-model correlation
    }


# ---------------------------------------------------------------------------
# Cross-model correlation analysis
# ---------------------------------------------------------------------------
def compute_cross_model_correlation(results: list) -> dict:
    """
    Compute pairwise prediction correlations across models for shared dates.
    Returns a correlation matrix dict.
    """
    if len(results) < 2:
        return {}

    # Collect all test dates common to all models
    all_date_sets = [set(r['test_dates']) for r in results if '_test_preds' in r]
    if not all_date_sets:
        return {}
    common_dates = sorted(set.intersection(*all_date_sets))

    if not common_dates:
        return {}

    model_names = [r['model'] for r in results if '_test_preds' in r]
    n_models = len(model_names)
    corr_matrix = np.full((n_models, n_models), np.nan)

    preds_by_model = []
    for r in results:
        if '_test_preds' not in r:
            continue
        day_preds = []
        for d in common_dates:
            if d in r['_test_preds']:
                day_preds.append(r['_test_preds'][d])
        if day_preds:
            preds_by_model.append(np.concatenate(day_preds))
        else:
            preds_by_model.append(np.array([]))

    for i in range(n_models):
        for j in range(n_models):
            pi, pj = preds_by_model[i], preds_by_model[j]
            if len(pi) == 0 or len(pj) == 0 or len(pi) != len(pj):
                continue
            mask = np.isfinite(pi) & np.isfinite(pj)
            if mask.sum() < 100:
                continue
            corr, _ = spearmanr(pi[mask], pj[mask])
            corr_matrix[i, j] = round(float(corr), 4) if np.isfinite(corr) else np.nan

    print(f"\n{'='*60}")
    print("CROSS-MODEL PREDICTION CORRELATION (Spearman rank)")
    print(f"{'='*60}")
    header = f"{'':25s}" + "".join(f"{m:>20s}" for m in model_names)
    print(header)
    for i, mi in enumerate(model_names):
        row = f"{mi:<25s}" + "".join(
            f"{corr_matrix[i, j]:>20.4f}" if np.isfinite(corr_matrix[i, j]) else f"{'N/A':>20s}"
            for j in range(n_models)
        )
        print(row)

    return {
        'models': model_names,
        'common_dates': common_dates,
        'correlation_matrix': corr_matrix.tolist(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='XGBoost/LightGBM walk-forward prediction on 340 MBO features'
    )
    parser.add_argument('--n-days', type=int, default=0,
                        help='Max days to use (0 = all available)')
    parser.add_argument('--horizon', choices=['3', '10', '30', 'all'], default='10',
                        help='Forecast horizon in seconds (default: 10)')
    parser.add_argument('--model', choices=['xgb', 'lgb', 'quantile', 'all'], default='all',
                        help='Model to run (default: all)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: 100 estimators instead of 500')
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Resolve which horizons / models to run
    # ------------------------------------------------------------------
    if args.horizon == 'all':
        horizons = ['3s', '10s', '30s']
    else:
        horizons = [f'{args.horizon}s']

    model_map = {
        'xgb': ['xgb'],
        'lgb': ['lgb'],
        'quantile': ['xgb_quantile'],
        'all': ['xgb', 'xgb_asymmetric', 'xgb_quantile', 'lgb'],
    }
    requested_models = model_map[args.model]

    # Filter based on availability
    active_models = []
    for m in requested_models:
        if m.startswith('xgb') and not XGB_AVAILABLE:
            print(f"Skipping {m} (xgboost not installed)")
            continue
        if m == 'lgb' and not LGB_AVAILABLE:
            print(f"Skipping {m} (lightgbm not installed)")
            continue
        active_models.append(m)

    if not active_models:
        print("ERROR: No models available. Install xgboost and/or lightgbm.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Find available dates from MBO features cache
    # ------------------------------------------------------------------
    mbo_files = sorted(MBO_FEAT_DIR.glob('*_mbo_features.npz'))
    if not mbo_files:
        # Try alternate naming
        mbo_files = sorted(MBO_FEAT_DIR.glob('*_features.npz'))
    if not mbo_files:
        mbo_files = sorted(MBO_FEAT_DIR.glob('*.npz'))

    # Extract date strings
    all_dates = []
    for f in mbo_files:
        stem = f.stem
        # Handle patterns: YYYY-MM-DD_mbo_features, YYYY-MM-DD_features, YYYY-MM-DD
        for suffix in ['_mbo_features', '_features']:
            stem = stem.replace(suffix, '')
        # Check it looks like a date
        parts = stem.split('-')
        if len(parts) == 3 and all(p.isdigit() for p in parts):
            all_dates.append(stem)

    all_dates = sorted(set(all_dates))

    if not all_dates:
        print(f"ERROR: No MBO feature files found in {MBO_FEAT_DIR}")
        print("       Check that the path is correct and files follow YYYY-MM-DD_mbo_features.npz naming.")
        sys.exit(1)

    if args.n_days > 0:
        all_dates = all_dates[:args.n_days]

    print(f"\nDates available: {len(all_dates)}  ({all_dates[0]} ... {all_dates[-1]})")
    print(f"Models: {active_models}")
    print(f"Horizons: {horizons}")
    print(f"Quick mode: {args.quick}")

    # ------------------------------------------------------------------
    # Load feature names once
    # ------------------------------------------------------------------
    feat_names = get_feature_names_safe()

    # ------------------------------------------------------------------
    # Run experiments
    # ------------------------------------------------------------------
    all_results = []
    ts = time.strftime('%Y%m%d_%H%M%S')

    for horizon_name in horizons:
        horizon_results = []
        for model_name in active_models:
            result = run_walkforward(
                dates=all_dates,
                horizon_name=horizon_name,
                model_name=model_name,
                feat_names=feat_names,
                quick=args.quick,
            )
            if 'error' not in result:
                horizon_results.append(result)
                all_results.append(result)

        # Cross-model correlation for this horizon
        if len(horizon_results) >= 2:
            corr = compute_cross_model_correlation(horizon_results)
            for r in horizon_results:
                r['cross_model_correlation'] = corr

    # ------------------------------------------------------------------
    # Final comparison table
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("FINAL COMPARISON TABLE")
    print(f"{'='*70}")
    header = f"{'Model':<22s}  {'Horizon':>8s}  {'MeanIC':>8s}  {'Std':>7s}  "
    header += f"{'t-stat':>7s}  {'Pos%':>6s}  {'Min':>8s}  {'Max':>8s}"
    print(header)
    print('-' * len(header))

    for r in all_results:
        sp = r.get('pearson_summary', {})
        print(f"{r['model']:<22s}  {r['horizon']:>8s}  "
              f"{sp.get('mean_ic', 0):>+8.4f}  {sp.get('std_ic', 0):>7.4f}  "
              f"{sp.get('t_stat', 0):>7.2f}  {sp.get('positive_pct', 0):>6.0%}  "
              f"{sp.get('min_ic', 0):>+8.4f}  {sp.get('max_ic', 0):>+8.4f}")

    print(f"\nReference (raw signals, test_raw_signals.py):")
    print(f"  microprice_dev      @1s  IC=0.195, 92% consistency, t-stat=8.57")
    print(f"  pressure_imbalance  @1s  IC=0.135, 86% consistency")
    print(f"  ask_slope           @1s  IC=0.084, 96% consistency")
    print(f"  LightGBM (OOS)      @10s  IC<0  — confirmed dead (OOD predictions)")

    # ------------------------------------------------------------------
    # Save JSON results (strip in-memory pred arrays before serialising)
    # ------------------------------------------------------------------
    serialisable = []
    for r in all_results:
        rc = {k: v for k, v in r.items() if k != '_test_preds'}
        serialisable.append(rc)

    out_file = RESULTS_DIR / f'xgboost_walkforward_{ts}.json'
    output = {
        'experiment': 'xgboost_walkforward',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'config': {
            'n_days_requested': args.n_days,
            'n_days_actual': len(all_dates),
            'horizons': horizons,
            'models': active_models,
            'quick': args.quick,
            'leakage_indices': LEAKAGE_INDICES,
            'n_features_total': 340,
            'n_features_after_exclusion': 340 - len(LEAKAGE_INDICES),
        },
        'results': serialisable,
        'es_constants': {
            'tick_size': TICK_SIZE,
            'tick_value': TICK_VALUE,
            'bars_per_sec': BARS_PER_SEC,
        },
    }

    with open(out_file, 'w') as fh:
        json.dump(output, fh, indent=2, default=str)
    print(f"\nSaved results: {out_file}")
    print(f"Prediction NPZ files: {SIGNAL_DIR}/xgb_*_{ts[:8]}*.npz")


if __name__ == '__main__':
    main()
