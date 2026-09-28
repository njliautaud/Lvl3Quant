"""
Lead-Lag and Cross-Feature Analysis — Lvl3Quant MBO Alpha Project
==================================================================

Discovers which features predict future price movements AND which features
predict EACH OTHER (leading indicators). Reveals the causal chain of the
order book.

Analyses:
  A) Raw Feature IC Heatmap — all 340 features vs 4 forward horizons
  B) Feature Lead-Lag Matrix — which features predict other features
  C) Granger Causality (simplified) — beyond lagged returns alone
  D) Temporal Decay Analysis — IC decay curves per feature
  E) Feature Interaction Discovery — synergistic A*B interactions
  F) Cross-Day Stability — which features are regime-stable vs regime-dependent

Usage:
    python alpha_discovery/lead_lag_analysis.py
    python alpha_discovery/lead_lag_analysis.py --n-days 20 --top-k 30
    python alpha_discovery/lead_lag_analysis.py --quick --n-days 10
"""

import sys
import gc
import json
import time
import argparse
import warnings
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

warnings.filterwarnings('ignore', category=RuntimeWarning)

# ============================================================================
# PATH SETUP
# ============================================================================

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Detect platform and set MBO cache path
if sys.platform == 'win32':
    MBO_DIR = ROOT / 'data' / 'processed' / 'mbo_features_cache'
else:
    MBO_DIR = Path.home() / 'lvl3quant' / 'data' / 'processed' / 'mbo_features_cache'

RESULTS_DIR = ROOT / 'alpha_discovery' / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Forward return horizons (in 100ms bars)
HORIZONS = {
    '1s':  10,
    '3s':  30,
    '10s': 100,
    '30s': 300,
}

# Extended horizons for temporal decay analysis (section D)
DECAY_HORIZONS = {
    '0.5s': 5,
    '1s':   10,
    '2s':   20,
    '3s':   30,
    '5s':   50,
    '10s':  100,
    '20s':  200,
    '30s':  300,
    '60s':  600,
}

# Feature columns to exclude (leakage set: raw prices, timestamps, and direct mid derivatives)
LEAKAGE_COLS = {0, 1, 8, 9, 3, 26, 27, 28, 29}
# col 0=mid, 1=spread, 3=microprice (contains mid), 8=best_bid, 9=best_ask,
# 26=hour_norm, 27=minute_norm, 28=time_since_rth, 29=time_to_close

# Lead-lag evaluation lags (in bars, negative = feature leads)
LEADLAG_LAGS = [-50, -20, -10, -5, 0, 5, 10, 20, 50]


# ============================================================================
# FEATURE LOADING
# ============================================================================

def get_feature_names_safe() -> List[str]:
    """Load feature names from mbo_features module."""
    try:
        from alpha_discovery.mbo_features import get_feature_names
        names = get_feature_names()
        return names
    except ImportError as e:
        print(f"WARNING: Could not import get_feature_names: {e}")
        return [f'feat_{i}' for i in range(340)]


def load_day_npz(npz_path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Load one day's MBO features NPZ.

    Returns:
        (features, mid_prices) — features shape (N, 340), mid_prices shape (N,)
        None if file is invalid.
    """
    try:
        data = np.load(str(npz_path), allow_pickle=False)
        feats = data['mbo_features'].astype(np.float32)
        # Mid prices are column 0 of the feature matrix
        mid = feats[:, 0].copy().astype(np.float64)
        return feats, mid
    except Exception as e:
        print(f"  WARNING: Failed to load {npz_path.name}: {e}")
        return None


def compute_forward_returns(mid_prices: np.ndarray, horizons: Dict[str, int]) -> Dict[str, np.ndarray]:
    """Compute forward returns at each horizon. NaN-fills at end."""
    n = len(mid_prices)
    fwd = {}
    for hz_name, hz_bars in horizons.items():
        ret = np.full(n, np.nan, dtype=np.float32)
        if hz_bars < n:
            # Simple arithmetic return: (p_{t+h} - p_t) / p_t
            ret[:n - hz_bars] = (
                (mid_prices[hz_bars:] - mid_prices[:n - hz_bars])
                / np.maximum(np.abs(mid_prices[:n - hz_bars]), 1e-6)
            ).astype(np.float32)
        fwd[hz_name] = ret
    return fwd


# ============================================================================
# IC COMPUTATION
# ============================================================================

def pearson_ic(signal: np.ndarray, target: np.ndarray) -> float:
    """Pearson IC, handles NaNs."""
    mask = np.isfinite(signal) & np.isfinite(target)
    if mask.sum() < 50:
        return 0.0
    s = signal[mask].astype(np.float64)
    r = target[mask].astype(np.float64)
    s -= s.mean()
    r -= r.mean()
    denom = np.sqrt((s * s).sum() * (r * r).sum())
    if denom < 1e-14:
        return 0.0
    return float((s * r).sum() / denom)


# ============================================================================
# ANALYSIS A: RAW FEATURE IC HEATMAP
# ============================================================================

def analysis_a_ic_heatmap(
    mbo_files: List[Path],
    feature_names: List[str],
    n_features: int,
    top_k: int = 20,
) -> Dict:
    """
    Per-day IC for every non-leakage feature vs 4 forward horizons.
    Aggregates across days to produce mean IC matrix.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS A: Raw Feature IC Heatmap (all 340 features)")
    print("=" * 70)

    hz_names = list(HORIZONS.keys())

    # Accumulate: ic_sum[feat_idx][hz] = list of per-day ICs
    ic_by_feat_hz: Dict[int, Dict[str, List[float]]] = {
        i: {hz: [] for hz in hz_names}
        for i in range(n_features)
        if i not in LEAKAGE_COLS
    }

    for day_idx, fpath in enumerate(mbo_files):
        result = load_day_npz(fpath)
        if result is None:
            continue
        feats, mid = result
        n = len(mid)
        if n < 500:
            continue

        fwd = compute_forward_returns(mid, HORIZONS)

        for feat_idx in ic_by_feat_hz:
            if feat_idx >= feats.shape[1]:
                continue
            signal = feats[:, feat_idx].astype(np.float64)
            for hz_name in hz_names:
                ic = pearson_ic(signal, fwd[hz_name].astype(np.float64))
                ic_by_feat_hz[feat_idx][hz_name].append(ic)

        if (day_idx + 1) % 5 == 0:
            print(f"  Processed {day_idx + 1}/{len(mbo_files)} days...")

        del feats, mid, fwd
        gc.collect()

    # Build mean IC matrix
    feat_indices = sorted(ic_by_feat_hz.keys())
    mean_ic = {}   # feat_idx -> {hz_name: mean_ic}
    std_ic = {}

    for feat_idx in feat_indices:
        mean_ic[feat_idx] = {}
        std_ic[feat_idx] = {}
        for hz_name in hz_names:
            ics = ic_by_feat_hz[feat_idx][hz_name]
            if ics:
                mean_ic[feat_idx][hz_name] = float(np.mean(ics))
                std_ic[feat_idx][hz_name] = float(np.std(ics))
            else:
                mean_ic[feat_idx][hz_name] = 0.0
                std_ic[feat_idx][hz_name] = 0.0

    # Print top features at each horizon
    all_results = {
        'mean_ic': {str(k): v for k, v in mean_ic.items()},
        'std_ic': {str(k): v for k, v in std_ic.items()},
        'top_per_horizon': {},
        'short_lived': [],
        'slow_burn': [],
    }

    for hz_name in hz_names:
        ranked = sorted(
            feat_indices,
            key=lambda i: abs(mean_ic[i][hz_name]),
            reverse=True,
        )[:top_k]

        print(f"\nTop {top_k} features by |IC| at {hz_name} horizon:")
        print(f"  {'Rank':>4s}  {'Feature':<35s}  {'Mean IC':>9s}  {'Std IC':>8s}  {'Consistency':>11s}")
        print(f"  {'-'*4}  {'-'*35}  {'-'*9}  {'-'*8}  {'-'*11}")

        top_list = []
        for rank, feat_idx in enumerate(ranked):
            fname = feature_names[feat_idx] if feat_idx < len(feature_names) else f'feat_{feat_idx}'
            mic = mean_ic[feat_idx][hz_name]
            sic = std_ic[feat_idx][hz_name]
            # Consistency: fraction of days with same sign as mean
            day_ics = ic_by_feat_hz[feat_idx][hz_name]
            if day_ics:
                sign = 1 if mic >= 0 else -1
                consistency = float(np.mean([1 if x * sign > 0 else 0 for x in day_ics]))
            else:
                consistency = 0.0
            print(f"  {rank+1:>4d}  {fname:<35s}  {mic:>+9.4f}  {sic:>8.4f}  {consistency:>10.1%}")
            top_list.append({
                'rank': rank + 1,
                'feat_idx': feat_idx,
                'feat_name': fname,
                'mean_ic': mic,
                'std_ic': sic,
                'consistency': consistency,
            })

        all_results['top_per_horizon'][hz_name] = top_list

    # Flag short-lived signals: strong IC at 1s, decays to <50% magnitude by 10s
    short_lived = []
    slow_burn = []

    for feat_idx in feat_indices:
        ic_1s = abs(mean_ic[feat_idx].get('1s', 0.0))
        ic_10s = abs(mean_ic[feat_idx].get('10s', 0.0))
        ic_30s = abs(mean_ic[feat_idx].get('30s', 0.0))
        fname = feature_names[feat_idx] if feat_idx < len(feature_names) else f'feat_{feat_idx}'

        if ic_1s > 0.02:
            if ic_10s < ic_1s * 0.5:
                short_lived.append({
                    'feat_idx': feat_idx,
                    'feat_name': fname,
                    'ic_1s': float(mean_ic[feat_idx]['1s']),
                    'ic_10s': float(mean_ic[feat_idx]['10s']),
                    'ic_30s': float(mean_ic[feat_idx]['30s']),
                    'decay_ratio': ic_10s / max(ic_1s, 1e-9),
                })

        if ic_10s > ic_1s * 1.2 and ic_10s > 0.01:
            slow_burn.append({
                'feat_idx': feat_idx,
                'feat_name': fname,
                'ic_1s': float(mean_ic[feat_idx]['1s']),
                'ic_10s': float(mean_ic[feat_idx]['10s']),
                'ic_30s': float(mean_ic[feat_idx]['30s']),
                'growth_ratio': ic_10s / max(ic_1s, 1e-9),
            })

    short_lived.sort(key=lambda x: abs(x['ic_1s']), reverse=True)
    slow_burn.sort(key=lambda x: x['growth_ratio'], reverse=True)

    print(f"\nSHORT-LIVED signals (IC decays >50% from 1s to 10s): {len(short_lived)}")
    for item in short_lived[:10]:
        print(f"  {item['feat_name']:<35s} 1s={item['ic_1s']:+.4f} -> 10s={item['ic_10s']:+.4f}"
              f" (decay={item['decay_ratio']:.1%})")

    print(f"\nSLOW-BURN signals (IC increases from 1s to 10s): {len(slow_burn)}")
    for item in slow_burn[:10]:
        print(f"  {item['feat_name']:<35s} 1s={item['ic_1s']:+.4f} -> 10s={item['ic_10s']:+.4f}"
              f" (growth={item['growth_ratio']:.1f}x)")

    all_results['short_lived'] = short_lived
    all_results['slow_burn'] = slow_burn

    return all_results


# ============================================================================
# ANALYSIS B: FEATURE LEAD-LAG MATRIX
# ============================================================================

def analysis_b_leadlag_matrix(
    mbo_files: List[Path],
    feature_names: List[str],
    top_feat_indices: List[int],
    top_k: int = 30,
) -> Dict:
    """
    For top-k features, compute cross-correlation at multiple lags.
    Identifies which features LEAD other features (causal chain).
    """
    print("\n" + "=" * 70)
    print("ANALYSIS B: Feature Lead-Lag Matrix")
    print("=" * 70)

    selected = top_feat_indices[:top_k]
    n_sel = len(selected)
    lags = LEADLAG_LAGS

    # Accumulate correlations across days: corr_sum[i][j][lag_idx] = list of corrs
    # corr(feat_selected[i] at t, feat_selected[j] at t+lag)
    # positive lag means j is in the future relative to i -> i LEADS j
    corr_accum = np.zeros((n_sel, n_sel, len(lags)), dtype=np.float64)
    n_days_counted = 0

    for day_idx, fpath in enumerate(mbo_files):
        result = load_day_npz(fpath)
        if result is None:
            continue
        feats, mid = result
        n = len(mid)
        if n < max(abs(l) for l in lags) * 2 + 100:
            continue

        day_corr = np.zeros((n_sel, n_sel, len(lags)), dtype=np.float64)

        # Extract selected feature vectors, demean and normalize
        feat_vecs = []
        for feat_idx in selected:
            if feat_idx < feats.shape[1]:
                v = feats[:, feat_idx].astype(np.float64)
            else:
                v = np.zeros(n)
            # Replace inf/nan with 0
            v = np.where(np.isfinite(v), v, 0.0)
            v -= v.mean()
            std = v.std()
            if std > 1e-12:
                v /= std
            feat_vecs.append(v)

        for i in range(n_sel):
            for j in range(n_sel):
                if i == j:
                    continue
                va = feat_vecs[i]
                vb = feat_vecs[j]
                for lag_k, lag in enumerate(lags):
                    # corr(A_t, B_{t+lag})
                    if lag >= 0:
                        if n - lag < 10:
                            continue
                        c = float(np.mean(va[:n - lag] * vb[lag:]))
                    else:
                        abs_lag = -lag
                        if n - abs_lag < 10:
                            continue
                        c = float(np.mean(va[abs_lag:] * vb[:n - abs_lag]))
                    day_corr[i, j, lag_k] = c

        corr_accum += day_corr
        n_days_counted += 1

        del feats, mid
        gc.collect()

    if n_days_counted > 0:
        corr_avg = corr_accum / n_days_counted
    else:
        corr_avg = corr_accum

    # Identify leading relationships
    # For pair (A, B): A leads B if corr(A_t, B_{t+k}) is maximized at k > 0
    lag0_idx = lags.index(0)
    leading_pairs = []

    for i in range(n_sel):
        for j in range(n_sel):
            if i == j:
                continue
            corr_curve = corr_avg[i, j, :]

            # Find lag at which |corr| is maximized
            max_lag_idx = int(np.argmax(np.abs(corr_curve)))
            max_lag = lags[max_lag_idx]
            max_corr = float(corr_curve[max_lag_idx])
            corr_at_zero = float(corr_curve[lag0_idx])

            # A strongly LEADS B if max is at lag > 0 and noticeably higher than lag=0
            if max_lag > 0 and abs(max_corr) > abs(corr_at_zero) * 1.3 and abs(max_corr) > 0.05:
                name_a = feature_names[selected[i]] if selected[i] < len(feature_names) else f'feat_{selected[i]}'
                name_b = feature_names[selected[j]] if selected[j] < len(feature_names) else f'feat_{selected[j]}'
                leading_pairs.append({
                    'leader': name_a,
                    'follower': name_b,
                    'leader_idx': int(selected[i]),
                    'follower_idx': int(selected[j]),
                    'optimal_lag_bars': int(max_lag),
                    'optimal_lag_sec': round(max_lag * 0.1, 1),
                    'corr_at_opt_lag': round(float(max_corr), 4),
                    'corr_at_zero': round(float(corr_at_zero), 4),
                    'lead_strength': round(abs(max_corr) / max(abs(corr_at_zero), 1e-9), 2),
                })

    leading_pairs.sort(key=lambda x: x['lead_strength'], reverse=True)

    print(f"\nTop leading relationships (A predicts B at future lag):")
    print(f"  {'Leader':<30s}  ->  {'Follower':<30s}  {'Lag(s)':>6s}  {'Corr@lag':>9s}  {'Strength':>8s}")
    print(f"  {'-'*30}      {'-'*30}  {'-'*6}  {'-'*9}  {'-'*8}")
    for pair in leading_pairs[:20]:
        print(f"  {pair['leader']:<30s}  ->  {pair['follower']:<30s}"
              f"  {pair['optimal_lag_sec']:>5.1f}s  {pair['corr_at_opt_lag']:>+9.4f}"
              f"  {pair['lead_strength']:>7.2f}x")

    # Build correlation matrix at key lags for JSON export
    corr_at_lags = {}
    for lag_k, lag in enumerate(lags):
        corr_at_lags[f'lag_{lag}'] = corr_avg[:, :, lag_k].tolist()

    sel_names = [
        feature_names[i] if i < len(feature_names) else f'feat_{i}'
        for i in selected
    ]

    return {
        'selected_features': list(selected),
        'selected_names': sel_names,
        'lags': lags,
        'corr_at_lags': {k: [[round(v, 4) for v in row] for row in mat]
                         for k, mat in corr_at_lags.items()},
        'leading_pairs': leading_pairs[:50],
        'n_days': n_days_counted,
    }


# ============================================================================
# ANALYSIS C: GRANGER CAUSALITY (SIMPLIFIED)
# ============================================================================

def analysis_c_granger(
    mbo_files: List[Path],
    feature_names: List[str],
    top_feat_indices: List[int],
    top_k: int = 20,
) -> Dict:
    """
    Simplified Granger test: does lagged feature_A improve prediction of
    forward_return beyond lagged forward_return alone?

    Model A (baseline): ret_{t+100} ~ a*ret_t
    Model B (with feature): ret_{t+100} ~ a*ret_t + b*feat_{t} + c*feat_{t-10} + d*feat_{t-50}

    Compare R² and report F-statistic.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS C: Granger Causality (simplified F-test)")
    print("=" * 70)

    selected = top_feat_indices[:top_k]
    hz_bars = HORIZONS['10s']  # Predict 10s forward return

    # Accumulate per-day results
    granger_results = {i: {'delta_r2_list': [], 'f_stat_list': []} for i in selected}

    for day_idx, fpath in enumerate(mbo_files):
        result = load_day_npz(fpath)
        if result is None:
            continue
        feats, mid = result
        n = len(mid)
        if n < 300:
            continue

        # Forward return at 10s
        fwd_ret = np.full(n, np.nan, dtype=np.float64)
        if hz_bars < n:
            fwd_ret[:n - hz_bars] = (
                (mid[hz_bars:] - mid[:n - hz_bars]) / np.maximum(np.abs(mid[:n - hz_bars]), 1e-6)
            )

        # Contemporaneous return (lag = 0, i.e., current 10-bar return looking back)
        cur_ret = np.full(n, np.nan, dtype=np.float64)
        if hz_bars < n:
            cur_ret[hz_bars:] = (mid[hz_bars:] - mid[:n - hz_bars]) / np.maximum(np.abs(mid[:n - hz_bars]), 1e-6)

        # Build common valid mask (no NaN in fwd_ret or cur_ret)
        for feat_idx in selected:
            if feat_idx >= feats.shape[1]:
                continue

            feat_t = feats[:, feat_idx].astype(np.float64)
            feat_lag10 = np.full(n, np.nan)
            feat_lag50 = np.full(n, np.nan)
            if 10 < n:
                feat_lag10[10:] = feat_t[:n - 10]
            if 50 < n:
                feat_lag50[50:] = feat_t[:n - 50]

            mask = (
                np.isfinite(fwd_ret) &
                np.isfinite(cur_ret) &
                np.isfinite(feat_t) &
                np.isfinite(feat_lag10) &
                np.isfinite(feat_lag50)
            )

            if mask.sum() < 100:
                continue

            y = fwd_ret[mask]
            x_base = cur_ret[mask]
            x_feat = feat_t[mask]
            x_feat_lag10 = feat_lag10[mask]
            x_feat_lag50 = feat_lag50[mask]

            # Standardize features
            def standardize(v):
                std = v.std()
                return (v - v.mean()) / max(std, 1e-12)

            y_s = standardize(y)
            xb_s = standardize(x_base)
            xf_s = standardize(x_feat)
            xl10_s = standardize(x_feat_lag10)
            xl50_s = standardize(x_feat_lag50)

            n_obs = len(y_s)

            # Model A: y ~ [1, ret_t]
            X_a = np.column_stack([np.ones(n_obs), xb_s])
            try:
                coef_a, res_a, _, _ = np.linalg.lstsq(X_a, y_s, rcond=None)
                ss_res_a = float(np.sum((y_s - X_a @ coef_a) ** 2))
            except Exception:
                continue

            # Model B: y ~ [1, ret_t, feat_t, feat_{t-10}, feat_{t-50}]
            X_b = np.column_stack([np.ones(n_obs), xb_s, xf_s, xl10_s, xl50_s])
            try:
                coef_b, res_b, _, _ = np.linalg.lstsq(X_b, y_s, rcond=None)
                ss_res_b = float(np.sum((y_s - X_b @ coef_b) ** 2))
            except Exception:
                continue

            ss_tot = float(np.sum((y_s - y_s.mean()) ** 2))
            if ss_tot < 1e-12:
                continue

            r2_a = 1.0 - ss_res_a / ss_tot
            r2_b = 1.0 - ss_res_b / ss_tot
            delta_r2 = r2_b - r2_a

            # F-test: H0 = the 3 extra feature coefficients are all zero
            # F = [(SS_res_A - SS_res_B) / q] / [SS_res_B / (n - k)]
            q = 3  # number of added regressors
            k = 5  # total regressors in model B
            if n_obs > k and ss_res_b > 1e-12:
                f_stat = ((ss_res_a - ss_res_b) / q) / (ss_res_b / (n_obs - k))
            else:
                f_stat = 0.0

            granger_results[feat_idx]['delta_r2_list'].append(float(delta_r2))
            granger_results[feat_idx]['f_stat_list'].append(float(f_stat))

        del feats, mid
        gc.collect()

    # Summarize
    summary = []
    for feat_idx in selected:
        dr2_list = granger_results[feat_idx]['delta_r2_list']
        fs_list = granger_results[feat_idx]['f_stat_list']
        fname = feature_names[feat_idx] if feat_idx < len(feature_names) else f'feat_{feat_idx}'

        if not dr2_list:
            continue

        mean_dr2 = float(np.mean(dr2_list))
        mean_fs = float(np.mean(fs_list))
        # Fraction of days with positive delta R2
        frac_pos = float(np.mean([1 if x > 0 else 0 for x in dr2_list]))

        summary.append({
            'feat_idx': int(feat_idx),
            'feat_name': fname,
            'mean_delta_r2': round(mean_dr2, 6),
            'mean_f_stat': round(mean_fs, 2),
            'frac_days_positive': round(frac_pos, 3),
            'n_days': len(dr2_list),
        })

    summary.sort(key=lambda x: x['mean_f_stat'], reverse=True)

    print(f"\nGranger causality results (top {top_k} features by 10s IC, F-stat ranking):")
    print(f"  {'Feature':<35s}  {'dR²':>8s}  {'F-stat':>8s}  {'Frac+':>6s}  {'Days':>5s}")
    print(f"  {'-'*35}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*5}")
    for item in summary[:20]:
        marker = " ***" if item['mean_f_stat'] > 10 else ("  **" if item['mean_f_stat'] > 5 else "")
        print(f"  {item['feat_name']:<35s}  {item['mean_delta_r2']:>+8.5f}"
              f"  {item['mean_f_stat']:>8.2f}  {item['frac_days_positive']:>6.1%}"
              f"  {item['n_days']:>5d}{marker}")

    return {
        'horizon': '10s',
        'summary': summary,
        'n_features_tested': len(summary),
    }


# ============================================================================
# ANALYSIS D: TEMPORAL DECAY ANALYSIS
# ============================================================================

def analysis_d_temporal_decay(
    mbo_files: List[Path],
    feature_names: List[str],
    top_feat_indices: List[int],
    top_k: int = 20,
) -> Dict:
    """
    IC decay curves at 9 horizons (0.5s to 60s).
    Identifies features with the SLOWEST decay (most tradeable with real fills).
    """
    print("\n" + "=" * 70)
    print("ANALYSIS D: Temporal IC Decay Analysis")
    print("=" * 70)

    selected = top_feat_indices[:top_k]
    hz_names = list(DECAY_HORIZONS.keys())

    # Accumulate IC per feature per horizon
    ic_accum: Dict[int, Dict[str, List[float]]] = {
        i: {hz: [] for hz in hz_names} for i in selected
    }

    for day_idx, fpath in enumerate(mbo_files):
        result = load_day_npz(fpath)
        if result is None:
            continue
        feats, mid = result
        n = len(mid)
        max_hz = max(DECAY_HORIZONS.values())
        if n < max_hz + 100:
            continue

        fwd = compute_forward_returns(mid, DECAY_HORIZONS)

        for feat_idx in selected:
            if feat_idx >= feats.shape[1]:
                continue
            signal = feats[:, feat_idx].astype(np.float64)
            for hz_name in hz_names:
                ic = pearson_ic(signal, fwd[hz_name].astype(np.float64))
                ic_accum[feat_idx][hz_name].append(ic)

        del feats, mid, fwd
        gc.collect()

    # Compute mean IC at each horizon
    decay_results = []
    for feat_idx in selected:
        fname = feature_names[feat_idx] if feat_idx < len(feature_names) else f'feat_{feat_idx}'
        mean_ic_curve = {}
        for hz_name in hz_names:
            ics = ic_accum[feat_idx][hz_name]
            mean_ic_curve[hz_name] = float(np.mean(ics)) if ics else 0.0

        # Compute decay speed: slope of |IC| from 1s to 30s
        ic_1s = abs(mean_ic_curve.get('1s', 0.0))
        ic_5s = abs(mean_ic_curve.get('5s', 0.0))
        ic_10s = abs(mean_ic_curve.get('10s', 0.0))
        ic_30s = abs(mean_ic_curve.get('30s', 0.0))
        ic_60s = abs(mean_ic_curve.get('60s', 0.0))

        # Decay ratio: how much IC remains at 30s relative to 1s
        peak_ic = max(abs(mean_ic_curve[hz]) for hz in hz_names)
        ic_at_1s = abs(mean_ic_curve.get('1s', 0.0))
        if peak_ic > 1e-6:
            decay_at_30s = ic_30s / peak_ic
            half_life_hz = None
            # Find horizon where IC drops to 50% of peak
            for hz_name in hz_names:
                if abs(mean_ic_curve[hz_name]) <= peak_ic * 0.5:
                    half_life_hz = hz_name
                    break
        else:
            decay_at_30s = 0.0
            half_life_hz = '0.5s'

        decay_results.append({
            'feat_idx': int(feat_idx),
            'feat_name': fname,
            'ic_curve': {hz: round(v, 5) for hz, v in mean_ic_curve.items()},
            'peak_ic': round(float(peak_ic), 5),
            'ic_at_1s': round(float(ic_at_1s), 5),
            'decay_at_30s': round(float(decay_at_30s), 3),
            'half_life_horizon': half_life_hz,
        })

    # Sort by slowest decay (largest decay_at_30s)
    decay_results.sort(key=lambda x: x['decay_at_30s'], reverse=True)

    print(f"\nFeatures with SLOWEST IC decay (most tradeable with real fills):")
    print(f"\n  {'Feature':<35s}  {'IC@1s':>7s}  {'IC@5s':>7s}  {'IC@10s':>7s}  {'IC@30s':>7s}  {'Decay@30s':>9s}")
    print(f"  {'-'*35}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*9}")
    for item in decay_results:
        c = item['ic_curve']
        print(f"  {item['feat_name']:<35s}"
              f"  {c.get('1s', 0):>+7.4f}"
              f"  {c.get('5s', 0):>+7.4f}"
              f"  {c.get('10s', 0):>+7.4f}"
              f"  {c.get('30s', 0):>+7.4f}"
              f"  {item['decay_at_30s']:>8.1%}")

    # Text-based IC decay chart for top 5 features
    print(f"\nIC Decay Curves (text plot, top 5 slowest-decay features):")
    bar_chars = " .-+*#@O"
    for item in decay_results[:5]:
        c = item['ic_curve']
        vals = [abs(c.get(hz, 0)) for hz in hz_names]
        max_val = max(vals) if max(vals) > 0 else 1.0
        bars = "".join(bar_chars[min(int(v / max_val * 7), 7)] for v in vals)
        print(f"  {item['feat_name']:<30s} [{bars}]  (peak={item['peak_ic']:+.4f})")

    print(f"  Horizons: {' -> '.join(hz_names)}")

    return {
        'decay_results': decay_results,
        'hz_names': hz_names,
        'hz_bars': {k: v for k, v in DECAY_HORIZONS.items()},
    }


# ============================================================================
# ANALYSIS E: FEATURE INTERACTION DISCOVERY
# ============================================================================

def analysis_e_interactions(
    mbo_files: List[Path],
    feature_names: List[str],
    top_feat_indices: List[int],
    top_k: int = 30,
) -> Dict:
    """
    For top-k features: compute IC of A*B interaction vs IC of A alone and B alone.
    Identifies synergistic feature combinations.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS E: Feature Interaction Discovery (A*B synergy)")
    print("=" * 70)

    selected = top_feat_indices[:top_k]
    n_sel = len(selected)
    hz_name = '10s'
    hz_bars = HORIZONS[hz_name]

    # Accumulate per-day ICs: single features and interactions
    single_ic: Dict[int, List[float]] = {i: [] for i in selected}
    interaction_ic: Dict[Tuple[int, int], List[float]] = {}

    for day_idx, fpath in enumerate(mbo_files):
        result = load_day_npz(fpath)
        if result is None:
            continue
        feats, mid = result
        n = len(mid)
        if n < hz_bars + 100:
            continue

        fwd_ret = np.full(n, np.nan, dtype=np.float64)
        fwd_ret[:n - hz_bars] = (
            (mid[hz_bars:] - mid[:n - hz_bars]) / np.maximum(np.abs(mid[:n - hz_bars]), 1e-6)
        )

        # Normalize each selected feature
        feat_vecs_norm = {}
        for feat_idx in selected:
            if feat_idx >= feats.shape[1]:
                feat_vecs_norm[feat_idx] = np.zeros(n)
                continue
            v = feats[:, feat_idx].astype(np.float64)
            v = np.where(np.isfinite(v), v, 0.0)
            v -= v.mean()
            std = v.std()
            if std > 1e-12:
                v /= std
            feat_vecs_norm[feat_idx] = v

        # Single feature ICs
        for feat_idx in selected:
            ic = pearson_ic(feat_vecs_norm[feat_idx], fwd_ret)
            single_ic[feat_idx].append(ic)

        # Pairwise interaction ICs
        for ii in range(n_sel):
            for jj in range(ii + 1, n_sel):
                fa_idx = selected[ii]
                fb_idx = selected[jj]
                interaction = feat_vecs_norm[fa_idx] * feat_vecs_norm[fb_idx]
                ic = pearson_ic(interaction, fwd_ret)
                key = (fa_idx, fb_idx)
                if key not in interaction_ic:
                    interaction_ic[key] = []
                interaction_ic[key].append(ic)

        del feats, mid
        gc.collect()

    # Compute mean ICs
    mean_single = {i: float(np.mean(v)) if v else 0.0 for i, v in single_ic.items()}

    interaction_results = []
    for (fa_idx, fb_idx), ics in interaction_ic.items():
        mean_int_ic = float(np.mean(ics)) if ics else 0.0
        ic_a = mean_single.get(fa_idx, 0.0)
        ic_b = mean_single.get(fb_idx, 0.0)
        max_single = max(abs(ic_a), abs(ic_b))

        name_a = feature_names[fa_idx] if fa_idx < len(feature_names) else f'feat_{fa_idx}'
        name_b = feature_names[fb_idx] if fb_idx < len(feature_names) else f'feat_{fb_idx}'

        # Synergy: interaction IC > both single ICs
        synergy = abs(mean_int_ic) / max(max_single, 1e-9)

        interaction_results.append({
            'feat_a': name_a,
            'feat_b': name_b,
            'feat_a_idx': int(fa_idx),
            'feat_b_idx': int(fb_idx),
            'ic_a': round(float(ic_a), 5),
            'ic_b': round(float(ic_b), 5),
            'ic_interaction': round(float(mean_int_ic), 5),
            'synergy_ratio': round(float(synergy), 3),
            'is_synergistic': abs(mean_int_ic) > max_single * 1.2,
            'n_days': len(ics),
        })

    interaction_results.sort(key=lambda x: x['synergy_ratio'], reverse=True)

    print(f"\nTop synergistic interactions (IC(A*B) >> max(IC(A), IC(B))):")
    print(f"  {'Feature A':<30s}  x  {'Feature B':<30s}  {'IC(A)':>7s}  {'IC(B)':>7s}  {'IC(A*B)':>8s}  {'Synergy':>7s}")
    print(f"  {'-'*30}     {'-'*30}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*7}")
    for item in interaction_results[:20]:
        marker = " ***" if item['is_synergistic'] else ""
        print(f"  {item['feat_a']:<30s}  x  {item['feat_b']:<30s}"
              f"  {item['ic_a']:>+7.4f}  {item['ic_b']:>+7.4f}"
              f"  {item['ic_interaction']:>+8.4f}  {item['synergy_ratio']:>6.2f}x{marker}")

    synergistic = [x for x in interaction_results if x['is_synergistic']]
    print(f"\nTotal synergistic pairs found: {len(synergistic)}")

    return {
        'horizon': hz_name,
        'interactions': interaction_results[:50],
        'n_synergistic': len(synergistic),
        'mean_single_ic': {str(k): round(v, 5) for k, v in mean_single.items()},
    }


# ============================================================================
# ANALYSIS F: CROSS-DAY STABILITY
# ============================================================================

def analysis_f_cross_day_stability(
    mbo_files: List[Path],
    feature_names: List[str],
    top_feat_indices: List[int],
    top_k: int = 30,
) -> Dict:
    """
    Per-day IC for top features. Reports stable vs regime-dependent features.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS F: Cross-Day IC Stability")
    print("=" * 70)

    selected = top_feat_indices[:top_k]
    hz_name = '10s'
    hz_bars = HORIZONS[hz_name]

    # Per-day IC for each selected feature
    per_day_ic: Dict[int, List[float]] = {i: [] for i in selected}
    day_labels = []

    for day_idx, fpath in enumerate(mbo_files):
        result = load_day_npz(fpath)
        if result is None:
            continue
        feats, mid = result
        n = len(mid)
        if n < hz_bars + 100:
            continue

        fwd_ret = np.full(n, np.nan, dtype=np.float64)
        fwd_ret[:n - hz_bars] = (
            (mid[hz_bars:] - mid[:n - hz_bars]) / np.maximum(np.abs(mid[:n - hz_bars]), 1e-6)
        )

        day_labels.append(fpath.stem.replace('_mbo_features', ''))

        for feat_idx in selected:
            if feat_idx >= feats.shape[1]:
                per_day_ic[feat_idx].append(0.0)
                continue
            signal = feats[:, feat_idx].astype(np.float64)
            ic = pearson_ic(signal, fwd_ret)
            per_day_ic[feat_idx].append(ic)

        del feats, mid
        gc.collect()

    # Compute stability metrics
    stability_results = []
    for feat_idx in selected:
        ics = per_day_ic[feat_idx]
        fname = feature_names[feat_idx] if feat_idx < len(feature_names) else f'feat_{feat_idx}'

        if len(ics) < 3:
            continue

        arr = np.array(ics)
        mean_ic = float(np.mean(arr))
        std_ic = float(np.std(arr))
        # Coefficient of variation (lower = more stable)
        cv = std_ic / max(abs(mean_ic), 1e-9)
        # Consistency: fraction of days with same sign as mean
        sign = 1 if mean_ic >= 0 else -1
        consistency = float(np.mean(arr * sign > 0))
        # Min/max day IC
        min_ic = float(arr.min())
        max_ic = float(arr.max())
        # IC range relative to mean
        ic_range_ratio = (max_ic - min_ic) / max(abs(mean_ic), 1e-9)

        # Regime days: days where IC has opposite sign to mean (>= 0.01 magnitude)
        regime_flip_days = [
            day_labels[i] for i, ic in enumerate(ics)
            if ic * sign < -0.01
        ]

        stability_results.append({
            'feat_idx': int(feat_idx),
            'feat_name': fname,
            'mean_ic': round(float(mean_ic), 5),
            'std_ic': round(float(std_ic), 5),
            'cv': round(float(cv), 3),
            'consistency': round(float(consistency), 3),
            'min_ic': round(float(min_ic), 5),
            'max_ic': round(float(max_ic), 5),
            'ic_range_ratio': round(float(ic_range_ratio), 2),
            'n_regime_flip_days': len(regime_flip_days),
            'regime_flip_days': regime_flip_days,
            'per_day_ic': [round(float(x), 5) for x in ics],
            'n_days': len(ics),
        })

    # Rank by stability (low CV = stable, high consistency = stable)
    stable = sorted(stability_results, key=lambda x: x['cv'])
    regime_dep = sorted(stability_results, key=lambda x: x['cv'], reverse=True)

    print(f"\nMOST STABLE features (low CV, consistent IC across days):")
    print(f"  {'Feature':<35s}  {'Mean IC':>8s}  {'Std IC':>7s}  {'CV':>6s}  {'Consist':>7s}  {'Flips':>5s}")
    print(f"  {'-'*35}  {'-'*8}  {'-'*7}  {'-'*6}  {'-'*7}  {'-'*5}")
    for item in stable[:15]:
        print(f"  {item['feat_name']:<35s}"
              f"  {item['mean_ic']:>+8.4f}"
              f"  {item['std_ic']:>7.4f}"
              f"  {item['cv']:>6.2f}"
              f"  {item['consistency']:>7.1%}"
              f"  {item['n_regime_flip_days']:>5d}")

    print(f"\nMOST REGIME-DEPENDENT features (high CV, IC flips across days):")
    print(f"  {'Feature':<35s}  {'Mean IC':>8s}  {'Std IC':>7s}  {'CV':>6s}  {'Consist':>7s}  {'Flips':>5s}")
    print(f"  {'-'*35}  {'-'*8}  {'-'*7}  {'-'*6}  {'-'*7}  {'-'*5}")
    for item in regime_dep[:10]:
        print(f"  {item['feat_name']:<35s}"
              f"  {item['mean_ic']:>+8.4f}"
              f"  {item['std_ic']:>7.4f}"
              f"  {item['cv']:>6.2f}"
              f"  {item['consistency']:>7.1%}"
              f"  {item['n_regime_flip_days']:>5d}")

    return {
        'horizon': hz_name,
        'stability_results': stability_results,
        'day_labels': day_labels,
        'n_days': len(day_labels),
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Lead-lag and cross-feature analysis for Lvl3Quant MBO data'
    )
    parser.add_argument('--n-days', type=int, default=30,
                        help='Max number of days to process (default: 30)')
    parser.add_argument('--top-k', type=int, default=30,
                        help='Number of top features for in-depth analysis (default: 30)')
    parser.add_argument('--quick', action='store_true',
                        help='Skip Granger causality and interaction analyses')
    parser.add_argument('--mbo-dir', type=str, default=None,
                        help='Override MBO features cache directory')
    args = parser.parse_args()

    mbo_dir = Path(args.mbo_dir) if args.mbo_dir else MBO_DIR

    print("=" * 70)
    print("LEAD-LAG AND CROSS-FEATURE ANALYSIS")
    print(f"MBO cache: {mbo_dir}")
    print(f"Max days: {args.n_days}  Top-k: {args.top_k}  Quick: {args.quick}")
    print("=" * 70)

    # Find all MBO NPZ files
    mbo_files = sorted(mbo_dir.glob('*_mbo_features.npz'))
    if not mbo_files:
        # Try alternate naming convention
        mbo_files = sorted(mbo_dir.glob('*.npz'))

    if not mbo_files:
        print(f"ERROR: No .npz files found in {mbo_dir}")
        sys.exit(1)

    mbo_files = mbo_files[:args.n_days]
    print(f"Found {len(mbo_files)} MBO feature files, using {len(mbo_files)}")

    # Verify first file structure
    print("\nVerifying first file structure...")
    sample = load_day_npz(mbo_files[0])
    if sample is None:
        print("ERROR: Could not load first file. Check NPZ format (needs 'mbo_features' key).")
        sys.exit(1)
    feats_sample, mid_sample = sample
    n_features = feats_sample.shape[1]
    print(f"  Shape: {feats_sample.shape}  (N={feats_sample.shape[0]}, features={n_features})")
    del feats_sample, mid_sample

    # Load feature names
    feature_names = get_feature_names_safe()
    print(f"  Feature names: {len(feature_names)} loaded")
    if len(feature_names) < n_features:
        # Pad with generic names
        feature_names += [f'feat_{i}' for i in range(len(feature_names), n_features)]

    t_start = time.time()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    results = {
        'timestamp': timestamp,
        'n_days': len(mbo_files),
        'n_features': n_features,
        'top_k': args.top_k,
        'quick_mode': args.quick,
        'mbo_dir': str(mbo_dir),
        'files': [f.name for f in mbo_files],
        'leakage_cols': sorted(LEAKAGE_COLS),
    }

    # ----------------------------------------------------------------
    # ANALYSIS A: IC Heatmap (all features)
    # ----------------------------------------------------------------
    t0 = time.time()
    result_a = analysis_a_ic_heatmap(
        mbo_files=mbo_files,
        feature_names=feature_names,
        n_features=n_features,
        top_k=20,
    )
    results['analysis_a'] = result_a
    print(f"\n[A complete in {time.time()-t0:.0f}s]")

    # Determine top features by 10s IC for subsequent analyses
    top_by_10s = sorted(
        [i for i in range(n_features) if i not in LEAKAGE_COLS],
        key=lambda i: abs(result_a['mean_ic'].get(str(i), {}).get('10s', 0.0)),
        reverse=True,
    )[:args.top_k]
    print(f"\nTop {args.top_k} features by |IC@10s| selected for in-depth analysis:")
    for rank, feat_idx in enumerate(top_by_10s[:10]):
        fname = feature_names[feat_idx] if feat_idx < len(feature_names) else f'feat_{feat_idx}'
        ic_10s = result_a['mean_ic'].get(str(feat_idx), {}).get('10s', 0.0)
        print(f"  {rank+1:>3d}. {fname:<35s}  IC@10s={ic_10s:+.4f}")

    # ----------------------------------------------------------------
    # ANALYSIS B: Lead-Lag Matrix
    # ----------------------------------------------------------------
    t0 = time.time()
    result_b = analysis_b_leadlag_matrix(
        mbo_files=mbo_files,
        feature_names=feature_names,
        top_feat_indices=top_by_10s,
        top_k=min(args.top_k, 30),
    )
    results['analysis_b'] = result_b
    print(f"\n[B complete in {time.time()-t0:.0f}s]")

    # ----------------------------------------------------------------
    # ANALYSIS C: Granger Causality (skipped in quick mode)
    # ----------------------------------------------------------------
    if not args.quick:
        t0 = time.time()
        result_c = analysis_c_granger(
            mbo_files=mbo_files,
            feature_names=feature_names,
            top_feat_indices=top_by_10s,
            top_k=20,
        )
        results['analysis_c'] = result_c
        print(f"\n[C complete in {time.time()-t0:.0f}s]")
    else:
        print("\n[C skipped — quick mode]")
        results['analysis_c'] = None

    # ----------------------------------------------------------------
    # ANALYSIS D: Temporal Decay
    # ----------------------------------------------------------------
    t0 = time.time()
    result_d = analysis_d_temporal_decay(
        mbo_files=mbo_files,
        feature_names=feature_names,
        top_feat_indices=top_by_10s,
        top_k=20,
    )
    results['analysis_d'] = result_d
    print(f"\n[D complete in {time.time()-t0:.0f}s]")

    # ----------------------------------------------------------------
    # ANALYSIS E: Feature Interactions (skipped in quick mode)
    # ----------------------------------------------------------------
    if not args.quick:
        t0 = time.time()
        result_e = analysis_e_interactions(
            mbo_files=mbo_files,
            feature_names=feature_names,
            top_feat_indices=top_by_10s,
            top_k=min(args.top_k, 30),
        )
        results['analysis_e'] = result_e
        print(f"\n[E complete in {time.time()-t0:.0f}s]")
    else:
        print("\n[E skipped — quick mode]")
        results['analysis_e'] = None

    # ----------------------------------------------------------------
    # ANALYSIS F: Cross-Day Stability
    # ----------------------------------------------------------------
    t0 = time.time()
    result_f = analysis_f_cross_day_stability(
        mbo_files=mbo_files,
        feature_names=feature_names,
        top_feat_indices=top_by_10s,
        top_k=args.top_k,
    )
    results['analysis_f'] = result_f
    print(f"\n[F complete in {time.time()-t0:.0f}s]")

    # ----------------------------------------------------------------
    # SUMMARY
    # ----------------------------------------------------------------
    elapsed = time.time() - t_start

    print("\n" + "=" * 70)
    print("SUMMARY OF KEY FINDINGS")
    print("=" * 70)

    print(f"\nDays analyzed: {len(mbo_files)}")
    print(f"Features analyzed: {n_features} total, {n_features - len(LEAKAGE_COLS)} non-leakage")

    # Best features by IC
    print(f"\nTop 5 features by |IC@1s|:")
    top_1s = result_a['top_per_horizon'].get('1s', [])[:5]
    for item in top_1s:
        print(f"  {item['rank']:>3d}. {item['feat_name']:<35s}  IC={item['mean_ic']:+.4f}  consist={item['consistency']:.0%}")

    print(f"\nTop 5 features by |IC@10s|:")
    top_10s = result_a['top_per_horizon'].get('10s', [])[:5]
    for item in top_10s:
        print(f"  {item['rank']:>3d}. {item['feat_name']:<35s}  IC={item['mean_ic']:+.4f}  consist={item['consistency']:.0%}")

    # Slow decay
    print(f"\nTop 5 SLOWEST-DECAY features (most tradeable):")
    for item in result_d['decay_results'][:5]:
        print(f"  {item['feat_name']:<35s}  IC@1s={item['ic_at_1s']:+.4f}  "
              f"still {item['decay_at_30s']:.0%} at 30s")

    # Stable features
    stable = sorted(
        result_f['stability_results'],
        key=lambda x: (x['cv'], -abs(x['mean_ic']))
    )
    print(f"\nTop 5 MOST STABLE features (low CV, works across regimes):")
    for item in stable[:5]:
        print(f"  {item['feat_name']:<35s}  IC={item['mean_ic']:+.4f}  "
              f"CV={item['cv']:.2f}  consist={item['consistency']:.0%}")

    # Leading pairs
    if result_b['leading_pairs']:
        print(f"\nTop 5 CAUSAL CHAIN relationships (A leads B):")
        for pair in result_b['leading_pairs'][:5]:
            print(f"  {pair['leader']:<30s} -> {pair['follower']:<30s}  "
                  f"lag={pair['optimal_lag_sec']}s  strength={pair['lead_strength']:.2f}x")

    # Granger significant features
    if results.get('analysis_c') and results['analysis_c'].get('summary'):
        sig_granger = [x for x in results['analysis_c']['summary'] if x['mean_f_stat'] > 5]
        print(f"\nGranger-significant features (F > 5): {len(sig_granger)}")
        for item in sig_granger[:5]:
            print(f"  {item['feat_name']:<35s}  F={item['mean_f_stat']:.2f}  dR²={item['mean_delta_r2']:+.5f}")

    # Synergistic interactions
    if results.get('analysis_e') and results['analysis_e'].get('interactions'):
        syn_interactions = [x for x in results['analysis_e']['interactions'] if x['is_synergistic']]
        print(f"\nSynergistic interactions found: {results['analysis_e']['n_synergistic']}")
        for item in syn_interactions[:5]:
            print(f"  {item['feat_a']:<25s} x {item['feat_b']:<25s}  "
                  f"IC(A*B)={item['ic_interaction']:+.4f}  synergy={item['synergy_ratio']:.2f}x")

    print(f"\nTotal elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # ----------------------------------------------------------------
    # SAVE RESULTS
    # ----------------------------------------------------------------
    results_file = RESULTS_DIR / f'lead_lag_analysis_{timestamp}.json'

    # Convert numpy types to native Python for JSON serialization
    def json_safe(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=json_safe)

    print(f"\nResults saved to: {results_file}")
    print("=" * 70)

    return results


if __name__ == '__main__':
    main()
