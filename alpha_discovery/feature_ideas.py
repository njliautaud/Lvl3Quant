#!/usr/bin/env python3
"""
Novel Feature Ideas Generator — Modular Feature Engineering for MBO Pipeline
=============================================================================

Explores new feature families that can be plugged into the existing
MBO feature pipeline. Each feature family is self-contained and returns
a dict of {feature_name: (n_bars,) array}.

Feature Families:
  1. Order Flow Imbalance Persistence (does OFI at T predict OFI at T+30min?)
  2. Microstructure Regime Detection (spread/volume/vol regime clustering)
  3. Cross-Asset Signal Stubs (VIX, bonds, sector ETFs — placeholders)
  4. Time-of-Day Interaction Features (morning vs afternoon behavior)
  5. Book Pressure Divergence (bid vs ask side dynamics)

All features are STRICTLY CAUSAL (only past data, no look-ahead).

Usage:
    # Generate features for a single day:
    python alpha_discovery/feature_ideas.py --date 2025-10-01

    # Test feature ICs against forward returns:
    python alpha_discovery/feature_ideas.py --test-ic --date 2025-10-01

    # As a module:
    from alpha_discovery.feature_ideas import generate_all_features
    features = generate_all_features(mid, spread, ofi_series, trade_volumes, ...)
"""

import sys
import json
import argparse
import logging
import numpy as np
from pathlib import Path
from scipy import stats

# ── Path setup ──
LVL3_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LVL3_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('feature_ideas')

# Constants
BARS_PER_SEC = 10    # 100ms bars
BARS_PER_MIN = 600
BARS_PER_DAY = 234000


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def rolling_mean(arr, window):
    """Causal rolling mean using cumsum (no look-ahead)."""
    out = np.full_like(arr, np.nan, dtype=np.float64)
    cs = np.cumsum(np.nan_to_num(arr, nan=0.0))
    valid = np.cumsum(~np.isnan(arr))
    for i in range(window, len(arr)):
        n = valid[i] - valid[i - window]
        if n > 0:
            out[i] = (cs[i] - cs[i - window]) / n
    return out


def rolling_std(arr, window):
    """Causal rolling standard deviation."""
    out = np.full_like(arr, np.nan, dtype=np.float64)
    cs = np.cumsum(np.nan_to_num(arr, nan=0.0))
    cs2 = np.cumsum(np.nan_to_num(arr, nan=0.0) ** 2)
    valid = np.cumsum(~np.isnan(arr))
    for i in range(window, len(arr)):
        n = valid[i] - valid[i - window]
        if n > 1:
            mean = (cs[i] - cs[i - window]) / n
            var = (cs2[i] - cs2[i - window]) / n - mean ** 2
            out[i] = np.sqrt(max(var, 0))
    return out


def expanding_percentile_rank(arr, min_obs=100):
    """Expanding-window percentile rank (0-1). Strictly causal."""
    import bisect
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    sorted_vals = []
    for i in range(n):
        if not np.isnan(arr[i]):
            bisect.insort(sorted_vals, arr[i])
        if len(sorted_vals) >= min_obs:
            rank = bisect.bisect_left(sorted_vals, arr[i])
            out[i] = rank / len(sorted_vals)
    return out


def compute_autocorrelation(arr, lag, window):
    """Rolling autocorrelation at given lag. Causal."""
    out = np.full(len(arr), np.nan, dtype=np.float64)
    for i in range(window + lag, len(arr)):
        x = arr[i - window:i]
        y = arr[i - window - lag:i - lag]
        valid = ~(np.isnan(x) | np.isnan(y))
        if valid.sum() > 10:
            xv = x[valid]
            yv = y[valid]
            if xv.std() > 1e-10 and yv.std() > 1e-10:
                out[i] = np.corrcoef(xv, yv)[0, 1]
    return out


# ============================================================================
# FAMILY 1: Order Flow Imbalance Persistence
# ============================================================================

def compute_ofi_persistence_features(mid, mbo_features=None, ofi_col_idx=None):
    """
    Order Flow Imbalance Persistence Features.

    Hypothesis: If OFI at time T is persistent (autocorrelated), the market
    is in a trending microstructure regime. If OFI mean-reverts quickly,
    the market is in a noise/ranging regime. This regime signal may predict
    whether future vol signals translate into directional moves.

    Features:
      - ofi_autocorr_1min: autocorrelation of OFI at 1-minute lag
      - ofi_autocorr_5min: autocorrelation of OFI at 5-minute lag
      - ofi_persistence_ratio: ratio of 5min autocorr to 1min autocorr
      - ofi_cumulative_sign: cumulative sign of OFI over trailing 30min
      - ofi_regime: 0=noise, 1=trending (based on persistence ratio)

    Args:
        mid: (n_bars,) mid prices
        mbo_features: (n_bars, n_feat) full MBO feature matrix (optional)
        ofi_col_idx: column index for OFI in mbo_features (default: approximate from returns)
    """
    n = len(mid)
    features = {}

    # Compute OFI proxy from price changes if not provided
    if mbo_features is not None and ofi_col_idx is not None:
        ofi = mbo_features[:, ofi_col_idx].astype(np.float64)
    else:
        # Approximate OFI as signed volume proxy: direction * magnitude of 1s returns
        ofi = np.zeros(n, dtype=np.float64)
        ofi[10:] = (mid[10:] - mid[:-10]) / np.maximum(mid[:-10], 1.0) * 10000

    # Autocorrelation at different lags
    lag_1min = BARS_PER_MIN       # 600 bars
    lag_5min = 5 * BARS_PER_MIN   # 3000 bars
    window = 5 * BARS_PER_MIN     # 5-minute estimation window

    features['ofi_autocorr_1min'] = compute_autocorrelation(ofi, lag_1min, window)
    features['ofi_autocorr_5min'] = compute_autocorrelation(ofi, lag_5min, window)

    # Persistence ratio
    ac1 = features['ofi_autocorr_1min']
    ac5 = features['ofi_autocorr_5min']
    features['ofi_persistence_ratio'] = np.where(
        (np.abs(ac1) > 0.01) & np.isfinite(ac1) & np.isfinite(ac5),
        ac5 / ac1,
        0.0
    )

    # Cumulative sign of OFI over trailing 30 minutes
    ofi_sign = np.sign(ofi)
    window_30min = 30 * BARS_PER_MIN
    features['ofi_cumulative_sign'] = np.full(n, 0.0)
    cs = np.cumsum(ofi_sign)
    for i in range(window_30min, n):
        features['ofi_cumulative_sign'][i] = (cs[i] - cs[i - window_30min]) / window_30min

    # Regime classification: trending if autocorr is high and persistent
    features['ofi_regime'] = np.where(
        (features['ofi_autocorr_1min'] > 0.1) & (features['ofi_persistence_ratio'] > 0.3),
        1.0, 0.0
    ).astype(np.float64)

    return features


# ============================================================================
# FAMILY 2: Microstructure Regime Detection
# ============================================================================

def compute_regime_features(mid, spread):
    """
    Microstructure Regime Detection Features.

    Identifies the current market microstructure regime using clustering
    of spread, volume, and volatility metrics. Different regimes have
    different signal characteristics.

    Features:
      - spread_regime: expanding percentile of spread (0=tight, 1=wide)
      - vol_regime_5min: expanding percentile of 5min realized vol
      - activity_regime: expanding percentile of price change frequency
      - regime_composite: weighted combination of all three
      - regime_transition: derivative of regime composite (regime changing?)
      - spread_vol_ratio: spread relative to vol (high = illiquid relative to vol)
    """
    n = len(mid)
    features = {}

    # Spread regime: expanding percentile
    features['spread_regime_pct'] = expanding_percentile_rank(spread, min_obs=100)

    # Volatility regime: 5-minute trailing realized vol
    ret_1s = np.zeros(n)
    ret_1s[10:] = (mid[10:] - mid[:-10]) / np.maximum(mid[:-10], 1.0) * 10000
    vol_5min = rolling_std(ret_1s, 5 * BARS_PER_MIN)
    features['vol_regime_5min_pct'] = expanding_percentile_rank(vol_5min, min_obs=100)

    # Activity regime: how often does price change in a 1-minute window?
    price_changes = np.zeros(n)
    price_changes[1:] = (mid[1:] != mid[:-1]).astype(float)
    activity = rolling_mean(price_changes, BARS_PER_MIN)
    features['activity_regime_pct'] = expanding_percentile_rank(activity, min_obs=100)

    # Composite regime score
    sr = np.nan_to_num(features['spread_regime_pct'], nan=0.5)
    vr = np.nan_to_num(features['vol_regime_5min_pct'], nan=0.5)
    ar = np.nan_to_num(features['activity_regime_pct'], nan=0.5)
    features['regime_composite'] = 0.4 * vr + 0.3 * sr + 0.3 * ar

    # Regime transition (rate of change over 5 minutes)
    rc = features['regime_composite']
    features['regime_transition'] = np.zeros(n, dtype=np.float64)
    lag = 5 * BARS_PER_MIN
    features['regime_transition'][lag:] = rc[lag:] - rc[:-lag]

    # Spread-to-vol ratio (illiquidity measure)
    features['spread_vol_ratio'] = np.where(
        (vol_5min > 1e-6) & np.isfinite(vol_5min),
        spread / vol_5min,
        0.0
    )

    return features


# ============================================================================
# FAMILY 3: Cross-Asset Signal Stubs
# ============================================================================

def compute_cross_asset_features(mid, vix_mid=None, bond_mid=None, spy_mid=None):
    """
    Cross-Asset Signal Features (Stubs).

    These are placeholders designed to accept external data feeds.
    When cross-asset data is not available, features are filled with NaN.

    Hypotheses:
      - VIX spikes predict ES vol regime changes (leading indicator)
      - Bond-equity correlation shifts signal risk regime changes
      - SPY-ES basis divergence signals institutional flow direction

    Features:
      - vix_zscore_5min: z-score of VIX over trailing 5 minutes
      - vix_regime: VIX level relative to expanding distribution
      - bond_equity_corr: rolling correlation between bond and ES returns
      - spy_es_basis: SPY-ES price basis (normalized)
      - cross_asset_signal: composite cross-asset signal
    """
    n = len(mid)
    features = {}

    # VIX features
    if vix_mid is not None and len(vix_mid) >= n:
        vix = vix_mid[:n].astype(np.float64)
        vix_mean = rolling_mean(vix, 5 * BARS_PER_MIN)
        vix_std = rolling_std(vix, 5 * BARS_PER_MIN)
        features['vix_zscore_5min'] = np.where(
            (vix_std > 1e-6) & np.isfinite(vix_std),
            (vix - vix_mean) / vix_std,
            0.0
        )
        features['vix_regime'] = expanding_percentile_rank(vix, min_obs=100)
    else:
        features['vix_zscore_5min'] = np.full(n, np.nan)
        features['vix_regime'] = np.full(n, np.nan)

    # Bond-equity correlation
    if bond_mid is not None and len(bond_mid) >= n:
        bond = bond_mid[:n].astype(np.float64)
        es_ret = np.zeros(n)
        es_ret[10:] = (mid[10:] - mid[:-10]) / np.maximum(mid[:-10], 1.0)
        bond_ret = np.zeros(n)
        bond_ret[10:] = (bond[10:] - bond[:-10]) / np.maximum(bond[:-10], 1.0)

        # Rolling 30-minute correlation
        window = 30 * BARS_PER_MIN
        features['bond_equity_corr'] = np.full(n, np.nan)
        for i in range(window, n, BARS_PER_MIN):  # compute every minute for efficiency
            x = es_ret[i - window:i:10]  # subsample to 1s
            y = bond_ret[i - window:i:10]
            valid = ~(np.isnan(x) | np.isnan(y))
            if valid.sum() > 20:
                features['bond_equity_corr'][i] = np.corrcoef(x[valid], y[valid])[0, 1]
        # Forward-fill
        last_val = np.nan
        for i in range(n):
            if np.isfinite(features['bond_equity_corr'][i]):
                last_val = features['bond_equity_corr'][i]
            else:
                features['bond_equity_corr'][i] = last_val
    else:
        features['bond_equity_corr'] = np.full(n, np.nan)

    # SPY-ES basis
    if spy_mid is not None and len(spy_mid) >= n:
        spy = spy_mid[:n].astype(np.float64)
        # Normalize: ES ~ SPY * 10 approximately
        basis = mid - spy * 10.0
        basis_mean = rolling_mean(basis, 30 * BARS_PER_MIN)
        basis_std = rolling_std(basis, 30 * BARS_PER_MIN)
        features['spy_es_basis'] = np.where(
            (basis_std > 1e-6) & np.isfinite(basis_std),
            (basis - basis_mean) / basis_std,
            0.0
        )
    else:
        features['spy_es_basis'] = np.full(n, np.nan)

    # Composite (only if any cross-asset data available)
    all_nan = all(np.all(np.isnan(features[k])) for k in features)
    if all_nan:
        features['cross_asset_signal'] = np.full(n, np.nan)
    else:
        signals = []
        for k, v in features.items():
            if not np.all(np.isnan(v)):
                signals.append(np.nan_to_num(v, nan=0.0))
        features['cross_asset_signal'] = np.mean(signals, axis=0) if signals else np.full(n, np.nan)

    return features


# ============================================================================
# FAMILY 4: Time-of-Day Interaction Features
# ============================================================================

def compute_tod_interaction_features(mid, spread):
    """
    Time-of-Day Interaction Features.

    Hypothesis: Market microstructure behaves differently in morning vs
    afternoon sessions. Features that interact signal strength with
    time-of-day may capture regime differences.

    The de-biased CNN sweep showed morning_afternoon filter improves Sharpe
    from 2.60 to 3.89 — confirming strong TOD effects.

    Features:
      - tod_normalized: normalized time (0=open, 1=close)
      - tod_session: 0=morning (9:30-11:30), 1=lunch (11:30-1:30), 2=afternoon (1:30-3:30), 3=close (3:30-4:00)
      - morning_vol_ratio: current vol / morning session average vol
      - afternoon_shift: how different is current spread from morning average
      - tod_vol_interaction: volatility * time interaction
      - opening_fade: decaying signal strength from market open
      - lunch_lull_indicator: indicator for the low-activity lunch period
    """
    n = len(mid)
    features = {}

    minutes = np.arange(n, dtype=np.float64) / BARS_PER_MIN

    # Normalized time
    total_minutes = 6.5 * 60  # RTH session
    features['tod_normalized'] = np.clip(minutes / total_minutes, 0, 1)

    # Session encoding
    features['tod_session'] = np.zeros(n, dtype=np.float64)
    features['tod_session'][(minutes >= 120) & (minutes < 240)] = 1.0  # lunch
    features['tod_session'][(minutes >= 240) & (minutes < 330)] = 2.0  # afternoon
    features['tod_session'][minutes >= 330] = 3.0  # close

    # Compute 1-second returns for vol calculation
    ret_1s = np.zeros(n, dtype=np.float64)
    ret_1s[10:] = np.abs(mid[10:] - mid[:-10]) / np.maximum(mid[:-10], 1.0) * 10000

    # Morning session stats (computed causally at each bar)
    morning_end = 120 * BARS_PER_MIN  # 2 hours = 120 minutes
    morning_vol_cumsum = np.zeros(n, dtype=np.float64)
    morning_vol_count = np.zeros(n, dtype=np.float64)

    for i in range(10, min(morning_end, n)):
        morning_vol_cumsum[i] = morning_vol_cumsum[i - 1] + ret_1s[i]
        morning_vol_count[i] = morning_vol_count[i - 1] + 1

    # Forward-fill morning averages for afternoon use
    if morning_end < n and morning_vol_count[morning_end - 1] > 0:
        morning_avg_vol = morning_vol_cumsum[morning_end - 1] / morning_vol_count[morning_end - 1]
    else:
        morning_avg_vol = 1.0

    # Current trailing vol ratio to morning average
    trailing_vol = rolling_mean(ret_1s, 5 * BARS_PER_MIN)
    features['morning_vol_ratio'] = np.where(
        (morning_avg_vol > 1e-6) & np.isfinite(trailing_vol),
        trailing_vol / morning_avg_vol,
        1.0
    )

    # Spread shift from morning average
    morning_spread_mean = rolling_mean(spread[:morning_end], morning_end) if morning_end > 0 else np.full(1, np.nan)
    ms = morning_spread_mean[-1] if len(morning_spread_mean) > 0 and np.isfinite(morning_spread_mean[-1]) else spread[:morning_end].mean() if morning_end > 0 else 1.0
    features['afternoon_spread_shift'] = (spread - ms) / max(ms, 1e-6)

    # Vol * time interaction
    vol_5min = rolling_std(ret_1s, 5 * BARS_PER_MIN)
    features['tod_vol_interaction'] = np.nan_to_num(vol_5min, nan=0.0) * features['tod_normalized']

    # Opening fade: exponential decay from market open
    decay_halflife = 30 * BARS_PER_MIN  # 30 minutes
    features['opening_fade'] = np.exp(-0.693 * minutes / 30.0)  # half-life at 30min

    # Lunch lull indicator (smooth)
    lunch_center = 180  # 12:00 noon = 150min from open
    lunch_width = 60    # +/- 60 minutes
    features['lunch_lull_indicator'] = np.exp(-0.5 * ((minutes - lunch_center) / lunch_width) ** 2)

    return features


# ============================================================================
# FAMILY 5: Book Pressure Divergence
# ============================================================================

def compute_book_pressure_features(mid, spread, mbo_features=None):
    """
    Book Pressure Divergence Features.

    Hypothesis: When bid-side and ask-side dynamics diverge (e.g., bid depth
    growing while ask depth thinning), this asymmetry predicts future direction.

    Features:
      - pressure_divergence_rate: rate of change of bid-ask pressure difference
      - depth_asymmetry_zscore: z-score of bid/ask depth ratio
      - spread_direction_interaction: spread change * price direction
      - book_renewal_rate: how fast is the book being refreshed (add/cancel ratio rate of change)
      - aggressive_flow_persistence: autocorrelation of aggressive order flow
    """
    n = len(mid)
    features = {}

    # Compute basic pressure metrics from mid/spread
    # Bid-ask pressure proxy: spread changes indicate one side being consumed faster
    spread_change = np.zeros(n, dtype=np.float64)
    spread_change[1:] = spread[1:] - spread[:-1]

    price_change = np.zeros(n, dtype=np.float64)
    price_change[1:] = mid[1:] - mid[:-1]

    # Spread * direction interaction: positive when spread widens on price moves
    features['spread_direction_interaction'] = spread_change * np.sign(price_change)

    # Rolling stats
    window = 5 * BARS_PER_MIN
    features['spread_change_zscore'] = np.zeros(n, dtype=np.float64)
    sc_mean = rolling_mean(spread_change, window)
    sc_std = rolling_std(spread_change, window)
    valid = (sc_std > 1e-10) & np.isfinite(sc_std)
    features['spread_change_zscore'][valid] = (spread_change[valid] - sc_mean[valid]) / sc_std[valid]

    # If MBO features available, compute richer metrics
    if mbo_features is not None:
        # Try to extract bid/ask pressure columns (indices depend on feature ordering)
        # From mbo_features.py: bid_pressure=col 18, ask_pressure=col 19
        try:
            bid_pressure = mbo_features[:, 18].astype(np.float64)
            ask_pressure = mbo_features[:, 19].astype(np.float64)

            pressure_diff = bid_pressure - ask_pressure
            features['pressure_divergence_rate'] = np.zeros(n, dtype=np.float64)
            lag = BARS_PER_MIN  # 1-minute lag
            features['pressure_divergence_rate'][lag:] = pressure_diff[lag:] - pressure_diff[:-lag]

            # Depth ratio z-score
            total_bid = mbo_features[:, 4].astype(np.float64)  # total_bid_vol (col 4)
            total_ask = mbo_features[:, 5].astype(np.float64)  # total_ask_vol (col 5)
            depth_ratio = np.where(total_ask > 0, total_bid / total_ask, 1.0)
            dr_mean = rolling_mean(depth_ratio, window)
            dr_std = rolling_std(depth_ratio, window)
            features['depth_asymmetry_zscore'] = np.where(
                (dr_std > 1e-10) & np.isfinite(dr_std),
                (depth_ratio - dr_mean) / dr_std,
                0.0
            )

            # Book renewal rate: add_count / cancel_count ratio change
            add_count = mbo_features[:, 13].astype(np.float64)  # add_count (col 13)
            cancel_count = mbo_features[:, 14].astype(np.float64)  # cancel_count (col 14)
            renewal = np.where(cancel_count > 0, add_count / cancel_count, 1.0)
            renewal_rate = np.zeros(n, dtype=np.float64)
            renewal_rate[BARS_PER_MIN:] = renewal[BARS_PER_MIN:] - renewal[:-BARS_PER_MIN]
            features['book_renewal_rate'] = renewal_rate

            # Aggressive flow persistence
            agg_buy = mbo_features[:, 34].astype(np.float64)  # aggressive_buy_count (col 34)
            agg_sell = mbo_features[:, 35].astype(np.float64)  # aggressive_sell_count (col 35)
            agg_imb = agg_buy - agg_sell
            features['aggressive_flow_persistence'] = compute_autocorrelation(
                agg_imb, lag=BARS_PER_MIN, window=5*BARS_PER_MIN
            )

        except (IndexError, ValueError) as e:
            logger.warning(f"Could not extract MBO columns: {e}. Using price-only features.")
            features.setdefault('pressure_divergence_rate', np.full(n, np.nan))
            features.setdefault('depth_asymmetry_zscore', np.full(n, np.nan))
            features.setdefault('book_renewal_rate', np.full(n, np.nan))
            features.setdefault('aggressive_flow_persistence', np.full(n, np.nan))
    else:
        features['pressure_divergence_rate'] = np.full(n, np.nan)
        features['depth_asymmetry_zscore'] = np.full(n, np.nan)
        features['book_renewal_rate'] = np.full(n, np.nan)
        features['aggressive_flow_persistence'] = np.full(n, np.nan)

    return features


# ============================================================================
# UNIFIED FEATURE GENERATOR
# ============================================================================

def generate_all_features(mid, spread, mbo_features=None, ofi_col_idx=None,
                          vix_mid=None, bond_mid=None, spy_mid=None):
    """
    Generate all novel features for a single day.

    Args:
        mid: (n_bars,) mid prices
        spread: (n_bars,) bid-ask spread
        mbo_features: (n_bars, n_feat) full MBO feature matrix (optional)
        ofi_col_idx: column index for OFI in mbo_features
        vix_mid, bond_mid, spy_mid: cross-asset mid prices (optional)

    Returns:
        dict: {feature_name: (n_bars,) np.ndarray}
    """
    all_features = {}

    logger.info("Computing OFI persistence features...")
    all_features.update(compute_ofi_persistence_features(mid, mbo_features, ofi_col_idx))

    logger.info("Computing regime detection features...")
    all_features.update(compute_regime_features(mid, spread))

    logger.info("Computing cross-asset features...")
    all_features.update(compute_cross_asset_features(mid, vix_mid, bond_mid, spy_mid))

    logger.info("Computing time-of-day interaction features...")
    all_features.update(compute_tod_interaction_features(mid, spread))

    logger.info("Computing book pressure features...")
    all_features.update(compute_book_pressure_features(mid, spread, mbo_features))

    logger.info(f"Generated {len(all_features)} novel features")
    return all_features


def features_to_matrix(features_dict, n_bars):
    """Convert feature dict to (n_bars, n_features) matrix + names list."""
    names = sorted(features_dict.keys())
    matrix = np.column_stack([features_dict[n][:n_bars] for n in names])
    return matrix, names


# ============================================================================
# IC TESTING
# ============================================================================

def test_feature_ics(features_dict, mid, horizon_bars=18000):
    """
    Test Spearman IC of each feature against forward returns.

    Args:
        features_dict: {name: (n_bars,) array}
        mid: (n_bars,) mid prices
        horizon_bars: forward return horizon (default: 30min = 18000)

    Returns:
        list of dicts with IC results
    """
    n = len(mid)

    # Forward return
    fwd_ret = np.full(n, np.nan)
    fwd_ret[:n - horizon_bars] = (mid[horizon_bars:] - mid[:n - horizon_bars]) / mid[:n - horizon_bars]

    results = []
    for name, feat in sorted(features_dict.items()):
        valid = np.isfinite(feat) & np.isfinite(fwd_ret)
        if valid.sum() < 1000:
            results.append({'feature': name, 'ic': np.nan, 'n_valid': int(valid.sum())})
            continue

        ic, p = stats.spearmanr(feat[valid], fwd_ret[valid])
        results.append({
            'feature': name,
            'ic': round(float(ic), 6),
            'abs_ic': round(abs(float(ic)), 6),
            'p_value': float(p),
            'n_valid': int(valid.sum()),
            'significant': bool(p < 0.01),
        })

    results.sort(key=lambda x: x.get('abs_ic', 0), reverse=True)
    return results


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Novel Feature Ideas Generator')
    parser.add_argument('--date', type=str, help='Date to process (YYYY-MM-DD)')
    parser.add_argument('--test-ic', action='store_true', help='Test feature ICs')
    parser.add_argument('--horizon', type=int, default=18000,
                        help='Forward return horizon in bars (default: 18000 = 30min)')
    args = parser.parse_args()

    # Load data
    FEAT_CACHE = LVL3_ROOT / 'data' / 'processed' / 'mbo_features_cache'

    if args.date:
        dates = [args.date]
    else:
        dates = sorted([f.stem.replace('_mbo_features', '') for f in FEAT_CACHE.glob('*.npz')])
        if not dates:
            print("No MBO data found. Provide --date or ensure data exists in data/processed/mbo_features_cache/")
            return
        dates = dates[:1]  # Just use first date for demo

    for date in dates:
        path = FEAT_CACHE / f'{date}_mbo_features.npz'
        if not path.exists():
            print(f"No data for {date}")
            continue

        print(f"\nProcessing {date}...")
        d = np.load(str(path), mmap_mode='r')
        mbo_feats = d['mbo_features']
        mid = mbo_feats[:, 0].astype(np.float64)
        spread = mbo_feats[:, 1].astype(np.float64)

        features = generate_all_features(mid, spread, mbo_features=mbo_feats)

        print(f"\nGenerated {len(features)} features:")
        for name in sorted(features.keys()):
            arr = features[name]
            valid = np.isfinite(arr).sum()
            print(f"  {name:40s} | valid: {valid:>8,}/{len(arr):,} | "
                  f"mean: {np.nanmean(arr):+.4f} std: {np.nanstd(arr):.4f}")

        if args.test_ic:
            print(f"\nTesting ICs against {args.horizon}-bar forward return...")
            ic_results = test_feature_ics(features, mid, horizon_bars=args.horizon)

            print(f"\n{'Feature':40s} | {'IC':>10} | {'p-value':>10} | {'Significant':>11}")
            print("-" * 80)
            for r in ic_results:
                sig = '***' if r.get('significant') else ''
                ic_str = f"{r['ic']:+.6f}" if np.isfinite(r.get('ic', np.nan)) else 'N/A'
                p_str = f"{r.get('p_value', 1.0):.2e}" if np.isfinite(r.get('p_value', 1.0)) else 'N/A'
                print(f"  {r['feature']:40s} | {ic_str:>10} | {p_str:>10} | {sig:>11}")

            # Save results
            results_dir = LVL3_ROOT / 'alpha_discovery' / 'results'
            results_dir.mkdir(parents=True, exist_ok=True)
            out = results_dir / f'feature_ideas_ic_{date}.json'
            with open(out, 'w') as f:
                json.dump({'date': date, 'horizon_bars': args.horizon, 'results': ic_results}, f, indent=2)
            print(f"\nSaved to {out}")


if __name__ == '__main__':
    main()
