"""
Multi-Bar Alpha Discovery Scan — ES Futures MBO
================================================

HYPOTHESIS: Our 100ms model has IC=0.1135 on 3s returns but signal value
(~0.11 * sigma_3s) is too small to overcome market order costs (~1.25 ticks).

Aggregating to longer timeframes may produce:
  - Stronger IC per bar (less noise, better feature averaging)
  - Larger target moves (easier to overcome costs)
  - Fewer trades/day (less commission impact)
  - More time for limit orders to fill (queue position less critical)

This script aggregates 100ms snapshots to multiple timeframes and runs
a walk-forward LightGBM scan for each (timeframe x target) combination,
then performs a detailed cost-adjusted PnL analysis.

TIMEFRAMES TESTED:
  - 1s   bars (every 10th 100ms snapshot)
  - 5s   bars (every 50th)
  - 10s  bars (every 100th)
  - 30s  bars (every 300th)
  - 1min bars (every 600th)

TARGETS PER TIMEFRAME:
  - Forward return at 1-bar, 2-bar, 5-bar, 10-bar horizons
  - Forward volatility at same horizons

EXECUTION COST MODEL:
  - Market order spread cost:   1 tick (0.25 pts = $3.125)
  - Commission round-trip:      0.24 ticks ($3.00 at $12.50/tick)
  - Total market order cost:    1.24 ticks
  - Limit entry + market exit:  0.24 ticks (net, entry edge cancels exit cost)
  - Limit entry + limit exit:   -0.76 ticks (net edge, ignoring dir move)

Usage:
    python alpha_discovery/run_multibar_alpha_scan.py
    python alpha_discovery/run_multibar_alpha_scan.py --n-days 10
    python alpha_discovery/run_multibar_alpha_scan.py --fast
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
from typing import Dict, List, Optional, Tuple
from scipy.stats import spearmanr, ttest_1samp

# ============================================================================
# PATHS
# ============================================================================
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from alpha_discovery.mbo_alpha_scan import MBOAlphaScanner, RESULTS_DIR
from alpha_discovery.run_return_multihorizon import load_feature_cache, EXCLUDE_FEATURES_DIRECTION
from alpha_discovery.run_model_refinement import walk_forward_evaluate

# ============================================================================
# LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(RESULTS_DIR / 'multibar_scan.log'),
                            mode='a', encoding='utf-8'),
    ]
)
log = logging.getLogger("multibar_scan")

# ============================================================================
# CONSTANTS
# ============================================================================
TICK_SIZE        = 0.25      # ES minimum price increment ($3.125 per tick)
TICK_VALUE       = 12.50     # Dollar value per tick per contract
BASE_INTERVAL_MS = 100       # 100ms base snapshots
BARS_PER_SEC     = 10        # 100ms = 10 bars/sec

# RTH hours: 6.5 hr/day
RTH_HOURS_PER_DAY = 6.5
RTH_SECS_PER_DAY  = RTH_HOURS_PER_DAY * 3600

# Execution cost model (in ticks)
COST_MARKET_SPREAD   = 1.0    # half-spread each leg = 1 tick round-trip
COST_COMMISSION_RT = 4.70  # HC #52: $4.70 RT (AMP) / TICK_VALUE  # $3.00 commission = 0.24 ticks
COST_MARKET_TOTAL    = COST_MARKET_SPREAD + COST_COMMISSION_RT  # 1.24 ticks

COST_LIM_MKT_NET     = COST_COMMISSION_RT                       # 0.24 ticks (entry edge cancels exit)
COST_LIM_LIM_NET     = COST_COMMISSION_RT - COST_MARKET_SPREAD  # -0.76 ticks (net edge: favorable!)

# Timeframes: (label, n_100ms_snapshots_per_bar)
TIMEFRAMES = [
    ('1s',  10),
    ('5s',  50),
    ('10s', 100),
    ('30s', 300),
    ('1min', 600),
]

# Target horizons expressed in number of aggregated bars
TARGET_HORIZONS_BARS = [1, 2, 5, 10]


# ============================================================================
# JSON SERIALIZATION HELPER
# ============================================================================
def to_safe(obj):
    """Convert numpy types and NaN/Inf to JSON-serializable form."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(obj, np.ndarray):
        return [to_safe(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {k: to_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_safe(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


# ============================================================================
# PHASE 1: TIMEFRAME AGGREGATION
# ============================================================================

def aggregate_to_timeframe(
    features_100ms: np.ndarray,
    mid_prices_100ms: np.ndarray,
    day_boundaries_100ms: List[int],
    n_per_bar: int,
    feature_names: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """
    Aggregate 100ms snapshots to a longer timeframe.

    Strategy:
    - Features: LAST snapshot in each bar window (point-in-time snapshot state).
      This is the cleanest for LOB-state features (no averaging over stale states).
      Additional bar-level features (OHLC, range, etc.) are computed and appended.
    - mid_price: CLOSE (last mid_price in the bar)
    - day_boundaries: converted to bar indices

    Returns:
        features_bar:   (M, F+extra) array of bar features
        mid_prices_bar: (M,) mid price at bar close
        hour_of_day_bar:(M,) hour of day at bar close
        day_boundaries_bar: list of bar-level day boundaries
    """
    n_total = len(mid_prices_100ms)
    n_days = len(day_boundaries_100ms) - 1

    # Find the hour_of_day column index
    try:
        hour_col = feature_names.index('hour_norm')
        has_hour = True
    except ValueError:
        has_hour = False

    all_bar_features = []
    all_bar_mids     = []
    all_bar_hours    = []
    bar_day_boundaries = [0]

    for d in range(n_days):
        ds = day_boundaries_100ms[d]
        de = day_boundaries_100ms[d + 1]
        day_len = de - ds

        if day_len < n_per_bar:
            log.debug(f"  Day {d}: only {day_len} snapshots < {n_per_bar}, skipping")
            # Still add a boundary (zero-length day segment)
            bar_day_boundaries.append(bar_day_boundaries[-1])
            continue

        feats_day = features_100ms[ds:de]
        mids_day  = mid_prices_100ms[ds:de]

        # Number of complete bars in this day
        n_bars = day_len // n_per_bar

        bar_features_list = []
        bar_mids_list     = []
        bar_hours_list    = []

        for b in range(n_bars):
            bar_start = b * n_per_bar
            bar_end   = (b + 1) * n_per_bar

            # Snapshot window for this bar
            window_feats = feats_day[bar_start:bar_end]   # (n_per_bar, F)
            window_mids  = mids_day[bar_start:bar_end]    # (n_per_bar,)

            # POINT-IN-TIME features: LAST snapshot in window
            last_feats = window_feats[-1].copy()           # (F,)

            # Bar-level OHLC features from mid price
            bar_open  = window_mids[0]
            bar_high  = window_mids.max()
            bar_low   = window_mids.min()
            bar_close = window_mids[-1]
            bar_range = (bar_high - bar_low) / TICK_SIZE  # in ticks
            bar_ret   = (bar_close - bar_open) / bar_open if bar_open > 0 else 0.0
            bar_mean  = window_mids.mean()
            bar_vwap_proxy = bar_mean  # VWAP proxy (no volume info, use mid mean)

            # Additional microstructure bar-level features
            # Count how many ticks moved up vs down within the bar
            price_changes = np.diff(window_mids)
            up_moves   = (price_changes > 0.1 * TICK_SIZE).sum()
            down_moves = (price_changes < -0.1 * TICK_SIZE).sum()
            no_move    = (np.abs(price_changes) <= 0.1 * TICK_SIZE).sum()
            tick_imbalance = (up_moves - down_moves) / max(up_moves + down_moves, 1)

            # Bar extra features: [open, high, low, close, range_ticks, ret, mean,
            #                      vwap_proxy, up_moves, down_moves, tick_imbalance]
            bar_extra = np.array([
                bar_open,
                bar_high,
                bar_low,
                bar_close,
                bar_range,
                bar_ret,
                bar_mean,
                bar_vwap_proxy,
                float(up_moves),
                float(down_moves),
                tick_imbalance,
            ], dtype=np.float32)

            # Concatenate: [last_snapshot_features, bar_extra]
            bar_vec = np.concatenate([last_feats, bar_extra])
            bar_features_list.append(bar_vec)
            bar_mids_list.append(bar_close)

            if has_hour:
                bar_hours_list.append(last_feats[hour_col] * 24.0)
            else:
                bar_hours_list.append(0.0)

        if n_bars > 0:
            day_bar_feats = np.array(bar_features_list, dtype=np.float32)
            day_bar_mids  = np.array(bar_mids_list, dtype=np.float64)
            day_bar_hours = np.array(bar_hours_list, dtype=np.float32)

            all_bar_features.append(day_bar_feats)
            all_bar_mids.append(day_bar_mids)
            all_bar_hours.append(day_bar_hours)
            bar_day_boundaries.append(bar_day_boundaries[-1] + n_bars)
        else:
            bar_day_boundaries.append(bar_day_boundaries[-1])

    if not all_bar_features:
        return (np.empty((0, features_100ms.shape[1] + 11), dtype=np.float32),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float32),
                bar_day_boundaries)

    features_bar   = np.concatenate(all_bar_features, axis=0)
    mid_prices_bar = np.concatenate(all_bar_mids, axis=0)
    hour_of_day_bar = np.concatenate(all_bar_hours, axis=0)

    return features_bar, mid_prices_bar, hour_of_day_bar, bar_day_boundaries


def get_bar_feature_names(base_names: List[str]) -> List[str]:
    """Return feature names for the aggregated bar (base + bar extras)."""
    bar_extras = [
        'bar_open', 'bar_high', 'bar_low', 'bar_close',
        'bar_range_ticks', 'bar_ret', 'bar_mid_mean', 'bar_vwap_proxy',
        'bar_up_moves', 'bar_down_moves', 'bar_tick_imbalance',
    ]
    return list(base_names) + bar_extras


# ============================================================================
# PHASE 2: TARGET COMPUTATION
# ============================================================================

def compute_bar_targets(
    mid_prices_bar: np.ndarray,
    day_boundaries_bar: List[int],
    horizon_bars: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute forward return and forward volatility targets for a given horizon.

    Forward return:   log(mid[t + horizon] / mid[t])
    Forward volatility: std of log returns over [t, t+horizon] window
    Both are NaN-filled at day boundaries.

    Returns:
        ret_target: (N,) forward log return
        vol_target: (N,) forward realized volatility
    """
    N = len(mid_prices_bar)
    n_days = len(day_boundaries_bar) - 1

    log_mid = np.log(np.maximum(mid_prices_bar.astype(np.float64), 1.0))
    ret_target = np.full(N, np.nan, dtype=np.float32)
    vol_target = np.full(N, np.nan, dtype=np.float32)

    for d in range(n_days):
        ds = day_boundaries_bar[d]
        de = day_boundaries_bar[d + 1]
        day_len = de - ds

        if day_len < horizon_bars + 2:
            continue

        # Last `horizon_bars` bars of the day cannot have a clean forward target
        valid_end = de - horizon_bars

        # Forward return: log(mid[t+h] / mid[t])
        for i in range(ds, valid_end):
            ret_target[i] = float(log_mid[i + horizon_bars] - log_mid[i])

        # Forward volatility: realized vol of bar returns over the horizon window
        # Use bar-level log returns within the forward window
        for i in range(ds, valid_end):
            window_log = log_mid[i:i + horizon_bars + 1]
            bar_rets = np.diff(window_log)
            if len(bar_rets) > 0:
                vol_target[i] = float(np.std(bar_rets))

    return ret_target, vol_target


# ============================================================================
# PHASE 3: WALK-FORWARD LIGHTGBM SCAN
# ============================================================================

def scan_single_timeframe(
    features_bar:    np.ndarray,
    mid_prices_bar:  np.ndarray,
    hour_of_day_bar: np.ndarray,
    day_boundaries_bar: List[int],
    feature_names_bar: List[str],
    tf_label:        str,
    n_per_bar:       int,
    fast:            bool = False,
    min_train_days:  int = 3,
) -> List[dict]:
    """
    Run walk-forward LightGBM scan for all (horizon, target_type) combinations.
    Returns list of result dicts.
    """
    n_days = len(day_boundaries_bar) - 1
    M = len(mid_prices_bar)
    tf_results = []

    # Feature filtering: exclude absolute prices, time-of-day, vol-proxies
    # for direction targets (same as multihorizon scan)
    keep_mask = np.array([fn not in EXCLUDE_FEATURES_DIRECTION
                          for fn in feature_names_bar])
    features_filtered = features_bar[:, keep_mask]
    feature_names_filtered = [fn for fn in feature_names_bar
                               if fn not in EXCLUDE_FEATURES_DIRECTION]

    lgbm_params = {
        'n_estimators':     300 if fast else 500,
        'max_depth':        6,
        'learning_rate':    0.03,
        'subsample':        0.8,
        'colsample_bytree': 0.7,
        'reg_alpha':        0.1,
        'reg_lambda':       1.0,
        'min_child_samples': max(20, M // (n_days * 500)),  # scale with bar count
        'verbose':          -1,
        'n_jobs':           -1,
    }

    log.info(f"  [{tf_label}] Scanning {len(TARGET_HORIZONS_BARS)} horizons x 2 targets "
             f"= {len(TARGET_HORIZONS_BARS) * 2} scans  "
             f"({M:,} bars, {n_days} days)")

    for horizon_bars in TARGET_HORIZONS_BARS:
        # Skip horizons that are too long relative to day length
        day_lengths = [day_boundaries_bar[d+1] - day_boundaries_bar[d]
                       for d in range(n_days)
                       if day_boundaries_bar[d+1] > day_boundaries_bar[d]]
        if not day_lengths or np.median(day_lengths) < horizon_bars * 3:
            log.warning(f"  [{tf_label}] Horizon {horizon_bars}bar too long for day length "
                        f"(median={int(np.median(day_lengths) if day_lengths else 0)} bars), skipping")
            continue

        log.info(f"  [{tf_label}] Computing targets for horizon={horizon_bars}bar...")
        ret_target, vol_target = compute_bar_targets(
            mid_prices_bar=mid_prices_bar,
            day_boundaries_bar=day_boundaries_bar,
            horizon_bars=horizon_bars,
        )

        # Compute target statistics for later analysis
        ret_valid = ret_target[np.isfinite(ret_target)]
        vol_valid = vol_target[np.isfinite(vol_target)]

        ret_sigma = float(np.std(ret_valid)) if len(ret_valid) > 10 else 0.0
        vol_sigma = float(np.std(vol_valid)) if len(vol_valid) > 10 else 0.0
        # ret_sigma_ticks: 1 unit of ret standard deviation in ticks
        mid_mean  = float(np.nanmean(mid_prices_bar))
        ret_sigma_ticks = ret_sigma * mid_mean / TICK_SIZE  # approx: log-ret * price / tick_size

        for target_type, target_arr, target_sigma in [
            ('return',     ret_target, ret_sigma),
            ('volatility', vol_target, vol_sigma),
        ]:
            scan_label = f"{tf_label}_h{horizon_bars}bar_{target_type}"
            log.info(f"    Scanning [{scan_label}]  "
                     f"n_valid={np.isfinite(target_arr).sum():,}  "
                     f"sigma={target_sigma:.6f}")

            result = walk_forward_evaluate(
                features=features_filtered,
                target=target_arr,
                day_boundaries=day_boundaries_bar,
                feature_names=feature_names_filtered,
                model_type='lgbm',
                min_train_days=min_train_days,
                hour_of_day=hour_of_day_bar,
                lgbm_params=lgbm_params,
            )

            if 'error' in result:
                log.warning(f"    [{scan_label}] FAILED: {result['error']}")
                tf_results.append({
                    'tf_label':      tf_label,
                    'n_per_bar':     n_per_bar,
                    'bar_sec':       n_per_bar * BASE_INTERVAL_MS / 1000,
                    'horizon_bars':  horizon_bars,
                    'horizon_sec':   horizon_bars * n_per_bar * BASE_INTERVAL_MS / 1000,
                    'target_type':   target_type,
                    'scan_label':    scan_label,
                    'error':         result['error'],
                })
                continue

            ic     = result.get('ic', float('nan'))
            icir   = result.get('icir', float('nan'))
            tstat  = result.get('tstat', float('nan'))
            n_preds = result.get('n_preds', 0)
            fold_ics = result.get('fold_ics', [])
            n_folds   = len(fold_ics)
            fold_con  = result.get('fold_con', float('nan'))

            # Top features from last fold
            top_features = result.get('top_features', [])[:10]

            log.info(f"    [{scan_label}] IC={ic:.4f}  ICIR={icir:.2f}  "
                     f"t-stat={tstat:.2f}  n_preds={n_preds:,}  "
                     f"fold_consistency={fold_con:.2f}")

            tf_results.append({
                'tf_label':      tf_label,
                'n_per_bar':     n_per_bar,
                'bar_sec':       n_per_bar * BASE_INTERVAL_MS / 1000,
                'horizon_bars':  horizon_bars,
                'horizon_sec':   horizon_bars * n_per_bar * BASE_INTERVAL_MS / 1000,
                'target_type':   target_type,
                'scan_label':    scan_label,
                'ic':            ic,
                'icir':          icir,
                'tstat':         tstat,
                'n_preds':       n_preds,
                'n_folds':       n_folds,
                'fold_ics':      fold_ics,
                'fold_consistency': fold_con,
                'target_sigma':  target_sigma,
                'ret_sigma_ticks': ret_sigma_ticks if target_type == 'return' else None,
                'top_features':  [(n, float(v)) for n, v in top_features],
            })

        gc.collect()

    return tf_results


# ============================================================================
# PHASE 4: COST-ADJUSTED PNL ANALYSIS
# ============================================================================

def compute_pnl_analysis(
    results: List[dict],
    n_days_total: int,
) -> List[dict]:
    """
    For each (timeframe, return-horizon) result, compute:
      - Signal value in ticks
      - Compare to execution costs
      - Net expected PnL per trade
      - Expected trades per day
      - Expected daily PnL

    Signal value = IC * sigma_target (in ticks)
    This is the expected gross edge per unit of position.

    All cost figures assume:
      - Market order: 1.24 tick round-trip cost
      - Limit entry + market exit: 0.24 tick net cost (entry edge offsets exit)
      - Limit entry + limit exit: -0.76 tick net cost (both legs earn passive fill)
    """
    pnl_results = []

    for r in results:
        if 'error' in r or r.get('target_type') != 'return':
            continue

        ic   = r.get('ic', float('nan'))
        icir = r.get('icir', float('nan'))
        ret_sigma_ticks = r.get('ret_sigma_ticks')

        if ret_sigma_ticks is None or not np.isfinite(ic):
            continue

        # Signal value = IC * sigma(target) in ticks
        # This is E[gross_pnl | top-X% signal] for the average bar
        # For a top-20% signal filter (quantile=0.8), expected IC contribution
        # is roughly IC * sigma_target for those bars.
        # More precisely, for Gaussian: E[signal | top p%] = sigma * phi(z_p) / (1-p)
        # But we use IC * sigma as a conservative proxy.
        signal_value_ticks = float(ic * ret_sigma_ticks)

        # Bars per day at this timeframe
        bar_sec = r.get('bar_sec', 0)
        if bar_sec <= 0:
            continue
        bars_per_day = RTH_SECS_PER_DAY / bar_sec

        # Fraction of bars we'd trade (e.g., top/bottom 30% = 60% of bars)
        # Using 70th-percentile signal filter (same as corrected_limit_study)
        signal_filter = 0.30  # trade top+bottom 30% = 60% of bars
        trades_per_day_estimate = bars_per_day * signal_filter * 2  # both directions

        # Net PnL per trade for each execution mode
        net_mkt_ticks    = signal_value_ticks - COST_MARKET_TOTAL   # market orders
        net_lim_mkt_ticks = signal_value_ticks - COST_LIM_MKT_NET   # limit entry + market exit
        net_lim_lim_ticks = signal_value_ticks - COST_LIM_LIM_NET   # limit + limit (best case)

        # Daily PnL estimates
        daily_pnl_mkt_dollars    = net_mkt_ticks * TICK_VALUE * trades_per_day_estimate
        daily_pnl_lim_mkt_dollars = net_lim_mkt_ticks * TICK_VALUE * trades_per_day_estimate
        daily_pnl_lim_lim_dollars = net_lim_lim_ticks * TICK_VALUE * trades_per_day_estimate

        # Breakeven IC needed for each execution mode
        be_ic_mkt    = COST_MARKET_TOTAL / ret_sigma_ticks if ret_sigma_ticks > 0 else float('nan')
        be_ic_lim_mkt = COST_LIM_MKT_NET / ret_sigma_ticks if ret_sigma_ticks > 0 else float('nan')
        be_ic_lim_lim = COST_LIM_LIM_NET / ret_sigma_ticks if ret_sigma_ticks > 0 else float('nan')

        # Viability assessment
        def assess(net_ticks, be_ic, mode):
            if not np.isfinite(ic) or not np.isfinite(net_ticks):
                return 'UNKNOWN'
            if net_ticks > 0.5:
                return f'VIABLE ({mode})'
            elif net_ticks > 0.1:
                return f'MARGINAL ({mode})'
            elif net_ticks > 0:
                return f'WEAK ({mode})'
            else:
                return f'NO_EDGE ({mode}) needs IC>{be_ic:.3f}'

        viability = {
            'market_order':   assess(net_mkt_ticks, be_ic_mkt, 'mkt'),
            'limit_mkt_exit': assess(net_lim_mkt_ticks, be_ic_lim_mkt, 'lim+mkt'),
            'limit_lim_exit': assess(net_lim_lim_ticks, be_ic_lim_lim, 'lim+lim'),
        }

        pnl_results.append({
            **{k: v for k, v in r.items() if k not in ('fold_ics', 'top_features')},
            'signal_value_ticks':      signal_value_ticks,
            'bars_per_day':            bars_per_day,
            'trades_per_day_estimate': trades_per_day_estimate,

            # Costs
            'cost_mkt_total_ticks':    COST_MARKET_TOTAL,
            'cost_lim_mkt_ticks':      COST_LIM_MKT_NET,
            'cost_lim_lim_ticks':      COST_LIM_LIM_NET,

            # Net edges per trade
            'net_mkt_ticks':           net_mkt_ticks,
            'net_lim_mkt_ticks':       net_lim_mkt_ticks,
            'net_lim_lim_ticks':       net_lim_lim_ticks,

            # Daily PnL (per contract)
            'daily_pnl_mkt_dollars':    daily_pnl_mkt_dollars,
            'daily_pnl_lim_mkt_dollars': daily_pnl_lim_mkt_dollars,
            'daily_pnl_lim_lim_dollars': daily_pnl_lim_lim_dollars,

            # Breakeven ICs
            'breakeven_ic_mkt':         be_ic_mkt,
            'breakeven_ic_lim_mkt':     be_ic_lim_mkt,
            'breakeven_ic_lim_lim':     be_ic_lim_lim,

            # Viability
            'viability':                viability,
        })

    return pnl_results


# ============================================================================
# SCOREBOARD FORMATTER
# ============================================================================

def format_scoreboard(all_results: List[dict], pnl_results: List[dict]) -> str:
    """Format a readable scoreboard of all scan results."""
    lines = [
        "",
        "=" * 90,
        "MULTI-BAR ALPHA SCAN SCOREBOARD",
        "=" * 90,
        "",
        "ALL TIMEFRAME x HORIZON COMBINATIONS (sorted by |IC| descending):",
        "",
        f"{'Label':<30s}  {'IC':>8s}  {'ICIR':>6s}  {'t-stat':>7s}  "
        f"{'sigma_t':>8s}  {'sig_val':>8s}  {'n_preds':>8s}  {'folds':>5s}",
        "-" * 90,
    ]

    # Sort by abs(IC) for return targets
    return_results = [r for r in all_results if r.get('target_type') == 'return'
                      and 'error' not in r]
    vol_results    = [r for r in all_results if r.get('target_type') == 'volatility'
                      and 'error' not in r]

    return_results.sort(key=lambda x: abs(x.get('ic', 0.0)), reverse=True)

    for r in return_results:
        ic   = r.get('ic', float('nan'))
        icir = r.get('icir', float('nan'))
        tstat = r.get('tstat', float('nan'))
        sig_t = r.get('ret_sigma_ticks', float('nan')) or float('nan')
        sig_val = ic * sig_t if np.isfinite(ic) and np.isfinite(sig_t) else float('nan')
        n_preds = r.get('n_preds', 0)
        n_folds = r.get('n_folds', 0)

        ic_str   = f"{ic:>+.4f}" if np.isfinite(ic) else "   NaN  "
        icir_str = f"{icir:>+.2f}" if np.isfinite(icir) else "  NaN "
        ts_str   = f"{tstat:>+.2f}" if np.isfinite(tstat) else "  NaN "
        sigt_str = f"{sig_t:>8.3f}" if np.isfinite(sig_t) else "     NaN"
        sigv_str = f"{sig_val:>+8.4f}t" if np.isfinite(sig_val) else "     NaN "

        lines.append(
            f"{r.get('scan_label', '?'):<30s}  "
            f"{ic_str:>8s}  {icir_str:>6s}  {ts_str:>7s}  "
            f"{sigt_str:>8s}  {sigv_str:>9s}  {n_preds:>8,d}  {n_folds:>5d}"
        )

    lines += ["", "VOLATILITY TARGETS (sorted by |IC|):", "-" * 90]
    vol_results.sort(key=lambda x: abs(x.get('ic', 0.0)), reverse=True)

    for r in vol_results:
        ic    = r.get('ic', float('nan'))
        icir  = r.get('icir', float('nan'))
        tstat = r.get('tstat', float('nan'))
        n_preds = r.get('n_preds', 0)
        n_folds  = r.get('n_folds', 0)
        ic_str   = f"{ic:>+.4f}" if np.isfinite(ic) else "   NaN  "
        icir_str = f"{icir:>+.2f}" if np.isfinite(icir) else "  NaN "
        ts_str   = f"{tstat:>+.2f}" if np.isfinite(tstat) else "  NaN "
        lines.append(
            f"{r.get('scan_label', '?'):<30s}  "
            f"{ic_str:>8s}  {icir_str:>6s}  {ts_str:>7s}  "
            f"{'':>8s}  {'':>9s}  {n_preds:>8,d}  {n_folds:>5d}"
        )

    # Cost-Adjusted PnL Analysis
    lines += [
        "", "=" * 90,
        "COST-ADJUSTED PNL ANALYSIS (return targets only, sorted by net_lim_mkt_ticks):",
        "",
        f"{'Label':<30s}  {'IC':>7s}  {'sigma_t':>8s}  {'sigval_t':>9s}  "
        f"{'net_mkt':>8s}  {'net_lm':>8s}  {'net_ll':>8s}  "
        f"{'$/day(lm)':>10s}  {'trades/d':>9s}",
        "-" * 90,
    ]

    if pnl_results:
        pnl_sorted = sorted(pnl_results,
                            key=lambda x: x.get('net_lim_mkt_ticks', -999),
                            reverse=True)
        for r in pnl_sorted:
            ic    = r.get('ic', float('nan'))
            sig_t = r.get('ret_sigma_ticks', float('nan')) or float('nan')
            sigv  = r.get('signal_value_ticks', float('nan'))
            n_mkt = r.get('net_mkt_ticks', float('nan'))
            n_lm  = r.get('net_lim_mkt_ticks', float('nan'))
            n_ll  = r.get('net_lim_lim_ticks', float('nan'))
            daily = r.get('daily_pnl_lim_mkt_dollars', float('nan'))
            tpd   = r.get('trades_per_day_estimate', float('nan'))

            def fmt(v, fmt_str='+.4f'):
                return f"{v:{fmt_str}}" if np.isfinite(v) else "  NaN  "

            lines.append(
                f"{r.get('scan_label', '?'):<30s}  "
                f"{fmt(ic, '+.4f'):>7s}  "
                f"{fmt(sig_t, '.3f'):>8s}  "
                f"{fmt(sigv, '+.4f')+'t':>9s}  "
                f"{fmt(n_mkt, '+.4f')+'t':>9s}  "
                f"{fmt(n_lm, '+.4f')+'t':>9s}  "
                f"{fmt(n_ll, '+.4f')+'t':>9s}  "
                f"${fmt(daily, '+.2f'):>10s}  "
                f"{fmt(tpd, '.1f'):>9s}"
            )
    else:
        lines.append("  (No valid PnL results)")

    # Verdict section
    viable = [r for r in pnl_results
              if r.get('net_lim_mkt_ticks', -999) > 0.1]
    strong = [r for r in pnl_results
              if r.get('net_lim_mkt_ticks', -999) > 0.5]

    lines += ["", "=" * 90, "VERDICT:", ""]
    if strong:
        lines.append(f"  STRONG EDGE FOUND ({len(strong)} combos with net_lim_mkt > 0.5t):")
        for r in sorted(strong, key=lambda x: x.get('net_lim_mkt_ticks', 0), reverse=True)[:5]:
            lines.append(
                f"    {r['scan_label']}: IC={r['ic']:+.4f}  "
                f"net_lim_mkt={r['net_lim_mkt_ticks']:+.4f}t  "
                f"$/day(lm)=${r['daily_pnl_lim_mkt_dollars']:+.2f}"
            )
    elif viable:
        lines.append(f"  MARGINAL EDGE ({len(viable)} combos with net_lim_mkt > 0.1t):")
        for r in sorted(viable, key=lambda x: x.get('net_lim_mkt_ticks', 0), reverse=True)[:5]:
            lines.append(
                f"    {r['scan_label']}: IC={r['ic']:+.4f}  "
                f"net_lim_mkt={r['net_lim_mkt_ticks']:+.4f}t  "
                f"$/day(lm)=${r['daily_pnl_lim_mkt_dollars']:+.2f}"
            )
    else:
        lines.append("  NO VIABLE EDGES FOUND at any timeframe.")
        lines.append("  Best result (by net_lim_mkt_ticks):")
        if pnl_results:
            best = max(pnl_results, key=lambda x: x.get('net_lim_mkt_ticks', -999))
            lines.append(
                f"    {best['scan_label']}: IC={best['ic']:+.4f}  "
                f"sigma_t={best.get('ret_sigma_ticks', float('nan')):.3f}t  "
                f"signal_val={best.get('signal_value_ticks', float('nan')):+.4f}t  "
                f"net_lim_mkt={best.get('net_lim_mkt_ticks', float('nan')):+.4f}t"
            )
        lines.append("")
        lines.append(
            "  INSIGHT: Longer timeframes produce larger sigma targets but IC may drop."
        )
        lines.append(
            "  If IC degrades proportionally with timeframe, the signal stays insufficient."
        )

    lines += ["", "=" * 90]
    return "\n".join(lines)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Multi-Bar Alpha Discovery Scan for ES Futures'
    )
    parser.add_argument('--n-days', type=int, default=None,
                        help='Number of trading days to use (default: all)')
    parser.add_argument('--fast', action='store_true',
                        help='Fewer LightGBM iterations, faster but less accurate')
    parser.add_argument('--min-train-days', type=int, default=3,
                        help='Minimum training days for walk-forward (default: 3)')
    parser.add_argument('--timeframes', nargs='+',
                        choices=['1s', '5s', '10s', '30s', '1min'],
                        default=None,
                        help='Specific timeframes to scan (default: all)')
    args = parser.parse_args()

    log.info("=" * 80)
    log.info("MULTI-BAR ALPHA DISCOVERY SCAN")
    log.info(f"  n_days:         {args.n_days or 'all'}")
    log.info(f"  fast_mode:      {args.fast}")
    log.info(f"  min_train_days: {args.min_train_days}")
    log.info(f"  timeframes:     {args.timeframes or 'all'}")
    log.info(f"  Execution cost model:")
    log.info(f"    Market order:            {COST_MARKET_TOTAL:.3f} ticks")
    log.info(f"    Limit entry + mkt exit:  {COST_LIM_MKT_NET:.3f} ticks (net)")
    log.info(f"    Limit entry + lim exit:  {COST_LIM_LIM_NET:.3f} ticks (net, best case)")
    log.info("=" * 80)

    t_start = time.time()

    # -----------------------------------------------------------------------
    # PHASE 1: Load data (use feature cache)
    # -----------------------------------------------------------------------
    log.info("\n--- PHASE 1: Load 100ms Base Data ---")

    scanner = MBOAlphaScanner(sample_interval_ms=BASE_INTERVAL_MS)
    stats = load_feature_cache(scanner)
    if stats is None:
        log.info("No feature cache found -- computing from snapshot cache...")
        stats = scanner.load_from_cache(n_days=args.n_days)
    else:
        # If n_days specified and cache has more, truncate
        if args.n_days is not None:
            n_avail = len(scanner.day_boundaries) - 1
            if n_avail > args.n_days:
                log.info(f"Truncating cache to first {args.n_days} days "
                         f"(cache has {n_avail})")
                cut_idx = scanner.day_boundaries[args.n_days]
                scanner.features      = scanner.features[:cut_idx]
                scanner.mid_prices    = scanner.mid_prices[:cut_idx]
                scanner.hour_of_day   = scanner.hour_of_day[:cut_idx]
                scanner.time_since_rth = scanner.time_since_rth[:cut_idx]
                scanner.day_boundaries = scanner.day_boundaries[:args.n_days + 1]

    n_days_total = len(scanner.day_boundaries) - 1
    N_base       = len(scanner.mid_prices)
    log.info(f"Base data: {N_base:,} snapshots @ 100ms, {n_days_total} days")
    log.info(f"Features:  {scanner.features.shape[1]} features")

    # Filter timeframes based on user input
    timeframes_to_scan = [
        (label, n)
        for label, n in TIMEFRAMES
        if args.timeframes is None or label in args.timeframes
    ]
    log.info(f"Timeframes to scan: {[tf[0] for tf in timeframes_to_scan]}")

    # -----------------------------------------------------------------------
    # PHASE 2: Aggregate and scan each timeframe
    # -----------------------------------------------------------------------
    log.info("\n--- PHASE 2: Aggregate and Scan Each Timeframe ---")

    all_results = []
    feature_names_base = scanner.feature_names
    feature_names_bar  = get_bar_feature_names(feature_names_base)

    for tf_label, n_per_bar in timeframes_to_scan:
        bar_sec = n_per_bar * BASE_INTERVAL_MS / 1000.0
        log.info(f"\n{'=' * 60}")
        log.info(f"TIMEFRAME: {tf_label} ({bar_sec:.1f}s/bar, every {n_per_bar}th snapshot)")
        log.info(f"{'=' * 60}")

        t_tf = time.time()

        # Aggregate to this timeframe
        log.info(f"  Aggregating {N_base:,} x 100ms snapshots to {tf_label} bars...")
        features_bar, mid_prices_bar, hour_of_day_bar, day_boundaries_bar = \
            aggregate_to_timeframe(
                features_100ms=scanner.features,
                mid_prices_100ms=scanner.mid_prices,
                day_boundaries_100ms=scanner.day_boundaries,
                n_per_bar=n_per_bar,
                feature_names=feature_names_base,
            )

        M = len(mid_prices_bar)
        n_days_bar = len(day_boundaries_bar) - 1
        log.info(f"  Aggregated: {M:,} bars, {n_days_bar} days  "
                 f"(~{M / max(n_days_bar, 1):.0f} bars/day)")

        if M < 50:
            log.warning(f"  Too few bars ({M}), skipping {tf_label}")
            continue

        # Run walk-forward scan
        tf_results = scan_single_timeframe(
            features_bar=features_bar,
            mid_prices_bar=mid_prices_bar,
            hour_of_day_bar=hour_of_day_bar,
            day_boundaries_bar=day_boundaries_bar,
            feature_names_bar=feature_names_bar,
            tf_label=tf_label,
            n_per_bar=n_per_bar,
            fast=args.fast,
            min_train_days=args.min_train_days,
        )

        all_results.extend(tf_results)

        elapsed_tf = time.time() - t_tf
        log.info(f"  [{tf_label}] Completed in {elapsed_tf:.1f}s "
                 f"({len(tf_results)} scan results)")

        # Free memory
        del features_bar, mid_prices_bar, hour_of_day_bar, day_boundaries_bar
        gc.collect()

    # -----------------------------------------------------------------------
    # PHASE 3: Cost-Adjusted PnL Analysis
    # -----------------------------------------------------------------------
    log.info("\n--- PHASE 3: Cost-Adjusted PnL Analysis ---")

    pnl_results = compute_pnl_analysis(
        results=all_results,
        n_days_total=n_days_total,
    )
    log.info(f"  PnL analysis: {len(pnl_results)} return-target results evaluated")

    # -----------------------------------------------------------------------
    # PHASE 4: Scoreboard and Report
    # -----------------------------------------------------------------------
    log.info("\n--- PHASE 4: Scoreboard ---")

    scoreboard = format_scoreboard(all_results, pnl_results)
    log.info(scoreboard)

    # -----------------------------------------------------------------------
    # Save Results
    # -----------------------------------------------------------------------
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"multibar_scan_{timestamp}.json"

    output = {
        'timestamp':         timestamp,
        'config': {
            'n_days':        args.n_days,
            'fast':          args.fast,
            'min_train_days': args.min_train_days,
            'timeframes':    [tf[0] for tf in timeframes_to_scan],
            'base_interval_ms': BASE_INTERVAL_MS,
            'target_horizons_bars': TARGET_HORIZONS_BARS,
        },
        'data_stats': {
            'n_snapshots':   int(N_base),
            'n_days':        int(n_days_total),
            'mid_price_mean': float(np.nanmean(scanner.mid_prices)),
            'n_features_base': int(scanner.features.shape[1]),
            'n_features_bar':  len(feature_names_bar),
        },
        'cost_model': {
            'tick_size':       TICK_SIZE,
            'tick_value':      TICK_VALUE,
            'cost_mkt_total':  COST_MARKET_TOTAL,
            'cost_lim_mkt':    COST_LIM_MKT_NET,
            'cost_lim_lim':    COST_LIM_LIM_NET,
        },
        'scan_results':   to_safe(all_results),
        'pnl_analysis':   to_safe(pnl_results),
        'scoreboard':     scoreboard,
        'elapsed_sec':    time.time() - t_start,
    }

    with open(str(out_file), 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

    log.info(f"\nResults saved to: {out_file}")
    log.info(f"Total elapsed: {(time.time() - t_start) / 60:.1f} minutes")

    print("\n" + "=" * 80)
    print(scoreboard)
    print(f"\nResults saved to: {out_file}")

    return output


if __name__ == '__main__':
    main()
