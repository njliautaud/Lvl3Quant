"""
Adaptive Trading Strategy Tester for ES Futures
================================================
Tests 4 adaptive approaches on ALL 100 days (Jul 14 - Nov 28, 2025).

Background:
- cancel_asym_chain (combo of cancel_asym_5 + ofi_5) worked in Jul-Sep (low vol)
  but FAILED in Oct-Nov (high vol, +39% volatility jump)
- ofi_5 (order flow imbalance) was positive IC EVERY single day
- Goal: find a strategy with positive PnL in BOTH first-50 and holdout-50 days

Strategies tested:
  1. Vol-Gated cancel_asym_chain
  2. ofi_5 Standalone (pure order flow)
  3. Rolling IC Gate (only trade when signal has recent predictive power)
  4. Vol-Weighted Signal (blend cancel_asym_chain + ofi_5 based on vol)

Trading sim (market orders):
  - TICK = 0.25, TICK_VAL = $12.50, COMM_RT = $3.00 (= 0.24 ticks)
  - Entry: buy at mid + spread/2, sell at mid - spread/2
  - Exit cost: spread/2 per side + commission
  - Cooldown: 10 bars between trades
"""

import sys
import time
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from scipy import stats as scipy_stats

import numpy as np

LVL3_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(LVL3_ROOT))

# Setup logging
_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
RESULTS_DIR = LVL3_ROOT / "alpha_discovery" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(str(RESULTS_DIR / f"adaptive_strategy_{_ts}.log"), mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("adaptive")

# ============================================================================
# CONSTANTS
# ============================================================================

TICK_SIZE    = 0.25
TICK_VALUE   = 12.50
COMMISSION_RT = 4.70  # HC #52: $4.70 RT (AMP)
COMMISSION_TICKS = COMMISSION_RT / TICK_VALUE   # 0.24 ticks
COOLDOWN_BARS = 10

FEATURE_CACHE = LVL3_ROOT / "data" / "processed" / "mbo_features_cache"
SIGNAL_DIR    = LVL3_ROOT / "data" / "processed" / "signal_predictions"

# Feature column indices (confirmed via get_feature_names())
OFI_5_COL          = 96    # ofi_5
CANCEL_ASYM_5_COL  = 158   # cancel_asym_5


# ============================================================================
# FAST VECTORIZED ROLLING UTILITIES
# ============================================================================

def _nan_safe_rolling(x: np.ndarray, window: int):
    """
    NaN-safe vectorized rolling sum, sum-of-squares, and valid-count using cumsum.
    Handles NaN values in input by zeroing them out and tracking valid count.
    Returns (sum_x, sum_x2, n_valid) arrays aligned to output indices [window-1 .. N-1].
    """
    nan_mask = np.isnan(x)
    x_clean  = np.where(nan_mask, 0.0, x)
    x2_clean = np.where(nan_mask, 0.0, x ** 2)

    cum_valid = np.concatenate([[0], np.cumsum(~nan_mask).astype(np.float64)])
    cum_x     = np.concatenate([[0.0], np.cumsum(x_clean)])
    cum_x2    = np.concatenate([[0.0], np.cumsum(x2_clean)])

    n = len(x)
    end_idx   = np.arange(window, n + 1)
    start_idx = end_idx - window

    n_valid = cum_valid[end_idx] - cum_valid[start_idx]
    sum_x   = cum_x[end_idx]    - cum_x[start_idx]
    sum_x2  = cum_x2[end_idx]   - cum_x2[start_idx]

    return sum_x, sum_x2, n_valid


def fast_rolling_std(x: np.ndarray, window: int, min_valid: int = 10) -> np.ndarray:
    """
    Vectorized NaN-safe rolling std. O(N). Returns nan where fewer than
    min_valid samples exist in the window.
    """
    n = len(x)
    out = np.full(n, np.nan)
    if n < window:
        return out

    sum_x, sum_x2, n_valid = _nan_safe_rolling(x, window)

    ok  = n_valid >= min_valid
    nv  = np.where(n_valid > 0, n_valid, 1.0)
    rm  = np.where(ok, sum_x  / nv, np.nan)
    rv  = np.where(ok, sum_x2 / nv - rm ** 2, np.nan)
    rstd = np.where(ok, np.sqrt(np.maximum(rv, 0.0)), np.nan)

    out[window - 1:] = rstd
    return out


def fast_rolling_mean(x: np.ndarray, window: int, min_valid: int = 10) -> np.ndarray:
    """Vectorized NaN-safe rolling mean."""
    n = len(x)
    out = np.full(n, np.nan)
    if n < window:
        return out

    sum_x, _, n_valid = _nan_safe_rolling(x, window)
    ok = n_valid >= min_valid
    nv = np.where(n_valid > 0, n_valid, 1.0)
    out[window - 1:] = np.where(ok, sum_x / nv, np.nan)
    return out


def fast_rolling_zscore(x: np.ndarray, window: int) -> np.ndarray:
    """
    Vectorized NaN-safe rolling z-score: (x - rolling_mean) / rolling_std.
    Returns nan for first (window-1) bars and where input is NaN.
    """
    rm   = fast_rolling_mean(x, window)
    rstd = fast_rolling_std(x, window)

    out = np.full(len(x), np.nan)
    valid = np.isfinite(rm) & np.isfinite(rstd) & np.isfinite(x) & (rstd > 1e-10)
    out[valid] = (x[valid] - rm[valid]) / rstd[valid]
    zero_std = np.isfinite(rm) & np.isfinite(rstd) & np.isfinite(x) & (rstd <= 1e-10)
    out[zero_std] = 0.0
    return out


def fast_rolling_ic(
    signal: np.ndarray,
    future_ret: np.ndarray,
    window: int,
    min_lag: int = 100,
    stride: int = 50,
) -> np.ndarray:
    """
    Rolling IC (Pearson correlation).

    ic[i] = corr(signal[i-window-min_lag : i-min_lag],
                 future_ret[i-window-min_lag : i-min_lag])

    Computed every `stride` bars and interpolated to avoid O(N*W) cost.
    The min_lag gap prevents lookahead.
    """
    n = len(signal)
    ic = np.full(n, np.nan)

    needed = window + min_lag
    # Compute at anchor points
    anchor_ic = {}
    for i in range(needed, n, stride):
        start = i - window - min_lag
        end   = i - min_lag
        s = signal[start:end]
        r = future_ret[start:end]
        valid = np.isfinite(s) & np.isfinite(r)
        if valid.sum() < 30:
            continue
        sv, rv = s[valid], r[valid]
        # Fast Pearson
        sv_dm = sv - sv.mean()
        rv_dm = rv - rv.mean()
        denom = np.sqrt((sv_dm ** 2).sum() * (rv_dm ** 2).sum())
        if denom > 1e-10:
            corr = float((sv_dm * rv_dm).sum() / denom)
            anchor_ic[i] = corr

    # Forward fill between anchors
    if not anchor_ic:
        return ic

    anchors = sorted(anchor_ic.keys())
    for k, idx in enumerate(anchors):
        next_idx = anchors[k + 1] if k + 1 < len(anchors) else n
        ic[idx:next_idx] = anchor_ic[idx]

    return ic


# ============================================================================
# DATA LOADING
# ============================================================================

def discover_days() -> List[str]:
    """Return sorted list of all available date strings."""
    dates = []
    for f in sorted(FEATURE_CACHE.glob("*_mbo_features.npz")):
        date_str = f.stem.replace("_mbo_features", "")
        dates.append(date_str)
    return dates


def load_day(date_str: str, signal_name: str) -> Optional[Dict]:
    """
    Load features + signal predictions for one day.
    Only keeps the columns we need: mid(0), spread(1), ofi_5(96), cancel_asym_5(158).
    """
    feat_path = FEATURE_CACHE / f"{date_str}_mbo_features.npz"
    sig_path  = SIGNAL_DIR    / f"{signal_name}_{date_str}.npz"

    if not feat_path.exists() or not sig_path.exists():
        return None

    feat_data = np.load(str(feat_path))
    feats_full = feat_data['mbo_features']   # (N, 340)

    # Extract only needed columns to minimize memory
    mid        = feats_full[:, 0].copy()
    spread     = feats_full[:, 1].copy()
    ofi_5      = feats_full[:, OFI_5_COL].copy()
    cancel_asym_5 = feats_full[:, CANCEL_ASYM_5_COL].copy()
    # Keep small feats array with just these 4 columns for memory efficiency
    feats = np.column_stack([mid, spread, ofi_5, cancel_asym_5])  # (N, 4)
    del feats_full

    sig_data  = np.load(str(sig_path))
    signal    = sig_data['predictions'].copy()

    n = min(len(mid), len(signal))
    return {
        'date':   date_str,
        'mid':    mid[:n],
        'spread': spread[:n],
        'feats':  feats[:n],    # cols: 0=mid, 1=spread, 2=ofi_5, 3=cancel_asym_5
        'signal': signal[:n],
    }


# ============================================================================
# CORE MARKET ORDER SIMULATOR
# ============================================================================

def simulate_market_orders(
    mid: np.ndarray,
    spread: np.ndarray,
    signal: np.ndarray,
    threshold: float,
    hold_bars: int,
    cooldown_bars: int = COOLDOWN_BARS,
    mask: Optional[np.ndarray] = None,
) -> Dict:
    """
    Market order simulation.

    Entry:
      signal > +threshold  -> BUY  at mid + spread/2
      signal < -threshold  -> SELL at mid - spread/2

    Exit: after hold_bars bars

    mask (optional): boolean array — only allow entries where mask[i] is True.
    """
    n = len(mid)
    if mask is None:
        mask = np.ones(n, dtype=bool)

    trades = []
    position       = 0
    entry_bar      = 0
    entry_price    = 0.0
    entry_dir      = 0
    last_exit_bar  = -cooldown_bars

    for i in range(n):
        if position != 0:
            bars_held = i - entry_bar
            if bars_held >= hold_bars:
                if entry_dir == 1:
                    exit_price  = mid[i] - spread[i] / 2.0
                    gross_pnl   = exit_price - entry_price
                else:
                    exit_price  = mid[i] + spread[i] / 2.0
                    gross_pnl   = entry_price - exit_price

                pnl_ticks   = gross_pnl / TICK_SIZE - COMMISSION_TICKS
                pnl_dollars = pnl_ticks * TICK_VALUE

                trades.append({'entry_bar': entry_bar, 'exit_bar': i,
                                'direction': entry_dir, 'pnl_dollars': pnl_dollars,
                                'pnl_ticks': pnl_ticks, 'bars_held': bars_held})
                position      = 0
                last_exit_bar = i
        else:
            if i - last_exit_bar < cooldown_bars:
                continue
            if not mask[i]:
                continue

            sig = signal[i]
            if np.isfinite(sig):
                if sig > threshold:
                    entry_price = mid[i] + spread[i] / 2.0
                    entry_dir   = 1
                    position    = 1
                    entry_bar   = i
                elif sig < -threshold:
                    entry_price = mid[i] - spread[i] / 2.0
                    entry_dir   = -1
                    position    = 1
                    entry_bar   = i

    # Force close end-of-day
    if position != 0:
        i = n - 1
        if entry_dir == 1:
            exit_price = mid[i] - spread[i] / 2.0
            gross_pnl  = exit_price - entry_price
        else:
            exit_price = mid[i] + spread[i] / 2.0
            gross_pnl  = entry_price - exit_price

        pnl_ticks   = gross_pnl / TICK_SIZE - COMMISSION_TICKS
        pnl_dollars = pnl_ticks * TICK_VALUE
        trades.append({'entry_bar': entry_bar, 'exit_bar': i,
                       'direction': entry_dir, 'pnl_dollars': pnl_dollars,
                       'pnl_ticks': pnl_ticks, 'bars_held': i - entry_bar})

    if not trades:
        return {'n_trades': 0, 'pnl_dollars': 0.0, 'win_rate': 0.0,
                'pnl_per_bar': 0.0, 'trades': []}

    pnl_list  = [t['pnl_dollars'] for t in trades]
    wins      = sum(1 for p in pnl_list if p > 0)
    total_pnl = sum(pnl_list)

    return {
        'n_trades': len(trades),
        'pnl_dollars': total_pnl,
        'win_rate': wins / len(trades),
        'pnl_per_bar': total_pnl / n,
        'trades': trades,
    }


# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

def compute_stats(daily_pnls: List[float], label: str = "") -> Dict:
    if not daily_pnls:
        return {}
    arr = np.array(daily_pnls, dtype=float)
    mean_pnl = float(np.mean(arr))
    med_pnl  = float(np.median(arr))
    std_pnl  = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    sharpe   = (mean_pnl / std_pnl * np.sqrt(252)) if std_pnl > 1e-6 else 0.0
    pct_pos  = float(np.mean(arr > 0))

    if len(arr) > 1 and std_pnl > 1e-6:
        t_stat, p_val = scipy_stats.ttest_1samp(arr, 0.0)
    else:
        t_stat, p_val = 0.0, 1.0

    return {
        'label': label, 'n_days': len(arr),
        'mean_pnl': mean_pnl, 'median_pnl': med_pnl,
        'std_pnl': std_pnl, 'sharpe_annual': sharpe,
        'pct_positive': pct_pos, 't_stat': t_stat, 'p_value': p_val,
    }


def split_stats(daily_pnls: List[float], n_first: int = 50) -> Dict:
    return {
        'first_50':   compute_stats(daily_pnls[:n_first], 'first_50'),
        'holdout_50': compute_stats(daily_pnls[n_first:], 'holdout_50'),
        'all_100':    compute_stats(daily_pnls, 'all_100'),
    }


def both_positive(split: Dict) -> bool:
    return (split['first_50'].get('mean_pnl', -1) > 0 and
            split['holdout_50'].get('mean_pnl', -1) > 0)


def fmt(s: Dict) -> str:
    return (f"mean=${s['mean_pnl']:+.1f} med=${s['median_pnl']:+.1f} "
            f"Sharpe={s['sharpe_annual']:.2f} pos%={s['pct_positive']:.0%} "
            f"t={s['t_stat']:.2f} p={s['p_value']:.3f}")


def print_result(name: str, params: str, split: Dict, n_trades: int):
    flag = "  *** BOTH POSITIVE ***" if both_positive(split) else ""
    log.info(
        f"\n{'='*68}\n"
        f"STRATEGY: {name} | {params}{flag}\n"
        f"  FIRST-50:   {fmt(split['first_50'])}\n"
        f"  HOLDOUT-50: {fmt(split['holdout_50'])}\n"
        f"  ALL-100:    {fmt(split['all_100'])} | trades={n_trades}\n"
        f"{'='*68}"
    )


# ============================================================================
# STRATEGY 1: VOL-GATED CANCEL_ASYM_CHAIN
# ============================================================================

def strategy1_vol_gated(
    all_data: List[Dict],
    vol_percentile: float,
    hold_bars: int = 3000,
    threshold: float = 3.5,
) -> Tuple[List[float], List[int]]:
    """
    Vol-Gated cancel_asym_chain.
    Only allow trades when rolling 5000-bar vol < vol_percentile of this day's vol.
    Uses vectorized rolling std.
    """
    daily_pnls, daily_trades = [], []

    for d in all_data:
        if d is None:
            daily_pnls.append(0.0); daily_trades.append(0); continue

        mid, spread, signal = d['mid'], d['spread'], d['signal']
        n = len(mid)

        # 1-bar returns in ticks
        rets = np.zeros(n)
        rets[1:] = (mid[1:] - mid[:-1]) / TICK_SIZE

        # Rolling 5000-bar vol (vectorized)
        rvol = fast_rolling_std(rets, 5000)

        # Vol threshold: percentile of this day's finite vol values
        valid_vol = rvol[np.isfinite(rvol)]
        if len(valid_vol) < 10:
            daily_pnls.append(0.0); daily_trades.append(0); continue

        vol_thresh = np.percentile(valid_vol, vol_percentile * 100)
        mask = np.isfinite(rvol) & (rvol < vol_thresh)

        result = simulate_market_orders(mid, spread, signal,
                                        threshold=threshold, hold_bars=hold_bars, mask=mask)
        daily_pnls.append(result['pnl_dollars'])
        daily_trades.append(result['n_trades'])

    return daily_pnls, daily_trades


# ============================================================================
# STRATEGY 2: OFI_5 STANDALONE
# ============================================================================

def strategy2_ofi5_standalone(
    all_data: List[Dict],
    threshold: float,
    hold_bars: int,
    zscore_window: int = 5000,
) -> Tuple[List[float], List[int]]:
    """
    ofi_5 standalone signal with rolling z-score normalization.
    ofi_5 is feature column 96. Uses vectorized rolling z-score.
    """
    daily_pnls, daily_trades = [], []

    for d in all_data:
        if d is None:
            daily_pnls.append(0.0); daily_trades.append(0); continue

        mid, spread, feats = d['mid'], d['spread'], d['feats']
        # feats cols: 0=mid, 1=spread, 2=ofi_5, 3=cancel_asym_5
        raw_ofi = feats[:, 2].copy()

        # Rolling z-score normalization (vectorized)
        ofi_z = fast_rolling_zscore(raw_ofi, zscore_window)
        mask  = np.isfinite(ofi_z)

        result = simulate_market_orders(mid, spread, ofi_z,
                                        threshold=threshold, hold_bars=hold_bars, mask=mask)
        daily_pnls.append(result['pnl_dollars'])
        daily_trades.append(result['n_trades'])

    return daily_pnls, daily_trades


# ============================================================================
# STRATEGY 3: ROLLING IC GATE
# ============================================================================

def strategy3_rolling_ic_gate(
    all_data: List[Dict],
    ic_threshold: float,
    hold_bars: int = 3000,
    threshold: float = 3.5,
    ic_window: int = 2000,
    ic_min_lag: int = 100,
) -> Tuple[List[float], List[int]]:
    """
    Rolling IC Gate on cancel_asym_chain signal.
    Only trade when rolling IC of signal vs realized returns > ic_threshold.
    """
    daily_pnls, daily_trades = [], []

    for d in all_data:
        if d is None:
            daily_pnls.append(0.0); daily_trades.append(0); continue

        mid, spread, signal = d['mid'], d['spread'], d['signal']
        n = len(mid)

        # Realized 1-bar forward returns
        future_ret = np.zeros(n)
        future_ret[:-1] = (mid[1:] - mid[:-1]) / TICK_SIZE
        future_ret[-1]  = 0.0

        # Rolling IC (with stride for speed)
        ic_arr = fast_rolling_ic(signal, future_ret, window=ic_window,
                                  min_lag=ic_min_lag, stride=100)
        mask = np.isfinite(ic_arr) & (ic_arr > ic_threshold)

        result = simulate_market_orders(mid, spread, signal,
                                        threshold=threshold, hold_bars=hold_bars, mask=mask)
        daily_pnls.append(result['pnl_dollars'])
        daily_trades.append(result['n_trades'])

    return daily_pnls, daily_trades


# ============================================================================
# STRATEGY 4: VOL-WEIGHTED SIGNAL BLEND
# ============================================================================

def strategy4_vol_weighted(
    all_data: List[Dict],
    vol_threshold_pct: float,
    threshold: float = 1.5,
    hold_bars: int = 1200,
    zscore_window: int = 5000,
) -> Tuple[List[float], List[int]]:
    """
    Vol-Weighted Signal Blend:
      w_cancel = max(0, 1 - rolling_vol / vol_threshold)
      signal   = w_cancel * (-0.5 * cancel_asym_5_z) + 0.5 * ofi_5_z

    In low vol:  cancel_asym_5 contributes fully
    In high vol: only ofi_5 contributes
    """
    daily_pnls, daily_trades = [], []

    for d in all_data:
        if d is None:
            daily_pnls.append(0.0); daily_trades.append(0); continue

        mid, spread, feats = d['mid'], d['spread'], d['feats']
        n = len(mid)

        # feats cols: 0=mid, 1=spread, 2=ofi_5, 3=cancel_asym_5
        raw_ofi    = feats[:, 2].copy()
        raw_cancel = feats[:, 3].copy()

        # 1-bar returns for vol
        rets = np.zeros(n)
        rets[1:] = (mid[1:] - mid[:-1]) / TICK_SIZE

        # Vectorized rolling computations
        ofi_z    = fast_rolling_zscore(raw_ofi,    zscore_window)
        cancel_z = fast_rolling_zscore(raw_cancel, zscore_window)
        rvol     = fast_rolling_std(rets,           zscore_window)

        # Vol threshold from this day
        valid_vol = rvol[np.isfinite(rvol)]
        if len(valid_vol) < 10:
            daily_pnls.append(0.0); daily_trades.append(0); continue

        vol_thresh = np.percentile(valid_vol, vol_threshold_pct * 100)

        # Blend signal (vectorized)
        valid = np.isfinite(rvol) & np.isfinite(ofi_z) & np.isfinite(cancel_z)
        blend = np.full(n, np.nan)

        idx = np.where(valid)[0]
        w_cancel = np.maximum(0.0, 1.0 - rvol[idx] / (vol_thresh + 1e-10))
        blend[idx] = w_cancel * (-0.5 * cancel_z[idx]) + 0.5 * ofi_z[idx]

        mask = np.isfinite(blend)

        result = simulate_market_orders(mid, spread, blend,
                                        threshold=threshold, hold_bars=hold_bars, mask=mask)
        daily_pnls.append(result['pnl_dollars'])
        daily_trades.append(result['n_trades'])

    return daily_pnls, daily_trades


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0 = time.time()
    log.info("=" * 68)
    log.info("ADAPTIVE STRATEGY TESTER - ES Futures, 100 Days")
    log.info("=" * 68)

    # ------------------------------------------------------------------
    # Load all 100 days
    # ------------------------------------------------------------------
    all_dates = discover_days()
    n_days    = len(all_dates)
    log.info(f"Found {n_days} days: {all_dates[0]} to {all_dates[-1]}")

    log.info("Loading all 100 days of features + signal predictions...")
    t_load = time.time()
    all_data = []
    for date_str in all_dates:
        d = load_day(date_str, 'cancel_asym_chain')
        all_data.append(d)
    loaded = sum(1 for d in all_data if d is not None)
    log.info(f"Loaded {loaded}/{n_days} days in {time.time()-t_load:.1f}s")

    results_summary = []
    best_candidates = []

    # ======================================================================
    # STRATEGY 1: VOL-GATED CANCEL_ASYM_CHAIN
    # ======================================================================
    log.info("\n" + "=" * 68)
    log.info("STRATEGY 1: Vol-Gated cancel_asym_chain")
    log.info("=" * 68)

    for vol_pct in [0.25, 0.50, 0.75]:
        t1 = time.time()
        pnls, trades = strategy1_vol_gated(all_data, vol_percentile=vol_pct,
                                           hold_bars=3000, threshold=3.5)
        log.info(f"  S1 vol_pct={vol_pct:.0%} done in {time.time()-t1:.1f}s")
        split = split_stats(pnls)
        n_tr  = sum(trades)
        label = f"s1_vol{int(vol_pct*100)}pct"
        params = f"vol_pct={vol_pct:.0%} hold=3000 thresh=3.5"
        print_result("S1 Vol-Gated", params, split, n_tr)
        results_summary.append({'strategy': label, 'params': params, 'split': split, 'n_trades': n_tr})
        if both_positive(split):
            best_candidates.append((label, params, split))

    # ======================================================================
    # STRATEGY 2: OFI_5 STANDALONE
    # ======================================================================
    log.info("\n" + "=" * 68)
    log.info("STRATEGY 2: ofi_5 Standalone")
    log.info("=" * 68)

    thresholds     = [1.5, 2.0, 2.5, 3.0, 3.5]
    hold_bars_list = [100, 300, 600, 1200, 3000]

    for thresh in thresholds:
        for hold in hold_bars_list:
            pnls, trades = strategy2_ofi5_standalone(all_data, threshold=thresh, hold_bars=hold)
            split = split_stats(pnls)
            n_tr  = sum(trades)
            label = f"s2_ofi5_t{thresh}_h{hold}"
            params = f"thresh={thresh} hold={hold}"
            print_result("S2 ofi_5", params, split, n_tr)
            results_summary.append({'strategy': label, 'params': params, 'split': split, 'n_trades': n_tr})
            if both_positive(split):
                best_candidates.append((label, params, split))

    log.info(f"  Strategy 2 complete: {len(thresholds)*len(hold_bars_list)} combos tested")

    # ======================================================================
    # STRATEGY 3: ROLLING IC GATE
    # ======================================================================
    log.info("\n" + "=" * 68)
    log.info("STRATEGY 3: Rolling IC Gate")
    log.info("=" * 68)

    ic_thresholds = [0.01, 0.02, 0.03, 0.05]
    for ic_thresh in ic_thresholds:
        t1 = time.time()
        pnls, trades = strategy3_rolling_ic_gate(all_data, ic_threshold=ic_thresh,
                                                  hold_bars=3000, threshold=3.5)
        log.info(f"  S3 ic_thresh={ic_thresh} done in {time.time()-t1:.1f}s")
        split = split_stats(pnls)
        n_tr  = sum(trades)
        label = f"s3_icgate_ic{ic_thresh}"
        params = f"ic_thresh={ic_thresh} hold=3000 thresh=3.5"
        print_result("S3 Rolling IC Gate", params, split, n_tr)
        results_summary.append({'strategy': label, 'params': params, 'split': split, 'n_trades': n_tr})
        if both_positive(split):
            best_candidates.append((label, params, split))

    # ======================================================================
    # STRATEGY 4: VOL-WEIGHTED SIGNAL BLEND
    # ======================================================================
    log.info("\n" + "=" * 68)
    log.info("STRATEGY 4: Vol-Weighted Signal Blend")
    log.info("=" * 68)

    vol_pcts    = [0.25, 0.50, 0.75]
    thresholds4 = [1.0, 1.5, 2.0]
    holds4      = [600, 1200, 3000]

    for vol_pct in vol_pcts:
        for thresh in thresholds4:
            for hold in holds4:
                pnls, trades = strategy4_vol_weighted(all_data, vol_threshold_pct=vol_pct,
                                                       threshold=thresh, hold_bars=hold)
                split = split_stats(pnls)
                n_tr  = sum(trades)
                label = f"s4_vp{int(vol_pct*100)}_t{thresh}_h{hold}"
                params = f"vol_pct={vol_pct:.0%} thresh={thresh} hold={hold}"
                print_result("S4 Vol-Weighted", params, split, n_tr)
                results_summary.append({'strategy': label, 'params': params, 'split': split, 'n_trades': n_tr})
                if both_positive(split):
                    best_candidates.append((label, params, split))

    log.info(f"  Strategy 4 complete: {len(vol_pcts)*len(thresholds4)*len(holds4)} combos tested")

    # ======================================================================
    # FINAL SUMMARY
    # ======================================================================
    elapsed = time.time() - t0

    log.info("\n" + "=" * 68)
    log.info(f"FINAL SUMMARY - {len(results_summary)} combos tested in {elapsed:.1f}s")
    log.info("=" * 68)

    sorted_results = sorted(results_summary,
                             key=lambda x: x['split']['all_100'].get('mean_pnl', -9999),
                             reverse=True)

    log.info("\nTOP 10 by all-100-day mean PnL:")
    for r in sorted_results[:10]:
        s_all = r['split']['all_100']
        s1    = r['split']['first_50']
        s2    = r['split']['holdout_50']
        flag  = "  *** BOTH POSITIVE ***" if both_positive(r['split']) else ""
        log.info(
            f"  {r['strategy']}: all=${s_all['mean_pnl']:+.1f}/day "
            f"(f50=${s1['mean_pnl']:+.1f}, h50=${s2['mean_pnl']:+.1f}) "
            f"Sharpe={s_all['sharpe_annual']:.2f} trades={r['n_trades']}{flag}"
        )

    if best_candidates:
        log.info(f"\n{'*'*68}")
        log.info(f"*** {len(best_candidates)} STRATEGIES WITH POSITIVE PNL IN BOTH PERIODS ***")
        log.info(f"{'*'*68}")
        for label, params, split in best_candidates:
            s1    = split['first_50']
            s2    = split['holdout_50']
            s_all = split['all_100']
            log.info(
                f"\n  STRATEGY: {label} | {params}\n"
                f"  FIRST-50:   mean=${s1['mean_pnl']:+.2f} Sharpe={s1['sharpe_annual']:.2f} "
                f"pos%={s1['pct_positive']:.0%} t={s1['t_stat']:.2f} p={s1['p_value']:.3f}\n"
                f"  HOLDOUT-50: mean=${s2['mean_pnl']:+.2f} Sharpe={s2['sharpe_annual']:.2f} "
                f"pos%={s2['pct_positive']:.0%} t={s2['t_stat']:.2f} p={s2['p_value']:.3f}\n"
                f"  ALL-100:    mean=${s_all['mean_pnl']:+.2f} Sharpe={s_all['sharpe_annual']:.2f} "
                f"pos%={s_all['pct_positive']:.0%} t={s_all['t_stat']:.2f} p={s_all['p_value']:.3f}"
            )
    else:
        log.info("\nNO strategies showed positive PnL in BOTH periods.")
        log.info("Best single-period results shown in top-10 above.")

    # Save JSON results
    out_path = RESULTS_DIR / f"adaptive_strategy_results_{_ts}.json"
    def ser(obj):
        if isinstance(obj, dict):  return {k: ser(v) for k, v in obj.items()}
        if isinstance(obj, list):  return [ser(v) for v in obj]
        if isinstance(obj, (np.floating, float)): return float(obj)
        if isinstance(obj, (np.integer, int)):    return int(obj)
        return obj

    with open(str(out_path), 'w') as f:
        json.dump(ser({'results': results_summary, 'best_candidates': [
            {'label': l, 'params': p, 'split': s} for l, p, s in best_candidates
        ]}), f, indent=2)
    log.info(f"\nResults saved: {out_path}")

    # ---- Build Discord summary ----
    n_both = len(best_candidates)
    discord_lines = [
        "**Adaptive Strategy Results - ES Futures (100 days)**",
        f"Tested {len(results_summary)} strategy/param combos in {elapsed:.0f}s",
        "",
        f"**Strategies with POSITIVE PnL in BOTH periods: {n_both}**",
    ]
    if best_candidates:
        for label, params, split in best_candidates[:8]:
            s1    = split['first_50']
            s2    = split['holdout_50']
            s_all = split['all_100']
            discord_lines.append(
                f"  [BOTH+] `{label}` [{params}]"
                f" f50=${s1['mean_pnl']:+.1f}/d (Sharpe={s1['sharpe_annual']:.2f}, t={s1['t_stat']:.2f})"
                f" h50=${s2['mean_pnl']:+.1f}/d (Sharpe={s2['sharpe_annual']:.2f}, t={s2['t_stat']:.2f})"
                f" all=${s_all['mean_pnl']:+.1f}/d"
            )
    else:
        discord_lines.append("  No strategy positive in both periods.")

    discord_lines += ["", "**Top 10 by all-100-day mean PnL:**"]
    for r in sorted_results[:10]:
        s_all = r['split']['all_100']
        s1    = r['split']['first_50']
        s2    = r['split']['holdout_50']
        flag  = " [BOTH+]" if both_positive(r['split']) else ""
        discord_lines.append(
            f"  `{r['strategy']}` ${s_all['mean_pnl']:+.1f}/d "
            f"Sharpe={s_all['sharpe_annual']:.2f} "
            f"(f50=${s1['mean_pnl']:+.1f}, h50=${s2['mean_pnl']:+.1f}){flag}"
        )

    discord_lines += ["", "**Bottom 3:**"]
    for r in sorted_results[-3:]:
        s_all = r['split']['all_100']
        discord_lines.append(
            f"  `{r['strategy']}` ${s_all['mean_pnl']:+.1f}/d Sharpe={s_all['sharpe_annual']:.2f}"
        )

    discord_msg = "\n".join(discord_lines)
    log.info("\nDISCORD MESSAGE:\n" + discord_msg)

    return discord_msg, best_candidates, sorted_results


if __name__ == '__main__':
    try:
        discord_msg, best_candidates, sorted_results = main()
        print("\n--- DISCORD SUMMARY ---")
        print(discord_msg)
    except Exception as e:
        import traceback
        err = traceback.format_exc()
        log.error(f"FATAL ERROR: {e}\n{err}")
        print(f"ERROR: {e}\n{err}")
        sys.exit(1)
