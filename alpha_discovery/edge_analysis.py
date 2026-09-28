"""
Edge Analysis — WHEN and WHY the MFE Prediction Model Works Best

Analyzes a saved predictions NPZ file (same format as mfe_path_analysis.py) to
decompose model performance across time, volatility, and signal-strength dimensions.

Analyses performed:
  A. Per-Day IC Decomposition        — which days drive the signal?
  B. Time-of-Day IC Curve            — 30-min buckets across the ES session
  C. Volatility Conditioning         — does the model work better in high/low vol?
  D. Signal Strength vs. Outcome     — do stronger predictions actually deliver?
  E. Market Order Sim (ultra-high conviction) — can MKT orders work at the extremes?

NPZ format (produced by mfe_path_analysis.py or magnitude_gated_sim.py):
    mid_prices        float32[N]   mid-price series (all days concatenated)
    direction_preds   float32[N]   direction model prediction (NaN where no pred)
    magnitude_preds   float32[N]   magnitude model prediction in ticks (NaN where no pred)
    direction_target  float32[N]   direction target (mfe_net_*s)
    magnitude_target  float32[N]   magnitude target (abs ticks to horizon)
    day_boundaries    int32[D+1]   start/end indices for each day

Usage:
    python alpha_discovery/edge_analysis.py \\
        --load-predictions results/predictions_mfe_path_TIMESTAMP.npz \\
        --n-days 27

    # With IS/OOS split:
    python alpha_discovery/edge_analysis.py \\
        --load-predictions results/predictions_mfe_path_TIMESTAMP.npz \\
        --oos-split-day 20

    # Run on server (no GPU needed):
    python alpha_discovery/edge_analysis.py \\
        --load-predictions /path/to/predictions.npz \\
        --n-days 100
"""

import gc
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
LVL3_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(LVL3_ROOT))

RESULTS_DIR = LVL3_ROOT / "alpha_discovery" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging — identical convention to mfe_path_analysis.py
# ---------------------------------------------------------------------------
_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
_log_file = RESULTS_DIR / f"edge_analysis_{_ts}.log"
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
logger = logging.getLogger('edge_analysis')

# ---------------------------------------------------------------------------
# Constants (ES futures, 100ms bars)
# ---------------------------------------------------------------------------
TICK_SIZE   = 0.25          # ES tick size in points
TICK_VALUE  = 12.50         # $ per tick (ES full contract)
COMMISSION_RT = 3.10        # $3.10 round-trip
COMMISSION_TICKS = COMMISSION_RT / TICK_VALUE   # 0.248 ticks
HALF_TICK   = TICK_SIZE / 2                      # 0.125 (half spread)
BARS_PER_SEC = 10           # 100ms resolution
BARS_PER_MIN = 600          # 60 × 10
BARS_PER_HALF_HOUR = 18_000 # 30 min × 60 s × 10 bars/s

# ES regular trading hours: 9:30 AM – 4:00 PM ET = 6.5 hours = 13 half-hour buckets
ES_SESSION_HOURS = 6.5
BUCKET_COUNT = 13           # 13 × 30-min = 6.5 hours
BARS_PER_DAY = 234_000      # 6.5h × 3600s/h × 10 bars/s
BUCKET_LABELS = [
    "09:30", "10:00", "10:30", "11:00", "11:30",
    "12:00", "12:30", "13:00", "13:30", "14:00",
    "14:30", "15:00", "15:30",
]

# Trade simulation parameters (Section E) — honest cost decomposition
# ES has 1-tick spread ($0.25 = 1t). For a 1-lot order, slippage ≈ 0.
ENTRY_SPREAD_TICKS    = 0.5    # cross half-spread on market entry
EXIT_SPREAD_TICKS     = 0.5    # cross half-spread on market exit
COMMISSION_TICKS_RT   = 0.248  # $3.10 RT / $12.50 per tick
# Cost scenarios (all in ticks, round-trip):
COST_MKT_IN_MKT_OUT   = ENTRY_SPREAD_TICKS + EXIT_SPREAD_TICKS + COMMISSION_TICKS_RT  # 1.248t
COST_MKT_IN_LMT_OUT   = ENTRY_SPREAD_TICKS + COMMISSION_TICKS_RT                      # 0.748t
COST_LMT_IN_MKT_OUT   = EXIT_SPREAD_TICKS + COMMISSION_TICKS_RT                       # 0.748t
COST_LMT_IN_LMT_OUT   = COMMISSION_TICKS_RT                                            # 0.248t
# Sim defaults
MFE_HORIZON_BARS      = 100    # 10 seconds lookahead
MAG_THRESHOLD_MKT     = 3.0    # magnitude prediction > 3 ticks (ultra-high conviction)
DIR_QUANTILE_MKT      = 0.90   # top 10% direction signal
# TP targets to sweep
TP_TARGETS            = [2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
# Horizons to test (bars)
SIM_HORIZONS          = [100, 200, 300, 600]  # 10s, 20s, 30s, 60s

QUINTILE_COUNT = 5


# ===========================================================================
# Helpers
# ===========================================================================

def _safe_ic(pred: np.ndarray, target: np.ndarray) -> float:
    """Spearman IC between pred and target, NaN-safe. Returns 0.0 if <10 valid pairs."""
    mask = np.isfinite(pred) & np.isfinite(target)
    if mask.sum() < 10:
        return 0.0
    try:
        r, _ = spearmanr(pred[mask], target[mask])
        return float(r) if np.isfinite(r) else 0.0
    except Exception:
        return 0.0


def _safe_mean(arr: np.ndarray) -> float:
    v = arr[np.isfinite(arr)]
    return float(v.mean()) if len(v) > 0 else 0.0


def _profit_factor(pnl_arr: np.ndarray) -> float:
    gross_profit = float(pnl_arr[pnl_arr > 0].sum()) if (pnl_arr > 0).any() else 0.0
    gross_loss   = float(abs(pnl_arr[pnl_arr < 0].sum())) if (pnl_arr < 0).any() else 1e-9
    return gross_profit / gross_loss


# ===========================================================================
# Data loading
# ===========================================================================

def load_predictions(path: str, n_days: Optional[int] = None) -> Dict:
    """
    Load predictions NPZ and optionally trim to n_days.
    Returns dict with keys: mid_prices, direction_preds, magnitude_preds,
    direction_target, magnitude_target, day_boundaries (list), n_days.
    """
    logger.info(f"Loading predictions from: {path}")
    data = np.load(path)

    mid_prices       = data['mid_prices'].astype(np.float32)
    direction_preds  = data['direction_preds'].astype(np.float32)
    magnitude_preds  = data['magnitude_preds'].astype(np.float32)
    direction_target = data['direction_target'].astype(np.float32)
    magnitude_target = data['magnitude_target'].astype(np.float32)
    day_boundaries   = data['day_boundaries'].tolist()

    total_days = len(day_boundaries) - 1
    logger.info(f"  Raw: {len(mid_prices):,} bars, {total_days} days")
    logger.info(f"  Direction preds valid: {np.isfinite(direction_preds).sum():,}")
    logger.info(f"  Magnitude preds valid: {np.isfinite(magnitude_preds).sum():,}")

    if n_days and total_days > n_days:
        cut = day_boundaries[n_days]
        mid_prices       = mid_prices[:cut]
        direction_preds  = direction_preds[:cut]
        magnitude_preds  = magnitude_preds[:cut]
        direction_target = direction_target[:cut]
        magnitude_target = magnitude_target[:cut]
        day_boundaries   = day_boundaries[:n_days + 1]
        logger.info(f"  Trimmed to {n_days} days ({len(mid_prices):,} bars)")

    return {
        'mid_prices':       mid_prices,
        'direction_preds':  direction_preds,
        'magnitude_preds':  magnitude_preds,
        'direction_target': direction_target,
        'magnitude_target': magnitude_target,
        'day_boundaries':   day_boundaries,
        'n_days':           len(day_boundaries) - 1,
    }


# ===========================================================================
# A. Per-Day IC Decomposition
# ===========================================================================

def analyze_per_day_ic(data: Dict) -> Dict:
    """
    Compute direction IC and magnitude IC for each day.
    Identifies best/worst days and any systematic patterns.
    """
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS A: PER-DAY IC DECOMPOSITION")
    logger.info("=" * 60)

    day_boundaries   = data['day_boundaries']
    direction_preds  = data['direction_preds']
    magnitude_preds  = data['magnitude_preds']
    direction_target = data['direction_target']
    magnitude_target = data['magnitude_target']
    n_days           = data['n_days']

    dir_ics  = []
    mag_ics  = []
    n_valids = []

    for d in range(n_days):
        s = day_boundaries[d]
        e = day_boundaries[d + 1]

        dp = direction_preds[s:e]
        dt = direction_target[s:e]
        mp = magnitude_preds[s:e]
        mt = magnitude_target[s:e]

        n_valid = int((np.isfinite(dp) & np.isfinite(dt)).sum())
        dir_ic  = _safe_ic(dp, dt)
        mag_ic  = _safe_ic(mp, mt)

        dir_ics.append(dir_ic)
        mag_ics.append(mag_ic)
        n_valids.append(n_valid)

    dir_ics_arr = np.array(dir_ics)
    mag_ics_arr = np.array(mag_ics)

    # Sort days by direction IC
    sorted_idx_dir = np.argsort(dir_ics_arr)
    worst5_dir = sorted_idx_dir[:5].tolist()
    best5_dir  = sorted_idx_dir[-5:][::-1].tolist()

    logger.info(f"\n  {'Day':>4s}  {'N_valid':>8s}  {'Dir_IC':>8s}  {'Mag_IC':>8s}")
    logger.info(f"  {'-'*40}")
    for d in range(n_days):
        flag = ""
        if d in best5_dir:
            flag = " <-- BEST"
        elif d in worst5_dir:
            flag = " <-- WORST"
        logger.info(f"  {d:>4d}  {n_valids[d]:>8,d}  {dir_ics[d]:>+8.4f}  {mag_ics[d]:>+8.4f}{flag}")

    logger.info(f"\n  DIRECTION IC SUMMARY:")
    logger.info(f"    Mean:   {dir_ics_arr.mean():+.4f}")
    logger.info(f"    Median: {float(np.median(dir_ics_arr)):+.4f}")
    logger.info(f"    Std:    {dir_ics_arr.std():.4f}")
    logger.info(f"    Min:    {dir_ics_arr.min():+.4f} (day {int(dir_ics_arr.argmin())})")
    logger.info(f"    Max:    {dir_ics_arr.max():+.4f} (day {int(dir_ics_arr.argmax())})")
    logger.info(f"    ICIR:   {dir_ics_arr.mean() / max(dir_ics_arr.std(), 1e-9):.2f}")
    logger.info(f"    % days positive: {(dir_ics_arr > 0).mean():.1%}")

    logger.info(f"\n  MAGNITUDE IC SUMMARY:")
    logger.info(f"    Mean:   {mag_ics_arr.mean():+.4f}")
    logger.info(f"    Median: {float(np.median(mag_ics_arr)):+.4f}")
    logger.info(f"    Std:    {mag_ics_arr.std():.4f}")
    logger.info(f"    Min:    {mag_ics_arr.min():+.4f} (day {int(mag_ics_arr.argmin())})")
    logger.info(f"    Max:    {mag_ics_arr.max():+.4f} (day {int(mag_ics_arr.argmax())})")
    logger.info(f"    ICIR:   {mag_ics_arr.mean() / max(mag_ics_arr.std(), 1e-9):.2f}")

    # Best/worst day details
    logger.info(f"\n  BEST 5 DAYS (direction IC):")
    for d in best5_dir:
        logger.info(f"    Day {d:>3d}: Dir IC={dir_ics[d]:+.4f}  Mag IC={mag_ics[d]:+.4f}  N={n_valids[d]:,}")

    logger.info(f"\n  WORST 5 DAYS (direction IC):")
    for d in worst5_dir:
        logger.info(f"    Day {d:>3d}: Dir IC={dir_ics[d]:+.4f}  Mag IC={mag_ics[d]:+.4f}  N={n_valids[d]:,}")

    return {
        'dir_ics':            dir_ics,
        'mag_ics':            mag_ics,
        'n_valids':           n_valids,
        'dir_ic_mean':        float(dir_ics_arr.mean()),
        'dir_ic_median':      float(np.median(dir_ics_arr)),
        'dir_ic_std':         float(dir_ics_arr.std()),
        'dir_ic_min':         float(dir_ics_arr.min()),
        'dir_ic_max':         float(dir_ics_arr.max()),
        'dir_ic_icir':        float(dir_ics_arr.mean() / max(dir_ics_arr.std(), 1e-9)),
        'dir_pct_positive':   float((dir_ics_arr > 0).mean()),
        'mag_ic_mean':        float(mag_ics_arr.mean()),
        'mag_ic_median':      float(np.median(mag_ics_arr)),
        'mag_ic_std':         float(mag_ics_arr.std()),
        'mag_ic_icir':        float(mag_ics_arr.mean() / max(mag_ics_arr.std(), 1e-9)),
        'best5_days_dir':     best5_dir,
        'worst5_days_dir':    worst5_dir,
    }


# ===========================================================================
# B. Time-of-Day IC Curve
# ===========================================================================

def analyze_time_of_day_ic(data: Dict) -> Dict:
    """
    Split each trading day into 30-minute buckets and compute IC per bucket.

    ES RTH: 9:30 AM – 4:00 PM ET = 6.5 hours = 13 buckets.
    At 100ms resolution: 234,000 bars/day → 18,000 bars/bucket.

    We don't have a clock — we infer time from bar position within each day.
    Bar 0 of each day = 9:30:00.000 ET.
    """
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS B: TIME-OF-DAY IC CURVE (30-min buckets)")
    logger.info("=" * 60)

    day_boundaries   = data['day_boundaries']
    direction_preds  = data['direction_preds']
    magnitude_preds  = data['magnitude_preds']
    direction_target = data['direction_target']
    magnitude_target = data['magnitude_target']
    n_days           = data['n_days']

    # Accumulate predictions and targets by bucket across all days
    bucket_dir_preds  = [[] for _ in range(BUCKET_COUNT)]
    bucket_dir_tgts   = [[] for _ in range(BUCKET_COUNT)]
    bucket_mag_preds  = [[] for _ in range(BUCKET_COUNT)]
    bucket_mag_tgts   = [[] for _ in range(BUCKET_COUNT)]
    bucket_n_days     = np.zeros(BUCKET_COUNT, dtype=np.int32)

    for d in range(n_days):
        s = day_boundaries[d]
        e = day_boundaries[d + 1]
        day_len = e - s

        for b in range(BUCKET_COUNT):
            bs = s + b * BARS_PER_HALF_HOUR
            be = s + (b + 1) * BARS_PER_HALF_HOUR
            if bs >= e:
                break
            be = min(be, e)
            if be <= bs:
                break

            dp = direction_preds[bs:be]
            dt = direction_target[bs:be]
            mp = magnitude_preds[bs:be]
            mt = magnitude_target[bs:be]

            mask = np.isfinite(dp) & np.isfinite(dt)
            if mask.sum() >= 20:
                bucket_dir_preds[b].append(dp[mask])
                bucket_dir_tgts[b].append(dt[mask])
                bucket_n_days[b] += 1

            mask_m = np.isfinite(mp) & np.isfinite(mt)
            if mask_m.sum() >= 20:
                bucket_mag_preds[b].append(mp[mask_m])
                bucket_mag_tgts[b].append(mt[mask_m])

    # Compute IC per bucket (pool all days together for stability)
    bucket_dir_ics = []
    bucket_mag_ics = []
    bucket_n_obs   = []

    for b in range(BUCKET_COUNT):
        if bucket_dir_preds[b]:
            all_p = np.concatenate(bucket_dir_preds[b])
            all_t = np.concatenate(bucket_dir_tgts[b])
            ic = _safe_ic(all_p, all_t)
            n_obs = len(all_p)
        else:
            ic = 0.0
            n_obs = 0
        bucket_dir_ics.append(ic)
        bucket_n_obs.append(n_obs)

        if bucket_mag_preds[b]:
            all_p = np.concatenate(bucket_mag_preds[b])
            all_t = np.concatenate(bucket_mag_tgts[b])
            mic = _safe_ic(all_p, all_t)
        else:
            mic = 0.0
        bucket_mag_ics.append(mic)

    # Logging
    dir_ics_arr = np.array(bucket_dir_ics)
    logger.info(f"\n  {'Bucket':>6s}  {'Time':>6s}  {'N_obs':>9s}  "
                f"{'Dir_IC':>8s}  {'Mag_IC':>8s}  {'Note':s}")
    logger.info(f"  {'-'*70}")

    best_b  = int(np.argmax(dir_ics_arr))
    worst_b = int(np.argmin(dir_ics_arr))

    for b in range(BUCKET_COUNT):
        note = ""
        if b == best_b:
            note = " <-- BEST"
        elif b == worst_b:
            note = " <-- WORST"
        logger.info(
            f"  {b:>6d}  {BUCKET_LABELS[b]:>6s}  {bucket_n_obs[b]:>9,d}  "
            f"{bucket_dir_ics[b]:>+8.4f}  {bucket_mag_ics[b]:>+8.4f}  {note}"
        )

    logger.info(f"\n  Overall Dir IC:  {dir_ics_arr.mean():+.4f}")
    logger.info(f"  Best bucket:     {BUCKET_LABELS[best_b]} (IC={bucket_dir_ics[best_b]:+.4f})")
    logger.info(f"  Worst bucket:    {BUCKET_LABELS[worst_b]} (IC={bucket_dir_ics[worst_b]:+.4f})")

    # Identify open/mid/close patterns
    open_ic  = float(np.mean(dir_ics_arr[:2]))   # 9:30–10:30
    mid_ic   = float(np.mean(dir_ics_arr[2:11])) # 10:30–14:30
    close_ic = float(np.mean(dir_ics_arr[11:]))   # 14:30–close
    logger.info(f"\n  IC by session segment:")
    logger.info(f"    Open  (9:30–10:30):  {open_ic:+.4f}")
    logger.info(f"    Mid   (10:30–14:30): {mid_ic:+.4f}")
    logger.info(f"    Close (14:30–16:00): {close_ic:+.4f}")

    return {
        'bucket_labels':   BUCKET_LABELS,
        'bucket_dir_ics':  bucket_dir_ics,
        'bucket_mag_ics':  bucket_mag_ics,
        'bucket_n_obs':    bucket_n_obs,
        'bucket_n_days':   bucket_n_days.tolist(),
        'best_bucket':     best_b,
        'worst_bucket':    worst_b,
        'best_bucket_time': BUCKET_LABELS[best_b],
        'open_ic':         open_ic,
        'mid_ic':          mid_ic,
        'close_ic':        close_ic,
    }


# ===========================================================================
# C. Volatility Conditioning
# ===========================================================================

def analyze_volatility_conditioning(data: Dict) -> Dict:
    """
    Compute per-day realized volatility (std of 100ms mid-price returns).
    Sort days into quintiles by vol and compute IC per quintile.
    Answers: does the model work better in high-vol or low-vol days?
    """
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS C: VOLATILITY CONDITIONING")
    logger.info("=" * 60)

    day_boundaries   = data['day_boundaries']
    mid_prices       = data['mid_prices']
    direction_preds  = data['direction_preds']
    magnitude_preds  = data['magnitude_preds']
    direction_target = data['direction_target']
    magnitude_target = data['magnitude_target']
    n_days           = data['n_days']

    # Compute per-day realized vol (std of log-returns at 100ms)
    day_vol = np.zeros(n_days, dtype=np.float64)
    for d in range(n_days):
        s = day_boundaries[d]
        e = day_boundaries[d + 1]
        prices = mid_prices[s:e]
        if len(prices) < 10:
            day_vol[d] = np.nan
            continue
        # log-returns (or simple returns — both fine at 100ms)
        rets = np.diff(prices.astype(np.float64)) / prices[:-1].astype(np.float64)
        day_vol[d] = float(np.std(rets)) if len(rets) > 0 else np.nan

    valid_vol_mask = np.isfinite(day_vol)
    logger.info(f"  Days with valid vol: {valid_vol_mask.sum()}/{n_days}")

    # Quintile breakpoints (on days with valid vol)
    valid_vols = day_vol[valid_vol_mask]
    quintile_edges = np.percentile(valid_vols, [0, 20, 40, 60, 80, 100])
    logger.info(f"  Vol quintile edges (returns std): "
                f"{' / '.join(f'{v:.2e}' for v in quintile_edges)}")

    # Assign each day to a quintile
    day_quintile = np.full(n_days, -1, dtype=np.int32)
    for d in range(n_days):
        if not np.isfinite(day_vol[d]):
            continue
        for q in range(QUINTILE_COUNT):
            lo = quintile_edges[q]
            hi = quintile_edges[q + 1]
            if lo <= day_vol[d] <= hi:
                day_quintile[d] = q
                break

    # Compute IC per quintile (pool all predictions from quintile days)
    quintile_dir_ics = []
    quintile_mag_ics = []
    quintile_n_days  = []
    quintile_n_obs   = []
    quintile_vol_mean = []

    for q in range(QUINTILE_COUNT):
        days_in_q = np.where(day_quintile == q)[0]
        if len(days_in_q) == 0:
            quintile_dir_ics.append(0.0)
            quintile_mag_ics.append(0.0)
            quintile_n_days.append(0)
            quintile_n_obs.append(0)
            quintile_vol_mean.append(0.0)
            continue

        all_dp, all_dt, all_mp, all_mt = [], [], [], []
        for d in days_in_q:
            s = day_boundaries[d]
            e = day_boundaries[d + 1]
            dp = direction_preds[s:e]
            dt = direction_target[s:e]
            mp = magnitude_preds[s:e]
            mt = magnitude_target[s:e]
            mask = np.isfinite(dp) & np.isfinite(dt)
            if mask.sum() > 0:
                all_dp.append(dp[mask])
                all_dt.append(dt[mask])
            mask_m = np.isfinite(mp) & np.isfinite(mt)
            if mask_m.sum() > 0:
                all_mp.append(mp[mask_m])
                all_mt.append(mt[mask_m])

        if all_dp:
            all_dp_arr = np.concatenate(all_dp)
            all_dt_arr = np.concatenate(all_dt)
            dir_ic = _safe_ic(all_dp_arr, all_dt_arr)
            n_obs  = len(all_dp_arr)
        else:
            dir_ic = 0.0
            n_obs  = 0

        if all_mp:
            mag_ic = _safe_ic(np.concatenate(all_mp), np.concatenate(all_mt))
        else:
            mag_ic = 0.0

        quintile_dir_ics.append(dir_ic)
        quintile_mag_ics.append(mag_ic)
        quintile_n_days.append(len(days_in_q))
        quintile_n_obs.append(n_obs)
        quintile_vol_mean.append(float(day_vol[days_in_q].mean()))

    logger.info(f"\n  {'Quintile':>8s}  {'Vol_mean':>10s}  {'N_days':>7s}  "
                f"{'N_obs':>9s}  {'Dir_IC':>8s}  {'Mag_IC':>8s}  Note")
    logger.info(f"  {'-'*72}")

    q_dir_arr = np.array(quintile_dir_ics)
    best_q  = int(np.argmax(q_dir_arr))
    worst_q = int(np.argmin(q_dir_arr))

    quintile_names = ['Q1 (lowest vol)', 'Q2', 'Q3', 'Q4', 'Q5 (highest vol)']
    for q in range(QUINTILE_COUNT):
        note = ""
        if q == best_q:
            note = " <-- BEST"
        elif q == worst_q:
            note = " <-- WORST"
        logger.info(
            f"  {quintile_names[q]:>8s}  {quintile_vol_mean[q]:>10.2e}  "
            f"{quintile_n_days[q]:>7d}  {quintile_n_obs[q]:>9,d}  "
            f"{quintile_dir_ics[q]:>+8.4f}  {quintile_mag_ics[q]:>+8.4f}  {note}"
        )

    logger.info(f"\n  Best vol quintile: {quintile_names[best_q]} "
                f"(Dir IC={quintile_dir_ics[best_q]:+.4f})")
    logger.info(f"  Interpretation: model performs best in "
                + ("HIGH volatility" if best_q >= 3 else
                   "LOW volatility" if best_q <= 1 else
                   "MODERATE volatility") + " regimes")

    # Trend check: is IC monotone in vol?
    ic_trend = np.corrcoef(np.arange(QUINTILE_COUNT), q_dir_arr)[0, 1]
    logger.info(f"  IC-vol correlation (rank): {ic_trend:+.3f}")
    if abs(ic_trend) > 0.7:
        logger.info(f"  -> Strong {'positive' if ic_trend > 0 else 'negative'} relationship "
                    f"between vol and model IC")

    return {
        'quintile_names':      quintile_names,
        'quintile_vol_mean':   quintile_vol_mean,
        'quintile_dir_ics':    quintile_dir_ics,
        'quintile_mag_ics':    quintile_mag_ics,
        'quintile_n_days':     quintile_n_days,
        'quintile_n_obs':      quintile_n_obs,
        'quintile_edges':      quintile_edges.tolist(),
        'best_vol_quintile':   best_q,
        'worst_vol_quintile':  worst_q,
        'ic_vol_trend_corr':   float(ic_trend),
        'day_vol':             day_vol.tolist(),
        'day_quintile':        day_quintile.tolist(),
    }


# ===========================================================================
# D. Signal Strength vs. Outcome
# ===========================================================================

def analyze_signal_strength_vs_outcome(data: Dict) -> Dict:
    """
    Split bars by |direction_pred| quintile.
    For each quintile compute: IC, mean MFE (realised), win rate.
    Answers: do stronger predictions actually lead to better outcomes?
    """
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS D: SIGNAL STRENGTH VS. OUTCOME")
    logger.info("=" * 60)

    direction_preds  = data['direction_preds']
    direction_target = data['direction_target']
    magnitude_target = data['magnitude_target']

    # Only bars with both prediction and target valid
    valid_mask = (np.isfinite(direction_preds) &
                  np.isfinite(direction_target) &
                  np.isfinite(magnitude_target))

    dp_valid = direction_preds[valid_mask]
    dt_valid = direction_target[valid_mask]
    mt_valid = magnitude_target[valid_mask]

    logger.info(f"  Valid bars: {valid_mask.sum():,}")

    if valid_mask.sum() < 50:
        logger.warning("  Too few valid bars — skipping signal strength analysis")
        return {'error': 'insufficient data'}

    # Compute absolute signal strength and quintile bins
    abs_signal = np.abs(dp_valid)
    quintile_edges = np.percentile(abs_signal, [0, 20, 40, 60, 80, 100])
    logger.info(f"  |Signal| quintile edges: "
                f"{' / '.join(f'{v:.4f}' for v in quintile_edges)}")

    q_bin = np.digitize(abs_signal, quintile_edges[1:-1])  # 0..4

    quintile_results = []
    logger.info(f"\n  {'Quintile':>12s}  {'Signal_range':>16s}  {'N_obs':>8s}  "
                f"{'Dir_IC':>8s}  {'Mean_MFE':>9s}  {'Win_Rate':>9s}")
    logger.info(f"  {'-'*74}")

    for q in range(QUINTILE_COUNT):
        mask_q = (q_bin == q)
        if mask_q.sum() < 10:
            continue

        dp_q = dp_valid[mask_q]
        dt_q = dt_valid[mask_q]
        mt_q = mt_valid[mask_q]

        dir_ic   = _safe_ic(dp_q, dt_q)
        mean_mfe = float(mt_q.mean())

        # Win rate: does realised move go in predicted direction?
        # direction_target > 0 means up is correct; direction_pred > 0 means model says up
        correct = ((dp_q > 0) & (dt_q > 0)) | ((dp_q < 0) & (dt_q < 0))
        win_rate = float(correct.mean())

        lo = float(quintile_edges[q])
        hi = float(quintile_edges[q + 1])
        q_result = {
            'quintile':      q,
            'signal_lo':     lo,
            'signal_hi':     hi,
            'n_obs':         int(mask_q.sum()),
            'dir_ic':        dir_ic,
            'mean_mfe_ticks': mean_mfe,
            'win_rate':      win_rate,
        }
        quintile_results.append(q_result)

        q_name = f"Q{q+1} (weakest)" if q == 0 else (f"Q{q+1} (strongest)" if q == 4 else f"Q{q+1}")
        logger.info(
            f"  {q_name:>12s}  [{lo:.4f},{hi:.4f}]  {mask_q.sum():>8,d}  "
            f"{dir_ic:>+8.4f}  {mean_mfe:>9.3f}t  {win_rate:>9.1%}"
        )

    # IC trend: is IC monotonically increasing with signal strength?
    q_ics = [r['dir_ic'] for r in quintile_results]
    if len(q_ics) >= 3:
        ic_monotone_corr = float(np.corrcoef(np.arange(len(q_ics)), q_ics)[0, 1])
        logger.info(f"\n  IC vs signal-strength rank correlation: {ic_monotone_corr:+.3f}")
        if ic_monotone_corr > 0.7:
            logger.info("  -> CONFIRMED: stronger signal = better IC (model is calibrated)")
        elif ic_monotone_corr < -0.3:
            logger.info("  -> WARNING: IC DECREASES with signal strength (overconfidence?)")
        else:
            logger.info("  -> MIXED: signal strength does not strongly predict IC")
    else:
        ic_monotone_corr = 0.0

    return {
        'quintile_results':          quintile_results,
        'ic_vs_strength_monotone':   ic_monotone_corr,
        'total_valid_bars':          int(valid_mask.sum()),
        'signal_quintile_edges':     quintile_edges.tolist(),
    }


# ===========================================================================
# E. Market Order Simulation on Ultra-High Conviction
# ===========================================================================

def analyze_market_order_sim(data: Dict, frozen_dir_threshold: float = None) -> Dict:
    """
    Comprehensive trade simulation v2 — critical audit of entries, exits, costs.

    Tests MULTIPLE variants:
      1. Raw edge diagnostic — avg signed MTM before ANY costs (the TRUE edge)
      2. Pure MTM exit — enter, hold to horizon end, exit (no TP/SL)
      3. Fixed TP sweep — test 2, 3, 4, 5, 7, 10t targets
      4. Multiple cost scenarios — market entry vs limit entry
      5. Multiple horizons — 10s, 20s, 30s, 60s

    Cost decomposition (ES, 1-tick spread, 1-lot, ZERO slippage):
      Market entry: 0.5t (cross half-spread)
      Limit entry:  0t   (sit on book)
      Market exit:  0.5t (cross half-spread)
      Limit exit:   0t   (TP hit fills passively)
      Commission:   0.248t ($3.10 RT)

    TP exit: when MFE from mid >= target, a limit sell at (entry_mid + target)
    sits at the ask when the mid reaches target. The bid at that moment is
    (target - 0.5t) from entry_mid. So actual captured profit from fill prices =
    target - entry_spread - implicit_exit_spread = target - 1.0t for mkt entry.
    Plus commission: net = target - 1.0t - 0.248t = target - 1.248t.

    This is numerically identical to the old model. The cost is CORRECT.

    Minimum bar spacing = horizon bars (non-overlapping forward paths).
    OOS uses frozen_dir_threshold from IS (no look-ahead).
    """
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS E: COMPREHENSIVE TRADE SIM v2")
    logger.info(f"  Gate: mag_pred > {MAG_THRESHOLD_MKT}t AND dir in top "
                f"{(1 - DIR_QUANTILE_MKT):.0%}")
    logger.info(f"  Cost model: entry_spread=0.5t, exit_spread=0.5t, commission=0.248t")
    logger.info(f"  Horizons: {[h // 10 for h in SIM_HORIZONS]}s")
    logger.info(f"  TP targets: {TP_TARGETS}")
    logger.info("=" * 60)

    day_boundaries  = data['day_boundaries']
    mid_prices      = data['mid_prices']
    direction_preds = data['direction_preds']
    magnitude_preds = data['magnitude_preds']
    direction_target = data.get('direction_target')
    n_days          = data['n_days']

    # --- Direction threshold (frozen from IS if provided) ---
    valid_dir = direction_preds[np.isfinite(direction_preds)]
    if len(valid_dir) < 100:
        logger.warning("  Insufficient direction predictions — skipping trade sim")
        return {'error': 'insufficient predictions'}

    if frozen_dir_threshold is not None:
        dir_threshold = frozen_dir_threshold
        logger.info(f"  Direction threshold (FROZEN from IS): {dir_threshold:.6f}")
    else:
        abs_dir_signal = np.abs(valid_dir)
        dir_threshold  = float(np.percentile(abs_dir_signal, DIR_QUANTILE_MKT * 100))
        logger.info(f"  Direction threshold (P{DIR_QUANTILE_MKT*100:.0f}): {dir_threshold:.6f}")

    # --- Step 1: Find ALL qualifying bar indices (no spacing yet) ---
    # Store (bar_idx, day_idx) for each qualifying bar
    all_qualifying = []
    for day_idx in range(n_days):
        s = day_boundaries[day_idx]
        e = day_boundaries[day_idx + 1]
        for i in range(s, e):
            if not (np.isfinite(direction_preds[i]) and np.isfinite(magnitude_preds[i])):
                continue
            if magnitude_preds[i] < MAG_THRESHOLD_MKT:
                continue
            if abs(direction_preds[i]) < dir_threshold:
                continue
            all_qualifying.append((i, day_idx))

    logger.info(f"  Qualifying bars (before spacing): {len(all_qualifying):,}")

    if len(all_qualifying) < 10:
        logger.warning("  Too few qualifying bars.")
        return {'error': 'too few qualifying bars', 'dir_threshold': dir_threshold,
                'n_qualifying': len(all_qualifying)}

    max_horizon = max(SIM_HORIZONS)

    # --- Step 2: Per-horizon analysis ---
    all_results = {}

    for H in SIM_HORIZONS:
        H_sec = H / BARS_PER_SEC

        # Apply H-bar spacing within each day
        last_entry_per_day = {}
        trade_indices = []  # bar indices surviving the spacing filter

        for bar_idx, day_idx in all_qualifying:
            if day_idx not in last_entry_per_day:
                last_entry_per_day[day_idx] = -H
            if bar_idx < last_entry_per_day[day_idx] + H:
                continue

            # Check enough forward room in the day
            e = day_boundaries[day_idx + 1]
            if bar_idx + H >= e:
                continue

            trade_indices.append(bar_idx)
            last_entry_per_day[day_idx] = bar_idx

        n_trades = len(trade_indices)
        logger.info(f"\n  {'='*50}")
        logger.info(f"  HORIZON {H_sec:.0f}s ({H} bars) | {n_trades:,} non-overlapping trades")
        logger.info(f"  {'='*50}")

        if n_trades < 10:
            logger.info(f"  Skipping (too few trades)")
            continue

        # --- Extract forward paths ---
        mfe_arr = np.zeros(n_trades, dtype=np.float64)   # max favorable from mid
        mae_arr = np.zeros(n_trades, dtype=np.float64)   # max adverse from mid
        mtm_arr = np.zeros(n_trades, dtype=np.float64)   # signed endpoint return from mid
        dir_pred_arr = np.zeros(n_trades, dtype=np.float64)
        mag_pred_arr = np.zeros(n_trades, dtype=np.float64)

        for j, bar_idx in enumerate(trade_indices):
            direction = 1 if direction_preds[bar_idx] > 0 else -1
            entry_mid = float(mid_prices[bar_idx])
            path = mid_prices[bar_idx + 1: bar_idx + H + 1].astype(np.float64)

            # Signed path: positive = price moved in predicted direction
            if direction == 1:
                signed_path = (path - entry_mid) / TICK_SIZE
            else:
                signed_path = (entry_mid - path) / TICK_SIZE

            mfe_arr[j] = max(0.0, float(np.max(signed_path)))
            mae_arr[j] = max(0.0, float(-np.min(signed_path)))
            mtm_arr[j] = float(signed_path[-1])  # endpoint in predicted direction
            dir_pred_arr[j] = float(direction_preds[bar_idx])
            mag_pred_arr[j] = float(magnitude_preds[bar_idx])

        # ==============================================
        # RAW EDGE DIAGNOSTIC (before any costs)
        # ==============================================
        avg_mtm = float(mtm_arr.mean())
        avg_mfe = float(mfe_arr.mean())
        avg_mae = float(mae_arr.mean())
        pct_correct = float((mtm_arr > 0).mean())  # % trades where direction was right
        mtm_std = float(mtm_arr.std())

        # IC of direction prediction vs endpoint return (NOT MFE-net)
        # Build raw (unsigned) endpoint returns for IC computation
        raw_endpoints = np.zeros(n_trades, dtype=np.float64)
        for j, bar_idx in enumerate(trade_indices):
            raw_endpoints[j] = (float(mid_prices[bar_idx + H]) -
                                float(mid_prices[bar_idx])) / TICK_SIZE
        ic_endpoint = _safe_ic(dir_pred_arr, raw_endpoints)

        # IC vs MFE-net (direction target from the data if available)
        if direction_target is not None:
            dir_tgt_vals = np.array([float(direction_target[bi]) for bi in trade_indices])
            ic_mfe_net = _safe_ic(dir_pred_arr, dir_tgt_vals)
        else:
            mfe_net_arr = mfe_arr - mae_arr
            raw_mfe_net = np.zeros(n_trades, dtype=np.float64)
            for j, bar_idx in enumerate(trade_indices):
                d = 1 if direction_preds[bar_idx] > 0 else -1
                raw_mfe_net[j] = (mfe_arr[j] if d == 1 else -mfe_arr[j])
            ic_mfe_net = _safe_ic(dir_pred_arr, raw_mfe_net)

        logger.info(f"\n  --- RAW EDGE (before costs) ---")
        logger.info(f"  Avg signed MTM at horizon end: {avg_mtm:+.4f}t  (${avg_mtm * TICK_VALUE:+.2f})")
        logger.info(f"  Std MTM:                       {mtm_std:.3f}t")
        logger.info(f"  Direction accuracy:             {pct_correct:.1%}")
        logger.info(f"  Avg MFE (favorable):            {avg_mfe:.3f}t")
        logger.info(f"  Avg MAE (adverse):              {avg_mae:.3f}t")
        logger.info(f"  MFE/MAE ratio:                  {avg_mfe / max(avg_mae, 0.01):.3f}")
        logger.info(f"  IC vs endpoint return:           {ic_endpoint:+.4f}")
        logger.info(f"  IC vs MFE-net (direction target):{ic_mfe_net:+.4f}")

        # ==============================================
        # VARIANT 1: Pure MTM exit (no TP, no SL)
        # ==============================================
        logger.info(f"\n  --- VARIANT 1: Pure MTM exit ---")
        for label, cost in [("Mkt entry + Mkt exit", COST_MKT_IN_MKT_OUT),
                            ("Mkt entry + Lmt exit", COST_MKT_IN_LMT_OUT),
                            ("Lmt entry + Mkt exit", COST_LMT_IN_MKT_OUT),
                            ("Lmt entry + Lmt exit", COST_LMT_IN_LMT_OUT)]:
            pnl = mtm_arr - cost
            mean_pnl = float(pnl.mean())
            pf = _profit_factor(pnl)
            wr = float((pnl > 0).mean())
            total = float(pnl.sum()) * TICK_VALUE
            n_days_trading = len(set(
                next(d for bi, d in all_qualifying if bi == trade_indices[j])
                for j in range(min(n_trades, 100))  # sample for speed
            )) if n_trades > 0 else 1
            daily_est = total / max(n_days, 1)
            logger.info(f"    {label:25s} | cost={cost:.3f}t | "
                        f"PnL={mean_pnl:+.4f}t (${mean_pnl * TICK_VALUE:+.2f}) | "
                        f"PF={pf:.2f} | WR={wr:.1%} | "
                        f"Total=${total:+,.0f} | ~${daily_est:+,.0f}/day")

        # ==============================================
        # VARIANT 2: Fixed TP exit sweep
        # ==============================================
        logger.info(f"\n  --- VARIANT 2: Fixed TP (no SL), mkt entry ---")
        logger.info(f"    {'TP':>4s} | {'WR':>6s} | {'AvgW':>7s} | {'AvgL':>7s} | "
                     f"{'PnL/t':>8s} | {'PF':>5s} | {'$/trade':>8s} | {'$/day':>8s}")
        logger.info(f"    {'-'*4}-+-{'-'*6}-+-{'-'*7}-+-{'-'*7}-+-{'-'*8}-+-"
                     f"{'-'*5}-+-{'-'*8}-+-{'-'*8}")

        tp_results = []
        for tp in TP_TARGETS:
            # Win = MFE from mid >= tp
            wins = mfe_arr >= tp
            n_w = int(wins.sum())
            n_l = n_trades - n_w
            wr = float(wins.mean())

            if n_w == 0 or n_l == 0:
                continue

            # Winner PnL: TP hit → limit exit
            # Cost = entry_spread + implicit_exit_spread + commission = 1.248t
            # (bid at MFE peak is 0.5t below mid, so effective capture = tp - 1.0t,
            #  minus commission = tp - 1.248t)
            w_pnl_ticks = tp - COST_MKT_IN_MKT_OUT

            # Loser PnL: timeout → market exit at MTM
            # Cost = entry_spread + exit_spread + commission = 1.248t
            l_mtm = mtm_arr[~wins]
            l_pnl_mean = float(l_mtm.mean()) - COST_MKT_IN_MKT_OUT

            # Combined
            all_pnl = np.concatenate([
                np.full(n_w, w_pnl_ticks),
                l_mtm - COST_MKT_IN_MKT_OUT
            ])
            mean_pnl = float(all_pnl.mean())
            pf = _profit_factor(all_pnl)
            total = float(all_pnl.sum()) * TICK_VALUE
            daily = total / max(n_days, 1)

            logger.info(f"    {tp:4.1f} | {wr:5.1%} | {w_pnl_ticks:+6.3f}t | "
                        f"{l_pnl_mean:+6.3f}t | {mean_pnl:+7.4f}t | "
                        f"{pf:5.2f} | ${mean_pnl * TICK_VALUE:+7.2f} | "
                        f"${daily:+7.0f}")

            tp_results.append({
                'tp_ticks': tp, 'win_rate': wr, 'n_wins': n_w,
                'winner_pnl_ticks': w_pnl_ticks, 'loser_pnl_mean_ticks': l_pnl_mean,
                'mean_pnl_ticks': mean_pnl, 'profit_factor': float(pf),
                'total_pnl_dollars': total, 'daily_pnl_dollars': daily,
            })

        # Also show limit-entry version of best TP
        if tp_results:
            logger.info(f"\n  --- VARIANT 2b: Fixed TP (no SL), LIMIT entry ---")
            logger.info(f"    {'TP':>4s} | {'WR':>6s} | {'AvgW':>7s} | {'AvgL':>7s} | "
                         f"{'PnL/t':>8s} | {'PF':>5s} | {'$/trade':>8s} | {'$/day':>8s}")
            logger.info(f"    {'-'*4}-+-{'-'*6}-+-{'-'*7}-+-{'-'*7}-+-{'-'*8}-+-"
                         f"{'-'*5}-+-{'-'*8}-+-{'-'*8}")

            for tp in TP_TARGETS:
                wins = mfe_arr >= tp
                n_w = int(wins.sum())
                n_l = n_trades - n_w
                wr = float(wins.mean())
                if n_w == 0 or n_l == 0:
                    continue

                # Limit entry: no entry spread cost
                # Winner: TP hit → exit at limit. Need MFE from mid >= tp.
                # With limit entry at mid (no spread), we capture full MFE.
                # But exit is also limit → cost = commission only = 0.248t
                # Captured = tp - commission = tp - 0.248t
                w_pnl_ticks = tp - COST_LMT_IN_LMT_OUT

                # Loser: timeout → need to exit at market (pay exit spread)
                # Cost = exit_spread + commission = 0.748t
                l_mtm = mtm_arr[~wins]
                l_pnl_mean = float(l_mtm.mean()) - COST_LMT_IN_MKT_OUT

                all_pnl = np.concatenate([
                    np.full(n_w, w_pnl_ticks),
                    l_mtm - COST_LMT_IN_MKT_OUT
                ])
                mean_pnl = float(all_pnl.mean())
                pf = _profit_factor(all_pnl)
                total = float(all_pnl.sum()) * TICK_VALUE
                daily = total / max(n_days, 1)

                logger.info(f"    {tp:4.1f} | {wr:5.1%} | {w_pnl_ticks:+6.3f}t | "
                            f"{l_pnl_mean:+6.3f}t | {mean_pnl:+7.4f}t | "
                            f"{pf:5.2f} | ${mean_pnl * TICK_VALUE:+7.2f} | "
                            f"${daily:+7.0f}")

        # ==============================================
        # Distribution diagnostics
        # ==============================================
        logger.info(f"\n  --- DISTRIBUTIONS ---")
        logger.info(f"  MFE distribution (favorable, from mid):")
        for pct in [10, 25, 50, 75, 90, 95]:
            logger.info(f"    P{pct:>2d}: {np.percentile(mfe_arr, pct):.3f}t")
        logger.info(f"  MTM distribution (endpoint, signed by direction):")
        for pct in [10, 25, 50, 75, 90]:
            logger.info(f"    P{pct:>2d}: {np.percentile(mtm_arr, pct):+.3f}t")
        logger.info(f"  MAE distribution (adverse, from mid):")
        for pct in [10, 25, 50, 75, 90]:
            logger.info(f"    P{pct:>2d}: {np.percentile(mae_arr, pct):.3f}t")

        # Profitability thresholds
        logger.info(f"\n  --- PROFITABILITY THRESHOLDS ---")
        logger.info(f"  Need avg MTM > cost for profitability:")
        logger.info(f"  Current avg MTM = {avg_mtm:+.4f}t = ${avg_mtm * TICK_VALUE:+.2f}")
        for label, cost in [("Mkt in + Mkt out", COST_MKT_IN_MKT_OUT),
                            ("Lmt in + Mkt out", COST_LMT_IN_MKT_OUT),
                            ("Lmt in + Lmt out", COST_LMT_IN_LMT_OUT)]:
            gap = avg_mtm - cost
            logger.info(f"    {label:20s}: cost={cost:.3f}t, "
                        f"gap={gap:+.4f}t, "
                        f"{'PROFITABLE' if gap > 0 else 'NOT profitable'}")

        # Store per-horizon results
        all_results[f'{int(H_sec)}s'] = {
            'horizon_bars': H,
            'horizon_sec': H_sec,
            'n_trades': n_trades,
            'raw_edge': {
                'avg_signed_mtm_ticks': avg_mtm,
                'avg_signed_mtm_dollars': avg_mtm * TICK_VALUE,
                'std_mtm_ticks': mtm_std,
                'direction_accuracy': pct_correct,
                'avg_mfe_ticks': avg_mfe,
                'avg_mae_ticks': avg_mae,
                'mfe_mae_ratio': avg_mfe / max(avg_mae, 0.01),
                'ic_vs_endpoint': ic_endpoint,
                'ic_vs_mfe_net': ic_mfe_net,
            },
            'pure_mtm': {
                'mkt_in_mkt_out': {
                    'cost': COST_MKT_IN_MKT_OUT,
                    'pnl_ticks': avg_mtm - COST_MKT_IN_MKT_OUT,
                    'pf': _profit_factor(mtm_arr - COST_MKT_IN_MKT_OUT),
                },
                'lmt_in_mkt_out': {
                    'cost': COST_LMT_IN_MKT_OUT,
                    'pnl_ticks': avg_mtm - COST_LMT_IN_MKT_OUT,
                    'pf': _profit_factor(mtm_arr - COST_LMT_IN_MKT_OUT),
                },
                'lmt_in_lmt_out': {
                    'cost': COST_LMT_IN_LMT_OUT,
                    'pnl_ticks': avg_mtm - COST_LMT_IN_LMT_OUT,
                    'pf': _profit_factor(mtm_arr - COST_LMT_IN_LMT_OUT),
                },
            },
            'tp_sweep': tp_results,
        }

    # ==============================================
    # Summary comparison across horizons
    # ==============================================
    logger.info(f"\n{'='*60}")
    logger.info(f"HORIZON COMPARISON SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"{'Horizon':>8s} | {'Trades':>7s} | {'AvgMTM':>8s} | {'DirAcc':>7s} | "
                f"{'IC_end':>7s} | {'IC_mfe':>7s} | "
                f"{'Mkt PnL':>8s} | {'Lmt PnL':>8s}")
    logger.info(f"{'-'*8}-+-{'-'*7}-+-{'-'*8}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*8}-+-{'-'*8}")

    for key in sorted(all_results.keys(), key=lambda x: int(x.replace('s', ''))):
        r = all_results[key]
        re = r['raw_edge']
        pm = r['pure_mtm']
        logger.info(f"  {key:>6s} | {r['n_trades']:>7,} | {re['avg_signed_mtm_ticks']:+7.4f}t | "
                    f"{re['direction_accuracy']:6.1%} | "
                    f"{re['ic_vs_endpoint']:+6.4f} | {re['ic_vs_mfe_net']:+6.4f} | "
                    f"${pm['mkt_in_mkt_out']['pnl_ticks'] * TICK_VALUE:+7.2f} | "
                    f"${pm['lmt_in_mkt_out']['pnl_ticks'] * TICK_VALUE:+7.2f}")

    return {
        'dir_threshold': dir_threshold,
        'n_qualifying_total': len(all_qualifying),
        'horizons': all_results,
    }


# ===========================================================================
# IS/OOS wrapper
# ===========================================================================

def _slice_data(data: Dict, start_day: int, end_day: int) -> Dict:
    """Return a sub-dict sliced to [start_day, end_day) with re-zero'd boundaries."""
    db = data['day_boundaries']
    s  = db[start_day]
    e  = db[end_day]
    new_bounds = [b - s for b in db[start_day: end_day + 1]]
    return {
        'mid_prices':       data['mid_prices'][s:e],
        'direction_preds':  data['direction_preds'][s:e],
        'magnitude_preds':  data['magnitude_preds'][s:e],
        'direction_target': data['direction_target'][s:e],
        'magnitude_target': data['magnitude_target'][s:e],
        'day_boundaries':   new_bounds,
        'n_days':           end_day - start_day,
    }


def run_all_analyses(data: Dict, label: str = "ALL",
                     frozen_dir_threshold: float = None) -> Dict:
    """Run all 5 analyses on the given data slice.

    If frozen_dir_threshold is provided, passes it to Analysis E so OOS
    uses the IS-computed direction threshold (prevents look-ahead bias).
    """
    logger.info(f"\n{'#' * 70}")
    logger.info(f"# RUNNING ALL ANALYSES — {label}  ({data['n_days']} days)")
    logger.info(f"{'#' * 70}")

    t0 = time.time()
    results_a = analyze_per_day_ic(data)
    results_b = analyze_time_of_day_ic(data)
    results_c = analyze_volatility_conditioning(data)
    results_d = analyze_signal_strength_vs_outcome(data)
    results_e = analyze_market_order_sim(data,
                                         frozen_dir_threshold=frozen_dir_threshold)
    elapsed   = time.time() - t0

    logger.info(f"\n  All analyses done in {elapsed:.1f}s")

    return {
        'label':            label,
        'n_days':           data['n_days'],
        'per_day_ic':       results_a,
        'time_of_day_ic':   results_b,
        'vol_conditioning': results_c,
        'signal_strength':  results_d,
        'market_order_sim': results_e,
        'elapsed_sec':      elapsed,
    }


# ===========================================================================
# Discord summary
# ===========================================================================

def _print_discord_summary(is_results: Dict, oos_results: Optional[Dict], total_time: float):
    """Print a compact Discord-ready summary to stdout."""

    def _fmt_section(r: Dict, label: str) -> List[str]:
        lines = [f"\n**EDGE ANALYSIS — {label}** ({r['n_days']} days)"]

        # A
        a = r.get('per_day_ic', {})
        if 'dir_ic_mean' in a:
            lines += [
                "**A. Per-Day IC:**",
                f"```",
                f"Dir IC:  mean={a['dir_ic_mean']:+.4f}  ICIR={a['dir_ic_icir']:.2f}  "
                f"std={a['dir_ic_std']:.4f}",
                f"Mag IC:  mean={a['mag_ic_mean']:+.4f}  ICIR={a['mag_ic_icir']:.2f}",
                f"% days positive dir IC: {a['dir_pct_positive']:.0%}",
                f"Best day IC:  {a['dir_ic_max']:+.4f}  Worst: {a['dir_ic_min']:+.4f}",
                f"```",
            ]

        # B
        b = r.get('time_of_day_ic', {})
        if 'best_bucket_time' in b:
            lines += [
                "**B. Time-of-Day IC (best/worst bucket):**",
                f"```",
                f"Best:  {b['best_bucket_time']} IC={b['bucket_dir_ics'][b['best_bucket']]:+.4f}",
                f"Worst: {b['bucket_labels'][b['worst_bucket']]} "
                f"IC={b['bucket_dir_ics'][b['worst_bucket']]:+.4f}",
                f"Open (9:30-10:30): {b['open_ic']:+.4f}",
                f"Mid  (10:30-14:30): {b['mid_ic']:+.4f}",
                f"Close(14:30-16:00): {b['close_ic']:+.4f}",
                f"```",
            ]

        # C
        c = r.get('vol_conditioning', {})
        if 'quintile_dir_ics' in c:
            best_q  = c['best_vol_quintile']
            worst_q = c['worst_vol_quintile']
            lines += [
                "**C. Volatility Conditioning:**",
                f"```",
                f"Best regime:  {c['quintile_names'][best_q]}  "
                f"IC={c['quintile_dir_ics'][best_q]:+.4f}",
                f"Worst regime: {c['quintile_names'][worst_q]}  "
                f"IC={c['quintile_dir_ics'][worst_q]:+.4f}",
                f"IC-vol trend corr: {c['ic_vol_trend_corr']:+.3f}",
                f"```",
            ]

        # D
        d = r.get('signal_strength', {})
        if 'quintile_results' in d and d['quintile_results']:
            qr = d['quintile_results']
            lines += [
                "**D. Signal Strength vs Outcome:**",
                f"```",
            ]
            for q in qr:
                lines.append(
                    f"  Q{q['quintile']+1}: IC={q['dir_ic']:+.4f}  "
                    f"MFE={q['mean_mfe_ticks']:.3f}t  WR={q['win_rate']:.1%}"
                )
            lines += [
                f"IC-strength monotone corr: {d.get('ic_vs_strength_monotone', 0):+.3f}",
                f"```",
            ]

        # E — Comprehensive Trade Sim v2
        e = r.get('market_order_sim', {})
        horizons = e.get('horizons', {})
        if horizons:
            lines += [
                "**E. Trade Sim v2 (mag>3t, top 10% dir):**",
                f"```",
            ]
            for key in sorted(horizons.keys(), key=lambda x: int(x.replace('s', ''))):
                h = horizons[key]
                re = h.get('raw_edge', {})
                pm = h.get('pure_mtm', {})
                mtm = re.get('avg_signed_mtm_ticks', 0)
                mkt_pnl = pm.get('mkt_in_mkt_out', {}).get('pnl_ticks', 0)
                lmt_pnl = pm.get('lmt_in_mkt_out', {}).get('pnl_ticks', 0)
                lines.append(
                    f"  {key:>4s}: {h['n_trades']:>5,}t | "
                    f"rawMTM={mtm:+.3f}t | "
                    f"dir={re.get('direction_accuracy', 0):.0%} | "
                    f"mkt=${mkt_pnl * TICK_VALUE:+.1f} | "
                    f"lmt=${lmt_pnl * TICK_VALUE:+.1f}"
                )
            lines.append(f"```")

        return lines

    lines: List[str] = [
        "",
        "--- DISCORD SUMMARY ---",
        f"**EDGE ANALYSIS COMPLETE** — total time: {total_time/60:.1f} min",
    ]
    lines += _fmt_section(is_results, "IS" if oos_results else "FULL")

    if oos_results:
        lines += _fmt_section(oos_results, "OOS")

    summary = "\n".join(lines)
    print(summary)
    logger.info(summary)


# ===========================================================================
# Main pipeline
# ===========================================================================

def run_pipeline(args):
    start_time = time.time()
    timestamp  = datetime.now().strftime('%Y%m%d_%H%M%S')

    logger.info("=" * 80)
    logger.info("EDGE ANALYSIS — WHEN AND WHY DOES THE MODEL WORK BEST")
    logger.info("=" * 80)
    logger.info(f"Timestamp:   {timestamp}")
    logger.info(f"Predictions: {args.load_predictions}")
    logger.info(f"N-days:      {args.n_days}")
    logger.info(f"OOS split:   {args.oos_split_day}")

    # ------------------------------------------------------------------
    # Load predictions
    # ------------------------------------------------------------------
    full_data = load_predictions(args.load_predictions, n_days=args.n_days)
    n_days    = full_data['n_days']

    # ------------------------------------------------------------------
    # IS / OOS split
    # ------------------------------------------------------------------
    oos_split_day = args.oos_split_day
    if oos_split_day is not None and oos_split_day >= n_days:
        logger.warning(f"--oos-split-day {oos_split_day} >= n_days {n_days}, "
                       "ignoring split and running full analysis")
        oos_split_day = None

    if oos_split_day:
        logger.info(f"\nIS: days 0..{oos_split_day-1}  |  OOS: days {oos_split_day}..{n_days-1}")
        is_data  = _slice_data(full_data, 0, oos_split_day)
        oos_data = _slice_data(full_data, oos_split_day, n_days)
        is_results  = run_all_analyses(is_data,  label="IS")

        # Freeze IS direction threshold for OOS (prevent look-ahead bias)
        is_dir_threshold = is_results.get('market_order_sim', {}).get('dir_threshold')
        if is_dir_threshold:
            logger.info(f"\n  Freezing IS dir_threshold={is_dir_threshold:.6f} for OOS")
        oos_results = run_all_analyses(oos_data, label="OOS",
                                       frozen_dir_threshold=is_dir_threshold)
    else:
        is_results  = run_all_analyses(full_data, label="FULL")
        oos_results = None

    # ------------------------------------------------------------------
    # Save results JSON
    # ------------------------------------------------------------------
    out_file = RESULTS_DIR / f"edge_analysis_{timestamp}.json"

    def _make_serializable(obj):
        """Recursively convert numpy types to Python native for JSON."""
        if isinstance(obj, dict):
            return {k: _make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_make_serializable(v) for v in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    save_data = _make_serializable({
        'timestamp':    timestamp,
        'predictions':  args.load_predictions,
        'n_days':       n_days,
        'oos_split_day': oos_split_day,
        'is_results':   is_results,
        'oos_results':  oos_results,
        'total_time_sec': time.time() - start_time,
    })

    with open(str(out_file), 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    logger.info(f"\nResults saved: {out_file}")
    logger.info(f"Log:           {_log_file}")

    total_time = time.time() - start_time
    logger.info(f"\n{'=' * 80}")
    logger.info(f"EDGE ANALYSIS COMPLETE — {total_time:.0f}s ({total_time / 60:.1f}m)")
    logger.info(f"{'=' * 80}")

    _print_discord_summary(is_results, oos_results, total_time)

    return save_data


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Edge Analysis — WHEN and WHY the MFE prediction model works best',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--load-predictions', type=str, required=True,
        help='Path to predictions NPZ file '
             '(same format as mfe_path_analysis.py / magnitude_gated_sim.py)',
    )
    parser.add_argument(
        '--n-days', type=int, default=None,
        help='Limit analysis to first N days (default: all)',
    )
    parser.add_argument(
        '--oos-split-day', type=int, default=None,
        help='Run IS analysis on days 0..N, OOS on days N..end (0-indexed, exclusive split)',
    )
    args = parser.parse_args()

    try:
        return run_pipeline(args)
    except Exception as e:
        logger.error(f"PIPELINE FAILED: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
