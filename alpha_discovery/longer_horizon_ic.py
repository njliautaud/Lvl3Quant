"""
Longer-Horizon IC Decay Analysis — Lvl3Quant ES Futures
=========================================================

PURPOSE:
  Investigate whether ES futures MBO data (340 features, 100ms bars) can support
  multi-minute to hourly trading strategies where transaction costs are manageable.

KEY QUESTIONS:
  1. At what horizon does the cost ratio drop below 10%? (cost / avg move < 0.10)
  2. Which features maintain predictive IC at 5-30 minute horizons?
  3. Do microprice_dev, cancel_asym_5, ofi_5, depth_ratio_l1 persist or decay?

METHODOLOGY:
  - Load each day's 340-feature cache independently (no look-ahead)
  - Compute forward returns in ticks at: 1m, 5m, 15m, 30m, 1h
  - Also include short horizons for baseline: 10s, 30s, 1m
  - For each of the TOP 20 features (by IC at 10s), compute per-day Spearman IC
  - Report: mean IC, std IC, t-stat, % positive days, IC decay curve
  - Compute: average |return| in ticks at each horizon
  - Compute: cost ratio = 1.24 ticks RT / avg |return|
  - Identify: features with "persistent" IC (>50% of their 10s IC at 5m+)

TOP FEATURES (from prior raw_feature_signals study on 100 days, 10s horizon):
  1. depth_ratio_l1      — IC=0.0943 (depth imbalance, raw L1)
  2. depth_ratio_l1_z500 — IC=0.0924 (z-score version, normalized)
  3. order_frag_asym     — IC=0.0862 (order fragmentation asymmetry)
  4. ask_L1_orders       — IC=0.0819 (ask side L1 order count, slowest decay)
  5. book_imb_zscore_50  — IC=0.0717 (short-term abnormal book imbalance)
  6. microprice_dev      — IC=0.0670 (microprice - mid, pure book physics)
  7. cancel_asym_5       — IC=0.0356 (cancel side imbalance, leads OFI by 0.5s)
  8. ofi_5               — IC from signal studies (order flow imbalance)
  9. pressure_imbalance  — raw signal from prior scans
 10. bid_slope / ask_slope — book depth slope features
  + additional composite candidates

USAGE:
    python alpha_discovery/longer_horizon_ic.py
    python alpha_discovery/longer_horizon_ic.py --n-days 50
    python alpha_discovery/longer_horizon_ic.py --quick  # fast mode, fewer features
"""

import gc
import sys
import json
import time
import logging
import argparse
import platform
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import numpy as np
from scipy.stats import spearmanr, ttest_1samp

# ============================================================================
# PATH SETUP
# ============================================================================

LVL3_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(LVL3_ROOT))

if platform.system() == "Windows":
    FEATURE_CACHE = LVL3_ROOT / "data" / "processed" / "mbo_features_cache"
else:
    FEATURE_CACHE = Path.home() / "lvl3quant" / "data" / "processed" / "mbo_features_cache"

RESULTS_DIR = LVL3_ROOT / "alpha_discovery" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# LOGGING
# ============================================================================

_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
_log_file = RESULTS_DIR / f"longer_horizon_ic_{_ts}.log"
_root = logging.getLogger()
_root.setLevel(logging.INFO)
for _h in _root.handlers[:]:
    _root.removeHandler(_h)
_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
_fh = logging.FileHandler(str(_log_file), mode="w")
_fh.setFormatter(_fmt)
_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(_fmt)
_root.addHandler(_fh)
_root.addHandler(_sh)
logger = logging.getLogger("long_hz_ic")

# ============================================================================
# CONSTANTS
# ============================================================================

TICK_SIZE = 0.25      # ES tick size in points
TICK_VALUE = 12.50    # $ per tick
COST_TICKS_RT = 3.00 / TICK_VALUE  # 0.24 ticks roundtrip commission
SPREAD_TICKS = 1.0    # typical 1-tick half-spread on entry (market order)
TOTAL_COST_TICKS = COST_TICKS_RT + SPREAD_TICKS  # ~1.24 ticks RT all-in
BARS_PER_SEC = 10     # 100ms bars

# Horizons: name -> bars count
HORIZONS = {
    "10s":  100,     # baseline (used in prior studies)
    "30s":  300,     # baseline
    "1m":   600,     # target: min manageable
    "5m":   3000,    # target: core analysis
    "15m":  9000,    # target: medium horizon
    "30m":  18000,   # target: swing
    "1h":   36000,   # target: intraday swing
}

# ============================================================================
# FEATURE DEFINITIONS
# ============================================================================
# Each entry: (display_name, computation_type, col_or_info)
# computation_type: "col" = direct column index,
#                   "diff" = col_a - col_b,
#                   "combo" = weighted sum of (col, weight) pairs

def get_feature_spec() -> List[Dict]:
    """
    Define the top ~20 features to analyze at longer horizons.
    Based on prior raw_feature_signals study (10s horizon, 100 days).
    """
    from alpha_discovery.mbo_features import get_feature_names
    names = get_feature_names()
    name_to_idx = {n: i for i, n in enumerate(names)}

    def col(feature_name: str) -> Optional[int]:
        idx = name_to_idx.get(feature_name)
        if idx is None:
            logger.warning(f"Feature not found: {feature_name}")
        return idx

    specs = []

    # --- TOP INDIVIDUAL FEATURES (by IC at 10s from prior studies) ---

    # 1. Depth ratio L1 — highest raw IC=0.0943
    if col("depth_ratio_l1") is not None:
        specs.append({
            "name": "depth_ratio_l1",
            "desc": "L1 bid/ask depth ratio (highest 10s IC=0.0943)",
            "type": "col",
            "col": col("depth_ratio_l1"),
            "sign": 1,
        })

    # 2. Depth ratio L1 z-score 500 — IC=0.0924 (normalized version)
    if col("depth_ratio_l1_zscore_500") is not None:
        specs.append({
            "name": "depth_ratio_l1_z500",
            "desc": "Depth ratio L1 z-score 500-bar (IC=0.0924 at 10s)",
            "type": "col",
            "col": col("depth_ratio_l1_zscore_500"),
            "sign": 1,
        })

    # 3. Order fragmentation asymmetry — IC=0.0862
    if col("order_frag_asym") is not None:
        specs.append({
            "name": "order_frag_asym",
            "desc": "Order fragmentation asymmetry (IC=0.0862 at 10s)",
            "type": "col",
            "col": col("order_frag_asym"),
            "sign": 1,
        })

    # 4. Ask L1 orders — IC=0.0819, slowest IC decay (25% at 30s)
    if col("ask_L1_orders") is not None:
        specs.append({
            "name": "ask_L1_orders",
            "desc": "Ask L1 order count (slowest decay: 25% remaining at 30s)",
            "type": "col",
            "col": col("ask_L1_orders"),
            "sign": -1,  # more ask orders = price goes down
        })

    # 5. Book imbalance z-score 50 — IC=0.0717
    if col("book_imb_zscore_50") is not None:
        specs.append({
            "name": "book_imb_z50",
            "desc": "Book imbalance z-score 50-bar (IC=0.0717 at 10s)",
            "type": "col",
            "col": col("book_imb_zscore_50"),
            "sign": 1,
        })

    # 6. Microprice deviation = microprice - mid (pure book physics signal)
    #    IC=0.195 at 1s in raw IC study (the BEST signal we found)
    mid_idx = col("mid")
    mp_idx = col("microprice")
    if mid_idx is not None and mp_idx is not None:
        specs.append({
            "name": "microprice_dev",
            "desc": "Microprice - mid (book physics, IC=0.195@1s, best raw signal)",
            "type": "diff",
            "col_a": mp_idx,   # microprice
            "col_b": mid_idx,  # mid
            "sign": 1,
        })

    # 7. Cancel asymmetry — leads OFI by 0.5s, early warning signal
    if col("cancel_asym_5") is not None:
        specs.append({
            "name": "cancel_asym_5",
            "desc": "Cancel-side asymmetry 5-bar (leads OFI by 0.5s)",
            "type": "col",
            "col": col("cancel_asym_5"),
            "sign": -1,
        })

    # 8. Cancel asym z-score 500 (normalized version)
    if col("cancel_asym_zscore_500") is not None:
        specs.append({
            "name": "cancel_asym_z500",
            "desc": "Cancel asym z-score 500-bar (longer-term cancel regime)",
            "type": "col",
            "col": col("cancel_asym_zscore_500"),
            "sign": -1,
        })

    # 9. OFI 5-bar — order flow imbalance (short horizon)
    if col("ofi_5") is not None:
        specs.append({
            "name": "ofi_5",
            "desc": "Order flow imbalance 5-bar (short-term flow)",
            "type": "col",
            "col": col("ofi_5"),
            "sign": 1,
        })

    # 10. OFI 50-bar — longer-term OFI
    if col("ofi_50") is not None:
        specs.append({
            "name": "ofi_50",
            "desc": "Order flow imbalance 50-bar (longer-term flow)",
            "type": "col",
            "col": col("ofi_50"),
            "sign": 1,
        })

    # 11. Pressure imbalance — bid vs ask pressure
    if col("pressure_imbalance") is not None:
        specs.append({
            "name": "pressure_imbalance",
            "desc": "Bid/ask pressure imbalance (raw book pressure)",
            "type": "col",
            "col": col("pressure_imbalance"),
            "sign": 1,
        })

    # 12. Depth ratio raw (all levels, not just L1)
    if col("depth_ratio") is not None:
        specs.append({
            "name": "depth_ratio",
            "desc": "Full book depth ratio (all levels)",
            "type": "col",
            "col": col("depth_ratio"),
            "sign": 1,
        })

    # 13. Bid slope (book shape feature)
    if col("bid_slope") is not None:
        specs.append({
            "name": "bid_slope",
            "desc": "Bid side book slope",
            "type": "col",
            "col": col("bid_slope"),
            "sign": 1,
        })

    # 14. Ask slope (book shape feature)
    if col("ask_slope") is not None:
        specs.append({
            "name": "ask_slope",
            "desc": "Ask side book slope",
            "type": "col",
            "col": col("ask_slope"),
            "sign": -1,
        })

    # 15. Weighted book imbalance
    if col("weighted_book_imb") is not None:
        specs.append({
            "name": "weighted_book_imb",
            "desc": "Depth-weighted book imbalance",
            "type": "col",
            "col": col("weighted_book_imb"),
            "sign": 1,
        })

    # 16. Depth ratio L1 z-score 50 (short-term normalized)
    if col("depth_ratio_l1_zscore_50") is not None:
        specs.append({
            "name": "depth_ratio_l1_z50",
            "desc": "Depth ratio L1 z-score 50-bar (short-term abnormal)",
            "type": "col",
            "col": col("depth_ratio_l1_zscore_50"),
            "sign": 1,
        })

    # 17. Aggressive imbalance (buy vs sell aggression)
    if col("aggressive_imbalance") is not None:
        specs.append({
            "name": "aggressive_imbalance",
            "desc": "Aggressive buy/sell imbalance (order flow direction)",
            "type": "col",
            "col": col("aggressive_imbalance"),
            "sign": 1,
        })

    # 18. Vol imbalance (L1 bid vs ask volume)
    if col("vol_imbalance") is not None:
        specs.append({
            "name": "vol_imbalance",
            "desc": "L1 bid/ask volume imbalance",
            "type": "col",
            "col": col("vol_imbalance"),
            "sign": 1,
        })

    # 19. Queue depletion asymmetry 5-bar
    if col("queue_depletion_asymmetry_5") is not None:
        specs.append({
            "name": "queue_dep_asym_5",
            "desc": "Queue depletion asymmetry 5-bar (bid vs ask depletion rate)",
            "type": "col",
            "col": col("queue_depletion_asymmetry_5"),
            "sign": -1,
        })

    # 20. OFI 5 z-score 500 (normalized longer-term OFI)
    if col("ofi_5_zscore_500") is not None:
        specs.append({
            "name": "ofi_5_z500",
            "desc": "OFI 5-bar z-score 500 (normalized long-term OFI)",
            "type": "col",
            "col": col("ofi_5_zscore_500"),
            "sign": 1,
        })

    # --- COMPOSITE SIGNALS (combos of above) ---

    # Top5 ensemble (from raw_feature_signals, IC=0.0896 at 10s)
    d1 = col("depth_ratio_l1")
    d1z500 = col("depth_ratio_l1_zscore_500")
    d1z50 = col("depth_ratio_l1_zscore_50")
    ofa = col("order_frag_asym")
    biz50 = col("book_imb_zscore_50")
    if all(x is not None for x in [d1, d1z500, d1z50, ofa, biz50]):
        specs.append({
            "name": "top5_ensemble",
            "desc": "Top5 equal-weight ensemble (IC=0.0896 at 10s from prior study)",
            "type": "combo",
            "components": [
                (d1,    +0.2),
                (d1z500, +0.2),
                (d1z50,  +0.2),
                (ofa,    +0.2),
                (biz50,  +0.2),
            ],
        })

    # Causal chain: cancel_asym -> OFI -> depth_ratio
    ca5 = col("cancel_asym_5")
    o5 = col("ofi_5")
    if all(x is not None for x in [ca5, o5, d1, d1z500]):
        specs.append({
            "name": "causal_chain",
            "desc": "Causal chain: cancel_asym -> OFI -> depth_ratio (IC=0.0773 at 10s)",
            "type": "combo",
            "components": [
                (ca5,    -0.3),   # cancel_asym sign=-1
                (o5,     +0.3),
                (d1,     +0.2),
                (d1z500, +0.2),
            ],
        })

    # Slow-decay combo (slowest decaying features for tradeable signals)
    al1 = col("ask_L1_orders")
    wbi = col("weighted_book_imb")
    pi = col("pressure_imbalance")
    biz500 = col("book_imb_zscore_500")
    qda5 = col("queue_depletion_asymmetry_5")
    if all(x is not None for x in [al1, wbi, pi, biz500, qda5]):
        specs.append({
            "name": "slow_decay_combo",
            "desc": "Slow-decay feature combo (IC=0.0801 at 10s, best persistence candidates)",
            "type": "combo",
            "components": [
                (al1,   -1.0),  # ask L1 sign=-1
                (wbi,   +1.0),
                (pi,    +1.0),
                (biz500, +1.0),
                (qda5,  -1.0),
            ],
        })

    logger.info(f"Defined {len(specs)} feature specs for analysis")
    return specs


# ============================================================================
# DATA LOADING
# ============================================================================

def discover_days(cache_dir: Path, n_days: Optional[int] = None) -> List[Tuple[str, Path]]:
    """Find all available days in the feature cache."""
    files = sorted(cache_dir.glob("*_mbo_features.npz"))
    if not files:
        raise FileNotFoundError(f"No *_mbo_features.npz files in {cache_dir}")
    if n_days is not None:
        files = files[:n_days]
    days = []
    for f in files:
        date_str = f.stem.replace("_mbo_features", "")
        days.append((date_str, f))
    logger.info(f"Found {len(days)} days: {days[0][0]} to {days[-1][0]}")
    return days


def load_day(fpath: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load one day's MBO features.

    Returns:
        features: (N, 340) float32
        mid:      (N,) float32 — column 0
    """
    data = np.load(str(fpath))
    features = data["mbo_features"]  # (N, 340) float32
    mid = features[:, 0].copy()      # col 0 = mid price
    return features, mid


# ============================================================================
# SIGNAL EXTRACTION
# ============================================================================

def extract_signal(features: np.ndarray, spec: Dict) -> np.ndarray:
    """Extract signal from features based on spec definition."""
    stype = spec["type"]
    if stype == "col":
        raw = features[:, spec["col"]].astype(np.float64)
        raw *= spec.get("sign", 1)
    elif stype == "diff":
        raw = (features[:, spec["col_a"]] - features[:, spec["col_b"]]).astype(np.float64)
        raw *= spec.get("sign", 1)
    elif stype == "combo":
        raw = np.zeros(len(features), dtype=np.float64)
        for col_idx, weight in spec["components"]:
            raw += features[:, col_idx].astype(np.float64) * weight
    else:
        raise ValueError(f"Unknown spec type: {stype}")
    # Winsorize: clip to [-10 std, +10 std] to remove outliers
    std = np.nanstd(raw)
    if std > 0:
        raw = np.clip(raw, -10 * std, 10 * std)
    return raw


# ============================================================================
# FORWARD RETURN COMPUTATION
# ============================================================================

def compute_forward_returns(
    mid: np.ndarray,
    horizons: Dict[str, int],
) -> Dict[str, np.ndarray]:
    """
    Compute forward returns at multiple horizons.

    Returns dict: horizon_name -> array of forward returns in ticks.
    NaN for bars without a full future window.
    """
    N = len(mid)
    results = {}
    for hz_name, hz_bars in horizons.items():
        ret = np.full(N, np.nan, dtype=np.float64)
        if N > hz_bars:
            future_mid = mid[hz_bars:]
            ret[: N - hz_bars] = (future_mid - mid[: N - hz_bars]) / TICK_SIZE
        results[hz_name] = ret
    return results


# ============================================================================
# IC COMPUTATION
# ============================================================================

def compute_ic(signal: np.ndarray, target: np.ndarray) -> Optional[float]:
    """Compute Spearman IC between signal and target. Returns None if insufficient data."""
    valid = np.isfinite(signal) & np.isfinite(target)
    n = valid.sum()
    if n < 200:
        return None
    try:
        ic, _ = spearmanr(signal[valid], target[valid])
        return float(ic) if np.isfinite(ic) else None
    except Exception:
        return None


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def run_analysis(
    n_days: Optional[int] = None,
    quick: bool = False,
) -> Dict:
    """
    Main analysis function.

    For each day, for each feature spec, compute IC at all horizons.
    Also compute statistics on return distributions.
    """
    t_start = time.time()
    logger.info("=" * 60)
    logger.info("LONGER-HORIZON IC DECAY ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"Feature cache: {FEATURE_CACHE}")
    logger.info(f"Tick size: {TICK_SIZE}, Total cost (RT): {TOTAL_COST_TICKS:.3f} ticks")
    logger.info("")

    # ---- Setup ----
    days = discover_days(FEATURE_CACHE, n_days=n_days)
    specs = get_feature_spec()
    if quick:
        # Keep only the top 10 most important features for quick mode
        specs = specs[:10]
        logger.info("QUICK MODE: analyzing top 10 features only")

    horizons_to_use = HORIZONS
    hz_names = list(horizons_to_use.keys())
    hz_bars = list(horizons_to_use.values())

    logger.info(f"Days: {len(days)}, Features: {len(specs)}, Horizons: {hz_names}")
    logger.info("")

    # ---- Per-day per-feature IC storage ----
    # per_day_ic[feat_name][hz_name] = list of ICs across days
    per_day_ic: Dict[str, Dict[str, List[float]]] = {
        s["name"]: {hz: [] for hz in hz_names} for s in specs
    }

    # Return distribution stats per horizon (pooled across days)
    ret_stats: Dict[str, Dict[str, List[float]]] = {
        hz: {"abs_rets": [], "mean_abs": [], "std": []} for hz in hz_names
    }

    # Track day dates for output
    day_dates = [d[0] for d in days]

    # ---- Process each day ----
    for day_idx, (date_str, fpath) in enumerate(days):
        t_day = time.time()

        try:
            features, mid = load_day(fpath)
        except Exception as e:
            logger.warning(f"  [{date_str}] Load failed: {e}")
            continue

        N = len(mid)
        if N < hz_bars[-1]:
            logger.warning(f"  [{date_str}] Only {N} bars, need {hz_bars[-1]} for 1h horizon. Skipping 1h.")

        # Compute forward returns at all horizons for this day
        fwd_rets = compute_forward_returns(mid, horizons_to_use)

        # Collect return distribution stats
        for hz_name, ret in fwd_rets.items():
            valid = ret[np.isfinite(ret)]
            if len(valid) > 100:
                abs_rets = np.abs(valid)
                ret_stats[hz_name]["abs_rets"].extend(abs_rets.tolist())
                ret_stats[hz_name]["mean_abs"].append(float(np.mean(abs_rets)))
                ret_stats[hz_name]["std"].append(float(np.std(valid)))

        # Compute IC for each feature spec at each horizon
        for spec in specs:
            feat_name = spec["name"]
            try:
                signal = extract_signal(features, spec)
            except Exception as e:
                logger.warning(f"  [{date_str}] {feat_name} extract failed: {e}")
                continue

            for hz_name, ret in fwd_rets.items():
                ic = compute_ic(signal, ret)
                if ic is not None:
                    per_day_ic[feat_name][hz_name].append(ic)

        # Progress update every 10 days or at key milestones
        if (day_idx + 1) % 10 == 0 or day_idx == 0:
            elapsed = time.time() - t_start
            eta_per_day = elapsed / (day_idx + 1)
            remaining = eta_per_day * (len(days) - day_idx - 1)
            logger.info(
                f"  Day {day_idx + 1}/{len(days)} [{date_str}] — "
                f"elapsed {elapsed:.0f}s, ETA {remaining:.0f}s"
            )

        # Free memory
        del features, mid, fwd_rets
        gc.collect()

    total_elapsed = time.time() - t_start
    logger.info(f"\nAll {len(days)} days processed in {total_elapsed:.1f}s")
    logger.info("")

    # ============================================================
    # COMPUTE SUMMARY STATISTICS
    # ============================================================

    logger.info("=" * 60)
    logger.info("RESULTS: COST RATIOS BY HORIZON")
    logger.info("=" * 60)

    cost_ratio_results = {}
    for hz_name in hz_names:
        abs_rets = ret_stats[hz_name]["abs_rets"]
        mean_abs_list = ret_stats[hz_name]["mean_abs"]
        if len(abs_rets) == 0:
            continue
        pooled_mean = float(np.mean(abs_rets))
        pooled_median = float(np.median(abs_rets))
        per_day_mean = float(np.mean(mean_abs_list)) if mean_abs_list else 0.0
        cost_ratio = TOTAL_COST_TICKS / pooled_mean if pooled_mean > 0 else 999.0
        cost_ratio_str = f"{cost_ratio:.1%}"
        viability = "VIABLE" if cost_ratio < 0.10 else ("MARGINAL" if cost_ratio < 0.20 else "COSTLY")

        logger.info(
            f"  {hz_name:>4s}: avg |ret| = {pooled_mean:6.3f}t  "
            f"median = {pooled_median:6.3f}t  "
            f"cost_ratio = {cost_ratio_str:>7s}  [{viability}]"
        )
        cost_ratio_results[hz_name] = {
            "pooled_mean_abs_ret_ticks": pooled_mean,
            "pooled_median_abs_ret_ticks": pooled_median,
            "per_day_mean_abs_ret_ticks": per_day_mean,
            "cost_ticks_rt": TOTAL_COST_TICKS,
            "cost_ratio": cost_ratio,
            "viability": viability,
        }

    logger.info("")
    logger.info("=" * 60)
    logger.info("RESULTS: IC DECAY CURVES PER FEATURE")
    logger.info("=" * 60)

    # Header
    hz_header = "  ".join(f"{h:>7s}" for h in hz_names)
    logger.info(f"{'Feature':<25s}  {hz_header}")
    logger.info("-" * (25 + 2 + len(hz_names) * 9))

    feature_summary = {}
    for spec in specs:
        feat_name = spec["name"]
        ic_by_hz = {}
        tstat_by_hz = {}
        pctpos_by_hz = {}
        n_days_by_hz = {}

        for hz_name in hz_names:
            ics = per_day_ic[feat_name][hz_name]
            if len(ics) < 5:
                ic_by_hz[hz_name] = float("nan")
                tstat_by_hz[hz_name] = float("nan")
                pctpos_by_hz[hz_name] = float("nan")
                n_days_by_hz[hz_name] = len(ics)
                continue
            mean_ic = float(np.mean(ics))
            std_ic = float(np.std(ics))
            n = len(ics)
            t_stat = mean_ic / (std_ic / np.sqrt(n)) if std_ic > 0 else 0.0
            pct_pos = float(np.mean([x > 0 for x in ics]) * 100)
            ic_by_hz[hz_name] = mean_ic
            tstat_by_hz[hz_name] = float(t_stat)
            pctpos_by_hz[hz_name] = pct_pos
            n_days_by_hz[hz_name] = n

        # IC decay curve string
        ic_vals = [ic_by_hz.get(h, float("nan")) for h in hz_names]
        ic_str = "  ".join(
            f"{v:+.4f}" if np.isfinite(v) else "    ——  " for v in ic_vals
        )
        logger.info(f"{feat_name:<25s}  {ic_str}")

        # Compute IC retention at each horizon vs 10s baseline
        baseline_ic = ic_by_hz.get("10s", float("nan"))
        ic_retention = {}
        for hz_name in hz_names:
            hz_ic = ic_by_hz.get(hz_name, float("nan"))
            if np.isfinite(baseline_ic) and baseline_ic != 0 and np.isfinite(hz_ic):
                ic_retention[hz_name] = float(hz_ic / baseline_ic)
            else:
                ic_retention[hz_name] = float("nan")

        # Check persistence: >50% of baseline IC at 5m
        ic_5m = ic_by_hz.get("5m", float("nan"))
        is_persistent = (
            np.isfinite(baseline_ic)
            and np.isfinite(ic_5m)
            and baseline_ic > 0
            and ic_5m > 0
            and (ic_5m / baseline_ic) > 0.5
        )

        feature_summary[feat_name] = {
            "description": spec["desc"],
            "ic_by_horizon": ic_by_hz,
            "tstat_by_horizon": tstat_by_hz,
            "pct_positive_by_horizon": pctpos_by_hz,
            "n_days_by_horizon": n_days_by_hz,
            "ic_retention_vs_10s": ic_retention,
            "ic_persistent_at_5m": is_persistent,
            "per_day_ic": {hz: per_day_ic[feat_name][hz] for hz in hz_names},
        }

    logger.info("")

    # ============================================================
    # PERSISTENCE ANALYSIS
    # ============================================================

    logger.info("=" * 60)
    logger.info("PERSISTENCE ANALYSIS: IC at 5m vs 10s baseline")
    logger.info("=" * 60)

    persistent_features = []
    decaying_features = []

    for feat_name, fsumm in feature_summary.items():
        ic_10s = fsumm["ic_by_horizon"].get("10s", float("nan"))
        ic_5m = fsumm["ic_by_horizon"].get("5m", float("nan"))
        ic_15m = fsumm["ic_by_horizon"].get("15m", float("nan"))
        ic_30m = fsumm["ic_by_horizon"].get("30m", float("nan"))
        retention_5m = fsumm["ic_retention_vs_10s"].get("5m", float("nan"))

        if not np.isfinite(ic_10s) or ic_10s <= 0:
            continue

        status = "PERSISTENT" if fsumm["ic_persistent_at_5m"] else "DECAYING"
        logger.info(
            f"  {feat_name:<25s}  10s={ic_10s:+.4f}  5m={ic_5m:+.4f}  "
            f"15m={ic_15m:+.4f}  30m={ic_30m:+.4f}  "
            f"retention@5m={retention_5m:.1%}  [{status}]"
        )
        if fsumm["ic_persistent_at_5m"]:
            persistent_features.append(feat_name)
        else:
            decaying_features.append(feat_name)

    logger.info("")
    logger.info(f"PERSISTENT features (>50% IC at 5m): {persistent_features}")
    logger.info(f"DECAYING  features (<50% IC at 5m): {decaying_features}")

    # ============================================================
    # SPECIFIC FEATURE CHECK: The 4 features mentioned in the task
    # ============================================================

    logger.info("")
    logger.info("=" * 60)
    logger.info("FOCUS: Key features from task spec")
    logger.info("  microprice_dev, cancel_asym_5, ofi_5, depth_ratio_l1")
    logger.info("=" * 60)

    focus_features = ["microprice_dev", "cancel_asym_5", "ofi_5", "depth_ratio_l1"]
    for feat in focus_features:
        if feat not in feature_summary:
            logger.warning(f"  {feat}: not in results")
            continue
        fsumm = feature_summary[feat]
        logger.info(f"\n  {feat} — {fsumm['description']}")
        logger.info(f"  {'Horizon':<8s}  {'Mean IC':>8s}  {'t-stat':>8s}  {'pct+':>6s}  {'cost_ratio':>10s}  {'verdict'}")
        logger.info(f"  {'-'*70}")
        for hz in hz_names:
            ic = fsumm["ic_by_horizon"].get(hz, float("nan"))
            t = fsumm["tstat_by_horizon"].get(hz, float("nan"))
            pp = fsumm["pct_positive_by_horizon"].get(hz, float("nan"))
            cr = cost_ratio_results.get(hz, {}).get("cost_ratio", float("nan"))
            if np.isfinite(ic):
                verdict = ""
                if np.isfinite(cr) and cr < 0.10 and ic > 0.02:
                    verdict = "<-- VIABLE: good IC + manageable cost"
                elif np.isfinite(cr) and cr < 0.10:
                    verdict = "<-- cost manageable (but IC low)"
                logger.info(
                    f"  {hz:<8s}  {ic:>+8.4f}  {t:>8.2f}  {pp:>5.1f}%  {cr:>9.1%}  {verdict}"
                )
            else:
                logger.info(f"  {hz:<8s}  {'N/A':>8s}  {'N/A':>8s}  {'N/A':>5s}  {'N/A':>9s}")

    # ============================================================
    # CONCLUSION / KEY FINDINGS
    # ============================================================

    logger.info("")
    logger.info("=" * 60)
    logger.info("KEY FINDINGS SUMMARY")
    logger.info("=" * 60)

    # Find viable horizon (cost_ratio < 0.10)
    viable_horizons = [
        hz for hz, stats in cost_ratio_results.items()
        if stats["cost_ratio"] < 0.10
    ]
    marginal_horizons = [
        hz for hz, stats in cost_ratio_results.items()
        if 0.10 <= stats["cost_ratio"] < 0.20
    ]

    logger.info(f"\n1. COST VIABILITY:")
    logger.info(f"   Viable horizons (cost < 10% of move): {viable_horizons}")
    logger.info(f"   Marginal horizons (10-20%): {marginal_horizons}")

    logger.info(f"\n2. FEATURE PERSISTENCE:")
    logger.info(f"   Features with persistent IC at 5m+: {persistent_features}")
    logger.info(f"   Decaying features: {decaying_features}")

    # Find features that are BOTH viable horizon AND have IC
    logger.info(f"\n3. COMBINED ASSESSMENT:")
    for feat_name, fsumm in feature_summary.items():
        for hz in viable_horizons:
            ic = fsumm["ic_by_horizon"].get(hz, float("nan"))
            t = fsumm["tstat_by_horizon"].get(hz, float("nan"))
            if np.isfinite(ic) and ic > 0.02 and np.isfinite(t) and t > 2.0:
                cr = cost_ratio_results[hz]["cost_ratio"]
                logger.info(
                    f"   ACTIONABLE: {feat_name} at {hz}: "
                    f"IC={ic:+.4f} t={t:.1f} cost_ratio={cr:.1%}"
                )

    logger.info(f"\n4. RECOMMENDATION:")
    if viable_horizons and persistent_features:
        logger.info(f"   YES: Longer-horizon strategy is viable.")
        logger.info(f"   Use features: {persistent_features}")
        logger.info(f"   At horizons: {viable_horizons}")
    elif viable_horizons:
        logger.info(f"   PARTIAL: Cost is manageable at {viable_horizons},")
        logger.info(f"   but IC decays quickly. Need stronger signals or models.")
    else:
        logger.info(f"   CAUTION: Even at 1h, costs may be problematic.")
        logger.info(f"   Consider increasing position size or using limit orders.")

    # ============================================================
    # SAVE RESULTS
    # ============================================================

    output = {
        "timestamp": _ts,
        "n_days": len(days),
        "day_dates": day_dates,
        "horizons": {hz: bars for hz, bars in horizons_to_use.items()},
        "constants": {
            "tick_size": TICK_SIZE,
            "tick_value": TICK_VALUE,
            "cost_ticks_rt_total": TOTAL_COST_TICKS,
            "bars_per_second": BARS_PER_SEC,
        },
        "cost_ratio_by_horizon": cost_ratio_results,
        "feature_results": feature_summary,
        "persistent_features": persistent_features,
        "decaying_features": decaying_features,
        "viable_horizons": viable_horizons,
        "marginal_horizons": marginal_horizons,
        "elapsed_sec": total_elapsed,
    }

    def _json_safe(obj):
        """Recursively convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_json_safe(v) for v in obj]
        elif isinstance(obj, bool):
            return bool(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            v = float(obj)
            return None if (v != v) else v  # NaN -> None for JSON
        elif isinstance(obj, float):
            return None if (obj != obj) else obj  # NaN -> None for JSON
        else:
            return obj

    out_path = RESULTS_DIR / f"longer_horizon_ic_{_ts}.json"
    with open(str(out_path), "w") as fp:
        json.dump(_json_safe(output), fp, indent=2)
    logger.info(f"\nResults saved to: {out_path}")
    logger.info(f"Log file: {_log_file}")

    return output


# ============================================================================
# DISCORD PROGRESS REPORTER
# ============================================================================

def send_discord_update(msg: str) -> None:
    """Send progress update to Discord. Gracefully no-ops if Discord unavailable."""
    try:
        sys.path.insert(0, str(LVL3_ROOT.parent / "teleclaude-main"))
        from lib.discord import send_message
        send_message(msg)
    except Exception:
        try:
            # Fallback: try the teleclaude discord lib directly
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "discord_lib",
                LVL3_ROOT.parent / "teleclaude-main" / "lib" / "discord.js",
            )
        except Exception:
            pass  # Discord not available, skip silently


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Longer-Horizon IC Decay Analysis for ES Futures MBO Data"
    )
    parser.add_argument(
        "--n-days", type=int, default=None,
        help="Number of days to analyze (default: all 100)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: analyze only top 10 features"
    )
    parser.add_argument(
        "--cache-dir", type=str, default=None,
        help="Override feature cache directory"
    )
    args = parser.parse_args()

    if args.cache_dir:
        global FEATURE_CACHE
        FEATURE_CACHE = Path(args.cache_dir)

    logger.info("Starting longer-horizon IC decay analysis...")
    logger.info(f"Args: n_days={args.n_days}, quick={args.quick}")
    logger.info("")

    try:
        results = run_analysis(n_days=args.n_days, quick=args.quick)

        # Summary for Discord
        n_days = results["n_days"]
        viable = results["viable_horizons"]
        persistent = results["persistent_features"]

        cr_summary = []
        for hz, stats in results["cost_ratio_by_horizon"].items():
            cr = stats["cost_ratio"]
            avg_move = stats["pooled_mean_abs_ret_ticks"]
            cr_summary.append(f"{hz}: avg={avg_move:.2f}t cost={cr:.0%}")

        ic_summary = []
        for feat in ["microprice_dev", "cancel_asym_5", "ofi_5", "depth_ratio_l1"]:
            if feat in results["feature_results"]:
                fres = results["feature_results"][feat]
                ic_5m = fres["ic_by_horizon"].get("5m", float("nan"))
                ic_30m = fres["ic_by_horizon"].get("30m", float("nan"))
                if np.isfinite(ic_5m):
                    ic_summary.append(f"{feat}@5m={ic_5m:+.4f} @30m={ic_30m:+.4f}")

        msg = (
            f"LONGER-HORIZON IC ANALYSIS COMPLETE ({n_days} days)\n"
            f"Cost ratios: {' | '.join(cr_summary)}\n"
            f"Viable horizons (<10% cost): {viable}\n"
            f"Persistent features (>50% IC at 5m): {persistent}\n"
            f"Key feature ICs:\n  " + "\n  ".join(ic_summary)
        )
        logger.info("\n" + msg)

    except Exception as e:
        logger.exception(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()
