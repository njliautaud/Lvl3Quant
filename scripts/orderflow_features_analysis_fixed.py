#!/usr/bin/env python3
"""
Orderflow Feature Engineering & Analysis
=========================================
Computes orderflow-based features from MBO event data and analyzes their
predictive power for identifying LARGE price moves (P75+ = abs(labels_10s) >= 3 ticks).

Data source:  /home/jupiter/Lvl3Quant/data/processed/mbo_events/
              Each NPZ has:
                events:     (N, 6) float32 — [time_delta_log, event_type_id, side_id,
                                               price_rel_ticks, qty_log, spread_ticks]
                labels_Xs:  (N,) float32 — price change in ticks at horizon X
                timestamps: (N,) int64 — nanoseconds

Event encoding (from metadata):
    action_encoding: A=0 (Add), C=1 (Cancel), M=2 (Modify), T=3 (Trade), F=4 (Fill)
    side_encoding:   B=0 (Bid/Buy), A=1 (Ask/Sell), N=2 (None)

Features computed:
    1. Trade Imbalance (multiple windows: 100, 500, 1000, 5000)
    2. Order Flow Delta — net aggressive volume
    3. Queue Pressure — bid vs ask limit adds/cancels
    4. Spread Dynamics — widening/tightening
    5. Large Trade Detection — P95+ qty cluster
    6. Volatility Regime — rolling realized vol

Analysis:
    - Correlation with |labels_10s| and signed labels_10s
    - Big-move (abs >= 3 ticks) vs small-move separability (AUC, KS)
    - LGBM models: binary big-move detection, direction, regression
    - IC, accuracy, confusion matrix

Usage:
    python scripts/orderflow_features_analysis.py
    python scripts/orderflow_features_analysis.py --date 20260201
    python scripts/orderflow_features_analysis.py --sample 2000000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

warnings.filterwarnings("ignore")

# ─── Optional imports ────────────────────────────────────────────────────────

try:
    from scipy.stats import spearmanr, ks_2samp
    from scipy.stats import rankdata
    HAS_SCIPY = True
    print("[OK] scipy available")
except ImportError:
    HAS_SCIPY = False
    print("[WARN] scipy not available — falling back to numpy-based stats")

try:
    import lightgbm as lgb
    HAS_LGB = True
    print("[OK] LightGBM available")
except ImportError:
    HAS_LGB = False
    print("[WARN] LightGBM not available — skipping ML section")

try:
    from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import label_binarize
    HAS_SKLEARN = True
    print("[OK] scikit-learn available")
except ImportError:
    HAS_SKLEARN = False
    print("[WARN] scikit-learn not available — simplified metrics only")

# ─── Constants ────────────────────────────────────────────────────────────────

DATA_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events")
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/fill_sim_test/orderflow_analysis")
DEFAULT_DATE = "20260128"

# Event type IDs (from metadata action_encoding)
ETYPE_ADD    = 0
ETYPE_CANCEL = 1
ETYPE_MODIFY = 2
ETYPE_TRADE  = 3
ETYPE_FILL   = 4

# Side IDs
SIDE_BID = 0  # Buy side
SIDE_ASK = 1  # Sell side

# Column indices for events array
COL_TIME_DELTA = 0
COL_ETYPE      = 1
COL_SIDE       = 2
COL_PRICE      = 3
COL_QTY_LOG    = 4
COL_SPREAD     = 5

BIG_MOVE_THRESHOLD = 3.0  # ticks — P75 threshold for "large move"

# Windows for rolling features (in events)
WINDOWS = [100, 500, 1000, 5000]


# ─── Utilities ────────────────────────────────────────────────────────────────

def print_section(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def spearman_ic(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman IC, handling NaNs and near-constant arrays."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10:
        return float("nan")
    x_m, y_m = x[mask], y[mask]
    # Guard: if either array is near-constant, correlation is undefined
    if x_m.std() < 1e-12 or y_m.std() < 1e-12:
        return float("nan")
    if HAS_SCIPY:
        r, _ = spearmanr(x_m, y_m)
        return float(r)
    # Numpy fallback: rank-based correlation
    rx = rankdata_numpy(x_m)
    ry = rankdata_numpy(y_m)
    r = np.corrcoef(rx, ry)[0, 1]
    return float(r)


def rankdata_numpy(x: np.ndarray) -> np.ndarray:
    """Simple rank transform using numpy."""
    order = x.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(x) + 1)
    return ranks


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation, handling NaNs."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def ks_statistic(x_big: np.ndarray, x_small: np.ndarray) -> Tuple[float, float]:
    """KS statistic between two distributions."""
    x_big   = x_big[np.isfinite(x_big)]
    x_small = x_small[np.isfinite(x_small)]
    if len(x_big) < 5 or len(x_small) < 5:
        return float("nan"), float("nan")
    if HAS_SCIPY:
        stat, pval = ks_2samp(x_big, x_small)
        return float(stat), float(pval)
    # Simple approximation: max CDF difference
    combined = np.sort(np.concatenate([x_big, x_small]))
    cdf_big   = np.searchsorted(np.sort(x_big),   combined, side="right") / len(x_big)
    cdf_small = np.searchsorted(np.sort(x_small), combined, side="right") / len(x_small)
    stat = float(np.max(np.abs(cdf_big - cdf_small)))
    return stat, float("nan")


def auc_from_feature(feat: np.ndarray, label_binary: np.ndarray) -> float:
    """Approximate AUC using rank correlation."""
    mask = np.isfinite(feat)
    if mask.sum() < 10 or label_binary[mask].sum() < 5:
        return float("nan")
    if HAS_SKLEARN:
        try:
            return float(roc_auc_score(label_binary[mask], feat[mask]))
        except Exception:
            pass
    # Fallback: Wilcoxon approximation
    pos  = feat[mask & (label_binary == 1)]
    neg  = feat[mask & (label_binary == 0)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Sample if too large
    if len(pos) > 5000:
        pos = np.random.choice(pos, 5000, replace=False)
    if len(neg) > 5000:
        neg = np.random.choice(neg, 5000, replace=False)
    wins = np.mean(pos[:, None] > neg[None, :])
    ties = np.mean(pos[:, None] == neg[None, :])
    return float(wins + 0.5 * ties)


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_date(date: str, sample: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Load MBO NPZ file for a given date string (YYYYMMDD)."""
    fpath = DATA_DIR / f"{date}_mbo_events.npz"
    if not fpath.exists():
        print(f"[ERROR] File not found: {fpath}")
        sys.exit(1)

    print(f"[INFO] Loading {fpath} ...")
    t0 = time.time()
    npz = np.load(str(fpath), allow_pickle=True)

    data: Dict[str, np.ndarray] = {
        "events":     npz["events"],
        "labels_1s":  npz.get("labels_1s",  np.full(len(npz["events"]), np.nan, dtype=np.float32)),
        "labels_5s":  npz.get("labels_5s",  np.full(len(npz["events"]), np.nan, dtype=np.float32)),
        "labels_10s": npz.get("labels_10s", np.full(len(npz["events"]), np.nan, dtype=np.float32)),
        "timestamps": npz["timestamps"],
    }

    n_total = len(data["events"])
    print(f"[INFO] Loaded {n_total:,} events in {time.time()-t0:.1f}s")

    # Filter to RTH of the target date only (avoids overnight/pre-market noise)
    try:
        yr, mo, dy = int(date[:4]), int(date[4:6]), int(date[6:8])
        rth_start = int(datetime(yr, mo, dy, 14, 30, 0, tzinfo=timezone.utc).timestamp() * 1e9)
        rth_end   = int(datetime(yr, mo, dy, 21,  0, 0, tzinfo=timezone.utc).timestamp() * 1e9)
        ts = data["timestamps"]
        rth_mask = (ts >= rth_start) & (ts <= rth_end)
        if rth_mask.sum() > 100_000:
            for key in data:
                data[key] = data[key][rth_mask]
            print(f"[INFO] RTH filter → {len(data['events']):,} events")
    except Exception as e:
        print(f"[WARN] RTH filter failed: {e} — using all events")

    # Optional downsample for speed
    if sample is not None and len(data["events"]) > sample:
        idx = np.linspace(0, len(data["events"]) - 1, sample, dtype=int)
        for key in data:
            data[key] = data[key][idx]
        print(f"[INFO] Sampled → {len(data['events']):,} events")

    return data


# ─── Feature Engineering ──────────────────────────────────────────────────────

def rolling_sum_fast(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Compute causal rolling sum using cumsum trick. No look-ahead.
    result[i] = sum(arr[max(0, i-window+1) : i+1])
    """
    cs = np.cumsum(arr, dtype=np.float64)
    out = cs.copy()
    out[window:] -= cs[:-window]
    return out.astype(np.float32)


def rolling_mean_fast(arr: np.ndarray, window: int) -> np.ndarray:
    cs = np.cumsum(arr, dtype=np.float64)
    count = np.minimum(np.arange(1, len(arr) + 1), window).astype(np.float64)
    out = cs.copy()
    out[window:] -= cs[:-window]
    return (out / count).astype(np.float32)


def rolling_std_fast(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling std using the identity Var[X] = E[X²] - E[X]²."""
    mu  = rolling_mean_fast(arr, window).astype(np.float64)
    mu2 = rolling_mean_fast(arr ** 2, window).astype(np.float64)
    var = np.maximum(mu2 - mu ** 2, 0.0)
    return np.sqrt(var).astype(np.float32)


def compute_features(data: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Compute all orderflow features. All features are strictly causal
    (look-behind only — no information from future events).

    Returns:
        features_dict: {feature_name: (N,) array}
        feature_names: ordered list
    """
    events     = data["events"]          # (N, 6)
    N          = len(events)
    feat: Dict[str, np.ndarray] = {}

    print_section("Feature Engineering")
    t0 = time.time()

    # ── Extract event columns ─────────────────────────────────────────────
    etype     = events[:, COL_ETYPE].astype(np.float32)
    side      = events[:, COL_SIDE].astype(np.float32)
    qty_log   = events[:, COL_QTY_LOG].astype(np.float32)
    spread    = events[:, COL_SPREAD].astype(np.float32)
    price_rel = events[:, COL_PRICE].astype(np.float32)

    # ── Boolean masks for event types ────────────────────────────────────
    is_trade  = ((etype == ETYPE_TRADE) | (etype == ETYPE_FILL)).astype(np.float32)
    is_add    = (etype == ETYPE_ADD).astype(np.float32)
    is_cancel = (etype == ETYPE_CANCEL).astype(np.float32)
    is_buy    = (side == SIDE_BID).astype(np.float32)
    is_sell   = (side == SIDE_ASK).astype(np.float32)

    # Aggressive trades broken out
    buy_trade  = (is_trade * is_buy)
    sell_trade = (is_trade * is_sell)

    # Actual volume (exp of log qty), clipped for outlier safety
    vol = np.clip(np.exp(qty_log.astype(np.float64)), 1.0, 1e6).astype(np.float32)
    buy_trade_vol  = buy_trade  * vol
    sell_trade_vol = sell_trade * vol

    # ── 1. Trade Imbalance (multiple windows) ────────────────────────────
    print(f"  [1/6] Trade Imbalance ({WINDOWS}) ...")
    for w in WINDOWS:
        rb = rolling_sum_fast(buy_trade,  w)
        rs = rolling_sum_fast(sell_trade, w)
        imb = (rb - rs) / (rb + rs + 1.0)
        feat[f"trade_imb_{w}"] = imb.astype(np.float32)

    # ── 2. Order Flow Delta — net aggressive volume ───────────────────────
    print(f"  [2/6] Order Flow Delta ({WINDOWS}) ...")
    for w in WINDOWS:
        bvol = rolling_sum_fast(buy_trade_vol,  w)
        svol = rolling_sum_fast(sell_trade_vol, w)
        # Raw delta
        feat[f"ofd_{w}"]  = (bvol - svol).astype(np.float32)
        # Normalized by total volume
        total_vol = bvol + svol + 1e-6
        feat[f"ofd_norm_{w}"] = ((bvol - svol) / total_vol).astype(np.float32)

    # ── 3. Queue Pressure ─────────────────────────────────────────────────
    print("  [3/6] Queue Pressure ...")
    # Bid-side adds = building bid queue (bullish pressure)
    # Ask-side cancels = weakening ask (also bullish)
    bid_add    = (is_add    * is_buy)
    ask_add    = (is_add    * is_sell)
    bid_cancel = (is_cancel * is_buy)
    ask_cancel = (is_cancel * is_sell)

    bid_add_vol    = bid_add    * vol
    ask_add_vol    = ask_add    * vol
    bid_cancel_vol = bid_cancel * vol
    ask_cancel_vol = ask_cancel * vol

    for w in [500, 1000, 5000]:
        rb_add  = rolling_sum_fast(bid_add_vol,    w)
        ra_add  = rolling_sum_fast(ask_add_vol,    w)
        rb_can  = rolling_sum_fast(bid_cancel_vol, w)
        ra_can  = rolling_sum_fast(ask_cancel_vol, w)

        # Net bid pressure: bid adds - bid cancels
        bid_pressure = rb_add - rb_can
        ask_pressure = ra_add - ra_can
        feat[f"bid_pressure_{w}"] = bid_pressure.astype(np.float32)
        feat[f"ask_pressure_{w}"] = ask_pressure.astype(np.float32)

        # Pressure imbalance: bid building vs ask building
        denom = np.abs(bid_pressure) + np.abs(ask_pressure) + 1.0
        feat[f"queue_imb_{w}"] = ((bid_pressure - ask_pressure) / denom).astype(np.float32)

        # Count-based imbalance (ignoring size)
        n_bid_add = rolling_sum_fast(bid_add,    w)
        n_ask_add = rolling_sum_fast(ask_add,    w)
        n_bid_can = rolling_sum_fast(bid_cancel, w)
        n_ask_can = rolling_sum_fast(ask_cancel, w)
        net_bid_q = n_bid_add - n_bid_can
        net_ask_q = n_ask_add - n_ask_can
        feat[f"queue_count_imb_{w}"] = ((net_bid_q - net_ask_q) / (
            np.abs(net_bid_q) + np.abs(net_ask_q) + 1.0)).astype(np.float32)

    # ── 4. Spread Dynamics ───────────────────────────────────────────────
    print("  [4/6] Spread Dynamics ...")
    feat["spread_cur"] = spread.copy()

    for w in [50, 200, 1000]:
        avg_spread = rolling_mean_fast(spread, w)
        feat[f"spread_avg_{w}"]    = avg_spread
        feat[f"spread_vs_avg_{w}"] = (spread - avg_spread).astype(np.float32)
        # Spread change: current vs rolling min (squeeze detection)
        # Use rolling std to detect volatility in spread
        std_spread = rolling_std_fast(spread, w)
        feat[f"spread_std_{w}"] = std_spread

    # Spread squeeze: current spread is tight relative to recent range
    # (low spread after elevated spread = breakout condition)
    spread_avg_slow = rolling_mean_fast(spread, 2000)
    spread_avg_fast = rolling_mean_fast(spread, 100)
    feat["spread_squeeze"] = (spread_avg_slow - spread_avg_fast).astype(np.float32)

    # Widening indicator: recent spread much higher than average
    feat["spread_widen"] = (spread_avg_fast - spread_avg_slow).astype(np.float32)

    # ── 5. Large Trade Detection ─────────────────────────────────────────
    print("  [5/6] Large Trade Detection ...")
    # Trade-only qty
    trade_qty_log = qty_log * is_trade  # 0 for non-trades

    # Rolling P95 of trade qty (use a proxy: mean + 1.65*std ≈ P95 for normal)
    for w in [500, 2000]:
        mu_qty  = rolling_mean_fast(trade_qty_log, w)
        std_qty = rolling_std_fast(trade_qty_log,  w)
        p95_proxy = mu_qty + 1.65 * std_qty

        is_large = ((trade_qty_log > p95_proxy) & (is_trade > 0)).astype(np.float32)
        is_large_buy  = is_large * is_buy
        is_large_sell = is_large * is_sell

        # Large trade imbalance
        lb = rolling_sum_fast(is_large_buy,  min(w, 500))
        ls = rolling_sum_fast(is_large_sell, min(w, 500))
        feat[f"large_trade_imb_{w}"] = ((lb - ls) / (lb + ls + 1.0)).astype(np.float32)

        # Large trade rate (fraction of recent trades that are large)
        n_trades = rolling_sum_fast(is_trade, min(w, 500))
        n_large  = rolling_sum_fast(is_large, min(w, 500))
        feat[f"large_trade_rate_{w}"] = (n_large / (n_trades + 1.0)).astype(np.float32)

        # Cluster: multiple consecutive large trades in same direction
        large_buy_cluster  = rolling_sum_fast(is_large_buy,  50)
        large_sell_cluster = rolling_sum_fast(is_large_sell, 50)
        feat[f"large_cluster_buy_{w}"]  = large_buy_cluster.astype(np.float32)
        feat[f"large_cluster_sell_{w}"] = large_sell_cluster.astype(np.float32)
        feat[f"large_cluster_net_{w}"]  = (large_buy_cluster - large_sell_cluster).astype(np.float32)

    # Large trade volume contribution
    large_buy_vol  = (is_large_buy  * vol)
    large_sell_vol = (is_large_sell * vol)
    for w in [500, 2000]:
        lbv = rolling_sum_fast(large_buy_vol,  w)
        lsv = rolling_sum_fast(large_sell_vol, w)
        feat[f"large_vol_imb_{w}"] = ((lbv - lsv) / (lbv + lsv + 1.0)).astype(np.float32)

    # ── 6. Volatility Regime ─────────────────────────────────────────────
    print("  [6/6] Volatility Regime ...")
    # Realized vol from price_rel_ticks changes (only for trades/fills)
    # price_rel_ticks = distance from mid (signed), so changes capture activity
    price_sq = price_rel ** 2

    for w in [100, 500, 2000]:
        rvol = rolling_mean_fast(price_sq, w)
        feat[f"rvol_{w}"] = np.sqrt(rvol).astype(np.float32)

    # Vol regime: fast vol vs slow vol
    rvol_fast = np.sqrt(rolling_mean_fast(price_sq, 200).astype(np.float64)).astype(np.float32)
    rvol_slow = np.sqrt(rolling_mean_fast(price_sq, 2000).astype(np.float64)).astype(np.float32)
    feat["rvol_regime"] = (rvol_fast / (rvol_slow + 1e-6)).astype(np.float32)

    # Trade intensity: trades per recent window
    for w in [100, 500]:
        feat[f"trade_intensity_{w}"] = rolling_mean_fast(is_trade, w)

    # Vol percentile: current rvol vs recent range
    # Approximate with rolling z-score
    rvol_cur = feat["rvol_500"]
    rvol_mu  = rolling_mean_fast(rvol_cur, 5000)
    rvol_std = rolling_std_fast(rvol_cur, 5000)
    feat["rvol_zscore"] = ((rvol_cur - rvol_mu) / (rvol_std + 1e-6)).astype(np.float32)

    # ── Composite / Interaction features ─────────────────────────────────
    print("  [+] Composite features ...")

    # Conviction signal: trade imbalance weighted by vol and intensity
    feat["conviction_500"]  = feat["trade_imb_500"]  * feat["ofd_norm_500"]
    feat["conviction_1000"] = feat["trade_imb_1000"] * feat["ofd_norm_1000"]

    # Momentum × volatility regime: high-vol periods where imbalance is strong = breakout
    # (more robust than spread_squeeze when spread is nearly constant)
    feat["breakout_signal"] = (feat["trade_imb_500"] * feat["rvol_regime"]).astype(np.float32)

    # Combined imbalance: trade + queue (same sign = stronger signal)
    feat["combo_imb_1000"] = (feat["trade_imb_1000"] + feat["queue_imb_1000"]).astype(np.float32)
    feat["combo_imb_5000"] = (feat["trade_imb_5000"] + feat["queue_imb_5000"]).astype(np.float32)

    # Acceleration: fast imbalance vs slow imbalance
    feat["imb_accel"] = (feat["trade_imb_100"] - feat["trade_imb_1000"]).astype(np.float32)
    feat["ofd_accel"] = (feat["ofd_norm_500"]  - feat["ofd_norm_5000"]).astype(np.float32)

    # Volume-weighted queue imbalance
    feat["vwq_imb_1000"] = (feat["queue_imb_1000"] * feat["ofd_norm_1000"]).astype(np.float32)

    elapsed = time.time() - t0
    print(f"  → {len(feat)} features computed in {elapsed:.1f}s")

    feature_names = sorted(feat.keys())
    return feat, feature_names


# ─── Analysis ─────────────────────────────────────────────────────────────────

def analyze_correlations(
    feat: Dict[str, np.ndarray],
    feature_names: List[str],
    labels_10s: np.ndarray,
    valid_mask: np.ndarray,
) -> Dict[str, Dict]:
    """Compute Pearson + Spearman IC with |labels_10s| and signed labels_10s."""
    print_section("Correlation Analysis")

    labels_abs    = np.abs(labels_10s)
    labels_signed = labels_10s

    results = {}
    for fname in feature_names:
        f = feat[fname][valid_mask]
        la  = labels_abs[valid_mask]
        ls  = labels_signed[valid_mask]

        ic_abs    = spearman_ic(f, la)
        ic_signed = spearman_ic(f, ls)
        pr_abs    = pearson_r(f, la)
        pr_signed = pearson_r(f, ls)

        results[fname] = {
            "spearman_abs":    ic_abs,
            "spearman_signed": ic_signed,
            "pearson_abs":     pr_abs,
            "pearson_signed":  pr_signed,
        }

    # Sort by |spearman_abs|
    sorted_feats = sorted(results.items(), key=lambda x: abs(x[1]["spearman_abs"] or 0), reverse=True)

    print(f"\n{'Feature':<35} {'SpearAbs':>9} {'SpearSgn':>9} {'PearAbs':>9} {'PearSgn':>9}")
    print("-" * 75)
    for fname, r in sorted_feats[:30]:
        sa  = r["spearman_abs"]    or 0
        ss  = r["spearman_signed"] or 0
        pa  = r["pearson_abs"]     or 0
        ps  = r["pearson_signed"]  or 0
        print(f"{fname:<35} {sa:>9.4f} {ss:>9.4f} {pa:>9.4f} {ps:>9.4f}")

    return results


def analyze_big_move_separability(
    feat: Dict[str, np.ndarray],
    feature_names: List[str],
    labels_10s: np.ndarray,
    valid_mask: np.ndarray,
    threshold: float = BIG_MOVE_THRESHOLD,
) -> Dict[str, Dict]:
    """
    For each feature, test if it can separate big-move vs small-move events.
    Reports KS statistic and AUC.
    """
    print_section(f"Big Move Separability (|labels_10s| >= {threshold} ticks)")

    labels_abs = np.abs(labels_10s)
    big_mask   = valid_mask & (labels_abs >= threshold)
    small_mask = valid_mask & (labels_abs < threshold)

    n_big   = big_mask.sum()
    n_small = small_mask.sum()
    print(f"  Big moves: {n_big:,} ({100*n_big/(n_big+n_small):.1f}%)")
    print(f"  Small moves: {n_small:,} ({100*n_small/(n_big+n_small):.1f}%)")

    # For AUC we need balanced approach — sample small for speed
    np.random.seed(42)
    max_per_class = min(n_big, n_small, 200_000)
    big_idx   = np.where(big_mask)[0]
    small_idx = np.where(small_mask)[0]
    if len(big_idx)   > max_per_class:
        big_idx   = np.random.choice(big_idx,   max_per_class, replace=False)
    if len(small_idx) > max_per_class:
        small_idx = np.random.choice(small_idx, max_per_class, replace=False)

    binary_label = np.zeros(len(labels_10s), dtype=np.int8)
    binary_label[big_mask] = 1

    results = {}
    for fname in feature_names:
        f       = feat[fname]
        f_big   = f[big_idx]
        f_small = f[small_idx]

        # Mean difference (normalized)
        mu_big   = np.nanmean(f_big)
        mu_small = np.nanmean(f_small)
        std_pool = (np.nanstd(f_big) + np.nanstd(f_small)) / 2 + 1e-10
        norm_diff = (mu_big - mu_small) / std_pool

        ks_stat, ks_pval = ks_statistic(f_big, f_small)

        # AUC: does higher feature value predict big move?
        combined_f = np.concatenate([f_big, f_small])
        combined_y = np.concatenate([
            np.ones(len(f_big), dtype=np.int8),
            np.zeros(len(f_small), dtype=np.int8)
        ])
        finite_mask = np.isfinite(combined_f)
        if finite_mask.sum() > 20:
            auc = auc_from_feature(combined_f[finite_mask], combined_y[finite_mask])
        else:
            auc = float("nan")

        # Reflect AUC > 0.5 (direction-agnostic)
        if not np.isnan(auc):
            auc = max(auc, 1.0 - auc)

        results[fname] = {
            "mu_big":    float(mu_big),
            "mu_small":  float(mu_small),
            "norm_diff": float(norm_diff),
            "ks_stat":   float(ks_stat),
            "ks_pval":   float(ks_pval) if not np.isnan(ks_pval) else None,
            "auc":       float(auc),
        }

    # Sort by AUC
    sorted_feats = sorted(results.items(), key=lambda x: x[1]["auc"] if not np.isnan(x[1]["auc"]) else 0, reverse=True)

    print(f"\n{'Feature':<35} {'AUC':>7} {'KS_stat':>9} {'NormDiff':>10} {'mu_big':>9} {'mu_small':>9}")
    print("-" * 82)
    for fname, r in sorted_feats[:30]:
        auc  = r["auc"]
        ks   = r["ks_stat"]
        nd   = r["norm_diff"]
        mb   = r["mu_big"]
        ms   = r["mu_small"]
        print(f"{fname:<35} {auc:>7.4f} {ks:>9.4f} {nd:>10.4f} {mb:>9.3f} {ms:>9.3f}")

    return results


def run_lgbm_models(
    feat: Dict[str, np.ndarray],
    feature_names: List[str],
    labels_10s: np.ndarray,
    valid_mask: np.ndarray,
    threshold: float = BIG_MOVE_THRESHOLD,
) -> Dict[str, Dict]:
    """
    Train LGBM models on orderflow features:
      (a) Binary: |labels_10s| >= threshold
      (b) Direction: sign(labels_10s)
      (c) Regression: labels_10s directly

    Uses time-series split (no random shuffle).
    Reports IC, accuracy, AUC, confusion matrix.
    """
    print_section("LGBM Model Analysis")

    if not HAS_LGB:
        print("  [SKIP] LightGBM not available.")
        return {}

    # Build feature matrix (only valid rows)
    valid_idx = np.where(valid_mask)[0]

    # For speed, subsample if very large
    max_samples = 1_500_000
    if len(valid_idx) > max_samples:
        step = len(valid_idx) // max_samples
        valid_idx = valid_idx[::step]
        print(f"  Subsampled to {len(valid_idx):,} events for ML")

    X = np.column_stack([feat[fname][valid_idx] for fname in feature_names]).astype(np.float32)
    y = labels_10s[valid_idx]

    print(f"  Feature matrix: {X.shape[0]:,} x {X.shape[1]}")

    # Replace any inf/nan in X
    X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)

    # Time-series split: 70% train, 30% test (no shuffle — preserve temporal order)
    split_idx = int(len(valid_idx) * 0.70)
    X_train, X_test = X[:split_idx],  X[split_idx:]
    y_train, y_test = y[:split_idx],  y[split_idx:]

    print(f"  Train: {len(X_train):,}  Test: {len(X_test):,}")

    lgb_params_base = {
        "n_estimators":    300,
        "learning_rate":   0.05,
        "num_leaves":      63,
        "max_depth":       -1,
        "subsample":       0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 50,
        "random_state":    42,
        "n_jobs":          -1,
        "verbose":         -1,
    }

    results = {}

    # ── (a) Binary: big move detection ────────────────────────────────────
    print("\n  (a) Binary Classification — Big Move Detection")
    y_bin_train = (np.abs(y_train) >= threshold).astype(np.int32)
    y_bin_test  = (np.abs(y_test)  >= threshold).astype(np.int32)

    pos_rate = y_bin_train.mean()
    scale_pos_weight = (1 - pos_rate) / (pos_rate + 1e-10)
    print(f"      Positive rate (train): {pos_rate:.3f} → scale_pos_weight={scale_pos_weight:.2f}")

    clf = lgb.LGBMClassifier(
        **lgb_params_base,
        objective="binary",
        scale_pos_weight=scale_pos_weight,
    )
    clf.fit(X_train, y_bin_train,
            eval_set=[(X_test, y_bin_test)],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])

    prob_test   = clf.predict_proba(X_test)[:, 1]
    pred_test   = (prob_test >= 0.5).astype(np.int32)
    acc_bin     = float(np.mean(pred_test == y_bin_test))

    if HAS_SKLEARN:
        auc_bin = float(roc_auc_score(y_bin_test, prob_test))
        cm      = confusion_matrix(y_bin_test, pred_test)
    else:
        auc_bin = auc_from_feature(prob_test, y_bin_test)
        tn = int(np.sum((pred_test == 0) & (y_bin_test == 0)))
        fp = int(np.sum((pred_test == 1) & (y_bin_test == 0)))
        fn = int(np.sum((pred_test == 0) & (y_bin_test == 1)))
        tp = int(np.sum((pred_test == 1) & (y_bin_test == 1)))
        cm = np.array([[tn, fp], [fn, tp]])

    print(f"      Accuracy:  {acc_bin:.4f}")
    print(f"      AUC:       {auc_bin:.4f}")
    print(f"      Confusion matrix (TN FP / FN TP):")
    print(f"        {cm[0,0]:>8,} {cm[0,1]:>8,}")
    print(f"        {cm[1,0]:>8,} {cm[1,1]:>8,}")

    # Feature importance (top 20)
    fi = clf.feature_importances_
    top_idx = np.argsort(fi)[::-1][:20]
    print("      Top-20 features by importance:")
    for rank, i in enumerate(top_idx):
        print(f"        {rank+1:2d}. {feature_names[i]:<35} {fi[i]:>6}")

    results["binary"] = {
        "accuracy": acc_bin,
        "auc":      auc_bin,
        "confusion_matrix": cm.tolist(),
        "top_features": [(feature_names[i], int(fi[i])) for i in top_idx],
    }

    # ── (b) Direction: sign prediction ────────────────────────────────────
    print("\n  (b) Multiclass — Direction Prediction (-, 0, +)")
    # 3-class: negative (-1), near-zero (0), positive (+1)
    dir_thresh = 1.0  # ticks
    y_dir_train = np.where(y_train > dir_thresh, 2,
                  np.where(y_train < -dir_thresh, 0, 1)).astype(np.int32)
    y_dir_test  = np.where(y_test  > dir_thresh, 2,
                  np.where(y_test  < -dir_thresh, 0, 1)).astype(np.int32)

    counts = np.bincount(y_dir_train, minlength=3)
    print(f"      Class distribution (train): neg={counts[0]:,} zero={counts[1]:,} pos={counts[2]:,}")

    clf_dir = lgb.LGBMClassifier(
        **lgb_params_base,
        objective="multiclass",
        num_class=3,
    )
    clf_dir.fit(X_train, y_dir_train,
                eval_set=[(X_test, y_dir_test)],
                callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])

    pred_dir  = clf_dir.predict(X_test)
    acc_dir   = float(np.mean(pred_dir == y_dir_test))

    # Directional accuracy (only on non-zero ground truth)
    nz_mask   = y_dir_test != 1
    acc_dir_nz = float(np.mean(pred_dir[nz_mask] == y_dir_test[nz_mask])) if nz_mask.sum() > 0 else float("nan")

    if HAS_SKLEARN:
        cm_dir = confusion_matrix(y_dir_test, pred_dir)
    else:
        cm_dir = None

    print(f"      Overall accuracy:       {acc_dir:.4f}")
    print(f"      Directional acc (±):    {acc_dir_nz:.4f}")
    if cm_dir is not None:
        print(f"      Confusion matrix (neg/zero/pos):")
        for row in cm_dir:
            print(f"        {row[0]:>8,} {row[1]:>8,} {row[2]:>8,}")

    results["direction"] = {
        "accuracy":           acc_dir,
        "directional_acc_nz": acc_dir_nz,
        "confusion_matrix":   cm_dir.tolist() if cm_dir is not None else None,
    }

    # ── (c) Regression: labels_10s directly ──────────────────────────────
    print("\n  (c) Regression — labels_10s Direct Prediction")

    reg = lgb.LGBMRegressor(
        **lgb_params_base,
        objective="regression",
    )
    reg.fit(X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])

    preds_reg = reg.predict(X_test)

    # IC (Spearman)
    ic_reg = spearman_ic(preds_reg, y_test)

    # MSE
    mse = float(np.mean((preds_reg - y_test) ** 2))

    # Directional accuracy from regression
    sign_pred = np.sign(preds_reg)
    sign_true = np.sign(y_test)
    nz_true   = sign_true != 0
    dir_acc_reg = float(np.mean(sign_pred[nz_true] == sign_true[nz_true])) if nz_true.sum() > 0 else float("nan")

    print(f"      Spearman IC:       {ic_reg:.4f}")
    print(f"      MSE:               {mse:.4f}")
    print(f"      Dir accuracy (nz): {dir_acc_reg:.4f}")

    # Pearson r on big moves only
    big_test = np.abs(y_test) >= threshold
    ic_big   = spearman_ic(preds_reg[big_test], y_test[big_test]) if big_test.sum() > 10 else float("nan")
    print(f"      IC (big moves):    {ic_big:.4f}  (n={big_test.sum():,})")

    results["regression"] = {
        "spearman_ic":   ic_reg,
        "mse":           mse,
        "dir_acc_nz":    dir_acc_reg,
        "ic_big_moves":  ic_big,
    }

    return results


# ─── Summary Report ───────────────────────────────────────────────────────────

def print_summary(
    corr_results: Dict[str, Dict],
    sep_results:  Dict[str, Dict],
    lgbm_results: Dict[str, Dict],
    feature_names: List[str],
) -> None:
    print_section("SUMMARY REPORT")

    # Top features by each criterion
    if corr_results:
        top_abs = sorted(corr_results.items(),
                         key=lambda x: abs(x[1]["spearman_abs"] or 0), reverse=True)[:5]
        top_sgn = sorted(corr_results.items(),
                         key=lambda x: abs(x[1]["spearman_signed"] or 0), reverse=True)[:5]
        print("\nTop 5 features by |Spearman IC| with |labels_10s|:")
        for fname, r in top_abs:
            print(f"  {fname:<35} {r['spearman_abs']:.4f}")

        print("\nTop 5 features by |Spearman IC| with signed labels_10s:")
        for fname, r in top_sgn:
            print(f"  {fname:<35} {r['spearman_signed']:.4f}")

    if sep_results:
        top_sep = sorted(sep_results.items(),
                         key=lambda x: x[1]["auc"] if not np.isnan(x[1]["auc"]) else 0, reverse=True)[:5]
        print("\nTop 5 features by AUC for big-move detection:")
        for fname, r in top_sep:
            print(f"  {fname:<35} AUC={r['auc']:.4f}  KS={r['ks_stat']:.4f}")

    if lgbm_results:
        print("\nLGBM Model Results:")
        if "binary" in lgbm_results:
            b = lgbm_results["binary"]
            print(f"  Binary (big move detection):  Acc={b['accuracy']:.4f}  AUC={b['auc']:.4f}")
        if "direction" in lgbm_results:
            d = lgbm_results["direction"]
            print(f"  Direction (3-class):           Acc={d['accuracy']:.4f}  "
                  f"DirAcc(±)={d['directional_acc_nz']:.4f}")
        if "regression" in lgbm_results:
            r = lgbm_results["regression"]
            print(f"  Regression:                    IC={r['spearman_ic']:.4f}  "
                  f"MSE={r['mse']:.4f}  DirAcc={r['dir_acc_nz']:.4f}")

    print()


# ─── Save Outputs ─────────────────────────────────────────────────────────────

def save_outputs(
    date: str,
    feat: Dict[str, np.ndarray],
    feature_names: List[str],
    corr_results: Dict[str, Dict],
    sep_results:  Dict[str, Dict],
    lgbm_results: Dict[str, Dict],
    valid_mask: np.ndarray,
    labels_10s: np.ndarray,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save feature arrays (valid only to save space)
    feat_arrays = {fname: feat[fname][valid_mask] for fname in feature_names}
    feat_arrays["labels_10s"] = labels_10s[valid_mask]
    np_out = OUTPUT_DIR / f"{date}_features.npz"
    np.savez_compressed(str(np_out), **feat_arrays)
    print(f"\n[SAVED] Feature arrays → {np_out}")

    # Save analysis results as JSON
    analysis = {
        "date":        date,
        "n_features":  len(feature_names),
        "feature_names": feature_names,
        "correlations":  corr_results,
        "big_move_separability": sep_results,
        "lgbm_results": lgbm_results,
        "threshold_ticks": BIG_MOVE_THRESHOLD,
    }
    json_out = OUTPUT_DIR / f"{date}_analysis.json"
    with open(str(json_out), "w") as f:
        json.dump(analysis, f, indent=2, default=lambda x: None if (isinstance(x, float) and np.isnan(x)) else x)
    print(f"[SAVED] Analysis JSON  → {json_out}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Orderflow Feature Engineering & Analysis")
    parser.add_argument("--date",   type=str, default=DEFAULT_DATE,
                        help=f"Date to process (default: {DEFAULT_DATE})")
    parser.add_argument("--sample", type=int, default=None,
                        help="Subsample N events for faster testing")
    parser.add_argument("--no-ml",  action="store_true",
                        help="Skip LGBM section")
    args = parser.parse_args()

    print("=" * 70)
    print("  ORDERFLOW FEATURE ENGINEERING & ANALYSIS")
    print(f"  Date: {args.date}  |  Big-move threshold: {BIG_MOVE_THRESHOLD} ticks")
    print("=" * 70)

    # 1. Load data
    data = load_date(args.date, sample=args.sample)
    events     = data["events"]
    labels_10s = data["labels_10s"]
    N          = len(events)

    # Valid mask: non-NaN labels
    valid_mask = np.isfinite(labels_10s)
    print(f"\n[INFO] Valid labels: {valid_mask.sum():,} / {N:,}  "
          f"({100*valid_mask.mean():.1f}%)")

    abs_labels = np.abs(labels_10s[valid_mask])
    n_big   = (abs_labels >= BIG_MOVE_THRESHOLD).sum()
    n_valid = len(abs_labels)
    print(f"[INFO] Big moves (|lbl| >= {BIG_MOVE_THRESHOLD}): "
          f"{n_big:,} / {n_valid:,}  ({100*n_big/n_valid:.1f}%)")
    for p in [50, 75, 90, 95, 99]:
        print(f"       P{p:2d} |labels_10s|: {np.percentile(abs_labels, p):.2f} ticks")

    # 2. Compute features
    feat, feature_names = compute_features(data)

    # 3. Correlation analysis
    corr_results = analyze_correlations(feat, feature_names, labels_10s, valid_mask)

    # 4. Big-move separability
    sep_results = analyze_big_move_separability(feat, feature_names, labels_10s, valid_mask)

    # 5. LGBM models
    lgbm_results = {}
    if not args.no_ml:
        lgbm_results = run_lgbm_models(feat, feature_names, labels_10s, valid_mask)
    else:
        print_section("LGBM Model Analysis")
        print("  [SKIP] --no-ml flag set.")

    # 6. Summary
    print_summary(corr_results, sep_results, lgbm_results, feature_names)

    # 7. Save outputs
    save_outputs(
        date=args.date,
        feat=feat,
        feature_names=feature_names,
        corr_results=corr_results,
        sep_results=sep_results,
        lgbm_results=lgbm_results,
        valid_mask=valid_mask,
        labels_10s=labels_10s,
    )

    print("\n[DONE]")


if __name__ == "__main__":
    main()
