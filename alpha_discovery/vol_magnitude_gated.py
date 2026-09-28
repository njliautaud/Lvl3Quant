"""
Volatility-Gated and Magnitude-Gated Trading Signal Generator
==============================================================

Core insight: Our signals have IC=0.097-0.180 but fail MBO fill sim because we trade
EVERY signal equally. Small signals move 1-2 ticks but cost 1.24 ticks RT. We need to
ONLY trade when expected magnitude > costs.

Four gating approaches:
  A) Vol-Regime Gated    — only trade in HIGH vol regime (top 30%)
  B) Magnitude Model     — train LightGBM to predict |forward_return|, gate on magnitude
  C) Toxicity-Vol Gate   — trade only when VPIN high AND toxicity high (informed flow)
  D) Composite Conviction — trade only when >= 4/5 independent signals agree

Feature column index reference (from mbo_features.py get_feature_names()):
  col  0: mid
  col  3: microprice
  col 10: trade_imbalance
  col 20: pressure_imbalance
  col 37: aggressive_imbalance
  col 96: ofi_5
  col 111: vpin_20
  col 112: vpin_50
  col 113: vpin_100
  col 124: rvol_10
  col 126: rvol_20
  col 128: rvol_50
  col 140: kyle_lambda_20
  col 141: kyle_lambda_50
  col 142: price_impact
  col 143: adverse_sel
  col 144: toxicity_score

Global features (medium_snapshots_cache global_features, shape N×96) match the
first 96 features of the 340-feature MBO set (the A-group static snapshot).

Usage:
    python alpha_discovery/vol_magnitude_gated.py
    python alpha_discovery/vol_magnitude_gated.py --n-days 20 --approach volgated
    python alpha_discovery/vol_magnitude_gated.py --approach all --quick
    python alpha_discovery/vol_magnitude_gated.py --approach magnitude --n-days 30
"""

import sys
import gc
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import pearsonr

# ============================================================================
# Paths and constants
# ============================================================================

LVL3_ROOT = Path(__file__).resolve().parent.parent

# Platform-agnostic path resolution
if sys.platform == "win32":
    MBO_DIR = LVL3_ROOT / "data" / "processed" / "mbo_features_cache"
    SNAP_DIR = LVL3_ROOT / "data" / "processed" / "medium_snapshots_cache"
else:
    # Linux (Jupiter/Saturn): ~/lvl3quant/data/processed/...
    _linux_root = Path.home() / "lvl3quant"
    if _linux_root.exists():
        LVL3_ROOT = _linux_root
    MBO_DIR = LVL3_ROOT / "data" / "processed" / "mbo_features_cache"
    SNAP_DIR = LVL3_ROOT / "data" / "processed" / "medium_snapshots_cache"

OUT_DIR = LVL3_ROOT / "data" / "processed" / "signal_predictions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ES futures constants
TICK_SIZE = 0.25
TICK_VALUE = 12.50          # $12.50 per tick
COMMISSION_RT = 4.70  # HC #52: $4.70 RT (AMP)        # $3.00 round-trip
COST_TICKS = COMMISSION_RT / TICK_VALUE  # ~0.24 ticks commission alone
HALF_SPREAD_TICKS = 1.0     # typical ES half-spread at entry + exit
TOTAL_COST_TICKS = COST_TICKS + HALF_SPREAD_TICKS  # ~1.24 ticks total
BARS_PER_SEC = 10           # 100ms bars
HORIZON_BARS_10S = 100      # 10-second forward return horizon

# Feature column indices in 340-feature MBO set
COL_MID = 0
COL_MICROPRICE = 3
COL_TRADE_IMBALANCE = 10
COL_PRESSURE_IMBALANCE = 20
COL_AGGRESSIVE_IMBALANCE = 37
COL_OFI_5 = 96
COL_VPIN_20 = 111
COL_VPIN_50 = 112
COL_VPIN_100 = 113
COL_RVOL_10 = 124
COL_RVOL_20 = 126
COL_RVOL_50 = 128
COL_KYLE_LAMBDA_20 = 140
COL_KYLE_LAMBDA_50 = 141
COL_PRICE_IMPACT = 142
COL_ADVERSE_SEL = 143
COL_TOXICITY_SCORE = 144

# In global_features (96-col snapshot cache), microprice_dev = col3 - col0
# but since mid is col0 and microprice is col3, we compute: microprice - mid
SNAP_COL_MID = 0
SNAP_COL_MICROPRICE = 3
SNAP_COL_TRADE_IMBALANCE = 10
SNAP_COL_PRESSURE_IMBALANCE = 20
SNAP_COL_AGGRESSIVE_IMBALANCE = 37  # only 45+extras in A-group, index 37 valid

# ============================================================================
# Logging
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("vol_magnitude_gated")


# ============================================================================
# Data loading
# ============================================================================

def find_day_files() -> List[Tuple[str, Path, Path]]:
    """Return sorted list of (date_str, mbo_path, snap_path) for days present in both caches."""
    mbo_files = {f.stem.replace("_mbo_features", ""): f
                 for f in sorted(MBO_DIR.glob("*_mbo_features.npz"))}
    snap_files = {f.stem.replace("_snapshots", ""): f
                  for f in sorted(SNAP_DIR.glob("*_snapshots.npz"))}
    common = sorted(set(mbo_files.keys()) & set(snap_files.keys()))
    return [(d, mbo_files[d], snap_files[d]) for d in common]


def load_day(mbo_path: Path, snap_path: Path) -> Optional[Dict]:
    """Load one day's MBO features and snapshot data.

    Returns dict with keys:
        mbo_feats   (N, 340) float32 — full 340-feature set
        global_feats (N, 96) float32 — static snapshot A-group features
        mid_prices   (N,)    float32
        N            int
    Returns None if either file is missing or corrupt.
    """
    try:
        mbo_data = np.load(str(mbo_path))
        snap_data = np.load(str(snap_path))
        mbo_feats = mbo_data["mbo_features"].astype(np.float32)      # (N, 340)
        global_feats = snap_data["global_features"].astype(np.float32)  # (N, 96)
        mid_prices = snap_data["mid_prices"].astype(np.float32)         # (N,)
        N = mbo_feats.shape[0]
        if N < 1000 or global_feats.shape[0] != N or mid_prices.shape[0] != N:
            return None
        return {
            "mbo_feats": mbo_feats,
            "global_feats": global_feats,
            "mid_prices": mid_prices,
            "N": N,
        }
    except Exception as exc:
        log.warning("Failed to load %s / %s: %s", mbo_path.name, snap_path.name, exc)
        return None


def compute_forward_return(mid_prices: np.ndarray, horizon: int = HORIZON_BARS_10S) -> np.ndarray:
    """Compute (mid[t+h] - mid[t]) / mid[t] forward return. Last `horizon` bars = NaN."""
    N = len(mid_prices)
    fwd = np.full(N, np.nan, dtype=np.float64)
    fwd[: N - horizon] = (
        (mid_prices[horizon:].astype(np.float64) - mid_prices[: N - horizon].astype(np.float64))
        / np.maximum(mid_prices[: N - horizon].astype(np.float64), 1.0)
    )
    return fwd


# ============================================================================
# IC computation utilities
# ============================================================================

def pearson_ic(signal: np.ndarray, target: np.ndarray) -> float:
    """Pearson correlation between signal and target, skipping NaN/Inf."""
    mask = np.isfinite(signal) & np.isfinite(target)
    if mask.sum() < 50:
        return np.nan
    try:
        corr, _ = pearsonr(signal[mask], target[mask])
        return float(corr) if np.isfinite(corr) else np.nan
    except Exception:
        return np.nan


def ic_summary(daily_ics: List[float]) -> Dict:
    """Compute summary statistics from a list of per-day IC values."""
    arr = np.array([x for x in daily_ics if np.isfinite(x)])
    if len(arr) == 0:
        return {"n": 0, "mean_ic": 0.0, "std_ic": 0.0, "t_stat": 0.0, "pct_positive": 0.0}
    n = len(arr)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    t_stat = (mean / std * np.sqrt(n)) if std > 1e-8 else 0.0
    pct_pos = float((arr > 0).mean())
    return {
        "n": n,
        "mean_ic": round(mean, 6),
        "std_ic": round(std, 6),
        "t_stat": round(t_stat, 3),
        "pct_positive": round(pct_pos, 3),
    }


def print_ic_summary(name: str, stats: Dict) -> None:
    """Print a formatted IC summary line."""
    log.info(
        "%s | n=%d  mean_IC=%.4f  std=%.4f  t=%.2f  pct+=%d%%",
        name.ljust(20),
        stats["n"],
        stats["mean_ic"],
        stats["std_ic"],
        stats["t_stat"],
        int(stats["pct_positive"] * 100),
    )


# ============================================================================
# Rolling percentile helper (causal, no lookahead)
# ============================================================================

def rolling_percentile_rank(x: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling percentile rank of each element within its lookback window.

    Returns values in [0, 1]. First `window-1` bars are filled with 0.5.
    Optimized using stride tricks for speed.
    """
    N = len(x)
    out = np.full(N, 0.5, dtype=np.float32)
    # Vectorized approach using sorted windows via cumulative method
    # For each bar i, rank x[i] among x[i-window+1 : i+1]
    # Use a simple O(N*window/batch) approach with batch processing
    batch = 500  # process in batches for memory efficiency
    for start in range(window - 1, N, batch):
        end = min(start + batch, N)
        for i in range(start, end):
            w_start = max(0, i - window + 1)
            window_vals = x[w_start: i + 1]
            rank = float(np.sum(window_vals < x[i])) / max(len(window_vals), 1)
            out[i] = rank
    return out


def rolling_percentile_rank_fast(x: np.ndarray, window: int) -> np.ndarray:
    """Faster causal rolling percentile using only boundary updates.

    For large windows (e.g. 500) on 234000-bar days, the full loop is too slow.
    This version uses a vectorized approximation: compute rolling quantile thresholds
    from expanding windows (for warm-up) then fixed windows via stride.
    Uses numpy's percentile on fixed strides.
    """
    N = len(x)
    out = np.full(N, 0.5, dtype=np.float32)
    x_f64 = x.astype(np.float64)

    # Compute rolling quantile by sampling — good enough for regime classification
    # Use 50-sample stride to keep performance reasonable
    stride = max(1, window // 50)
    for i in range(window - 1, N, stride):
        w_start = max(0, i - window + 1)
        wnd = x_f64[w_start: i + 1]
        pct = float(np.sum(wnd < x_f64[i])) / max(len(wnd), 1)
        # Fill forward until next stride
        fill_end = min(i + stride, N)
        out[i: fill_end] = pct

    return out


# ============================================================================
# Approach A: Vol-Regime Gated Signal
# ============================================================================

def generate_volgated_signal(
    day_data: Dict,
    vol_window: int = 500,          # bars for rolling vol (500 bars = 50s)
    vol_high_pct: float = 0.70,     # top 30% = regime HIGH (rolling percentile >= 0.70)
    vol_rank_window: int = 5000,    # bars for rolling percentile rank of vol
) -> np.ndarray:
    """Approach A: Vol-Regime Gated signal.

    Logic:
        - Compute realized vol as rolling std of mid returns (window=500 bars = 50s)
        - Classify regime: HIGH = rolling vol in top 30% of vol_rank_window history
        - Directional signal: microprice_dev (microprice - mid)
        - Output: microprice_dev when vol_regime == HIGH, else 0.0

    Rationale: HIGH vol periods mean larger tick moves per bar, so signal × magnitude
    exceeds round-trip costs more often.
    """
    mbo_feats = day_data["mbo_feats"]      # (N, 340)
    mid_prices = day_data["mid_prices"]    # (N,)
    N = day_data["N"]

    # --- Step 1: microprice_dev as directional signal ---
    # microprice is col 3 of mbo_feats; mid is col 0
    # microprice_dev = microprice - mid (positive → price should go up)
    microprice = mbo_feats[:, COL_MICROPRICE]
    mid = mbo_feats[:, COL_MID]
    microprice_dev = (microprice - mid).astype(np.float64)

    # --- Step 2: Realized vol (rolling std of log returns) ---
    log_ret = np.diff(np.log(np.maximum(mid_prices.astype(np.float64), 1.0)))
    log_ret = np.concatenate([[0.0], log_ret])  # prepend 0 for alignment

    # Rolling std using cumsum trick — O(N) computation
    rvol = np.zeros(N, dtype=np.float32)
    cumsum = np.cumsum(log_ret)
    cumsum_sq = np.cumsum(log_ret ** 2)
    for i in range(vol_window, N):
        s = cumsum[i] - cumsum[i - vol_window]
        s2 = cumsum_sq[i] - cumsum_sq[i - vol_window]
        var = s2 / vol_window - (s / vol_window) ** 2
        rvol[i] = float(np.sqrt(max(var, 0.0))) * 1e4  # in bps

    # --- Step 3: Rolling percentile rank of vol (causal) ---
    # Use fast stride-based approximation for performance on 234k bars
    vol_rank = rolling_percentile_rank_fast(rvol, vol_rank_window)

    # --- Step 4: Gate signal ---
    high_vol_mask = vol_rank >= vol_high_pct  # top 30% vol regime
    signal = np.where(high_vol_mask, microprice_dev, 0.0).astype(np.float32)
    return signal


# ============================================================================
# Approach B: Magnitude Model (Walk-Forward LightGBM)
# ============================================================================

def generate_magnitude_signal_one_fold(
    X_train: np.ndarray,
    y_mag_train: np.ndarray,
    y_dir_train: np.ndarray,
    X_test: np.ndarray,
    cost_ticks: float = TOTAL_COST_TICKS,
    quick: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Train direction + magnitude LightGBM models on train, predict on test.

    Returns:
        magnitude_pred (N_test,)  — predicted |forward_return| in ticks
        combined_signal (N_test,) — direction * magnitude, 0 when magnitude <= 2*cost
    """
    try:
        import lightgbm as lgb
    except ImportError:
        log.error("lightgbm not installed — magnitude approach unavailable")
        return np.zeros(len(X_test)), np.zeros(len(X_test))

    n_est = 100 if quick else 300
    lgb_params = {
        "n_estimators": n_est,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.6,
        "min_child_samples": 200,
        "verbose": -1,
        "n_jobs": -1,
        "objective": "regression",
        "metric": "rmse",
    }

    # Mask out NaN/Inf
    tr_valid = np.isfinite(y_mag_train) & np.isfinite(y_dir_train) & np.all(np.isfinite(X_train), axis=1)
    te_valid = np.all(np.isfinite(X_test), axis=1)

    if tr_valid.sum() < 500:
        return np.zeros(len(X_test)), np.zeros(len(X_test))

    # --- Magnitude model: predict |forward_return| ---
    mag_model = lgb.LGBMRegressor(**lgb_params)
    try:
        mag_model.fit(X_train[tr_valid], y_mag_train[tr_valid])
        mag_pred = np.zeros(len(X_test), dtype=np.float32)
        if te_valid.sum() > 0:
            mag_pred[te_valid] = mag_model.predict(X_test[te_valid]).astype(np.float32)
    except Exception as exc:
        log.debug("Magnitude model fit failed: %s", exc)
        return np.zeros(len(X_test)), np.zeros(len(X_test))
    finally:
        del mag_model
        gc.collect()

    # --- Direction model: predict sign(forward_return) → {-1, +1} ---
    dir_model = lgb.LGBMRegressor(**lgb_params)
    try:
        dir_model.fit(X_train[tr_valid], y_dir_train[tr_valid])
        dir_pred = np.zeros(len(X_test), dtype=np.float32)
        if te_valid.sum() > 0:
            dir_pred[te_valid] = dir_model.predict(X_test[te_valid]).astype(np.float32)
    except Exception as exc:
        log.debug("Direction model fit failed: %s", exc)
        return mag_pred, np.zeros(len(X_test))
    finally:
        del dir_model
        gc.collect()

    # --- Combined signal: direction × magnitude, gated on magnitude ---
    # Convert forward return to ticks: ret (fractional) × price / TICK_SIZE
    # We use relative magnitude so cost comparison is tick-normalized
    cost_gate = 2.0 * cost_ticks * TICK_SIZE  # in price units (2 × 1.24 ticks × $0.25)
    combined = np.where(
        mag_pred > cost_gate,
        np.sign(dir_pred) * mag_pred,
        0.0,
    ).astype(np.float32)

    return mag_pred, combined


def run_magnitude_approach(
    days: List[Tuple[str, Path, Path]],
    min_train_days: int = 10,
    purge_days: int = 1,
    quick: bool = False,
) -> Tuple[List[float], Dict[str, np.ndarray]]:
    """Walk-forward magnitude-gated signal generation across all days.

    Uses the full 340-feature MBO feature set as input.
    Train on days 0..d-purge-1, predict on day d.
    Returns (daily_ics, date_predictions).
    """
    log.info("Approach B: Magnitude Model (walk-forward, %d days)", len(days))
    log.info("  Min train days=%d, purge=%d", min_train_days, purge_days)

    # Load all days
    loaded = []
    for date_str, mbo_path, snap_path in days:
        d = load_day(mbo_path, snap_path)
        if d is not None:
            d["date"] = date_str
            loaded.append(d)

    if len(loaded) < min_train_days + purge_days + 1:
        log.warning("Not enough days for walk-forward (%d loaded)", len(loaded))
        return [], {}

    daily_ics = []
    date_predictions: Dict[str, np.ndarray] = {}

    for test_idx in range(min_train_days + purge_days, len(loaded)):
        train_days_data = loaded[: test_idx - purge_days]
        test_day_data = loaded[test_idx]
        date_str = test_day_data["date"]

        # Build train arrays
        X_parts, y_mag_parts, y_dir_parts = [], [], []
        for td in train_days_data:
            N = td["N"]
            fwd = compute_forward_return(td["mid_prices"], HORIZON_BARS_10S)
            # Convert fractional return to approximate ticks
            fwd_ticks = fwd * td["mid_prices"].astype(np.float64) / TICK_SIZE
            valid = np.isfinite(fwd_ticks)
            if valid.sum() < 100:
                continue
            X_parts.append(td["mbo_feats"][valid])
            y_mag_parts.append(np.abs(fwd_ticks[valid]).astype(np.float32))
            y_dir_parts.append(np.sign(fwd_ticks[valid]).astype(np.float32))

        if not X_parts:
            log.debug("No train data for test day %s — skipping", date_str)
            continue

        X_train = np.vstack(X_parts)
        y_mag_train = np.concatenate(y_mag_parts)
        y_dir_train = np.concatenate(y_dir_parts)

        # Replace inf/NaN in X_train with 0
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

        # Test day
        X_test = np.nan_to_num(
            test_day_data["mbo_feats"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
        fwd_test = compute_forward_return(test_day_data["mid_prices"], HORIZON_BARS_10S)

        _, combined_signal = generate_magnitude_signal_one_fold(
            X_train, y_mag_train, y_dir_train, X_test, quick=quick
        )

        ic = pearson_ic(combined_signal.astype(np.float64), fwd_test)
        if np.isfinite(ic):
            daily_ics.append(ic)
            log.info("  %s  IC=%.4f  trades=%d", date_str, ic, int((combined_signal != 0).sum()))

        date_predictions[date_str] = combined_signal

        del X_train, X_test, X_parts, y_mag_parts, y_dir_parts
        gc.collect()

    return daily_ics, date_predictions


# ============================================================================
# Approach C: Toxicity-Volatility Gate
# ============================================================================

def generate_toxgate_signal(
    day_data: Dict,
    vpin_window: int = 5000,        # rolling window for VPIN percentile
    tox_window: int = 5000,         # rolling window for toxicity percentile
    vpin_threshold_pct: float = 0.80,   # 80th percentile gate for VPIN
    tox_threshold_pct: float = 0.70,    # 70th percentile gate for toxicity
) -> np.ndarray:
    """Approach C: Toxicity-Volatility gated signal.

    Logic:
        - VPIN (Volume-Synchronized Probability of Informed Trading) measures
          the probability that a counterparty is informationally advantaged.
        - toxicity_score is the composite adverse selection measure.
        - When BOTH are elevated, trade in the direction of: sign(pressure_imbalance + microprice_dev)
        - Rationale: high VPIN + high toxicity = informed flow = follow the smart money.
    """
    mbo_feats = day_data["mbo_feats"]   # (N, 340)
    N = day_data["N"]

    # --- Directional signal: pressure_imbalance + microprice_dev ---
    microprice = mbo_feats[:, COL_MICROPRICE].astype(np.float64)
    mid = mbo_feats[:, COL_MID].astype(np.float64)
    microprice_dev = microprice - mid

    pressure_imb = mbo_feats[:, COL_PRESSURE_IMBALANCE].astype(np.float64)
    combined_dir = pressure_imb + microprice_dev

    # --- VPIN rolling percentile rank ---
    vpin_20 = mbo_feats[:, COL_VPIN_20].astype(np.float32)
    # Replace NaN/Inf with 0 (uninformative)
    vpin_20 = np.nan_to_num(vpin_20, nan=0.0, posinf=1.0, neginf=0.0)
    vpin_rank = rolling_percentile_rank_fast(vpin_20, vpin_window)

    # --- Toxicity rolling percentile rank ---
    toxicity = mbo_feats[:, COL_TOXICITY_SCORE].astype(np.float32)
    toxicity = np.nan_to_num(toxicity, nan=0.0, posinf=1.0, neginf=0.0)
    tox_rank = rolling_percentile_rank_fast(toxicity, tox_window)

    # --- Gate: both VPIN and toxicity must be elevated ---
    gate = (vpin_rank >= vpin_threshold_pct) & (tox_rank >= tox_threshold_pct)

    # Signal: direction of informed flow when gate is active
    signal = np.where(gate, np.sign(combined_dir), 0.0).astype(np.float32)
    return signal


# ============================================================================
# Approach D: Composite Conviction Gate
# ============================================================================

def generate_conviction_signal(
    day_data: Dict,
    min_agreement: int = 4,         # require >= 4/5 signals to agree
    vol_window: int = 5000,         # rolling window for vol percentile
    vol_min_pct: float = 0.50,      # vol must be above median
) -> np.ndarray:
    """Approach D: Composite conviction gate (multi-signal agreement).

    Five independent signals:
        1. microprice_dev:       col 3 - col 0 (from MBO)
        2. pressure_imbalance:   col 20 (from MBO)
        3. ofi_5:                col 96 (from MBO)
        4. aggressive_imbalance: col 37 (from MBO)
        5. trade_imbalance:      col 10 (from MBO)

    ONLY trade when:
        - >= min_agreement (4/5) signals agree on direction (all positive or all negative)
        - AND vol is above rolling median (vol_rank >= 0.5)

    Signal strength: count_agreeing / 5 (fractional conviction)
    """
    mbo_feats = day_data["mbo_feats"]   # (N, 340)
    mid_prices = day_data["mid_prices"]
    N = day_data["N"]

    # --- Extract five signals ---
    sig1 = (mbo_feats[:, COL_MICROPRICE] - mbo_feats[:, COL_MID]).astype(np.float64)
    sig2 = mbo_feats[:, COL_PRESSURE_IMBALANCE].astype(np.float64)
    sig3 = mbo_feats[:, COL_OFI_5].astype(np.float64)
    sig4 = mbo_feats[:, COL_AGGRESSIVE_IMBALANCE].astype(np.float64)
    sig5 = mbo_feats[:, COL_TRADE_IMBALANCE].astype(np.float64)

    # --- Vol regime (rolling percentile) ---
    log_ret = np.diff(np.log(np.maximum(mid_prices.astype(np.float64), 1.0)))
    log_ret = np.concatenate([[0.0], log_ret])
    cumsum = np.cumsum(log_ret)
    cumsum_sq = np.cumsum(log_ret ** 2)
    rvol = np.zeros(N, dtype=np.float32)
    for i in range(50, N):
        s = cumsum[i] - cumsum[max(0, i - 50)]
        s2 = cumsum_sq[i] - cumsum_sq[max(0, i - 50)]
        wlen = min(i, 50)
        var = s2 / wlen - (s / wlen) ** 2
        rvol[i] = float(np.sqrt(max(var, 0.0))) * 1e4
    vol_rank = rolling_percentile_rank_fast(rvol, vol_window)
    above_median = vol_rank >= vol_min_pct

    # --- Count agreements ---
    # For each bar, sign of each signal (+1, -1, 0)
    signs = np.stack([
        np.sign(sig1), np.sign(sig2), np.sign(sig3),
        np.sign(sig4), np.sign(sig5)
    ], axis=1)  # (N, 5)

    # Count positives and negatives
    n_pos = (signs > 0).sum(axis=1).astype(np.float32)   # (N,)
    n_neg = (signs < 0).sum(axis=1).astype(np.float32)   # (N,)
    max_agree = np.maximum(n_pos, n_neg)                  # (N,)
    direction = np.where(n_pos >= n_neg, 1.0, -1.0)       # majority direction

    # Gate conditions
    high_conviction = max_agree >= min_agreement
    vol_ok = above_median

    # Signal: direction × (count_agreeing / 5) when both gates active
    strength = max_agree / 5.0
    signal = np.where(
        high_conviction & vol_ok,
        direction * strength,
        0.0,
    ).astype(np.float32)

    return signal


# ============================================================================
# Saving predictions
# ============================================================================

def save_predictions(approach: str, date_str: str, predictions: np.ndarray) -> Path:
    """Save predictions NPZ file. Returns output path."""
    fname = f"{approach}_{date_str}.npz"
    out_path = OUT_DIR / fname
    np.savez_compressed(str(out_path), predictions=predictions.astype(np.float32))
    return out_path


# ============================================================================
# Per-approach runner functions
# ============================================================================

def run_volgated(
    days: List[Tuple[str, Path, Path]],
    quick: bool = False,
) -> List[float]:
    """Run Approach A across all days. Returns daily IC list."""
    log.info("=" * 60)
    log.info("Approach A: Vol-Regime Gated Signal")
    log.info("  Vol window=500bars, high regime=top 30%%")
    log.info("=" * 60)

    daily_ics = []
    for date_str, mbo_path, snap_path in days:
        d = load_day(mbo_path, snap_path)
        if d is None:
            continue

        fwd = compute_forward_return(d["mid_prices"], HORIZON_BARS_10S)
        signal = generate_volgated_signal(d)

        ic = pearson_ic(signal.astype(np.float64), fwd)
        if np.isfinite(ic):
            daily_ics.append(ic)
            n_active = int((signal != 0).sum())
            log.info("  %s  IC=%+.4f  active_bars=%d (%.1f%%)",
                     date_str, ic, n_active, 100.0 * n_active / max(d["N"], 1))

        save_predictions("volgated", date_str, signal)
        del d
        gc.collect()

    stats = ic_summary(daily_ics)
    log.info("")
    log.info("VOLGATED SUMMARY:")
    print_ic_summary("volgated", stats)
    return daily_ics


def run_toxgate(
    days: List[Tuple[str, Path, Path]],
    quick: bool = False,
) -> List[float]:
    """Run Approach C across all days. Returns daily IC list."""
    log.info("=" * 60)
    log.info("Approach C: Toxicity-Volatility Gate")
    log.info("  VPIN >= 80th pct AND toxicity >= 70th pct")
    log.info("=" * 60)

    daily_ics = []
    for date_str, mbo_path, snap_path in days:
        d = load_day(mbo_path, snap_path)
        if d is None:
            continue

        fwd = compute_forward_return(d["mid_prices"], HORIZON_BARS_10S)
        signal = generate_toxgate_signal(d)

        ic = pearson_ic(signal.astype(np.float64), fwd)
        if np.isfinite(ic):
            daily_ics.append(ic)
            n_active = int((signal != 0).sum())
            log.info("  %s  IC=%+.4f  active_bars=%d (%.1f%%)",
                     date_str, ic, n_active, 100.0 * n_active / max(d["N"], 1))

        save_predictions("toxgate", date_str, signal)
        del d
        gc.collect()

    stats = ic_summary(daily_ics)
    log.info("")
    log.info("TOXGATE SUMMARY:")
    print_ic_summary("toxgate", stats)
    return daily_ics


def run_conviction(
    days: List[Tuple[str, Path, Path]],
    quick: bool = False,
) -> List[float]:
    """Run Approach D across all days. Returns daily IC list."""
    log.info("=" * 60)
    log.info("Approach D: Composite Conviction Gate")
    log.info("  >= 4/5 signals agree AND vol > median")
    log.info("=" * 60)

    daily_ics = []
    for date_str, mbo_path, snap_path in days:
        d = load_day(mbo_path, snap_path)
        if d is None:
            continue

        fwd = compute_forward_return(d["mid_prices"], HORIZON_BARS_10S)
        signal = generate_conviction_signal(d)

        ic = pearson_ic(signal.astype(np.float64), fwd)
        if np.isfinite(ic):
            daily_ics.append(ic)
            n_active = int((signal != 0).sum())
            log.info("  %s  IC=%+.4f  active_bars=%d (%.1f%%)",
                     date_str, ic, n_active, 100.0 * n_active / max(d["N"], 1))

        save_predictions("conviction", date_str, signal)
        del d
        gc.collect()

    stats = ic_summary(daily_ics)
    log.info("")
    log.info("CONVICTION SUMMARY:")
    print_ic_summary("conviction", stats)
    return daily_ics


# ============================================================================
# Main entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Volatility-gated and magnitude-gated trading signal generator"
    )
    parser.add_argument(
        "--n-days",
        type=int,
        default=0,
        help="Number of days to process (0 = all available)",
    )
    parser.add_argument(
        "--approach",
        choices=["all", "volgated", "magnitude", "toxgate", "conviction"],
        default="all",
        help="Which signal approach to run (default: all)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: fewer LightGBM estimators, faster but noisier",
    )
    args = parser.parse_args()

    log.info("Vol-Magnitude Gated Signal Generator")
    log.info("  MBO cache:  %s", MBO_DIR)
    log.info("  Snap cache: %s", SNAP_DIR)
    log.info("  Output dir: %s", OUT_DIR)
    log.info("  Approach:   %s", args.approach)
    log.info("  Quick mode: %s", args.quick)
    log.info("")

    # Discover available data files
    all_days = find_day_files()
    if not all_days:
        log.error("No matching day files found in %s / %s", MBO_DIR, SNAP_DIR)
        sys.exit(1)

    if args.n_days > 0:
        all_days = all_days[: args.n_days]

    log.info("Found %d days: %s ... %s", len(all_days), all_days[0][0], all_days[-1][0])
    log.info("Output: %s", OUT_DIR)
    log.info("")

    results = {}

    # -------------------------------------------------------------------------
    # A: Vol-Regime Gated
    # -------------------------------------------------------------------------
    if args.approach in ("all", "volgated"):
        ics_a = run_volgated(all_days, quick=args.quick)
        results["volgated"] = ic_summary(ics_a)

    # -------------------------------------------------------------------------
    # B: Magnitude Model (walk-forward LightGBM)
    # -------------------------------------------------------------------------
    if args.approach in ("all", "magnitude"):
        log.info("=" * 60)
        log.info("Approach B: Magnitude Model (Walk-Forward LightGBM)")
        log.info("  Train: min 10 days, 1-day purge gap")
        log.info("  Gate: magnitude_pred > 2 × %.2f ticks", TOTAL_COST_TICKS)
        log.info("=" * 60)

        min_train = 5 if args.quick else 10
        ics_b, preds_b = run_magnitude_approach(
            all_days,
            min_train_days=min_train,
            purge_days=1,
            quick=args.quick,
        )

        # Save magnitude predictions
        for date_str, sig in preds_b.items():
            save_predictions("magnitude", date_str, sig)

        stats_b = ic_summary(ics_b)
        log.info("")
        log.info("MAGNITUDE MODEL SUMMARY:")
        print_ic_summary("magnitude", stats_b)
        results["magnitude"] = stats_b

    # -------------------------------------------------------------------------
    # C: Toxicity-Vol Gate
    # -------------------------------------------------------------------------
    if args.approach in ("all", "toxgate"):
        ics_c = run_toxgate(all_days, quick=args.quick)
        results["toxgate"] = ic_summary(ics_c)

    # -------------------------------------------------------------------------
    # D: Composite Conviction Gate
    # -------------------------------------------------------------------------
    if args.approach in ("all", "conviction"):
        ics_d = run_conviction(all_days, quick=args.quick)
        results["conviction"] = ic_summary(ics_d)

    # -------------------------------------------------------------------------
    # Final summary table
    # -------------------------------------------------------------------------
    log.info("")
    log.info("=" * 70)
    log.info("FINAL SUMMARY — IC vs 10s forward return (Pearson)")
    log.info("Cost threshold: %.2f ticks ($%.2f RT, ES)", TOTAL_COST_TICKS, COMMISSION_RT)
    log.info("=" * 70)
    log.info("%-20s  %6s  %8s  %7s  %6s  %8s", "Approach", "n_days", "mean_IC", "std_IC", "t-stat", "pct_pos")
    log.info("-" * 70)
    for name, s in results.items():
        if s["n"] == 0:
            continue
        log.info(
            "%-20s  %6d  %+8.4f  %7.4f  %6.2f  %7.1f%%",
            name,
            s["n"],
            s["mean_ic"],
            s["std_ic"],
            s["t_stat"],
            s["pct_positive"] * 100,
        )
    log.info("=" * 70)
    log.info("")
    log.info("Predictions saved to: %s", OUT_DIR)
    log.info("Next step: pipe predictions through production/mbo_fill_sim.py")
    log.info("")
    log.info("Expected improvement over baseline (all-signal approach):")
    log.info("  Baseline IC ~0.09-0.18 but flat/negative PnL after costs")
    log.info("  Vol-gated: fewer trades but each in higher-vol window = more ticks/trade")
    log.info("  Magnitude-gated: only trade when LightGBM predicts > 2×cost")
    log.info("  Toxicity-gated: follow informed flow during HFT-active moments")
    log.info("  Conviction-gated: only trade when 4/5 independent signals align")


if __name__ == "__main__":
    main()
