"""
Novel Targets MBO Sweep — Walk-Forward LightGBM with 4 Novel Training Objectives

Trains walk-forward LightGBM models for 4 novel targets and saves per-day
predictions as NPZ files to data/processed/signal_predictions/.

Targets:
  1. risk_adjusted_return  — forward_return / realized_vol (vol-normalized)
  2. time_to_move          — I(|max_move_30s| > 3 ticks) * direction_sign
  3. direct_pnl            — simulated limit-order P&L in ticks
  4. optimal_action_value  — max(buy_pnl, sell_pnl, 0) * sign(best action)

Walk-forward: min 10 days training, 1-day purge gap, predict day N+2.

Output format: novel_{target_name}_YYYY-MM-DD.npz with key 'predictions' (N,) float32

Usage:
    python alpha_discovery/novel_targets_mbo_sweep.py
    python alpha_discovery/novel_targets_mbo_sweep.py --n-days 30 --quick
    python alpha_discovery/novel_targets_mbo_sweep.py --horizon 100 --n-days 50
"""

import gc
import sys
import time
import logging
import argparse
import platform
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Dict

import numpy as np
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
LVL3_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(LVL3_ROOT))

if platform.system() == "Windows":
    DEFAULT_FEATURE_CACHE = str(LVL3_ROOT / "data" / "processed" / "mbo_features_cache")
else:
    DEFAULT_FEATURE_CACHE = str(Path.home() / "lvl3quant" / "data" / "processed" / "mbo_features_cache")

SIGNAL_PRED_DIR = LVL3_ROOT / "data" / "processed" / "signal_predictions"
SIGNAL_PRED_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = LVL3_ROOT / "alpha_discovery" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# ES Futures constants
# ---------------------------------------------------------------------------
TICK_SIZE = 0.25          # 1 tick = $12.50
TICK_VALUE = 12.50
COMMISSION_RT = 4.70  # HC #52: $4.70 RT (AMP)
COMMISSION_TICKS = COMMISSION_RT / TICK_VALUE   # 0.24 ticks RT
HALF_SPREAD_TICKS = 0.5                         # ES spread = 1 tick, half = 0.5t
LIMIT_ENTRY_EDGE = 0.5                          # earn half spread on limit entry
NET_LIMIT_COST = COMMISSION_TICKS               # 0.24t (limit in + limit out)
BARS_PER_SEC = 10                               # 100ms bars
COL_MID = 0                                     # mid price is column 0

# Feature columns to exclude (leakage risk):
# 0=mid, 1=spread, 3=microprice, 8=best_bid, 9=best_ask,
# 26=hour_norm, 27=minute_norm, 28=time_since_rth, 29=time_to_close
EXCLUDED_COLS = [0, 1, 3, 8, 9, 26, 27, 28, 29]

# Walk-forward settings
MIN_TRAIN_DAYS = 10
PURGE_GAP = 1           # 1-day purge gap between train and test
MAX_TRAIN_ROWS = 500_000

# LightGBM params (as specified)
LGBM_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=1000,
    reg_alpha=0.1,
    reg_lambda=1.0,
    verbose=-1,
    n_jobs=-1,
    device="cpu",
    force_row_wise=True,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
_log_file = RESULTS_DIR / f"novel_targets_mbo_sweep_{_ts}.log"
_root = logging.getLogger()
_root.setLevel(logging.INFO)
for _h in list(_root.handlers):
    _root.removeHandler(_h)
_fmt = logging.Formatter("%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S")
_fh = logging.FileHandler(str(_log_file), mode="w")
_fh.setFormatter(_fmt)
_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(_fmt)
_root.addHandler(_fh)
_root.addHandler(_sh)
log = logging.getLogger("novel_mbo_sweep")


# ===========================================================================
# Data Loading
# ===========================================================================

def load_mbo_cache(cache_dir: str, n_days: Optional[int] = None) -> Tuple[
    np.ndarray, np.ndarray, List[str], np.ndarray
]:
    """
    Load all NPZ files from cache_dir sorted by date.

    Returns
    -------
    features     : (N_total, 340) float32 — all days concatenated
    mid_prices   : (N_total,) float32 — column 0 extracted
    dates        : list of str 'YYYY-MM-DD'
    day_boundaries : (n_days+1,) int — start/end index of each day
    """
    cache_path = Path(cache_dir)
    files = sorted(cache_path.glob("*_mbo_features.npz"))

    if not files:
        raise FileNotFoundError(f"No *_mbo_features.npz files found in {cache_dir}")

    if n_days is not None:
        files = files[:n_days]

    log.info(f"Loading {len(files)} days from {cache_dir}")

    all_features: List[np.ndarray] = []
    dates: List[str] = []
    boundaries: List[int] = [0]

    for f in files:
        # Filename: YYYY-MM-DD_mbo_features.npz
        date_str = f.stem.split("_mbo_features")[0]
        try:
            data = np.load(str(f))
            arr = data["mbo_features"].astype(np.float32)
        except Exception as exc:
            log.warning(f"Skipping {f.name}: {exc}")
            continue

        if arr.ndim != 2 or arr.shape[1] != 340:
            log.warning(f"Skipping {f.name}: unexpected shape {arr.shape}")
            continue

        all_features.append(arr)
        dates.append(date_str)
        boundaries.append(boundaries[-1] + arr.shape[0])

    if not all_features:
        raise RuntimeError("No valid feature files loaded.")

    features = np.concatenate(all_features, axis=0)
    mid_prices = features[:, COL_MID].copy()
    day_boundaries = np.array(boundaries, dtype=np.int64)

    log.info(
        f"Loaded {len(dates)} days, {len(features):,} bars total "
        f"(avg {len(features)//len(dates):,} bars/day)"
    )
    return features, mid_prices, dates, day_boundaries


def build_feature_matrix(features: np.ndarray) -> np.ndarray:
    """Return feature matrix with leakage columns removed."""
    n_cols = features.shape[1]
    keep_cols = [c for c in range(n_cols) if c not in EXCLUDED_COLS]
    return features[:, keep_cols]


# ===========================================================================
# Target Computation
# ===========================================================================

def compute_forward_return_ticks(
    mid_prices: np.ndarray,
    horizon_bars: int,
    day_boundaries: np.ndarray,
) -> np.ndarray:
    """Simple forward return in ticks from current bar."""
    N = len(mid_prices)
    n_days = len(day_boundaries) - 1
    ret = np.full(N, np.nan, dtype=np.float32)
    for d in range(n_days):
        s, e = int(day_boundaries[d]), int(day_boundaries[d + 1])
        day_len = e - s
        if day_len <= horizon_bars:
            continue
        valid_len = day_len - horizon_bars
        future_prices = mid_prices[s + horizon_bars : e]
        current_prices = mid_prices[s : s + valid_len]
        ret[s : s + valid_len] = (future_prices - current_prices) / TICK_SIZE
    return ret


def compute_realized_vol(
    mid_prices: np.ndarray,
    window_bars: int,
    day_boundaries: np.ndarray,
) -> np.ndarray:
    """Rolling std of 1-bar returns (in ticks) with given window."""
    N = len(mid_prices)
    n_days = len(day_boundaries) - 1
    vol = np.full(N, np.nan, dtype=np.float32)
    for d in range(n_days):
        s, e = int(day_boundaries[d]), int(day_boundaries[d + 1])
        day_prices = mid_prices[s:e]
        if len(day_prices) < window_bars + 1:
            continue
        rets = np.diff(day_prices) / TICK_SIZE  # length day_len-1
        # Rolling std using stride tricks for speed
        for i in range(window_bars, len(day_prices)):
            window = rets[i - window_bars : i]
            vol[s + i] = float(np.std(window))
    return vol


def target_risk_adjusted_return(
    mid_prices: np.ndarray,
    horizon_bars: int,
    day_boundaries: np.ndarray,
    vol_window: int = 50,
) -> np.ndarray:
    """
    Target 1: forward_return / realized_vol

    Normalizes returns by recent realized volatility. Model learns
    to pick directional trades with good risk/reward ratio.

    vol_window = 50 bars (5 seconds at 100ms resolution)
    """
    ret = compute_forward_return_ticks(mid_prices, horizon_bars, day_boundaries)
    vol = compute_realized_vol(mid_prices, vol_window, day_boundaries)

    # Safe division — floor vol at 0.1 ticks to avoid division by near-zero
    vol_safe = np.where(np.isfinite(vol) & (vol > 0.1), vol, np.nan)
    risk_adj = ret / vol_safe

    # Clip extreme values (e.g. ±10 Sharpe ratio)
    risk_adj = np.clip(risk_adj, -10.0, 10.0)
    return risk_adj.astype(np.float32)


def target_time_to_move(
    mid_prices: np.ndarray,
    horizon_bars: int,
    day_boundaries: np.ndarray,
    threshold_ticks: float = 3.0,
    ttm_horizon_bars: int = 300,  # 30s at 100ms
) -> np.ndarray:
    """
    Target 2: I(|max_price_move_in_30s| > 3 ticks) * direction_sign

    Binary component: does a big move happen within 30 seconds?
    Direction component: sign of the forward return at horizon_bars.

    Combined: +1 if big up move, -1 if big down move, 0 if no big move.
    This teaches the model to find explosive directional moments.
    """
    N = len(mid_prices)
    n_days = len(day_boundaries) - 1
    target = np.full(N, np.nan, dtype=np.float32)

    for d in range(n_days):
        s, e = int(day_boundaries[d]), int(day_boundaries[d + 1])
        day_len = e - s
        day_prices = mid_prices[s:e]

        # We need room for both horizons
        safe_len = day_len - max(ttm_horizon_bars, horizon_bars)
        if safe_len <= 0:
            continue

        for i in range(safe_len):
            p0 = day_prices[i]
            # Check for big move within ttm_horizon_bars (30s)
            future_window = day_prices[i : i + ttm_horizon_bars + 1]
            max_move = float(np.max(np.abs(future_window - p0))) / TICK_SIZE
            if max_move < threshold_ticks:
                target[s + i] = 0.0
            else:
                # Direction from forward return at shorter horizon_bars
                future_price = day_prices[i + min(horizon_bars, ttm_horizon_bars)]
                direction = np.sign(future_price - p0)
                target[s + i] = float(direction)

    return target


def target_direct_pnl(
    mid_prices: np.ndarray,
    horizon_bars: int,
    day_boundaries: np.ndarray,
) -> np.ndarray:
    """
    Target 3: Simulated limit-order P&L in ticks.

    Assumes:
    - Entry: limit order at mid - 0.5*tick (buy) or mid + 0.5*tick (sell)
    - Fill: immediate (optimistic for training signal quality)
    - Hold: horizon_bars
    - Exit: limit order at mid (earn half spread back on exit too)
    - Costs: commission only = 0.24 ticks RT

    For each bar, computes net P&L if we enter long vs short,
    then returns the best directional net P&L (signed).

    Limit entry edge: earn 0.5 ticks on entry (passive fill)
    Limit exit edge: earn 0.5 ticks on exit (passive fill)
    Total edge = 1.0 tick - 0.24 commission = 0.76 ticks net edge bonus
    """
    ret = compute_forward_return_ticks(mid_prices, horizon_bars, day_boundaries)
    # Limit entry earns 0.5t, limit exit earns 0.5t → total limit rebate = 1.0t
    # But we assume exit at mid (not limit), so only entry edge = 0.5t
    # Net: ret + 0.5 (entry edge) - 0.24 (commission) = ret + 0.26 for long
    # Net: -ret + 0.5 (entry edge) - 0.24 (commission) = -ret + 0.26 for short
    entry_edge = LIMIT_ENTRY_EDGE - COMMISSION_TICKS  # 0.5 - 0.24 = 0.26 ticks

    long_pnl = ret + entry_edge    # P&L if we go long
    short_pnl = -ret + entry_edge  # P&L if we go short

    # Return the signed best-direction P&L
    target = np.full_like(ret, np.nan)
    valid = np.isfinite(ret)

    long_better = valid & (long_pnl >= short_pnl)
    short_better = valid & (short_pnl > long_pnl)

    target[long_better] = long_pnl[long_better]
    target[short_better] = -short_pnl[short_better]  # negative = short

    return target.astype(np.float32)


def target_optimal_action_value(
    mid_prices: np.ndarray,
    horizon_bars: int,
    day_boundaries: np.ndarray,
) -> np.ndarray:
    """
    Target 4: max(buy_pnl, sell_pnl, 0) * sign(best_action)

    For each bar:
      buy_pnl  = forward_return + LIMIT_ENTRY_EDGE - COMMISSION_TICKS
      sell_pnl = -forward_return + LIMIT_ENTRY_EDGE - COMMISSION_TICKS
      nothing_value = 0

    target = sign(best_action) * value_of_best_action

    Teaches the model to output SIGNED expected net profit
    of the optimal action, including the option to do nothing.
    Bars where neither buy nor sell is profitable yield target=0.
    """
    ret = compute_forward_return_ticks(mid_prices, horizon_bars, day_boundaries)
    net_edge = LIMIT_ENTRY_EDGE - COMMISSION_TICKS  # 0.26 ticks

    buy_value = ret + net_edge
    sell_value = -ret + net_edge

    target = np.zeros_like(ret)
    valid = np.isfinite(ret)

    # Best action selection
    buy_best = valid & (buy_value > sell_value) & (buy_value > 0)
    sell_best = valid & (sell_value >= buy_value) & (sell_value > 0)

    target[buy_best] = buy_value[buy_best]
    target[sell_best] = -sell_value[sell_best]  # negative sign = short signal

    return target.astype(np.float32)


# ===========================================================================
# Walk-Forward Training
# ===========================================================================

def build_lgbm_params(quick: bool = False) -> dict:
    """Return LightGBM params, optionally reduced for quick testing."""
    params = dict(LGBM_PARAMS)
    if quick:
        params["n_estimators"] = 100
        params["min_child_samples"] = 500
    return params


def walk_forward_train(
    X: np.ndarray,
    y: np.ndarray,
    day_boundaries: np.ndarray,
    dates: List[str],
    target_name: str,
    quick: bool = False,
    progress_every: int = 5,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Walk-forward LightGBM training.

    Train on days 0..N-1 (min MIN_TRAIN_DAYS), skip PURGE_GAP days,
    predict on day N + PURGE_GAP.

    Uses native lgb.train() API to avoid sklearn compatibility issues
    (LightGBM 4.x + sklearn 1.8+ dropped force_all_finite).

    Returns
    -------
    predictions  : (N_total,) float32 — NaN where not predicted
    day_ic       : dict[date_str -> IC] for predicted days
    """
    import lightgbm as lgb

    n_days = len(day_boundaries) - 1
    N = len(y)
    predictions = np.full(N, np.nan, dtype=np.float32)
    day_ic: Dict[str, float] = {}
    params = build_lgbm_params(quick=quick)

    # Convert sklearn-style params to native lgb.train() params
    n_estimators = params.pop("n_estimators", 300)
    # n_jobs -> num_threads in native API
    n_jobs = params.pop("n_jobs", -1)
    # device stays
    # force_row_wise stays
    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": params.get("learning_rate", 0.05),
        "max_depth": params.get("max_depth", 6),
        "num_leaves": params.get("num_leaves", 31),
        "subsample": params.get("subsample", 0.8),
        "feature_fraction": params.get("colsample_bytree", 0.8),
        "min_child_samples": params.get("min_child_samples", 1000),
        "reg_alpha": params.get("reg_alpha", 0.1),
        "reg_lambda": params.get("reg_lambda", 1.0),
        "verbose": -1,
        "num_threads": n_jobs if n_jobs > 0 else 0,
        "device": params.get("device", "cpu"),
        "force_row_wise": params.get("force_row_wise", True),
    }

    log.info(
        f"[{target_name}] Walk-forward: {n_days} days, min_train={MIN_TRAIN_DAYS}, "
        f"purge={PURGE_GAP}, n_estimators={n_estimators}, lr={lgb_params['learning_rate']}"
    )

    n_folds = 0
    fold_ics = []
    t0 = time.time()

    # test_day is the day we predict
    for test_day in range(MIN_TRAIN_DAYS + PURGE_GAP, n_days):
        # Train window: days [0, test_day - PURGE_GAP - 1]
        train_end_day = test_day - PURGE_GAP - 1
        train_start_day = 0  # full history (no sliding window)

        ts = int(day_boundaries[train_start_day])
        te = int(day_boundaries[train_end_day + 1])
        vs = int(day_boundaries[test_day])
        ve = int(day_boundaries[test_day + 1])

        # Extract train targets and filter valid rows
        y_tr_full = y[ts:te]
        y_te_full = y[vs:ve]

        tr_valid = np.isfinite(y_tr_full)
        te_valid = np.isfinite(y_te_full)

        if tr_valid.sum() < 1000:
            continue
        if te_valid.sum() < 50:
            continue

        # Build training data
        tr_idx = np.where(tr_valid)[0]
        if len(tr_idx) > MAX_TRAIN_ROWS:
            rng = np.random.default_rng(seed=test_day)
            tr_idx = np.sort(rng.choice(tr_idx, MAX_TRAIN_ROWS, replace=False))

        X_tr = X[ts:te][tr_idx].astype(np.float32)
        y_tr = y_tr_full[tr_idx].astype(np.float32)
        X_te = X[vs:ve][te_valid].astype(np.float32)
        y_te = y_te_full[te_valid].astype(np.float32)

        # Train/val split for early stopping (80/20)
        split = int(len(X_tr) * 0.8)
        use_early_stopping = split >= 500 and split < len(X_tr)

        try:
            dtrain = lgb.Dataset(X_tr[:split] if use_early_stopping else X_tr,
                                 label=y_tr[:split] if use_early_stopping else y_tr,
                                 free_raw_data=True)
            callbacks = [lgb.log_evaluation(period=-1)]  # suppress output
            valid_sets = [dtrain]
            valid_names = ["train"]

            if use_early_stopping:
                dval = lgb.Dataset(X_tr[split:], label=y_tr[split:],
                                   reference=dtrain, free_raw_data=True)
                valid_sets = [dtrain, dval]
                valid_names = ["train", "val"]
                callbacks.append(lgb.early_stopping(stopping_rounds=30, verbose=False))

            booster = lgb.train(
                lgb_params,
                dtrain,
                num_boost_round=n_estimators,
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=callbacks,
            )
            p = booster.predict(X_te, num_iteration=booster.best_iteration).astype(np.float32)
        except Exception as exc:
            log.warning(f"[{target_name}] Day {test_day} fold failed: {exc}")
            del X_tr, y_tr
            gc.collect()
            continue

        # Write predictions back to global array at te_valid positions
        te_positions = np.where(te_valid)[0]
        n_write = min(len(te_positions), len(p))
        global_positions = vs + te_positions[:n_write]
        predictions[global_positions] = p[:n_write]

        # Day-level IC (Pearson correlation with actual target)
        date_str = dates[test_day]
        if len(p) > 10:
            try:
                ic_val, _ = pearsonr(p, y_te)
                if np.isfinite(ic_val):
                    fold_ics.append(float(ic_val))
                    day_ic[date_str] = float(ic_val)
            except Exception:
                day_ic[date_str] = float("nan")
        else:
            day_ic[date_str] = float("nan")

        n_folds += 1
        if n_folds % progress_every == 0:
            mean_ic = float(np.nanmean(fold_ics)) if fold_ics else 0.0
            elapsed = time.time() - t0
            log.info(
                f"  [{target_name}] Day {test_day}/{n_days-1} "
                f"(fold {n_folds}): mean_IC={mean_ic:.4f} "
                f"[{elapsed:.0f}s elapsed]"
            )

        del X_tr, y_tr, booster, dtrain
        if use_early_stopping:
            del dval
        gc.collect()

    # Summary
    mean_ic = float(np.nanmean(fold_ics)) if fold_ics else 0.0
    elapsed = time.time() - t0
    log.info(
        f"[{target_name}] COMPLETE: {n_folds} folds, "
        f"mean_IC={mean_ic:.4f}, elapsed={elapsed:.0f}s"
    )
    return predictions, day_ic


# ===========================================================================
# Save Predictions
# ===========================================================================

def save_per_day_predictions(
    predictions: np.ndarray,
    day_boundaries: np.ndarray,
    dates: List[str],
    target_name: str,
    output_dir: Path,
) -> List[Path]:
    """
    Save per-day prediction arrays as NPZ files.

    Output filename: novel_{target_name}_YYYY-MM-DD.npz
    NPZ key: 'predictions' — (N_day,) float32

    Only saves days that have at least some non-NaN predictions.
    Returns list of saved file paths.
    """
    saved = []
    n_days = len(day_boundaries) - 1

    for d in range(n_days):
        s = int(day_boundaries[d])
        e = int(day_boundaries[d + 1])
        date_str = dates[d]
        day_preds = predictions[s:e].astype(np.float32)

        n_valid = int(np.sum(np.isfinite(day_preds)))
        if n_valid == 0:
            continue

        fname = output_dir / f"novel_{target_name}_{date_str}.npz"
        np.savez_compressed(str(fname), predictions=day_preds)
        saved.append(fname)

    log.info(
        f"[{target_name}] Saved {len(saved)} prediction files to {output_dir}"
    )
    return saved


# ===========================================================================
# IC Evaluation
# ===========================================================================

def compute_ic_vs_forward_return(
    predictions: np.ndarray,
    mid_prices: np.ndarray,
    day_boundaries: np.ndarray,
    dates: List[str],
    horizon_bars: int = 100,  # default 10s = 100 bars
) -> Dict[str, float]:
    """
    Compute per-day Pearson IC between predictions and actual 10s forward return.

    This is the primary post-hoc quality check: regardless of training target,
    do the predictions correlate with actual future price moves?
    """
    fwd_ret = compute_forward_return_ticks(mid_prices, horizon_bars, day_boundaries)
    n_days = len(day_boundaries) - 1
    day_ic: Dict[str, float] = {}

    for d in range(n_days):
        s = int(day_boundaries[d])
        e = int(day_boundaries[d + 1])
        date_str = dates[d]

        p = predictions[s:e]
        r = fwd_ret[s:e]
        valid = np.isfinite(p) & np.isfinite(r)

        if valid.sum() < 50:
            continue

        try:
            ic_val, _ = pearsonr(p[valid], r[valid])
            day_ic[date_str] = float(ic_val) if np.isfinite(ic_val) else float("nan")
        except Exception:
            day_ic[date_str] = float("nan")

    return day_ic


def print_ic_summary(target_name: str, day_ic: Dict[str, float]) -> None:
    """Print a compact per-day and aggregate IC summary."""
    values = [v for v in day_ic.values() if np.isfinite(v)]
    if not values:
        log.info(f"[{target_name}] No valid IC values to summarize.")
        return

    mean_ic = float(np.mean(values))
    std_ic = float(np.std(values))
    n_pos = sum(1 for v in values if v > 0)
    pct_pos = 100 * n_pos / len(values)

    log.info(
        f"\n{'='*60}\n"
        f"[{target_name}] IC vs 10s forward return summary:\n"
        f"  Days evaluated : {len(values)}\n"
        f"  Mean IC        : {mean_ic:+.4f}\n"
        f"  Std IC         : {std_ic:.4f}\n"
        f"  Positive days  : {n_pos}/{len(values)} ({pct_pos:.1f}%)\n"
        f"  Min/Max        : {min(values):+.4f} / {max(values):+.4f}\n"
        f"{'='*60}"
    )

    # Per-day table (last 20 days)
    sorted_dates = sorted(day_ic.keys())
    recent = sorted_dates[-20:]
    log.info(f"[{target_name}] Per-day IC (most recent {len(recent)}):")
    for date_str in recent:
        ic = day_ic.get(date_str, float("nan"))
        bar = "#" * int(max(0, ic * 40))
        log.info(f"  {date_str}  {ic:+.4f}  {bar}")


# ===========================================================================
# Main Pipeline
# ===========================================================================

TARGET_REGISTRY = {
    "risk_adjusted_return": target_risk_adjusted_return,
    "time_to_move": target_time_to_move,
    "direct_pnl": target_direct_pnl,
    "optimal_action_value": target_optimal_action_value,
}


def run_sweep(
    cache_dir: str,
    n_days: Optional[int],
    horizon_bars: int,
    quick: bool,
    targets: Optional[List[str]] = None,
) -> None:
    """
    Main entry point: load data, compute targets, train walk-forward models,
    save predictions, and print IC summary.
    """
    t_start = time.time()

    # 1. Load feature cache
    features_full, mid_prices, dates, day_boundaries = load_mbo_cache(
        cache_dir, n_days=n_days
    )
    n_days_loaded = len(dates)
    log.info(f"Feature shape: {features_full.shape}")

    # 2. Build feature matrix (remove leakage columns)
    X = build_feature_matrix(features_full)
    log.info(
        f"Feature matrix after exclusions: {X.shape} "
        f"(removed {features_full.shape[1] - X.shape[1]} leakage columns)"
    )
    del features_full
    gc.collect()

    # 3. Select targets
    if targets is None:
        targets_to_run = list(TARGET_REGISTRY.keys())
    else:
        targets_to_run = [t for t in targets if t in TARGET_REGISTRY]

    if not targets_to_run:
        raise ValueError(f"No valid targets in {targets}. Available: {list(TARGET_REGISTRY.keys())}")

    log.info(f"Targets to run: {targets_to_run}")
    log.info(f"Horizon: {horizon_bars} bars = {horizon_bars / BARS_PER_SEC:.1f}s")
    log.info(f"Quick mode: {quick}")

    all_ic_summaries: Dict[str, Dict[str, float]] = {}

    for target_name in targets_to_run:
        log.info(f"\n{'='*60}")
        log.info(f"Running target: {target_name}")
        log.info(f"{'='*60}")
        t_target = time.time()

        # 4. Compute training target
        target_fn = TARGET_REGISTRY[target_name]
        log.info(f"[{target_name}] Computing target labels...")

        # All targets accept (mid_prices, horizon_bars, day_boundaries)
        # time_to_move has an extra ttm_horizon_bars arg but defaults are fine
        y = target_fn(mid_prices, horizon_bars, day_boundaries)

        n_valid = int(np.sum(np.isfinite(y)))
        n_nonzero = int(np.sum(y != 0) if np.sum(np.isfinite(y)) > 0 else 0)
        log.info(
            f"[{target_name}] Target: {n_valid:,} valid bars, "
            f"{n_nonzero:,} non-zero ({100*n_nonzero/max(n_valid,1):.1f}%)"
        )

        # 5. Walk-forward training
        predictions, train_day_ic = walk_forward_train(
            X=X,
            y=y,
            day_boundaries=day_boundaries,
            dates=dates,
            target_name=target_name,
            quick=quick,
            progress_every=5,
        )

        n_pred = int(np.sum(np.isfinite(predictions)))
        log.info(f"[{target_name}] Generated {n_pred:,} predictions")

        # 6. Save per-day predictions
        save_per_day_predictions(
            predictions=predictions,
            day_boundaries=day_boundaries,
            dates=dates,
            target_name=target_name,
            output_dir=SIGNAL_PRED_DIR,
        )

        # 7. Compute IC vs actual 10s forward return (post-hoc quality check)
        log.info(f"[{target_name}] Computing IC vs 10s forward return...")
        ic_10s_bars = 100  # 10s = 100 bars at 100ms
        day_ic_vs_ret = compute_ic_vs_forward_return(
            predictions=predictions,
            mid_prices=mid_prices,
            day_boundaries=day_boundaries,
            dates=dates,
            horizon_bars=ic_10s_bars,
        )

        all_ic_summaries[target_name] = day_ic_vs_ret
        print_ic_summary(target_name, day_ic_vs_ret)

        elapsed = time.time() - t_target
        log.info(f"[{target_name}] Target completed in {elapsed:.0f}s")

        del y, predictions
        gc.collect()

    # 8. Final aggregate summary
    log.info(f"\n{'='*70}")
    log.info("AGGREGATE IC SUMMARY (Pearson IC vs 10s forward return)")
    log.info(f"{'='*70}")
    for target_name, day_ic in all_ic_summaries.items():
        values = [v for v in day_ic.values() if np.isfinite(v)]
        if not values:
            log.info(f"  {target_name:<30} — no valid IC")
            continue
        mean_ic = float(np.mean(values))
        n_pos = sum(1 for v in values if v > 0)
        pct_pos = 100 * n_pos / len(values)
        log.info(
            f"  {target_name:<30}  IC={mean_ic:+.4f}  "
            f"pos={n_pos}/{len(values)} ({pct_pos:.0f}%)"
        )

    total_elapsed = time.time() - t_start
    log.info(f"\nTotal sweep completed in {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    log.info(f"Predictions saved to: {SIGNAL_PRED_DIR}")
    log.info(f"Log saved to: {_log_file}")


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Novel Targets MBO Sweep — Walk-Forward LightGBM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n-days", type=int, default=None,
        help="Number of days to load (default: all available)",
    )
    parser.add_argument(
        "--horizon", type=int, default=100,
        help="Forward return horizon in bars (100 = 10s at 100ms resolution)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: fewer trees, faster run for testing",
    )
    parser.add_argument(
        "--targets", nargs="+", default=None,
        choices=list(TARGET_REGISTRY.keys()),
        help="Which targets to run (default: all 4)",
    )
    parser.add_argument(
        "--cache-dir", type=str, default=DEFAULT_FEATURE_CACHE,
        help="Path to mbo_features_cache directory",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(SIGNAL_PRED_DIR),
        help="Output directory for prediction NPZ files",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Override output dir if specified
    if args.output_dir != str(SIGNAL_PRED_DIR):
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        globals()["SIGNAL_PRED_DIR"] = out_dir

    log.info("=" * 70)
    log.info("Novel Targets MBO Sweep")
    log.info(f"  cache_dir   : {args.cache_dir}")
    log.info(f"  n_days      : {args.n_days or 'all'}")
    log.info(f"  horizon     : {args.horizon} bars = {args.horizon / BARS_PER_SEC:.1f}s")
    log.info(f"  quick       : {args.quick}")
    log.info(f"  targets     : {args.targets or 'all'}")
    log.info(f"  output_dir  : {SIGNAL_PRED_DIR}")
    log.info("=" * 70)

    run_sweep(
        cache_dir=args.cache_dir,
        n_days=args.n_days,
        horizon_bars=args.horizon,
        quick=args.quick,
        targets=args.targets,
    )
