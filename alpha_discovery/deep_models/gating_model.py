"""
gating_model.py — IC Regime Gating Model for BookSpatialCNN

Predicts whether the CNN's directional signal will be accurate in the current
market regime. Allows filtering low-confidence trades.

Walk-forward methodology: for each OOT fold, train only on past data (IS + prior OOT folds).

Data sources:
  - Book cache OOT: data/processed/dl_book_cache_oot/YYYY-MM-DD_book_tensors.npz
  - WF predictions:  alpha_discovery/deep_models/results/oot_wf_predictions_incremental.npz

Usage:
    python gating_model.py
    python gating_model.py --local-ic-window 500 --threshold 0.08 --min-train-days 5
"""

import argparse
import json
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

FILE_DIR = Path(__file__).parent.resolve()
ROOT = FILE_DIR.parent.parent
OOT_DIR = ROOT / "data/processed/dl_book_cache_oot"
PRED_PATH = FILE_DIR / "results/oot_wf_predictions_incremental.npz"
RESULTS_DIR = FILE_DIR / "results"

# ─────────────────────────────────────────────────────────────
#  Feature engineering on raw book data
# ─────────────────────────────────────────────────────────────

def compute_rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling std with Welford's algorithm via cumsum (fast, no NaN padding needed)."""
    result = np.full(len(arr), np.nan, dtype=np.float32)
    if len(arr) < window:
        return result
    # Use pandas-style efficient rolling via np.lib.stride_tricks
    # For large arrays just use a simple cumulative approach
    cumsum = np.cumsum(arr)
    cumsum2 = np.cumsum(arr ** 2)
    n = window
    for i in range(n - 1, len(arr)):
        s = cumsum[i] - (cumsum[i - n] if i >= n else 0.0)
        s2 = cumsum2[i] - (cumsum2[i - n] if i >= n else 0.0)
        var = (s2 - s * s / n) / (n - 1) if n > 1 else 0.0
        result[i] = np.sqrt(max(var, 0.0))
    return result


def compute_rolling_std_fast(arr: np.ndarray, window: int) -> np.ndarray:
    """Vectorized rolling std using stride tricks."""
    result = np.full(len(arr), np.nan, dtype=np.float32)
    if len(arr) < window:
        return result
    # Cumulative sum approach (O(N), numerically stable enough for our purposes)
    a = arr.astype(np.float64)
    cs = np.cumsum(a)
    cs2 = np.cumsum(a * a)
    # For i >= window-1: sum_window = cs[i] - cs[i-window]
    i_end = np.arange(window - 1, len(arr))
    i_start = i_end - window  # -1 means use 0 as start
    s = cs[i_end] - np.where(i_start >= 0, cs[i_start], 0.0)
    s2 = cs2[i_end] - np.where(i_start >= 0, cs2[i_start], 0.0)
    var = (s2 - s * s / window) / max(window - 1, 1)
    var = np.maximum(var, 0.0)
    result[window - 1:] = np.sqrt(var).astype(np.float32)
    return result


def build_regime_features(
    book_tensors: np.ndarray,   # (N, 20, 4)
    mid_prices: np.ndarray,     # (N,)
    preds: np.ndarray,          # (N,) CNN predictions
    timestamps: np.ndarray,     # (N,) nanosecond timestamps
    horizon: int = 100,         # bars, must match CNN training
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-bar regime features from book + prediction data.

    Returns:
        features: (N, F) float32
        targets:  (N,)  int8   — 1 if local IC good, 0 if bad (or NaN)
        valid_mask: (N,) bool
    """
    N = len(mid_prices)

    # ── 1. Realized volatility at multiple windows ──────────────────────────
    mid_ret = np.diff(mid_prices / 0.25, prepend=mid_prices[0] / 0.25).astype(np.float32)  # in ticks

    rvol_100  = compute_rolling_std_fast(mid_ret, 100)
    rvol_500  = compute_rolling_std_fast(mid_ret, 500)
    rvol_1000 = compute_rolling_std_fast(mid_ret, 1000)
    rvol_3000 = compute_rolling_std_fast(mid_ret, 3000)

    # ── 2. Bid-ask spread ───────────────────────────────────────────────────
    # Feature 0: price_relative_to_mid (in ticks, bid levels 0-9, ask levels 10-19)
    # bid_best = level 0 feature 0 (negative = below mid)
    # ask_best = level 10 feature 0 (positive = above mid)
    bid_best = book_tensors[:, 0, 0]    # negative ticks from mid
    ask_best = book_tensors[:, 10, 0]   # positive ticks from mid
    spread_cur = (ask_best - bid_best).astype(np.float32)  # ticks
    spread_ma = compute_rolling_std_fast(spread_cur, 500)  # use rolling mean instead

    # Rolling mean for spread (not std)
    spread_roll_mean = np.full(N, np.nan, dtype=np.float32)
    cs = np.cumsum(spread_cur.astype(np.float64))
    for w in [500]:
        i_end = np.arange(w - 1, N)
        i_start = i_end - w
        s = cs[i_end] - np.where(i_start >= 0, cs[i_start], 0.0)
        spread_roll_mean[w - 1:] = (s / w).astype(np.float32)

    # ── 3. Book imbalance at multiple depth levels ──────────────────────────
    bid_depth_l1  = book_tensors[:, 0, 1]    # best bid depth
    ask_depth_l1  = book_tensors[:, 10, 1]   # best ask depth
    bid_depth_5   = book_tensors[:, :5, 1].sum(axis=1)
    ask_depth_5   = book_tensors[:, 10:15, 1].sum(axis=1)
    bid_depth_all = book_tensors[:, :10, 1].sum(axis=1)
    ask_depth_all = book_tensors[:, 10:, 1].sum(axis=1)
    total_all = bid_depth_all + ask_depth_all + 1e-8

    imbalance_l1  = ((bid_depth_l1 - ask_depth_l1) / (bid_depth_l1 + ask_depth_l1 + 1e-8)).astype(np.float32)
    imbalance_5   = ((bid_depth_5 - ask_depth_5) / (bid_depth_5 + ask_depth_5 + 1e-8)).astype(np.float32)
    imbalance_all = ((bid_depth_all - ask_depth_all) / total_all).astype(np.float32)

    # Book depth asymmetry (log ratio of total depth)
    depth_asym = np.log1p(bid_depth_all + 1e-8) - np.log1p(ask_depth_all + 1e-8)
    depth_asym = depth_asym.astype(np.float32)

    # ── 4. Time of day (minutes since 9:30 ET) ──────────────────────────────
    # timestamps are nanoseconds; session opens at 9:30 ET
    # Bars start at 0 = bar 0 corresponds to 9:30:00 ET
    # 234000 bars / 6.5 hrs = 36000 bars/hr = 600 bars/min
    bar_idx = np.arange(N, dtype=np.float32)
    # 1 bar = 100ms → minutes = bar_idx / 600
    minutes_since_open = bar_idx / 600.0  # 0 to 390 minutes

    # Cyclic encoding (cycle over full 390-min day)
    tod_sin = np.sin(2 * np.pi * minutes_since_open / 390.0).astype(np.float32)
    tod_cos = np.cos(2 * np.pi * minutes_since_open / 390.0).astype(np.float32)

    # ── 5. Rolling prediction accuracy (IC-like correlation over window) ────
    # Compute MFE targets same as training (horizon=100 bars)
    mfe_targets = np.full(N, np.nan, dtype=np.float32)
    mid64 = mid_prices.astype(np.float64)
    for i in range(N - horizon):
        future = mid64[i + 1: i + 1 + horizon]
        mfe_targets[i] = float(np.max(future) - mid64[i] - (mid64[i] - np.min(future)))

    # Rolling IC over past windows using Spearman rank correlation approx
    # (use Pearson on raw values for speed — good enough for regime features)
    def rolling_ic(pred_arr, tgt_arr, window):
        out = np.full(N, np.nan, dtype=np.float32)
        pred64 = pred_arr.astype(np.float64)
        tgt64  = tgt_arr.astype(np.float64)
        for i in range(window - 1, N):
            p = pred64[i - window + 1: i + 1]
            t = tgt64[i - window + 1: i + 1]
            mask = np.isfinite(p) & np.isfinite(t)
            if mask.sum() < 10:
                continue
            pm, tm = p[mask], t[mask]
            if pm.std() < 1e-10 or tm.std() < 1e-10:
                continue
            out[i] = float(np.corrcoef(pm, tm)[0, 1])
        return out

    # Rolling IC at 100, 500, 1000 bar windows (vectorized using numpy for speed)
    def rolling_ic_fast(pred_arr, tgt_arr, window):
        """Vectorized rolling IC using cumsum method."""
        out = np.full(N, np.nan, dtype=np.float32)
        valid = np.isfinite(pred_arr) & np.isfinite(tgt_arr)
        p = np.where(valid, pred_arr.astype(np.float64), 0.0)
        t = np.where(valid, tgt_arr.astype(np.float64), 0.0)
        v = valid.astype(np.float64)

        cp  = np.cumsum(p)
        ct  = np.cumsum(t)
        cp2 = np.cumsum(p * p)
        ct2 = np.cumsum(t * t)
        cpt = np.cumsum(p * t)
        cv  = np.cumsum(v)

        for i in range(window - 1, N):
            j = i - window
            n_v = cv[i] - (cv[j] if j >= 0 else 0.0)
            if n_v < 10:
                continue
            sp  = cp[i]  - (cp[j]  if j >= 0 else 0.0)
            st  = ct[i]  - (ct[j]  if j >= 0 else 0.0)
            sp2 = cp2[i] - (cp2[j] if j >= 0 else 0.0)
            st2 = ct2[i] - (ct2[j] if j >= 0 else 0.0)
            spt = cpt[i] - (cpt[j] if j >= 0 else 0.0)
            # Pearson
            num = n_v * spt - sp * st
            den = np.sqrt(max(n_v * sp2 - sp * sp, 0.0) * max(n_v * st2 - st * st, 0.0))
            if den < 1e-12:
                continue
            out[i] = float(num / den)
        return out

    ic_100  = rolling_ic_fast(preds, mfe_targets, 100)
    ic_500  = rolling_ic_fast(preds, mfe_targets, 500)
    ic_1000 = rolling_ic_fast(preds, mfe_targets, 1000)

    # ── 6. Order flow imbalance (momentum proxy) ────────────────────────────
    # Approximated by signed mid-price changes over short windows
    # Cumulative signed flow: positive if up-moves dominate
    signed_flow = np.sign(mid_ret).astype(np.float32)
    # Rolling sum over windows
    def rolling_sum_fast(arr, window):
        cs = np.cumsum(arr.astype(np.float64))
        out = np.full(len(arr), np.nan, dtype=np.float32)
        i_end = np.arange(window - 1, len(arr))
        i_start = i_end - window
        s = cs[i_end] - np.where(i_start >= 0, cs[i_start], 0.0)
        out[window - 1:] = (s / window).astype(np.float32)
        return out

    ofi_100  = rolling_sum_fast(signed_flow, 100)
    ofi_500  = rolling_sum_fast(signed_flow, 500)

    # ── 7. Price momentum at multiple lookbacks ──────────────────────────────
    def price_momentum(mid, lookback):
        """Return in ticks over lookback bars."""
        out = np.full(N, np.nan, dtype=np.float32)
        out[lookback:] = ((mid[lookback:] - mid[:-lookback]) / 0.25).astype(np.float32)
        return out

    mom_100  = price_momentum(mid_prices, 100)
    mom_500  = price_momentum(mid_prices, 500)
    mom_1000 = price_momentum(mid_prices, 1000)

    # ── 8. Prediction magnitude and statistics ───────────────────────────────
    abs_pred = np.abs(preds).astype(np.float32)
    pred_sign = np.sign(preds).astype(np.float32)

    # Rolling mean abs prediction (higher = model more confident on average)
    abs_pred_ma_100  = rolling_sum_fast(abs_pred, 100)
    abs_pred_ma_500  = rolling_sum_fast(abs_pred, 500)

    # Prediction std (consistency)
    pred_std_100  = compute_rolling_std_fast(preds.astype(np.float32), 100)
    pred_std_500  = compute_rolling_std_fast(preds.astype(np.float32), 500)

    # ── 9. Queue age features (staleness of liquidity) ──────────────────────
    # Feature 3 is queue_age_seconds
    bid_age_best = book_tensors[:, 0, 3].astype(np.float32)
    ask_age_best = book_tensors[:, 10, 3].astype(np.float32)
    age_imbalance = (bid_age_best - ask_age_best) / (bid_age_best + ask_age_best + 1e-8)

    # ── Assemble feature matrix ──────────────────────────────────────────────
    feature_list = [
        # Volatility
        rvol_100, rvol_500, rvol_1000, rvol_3000,
        # Spread
        spread_cur, spread_roll_mean,
        # Book imbalance
        imbalance_l1, imbalance_5, imbalance_all, depth_asym,
        # Time of day
        tod_sin, tod_cos,
        # Rolling IC
        ic_100, ic_500, ic_1000,
        # Order flow
        ofi_100, ofi_500,
        # Momentum
        mom_100, mom_500, mom_1000,
        # Prediction features
        abs_pred, abs_pred_ma_100, abs_pred_ma_500,
        pred_std_100, pred_std_500,
        # Queue age
        bid_age_best, ask_age_best, age_imbalance,
    ]

    feature_names = [
        "rvol_100", "rvol_500", "rvol_1000", "rvol_3000",
        "spread_cur", "spread_roll_mean",
        "imbalance_l1", "imbalance_5", "imbalance_all", "depth_asym",
        "tod_sin", "tod_cos",
        "ic_100", "ic_500", "ic_1000",
        "ofi_100", "ofi_500",
        "mom_100", "mom_500", "mom_1000",
        "abs_pred", "abs_pred_ma_100", "abs_pred_ma_500",
        "pred_std_100", "pred_std_500",
        "bid_age_best", "ask_age_best", "age_imbalance",
    ]

    X = np.stack(feature_list, axis=1)  # (N, F)

    # ── Compute target labels ────────────────────────────────────────────────
    # Local IC computed over rolling windows AHEAD (future data — leakage-free in WF mode)
    # For the gating model, we want: "if I look forward N bars from here, is this a good regime?"
    # We use the actual sign prediction accuracy as a binary label.
    # Label: is the next `local_window` bars' IC above threshold?
    # We'll compute this at the daily level (per-bar target = daily IC binary)
    # This approach avoids forward-looking IC within a day when used in WF context.
    # Daily IC is pre-computed from fold stats but we build per-bar targets here.

    # Per-bar target: future return sign matches prediction sign (hit rate proxy)
    # We assign each bar a "local quality" = hit rate over next 500 bars
    # This IS forward-looking during feature computation, but we handle leakage
    # at the FOLD level: we only use past days' features to predict current day labels.
    local_win = 500
    local_quality = np.full(N, np.nan, dtype=np.float32)

    # For each bar, compute hit rate of FUTURE bars (forward-looking, handled by WF split)
    # sign(pred) == sign(actual_return) over next local_win bars
    actual_sign = np.sign(mid_ret)
    pred_sign_arr = np.sign(preds)

    # Compute rolling forward hit rate
    hits = (pred_sign_arr == actual_sign).astype(np.float32)
    for i in range(N - local_win):
        window_hits = hits[i: i + local_win]
        # Only where both pred and actual are non-zero (directional)
        valid_w = (pred_sign_arr[i: i + local_win] != 0) & (actual_sign[i: i + local_win] != 0)
        if valid_w.sum() > 20:
            local_quality[i] = hits[i: i + local_win][valid_w].mean()

    # Also compute forward rolling IC (more informative than hit rate)
    forward_ic = np.full(N, np.nan, dtype=np.float32)
    p64 = preds.astype(np.float64)
    t64 = mfe_targets.astype(np.float64)
    for i in range(N - local_win):
        pw = p64[i: i + local_win]
        tw = t64[i: i + local_win]
        mask_w = np.isfinite(pw) & np.isfinite(tw)
        if mask_w.sum() < 20:
            continue
        pm, tm = pw[mask_w], tw[mask_w]
        if pm.std() < 1e-10 or tm.std() < 1e-10:
            continue
        forward_ic[i] = float(np.corrcoef(pm, tm)[0, 1])

    return X, forward_ic, local_quality, mfe_targets, feature_names


# ─────────────────────────────────────────────────────────────
#  Data loading
# ─────────────────────────────────────────────────────────────

def load_oot_data(oot_dir: Path, pred_path: Path) -> List[Dict]:
    """Load book cache + WF predictions for each OOT day with predictions."""
    preds_npz = np.load(str(pred_path), allow_pickle=True)
    available_dates = set(
        k.replace("_preds", "") for k in preds_npz.files if k.endswith("_preds")
    )

    book_files = sorted(oot_dir.glob("*_book_tensors.npz"))
    days = []

    for bf in book_files:
        date_str = bf.name.replace("_book_tensors.npz", "")
        if date_str not in available_dates:
            continue
        try:
            data = np.load(str(bf), allow_pickle=False)
            bt      = data["book_tensors"].astype(np.float32)   # (N, 20, 4)
            mid     = data["mid_prices"].astype(np.float64)      # (N,)
            ts      = data["timestamps"].astype(np.int64)        # (N,)
            data.close()

            p_key = f"{date_str}_preds"
            m_key = f"{date_str}_mid"
            pred_arr = preds_npz[p_key].astype(np.float32)       # (N,)
            # mid from predictions file (should match, but use book cache mid for accuracy)

            if len(pred_arr) != len(mid):
                print(f"  WARNING: {date_str} pred length {len(pred_arr)} != book length {len(mid)}")
                # Truncate to shorter
                n_min = min(len(pred_arr), len(mid))
                pred_arr = pred_arr[:n_min]
                mid      = mid[:n_min]
                bt       = bt[:n_min]
                ts       = ts[:n_min]

            days.append({
                "date": date_str,
                "bt": bt,
                "mid": mid,
                "ts": ts,
                "preds": pred_arr,
                "n": len(mid),
            })
        except Exception as e:
            print(f"  Error loading {date_str}: {e}")

    preds_npz.close()
    print(f"Loaded {len(days)} days with predictions.")
    return days


# ─────────────────────────────────────────────────────────────
#  Walk-Forward Gating Model Training
# ─────────────────────────────────────────────────────────────

def run_walkforward_gating(
    days: List[Dict],
    ic_threshold: float = 0.08,
    use_median_threshold: bool = True,
    local_window: int = 500,
    min_train_days: int = 5,
    subsample_train: int = 10,
    n_estimators: int = 200,
    num_leaves: int = 31,
    verbose: bool = True,
) -> Dict:
    """
    Walk-forward gating model training.

    For each fold day D:
        - Build features from all prior days (IS training set)
        - Train LightGBM to predict: will local IC > threshold?
        - Apply to day D (truly out-of-sample)
        - Measure: does gating improve Sharpe vs "trade all"?
    """
    import lightgbm as lgb
    from sklearn.metrics import classification_report, roc_auc_score
    from sklearn.calibration import calibration_curve

    print("\n" + "=" * 70)
    print("IC Regime Gating Model - Walk-Forward Training")
    print(f"IC threshold: {ic_threshold}")
    print(f"Local IC window: {local_window} bars ({local_window * 0.1:.0f}s)")
    print(f"Min train days: {min_train_days}")
    print("=" * 70)

    # ── Step 1: Compute features for all days ────────────────────────────────
    print("\n[1/4] Computing regime features for all days...")
    all_day_data = []
    for i, day in enumerate(days):
        if verbose:
            print(f"  Processing {day['date']} (day {i+1}/{len(days)})...", end=" ")
        X, fwd_ic, local_qual, targets, feat_names = build_regime_features(
            day["bt"], day["mid"], day["preds"], day["ts"], horizon=100
        )
        all_day_data.append({
            "date":    day["date"],
            "X":       X,
            "fwd_ic":  fwd_ic,
            "targets": targets,
            "preds":   day["preds"],
            "mid":     day["mid"],
            "n":       day["n"],
        })
        # Daily IC summary
        valid_ic = fwd_ic[np.isfinite(fwd_ic)]
        if verbose:
            print(f"  fwd_IC: mean={valid_ic.mean():.4f} std={valid_ic.std():.4f} n={len(valid_ic)}" if len(valid_ic) > 0 else "no valid IC")

    feature_names = feat_names

    # ── Step 2: Walk-forward fold loop ──────────────────────────────────────
    print(f"\n[2/4] Walk-forward training ({len(all_day_data)} folds)...")

    fold_results = []
    oof_preds_proba  = []
    oof_labels       = []
    oof_trades_all   = []  # "trade all" baseline
    oof_trades_gated = []  # gated trades

    for fold_idx in range(min_train_days, len(all_day_data)):
        test_day = all_day_data[fold_idx]
        train_days_data = all_day_data[:fold_idx]  # all prior days

        # ── Compute dynamic threshold from training data ─────────────────────
        # Collect all valid forward ICs from training days for threshold
        if use_median_threshold:
            all_train_fwd_ics = []
            for td in train_days_data:
                fwd_raw = td["fwd_ic"]
                valid_raw = fwd_raw[np.isfinite(fwd_raw)]
                all_train_fwd_ics.extend(valid_raw.tolist())
            if all_train_fwd_ics:
                fold_threshold = float(np.median(all_train_fwd_ics))
            else:
                fold_threshold = ic_threshold
        else:
            fold_threshold = ic_threshold

        # Build training matrix
        X_parts, y_parts = [], []
        for td in train_days_data:
            # Label: 1 if forward IC > threshold at this bar, else 0
            fwd = td["fwd_ic"]
            valid = np.isfinite(fwd)
            if valid.sum() < 20:
                continue
            # Use dynamic threshold to create binary label
            labels = (fwd >= fold_threshold).astype(np.int8)
            X_day  = td["X"]

            # Mask: only bars with valid features and labels
            feat_valid = np.all(np.isfinite(X_day), axis=1)
            mask = valid & feat_valid
            if mask.sum() < 20:
                continue

            # Subsample training data (every N-th bar to reduce autocorrelation)
            indices = np.where(mask)[0][::subsample_train]
            X_parts.append(X_day[indices])
            y_parts.append(labels[indices])

        if not X_parts:
            continue

        X_train = np.concatenate(X_parts, axis=0)
        y_train = np.concatenate(y_parts, axis=0)

        if len(X_train) < 100:
            continue

        # Class balance check
        pos_rate = y_train.mean()
        if verbose:
            thresh_disp = fold_threshold if use_median_threshold else ic_threshold
            print(f"  Fold {fold_idx+1}/{len(all_day_data)} - {test_day['date']} | "
                  f"train: {len(X_train):,} bars, pos_rate={pos_rate:.2%}, "
                  f"thresh={thresh_disp:.4f}")

        # ── Train LightGBM ──────────────────────────────────────────────────
        scale_pos_weight = (1 - pos_rate) / (pos_rate + 1e-8)
        params = {
            "objective":         "binary",
            "metric":            "auc",
            "n_estimators":      n_estimators,
            "num_leaves":        num_leaves,
            "learning_rate":     0.05,
            "feature_fraction":  0.8,
            "bagging_fraction":  0.8,
            "bagging_freq":      5,
            "min_child_samples": 50,
            "scale_pos_weight":  scale_pos_weight,
            "verbosity":         -1,
            "n_jobs":            -1,
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)

        # ── Inference on test day ──────────────────────────────────────────
        X_test  = test_day["X"]
        fwd_test = test_day["fwd_ic"]
        preds_cnn = test_day["preds"]
        mid = test_day["mid"]
        N_test = test_day["n"]

        feat_valid_test = np.all(np.isfinite(X_test), axis=1)
        label_valid_test = np.isfinite(fwd_test)
        eval_mask = feat_valid_test & label_valid_test

        if eval_mask.sum() < 50:
            continue

        y_test = (fwd_test[eval_mask] >= fold_threshold).astype(np.int8)
        proba  = model.predict_proba(X_test[eval_mask])[:, 1]

        # ── Metrics ───────────────────────────────────────────────────────
        y_pred_bin = (proba >= 0.5).astype(np.int8)
        try:
            auc = roc_auc_score(y_test, proba)
        except Exception:
            auc = float("nan")

        acc = (y_pred_bin == y_test).mean()
        tp  = ((y_pred_bin == 1) & (y_test == 1)).sum()
        fp  = ((y_pred_bin == 1) & (y_test == 0)).sum()
        fn  = ((y_pred_bin == 0) & (y_test == 1)).sum()
        tn  = ((y_pred_bin == 0) & (y_test == 0)).sum()
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)

        # ── Sharpe comparison: trade all vs gated ────────────────────────
        # We simulate a simple strategy: go long if CNN pred > 0, short if < 0
        # Horizon = 100 bars (same as CNN training target)
        horizon_bars = 100
        trade_returns = []
        gated_returns = []

        # Get gating proba for all bars (not just eval_mask)
        full_proba = np.zeros(N_test, dtype=np.float32)
        try:
            fp_raw = model.predict_proba(X_test)[:, 1]
            full_proba = fp_raw
        except Exception:
            full_proba[feat_valid_test] = proba[: feat_valid_test.sum()]

        gate_threshold = 0.55  # Only trade when confident it's good regime

        for i in range(0, N_test - horizon_bars, horizon_bars):
            p_cnn = preds_cnn[i]
            if not np.isfinite(p_cnn) or abs(p_cnn) < 1e-6:
                continue

            future_ret = (mid[i + horizon_bars] - mid[i]) / 0.25  # ticks
            direction  = np.sign(p_cnn)
            ret_ticks  = direction * future_ret  # positive = correct direction

            trade_returns.append(ret_ticks)

            # Gated: only take trade if model predicts good regime
            gate_val = full_proba[i] if i < len(full_proba) else 0.5
            if gate_val >= gate_threshold:
                gated_returns.append(ret_ticks)

        def sharpe(rets):
            """Information ratio: mean/std. Not annualized — makes per-day comparable."""
            arr = np.array(rets)
            if len(arr) < 5:
                return float("nan")
            return float(arr.mean() / (arr.std() + 1e-8))

        def mean_ret(rets):
            return float(np.mean(rets)) if len(rets) > 0 else float("nan")

        fold_result = {
            "date":           test_day["date"],
            "fold_idx":       fold_idx,
            "train_samples":  int(len(X_train)),
            "test_bars":      int(N_test),
            "eval_bars":      int(eval_mask.sum()),
            "auc":            float(auc),
            "accuracy":       float(acc),
            "precision":      float(prec),
            "recall":         float(rec),
            "f1":             float(f1),
            "pos_rate_train": float(pos_rate),
            "pos_rate_test":  float(y_test.mean()),
            "n_trades_all":   len(trade_returns),
            "n_trades_gated": len(gated_returns),
            "sharpe_all":     sharpe(trade_returns),
            "sharpe_gated":   sharpe(gated_returns),
            "mean_ret_all":   mean_ret(trade_returns),
            "mean_ret_gated": mean_ret(gated_returns),
            "gate_ratio":     len(gated_returns) / (len(trade_returns) + 1e-8),
            "fold_threshold":  float(fold_threshold),
        }
        fold_results.append(fold_result)

        if verbose:
            print(f"    AUC={auc:.3f} Acc={acc:.2%} F1={f1:.3f} | "
                  f"Sharpe all={fold_result['sharpe_all']:.3f} "
                  f"gated={fold_result['sharpe_gated']:.3f} "
                  f"({fold_result['gate_ratio']:.0%} trades kept)")

        # Collect OOF predictions
        oof_preds_proba.extend(proba.tolist())
        oof_labels.extend(y_test.tolist())

    # ── Step 3: Aggregate results ────────────────────────────────────────────
    print(f"\n[3/4] Aggregating results across {len(fold_results)} folds...")

    if not fold_results:
        print("ERROR: No fold results! Need more training data.")
        return {}

    auc_arr     = np.array([r["auc"]           for r in fold_results if np.isfinite(r["auc"])])
    acc_arr     = np.array([r["accuracy"]       for r in fold_results])
    f1_arr      = np.array([r["f1"]             for r in fold_results])
    sh_all_arr  = np.array([r["sharpe_all"]     for r in fold_results if np.isfinite(r["sharpe_all"])])
    sh_gate_arr = np.array([r["sharpe_gated"]   for r in fold_results if np.isfinite(r["sharpe_gated"])])
    gate_ratio  = np.array([r["gate_ratio"]     for r in fold_results])

    # Feature importance from last fold's model
    feat_importance = {}
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        for name, val in sorted(zip(feature_names, imp), key=lambda x: -x[1]):
            feat_importance[name] = int(val)

    # ── Overall OOF AUC ──────────────────────────────────────────────────────
    oof_auc = float("nan")
    if oof_labels and oof_preds_proba:
        try:
            oof_auc = roc_auc_score(oof_labels, oof_preds_proba)
        except Exception:
            pass

    summary = {
        "n_folds":          len(fold_results),
        "oof_auc":          float(oof_auc),
        "mean_auc":         float(auc_arr.mean()) if len(auc_arr) > 0 else float("nan"),
        "mean_accuracy":    float(acc_arr.mean()),
        "mean_f1":          float(f1_arr.mean()),
        "mean_sharpe_all":    float(sh_all_arr.mean()) if len(sh_all_arr) > 0 else float("nan"),
        "mean_sharpe_gated":  float(sh_gate_arr.mean()) if len(sh_gate_arr) > 0 else float("nan"),
        "sharpe_improvement": float(sh_gate_arr.mean() - sh_all_arr.mean()) if (len(sh_all_arr) > 0 and len(sh_gate_arr) > 0) else float("nan"),
        "mean_gate_ratio":    float(gate_ratio.mean()),
        "ic_threshold":       ic_threshold,
        "local_window_bars":  local_window,
        "feature_importance": feat_importance,
        "fold_results":       fold_results,
        "timestamp":          datetime.now().isoformat(),
    }

    # ── Step 4: Print summary ────────────────────────────────────────────────
    print("\n[4/4] RESULTS SUMMARY")
    print("=" * 70)
    print(f"Folds evaluated:       {len(fold_results)}")
    print(f"OOF AUC:               {oof_auc:.4f}")
    print(f"Mean per-fold AUC:     {summary['mean_auc']:.4f}")
    print(f"Mean accuracy:         {summary['mean_accuracy']:.2%}")
    print(f"Mean F1:               {summary['mean_f1']:.4f}")
    print()
    print(f"Sharpe (trade all):    {summary['mean_sharpe_all']:.4f}")
    print(f"Sharpe (gated):        {summary['mean_sharpe_gated']:.4f}")
    print(f"Sharpe improvement:    {summary['sharpe_improvement']:+.4f}")
    print(f"Trades kept by gate:   {summary['mean_gate_ratio']:.1%}")
    print()
    print("Top 10 Feature Importance:")
    for name, val in list(feat_importance.items())[:10]:
        print(f"  {name:30s}: {val}")
    print()

    # Per-fold results table
    print(f"{'Date':<12} {'AUC':>6} {'F1':>6} {'ShAll':>8} {'ShGate':>8} {'Gate%':>7}")
    print("-" * 55)
    for r in fold_results:
        auc_s  = f"{r['auc']:.3f}" if np.isfinite(r["auc"]) else "  nan"
        sh_a   = f"{r['sharpe_all']:.3f}" if np.isfinite(r["sharpe_all"]) else "  nan"
        sh_g   = f"{r['sharpe_gated']:.3f}" if np.isfinite(r["sharpe_gated"]) else "  nan"
        print(f"{r['date']:<12} {auc_s:>6} {r['f1']:>6.3f} {sh_a:>8} {sh_g:>8} {r['gate_ratio']:>6.0%}")

    return summary


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="IC Regime Gating Model")
    parser.add_argument("--oot-dir",       type=Path, default=OOT_DIR)
    parser.add_argument("--pred-path",     type=Path, default=PRED_PATH)
    parser.add_argument("--results-dir",   type=Path, default=RESULTS_DIR)
    parser.add_argument("--threshold",     type=float, default=0.08,
                        help="IC threshold for 'good regime' label (default: 0.08)")
    parser.add_argument("--local-ic-window", type=int, default=500,
                        help="Forward window in bars to compute local IC (default: 500 = 50s)")
    parser.add_argument("--min-train-days", type=int, default=5,
                        help="Minimum days before first test fold (default: 5)")
    parser.add_argument("--subsample",    type=int, default=10,
                        help="Training bar subsampling (default: every 10th bar)")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--num-leaves",   type=int, default=31)
    parser.add_argument("--quiet",        action="store_true")
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)

    print("IC Regime Gating Model")
    print(f"OOT dir:     {args.oot_dir}")
    print(f"Predictions: {args.pred_path}")
    print(f"Results dir: {args.results_dir}")
    print()

    # ── Load data ─────────────────────────────────────────────────────────
    print("[0/4] Loading data...")
    days = load_oot_data(args.oot_dir, args.pred_path)
    if not days:
        print("ERROR: No data found!")
        sys.exit(1)
    print(f"  {len(days)} days available: {days[0]['date']} to {days[-1]['date']}")

    # ── Run gating model ──────────────────────────────────────────────────
    summary = run_walkforward_gating(
        days=days,
        ic_threshold=args.threshold,
        use_median_threshold=True,
        local_window=args.local_ic_window,
        min_train_days=args.min_train_days,
        subsample_train=args.subsample,
        n_estimators=args.n_estimators,
        num_leaves=args.num_leaves,
        verbose=not args.quiet,
    )

    # ── Save results ──────────────────────────────────────────────────────
    out_path = args.results_dir / "gating_model_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
