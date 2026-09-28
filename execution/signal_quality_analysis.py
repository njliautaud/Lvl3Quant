#!/usr/bin/env python3
"""
CNN-Mamba Signal Quality Deep Analysis — HC #44 Model Evaluation Checklist
Comprehensive per-fold OOT analysis across 10 checklist dimensions.

Inputs:
  - CNN-Mamba v2 OOT predictions: output/cnn_mamba_v2_smart_v3_mar/fold_XX_oot_predictions.npz
  - Exec features: output/exec_features_v1/YYYYMMDD_exec_features.npz

Output:
  - JSON report + human-readable log in output/signal_quality_analysis/
"""

import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy import stats

# --- Config ---
BASE = Path("/home/jupiter/Lvl3Quant")
PRED_DIR = BASE / "output" / "cnn_mamba_v2_smart_v3_mar"
EXEC_DIR = BASE / "output" / "exec_features_v1"
OUT_DIR = BASE / "output" / "signal_quality_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = ["1s", "5s", "10s"]
HORIZON_IDX = {h: i for i, h in enumerate(HORIZONS)}
COMMISSION_TICKS = 0.376
NUM_WORKERS = 16

# Market hours for TOD derivation (ET)
MARKET_OPEN_MIN = 9 * 60 + 30   # 9:30
MARKET_CLOSE_MIN = 16 * 60      # 16:00
TOTAL_MARKET_MIN = MARKET_CLOSE_MIN - MARKET_OPEN_MIN  # 390

# TOD hour bins
HOUR_BINS = [
    ("09:30-10:00", 9.5, 10.0),
    ("10:00-11:00", 10.0, 11.0),
    ("11:00-12:00", 11.0, 12.0),
    ("12:00-13:00", 12.0, 13.0),
    ("13:00-14:00", 13.0, 14.0),
    ("14:00-15:00", 14.0, 15.0),
    ("15:00-16:00", 15.0, 16.0),
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUT_DIR / "analysis.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)


# ============================================================
# Utility functions
# ============================================================

def pearson_ic(pred, label):
    """Pearson correlation (IC) with NaN safety."""
    mask = np.isfinite(pred) & np.isfinite(label)
    if mask.sum() < 10:
        return np.nan
    return np.corrcoef(pred[mask], label[mask])[0, 1]


def rank_ic(pred, label):
    """Spearman rank IC."""
    mask = np.isfinite(pred) & np.isfinite(label)
    if mask.sum() < 10:
        return np.nan
    return stats.spearmanr(pred[mask], label[mask]).correlation


def hit_rate(pred, label):
    """Directional accuracy."""
    mask = (pred != 0) & (label != 0) & np.isfinite(pred) & np.isfinite(label)
    if mask.sum() < 10:
        return np.nan
    return np.mean(np.sign(pred[mask]) == np.sign(label[mask]))


def extract_date_from_oot(oot_files):
    """Extract YYYYMMDD date string from oot_files paths."""
    for f in oot_files:
        m = re.search(r"(\d{8})", str(f))
        if m:
            return m.group(1)
    return None


def derive_tod_hours(n_events):
    """Derive approximate time-of-day (fractional hours) from index position.
    Events are sequential within a trading day 9:30-16:00."""
    fracs = np.linspace(0, 1, n_events, endpoint=False)
    hours = 9.5 + fracs * (16.0 - 9.5)  # 9.5 = 9:30, 16.0 = 16:00
    return hours


def load_exec_features(date_str):
    """Load exec features for a date, return dict with feature arrays and names."""
    path = EXEC_DIR / f"{date_str}_exec_features.npz"
    if not path.exists():
        return None
    f = np.load(path, allow_pickle=True)
    feat_names = [str(n) for n in f["feature_names"]]
    features = f["features"]
    n_windows = int(f["n_windows"])
    decision_stride = int(f["decision_stride"])
    return {
        "features": features,
        "feature_names": feat_names,
        "n_windows": n_windows,
        "decision_stride": decision_stride,
    }


def get_exec_feature_for_events(exec_data, n_events, feature_name):
    """Map exec feature windows to event-level using nearest-window assignment."""
    if exec_data is None:
        return None
    feat_names = exec_data["feature_names"]
    if feature_name not in feat_names:
        return None
    idx = feat_names.index(feature_name)
    feat_vals = exec_data["features"][:, idx]
    n_windows = len(feat_vals)
    # Map each event to nearest window
    event_window_idx = np.linspace(0, n_windows - 1, n_events).astype(int)
    return feat_vals[event_window_idx]


# ============================================================
# Per-fold analysis (runs in parallel)
# ============================================================

def analyze_fold(fold_idx):
    """Analyze a single fold. Returns dict with all metrics."""
    path = PRED_DIR / f"fold_{fold_idx:02d}_oot_predictions.npz"
    if not path.exists():
        return None

    f = np.load(path, allow_pickle=True)
    preds = f["predictions"]  # (N, 3)
    labels = f["labels"]      # (N, 3)
    oot_files = f["oot_files"]
    n_events = preds.shape[0]
    date_str = extract_date_from_oot(oot_files)

    if n_events < 50:
        log.warning(f"Fold {fold_idx}: only {n_events} events, skipping")
        return None

    # Load exec features for this date
    exec_data = load_exec_features(date_str) if date_str else None

    result = {
        "fold": fold_idx,
        "date": date_str,
        "n_events": n_events,
    }

    # ---- 1. IC per horizon ----
    ic_data = {}
    for h in HORIZONS:
        hi = HORIZON_IDX[h]
        p, l = preds[:, hi], labels[:, hi]
        ic_data[h] = {
            "pearson_ic": float(pearson_ic(p, l)),
            "rank_ic": float(rank_ic(p, l)),
            "stored_ic": float(f[f"ic_{h}"].item()),
        }
    result["ic_per_horizon"] = ic_data

    # ---- 2. Confidence-conditional IC ----
    conf_ic = {}
    for h in HORIZONS:
        hi = HORIZON_IDX[h]
        p, l = preds[:, hi], labels[:, hi]
        abs_p = np.abs(p)
        conf_ic[h] = {}
        for pct_label, pct in [("top_10pct", 90), ("top_5pct", 95), ("top_1pct", 99), ("top_0.5pct", 99.5)]:
            thresh = np.percentile(abs_p, pct)
            mask = abs_p >= thresh
            n_sel = mask.sum()
            if n_sel < 10:
                conf_ic[h][pct_label] = {"ic": np.nan, "n": int(n_sel)}
                continue
            conf_ic[h][pct_label] = {
                "ic": float(pearson_ic(p[mask], l[mask])),
                "rank_ic": float(rank_ic(p[mask], l[mask])),
                "hit_rate": float(hit_rate(p[mask], l[mask])),
                "n": int(n_sel),
                "mean_abs_pred": float(np.mean(abs_p[mask])),
                "mean_abs_label": float(np.mean(np.abs(l[mask]))),
            }
    result["confidence_conditional_ic"] = conf_ic

    # ---- 3. Magnitude IC: |pred| vs |label| ----
    mag_ic = {}
    for h in HORIZONS:
        hi = HORIZON_IDX[h]
        abs_p = np.abs(preds[:, hi])
        abs_l = np.abs(labels[:, hi])
        mag_ic[h] = {
            "magnitude_pearson_ic": float(pearson_ic(abs_p, abs_l)),
            "magnitude_rank_ic": float(rank_ic(abs_p, abs_l)),
        }
    result["magnitude_ic"] = mag_ic

    # ---- 4. Per-decile hit rate ----
    decile_data = {}
    for h in HORIZONS:
        hi = HORIZON_IDX[h]
        p, l = preds[:, hi], labels[:, hi]
        deciles = np.percentile(p, np.arange(10, 100, 10))
        bin_idx = np.digitize(p, deciles)  # 0-9
        dec_results = []
        for d in range(10):
            mask = bin_idx == d
            n_in = mask.sum()
            if n_in < 5:
                dec_results.append({"decile": d, "n": int(n_in), "hit_rate": np.nan,
                                    "mean_pred": np.nan, "mean_label": np.nan})
                continue
            hr = hit_rate(p[mask], l[mask])
            dec_results.append({
                "decile": d,
                "n": int(n_in),
                "hit_rate": float(hr) if not np.isnan(hr) else None,
                "mean_pred": float(np.mean(p[mask])),
                "mean_label": float(np.mean(l[mask])),
                "mean_abs_label": float(np.mean(np.abs(l[mask]))),
            })
        decile_data[h] = dec_results
    result["per_decile_hit_rate"] = decile_data

    # ---- 5. Long vs Short ----
    long_short = {}
    for h in HORIZONS:
        hi = HORIZON_IDX[h]
        p, l = preds[:, hi], labels[:, hi]
        long_mask = p > 0
        short_mask = p < 0
        neutral_mask = p == 0
        long_short[h] = {
            "pct_long": float(long_mask.mean()),
            "pct_short": float(short_mask.mean()),
            "pct_neutral": float(neutral_mask.mean()),
            "long_ic": float(pearson_ic(p[long_mask], l[long_mask])) if long_mask.sum() > 10 else None,
            "short_ic": float(pearson_ic(p[short_mask], l[short_mask])) if short_mask.sum() > 10 else None,
            "long_hit_rate": float(hit_rate(p[long_mask], l[long_mask])) if long_mask.sum() > 10 else None,
            "short_hit_rate": float(hit_rate(p[short_mask], l[short_mask])) if short_mask.sum() > 10 else None,
            "long_mean_pnl": float(np.mean(l[long_mask])) if long_mask.sum() > 0 else None,
            "short_mean_pnl": float(np.mean(-l[short_mask])) if short_mask.sum() > 0 else None,
        }
    result["long_vs_short"] = long_short

    # ---- 6. Signal autocorrelation ----
    autocorr = {}
    for h in HORIZONS:
        hi = HORIZON_IDX[h]
        p = preds[:, hi]
        ac = {}
        for lag in [1, 5, 10, 50, 100]:
            if len(p) > lag + 10:
                ac[f"lag_{lag}"] = float(np.corrcoef(p[:-lag], p[lag:])[0, 1])
            else:
                ac[f"lag_{lag}"] = None
        # Sign persistence: fraction where sign(t) == sign(t-1)
        signs = np.sign(p)
        sign_persist = float(np.mean(signs[1:] == signs[:-1]))
        ac["sign_persistence"] = sign_persist
        autocorr[h] = ac
    result["signal_autocorrelation"] = autocorr

    # ---- 7. Per-date IC (this fold = 1 date, but store for aggregation) ----
    result["per_date_ic"] = {
        "date": date_str,
        "n_events": n_events,
        "ic": {h: float(pearson_ic(preds[:, HORIZON_IDX[h]], labels[:, HORIZON_IDX[h]])) for h in HORIZONS},
        "rank_ic": {h: float(rank_ic(preds[:, HORIZON_IDX[h]], labels[:, HORIZON_IDX[h]])) for h in HORIZONS},
    }

    # ---- 8. Time-of-day analysis ----
    tod_hours = derive_tod_hours(n_events)

    # Also try to use exec feature tod_sin/tod_cos if available
    tod_sin = get_exec_feature_for_events(exec_data, n_events, "tod_sin")
    tod_cos = get_exec_feature_for_events(exec_data, n_events, "tod_cos")
    minutes_from_open = get_exec_feature_for_events(exec_data, n_events, "minutes_from_open")
    session_progress = get_exec_feature_for_events(exec_data, n_events, "session_progress")

    # If we have minutes_from_open, use it for more accurate TOD
    if minutes_from_open is not None:
        # minutes_from_open is normalized [0,1] based on exec windows
        # Convert back: session_progress * 390 minutes + 9:30
        if session_progress is not None:
            tod_hours = 9.5 + session_progress * 6.5  # 6.5 hours = 390 min

    tod_data = {}
    for label, h_start, h_end in HOUR_BINS:
        mask = (tod_hours >= h_start) & (tod_hours < h_end)
        n_in = mask.sum()
        if n_in < 10:
            tod_data[label] = {"n": int(n_in), "ic": {}, "hit_rate": {}, "mean_abs_pred": {}}
            continue
        bin_result = {"n": int(n_in), "ic": {}, "hit_rate": {}, "mean_abs_pred": {}, "mean_abs_label": {}}
        for h in HORIZONS:
            hi = HORIZON_IDX[h]
            p, l = preds[mask, hi], labels[mask, hi]
            bin_result["ic"][h] = float(pearson_ic(p, l))
            bin_result["hit_rate"][h] = float(hit_rate(p, l))
            bin_result["mean_abs_pred"][h] = float(np.mean(np.abs(p)))
            bin_result["mean_abs_label"][h] = float(np.mean(np.abs(l)))
        tod_data[label] = bin_result
    result["time_of_day"] = tod_data

    # ---- 9. Volatility regime ----
    # Use realized label volatility (rolling window of |label_10s|) as vol proxy
    # since price_volatility_window from exec features is all zeros for these dates
    vol_data = {}
    label_10s = labels[:, HORIZON_IDX["10s"]]
    # Rolling realized vol: use 500-event window of |label|
    window = 500
    if n_events >= window * 2:
        abs_labels_padded = np.abs(label_10s)
        # Compute rolling mean of |label| as realized vol proxy
        cumsum = np.cumsum(np.insert(abs_labels_padded, 0, 0))
        rolling_vol = (cumsum[window:] - cumsum[:-window]) / window
        # Align: rolling_vol[i] corresponds to event i+window//2
        offset = window // 2
        # Trim predictions to match
        valid_start = offset
        valid_end = offset + len(rolling_vol)
        if valid_end > n_events:
            valid_end = n_events
            rolling_vol = rolling_vol[:valid_end - valid_start]
        vol_median = np.median(rolling_vol)
        high_vol = rolling_vol >= vol_median
        low_vol = ~high_vol
        for regime_label, regime_mask in [("high_vol", high_vol), ("low_vol", low_vol)]:
            n_in = regime_mask.sum()
            regime_result = {"n": int(n_in), "vol_mean": float(np.mean(rolling_vol[regime_mask])), "ic": {}, "hit_rate": {}}
            for h in HORIZONS:
                hi = HORIZON_IDX[h]
                p = preds[valid_start:valid_end, hi]
                l = labels[valid_start:valid_end, hi]
                regime_result["ic"][h] = float(pearson_ic(p[regime_mask], l[regime_mask]))
                regime_result["hit_rate"][h] = float(hit_rate(p[regime_mask], l[regime_mask]))
            vol_data[regime_label] = regime_result
        vol_data["vol_median"] = float(vol_median)
        vol_data["method"] = "rolling_abs_label_10s_window500"
    else:
        vol_data["note"] = f"too few events ({n_events}) for vol regime analysis"
    # Also use event_rate from exec features as activity proxy
    event_rate = get_exec_feature_for_events(exec_data, n_events, "event_rate")
    if event_rate is not None and np.any(event_rate > 0):
        er_median = np.median(event_rate)
        high_activity = event_rate >= er_median
        low_activity = ~high_activity
        activity_data = {}
        for act_label, act_mask in [("high_activity", high_activity), ("low_activity", low_activity)]:
            n_in = act_mask.sum()
            act_result = {"n": int(n_in), "event_rate_mean": float(np.mean(event_rate[act_mask])), "ic": {}, "hit_rate": {}}
            for h in HORIZONS:
                hi = HORIZON_IDX[h]
                p, l = preds[act_mask, hi], labels[act_mask, hi]
                act_result["ic"][h] = float(pearson_ic(p, l))
                act_result["hit_rate"][h] = float(hit_rate(p, l))
            activity_data[act_label] = act_result
        activity_data["event_rate_median"] = float(er_median)
        vol_data["activity_regime"] = activity_data
    result["volatility_regime"] = vol_data

    # ---- 10. Coverage (events passing z-score thresholds) ----
    coverage = {}
    for h in HORIZONS:
        hi = HORIZON_IDX[h]
        p = preds[:, hi]
        p_std = np.std(p)
        p_mean = np.mean(p)
        if p_std < 1e-12:
            coverage[h] = {"note": "zero variance predictions"}
            continue
        z = (p - p_mean) / p_std
        abs_z = np.abs(z)
        cov = {}
        for zt in [1.0, 1.5, 2.0, 2.5]:
            mask = abs_z >= zt
            n_pass = mask.sum()
            cov[f"z>={zt}"] = {
                "n_events": int(n_pass),
                "pct": float(n_pass / n_events * 100),
                "ic": float(pearson_ic(p[mask], labels[mask, hi])) if n_pass > 10 else None,
                "hit_rate": float(hit_rate(p[mask], labels[mask, hi])) if n_pass > 10 else None,
            }
        coverage[h] = cov
    result["coverage"] = coverage

    log.info(f"Fold {fold_idx} ({date_str}): {n_events} events, IC_10s={ic_data['10s']['pearson_ic']:.4f}")
    return result


# ============================================================
# Aggregation
# ============================================================

def aggregate_results(fold_results):
    """Aggregate per-fold results into a comprehensive report."""
    report = {
        "model": "CNN-Mamba v2 (smart_v3_mar)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "commission_ticks_rt": COMMISSION_TICKS,
        "n_folds": len(fold_results),
        "total_events": sum(r["n_events"] for r in fold_results),
    }

    # Concatenate all predictions and labels for aggregate metrics
    all_preds = {h: [] for h in HORIZONS}
    all_labels = {h: [] for h in HORIZONS}
    all_tod = {h: [] for h in HORIZONS}

    for r in fold_results:
        fold_path = PRED_DIR / f"fold_{r['fold']:02d}_oot_predictions.npz"
        f = np.load(fold_path, allow_pickle=True)
        n = f["predictions"].shape[0]
        tod = derive_tod_hours(n)
        for h in HORIZONS:
            hi = HORIZON_IDX[h]
            all_preds[h].append(f["predictions"][:, hi])
            all_labels[h].append(f["labels"][:, hi])
            all_tod[h].append(tod)

    for h in HORIZONS:
        all_preds[h] = np.concatenate(all_preds[h])
        all_labels[h] = np.concatenate(all_labels[h])
        all_tod[h] = np.concatenate(all_tod[h])

    # ---- 1. Overall IC (concat) ----
    overall_ic = {}
    for h in HORIZONS:
        overall_ic[h] = {
            "concat_pearson_ic": float(pearson_ic(all_preds[h], all_labels[h])),
            "concat_rank_ic": float(rank_ic(all_preds[h], all_labels[h])),
            "mean_fold_ic": float(np.nanmean([r["ic_per_horizon"][h]["pearson_ic"] for r in fold_results])),
            "std_fold_ic": float(np.nanstd([r["ic_per_horizon"][h]["pearson_ic"] for r in fold_results])),
        }
    report["overall_ic"] = overall_ic

    # ---- 2. Confidence-conditional IC (aggregate) ----
    conf_ic_agg = {}
    for h in HORIZONS:
        p, l = all_preds[h], all_labels[h]
        abs_p = np.abs(p)
        conf_ic_agg[h] = {}
        for pct_label, pct in [("top_10pct", 90), ("top_5pct", 95), ("top_1pct", 99), ("top_0.5pct", 99.5)]:
            thresh = np.percentile(abs_p, pct)
            mask = abs_p >= thresh
            n_sel = mask.sum()
            if n_sel < 10:
                conf_ic_agg[h][pct_label] = {"ic": None, "n": int(n_sel)}
                continue
            conf_ic_agg[h][pct_label] = {
                "ic": float(pearson_ic(p[mask], l[mask])),
                "rank_ic": float(rank_ic(p[mask], l[mask])),
                "hit_rate": float(hit_rate(p[mask], l[mask])),
                "n": int(n_sel),
                "mean_abs_pred": float(np.mean(abs_p[mask])),
                "mean_abs_label": float(np.mean(np.abs(l[mask]))),
                "threshold": float(thresh),
            }
    report["confidence_conditional_ic"] = conf_ic_agg

    # ---- 3. Magnitude IC (aggregate) ----
    mag_ic_agg = {}
    for h in HORIZONS:
        abs_p = np.abs(all_preds[h])
        abs_l = np.abs(all_labels[h])
        mag_ic_agg[h] = {
            "magnitude_pearson_ic": float(pearson_ic(abs_p, abs_l)),
            "magnitude_rank_ic": float(rank_ic(abs_p, abs_l)),
        }
    report["magnitude_ic"] = mag_ic_agg

    # ---- 4. Per-decile hit rate (aggregate) ----
    decile_agg = {}
    for h in HORIZONS:
        p, l = all_preds[h], all_labels[h]
        deciles = np.percentile(p, np.arange(10, 100, 10))
        bin_idx = np.digitize(p, deciles)
        dec_results = []
        for d in range(10):
            mask = bin_idx == d
            n_in = mask.sum()
            hr = hit_rate(p[mask], l[mask]) if n_in > 10 else np.nan
            dec_results.append({
                "decile": d,
                "n": int(n_in),
                "hit_rate": float(hr) if not np.isnan(hr) else None,
                "mean_pred": float(np.mean(p[mask])) if n_in > 0 else None,
                "mean_label": float(np.mean(l[mask])) if n_in > 0 else None,
                "mean_abs_label": float(np.mean(np.abs(l[mask]))) if n_in > 0 else None,
            })
        decile_agg[h] = dec_results
    report["per_decile_hit_rate"] = decile_agg

    # ---- 5. Long vs Short (aggregate) ----
    long_short_agg = {}
    for h in HORIZONS:
        p, l = all_preds[h], all_labels[h]
        long_m = p > 0
        short_m = p < 0
        long_short_agg[h] = {
            "pct_long": float(long_m.mean()),
            "pct_short": float(short_m.mean()),
            "long_ic": float(pearson_ic(p[long_m], l[long_m])) if long_m.sum() > 10 else None,
            "short_ic": float(pearson_ic(p[short_m], l[short_m])) if short_m.sum() > 10 else None,
            "long_hit_rate": float(hit_rate(p[long_m], l[long_m])) if long_m.sum() > 10 else None,
            "short_hit_rate": float(hit_rate(p[short_m], l[short_m])) if short_m.sum() > 10 else None,
            "long_mean_pnl_ticks": float(np.mean(l[long_m])) if long_m.sum() > 0 else None,
            "short_mean_pnl_ticks": float(np.mean(-l[short_m])) if short_m.sum() > 0 else None,
        }
    report["long_vs_short"] = long_short_agg

    # ---- 6. Signal autocorrelation (aggregate across all concat preds) ----
    autocorr_agg = {}
    for h in HORIZONS:
        p = all_preds[h]
        ac = {}
        for lag in [1, 5, 10, 50, 100, 500]:
            if len(p) > lag + 10:
                ac[f"lag_{lag}"] = float(np.corrcoef(p[:-lag], p[lag:])[0, 1])
        signs = np.sign(p)
        ac["sign_persistence"] = float(np.mean(signs[1:] == signs[:-1]))
        autocorr_agg[h] = ac
    report["signal_autocorrelation"] = autocorr_agg

    # ---- 7. Per-date IC breakdown ----
    per_date = []
    for r in fold_results:
        per_date.append(r["per_date_ic"])
    # Sort by date
    per_date.sort(key=lambda x: x["date"] or "")
    report["per_date_ic"] = per_date

    # Best/worst date
    for h in HORIZONS:
        ics = [(d["date"], d["ic"][h]) for d in per_date if d["ic"][h] is not None and not np.isnan(d["ic"][h])]
        if ics:
            best = max(ics, key=lambda x: x[1])
            worst = min(ics, key=lambda x: x[1])
            report[f"best_date_{h}"] = {"date": best[0], "ic": best[1]}
            report[f"worst_date_{h}"] = {"date": worst[0], "ic": worst[1]}

    # ---- 8. Time-of-day (aggregate) ----
    tod_agg = {}
    for label, h_start, h_end in HOUR_BINS:
        # Use 10s horizon TOD array (same for all horizons)
        mask = (all_tod["10s"] >= h_start) & (all_tod["10s"] < h_end)
        n_in = mask.sum()
        if n_in < 20:
            tod_agg[label] = {"n": int(n_in)}
            continue
        bin_res = {"n": int(n_in), "ic": {}, "hit_rate": {}, "mean_abs_pred": {}, "mean_abs_label": {}}
        for h in HORIZONS:
            p, l = all_preds[h][mask], all_labels[h][mask]
            bin_res["ic"][h] = float(pearson_ic(p, l))
            bin_res["hit_rate"][h] = float(hit_rate(p, l))
            bin_res["mean_abs_pred"][h] = float(np.mean(np.abs(p)))
            bin_res["mean_abs_label"][h] = float(np.mean(np.abs(l)))
        tod_agg[label] = bin_res
    report["time_of_day"] = tod_agg

    # ---- 9. Volatility regime (aggregate from fold results) ----
    vol_agg = {"per_fold": [], "method": "rolling_abs_label_10s_window500"}
    for r in fold_results:
        vr = r.get("volatility_regime", {})
        if "high_vol" in vr:
            entry = {
                "date": r["date"],
                "high_vol_ic": vr["high_vol"]["ic"],
                "low_vol_ic": vr["low_vol"]["ic"],
                "high_vol_n": vr["high_vol"]["n"],
                "low_vol_n": vr["low_vol"]["n"],
                "vol_median": vr.get("vol_median"),
            }
            if "activity_regime" in vr:
                ar = vr["activity_regime"]
                entry["high_activity_ic"] = ar.get("high_activity", {}).get("ic", {})
                entry["low_activity_ic"] = ar.get("low_activity", {}).get("ic", {})
            vol_agg["per_fold"].append(entry)
    # Average across folds
    if vol_agg["per_fold"]:
        for h in HORIZONS:
            high_ics = [v["high_vol_ic"][h] for v in vol_agg["per_fold"] if h in v.get("high_vol_ic", {}) and v["high_vol_ic"][h] is not None and not np.isnan(v["high_vol_ic"][h])]
            low_ics = [v["low_vol_ic"][h] for v in vol_agg["per_fold"] if h in v.get("low_vol_ic", {}) and v["low_vol_ic"][h] is not None and not np.isnan(v["low_vol_ic"][h])]
            vol_agg[f"mean_high_vol_ic_{h}"] = float(np.nanmean(high_ics)) if high_ics else None
            vol_agg[f"mean_low_vol_ic_{h}"] = float(np.nanmean(low_ics)) if low_ics else None
            # Activity regime
            ha_ics = [v["high_activity_ic"][h] for v in vol_agg["per_fold"] if "high_activity_ic" in v and h in v["high_activity_ic"] and v["high_activity_ic"][h] is not None and not np.isnan(v["high_activity_ic"][h])]
            la_ics = [v["low_activity_ic"][h] for v in vol_agg["per_fold"] if "low_activity_ic" in v and h in v["low_activity_ic"] and v["low_activity_ic"][h] is not None and not np.isnan(v["low_activity_ic"][h])]
            vol_agg[f"mean_high_activity_ic_{h}"] = float(np.nanmean(ha_ics)) if ha_ics else None
            vol_agg[f"mean_low_activity_ic_{h}"] = float(np.nanmean(la_ics)) if la_ics else None
    report["volatility_regime"] = vol_agg

    # ---- 10. Coverage (aggregate) ----
    cov_agg = {}
    total_events = report["total_events"]
    for h in HORIZONS:
        p = all_preds[h]
        l = all_labels[h]
        p_std = np.std(p)
        p_mean = np.mean(p)
        z = (p - p_mean) / p_std if p_std > 1e-12 else np.zeros_like(p)
        abs_z = np.abs(z)
        n_dates = len(fold_results)
        cov = {}
        for zt in [1.0, 1.5, 2.0, 2.5]:
            mask = abs_z >= zt
            n_pass = mask.sum()
            cov[f"z>={zt}"] = {
                "total_events": int(n_pass),
                "pct_of_all": float(n_pass / total_events * 100),
                "events_per_day": float(n_pass / n_dates) if n_dates > 0 else 0,
                "ic": float(pearson_ic(p[mask], l[mask])) if n_pass > 10 else None,
                "hit_rate": float(hit_rate(p[mask], l[mask])) if n_pass > 10 else None,
            }
        cov_agg[h] = cov
    report["coverage"] = cov_agg

    # ---- Profitability estimate (simple) ----
    prof = {}
    for h in HORIZONS:
        p, l = all_preds[h], all_labels[h]
        # Use z>2 as trading threshold
        abs_z = np.abs((p - np.mean(p)) / (np.std(p) + 1e-12))
        mask = abs_z >= 2.0
        if mask.sum() < 10:
            prof[h] = {"note": "too few signals at z>=2"}
            continue
        trades = l[mask] * np.sign(p[mask])  # PnL in ticks per trade
        gross_per_trade = float(np.mean(trades))
        net_per_trade = gross_per_trade - COMMISSION_TICKS
        n_trades = int(mask.sum())
        n_days = len(fold_results)
        prof[h] = {
            "n_trades_total": n_trades,
            "trades_per_day": float(n_trades / n_days),
            "gross_per_trade_ticks": round(gross_per_trade, 4),
            "net_per_trade_ticks": round(net_per_trade, 4),
            "gross_total_ticks": round(gross_per_trade * n_trades, 2),
            "net_total_ticks": round(net_per_trade * n_trades, 2),
            "win_rate": float(np.mean(trades > 0)),
            "sortino": float(np.mean(trades) / (np.std(trades[trades < 0]) + 1e-12)) if (trades < 0).sum() > 0 else None,
        }
    report["profitability_estimate_z2"] = prof

    return report


def format_human_readable(report):
    """Format report as human-readable text."""
    lines = []
    lines.append("=" * 80)
    lines.append("CNN-MAMBA v2 SIGNAL QUALITY DEEP ANALYSIS — HC #44 Checklist")
    lines.append(f"Generated: {report['timestamp']}")
    lines.append(f"Folds: {report['n_folds']}, Total events: {report['total_events']:,}")
    lines.append(f"Commission: {report['commission_ticks_rt']} ticks RT")
    lines.append("=" * 80)

    # 1. Overall IC
    lines.append("\n--- 1. OVERALL IC (Concat across all folds) ---")
    lines.append(f"{'Horizon':<10} {'Pearson IC':>12} {'Rank IC':>12} {'Mean Fold IC':>14} {'Std':>8}")
    for h in HORIZONS:
        d = report["overall_ic"][h]
        lines.append(f"{h:<10} {d['concat_pearson_ic']:>12.4f} {d['concat_rank_ic']:>12.4f} "
                      f"{d['mean_fold_ic']:>14.4f} {d['std_fold_ic']:>8.4f}")

    # 2. Confidence-conditional IC
    lines.append("\n--- 2. CONFIDENCE-CONDITIONAL IC ---")
    for h in HORIZONS:
        lines.append(f"\n  Horizon: {h}")
        lines.append(f"  {'Bucket':<12} {'IC':>8} {'Rank IC':>8} {'Hit Rate':>10} {'N':>8} {'Mean|pred|':>12} {'Mean|label|':>12}")
        for pct_label in ["top_10pct", "top_5pct", "top_1pct", "top_0.5pct"]:
            d = report["confidence_conditional_ic"][h].get(pct_label, {})
            ic = d.get("ic")
            ric = d.get("rank_ic")
            hr = d.get("hit_rate")
            n = d.get("n", 0)
            mp = d.get("mean_abs_pred")
            ml = d.get("mean_abs_label")
            lines.append(f"  {pct_label:<12} {_fmt(ic):>8} {_fmt(ric):>8} {_fmt(hr, pct=True):>10} "
                          f"{n:>8} {_fmt(mp):>12} {_fmt(ml):>12}")

    # 3. Magnitude IC
    lines.append("\n--- 3. MAGNITUDE IC (|pred| vs |label| correlation) ---")
    lines.append(f"{'Horizon':<10} {'Pearson':>10} {'Rank':>10}")
    for h in HORIZONS:
        d = report["magnitude_ic"][h]
        lines.append(f"{h:<10} {d['magnitude_pearson_ic']:>10.4f} {d['magnitude_rank_ic']:>10.4f}")

    # 4. Per-decile hit rate
    lines.append("\n--- 4. PER-DECILE HIT RATE (10s horizon) ---")
    lines.append(f"{'Decile':<8} {'N':>8} {'Hit Rate':>10} {'Mean Pred':>12} {'Mean Label':>12} {'Mean|Label|':>12}")
    for d in report["per_decile_hit_rate"]["10s"]:
        lines.append(f"{d['decile']:<8} {d['n']:>8} {_fmt(d['hit_rate'], pct=True):>10} "
                      f"{_fmt(d['mean_pred']):>12} {_fmt(d['mean_label']):>12} {_fmt(d['mean_abs_label']):>12}")

    # 5. Long vs Short
    lines.append("\n--- 5. LONG VS SHORT ---")
    lines.append(f"{'Horizon':<8} {'%Long':>8} {'%Short':>8} {'Long IC':>10} {'Short IC':>10} {'Long HR':>10} {'Short HR':>10} {'Long PnL':>10} {'Short PnL':>10}")
    for h in HORIZONS:
        d = report["long_vs_short"][h]
        lines.append(f"{h:<8} {d['pct_long']*100:>7.1f}% {d['pct_short']*100:>7.1f}% "
                      f"{_fmt(d['long_ic']):>10} {_fmt(d['short_ic']):>10} "
                      f"{_fmt(d['long_hit_rate'], pct=True):>10} {_fmt(d['short_hit_rate'], pct=True):>10} "
                      f"{_fmt(d['long_mean_pnl_ticks']):>10} {_fmt(d['short_mean_pnl_ticks']):>10}")

    # 6. Signal autocorrelation
    lines.append("\n--- 6. SIGNAL AUTOCORRELATION ---")
    for h in HORIZONS:
        ac = report["signal_autocorrelation"][h]
        lags = " | ".join(f"{k}={_fmt(v)}" for k, v in ac.items() if k != "sign_persistence")
        lines.append(f"  {h}: {lags}")
        lines.append(f"       sign_persistence={_fmt(ac.get('sign_persistence'))}")

    # 7. Per-date IC
    lines.append("\n--- 7. PER-DATE IC BREAKDOWN ---")
    lines.append(f"{'Date':<12} {'N Events':>10} {'IC_1s':>8} {'IC_5s':>8} {'IC_10s':>8}")
    for d in report["per_date_ic"]:
        lines.append(f"{d['date']:<12} {d['n_events']:>10,} {d['ic']['1s']:>8.4f} {d['ic']['5s']:>8.4f} {d['ic']['10s']:>8.4f}")
    for h in HORIZONS:
        if f"best_date_{h}" in report:
            b = report[f"best_date_{h}"]
            w = report[f"worst_date_{h}"]
            lines.append(f"  {h}: Best={b['date']} (IC={b['ic']:.4f}), Worst={w['date']} (IC={w['ic']:.4f})")

    # 8. Time-of-day
    lines.append("\n--- 8. TIME-OF-DAY ANALYSIS ---")
    lines.append(f"{'Period':<14} {'N':>8} {'IC_1s':>8} {'IC_5s':>8} {'IC_10s':>8} {'HR_10s':>8} {'|pred|_10s':>10} {'|label|_10s':>12}")
    for label in [b[0] for b in HOUR_BINS]:
        d = report["time_of_day"].get(label, {})
        n = d.get("n", 0)
        if n < 20:
            lines.append(f"{label:<14} {n:>8} {'---':>8} {'---':>8} {'---':>8}")
            continue
        lines.append(f"{label:<14} {n:>8} {_fmt(d['ic'].get('1s')):>8} {_fmt(d['ic'].get('5s')):>8} "
                      f"{_fmt(d['ic'].get('10s')):>8} {_fmt(d['hit_rate'].get('10s'), pct=True):>8} "
                      f"{_fmt(d['mean_abs_pred'].get('10s')):>10} {_fmt(d['mean_abs_label'].get('10s')):>12}")

    # 9. Volatility regime
    lines.append("\n--- 9. VOLATILITY REGIME (realized vol = rolling |label_10s|, activity = event_rate) ---")
    vr = report["volatility_regime"]
    lines.append(f"  {'Horizon':<8} {'Hi-Vol IC':>10} {'Lo-Vol IC':>10} {'Hi-Act IC':>10} {'Lo-Act IC':>10}")
    for h in HORIZONS:
        hv = vr.get(f"mean_high_vol_ic_{h}")
        lv = vr.get(f"mean_low_vol_ic_{h}")
        ha = vr.get(f"mean_high_activity_ic_{h}")
        la = vr.get(f"mean_low_activity_ic_{h}")
        lines.append(f"  {h:<8} {_fmt(hv):>10} {_fmt(lv):>10} {_fmt(ha):>10} {_fmt(la):>10}")
    if vr.get("per_fold"):
        lines.append(f"  Per-fold breakdown:")
        for pf in vr["per_fold"]:
            hi_act = _fmt(pf.get("high_activity_ic", {}).get("10s"))
            lo_act = _fmt(pf.get("low_activity_ic", {}).get("10s"))
            lines.append(f"    {pf['date']}: Hi-vol IC_10s={_fmt(pf['high_vol_ic'].get('10s'))}, "
                          f"Lo-vol IC_10s={_fmt(pf['low_vol_ic'].get('10s'))}, "
                          f"Hi-act={hi_act}, Lo-act={lo_act}, "
                          f"vol_median={_fmt(pf.get('vol_median'))}")

    # 10. Coverage
    lines.append("\n--- 10. COVERAGE (events passing z-score thresholds) ---")
    for h in HORIZONS:
        lines.append(f"\n  Horizon: {h}")
        lines.append(f"  {'Threshold':<10} {'N Events':>10} {'%':>8} {'Per Day':>10} {'IC':>8} {'Hit Rate':>10}")
        for zt in ["z>=1.0", "z>=1.5", "z>=2.0", "z>=2.5"]:
            d = report["coverage"][h].get(zt, {})
            lines.append(f"  {zt:<10} {d.get('total_events', 0):>10} {d.get('pct_of_all', 0):>7.1f}% "
                          f"{d.get('events_per_day', 0):>10.0f} {_fmt(d.get('ic')):>8} {_fmt(d.get('hit_rate'), pct=True):>10}")

    # Profitability
    lines.append("\n--- PROFITABILITY ESTIMATE (z>=2 threshold) ---")
    lines.append(f"{'Horizon':<8} {'Trades':>8} {'/Day':>8} {'Gross/Trade':>12} {'Net/Trade':>12} {'Win Rate':>10} {'Sortino':>10}")
    for h in HORIZONS:
        d = report.get("profitability_estimate_z2", {}).get(h, {})
        if "note" in d:
            lines.append(f"{h:<8} {d['note']}")
            continue
        lines.append(f"{h:<8} {d['n_trades_total']:>8} {d['trades_per_day']:>8.0f} "
                      f"{d['gross_per_trade_ticks']:>12.4f} {d['net_per_trade_ticks']:>12.4f} "
                      f"{d['win_rate']*100:>9.1f}% {_fmt(d.get('sortino')):>10}")

    lines.append("\n" + "=" * 80)
    return "\n".join(lines)


def _fmt(val, pct=False):
    """Format a value for display."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "---"
    if pct:
        return f"{val*100:.1f}%"
    return f"{val:.4f}"


# ============================================================
# Main
# ============================================================

def main():
    log.info("Starting CNN-Mamba Signal Quality Deep Analysis")
    log.info(f"Prediction dir: {PRED_DIR}")
    log.info(f"Exec features dir: {EXEC_DIR}")

    # Find all folds
    fold_files = sorted(PRED_DIR.glob("fold_*_oot_predictions.npz"))
    fold_indices = []
    for f in fold_files:
        m = re.search(r"fold_(\d+)_", f.name)
        if m:
            fold_indices.append(int(m.group(1)))
    log.info(f"Found {len(fold_indices)} folds: {fold_indices}")

    # Parallel per-fold analysis
    fold_results = []
    with ProcessPoolExecutor(max_workers=min(NUM_WORKERS, len(fold_indices))) as executor:
        futures = {executor.submit(analyze_fold, fi): fi for fi in fold_indices}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                fold_results.append(result)

    log.info(f"Successfully analyzed {len(fold_results)} folds")

    if not fold_results:
        log.error("No valid fold results!")
        return

    # Sort by fold index
    fold_results.sort(key=lambda r: r["fold"])

    # Aggregate
    report = aggregate_results(fold_results)
    report["per_fold_results"] = fold_results

    # Write JSON report
    json_path = OUT_DIR / "signal_quality_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info(f"JSON report written to {json_path}")

    # Write human-readable report
    human_text = format_human_readable(report)
    text_path = OUT_DIR / "signal_quality_report.txt"
    with open(text_path, "w") as f:
        f.write(human_text)
    log.info(f"Human-readable report written to {text_path}")

    # Print summary to log
    print("\n" + human_text)

    log.info("Analysis complete!")


if __name__ == "__main__":
    main()
