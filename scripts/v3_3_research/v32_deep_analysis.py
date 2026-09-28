"""
v3.2 Deep Analysis — full HC #307 + HC #306 deliverable.

Ingests a `fold_NN_oot_predictions.npz` produced by v32_run_oot_inference.py
(or by the trainer's fold-end dump) and produces every metric the user asked for:

  1. Confidence-band table (DA + IC + MagCorr + Sharpe per Top0.1/0.5/1/5/10/20%
     × Bottom mirrors × horizon, long/short split) per HC #306
  2. MFE / MAE band tables — predicted-confidence vs realized path extent
  3. Price-path-from-predictions — average forward path of top-X% signals
     (uses target_log_ret_* heads as proxy when raw price unavailable)
  4. Rolling-average 1s pred exit confluence — K in {1,3,5,10,20,50} smoothing
     windows over the 1s head, DA + IC on each horizon, find optimal K
  5. Timing alignment — does pred_mfe_30s sharpness vary with WHEN the
     realized max occurred? Buckets pred_time_to_mfe_secs vs realized
  6. Multi-head agreement / confluence — DA when N of {1s,5s,10s,30s} agree
  7. Quantile calibration — empirical CDF vs predicted q10/q50/q90
  8. Reversal head value — when p_reversal_Ns > 0.5, hit rate of actual reversal
  9. Prediction distribution stats — histograms, mean/median/std/skew/kurt
  10. Aggregate IC + horizon summary (legacy compatibility)

Output structure (under --output-dir):
    confidence_bands.csv / .json              (item 1)
    mfe_mae_bands.csv                         (item 2)
    price_path_from_preds.csv                 (item 3)
    rolling_avg_exit_confluence.csv           (item 4)
    timing_alignment.csv                      (item 5)
    multi_head_agreement.csv                  (item 6)
    quantile_calibration.csv                  (item 7)
    reversal_head_value.csv                   (item 8)
    pred_distributions.json                   (item 9)
    aggregate_ic.json                         (item 10)
    morning_briefing.md                       (synthesis report)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sstats


# ============================================================
# Constants
# ============================================================
DIR_REG_HEADS = ["log_ret_1s", "log_ret_5s", "log_ret_10s",
                 "log_ret_30s", "log_ret_60s", "log_ret_5min"]
P_UP_HEADS = ["p_up_5s", "p_up_10s", "p_up_30s", "p_up_60s"]
QUANTILE_HEADS = [
    "log_ret_10s_q10", "log_ret_10s_q50", "log_ret_10s_q90",
    "log_ret_30s_q10", "log_ret_30s_q50", "log_ret_30s_q90",
    "log_ret_60s_q10", "log_ret_60s_q50", "log_ret_60s_q90",
]
PATH_HEADS = ["pred_mfe_30s_ticks", "pred_mae_30s_ticks",
              "pred_mfe_60s_ticks", "pred_mae_60s_ticks"]
REVERSAL_HEADS = ["p_reversal_15s", "p_reversal_30s", "p_reversal_60s"]

CONF_BANDS = [
    ("Top 0.1%", 0.999, 1.0001),
    ("Top 0.5%", 0.995, 1.0001),
    ("Top 1%",   0.99,  1.0001),
    ("Top 5%",   0.95,  1.0001),
    ("Top 10%",  0.90,  1.0001),
    ("Top 20%",  0.80,  1.0001),
    ("All",      0.0,   1.0001),
    ("Bottom 20%", -0.0001, 0.20),
    ("Bottom 10%", -0.0001, 0.10),
    ("Bottom 5%",  -0.0001, 0.05),
    ("Bottom 1%",  -0.0001, 0.01),
    ("Bottom 0.5%",-0.0001, 0.005),
    ("Bottom 0.1%",-0.0001, 0.001),
]


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 5:
        return float("nan")
    if np.allclose(x.std(), 0) or np.allclose(y.std(), 0):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def directional_accuracy(pred: np.ndarray, real: np.ndarray) -> Tuple[float, float, float]:
    """Returns (DA_all, DA_long, DA_short). DA_long uses pred>0 subset."""
    if pred.size == 0:
        return float("nan"), float("nan"), float("nan")
    nz = (pred != 0) & (real != 0)
    if nz.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    da_all = float((np.sign(pred[nz]) == np.sign(real[nz])).mean())
    long_mask = (pred > 0)
    short_mask = (pred < 0)
    da_long = float((real[long_mask] > 0).mean()) if long_mask.sum() > 0 else float("nan")
    da_short = float((real[short_mask] < 0).mean()) if short_mask.sum() > 0 else float("nan")
    return da_all, da_long, da_short


def compute_band_subset(pred: np.ndarray, conf_score: np.ndarray, band: Tuple[str, float, float]) -> np.ndarray:
    """Returns boolean mask of which rows fall in this confidence band.

    `conf_score` is a 0..1 percentile rank (per-row) of |pred|. Top bands
    take HIGH percentile rows, Bottom bands take LOW percentile rows."""
    name, lo, hi = band
    return (conf_score >= lo) & (conf_score <= hi)


def percentile_rank(x: np.ndarray) -> np.ndarray:
    n = x.size
    if n == 0:
        return x
    return (sstats.rankdata(x, method="average") - 1.0) / max(n - 1, 1)


# ============================================================
# Analysis modules
# ============================================================
def analyze_confidence_bands(
    pred: np.ndarray, real: np.ndarray, mask: np.ndarray, head_name: str
) -> pd.DataFrame:
    """HC #306 confidence-band table for one head."""
    valid = mask > 0
    p, r = pred[valid], real[valid]
    if p.size < 50:
        return pd.DataFrame()

    abs_p = np.abs(p)
    conf = percentile_rank(abs_p)
    rows = []
    for band in CONF_BANDS:
        m = compute_band_subset(p, conf, band)
        n_band = int(m.sum())
        if n_band < 5:
            rows.append({"head": head_name, "band": band[0], "n": n_band})
            continue
        p_b, r_b = p[m], r[m]
        da_all, da_long, da_short = directional_accuracy(p_b, r_b)
        ic = safe_corr(p_b, r_b)
        mag_corr = safe_corr(np.abs(p_b), np.abs(r_b))
        avg_pred = float(p_b.mean())
        avg_real = float(r_b.mean())
        # Toy Sharpe: per-trade pnl = sign(pred) * real
        pnl = np.sign(p_b) * r_b
        sharpe = float(pnl.mean() / (pnl.std() + 1e-12) * np.sqrt(len(pnl))) if pnl.size > 1 else float("nan")
        rows.append({
            "head": head_name, "band": band[0], "n": n_band,
            "DA_all": da_all, "DA_long": da_long, "DA_short": da_short,
            "IC": ic, "MagCorr": mag_corr,
            "avg_pred": avg_pred, "avg_real": avg_real,
            "sharpe_toy": sharpe,
        })
    return pd.DataFrame(rows)


def analyze_mfe_mae_bands(data: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Predicted MFE / MAE vs realized for high-confidence directional signals."""
    rows = []
    for dir_head in ["log_ret_30s", "log_ret_60s"]:
        if f"pred_{dir_head}" not in data:
            continue
        pred = data[f"pred_{dir_head}"]
        mask = data.get(f"mask_{dir_head}", np.ones_like(pred))
        valid = mask > 0
        p = pred[valid]
        if p.size < 50:
            continue
        conf = percentile_rank(np.abs(p))

        for mfe_head, mae_head, horizon in [
            ("pred_mfe_30s_ticks", "pred_mae_30s_ticks", "30s"),
            ("pred_mfe_60s_ticks", "pred_mae_60s_ticks", "60s"),
        ]:
            if f"pred_{mfe_head}" not in data:
                continue
            pred_mfe = data[f"pred_{mfe_head}"][valid]
            real_mfe = data[f"target_{mfe_head}"][valid]
            pred_mae = data[f"pred_{mae_head}"][valid]
            real_mae = data[f"target_{mae_head}"][valid]

            for band in CONF_BANDS:
                m = compute_band_subset(p, conf, band)
                if m.sum() < 5:
                    continue
                rows.append({
                    "entry_head": dir_head, "path_horizon": horizon, "band": band[0],
                    "n": int(m.sum()),
                    "avg_pred_mfe": float(pred_mfe[m].mean()),
                    "avg_real_mfe": float(real_mfe[m].mean()),
                    "corr_mfe": safe_corr(pred_mfe[m], real_mfe[m]),
                    "avg_pred_mae": float(pred_mae[m].mean()),
                    "avg_real_mae": float(real_mae[m].mean()),
                    "corr_mae": safe_corr(pred_mae[m], real_mae[m]),
                    "mfe_calibration_ratio": float(pred_mfe[m].mean() / max(real_mfe[m].mean(), 1e-6)),
                    "mae_calibration_ratio": float(pred_mae[m].mean() / max(real_mae[m].mean(), 1e-6)),
                })
    return pd.DataFrame(rows)


def analyze_price_path(data: Dict[str, np.ndarray]) -> pd.DataFrame:
    """For top-X% signals of each entry head, compute average realized forward
    return at each horizon (price-path from predictions)."""
    rows = []
    target_horizons = {"1s": "log_ret_1s", "5s": "log_ret_5s", "10s": "log_ret_10s",
                       "30s": "log_ret_30s", "60s": "log_ret_60s", "5min": "log_ret_5min"}

    for entry_head in DIR_REG_HEADS:
        if f"pred_{entry_head}" not in data:
            continue
        pred = data[f"pred_{entry_head}"]
        mask = data.get(f"mask_{entry_head}", np.ones_like(pred))
        valid = mask > 0
        p = pred[valid]
        if p.size < 100:
            continue
        conf = percentile_rank(np.abs(p))
        sign_p = np.sign(p)

        for band in [("Top 0.1%", 0.999), ("Top 0.5%", 0.995), ("Top 1%", 0.99),
                     ("Top 5%", 0.95), ("Top 10%", 0.9), ("Top 20%", 0.8), ("All", 0.0)]:
            band_name, lo = band
            m = conf >= lo
            if m.sum() < 5:
                continue
            n = int(m.sum())
            n_long = int(((sign_p > 0) & m).sum())
            n_short = int(((sign_p < 0) & m).sum())
            row = {"entry_head": entry_head, "band": band_name, "n": n,
                   "n_long": n_long, "n_short": n_short}
            for hz, tgt_head in target_horizons.items():
                tgt_key = f"target_{tgt_head}"
                if tgt_key not in data:
                    continue
                tgt = data[tgt_key][valid]
                # Signed forward return in trade direction
                signed_ret = sign_p[m] * tgt[m]
                row[f"avg_signed_ret_{hz}"] = float(signed_ret.mean())
                row[f"median_signed_ret_{hz}"] = float(np.median(signed_ret))
                # Long/short separately
                long_mask = (sign_p[m] > 0)
                short_mask = (sign_p[m] < 0)
                if long_mask.sum() > 0:
                    row[f"long_avg_ret_{hz}"] = float(tgt[m][long_mask].mean())
                if short_mask.sum() > 0:
                    row[f"short_avg_ret_{hz}"] = float(-tgt[m][short_mask].mean())  # short pnl = -ret
            rows.append(row)
    return pd.DataFrame(rows)


def analyze_rolling_avg_exit_confluence(data: Dict[str, np.ndarray]) -> pd.DataFrame:
    """USER IDEA (HC #307): rolling-avg 1s preds to find exit confluence.

    For K in {1,3,5,10,20,50} eval steps (each step = 1 stride = 250ms ≈),
    smooth the 1s pred series, compute:
      - DA + IC of smoothed pred vs realized 1s ret
      - DA + IC of smoothed pred vs realized 5s/10s/30s ret (forward windows)
      - Sign-flip rate (exit signal)
      - When smoothed flips sign, what's the realized 5s/10s ret in OLD direction?
    """
    rows = []
    if "pred_log_ret_1s" not in data:
        return pd.DataFrame()
    pred_1s = data["pred_log_ret_1s"]
    mask_1s = data.get("mask_log_ret_1s", np.ones_like(pred_1s))
    valid = mask_1s > 0
    p = pred_1s[valid]
    targets = {hz: data.get(f"target_log_ret_{hz}", None) for hz in ["1s", "5s", "10s", "30s"]}
    targets = {hz: (t[valid] if t is not None else None) for hz, t in targets.items()}

    for k in [1, 3, 5, 10, 20, 50]:
        if k > p.size // 4:
            continue
        # Rolling mean (causal, trailing window)
        kernel = np.ones(k) / k
        smoothed = np.convolve(p, kernel, mode="same")
        # Trim edges to avoid bias
        edge = max(k // 2, 1)
        s = smoothed[edge:-edge] if edge > 0 else smoothed
        for hz, t in targets.items():
            if t is None:
                continue
            tt = t[edge:-edge] if edge > 0 else t
            if s.size < 50:
                continue
            da_all, da_long, da_short = directional_accuracy(s, tt)
            ic = safe_corr(s, tt)
            row = {"K_smooth": k, "approx_smooth_ms": k * 250, "target_horizon": hz,
                   "n": int(s.size), "DA_all": da_all, "DA_long": da_long,
                   "DA_short": da_short, "IC": ic}
            # Sign flip rate of smoothed series (exit signal density)
            sign_flips = int(np.sum(np.diff(np.sign(s)) != 0))
            row["sign_flip_rate"] = sign_flips / max(s.size - 1, 1)
            rows.append(row)
    return pd.DataFrame(rows)


def analyze_timing_alignment(data: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Does pred_mfe_30s sharpness vary with WHEN max occurred (time_to_mfe)?

    Buckets predictions by pred_time_to_mfe_secs and reports MFE prediction
    accuracy in each bucket. Tells us if the model knows WHEN to exit."""
    rows = []
    if "pred_pred_time_to_mfe_secs" not in data or "pred_pred_mfe_30s_ticks" not in data:
        return pd.DataFrame()
    pred_time = data["pred_pred_time_to_mfe_secs"]
    pred_mfe = data["pred_pred_mfe_30s_ticks"]
    real_mfe = data["target_pred_mfe_30s_ticks"]
    mask = data.get("mask_pred_mfe_30s_ticks", np.ones_like(pred_mfe))
    valid = mask > 0
    pt = pred_time[valid]
    pm = pred_mfe[valid]
    rm = real_mfe[valid]

    # Bucket by predicted time-to-mfe
    bins = [0, 5, 10, 15, 20, 25, 30, 1e6]
    labels = ["0-5s", "5-10s", "10-15s", "15-20s", "20-25s", "25-30s", ">30s"]
    bidx = np.digitize(pt, bins) - 1
    for i, label in enumerate(labels):
        m = (bidx == i)
        if m.sum() < 20:
            continue
        rows.append({
            "time_bucket": label, "n": int(m.sum()),
            "avg_pred_time": float(pt[m].mean()),
            "avg_pred_mfe": float(pm[m].mean()),
            "avg_real_mfe": float(rm[m].mean()),
            "corr_mfe": safe_corr(pm[m], rm[m]),
            "calibration_ratio": float(pm[m].mean() / max(rm[m].mean(), 1e-6)),
        })
    return pd.DataFrame(rows)


def analyze_multi_head_agreement(data: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Confluence: when N heads agree on sign, what's DA + IC of the trade?

    Uses log_ret_1s/5s/10s/30s heads."""
    rows = []
    heads = ["log_ret_1s", "log_ret_5s", "log_ret_10s", "log_ret_30s"]
    pred_stack = []
    mask_stack = []
    for h in heads:
        if f"pred_{h}" not in data:
            return pd.DataFrame()
        pred_stack.append(np.sign(data[f"pred_{h}"]))
        mask_stack.append(data.get(f"mask_{h}", np.ones_like(data[f"pred_{h}"])))
    P = np.stack(pred_stack, axis=1)  # (N, 4)
    M = np.stack(mask_stack, axis=1)
    valid = (M.min(axis=1) > 0)
    P = P[valid]
    if P.shape[0] < 100:
        return pd.DataFrame()

    # For each target horizon, compute DA stratified by agreement count
    for tgt_hz in heads:
        if f"target_{tgt_hz}" not in data:
            continue
        tgt = data[f"target_{tgt_hz}"]
        tgt_v = tgt[valid] if valid.size == tgt.size else tgt
        # Sign of majority direction
        pos_count = (P > 0).sum(axis=1)
        neg_count = (P < 0).sum(axis=1)
        majority_sign = np.where(pos_count > neg_count, 1, np.where(neg_count > pos_count, -1, 0))
        agreement_count = np.maximum(pos_count, neg_count)  # 1..4

        for ac in [1, 2, 3, 4]:
            m = (agreement_count == ac) & (majority_sign != 0)
            if m.sum() < 20:
                continue
            ms = majority_sign[m]
            tt = tgt_v[m]
            da_all = float((ms == np.sign(tt)).mean())
            ic = safe_corr(ms.astype(float), tt)
            avg_ret = float((ms * tt).mean())
            rows.append({
                "target_horizon": tgt_hz, "agreement_count": ac, "n": int(m.sum()),
                "DA": da_all, "IC": ic, "avg_signed_ret": avg_ret,
            })
    return pd.DataFrame(rows)


def analyze_quantile_calibration(data: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Empirical CDF check: when pred q90=X, is realized return ≤X 90% of time?"""
    rows = []
    for horizon in ["10s", "30s", "60s"]:
        q10_k = f"pred_log_ret_{horizon}_q10"
        q50_k = f"pred_log_ret_{horizon}_q50"
        q90_k = f"pred_log_ret_{horizon}_q90"
        tgt_k = f"target_log_ret_{horizon}_q50"  # same target across q10/q50/q90
        if q10_k not in data or q50_k not in data or q90_k not in data:
            continue
        if tgt_k not in data:
            tgt_k = f"target_log_ret_{horizon}"
            if tgt_k not in data:
                continue
        mask_k = f"mask_log_ret_{horizon}_q50"
        if mask_k not in data:
            mask_k = f"mask_log_ret_{horizon}"
        m = data.get(mask_k, np.ones(data[q10_k].size)) > 0
        q10, q50, q90 = data[q10_k][m], data[q50_k][m], data[q90_k][m]
        real = data[tgt_k][m]
        if q10.size < 100:
            continue
        rows.append({
            "horizon": horizon, "n": int(q10.size),
            "p_real_below_q10": float((real <= q10).mean()),  # should be ~0.10
            "p_real_below_q50": float((real <= q50).mean()),  # should be ~0.50
            "p_real_below_q90": float((real <= q90).mean()),  # should be ~0.90
            "avg_q90_minus_q10": float((q90 - q10).mean()),
            "avg_realized": float(real.mean()),
            "median_q50": float(np.median(q50)),
            "median_realized": float(np.median(real)),
        })
    return pd.DataFrame(rows)


def analyze_reversal_head(data: Dict[str, np.ndarray]) -> pd.DataFrame:
    """When p_reversal_Ns > threshold, does the signal actually reverse?"""
    rows = []
    for rev_head, horizon_secs in [("p_reversal_15s", 15), ("p_reversal_30s", 30), ("p_reversal_60s", 60)]:
        if f"pred_{rev_head}" not in data:
            continue
        p = data[f"pred_{rev_head}"]
        # Sigmoid if logits, else identity
        if (p < 0).any() or p.max() > 1:
            p = 1 / (1 + np.exp(-p))
        t = data.get(f"target_{rev_head}", None)
        mask = data.get(f"mask_{rev_head}", np.ones_like(p)) > 0
        if t is None:
            continue
        p, t = p[mask], t[mask]
        for thr in [0.3, 0.4, 0.5, 0.6, 0.7]:
            m = p > thr
            if m.sum() < 20:
                continue
            hit_rate = float(t[m].mean())  # fraction that actually reversed
            rows.append({
                "head": rev_head, "horizon_secs": horizon_secs, "threshold": thr,
                "n": int(m.sum()), "hit_rate": hit_rate,
                "base_rate": float(t.mean()),
                "lift": hit_rate - float(t.mean()),
            })
    return pd.DataFrame(rows)


def analyze_pred_distributions(data: Dict[str, np.ndarray]) -> Dict:
    """Distribution stats for every head (the 'we need that data' deliverable)."""
    out = {}
    for k in sorted(data.keys()):
        if not k.startswith("pred_"):
            continue
        x = data[k]
        mask_k = k.replace("pred_", "mask_")
        if mask_k in data:
            x = x[data[mask_k] > 0]
        if x.size == 0:
            continue
        out[k] = {
            "n": int(x.size),
            "mean": float(x.mean()),
            "median": float(np.median(x)),
            "std": float(x.std()),
            "min": float(x.min()),
            "p01": float(np.percentile(x, 1)),
            "p05": float(np.percentile(x, 5)),
            "p25": float(np.percentile(x, 25)),
            "p75": float(np.percentile(x, 75)),
            "p95": float(np.percentile(x, 95)),
            "p99": float(np.percentile(x, 99)),
            "max": float(x.max()),
            "skew": float(sstats.skew(x)) if x.size > 3 else None,
            "kurt": float(sstats.kurtosis(x)) if x.size > 3 else None,
            "frac_zero": float((x == 0).mean()),
            "frac_pos": float((x > 0).mean()),
            "frac_neg": float((x < 0).mean()),
        }
        # Corresponding target distribution
        tgt_k = k.replace("pred_", "target_")
        if tgt_k in data:
            t = data[tgt_k]
            if mask_k in data:
                t = t[data[mask_k] > 0]
            if t.size > 0:
                out[k]["target_mean"] = float(t.mean())
                out[k]["target_std"] = float(t.std())
                out[k]["target_p05"] = float(np.percentile(t, 5))
                out[k]["target_p95"] = float(np.percentile(t, 95))
    return out


def analyze_aggregate_ic(data: Dict[str, np.ndarray]) -> Dict:
    """Aggregate IC + MagCorr per head for legacy compat with HC #306 (E)."""
    out = {}
    for h in DIR_REG_HEADS + PATH_HEADS + QUANTILE_HEADS:
        pk = f"pred_{h}"
        tk = f"target_{h}"
        mk = f"mask_{h}"
        if pk not in data or tk not in data:
            continue
        m = data.get(mk, np.ones(data[pk].size)) > 0
        p = data[pk][m]
        t = data[tk][m]
        if p.size < 50:
            continue
        out[h] = {
            "n": int(p.size),
            "IC": safe_corr(p, t),
            "MagCorr": safe_corr(np.abs(p), np.abs(t)),
            "DA_all": directional_accuracy(p, t)[0],
        }
    return out


def write_morning_briefing(
    output_dir: Path,
    conf_bands: pd.DataFrame,
    mfe_mae: pd.DataFrame,
    price_path: pd.DataFrame,
    rolling_avg: pd.DataFrame,
    timing: pd.DataFrame,
    agreement: pd.DataFrame,
    quantile_cal: pd.DataFrame,
    reversal: pd.DataFrame,
    pred_dist: Dict,
    agg_ic: Dict,
    meta: Dict,
) -> None:
    md = []
    md.append(f"# v3.2 Fold 0 OOT Deep Analysis — Morning Briefing")
    md.append(f"")
    md.append(f"**Generated**: {pd.Timestamp.now().isoformat()}")
    md.append(f"**Source**: `{meta.get('predictions_npz', 'unknown')}`")
    md.append(f"**Samples analyzed**: {meta.get('n_samples', 'unknown')}")
    md.append(f"**OOT dates**: {meta.get('oot_dates', 'unknown')}")
    md.append("")
    md.append("## TL;DR — bottom-line findings per HC #307 deliverable")
    md.append("")
    # Aggregate IC summary
    md.append("### 1. Aggregate IC by horizon (legacy compat)")
    md.append("")
    md.append("| Head | n | IC | MagCorr | DA |")
    md.append("|---|---|---|---|---|")
    for h, m in agg_ic.items():
        ic = m.get("IC", float("nan"))
        mc = m.get("MagCorr", float("nan"))
        da = m.get("DA_all", float("nan"))
        md.append(f"| {h} | {m.get('n', 0)} | {ic:.4f} | {mc:.4f} | {da:.4f} |")
    md.append("")

    # Top-band DA
    md.append("### 2. CONFIDENCE BANDS — DA @ Top 1% (per HC #306 — the key cells)")
    md.append("")
    md.append("| Head | n | DA_all | DA_long | DA_short | IC | MagCorr | Sharpe_toy |")
    md.append("|---|---|---|---|---|---|---|---|")
    if not conf_bands.empty:
        top1 = conf_bands[conf_bands["band"] == "Top 1%"]
        for _, r in top1.iterrows():
            md.append(
                f"| {r['head']} | {r.get('n', 0)} | {r.get('DA_all', float('nan')):.4f} | "
                f"{r.get('DA_long', float('nan')):.4f} | {r.get('DA_short', float('nan')):.4f} | "
                f"{r.get('IC', float('nan')):.4f} | {r.get('MagCorr', float('nan')):.4f} | "
                f"{r.get('sharpe_toy', float('nan')):.3f} |"
            )
    md.append("")

    # Rolling-avg exit confluence (user's idea)
    md.append("### 3. ROLLING-AVG 1s EXIT CONFLUENCE (USER IDEA — HC #307)")
    md.append("")
    md.append("Hypothesis: smoothing 1s preds via rolling mean cleans tick noise → sharper entry+exit.")
    md.append("")
    md.append("| K_smooth | smooth_ms | tgt_horizon | n | DA_all | DA_long | DA_short | IC | sign_flip_rate |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    if not rolling_avg.empty:
        for _, r in rolling_avg.iterrows():
            md.append(
                f"| {int(r['K_smooth'])} | {int(r['approx_smooth_ms'])} | {r['target_horizon']} | "
                f"{int(r['n'])} | {r.get('DA_all', float('nan')):.4f} | "
                f"{r.get('DA_long', float('nan')):.4f} | {r.get('DA_short', float('nan')):.4f} | "
                f"{r.get('IC', float('nan')):.4f} | {r.get('sign_flip_rate', float('nan')):.4f} |"
            )
    md.append("")

    # Price path
    md.append("### 4. PRICE-PATH FROM PREDICTIONS — avg signed forward return per band")
    md.append("")
    md.append("Per top-X% confidence signal of each entry head, what's the realized signed return at each forward horizon?")
    md.append("(Signed = direction-correct trade pnl in log-ret units)")
    md.append("")
    if not price_path.empty:
        cols_of_interest = ["entry_head", "band", "n", "avg_signed_ret_1s",
                            "avg_signed_ret_5s", "avg_signed_ret_10s",
                            "avg_signed_ret_30s", "avg_signed_ret_60s"]
        cols_present = [c for c in cols_of_interest if c in price_path.columns]
        md.append("| " + " | ".join(cols_present) + " |")
        md.append("|" + "|".join(["---"] * len(cols_present)) + "|")
        for _, r in price_path[price_path["band"].isin(["Top 0.1%", "Top 1%", "Top 5%", "All"])].iterrows():
            md.append("| " + " | ".join(
                f"{r[c]:.6f}" if isinstance(r[c], (int, float, np.floating)) and not isinstance(r[c], (int, np.integer)) and c.startswith("avg_") else str(r[c])
                for c in cols_present
            ) + " |")
    md.append("")

    # Multi-head agreement
    md.append("### 5. MULTI-HEAD AGREEMENT / CONFLUENCE")
    md.append("")
    md.append("When N of {1s,5s,10s,30s} heads agree on sign, what's DA on each target?")
    md.append("")
    md.append("| target_horizon | agreement_count | n | DA | IC | avg_signed_ret |")
    md.append("|---|---|---|---|---|---|")
    if not agreement.empty:
        for _, r in agreement.iterrows():
            md.append(
                f"| {r['target_horizon']} | {int(r['agreement_count'])} | {int(r['n'])} | "
                f"{r['DA']:.4f} | {r['IC']:.4f} | {r['avg_signed_ret']:.6f} |"
            )
    md.append("")

    # Quantile calibration
    md.append("### 6. QUANTILE CALIBRATION (is q90 actually the 90th percentile?)")
    md.append("")
    md.append("| horizon | n | p_real_below_q10 (want ~0.10) | p_real_below_q50 (want ~0.50) | p_real_below_q90 (want ~0.90) | avg(q90-q10) |")
    md.append("|---|---|---|---|---|---|")
    if not quantile_cal.empty:
        for _, r in quantile_cal.iterrows():
            md.append(
                f"| {r['horizon']} | {int(r['n'])} | {r['p_real_below_q10']:.4f} | "
                f"{r['p_real_below_q50']:.4f} | {r['p_real_below_q90']:.4f} | "
                f"{r['avg_q90_minus_q10']:.6f} |"
            )
    md.append("")

    # MFE/MAE bands
    md.append("### 7. MFE/MAE BAND CALIBRATION (do path heads predict path correctly?)")
    md.append("")
    md.append("(See `mfe_mae_bands.csv` for full per-band breakdown — summary below)")
    md.append("")
    if not mfe_mae.empty:
        top1 = mfe_mae[mfe_mae["band"] == "Top 1%"]
        md.append("| entry_head | path_horizon | n | avg_pred_mfe | avg_real_mfe | corr_mfe | mfe_cal_ratio |")
        md.append("|---|---|---|---|---|---|---|")
        for _, r in top1.iterrows():
            md.append(
                f"| {r['entry_head']} | {r['path_horizon']} | {int(r['n'])} | "
                f"{r['avg_pred_mfe']:.4f} | {r['avg_real_mfe']:.4f} | "
                f"{r['corr_mfe']:.4f} | {r['mfe_calibration_ratio']:.4f} |"
            )
    md.append("")

    # Reversal
    md.append("### 8. REVERSAL HEAD VALUE (when p_reversal > thr, does it reverse?)")
    md.append("")
    md.append("| head | threshold | n | hit_rate | base_rate | lift |")
    md.append("|---|---|---|---|---|---|")
    if not reversal.empty:
        for _, r in reversal.iterrows():
            md.append(
                f"| {r['head']} | {r['threshold']:.2f} | {int(r['n'])} | "
                f"{r['hit_rate']:.4f} | {r['base_rate']:.4f} | {r['lift']:+.4f} |"
            )
    md.append("")

    # Timing alignment
    md.append("### 9. TIMING ALIGNMENT (does pred_time_to_mfe sharpen MFE accuracy?)")
    md.append("")
    md.append("| time_bucket | n | avg_pred_time | avg_pred_mfe | avg_real_mfe | corr_mfe | calibration |")
    md.append("|---|---|---|---|---|---|---|")
    if not timing.empty:
        for _, r in timing.iterrows():
            md.append(
                f"| {r['time_bucket']} | {int(r['n'])} | {r['avg_pred_time']:.2f} | "
                f"{r['avg_pred_mfe']:.4f} | {r['avg_real_mfe']:.4f} | "
                f"{r['corr_mfe']:.4f} | {r['calibration_ratio']:.4f} |"
            )
    md.append("")

    # Pred distribution summary
    md.append("### 10. PREDICTION DISTRIBUTIONS (we need this data, can't lose it)")
    md.append("")
    md.append("(Full per-head stats in `pred_distributions.json` — top heads summarized below)")
    md.append("")
    md.append("| head | n | mean | std | p05 | p95 | frac_pos | target_mean | target_std |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for k in ["pred_log_ret_1s", "pred_log_ret_5s", "pred_log_ret_10s",
              "pred_log_ret_30s", "pred_pred_mfe_30s_ticks", "pred_pred_mae_30s_ticks"]:
        if k in pred_dist:
            d = pred_dist[k]
            md.append(
                f"| {k} | {d['n']} | {d['mean']:.6f} | {d['std']:.6f} | "
                f"{d['p05']:.6f} | {d['p95']:.6f} | {d['frac_pos']:.3f} | "
                f"{d.get('target_mean', float('nan')):.6f} | {d.get('target_std', float('nan')):.6f} |"
            )
    md.append("")

    md.append("## Files written")
    md.append("")
    for f in sorted(output_dir.glob("*")):
        md.append(f"- `{f.name}` ({f.stat().st_size / 1024:.1f} KB)")
    md.append("")

    (output_dir / "morning_briefing.md").write_text("\n".join(md))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions-npz", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--meta-json", default=None, type=Path,
                    help="Optional metadata json (e.g. from inference run).")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[v32_deep_analysis] loading {args.predictions_npz}", flush=True)
    raw = np.load(args.predictions_npz, allow_pickle=True)
    data = {k: raw[k] for k in raw.files}
    print(f"[v32_deep_analysis] loaded {len(data)} arrays. keys[:10]={list(data.keys())[:10]}", flush=True)

    meta = {"predictions_npz": str(args.predictions_npz)}
    if args.meta_json and args.meta_json.exists():
        meta.update(json.loads(args.meta_json.read_text()))
    if "oot_dates" in data:
        meta["oot_dates"] = data["oot_dates"].tolist() if hasattr(data["oot_dates"], "tolist") else str(data["oot_dates"])
    if "n_samples" in data:
        meta["n_samples"] = int(data["n_samples"])

    # ---- Module 1: Confidence bands (HC #306) ----
    print("[deep] confidence bands...", flush=True)
    all_band_dfs = []
    for h in DIR_REG_HEADS + PATH_HEADS:
        pk = f"pred_{h}"
        tk = f"target_{h}"
        mk = f"mask_{h}"
        if pk not in data or tk not in data:
            continue
        df = analyze_confidence_bands(data[pk], data[tk], data.get(mk, np.ones(data[pk].size)), h)
        if not df.empty:
            all_band_dfs.append(df)
    conf_bands = pd.concat(all_band_dfs, ignore_index=True) if all_band_dfs else pd.DataFrame()
    if not conf_bands.empty:
        conf_bands.to_csv(args.output_dir / "confidence_bands.csv", index=False)
        conf_bands.to_json(args.output_dir / "confidence_bands.json", orient="records", indent=2)
    print(f"  → {len(conf_bands)} rows", flush=True)

    # ---- Module 2: MFE/MAE bands ----
    print("[deep] mfe/mae bands...", flush=True)
    mfe_mae = analyze_mfe_mae_bands(data)
    if not mfe_mae.empty:
        mfe_mae.to_csv(args.output_dir / "mfe_mae_bands.csv", index=False)
    print(f"  → {len(mfe_mae)} rows", flush=True)

    # ---- Module 3: Price path ----
    print("[deep] price path from preds...", flush=True)
    price_path = analyze_price_path(data)
    if not price_path.empty:
        price_path.to_csv(args.output_dir / "price_path_from_preds.csv", index=False)
    print(f"  → {len(price_path)} rows", flush=True)

    # ---- Module 4: Rolling avg exit confluence (USER IDEA) ----
    print("[deep] rolling-avg exit confluence (USER IDEA)...", flush=True)
    rolling_avg = analyze_rolling_avg_exit_confluence(data)
    if not rolling_avg.empty:
        rolling_avg.to_csv(args.output_dir / "rolling_avg_exit_confluence.csv", index=False)
    print(f"  → {len(rolling_avg)} rows", flush=True)

    # ---- Module 5: Timing alignment ----
    print("[deep] timing alignment...", flush=True)
    timing = analyze_timing_alignment(data)
    if not timing.empty:
        timing.to_csv(args.output_dir / "timing_alignment.csv", index=False)
    print(f"  → {len(timing)} rows", flush=True)

    # ---- Module 6: Multi-head agreement ----
    print("[deep] multi-head agreement...", flush=True)
    agreement = analyze_multi_head_agreement(data)
    if not agreement.empty:
        agreement.to_csv(args.output_dir / "multi_head_agreement.csv", index=False)
    print(f"  → {len(agreement)} rows", flush=True)

    # ---- Module 7: Quantile calibration ----
    print("[deep] quantile calibration...", flush=True)
    quantile_cal = analyze_quantile_calibration(data)
    if not quantile_cal.empty:
        quantile_cal.to_csv(args.output_dir / "quantile_calibration.csv", index=False)
    print(f"  → {len(quantile_cal)} rows", flush=True)

    # ---- Module 8: Reversal head ----
    print("[deep] reversal head value...", flush=True)
    reversal = analyze_reversal_head(data)
    if not reversal.empty:
        reversal.to_csv(args.output_dir / "reversal_head_value.csv", index=False)
    print(f"  → {len(reversal)} rows", flush=True)

    # ---- Module 9: Pred distributions ----
    print("[deep] pred distributions...", flush=True)
    pred_dist = analyze_pred_distributions(data)
    (args.output_dir / "pred_distributions.json").write_text(json.dumps(pred_dist, indent=2))
    print(f"  → {len(pred_dist)} heads", flush=True)

    # ---- Module 10: Aggregate IC ----
    print("[deep] aggregate IC...", flush=True)
    agg_ic = analyze_aggregate_ic(data)
    (args.output_dir / "aggregate_ic.json").write_text(json.dumps(agg_ic, indent=2))
    print(f"  → {len(agg_ic)} heads", flush=True)

    # ---- Morning briefing synthesis ----
    print("[deep] writing morning_briefing.md...", flush=True)
    write_morning_briefing(
        args.output_dir, conf_bands, mfe_mae, price_path, rolling_avg,
        timing, agreement, quantile_cal, reversal, pred_dist, agg_ic, meta
    )

    print(f"[v32_deep_analysis] DONE — outputs in {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
