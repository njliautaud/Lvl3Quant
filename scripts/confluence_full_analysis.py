#!/usr/bin/env python3
"""
Full Confluence Analysis: CNN-Mamba v2 + PatchTST + Vol LGBM
=============================================================
MIDPOINT-BASED analysis (no FIFO sim). Per HC #57.
Commission: $4.70 RT = 0.376 ticks (ES, $12.50/tick).

Analyzes when CNN-Mamba v2 and PatchTST predictions AGREE (confluence)
vs disagree, and tests whether vol regime filtering helps.

Outputs:
  - Detailed metrics for each confluence configuration
  - MFE/MAE analysis at 1s, 3s, 5s, 10s post-signal
  - Time-of-day effects
  - Recommended trading configs

Usage:
    python scripts/confluence_full_analysis.py
"""

import sys
import json
import re
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# ── Paths ──
LVL3_ROOT = Path(__file__).resolve().parent.parent
CM_DIR = LVL3_ROOT / "output" / "cnn_mamba_v2_smart_v3_mar"
PT_DIR = LVL3_ROOT / "output" / "patchtst_smart_v3_mar"
VOL_DIR = LVL3_ROOT / "output" / "vol_lgbm_v3"
OUTPUT_DIR = LVL3_ROOT / "output" / "confluence_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──
TICK_VALUE = 12.50
COST_RT_TICKS = 0.376       # $4.70 / $12.50, commission only
COST_RT_USD = 4.70
HORIZONS = ["1s", "5s", "10s"]
HORIZON_IDX = {"1s": 0, "5s": 1, "10s": 2}  # index into (N,3) predictions/labels

# Labels are in ticks (midpoint price change). PnL = direction * label - cost.
# MFE/MAE: we use 1s, 5s, 10s labels as proxy for price path at those horizons.

_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
log = logging.getLogger("confluence")
log.setLevel(logging.INFO)
_fh = logging.FileHandler(str(OUTPUT_DIR / f"confluence_{_ts}.log"), mode="w")
_sh = logging.StreamHandler(sys.stdout)
for h in [_fh, _sh]:
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    log.addHandler(h)


# ── Discover folds and dates ──

def extract_date_from_oot(oot_path: str) -> str:
    """Extract YYYYMMDD from oot_files path (handles Windows and Linux paths)."""
    m = re.search(r"(\d{8})_mbo_events", oot_path)
    if m:
        return m.group(1)
    m = re.search(r"(\d{8})", Path(oot_path).name)
    return m.group(1) if m else "unknown"


def discover_folds():
    """Discover all folds for each model and map fold -> date."""
    cm_folds = {}  # date -> fold_idx
    for p in sorted(CM_DIR.glob("fold_*_oot_predictions.npz")):
        fold_idx = int(p.name.split("_")[1])
        d = np.load(p, allow_pickle=True)
        date = extract_date_from_oot(str(d["oot_files"][0]))
        cm_folds[date] = fold_idx

    pt_folds = {}
    for p in sorted(PT_DIR.glob("fold_*_oot_predictions.npz")):
        fold_idx = int(p.name.split("_")[1])
        d = np.load(p, allow_pickle=True)
        date = extract_date_from_oot(str(d["oot_files"][0]))
        pt_folds[date] = fold_idx

    vol_files = {}  # date -> filepath
    for p in sorted(VOL_DIR.glob("vol_v3_*_predictions.npz")):
        d = np.load(p, allow_pickle=True)
        date = str(d["test_date"][0])
        vol_files[date] = p

    common_dates = sorted(set(cm_folds.keys()) & set(pt_folds.keys()))
    vol_dates = sorted(set(common_dates) & set(vol_files.keys()))

    return cm_folds, pt_folds, vol_files, common_dates, vol_dates


# ── Alignment ──
# CNN-Mamba and PatchTST are sampled at different strides from the same
# underlying event stream.  Empirically verified: CM[i] aligns perfectly
# with PT[2*i + 2] for all common dates (label match = 100%).

def align_cm_pt(cm_data: dict, pt_data: dict) -> dict:
    """
    Align CNN-Mamba and PatchTST predictions.
    Returns dict with aligned arrays: cm_preds (N,3), pt_preds (N,3),
    labels (N,3), plus per-horizon z-scores.
    """
    cm_preds = cm_data["predictions"]  # (N_cm, 3)
    pt_preds = pt_data["predictions"]  # (N_pt, 3)
    cm_labels = cm_data["labels"]      # (N_cm, 3)

    n_cm = len(cm_preds)
    # PT[2*i+2] maps to CM[i]
    n_available = (len(pt_preds) - 2) // 2
    n = min(n_cm, n_available)

    pt_aligned = pt_preds[2::2][:n]
    cm_aligned = cm_preds[:n]
    labels_aligned = cm_labels[:n]

    # Z-score normalize each model's predictions (per-date, expanding would need
    # per-sample, but for OOS evaluation, per-date z-score is standard)
    result = {
        "cm_preds": cm_aligned,
        "pt_preds": pt_aligned,
        "labels": labels_aligned,
        "n_samples": n,
    }

    for h_name, h_idx in HORIZON_IDX.items():
        cm_h = cm_aligned[:, h_idx]
        pt_h = pt_aligned[:, h_idx]

        cm_std = cm_h.std()
        pt_std = pt_h.std()

        if cm_std > 1e-9:
            result[f"cm_z_{h_name}"] = (cm_h - cm_h.mean()) / cm_std
        else:
            result[f"cm_z_{h_name}"] = np.zeros_like(cm_h)

        if pt_std > 1e-9:
            result[f"pt_z_{h_name}"] = (pt_h - pt_h.mean()) / pt_std
        else:
            result[f"pt_z_{h_name}"] = np.zeros_like(pt_h)

    return result


def align_vol(aligned: dict, vol_data: dict) -> np.ndarray:
    """
    Align vol LGBM predictions with the CM/PT aligned data.
    Vol has anchor stride 500, starting at 999. CM has similar count.
    Returns vol predictions (N_aligned,) for 10s horizon, or None.
    """
    vol_preds = vol_data["predictions"][:, 0]  # 10s horizon vol prediction
    vol_labels = vol_data["labels"][:, 0]       # 10s realized vol
    n_cm = aligned["n_samples"]
    n_vol = len(vol_preds)

    # Both have ~same count for same date, slight offset possible.
    # Use closest index mapping: CM has N samples, Vol has M samples.
    # Map CM[i] -> Vol[round(i * M/N)]
    if n_vol == 0:
        return None, None

    ratio = n_vol / n_cm
    vol_idx = np.clip(np.round(np.arange(n_cm) * ratio).astype(int), 0, n_vol - 1)
    return vol_preds[vol_idx], vol_labels[vol_idx]


# ── Metrics ──

def compute_metrics(pnls: np.ndarray) -> dict:
    """Compute trading metrics from tick PnL array."""
    if len(pnls) == 0:
        return dict(n_trades=0, win_rate=0, sharpe=0, sortino=0,
                    profit_factor=0, rr_ratio=0, total_pnl_ticks=0,
                    total_pnl_usd=0, mean_win=0, mean_loss=0, avg_pnl=0)

    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    n = len(pnls)

    win_rate = len(wins) / n
    mean = pnls.mean()
    std = pnls.std(ddof=1) if n > 1 else 1e-9

    sharpe = mean / std if std > 1e-9 else 0

    downside = pnls[pnls < 0]
    down_std = np.sqrt(np.mean(downside ** 2)) if len(downside) > 0 else 1e-9
    sortino = mean / down_std if down_std > 1e-9 else 0

    gross_profit = wins.sum() if len(wins) > 0 else 0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 1e-9
    profit_factor = gross_profit / gross_loss if gross_loss > 1e-9 else float("inf")

    mean_win = wins.mean() if len(wins) > 0 else 0
    mean_loss = abs(losses.mean()) if len(losses) > 0 else 1e-9
    rr_ratio = mean_win / mean_loss if mean_loss > 1e-9 else float("inf")

    return dict(
        n_trades=n,
        win_rate=round(win_rate, 4),
        sharpe=round(sharpe, 4),
        sortino=round(sortino, 4),
        profit_factor=round(profit_factor, 4),
        rr_ratio=round(rr_ratio, 4),
        total_pnl_ticks=round(float(pnls.sum()), 2),
        total_pnl_usd=round(float(pnls.sum()) * TICK_VALUE, 2),
        mean_win=round(float(mean_win), 4),
        mean_loss=round(float(mean_loss), 4),
        avg_pnl=round(float(mean), 4),
    )


def compute_mfe_mae(labels: np.ndarray, direction: np.ndarray) -> dict:
    """
    Compute MFE and MAE at each horizon.
    labels: (N, 3) with [1s, 5s, 10s] in ticks.
    direction: (N,) with +1/-1.

    MFE = max favorable excursion = max(direction * price_path)
    MAE = max adverse excursion = min(direction * price_path)

    We only have snapshots at 1s, 5s, 10s, so we compute excursion at each.
    """
    if len(labels) == 0:
        return {}

    result = {}
    for h_name, h_idx in HORIZON_IDX.items():
        excursion = direction * labels[:, h_idx]  # positive = favorable
        favorable = excursion.copy()
        adverse = excursion.copy()

        result[f"mfe_{h_name}"] = {
            "mean": round(float(favorable.mean()), 4),
            "median": round(float(np.median(favorable)), 4),
            "p25": round(float(np.percentile(favorable, 25)), 4),
            "p75": round(float(np.percentile(favorable, 75)), 4),
            "p90": round(float(np.percentile(favorable, 90)), 4),
            "std": round(float(favorable.std()), 4),
        }

        # For MAE, we want to know the worst adverse excursion
        # at each horizon on the way to the final 10s
        # adverse excursion = negative moves
        adverse_only = excursion[excursion < 0]
        if len(adverse_only) > 0:
            result[f"mae_{h_name}"] = {
                "mean": round(float(adverse_only.mean()), 4),
                "median": round(float(np.median(adverse_only)), 4),
                "p10": round(float(np.percentile(adverse_only, 10)), 4),
                "p25": round(float(np.percentile(adverse_only, 25)), 4),
                "pct_adverse": round(float(len(adverse_only) / len(excursion)), 4),
            }
        else:
            result[f"mae_{h_name}"] = {
                "mean": 0, "median": 0, "p10": 0, "p25": 0, "pct_adverse": 0,
            }

    # Optimal TP/SL from MFE/MAE distributions
    # TP: set at p75 of favorable excursion at 10s (captures 75% of winners)
    # SL: set at p25 of adverse excursion at 10s (limits 75% of drawdowns)
    exc_10s = direction * labels[:, 2]
    fav = exc_10s[exc_10s > 0]
    adv = exc_10s[exc_10s < 0]

    if len(fav) > 10 and len(adv) > 10:
        result["optimal_tp_ticks"] = round(float(np.percentile(fav, 50)), 2)
        result["optimal_sl_ticks"] = round(float(abs(np.percentile(adv, 25))), 2)
        # Also compute for aggressive and conservative
        result["aggressive_tp"] = round(float(np.percentile(fav, 25)), 2)
        result["conservative_tp"] = round(float(np.percentile(fav, 75)), 2)
        result["tight_sl"] = round(float(abs(np.percentile(adv, 50))), 2)
        result["wide_sl"] = round(float(abs(np.percentile(adv, 10))), 2)
    else:
        result["optimal_tp_ticks"] = 0
        result["optimal_sl_ticks"] = 0

    return result


def estimate_time_of_day(sample_idx: int, n_total: int) -> str:
    """
    Estimate time of day from sample position.
    Trading day ~6.5 hours = 390 min. Samples evenly distributed.
    RTH: 9:30 - 16:00 ET.
    """
    frac = sample_idx / max(n_total, 1)
    minutes = int(frac * 390)
    hour = 9 + (30 + minutes) // 60
    minute = (30 + minutes) % 60
    return f"{hour:02d}:{minute:02d}"


def time_bucket(sample_idx: int, n_total: int) -> str:
    """Map sample position to hour bucket."""
    frac = sample_idx / max(n_total, 1)
    minutes = int(frac * 390)
    hour = 9 + (30 + minutes) // 60
    if hour < 10:
        return "09:30-10:00"
    elif hour < 11:
        return "10:00-11:00"
    elif hour < 12:
        return "11:00-12:00"
    elif hour < 13:
        return "12:00-13:00"
    elif hour < 14:
        return "13:00-14:00"
    elif hour < 15:
        return "14:00-15:00"
    else:
        return "15:00-16:00"


# ── Confluence Configurations ──

CONFIGS = {
    "baseline_cm_z2.0": {
        "desc": "CNN-Mamba |z|>2.0 only (baseline)",
        "cm_z_thresh": 2.0,
        "require_pt_agree": False,
        "pt_z_thresh": None,
        "require_vol": False,
    },
    "baseline_cm_z2.5": {
        "desc": "CNN-Mamba |z|>2.5 only (high-conviction baseline)",
        "cm_z_thresh": 2.5,
        "require_pt_agree": False,
        "pt_z_thresh": None,
        "require_vol": False,
    },
    "confluence_z2.0_agree": {
        "desc": "CM z>2.0 AND PatchTST agrees direction",
        "cm_z_thresh": 2.0,
        "require_pt_agree": True,
        "pt_z_thresh": None,
        "require_vol": False,
    },
    "confluence_z2.5_agree": {
        "desc": "CM z>2.5 AND PatchTST agrees direction",
        "cm_z_thresh": 2.5,
        "require_pt_agree": True,
        "pt_z_thresh": None,
        "require_vol": False,
    },
    "strong_z2.0_pt1.0": {
        "desc": "CM z>2.0 AND PatchTST z>1.0 same dir",
        "cm_z_thresh": 2.0,
        "require_pt_agree": True,
        "pt_z_thresh": 1.0,
        "require_vol": False,
    },
    "strong_z2.5_pt1.5": {
        "desc": "CM z>2.5 AND PatchTST z>1.5 same dir",
        "cm_z_thresh": 2.5,
        "require_pt_agree": True,
        "pt_z_thresh": 1.5,
        "require_vol": False,
    },
    "strong_z2.0_pt1.5": {
        "desc": "CM z>2.0 AND PatchTST z>1.5 same dir",
        "cm_z_thresh": 2.0,
        "require_pt_agree": True,
        "pt_z_thresh": 1.5,
        "require_vol": False,
    },
    "confluence_z2.0_vol_confirm": {
        "desc": "CM z>2.0 + PT agrees + Vol LGBM high-vol regime",
        "cm_z_thresh": 2.0,
        "require_pt_agree": True,
        "pt_z_thresh": None,
        "require_vol": True,
        "vol_quantile": 0.5,  # above median vol = "high vol regime"
    },
    "confluence_z2.0_vol_high": {
        "desc": "CM z>2.0 + PT agrees + Vol LGBM top-25% vol",
        "cm_z_thresh": 2.0,
        "require_pt_agree": True,
        "pt_z_thresh": None,
        "require_vol": True,
        "vol_quantile": 0.75,
    },
    "anti_confluence_z2.0": {
        "desc": "CM z>2.0 AND PatchTST DISAGREES (control)",
        "cm_z_thresh": 2.0,
        "require_pt_agree": False,
        "require_pt_disagree": True,
        "pt_z_thresh": None,
        "require_vol": False,
    },
    "combined_score": {
        "desc": "Combined z-score (CM_z + PT_z)/2 > 2.0",
        "combined_z_thresh": 2.0,
        "require_vol": False,
    },
    "combined_score_2.5": {
        "desc": "Combined z-score (CM_z + PT_z)/2 > 2.5",
        "combined_z_thresh": 2.5,
        "require_vol": False,
    },
}


def apply_config(config: dict, aligned: dict, vol_pred: np.ndarray = None,
                 vol_label: np.ndarray = None, horizon: str = "10s") -> dict:
    """
    Apply a confluence config and return trade-level results.
    Returns dict with mask, direction, pnls, labels, etc.
    """
    h_idx = HORIZON_IDX[horizon]
    cm_z = aligned[f"cm_z_{horizon}"]
    pt_z = aligned[f"pt_z_{horizon}"]
    labels = aligned["labels"]
    n = aligned["n_samples"]

    # Combined score configs
    if "combined_z_thresh" in config:
        # Both must agree on direction for combined score
        agree = np.sign(cm_z) == np.sign(pt_z)
        combined_z = (np.abs(cm_z) + np.abs(pt_z)) / 2.0
        mask = agree & (combined_z > config["combined_z_thresh"])
        direction = np.sign(cm_z[mask])
    else:
        # CM threshold
        cm_thresh = config["cm_z_thresh"]
        mask = np.abs(cm_z) > cm_thresh

        if config.get("require_pt_agree"):
            mask = mask & (np.sign(cm_z) == np.sign(pt_z))

        if config.get("require_pt_disagree"):
            mask = mask & (np.sign(cm_z) != np.sign(pt_z)) & (np.sign(pt_z) != 0)

        if config.get("pt_z_thresh") is not None:
            mask = mask & (np.abs(pt_z) > config["pt_z_thresh"])

        if config.get("require_vol") and vol_pred is not None:
            vol_thresh = np.quantile(vol_pred, config.get("vol_quantile", 0.5))
            mask = mask & (vol_pred[:n] > vol_thresh)

        direction = np.sign(cm_z[mask])

    if mask.sum() == 0:
        return {"n_trades": 0, "pnls": np.array([]), "mask": mask,
                "direction": np.array([]), "labels_at_trade": np.zeros((0, 3))}

    raw_pnl = direction * labels[mask, h_idx]
    pnl = raw_pnl - COST_RT_TICKS

    return {
        "n_trades": int(mask.sum()),
        "pnls": pnl,
        "raw_pnls": raw_pnl,
        "mask": mask,
        "direction": direction,
        "labels_at_trade": labels[mask],
        "cm_z_at_trade": cm_z[mask],
        "pt_z_at_trade": pt_z[mask],
        "trade_indices": np.where(mask)[0],
    }


# ── Per-date processing ──

def process_date(args) -> Optional[dict]:
    """Process a single date: load, align, run all configs."""
    date_str, cm_fold, pt_fold, vol_path = args

    cm_path = CM_DIR / f"fold_{cm_fold:02d}_oot_predictions.npz"
    pt_path = PT_DIR / f"fold_{pt_fold:02d}_oot_predictions.npz"

    try:
        cm_data = dict(np.load(cm_path, allow_pickle=True))
        pt_data = dict(np.load(pt_path, allow_pickle=True))
    except Exception as e:
        log.warning(f"Failed to load {date_str}: {e}")
        return None

    aligned = align_cm_pt(cm_data, pt_data)

    if aligned["n_samples"] < 100:
        log.warning(f"{date_str}: only {aligned['n_samples']} aligned samples, skipping")
        return None

    # Load vol if available
    vol_pred = None
    vol_label = None
    if vol_path and Path(vol_path).exists():
        try:
            vol_data = dict(np.load(vol_path, allow_pickle=True))
            vol_pred, vol_label = align_vol(aligned, vol_data)
        except Exception as e:
            log.warning(f"Failed to load vol for {date_str}: {e}")

    # Run all configs
    config_results = {}
    for cfg_name, cfg in CONFIGS.items():
        if cfg.get("require_vol") and vol_pred is None:
            continue

        trade_result = apply_config(cfg, aligned, vol_pred, vol_label)

        metrics = compute_metrics(trade_result["pnls"])

        # MFE/MAE analysis
        mfe_mae = {}
        if trade_result["n_trades"] > 10:
            mfe_mae = compute_mfe_mae(
                trade_result["labels_at_trade"],
                trade_result["direction"]
            )

        # Time of day analysis
        tod = defaultdict(list)
        if trade_result["n_trades"] > 0:
            n_total = aligned["n_samples"]
            for idx, pnl_val in zip(trade_result["trade_indices"],
                                     trade_result["pnls"]):
                bucket = time_bucket(int(idx), n_total)
                tod[bucket].append(float(pnl_val))

        tod_metrics = {}
        for bucket, pnls in sorted(tod.items()):
            pnls_arr = np.array(pnls)
            tod_metrics[bucket] = {
                "n_trades": len(pnls),
                "win_rate": round(float((pnls_arr > 0).mean()), 4),
                "avg_pnl": round(float(pnls_arr.mean()), 4),
                "total_pnl": round(float(pnls_arr.sum()), 4),
            }

        config_results[cfg_name] = {
            "metrics": metrics,
            "mfe_mae": mfe_mae,
            "time_of_day": tod_metrics,
        }

    return {
        "date": date_str,
        "n_cm_samples": len(cm_data["predictions"]),
        "n_pt_samples": len(pt_data["predictions"]),
        "n_aligned": aligned["n_samples"],
        "has_vol": vol_pred is not None,
        "cm_ic_10s": float(cm_data.get("ic_10s", 0)),
        "pt_ic_10s": float(pt_data.get("ic_10s", 0)),
        "configs": config_results,
    }


# ── Aggregation ──

def aggregate_all(date_results: List[dict]) -> dict:
    """Aggregate per-date results into global summaries per config."""
    agg = {}

    # Get baseline trade count for coverage
    baseline_key = "baseline_cm_z2.0"

    for cfg_name in CONFIGS.keys():
        all_pnls = []
        all_labels = []
        all_directions = []
        all_tod = defaultdict(list)
        per_date = {}
        mfe_mae_accum = defaultdict(list)

        for dr in date_results:
            if cfg_name not in dr["configs"]:
                continue
            cr = dr["configs"][cfg_name]
            m = cr["metrics"]

            # Reconstruct pnls from per-date metrics
            # (We stored metrics but not raw pnls to keep memory down)
            per_date[dr["date"]] = {
                "n_trades": m["n_trades"],
                "win_rate": m["win_rate"],
                "sharpe": m["sharpe"],
                "sortino": m["sortino"],
                "profit_factor": m["profit_factor"],
                "total_pnl_ticks": m["total_pnl_ticks"],
            }

            # For time-of-day aggregation
            for bucket, td in cr["time_of_day"].items():
                all_tod[bucket].append(td)

        agg[cfg_name] = {
            "per_date": per_date,
            "time_of_day_agg": {},
        }

        # Aggregate TOD
        for bucket in sorted(all_tod.keys()):
            entries = all_tod[bucket]
            total_trades = sum(e["n_trades"] for e in entries)
            total_pnl = sum(e["total_pnl"] for e in entries)
            total_wins = sum(e["n_trades"] * e["win_rate"] for e in entries)
            agg[cfg_name]["time_of_day_agg"][bucket] = {
                "n_trades": total_trades,
                "win_rate": round(total_wins / max(total_trades, 1), 4),
                "avg_pnl": round(total_pnl / max(total_trades, 1), 4),
                "total_pnl": round(total_pnl, 4),
            }

    return agg


def process_date_raw(args) -> Optional[dict]:
    """Process date returning raw pnls for proper aggregation."""
    date_str, cm_fold, pt_fold, vol_path = args

    cm_path = CM_DIR / f"fold_{cm_fold:02d}_oot_predictions.npz"
    pt_path = PT_DIR / f"fold_{pt_fold:02d}_oot_predictions.npz"

    try:
        cm_data = dict(np.load(cm_path, allow_pickle=True))
        pt_data = dict(np.load(pt_path, allow_pickle=True))
    except Exception as e:
        return None

    aligned = align_cm_pt(cm_data, pt_data)
    if aligned["n_samples"] < 100:
        return None

    vol_pred = None
    vol_label = None
    if vol_path and Path(vol_path).exists():
        try:
            vol_data = dict(np.load(vol_path, allow_pickle=True))
            vol_pred, vol_label = align_vol(aligned, vol_data)
        except:
            pass

    results = {}
    for cfg_name, cfg in CONFIGS.items():
        if cfg.get("require_vol") and vol_pred is None:
            continue

        tr = apply_config(cfg, aligned, vol_pred, vol_label)
        results[cfg_name] = {
            "pnls": tr["pnls"].tolist() if len(tr["pnls"]) > 0 else [],
            "n_trades": tr["n_trades"],
            "labels_at_trade": tr["labels_at_trade"].tolist() if tr["n_trades"] > 0 else [],
            "directions": tr["direction"].tolist() if tr["n_trades"] > 0 else [],
            "trade_indices": tr["trade_indices"].tolist() if tr["n_trades"] > 0 else [],
        }

    return {
        "date": date_str,
        "n_aligned": aligned["n_samples"],
        "has_vol": vol_pred is not None,
        "cm_ic_10s": float(cm_data.get("ic_10s", 0)),
        "pt_ic_10s": float(pt_data.get("ic_10s", 0)),
        "configs": results,
    }


# ── Main ──

def main():
    log.info("=" * 80)
    log.info("FULL CONFLUENCE ANALYSIS: CNN-Mamba v2 + PatchTST + Vol LGBM")
    log.info("MIDPOINT-BASED (no FIFO sim). Cost: 0.376 ticks RT ($4.70)")
    log.info("=" * 80)

    cm_folds, pt_folds, vol_files, common_dates, vol_dates = discover_folds()

    log.info(f"CNN-Mamba v2 folds: {len(cm_folds)} dates")
    log.info(f"PatchTST folds:     {len(pt_folds)} dates")
    log.info(f"Vol LGBM files:     {len(vol_files)} dates")
    log.info(f"Common CM+PT dates: {len(common_dates)} -> {common_dates}")
    log.info(f"With Vol overlay:   {len(vol_dates)} -> {vol_dates}")
    log.info(f"Configs to test:    {len(CONFIGS)}")

    if len(common_dates) == 0:
        log.error("No common dates found!")
        sys.exit(1)

    # Build work items
    work_items = []
    for date in common_dates:
        vol_path = str(vol_files[date]) if date in vol_files else None
        work_items.append((date, cm_folds[date], pt_folds[date], vol_path))

    # Phase 1: Per-date processing with raw pnl collection
    n_workers = min(16, cpu_count())
    log.info(f"Processing {len(work_items)} dates with {n_workers} workers...")

    t0 = datetime.now()
    with Pool(processes=n_workers) as pool:
        raw_results = pool.map(process_date_raw, work_items)

    # Also run the detailed version for MFE/MAE
    with Pool(processes=n_workers) as pool:
        detailed_results = pool.map(process_date, work_items)

    elapsed = (datetime.now() - t0).total_seconds()
    log.info(f"Processing complete in {elapsed:.1f}s")

    # Filter None
    raw_results = [r for r in raw_results if r is not None]
    detailed_results = [r for r in detailed_results if r is not None]
    log.info(f"Valid dates: {len(raw_results)}")

    if len(raw_results) == 0:
        log.error("No valid dates after processing!")
        sys.exit(1)

    # Phase 2: Aggregate metrics from raw pnls
    log.info("Aggregating results...")

    # Aggregate raw pnls across all dates
    global_metrics = {}
    for cfg_name in CONFIGS.keys():
        all_pnls = []
        all_labels = []
        all_directions = []
        all_indices = []  # for TOD
        n_total_per_date = []
        per_date_metrics = {}

        for dr in raw_results:
            if cfg_name not in dr["configs"]:
                continue
            cr = dr["configs"][cfg_name]
            pnls = cr["pnls"]
            all_pnls.extend(pnls)

            if cr["labels_at_trade"]:
                all_labels.extend(cr["labels_at_trade"])
                all_directions.extend(cr["directions"])

            # Per-date metrics
            if len(pnls) > 0:
                pnl_arr = np.array(pnls)
                dm = compute_metrics(pnl_arr)
                per_date_metrics[dr["date"]] = dm
            else:
                per_date_metrics[dr["date"]] = {"n_trades": 0}

        all_pnls = np.array(all_pnls, dtype=np.float64)
        global_m = compute_metrics(all_pnls)

        # Compute trades per day
        n_valid_dates = sum(1 for dr in raw_results if cfg_name in dr["configs"])
        global_m["trades_per_day"] = round(global_m["n_trades"] / max(n_valid_dates, 1), 1)

        # MFE/MAE from labels
        mfe_mae = {}
        if len(all_labels) > 10:
            all_labels_arr = np.array(all_labels, dtype=np.float64)
            all_dirs_arr = np.array(all_directions, dtype=np.float64)
            mfe_mae = compute_mfe_mae(all_labels_arr, all_dirs_arr)

        global_metrics[cfg_name] = {
            "config": CONFIGS[cfg_name] if cfg_name in CONFIGS else {},
            "global": global_m,
            "mfe_mae": mfe_mae,
            "per_date": per_date_metrics,
        }

    # Phase 3: Time-of-day from detailed results
    tod_agg = aggregate_all(detailed_results)
    for cfg_name in global_metrics:
        if cfg_name in tod_agg:
            global_metrics[cfg_name]["time_of_day"] = tod_agg[cfg_name]["time_of_day_agg"]

    # ── Print Results ──
    print("\n" + "=" * 120)
    print("  FULL CONFLUENCE ANALYSIS — CNN-Mamba v2 + PatchTST (MIDPOINT-BASED)")
    print(f"  Cost: 0.376 ticks RT ($4.70 commission, no spread) | {len(raw_results)} OOS dates")
    print(f"  Dates: {', '.join(d['date'] for d in raw_results)}")
    print("=" * 120)

    header = (f"{'Config':<32} | {'Trades':>6} | {'T/Day':>5} | {'WR':>6} | "
              f"{'Sharpe':>7} | {'Sortino':>8} | {'PF':>6} | {'R:R':>5} | "
              f"{'AvgPnL':>7} | {'PnL(t)':>9} | {'PnL($)':>10}")
    sep = "-" * len(header)
    print(header)
    print(sep)

    # Sort by Sharpe for easy reading
    sorted_configs = sorted(global_metrics.items(),
                           key=lambda x: x[1]["global"]["sharpe"], reverse=True)

    for cfg_name, data in sorted_configs:
        m = data["global"]
        desc = CONFIGS.get(cfg_name, {}).get("desc", cfg_name)[:32]
        print(
            f"{desc:<32} | {m['n_trades']:>6} | {m['trades_per_day']:>5} | "
            f"{m['win_rate']:>5.1%} | {m['sharpe']:>7.4f} | {m['sortino']:>8.4f} | "
            f"{m['profit_factor']:>6.2f} | {m['rr_ratio']:>5.2f} | "
            f"{m['avg_pnl']:>7.4f} | {m['total_pnl_ticks']:>9.2f} | "
            f"{m['total_pnl_usd']:>10.2f}"
        )
    print(sep)

    # ── MFE/MAE Summary ──
    print("\n" + "=" * 100)
    print("  MFE/MAE ANALYSIS (all horizons)")
    print("=" * 100)

    for cfg_name, data in sorted_configs[:6]:  # top 6 configs
        mfe = data.get("mfe_mae", {})
        if not mfe:
            continue
        desc = CONFIGS.get(cfg_name, {}).get("desc", cfg_name)[:40]
        print(f"\n  {desc}")
        print(f"  {'Horizon':<6} | {'MFE_mean':>9} | {'MFE_med':>8} | {'MFE_p75':>8} | "
              f"{'MFE_p90':>8} | {'MAE_mean':>9} | {'MAE_med':>8} | {'%Adverse':>8}")
        for h in HORIZONS:
            mfe_h = mfe.get(f"mfe_{h}", {})
            mae_h = mfe.get(f"mae_{h}", {})
            if mfe_h:
                print(
                    f"  {h:<6} | {mfe_h.get('mean',0):>9.3f} | {mfe_h.get('median',0):>8.3f} | "
                    f"{mfe_h.get('p75',0):>8.3f} | {mfe_h.get('p90',0):>8.3f} | "
                    f"{mae_h.get('mean',0):>9.3f} | {mae_h.get('median',0):>8.3f} | "
                    f"{mae_h.get('pct_adverse',0):>7.1%}"
                )

        if "optimal_tp_ticks" in mfe:
            print(f"  Optimal TP: {mfe['optimal_tp_ticks']} ticks | "
                  f"Optimal SL: {mfe['optimal_sl_ticks']} ticks")
            print(f"  Aggressive TP: {mfe.get('aggressive_tp', '?')} | "
                  f"Conservative TP: {mfe.get('conservative_tp', '?')}")
            print(f"  Tight SL: {mfe.get('tight_sl', '?')} | "
                  f"Wide SL: {mfe.get('wide_sl', '?')}")

    # ── Time of Day Analysis ──
    print("\n" + "=" * 100)
    print("  TIME-OF-DAY ANALYSIS (top configs)")
    print("=" * 100)

    for cfg_name, data in sorted_configs[:4]:
        tod = data.get("time_of_day", {})
        if not tod:
            continue
        desc = CONFIGS.get(cfg_name, {}).get("desc", cfg_name)[:40]
        print(f"\n  {desc}")
        print(f"  {'Bucket':<15} | {'Trades':>7} | {'WR':>6} | {'AvgPnL':>8} | {'TotalPnL':>9}")
        for bucket in sorted(tod.keys()):
            t = tod[bucket]
            print(f"  {bucket:<15} | {t['n_trades']:>7} | {t['win_rate']:>5.1%} | "
                  f"{t['avg_pnl']:>8.4f} | {t['total_pnl']:>9.2f}")

    # ── Per-Date Breakdown ──
    print("\n" + "=" * 100)
    print("  PER-DATE BREAKDOWN (top 4 configs)")
    print("=" * 100)

    for cfg_name, data in sorted_configs[:4]:
        desc = CONFIGS.get(cfg_name, {}).get("desc", cfg_name)[:40]
        print(f"\n  {desc}")
        print(f"  {'Date':<10} | {'Trades':>6} | {'WR':>6} | {'Sharpe':>7} | "
              f"{'Sortino':>8} | {'PF':>6} | {'PnL(t)':>9}")
        for date in sorted(data["per_date"].keys()):
            pd = data["per_date"][date]
            if isinstance(pd, dict) and pd.get("n_trades", 0) > 0:
                print(
                    f"  {date:<10} | {pd['n_trades']:>6} | {pd['win_rate']:>5.1%} | "
                    f"{pd.get('sharpe',0):>7.4f} | {pd.get('sortino',0):>8.4f} | "
                    f"{pd.get('profit_factor',0):>6.2f} | {pd.get('total_pnl_ticks',0):>9.2f}"
                )
            else:
                print(f"  {date:<10} | {pd.get('n_trades',0):>6} | {'N/A':>6}")

    # ── Confluence Value Assessment ──
    print("\n" + "=" * 100)
    print("  CONFLUENCE VALUE ASSESSMENT")
    print("=" * 100)

    baseline = global_metrics.get("baseline_cm_z2.0", {}).get("global", {})
    for cfg_name, data in sorted_configs:
        if cfg_name.startswith("baseline"):
            continue
        m = data["global"]
        if m["n_trades"] == 0:
            continue
        base_sharpe = baseline.get("sharpe", 0)
        base_wr = baseline.get("win_rate", 0)
        base_trades = baseline.get("n_trades", 1)

        sharpe_delta = m["sharpe"] - base_sharpe
        wr_delta = m["win_rate"] - base_wr
        coverage = m["n_trades"] / max(base_trades, 1)

        desc = CONFIGS.get(cfg_name, {}).get("desc", cfg_name)[:45]
        verdict = "IMPROVES" if sharpe_delta > 0 else "DEGRADES"
        print(f"  {desc}")
        print(f"    Sharpe: {base_sharpe:.4f} -> {m['sharpe']:.4f} ({sharpe_delta:+.4f}) [{verdict}]")
        print(f"    WR:     {base_wr:.1%} -> {m['win_rate']:.1%} ({wr_delta:+.1%})")
        print(f"    Coverage: {coverage:.1%} of baseline trades")
        print(f"    Sortino: {m['sortino']:.4f}, PF: {m['profit_factor']:.2f}")
        print()

    # ── Recommended Trading Configs ──
    print("=" * 100)
    print("  RECOMMENDED TRADING CONFIGURATIONS")
    print("=" * 100)

    # Score configs: Sharpe * sqrt(trades_per_day) to balance quality vs quantity
    scored = []
    for cfg_name, data in global_metrics.items():
        m = data["global"]
        if m["n_trades"] < 20 or m["sharpe"] <= 0:
            continue
        # Quality-adjusted score
        score = m["sharpe"] * np.sqrt(m["trades_per_day"])
        scored.append((score, cfg_name, data))

    scored.sort(reverse=True)

    for rank, (score, cfg_name, data) in enumerate(scored[:4], 1):
        m = data["global"]
        mfe = data.get("mfe_mae", {})
        desc = CONFIGS.get(cfg_name, {}).get("desc", cfg_name)
        cfg = CONFIGS.get(cfg_name, {})

        print(f"\n  CONFIG #{rank}: {desc}")
        print(f"  {'─' * 60}")
        print(f"  Sharpe: {m['sharpe']:.4f} | Sortino: {m['sortino']:.4f} | PF: {m['profit_factor']:.2f}")
        print(f"  WR: {m['win_rate']:.1%} | R:R: {m['rr_ratio']:.2f} | Trades/Day: {m['trades_per_day']}")
        print(f"  Quality Score: {score:.3f} (Sharpe * sqrt(T/D))")

        if mfe:
            tp = mfe.get("optimal_tp_ticks", "?")
            sl = mfe.get("optimal_sl_ticks", "?")
            print(f"  Suggested TP: {tp} ticks | SL: {sl} ticks")
            print(f"  Aggressive TP: {mfe.get('aggressive_tp', '?')} | "
                  f"Conservative TP: {mfe.get('conservative_tp', '?')}")
            print(f"  Tight SL: {mfe.get('tight_sl', '?')} | Wide SL: {mfe.get('wide_sl', '?')}")

        # Best time of day
        tod = data.get("time_of_day", {})
        if tod:
            best_bucket = max(tod.items(), key=lambda x: x[1].get("avg_pnl", 0))
            worst_bucket = min(tod.items(), key=lambda x: x[1].get("avg_pnl", 0))
            print(f"  Best TOD: {best_bucket[0]} (avg={best_bucket[1]['avg_pnl']:.3f})")
            print(f"  Worst TOD: {worst_bucket[0]} (avg={worst_bucket[1]['avg_pnl']:.3f})")

        # Parameters
        print(f"  Parameters:")
        if "cm_z_thresh" in cfg:
            print(f"    CNN-Mamba z threshold: {cfg['cm_z_thresh']}")
        if cfg.get("require_pt_agree"):
            print(f"    PatchTST: must agree on direction")
        if cfg.get("pt_z_thresh"):
            print(f"    PatchTST z threshold: {cfg['pt_z_thresh']}")
        if "combined_z_thresh" in cfg:
            print(f"    Combined z threshold: {cfg['combined_z_thresh']}")
        if cfg.get("require_vol"):
            print(f"    Vol regime: top {(1-cfg.get('vol_quantile',0.5))*100:.0f}%")

    # ── Save full results ──
    output = {
        "metadata": {
            "generated": _ts,
            "analysis": "MIDPOINT-BASED (no FIFO sim)",
            "cost_rt_ticks": COST_RT_TICKS,
            "cost_rt_usd": COST_RT_USD,
            "horizon": "10s",
            "n_common_dates": len(common_dates),
            "n_valid_dates": len(raw_results),
            "dates": [d["date"] for d in raw_results],
            "n_configs": len(CONFIGS),
        },
        "configs": {k: v.get("desc", k) for k, v in CONFIGS.items()},
        "global_metrics": {},
    }

    for cfg_name, data in global_metrics.items():
        # Make JSON-serializable
        serializable = {
            "global": data["global"],
            "mfe_mae": data.get("mfe_mae", {}),
            "time_of_day": data.get("time_of_day", {}),
            "per_date": data.get("per_date", {}),
        }
        output["global_metrics"][cfg_name] = serializable

    # Add recommendations
    output["recommendations"] = []
    for rank, (score, cfg_name, data) in enumerate(scored[:4], 1):
        m = data["global"]
        mfe = data.get("mfe_mae", {})
        output["recommendations"].append({
            "rank": rank,
            "config": cfg_name,
            "description": CONFIGS.get(cfg_name, {}).get("desc", cfg_name),
            "quality_score": round(score, 3),
            "sharpe": m["sharpe"],
            "sortino": m["sortino"],
            "profit_factor": m["profit_factor"],
            "win_rate": m["win_rate"],
            "rr_ratio": m["rr_ratio"],
            "trades_per_day": m["trades_per_day"],
            "optimal_tp_ticks": mfe.get("optimal_tp_ticks"),
            "optimal_sl_ticks": mfe.get("optimal_sl_ticks"),
            "parameters": CONFIGS.get(cfg_name, {}),
        })

    out_path = OUTPUT_DIR / f"confluence_full_{_ts}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info(f"Full results saved to {out_path}")

    print(f"\n  Output: {OUTPUT_DIR}")
    print("  NOTE: All PnL figures are MIDPOINT-BASED. Requires FIFO sim for realistic execution.")
    print()


if __name__ == "__main__":
    main()
