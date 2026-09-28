#!/usr/bin/env python3
"""
Validate Smart Execution v4 (XGBoost TP/SL/Gate/Exit) on 39 OOS Decay Dates

Loads the fold_09 XGBoost models from smart_exec_v4 and applies them to
CNN-Mamba v2 decay predictions across 39 new out-of-sample dates.

For each date:
  1. Load CNN-Mamba v2 predictions (preds, labels)
  2. Build the same V3 feature pipeline as train_smart_exec_v4.py
  3. Apply TP/SL models -> augment features
  4. Apply Gate model -> gate confidence
  5. Apply Exit model -> hold time
  6. Compute PnL with 0.376 ticks RT cost (HC #52)
  7. Compare gated vs ungated performance

Uses multiprocessing to saturate all 16 cores (HC #62).

Output: /home/jupiter/Lvl3Quant/output/smart_exec_v4_oos_validation/
"""

import os
import sys
import json
import time
import logging
import argparse
import warnings
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np

warnings.filterwarnings("ignore")

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

logging.basicConfig(
    force=True,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ============================================================
# Constants (from train_smart_exec_v4.py)
# ============================================================
TICK_VAL = 12.50           # USD per tick
RT_COST_TICKS = 0.376      # HC #52: round-trip commission in ticks
ROUND_TRIP_COST = RT_COST_TICKS  # For gate target alignment

ET_OFFSET_HOURS = -5       # EST

SESSION_BOUNDS = [
    ("overnight",   0.0,  2.0),
    ("pre_market",  2.0,  9.5),
    ("rth_open",    9.5,  10.5),
    ("rth_core",    10.5, 15.0),
    ("rth_close",   15.0, 16.0),
    ("post_market", 16.0, 17.0),
    ("maintenance", 17.0, 17.75),
    ("evening",     17.75, 24.0),
]

SESSION_TO_IDX = {s[0]: i for i, s in enumerate(SESSION_BOUNDS)}

GAP_WINDOWS_ET = [
    (16.95, 17.75),
]

LOG_TPSL_MIN_FLOOR = 0.5
LOG_TPSL_CLIP = 6.0


# ============================================================
# Time helpers (replicated from training script)
# ============================================================

def ts_ns_to_et_hour(ts_ns: int) -> float:
    """Convert nanosecond timestamp to fractional hour in ET."""
    from datetime import datetime, timezone
    utc_sec = ts_ns / 1e9
    et_sec = utc_sec + ET_OFFSET_HOURS * 3600
    dt = datetime.fromtimestamp(et_sec, tz=timezone.utc)
    return dt.hour + dt.minute / 60.0 + dt.second / 3600.0


def is_in_gap(et_hour: float) -> bool:
    for start, end in GAP_WINDOWS_ET:
        if start <= et_hour < end:
            return True
    return False


def get_session_idx(et_hour: float) -> int:
    for name, start, end in SESSION_BOUNDS:
        if start <= et_hour < end:
            return SESSION_TO_IDX[name]
    return SESSION_TO_IDX["overnight"]


# ============================================================
# Rolling statistics (vectorized, from training script)
# ============================================================

def _rolling_zscore(arr: np.ndarray, window: int) -> np.ndarray:
    n = len(arr)
    z = np.zeros(n, dtype=np.float32)
    cumsum = np.cumsum(arr)
    cumsum2 = np.cumsum(arr ** 2)
    for i in range(n):
        start = max(0, i - window + 1)
        count = i - start + 1
        if count < 30:
            continue
        s = cumsum[i] - (cumsum[start - 1] if start > 0 else 0)
        s2 = cumsum2[i] - (cumsum2[start - 1] if start > 0 else 0)
        mean = s / count
        var = s2 / count - mean ** 2
        std = np.sqrt(max(var, 1e-10))
        z[i] = (arr[i] - mean) / std
    return z


def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    n = len(arr)
    out = np.zeros(n, dtype=np.float32)
    cs = np.cumsum(arr)
    cs2 = np.cumsum(arr ** 2)
    for i in range(n):
        start = max(0, i - window + 1)
        count = i - start + 1
        if count < 5:
            continue
        s = cs[i] - (cs[start - 1] if start > 0 else 0)
        s2 = cs2[i] - (cs2[start - 1] if start > 0 else 0)
        mean = s / count
        var = max(s2 / count - mean ** 2, 0.0)
        out[i] = np.sqrt(var)
    return out


def _rolling_percentile_rank(arr: np.ndarray, window: int) -> np.ndarray:
    n = len(arr)
    out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        start = max(0, i - window + 1)
        chunk = arr[start:i + 1]
        if len(chunk) < 10:
            out[i] = 0.5
        else:
            out[i] = np.searchsorted(np.sort(chunk), arr[i]) / len(chunk)
    return out


# ============================================================
# Feature Engineering V3 (replicated from train_smart_exec_v4.py)
# ============================================================

def build_exec_features_v3(
    cnn_preds: np.ndarray,      # (N, 3) predictions for 1s/5s/10s
    ptst_preds: np.ndarray,     # (N, 3) — zeros if unavailable
    vol_preds: np.ndarray,      # (N, 3) — zeros if unavailable
    timestamps: np.ndarray,     # (N,) int64 nanoseconds
    events: Optional[np.ndarray],  # None for OOS decay dates
    anchor_idxs: np.ndarray,    # (N,)
    zscore_window: int = 3000,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build feature vectors identical to train_smart_exec_v4.py.
    Returns (N, F) float32 feature matrix and list of feature names.
    """
    N = len(cnn_preds)
    features = []
    names = []

    # GROUP 1: CNN z-scores (3)
    for h, label in enumerate(["1s", "5s", "10s"]):
        z = _rolling_zscore(cnn_preds[:, h], zscore_window)
        features.append(z)
        names.append(f"cnn_z_{label}")

    # GROUP 2: PatchTST z-scores (3)
    for h, label in enumerate(["1s", "5s", "10s"]):
        z = _rolling_zscore(ptst_preds[:, h], zscore_window)
        features.append(z)
        names.append(f"ptst_z_{label}")

    # GROUP 3: Vol predictions (3)
    for h, label in enumerate(["1s", "5s", "10s"]):
        features.append(vol_preds[:, h])
        names.append(f"vol_pred_{label}")

    # GROUP 4: Model agreement per horizon (3)
    for h, label in enumerate(["1s", "5s", "10s"]):
        agreement = np.sign(cnn_preds[:, h]) * np.sign(ptst_preds[:, h])
        features.append(agreement)
        names.append(f"agreement_{label}")

    # GROUP 5: Conviction (3)
    for h, label in enumerate(["1s", "5s", "10s"]):
        cnn_z = _rolling_zscore(cnn_preds[:, h], zscore_window)
        ptst_z = _rolling_zscore(ptst_preds[:, h], zscore_window)
        conviction = np.abs(cnn_z) * np.sign(cnn_z) * np.sign(ptst_z)
        features.append(conviction)
        names.append(f"conviction_{label}")

    # GROUP 6: Cross-model features (4)
    cnn_z_10s = _rolling_zscore(cnn_preds[:, 2], zscore_window)
    ptst_z_10s = _rolling_zscore(ptst_preds[:, 2], zscore_window)

    conv_diff = np.abs(cnn_z_10s) - np.abs(ptst_z_10s)
    features.append(conv_diff)
    names.append("conv_diff_10s")

    agreement_score = cnn_z_10s * ptst_z_10s
    features.append(agreement_score)
    names.append("agreement_score_10s")

    best_conv = np.maximum(np.abs(cnn_z_10s), np.abs(ptst_z_10s))
    features.append(best_conv)
    names.append("best_conv_10s")

    vol_weighted_conv = vol_preds[:, 2] * np.abs(cnn_z_10s)
    features.append(vol_weighted_conv)
    names.append("vol_weighted_conv_10s")

    # GROUP 7: Time-of-day (12)
    et_hours = np.array([ts_ns_to_et_hour(t) for t in timestamps])

    tod_sin = np.sin(2 * np.pi * et_hours / 24.0).astype(np.float32)
    tod_cos = np.cos(2 * np.pi * et_hours / 24.0).astype(np.float32)
    features.append(tod_sin)
    names.append("tod_sin")
    features.append(tod_cos)
    names.append("tod_cos")

    session_idxs = np.array([get_session_idx(h) for h in et_hours])
    for s_idx, (s_name, _, _) in enumerate(SESSION_BOUNDS):
        oh = (session_idxs == s_idx).astype(np.float32)
        features.append(oh)
        names.append(f"session_{s_name}")

    mins_since_session = np.zeros(N, dtype=np.float32)
    for i, eh in enumerate(et_hours):
        s_idx = session_idxs[i]
        s_start = SESSION_BOUNDS[s_idx][1]
        mins_since_session[i] = (eh - s_start) * 60.0
    features.append(mins_since_session)
    names.append("mins_since_session_start")

    mins_until_maint = np.clip((17.0 - et_hours) * 60.0, -60.0, 600.0).astype(np.float32)
    features.append(mins_until_maint)
    names.append("mins_until_maintenance")

    gap_flag = np.array([is_in_gap(h) for h in et_hours], dtype=np.float32)
    features.append(gap_flag)
    names.append("gap_flag")

    # GROUP 8: Microstructure features (26) — all zeros for OOS decay dates
    # (no MBO events available in decay prediction files)
    micro_names = [
        "spread", "vol_100", "vol_500", "vol_2000",
        "vol_ratio_vs_session", "vol_acceleration", "vol_percentile",
        "momentum_5", "momentum_20", "momentum_50",
        "momentum_100", "momentum_500", "momentum_accel",
        "event_rate",
        "realized_vol_50", "realized_vol_200", "realized_vol_1000",
        "vol_of_vol", "vol_pctile_rank",
        "net_flow_50", "net_flow_200",
        "order_arrival_rate", "cancel_rate",
        "spread_zscore", "spread_percentile",
        "bid_ask_ratio",
    ]
    for mname in micro_names:
        features.append(np.zeros(N, dtype=np.float32))
        names.append(mname)

    # GROUP 9: Volatility regime from predictions (1)
    vol_regime = np.zeros(N, dtype=np.float32)
    window = min(500, N)
    for i in range(window, N):
        vol_regime[i] = np.std(cnn_preds[i - window:i, 2])
    features.append(vol_regime)
    names.append("vol_regime_pred")

    # Stack
    feature_matrix = np.column_stack(features).astype(np.float32)
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=10.0, neginf=-10.0)

    return feature_matrix, names


def augment_features_with_tpsl(
    base_features: np.ndarray,
    tp_model: "xgb.XGBRegressor",
    sl_model: "xgb.XGBRegressor",
) -> np.ndarray:
    """Augment base features with TP/SL predictions (5 extra columns)."""
    log_tp = tp_model.predict(base_features).astype(np.float32)
    log_sl = sl_model.predict(base_features).astype(np.float32)
    tp_ticks = np.exp(np.clip(log_tp, -2.0, LOG_TPSL_CLIP))
    sl_ticks = np.exp(np.clip(log_sl, -2.0, LOG_TPSL_CLIP))
    rr_ratio = tp_ticks / np.maximum(sl_ticks, 0.1)
    augmented = np.column_stack([
        base_features, log_tp, log_sl, tp_ticks, sl_ticks, rr_ratio,
    ]).astype(np.float32)
    return np.nan_to_num(augmented, nan=0.0, posinf=10.0, neginf=-10.0)


# ============================================================
# Performance metrics
# ============================================================

def compute_sortino(pnl: np.ndarray) -> float:
    if len(pnl) < 2:
        return 0.0
    downside = pnl[pnl < 0]
    if len(downside) < 2:
        return float(pnl.mean()) if pnl.mean() > 0 else 0.0
    return float(pnl.mean() / (np.std(downside) + 1e-8))


def compute_profit_factor(pnl: np.ndarray) -> float:
    gross_profit = pnl[pnl > 0].sum()
    gross_loss = np.abs(pnl[pnl < 0].sum())
    if gross_loss < 1e-8:
        return float("inf") if gross_profit > 0 else 0.0
    return float(gross_profit / gross_loss)


def compute_directional_accuracy(preds_10s: np.ndarray, labels_10s: np.ndarray) -> float:
    """DA = fraction where sign(pred) == sign(label), excluding zeros."""
    mask = labels_10s != 0
    if mask.sum() == 0:
        return 0.5
    return float((np.sign(preds_10s[mask]) == np.sign(labels_10s[mask])).mean())


# ============================================================
# Single-date validation worker
# ============================================================

def validate_single_date(
    date_str: str,
    decay_dir: Path,
    models: Dict,
    gate_thresholds: List[float],
) -> Optional[Dict]:
    """
    Validate all 4 smart exec models on a single decay date.
    Returns a dict of metrics or None on failure.
    """
    pred_path = decay_dir / date_str / "predictions.npz"
    if not pred_path.exists():
        return None

    try:
        data = np.load(pred_path, allow_pickle=True)
        preds = data["preds"]          # (N, 3)
        labels_1s = data["labels_1s"]  # (N,)
        labels_5s = data["labels_5s"]
        labels_10s = data["labels_10s"]
        valid_indices = data["valid_indices"]
        N = len(preds)

        if N < 10:
            return None

        # Build labels array (N, 3)
        labels = np.column_stack([labels_1s, labels_5s, labels_10s])

        # PatchTST and vol preds unavailable for decay dates
        ptst_preds = np.zeros((N, 3), dtype=np.float32)
        vol_preds = np.zeros((N, 3), dtype=np.float32)

        # Generate placeholder timestamps from valid_indices
        # valid_indices are event indices; use them as proxy nanosecond timestamps
        # Spread them across a trading day for time-of-day features
        timestamps = valid_indices.astype(np.int64)
        anchor_idxs = np.arange(N, dtype=np.int64)

        # Build features (same pipeline as training)
        features, feature_names = build_exec_features_v3(
            preds, ptst_preds, vol_preds,
            timestamps, None, anchor_idxs,
        )

        # Apply TP/SL models
        tp_model = models["tp"]
        sl_model = models["sl"]
        augmented = augment_features_with_tpsl(features, tp_model, sl_model)

        # Apply Gate model
        gate_model = models["gate"]
        gate_prob = gate_model.predict_proba(augmented)[:, 1].astype(np.float32)

        # Apply Exit model
        exit_model = models["exit"]
        exit_hold = exit_model.predict(augmented).astype(np.float32)

        # Compute direction and PnL
        direction = np.sign(preds[:, 2])  # 10s prediction direction
        realized_pnl_ticks = labels_10s * direction  # directional PnL

        # After-cost PnL
        pnl_after_cost = realized_pnl_ticks - RT_COST_TICKS

        # Ungated metrics
        ungated_da = compute_directional_accuracy(preds[:, 2], labels_10s)
        ungated_sortino = compute_sortino(pnl_after_cost)
        ungated_pf = compute_profit_factor(pnl_after_cost)
        ungated_avg_pnl = float(pnl_after_cost.mean())
        ungated_total_pnl = float(pnl_after_cost.sum())

        result = {
            "date": date_str,
            "n_samples": N,
            "ungated_da": ungated_da,
            "ungated_sortino": ungated_sortino,
            "ungated_pf": ungated_pf,
            "ungated_avg_pnl_ticks": ungated_avg_pnl,
            "ungated_total_pnl_ticks": ungated_total_pnl,
            "ungated_total_pnl_usd": ungated_total_pnl * TICK_VAL,
            "gate_prob_mean": float(gate_prob.mean()),
            "gate_prob_std": float(gate_prob.std()),
            "gate_prob_min": float(gate_prob.min()),
            "gate_prob_max": float(gate_prob.max()),
            "exit_hold_mean": float(exit_hold.mean()),
            "exit_hold_std": float(exit_hold.std()),
        }

        # Gated metrics at each threshold
        for thresh in gate_thresholds:
            t_key = f"{thresh:.2f}".replace(".", "")
            mask = gate_prob > thresh
            n_gated = int(mask.sum())
            rate = float(mask.mean())

            if n_gated > 0:
                gated_pnl = pnl_after_cost[mask]
                gated_da = compute_directional_accuracy(preds[mask, 2], labels_10s[mask])
                gated_sortino = compute_sortino(gated_pnl)
                gated_pf = compute_profit_factor(gated_pnl)
                gated_avg_pnl = float(gated_pnl.mean())
                gated_total_pnl = float(gated_pnl.sum())
            else:
                gated_da = 0.0
                gated_sortino = 0.0
                gated_pf = 0.0
                gated_avg_pnl = 0.0
                gated_total_pnl = 0.0

            result[f"gated_{t_key}_n"] = n_gated
            result[f"gated_{t_key}_rate"] = rate
            result[f"gated_{t_key}_da"] = gated_da
            result[f"gated_{t_key}_sortino"] = gated_sortino
            result[f"gated_{t_key}_pf"] = gated_pf
            result[f"gated_{t_key}_avg_pnl_ticks"] = gated_avg_pnl
            result[f"gated_{t_key}_total_pnl_ticks"] = gated_total_pnl
            result[f"gated_{t_key}_total_pnl_usd"] = gated_total_pnl * TICK_VAL

        return result

    except Exception as e:
        logger.error(f"Error processing {date_str}: {e}")
        import traceback
        traceback.print_exc()
        return None


def validate_single_date_wrapper(args):
    """Wrapper for multiprocessing — unpacks tuple args."""
    date_str, decay_dir, models_path, gate_thresholds = args

    # Load models inside each worker (XGBoost models are not picklable across processes)
    models = load_models(models_path)
    if models is None:
        return None

    return validate_single_date(
        date_str,
        Path(decay_dir),
        models,
        gate_thresholds,
    )


def load_models(model_dir: Path) -> Optional[Dict]:
    """Load fold_09 XGBoost models."""
    model_dir = Path(model_dir)

    tp_path = model_dir / "fold_09_tp_xgb.json"
    sl_path = model_dir / "fold_09_sl_xgb.json"
    gate_path = model_dir / "fold_09_gate_xgb.json"
    exit_path = model_dir / "fold_09_exit_xgb.json"

    for p in [tp_path, sl_path, gate_path, exit_path]:
        if not p.exists():
            logger.error(f"Model not found: {p}")
            return None

    tp_model = xgb.XGBRegressor()
    tp_model.load_model(str(tp_path))

    sl_model = xgb.XGBRegressor()
    sl_model.load_model(str(sl_path))

    gate_model = xgb.XGBClassifier()
    gate_model.load_model(str(gate_path))

    exit_model = xgb.XGBRegressor()
    exit_model.load_model(str(exit_path))

    return {
        "tp": tp_model,
        "sl": sl_model,
        "gate": gate_model,
        "exit": exit_model,
    }


# ============================================================
# Summary & Reporting
# ============================================================

def print_summary(results: List[Dict], gate_thresholds: List[float]):
    """Print a clean summary table of OOS validation results."""
    n_dates = len(results)
    if n_dates == 0:
        logger.error("No results to summarize.")
        return

    print("\n" + "=" * 120)
    print(f"  SMART EXEC v4 OOS VALIDATION — {n_dates} dates")
    print(f"  Cost: {RT_COST_TICKS} ticks RT (${RT_COST_TICKS * TICK_VAL:.2f})")
    print("=" * 120)

    # ---- Per-date table ----
    print(f"\n{'Date':<12} {'N':>6} {'DA':>6} {'Sortino':>8} {'PF':>6} {'AvgPnL':>8} "
          f"{'G50_N':>6} {'G50_DA':>6} {'G50_Sort':>8} {'G50_PF':>6} {'G50_Avg':>8} "
          f"{'G60_N':>6} {'G60_DA':>6} {'G60_Sort':>8}")
    print("-" * 120)

    for r in sorted(results, key=lambda x: x["date"]):
        print(f"{r['date']:<12} "
              f"{r['n_samples']:>6} "
              f"{r['ungated_da']:>6.3f} "
              f"{r['ungated_sortino']:>8.3f} "
              f"{r['ungated_pf']:>6.2f} "
              f"{r['ungated_avg_pnl_ticks']:>8.3f} "
              f"{r.get('gated_050_n', 0):>6} "
              f"{r.get('gated_050_da', 0):>6.3f} "
              f"{r.get('gated_050_sortino', 0):>8.3f} "
              f"{r.get('gated_050_pf', 0):>6.2f} "
              f"{r.get('gated_050_avg_pnl_ticks', 0):>8.3f} "
              f"{r.get('gated_060_n', 0):>6} "
              f"{r.get('gated_060_da', 0):>6.3f} "
              f"{r.get('gated_060_sortino', 0):>8.3f}")

    print("-" * 120)

    # ---- Aggregate summary ----
    total_samples = sum(r["n_samples"] for r in results)

    # Concatenate all per-date PnLs for proper aggregate metrics
    # We need to recompute aggregates from per-date stats
    avg_da = np.mean([r["ungated_da"] for r in results])
    avg_sortino = np.mean([r["ungated_sortino"] for r in results])
    total_pnl = sum(r["ungated_total_pnl_ticks"] for r in results)
    avg_pnl = total_pnl / total_samples if total_samples > 0 else 0

    print(f"\n{'AGGREGATE':<12} "
          f"{total_samples:>6} "
          f"{avg_da:>6.3f} "
          f"{avg_sortino:>8.3f} "
          f"{'':>6} "
          f"{avg_pnl:>8.3f} ")

    # ---- Threshold comparison table ----
    print(f"\n{'='*90}")
    print(f"  GATE THRESHOLD COMPARISON (aggregate across {n_dates} dates)")
    print(f"{'='*90}")
    print(f"{'Threshold':<12} {'Trades':>8} {'Rate':>6} {'Avg DA':>8} {'Avg Sort':>10} "
          f"{'Avg PF':>8} {'Avg PnL':>10} {'Total PnL':>12} {'Total USD':>12}")
    print("-" * 90)

    # Ungated row
    print(f"{'Ungated':<12} "
          f"{total_samples:>8} "
          f"{'100.0%':>6} "
          f"{avg_da:>8.4f} "
          f"{avg_sortino:>10.4f} "
          f"{np.mean([r['ungated_pf'] for r in results]):>8.3f} "
          f"{avg_pnl:>10.4f} "
          f"{total_pnl:>12.2f} "
          f"{total_pnl * TICK_VAL:>12.2f}")

    for thresh in gate_thresholds:
        t_key = f"{thresh:.2f}".replace(".", "")
        n_trades = sum(r.get(f"gated_{t_key}_n", 0) for r in results)
        rate = n_trades / total_samples if total_samples > 0 else 0
        avg_da_g = np.mean([r.get(f"gated_{t_key}_da", 0) for r in results
                            if r.get(f"gated_{t_key}_n", 0) > 0]) if any(
                                r.get(f"gated_{t_key}_n", 0) > 0 for r in results) else 0
        avg_sort_g = np.mean([r.get(f"gated_{t_key}_sortino", 0) for r in results
                              if r.get(f"gated_{t_key}_n", 0) > 0]) if any(
                                  r.get(f"gated_{t_key}_n", 0) > 0 for r in results) else 0
        avg_pf_g = np.mean([r.get(f"gated_{t_key}_pf", 0) for r in results
                            if r.get(f"gated_{t_key}_n", 0) > 0]) if any(
                                r.get(f"gated_{t_key}_n", 0) > 0 for r in results) else 0
        total_pnl_g = sum(r.get(f"gated_{t_key}_total_pnl_ticks", 0) for r in results)
        avg_pnl_g = total_pnl_g / n_trades if n_trades > 0 else 0

        print(f"{thresh:<12.2f} "
              f"{n_trades:>8} "
              f"{rate*100:>5.1f}% "
              f"{avg_da_g:>8.4f} "
              f"{avg_sort_g:>10.4f} "
              f"{avg_pf_g:>8.3f} "
              f"{avg_pnl_g:>10.4f} "
              f"{total_pnl_g:>12.2f} "
              f"{total_pnl_g * TICK_VAL:>12.2f}")

    print("-" * 90)

    # Win rate by date
    winning_dates_ungated = sum(1 for r in results if r["ungated_total_pnl_ticks"] > 0)
    print(f"\nDate win rate (ungated): {winning_dates_ungated}/{n_dates} "
          f"({winning_dates_ungated/n_dates*100:.1f}%)")

    for thresh in [0.50, 0.60, 0.70]:
        t_key = f"{thresh:.2f}".replace(".", "")
        winning = sum(1 for r in results
                      if r.get(f"gated_{t_key}_total_pnl_ticks", 0) > 0
                      and r.get(f"gated_{t_key}_n", 0) > 0)
        active = sum(1 for r in results if r.get(f"gated_{t_key}_n", 0) > 0)
        print(f"Date win rate (gate>{thresh:.2f}): {winning}/{active} "
              f"({winning/active*100:.1f}%)" if active > 0 else
              f"Date win rate (gate>{thresh:.2f}): 0/0")

    print()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Validate Smart Exec v4 on OOS Decay Dates"
    )
    parser.add_argument("--model-dir", type=str,
                        default="/home/jupiter/Lvl3Quant/output/smart_exec_v4",
                        help="Directory containing fold_09 XGBoost models")
    parser.add_argument("--decay-dir", type=str,
                        default="/home/jupiter/Lvl3Quant/output/decay_v4_comprehensive/CNN-Mamba_v2",
                        help="Directory containing per-date decay predictions")
    parser.add_argument("--output-dir", type=str,
                        default="/home/jupiter/Lvl3Quant/output/smart_exec_v4_oos_validation",
                        help="Output directory for results")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Number of parallel workers (0=auto, uses all cores)")
    parser.add_argument("--gate-thresholds", type=str,
                        default="0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80",
                        help="Comma-separated gate thresholds to evaluate")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    decay_dir = Path(args.decay_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gate_thresholds = [float(x) for x in args.gate_thresholds.split(",")]

    n_workers = args.num_workers if args.num_workers > 0 else cpu_count()
    n_workers = min(n_workers, 16)  # Cap at 16 cores

    logger.info("=" * 70)
    logger.info("Smart Exec v4 OOS Validation")
    logger.info("=" * 70)
    logger.info(f"Model dir:     {model_dir}")
    logger.info(f"Decay dir:     {decay_dir}")
    logger.info(f"Output dir:    {output_dir}")
    logger.info(f"Workers:       {n_workers}")
    logger.info(f"RT cost:       {RT_COST_TICKS} ticks (${RT_COST_TICKS * TICK_VAL:.2f})")
    logger.info(f"Thresholds:    {gate_thresholds}")
    logger.info("=" * 70)

    if not XGBOOST_AVAILABLE:
        logger.error("XGBoost not available. Install with: pip install xgboost")
        return

    # Verify models exist
    models_check = load_models(model_dir)
    if models_check is None:
        logger.error("Failed to load models. Exiting.")
        return
    del models_check
    logger.info("Models verified: fold_09 TP/SL/Gate/Exit loaded successfully.")

    # Discover decay dates
    date_dirs = sorted([
        d.name for d in decay_dir.iterdir()
        if d.is_dir() and (d / "predictions.npz").exists()
    ])
    logger.info(f"Found {len(date_dirs)} decay dates to validate.")

    if len(date_dirs) == 0:
        logger.error("No decay dates found. Exiting.")
        return

    # Dispatch to workers
    t0 = time.time()

    worker_args = [
        (date_str, str(decay_dir), str(model_dir), gate_thresholds)
        for date_str in date_dirs
    ]

    if n_workers > 1 and len(date_dirs) > 1:
        logger.info(f"Launching {n_workers} workers for {len(date_dirs)} dates...")
        with Pool(processes=n_workers) as pool:
            raw_results = pool.map(validate_single_date_wrapper, worker_args)
    else:
        logger.info("Running single-threaded...")
        raw_results = [validate_single_date_wrapper(a) for a in worker_args]

    # Filter failures
    results = [r for r in raw_results if r is not None]
    elapsed = time.time() - t0
    logger.info(f"Validation complete: {len(results)}/{len(date_dirs)} dates "
                f"in {elapsed:.1f}s")

    if len(results) == 0:
        logger.error("All dates failed. Check logs above.")
        return

    # Print summary
    print_summary(results, gate_thresholds)

    # Save detailed results
    results_path = output_dir / "oos_validation_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "meta": {
                "model_dir": str(model_dir),
                "decay_dir": str(decay_dir),
                "rt_cost_ticks": RT_COST_TICKS,
                "gate_thresholds": gate_thresholds,
                "n_dates": len(results),
                "n_dates_attempted": len(date_dirs),
                "elapsed_s": elapsed,
                "n_workers": n_workers,
            },
            "per_date": results,
        }, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    # Save per-date CSV for easy analysis
    csv_path = output_dir / "oos_validation_per_date.csv"
    with open(csv_path, "w") as f:
        if results:
            headers = sorted(results[0].keys())
            f.write(",".join(headers) + "\n")
            for r in sorted(results, key=lambda x: x["date"]):
                f.write(",".join(str(r.get(h, "")) for h in headers) + "\n")
    logger.info(f"CSV saved to {csv_path}")


if __name__ == "__main__":
    main()
