#!/usr/bin/env python3
"""
LGBM-Gated FIFO Validation Test
================================

KEY QUESTION: Does the LGBM execution filter (AUC=0.551 on midpoint) actually
improve FIFO outcomes? Or does it just re-rank trades that are all negative on FIFO?

APPROACH:
1. Run Rust FIFO fill sim on ALL predictions for each OOT date (low signal threshold)
2. Train walk-forward LGBM to predict FIFO profitability (not midpoint)
3. Use LGBM scores to gate: only take top N% by LGBM score
4. Compare FIFO P&L: ungated vs LGBM-gated at various thresholds

This tests TWO hypotheses:
A) LGBM trained on midpoint → does top 1% have better FIFO results?
B) LGBM trained on FIFO outcomes → can it identify the rare profitable FIFO trades?

Author: Claude (Infrastructure Builder)
Date: 2026-05-08
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import lightgbm as lgb
except ImportError:
    print("ERROR: pip install lightgbm")
    sys.exit(1)

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LVL3_ROOT = Path(os.environ.get("LVL3_ROOT", "/home/nick/Lvl3Quant"))
FILL_SIM_CLI = LVL3_ROOT / "rust_cache_builder/target/release/fill_sim_cli"
MBO_DIR = LVL3_ROOT / "data/raw/mbo"
PRED_DIR = LVL3_ROOT / "output/cnn_mamba_v2_all_oot"
MBO_EVENTS_DIR = LVL3_ROOT / "data/processed/mbo_events_smart_v3"

COMMISSION_TICKS = 0.376  # $4.70 / $12.50
LOG = logging.getLogger("LGBM_FIFO")

# Fill sim configs to test
FILL_SIM_CONFIGS = {
    "base_tp4_sl4": {
        "signal_threshold": 0.5,
        "take_profit_ticks": 4,
        "stop_loss_ticks": 4,
        "hold_ms": 30000,
        "max_wait_bars": 10,
    },
    "tight_tp3_sl3": {
        "signal_threshold": 0.5,
        "take_profit_ticks": 3,
        "stop_loss_ticks": 3,
        "hold_ms": 15000,
        "max_wait_bars": 5,
    },
    "wide_tp6_sl4": {
        "signal_threshold": 0.5,
        "take_profit_ticks": 6,
        "stop_loss_ticks": 4,
        "hold_ms": 60000,
        "max_wait_bars": 10,
    },
}

# Feature names for FIFO LGBM (per-trade features extracted from fill sim output + signal features)
FIFO_FEATURE_NAMES = [
    # Signal features
    "signal_strength",
    "abs_signal_strength",
    "is_long",

    # Queue/book features
    "queue_position_at_post",
    "book_size_at_post",
    "queue_fraction",  # queue_pos / book_size

    # Time features
    "time_of_day",       # 0=open, 1=close
    "minutes_since_open",
    "is_first_30min",
    "is_last_30min",
    "is_lunch",          # 11:30-13:00 ET (typically quieter)

    # Fill timing
    "fill_latency_ms",
    "fill_latency_log",

    # Derived signal features
    "signal_x_queue",    # |signal| * (1/queue_pos) — strong signal + good queue
    "signal_x_book",     # |signal| * book_size — strong signal + thick book
]

N_FIFO_FEATURES = len(FIFO_FEATURE_NAMES)


def find_oot_dates() -> List[str]:
    """Find all dates that have both MBO raw data and CNN-Mamba predictions."""
    pred_dates = set()
    for f in PRED_DIR.glob("*_predictions.npz"):
        date = f.stem.replace("_predictions", "").replace("_oot_predictions", "")
        if len(date) == 8 and date.isdigit():
            pred_dates.add(date)

    # Also check fold files
    for f in PRED_DIR.glob("fold_*_oot_predictions.npz"):
        try:
            data = np.load(str(f), allow_pickle=True)
            if "date" in data:
                d = str(data["date"])
                if len(d) == 8:
                    pred_dates.add(d)
        except Exception:
            pass

    # Filter to dates that have MBO raw data
    valid_dates = []
    for d in sorted(pred_dates):
        mbo_patterns = [
            MBO_DIR / f"glbx-mdp3-{d}.mbo.dbn.zst",
            MBO_DIR / f"glbx-mdp3-{d}.mbo.dbn",
        ]
        for mp in mbo_patterns:
            if mp.exists():
                valid_dates.append(d)
                break

    return valid_dates


def find_mbo_file(date: str) -> Optional[Path]:
    """Find MBO raw data file for a date."""
    for ext in [".mbo.dbn.zst", ".mbo.dbn"]:
        p = MBO_DIR / f"glbx-mdp3-{date}{ext}"
        if p.exists():
            return p
    return None


def find_pred_file(date: str) -> Optional[Path]:
    """Find prediction file for a date."""
    for pattern in [f"{date}_predictions.npz", f"{date}_oot_predictions.npz"]:
        p = PRED_DIR / pattern
        if p.exists():
            return p
    return None


def run_fill_sim(date: str, config_name: str, config: dict, output_dir: Path) -> Optional[dict]:
    """Run Rust fill sim for one date with one config."""
    mbo_file = find_mbo_file(date)
    pred_file = find_pred_file(date)

    if not mbo_file or not pred_file:
        return None

    output_file = output_dir / f"{config_name}_{date}.json"
    if output_file.exists():
        try:
            return json.load(open(output_file))
        except Exception:
            pass

    cmd = [
        str(FILL_SIM_CLI),
        "--mbo-file", str(mbo_file),
        "--predictions", str(pred_file),
        "--output", str(output_file),
        "--quiet",
        "--signal-threshold", str(config.get("signal_threshold", 0.5)),
        "--take-profit-ticks", str(config.get("take_profit_ticks", 4)),
        "--stop-loss-ticks", str(config.get("stop_loss_ticks", 4)),
        "--hold-ms", str(config.get("hold_ms", 30000)),
        "--max-wait-bars", str(config.get("max_wait_bars", 10)),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0 and output_file.exists():
            return json.load(open(output_file))
        else:
            LOG.warning(f"  Fill sim failed for {date}: {result.stderr[:200]}")
            return None
    except Exception as e:
        LOG.warning(f"  Fill sim error for {date}: {e}")
        return None


def extract_trade_features(trade: dict) -> np.ndarray:
    """Extract features from a single fill sim trade."""
    sig = trade.get("signal_strength", 0.0)
    abs_sig = abs(sig)
    is_long = 1.0 if trade.get("side") == "BUY" else 0.0

    queue_pos = trade.get("queue_position_at_post", 1.0)
    book_size = trade.get("book_size_at_post", 1.0)
    queue_frac = queue_pos / max(book_size, 1.0)

    # Time from signal timestamp
    signal_ns = trade.get("signal_time_ns", 0)
    if signal_ns > 0:
        # Convert to seconds since midnight UTC, then to ET
        sec_in_day = (signal_ns / 1e9) % 86400
        et_sec = (sec_in_day - 4 * 3600) % 86400  # UTC-4 for ET
        rth_start = 9.5 * 3600  # 9:30 AM
        rth_end = 16.0 * 3600   # 4:00 PM
        tod = max(0, min(1, (et_sec - rth_start) / (rth_end - rth_start)))
        minutes = tod * 390
    else:
        tod = 0.5
        minutes = 195

    is_first_30 = 1.0 if minutes < 30 else 0.0
    is_last_30 = 1.0 if minutes > 360 else 0.0
    is_lunch = 1.0 if 120 < minutes < 210 else 0.0  # ~11:30-13:00

    fill_lat_ns = trade.get("fill_latency_ns", 0)
    fill_lat_ms = fill_lat_ns / 1e6
    fill_lat_log = np.log1p(fill_lat_ms)

    sig_x_queue = abs_sig * (1.0 / max(queue_pos, 0.1))
    sig_x_book = abs_sig * book_size

    return np.array([
        sig, abs_sig, is_long,
        queue_pos, book_size, queue_frac,
        tod, minutes, is_first_30, is_last_30, is_lunch,
        fill_lat_ms, fill_lat_log,
        sig_x_queue, sig_x_book,
    ], dtype=np.float32)


def collect_fifo_data(dates: List[str], config_name: str, config: dict,
                       output_dir: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Run fill sim and collect per-trade features and labels for all dates."""
    features_by_date = {}
    labels_by_date = {}  # 1=profitable, 0=not

    for i, date in enumerate(dates):
        LOG.info(f"  [{i+1}/{len(dates)}] {date}...")
        result = run_fill_sim(date, config_name, config, output_dir)

        if not result or not result.get("trades"):
            LOG.info(f"    No trades for {date}")
            continue

        trades = result["trades"]
        n_trades = len(trades)

        feats = np.zeros((n_trades, N_FIFO_FEATURES), dtype=np.float32)
        labs = np.zeros(n_trades, dtype=np.float32)

        for j, trade in enumerate(trades):
            feats[j] = extract_trade_features(trade)
            pnl = trade.get("pnl_ticks", 0.0)
            labs[j] = 1.0 if pnl > 0 else 0.0

        features_by_date[date] = feats
        labels_by_date[date] = labs
        LOG.info(f"    {n_trades} trades, WR={labs.mean():.3f}, avg_pnl={np.mean([t['pnl_ticks'] for t in trades]):.3f}")

    return features_by_date, labels_by_date


def train_fifo_lgbm_walkforward(
    features_by_date: Dict[str, np.ndarray],
    labels_by_date: Dict[str, np.ndarray],
    all_dates: List[str],
    n_train_days: int = 30,
    n_oot_days: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Walk-forward train LGBM to predict FIFO profitability."""

    available_dates = [d for d in all_dates if d in features_by_date and len(features_by_date[d]) > 0]

    if len(available_dates) < n_train_days + n_oot_days:
        LOG.error(f"Not enough dates with trades: {len(available_dates)} < {n_train_days + n_oot_days}")
        return np.array([]), np.array([]), np.array([])

    # Walk-forward folds
    folds = []
    start = 0
    while start + n_train_days + n_oot_days <= len(available_dates):
        train_dates = available_dates[start:start + n_train_days]
        oot_dates = available_dates[start + n_train_days:start + n_train_days + n_oot_days]
        folds.append((train_dates, oot_dates))
        start += n_oot_days

    LOG.info(f"Walk-forward: {len(folds)} folds ({n_train_days} train, {n_oot_days} OOT)")

    all_oot_preds = []
    all_oot_labels = []
    all_oot_features = []
    fold_aucs = []

    for fold_idx, (train_dates, oot_dates) in enumerate(folds):
        X_train = np.concatenate([features_by_date[d] for d in train_dates], axis=0)
        y_train = np.concatenate([labels_by_date[d] for d in train_dates])
        X_oot = np.concatenate([features_by_date[d] for d in oot_dates], axis=0)
        y_oot = np.concatenate([labels_by_date[d] for d in oot_dates])

        if len(X_train) < 100 or len(X_oot) < 10:
            LOG.warning(f"Fold {fold_idx}: too few samples ({len(X_train)} train, {len(X_oot)} OOT)")
            continue

        params = {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 5,
            "min_child_samples": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "verbose": -1,
            "n_jobs": -1,
            "seed": 42 + fold_idx,
        }

        dtrain = lgb.Dataset(X_train, label=y_train, feature_name=FIFO_FEATURE_NAMES)
        dval = lgb.Dataset(X_oot, label=y_oot, feature_name=FIFO_FEATURE_NAMES, reference=dtrain)

        model = lgb.train(
            params, dtrain, num_boost_round=300,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(999)],
        )

        preds = model.predict(X_oot)

        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(y_oot, preds)
        except ValueError:
            auc = 0.5

        fold_aucs.append(auc)
        all_oot_preds.append(preds)
        all_oot_labels.append(y_oot)
        all_oot_features.append(X_oot)

        LOG.info(f"  Fold {fold_idx}: Train={len(X_train):,} OOT={len(X_oot):,} "
                 f"AUC={auc:.4f} WR_train={y_train.mean():.3f} WR_oot={y_oot.mean():.3f}")

    if not all_oot_preds:
        return np.array([]), np.array([]), np.array([])

    all_preds = np.concatenate(all_oot_preds)
    all_labels = np.concatenate(all_oot_labels)
    all_feats = np.concatenate(all_oot_features)

    LOG.info(f"\nConcat AUC: {roc_auc_score(all_labels, all_preds):.4f}")
    LOG.info(f"Mean fold AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")

    return all_preds, all_labels, all_feats


def analyze_gated_results(
    all_results: Dict[str, List[dict]],  # config -> list of (date, trades)
    lgbm_scores: Optional[np.ndarray],
    lgbm_labels: Optional[np.ndarray],
    config_name: str,
):
    """Analyze FIFO results with and without LGBM gating."""

    LOG.info(f"\n{'='*70}")
    LOG.info(f"FIFO RESULTS: {config_name}")
    LOG.info(f"{'='*70}")

    # Flatten all trades
    all_trades = []
    for date, trades in all_results.items():
        for t in trades:
            t["date"] = date
            all_trades.append(t)

    if not all_trades:
        LOG.info("  No trades")
        return {}

    # Ungated results
    pnls = np.array([t["pnl_ticks"] for t in all_trades])
    n = len(pnls)
    wr = (pnls > 0).mean()
    avg_pnl = pnls.mean()
    total_pnl = pnls.sum()
    pf = abs(pnls[pnls > 0].sum() / pnls[pnls < 0].sum()) if (pnls < 0).any() and (pnls > 0).any() else 0

    # Daily P&L for Sortino
    daily_pnl = defaultdict(float)
    for t in all_trades:
        daily_pnl[t["date"]] += t["pnl_ticks"]
    daily_vals = np.array(list(daily_pnl.values()))
    neg_rets = daily_vals[daily_vals < 0]
    downside_std = np.sqrt(np.mean(neg_rets**2)) if len(neg_rets) > 0 else 1e-8
    sortino = daily_vals.mean() / downside_std if downside_std > 1e-8 else 0
    green_pct = (daily_vals > 0).mean()

    LOG.info(f"\n  UNGATED (all trades):")
    LOG.info(f"    Trades: {n:,}")
    LOG.info(f"    WR: {wr:.3f}")
    LOG.info(f"    Avg P&L: {avg_pnl:+.3f} ticks")
    LOG.info(f"    Total P&L: {total_pnl:+.1f} ticks (${total_pnl * 12.50:+,.0f})")
    LOG.info(f"    PF: {pf:.2f}")
    LOG.info(f"    Sortino: {sortino:.3f}")
    LOG.info(f"    Green days: {green_pct:.1%} ({int(green_pct * len(daily_vals))}/{len(daily_vals)})")
    LOG.info(f"    Avg Win: {pnls[pnls > 0].mean():+.2f}  Avg Loss: {pnls[pnls < 0].mean():+.2f}")

    results = {
        "ungated": {
            "n_trades": n, "wr": float(wr), "avg_pnl": float(avg_pnl),
            "total_pnl": float(total_pnl), "pf": float(pf), "sortino": float(sortino),
            "green_pct": float(green_pct),
        }
    }

    # Gate by signal strength (baseline)
    LOG.info(f"\n  SIGNAL-STRENGTH GATE (baseline):")
    abs_sigs = np.array([abs(t.get("signal_strength", 0)) for t in all_trades])

    for pct_name, pct in [("top_50pct", 50), ("top_20pct", 20), ("top_10pct", 10),
                           ("top_5pct", 5), ("top_1pct", 1)]:
        thresh = np.percentile(abs_sigs, 100 - pct)
        mask = abs_sigs >= thresh
        gated_pnls = pnls[mask]
        if len(gated_pnls) == 0:
            continue
        g_wr = (gated_pnls > 0).mean()
        g_avg = gated_pnls.mean()
        g_pf = abs(gated_pnls[gated_pnls > 0].sum() / gated_pnls[gated_pnls < 0].sum()) if (gated_pnls < 0).any() and (gated_pnls > 0).any() else 0
        LOG.info(f"    {pct_name}: n={len(gated_pnls):5d}  WR={g_wr:.3f}  PnL={g_avg:+.3f}t  PF={g_pf:.2f}")
        results[f"signal_{pct_name}"] = {
            "n": int(mask.sum()), "wr": float(g_wr), "avg_pnl": float(g_avg), "pf": float(g_pf)
        }

    # Gate by LGBM score (if available)
    if lgbm_scores is not None and len(lgbm_scores) == len(all_trades):
        LOG.info(f"\n  LGBM-GATED (trained on FIFO outcomes):")
        for pct_name, pct in [("top_50pct", 50), ("top_20pct", 20), ("top_10pct", 10),
                               ("top_5pct", 5), ("top_1pct", 1)]:
            thresh = np.percentile(lgbm_scores, 100 - pct)
            mask = lgbm_scores >= thresh
            gated_pnls = pnls[mask]
            if len(gated_pnls) == 0:
                continue
            g_wr = (gated_pnls > 0).mean()
            g_avg = gated_pnls.mean()
            g_pf = abs(gated_pnls[gated_pnls > 0].sum() / gated_pnls[gated_pnls < 0].sum()) if (gated_pnls < 0).any() and (gated_pnls > 0).any() else 0
            LOG.info(f"    {pct_name}: n={len(gated_pnls):5d}  WR={g_wr:.3f}  PnL={g_avg:+.3f}t  PF={g_pf:.2f}")
            results[f"lgbm_{pct_name}"] = {
                "n": int(mask.sum()), "wr": float(g_wr), "avg_pnl": float(g_avg), "pf": float(g_pf)
            }

    # Gate by queue position (microstructure)
    LOG.info(f"\n  QUEUE-POSITION GATE:")
    queue_pos = np.array([t.get("queue_position_at_post", 999) for t in all_trades])
    for max_q in [1, 2, 3, 5, 10]:
        mask = queue_pos <= max_q
        gated_pnls = pnls[mask]
        if len(gated_pnls) == 0:
            continue
        g_wr = (gated_pnls > 0).mean()
        g_avg = gated_pnls.mean()
        g_pf = abs(gated_pnls[gated_pnls > 0].sum() / gated_pnls[gated_pnls < 0].sum()) if (gated_pnls < 0).any() and (gated_pnls > 0).any() else 0
        LOG.info(f"    queue<={max_q}: n={len(gated_pnls):5d}  WR={g_wr:.3f}  PnL={g_avg:+.3f}t  PF={g_pf:.2f}")
        results[f"queue_le_{max_q}"] = {
            "n": int(mask.sum()), "wr": float(g_wr), "avg_pnl": float(g_avg), "pf": float(g_pf)
        }

    # Combined: signal + queue
    LOG.info(f"\n  COMBINED GATES (signal + queue):")
    for pct in [10, 5, 1]:
        sig_thresh = np.percentile(abs_sigs, 100 - pct)
        for max_q in [2, 5]:
            mask = (abs_sigs >= sig_thresh) & (queue_pos <= max_q)
            gated_pnls = pnls[mask]
            if len(gated_pnls) < 5:
                continue
            g_wr = (gated_pnls > 0).mean()
            g_avg = gated_pnls.mean()
            g_pf = abs(gated_pnls[gated_pnls > 0].sum() / gated_pnls[gated_pnls < 0].sum()) if (gated_pnls < 0).any() and (gated_pnls > 0).any() else 0
            LOG.info(f"    top{pct}%+queue<={max_q}: n={len(gated_pnls):5d}  WR={g_wr:.3f}  PnL={g_avg:+.3f}t  PF={g_pf:.2f}")
            results[f"sig_top{pct}_q{max_q}"] = {
                "n": int(mask.sum()), "wr": float(g_wr), "avg_pnl": float(g_avg), "pf": float(g_pf)
            }

    # Time-of-day analysis
    LOG.info(f"\n  TIME-OF-DAY ANALYSIS:")
    tods = np.array([extract_trade_features(t)[6] for t in all_trades])  # time_of_day index
    for period_name, start_min, end_min in [
        ("open_30min", 0, 30), ("mid_morning", 30, 120),
        ("lunch", 120, 210), ("afternoon", 210, 360), ("close_30min", 360, 390)
    ]:
        minutes_arr = tods * 390
        mask = (minutes_arr >= start_min) & (minutes_arr < end_min)
        gated_pnls = pnls[mask]
        if len(gated_pnls) < 10:
            continue
        g_wr = (gated_pnls > 0).mean()
        g_avg = gated_pnls.mean()
        LOG.info(f"    {period_name:15s}: n={len(gated_pnls):5d}  WR={g_wr:.3f}  PnL={g_avg:+.3f}t")
        results[f"time_{period_name}"] = {
            "n": int(mask.sum()), "wr": float(g_wr), "avg_pnl": float(g_avg)
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="LGBM-Gated FIFO Validation")
    parser.add_argument("--output-dir", type=str, default=str(LVL3_ROOT / "output/lgbm_fifo_gate_v1"))
    parser.add_argument("--n-workers", type=int, default=4)
    parser.add_argument("--n-train-days", type=int, default=25)
    parser.add_argument("--n-oot-days", type=int, default=3)
    parser.add_argument("--configs", type=str, nargs="+",
                        default=list(FILL_SIM_CONFIGS.keys()),
                        help="Which fill sim configs to test")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(output_dir) + ".log", mode="w"),
        ],
    )

    LOG.info("=" * 70)
    LOG.info("LGBM-Gated FIFO Validation Test")
    LOG.info("=" * 70)
    LOG.info(f"Output: {output_dir}")
    LOG.info(f"Fill sim: {FILL_SIM_CLI}")
    LOG.info(f"MBO dir: {MBO_DIR}")
    LOG.info(f"Pred dir: {PRED_DIR}")

    # MLflow tracking
    if HAS_MLFLOW:
        mlflow.set_tracking_uri("http://jupiter:5000")
        mlflow.set_experiment("lgbm_fifo_gate")
        mlflow.start_run(run_name="lgbm_fifo_gate_v1")

    # Find available dates
    dates = find_oot_dates()
    LOG.info(f"\nFound {len(dates)} dates with MBO + predictions")
    LOG.info(f"Date range: {dates[0]} - {dates[-1]}")

    all_config_results = {}

    for config_name in args.configs:
        if config_name not in FILL_SIM_CONFIGS:
            LOG.warning(f"Unknown config: {config_name}")
            continue

        config = FILL_SIM_CONFIGS[config_name]
        LOG.info(f"\n{'#'*70}")
        LOG.info(f"CONFIG: {config_name}")
        LOG.info(f"{'#'*70}")
        LOG.info(f"  Params: {json.dumps(config, indent=2)}")

        sim_dir = output_dir / "fill_sim_results"
        sim_dir.mkdir(exist_ok=True)

        # Step 1: Run fill sim for all dates
        LOG.info(f"\nStep 1: Running FIFO fill sim for {len(dates)} dates...")
        date_trades = {}
        for date in dates:
            result = run_fill_sim(date, config_name, config, sim_dir)
            if result and result.get("trades"):
                date_trades[date] = result["trades"]

        available_dates = sorted(date_trades.keys())
        LOG.info(f"  Dates with trades: {len(available_dates)}/{len(dates)}")
        total_trades = sum(len(t) for t in date_trades.values())
        LOG.info(f"  Total trades: {total_trades:,}")

        if total_trades < 100:
            LOG.warning(f"  Too few trades for analysis")
            continue

        # Step 2: Extract features and train LGBM on FIFO outcomes
        LOG.info(f"\nStep 2: Extracting FIFO features...")
        features_by_date = {}
        labels_by_date = {}

        for date in available_dates:
            trades = date_trades[date]
            feats = np.array([extract_trade_features(t) for t in trades], dtype=np.float32)
            labs = np.array([1.0 if t["pnl_ticks"] > 0 else 0.0 for t in trades], dtype=np.float32)
            features_by_date[date] = feats
            labels_by_date[date] = labs

        LOG.info(f"\nStep 3: Training walk-forward LGBM on FIFO outcomes...")
        lgbm_scores, lgbm_labels, lgbm_features = train_fifo_lgbm_walkforward(
            features_by_date, labels_by_date, available_dates,
            n_train_days=args.n_train_days, n_oot_days=args.n_oot_days,
        )

        # Step 4: Analyze results with various gates
        LOG.info(f"\nStep 4: Analyzing gated results...")

        # For analysis, we need to align lgbm_scores with trades
        # The walk-forward only covers a subset of dates (after warm-up period)
        # For ungated analysis, use ALL dates
        results = analyze_gated_results(
            date_trades,
            lgbm_scores if len(lgbm_scores) > 0 else None,
            lgbm_labels if len(lgbm_labels) > 0 else None,
            config_name,
        )

        all_config_results[config_name] = results

        if HAS_MLFLOW:
            for key, val in results.items():
                if isinstance(val, dict):
                    for k, v in val.items():
                        mlflow.log_metric(f"{config_name}_{key}_{k}", v)

    # Save all results
    results_file = output_dir / "all_results.json"
    with open(results_file, "w") as f:
        json.dump(all_config_results, f, indent=2)
    LOG.info(f"\nAll results saved to {results_file}")

    # Final summary
    LOG.info(f"\n{'='*70}")
    LOG.info(f"FINAL SUMMARY")
    LOG.info(f"{'='*70}")

    for cfg_name, results in all_config_results.items():
        LOG.info(f"\n{cfg_name}:")
        ungated = results.get("ungated", {})
        LOG.info(f"  Ungated: {ungated.get('n_trades',0):,} trades, WR={ungated.get('wr',0):.3f}, "
                 f"PnL={ungated.get('avg_pnl',0):+.3f}t, PF={ungated.get('pf',0):.2f}, "
                 f"Sortino={ungated.get('sortino',0):.3f}")

        # Show best gated result
        best_key = None
        best_pnl = -999
        for key, val in results.items():
            if key == "ungated":
                continue
            if isinstance(val, dict) and val.get("avg_pnl", -999) > best_pnl and val.get("n", 0) > 20:
                best_pnl = val["avg_pnl"]
                best_key = key

        if best_key:
            bv = results[best_key]
            LOG.info(f"  Best gate [{best_key}]: n={bv.get('n',0):,}, WR={bv.get('wr',0):.3f}, "
                     f"PnL={bv.get('avg_pnl',0):+.3f}t, PF={bv.get('pf',0):.2f}")

    if HAS_MLFLOW:
        mlflow.end_run()

    LOG.info("\nDone!")


if __name__ == "__main__":
    main()
