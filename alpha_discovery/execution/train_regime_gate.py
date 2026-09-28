#!/usr/bin/env python3
"""
Regime Gate — Find WHEN to trade, not WHAT to trade
=====================================================

KEY INSIGHT FROM LGBM v1:
- Signal confidence has ~0 correlation with individual trade outcomes
- BUT time_of_day, volatility, and signal persistence ARE predictive
- This means the signal works in SOME regimes and not others

APPROACH:
Instead of predicting individual trade outcomes, cluster the data into regimes
and find which regimes have positive edge. Then build a simple gate:
"Is the current microstructure state in a favorable regime?"

TWO MODELS:
1. EXTREME CLASSIFIER: Train on top 10% best outcomes vs bottom 10% worst
   outcomes (remove the noisy middle). What features separate clearly
   profitable trades from clearly unprofitable ones?

2. REGIME ANALYSIS: Bin trades by time-of-day × volatility × signal-persistence
   and find which bins have positive expected P&L after commission. Output a
   lookup table for the paper trader.

Walk-forward validated, FIFO-based, commission=0.376tk.

Author: Claude (Infrastructure Builder)
Date: 2026-05-08
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

try:
    import lightgbm as lgb
except ImportError:
    print("ERROR: lightgbm not installed")
    sys.exit(1)

# Import shared feature extraction from lgbm_exec
from alpha_discovery.execution.train_lgbm_exec import (
    FEATURE_NAMES, N_FEATURES, HORIZONS, TARGET_NAMES_PER_HORIZON,
    COMMISSION_TICKS, PRED_STRIDE, PRED_WINDOW,
    extract_features_for_date, build_date_pred_index,
    _extract_one_date,
)

LOG = logging.getLogger("REGIME_GATE")
LVL3_ROOT = Path(os.environ.get("LVL3_ROOT", "/home/nick/Lvl3Quant"))


def load_all_data(data_dir, pred_dir, n_workers=8):
    """Load features and targets for all available dates."""
    pred_index = build_date_pred_index(pred_dir)
    LOG.info(f"Prediction index: {len(pred_index)} dates")

    mbo_files = {}
    for d in sorted(data_dir.glob("*_mbo_events.npz")):
        date = d.stem.replace("_mbo_events", "")
        if date in pred_index:
            mbo_files[date] = d

    LOG.info(f"Dates with both MBO + predictions: {len(mbo_files)}")

    tasks = [
        (str(mbo_files[d]), str(pred_index[d]), PRED_STRIDE, PRED_WINDOW)
        for d in sorted(mbo_files.keys())
    ]

    features_by_date = {}
    targets_by_date = {}
    meta_by_date = {}
    all_dates = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_extract_one_date, t): t for t in tasks}
        for fut in as_completed(futures):
            result = fut.result()
            date = result["date"]
            if result["status"] == "ok" and result["n_samples"] > 0:
                features_by_date[date] = result["features"]
                targets_by_date[date] = result["targets"]
                meta_by_date[date] = result["meta"]
                all_dates.append(date)
                LOG.info(f"  {date}: {result['n_samples']:,} samples")
            elif result["n_samples"] == 0:
                LOG.warning(f"  {date}: 0 samples")

    all_dates.sort()
    total = sum(len(features_by_date[d]) for d in all_dates)
    LOG.info(f"Total: {total:,} samples from {len(all_dates)} dates")
    return features_by_date, targets_by_date, meta_by_date, all_dates


def run_extreme_classifier(
    features_by_date, targets_by_date, meta_by_date, all_dates,
    n_train_days=30, n_oot_days=5, output_dir=None,
    extreme_pct=10, horizon="10s",
):
    """
    EXTREME CLASSIFIER: Only train on clearly good vs clearly bad trades.

    Remove the noisy middle and see if LightGBM can separate extremes.
    If it can → the features that separate them are regime indicators.
    """
    LOG.info(f"\n{'='*70}")
    LOG.info(f"EXTREME CLASSIFIER (top/bottom {extreme_pct}% at {horizon})")
    LOG.info(f"{'='*70}")

    n_dates = len(all_dates)
    if n_dates < n_train_days + n_oot_days:
        LOG.error(f"Not enough dates: {n_dates} < {n_train_days + n_oot_days}")
        return {}

    # Generate walk-forward folds
    folds = []
    start = 0
    while start + n_train_days + n_oot_days <= n_dates:
        train_dates = all_dates[start:start + n_train_days]
        oot_dates = all_dates[start + n_train_days:start + n_train_days + n_oot_days]
        folds.append((train_dates, oot_dates))
        start += n_oot_days

    LOG.info(f"Walk-forward: {len(folds)} folds")

    all_oot_X = []
    all_oot_y = []
    all_oot_preds = []
    all_oot_pnl = []
    all_oot_meta = []

    pnl_key = f"pnl_{horizon}"

    for fold_idx, (train_dates, oot_dates) in enumerate(folds):
        LOG.info(f"\nFOLD {fold_idx}: Train [{train_dates[0]}..{train_dates[-1]}] "
                 f"Eval [{oot_dates[0]}..{oot_dates[-1]}]")

        # Training data
        X_train_all = np.concatenate([features_by_date[d] for d in train_dates if d in features_by_date])
        y_train_pnl = np.concatenate([targets_by_date[d][pnl_key] for d in train_dates if d in targets_by_date])

        # Select extremes for training
        top_thresh = np.percentile(y_train_pnl, 100 - extreme_pct)
        bot_thresh = np.percentile(y_train_pnl, extreme_pct)

        top_mask = y_train_pnl >= top_thresh
        bot_mask = y_train_pnl <= bot_thresh

        X_train = np.concatenate([X_train_all[top_mask], X_train_all[bot_mask]])
        y_train = np.concatenate([np.ones(top_mask.sum()), np.zeros(bot_mask.sum())])

        # Shuffle
        shuffle_idx = np.random.RandomState(42 + fold_idx).permutation(len(X_train))
        X_train = X_train[shuffle_idx]
        y_train = y_train[shuffle_idx]

        LOG.info(f"  Extreme training: {len(X_train):,} samples "
                 f"({top_mask.sum():,} top + {bot_mask.sum():,} bottom), "
                 f"thresholds=[{bot_thresh:+.2f}, {top_thresh:+.2f}]")

        # OOT data - evaluate on ALL samples (not just extremes)
        X_oot = np.concatenate([features_by_date[d] for d in oot_dates if d in features_by_date])
        y_oot_pnl = np.concatenate([targets_by_date[d][pnl_key] for d in oot_dates if d in targets_by_date])
        meta_oot = np.concatenate([meta_by_date[d] for d in oot_dates if d in meta_by_date])

        if len(X_oot) == 0:
            continue

        # Train LightGBM
        params = {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.03,
            "num_leaves": 31,
            "max_depth": 5,
            "min_child_samples": 500,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "reg_alpha": 0.5,
            "reg_lambda": 2.0,
            "verbose": -1,
            "n_jobs": 8,
            "seed": 42 + fold_idx,
            "is_unbalance": True,
        }

        dtrain = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)

        # Use 20% of training as validation
        n_val = len(X_train) // 5
        dval = lgb.Dataset(X_train[-n_val:], label=y_train[-n_val:],
                          feature_name=FEATURE_NAMES, reference=dtrain)

        model = lgb.train(
            params, dtrain, num_boost_round=300,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(100)],
        )

        # Predict on ALL OOT samples
        preds = model.predict(X_oot)

        # Feature importance
        imp = dict(zip(FEATURE_NAMES, model.feature_importance(importance_type="gain")))
        sorted_imp = sorted(imp.items(), key=lambda x: -x[1])
        LOG.info(f"  Feature importance (top 10):")
        for fname, fval in sorted_imp[:10]:
            LOG.info(f"    {fname:30s}: {fval:.1f}")

        # Evaluate: use the extreme-trained model as a GATE on all OOT trades
        LOG.info(f"\n  OOT evaluation (model as trade gate):")
        LOG.info(f"  {'Thresh':>7s}  {'Select%':>8s}  {'Trades':>7s}  {'WR':>6s}  {'AvgPnL':>8s}  {'TotalPnL':>10s}  {'PF':>6s}")

        prof_actual = (y_oot_pnl > 0).astype(float)

        for thresh in [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
            selected = preds >= thresh
            n_sel = int(selected.sum())
            if n_sel > 10:
                sel_pct = n_sel / len(X_oot) * 100
                wr = float(prof_actual[selected].mean())
                avg_pnl = float(y_oot_pnl[selected].mean())
                total_pnl = float(y_oot_pnl[selected].sum())
                winners = y_oot_pnl[selected] > 0
                losers = y_oot_pnl[selected] <= 0
                gp = float(y_oot_pnl[selected][winners].sum()) if winners.any() else 0
                gl = abs(float(y_oot_pnl[selected][losers].sum())) if losers.any() else 1e-8
                pf = gp / (gl + 1e-8)
                LOG.info(f"    {thresh:5.2f}    {sel_pct:6.1f}%  {n_sel:7,}  {wr:.3f}  {avg_pnl:+7.3f}tk  {total_pnl:+9.1f}tk  {pf:.2f}")

        # Compare: what's the P&L if we DON'T use any gate?
        ungated_wr = float(prof_actual.mean())
        ungated_pnl = float(y_oot_pnl.mean())
        LOG.info(f"  UNGATED: {len(X_oot):,} trades, WR={ungated_wr:.3f}, AvgPnL={ungated_pnl:+.3f}tk")

        all_oot_X.append(X_oot)
        all_oot_y.append(prof_actual)
        all_oot_preds.append(preds)
        all_oot_pnl.append(y_oot_pnl)
        all_oot_meta.append(meta_oot)

        # Save model
        if output_dir:
            fold_dir = output_dir / f"extreme_fold_{fold_idx}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            model.save_model(str(fold_dir / f"model_extreme_{horizon}.txt"))

    # Concat results
    if all_oot_X:
        X_all = np.concatenate(all_oot_X)
        y_all = np.concatenate(all_oot_y)
        preds_all = np.concatenate(all_oot_preds)
        pnl_all = np.concatenate(all_oot_pnl)

        LOG.info(f"\n{'='*60}")
        LOG.info(f"CONCAT EXTREME CLASSIFIER RESULTS ({len(folds)} folds, {len(X_all):,} OOT samples)")
        LOG.info(f"{'='*60}")
        LOG.info(f"  {'Thresh':>7s}  {'Select%':>8s}  {'Trades':>7s}  {'WR':>6s}  {'AvgPnL':>8s}  {'TotalPnL':>10s}  {'PF':>6s}")

        for thresh in [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
            selected = preds_all >= thresh
            n_sel = int(selected.sum())
            if n_sel > 10:
                sel_pct = n_sel / len(X_all) * 100
                wr = float(y_all[selected].mean())
                avg_pnl = float(pnl_all[selected].mean())
                total_pnl = float(pnl_all[selected].sum())
                winners = pnl_all[selected] > 0
                losers = pnl_all[selected] <= 0
                gp = float(pnl_all[selected][winners].sum()) if winners.any() else 0
                gl = abs(float(pnl_all[selected][losers].sum())) if losers.any() else 1e-8
                pf = gp / (gl + 1e-8)
                LOG.info(f"    {thresh:5.2f}    {sel_pct:6.1f}%  {n_sel:7,}  {wr:.3f}  {avg_pnl:+7.3f}tk  {total_pnl:+9.1f}tk  {pf:.2f}")

        ungated_wr = float(y_all.mean())
        ungated_pnl = float(pnl_all.mean())
        LOG.info(f"  UNGATED: {len(X_all):,} trades, WR={ungated_wr:.3f}, AvgPnL={ungated_pnl:+.3f}tk")

    return {"n_folds": len(folds)}


def run_regime_analysis(
    features_by_date, targets_by_date, meta_by_date, all_dates,
    n_train_days=30, n_oot_days=5, output_dir=None,
    horizon="10s",
):
    """
    REGIME ANALYSIS: Bin trades by key features and find profitable regimes.

    Key features (from LGBM importance):
    1. time_of_day
    2. pred_ratio_10s_1s (signal persistence)
    3. recent_volatility
    4. signal_agreement
    5. abs_pred_1s (confidence)
    6. conf_x_imbalance (confidence aligned with book)
    """
    LOG.info(f"\n{'='*70}")
    LOG.info(f"REGIME ANALYSIS — Finding profitable trading conditions ({horizon})")
    LOG.info(f"{'='*70}")

    # Combine all data for regime discovery
    X_all = np.concatenate([features_by_date[d] for d in all_dates if d in features_by_date])
    pnl_key = f"pnl_{horizon}"
    pnl_all = np.concatenate([targets_by_date[d][pnl_key] for d in all_dates if d in targets_by_date])
    prof_key = f"profitable_{horizon}"
    prof_all = np.concatenate([targets_by_date[d][prof_key] for d in all_dates if d in targets_by_date])

    LOG.info(f"Total samples: {len(X_all):,}")
    LOG.info(f"Overall WR: {prof_all.mean():.3f}, AvgPnL: {pnl_all.mean():+.3f}tk")

    # Feature indices
    idx = {name: i for i, name in enumerate(FEATURE_NAMES)}

    # 1. Time-of-day analysis (30-minute buckets)
    LOG.info(f"\n--- TIME-OF-DAY ANALYSIS ---")
    tod = X_all[:, idx["time_of_day"]]
    tod_bins = np.digitize(tod, np.linspace(0, 1, 14))  # ~30 min buckets
    LOG.info(f"  {'Period':>12s}  {'N':>7s}  {'WR':>6s}  {'AvgPnL':>8s}  {'TotalPnL':>10s}  {'PF':>6s}")
    for b in sorted(np.unique(tod_bins)):
        mask = tod_bins == b
        n = mask.sum()
        if n > 100:
            wr = prof_all[mask].mean()
            avg_pnl = pnl_all[mask].mean()
            total_pnl = pnl_all[mask].sum()
            w = pnl_all[mask] > 0
            l = pnl_all[mask] <= 0
            gp = pnl_all[mask][w].sum() if w.any() else 0
            gl = abs(pnl_all[mask][l].sum()) if l.any() else 1e-8
            pf = gp / (gl + 1e-8)
            time_start = 9.5 + b * 0.5
            h = int(time_start)
            m = int((time_start - h) * 60)
            LOG.info(f"  {h:02d}:{m:02d}         {n:7,}  {wr:.3f}  {avg_pnl:+7.3f}tk  {total_pnl:+9.1f}tk  {pf:.2f}")

    # 2. Volatility regime analysis
    LOG.info(f"\n--- VOLATILITY REGIME ANALYSIS ---")
    vol = X_all[:, idx["recent_volatility_200"]]
    vol_percentiles = [0, 20, 40, 60, 80, 90, 95, 100]
    vol_thresholds = np.percentile(vol, vol_percentiles)
    LOG.info(f"  {'Vol Pctile':>12s}  {'N':>7s}  {'WR':>6s}  {'AvgPnL':>8s}  {'TotalPnL':>10s}  {'AvgVol':>8s}")
    for i in range(len(vol_percentiles) - 1):
        mask = (vol >= vol_thresholds[i]) & (vol < vol_thresholds[i+1] + 1e-8)
        if i == len(vol_percentiles) - 2:
            mask = vol >= vol_thresholds[i]
        n = mask.sum()
        if n > 100:
            wr = prof_all[mask].mean()
            avg_pnl = pnl_all[mask].mean()
            total_pnl = pnl_all[mask].sum()
            avg_vol = vol[mask].mean()
            label = f"{vol_percentiles[i]}-{vol_percentiles[i+1]}%"
            LOG.info(f"  {label:>12s}  {n:7,}  {wr:.3f}  {avg_pnl:+7.3f}tk  {total_pnl:+9.1f}tk  {avg_vol:7.4f}")

    # 3. Signal persistence analysis
    LOG.info(f"\n--- SIGNAL PERSISTENCE ANALYSIS ---")
    persistence = X_all[:, idx["pred_ratio_10s_1s"]]
    pers_bins = [(-100, 0), (0, 0.3), (0.3, 0.6), (0.6, 0.8), (0.8, 1.0), (1.0, 1.5), (1.5, 100)]
    LOG.info(f"  {'Ratio 10s/1s':>14s}  {'N':>7s}  {'WR':>6s}  {'AvgPnL':>8s}  {'TotalPnL':>10s}")
    for lo, hi in pers_bins:
        mask = (persistence >= lo) & (persistence < hi)
        n = mask.sum()
        if n > 100:
            wr = prof_all[mask].mean()
            avg_pnl = pnl_all[mask].mean()
            total_pnl = pnl_all[mask].sum()
            label = f"[{lo:.1f},{hi:.1f})"
            LOG.info(f"  {label:>14s}  {n:7,}  {wr:.3f}  {avg_pnl:+7.3f}tk  {total_pnl:+9.1f}tk")

    # 4. Signal agreement analysis
    LOG.info(f"\n--- SIGNAL AGREEMENT ANALYSIS ---")
    agreement = X_all[:, idx["signal_agreement"]]
    for agr_val, agr_label in [(0.0, "Disagree"), (1.0, "Agree")]:
        mask = agreement == agr_val
        n = mask.sum()
        if n > 100:
            wr = prof_all[mask].mean()
            avg_pnl = pnl_all[mask].mean()
            total_pnl = pnl_all[mask].sum()
            LOG.info(f"  {agr_label:>12s}  {n:7,}  WR={wr:.3f}  AvgPnL={avg_pnl:+.3f}tk  TotalPnL={total_pnl:+.1f}tk")

    # 5. Confidence + alignment interaction
    LOG.info(f"\n--- CONFIDENCE × BOOK ALIGNMENT ---")
    conf = X_all[:, idx["abs_pred_1s"]]
    aligned = X_all[:, idx["signal_aligned_with_imbalance"]]

    conf_bins = [(0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 100)]
    for lo, hi in conf_bins:
        for align_val, align_label in [(0.0, "Against"), (1.0, "With")]:
            mask = (conf >= lo) & (conf < hi) & (aligned == align_val)
            n = mask.sum()
            if n > 100:
                wr = prof_all[mask].mean()
                avg_pnl = pnl_all[mask].mean()
                label = f"Conf[{lo:.1f},{hi:.1f}) {align_label}"
                LOG.info(f"  {label:>30s}  n={n:7,}  WR={wr:.3f}  AvgPnL={avg_pnl:+.3f}tk")

    # 6. Combined regime: high confidence + agreeing horizons + aligned with book
    LOG.info(f"\n--- BEST REGIME CANDIDATES ---")
    conf_thresh_list = [0.1, 0.25, 0.5, 0.75, 1.0]
    for ct in conf_thresh_list:
        # High confidence + agreeing + aligned with book
        mask_full = (conf >= ct) & (agreement == 1.0) & (aligned == 1.0)
        # High confidence + agreeing only
        mask_agree = (conf >= ct) & (agreement == 1.0)
        # High confidence only
        mask_conf = conf >= ct

        for mask, label in [
            (mask_conf, f"Conf≥{ct:.2f}"),
            (mask_agree, f"Conf≥{ct:.2f}+Agree"),
            (mask_full, f"Conf≥{ct:.2f}+Agree+AlignBook"),
        ]:
            n = mask.sum()
            if n > 50:
                wr = prof_all[mask].mean()
                avg_pnl = pnl_all[mask].mean()
                total_pnl = pnl_all[mask].sum()
                w = pnl_all[mask] > 0
                l = pnl_all[mask] <= 0
                gp = pnl_all[mask][w].sum() if w.any() else 0
                gl = abs(pnl_all[mask][l].sum()) if l.any() else 1e-8
                pf = gp / (gl + 1e-8)
                LOG.info(f"  {label:>35s}  n={n:6,}  WR={wr:.3f}  AvgPnL={avg_pnl:+.3f}  PF={pf:.2f}")

    # 7. Side-specific regime analysis
    LOG.info(f"\n--- SIDE-SPECIFIC REGIME ANALYSIS ---")
    sig_dir = X_all[:, idx["signal_direction"]]
    for side_val, side_label in [(1.0, "LONG"), (-1.0, "SHORT")]:
        side_mask = sig_dir == side_val
        LOG.info(f"\n  {side_label}:")
        for ct in [0.25, 0.5, 0.75, 1.0]:
            for agr in [None, 1.0]:
                mask = side_mask & (conf >= ct)
                label = f"Conf≥{ct:.2f}"
                if agr is not None:
                    mask = mask & (agreement == agr)
                    label += "+Agree"
                n = mask.sum()
                if n > 50:
                    wr = prof_all[mask].mean()
                    avg_pnl = pnl_all[mask].mean()
                    total_pnl = pnl_all[mask].sum()
                    LOG.info(f"    {label:>25s}  n={n:6,}  WR={wr:.3f}  AvgPnL={avg_pnl:+.3f}  Total={total_pnl:+.1f}tk")

    # Save regime analysis as JSON lookup table
    if output_dir:
        regime_table = {}
        # Build lookup: (tod_bucket, vol_bucket, conf_bucket) -> stats
        tod_edges = np.linspace(0, 1, 14)
        vol_edges = np.percentile(vol[vol > 0], [0, 25, 50, 75, 90, 100])
        conf_edges = [0, 0.1, 0.25, 0.5, 1.0, 100]

        tod_b = np.digitize(tod, tod_edges)
        vol_b = np.digitize(vol, vol_edges)
        conf_b = np.digitize(conf, conf_edges)

        for tb in range(1, len(tod_edges) + 1):
            for vb in range(1, len(vol_edges)):
                for cb in range(1, len(conf_edges)):
                    mask = (tod_b == tb) & (vol_b == vb) & (conf_b == cb)
                    n = mask.sum()
                    if n > 30:
                        wr = float(prof_all[mask].mean())
                        avg_pnl = float(pnl_all[mask].mean())
                        key = f"tod{tb}_vol{vb}_conf{cb}"
                        regime_table[key] = {
                            "n": int(n), "wr": round(wr, 3),
                            "avg_pnl": round(avg_pnl, 3),
                            "profitable": avg_pnl > 0,
                        }

        with open(output_dir / "regime_lookup.json", "w") as f:
            json.dump(regime_table, f, indent=2)
        LOG.info(f"\nRegime lookup saved: {len(regime_table)} cells")

        # Count profitable regimes
        n_prof = sum(1 for v in regime_table.values() if v["profitable"])
        n_total = len(regime_table)
        LOG.info(f"Profitable regimes: {n_prof}/{n_total} ({n_prof/max(n_total,1)*100:.0f}%)")


def main():
    parser = argparse.ArgumentParser(description="Regime Gate Analysis")
    parser.add_argument("--data-dir", type=str,
                       default=str(LVL3_ROOT / "data/processed/mbo_events_smart_v3"))
    parser.add_argument("--pred-dir", type=str,
                       default=str(LVL3_ROOT / "output/cnn_mamba_v2_all_oot"))
    parser.add_argument("--output-dir", type=str,
                       default=str(LVL3_ROOT / "output/regime_gate_v1"))
    parser.add_argument("--n-train-days", type=int, default=30)
    parser.add_argument("--n-oot-days", type=int, default=5)
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--horizon", type=str, default="10s",
                       choices=["1s", "5s", "10s"])
    parser.add_argument("--extreme-pct", type=int, default=10)
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
    LOG.info("REGIME GATE ANALYSIS")
    LOG.info("=" * 70)

    data_dir = Path(args.data_dir)
    pred_dir = Path(args.pred_dir)

    features_by_date, targets_by_date, meta_by_date, all_dates = load_all_data(
        data_dir, pred_dir, args.n_workers,
    )

    if not all_dates:
        LOG.error("No data loaded!")
        return

    # Run both analyses
    # 1. Extreme classifier (walk-forward)
    run_extreme_classifier(
        features_by_date, targets_by_date, meta_by_date, all_dates,
        n_train_days=args.n_train_days,
        n_oot_days=args.n_oot_days,
        output_dir=output_dir,
        extreme_pct=args.extreme_pct,
        horizon=args.horizon,
    )

    # Also run for 1s and 5s horizons
    for h in ["1s", "5s"]:
        if h != args.horizon:
            run_extreme_classifier(
                features_by_date, targets_by_date, meta_by_date, all_dates,
                n_train_days=args.n_train_days,
                n_oot_days=args.n_oot_days,
                output_dir=output_dir,
                extreme_pct=args.extreme_pct,
                horizon=h,
            )

    # 2. Regime analysis (full data, no walk-forward — this is descriptive stats)
    for h in HORIZONS:
        run_regime_analysis(
            features_by_date, targets_by_date, meta_by_date, all_dates,
            n_train_days=args.n_train_days,
            n_oot_days=args.n_oot_days,
            output_dir=output_dir,
            horizon=h,
        )

    LOG.info("\n" + "=" * 70)
    LOG.info("REGIME GATE ANALYSIS COMPLETE")
    LOG.info("=" * 70)


if __name__ == "__main__":
    main()
