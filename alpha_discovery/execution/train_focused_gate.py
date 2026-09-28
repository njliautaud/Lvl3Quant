#!/usr/bin/env python3
"""
Focused Gate Experiment — Validate thin positive edges with more WF folds
==========================================================================

From regime_gate_v1, we found thin positive edges in:
1. Extreme classifier at 1s/5s horizons (very selective)
2. Combined regime: high confidence + signal agreement + book alignment
3. Long side outperforms short side

This experiment does exhaustive walk-forward validation with DAILY sliding
(not 5-day OOT blocks) to get maximum statistical power.

Two approaches:
A) PURE RULES: No ML model. Just apply regime rules and measure OOT P&L.
   Tests: confidence thresholds × side × agreement × persistence × time_of_day
B) LGBM GATE with daily WF: train on N days, predict day N+1, slide.
   Maximum number of OOT points for statistical significance.

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
from scipy.stats import spearmanr

try:
    import lightgbm as lgb
except ImportError:
    print("ERROR: lightgbm not installed")
    sys.exit(1)

from alpha_discovery.execution.train_lgbm_exec import (
    FEATURE_NAMES, N_FEATURES, HORIZONS,
    COMMISSION_TICKS, PRED_STRIDE, PRED_WINDOW,
    extract_features_for_date, build_date_pred_index,
    _extract_one_date,
)

LOG = logging.getLogger("FOCUSED_GATE")
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

    all_dates.sort()
    total = sum(len(features_by_date[d]) for d in all_dates)
    LOG.info(f"Total: {total:,} samples from {len(all_dates)} dates")
    return features_by_date, targets_by_date, meta_by_date, all_dates


def run_rules_based_analysis(
    features_by_date, targets_by_date, meta_by_date, all_dates,
    output_dir,
):
    """
    Test pure rules-based gates with per-day P&L tracking.
    No ML model — just regime rules applied to each OOT day.
    """
    LOG.info(f"\n{'='*70}")
    LOG.info(f"RULES-BASED GATE ANALYSIS — Per-Day P&L")
    LOG.info(f"{'='*70}")

    idx = {name: i for i, name in enumerate(FEATURE_NAMES)}

    # Combine all data
    X_all = np.concatenate([features_by_date[d] for d in all_dates])
    n_total = len(X_all)

    # Precompute feature columns
    abs_pred_1s = X_all[:, idx["abs_pred_1s"]]
    sig_dir = X_all[:, idx["signal_direction"]]
    agreement = X_all[:, idx["signal_agreement"]]
    aligned_imb = X_all[:, idx["signal_aligned_with_imbalance"]]
    aligned_mom = X_all[:, idx["signal_aligned_with_momentum"]]
    persistence = X_all[:, idx["pred_ratio_10s_1s"]]
    tod = X_all[:, idx["time_of_day"]]

    # Define rule combinations to test
    rules = []

    # Confidence levels
    for conf_thresh in [0.25, 0.50, 0.75, 1.0, 1.5]:
        for side in ["both", "long", "short"]:
            for agree in [False, True]:
                for align in [False, True]:
                    for persist_min in [None, 0.5, 0.8]:
                        for tod_filter in [None, "mid_day"]:
                            rules.append({
                                "conf": conf_thresh,
                                "side": side,
                                "agree": agree,
                                "align": align,
                                "persist": persist_min,
                                "tod": tod_filter,
                            })

    # Test each rule combination across all horizons
    results = []

    for horizon in HORIZONS:
        pnl_key = f"pnl_{horizon}"
        prof_key = f"profitable_{horizon}"

        pnl_all = np.concatenate([targets_by_date[d][pnl_key] for d in all_dates])
        prof_all = np.concatenate([targets_by_date[d][prof_key] for d in all_dates])

        LOG.info(f"\n--- {horizon} HORIZON (n={n_total:,}, base WR={prof_all.mean():.3f}) ---")

        best_rules = []

        for rule in rules:
            # Build mask
            mask = abs_pred_1s >= rule["conf"]

            if rule["side"] == "long":
                mask &= sig_dir > 0
            elif rule["side"] == "short":
                mask &= sig_dir < 0

            if rule["agree"]:
                mask &= agreement == 1.0

            if rule["align"]:
                mask &= aligned_imb == 1.0

            if rule["persist"] is not None:
                mask &= persistence >= rule["persist"]

            if rule["tod"] == "mid_day":
                # Avoid first/last 30 min
                mask &= (tod > 0.077) & (tod < 0.923)  # ~30 min buffer

            n_sel = mask.sum()
            if n_sel < 50:
                continue

            wr = float(prof_all[mask].mean())
            avg_pnl = float(pnl_all[mask].mean())
            total_pnl = float(pnl_all[mask].sum())

            # Profit factor
            winners = pnl_all[mask] > 0
            losers = pnl_all[mask] <= 0
            gp = float(pnl_all[mask][winners].sum()) if winners.any() else 0
            gl = abs(float(pnl_all[mask][losers].sum())) if losers.any() else 1e-8
            pf = gp / (gl + 1e-8)

            # Per-day consistency
            day_pnls = []
            day_idx = 0
            for d in all_dates:
                n_d = len(features_by_date[d])
                day_mask = mask[day_idx:day_idx + n_d]
                day_pnl_arr = pnl_all[day_idx:day_idx + n_d]
                n_trades = day_mask.sum()
                if n_trades > 0:
                    day_pnls.append(float(day_pnl_arr[day_mask].sum()))
                day_idx += n_d

            if len(day_pnls) >= 5:
                day_pnls = np.array(day_pnls)
                win_days = (day_pnls > 0).sum()
                total_days = len(day_pnls)
                day_wr = win_days / total_days
                daily_mean = day_pnls.mean()
                daily_std = day_pnls.std()
                sortino_denom = np.sqrt((day_pnls[day_pnls < 0] ** 2).mean()) if (day_pnls < 0).any() else 1e-8
                sortino = daily_mean / sortino_denom if sortino_denom > 0 else 0
            else:
                day_wr = 0
                sortino = 0
                daily_mean = 0

            if avg_pnl > -0.05:  # Only log near-profitable rules
                best_rules.append({
                    "rule": rule,
                    "n": n_sel,
                    "wr": wr,
                    "avg_pnl": avg_pnl,
                    "total_pnl": total_pnl,
                    "pf": pf,
                    "day_wr": day_wr,
                    "sortino": sortino,
                    "daily_mean": daily_mean,
                    "n_days": len(day_pnls),
                })

        # Sort by Sortino
        best_rules.sort(key=lambda x: -x["sortino"])

        LOG.info(f"\n  TOP 20 RULES BY SORTINO (near-profitable, n≥50):")
        LOG.info(f"  {'#':>3s}  {'N':>6s}  {'WR':>6s}  {'PnL/tr':>8s}  {'PF':>5s}  {'Sortino':>8s}  {'DayWR':>6s}  {'Days':>4s}  Rule")

        for rank, r in enumerate(best_rules[:20]):
            rule = r["rule"]
            rule_str = f"C≥{rule['conf']:.2f}"
            if rule["side"] != "both":
                rule_str += f" {rule['side'].upper()}"
            if rule["agree"]:
                rule_str += " +Agr"
            if rule["align"]:
                rule_str += " +AlnBk"
            if rule["persist"]:
                rule_str += f" +Per≥{rule['persist']}"
            if rule["tod"]:
                rule_str += f" +MidDay"

            LOG.info(f"  {rank+1:3d}  {r['n']:6,}  {r['wr']:.3f}  {r['avg_pnl']:+7.3f}  {r['pf']:.2f}  {r['sortino']:+7.3f}  {r['day_wr']:.3f}  {r['n_days']:4d}  {rule_str}")

            results.append({
                "horizon": horizon,
                "rank": rank + 1,
                **r,
            })

    # Save results
    if output_dir:
        # Serialize — convert numpy types
        def _clean(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        clean_results = json.loads(json.dumps(results, default=_clean))
        with open(output_dir / "rules_results.json", "w") as f:
            json.dump(clean_results, f, indent=2)

    return results


def run_daily_wf_lgbm(
    features_by_date, targets_by_date, meta_by_date, all_dates,
    n_train_days=25, output_dir=None,
):
    """
    Daily walk-forward LightGBM: train on N days, predict day N+1, slide by 1.
    Maximizes number of OOT evaluation points.
    """
    LOG.info(f"\n{'='*70}")
    LOG.info(f"DAILY WALK-FORWARD LGBM ({n_train_days} train days, 1 OOT day)")
    LOG.info(f"{'='*70}")

    n_dates = len(all_dates)
    if n_dates < n_train_days + 1:
        LOG.error(f"Not enough dates: {n_dates}")
        return {}

    all_oot_preds = {h: [] for h in HORIZONS}
    all_oot_actuals = {h: [] for h in HORIZONS}
    all_oot_pnl = {h: [] for h in HORIZONS}
    all_oot_features = []
    oot_dates_list = []

    for start in range(0, n_dates - n_train_days):
        train_dates = all_dates[start:start + n_train_days]
        oot_date = all_dates[start + n_train_days]

        if oot_date not in features_by_date:
            continue

        X_train = np.concatenate([features_by_date[d] for d in train_dates if d in features_by_date])
        X_oot = features_by_date[oot_date]

        if len(X_train) == 0 or len(X_oot) == 0:
            continue

        for horizon in HORIZONS:
            prof_key = f"profitable_{horizon}"
            pnl_key = f"pnl_{horizon}"

            y_train = np.concatenate([targets_by_date[d][prof_key] for d in train_dates if d in targets_by_date])
            y_oot = targets_by_date[oot_date][prof_key]
            pnl_oot = targets_by_date[oot_date][pnl_key]

            params = {
                "objective": "binary",
                "metric": "auc",
                "learning_rate": 0.05,
                "num_leaves": 31,
                "max_depth": 5,
                "min_child_samples": 200,
                "subsample": 0.7,
                "colsample_bytree": 0.7,
                "reg_alpha": 0.5,
                "reg_lambda": 2.0,
                "verbose": -1,
                "n_jobs": 8,
                "seed": 42,
            }

            dtrain = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
            model = lgb.train(params, dtrain, num_boost_round=200)
            preds = model.predict(X_oot)

            all_oot_preds[horizon].append(preds)
            all_oot_actuals[horizon].append(y_oot)
            all_oot_pnl[horizon].append(pnl_oot)

        all_oot_features.append(X_oot)
        oot_dates_list.append(oot_date)

        if (start + 1) % 5 == 0:
            LOG.info(f"  Completed {start + 1}/{n_dates - n_train_days} OOT days")

    LOG.info(f"\nCompleted {len(oot_dates_list)} OOT days")

    # Analyze concat results
    idx = {name: i for i, name in enumerate(FEATURE_NAMES)}

    for horizon in HORIZONS:
        if not all_oot_preds[horizon]:
            continue

        preds_concat = np.concatenate(all_oot_preds[horizon])
        actuals_concat = np.concatenate(all_oot_actuals[horizon])
        pnl_concat = np.concatenate(all_oot_pnl[horizon])
        X_concat = np.concatenate(all_oot_features)

        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(actuals_concat, preds_concat)
        except:
            auc = 0.5

        sp, _ = spearmanr(preds_concat, pnl_concat)

        LOG.info(f"\n--- DAILY WF RESULTS: {horizon} ({len(preds_concat):,} OOT samples, {len(oot_dates_list)} days) ---")
        LOG.info(f"  AUC: {auc:.4f} | Spearman(pred, P&L): {sp:.4f}")

        # Threshold analysis
        LOG.info(f"  {'Thresh':>7s}  {'Select%':>8s}  {'Trades':>7s}  {'WR':>6s}  {'AvgPnL':>8s}  {'TotalPnL':>10s}  {'PF':>6s}")

        for thresh in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
            selected = preds_concat >= thresh
            n_sel = int(selected.sum())
            if n_sel > 10:
                sel_pct = n_sel / len(preds_concat) * 100
                wr = float(actuals_concat[selected].mean())
                avg_pnl = float(pnl_concat[selected].mean())
                total_pnl = float(pnl_concat[selected].sum())
                w = pnl_concat[selected] > 0
                l = pnl_concat[selected] <= 0
                gp = float(pnl_concat[selected][w].sum()) if w.any() else 0
                gl = abs(float(pnl_concat[selected][l].sum())) if l.any() else 1e-8
                pf = gp / (gl + 1e-8)
                LOG.info(f"    {thresh:5.2f}    {sel_pct:6.1f}%  {n_sel:7,}  {wr:.3f}  {avg_pnl:+7.3f}tk  {total_pnl:+9.1f}tk  {pf:.2f}")

        # Per-day performance of top 50% selection
        LOG.info(f"\n  Per-day P&L (threshold=0.55):")
        day_pnls = []
        offset = 0
        for d_idx, oot_date in enumerate(oot_dates_list):
            n_d = len(all_oot_preds[horizon][d_idx])
            day_preds = preds_concat[offset:offset + n_d]
            day_pnl = pnl_concat[offset:offset + n_d]
            selected = day_preds >= 0.55
            n_sel = selected.sum()
            if n_sel > 0:
                day_total = float(day_pnl[selected].sum())
                day_wr = float((day_pnl[selected] > 0).mean())
                day_pnls.append(day_total)
                LOG.info(f"    {oot_date}: {n_sel:4d} trades, WR={day_wr:.3f}, P&L={day_total:+.1f}tk")
            offset += n_d

        if day_pnls:
            day_pnls = np.array(day_pnls)
            win_days = (day_pnls > 0).sum()
            LOG.info(f"\n  Day-level stats: {win_days}/{len(day_pnls)} green days "
                     f"({win_days/len(day_pnls)*100:.0f}%), "
                     f"mean={day_pnls.mean():+.1f}tk, std={day_pnls.std():.1f}tk")

        # Side-split analysis
        sig_dir = X_concat[:, idx["signal_direction"]]
        for side_val, side_label in [(1.0, "LONG"), (-1.0, "SHORT")]:
            side_mask = sig_dir == side_val
            if side_mask.sum() > 100:
                sp_side, _ = spearmanr(preds_concat[side_mask], pnl_concat[side_mask])

                # Best threshold for this side
                best_pnl = -999
                best_thresh = 0.5
                for thresh in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
                    sel = (preds_concat >= thresh) & side_mask
                    if sel.sum() > 50:
                        avg = pnl_concat[sel].mean()
                        if avg > best_pnl:
                            best_pnl = avg
                            best_thresh = thresh

                sel = (preds_concat >= best_thresh) & side_mask
                n_sel = sel.sum()
                wr = actuals_concat[sel].mean() if n_sel > 0 else 0
                LOG.info(f"\n  {side_label} (Spearman={sp_side:.4f}): "
                         f"Best @{best_thresh:.2f}: {n_sel:,} trades, WR={wr:.3f}, "
                         f"AvgPnL={best_pnl:+.3f}tk")


def main():
    parser = argparse.ArgumentParser(description="Focused Gate Validation")
    parser.add_argument("--data-dir", type=str,
                       default=str(LVL3_ROOT / "data/processed/mbo_events_smart_v3"))
    parser.add_argument("--pred-dir", type=str,
                       default=str(LVL3_ROOT / "output/cnn_mamba_v2_all_oot"))
    parser.add_argument("--output-dir", type=str,
                       default=str(LVL3_ROOT / "output/focused_gate_v1"))
    parser.add_argument("--n-train-days", type=int, default=25)
    parser.add_argument("--n-workers", type=int, default=8)
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
    LOG.info("FOCUSED GATE VALIDATION")
    LOG.info("=" * 70)

    features_by_date, targets_by_date, meta_by_date, all_dates = load_all_data(
        Path(args.data_dir), Path(args.pred_dir), args.n_workers,
    )

    if not all_dates:
        LOG.error("No data loaded!")
        return

    # Part 1: Pure rules analysis (no ML, uses all data descriptively)
    run_rules_based_analysis(
        features_by_date, targets_by_date, meta_by_date, all_dates,
        output_dir,
    )

    # Part 2: Daily walk-forward LGBM
    run_daily_wf_lgbm(
        features_by_date, targets_by_date, meta_by_date, all_dates,
        n_train_days=args.n_train_days,
        output_dir=output_dir,
    )

    LOG.info("\n" + "=" * 70)
    LOG.info("FOCUSED GATE VALIDATION COMPLETE")
    LOG.info("=" * 70)


if __name__ == "__main__":
    main()
