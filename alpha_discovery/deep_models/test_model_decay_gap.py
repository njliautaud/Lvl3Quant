#!/usr/bin/env python3
"""
Model Decay Test — GAP FILL (Mar 30 → Apr 8, 2026)

Fills the critical gap between day 14 (Mar 29) and day 37 (Apr 21) to find exact decay point.
Reuses all model/inference code from test_model_decay.py.
These 9 dates cover days 15-24 post-training — the expected transition zone.

CPU inference on Jupiter (no GPU).
"""

import os
import sys
import time
import json
import warnings
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List

warnings.filterwarnings("ignore")

import functools
print = functools.partial(print, flush=True)

# =============================================================================
# Configuration — GAP DATES ONLY
# =============================================================================

DATA_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3")

# Gap dates: Mar 30 → Apr 8 (days 15-24 post-training)
TEST_DATES = [
    "20260330", "20260331",
    "20260401", "20260402", "20260403",
    "20260405", "20260406", "20260407", "20260408",
]

# CNN-Mamba v2 config
CNNMAMBA_WEIGHTS = Path("/home/jupiter/Lvl3Quant/output/cnn_mamba_v2_smart_v3_mar/fold_10_best.pt")
CNNMAMBA_STATS   = Path("/home/jupiter/Lvl3Quant/output/cnn_mamba_v2_smart_v3_mar/fold_09_feature_stats.npz")
CNNMAMBA_WINDOW  = 3000
CNNMAMBA_STRIDE  = 3000

# PatchTST config
PATCHTST_WEIGHTS = Path("/home/jupiter/Lvl3Quant/output/patchtst_razer_weights/fold_17_best.pt")
PATCHTST_WINDOW  = 500
PATCHTST_STRIDE  = 500

BASELINE_IC = {
    "CNN-Mamba v2": {"IC_1s": 0.040, "IC_5s": 0.065, "IC_10s": 0.085},
    "PatchTST":     {"IC_1s": 0.035, "IC_5s": 0.055, "IC_10s": 0.075},
}

DEVICE = torch.device("cpu")
MAX_WINDOWS_PER_DAY = 500

# =============================================================================
# Import model classes and utilities from main script
# =============================================================================

_DEEP_MODELS_DIR = str(Path(__file__).resolve().parent)
if _DEEP_MODELS_DIR not in sys.path:
    sys.path.insert(0, _DEEP_MODELS_DIR)

# Import all the model classes and functions we need
from test_model_decay import (
    CNNMambaV2, SelectiveSSM, MambaBlock,
    load_cnn_mamba_v2, load_patchtst,
    run_sliding_window_inference,
    compute_ic, compute_directional_accuracy, compute_conditional_ic,
    load_day_data, get_days_since_training, get_week_group,
)


def run_gap_test():
    print("=" * 80)
    print(f"MODEL DECAY TEST — GAP FILL (Mar 30 → Apr 8, {len(TEST_DATES)} dates)")
    print("Fills days 15-24 post-training to find exact decay point")
    print("Models trained on data up to ~Mar 15, 2026")
    print("=" * 80)
    print()

    # Verify data files exist
    available = []
    for d in TEST_DATES:
        fpath = DATA_DIR / f"{d}_mbo_events.npz"
        if fpath.exists():
            available.append(d)
        else:
            print(f"  MISSING: {d}")
    print(f"  Available: {len(available)}/{len(TEST_DATES)} dates")
    print()

    # Load models
    cnn_mamba = load_cnn_mamba_v2()
    patchtst = load_patchtst()
    print()

    models = {
        "CNN-Mamba v2": (cnn_mamba, CNNMAMBA_WINDOW, CNNMAMBA_STRIDE, 4),
        "PatchTST":     (patchtst, PATCHTST_WINDOW, PATCHTST_STRIDE, 32),
    }

    results = defaultdict(dict)

    for model_name, (model, window, stride, bsz) in models.items():
        print(f"\n{'='*60}")
        print(f"  {model_name}  (window={window}, stride={stride})")
        print(f"{'='*60}")

        all_preds = {h: [] for h in ["1s", "5s", "10s"]}
        all_labels = {h: [] for h in ["1s", "5s", "10s"]}

        for date_str in available:
            days_since = get_days_since_training(date_str)
            week_group = get_week_group(date_str)
            print(f"\n  Date: {date_str} (day +{days_since}, {week_group})")
            t0 = time.time()

            try:
                data = load_day_data(date_str)
            except FileNotFoundError as e:
                print(f"    SKIP: {e}")
                continue

            events = data["events"]
            print(f"    Events: {len(events):,}")

            preds, valid_mask = run_sliding_window_inference(
                model, events, window, stride, batch_size=bsz
            )
            elapsed = time.time() - t0
            n_valid = valid_mask.sum()
            print(f"    Inference: {elapsed:.1f}s, {n_valid:,} predictions")

            date_results = {"days_since_training": days_since, "week_group": week_group}
            for hi, horizon in enumerate(["1s", "5s", "10s"]):
                label_key = f"labels_{horizon}"
                labels = data[label_key][:len(preds)]
                p = preds[valid_mask, hi]
                l = labels[valid_mask]

                ic = compute_ic(p, l)
                da = compute_directional_accuracy(p, l)
                cond_ic_5 = compute_conditional_ic(p, l, 0.05)
                cond_ic_1 = compute_conditional_ic(p, l, 0.01)

                date_results[f"IC_{horizon}"] = ic
                date_results[f"DA_{horizon}"] = da
                date_results[f"CondIC5%_{horizon}"] = cond_ic_5
                date_results[f"CondIC1%_{horizon}"] = cond_ic_1

                all_preds[horizon].append(p)
                all_labels[horizon].append(l)

                print(f"    {horizon}: IC={ic:+.4f}  DA={da:.3f}  "
                      f"Top5%IC={cond_ic_5:+.4f}  Top1%IC={cond_ic_1:+.4f}")

            results[model_name][date_str] = date_results

        # Concat IC for gap period
        print(f"\n  --- Concat (gap period, {len(available)} dates) ---")
        for horizon in ["1s", "5s", "10s"]:
            if not all_preds[horizon]:
                continue
            p_all = np.concatenate(all_preds[horizon])
            l_all = np.concatenate(all_labels[horizon])
            ic = compute_ic(p_all, l_all)
            da = compute_directional_accuracy(p_all, l_all)
            bl = BASELINE_IC[model_name][f"IC_{horizon}"]
            decay_pct = ((ic - bl) / abs(bl)) * 100 if bl != 0 else 0
            print(f"  {horizon}: IC={ic:+.4f} (baseline={bl:+.4f}, "
                  f"delta={ic-bl:+.4f} [{decay_pct:+.1f}%])  DA={da:.3f}")

    # Summary table
    print("\n")
    print("=" * 100)
    print("GAP FILL SUMMARY — Per-Date IC_10s")
    print("=" * 100)
    header = f"{'Model':<16} {'Date':<10} {'Day+':>5} {'IC_1s':>8} {'IC_5s':>8} {'IC_10s':>8} {'DA_10s':>7}"
    print(header)
    print("-" * 80)

    for model_name in models:
        for date_str in available:
            if date_str not in results[model_name]:
                continue
            r = results[model_name][date_str]
            days = r.get("days_since_training", "?")
            print(f"{model_name:<16} {date_str:<10} {days:>5} "
                  f"{r['IC_1s']:>+8.4f} {r['IC_5s']:>+8.4f} {r['IC_10s']:>+8.4f} "
                  f"{r['DA_10s']:>7.3f}")
        print("-" * 80)

    # Save results
    out_path = Path("/home/jupiter/Lvl3Quant/output/decay_analysis_gap_results.json")
    save_data = {}
    for model_name in models:
        save_data[model_name] = {
            k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                for kk, vv in v.items()}
            for k, v in results[model_name].items()
        }
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print("\nDone — merge with main decay results to get complete decay curve.")


if __name__ == "__main__":
    run_gap_test()
