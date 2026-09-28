#!/usr/bin/env python3
"""
Passive Execution Optimizer — Permutation Validation (GPU)
==========================================================
Retrains XGBoost with shuffled labels N times to establish
null distribution. Tests if real Spearman 0.784 is genuine.

HC #720: Prove before celebrating. 10/10 past results were leakage.
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ============ CONFIG ============
BASE_DIR = Path("/home/nick/Lvl3Quant")
LABELS_DIR = BASE_DIR / "output" / "mbo_walker_labels"
FEATURES_DIR = BASE_DIR / "output" / "queue_augmented_features"
FILL_PROB_DIR = BASE_DIR / "output" / "fill_prob_v3_honest_xgb"
OUTPUT_DIR = BASE_DIR / "output" / "passive_exec_optimizer_v1" / "permtest"
REAL_RESULTS = BASE_DIR / "output" / "passive_exec_optimizer_v1" / "results.json"

N_PERMUTATIONS = 100
TRAIN_WINDOW = 25
COMMISSION_PASSIVE = 0.376
HORIZON = "10s"

XGB_PARAMS = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "device": "cuda",
    "max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "min_child_weight": 50,
    "early_stopping_rounds": 30,
    "eval_metric": "rmse",
    "verbosity": 0,
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

SEP = "=" * 60

def get_available_dates():
    label_dates = {f.stem.replace("labels_", "") for f in LABELS_DIR.glob("labels_*.parquet")}
    feat_dates = {f.stem.replace("features_", "") for f in FEATURES_DIR.glob("features_*.parquet")}
    return sorted(label_dates & feat_dates)


def load_fill_prob_model():
    model_path = FILL_PROB_DIR / f"xgb_fill_{HORIZON}_full.json"
    if not model_path.exists():
        return None
    model = xgb.Booster()
    model.load_model(str(model_path))
    feats_path = FILL_PROB_DIR / "feats.json"
    fill_features = None
    if feats_path.exists():
        with open(feats_path) as f:
            feats_data = json.load(f)
            fill_features = feats_data if isinstance(feats_data, list) else feats_data.get(HORIZON, feats_data.get("features", None))
    return model, fill_features


def load_day(date_str):
    labels = pd.read_parquet(LABELS_DIR / f"labels_{date_str}.parquet")
    features = pd.read_parquet(FEATURES_DIR / f"features_{date_str}.parquet")
    label_cols = ["event_id", "price", "queue_depth_at_touch",
                  "queue_rank_at_10s", "filled_10s", "time_to_fill_s_10s"]
    df = features.merge(labels[label_cols], on="event_id", how="inner")
    df = df.rename(columns={
        "queue_rank_at_10s": "queue_rank",
        "filled_10s": "filled",
        "time_to_fill_s_10s": "time_to_fill_s",
    })
    return df


def construct_target(df):
    signal_magnitude = df["pred_10s"].abs()
    df["net_pnl_ticks"] = df["filled"] * (signal_magnitude - COMMISSION_PASSIVE)
    return df


def compute_fill_prob_feature(df, fill_model_info):
    if fill_model_info is None:
        df["fill_prob_pred"] = np.nan
        return df
    model, fill_features = fill_model_info
    if fill_features is not None:
        available = [c for c in fill_features if c in df.columns]
        if len(available) < 3:
            df["fill_prob_pred"] = np.nan
            return df
        X_fill = df[available].copy()
    else:
        queue_cols = [c for c in df.columns if "queue" in c or "bid_q" in c or "ask_q" in c
                     or "depth" in c or "qty" in c]
        X_fill = df[queue_cols].copy()
    dmat = xgb.DMatrix(X_fill)
    df["fill_prob_pred"] = model.predict(dmat)
    return df


def get_feature_columns(df):
    exclude = {"event_id", "ts_ns", "price", "net_pnl_ticks", "filled",
               "time_to_fill_s", "queue_rank", "queue_depth_at_touch"}
    return [c for c in df.columns if c not in exclude]


def run_single_wf(all_data, dates, feature_cols, shuffle_labels=False, seed=None):
    """Run one full walk-forward pass. If shuffle_labels=True, shuffle targets within each training fold."""
    rng = np.random.RandomState(seed)
    all_preds = []
    all_targets = []
    
    for fold_idx in range(TRAIN_WINDOW, len(dates)):
        oot_date = dates[fold_idx]
        train_dates = dates[fold_idx - TRAIN_WINDOW:fold_idx]
        
        train_dfs = [all_data[d] for d in train_dates]
        train_df = pd.concat(train_dfs, ignore_index=True)
        oot_df = all_data[oot_date].copy()
        
        X_train = np.nan_to_num(train_df[feature_cols].values, nan=0.0)
        y_train = train_df["net_pnl_ticks"].values.copy()
        X_oot = np.nan_to_num(oot_df[feature_cols].values, nan=0.0)
        y_oot = oot_df["net_pnl_ticks"].values
        
        if shuffle_labels:
            rng.shuffle(y_train)
        
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
        dval = xgb.DMatrix(X_oot, label=y_oot, feature_names=feature_cols)
        
        params = {k: v for k, v in XGB_PARAMS.items()
                 if k not in ("n_estimators", "early_stopping_rounds", "eval_metric")}
        params["eval_metric"] = XGB_PARAMS["eval_metric"]
        
        bst = xgb.train(
            params, dtrain,
            num_boost_round=XGB_PARAMS["n_estimators"],
            evals=[(dval, "oot")],
            early_stopping_rounds=XGB_PARAMS["early_stopping_rounds"],
            verbose_eval=False,
        )
        
        preds = bst.predict(dval)
        all_preds.extend(preds.tolist())
        all_targets.extend(y_oot.tolist())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    concat_spearman, _ = spearmanr(all_preds, all_targets)
    if np.isnan(concat_spearman):
        concat_spearman = 0.0
    
    # Top decile analysis
    n = len(all_preds)
    top_dec_idx = np.argsort(all_preds)[-n // 10:]
    top_decile_net = all_targets[top_dec_idx].mean()
    bot_dec_idx = np.argsort(all_preds)[:n // 10]
    bot_decile_net = all_targets[bot_dec_idx].mean()
    
    return {
        "spearman": concat_spearman,
        "top_decile_net": top_decile_net,
        "bot_decile_net": bot_decile_net,
        "spread": top_decile_net - bot_decile_net,
    }


def main():
    print(SEP)
    print("PASSIVE EXEC OPTIMIZER — PERMUTATION VALIDATION")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"N_PERMUTATIONS: {N_PERMUTATIONS}")
    print(SEP)
    
    # Load real results
    with open(REAL_RESULTS) as f:
        real = json.load(f)
    real_spearman = real["concat_spearman"]
    real_top_net = real["concat_top_decile_net"]
    print(f"Real Spearman: {real_spearman:.4f}")
    print(f"Real Top Decile Net: {real_top_net:.4f} ticks")
    
    # Load data
    dates = get_available_dates()
    print(f"Available dates: {len(dates)}")
    
    fill_model_info = load_fill_prob_model()
    print(f"Fill prob model: {'loaded' if fill_model_info else 'MISSING'}")
    
    print("Pre-loading all dates...")
    all_data = {}
    for d in dates:
        df = load_day(d)
        df = construct_target(df)
        df = compute_fill_prob_feature(df, fill_model_info)
        all_data[d] = df
    
    feature_cols = get_feature_columns(all_data[dates[0]])
    print(f"Features: {len(feature_cols)}")
    print(f"OOT folds per pass: {len(dates) - TRAIN_WINDOW}")
    
    # Run permutation tests
    perm_spearmans = []
    perm_top_nets = []
    
    for i in range(N_PERMUTATIONS):
        t0 = time.time()
        result = run_single_wf(all_data, dates, feature_cols, shuffle_labels=True, seed=i)
        elapsed = time.time() - t0
        
        perm_spearmans.append(result["spearman"])
        perm_top_nets.append(result["top_decile_net"])
        
        print(f"  Perm {i+1}/{N_PERMUTATIONS}: Spearman={result['spearman']:.4f}, "
              f"TopDecile={result['top_decile_net']:.4f}, time={elapsed:.1f}s", flush=True)
    
    # Compute p-values
    perm_spearmans = np.array(perm_spearmans)
    perm_top_nets = np.array(perm_top_nets)
    
    p_spearman = (np.sum(perm_spearmans >= real_spearman) + 1) / (N_PERMUTATIONS + 1)
    p_top_net = (np.sum(perm_top_nets >= real_top_net) + 1) / (N_PERMUTATIONS + 1)
    
    print(SEP)
    print("PERMUTATION TEST RESULTS")
    print(SEP)
    print(f"  Real Spearman: {real_spearman:.4f}")
    print(f"  Perm Spearman: mean={perm_spearmans.mean():.4f}, std={perm_spearmans.std():.4f}, "
          f"max={perm_spearmans.max():.4f}")
    print(f"  p-value (Spearman): {p_spearman:.4f} ({'PASS' if p_spearman < 0.05 else 'FAIL'})")
    print()
    print(f"  Real Top Decile Net: {real_top_net:.4f} ticks")
    print(f"  Perm Top Decile Net: mean={perm_top_nets.mean():.4f}, std={perm_top_nets.std():.4f}, "
          f"max={perm_top_nets.max():.4f}")
    print(f"  p-value (TopDecile): {p_top_net:.4f} ({'PASS' if p_top_net < 0.05 else 'FAIL'})")
    print()
    print(f"  Edge (Spearman): {real_spearman - perm_spearmans.mean():.4f}")
    print(f"  Edge (TopDecile): {real_top_net - perm_top_nets.mean():.4f} ticks")
    
    overall = "PASS" if (p_spearman < 0.05 and p_top_net < 0.05) else "FAIL"
    print(f"\n  OVERALL: {overall}")
    
    # Save
    results = {
        "real_spearman": real_spearman,
        "real_top_decile_net": real_top_net,
        "perm_spearman_mean": float(perm_spearmans.mean()),
        "perm_spearman_std": float(perm_spearmans.std()),
        "perm_spearman_max": float(perm_spearmans.max()),
        "perm_top_net_mean": float(perm_top_nets.mean()),
        "perm_top_net_std": float(perm_top_nets.std()),
        "perm_top_net_max": float(perm_top_nets.max()),
        "p_spearman": float(p_spearman),
        "p_top_net": float(p_top_net),
        "n_permutations": N_PERMUTATIONS,
        "overall": overall,
        "perm_spearmans": perm_spearmans.tolist(),
        "perm_top_nets": perm_top_nets.tolist(),
    }
    
    with open(OUTPUT_DIR / "permtest_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {OUTPUT_DIR / 'permtest_results.json'}")


if __name__ == "__main__":
    main()
