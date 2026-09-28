#!/usr/bin/env python3
"""
2h LGBM Model — HC #659 PROPER Permutation Test (100 trials)
=============================================================
Prior test used only 3 shuffled trials. HC #659 requires 100+.
This script runs 100 shuffled walk-forward trials to get a real p-value.

Method: Same as lh_2h_permutation_test.py but with 100 seeds.
Parallelized across CPU cores for speed.
"""

import sys, os, json, time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "live_trading_linux"))

from lh_2h_paper_engine import (
    compute_enhanced_hourly, add_rolling_features, add_regime_context,
    get_feature_cols, TRAIN_DAYS, PURGE_DAYS, HORIZON_BARS, LGBM_PARAMS,
    MINUTE_BAR_DIR, LOOKBACK_DAYS, LH2hPaperEngine
)

OUT_DIR = ROOT / "output" / "lh_2h_permutation_100"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SHUFFLED = 100
COST = 1.376  # ticks RT (market order)
COST_PASSIVE = 0.376  # ticks RT (passive limit)

def load_and_prepare_data():
    """Load data once and return prepared hourly DataFrame."""
    engine = LH2hPaperEngine()
    print("Loading minute bar data...")
    minute_df = engine._load_minute_bars(n_days=999)
    if minute_df.empty:
        sys.exit("No data loaded")
    print(f"Loaded {len(minute_df)} minute bars")

    hourly = engine._build_hourly_df(minute_df)
    print(f"Built {len(hourly)} hourly bars with {len(hourly.columns)} columns")

    # Add forward labels
    hourly = hourly.sort_values("ts").reset_index(drop=True)
    hourly["fwd_ticks"] = hourly["close"].shift(-HORIZON_BARS) - hourly["close"]

    # Null out overnight gaps
    for i in range(len(hourly) - HORIZON_BARS):
        ts_now = hourly["ts"].iloc[i]
        ts_fwd = hourly["ts"].iloc[i + HORIZON_BARS]
        diff_s = (ts_fwd - ts_now).total_seconds()
        if diff_s > 8 * 3600:
            hourly.loc[hourly.index[i], "fwd_ticks"] = np.nan

    # Clean
    hourly_clean = hourly[~hourly["hour"].isin([19, 20])].copy()
    hourly_clean = hourly_clean.dropna(subset=["fwd_ticks"])
    dates = sorted(hourly_clean["date"].unique())
    feature_cols = get_feature_cols(hourly_clean)

    print(f"Clean data: {len(hourly_clean)} bars, {len(dates)} days, {len(feature_cols)} features")
    return hourly_clean, dates, feature_cols


def walk_forward(hourly_clean, dates, feature_cols, shuffle_labels=False, seed=42):
    """Walk-forward LGBM training. Returns predictions, actuals, dates."""
    import lightgbm as lgb

    all_preds = []
    all_actuals = []
    all_dates_oot = []

    min_start = TRAIN_DAYS + PURGE_DAYS

    for day_idx in range(min_start, len(dates)):
        oot_date = dates[day_idx]
        train_dates = dates[day_idx - TRAIN_DAYS - PURGE_DAYS : day_idx - PURGE_DAYS]

        train_mask = hourly_clean["date"].isin(train_dates)
        oot_mask = hourly_clean["date"] == oot_date

        train_df = hourly_clean[train_mask]
        oot_df = hourly_clean[oot_mask]

        if len(train_df) < 100 or len(oot_df) == 0:
            continue

        X_train = train_df[feature_cols].fillna(0).values.astype(np.float32)
        y_train = train_df["fwd_ticks"].values.astype(np.float32)
        X_oot = oot_df[feature_cols].fillna(0).values.astype(np.float32)
        y_oot = oot_df["fwd_ticks"].values.astype(np.float32)

        if shuffle_labels:
            rng = np.random.RandomState(seed + day_idx)
            y_train = rng.permutation(y_train)

        # Split for early stopping
        split = int(len(X_train) * 0.8)
        X_tr, X_val = X_train[:split], X_train[split:]
        y_tr, y_val = y_train[:split], y_train[split:]

        params = {**LGBM_PARAMS, "seed": seed, "verbosity": -1}
        model = lgb.LGBMRegressor(**params, early_stopping_rounds=50)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.log_evaluation(0)])

        preds = model.predict(X_oot)
        all_preds.extend(preds)
        all_actuals.extend(y_oot)
        all_dates_oot.extend([oot_date] * len(y_oot))

    return np.array(all_preds), np.array(all_actuals), all_dates_oot


def evaluate_run(preds, actuals, cost=COST):
    """Compute IC, direction accuracy, net ticks, Sharpe."""
    if len(preds) == 0:
        return {"ic": 0, "dir_acc": 50, "avg_net": 0, "sharpe": 0, "n": 0}

    ic = float(stats.spearmanr(preds, actuals)[0])
    dir_acc = float(((preds > 0) == (actuals > 0)).mean() * 100)
    gross = np.where(preds > 0, actuals, -actuals)
    net = gross - cost
    sharpe = float(net.mean() / net.std() * np.sqrt(252 * 7)) if net.std() > 0 else 0

    return {
        "ic": round(ic, 4),
        "dir_acc": round(dir_acc, 1),
        "avg_net": round(float(net.mean()), 3),
        "avg_gross": round(float(gross.mean()), 3),
        "sharpe": round(sharpe, 2),
        "n": len(preds),
        "wr": round(float((net > 0).mean() * 100), 1),
    }


def run_single_shuffled(args):
    """Worker function for parallel shuffled trials."""
    hourly_clean, dates, feature_cols, seed = args
    preds, actuals, _ = walk_forward(hourly_clean, dates, feature_cols, shuffle_labels=True, seed=seed)
    result = evaluate_run(preds, actuals)
    result["seed"] = seed
    return result


if __name__ == "__main__":
    t0 = time.time()

    # Load data
    hourly_clean, dates, feature_cols = load_and_prepare_data()

    # Run REAL labels first
    print("\n" + "=" * 60)
    print("REAL LABELS (no shuffle)")
    print("=" * 60)
    real_preds, real_actuals, real_dates = walk_forward(hourly_clean, dates, feature_cols, shuffle_labels=False)
    real_result = evaluate_run(real_preds, real_actuals)
    real_result_passive = evaluate_run(real_preds, real_actuals, cost=COST_PASSIVE)

    print(f"IC: {real_result['ic']:.4f}")
    print(f"Direction: {real_result['dir_acc']:.1f}%")
    print(f"Avg net (market): {real_result['avg_net']:.3f} ticks")
    print(f"Avg net (passive): {real_result_passive['avg_net']:.3f} ticks")
    print(f"Sharpe (market): {real_result['sharpe']:.2f}")
    print(f"Sharpe (passive): {real_result_passive['sharpe']:.2f}")
    print(f"N trades: {real_result['n']}")

    t_real = time.time()
    real_seconds = t_real - t0
    print(f"\nReal run took {real_seconds:.0f}s")
    est_total = real_seconds * (N_SHUFFLED + 1)
    print(f"Estimated total time: {est_total / 60:.0f} minutes")

    # Run SHUFFLED labels — sequential (LGBM uses all cores internally)
    print(f"\n{'=' * 60}")
    print(f"SHUFFLED LABELS ({N_SHUFFLED} trials, sequential)")
    print(f"{'=' * 60}")

    shuffled_results = []
    seeds = list(range(1000, 1000 + N_SHUFFLED))

    for i, seed in enumerate(seeds):
        t_trial = time.time()
        preds, actuals, _ = walk_forward(hourly_clean, dates, feature_cols, shuffle_labels=True, seed=seed)
        result = evaluate_run(preds, actuals)
        result["seed"] = seed
        shuffled_results.append(result)

        elapsed = time.time() - t_trial
        total_elapsed = time.time() - t_real
        avg_per = total_elapsed / (i + 1)
        remaining = avg_per * (N_SHUFFLED - i - 1)

        if (i + 1) % 5 == 0:
            print(f"  Trial {i + 1}/{N_SHUFFLED}: IC={result['ic']:.4f}, "
                  f"net={result['avg_net']:.3f}, Sharpe={result['sharpe']:.2f} "
                  f"[{elapsed:.0f}s, ETA {remaining / 60:.0f}m]")

    # Compute p-value
    shuffled_nets = [r["avg_net"] for r in shuffled_results]
    shuffled_sharpes = [r["sharpe"] for r in shuffled_results]
    shuffled_ics = [r["ic"] for r in shuffled_results]

    p_value_net = sum(1 for sn in shuffled_nets if sn >= real_result["avg_net"]) / N_SHUFFLED
    p_value_sharpe = sum(1 for ss in shuffled_sharpes if ss >= real_result["sharpe"]) / N_SHUFFLED
    p_value_ic = sum(1 for si in shuffled_ics if si >= real_result["ic"]) / N_SHUFFLED

    total_time = time.time() - t0

    # Print summary
    print(f"\n{'=' * 60}")
    print("PERMUTATION TEST RESULTS (HC #659 compliant)")
    print(f"{'=' * 60}")
    print(f"\nReal model:")
    print(f"  IC={real_result['ic']:.4f}, Dir%={real_result['dir_acc']:.1f}%")
    print(f"  Avg net (market order): {real_result['avg_net']:.3f} ticks")
    print(f"  Avg net (passive limit): {real_result_passive['avg_net']:.3f} ticks")
    print(f"  Sharpe (market): {real_result['sharpe']:.2f}")
    print(f"  Sharpe (passive): {real_result_passive['sharpe']:.2f}")

    print(f"\nShuffled ({N_SHUFFLED} trials):")
    print(f"  IC: mean={np.mean(shuffled_ics):.4f}, std={np.std(shuffled_ics):.4f}, "
          f"max={np.max(shuffled_ics):.4f}")
    print(f"  Net: mean={np.mean(shuffled_nets):.3f}, std={np.std(shuffled_nets):.3f}, "
          f"max={np.max(shuffled_nets):.3f}")
    print(f"  Sharpe: mean={np.mean(shuffled_sharpes):.2f}, std={np.std(shuffled_sharpes):.2f}, "
          f"max={np.max(shuffled_sharpes):.2f}")

    print(f"\np-values:")
    print(f"  p(IC): {p_value_ic:.3f}")
    print(f"  p(net_ticks): {p_value_net:.3f}")
    print(f"  p(Sharpe): {p_value_sharpe:.3f}")

    verdict = "PASS" if p_value_net == 0 and p_value_ic == 0 else \
              "PASS (marginal)" if p_value_net < 0.05 else "FAIL"
    print(f"\nVERDICT: {verdict}")
    print(f"Total time: {total_time / 60:.1f} minutes")

    # Save results
    output = {
        "model": "2h LGBM (HC #659 100-trial permutation)",
        "generated": pd.Timestamp.now().isoformat(),
        "n_oot_days": len(dates) - TRAIN_DAYS - PURGE_DAYS,
        "n_shuffled_trials": N_SHUFFLED,
        "total_seconds": round(total_time, 1),
        "real": real_result,
        "real_passive": real_result_passive,
        "shuffled_summary": {
            "ic_mean": round(float(np.mean(shuffled_ics)), 4),
            "ic_std": round(float(np.std(shuffled_ics)), 4),
            "ic_max": round(float(np.max(shuffled_ics)), 4),
            "net_mean": round(float(np.mean(shuffled_nets)), 3),
            "net_std": round(float(np.std(shuffled_nets)), 3),
            "net_max": round(float(np.max(shuffled_nets)), 3),
            "sharpe_mean": round(float(np.mean(shuffled_sharpes)), 2),
            "sharpe_std": round(float(np.std(shuffled_sharpes)), 2),
            "sharpe_max": round(float(np.max(shuffled_sharpes)), 2),
        },
        "p_values": {
            "ic": p_value_ic,
            "net_ticks": p_value_net,
            "sharpe": p_value_sharpe,
        },
        "verdict": verdict,
        "shuffled_raw": shuffled_results,
    }

    out_path = OUT_DIR / "permutation_100_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")
