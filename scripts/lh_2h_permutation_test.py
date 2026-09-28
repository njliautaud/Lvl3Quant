#!/usr/bin/env python3
"""
2h LGBM Model — HC #659 Permutation Test
=========================================

The longer_horizon_v2 permutation diagnostic showed ~75% artifact.
This test checks whether the CURRENT 2h paper engine model has
the same problem using the exact same training framework.

Method:
  - Load minute bar data → aggregate to hourly
  - Walk-forward: train on 60 days, predict next day, slide by 1
  - Compare REAL labels vs SHUFFLED labels (5 random seeds)
  - If shuffled labels produce similar IC/Sharpe → model is overfitting

HC #659 gate: random directions must LOSE money.
"""

import sys, os, json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(ROOT))

# Import the paper engine's feature builder
sys.path.insert(0, str(ROOT / "live_trading_linux"))
from lh_2h_paper_engine import (
    compute_enhanced_hourly, add_rolling_features, add_regime_context,
    get_feature_cols, TRAIN_DAYS, PURGE_DAYS, HORIZON_BARS, LGBM_PARAMS,
    MINUTE_BAR_DIR, LOOKBACK_DAYS, LH2hPaperEngine
)

OUT_DIR = ROOT / "output" / "lh_2h_permutation_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Use the paper engine's own data loading pipeline
engine = LH2hPaperEngine()
print("Loading minute bar data via engine._load_minute_bars()...")
minute_df = engine._load_minute_bars(n_days=999)  # load ALL available data
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

# Walk-forward with real and shuffled labels
import lightgbm as lgb

def walk_forward(hourly_clean, dates, feature_cols, shuffle_labels=False, seed=42):
    """Run walk-forward and return per-day OOT predictions."""
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

# Run REAL labels
print("\n=== REAL LABELS ===")
real_preds, real_actuals, real_dates = walk_forward(hourly_clean, dates, feature_cols, shuffle_labels=False)
real_ic = float(stats.spearmanr(real_preds, real_actuals)[0])
real_dir_accuracy = ((real_preds > 0) == (real_actuals > 0)).mean()

# Simple trading sim: go direction of prediction, earn/lose fwd_ticks
real_gross = np.where(real_preds > 0, real_actuals, -real_actuals)
COST = 1.376  # ticks RT
real_net = real_gross - COST
real_sharpe = real_net.mean() / real_net.std() * np.sqrt(252 * 7) if real_net.std() > 0 else 0  # ~7 trades/day

print(f"OOT IC: {real_ic:.4f}")
print(f"Direction accuracy: {real_dir_accuracy*100:.1f}%")
print(f"Avg gross ticks/trade: {real_gross.mean():.2f}")
print(f"Avg net ticks/trade: {real_net.mean():.2f}")
print(f"Sharpe (ann.): {real_sharpe:.2f}")
print(f"Total trades: {len(real_preds)}")

# Run SHUFFLED labels (3 seeds)
shuffle_results = []
for seed in [42, 123, 999]:
    print(f"\n=== SHUFFLED LABELS (seed={seed}) ===")
    sh_preds, sh_actuals, sh_dates = walk_forward(hourly_clean, dates, feature_cols, shuffle_labels=True, seed=seed)
    sh_ic = float(stats.spearmanr(sh_preds, sh_actuals)[0])
    sh_dir = ((sh_preds > 0) == (sh_actuals > 0)).mean()
    sh_gross = np.where(sh_preds > 0, sh_actuals, -sh_actuals)
    sh_net = sh_gross - COST
    sh_sharpe = sh_net.mean() / sh_net.std() * np.sqrt(252 * 7) if sh_net.std() > 0 else 0

    print(f"OOT IC: {sh_ic:.4f}")
    print(f"Direction accuracy: {sh_dir*100:.1f}%")
    print(f"Avg net ticks/trade: {sh_net.mean():.2f}")
    print(f"Sharpe (ann.): {sh_sharpe:.2f}")

    shuffle_results.append({
        "seed": seed,
        "ic": round(sh_ic, 4),
        "direction_accuracy": round(sh_dir * 100, 1),
        "avg_net_ticks": round(float(sh_net.mean()), 3),
        "sharpe": round(sh_sharpe, 2),
    })

# Compute genuine edge
avg_shuffle_ic = np.mean([r["ic"] for r in shuffle_results])
avg_shuffle_sharpe = np.mean([r["sharpe"] for r in shuffle_results])
genuine_ic = real_ic - avg_shuffle_ic
genuine_sharpe = real_sharpe - avg_shuffle_sharpe

print(f"\n{'='*60}")
print(f"PERMUTATION TEST RESULT")
print(f"{'='*60}")
print(f"Real IC:      {real_ic:.4f}")
print(f"Shuffle IC:   {avg_shuffle_ic:.4f} (avg of 3)")
print(f"Genuine IC:   {genuine_ic:.4f}")
print(f"")
print(f"Real Sharpe:      {real_sharpe:.2f}")
print(f"Shuffle Sharpe:   {avg_shuffle_sharpe:.2f} (avg of 3)")
print(f"Genuine Sharpe:   {genuine_sharpe:.2f}")
print(f"")
artifact_pct = abs(avg_shuffle_ic / real_ic) * 100 if real_ic != 0 else 0
print(f"Artifact %: {artifact_pct:.0f}%")

if avg_shuffle_sharpe > 0:
    print(f"\n⚠ SHUFFLED LABELS ARE PROFITABLE → result is ARTIFACT per HC #659")
    verdict = "FAIL — shuffled labels profitable"
elif genuine_ic < 0.02:
    print(f"\n⚠ Genuine IC < 0.02 → effectively no real signal")
    verdict = "FAIL — genuine IC too small"
else:
    print(f"\n✅ PASSES — shuffled labels lose money, genuine IC is real")
    verdict = "PASS"

# Save results
report = {
    "model": "2h LGBM (lh_2h_paper_engine framework)",
    "test_type": "permutation (HC #659)",
    "n_oot_days": len(set(real_dates)),
    "n_trades": len(real_preds),
    "real": {
        "ic": round(real_ic, 4),
        "direction_accuracy": round(real_dir_accuracy * 100, 1),
        "avg_net_ticks": round(float(real_net.mean()), 3),
        "sharpe": round(real_sharpe, 2),
    },
    "shuffled": shuffle_results,
    "genuine_ic": round(genuine_ic, 4),
    "genuine_sharpe": round(genuine_sharpe, 2),
    "artifact_pct": round(artifact_pct, 1),
    "verdict": verdict,
}

with open(OUT_DIR / "permutation_test.json", "w") as f:
    json.dump(report, f, indent=2)

print(f"\nSaved: {OUT_DIR / 'permutation_test.json'}")
