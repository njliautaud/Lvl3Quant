"""
MBO Alpha Phase 2 — Model A/B Test: Does MBO data add alpha beyond OHLCV?

Phase 1 found only 6 feature-horizon pairs with |IC|>0.03, all at 1h/2h
horizons, max IC=-0.064. The strongest features (vwap, microprice_close) are
just price-level proxies available from any OHLCV feed.

This script answers: "Is there genuine alpha in MBO-specific features, or is
everything we found just OHLCV information repackaged?"

Method:
  - Model A: LGBM on OHLCV features only (open, high, low, close, volume)
  - Model B: LGBM on OHLCV + all MBO-derived features
  - Walk-forward: 60-day sliding train window, 1-day OOT (HC #0: NEVER expanding)
  - Target: forward returns at 5min, 15min, 30min, 1h
  - Metric: Spearman IC per OOT day, then paired t-test on IC_B - IC_A
  - Verdict: MBO features add alpha only if IC difference is statistically
    significant (p < 0.05) AND practically meaningful (delta IC > 0.005)

Data: /home/jupiter/Lvl3Quant/data/processed/mbo_minute_bars_v1/*.parquet
"""

import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_rel, ttest_1samp

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_minute_bars_v1")
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/mbo_alpha_phase2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_WINDOW = 60  # sliding window size in days (HC #0: SLIDING, never expanding)
HORIZONS_MIN = {"5min": 5, "15min": 15, "30min": 30, "1h": 60}

OHLCV_COLS = ["open", "high", "low", "close", "volume"]

# Raw MBO columns from parquet (not derived)
MBO_RAW_COLS = [
    "spread_mean", "ofi_1min", "microprice_close",
    "signed_volume", "trade_count", "vwap",
]

LGBM_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "min_child_samples": 50,
    "verbose": -1,
    "n_jobs": -1,
}


# ---------------------------------------------------------------------------
# Feature engineering (derived from raw columns, no look-ahead)
# ---------------------------------------------------------------------------
def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive MBO-specific features from raw minute bar columns.
    All features use only past data (rolling backwards). No look-ahead.
    """
    out = df.copy()

    # Microprice deviation from close (how far the MBO-derived microprice
    # deviates from the last trade price — measures order book pressure)
    if "microprice_close" in out.columns and "close" in out.columns:
        out["microprice_dev"] = out["microprice_close"] - out["close"]

    # VWAP deviation from close (institutional flow indicator)
    if "vwap" in out.columns and "close" in out.columns:
        out["vwap_dev"] = out["vwap"] - out["close"]

    # OFI momentum: 5-bar rolling sum of order flow imbalance
    if "ofi_1min" in out.columns:
        out["ofi_momentum_5"] = out["ofi_1min"].rolling(5, min_periods=1).sum()

    # Volume ratio: current volume vs 5-bar rolling mean
    if "volume" in out.columns:
        vol_ma = out["volume"].rolling(5, min_periods=1).mean()
        out["volume_ratio_5"] = out["volume"] / vol_ma.replace(0, np.nan)

    # Signed volume ratio: fraction of volume that is signed (buy-sell imbalance)
    if "signed_volume" in out.columns and "volume" in out.columns:
        out["signed_volume_ratio"] = out["signed_volume"] / out["volume"].replace(0, np.nan)

    # Spread change (1-bar diff of spread — detects widening/tightening)
    if "spread_mean" in out.columns:
        out["spread_change"] = out["spread_mean"].diff()

    # Trade intensity: trade_count relative to its 5-bar moving average
    if "trade_count" in out.columns:
        tc_ma = out["trade_count"].rolling(5, min_periods=1).mean()
        out["trade_intensity"] = out["trade_count"] / tc_ma.replace(0, np.nan)

    return out


# Derived MBO feature names (added by derive_features)
MBO_DERIVED_COLS = [
    "microprice_dev", "vwap_dev", "ofi_momentum_5",
    "volume_ratio_5", "signed_volume_ratio", "spread_change",
    "trade_intensity",
]

# Also add OHLCV-derived technical features for a fair baseline
def derive_ohlcv_technicals(df: pd.DataFrame) -> pd.DataFrame:
    """Add basic technical features from OHLCV so Model A isn't trivially weak."""
    out = df.copy()
    # Returns
    out["ret_1"] = out["close"].pct_change()
    out["ret_5"] = out["close"].pct_change(5)
    # Range
    out["bar_range"] = out["high"] - out["low"]
    # Close position within bar
    rng = out["high"] - out["low"]
    out["close_position"] = ((out["close"] - out["low"]) / rng.replace(0, np.nan))
    # Volume change
    out["volume_change"] = out["volume"].pct_change()
    # 5-bar rolling volatility
    out["volatility_5"] = out["close"].pct_change().rolling(5, min_periods=1).std()
    # 10-bar rolling mean return
    out["ret_ma_10"] = out["close"].pct_change().rolling(10, min_periods=1).mean()
    return out

OHLCV_TECH_COLS = [
    "ret_1", "ret_5", "bar_range", "close_position",
    "volume_change", "volatility_5", "ret_ma_10",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_all_days() -> dict:
    """Load all parquet files, one DataFrame per day. Returns {date_str: df}."""
    files = sorted(DATA_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {DATA_DIR}")

    days = {}
    for f in files:
        date_str = f.stem  # e.g. "20250714"
        df = pd.read_parquet(f)

        # Drop first and last 5 bars (potential overnight artifacts)
        if len(df) > 20:
            df = df.iloc[5:-5].reset_index(drop=True)

        if len(df) < 30:
            continue

        # Derive all features
        df = derive_features(df)
        df = derive_ohlcv_technicals(df)

        days[date_str] = df

    print(f"Loaded {len(days)} trading days from {min(days.keys())} to {max(days.keys())}")
    return days


# ---------------------------------------------------------------------------
# Forward return computation (within-day only, no overnight leakage)
# ---------------------------------------------------------------------------
def compute_forward_returns(df: pd.DataFrame, horizon_min: int) -> np.ndarray:
    """
    Compute forward return over `horizon_min` minute bars.
    Returns NaN where the forward window would exceed the day boundary.
    """
    closes = df["close"].values
    n = len(closes)
    fwd_ret = np.full(n, np.nan)
    if horizon_min < n:
        fwd_ret[:n - horizon_min] = (
            closes[horizon_min:] - closes[:n - horizon_min]
        ) / closes[:n - horizon_min]
    return fwd_ret


# ---------------------------------------------------------------------------
# Walk-forward evaluation
# ---------------------------------------------------------------------------
def walk_forward_ab_test(
    days: dict,
    horizon_name: str,
    horizon_min: int,
) -> dict:
    """
    Walk-forward A/B model test for one horizon.

    SLIDING window: 60-day train, 1-day OOT, drop oldest day each step.
    (HC #0: NEVER expanding window.)

    Returns per-day ICs for Model A and Model B, plus feature importances.
    """
    import lightgbm as lgb

    sorted_dates = sorted(days.keys())
    n_days = len(sorted_dates)

    if n_days < TRAIN_WINDOW + 1:
        return {"error": f"Need {TRAIN_WINDOW+1} days, have {n_days}"}

    # Define feature sets
    model_a_features = OHLCV_COLS + OHLCV_TECH_COLS
    model_b_features = (
        OHLCV_COLS + OHLCV_TECH_COLS +
        MBO_RAW_COLS + MBO_DERIVED_COLS
    )

    # Filter to columns that actually exist
    sample_df = days[sorted_dates[0]]
    model_a_features = [c for c in model_a_features if c in sample_df.columns]
    model_b_features = [c for c in model_b_features if c in sample_df.columns]

    mbo_only_features = [c for c in model_b_features if c not in model_a_features]
    print(f"  Model A features ({len(model_a_features)}): {model_a_features}")
    print(f"  Model B features ({len(model_b_features)}): {model_b_features}")
    print(f"  MBO-only features ({len(mbo_only_features)}): {mbo_only_features}")

    ics_a = []
    ics_b = []
    oot_dates = []
    feat_importance_b = np.zeros(len(model_b_features))
    n_oot_samples = []

    n_oot_days = n_days - TRAIN_WINDOW
    print(f"  Walk-forward: {n_oot_days} OOT days (sliding {TRAIN_WINDOW}-day train window)")

    for oot_idx in range(TRAIN_WINDOW, n_days):
        oot_date = sorted_dates[oot_idx]

        # SLIDING: train on the most recent TRAIN_WINDOW days before OOT
        train_start = oot_idx - TRAIN_WINDOW
        train_dates = sorted_dates[train_start:oot_idx]

        # Build training set
        train_dfs = []
        for td in train_dates:
            tdf = days[td].copy()
            tdf["__fwd_ret"] = compute_forward_returns(tdf, horizon_min)
            train_dfs.append(tdf)
        train_all = pd.concat(train_dfs, ignore_index=True)

        # Build OOT set
        oot_df = days[oot_date].copy()
        oot_df["__fwd_ret"] = compute_forward_returns(oot_df, horizon_min)

        # Drop rows with NaN target
        train_all = train_all.dropna(subset=["__fwd_ret"])
        oot_df = oot_df.dropna(subset=["__fwd_ret"])

        if len(train_all) < 200 or len(oot_df) < 20:
            continue

        y_train = train_all["__fwd_ret"].values
        y_oot = oot_df["__fwd_ret"].values

        # --- Model A (OHLCV only) ---
        X_train_a = train_all[model_a_features].values.astype(np.float32)
        X_oot_a = oot_df[model_a_features].values.astype(np.float32)

        # Handle NaN in features (from rolling calcs at start of each day)
        train_mask_a = np.all(np.isfinite(X_train_a), axis=1)
        oot_mask_a = np.all(np.isfinite(X_oot_a), axis=1)

        if train_mask_a.sum() < 200 or oot_mask_a.sum() < 20:
            continue

        # Train Model A
        model_a = lgb.LGBMRegressor(**LGBM_PARAMS)
        try:
            # Use last 20% of training data as validation for early stopping
            n_tr = train_mask_a.sum()
            split = int(n_tr * 0.8)
            X_tr_a = X_train_a[train_mask_a]
            y_tr_a = y_train[train_mask_a]
            model_a.fit(
                X_tr_a[:split], y_tr_a[:split],
                eval_set=[(X_tr_a[split:], y_tr_a[split:])],
                callbacks=[lgb.early_stopping(20, verbose=False)],
            )
            preds_a = model_a.predict(X_oot_a[oot_mask_a])
        except Exception as e:
            print(f"    Model A failed on {oot_date}: {e}")
            continue

        # --- Model B (OHLCV + MBO) ---
        X_train_b = train_all[model_b_features].values.astype(np.float32)
        X_oot_b = oot_df[model_b_features].values.astype(np.float32)

        train_mask_b = np.all(np.isfinite(X_train_b), axis=1)
        oot_mask_b = np.all(np.isfinite(X_oot_b), axis=1)

        if train_mask_b.sum() < 200 or oot_mask_b.sum() < 20:
            continue

        model_b = lgb.LGBMRegressor(**LGBM_PARAMS)
        try:
            n_tr_b = train_mask_b.sum()
            split_b = int(n_tr_b * 0.8)
            X_tr_b = X_train_b[train_mask_b]
            y_tr_b = y_train[train_mask_b]
            model_b.fit(
                X_tr_b[:split_b], y_tr_b[:split_b],
                eval_set=[(X_tr_b[split_b:], y_tr_b[split_b:])],
                callbacks=[lgb.early_stopping(20, verbose=False)],
            )
            preds_b = model_b.predict(X_oot_b[oot_mask_b])
        except Exception as e:
            print(f"    Model B failed on {oot_date}: {e}")
            continue

        # Compute ICs using the intersection of valid masks
        # (both models must have predictions on the same rows for fair comparison)
        common_mask = oot_mask_a & oot_mask_b
        if common_mask.sum() < 20:
            continue

        # Re-predict on common mask to ensure same rows
        X_common_a = X_oot_a[common_mask]
        X_common_b = X_oot_b[common_mask]
        y_common = y_oot[common_mask]

        preds_a_common = model_a.predict(X_common_a)
        preds_b_common = model_b.predict(X_common_b)

        try:
            ic_a = spearmanr(preds_a_common, y_common)[0]
            ic_b = spearmanr(preds_b_common, y_common)[0]
        except Exception:
            continue

        if not (np.isfinite(ic_a) and np.isfinite(ic_b)):
            continue

        ics_a.append(float(ic_a))
        ics_b.append(float(ic_b))
        oot_dates.append(oot_date)
        n_oot_samples.append(int(common_mask.sum()))

        # Accumulate feature importance from Model B
        if hasattr(model_b, "feature_importances_"):
            feat_importance_b += model_b.feature_importances_

        # Progress
        done = oot_idx - TRAIN_WINDOW + 1
        if done <= 3 or done % 10 == 0 or done == n_oot_days:
            print(
                f"    [{done}/{n_oot_days}] {oot_date}: "
                f"IC_A={ic_a:+.4f}  IC_B={ic_b:+.4f}  "
                f"delta={ic_b - ic_a:+.4f}  n={common_mask.sum()}"
            )

        del model_a, model_b, train_all, oot_df
        gc.collect()

    if len(ics_a) < 5:
        return {"error": f"Only {len(ics_a)} valid OOT days"}

    # -----------------------------------------------------------------------
    # Statistical analysis
    # -----------------------------------------------------------------------
    ics_a = np.array(ics_a)
    ics_b = np.array(ics_b)
    ic_diff = ics_b - ics_a

    # Paired t-test: does Model B systematically beat Model A?
    t_stat_paired, p_val_paired = ttest_rel(ics_b, ics_a)

    # One-sample t-tests: are ICs significantly different from zero?
    t_stat_a, p_val_a = ttest_1samp(ics_a, 0)
    t_stat_b, p_val_b = ttest_1samp(ics_b, 0)

    # Feature importance ranking
    feat_imp_sorted = sorted(
        zip(model_b_features, feat_importance_b),
        key=lambda x: x[1],
        reverse=True,
    )

    # Separate MBO vs OHLCV importance
    total_imp = feat_importance_b.sum()
    mbo_imp = sum(v for k, v in feat_imp_sorted if k in mbo_only_features)
    ohlcv_imp = total_imp - mbo_imp

    result = {
        "horizon": horizon_name,
        "horizon_min": horizon_min,
        "n_oot_days": len(ics_a),
        "oot_dates": oot_dates,
        "n_oot_samples": n_oot_samples,
        # Model A stats
        "model_a_mean_ic": float(np.mean(ics_a)),
        "model_a_std_ic": float(np.std(ics_a)),
        "model_a_median_ic": float(np.median(ics_a)),
        "model_a_tstat": float(t_stat_a),
        "model_a_pval": float(p_val_a),
        "model_a_ics": ics_a.tolist(),
        # Model B stats
        "model_b_mean_ic": float(np.mean(ics_b)),
        "model_b_std_ic": float(np.std(ics_b)),
        "model_b_median_ic": float(np.median(ics_b)),
        "model_b_tstat": float(t_stat_b),
        "model_b_pval": float(p_val_b),
        "model_b_ics": ics_b.tolist(),
        # A/B comparison
        "delta_ic_mean": float(np.mean(ic_diff)),
        "delta_ic_std": float(np.std(ic_diff)),
        "delta_ic_median": float(np.median(ic_diff)),
        "paired_tstat": float(t_stat_paired),
        "paired_pval": float(p_val_paired),
        "b_wins_pct": float((ic_diff > 0).mean()),
        # Feature importance
        "feature_importance": feat_imp_sorted,
        "mbo_importance_pct": float(mbo_imp / total_imp * 100) if total_imp > 0 else 0.0,
        "ohlcv_importance_pct": float(ohlcv_imp / total_imp * 100) if total_imp > 0 else 0.0,
    }

    return result


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------
def generate_verdict(all_results: list) -> str:
    """Generate a plain-English verdict from all horizon results."""
    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append("MBO ALPHA PHASE 2 — A/B MODEL TEST RESULTS")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Question: Do MBO-derived features add predictive alpha beyond OHLCV?")
    lines.append(f"Method: 60-day sliding LGBM, {len(HORIZONS_MIN)} horizons, paired t-test on IC")
    lines.append("")

    any_sig = False
    for r in all_results:
        if "error" in r:
            lines.append(f"  {r['horizon']}: ERROR — {r['error']}")
            continue

        hz = r["horizon"]
        sig = r["paired_pval"] < 0.05 and r["delta_ic_mean"] > 0.005
        sig_str = "*** SIGNIFICANT ***" if sig else "(not significant)"
        if sig:
            any_sig = True

        lines.append(f"--- {hz} horizon ---")
        lines.append(
            f"  Model A (OHLCV):      mean IC = {r['model_a_mean_ic']:+.4f}  "
            f"(t={r['model_a_tstat']:.2f}, p={r['model_a_pval']:.4f})"
        )
        lines.append(
            f"  Model B (OHLCV+MBO):  mean IC = {r['model_b_mean_ic']:+.4f}  "
            f"(t={r['model_b_tstat']:.2f}, p={r['model_b_pval']:.4f})"
        )
        lines.append(
            f"  IC difference (B-A):  mean = {r['delta_ic_mean']:+.4f}  "
            f"(paired t={r['paired_tstat']:.2f}, p={r['paired_pval']:.4f}) {sig_str}"
        )
        lines.append(
            f"  Model B wins: {r['b_wins_pct']:.0%} of OOT days  "
            f"({r['n_oot_days']} days tested)"
        )
        lines.append(
            f"  Feature importance split: "
            f"OHLCV={r['ohlcv_importance_pct']:.1f}%  MBO={r['mbo_importance_pct']:.1f}%"
        )

        # Top 5 features
        lines.append(f"  Top 5 features (Model B):")
        for fname, fimp in r["feature_importance"][:5]:
            marker = " [MBO]" if fname not in (OHLCV_COLS + OHLCV_TECH_COLS) else ""
            lines.append(f"    {fname:<25s}: {fimp:>10.0f}{marker}")
        lines.append("")

    # Final verdict
    lines.append("=" * 80)
    lines.append("VERDICT")
    lines.append("=" * 80)

    if any_sig:
        sig_horizons = [
            r["horizon"] for r in all_results
            if "error" not in r and r["paired_pval"] < 0.05 and r["delta_ic_mean"] > 0.005
        ]
        lines.append(
            f"MBO features ADD significant alpha at: {', '.join(sig_horizons)}"
        )
        lines.append(
            "Recommendation: Investigate the top MBO features at these horizons "
            "for integration into the execution stack."
        )
    else:
        # Check if any model has signal at all
        any_model_sig = any(
            r.get("model_a_pval", 1) < 0.05 or r.get("model_b_pval", 1) < 0.05
            for r in all_results if "error" not in r
        )
        if any_model_sig:
            lines.append(
                "MBO features DO NOT add significant alpha beyond OHLCV."
            )
            lines.append(
                "Some predictability exists, but it's captured equally well by "
                "price/volume alone. The MBO-specific features (OFI, microprice, "
                "signed volume) are not providing incremental information."
            )
        else:
            lines.append(
                "NEITHER model shows significant predictive signal at these horizons."
            )
            lines.append(
                "At minute-bar granularity, forward returns appear essentially "
                "unpredictable by both OHLCV and MBO features. The MBO alpha "
                "(if any) may exist only at sub-second timescales, consistent "
                "with Phase 1 findings that signal decays rapidly."
            )

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("MBO Alpha Phase 2 — A/B Model Test")
    print("=" * 60)
    t_start = time.time()

    # Load data
    print("\n[1/3] Loading data...")
    days = load_all_days()

    # Run walk-forward for each horizon
    print(f"\n[2/3] Running walk-forward A/B tests ({len(HORIZONS_MIN)} horizons)...")
    all_results = []
    for hz_name, hz_min in HORIZONS_MIN.items():
        print(f"\n{'='*60}")
        print(f"Horizon: {hz_name} ({hz_min} minute bars forward)")
        print(f"{'='*60}")
        result = walk_forward_ab_test(days, hz_name, hz_min)
        all_results.append(result)

        if "error" in result:
            print(f"  ERROR: {result['error']}")
        else:
            print(
                f"\n  RESULT: Model A IC={result['model_a_mean_ic']:+.4f}  "
                f"Model B IC={result['model_b_mean_ic']:+.4f}  "
                f"delta={result['delta_ic_mean']:+.4f}  "
                f"p={result['paired_pval']:.4f}"
            )

    # Generate verdict
    print("\n[3/3] Generating verdict...")
    verdict = generate_verdict(all_results)
    print(verdict)

    # Save results
    elapsed = time.time() - t_start
    output = {
        "run_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": elapsed,
        "config": {
            "train_window": TRAIN_WINDOW,
            "horizons": HORIZONS_MIN,
            "lgbm_params": LGBM_PARAMS,
            "model_a_features": OHLCV_COLS + OHLCV_TECH_COLS,
            "model_b_features": OHLCV_COLS + OHLCV_TECH_COLS + MBO_RAW_COLS + MBO_DERIVED_COLS,
        },
        "results": [],
    }

    # Serialize results (convert numpy arrays to lists for JSON)
    for r in all_results:
        r_clean = {}
        for k, v in r.items():
            if isinstance(v, np.ndarray):
                r_clean[k] = v.tolist()
            elif isinstance(v, (np.floating, np.integer)):
                r_clean[k] = float(v)
            else:
                r_clean[k] = v
        output["results"].append(r_clean)

    output["verdict"] = verdict

    results_path = OUTPUT_DIR / "phase2_ab_test_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Save verdict as text too
    verdict_path = OUTPUT_DIR / "phase2_verdict.txt"
    with open(verdict_path, "w") as f:
        f.write(verdict)
    print(f"Verdict saved to {verdict_path}")

    print(f"\nTotal elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
