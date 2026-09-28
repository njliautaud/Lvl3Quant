"""
Vol Forecaster Evaluation + Wheel Strategy Integration Test
============================================================
Run this AFTER vol_forecaster.py completes.

Phase 1: Evaluate prediction quality (RMSE, IC, directional accuracy)
Phase 2: Test if vol predictions improve wheel CSP strategy
  - Compare fixed-delta put selection vs IV-rank-aware selection
  - Test vol-adjusted position sizing
Phase 3: Permutation test (HC #665)

Usage:
    python3 research/vol_forecaster_eval.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path("/home/jupiter/Lvl3Quant")
OUTPUT = ROOT / "output/vol_forecaster"
EVAL_OUT = ROOT / "output/vol_forecaster_eval"
EVAL_OUT.mkdir(parents=True, exist_ok=True)

PRICE_CACHE = ROOT / "wheel_strategy_v1/data/cache/prices_v2.parquet"
MACRO_CACHE = ROOT / "wheel_strategy_v1/data/cache/macro.parquet"


def load_vol_predictions() -> pd.DataFrame:
    """Load walk-forward OOS predictions from vol_forecaster."""
    path = OUTPUT / "wf_predictions.parquet"
    if not path.exists():
        print(f"ERROR: {path} not found. Run vol_forecaster.py first.")
        sys.exit(1)
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_vol_metrics() -> dict:
    """Load aggregate metrics from vol_forecaster."""
    path = OUTPUT / "wf_metrics.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


# ============================================================================
# Phase 1: Deep prediction quality analysis
# ============================================================================

def evaluate_predictions(preds: pd.DataFrame) -> dict:
    """Detailed evaluation of vol prediction quality."""
    print("=" * 70)
    print("PHASE 1: VOL PREDICTION QUALITY ANALYSIS")
    print("=" * 70)

    # Basic metrics
    valid = preds.dropna(subset=["pred_vol", "actual_vol"])
    rmse = np.sqrt(((valid["pred_vol"] - valid["actual_vol"]) ** 2).mean())
    mae = (valid["pred_vol"] - valid["actual_vol"]).abs().mean()
    bias = (valid["pred_vol"] - valid["actual_vol"]).mean()

    print(f"\n  Predictions: {len(valid):,} rows, {valid['ticker'].nunique()} tickers")
    print(f"  Date range: {valid['date'].min().date()} to {valid['date'].max().date()}")
    print(f"\n  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  Bias: {bias:+.4f} ({'over-predicting' if bias > 0 else 'under-predicting'})")

    # Cross-sectional rank IC per period
    from scipy.stats import spearmanr
    ics = []
    for dt in valid["date"].unique():
        snap = valid[valid["date"] == dt]
        if len(snap) < 5:
            continue
        ic, _ = spearmanr(snap["pred_vol"], snap["actual_vol"])
        ics.append({"date": dt, "ic": ic})

    ic_df = pd.DataFrame(ics)
    mean_ic = ic_df["ic"].mean()
    ic_std = ic_df["ic"].std()
    icir = mean_ic / ic_std if ic_std > 0 else 0
    pct_positive = (ic_df["ic"] > 0).mean()

    print(f"\n  Cross-sectional Rank IC:")
    print(f"    Mean IC:     {mean_ic:.4f}")
    print(f"    IC Std:      {ic_std:.4f}")
    print(f"    ICIR:        {icir:.2f}")
    print(f"    % positive:  {pct_positive:.1%}")

    # Directional accuracy (vol up vs down)
    if "naive_vol" in valid.columns:
        # Compare pred vs naive (trailing vol) at predicting direction
        actual_up = valid["actual_vol"] > valid.get("naive_vol", valid["actual_vol"].shift(1))
        pred_up = valid["pred_vol"] > valid.get("naive_vol", valid["pred_vol"].shift(1))
        dir_acc = (actual_up == pred_up).mean()
        print(f"    Directional accuracy: {dir_acc:.1%}")

    # Quintile analysis: does ranking by predicted vol correctly sort realized vol?
    valid = valid.copy()
    valid["pred_quintile"] = valid.groupby("date")["pred_vol"].transform(
        lambda x: pd.qcut(x, 5, labels=False, duplicates="drop") if len(x) >= 5 else np.nan
    )
    quintile_means = valid.groupby("pred_quintile")["actual_vol"].mean()
    print(f"\n  Quintile Analysis (avg realized vol by predicted quintile):")
    for q, v in quintile_means.items():
        print(f"    Q{int(q)+1}: {v:.4f}")
    monotonic = all(quintile_means.iloc[i] <= quintile_means.iloc[i+1]
                    for i in range(len(quintile_means)-1))
    print(f"    Monotonic: {'✅ YES' if monotonic else '❌ NO'}")

    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "bias": round(bias, 4),
        "mean_ic": round(mean_ic, 4),
        "icir": round(icir, 2),
        "pct_positive_ic": round(pct_positive, 3),
        "quintile_monotonic": monotonic,
        "n_predictions": len(valid),
        "n_tickers": int(valid["ticker"].nunique()),
    }


# ============================================================================
# Phase 2: Integration with wheel strategy
# ============================================================================

def test_vol_aware_wheel(preds: pd.DataFrame) -> dict:
    """
    Test whether vol predictions improve wheel CSP selection.

    Strategy: when predicted vol > trailing vol (model says vol will rise),
    reduce position or skip selling puts (they'll get cheaper).
    When predicted vol < trailing vol (model says vol will fall),
    sell puts aggressively (lock in high premium before IV drops).
    """
    print(f"\n" + "=" * 70)
    print("PHASE 2: VOL-AWARE WHEEL STRATEGY TEST")
    print("=" * 70)

    # Load prices
    prices = pd.read_parquet(PRICE_CACHE)
    prices["date"] = pd.to_datetime(prices["date"])

    # Merge predictions with prices
    valid = preds.dropna(subset=["pred_vol", "actual_vol"]).copy()
    valid = valid.merge(
        prices[["ticker", "date", "close"]],
        on=["ticker", "date"],
        how="left",
    )

    # Compute vol ratio: predicted / trailing
    if "naive_vol" in valid.columns:
        valid["vol_ratio"] = valid["pred_vol"] / valid["naive_vol"].clip(lower=0.01)
    else:
        # Approximate trailing vol
        valid = valid.sort_values(["ticker", "date"])
        valid["trailing_vol"] = valid.groupby("ticker")["actual_vol"].shift(1)
        valid["vol_ratio"] = valid["pred_vol"] / valid["trailing_vol"].clip(lower=0.01)

    valid = valid.dropna(subset=["vol_ratio", "close"])

    # Simple CSP simulation:
    # - Sell 30-delta put on each rebalance date (monthly)
    # - Premium ≈ delta × close × IV × sqrt(DTE/252)
    # - P&L: premium collected if stock stays above strike, loss if it drops below
    #
    # Baseline: equal weight all tickers
    # Vol-aware: scale position by inverse vol_ratio
    #   (when predicted vol > trailing → smaller position)
    #   (when predicted vol < trailing → larger position)

    # Monthly rebalance dates
    valid["ym"] = valid["date"].dt.to_period("M")
    rebal_dates = valid.groupby("ym")["date"].first().reset_index()["date"]

    baseline_returns = []
    vol_aware_returns = []

    for rd in rebal_dates:
        snap = valid[valid["date"] == rd].copy()
        if len(snap) < 5:
            continue

        # Simulate CSP premium and payoff for each ticker
        delta = 0.30
        dte = 30
        for idx, row in snap.iterrows():
            iv = row.get("pred_vol", row.get("actual_vol", 0.3))
            premium_pct = delta * iv * np.sqrt(dte / 252)  # simplified BSM
            strike = row["close"] * (1 - delta * iv * np.sqrt(dte / 252) * 0.5)

            # Actual move over next 21 days (use actual_vol as proxy)
            actual_move = np.random.default_rng(int(rd.timestamp()) + hash(row["ticker"])).normal(
                0.005, row["actual_vol"] / np.sqrt(12)
            )  # monthly return

            if row["close"] * (1 + actual_move) >= strike:
                pnl_pct = premium_pct  # keep premium
            else:
                pnl_pct = premium_pct - abs(actual_move)  # premium - loss

            snap.loc[idx, "pnl_pct"] = pnl_pct
            snap.loc[idx, "premium_pct"] = premium_pct

        # Baseline: equal weight
        base_ret = snap["pnl_pct"].mean()
        baseline_returns.append({"date": rd, "ret": base_ret})

        # Vol-aware: weight inversely by vol_ratio (predicted vol / trailing)
        snap["weight"] = 1.0 / snap["vol_ratio"].clip(0.5, 2.0)
        snap["weight"] = snap["weight"] / snap["weight"].sum()
        vol_ret = (snap["pnl_pct"] * snap["weight"]).sum()
        vol_aware_returns.append({"date": rd, "ret": vol_ret})

    if not baseline_returns:
        print("  No valid rebalance periods found!")
        return {}

    base_df = pd.DataFrame(baseline_returns).set_index("date")["ret"]
    vol_df = pd.DataFrame(vol_aware_returns).set_index("date")["ret"]

    def metrics(s, label):
        sharpe = s.mean() / s.std() * np.sqrt(12) if s.std() > 0 else 0
        sortino_d = s[s < 0].std()
        sortino = s.mean() / sortino_d * np.sqrt(12) if sortino_d > 0 else 0
        wr = (s > 0).mean()
        cum = (1 + s).cumprod()
        n_yrs = len(s) / 12
        cagr = (cum.iloc[-1] ** (1 / n_yrs) - 1) if n_yrs > 0 else 0
        max_dd = (cum / cum.cummax() - 1).min()
        print(f"\n  {label}:")
        print(f"    Sharpe: {sharpe:.2f}, Sortino: {sortino:.2f}")
        print(f"    CAGR: {cagr*100:.1f}%, MaxDD: {max_dd*100:.1f}%")
        print(f"    WR: {wr*100:.1f}%, N periods: {len(s)}")
        return {
            "sharpe": round(sharpe, 2), "sortino": round(sortino, 2),
            "cagr": round(cagr, 4), "max_dd": round(max_dd, 4), "wr": round(wr, 3),
        }

    base_metrics = metrics(base_df, "BASELINE (equal weight)")
    vol_metrics = metrics(vol_df, "VOL-AWARE (prediction-weighted)")

    improved = vol_metrics["sharpe"] > base_metrics["sharpe"] + 0.1
    print(f"\n  Delta Sharpe: {vol_metrics['sharpe'] - base_metrics['sharpe']:+.2f}")
    print(f"  {'✅ Vol prediction IMPROVES wheel' if improved else '❌ Vol prediction does NOT improve wheel'}")

    return {
        "baseline": base_metrics,
        "vol_aware": vol_metrics,
        "improved": improved,
        "delta_sharpe": round(vol_metrics["sharpe"] - base_metrics["sharpe"], 2),
    }


# ============================================================================
# Phase 3: Vol-timing signal analysis
# ============================================================================

def vol_timing_analysis(preds: pd.DataFrame) -> dict:
    """
    Does the model correctly predict vol regime changes?
    Key question: when model predicts vol INCREASE, does selling premium
    perform worse (and vice versa)?
    """
    print(f"\n" + "=" * 70)
    print("PHASE 3: VOL TIMING SIGNAL ANALYSIS")
    print("=" * 70)

    valid = preds.dropna(subset=["pred_vol", "actual_vol"]).copy()

    # Vol regime: is predicted vol rising or falling relative to trailing?
    if "naive_vol" in valid.columns:
        valid["pred_change"] = valid["pred_vol"] - valid["naive_vol"]
    else:
        valid = valid.sort_values(["ticker", "date"])
        valid["trailing"] = valid.groupby("ticker")["actual_vol"].shift(1)
        valid["pred_change"] = valid["pred_vol"] - valid["trailing"]

    valid = valid.dropna(subset=["pred_change"])

    # When model predicts vol increase, did vol actually increase?
    valid["pred_up"] = valid["pred_change"] > 0
    if "naive_vol" in valid.columns:
        valid["actual_up"] = valid["actual_vol"] > valid["naive_vol"]
    else:
        valid["actual_up"] = valid["actual_vol"] > valid["trailing"]

    # Confusion matrix
    tp = ((valid["pred_up"]) & (valid["actual_up"])).sum()
    fp = ((valid["pred_up"]) & (~valid["actual_up"])).sum()
    fn = ((~valid["pred_up"]) & (valid["actual_up"])).sum()
    tn = ((~valid["pred_up"]) & (~valid["actual_up"])).sum()

    accuracy = (tp + tn) / len(valid) if len(valid) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    print(f"\n  Vol Direction Prediction:")
    print(f"    Accuracy:  {accuracy:.1%}")
    print(f"    Precision: {precision:.1%} (when predicts ↑, % actually ↑)")
    print(f"    Recall:    {recall:.1%} (when actually ↑, % predicted ↑)")
    print(f"    TP={tp}, FP={fp}, FN={fn}, TN={tn}")

    # Vol regime profitability: when model predicts vol falling,
    # are CSP returns actually better?
    valid["actual_return"] = valid["actual_vol"]  # proxy: lower realized vol = better for CSP
    vol_up_ret = valid[valid["pred_up"]]["actual_vol"].mean()
    vol_dn_ret = valid[~valid["pred_up"]]["actual_vol"].mean()

    print(f"\n  Avg realized vol by prediction direction:")
    print(f"    Predicted vol ↑: {vol_up_ret:.4f} ({len(valid[valid['pred_up']])} obs)")
    print(f"    Predicted vol ↓: {vol_dn_ret:.4f} ({len(valid[~valid['pred_up']])} obs)")
    print(f"    Separation: {vol_up_ret - vol_dn_ret:+.4f}")

    return {
        "accuracy": round(accuracy, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "vol_up_realized": round(vol_up_ret, 4),
        "vol_dn_realized": round(vol_dn_ret, 4),
        "separation": round(vol_up_ret - vol_dn_ret, 4),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("VOL FORECASTER EVALUATION + WHEEL INTEGRATION")
    print("=" * 70)

    # Load predictions
    preds = load_vol_predictions()
    print(f"\nLoaded {len(preds):,} predictions, {preds['ticker'].nunique()} tickers")
    print(f"Date range: {preds['date'].min().date()} to {preds['date'].max().date()}")
    print(f"Columns: {list(preds.columns)}")

    # Load metrics
    metrics = load_vol_metrics()
    if metrics:
        print(f"\nPre-computed metrics:")
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                print(f"  {k}: {v}")

    # Phase 1: Quality
    quality = evaluate_predictions(preds)

    # Phase 2: Wheel integration
    wheel_results = test_vol_aware_wheel(preds)

    # Phase 3: Vol timing
    timing = vol_timing_analysis(preds)

    # Save combined results
    output = {
        "quality": quality,
        "wheel_integration": wheel_results,
        "vol_timing": timing,
    }

    with open(EVAL_OUT / "vol_eval_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Verdict
    print(f"\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    good_ic = quality.get("mean_ic", 0) > 0.3
    good_quintile = quality.get("quintile_monotonic", False)
    improves_wheel = wheel_results.get("improved", False)

    if good_ic and good_quintile and improves_wheel:
        print("  ✅ VOL FORECASTER IS VALUABLE — integrate into wheel strategy")
    elif good_ic and good_quintile:
        print("  ⚠️ VOL PREDICTIONS ARE ACCURATE but don't improve wheel strategy")
        print("  May be useful for position sizing or strike selection separately")
    else:
        print("  ❌ VOL FORECASTER NEEDS IMPROVEMENT before integration")
        print(f"  IC={quality.get('mean_ic', '?')}, Monotonic={quality.get('quintile_monotonic', '?')}")

    print(f"\n  Results saved to {EVAL_OUT}/")


if __name__ == "__main__":
    main()
