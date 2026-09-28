#!/usr/bin/env python3
"""
Train Final Sub-Sector Rotation LGBM Model
============================================
Trains on ALL available data for the 5 adversarial-validated pairs and saves
a production model to models/subsector_rotation_lgbm.pkl.

Validated pairs (from adversarial validation):
  1. VNQ vs XLRE (REITs vs Real Estate) — 5d horizon, Sharpe 4.6
  2. GDX vs XME (Gold Miners vs Metals) — 5d horizon, Sharpe 2.6
  3. KRE vs XLF (Regional Banks vs Financials) — 10d horizon, Sharpe 2.1
  4. XLY vs XLP (Consumer Disc vs Staples) — 5d horizon, Sharpe 2.1
  5. KBE vs KIE (Banks vs Insurance) — 5d horizon, Sharpe 2.0

Usage:
    python3 scripts/growth_research/train_subsector_rotation_model.py
"""

import json
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import yfinance as yf

warnings.filterwarnings("ignore")

BASE_DIR = Path("/home/jupiter/Lvl3Quant")
MODEL_PATH = BASE_DIR / "models" / "subsector_rotation_lgbm.pkl"

# The 5 validated pairs with their optimal horizons
VALIDATED_PAIRS = {
    "VNQ_vs_XLRE": {"etf_a": "VNQ", "etf_b": "XLRE", "horizon": 5,
                     "label_a": "REITs", "label_b": "Real Estate"},
    "GDX_vs_XME": {"etf_a": "GDX", "etf_b": "XME", "horizon": 5,
                    "label_a": "Gold Miners", "label_b": "Metals/Mining"},
    "KRE_vs_XLF": {"etf_a": "KRE", "etf_b": "XLF", "horizon": 10,
                    "label_a": "Regional Banks", "label_b": "Financials"},
    "XLY_vs_XLP": {"etf_a": "XLY", "etf_b": "XLP", "horizon": 5,
                    "label_a": "Consumer Disc", "label_b": "Consumer Staples"},
    "KBE_vs_KIE": {"etf_a": "KBE", "etf_b": "KIE", "horizon": 5,
                    "label_a": "Banks", "label_b": "Insurance"},
}

TRAIN_DAYS = 252
START_DATE = "2020-01-01"

LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "verbosity": -1,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "n_estimators": 200,
    "random_state": 42,
}


def compute_features(close_a, close_b, spy_close):
    """Compute rotation features for a sub-sector pair."""
    ret_a = close_a.pct_change()
    ret_b = close_b.pct_change()
    ratio = close_a / close_b

    features = pd.DataFrame(index=close_a.index)

    # Relative returns over multiple windows
    for w in [5, 10, 21, 63]:
        features[f"rel_ret_{w}d"] = (close_a / close_a.shift(w)) / (close_b / close_b.shift(w)) - 1

    # RSI of the ratio (14-day)
    delta = ratio.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    features["ratio_rsi_14"] = 100 - (100 / (1 + rs))

    # Z-score of relative returns
    ratio_ret = ratio.pct_change()
    for w in [21, 63]:
        roll_mean = ratio_ret.rolling(w).mean()
        roll_std = ratio_ret.rolling(w).std()
        features[f"rel_zscore_{w}d"] = (ratio_ret - roll_mean) / roll_std.replace(0, np.nan)

    # Rolling correlation
    for w in [21, 63]:
        features[f"corr_{w}d"] = ret_a.rolling(w).corr(ret_b)

    # Correlation change
    features["corr_change_21d"] = features["corr_21d"] - features["corr_21d"].shift(21)

    # Volatility ratio
    vol_a = ret_a.rolling(21).std()
    vol_b = ret_b.rolling(21).std()
    features["vol_ratio_21d"] = vol_a / vol_b.replace(0, np.nan)

    # Momentum divergence
    mom_a = close_a / close_a.shift(21) - 1
    mom_b = close_b / close_b.shift(21) - 1
    features["mom_divergence_21d"] = mom_a - mom_b

    # Ratio distance from 63d SMA
    ratio_sma = ratio.rolling(63).mean()
    features["ratio_dist_sma63"] = (ratio / ratio_sma) - 1

    # SPY regime
    spy_sma200 = spy_close.rolling(200).mean()
    features["spy_regime"] = (spy_close > spy_sma200).astype(int)

    # SPY recent performance
    features["spy_ret_21d"] = spy_close.pct_change(21)

    return features


def compute_target(close_a, close_b, horizon):
    """Target: does mean reversion happen over the next horizon days?"""
    past_rel = (close_a / close_a.shift(21)) / (close_b / close_b.shift(21)) - 1
    fwd_ret_a = close_a.pct_change(horizon).shift(-horizon)
    fwd_ret_b = close_b.pct_change(horizon).shift(-horizon)

    reversal = np.where(
        past_rel < 0,
        fwd_ret_a > fwd_ret_b,
        fwd_ret_b > fwd_ret_a,
    ).astype(float)

    return pd.Series(reversal, index=close_a.index)


def download_data():
    """Download all needed ETF data."""
    tickers = set()
    for pair in VALIDATED_PAIRS.values():
        tickers.add(pair["etf_a"])
        tickers.add(pair["etf_b"])
    tickers.add("SPY")
    tickers = sorted(tickers)

    print(f"Downloading {len(tickers)} ETFs: {', '.join(tickers)}")
    data = yf.download(tickers, start=START_DATE, auto_adjust=True, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
    else:
        close = data

    print(f"Data shape: {close.shape}, range: {close.index[0].date()} to {close.index[-1].date()}")
    return close


def train_pair_model(close_a, close_b, spy_close, horizon, pair_name):
    """Train LGBM on full available data for a single pair."""
    features = compute_features(close_a, close_b, spy_close)
    target = compute_target(close_a, close_b, horizon)

    combined = features.copy()
    combined["target"] = target
    combined = combined.dropna()

    if len(combined) < TRAIN_DAYS:
        print(f"  WARNING: {pair_name} has insufficient data ({len(combined)} rows)")
        return None

    feature_cols = [c for c in features.columns if c in combined.columns]

    X = combined[feature_cols].values
    y = combined["target"].values

    if len(np.unique(y)) < 2:
        print(f"  WARNING: {pair_name} has no class variation")
        return None

    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(X, y)

    # Report in-sample stats
    proba = model.predict_proba(X)[:, 1]
    from sklearn.metrics import accuracy_score, roc_auc_score
    acc = accuracy_score(y, (proba >= 0.5).astype(int))
    auc = roc_auc_score(y, proba)
    high_conf = proba >= 0.6
    high_conf_acc = accuracy_score(y[high_conf], (proba[high_conf] >= 0.5).astype(int)) if high_conf.sum() > 0 else 0

    print(f"  {pair_name} ({horizon}d): {len(combined)} samples, "
          f"ACC={acc:.3f}, AUC={auc:.3f}, "
          f"HighConf(>60%)={high_conf.sum()} trades, ACC={high_conf_acc:.3f}")

    return {
        "model": model,
        "feature_cols": feature_cols,
        "n_samples": len(combined),
        "accuracy": round(acc, 4),
        "auc": round(auc, 4),
    }


def main():
    print("=" * 70)
    print("TRAINING SUBSECTOR ROTATION LGBM — PRODUCTION MODEL")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    close = download_data()
    spy_close = close["SPY"] if "SPY" in close.columns else None

    models = {}
    metadata = {
        "trained_at": datetime.now().isoformat(),
        "start_date": START_DATE,
        "end_date": str(close.index[-1].date()),
        "lgbm_params": LGBM_PARAMS,
        "pairs": {},
    }

    for pair_name, pair_info in VALIDATED_PAIRS.items():
        etf_a = pair_info["etf_a"]
        etf_b = pair_info["etf_b"]
        horizon = pair_info["horizon"]

        if etf_a not in close.columns or etf_b not in close.columns:
            print(f"  SKIPPING {pair_name}: missing data for {etf_a} or {etf_b}")
            continue

        close_a = close[etf_a].dropna()
        close_b = close[etf_b].dropna()
        common_idx = close_a.index.intersection(close_b.index)
        if spy_close is not None:
            common_idx = common_idx.intersection(spy_close.index)
        close_a = close_a.loc[common_idx]
        close_b = close_b.loc[common_idx]
        spy_aligned = spy_close.loc[common_idx] if spy_close is not None else close_a * 0 + 1

        result = train_pair_model(close_a, close_b, spy_aligned, horizon, pair_name)
        if result:
            models[pair_name] = {
                "model": result["model"],
                "feature_cols": result["feature_cols"],
                "horizon": horizon,
                "etf_a": etf_a,
                "etf_b": etf_b,
                "label_a": pair_info["label_a"],
                "label_b": pair_info["label_b"],
            }
            metadata["pairs"][pair_name] = {
                "etf_a": etf_a,
                "etf_b": etf_b,
                "horizon": horizon,
                "n_samples": result["n_samples"],
                "accuracy": result["accuracy"],
                "auc": result["auc"],
            }

    # Save everything as a pickle
    save_obj = {
        "models": models,
        "metadata": metadata,
        "validated_pairs": VALIDATED_PAIRS,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(save_obj, f)

    print(f"\nSaved {len(models)} pair models to {MODEL_PATH}")
    print(f"Metadata: {json.dumps(metadata, indent=2, default=str)}")

    # Also save metadata as JSON for easy inspection
    meta_path = MODEL_PATH.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Metadata JSON saved to {meta_path}")


if __name__ == "__main__":
    main()
