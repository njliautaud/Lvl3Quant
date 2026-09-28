"""
FIFO Outcome Predictor — Train LGBM/XGBoost on per-trade FIFO data.

The key question: are there CONDITIONS where FIFO trades are profitable?
The average is negative (-0.6 to -1.2 ticks/trade), but maybe there are
pockets (thin book + strong signal + favorable time) where trades work.

Approach:
1. Collect per-trade data from FIFO validation runs
2. Extract features: signal_strength, queue_position, book_size, fill_latency, time_of_day
3. Also load gated predictions .npz for additional signal features
4. Train LGBM to predict: binary (profitable vs not) and regression (pnl_ticks)
5. Walk-forward validation: train on first 70% of dates, test on last 30%
6. Check if top-scored trades have positive edge

Usage:
    python3 -m alpha_discovery.execution.fifo_outcome_predictor \
        --input-dirs output/fifo_validate_v3/chase_conv30_short_c03 output/fifo_validate_v3/chase_conv30_long_c03 \
        --output-dir output/fifo_outcome_predictor_v1
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

LOG = logging.getLogger("FIFO_OUTCOME")


def load_trades_from_dir(result_dir: str) -> pd.DataFrame:
    """Load all per-trade data from a FIFO validation result directory."""
    result_dir = Path(result_dir)
    all_trades = []

    for json_file in sorted(result_dir.glob("*_result.json")):
        date_str = json_file.stem.replace("_result", "")
        try:
            with open(json_file) as f:
                data = json.load(f)
            trades = data.get("trades", [])
            if not trades:
                continue
            for t in trades:
                t["date"] = date_str
                t["config"] = data.get("config", {})
            all_trades.extend(trades)
        except Exception as e:
            LOG.warning(f"Failed to load {json_file}: {e}")

    if not all_trades:
        return pd.DataFrame()

    df = pd.DataFrame(all_trades)
    LOG.info(f"Loaded {len(df)} trades from {result_dir.name}")
    return df


def load_prediction_features(result_dir: str) -> dict:
    """Load gated prediction .npz files for additional signal features."""
    result_dir = Path(result_dir)
    pred_data = {}

    for npz_file in sorted(result_dir.glob("*_gated.npz")):
        date_str = npz_file.stem.replace("_gated", "")
        try:
            data = np.load(npz_file)
            pred_data[date_str] = {
                "timestamps": data.get("timestamps", np.array([])),
                "predictions": data.get("predictions", np.array([])),
                "confidences": data.get("confidences", np.array([])),
            }
        except Exception as e:
            LOG.warning(f"Failed to load {npz_file}: {e}")

    return pred_data


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features from raw trade data."""
    if df.empty:
        return df

    feat = pd.DataFrame()

    # Signal features
    feat["signal_strength"] = df["signal_strength"].astype(float)
    feat["abs_signal"] = feat["signal_strength"].abs()

    # Queue/book features
    feat["queue_position"] = df["queue_position_at_post"].astype(float)
    feat["book_size"] = df["book_size_at_post"].astype(float)
    feat["queue_pct"] = (feat["queue_position"] / feat["book_size"].clip(lower=1)).clip(0, 1)

    # Fill latency
    feat["fill_latency_ms"] = df["fill_latency_ns"].astype(float) / 1e6
    feat["log_fill_latency"] = np.log1p(feat["fill_latency_ms"])

    # Hold duration
    feat["hold_duration_ms"] = df["hold_duration_ns"].astype(float) / 1e6
    feat["log_hold_duration"] = np.log1p(feat["hold_duration_ms"])

    # MFE/MAE
    feat["mfe_ticks"] = df["mfe_ticks"].astype(float)
    feat["mae_ticks"] = df["mae_ticks"].astype(float)
    feat["mfe_mae_ratio"] = (feat["mfe_ticks"] / feat["mae_ticks"].clip(lower=0.25))

    # Time of day (extract from signal_time_ns)
    # ES RTH: 9:30 AM - 4:00 PM ET = 34200s - 57600s from midnight
    signal_ns = df["signal_time_ns"].astype(float)
    # Convert nanoseconds to seconds-since-midnight (approximate)
    # signal_time_ns is Unix epoch nanoseconds
    signal_secs = (signal_ns / 1e9).astype(int)
    # Get seconds since midnight ET (UTC-4 during EDT, UTC-5 during EST)
    # For simplicity, use modulo 86400 and subtract 4 hours (EDT)
    tod_secs = (signal_secs - 4 * 3600) % 86400
    feat["time_of_day_secs"] = tod_secs
    feat["time_of_day_frac"] = tod_secs / 86400.0

    # Time periods
    feat["is_open_30min"] = ((tod_secs >= 34200) & (tod_secs < 36000)).astype(int)  # 9:30-10:00
    feat["is_close_30min"] = ((tod_secs >= 55800) & (tod_secs < 57600)).astype(int)  # 3:30-4:00
    feat["is_lunch"] = ((tod_secs >= 43200) & (tod_secs < 48600)).astype(int)  # 12:00-1:30
    feat["is_prime"] = ((tod_secs >= 36000) & (tod_secs < 43200)).astype(int)  # 10:00-12:00

    # Side
    feat["is_buy"] = (df["side"] == "BUY").astype(int)

    # Entry price level (relative, normalized per day)
    feat["entry_price"] = df["entry_price"].astype(float)

    # Exit reason encoding
    exit_reasons = df["exit_reason"].astype(str)
    feat["exit_tp"] = (exit_reasons == "TakeProfit").astype(int)
    feat["exit_sl"] = (exit_reasons == "StopLoss").astype(int)
    feat["exit_timeout"] = (exit_reasons.str.contains("Timeout|Hold|Conviction")).astype(int)

    # Target variables
    feat["pnl_ticks"] = df["pnl_ticks"].astype(float)
    feat["is_winner"] = (feat["pnl_ticks"] > 0).astype(int)
    feat["date"] = df["date"].values

    return feat


def walk_forward_train_eval(feat_df: pd.DataFrame, output_dir: Path) -> dict:
    """Walk-forward training and evaluation."""
    if feat_df.empty:
        LOG.error("No features to train on")
        return {}

    dates = sorted(feat_df["date"].unique())
    n_dates = len(dates)

    if n_dates < 10:
        LOG.error(f"Only {n_dates} dates — need at least 10 for walk-forward")
        return {}

    # Split: first 70% train, last 30% test
    split_idx = int(n_dates * 0.7)
    train_dates = dates[:split_idx]
    test_dates = dates[split_idx:]

    LOG.info(f"Walk-forward split: {len(train_dates)} train dates, {len(test_dates)} test dates")
    LOG.info(f"Train: {train_dates[0]} to {train_dates[-1]}")
    LOG.info(f"Test:  {test_dates[0]} to {test_dates[-1]}")

    # Feature columns (exclude targets and metadata)
    exclude_cols = {"pnl_ticks", "is_winner", "date", "entry_price",
                    "exit_tp", "exit_sl", "exit_timeout",
                    "mfe_ticks", "mae_ticks", "mfe_mae_ratio",
                    "hold_duration_ms", "log_hold_duration", "fill_latency_ms", "log_fill_latency"}

    # IMPORTANT: Only use features available BEFORE the trade executes
    # Post-trade features (MFE, MAE, hold duration, fill latency, exit reason) are LEAKY
    feature_cols = [c for c in feat_df.columns if c not in exclude_cols]

    LOG.info(f"Feature columns ({len(feature_cols)}): {feature_cols}")

    train_df = feat_df[feat_df["date"].isin(train_dates)]
    test_df = feat_df[feat_df["date"].isin(test_dates)]

    X_train = train_df[feature_cols].values
    y_train_cls = train_df["is_winner"].values
    y_train_reg = train_df["pnl_ticks"].values

    X_test = test_df[feature_cols].values
    y_test_cls = test_df["is_winner"].values
    y_test_reg = test_df["pnl_ticks"].values

    LOG.info(f"Train: {len(X_train)} trades, WR={y_train_cls.mean():.3f}, avg PnL={y_train_reg.mean():.3f}tk")
    LOG.info(f"Test:  {len(X_test)} trades, WR={y_test_cls.mean():.3f}, avg PnL={y_test_reg.mean():.3f}tk")

    results = {}

    # === LGBM Classifier ===
    if HAS_LGBM:
        LOG.info("\n=== Training LGBM Classifier ===")
        lgb_params = {
            "objective": "binary",
            "metric": "auc",
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 50,
            "verbose": -1,
            "random_state": 42,
        }

        lgb_model = lgb.LGBMClassifier(**lgb_params)
        lgb_model.fit(X_train, y_train_cls)

        lgb_probs = lgb_model.predict_proba(X_test)[:, 1]
        lgb_results = evaluate_predictions(lgb_probs, y_test_cls, y_test_reg, test_df, "LGBM")
        results["lgbm"] = lgb_results

        # Feature importance
        importances = lgb_model.feature_importances_
        feat_imp = sorted(zip(feature_cols, importances), key=lambda x: -x[1])
        LOG.info("Feature importances:")
        for name, imp in feat_imp[:10]:
            LOG.info(f"  {name}: {imp}")

        results["lgbm"]["feature_importances"] = {n: int(v) for n, v in feat_imp}

    # === XGBoost Classifier ===
    if HAS_XGB:
        LOG.info("\n=== Training XGBoost Classifier ===")
        xgb_params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 50,
            "verbosity": 0,
            "random_state": 42,
        }

        xgb_model = xgb.XGBClassifier(**xgb_params)
        xgb_model.fit(X_train, y_train_cls)

        xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
        xgb_results = evaluate_predictions(xgb_probs, y_test_cls, y_test_reg, test_df, "XGBoost")
        results["xgboost"] = xgb_results

    # === LGBM Regressor (predict PnL ticks directly) ===
    if HAS_LGBM:
        LOG.info("\n=== Training LGBM Regressor (PnL) ===")
        lgb_reg_params = {
            "objective": "regression",
            "metric": "rmse",
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 50,
            "verbose": -1,
            "random_state": 42,
        }

        lgb_reg = lgb.LGBMRegressor(**lgb_reg_params)
        lgb_reg.fit(X_train, y_train_reg)

        lgb_reg_preds = lgb_reg.predict(X_test)
        lgb_reg_results = evaluate_regression(lgb_reg_preds, y_test_cls, y_test_reg, test_df, "LGBM_Reg")
        results["lgbm_reg"] = lgb_reg_results

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


def evaluate_predictions(probs: np.ndarray, y_cls: np.ndarray, y_reg: np.ndarray,
                         test_df: pd.DataFrame, model_name: str) -> dict:
    """Evaluate model predictions with percentile analysis."""
    from sklearn.metrics import roc_auc_score

    n = len(probs)
    auc = roc_auc_score(y_cls, probs)
    LOG.info(f"\n{model_name} AUC: {auc:.4f}")

    results = {"auc": auc, "n_test": n}

    # Percentile analysis
    for pct_label, pct in [("top1%", 99), ("top5%", 95), ("top10%", 90), ("top20%", 80), ("top50%", 50)]:
        threshold = np.percentile(probs, pct)
        mask = probs >= threshold
        n_selected = mask.sum()
        if n_selected < 10:
            continue

        wr = y_cls[mask].mean()
        avg_pnl = y_reg[mask].mean()
        total_pnl = y_reg[mask].sum()
        pf = compute_profit_factor(y_reg[mask])

        # Per-day analysis
        dates_selected = test_df["date"].values[mask]
        unique_dates = np.unique(dates_selected)
        green_days = 0
        for d in unique_dates:
            day_mask = dates_selected == d
            if y_reg[mask][day_mask].sum() > 0:
                green_days += 1

        LOG.info(f"  {pct_label} (n={n_selected}, thresh={threshold:.3f}): "
                 f"WR={wr:.3f}, AvgPnL={avg_pnl:+.3f}tk, TotalPnL={total_pnl:+.1f}tk, "
                 f"PF={pf:.2f}, Green={green_days}/{len(unique_dates)}")

        results[pct_label] = {
            "n": int(n_selected),
            "threshold": float(threshold),
            "wr": float(wr),
            "avg_pnl": float(avg_pnl),
            "total_pnl": float(total_pnl),
            "profit_factor": float(pf),
            "green_days": green_days,
            "total_days": len(unique_dates),
        }

    # Bottom percentile (worst trades — does model correctly identify losers?)
    bottom_mask = probs <= np.percentile(probs, 10)
    if bottom_mask.sum() >= 10:
        bot_wr = y_cls[bottom_mask].mean()
        bot_pnl = y_reg[bottom_mask].mean()
        LOG.info(f"  bottom10% (n={bottom_mask.sum()}): WR={bot_wr:.3f}, AvgPnL={bot_pnl:+.3f}tk")
        results["bottom10%"] = {"wr": float(bot_wr), "avg_pnl": float(bot_pnl)}

    return results


def evaluate_regression(preds: np.ndarray, y_cls: np.ndarray, y_reg: np.ndarray,
                        test_df: pd.DataFrame, model_name: str) -> dict:
    """Evaluate regression predictions — select trades where predicted PnL > 0."""
    from scipy.stats import spearmanr

    corr, pval = spearmanr(preds, y_reg)
    LOG.info(f"\n{model_name} Spearman(pred, actual): {corr:.4f} (p={pval:.4e})")

    results = {"spearman": float(corr), "spearman_pval": float(pval)}

    # Trade only when predicted PnL > threshold
    for thresh_label, thresh in [("pred>0", 0), ("pred>0.5", 0.5), ("pred>1.0", 1.0)]:
        mask = preds > thresh
        n_selected = mask.sum()
        if n_selected < 10:
            continue

        wr = y_cls[mask].mean()
        avg_pnl = y_reg[mask].mean()
        total_pnl = y_reg[mask].sum()
        pf = compute_profit_factor(y_reg[mask])

        LOG.info(f"  {thresh_label} (n={n_selected}): "
                 f"WR={wr:.3f}, AvgPnL={avg_pnl:+.3f}tk, PF={pf:.2f}")

        results[thresh_label] = {
            "n": int(n_selected),
            "wr": float(wr),
            "avg_pnl": float(avg_pnl),
            "total_pnl": float(total_pnl),
            "profit_factor": float(pf),
        }

    return results


def compute_profit_factor(pnl_array: np.ndarray) -> float:
    """Compute profit factor = gross profits / gross losses."""
    gross_profit = pnl_array[pnl_array > 0].sum()
    gross_loss = abs(pnl_array[pnl_array < 0].sum())
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def main():
    parser = argparse.ArgumentParser(description="Train execution filter on FIFO trade outcomes")
    parser.add_argument("--input-dirs", nargs="+", required=True,
                        help="Directories containing FIFO validation results")
    parser.add_argument("--output-dir", default="output/fifo_outcome_predictor_v1",
                        help="Output directory")
    parser.add_argument("--min-trades", type=int, default=500,
                        help="Minimum total trades required")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"{args.output_dir}.log", mode="w"),
        ],
    )

    LOG.info(f"Loading trades from {len(args.input_dirs)} directories")

    all_dfs = []
    for d in args.input_dirs:
        df = load_trades_from_dir(d)
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        LOG.error("No trade data found!")
        return

    raw_df = pd.concat(all_dfs, ignore_index=True)
    LOG.info(f"Total raw trades: {len(raw_df)}")
    LOG.info(f"Dates: {sorted(raw_df['date'].unique())}")
    LOG.info(f"Overall WR: {(raw_df['pnl_ticks'].astype(float) > 0).mean():.3f}")
    LOG.info(f"Overall avg PnL: {raw_df['pnl_ticks'].astype(float).mean():.3f} ticks")

    if len(raw_df) < args.min_trades:
        LOG.error(f"Only {len(raw_df)} trades, need {args.min_trades}")
        return

    # Engineer features
    feat_df = engineer_features(raw_df)
    LOG.info(f"Engineered {len(feat_df)} feature rows with {len(feat_df.columns)} columns")

    # Train and evaluate
    output_dir = Path(args.output_dir)
    results = walk_forward_train_eval(feat_df, output_dir)

    if results:
        LOG.info("\n" + "=" * 60)
        LOG.info("FINAL SUMMARY")
        LOG.info("=" * 60)
        for model_name, model_results in results.items():
            LOG.info(f"\n{model_name}:")
            if "auc" in model_results:
                LOG.info(f"  AUC: {model_results['auc']:.4f}")
                for pct in ["top1%", "top5%", "top10%"]:
                    if pct in model_results:
                        r = model_results[pct]
                        LOG.info(f"  {pct}: WR={r['wr']:.3f}, PnL={r['avg_pnl']:+.3f}tk, PF={r['profit_factor']:.2f}")
            if "spearman" in model_results:
                LOG.info(f"  Spearman: {model_results['spearman']:.4f}")

    LOG.info("\nDone!")


if __name__ == "__main__":
    main()
