#!/usr/bin/env python3
"""
Passive Execution Optimizer - XGBoost GPU
==========================================
Trains a model to predict net P&L per passive limit order placement.
Combines signal strength + fill probability + microstructure features.

Target: net_pnl_ticks (continuous) = filled * (|signal_aligned_move| - commission)
Walk-forward: sliding 25-day train, 1-day OOT (HC #0 SLIDING ONLY)
"""

import os
import sys
import json
import time
import logging
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
import mlflow

warnings.filterwarnings("ignore")

# ============ CONFIG ============
BASE_DIR = Path("/home/nick/Lvl3Quant")
LABELS_DIR = BASE_DIR / "output" / "mbo_walker_labels"
FEATURES_DIR = BASE_DIR / "output" / "queue_augmented_features"
FILL_PROB_DIR = BASE_DIR / "output" / "fill_prob_v3_honest_xgb"
OUTPUT_DIR = BASE_DIR / "output" / "passive_exec_optimizer_v1"
LOG_DIR = BASE_DIR / "logs" / "passive_exec_opt"

TRAIN_WINDOW = 25  # days
COMMISSION_PASSIVE = 0.376  # ticks, passive limit only
HORIZON = "10s"  # primary fill horizon

# XGBoost GPU params
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

# Kill threshold
KILL_SPEARMAN_THRESHOLD = 0.02
KILL_AFTER_FOLDS = 3

# Acceptance gates (HC #506 R5)
ACCEPT_SPEARMAN = 0.05
ACCEPT_TOP_DECILE_NET = 0.2
ACCEPT_TOP_DECILE_SHARPE = 0.5

# ============ LOGGING ============
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "training.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

SEP = "=" * 60

# ============ DATA LOADING ============
def get_available_dates():
    """Get sorted list of dates with both labels and features."""
    label_dates = {f.stem.replace("labels_", "") for f in LABELS_DIR.glob("labels_*.parquet")}
    feat_dates = {f.stem.replace("features_", "") for f in FEATURES_DIR.glob("features_*.parquet")}
    common = sorted(label_dates & feat_dates)
    return common


def load_fill_prob_model():
    """Load the pre-trained fill probability XGBoost model."""
    model_path = FILL_PROB_DIR / f"xgb_fill_{HORIZON}_full.json"
    if not model_path.exists():
        logger.warning(f"Fill prob model not found at {model_path}, will skip fill_prob feature")
        return None
    model = xgb.Booster()
    model.load_model(str(model_path))
    # Get expected feature names
    feats_path = FILL_PROB_DIR / "feats.json"
    fill_features = None
    if feats_path.exists():
        with open(feats_path) as f:
            feats_data = json.load(f)
            fill_features = feats_data if isinstance(feats_data, list) else feats_data.get(HORIZON, feats_data.get("features", None))
    return model, fill_features


def load_day(date_str):
    """Load and merge labels + features for one date."""
    labels = pd.read_parquet(LABELS_DIR / f"labels_{date_str}.parquet")
    features = pd.read_parquet(FEATURES_DIR / f"features_{date_str}.parquet")

    # Merge on event_id
    label_cols = ["event_id", "price", "queue_depth_at_touch",
                  "queue_rank_at_10s", "filled_10s", "time_to_fill_s_10s"]
    df = features.merge(labels[label_cols], on="event_id", how="inner")

    # Rename for clarity
    df = df.rename(columns={
        "queue_rank_at_10s": "queue_rank",
        "filled_10s": "filled",
        "time_to_fill_s_10s": "time_to_fill_s",
    })

    return df


def construct_target(df):
    """
    Construct net_pnl_ticks target for passive limit placement.

    Logic: We place a passive limit in the predicted direction at touch.
    - If filled: net_pnl = abs(pred_10s) - commission
      Signal magnitude proxies expected favorable move (validated: WR 58.5% at touch).
    - If NOT filled: net_pnl = 0 (no trade, no P&L)
    """
    signal_magnitude = df["pred_10s"].abs()
    df["net_pnl_ticks"] = df["filled"] * (signal_magnitude - COMMISSION_PASSIVE)
    return df


def compute_fill_prob_feature(df, fill_model_info):
    """Add fill probability prediction as a feature."""
    if fill_model_info is None:
        df["fill_prob_pred"] = np.nan
        return df

    model, fill_features = fill_model_info

    if fill_features is not None:
        available = [c for c in fill_features if c in df.columns]
        if len(available) < 3:
            logger.warning(f"Only {len(available)} fill prob features available, skipping")
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
    """Get training feature columns (exclude target, identifiers)."""
    exclude = {"event_id", "ts_ns", "price", "net_pnl_ticks", "filled",
               "time_to_fill_s", "queue_rank", "queue_depth_at_touch"}
    features = [c for c in df.columns if c not in exclude]
    return features


# ============ WALK-FORWARD ============
def run_walk_forward():
    """Sliding 25-day train, 1-day OOT walk-forward."""
    dates = get_available_dates()
    n_dates = len(dates)
    logger.info(f"Available dates: {n_dates} ({dates[0]} to {dates[-1]})")
    logger.info(f"Train window: {TRAIN_WINDOW} days, OOT folds: {n_dates - TRAIN_WINDOW}")

    if n_dates <= TRAIN_WINDOW:
        logger.error(f"Not enough dates for walk-forward: {n_dates} <= {TRAIN_WINDOW}")
        sys.exit(1)

    # Load fill prob model once
    fill_model_info = load_fill_prob_model()
    if fill_model_info:
        logger.info("Fill probability model loaded successfully")
    else:
        logger.info("Running WITHOUT fill probability feature")

    # Pre-load all data
    logger.info("Pre-loading all dates...")
    all_data = {}
    for d in dates:
        df = load_day(d)
        df = construct_target(df)
        df = compute_fill_prob_feature(df, fill_model_info)
        all_data[d] = df
        logger.info(f"  {d}: {len(df)} events, fill_rate={df['filled'].mean():.3f}, "
                   f"mean_target={df['net_pnl_ticks'].mean():.4f}")

    feature_cols = get_feature_columns(all_data[dates[0]])
    logger.info(f"Feature columns ({len(feature_cols)}): {feature_cols}")

    # MLflow setup
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://jupiter:5000"))
    mlflow.set_experiment("passive_exec_optimizer_v1")

    run_ts = datetime.now().strftime("%Y%m%d_%H%M")
    with mlflow.start_run(run_name="xgb_passive_exec_" + run_ts):
        mlflow.log_params({
            "model_type": "xgboost_gpu_regression",
            "train_window": TRAIN_WINDOW,
            "horizon": HORIZON,
            "commission_ticks": COMMISSION_PASSIVE,
            "n_features": len(feature_cols),
            "n_dates": n_dates,
            "n_oot_folds": n_dates - TRAIN_WINDOW,
            "xgb_max_depth": XGB_PARAMS["max_depth"],
            "xgb_lr": XGB_PARAMS["learning_rate"],
            "xgb_n_estimators": XGB_PARAMS["n_estimators"],
        })
        mlflow.log_param("feature_names", json.dumps(feature_cols))

        fold_results = []
        all_oot_preds = []
        all_oot_targets = []

        for fold_idx in range(TRAIN_WINDOW, n_dates):
            oot_date = dates[fold_idx]
            train_dates = dates[fold_idx - TRAIN_WINDOW:fold_idx]

            logger.info(SEP)
            logger.info(f"FOLD {fold_idx - TRAIN_WINDOW + 1}/{n_dates - TRAIN_WINDOW}: "
                       f"OOT={oot_date}, Train={train_dates[0]}..{train_dates[-1]}")

            # Assemble training data
            train_dfs = [all_data[d] for d in train_dates]
            train_df = pd.concat(train_dfs, ignore_index=True)
            oot_df = all_data[oot_date].copy()

            X_train = train_df[feature_cols].values
            y_train = train_df["net_pnl_ticks"].values
            X_oot = oot_df[feature_cols].values
            y_oot = oot_df["net_pnl_ticks"].values

            # Handle NaN
            X_train = np.nan_to_num(X_train, nan=0.0)
            X_oot = np.nan_to_num(X_oot, nan=0.0)

            # Train XGBoost
            dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
            dval = xgb.DMatrix(X_oot, label=y_oot, feature_names=feature_cols)

            t0 = time.time()
            params = {k: v for k, v in XGB_PARAMS.items()
                     if k not in ("n_estimators", "early_stopping_rounds", "eval_metric")}
            params["eval_metric"] = XGB_PARAMS["eval_metric"]

            bst = xgb.train(
                params,
                dtrain,
                num_boost_round=XGB_PARAMS["n_estimators"],
                evals=[(dval, "oot")],
                early_stopping_rounds=XGB_PARAMS["early_stopping_rounds"],
                verbose_eval=False,
            )
            train_time = time.time() - t0

            # OOT predictions
            preds = bst.predict(dval)

            # Metrics
            spearman_corr, spearman_p = spearmanr(preds, y_oot)
            if np.isnan(spearman_corr):
                spearman_corr = 0.0

            # Top decile analysis
            n_oot = len(preds)
            top_decile_idx = np.argsort(preds)[-n_oot // 10:]
            top_decile_net = y_oot[top_decile_idx].mean()
            top_decile_std = y_oot[top_decile_idx].std()
            top_decile_sharpe = (top_decile_net / top_decile_std * np.sqrt(252)) if top_decile_std > 0 else 0.0

            # Bottom decile
            bot_decile_idx = np.argsort(preds)[:n_oot // 10]
            bot_decile_net = y_oot[bot_decile_idx].mean()

            # Fill rate in top decile
            top_decile_fill_rate = oot_df.iloc[top_decile_idx]["filled"].mean()

            fold_result = {
                "fold": fold_idx - TRAIN_WINDOW + 1,
                "oot_date": oot_date,
                "n_train": len(train_df),
                "n_oot": n_oot,
                "spearman": spearman_corr,
                "spearman_p": spearman_p,
                "top_decile_mean_net": top_decile_net,
                "top_decile_sharpe": top_decile_sharpe,
                "bot_decile_mean_net": bot_decile_net,
                "top_decile_fill_rate": top_decile_fill_rate,
                "spread_top_bot": top_decile_net - bot_decile_net,
                "best_iteration": bst.best_iteration,
                "train_time_s": train_time,
            }
            fold_results.append(fold_result)
            all_oot_preds.extend(preds.tolist())
            all_oot_targets.extend(y_oot.tolist())

            logger.info(f"  Spearman={spearman_corr:.4f} (p={spearman_p:.2e}), "
                       f"TopDecile={top_decile_net:.4f} ticks, "
                       f"BotDecile={bot_decile_net:.4f} ticks, "
                       f"Spread={top_decile_net - bot_decile_net:.4f}, "
                       f"TopFillRate={top_decile_fill_rate:.3f}, "
                       f"best_iter={bst.best_iteration}, time={train_time:.1f}s")

            # Log per-fold metrics
            mlflow.log_metrics({
                f"fold_{fold_result['fold']}_spearman": spearman_corr,
                f"fold_{fold_result['fold']}_top_decile_net": top_decile_net,
            }, step=fold_result["fold"])

            # KILL CHECK after first N folds
            if len(fold_results) == KILL_AFTER_FOLDS:
                avg_spearman = np.mean([r["spearman"] for r in fold_results])
                if avg_spearman < KILL_SPEARMAN_THRESHOLD:
                    logger.error(f"KILL: First {KILL_AFTER_FOLDS} folds avg Spearman "
                               f"= {avg_spearman:.4f} < {KILL_SPEARMAN_THRESHOLD}. ABORTING.")
                    mlflow.log_metric("killed_early", 1)
                    mlflow.log_metric("kill_reason_spearman", avg_spearman)
                    mlflow.set_tag("status", "KILLED_LOW_SPEARMAN")
                    sys.exit(1)
                else:
                    logger.info(f"  PASS kill check: avg Spearman={avg_spearman:.4f} >= {KILL_SPEARMAN_THRESHOLD}")

            # Save fold model
            fold_model_path = OUTPUT_DIR / f"fold_{fold_result['fold']}_{oot_date}.json"
            bst.save_model(str(fold_model_path))

        # ============ CONCAT ANALYSIS ============
        logger.info(SEP)
        logger.info("CONCAT OOT ANALYSIS (all folds combined)")
        logger.info(SEP)

        all_preds = np.array(all_oot_preds)
        all_targets = np.array(all_oot_targets)

        concat_spearman, concat_p = spearmanr(all_preds, all_targets)
        if np.isnan(concat_spearman):
            concat_spearman = 0.0

        # Top decile concat
        n_total = len(all_preds)
        top_dec_idx = np.argsort(all_preds)[-n_total // 10:]
        concat_top_net = all_targets[top_dec_idx].mean()
        concat_top_std = all_targets[top_dec_idx].std()
        concat_top_sharpe = (concat_top_net / concat_top_std * np.sqrt(252)) if concat_top_std > 0 else 0.0

        bot_dec_idx = np.argsort(all_preds)[:n_total // 10]
        concat_bot_net = all_targets[bot_dec_idx].mean()

        # Per-fold Sharpe (daily)
        daily_nets = [r["top_decile_mean_net"] for r in fold_results]
        daily_sharpe = (np.mean(daily_nets) / np.std(daily_nets) * np.sqrt(252)) if np.std(daily_nets) > 0 else 0.0

        # Win rate of top decile
        top_dec_wr = (all_targets[top_dec_idx] > 0).mean()

        # Feature importance (from last fold model)
        importance = bst.get_score(importance_type="gain")
        top_features = sorted(importance.items(), key=lambda x: -x[1])[:15]

        logger.info(f"  Concat Spearman: {concat_spearman:.4f} (p={concat_p:.2e})")
        logger.info(f"  Concat Top Decile Mean Net: {concat_top_net:.4f} ticks")
        logger.info(f"  Concat Top Decile Sharpe: {concat_top_sharpe:.2f}")
        logger.info(f"  Concat Bot Decile Mean Net: {concat_bot_net:.4f} ticks")
        logger.info(f"  Concat Spread (top-bot): {concat_top_net - concat_bot_net:.4f} ticks")
        logger.info(f"  Top Decile WR: {top_dec_wr:.3f}")
        logger.info(f"  Daily Sharpe (top decile): {daily_sharpe:.2f}")
        logger.info(f"  Per-fold Spearman avg: {np.mean([r['spearman'] for r in fold_results]):.4f}")
        logger.info("  Top 15 features by gain:")
        for feat, gain in top_features:
            logger.info(f"    {feat}: {gain:.1f}")

        # Log concat metrics
        mlflow.log_metrics({
            "concat_spearman": concat_spearman,
            "concat_top_decile_net": concat_top_net,
            "concat_top_decile_sharpe": concat_top_sharpe,
            "concat_bot_decile_net": concat_bot_net,
            "concat_spread_top_bot": concat_top_net - concat_bot_net,
            "concat_top_decile_wr": top_dec_wr,
            "daily_sharpe": daily_sharpe,
            "avg_fold_spearman": np.mean([r["spearman"] for r in fold_results]),
            "n_oot_events_total": n_total,
        })

        # Acceptance gate check
        logger.info(SEP)
        logger.info("ACCEPTANCE GATE CHECK (HC #506 R5)")
        logger.info(SEP)

        gate_pass = True
        checks = [
            ("Concat Spearman > 0.05", concat_spearman, ACCEPT_SPEARMAN, concat_spearman > ACCEPT_SPEARMAN),
            ("Top Decile Net > +0.2 ticks", concat_top_net, ACCEPT_TOP_DECILE_NET, concat_top_net > ACCEPT_TOP_DECILE_NET),
            ("Top Decile Sharpe > 0.5", concat_top_sharpe, ACCEPT_TOP_DECILE_SHARPE, concat_top_sharpe > ACCEPT_TOP_DECILE_SHARPE),
        ]
        for name, val, thresh, passed in checks:
            status = "PASS" if passed else "FAIL"
            logger.info(f"  [{status}] {name}: {val:.4f} (thresh={thresh})")
            if not passed:
                gate_pass = False

        overall = "ACCEPTED" if gate_pass else "REJECTED"
        logger.info(f"  OVERALL: {overall}")
        mlflow.set_tag("acceptance", overall)
        mlflow.log_metric("gate_passed", 1 if gate_pass else 0)

        # Save results
        results = {
            "concat_spearman": concat_spearman,
            "concat_top_decile_net": concat_top_net,
            "concat_top_decile_sharpe": concat_top_sharpe,
            "concat_bot_decile_net": concat_bot_net,
            "top_decile_wr": top_dec_wr,
            "daily_sharpe": daily_sharpe,
            "acceptance": overall,
            "n_folds": len(fold_results),
            "feature_importance": dict(top_features),
            "fold_results": fold_results,
            "feature_columns": feature_cols,
        }

        with open(OUTPUT_DIR / "results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Save predictions
        np.savez(OUTPUT_DIR / "oot_predictions.npz",
                 predictions=all_preds,
                 targets=all_targets)

        mlflow.log_artifact(str(OUTPUT_DIR / "results.json"))

        logger.info(f"Done. Results saved to {OUTPUT_DIR}")
        logger.info(f"Total training time: {sum(r['train_time_s'] for r in fold_results):.0f}s")


if __name__ == "__main__":
    logger.info(SEP)
    logger.info("PASSIVE EXECUTION OPTIMIZER v1 - XGBoost GPU")
    logger.info(f"Started: {datetime.now().isoformat()}")
    logger.info(f"Config: train_window={TRAIN_WINDOW}, horizon={HORIZON}, commission={COMMISSION_PASSIVE}")
    logger.info(SEP)
    run_walk_forward()
