#!/usr/bin/env python3
"""
Passive Execution Optimizer v1 — MLP (PyTorch GPU)
====================================================
MLP version of the proven XGBoost passive exec optimizer.
Predicts net P&L per passive limit order placement.

Architecture: 30 → 128 → 64 → 1 (BatchNorm + ReLU + Dropout(0.3))
Walk-forward: sliding 25-day train, 1-day OOT (HC #0 SLIDING ONLY)
Target: net_pnl_ticks = filled * (|pred_10s| - 0.376)

Built-in permutation test: 100 shuffles to validate statistical significance.
"""

import os
import sys
import json
import time
import copy
import logging
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import xgboost as xgb
from scipy.stats import spearmanr
import mlflow

warnings.filterwarnings("ignore")

# ============ CONFIG ============
BASE_DIR = Path("/home/nick/Lvl3Quant")
LABELS_DIR = BASE_DIR / "output" / "mbo_walker_labels"
FEATURES_DIR = BASE_DIR / "output" / "queue_augmented_features"
FILL_PROB_DIR = BASE_DIR / "output" / "fill_prob_v3_honest_xgb"
OUTPUT_DIR = BASE_DIR / "output" / "passive_exec_optimizer_mlp_v1"
LOG_DIR = BASE_DIR / "logs" / "passive_exec_opt_mlp"

TRAIN_WINDOW = 25   # days (SLIDING, not expanding — HC #0)
COMMISSION_PASSIVE = 0.376  # ticks, passive limit only
HORIZON = "10s"

# MLP hyperparams
HIDDEN_DIMS = [128, 64]
DROPOUT = 0.3
LR = 1e-3
EPOCHS = 50
BATCH_SIZE = 4096
VAL_FRACTION = 0.20  # last 20% of training data for early stopping
PATIENCE = 8         # early stopping patience (epochs)

# Kill threshold
KILL_SPEARMAN_THRESHOLD = 0.02
KILL_AFTER_FOLDS = 3

# Acceptance gates
ACCEPT_SPEARMAN = 0.05
ACCEPT_TOP_DECILE_NET = 0.2
ACCEPT_TOP_DECILE_SHARPE = 0.5

# Permutation test
N_PERMUTATIONS = 100

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


# ============ MODEL ============
class PassiveExecMLP(nn.Module):
    """Simple 3-layer MLP: input → 128 → 64 → 1, with BN + ReLU + Dropout."""

    def __init__(self, n_features, hidden_dims=None, dropout=0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        layers = []
        in_dim = n_features
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ============ DATA LOADING (same as v1 XGBoost) ============
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
    """net_pnl_ticks = filled * (|pred_10s| - commission)"""
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


# ============ TRAINING UTILITIES ============
def standardize(X_train, X_oot):
    """Per-fold standardization. Fit on train, transform both."""
    mean = np.nanmean(X_train, axis=0)
    std = np.nanstd(X_train, axis=0)
    std[std < 1e-8] = 1.0  # prevent div by zero
    X_train_s = (X_train - mean) / std
    X_oot_s = (X_oot - mean) / std
    return X_train_s, X_oot_s, mean, std


def train_one_fold(X_train_np, y_train_np, X_oot_np, n_features, fold_idx,
                   shuffle_target=False, quiet=False):
    """
    Train MLP for one fold. Returns (oot_preds, best_val_loss, best_epoch, train_time, model_state).
    If shuffle_target=True, shuffles y_train for permutation test.
    """
    # Standardize
    X_train_s, X_oot_s, mean, std = standardize(X_train_np, X_oot_np)

    # NaN → 0 after standardization
    X_train_s = np.nan_to_num(X_train_s, nan=0.0)
    X_oot_s = np.nan_to_num(X_oot_s, nan=0.0)

    y_train = y_train_np.copy()
    if shuffle_target:
        np.random.shuffle(y_train)

    # Train/val split (last 20% of training data = validation)
    n_train = len(X_train_s)
    n_val = int(n_train * VAL_FRACTION)
    n_fit = n_train - n_val

    X_fit = torch.tensor(X_train_s[:n_fit], dtype=torch.float32, device=DEVICE)
    y_fit = torch.tensor(y_train[:n_fit], dtype=torch.float32, device=DEVICE)
    X_val = torch.tensor(X_train_s[n_fit:], dtype=torch.float32, device=DEVICE)
    y_val = torch.tensor(y_train[n_fit:], dtype=torch.float32, device=DEVICE)
    X_oot_t = torch.tensor(X_oot_s, dtype=torch.float32, device=DEVICE)

    fit_ds = TensorDataset(X_fit, y_fit)
    fit_loader = DataLoader(fit_ds, batch_size=BATCH_SIZE, shuffle=True,
                            drop_last=False, pin_memory=False)

    # Model
    model = PassiveExecMLP(n_features, HIDDEN_DIMS, DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 0.01)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0

    t0 = time.time()
    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss_sum = 0.0
        n_batches = 0
        for xb, yb in fit_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            n_batches += 1
        scheduler.step()

        # Val
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            break

    train_time = time.time() - t0

    # Restore best model and predict OOT
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        oot_preds = model(X_oot_t).cpu().numpy()

    return oot_preds, best_val_loss, best_epoch, train_time, best_state, mean, std


# ============ PERMUTATION TEST ============
def run_permutation_test(X_train_np, y_train_np, X_oot_np, y_oot_np,
                         n_features, real_spearman, fold_idx):
    """
    Run N_PERMUTATIONS with shuffled labels.
    Returns (p_value, null_spearmans).
    """
    null_spearmans = []
    for perm_i in range(N_PERMUTATIONS):
        preds, _, _, _, _, _, _ = train_one_fold(
            X_train_np, y_train_np, X_oot_np, n_features,
            fold_idx, shuffle_target=True, quiet=True
        )
        sp, _ = spearmanr(preds, y_oot_np)
        if np.isnan(sp):
            sp = 0.0
        null_spearmans.append(sp)
        if (perm_i + 1) % 25 == 0:
            logger.info(f"    Permutation {perm_i + 1}/{N_PERMUTATIONS}, "
                       f"null Spearman mean={np.mean(null_spearmans):.4f}")

    null_spearmans = np.array(null_spearmans)
    p_value = (np.sum(null_spearmans >= real_spearman) + 1) / (N_PERMUTATIONS + 1)
    return p_value, null_spearmans


# ============ WALK-FORWARD ============
def run_walk_forward():
    """Sliding 25-day train, 1-day OOT walk-forward with MLP."""
    dates = get_available_dates()
    n_dates = len(dates)
    n_oot = n_dates - TRAIN_WINDOW
    logger.info(f"Available dates: {n_dates} ({dates[0]} to {dates[-1]})")
    logger.info(f"Train window: {TRAIN_WINDOW} days, OOT folds: {n_oot}")
    logger.info(f"Device: {DEVICE}")

    if n_dates <= TRAIN_WINDOW:
        logger.error(f"Not enough dates: {n_dates} <= {TRAIN_WINDOW}")
        sys.exit(1)

    # Load fill prob model once
    fill_model_info = load_fill_prob_model()
    if fill_model_info:
        logger.info("Fill probability model loaded")
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
    n_features = len(feature_cols)
    logger.info(f"Feature columns ({n_features}): {feature_cols}")

    # MLflow setup
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://jupiter:5000"))
    mlflow.set_experiment("passive_exec_optimizer_mlp_v1")

    run_ts = datetime.now().strftime("%Y%m%d_%H%M")
    with mlflow.start_run(run_name=f"mlp_passive_exec_{run_ts}"):
        mlflow.log_params({
            "model_type": "mlp_pytorch",
            "architecture": f"{n_features}→{'→'.join(str(h) for h in HIDDEN_DIMS)}→1",
            "dropout": DROPOUT,
            "lr": LR,
            "epochs_max": EPOCHS,
            "batch_size": BATCH_SIZE,
            "val_fraction": VAL_FRACTION,
            "patience": PATIENCE,
            "train_window": TRAIN_WINDOW,
            "horizon": HORIZON,
            "commission_ticks": COMMISSION_PASSIVE,
            "n_features": n_features,
            "n_dates": n_dates,
            "n_oot_folds": n_oot,
            "device": str(DEVICE),
            "n_permutations": N_PERMUTATIONS,
        })
        mlflow.log_param("feature_names", json.dumps(feature_cols))

        fold_results = []
        all_oot_preds = []
        all_oot_targets = []

        for fold_idx in range(TRAIN_WINDOW, n_dates):
            oot_date = dates[fold_idx]
            train_dates = dates[fold_idx - TRAIN_WINDOW:fold_idx]
            fold_num = fold_idx - TRAIN_WINDOW + 1

            logger.info(SEP)
            logger.info(f"FOLD {fold_num}/{n_oot}: "
                       f"OOT={oot_date}, Train={train_dates[0]}..{train_dates[-1]}")

            # Assemble training data
            train_dfs = [all_data[d] for d in train_dates]
            train_df = pd.concat(train_dfs, ignore_index=True)
            oot_df = all_data[oot_date].copy()

            X_train = train_df[feature_cols].values.astype(np.float32)
            y_train = train_df["net_pnl_ticks"].values.astype(np.float32)
            X_oot = oot_df[feature_cols].values.astype(np.float32)
            y_oot = oot_df["net_pnl_ticks"].values.astype(np.float32)

            # NaN → 0 before standardization
            X_train = np.nan_to_num(X_train, nan=0.0)
            X_oot = np.nan_to_num(X_oot, nan=0.0)

            # Train
            preds, best_val_loss, best_epoch, train_time, best_state, mean, std = \
                train_one_fold(X_train, y_train, X_oot, n_features, fold_idx)

            # Metrics
            spearman_corr, spearman_p = spearmanr(preds, y_oot)
            if np.isnan(spearman_corr):
                spearman_corr = 0.0

            # Top/bottom decile
            n_oot_events = len(preds)
            top_dec_idx = np.argsort(preds)[-n_oot_events // 10:]
            top_dec_net = y_oot[top_dec_idx].mean()
            top_dec_std = y_oot[top_dec_idx].std()
            top_dec_sharpe = (top_dec_net / top_dec_std * np.sqrt(252)) if top_dec_std > 0 else 0.0
            top_dec_fill_rate = oot_df.iloc[top_dec_idx]["filled"].mean()

            bot_dec_idx = np.argsort(preds)[:n_oot_events // 10]
            bot_dec_net = y_oot[bot_dec_idx].mean()

            fold_result = {
                "fold": fold_num,
                "oot_date": oot_date,
                "n_train": len(train_df),
                "n_oot": n_oot_events,
                "spearman": float(spearman_corr),
                "spearman_p": float(spearman_p) if not np.isnan(spearman_p) else 1.0,
                "top_decile_mean_net": float(top_dec_net),
                "top_decile_sharpe": float(top_dec_sharpe),
                "bot_decile_mean_net": float(bot_dec_net),
                "top_decile_fill_rate": float(top_dec_fill_rate),
                "spread_top_bot": float(top_dec_net - bot_dec_net),
                "best_val_loss": float(best_val_loss),
                "best_epoch": best_epoch,
                "train_time_s": train_time,
            }
            fold_results.append(fold_result)
            all_oot_preds.extend(preds.tolist())
            all_oot_targets.extend(y_oot.tolist())

            logger.info(f"  Spearman={spearman_corr:.4f} (p={spearman_p:.2e}), "
                       f"TopDec={top_dec_net:.4f}t, BotDec={bot_dec_net:.4f}t, "
                       f"Spread={top_dec_net - bot_dec_net:.4f}, "
                       f"FillRate={top_dec_fill_rate:.3f}, "
                       f"best_ep={best_epoch}, val_loss={best_val_loss:.6f}, "
                       f"time={train_time:.1f}s")

            mlflow.log_metrics({
                f"fold_{fold_num}_spearman": spearman_corr,
                f"fold_{fold_num}_top_decile_net": top_dec_net,
                f"fold_{fold_num}_best_epoch": best_epoch,
            }, step=fold_num)

            # KILL CHECK after 3 folds
            if len(fold_results) == KILL_AFTER_FOLDS:
                avg_sp = np.mean([r["spearman"] for r in fold_results])
                if avg_sp < KILL_SPEARMAN_THRESHOLD:
                    logger.error(f"KILL: First {KILL_AFTER_FOLDS} folds avg Spearman "
                               f"= {avg_sp:.4f} < {KILL_SPEARMAN_THRESHOLD}. ABORTING.")
                    mlflow.log_metric("killed_early", 1)
                    mlflow.set_tag("status", "KILLED_LOW_SPEARMAN")
                    sys.exit(1)
                else:
                    logger.info(f"  PASS kill check: avg Spearman={avg_sp:.4f} >= {KILL_SPEARMAN_THRESHOLD}")

            # Save fold model + scaler
            torch.save({
                "model_state": best_state,
                "mean": mean,
                "std": std,
                "feature_cols": feature_cols,
                "n_features": n_features,
                "hidden_dims": HIDDEN_DIMS,
                "dropout": DROPOUT,
            }, OUTPUT_DIR / f"fold_{fold_num}_{oot_date}.pt")

        # ============ CONCAT ANALYSIS ============
        logger.info(SEP)
        logger.info("CONCAT OOT ANALYSIS (all folds combined)")
        logger.info(SEP)

        all_preds = np.array(all_oot_preds)
        all_targets = np.array(all_oot_targets)

        concat_spearman, concat_p = spearmanr(all_preds, all_targets)
        if np.isnan(concat_spearman):
            concat_spearman = 0.0

        n_total = len(all_preds)
        top_dec_idx = np.argsort(all_preds)[-n_total // 10:]
        concat_top_net = all_targets[top_dec_idx].mean()
        concat_top_std = all_targets[top_dec_idx].std()
        concat_top_sharpe = (concat_top_net / concat_top_std * np.sqrt(252)) if concat_top_std > 0 else 0.0
        top_dec_wr = (all_targets[top_dec_idx] > 0).mean()

        bot_dec_idx = np.argsort(all_preds)[:n_total // 10]
        concat_bot_net = all_targets[bot_dec_idx].mean()

        # Daily Sharpe from fold top-decile nets
        daily_nets = [r["top_decile_mean_net"] for r in fold_results]
        daily_sharpe = (np.mean(daily_nets) / np.std(daily_nets) * np.sqrt(252)) if np.std(daily_nets) > 0 else 0.0

        logger.info(f"  Concat Spearman: {concat_spearman:.4f} (p={concat_p:.2e})")
        logger.info(f"  Concat Top Decile Mean Net: {concat_top_net:.4f} ticks")
        logger.info(f"  Concat Top Decile Sharpe: {concat_top_sharpe:.2f}")
        logger.info(f"  Concat Bot Decile Mean Net: {concat_bot_net:.4f} ticks")
        logger.info(f"  Concat Spread (top-bot): {concat_top_net - concat_bot_net:.4f} ticks")
        logger.info(f"  Top Decile WR: {top_dec_wr:.3f}")
        logger.info(f"  Daily Sharpe (top decile): {daily_sharpe:.2f}")
        logger.info(f"  Per-fold Spearman avg: {np.mean([r['spearman'] for r in fold_results]):.4f}")

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

        # ============ ACCEPTANCE GATE CHECK ============
        logger.info(SEP)
        logger.info("ACCEPTANCE GATE CHECK")
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

        # ============ PERMUTATION TEST (100 shuffles) ============
        logger.info(SEP)
        logger.info(f"PERMUTATION TEST ({N_PERMUTATIONS} permutations)")
        logger.info("Using LAST fold data for permutation test")
        logger.info(SEP)

        # Use last fold for permutation test (most representative of current regime)
        last_fold_idx = n_dates - 1
        last_oot_date = dates[last_fold_idx]
        last_train_dates = dates[last_fold_idx - TRAIN_WINDOW:last_fold_idx]

        last_train_dfs = [all_data[d] for d in last_train_dates]
        last_train_df = pd.concat(last_train_dfs, ignore_index=True)
        last_oot_df = all_data[last_oot_date]

        X_train_perm = np.nan_to_num(last_train_df[feature_cols].values.astype(np.float32), nan=0.0)
        y_train_perm = last_train_df["net_pnl_ticks"].values.astype(np.float32)
        X_oot_perm = np.nan_to_num(last_oot_df[feature_cols].values.astype(np.float32), nan=0.0)
        y_oot_perm = last_oot_df["net_pnl_ticks"].values.astype(np.float32)

        # Real Spearman for this fold
        real_preds, _, _, _, _, _, _ = train_one_fold(
            X_train_perm, y_train_perm, X_oot_perm, n_features, last_fold_idx
        )
        real_sp, _ = spearmanr(real_preds, y_oot_perm)
        if np.isnan(real_sp):
            real_sp = 0.0

        logger.info(f"  Real Spearman (last fold): {real_sp:.4f}")
        perm_p, null_spearmans = run_permutation_test(
            X_train_perm, y_train_perm, X_oot_perm, y_oot_perm,
            n_features, real_sp, last_fold_idx
        )

        logger.info(f"  Permutation p-value: {perm_p:.4f}")
        logger.info(f"  Null Spearman: mean={null_spearmans.mean():.4f}, "
                   f"std={null_spearmans.std():.4f}, "
                   f"max={null_spearmans.max():.4f}")
        logger.info(f"  Real Spearman vs null: {real_sp:.4f} vs {null_spearmans.mean():.4f}")

        perm_significant = perm_p < 0.05
        logger.info(f"  Significant at p<0.05: {'YES' if perm_significant else 'NO'}")

        mlflow.log_metrics({
            "perm_p_value": perm_p,
            "perm_real_spearman": real_sp,
            "perm_null_mean": null_spearmans.mean(),
            "perm_null_std": null_spearmans.std(),
            "perm_significant": 1.0 if perm_significant else 0.0,
        })

        # ============ SAVE RESULTS ============
        results = {
            "model_type": "mlp_pytorch",
            "architecture": f"{n_features}→{'→'.join(str(h) for h in HIDDEN_DIMS)}→1",
            "concat_spearman": float(concat_spearman),
            "concat_top_decile_net": float(concat_top_net),
            "concat_top_decile_sharpe": float(concat_top_sharpe),
            "concat_bot_decile_net": float(concat_bot_net),
            "top_decile_wr": float(top_dec_wr),
            "daily_sharpe": float(daily_sharpe),
            "acceptance": overall,
            "permutation_p_value": float(perm_p),
            "permutation_significant": perm_significant,
            "n_folds": len(fold_results),
            "fold_results": fold_results,
            "feature_columns": feature_cols,
            "hyperparams": {
                "hidden_dims": HIDDEN_DIMS,
                "dropout": DROPOUT,
                "lr": LR,
                "epochs_max": EPOCHS,
                "batch_size": BATCH_SIZE,
                "val_fraction": VAL_FRACTION,
                "patience": PATIENCE,
            },
        }

        with open(OUTPUT_DIR / "results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        np.savez(OUTPUT_DIR / "oot_predictions.npz",
                 predictions=all_preds,
                 targets=all_targets,
                 null_spearmans=null_spearmans)

        try:
            mlflow.log_artifact(str(OUTPUT_DIR / "results.json"))
        except Exception as e:
            logger.warning(f"Artifact upload failed (non-fatal): {e}")

        logger.info(SEP)
        logger.info(f"DONE. Results saved to {OUTPUT_DIR}")
        logger.info(f"Total fold training time: {sum(r['train_time_s'] for r in fold_results):.0f}s")
        logger.info(f"Acceptance: {overall} | Permutation p={perm_p:.4f}")
        logger.info(SEP)


if __name__ == "__main__":
    logger.info(SEP)
    logger.info("PASSIVE EXECUTION OPTIMIZER v1 — MLP (PyTorch GPU)")
    logger.info(f"Started: {datetime.now().isoformat()}")
    logger.info(f"Config: window={TRAIN_WINDOW}d, lr={LR}, epochs={EPOCHS}, "
               f"arch={HIDDEN_DIMS}, dropout={DROPOUT}")
    logger.info(f"Commission: {COMMISSION_PASSIVE} ticks (passive limit)")
    logger.info(SEP)
    run_walk_forward()
