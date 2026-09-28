#!/usr/bin/env python3
"""
Trade Management v7 — Regime-Aware Mid-Trade MLP
=================================================

Motivation: v6 continuous edge scoring achieved IC 0.398 and Sharpe 3.77, but
FAILED the regime gate (HC #428 R1) with a gap of 1.91 (green Sharpe +4.92,
red Sharpe -5.42). This experiment answers:

  Q: Does the regime bias come from the ENTRY SIGNAL or the MID-TRADE MLP?

Approach (two-phase):

  Phase 1 — REGIME DECOMPOSITION:
    - Load v6 predictions + trades
    - Classify each day as green/red/flat (ES close-to-close from enhanced_daily)
    - Compute per-feature importance separately on green vs red days
    - Identify which features drive the asymmetry

  Phase 2 — REGIME-AWARE v7 MLP:
    (a) Add regime indicator features: prior-day return, realized vol 5d,
        vol_regime_5d, price_range, volume_concentration
    (b) Regime-balanced sampling: equal weight green/red days in training
    (c) Both (a) + (b)
    - Walk-forward with SLIDING windows (HC #0)
    - Log everything to MLflow

Cost: passive entry 0.376 ticks, market exit 1.376 ticks (FIFO).
Regime gate: |Sharpe_green - Sharpe_red| / max(|Sharpe_green|,|Sharpe_red|) <= 0.50

Usage:
  cd /home/nick/Lvl3Quant && \\
  /home/nick/miniconda3/envs/py311-train/bin/python \\
      alpha_discovery/trade_management_v7_regime_aware.py

Author: Claude (autonomous research)
"""

import gc
import json
import logging
import os
import pickle
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
V6_DIR = ROOT / "output" / "trade_management_v6_continuous"
OUTPUT_DIR = ROOT / "output" / "trade_management_v7_regime_aware"
LOG_DIR = ROOT / "logs"
MODEL_DIR = OUTPUT_DIR / "models"
DAILY_FEATURES_PATH = ROOT / "output" / "long_horizon_flow_v2" / "enhanced_daily_features.parquet"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [TM-v7] %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_DIR / "trade_management_v7_regime_aware.log")),
    ],
)
log = logging.getLogger("TM-v7")

# ─────────────────────────────────────────────
#  COST CONSTANTS (ES Futures — AMP/Rithmic)
# ─────────────────────────────────────────────
ES_TICK_VALUE = 12.50
COST_RT_TICKS = 1.376  # passive entry + market exit
COST_PARTIAL_EXIT_TICKS = 1.376

# Walk-forward: 20d/10d/5d SLIDING (HC #0)
TRAIN_DAYS = 20
VAL_DAYS = 10
SLIDE_DAYS = 5

# MLP config (same as v6 baseline for fair comparison)
MLP_HIDDEN_1 = 64
MLP_HIDDEN_2 = 32
MLP_DROPOUT = 0.2
MLP_LR = 1e-3
MLP_EPOCHS = 100
MLP_BATCH_SIZE = 512
MLP_PATIENCE = 15

# Regime classification threshold (in price points, ~4 ticks = 1 point)
REGIME_GREEN_THRESHOLD = 20   # > +20 pts = green day
REGIME_RED_THRESHOLD = -20    # < -20 pts = red day

# Deferred imports
lgb = None
torch = None
nn = None


def _import_lightgbm():
    global lgb
    if lgb is not None:
        return
    import lightgbm as _lgb
    lgb = _lgb


def _import_torch():
    global torch, nn
    if torch is not None:
        return
    import torch as _torch
    import torch.nn as _nn
    torch = _torch
    nn = _nn


# ═══════════════════════════════════════════════════════════════════
#  SECTION 1: DATA LOADING + REGIME CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════


def load_v6_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load v6 tick samples and predictions."""
    tick_path = V6_DIR / "tick_samples_v6.parquet"
    pred_path = V6_DIR / "predictions_oot.parquet"

    if not tick_path.exists():
        raise FileNotFoundError(f"v6 tick samples not found: {tick_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"v6 predictions not found: {pred_path}")

    tick_df = pd.read_parquet(tick_path)
    pred_df = pd.read_parquet(pred_path)

    log.info(f"Loaded v6 tick samples: {len(tick_df):,} rows, {tick_df['date'].nunique()} dates")
    log.info(f"Loaded v6 predictions: {len(pred_df):,} rows")

    return tick_df, pred_df


def load_daily_features() -> pd.DataFrame:
    """Load enhanced daily features for regime classification."""
    if not DAILY_FEATURES_PATH.exists():
        raise FileNotFoundError(f"Daily features not found: {DAILY_FEATURES_PATH}")

    df = pd.read_parquet(DAILY_FEATURES_PATH)
    df["date_str"] = df["date"].dt.strftime("%Y%m%d")
    df = df.sort_values("date_str").reset_index(drop=True)

    # Compute close-to-close return
    df["close_return"] = df["close"].diff()

    # Classify regime
    df["regime"] = "flat"
    df.loc[df["close_return"] > REGIME_GREEN_THRESHOLD, "regime"] = "green"
    df.loc[df["close_return"] < REGIME_RED_THRESHOLD, "regime"] = "red"

    # Compute lookback features that can be used as regime indicators
    # (only using PRIOR day info to avoid lookahead)
    df["prior_day_return"] = df["close_return"].shift(1)
    df["realized_vol_5d"] = df["close_return"].rolling(5, min_periods=2).std().shift(1)
    df["realized_vol_10d"] = df["close_return"].rolling(10, min_periods=3).std().shift(1)
    df["return_3d_lag"] = df["close_return"].rolling(3, min_periods=1).sum().shift(1)
    df["return_5d_lag"] = df["close_return"].rolling(5, min_periods=2).sum().shift(1)
    df["range_ma5"] = df["price_range_ticks"].rolling(5, min_periods=2).mean().shift(1)
    df["vol_conc_ma5"] = df["volume_concentration"].rolling(5, min_periods=2).mean().shift(1)

    # Fill NaN in lookback features with column median
    lookback_cols = [
        "prior_day_return", "realized_vol_5d", "realized_vol_10d",
        "return_3d_lag", "return_5d_lag", "range_ma5", "vol_conc_ma5",
    ]
    for col in lookback_cols:
        df[col] = df[col].fillna(df[col].median())

    log.info(f"Daily features: {len(df)} days, regime dist: "
             f"green={sum(df['regime']=='green')}, "
             f"red={sum(df['regime']=='red')}, "
             f"flat={sum(df['regime']=='flat')}")

    return df


def merge_regime_data(tick_df: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
    """Merge regime classification and daily features into tick samples."""
    # Build date -> regime + features mapping
    regime_cols = [
        "date_str", "regime", "close_return", "prior_day_return",
        "realized_vol_5d", "realized_vol_10d", "return_3d_lag",
        "return_5d_lag", "range_ma5", "vol_conc_ma5",
        "price_range_ticks", "volume_concentration", "close_vs_open_ticks",
    ]
    # Only keep cols that exist
    regime_cols = [c for c in regime_cols if c in daily_df.columns]
    regime_map = daily_df[regime_cols].copy()
    regime_map = regime_map.rename(columns={"date_str": "date"})

    # Merge on date
    merged = tick_df.merge(regime_map, on="date", how="left")

    n_green = (merged["regime"] == "green").sum()
    n_red = (merged["regime"] == "red").sum()
    n_flat = (merged["regime"] == "flat").sum()
    log.info(f"Merged regime data: green={n_green:,}, red={n_red:,}, flat={n_flat:,} samples")

    return merged


# ═══════════════════════════════════════════════════════════════════
#  SECTION 2: REGIME DECOMPOSITION ANALYSIS
# ═══════════════════════════════════════════════════════════════════


def regime_decomposition(
    tick_df: pd.DataFrame, pred_df: pd.DataFrame, daily_df: pd.DataFrame
) -> Dict:
    """
    Phase 1: Analyze WHERE the regime bias comes from.

    1. Compute feature importance on green vs red days separately (LightGBM)
    2. Compare feature distributions green vs red
    3. Analyze entry signal vs mid-trade feature contribution
    """
    _import_lightgbm()
    log.info("\n" + "=" * 70)
    log.info("PHASE 1: REGIME DECOMPOSITION ANALYSIS")
    log.info("=" * 70)

    # Merge regime info
    merged = merge_regime_data(tick_df, daily_df)

    # Original v6 feature columns (mid-trade features only)
    feature_cols = _get_base_feature_columns()
    feature_cols = [c for c in feature_cols if c in merged.columns]
    target_col = "label_remaining_favorable_ticks"

    results = {}

    # ── 2a: Feature importance on GREEN vs RED days ──
    log.info("\n--- Feature importance by regime ---")

    for regime_name in ["green", "red"]:
        regime_data = merged[merged["regime"] == regime_name].copy()
        if len(regime_data) < 100:
            log.warning(f"  {regime_name}: only {len(regime_data)} samples, skipping")
            continue

        X = regime_data[feature_cols].values.astype(np.float32)
        y = regime_data[target_col].values.astype(np.float32)

        # Clean
        valid = ~np.isnan(y) & ~np.isinf(y) & ~np.isnan(X).any(axis=1) & ~np.isinf(X).any(axis=1)
        X, y = X[valid], y[valid]

        if len(X) < 100:
            continue

        # Train a quick LightGBM to get feature importance
        dtrain = lgb.Dataset(X, label=y, feature_name=feature_cols, free_raw_data=False)
        params = {
            "objective": "regression", "metric": "mae",
            "learning_rate": 0.05, "num_leaves": 31,
            "min_child_samples": 30, "verbose": -1, "n_jobs": -1,
        }
        model = lgb.train(params, dtrain, num_boost_round=300)
        imp = model.feature_importance(importance_type="gain")
        imp_dict = dict(zip(feature_cols, imp.tolist()))

        # Sort by importance
        sorted_imp = sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)
        results[f"importance_{regime_name}"] = imp_dict

        log.info(f"\n  {regime_name.upper()} day feature importance (top 15):")
        for feat, val in sorted_imp[:15]:
            log.info(f"    {feat:35s} {val:10.1f}")

        del model, dtrain
        gc.collect()

    # ── 2b: Importance DIFFERENCE (what drives asymmetry) ──
    if "importance_green" in results and "importance_red" in results:
        log.info("\n--- Feature importance DIVERGENCE (green - red) ---")
        green_imp = results["importance_green"]
        red_imp = results["importance_red"]

        divergence = {}
        for feat in feature_cols:
            g = green_imp.get(feat, 0)
            r = red_imp.get(feat, 0)
            total = g + r
            if total > 0:
                divergence[feat] = {
                    "green": g, "red": r,
                    "diff": g - r,
                    "ratio": g / max(r, 1),
                    "pct_of_total_green": g / max(sum(green_imp.values()), 1) * 100,
                    "pct_of_total_red": r / max(sum(red_imp.values()), 1) * 100,
                }

        # Sort by absolute difference
        sorted_div = sorted(divergence.items(), key=lambda x: abs(x[1]["diff"]), reverse=True)
        log.info(f"\n  Top 15 DIVERGENT features (biggest regime-dependent importance):")
        for feat, vals in sorted_div[:15]:
            log.info(
                f"    {feat:35s} green={vals['green']:8.1f} red={vals['red']:8.1f} "
                f"diff={vals['diff']:+8.1f} ratio={vals['ratio']:.2f}"
            )
        results["divergence"] = {k: v for k, v in sorted_div}

        # ── 2c: Classify features as ENTRY vs MID-TRADE ──
        entry_features = {"entry_confidence", "entry_direction"}
        mid_trade_features = set(feature_cols) - entry_features

        entry_div = sum(abs(divergence[f]["diff"]) for f in entry_features if f in divergence)
        midtrade_div = sum(abs(divergence[f]["diff"]) for f in mid_trade_features if f in divergence)
        total_div = entry_div + midtrade_div

        log.info(f"\n  ENTRY SIGNAL divergence contribution: {entry_div:.1f} ({entry_div/max(total_div,1)*100:.1f}%)")
        log.info(f"  MID-TRADE divergence contribution:   {midtrade_div:.1f} ({midtrade_div/max(total_div,1)*100:.1f}%)")

        if entry_div / max(total_div, 1) > 0.3:
            log.info("  CONCLUSION: Entry signal contributes significantly to regime bias")
            results["bias_source"] = "entry_signal_significant"
        else:
            log.info("  CONCLUSION: Regime bias is primarily in mid-trade features")
            results["bias_source"] = "mid_trade_features"

    # ── 2d: Feature distribution comparison ──
    log.info("\n--- Feature distribution green vs red (KS test) ---")
    green_data = merged[merged["regime"] == "green"]
    red_data = merged[merged["regime"] == "red"]

    ks_results = {}
    for feat in feature_cols:
        g = green_data[feat].dropna().values
        r = red_data[feat].dropna().values
        if len(g) > 10 and len(r) > 10:
            ks_stat, ks_p = stats.ks_2samp(g, r)
            mean_diff = np.mean(g) - np.mean(r)
            std_pool = np.sqrt((np.std(g)**2 + np.std(r)**2) / 2)
            cohens_d = mean_diff / max(std_pool, 1e-8)
            ks_results[feat] = {
                "ks_stat": ks_stat, "ks_p": ks_p,
                "mean_green": float(np.mean(g)), "mean_red": float(np.mean(r)),
                "cohens_d": cohens_d,
            }

    sorted_ks = sorted(ks_results.items(), key=lambda x: x[1]["ks_stat"], reverse=True)
    log.info(f"\n  Top 10 features with LARGEST distribution shift green vs red:")
    for feat, vals in sorted_ks[:10]:
        log.info(
            f"    {feat:35s} KS={vals['ks_stat']:.3f} p={vals['ks_p']:.2e} "
            f"d={vals['cohens_d']:+.3f} mean_g={vals['mean_green']:.3f} mean_r={vals['mean_red']:.3f}"
        )
    results["ks_tests"] = ks_results

    # ── 2e: Per-regime P&L decomposition ──
    log.info("\n--- Per-regime trade P&L from v6 ---")
    # Get per-trade final P&L
    trade_pnl = merged.groupby("trade_idx").agg({
        "label_final_pnl_ticks": "first",
        "regime": "first",
        "date": "first",
    }).reset_index()

    for regime in ["green", "red", "flat"]:
        rdata = trade_pnl[trade_pnl["regime"] == regime]
        if len(rdata) == 0:
            continue
        pnl = rdata["label_final_pnl_ticks"].values - COST_RT_TICKS
        pnl = pnl[~np.isnan(pnl)]
        if len(pnl) == 0:
            continue
        sharpe = np.mean(pnl) / max(np.std(pnl), 1e-8) * np.sqrt(252)
        wr = np.mean(pnl > 0)
        log.info(
            f"  {regime:5s}: n={len(pnl):3d} mean={np.mean(pnl):.2f}t "
            f"std={np.std(pnl):.2f}t Sharpe={sharpe:.2f} WR={wr:.1%}"
        )
        results[f"regime_{regime}_sharpe"] = sharpe
        results[f"regime_{regime}_n"] = len(pnl)
        results[f"regime_{regime}_mean_pnl"] = float(np.mean(pnl))

    return results


# ═══════════════════════════════════════════════════════════════════
#  SECTION 3: REGIME-AWARE MLP (V7)
# ═══════════════════════════════════════════════════════════════════


def _get_base_feature_columns() -> List[str]:
    """Base v6 feature columns (without regime features)."""
    return [
        "time_in_trade_seconds", "unrealized_pnl_ticks", "mfe_so_far_ticks",
        "mae_so_far_ticks", "drawdown_from_peak", "our_side_qty",
        "against_side_qty", "queue_ratio", "queue_ratio_change",
        "our_side_add_rate", "against_side_cancel_rate", "ofi_since_entry",
        "ofi_recent_5s", "ofi_alignment", "imbalance_now", "imbalance_vs_entry",
        "microprice_offset_now", "microprice_trend", "trade_rate_our_side",
        "trade_rate_against_side", "level_age_our_side", "cancel_spike",
        "our_side_n_orders", "against_side_n_orders", "entry_confidence",
        "entry_direction",
        # v6 engineered features
        "queue_ratio_velocity", "ofi_acceleration", "microprice_velocity",
        "imbalance_velocity", "drawdown_speed", "gain_speed", "pnl_per_second",
        "mae_recovery_ratio", "flow_agreement", "pressure_momentum",
        "queue_flow_divergence", "net_queue_drain", "against_our_ratio",
        "spread_proxy_trend", "decayed_confidence", "confidence_weighted_pnl",
        "time_fraction",
    ]


def _get_regime_feature_columns() -> List[str]:
    """New regime indicator features (all use prior-day data, no lookahead)."""
    return [
        "prior_day_return",
        "realized_vol_5d",
        "realized_vol_10d",
        "return_3d_lag",
        "return_5d_lag",
        "range_ma5",
        "vol_conc_ma5",
    ]


def _build_v7_mlp(n_features: int, device: str = "cpu"):
    """Build regime-aware MLP (same architecture, more input features)."""
    _import_torch()

    class RegimeAwareMLP(nn.Module):
        def __init__(self, n_in):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_in, MLP_HIDDEN_1),
                nn.ReLU(),
                nn.Dropout(MLP_DROPOUT),
                nn.Linear(MLP_HIDDEN_1, MLP_HIDDEN_2),
                nn.ReLU(),
                nn.Dropout(MLP_DROPOUT),
                nn.Linear(MLP_HIDDEN_2, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    model = RegimeAwareMLP(n_features).to(device)
    return model


def _train_v7_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    sample_weights: Optional[np.ndarray] = None,
    device: str = "cpu",
) -> Tuple[Any, float, float, List[float]]:
    """Train MLP with early stopping + optional sample weighting. Returns (model, val_mae, val_ic, loss_history)."""
    _import_torch()

    n_features = X_train.shape[1]
    model = _build_v7_mlp(n_features, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=MLP_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7, min_lr=1e-5
    )
    criterion = nn.SmoothL1Loss(reduction="none")

    # Normalize features
    train_mean = np.nanmean(X_train, axis=0)
    train_std = np.nanstd(X_train, axis=0)
    train_std[train_std < 1e-8] = 1.0

    X_tr_norm = (X_train - train_mean) / train_std
    X_v_norm = (X_val - train_mean) / train_std

    X_tr_t = torch.tensor(X_tr_norm, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_v_t = torch.tensor(X_v_norm, dtype=torch.float32, device=device)
    y_v_t = torch.tensor(y_val, dtype=torch.float32, device=device)

    if sample_weights is not None:
        w_t = torch.tensor(sample_weights, dtype=torch.float32, device=device)
    else:
        w_t = torch.ones(len(y_train), dtype=torch.float32, device=device)

    dataset = torch.utils.data.TensorDataset(X_tr_t, y_tr_t, w_t)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=MLP_BATCH_SIZE, shuffle=True, drop_last=False
    )

    best_val_mae = float("inf")
    best_state = None
    patience_counter = 0
    loss_history = []

    for epoch in range(MLP_EPOCHS):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb, wb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss_per_sample = criterion(pred, yb)
            loss = (loss_per_sample * wb).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        loss_history.append(avg_loss)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_v_t)
            val_mae = torch.mean(torch.abs(val_pred - y_v_t)).item()

        scheduler.step(val_mae)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= MLP_PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    model._norm_mean = train_mean
    model._norm_std = train_std

    # Compute val IC
    with torch.no_grad():
        val_pred_np = model(X_v_t).cpu().numpy()
    val_ic = float(np.corrcoef(y_val, val_pred_np)[0, 1]) if len(y_val) > 5 else 0.0

    return model, best_val_mae, val_ic, loss_history


def _predict_v7_mlp(model, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    _import_torch()
    X_norm = (X - model._norm_mean) / model._norm_std
    X_t = torch.tensor(X_norm, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        preds = model(X_t).cpu().numpy()
    return preds


def compute_regime_weights(dates: np.ndarray, date_regime_map: Dict[str, str]) -> np.ndarray:
    """
    Compute per-sample weights for regime-balanced training.
    Upweight minority regime so green and red contribute equally.
    """
    regimes = np.array([date_regime_map.get(d, "flat") for d in dates])
    n_green = np.sum(regimes == "green")
    n_red = np.sum(regimes == "red")
    n_flat = np.sum(regimes == "flat")

    total = len(regimes)
    weights = np.ones(total, dtype=np.float32)

    if n_green > 0 and n_red > 0:
        # Balance green and red to equal effective weight
        max_count = max(n_green, n_red)
        weights[regimes == "green"] = max_count / n_green
        weights[regimes == "red"] = max_count / n_red
        weights[regimes == "flat"] = max_count / max(n_flat, 1)

    # Normalize so mean weight = 1
    weights /= weights.mean()
    return weights


def train_v7_models(
    merged_df: pd.DataFrame,
    daily_df: pd.DataFrame,
) -> Tuple[Dict, pd.DataFrame]:
    """
    Walk-forward train three v7 MLP variants:
      A: base features + regime indicator features (no rebalancing)
      B: base features only + regime-balanced sampling
      C: base features + regime features + regime-balanced sampling

    20d train / 10d val / 5d slide — SLIDING only.
    """
    _import_torch()

    base_cols = _get_base_feature_columns()
    base_cols = [c for c in base_cols if c in merged_df.columns]
    regime_cols = _get_regime_feature_columns()
    regime_cols = [c for c in regime_cols if c in merged_df.columns]

    all_cols = base_cols + regime_cols
    target_col = "label_remaining_favorable_ticks"

    log.info(f"\n{'='*70}")
    log.info("PHASE 2: REGIME-AWARE V7 MLP TRAINING")
    log.info(f"{'='*70}")
    log.info(f"Base features: {len(base_cols)}, Regime features: {len(regime_cols)}")
    log.info(f"Regime features: {regime_cols}")

    dates = sorted(merged_df["date"].unique())
    log.info(f"Total dates: {len(dates)}")

    # Build date->regime map
    date_regime_map = dict(zip(
        daily_df["date_str"].values,
        daily_df["regime"].values,
    ))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    # Track predictions for each variant
    variants = {
        "A_regime_features": {"cols": all_cols, "balanced": False},
        "B_balanced_sampling": {"cols": base_cols, "balanced": True},
        "C_both": {"cols": all_cols, "balanced": True},
    }

    for vname in variants:
        merged_df[f"pred_{vname}"] = np.nan

    all_fold_results = {vname: [] for vname in variants}

    fold_idx = 0
    start = 0

    while start + TRAIN_DAYS + VAL_DAYS <= len(dates):
        train_dates = dates[start:start + TRAIN_DAYS]
        val_dates = dates[start + TRAIN_DAYS:start + TRAIN_DAYS + VAL_DAYS]

        train_mask = merged_df["date"].isin(train_dates)
        val_mask = merged_df["date"].isin(val_dates)

        log.info(f"\n--- Fold {fold_idx}: train {train_dates[0]}..{train_dates[-1]} | val {val_dates[0]}..{val_dates[-1]} ---")

        for vname, vconfig in variants.items():
            feat_cols = vconfig["cols"]
            use_balanced = vconfig["balanced"]

            X_train = merged_df.loc[train_mask, feat_cols].values.astype(np.float32)
            y_train = merged_df.loc[train_mask, target_col].values.astype(np.float32)
            X_val = merged_df.loc[val_mask, feat_cols].values.astype(np.float32)
            y_val = merged_df.loc[val_mask, target_col].values.astype(np.float32)

            # Clean
            valid_train = (
                ~np.isnan(y_train) & ~np.isinf(y_train) &
                ~np.isnan(X_train).any(axis=1) & ~np.isinf(X_train).any(axis=1)
            )
            valid_val = (
                ~np.isnan(y_val) & ~np.isinf(y_val) &
                ~np.isnan(X_val).any(axis=1) & ~np.isinf(X_val).any(axis=1)
            )

            X_tr = X_train[valid_train]
            y_tr = y_train[valid_train]
            X_v = X_val[valid_val]
            y_v = y_val[valid_val]

            if len(X_tr) < 200 or len(X_v) < 20:
                log.warning(f"  {vname}: insufficient data (tr={len(X_tr)}, val={len(X_v)})")
                start += SLIDE_DAYS
                continue

            # Compute sample weights if balanced
            sample_weights = None
            if use_balanced:
                train_dates_arr = merged_df.loc[train_mask, "date"].values[valid_train]
                sample_weights = compute_regime_weights(train_dates_arr, date_regime_map)
                n_regimes = {}
                for d in train_dates_arr:
                    r = date_regime_map.get(d, "flat")
                    n_regimes[r] = n_regimes.get(r, 0) + 1
                log.info(f"  {vname}: regime balance in train: {n_regimes}")

            # Train MLP
            model, val_mae, val_ic, loss_hist = _train_v7_mlp(
                X_tr, y_tr, X_v, y_v,
                sample_weights=sample_weights,
                device=device,
            )

            # Store OOT predictions
            val_indices = merged_df.index[val_mask]
            valid_val_positions = np.where(valid_val)[0]
            preds = _predict_v7_mlp(model, X_v, device)

            for i, pos in enumerate(valid_val_positions):
                if pos < len(val_indices):
                    merged_df.loc[val_indices[pos], f"pred_{vname}"] = preds[i]

            all_fold_results[vname].append({
                "fold": fold_idx,
                "val_start": val_dates[0],
                "val_end": val_dates[-1],
                "n_train": len(X_tr),
                "n_val": len(X_v),
                "val_mae": val_mae,
                "val_ic": val_ic,
                "epochs": len(loss_hist),
            })

            log.info(f"  {vname}: MAE={val_mae:.3f} IC={val_ic:.4f} epochs={len(loss_hist)}")

            # Save last fold model
            if start + SLIDE_DAYS + TRAIN_DAYS + VAL_DAYS > len(dates):
                model_path = MODEL_DIR / f"mlp_{vname}_latest.pt"
                torch.save({
                    "state_dict": model.state_dict(),
                    "norm_mean": model._norm_mean,
                    "norm_std": model._norm_std,
                    "n_features": len(feat_cols),
                    "feature_cols": feat_cols,
                    "variant": vname,
                }, str(model_path))
                log.info(f"  Saved model: {model_path.name}")

            del model
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

        fold_idx += 1
        start += SLIDE_DAYS

    # Summary
    log.info(f"\n{'='*70}")
    log.info("TRAINING SUMMARY")
    log.info(f"{'='*70}")

    summary = {}
    for vname, folds in all_fold_results.items():
        if not folds:
            continue
        avg_ic = np.mean([f["val_ic"] for f in folds])
        avg_mae = np.mean([f["val_mae"] for f in folds])
        log.info(f"  {vname}: {len(folds)} folds, avg IC={avg_ic:.4f}, avg MAE={avg_mae:.3f}")
        summary[vname] = {"avg_ic": avg_ic, "avg_mae": avg_mae, "n_folds": len(folds)}

    return summary, merged_df


# ═══════════════════════════════════════════════════════════════════
#  SECTION 4: SIMULATION + REGIME GATE CHECK
# ═══════════════════════════════════════════════════════════════════


def _compute_strategy_metrics(pnl_arr: np.ndarray, name: str) -> Dict:
    """Compute Sharpe, Sortino, PF, WR from P&L array."""
    pnl = pnl_arr[~np.isnan(pnl_arr)]
    if len(pnl) < 3:
        return {"name": name, "sharpe": 0, "sortino": 0, "profit_factor": 0,
                "win_rate": 0, "mean_pnl_ticks": 0, "n_trades": 0}

    mean_pnl = np.mean(pnl)
    std_pnl = np.std(pnl)
    sharpe = mean_pnl / max(std_pnl, 1e-8) * np.sqrt(252)

    downside = pnl[pnl < 0]
    downside_std = np.std(downside) if len(downside) > 1 else std_pnl
    sortino = mean_pnl / max(downside_std, 1e-8) * np.sqrt(252)

    gross_profit = np.sum(pnl[pnl > 0])
    gross_loss = abs(np.sum(pnl[pnl < 0]))
    pf = gross_profit / max(gross_loss, 1e-8)

    wr = np.mean(pnl > 0)

    return {
        "name": name, "sharpe": float(sharpe), "sortino": float(sortino),
        "profit_factor": float(pf), "win_rate": float(wr),
        "mean_pnl_ticks": float(mean_pnl), "n_trades": int(len(pnl)),
        "total_pnl_ticks": float(np.sum(pnl)),
    }


def simulate_and_check_regime(
    merged_df: pd.DataFrame,
    daily_df: pd.DataFrame,
) -> Dict:
    """
    Simulate threshold-exit strategies for each v7 variant.
    Check HC #428 R1 regime gate for each.
    """
    log.info(f"\n{'='*70}")
    log.info("PHASE 3: SIMULATION + REGIME GATE CHECK")
    log.info(f"{'='*70}")

    date_regime_map = dict(zip(
        daily_df["date_str"].values,
        daily_df["regime"].values,
    ))

    # Get per-trade data
    trade_groups = merged_df.groupby("trade_idx")
    trade_ids = sorted(merged_df["trade_idx"].unique())

    trade_meta = []
    for tid in trade_ids:
        grp = trade_groups.get_group(tid)
        trade_meta.append({
            "trade_idx": tid,
            "date": grp["date"].iloc[0],
            "direction": grp["entry_direction"].iloc[0] if "entry_direction" in grp.columns else grp["direction"].iloc[0],
            "final_pnl_ticks": grp["label_final_pnl_ticks"].iloc[0] if "label_final_pnl_ticks" in grp.columns else np.nan,
            "max_time_s": grp["time_in_trade_seconds"].max(),
            "regime": date_regime_map.get(grp["date"].iloc[0], "flat"),
        })
    meta_df = pd.DataFrame(trade_meta)

    results = {}
    pred_cols = [c for c in merged_df.columns if c.startswith("pred_")]

    for pred_col in pred_cols:
        vname = pred_col.replace("pred_", "")
        valid = merged_df.dropna(subset=[pred_col])
        if len(valid) == 0:
            continue

        log.info(f"\n--- Variant: {vname} ({len(valid):,} valid samples) ---")

        # Try multiple exit thresholds
        best_sharpe = -999
        best_config = None
        best_config_results = None

        for exit_thresh in [0.0, 0.5, 1.0, 1.5, 2.0]:
            config_key = f"{vname}_thresh_{exit_thresh:.1f}"
            trade_pnls = []

            for _, tinfo in meta_df.iterrows():
                tid = tinfo["trade_idx"]
                final_pnl = tinfo["final_pnl_ticks"]
                if np.isnan(final_pnl):
                    continue

                trade_samples = valid[valid["trade_idx"] == tid].sort_values("time_in_trade_seconds")
                if len(trade_samples) == 0:
                    pnl = final_pnl - COST_RT_TICKS
                    trade_pnls.append({
                        "trade_idx": tid, "pnl": pnl,
                        "hold_sec": tinfo["max_time_s"],
                        "early_exit": False, "regime": tinfo["regime"],
                    })
                    continue

                exited_early = False
                for _, sample in trade_samples.iterrows():
                    t = sample["time_in_trade_seconds"]
                    pred_edge = sample[pred_col]
                    if np.isnan(pred_edge) or t < 30.0:
                        continue
                    if pred_edge < exit_thresh:
                        exit_pnl = sample["unrealized_pnl_ticks"] - COST_RT_TICKS
                        trade_pnls.append({
                            "trade_idx": tid, "pnl": exit_pnl,
                            "hold_sec": t, "early_exit": True,
                            "regime": tinfo["regime"],
                        })
                        exited_early = True
                        break

                if not exited_early:
                    pnl = final_pnl - COST_RT_TICKS
                    trade_pnls.append({
                        "trade_idx": tid, "pnl": pnl,
                        "hold_sec": tinfo["max_time_s"],
                        "early_exit": False, "regime": tinfo["regime"],
                    })

            if not trade_pnls:
                continue

            pnl_arr = np.array([t["pnl"] for t in trade_pnls])
            overall = _compute_strategy_metrics(pnl_arr, config_key)

            # Regime decomposition
            green_pnl = np.array([t["pnl"] for t in trade_pnls if t["regime"] == "green"])
            red_pnl = np.array([t["pnl"] for t in trade_pnls if t["regime"] == "red"])
            flat_pnl = np.array([t["pnl"] for t in trade_pnls if t["regime"] == "flat"])

            green_metrics = _compute_strategy_metrics(green_pnl, f"{config_key}_green")
            red_metrics = _compute_strategy_metrics(red_pnl, f"{config_key}_red")

            # HC #428 R1 regime gate
            sharpe_g = green_metrics["sharpe"]
            sharpe_r = red_metrics["sharpe"]
            max_abs = max(abs(sharpe_g), abs(sharpe_r), 1e-8)
            regime_gap = abs(sharpe_g - sharpe_r) / max_abs

            overall["sharpe_green"] = sharpe_g
            overall["sharpe_red"] = sharpe_r
            overall["regime_gap"] = regime_gap
            overall["regime_pass"] = regime_gap <= 0.50
            overall["exit_rate"] = float(np.mean([1 if t.get("early_exit") else 0 for t in trade_pnls]))

            log.info(
                f"  thresh={exit_thresh:.1f}: Sharpe={overall['sharpe']:.2f} "
                f"WR={overall['win_rate']:.1%} PF={overall['profit_factor']:.2f} "
                f"| green={sharpe_g:.2f} red={sharpe_r:.2f} gap={regime_gap:.2f} "
                f"{'PASS' if overall['regime_pass'] else 'FAIL'}"
            )

            results[config_key] = overall

            if overall["sharpe"] > best_sharpe:
                best_sharpe = overall["sharpe"]
                best_config = config_key
                best_config_results = overall

        if best_config:
            results[f"{vname}_best"] = best_config
            log.info(f"  BEST for {vname}: {best_config} (Sharpe={best_sharpe:.2f})")

    return results


# ═══════════════════════════════════════════════════════════════════
#  SECTION 5: MLFLOW LOGGING
# ═══════════════════════════════════════════════════════════════════


def log_to_mlflow(
    decomposition_results: Dict,
    training_summary: Dict,
    simulation_results: Dict,
    elapsed_seconds: float,
) -> Optional[str]:
    """Log all results to MLflow."""
    try:
        import mlflow
    except ImportError:
        log.warning("MLflow not available, skipping logging")
        return None

    mlflow_uri = "http://jupiter:5000"
    mlflow.set_tracking_uri(mlflow_uri)
    experiment_name = "trade_management_v7_regime_aware"

    try:
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        log.warning(f"Could not set MLflow experiment: {e}")
        return None

    try:
        with mlflow.start_run(run_name=f"v7_regime_{datetime.now().strftime('%Y%m%d_%H%M')}") as run:
            # Log params
            mlflow.log_param("train_days", TRAIN_DAYS)
            mlflow.log_param("val_days", VAL_DAYS)
            mlflow.log_param("slide_days", SLIDE_DAYS)
            mlflow.log_param("mlp_hidden_1", MLP_HIDDEN_1)
            mlflow.log_param("mlp_hidden_2", MLP_HIDDEN_2)
            mlflow.log_param("mlp_dropout", MLP_DROPOUT)
            mlflow.log_param("mlp_lr", MLP_LR)
            mlflow.log_param("cost_rt_ticks", COST_RT_TICKS)
            mlflow.log_param("regime_green_thresh", REGIME_GREEN_THRESHOLD)
            mlflow.log_param("regime_red_thresh", REGIME_RED_THRESHOLD)

            # Decomposition results
            if "bias_source" in decomposition_results:
                mlflow.log_param("bias_source", decomposition_results["bias_source"])
            for key in ["regime_green_sharpe", "regime_red_sharpe", "regime_flat_sharpe",
                        "regime_green_n", "regime_red_n"]:
                if key in decomposition_results:
                    mlflow.log_metric(f"decomp_{key}", decomposition_results[key])

            # Training summary
            for vname, vsummary in training_summary.items():
                mlflow.log_metric(f"train_{vname}_avg_ic", vsummary["avg_ic"])
                mlflow.log_metric(f"train_{vname}_avg_mae", vsummary["avg_mae"])
                mlflow.log_metric(f"train_{vname}_n_folds", vsummary["n_folds"])

            # Simulation results
            for config_key, config_results in simulation_results.items():
                if isinstance(config_results, dict) and "sharpe" in config_results:
                    prefix = f"sim_{config_key}"
                    mlflow.log_metric(f"{prefix}_sharpe", config_results["sharpe"])
                    mlflow.log_metric(f"{prefix}_sortino", config_results.get("sortino", 0))
                    mlflow.log_metric(f"{prefix}_pf", config_results.get("profit_factor", 0))
                    mlflow.log_metric(f"{prefix}_wr", config_results.get("win_rate", 0))
                    if "regime_gap" in config_results:
                        mlflow.log_metric(f"{prefix}_regime_gap", config_results["regime_gap"])
                        mlflow.log_metric(f"{prefix}_regime_pass", int(config_results["regime_pass"]))
                        mlflow.log_metric(f"{prefix}_sharpe_green", config_results["sharpe_green"])
                        mlflow.log_metric(f"{prefix}_sharpe_red", config_results["sharpe_red"])

            mlflow.log_metric("elapsed_seconds", elapsed_seconds)

            # Log artifacts
            summary_path = OUTPUT_DIR / "summary.json"
            if summary_path.exists():
                mlflow.log_artifact(str(summary_path))

            log.info(f"MLflow run logged: {run.info.run_id}")
            return run.info.run_id

    except Exception as e:
        log.error(f"MLflow logging failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════


def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("Trade Management v7 — Regime-Aware Mid-Trade MLP")
    log.info("=" * 70)

    # Load data
    tick_df, pred_df = load_v6_data()
    daily_df = load_daily_features()

    # Phase 1: Regime decomposition
    decomp_results = regime_decomposition(tick_df, pred_df, daily_df)

    # Merge regime features into tick data
    merged_df = merge_regime_data(tick_df, daily_df)

    # Phase 2: Train v7 variants
    training_summary, merged_df = train_v7_models(merged_df, daily_df)

    # Phase 3: Simulate + regime gate
    sim_results = simulate_and_check_regime(merged_df, daily_df)

    elapsed = time.time() - t0

    # Save predictions
    pred_cols = [c for c in merged_df.columns if c.startswith("pred_")]
    save_cols = ["trade_idx", "date", "time_in_trade_seconds",
                 "label_remaining_favorable_ticks", "regime"] + pred_cols
    save_cols = [c for c in save_cols if c in merged_df.columns]
    pred_out = merged_df[save_cols].copy()
    pred_path = OUTPUT_DIR / "predictions_oot.parquet"
    pred_out.to_parquet(pred_path, index=False)
    log.info(f"Saved predictions: {pred_path}")

    # Save full summary
    summary = {
        "completed_at": datetime.now().isoformat(),
        "elapsed_seconds": elapsed,
        "decomposition": {
            k: v for k, v in decomp_results.items()
            if not isinstance(v, dict) or k.startswith("regime_")
        },
        "training": training_summary,
        "simulation": {
            k: v for k, v in sim_results.items()
            if isinstance(v, dict)
        },
        "best_configs": {
            k: v for k, v in sim_results.items()
            if isinstance(v, str)
        },
    }

    # Find best regime-passing config
    best_passing = None
    best_passing_sharpe = -999
    for k, v in sim_results.items():
        if isinstance(v, dict) and v.get("regime_pass") and v.get("sharpe", -999) > best_passing_sharpe:
            best_passing = k
            best_passing_sharpe = v["sharpe"]

    summary["best_regime_passing_config"] = best_passing
    summary["best_regime_passing_sharpe"] = best_passing_sharpe if best_passing else None

    # Also find best overall (may not pass gate)
    best_overall = None
    best_overall_sharpe = -999
    for k, v in sim_results.items():
        if isinstance(v, dict) and "sharpe" in v and v["sharpe"] > best_overall_sharpe:
            best_overall = k
            best_overall_sharpe = v["sharpe"]
    summary["best_overall_config"] = best_overall
    summary["best_overall_sharpe"] = best_overall_sharpe if best_overall else None

    summary_path = OUTPUT_DIR / "summary.json"

    # Custom JSON serializer for numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)
    log.info(f"Saved summary: {summary_path}")

    # Log to MLflow
    mlflow_run_id = log_to_mlflow(decomp_results, training_summary, sim_results, elapsed)

    # Final report
    log.info(f"\n{'='*70}")
    log.info("FINAL REPORT")
    log.info(f"{'='*70}")
    log.info(f"Elapsed: {elapsed:.1f}s")
    log.info(f"Bias source: {decomp_results.get('bias_source', 'unknown')}")

    for vname in ["A_regime_features", "B_balanced_sampling", "C_both"]:
        if vname in training_summary:
            s = training_summary[vname]
            log.info(f"  {vname}: IC={s['avg_ic']:.4f} MAE={s['avg_mae']:.3f}")

    if best_passing:
        bp = sim_results[best_passing]
        log.info(f"\nBEST REGIME-PASSING CONFIG: {best_passing}")
        log.info(f"  Sharpe={bp['sharpe']:.2f} Sortino={bp['sortino']:.2f} "
                 f"PF={bp['profit_factor']:.2f} WR={bp['win_rate']:.1%}")
        log.info(f"  Green Sharpe={bp['sharpe_green']:.2f} Red Sharpe={bp['sharpe_red']:.2f} "
                 f"Gap={bp['regime_gap']:.2f} (<=0.50 required)")
    else:
        log.info("\nNO CONFIG PASSED REGIME GATE (gap <= 0.50)")
        if best_overall:
            bo = sim_results[best_overall]
            log.info(f"  Best overall: {best_overall} Sharpe={bo['sharpe']:.2f} "
                     f"gap={bo.get('regime_gap', 'N/A')}")

    if mlflow_run_id:
        log.info(f"\nMLflow run: {mlflow_run_id}")

    log.info("DONE")
    return summary


if __name__ == "__main__":
    main()
