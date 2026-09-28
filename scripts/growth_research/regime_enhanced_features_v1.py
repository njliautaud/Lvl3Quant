#!/usr/bin/env python3
"""
Regime Enhanced Features v1 — Keep v1 GRU architecture, improve features.

Hypothesis: The v1 GRU (2-layer, 64 hidden, 60-day lookback) is near-optimal
architecturally. But adding cross-asset and sentiment features should improve
transition recall — the key metric that v2 architectural changes failed to fix.

Tests 3 variants (same architecture):
  A. Baseline: original ~20 features (reproduction of v1)
  B. Baseline + 10 new cross-asset/sentiment features (30 total)
  C. Baseline + best 5 new features (selected by importance from B)

Walk-forward: 1000d train, 250d val, 20d step, sliding (NO expanding).
PyTorch GRU on CUDA. Logs to MLflow. Saves .npz predictions.

Author: Claude (Head of Quant)
Date: 2026-07-27
"""

import os
import sys
import time
import warnings
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import mlflow
import mlflow.pytorch

warnings.filterwarnings("ignore")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
CFG = {
    # Data — expanded ticker list for cross-asset features
    "tickers": [
        "SPY", "^VIX", "^VIX3M",
        "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY",
        "TLT", "SHY", "HYG", "GLD",
    ],
    "start_date": "2007-01-01",
    "end_date": None,

    # Architecture (FROZEN — same as v1 production winner)
    "model_type": "GRU",
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.3,
    "bidirectional": False,
    "lookback": 60,

    # Target
    "fwd_vol_window": 20,

    # Walk-forward
    "train_days": 1000,
    "val_days": 250,
    "step_days": 20,

    # Training
    "epochs": 80,
    "batch_size": 64,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "patience": 15,
    "min_lr": 1e-6,

    # MLflow
    "mlflow_uri": "http://jupiter:5000",
    "experiment_name": "regime_enhanced_features_v1",

    # Output
    "output_dir": "/home/nick/Lvl3Quant/output/regime_enhanced_features_v1",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INIT] Device: {DEVICE}")


# ─── DATA DOWNLOAD ──────────────────────────────────────────────────────────

def download_data(cfg):
    """Download daily OHLCV data for all tickers via yfinance."""
    print(f"[DATA] Downloading {len(cfg['tickers'])} tickers from {cfg['start_date']}...")
    end = cfg["end_date"] or dt.date.today().isoformat()

    data = {}
    for ticker in cfg["tickers"]:
        try:
            df = yf.download(ticker, start=cfg["start_date"], end=end, auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 100:
                data[ticker] = df
                print(f"  {ticker}: {len(df)} days")
            else:
                print(f"  {ticker}: SKIPPED (only {len(df)} days)")
        except Exception as e:
            print(f"  {ticker}: FAILED ({e})")

    return data


def _col(df, col):
    """Safely extract a single column as a Series."""
    s = df[col]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return s


# ─── FEATURE ENGINEERING ─────────────────────────────────────────────────────

def build_baseline_features(data, cfg):
    """Build the original v1 baseline features (~20 features)."""
    spy = _col(data["SPY"], "Close").copy()
    spy.name = "SPY_close"
    idx = spy.index
    features = pd.DataFrame(index=idx)

    # 1. VIX level (raw and log)
    if "^VIX" in data:
        vix = _col(data["^VIX"], "Close").reindex(idx).ffill()
        features["vix"] = vix
        features["vix_log"] = np.log(vix.clip(lower=1))

    # 2. VIX term structure ratio
    if "^VIX" in data and "^VIX3M" in data:
        vix3m = _col(data["^VIX3M"], "Close").reindex(idx).ffill()
        features["vix_term_ratio"] = (vix / vix3m.clip(lower=1)).clip(0.5, 2.0)

    # 3. SPY returns at multiple horizons
    spy_ret = spy.pct_change()
    for w in [5, 10, 21, 63]:
        features[f"spy_ret_{w}d"] = spy.pct_change(w)

    # 4. SPY distance from 200d SMA
    sma200 = spy.rolling(200).mean()
    features["spy_sma200_dist"] = (spy - sma200) / sma200

    # 5. SPY realized vol (trailing)
    features["spy_rvol_10d"] = spy_ret.rolling(10).std() * np.sqrt(252)
    features["spy_rvol_21d"] = spy_ret.rolling(21).std() * np.sqrt(252)

    # 6. Sector dispersion
    sector_tickers = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
    sector_rets = pd.DataFrame()
    for t in sector_tickers:
        if t in data:
            sector_rets[t] = _col(data[t], "Close").reindex(idx).ffill().pct_change()
    if len(sector_rets.columns) >= 5:
        features["sector_dispersion"] = sector_rets.std(axis=1)
        features["sector_dispersion_5d"] = features["sector_dispersion"].rolling(5).mean()
        ranks = sector_rets.rolling(5).mean().rank(axis=1)
        rank_changes = ranks.diff().abs().mean(axis=1)
        features["sector_rotation"] = rank_changes

    # 7. Credit spread proxy: HYG - TLT return differential
    if "HYG" in data and "TLT" in data:
        hyg_ret = _col(data["HYG"], "Close").reindex(idx).ffill().pct_change()
        tlt_ret = _col(data["TLT"], "Close").reindex(idx).ffill().pct_change()
        features["credit_spread_1d"] = hyg_ret - tlt_ret
        features["credit_spread_5d"] = features["credit_spread_1d"].rolling(5).mean()
        features["credit_spread_21d"] = features["credit_spread_1d"].rolling(21).mean()

    # 8. Gold momentum
    if "GLD" in data:
        gld = _col(data["GLD"], "Close").reindex(idx).ffill()
        features["gld_ret_5d"] = gld.pct_change(5)
        features["gld_ret_21d"] = gld.pct_change(21)

    # 9. SPY-TLT correlation (rolling 21d)
    if "TLT" in data:
        tlt_ret = _col(data["TLT"], "Close").reindex(idx).ffill().pct_change()
        features["spy_tlt_corr_21d"] = spy_ret.rolling(21).corr(tlt_ret)

    # 10. Intraday range proxy
    if "High" in data["SPY"].columns and "Low" in data["SPY"].columns:
        features["spy_range"] = (_col(data["SPY"], "High").reindex(idx) - _col(data["SPY"], "Low").reindex(idx)) / spy

    return features


def build_new_features(data, cfg):
    """Build the 10 NEW candidate features for testing."""
    spy = _col(data["SPY"], "Close").copy()
    idx = spy.index
    spy_ret = spy.pct_change()
    new_features = pd.DataFrame(index=idx)

    vix = _col(data["^VIX"], "Close").reindex(idx).ffill() if "^VIX" in data else None
    vix3m = _col(data["^VIX3M"], "Close").reindex(idx).ffill() if "^VIX3M" in data else None

    # 1. VIX_ratio (VIX/VIX3M — term structure, proven independent signal)
    #    NOTE: baseline already has vix_term_ratio, but this is the RAW ratio without clipping
    #    Actually the baseline clips to [0.5, 2.0]. We add the unclipped + momentum.
    if vix is not None and vix3m is not None:
        vix_ratio_raw = vix / vix3m.clip(lower=1)
        new_features["vix_ratio"] = vix_ratio_raw

    # 2. VIX_slope_change_5d (term structure momentum — how fast is the curve inverting/normalizing)
    if vix is not None and vix3m is not None:
        vix_ratio_raw = vix / vix3m.clip(lower=1)
        new_features["vix_slope_change_5d"] = vix_ratio_raw.diff(5)

    # 3. SPY-TLT rolling 21d correlation (flight-to-safety regime indicator)
    #    NOTE: baseline has spy_tlt_corr_21d already. We add the CHANGE in correlation.
    if "TLT" in data:
        tlt_ret = _col(data["TLT"], "Close").reindex(idx).ffill().pct_change()
        spy_tlt_corr = spy_ret.rolling(21).corr(tlt_ret)
        new_features["spy_tlt_corr_change_5d"] = spy_tlt_corr.diff(5)

    # 4. SPY-HYG rolling 21d correlation (credit stress indicator)
    if "HYG" in data:
        hyg_ret = _col(data["HYG"], "Close").reindex(idx).ffill().pct_change()
        new_features["spy_hyg_corr_21d"] = spy_ret.rolling(21).corr(hyg_ret)

    # 5. SPY-GLD rolling 21d correlation (safe haven demand)
    if "GLD" in data:
        gld_ret = _col(data["GLD"], "Close").reindex(idx).ffill().pct_change()
        new_features["spy_gld_corr_21d"] = spy_ret.rolling(21).corr(gld_ret)

    # 6. Cross-sector dispersion 21d (longer-term sector divergence — captures structural shifts)
    sector_tickers = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
    sector_rets = pd.DataFrame()
    for t in sector_tickers:
        if t in data:
            sector_rets[t] = _col(data[t], "Close").reindex(idx).ffill().pct_change()
    if len(sector_rets.columns) >= 5:
        new_features["sector_dispersion_21d"] = sector_rets.rolling(21).std().mean(axis=1)

    # 7. HYG-TLT spread 5d velocity (credit stress acceleration)
    if "HYG" in data and "TLT" in data:
        hyg = _col(data["HYG"], "Close").reindex(idx).ffill()
        tlt = _col(data["TLT"], "Close").reindex(idx).ffill()
        # Use log ratio as spread proxy
        spread = np.log(hyg / tlt)
        spread_change = spread.diff(5)
        new_features["hyg_tlt_spread_velocity_5d"] = spread_change

    # 8. GLD momentum 21d (gold as fear proxy — already in baseline as gld_ret_21d)
    #    We add a RELATIVE gold momentum: GLD vs SPY
    if "GLD" in data:
        gld = _col(data["GLD"], "Close").reindex(idx).ffill()
        gld_rel = (gld.pct_change(21)) - (spy.pct_change(21))
        new_features["gld_relative_momentum_21d"] = gld_rel

    # 9. Realized vol of vol (rolling stdev of 21d VIX changes — vol regime instability)
    if vix is not None:
        vix_changes = vix.pct_change()
        new_features["vol_of_vol_21d"] = vix_changes.rolling(21).std()

    # 10. Put-call ratio proxy via VIX momentum (5d change in VIX as sentiment proxy)
    #     Note: actual put-call ratio not in yfinance. VIX 5d change captures same sentiment shift.
    if vix is not None:
        new_features["vix_momentum_5d"] = vix.pct_change(5)

    return new_features


# ─── DATASET ──────────────────────────────────────────────────────────────────

class RegimeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_sequences(features, targets, lookback):
    X, y = [], []
    for i in range(lookback, len(features)):
        X.append(features[i - lookback:i])
        y.append(targets[i])
    return np.array(X), np.array(y)


# ─── MODEL (IDENTICAL TO v1) ─────────────────────────────────────────────────

class RegimeDetector(nn.Module):
    def __init__(self, n_features, hidden_size=64, num_layers=2, dropout=0.3,
                 model_type="GRU", bidirectional=False):
        super().__init__()
        self.model_type = model_type

        RNNClass = nn.GRU if model_type == "GRU" else nn.LSTM
        self.rnn = RNNClass(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        dir_mult = 2 if bidirectional else 1
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size * dir_mult),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * dir_mult, 32),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


# ─── TRAINING ─────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    n = 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
        n += len(y)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    preds, actuals = [], []
    n = 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        pred = model(X)
        loss = criterion(pred, y)
        total_loss += loss.item() * len(y)
        n += len(y)
        preds.append(pred.cpu().numpy())
        actuals.append(y.cpu().numpy())
    preds = np.concatenate(preds)
    actuals = np.concatenate(actuals)
    mae = mean_absolute_error(actuals, preds)
    r2 = r2_score(actuals, preds) if len(actuals) > 2 else 0
    corr = np.corrcoef(actuals, preds)[0, 1] if len(actuals) > 2 else 0
    return total_loss / max(n, 1), mae, r2, corr, preds, actuals


# ─── FEATURE IMPORTANCE (for variant C selection) ─────────────────────────────

def compute_permutation_importance(model, X_val, y_val, feat_cols, new_feat_names, lookback, criterion):
    """
    Permutation importance: for each new feature, shuffle it across all timesteps
    and measure degradation in validation loss.
    """
    model.eval()
    ds = RegimeDataset(X_val, y_val)
    loader = DataLoader(ds, batch_size=256, shuffle=False)
    base_loss, _, _, _, _, _ = evaluate(model, loader, criterion)

    importances = {}
    for feat_name in new_feat_names:
        if feat_name not in feat_cols:
            continue
        feat_idx = feat_cols.index(feat_name)

        # Shuffle this feature across samples
        X_perm = X_val.copy()
        perm_idx = np.random.permutation(len(X_perm))
        X_perm[:, :, feat_idx] = X_perm[perm_idx, :, feat_idx]

        ds_perm = RegimeDataset(X_perm, y_val)
        loader_perm = DataLoader(ds_perm, batch_size=256, shuffle=False)
        perm_loss, _, _, _, _, _ = evaluate(model, loader_perm, criterion)

        importances[feat_name] = perm_loss - base_loss  # Higher = more important

    return importances


# ─── WALK-FORWARD ENGINE ─────────────────────────────────────────────────────

def run_variant(variant_name, combined, feat_cols, cfg, run_name_suffix=""):
    """Run a single walk-forward variant."""
    output_dir = Path(cfg["output_dir"]) / variant_name
    output_dir.mkdir(parents=True, exist_ok=True)

    lookback = cfg["lookback"]
    train_days = cfg["train_days"]
    val_days = cfg["val_days"]
    step_days = cfg["step_days"]

    features_raw = combined[feat_cols].values
    targets_raw = combined["fwd_rvol_20d"].values
    dates = combined.index

    total_samples = len(features_raw)
    min_required = lookback + train_days + val_days
    if total_samples < min_required:
        raise ValueError(f"Need {min_required} samples, got {total_samples}")

    fold_starts = list(range(0, total_samples - min_required, step_days))
    n_folds = len(fold_starts)
    print(f"\n[{variant_name}] {n_folds} walk-forward folds, {len(feat_cols)} features, {total_samples} samples")
    print(f"  Features: {feat_cols}")

    all_oot_preds = []
    all_oot_actuals = []
    all_oot_dates = []
    fold_metrics = []

    # For variant B: accumulate importance across folds
    accumulated_importance = {} if variant_name == "variant_B" else None

    with mlflow.start_run(run_name=f"{variant_name}_{run_name_suffix}", nested=True) as run:
        mlflow.log_params({
            "variant": variant_name,
            "model_type": cfg["model_type"],
            "hidden_size": cfg["hidden_size"],
            "num_layers": cfg["num_layers"],
            "dropout": cfg["dropout"],
            "lookback": lookback,
            "train_days": train_days,
            "val_days": val_days,
            "step_days": step_days,
            "n_features": len(feat_cols),
            "n_folds": n_folds,
            "epochs": cfg["epochs"],
            "batch_size": cfg["batch_size"],
            "lr": cfg["lr"],
            "features": str(feat_cols[:50]),  # Truncate for MLflow
        })

        for fold_idx, fold_start in enumerate(fold_starts):
            train_end = fold_start + train_days
            val_end = train_end + val_days

            if val_end > total_samples:
                break

            # SLIDING window
            train_feat = features_raw[fold_start:train_end]
            train_tgt = targets_raw[fold_start:train_end]
            val_feat = features_raw[train_end:val_end]
            val_tgt = targets_raw[train_end:val_end]

            # Scale features (fit on train only)
            scaler = RobustScaler()
            train_feat_scaled = scaler.fit_transform(train_feat)
            val_feat_scaled = scaler.transform(val_feat)

            # Scale targets
            tgt_mean = train_tgt.mean()
            tgt_std = train_tgt.std() + 1e-8
            train_tgt_norm = (train_tgt - tgt_mean) / tgt_std
            val_tgt_norm = (val_tgt - tgt_mean) / tgt_std

            # Create sequences
            X_train, y_train = create_sequences(train_feat_scaled, train_tgt_norm, lookback)
            X_val, y_val = create_sequences(
                np.vstack([train_feat_scaled[-lookback:], val_feat_scaled]),
                np.concatenate([train_tgt_norm[-lookback:], val_tgt_norm]),
                lookback
            )
            X_val = X_val[lookback:]
            y_val = y_val[lookback:]

            if len(X_train) < 50 or len(X_val) < 10:
                continue

            train_ds = RegimeDataset(X_train, y_train)
            val_ds = RegimeDataset(X_val, y_val)
            train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                                      num_workers=4, pin_memory=True)
            val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                                    num_workers=4, pin_memory=True)

            # Model
            model = RegimeDetector(
                n_features=len(feat_cols),
                hidden_size=cfg["hidden_size"],
                num_layers=cfg["num_layers"],
                dropout=cfg["dropout"],
                model_type=cfg["model_type"],
                bidirectional=cfg["bidirectional"],
            ).to(DEVICE)

            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                          weight_decay=cfg["weight_decay"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg["epochs"], eta_min=cfg["min_lr"]
            )
            criterion = nn.HuberLoss(delta=1.0)

            best_val_loss = float("inf")
            patience_counter = 0
            best_state = None

            for epoch in range(cfg["epochs"]):
                train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
                val_loss, val_mae, val_r2, val_corr, _, _ = evaluate(model, val_loader, criterion)
                scheduler.step()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= cfg["patience"]:
                        break

            # Load best model
            model.load_state_dict(best_state)
            val_loss, val_mae, val_r2, val_corr, val_preds_norm, val_actuals_norm = evaluate(
                model, val_loader, criterion
            )

            # Inverse normalize
            val_preds = val_preds_norm * tgt_std + tgt_mean
            val_actuals = val_actuals_norm * tgt_std + tgt_mean

            val_mae_raw = mean_absolute_error(val_actuals, val_preds)
            val_r2_raw = r2_score(val_actuals, val_preds) if len(val_actuals) > 2 else 0
            val_corr_raw = np.corrcoef(val_actuals, val_preds)[0, 1] if len(val_actuals) > 2 else 0

            # OOT dates
            val_date_start_idx = train_end + lookback
            val_date_end_idx = val_end
            if val_date_end_idx <= len(dates):
                oot_dates = dates[val_date_start_idx:val_date_end_idx]
                min_len = min(len(val_preds), len(oot_dates))
                all_oot_preds.extend(val_preds[:min_len].tolist())
                all_oot_actuals.extend(val_actuals[:min_len].tolist())
                all_oot_dates.extend(oot_dates[:min_len].tolist())

            fold_metrics.append({
                "fold": fold_idx,
                "train_start": str(dates[fold_start].date()),
                "val_start": str(dates[train_end].date()),
                "val_mae": val_mae_raw,
                "val_r2": val_r2_raw,
                "val_corr": val_corr_raw,
                "best_epoch": cfg["epochs"] - patience_counter,
                "n_val": len(X_val),
            })

            # Permutation importance for variant B (every 20th fold to save time)
            if variant_name == "variant_B" and fold_idx % 20 == 0:
                new_feat_names = [f for f in feat_cols if f.startswith(("vix_ratio", "vix_slope",
                    "spy_tlt_corr_change", "spy_hyg_corr", "spy_gld_corr",
                    "sector_dispersion_21d", "hyg_tlt_spread", "gld_relative",
                    "vol_of_vol", "vix_momentum"))]
                imp = compute_permutation_importance(
                    model, X_val, y_val, feat_cols, new_feat_names, lookback, criterion
                )
                for k, v in imp.items():
                    accumulated_importance[k] = accumulated_importance.get(k, []) + [v]

            if fold_idx % 10 == 0 or fold_idx == n_folds - 1:
                print(f"  Fold {fold_idx}/{n_folds}: MAE={val_mae_raw:.4f}, R2={val_r2_raw:.3f}, "
                      f"Corr={val_corr_raw:.3f}, Ep={cfg['epochs'] - patience_counter}")

        # ─── AGGREGATE METRICS ────────────────────────────────────────────
        all_oot_preds = np.array(all_oot_preds)
        all_oot_actuals = np.array(all_oot_actuals)

        results = {}
        if len(all_oot_preds) > 10:
            concat_mae = mean_absolute_error(all_oot_actuals, all_oot_preds)
            concat_r2 = r2_score(all_oot_actuals, all_oot_preds)
            concat_corr = np.corrcoef(all_oot_actuals, all_oot_preds)[0, 1]

            # Regime classification
            threshold = 0.20
            pred_high = all_oot_preds > threshold
            actual_high = all_oot_actuals > threshold
            regime_accuracy = (pred_high == actual_high).mean()

            # Transition detection (±5 day window)
            actual_transitions = np.diff(actual_high.astype(int)) != 0
            pred_transitions = np.diff(pred_high.astype(int)) != 0
            if actual_transitions.sum() > 0:
                transition_recall = 0
                actual_trans_idx = np.where(actual_transitions)[0]
                pred_trans_idx = np.where(pred_transitions)[0]
                for at in actual_trans_idx:
                    if any(abs(at - pt) <= 5 for pt in pred_trans_idx):
                        transition_recall += 1
                transition_recall /= len(actual_trans_idx)
            else:
                transition_recall = np.nan

            # Transition precision: of predicted transitions, how many were real?
            if pred_transitions.sum() > 0:
                transition_precision = 0
                for pt in pred_trans_idx:
                    if any(abs(at - pt) <= 5 for at in actual_trans_idx):
                        transition_precision += 1
                transition_precision /= len(pred_trans_idx)
            else:
                transition_precision = np.nan

            print(f"\n{'='*60}")
            print(f"[{variant_name}] AGGREGATE OOT RESULTS ({len(all_oot_preds)} predictions)")
            print(f"{'='*60}")
            print(f"  MAE:                {concat_mae:.4f}")
            print(f"  R2:                 {concat_r2:.4f}")
            print(f"  Correlation:        {concat_corr:.4f}")
            print(f"  Regime Accuracy:    {regime_accuracy:.3f}")
            print(f"  Transition Recall:  {transition_recall:.3f}")
            print(f"  Transition Precision: {transition_precision:.3f}")

            # Quintile analysis
            print(f"\n  Quintile Analysis:")
            quintiles = pd.qcut(all_oot_preds, 5, labels=False, duplicates="drop")
            for q in sorted(np.unique(quintiles)):
                mask = quintiles == q
                print(f"    Q{q}: pred={all_oot_preds[mask].mean():.4f}, "
                      f"actual={all_oot_actuals[mask].mean():.4f}, n={mask.sum()}")

            results = {
                "concat_mae": concat_mae,
                "concat_r2": concat_r2,
                "concat_corr": concat_corr,
                "regime_accuracy": regime_accuracy,
                "transition_recall": float(transition_recall) if not np.isnan(transition_recall) else 0,
                "transition_precision": float(transition_precision) if not np.isnan(transition_precision) else 0,
                "n_predictions": len(all_oot_preds),
                "n_folds": len(fold_metrics),
            }

            mlflow.log_metrics({f"{variant_name}_{k}": v for k, v in results.items()})

            # Save fold metrics
            fold_df = pd.DataFrame(fold_metrics)
            fold_csv = output_dir / "fold_metrics.csv"
            fold_df.to_csv(fold_csv, index=False)
            mlflow.log_metrics({
                f"{variant_name}_mean_fold_r2": fold_df["val_r2"].mean(),
                f"{variant_name}_mean_fold_corr": fold_df["val_corr"].mean(),
                f"{variant_name}_std_fold_corr": fold_df["val_corr"].std(),
            })

        # Save predictions
        date_strings = [str(d.date()) if hasattr(d, 'date') else str(d) for d in all_oot_dates]
        npz_path = output_dir / f"predictions_{variant_name}.npz"
        np.savez(
            npz_path,
            predictions=all_oot_preds,
            actuals=all_oot_actuals,
            dates=np.array(date_strings),
            threshold=0.20,
            regime_scores=np.clip(all_oot_preds / 0.40, 0, 1),
        )
        print(f"[SAVED] {npz_path}")

        try:
            mlflow.log_artifact(str(npz_path))
            mlflow.log_artifact(str(fold_csv))
        except Exception as e:
            print(f"[WARN] MLflow artifact upload: {e}")

    return results, accumulated_importance


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    ts = dt.datetime.now().strftime('%Y%m%d_%H%M')
    print(f"{'='*70}")
    print(f"REGIME ENHANCED FEATURES v1 — Feature Ablation Study")
    print(f"Architecture: GRU 2-layer 64-hidden 60-day lookback (FROZEN from v1)")
    print(f"{'='*70}")
    print(f"Started: {dt.datetime.now().isoformat()}")

    # 1. Download data
    data = download_data(CFG)
    if "SPY" not in data or "^VIX" not in data:
        print("[FATAL] Missing SPY or VIX data. Aborting.")
        sys.exit(1)

    # 2. Build feature sets
    baseline_features = build_baseline_features(data, CFG)
    new_features = build_new_features(data, CFG)
    print(f"\n[FEATURES] Baseline: {len(baseline_features.columns)} features")
    print(f"[FEATURES] New candidates: {len(new_features.columns)} features")
    print(f"  New: {list(new_features.columns)}")

    # 3. Build target
    spy = _col(data["SPY"], "Close").copy()
    spy_ret = spy.pct_change()
    fwd_ret = spy_ret.shift(-1)
    target = fwd_ret.rolling(CFG["fwd_vol_window"]).std().shift(-(CFG["fwd_vol_window"] - 1)) * np.sqrt(252)
    target.name = "fwd_rvol_20d"

    # MLflow parent run
    mlflow.set_tracking_uri(CFG["mlflow_uri"])
    mlflow.set_experiment(CFG["experiment_name"])

    with mlflow.start_run(run_name=f"enhanced_features_ablation_{ts}") as parent_run:
        mlflow.log_param("study_type", "feature_ablation")
        mlflow.log_param("architecture", "GRU_2layer_64hidden_frozen")

        # ─── VARIANT A: Baseline (original ~20 features) ─────────────
        print(f"\n{'='*70}")
        print(f"VARIANT A: Baseline (original features)")
        print(f"{'='*70}")
        combined_a = pd.concat([baseline_features, target], axis=1).dropna()
        feat_cols_a = [c for c in combined_a.columns if c != "fwd_rvol_20d"]
        print(f"  {len(feat_cols_a)} features, {len(combined_a)} samples")

        results_a, _ = run_variant("variant_A", combined_a, feat_cols_a, CFG, ts)

        # ─── VARIANT B: Baseline + all 10 new features ───────────────
        print(f"\n{'='*70}")
        print(f"VARIANT B: Baseline + 10 new features")
        print(f"{'='*70}")
        all_features = pd.concat([baseline_features, new_features], axis=1)
        combined_b = pd.concat([all_features, target], axis=1).dropna()
        feat_cols_b = [c for c in combined_b.columns if c != "fwd_rvol_20d"]
        print(f"  {len(feat_cols_b)} features, {len(combined_b)} samples")

        results_b, importance_b = run_variant("variant_B", combined_b, feat_cols_b, CFG, ts)

        # ─── Select top 5 features from importance ────────────────────
        if importance_b:
            avg_importance = {k: np.mean(v) for k, v in importance_b.items()}
            sorted_imp = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
            print(f"\n[IMPORTANCE] New feature importance (permutation, higher = better):")
            for feat, imp in sorted_imp:
                print(f"  {feat}: {imp:.6f}")
            top5_features = [feat for feat, _ in sorted_imp[:5]]
            print(f"\n[SELECTED] Top 5 new features for variant C: {top5_features}")

            # Log importance to MLflow
            for feat, imp in sorted_imp:
                mlflow.log_metric(f"importance_{feat}", imp)
        else:
            # Fallback: use features with strongest theoretical priors
            top5_features = [
                "vix_ratio", "vix_slope_change_5d", "spy_hyg_corr_21d",
                "vol_of_vol_21d", "hyg_tlt_spread_velocity_5d"
            ]
            print(f"\n[FALLBACK] Using theory-based top 5: {top5_features}")

        # ─── VARIANT C: Baseline + best 5 new features ───────────────
        print(f"\n{'='*70}")
        print(f"VARIANT C: Baseline + best 5 new features")
        print(f"{'='*70}")
        # Filter to only top 5 new features that exist
        available_top5 = [f for f in top5_features if f in new_features.columns]
        selected_new = new_features[available_top5]
        combined_c_feats = pd.concat([baseline_features, selected_new], axis=1)
        combined_c = pd.concat([combined_c_feats, target], axis=1).dropna()
        feat_cols_c = [c for c in combined_c.columns if c != "fwd_rvol_20d"]
        print(f"  {len(feat_cols_c)} features, {len(combined_c)} samples")
        print(f"  Selected new features: {available_top5}")

        results_c, _ = run_variant("variant_C", combined_c, feat_cols_c, CFG, ts)

        # ─── COMPARISON TABLE ─────────────────────────────────────────
        print(f"\n{'='*70}")
        print(f"FINAL COMPARISON")
        print(f"{'='*70}")
        print(f"{'Metric':<25} {'A (Baseline)':>14} {'B (+10 feat)':>14} {'C (+5 best)':>14}")
        print(f"{'-'*70}")
        for metric in ["concat_r2", "concat_corr", "concat_mae", "regime_accuracy",
                       "transition_recall", "transition_precision"]:
            va = results_a.get(metric, float('nan'))
            vb = results_b.get(metric, float('nan'))
            vc = results_c.get(metric, float('nan'))
            print(f"  {metric:<23} {va:>14.4f} {vb:>14.4f} {vc:>14.4f}")

        # Determine winner
        best_variant = "A"
        best_score = results_a.get("transition_recall", 0) + results_a.get("concat_r2", 0)
        for name, res in [("B", results_b), ("C", results_c)]:
            score = res.get("transition_recall", 0) + res.get("concat_r2", 0)
            if score > best_score:
                best_score = score
                best_variant = name

        print(f"\n  WINNER: Variant {best_variant} (R2 + transition_recall = {best_score:.4f})")

        mlflow.log_param("winner", best_variant)
        mlflow.log_metric("best_combined_score", best_score)
        if available_top5:
            mlflow.log_param("top5_new_features", str(available_top5))

    elapsed = time.time() - t0
    print(f"\n[DONE] Total time: {elapsed/60:.1f} minutes ({elapsed/3600:.1f} hours)")
    print(f"Completed: {dt.datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
