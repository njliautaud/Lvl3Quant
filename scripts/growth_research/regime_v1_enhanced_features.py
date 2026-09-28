#!/usr/bin/env python3
"""
Regime Detector v1 — Enhanced Features Experiment

Tests whether BETTER FEATURES (not better architecture) improve the GRU regime model.
Validated that architecture is near-optimal. Now testing cross-asset features.

Three variants:
  A. v1 baseline (20 features) — for fair comparison on same run
  B. v1 + 5 cross-asset correlation features (25 total)
  C. v1 + 3 VIX term structure features (23 total)

Same architecture as v1: 2-layer GRU, 64 hidden, 60d lookback
Walk-forward: 1000d train, 250d val, 20d step, sliding window

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
    "tickers": [
        "SPY", "^VIX", "^VIX3M",
        "TLT", "SHY", "HYG", "GLD",
        "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC",
    ],
    "start_date": "2007-01-01",
    "end_date": None,

    # Sequence
    "lookback": 60,

    # Target
    "fwd_vol_window": 21,  # 21d forward realized vol

    # Model (same as v1 — architecture is near-optimal)
    "model_type": "GRU",
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.3,
    "bidirectional": False,

    # Walk-forward (sliding)
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
    "experiment_name": "regime_v1_enhanced_features",

    # Output
    "output_base": "/home/nick/Lvl3Quant/output/regime_v1_enhanced_features",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INIT] Device: {DEVICE}")


# ─── DATA ─────────────────────────────────────────────────────────────────────

def download_data(cfg):
    """Download daily OHLCV data for all tickers."""
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
    s = df[col]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return s


# ─── FEATURE ENGINEERING ─────────────────────────────────────────────────────

def build_base_features(data):
    """Build the 20 v1 baseline features."""
    spy = _col(data["SPY"], "Close").copy()
    spy.name = "SPY_close"
    idx = spy.index
    features = pd.DataFrame(index=idx)
    spy_ret = spy.pct_change()

    # 1. VIX level + changes
    vix = _col(data["^VIX"], "Close").reindex(idx).ffill()
    features["vix"] = vix
    features["vix_5d_chg"] = vix.pct_change(5)
    features["vix_10d_chg"] = vix.pct_change(10)
    features["vix_21d_chg"] = vix.pct_change(21)

    # 2. SPY returns
    features["spy_ret_5d"] = spy.pct_change(5)
    features["spy_ret_10d"] = spy.pct_change(10)
    features["spy_ret_21d"] = spy.pct_change(21)

    # 3. SPY realized vol
    features["spy_vol_21d"] = spy_ret.rolling(21).std() * np.sqrt(252)
    features["spy_vol_63d"] = spy_ret.rolling(63).std() * np.sqrt(252)

    # 4. Yield curve proxy (TLT/SHY ratio)
    if "TLT" in data and "SHY" in data:
        tlt = _col(data["TLT"], "Close").reindex(idx).ffill()
        shy = _col(data["SHY"], "Close").reindex(idx).ffill()
        features["yield_curve_ratio"] = tlt / shy.clip(lower=1)

    # 5. Credit spread proxy (HYG/TLT ratio)
    if "HYG" in data and "TLT" in data:
        hyg = _col(data["HYG"], "Close").reindex(idx).ffill()
        tlt = _col(data["TLT"], "Close").reindex(idx).ffill()
        features["credit_spread_ratio"] = hyg / tlt.clip(lower=1)

    # 6. Gold momentum
    if "GLD" in data:
        gld = _col(data["GLD"], "Close").reindex(idx).ffill()
        features["gld_momentum"] = gld.pct_change(21)

    # 7. Sector dispersion
    sector_tickers = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
    sector_rets = pd.DataFrame()
    for t in sector_tickers:
        if t in data:
            sector_rets[t] = _col(data[t], "Close").reindex(idx).ffill().pct_change()
    if len(sector_rets.columns) >= 5:
        features["sector_dispersion"] = sector_rets.std(axis=1)

    # 8. Put-call ratio proxy (VIX skew — vix level relative to realized vol)
    features["put_call_proxy"] = vix / (features["spy_vol_21d"].clip(lower=0.01) * 100)

    # 9. HYG ret 5d
    if "HYG" in data:
        hyg_close = _col(data["HYG"], "Close").reindex(idx).ffill()
        features["hyg_ret_5d"] = hyg_close.pct_change(5)

    # 10. SPY-TLT correlation 21d
    if "TLT" in data:
        tlt_ret = _col(data["TLT"], "Close").reindex(idx).ffill().pct_change()
        features["spy_tlt_corr_21d"] = spy_ret.rolling(21).corr(tlt_ret)

    # 11. Skew proxy (absolute returns / vol — tail thickness)
    features["skew_proxy"] = spy_ret.abs().rolling(21).mean() / (features["spy_vol_21d"] / np.sqrt(252)).clip(lower=1e-6)

    # 12. Momentum breadth (% of sectors with positive 21d returns)
    if len(sector_rets.columns) >= 5:
        sector_mom_21d = pd.DataFrame()
        for t in sector_tickers:
            if t in data:
                sector_mom_21d[t] = _col(data[t], "Close").reindex(idx).ffill().pct_change(21)
        features["momentum_breadth"] = (sector_mom_21d > 0).mean(axis=1)

    # 13. Vol of vol (21d rolling std of 5d vol changes)
    vol_5d = spy_ret.rolling(5).std()
    features["vol_of_vol"] = vol_5d.pct_change(5).rolling(21).std()

    return features, spy_ret, idx


def build_variant_a(data):
    """Variant A: v1 baseline (20 features)."""
    features, _, _ = build_base_features(data)
    return features


def build_variant_b(data):
    """Variant B: v1 + 5 cross-asset correlation features (25 total)."""
    features, spy_ret, idx = build_base_features(data)

    # Cross-asset correlations
    if "TLT" in data:
        tlt_ret = _col(data["TLT"], "Close").reindex(idx).ffill().pct_change()
        # SPY-TLT corr already in base, but we ensure it's there
        if "spy_tlt_corr_21d" not in features.columns:
            features["spy_tlt_corr_21d"] = spy_ret.rolling(21).corr(tlt_ret)

    if "HYG" in data:
        hyg_ret = _col(data["HYG"], "Close").reindex(idx).ffill().pct_change()
        features["spy_hyg_corr_21d"] = spy_ret.rolling(21).corr(hyg_ret)

    if "GLD" in data:
        gld_ret = _col(data["GLD"], "Close").reindex(idx).ffill().pct_change()
        features["spy_gld_corr_21d"] = spy_ret.rolling(21).corr(gld_ret)

    # TLT-HYG spread change 5d (credit stress velocity)
    if "TLT" in data and "HYG" in data:
        tlt = _col(data["TLT"], "Close").reindex(idx).ffill()
        hyg = _col(data["HYG"], "Close").reindex(idx).ffill()
        spread = (tlt / tlt.iloc[0]) - (hyg / hyg.iloc[0])  # normalized spread
        features["tlt_hyg_spread_chg_5d"] = spread.diff(5)

    # GLD momentum 21d (may overlap with gld_momentum — kept separate for clarity)
    if "GLD" in data:
        gld = _col(data["GLD"], "Close").reindex(idx).ffill()
        features["gld_momentum_21d"] = gld.pct_change(21)

    return features


def build_variant_c(data):
    """Variant C: v1 + 3 VIX term structure features (23 total)."""
    features, _, idx = build_base_features(data)

    vix = _col(data["^VIX"], "Close").reindex(idx).ffill()

    if "^VIX3M" in data:
        vix3m = _col(data["^VIX3M"], "Close").reindex(idx).ffill()
        # VIX/VIX3M ratio (< 1 = contango/calm, > 1 = backwardation/stress)
        features["vix_ratio"] = (vix / vix3m.clip(lower=1)).clip(0.5, 2.0)
        # VIX slope (simple difference, annualized units)
        features["vix_slope"] = vix - vix3m
        # VIX slope change 5d (acceleration of term structure shift)
        features["vix_slope_chg_5d"] = features["vix_slope"].diff(5)
    else:
        print("[WARN] ^VIX3M not available — Variant C will use NaN-filled term structure features")
        features["vix_ratio"] = np.nan
        features["vix_slope"] = np.nan
        features["vix_slope_chg_5d"] = np.nan

    return features


VARIANT_BUILDERS = {
    "A_baseline": build_variant_a,
    "B_cross_asset": build_variant_b,
    "C_vix_term": build_variant_c,
}


# ─── DATASET & MODEL (same as v1) ────────────────────────────────────────────

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
    total_loss, n = 0, 0
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
    total_loss, n = 0, 0
    preds, actuals = [], []
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


# ─── WALK-FORWARD ─────────────────────────────────────────────────────────────

def compute_transition_metrics(all_preds, all_actuals, threshold=0.20):
    """Compute transition recall and precision."""
    pred_high = all_preds > threshold
    actual_high = all_actuals > threshold

    actual_transitions = np.diff(actual_high.astype(int)) != 0
    pred_transitions = np.diff(pred_high.astype(int)) != 0

    # Recall: of actual transitions, how many predicted within ±5 days?
    actual_trans_idx = np.where(actual_transitions)[0]
    pred_trans_idx = np.where(pred_transitions)[0]

    transition_recall = 0.0
    transition_precision = 0.0

    if len(actual_trans_idx) > 0 and len(pred_trans_idx) > 0:
        hits = sum(1 for at in actual_trans_idx if any(abs(at - pt) <= 5 for pt in pred_trans_idx))
        transition_recall = hits / len(actual_trans_idx)

        # Precision: of predicted transitions, how many matched actual within ±5 days?
        prec_hits = sum(1 for pt in pred_trans_idx if any(abs(at - pt) <= 5 for at in actual_trans_idx))
        transition_precision = prec_hits / len(pred_trans_idx)
    elif len(actual_trans_idx) == 0:
        transition_recall = np.nan
        transition_precision = np.nan

    return transition_recall, transition_precision


def compute_quintile_metrics(all_preds, all_actuals):
    """Quintile monotonicity: Q0→Q4 should map to increasing actual vol."""
    quintiles = pd.qcut(all_preds, 5, labels=False, duplicates="drop")
    q_means = []
    q_report = []
    for q in sorted(np.unique(quintiles)):
        mask = quintiles == q
        q_actual_mean = all_actuals[mask].mean()
        q_pred_mean = all_preds[mask].mean()
        q_means.append(q_actual_mean)
        q_report.append({
            "quintile": int(q),
            "pred_mean": float(q_pred_mean),
            "actual_mean": float(q_actual_mean),
            "n": int(mask.sum()),
        })

    # Monotonicity: Spearman rank correlation of quintile actual means
    if len(q_means) >= 3:
        from scipy.stats import spearmanr
        mono_corr, _ = spearmanr(range(len(q_means)), q_means)
    else:
        mono_corr = np.nan

    return mono_corr, q_report


def run_variant(variant_name, features_df, data, cfg):
    """Run walk-forward for a single variant."""
    print(f"\n{'='*70}")
    print(f"  VARIANT {variant_name}: {len(features_df.columns)} features")
    print(f"{'='*70}")

    spy = _col(data["SPY"], "Close").copy()
    spy_ret = spy.pct_change()

    # Target: 21d forward realized vol (annualized)
    fwd_ret = spy_ret.shift(-1)
    target = fwd_ret.rolling(cfg["fwd_vol_window"]).std().shift(-(cfg["fwd_vol_window"] - 1)) * np.sqrt(252)
    target.name = "fwd_rvol_21d"

    combined = pd.concat([features_df, target], axis=1).dropna()
    feat_cols = [c for c in combined.columns if c != "fwd_rvol_21d"]

    print(f"[FEATURES] {len(feat_cols)} features, {len(combined)} valid samples")
    print(f"  Features: {feat_cols}")

    output_dir = Path(cfg["output_base"]) / variant_name
    output_dir.mkdir(parents=True, exist_ok=True)

    lookback = cfg["lookback"]
    train_days = cfg["train_days"]
    val_days = cfg["val_days"]
    step_days = cfg["step_days"]

    features_raw = combined[feat_cols].values
    targets_raw = combined["fwd_rvol_21d"].values
    dates = combined.index

    total_samples = len(features_raw)
    min_required = lookback + train_days + val_days
    if total_samples < min_required:
        print(f"[ERROR] Need {min_required} samples, got {total_samples}. Skipping.")
        return None

    fold_starts = list(range(0, total_samples - min_required, step_days))
    n_folds = len(fold_starts)
    print(f"[WF] {n_folds} walk-forward folds")

    all_oot_preds = []
    all_oot_actuals = []
    all_oot_dates = []
    fold_metrics = []

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

        # Scale
        scaler = RobustScaler()
        train_feat_scaled = scaler.fit_transform(train_feat)
        val_feat_scaled = scaler.transform(val_feat)

        tgt_mean = train_tgt.mean()
        tgt_std = train_tgt.std() + 1e-8
        train_tgt_norm = (train_tgt - tgt_mean) / tgt_std
        val_tgt_norm = (val_tgt - tgt_mean) / tgt_std

        # Sequences
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
            val_loss, _, _, _, _, _ = evaluate(model, val_loader, criterion)
            scheduler.step()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= cfg["patience"]:
                    break

        # Final evaluation
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
            "train_end": str(dates[train_end - 1].date()),
            "val_start": str(dates[train_end].date()),
            "val_end": str(dates[min(val_end - 1, len(dates) - 1)].date()),
            "val_mae": val_mae_raw,
            "val_r2": val_r2_raw,
            "val_corr": val_corr_raw,
            "best_epoch": cfg["epochs"] - patience_counter,
            "n_train": len(X_train),
            "n_val": len(X_val),
        })

        if fold_idx % 20 == 0 or fold_idx == n_folds - 1:
            print(f"  Fold {fold_idx}/{n_folds}: MAE={val_mae_raw:.4f}, R2={val_r2_raw:.3f}, "
                  f"Corr={val_corr_raw:.3f}, Epochs={cfg['epochs'] - patience_counter}")

        # Checkpoint every 50 folds
        if fold_idx % 50 == 0 and best_state:
            torch.save({
                "model_state": best_state,
                "scaler_center": scaler.center_,
                "scaler_scale": scaler.scale_,
                "tgt_mean": tgt_mean,
                "tgt_std": tgt_std,
                "fold": fold_idx,
            }, output_dir / f"fold_{fold_idx:04d}.pt")

    # ─── AGGREGATE RESULTS ────────────────────────────────────────────────
    all_oot_preds = np.array(all_oot_preds)
    all_oot_actuals = np.array(all_oot_actuals)

    if len(all_oot_preds) < 10:
        print(f"[ERROR] Too few OOT predictions ({len(all_oot_preds)}). Variant failed.")
        return None

    concat_mae = mean_absolute_error(all_oot_actuals, all_oot_preds)
    concat_r2 = r2_score(all_oot_actuals, all_oot_preds)
    concat_corr = np.corrcoef(all_oot_actuals, all_oot_preds)[0, 1]

    # Regime classification
    threshold = 0.20
    pred_high = all_oot_preds > threshold
    actual_high = all_oot_actuals > threshold
    regime_accuracy = (pred_high == actual_high).mean()

    # Transition metrics
    transition_recall, transition_precision = compute_transition_metrics(all_oot_preds, all_oot_actuals, threshold)

    # Quintile monotonicity
    mono_corr, q_report = compute_quintile_metrics(all_oot_preds, all_oot_actuals)

    # Fold stats
    fold_df = pd.DataFrame(fold_metrics)

    results = {
        "variant": variant_name,
        "n_features": len(feat_cols),
        "feature_names": feat_cols,
        "n_folds": len(fold_metrics),
        "n_oot_predictions": len(all_oot_preds),
        "concat_mae": concat_mae,
        "concat_r2": concat_r2,
        "concat_corr": concat_corr,
        "regime_accuracy": regime_accuracy,
        "transition_recall": float(transition_recall) if not np.isnan(transition_recall) else 0.0,
        "transition_precision": float(transition_precision) if not np.isnan(transition_precision) else 0.0,
        "quintile_monotonicity": float(mono_corr) if not np.isnan(mono_corr) else 0.0,
        "quintile_detail": q_report,
        "mean_fold_mae": fold_df["val_mae"].mean(),
        "mean_fold_r2": fold_df["val_r2"].mean(),
        "mean_fold_corr": fold_df["val_corr"].mean(),
        "std_fold_corr": fold_df["val_corr"].std(),
    }

    # Print results
    print(f"\n{'─'*50}")
    print(f"  {variant_name} RESULTS ({len(all_oot_preds)} OOT predictions)")
    print(f"{'─'*50}")
    print(f"  Concat MAE:             {concat_mae:.4f}")
    print(f"  Concat R²:              {concat_r2:.4f}")
    print(f"  Concat Correlation:     {concat_corr:.4f}")
    print(f"  Regime Accuracy (20%):  {regime_accuracy:.3f}")
    print(f"  Transition Recall:      {transition_recall:.3f}")
    print(f"  Transition Precision:   {transition_precision:.3f}")
    print(f"  Quintile Monotonicity:  {mono_corr:.3f}")
    print(f"  Mean Fold Corr:         {fold_df['val_corr'].mean():.3f} ± {fold_df['val_corr'].std():.3f}")
    print(f"\n  Quintile breakdown:")
    for q in q_report:
        print(f"    Q{q['quintile']}: pred={q['pred_mean']:.4f}, actual={q['actual_mean']:.4f}, n={q['n']}")

    # Save predictions
    date_strings = [str(d.date()) if hasattr(d, 'date') else str(d) for d in all_oot_dates]
    npz_path = output_dir / f"predictions_{variant_name}.npz"
    np.savez(
        npz_path,
        predictions=all_oot_preds,
        actuals=all_oot_actuals,
        dates=np.array(date_strings),
        regime_scores=np.clip(all_oot_preds / 0.40, 0, 1),
    )
    print(f"  Saved predictions: {npz_path}")

    # Save fold metrics
    fold_df.to_csv(output_dir / "fold_metrics.csv", index=False)

    # Save final model
    if best_state:
        torch.save({
            "model_state": best_state,
            "scaler_center": scaler.center_,
            "scaler_scale": scaler.scale_,
            "tgt_mean": tgt_mean,
            "tgt_std": tgt_std,
            "feat_cols": feat_cols,
            "cfg": cfg,
            "results": {k: v for k, v in results.items() if k != "quintile_detail"},
        }, output_dir / f"model_final_{variant_name}.pt")

    return results


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print(f"{'='*70}")
    print(f"REGIME v1 ENHANCED FEATURES EXPERIMENT")
    print(f"Hypothesis: Better features > better architecture")
    print(f"{'='*70}")
    print(f"Started: {dt.datetime.now().isoformat()}")
    print(f"Device: {DEVICE}")

    # 1. Download data (shared across all variants)
    data = download_data(CFG)
    missing = [t for t in ["SPY", "^VIX"] if t not in data]
    if missing:
        print(f"[FATAL] Missing critical tickers: {missing}")
        sys.exit(1)

    # 2. MLflow setup
    mlflow.set_tracking_uri(CFG["mlflow_uri"])
    mlflow.set_experiment(CFG["experiment_name"])

    # 3. Run all variants
    all_results = {}

    with mlflow.start_run(run_name=f"enhanced_features_{dt.datetime.now().strftime('%Y%m%d_%H%M')}") as parent_run:
        mlflow.log_params({
            "model_type": CFG["model_type"],
            "hidden_size": CFG["hidden_size"],
            "num_layers": CFG["num_layers"],
            "lookback": CFG["lookback"],
            "train_days": CFG["train_days"],
            "val_days": CFG["val_days"],
            "step_days": CFG["step_days"],
            "fwd_vol_window": CFG["fwd_vol_window"],
        })

        for variant_name, builder_fn in VARIANT_BUILDERS.items():
            variant_t0 = time.time()
            print(f"\n\n{'#'*70}")
            print(f"# Starting variant: {variant_name}")
            print(f"{'#'*70}")

            with mlflow.start_run(run_name=variant_name, nested=True) as child_run:
                features_df = builder_fn(data)
                results = run_variant(variant_name, features_df, data, CFG)

                if results:
                    all_results[variant_name] = results

                    # Log to MLflow
                    mlflow.log_params({
                        "variant": variant_name,
                        "n_features": results["n_features"],
                    })
                    mlflow.log_metrics({
                        "concat_mae": results["concat_mae"],
                        "concat_r2": results["concat_r2"],
                        "concat_corr": results["concat_corr"],
                        "regime_accuracy": results["regime_accuracy"],
                        "transition_recall": results["transition_recall"],
                        "transition_precision": results["transition_precision"],
                        "quintile_monotonicity": results["quintile_monotonicity"],
                        "mean_fold_corr": results["mean_fold_corr"],
                        "std_fold_corr": results["std_fold_corr"],
                        "n_folds": results["n_folds"],
                    })

                    # Log artifacts
                    variant_dir = Path(CFG["output_base"]) / variant_name
                    for f in variant_dir.glob("*.npz"):
                        try:
                            mlflow.log_artifact(str(f))
                        except Exception as e:
                            print(f"[WARN] Artifact upload failed: {e}")
                    for f in variant_dir.glob("*.csv"):
                        try:
                            mlflow.log_artifact(str(f))
                        except Exception as e:
                            print(f"[WARN] Artifact upload failed: {e}")

            elapsed_v = time.time() - variant_t0
            print(f"  Variant {variant_name} completed in {elapsed_v/60:.1f} min")

        # ─── COMPARISON TABLE ─────────────────────────────────────────────
        print(f"\n\n{'='*70}")
        print(f"  COMPARISON: ENHANCED FEATURES EXPERIMENT")
        print(f"{'='*70}")

        if len(all_results) >= 2:
            header = f"{'Variant':<20} {'Feats':>5} {'MAE':>7} {'R²':>7} {'Corr':>7} {'RegAcc':>7} {'TrRecl':>7} {'TrPrec':>7} {'QMono':>7}"
            print(header)
            print("─" * len(header))

            baseline = all_results.get("A_baseline")

            for name, r in all_results.items():
                line = (f"{name:<20} {r['n_features']:>5} {r['concat_mae']:>7.4f} "
                        f"{r['concat_r2']:>7.4f} {r['concat_corr']:>7.4f} "
                        f"{r['regime_accuracy']:>7.3f} {r['transition_recall']:>7.3f} "
                        f"{r['transition_precision']:>7.3f} {r['quintile_monotonicity']:>7.3f}")
                print(line)

            if baseline:
                print(f"\n  Deltas vs Baseline (A):")
                for name, r in all_results.items():
                    if name == "A_baseline":
                        continue
                    d_mae = r["concat_mae"] - baseline["concat_mae"]
                    d_r2 = r["concat_r2"] - baseline["concat_r2"]
                    d_corr = r["concat_corr"] - baseline["concat_corr"]
                    d_acc = r["regime_accuracy"] - baseline["regime_accuracy"]
                    d_recall = r["transition_recall"] - baseline["transition_recall"]
                    d_prec = r["transition_precision"] - baseline["transition_precision"]
                    d_mono = r["quintile_monotonicity"] - baseline["quintile_monotonicity"]
                    print(f"    {name}: MAE {d_mae:+.4f}, R² {d_r2:+.4f}, Corr {d_corr:+.4f}, "
                          f"RegAcc {d_acc:+.3f}, TrRecl {d_recall:+.3f}, TrPrec {d_prec:+.3f}, "
                          f"QMono {d_mono:+.3f}")

                    verdict = "BETTER" if d_corr > 0.005 and d_mono > 0 else "WORSE" if d_corr < -0.005 else "NEUTRAL"
                    print(f"    → Verdict: {verdict}")

            # Log comparison to parent run
            for name, r in all_results.items():
                for metric in ["concat_mae", "concat_r2", "concat_corr", "regime_accuracy",
                               "transition_recall", "transition_precision", "quintile_monotonicity"]:
                    mlflow.log_metric(f"{name}_{metric}", r[metric])

    elapsed = time.time() - t0
    print(f"\n[DONE] Total experiment time: {elapsed/60:.1f} minutes ({elapsed/3600:.1f} hours)")
    print(f"Completed: {dt.datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
