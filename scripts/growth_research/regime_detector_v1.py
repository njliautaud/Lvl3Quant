#!/usr/bin/env python3
"""
Regime Detector v1 — Continuous regime scoring via realized volatility forecasting.

Replaces crude VIX>20 binary threshold with a continuous 0-1 regime intensity score.
Uses LSTM/GRU on daily market data to predict next-20-day realized volatility of SPY.

Walk-forward: 1000d train, 250d val, 20d step, sliding window (NO expanding).
Logs to MLflow on Jupiter. Saves predictions as .npz.

Author: Claude (Head of Quant)
Date: 2026-07-26
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
    # Data
    "tickers": [
        "SPY", "^VIX", "^VIX3M",
        "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY",
        "TLT", "HYG", "GLD",
    ],
    "start_date": "2007-01-01",  # Enough history for walk-forward
    "end_date": None,  # Today

    # Features
    "spy_momentum_windows": [5, 10, 21, 63],
    "sma_window": 200,
    "lookback": 60,  # Input sequence length (trading days)

    # Target
    "fwd_vol_window": 20,  # Next 20-day realized vol

    # Model
    "model_type": "GRU",  # GRU tends to train faster than LSTM with similar quality
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.3,
    "bidirectional": False,

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
    "experiment_name": "regime-detector-v1",

    # Output
    "output_dir": "/home/nick/Lvl3Quant/output/regime_detector_v1",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INIT] Device: {DEVICE}")


# ─── DATA DOWNLOAD & FEATURE ENGINEERING ─────────────────────────────────────

def download_data(cfg):
    """Download daily OHLCV data for all tickers via yfinance."""
    print(f"[DATA] Downloading {len(cfg['tickers'])} tickers from {cfg['start_date']}...")
    end = cfg["end_date"] or dt.date.today().isoformat()

    data = {}
    for ticker in cfg["tickers"]:
        try:
            df = yf.download(ticker, start=cfg["start_date"], end=end, auto_adjust=True, progress=False)
            # yfinance may return MultiIndex columns for single ticker — flatten
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
    """Safely extract a single column as a Series, even if columns have duplicates."""
    s = df[col]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return s


def build_features(data, cfg):
    """Engineer features from raw price data."""
    spy = _col(data["SPY"], "Close").copy()
    spy.name = "SPY_close"

    # Align all series to SPY's index
    idx = spy.index

    features = pd.DataFrame(index=idx)

    # 1. VIX level (raw and log)
    if "^VIX" in data:
        vix = _col(data["^VIX"], "Close").reindex(idx).ffill()
        features["vix"] = vix
        features["vix_log"] = np.log(vix.clip(lower=1))

    # 2. VIX term structure: VIX / VIX3M ratio (< 1 = contango = calm)
    if "^VIX" in data and "^VIX3M" in data:
        vix3m = _col(data["^VIX3M"], "Close").reindex(idx).ffill()
        features["vix_term_ratio"] = (vix / vix3m.clip(lower=1)).clip(0.5, 2.0)

    # 3. SPY returns at multiple horizons
    spy_ret = spy.pct_change()
    for w in cfg["spy_momentum_windows"]:
        features[f"spy_ret_{w}d"] = spy.pct_change(w)

    # 4. SPY distance from 200d SMA (normalized)
    sma200 = spy.rolling(cfg["sma_window"]).mean()
    features["spy_sma200_dist"] = (spy - sma200) / sma200

    # 5. SPY realized vol (trailing, for context)
    features["spy_rvol_10d"] = spy_ret.rolling(10).std() * np.sqrt(252)
    features["spy_rvol_21d"] = spy_ret.rolling(21).std() * np.sqrt(252)

    # 6. Sector dispersion: cross-sector daily return std
    sector_tickers = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
    sector_rets = pd.DataFrame()
    for t in sector_tickers:
        if t in data:
            sector_rets[t] = _col(data[t], "Close").reindex(idx).ffill().pct_change()
    if len(sector_rets.columns) >= 5:
        features["sector_dispersion"] = sector_rets.std(axis=1)
        features["sector_dispersion_5d"] = features["sector_dispersion"].rolling(5).mean()
        # Sector rotation intensity: avg absolute rank change
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

    # 8. Gold momentum (flight to safety indicator)
    if "GLD" in data:
        gld = _col(data["GLD"], "Close").reindex(idx).ffill()
        features["gld_ret_5d"] = gld.pct_change(5)
        features["gld_ret_21d"] = gld.pct_change(21)

    # 9. SPY-TLT correlation (rolling 21d) — risk-on/risk-off regime indicator
    if "TLT" in data:
        tlt_ret = _col(data["TLT"], "Close").reindex(idx).ffill().pct_change()
        features["spy_tlt_corr_21d"] = spy_ret.rolling(21).corr(tlt_ret)

    # 10. Intraday range proxy (High-Low / Close)
    if "High" in data["SPY"].columns and "Low" in data["SPY"].columns:
        features["spy_range"] = (_col(data["SPY"], "High").reindex(idx) - _col(data["SPY"], "Low").reindex(idx)) / spy

    # ─── TARGET: Forward 20-day realized vol (annualized) ─────────────────
    fwd_ret = spy_ret.shift(-1)  # Next day's return
    target = fwd_ret.rolling(cfg["fwd_vol_window"]).std().shift(-(cfg["fwd_vol_window"] - 1)) * np.sqrt(252)
    target.name = "fwd_rvol_20d"

    # Drop NaN rows
    combined = pd.concat([features, target], axis=1).dropna()
    feat_cols = [c for c in combined.columns if c != "fwd_rvol_20d"]

    print(f"[FEATURES] {len(feat_cols)} features, {len(combined)} valid samples")
    print(f"  Features: {feat_cols}")
    print(f"  Target range: [{combined['fwd_rvol_20d'].min():.4f}, {combined['fwd_rvol_20d'].max():.4f}]")
    print(f"  Target mean: {combined['fwd_rvol_20d'].mean():.4f}, std: {combined['fwd_rvol_20d'].std():.4f}")

    return combined, feat_cols


# ─── DATASET ──────────────────────────────────────────────────────────────────

class RegimeDataset(Dataset):
    def __init__(self, X, y):
        """X: (N, seq_len, n_features), y: (N,)"""
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_sequences(features, targets, lookback):
    """Create sliding window sequences."""
    X, y = [], []
    for i in range(lookback, len(features)):
        X.append(features[i - lookback:i])
        y.append(targets[i])
    return np.array(X), np.array(y)


# ─── MODEL ────────────────────────────────────────────────────────────────────

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
        # x: (batch, seq_len, n_features)
        out, _ = self.rnn(x)
        # Use last timestep
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


# ─── WALK-FORWARD ENGINE ─────────────────────────────────────────────────────

def run_walk_forward(combined, feat_cols, cfg):
    """Sliding window walk-forward training and prediction."""
    output_dir = Path(cfg["output_dir"])
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

    # Walk-forward folds
    fold_starts = list(range(0, total_samples - min_required, step_days))
    n_folds = len(fold_starts)
    print(f"\n[WF] {n_folds} walk-forward folds, {total_samples} total samples")

    # MLflow setup
    mlflow.set_tracking_uri(cfg["mlflow_uri"])
    mlflow.set_experiment(cfg["experiment_name"])

    all_oot_preds = []
    all_oot_actuals = []
    all_oot_dates = []
    fold_metrics = []

    with mlflow.start_run(run_name=f"regime_detector_v1_{dt.datetime.now().strftime('%Y%m%d_%H%M')}") as run:
        mlflow.log_params({
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
            "target": "fwd_rvol_20d",
            "features": str(feat_cols),
        })

        for fold_idx, fold_start in enumerate(fold_starts):
            train_end = fold_start + train_days
            val_end = train_end + val_days

            if val_end > total_samples:
                break

            # SLIDING window — no expanding
            train_feat = features_raw[fold_start:train_end]
            train_tgt = targets_raw[fold_start:train_end]
            val_feat = features_raw[train_end:val_end]
            val_tgt = targets_raw[train_end:val_end]

            # Fit scaler on TRAIN only
            scaler = RobustScaler()
            train_feat_scaled = scaler.fit_transform(train_feat)
            val_feat_scaled = scaler.transform(val_feat)

            # Also scale targets for stable training (inverse later)
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
            # Only keep val portion
            X_val = X_val[lookback:]
            y_val = y_val[lookback:]

            if len(X_train) < 50 or len(X_val) < 10:
                print(f"  Fold {fold_idx}: skipped (too few samples)")
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

            # Load best model and get final predictions
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

            # Get OOT dates
            val_date_start_idx = train_end + lookback
            val_date_end_idx = val_end
            if val_date_end_idx <= len(dates):
                oot_dates = dates[val_date_start_idx:val_date_end_idx]
                # Align lengths
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

            if fold_idx % 10 == 0 or fold_idx == n_folds - 1:
                print(f"  Fold {fold_idx}/{n_folds}: MAE={val_mae_raw:.4f}, R2={val_r2_raw:.3f}, "
                      f"Corr={val_corr_raw:.3f}, Epochs={cfg['epochs'] - patience_counter}")

            # Save checkpoint every 50 folds
            if fold_idx % 50 == 0 and best_state:
                ckpt_path = output_dir / f"fold_{fold_idx:04d}.pt"
                torch.save({
                    "model_state": best_state,
                    "scaler_center": scaler.center_,
                    "scaler_scale": scaler.scale_,
                    "tgt_mean": tgt_mean,
                    "tgt_std": tgt_std,
                    "fold": fold_idx,
                    "cfg": cfg,
                }, ckpt_path)

        # ─── AGGREGATE METRICS ────────────────────────────────────────────
        all_oot_preds = np.array(all_oot_preds)
        all_oot_actuals = np.array(all_oot_actuals)

        if len(all_oot_preds) > 10:
            concat_mae = mean_absolute_error(all_oot_actuals, all_oot_preds)
            concat_r2 = r2_score(all_oot_actuals, all_oot_preds)
            concat_corr = np.corrcoef(all_oot_actuals, all_oot_preds)[0, 1]

            # Regime classification accuracy: predict high-vol vs low-vol
            # Using 20% annualized as threshold (comparable to VIX=20)
            threshold = 0.20
            pred_high = all_oot_preds > threshold
            actual_high = all_oot_actuals > threshold
            regime_accuracy = (pred_high == actual_high).mean()

            # Transition detection: did we predict regime CHANGES?
            actual_transitions = np.diff(actual_high.astype(int)) != 0
            pred_transitions = np.diff(pred_high.astype(int)) != 0
            if actual_transitions.sum() > 0:
                # Of actual transitions, how many did we predict within +/- 5 days?
                transition_recall = 0
                actual_trans_idx = np.where(actual_transitions)[0]
                pred_trans_idx = np.where(pred_transitions)[0]
                for at in actual_trans_idx:
                    if any(abs(at - pt) <= 5 for pt in pred_trans_idx):
                        transition_recall += 1
                transition_recall /= len(actual_trans_idx)
            else:
                transition_recall = np.nan

            print(f"\n{'='*60}")
            print(f"AGGREGATE OOT RESULTS ({len(all_oot_preds)} predictions)")
            print(f"{'='*60}")
            print(f"  MAE:              {concat_mae:.4f}")
            print(f"  R2:               {concat_r2:.4f}")
            print(f"  Correlation:      {concat_corr:.4f}")
            print(f"  Regime Accuracy:  {regime_accuracy:.3f} (at 20% threshold)")
            print(f"  Transition Recall: {transition_recall:.3f} (within ±5 days)")

            # Quintile analysis
            print(f"\n  Quintile Analysis (predicted vol → actual vol):")
            quintiles = pd.qcut(all_oot_preds, 5, labels=False, duplicates="drop")
            for q in sorted(np.unique(quintiles)):
                mask = quintiles == q
                print(f"    Q{q}: pred_mean={all_oot_preds[mask].mean():.4f}, "
                      f"actual_mean={all_oot_actuals[mask].mean():.4f}, "
                      f"n={mask.sum()}")

            # Log to MLflow
            mlflow.log_metrics({
                "concat_mae": concat_mae,
                "concat_r2": concat_r2,
                "concat_corr": concat_corr,
                "regime_accuracy": regime_accuracy,
                "transition_recall": float(transition_recall) if not np.isnan(transition_recall) else 0,
                "n_oot_predictions": len(all_oot_preds),
                "n_folds_completed": len(fold_metrics),
            })

            # Per-fold metrics
            fold_df = pd.DataFrame(fold_metrics)
            mlflow.log_metrics({
                "mean_fold_mae": fold_df["val_mae"].mean(),
                "mean_fold_r2": fold_df["val_r2"].mean(),
                "mean_fold_corr": fold_df["val_corr"].mean(),
                "std_fold_corr": fold_df["val_corr"].std(),
            })

            # Save fold metrics
            fold_df.to_csv(output_dir / "fold_metrics.csv", index=False)
            try:
                mlflow.log_artifact(str(output_dir / "fold_metrics.csv"))
            except Exception as e:
                print(f"[WARN] MLflow artifact upload failed (fold_metrics): {e}")

        # ─── SAVE PREDICTIONS ─────────────────────────────────────────────
        # Convert dates to strings for .npz storage
        date_strings = [str(d.date()) if hasattr(d, 'date') else str(d) for d in all_oot_dates]

        npz_path = output_dir / "regime_predictions_v1.npz"
        np.savez(
            npz_path,
            predictions=all_oot_preds,
            actuals=all_oot_actuals,
            dates=np.array(date_strings),
            threshold=threshold,
            regime_scores=np.clip(all_oot_preds / 0.40, 0, 1),  # Normalize to 0-1 (40% vol = max)
        )
        print(f"\n[SAVED] Predictions: {npz_path}")

        # Save last model checkpoint
        if best_state:
            final_path = output_dir / "regime_detector_final.pt"
            torch.save({
                "model_state": best_state,
                "scaler_center": scaler.center_,
                "scaler_scale": scaler.scale_,
                "tgt_mean": tgt_mean,
                "tgt_std": tgt_std,
                "cfg": cfg,
                "feat_cols": feat_cols,
                "concat_metrics": {
                    "mae": concat_mae,
                    "r2": concat_r2,
                    "corr": concat_corr,
                    "regime_accuracy": regime_accuracy,
                },
            }, final_path)
            try:
                mlflow.log_artifact(str(final_path))
            except Exception as e:
                print(f"[WARN] MLflow artifact upload failed (model): {e}")
            print(f"[SAVED] Final model: {final_path}")

        try:
            mlflow.log_artifact(str(npz_path))
        except Exception as e:
            print(f"[WARN] MLflow artifact upload failed (predictions): {e}")
        print(f"[MLFLOW] Run ID: {run.info.run_id}")

    return fold_metrics, all_oot_preds, all_oot_actuals


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print(f"{'='*60}")
    print(f"REGIME DETECTOR v1 — Continuous Volatility Regime Scoring")
    print(f"{'='*60}")
    print(f"Started: {dt.datetime.now().isoformat()}")

    # 1. Download data
    data = download_data(CFG)
    if "SPY" not in data or "^VIX" not in data:
        print("[FATAL] Missing SPY or VIX data. Aborting.")
        sys.exit(1)

    # 2. Build features
    combined, feat_cols = build_features(data, CFG)

    # 3. Run walk-forward
    fold_metrics, preds, actuals = run_walk_forward(combined, feat_cols, CFG)

    elapsed = time.time() - t0
    print(f"\n[DONE] Total time: {elapsed/60:.1f} minutes")
    print(f"Completed: {dt.datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
