#!/usr/bin/env python3
"""
Regime Detector v2 — Improved transition detection.

5 variants tested against v1 baseline (83.9% accuracy, 29.3% transition recall):
  A. Transition-weighted loss (3x weight near regime changes)
  B. Multi-horizon prediction (t+1, t+5, t+10 heads)
  C. Attention-GRU (self-attention on GRU outputs)
  D. Bidirectional GRU (bidir train, forward-only inference)
  E. Change-point features (explicit transition velocity features)

Walk-forward: 1000d train, 250d val, 20d step, sliding window.
Logs to MLflow. Saves predictions as .npz.

Author: Claude (Head of Quant)
Date: 2026-07-27
"""

import os
import sys
import time
import warnings
import datetime as dt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, r2_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import mlflow
import mlflow.pytorch

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INIT] Device: {DEVICE}")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
BASE_CFG = {
    "tickers": [
        "SPY", "^VIX",
        "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY",
        "TLT", "SHY", "HYG", "GLD",
    ],
    "start_date": "2007-01-01",
    "end_date": None,
    "spy_momentum_windows": [5, 10, 21, 63],
    "sma_window": 200,
    "lookback": 60,
    "fwd_vol_window": 20,
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.3,
    "train_days": 1000,
    "val_days": 250,
    "step_days": 20,
    "epochs": 80,
    "batch_size": 64,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "patience": 15,
    "min_lr": 1e-6,
    "mlflow_uri": "http://jupiter:5000",
    "experiment_name": "regime_detector_v2",
    "output_dir": "/home/nick/Lvl3Quant/output/regime_detector_v2",
}


# ─── DATA ─────────────────────────────────────────────────────────────────────

def download_data(cfg):
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
                print(f"  {ticker}: SKIPPED ({len(df)} days)")
        except Exception as e:
            print(f"  {ticker}: FAILED ({e})")
    return data


def _col(df, col):
    s = df[col]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return s


def build_features(data, cfg, add_changepoint=False):
    """Build feature matrix. If add_changepoint=True, add change-point velocity features (variant E)."""
    spy = _col(data["SPY"], "Close").copy()
    spy.name = "SPY_close"
    idx = spy.index
    features = pd.DataFrame(index=idx)

    # 1. VIX
    if "^VIX" in data:
        vix = _col(data["^VIX"], "Close").reindex(idx).ffill()
        features["vix"] = vix
        features["vix_log"] = np.log(vix.clip(lower=1))
        # VIX changes (used by all, explicit for variant E)
        features["vix_chg_5d"] = vix.pct_change(5)
        features["vix_chg_10d"] = vix.pct_change(10)
        features["vix_chg_21d"] = vix.pct_change(21)

    # 2. SPY returns
    spy_ret = spy.pct_change()
    for w in cfg["spy_momentum_windows"]:
        features[f"spy_ret_{w}d"] = spy.pct_change(w)

    # 3. SPY distance from SMA200
    sma200 = spy.rolling(cfg["sma_window"]).mean()
    features["spy_sma200_dist"] = (spy - sma200) / sma200

    # 4. Realized vol
    features["spy_rvol_21d"] = spy_ret.rolling(21).std() * np.sqrt(252)
    features["spy_rvol_63d"] = spy_ret.rolling(63).std() * np.sqrt(252)

    # 5. Sector dispersion
    sector_tickers = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
    sector_rets = pd.DataFrame()
    for t in sector_tickers:
        if t in data:
            sector_rets[t] = _col(data[t], "Close").reindex(idx).ffill().pct_change()
    if len(sector_rets.columns) >= 5:
        features["sector_dispersion"] = sector_rets.std(axis=1)

    # 6. Yield curve proxy (TLT/SHY ratio)
    if "TLT" in data and "SHY" in data:
        tlt = _col(data["TLT"], "Close").reindex(idx).ffill()
        shy = _col(data["SHY"], "Close").reindex(idx).ffill()
        features["yield_curve_ratio"] = tlt / shy.clip(lower=1)

    # 7. Credit spread proxy (HYG/TLT)
    if "HYG" in data and "TLT" in data:
        hyg = _col(data["HYG"], "Close").reindex(idx).ffill()
        tlt = _col(data["TLT"], "Close").reindex(idx).ffill()
        features["credit_spread_ratio"] = hyg / tlt.clip(lower=1)
        hyg_ret = hyg.pct_change()
        tlt_ret = tlt.pct_change()
        features["credit_spread_ret_5d"] = hyg_ret.rolling(5).mean() - tlt_ret.rolling(5).mean()

    # 8. Gold momentum
    if "GLD" in data:
        gld = _col(data["GLD"], "Close").reindex(idx).ffill()
        features["gld_ret_5d"] = gld.pct_change(5)

    # 9. Put-call proxy: VIX / realized vol ratio
    if "^VIX" in data:
        rvol21 = features.get("spy_rvol_21d")
        if rvol21 is not None:
            features["put_call_proxy"] = vix / (rvol21.clip(lower=1) * 100)

    # 10. SPY-TLT correlation
    if "TLT" in data:
        tlt_ret = _col(data["TLT"], "Close").reindex(idx).ffill().pct_change()
        features["spy_tlt_corr_21d"] = spy_ret.rolling(21).corr(tlt_ret)

    # 11. Skew proxy: rolling skewness of SPY returns
    features["spy_skew_21d"] = spy_ret.rolling(21).skew()

    # 12. Momentum breadth: fraction of sectors with positive 21d return
    if len(sector_rets.columns) >= 5:
        sector_mom = sector_rets.rolling(21).mean()
        features["momentum_breadth"] = (sector_mom > 0).mean(axis=1)

    # 13. Vol of vol
    if "^VIX" in data:
        features["vol_of_vol"] = vix.pct_change().rolling(21).std()

    # ─── VARIANT E: Change-point features ─────────────────────────────────
    if add_changepoint:
        if "^VIX" in data:
            # VIX acceleration (2nd derivative)
            vix_diff = vix.diff()
            features["vix_accel_5d"] = vix_diff.rolling(5).mean().diff()
            # VIX regime change velocity: rolling std of VIX daily changes
            features["vix_change_velocity"] = vix_diff.rolling(10).std()
            # VIX breakout: current vs 20d range
            vix_hi20 = vix.rolling(20).max()
            vix_lo20 = vix.rolling(20).min()
            vix_range = (vix_hi20 - vix_lo20).clip(lower=0.1)
            features["vix_breakout"] = (vix - vix_lo20) / vix_range

        # HYG-TLT spread velocity
        if "HYG" in data and "TLT" in data:
            cs = features.get("credit_spread_ratio")
            if cs is not None:
                features["credit_velocity_5d"] = cs.diff(5)
                features["credit_velocity_10d"] = cs.diff(10)

        # Realized vol acceleration
        rvol = features.get("spy_rvol_21d")
        if rvol is not None:
            features["rvol_accel"] = rvol.diff().rolling(5).mean()
            features["rvol_regime_chg"] = rvol.diff(10)  # 10d vol change (for transition ID)

        # Sector dispersion velocity
        sd = features.get("sector_dispersion")
        if sd is not None:
            features["dispersion_velocity"] = sd.diff(5)

    # ─── TARGET ───────────────────────────────────────────────────────────
    fwd_ret = spy_ret.shift(-1)
    target = fwd_ret.rolling(cfg["fwd_vol_window"]).std().shift(-(cfg["fwd_vol_window"] - 1)) * np.sqrt(252)
    target.name = "fwd_rvol_20d"

    combined = pd.concat([features, target], axis=1).dropna()
    feat_cols = [c for c in combined.columns if c != "fwd_rvol_20d"]

    print(f"[FEATURES] {len(feat_cols)} features, {len(combined)} valid samples")
    print(f"  Features: {feat_cols}")

    return combined, feat_cols


# ─── DATASET ──────────────────────────────────────────────────────────────────

class RegimeDataset(Dataset):
    def __init__(self, X, y, weights=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.weights = torch.FloatTensor(weights) if weights is not None else None

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if self.weights is not None:
            return self.X[idx], self.y[idx], self.weights[idx]
        return self.X[idx], self.y[idx]


class MultiHorizonDataset(Dataset):
    """For variant B: multiple target horizons."""
    def __init__(self, X, y_dict):
        self.X = torch.FloatTensor(X)
        self.y = {k: torch.FloatTensor(v) for k, v in y_dict.items()}

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        targets = {k: v[idx] for k, v in self.y.items()}
        return self.X[idx], targets


def create_sequences(features, targets, lookback):
    X, y = [], []
    for i in range(lookback, len(features)):
        X.append(features[i - lookback:i])
        y.append(targets[i])
    return np.array(X), np.array(y)


def create_sequences_weighted(features, targets, weights, lookback):
    X, y, w = [], [], []
    for i in range(lookback, len(features)):
        X.append(features[i - lookback:i])
        y.append(targets[i])
        w.append(weights[i])
    return np.array(X), np.array(y), np.array(w)


def create_sequences_multi(features, target_dict, lookback):
    X = []
    y_dict = {k: [] for k in target_dict}
    n = len(features)
    min_len = min(len(v) for v in target_dict.values())
    n = min(n, min_len)
    for i in range(lookback, n):
        X.append(features[i - lookback:i])
        for k in target_dict:
            y_dict[k].append(target_dict[k][i])
    return np.array(X), {k: np.array(v) for k, v in y_dict.items()}


# ─── MODELS ───────────────────────────────────────────────────────────────────

class RegimeGRU(nn.Module):
    """Baseline GRU (same as v1)."""
    def __init__(self, n_features, hidden_size=64, num_layers=2, dropout=0.3, bidirectional=False):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=n_features, hidden_size=hidden_size, num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0, batch_first=True, bidirectional=bidirectional,
        )
        d = 2 if bidirectional else 1
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size * d), nn.Dropout(dropout),
            nn.Linear(hidden_size * d, 32), nn.GELU(),
            nn.Dropout(dropout * 0.5), nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class MultiHorizonGRU(nn.Module):
    """Variant B: Predict regime at t+1, t+5, t+10 simultaneously."""
    def __init__(self, n_features, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=n_features, hidden_size=hidden_size, num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0, batch_first=True,
        )
        self.shared = nn.Sequential(
            nn.LayerNorm(hidden_size), nn.Dropout(dropout),
            nn.Linear(hidden_size, 48), nn.GELU(),
        )
        # 3 separate heads for different horizons
        self.head_1 = nn.Linear(48, 1)   # t+1
        self.head_5 = nn.Linear(48, 1)   # t+5
        self.head_10 = nn.Linear(48, 1)  # t+10

    def forward(self, x):
        out, _ = self.rnn(x)
        h = self.shared(out[:, -1, :])
        return {
            "h1": self.head_1(h).squeeze(-1),
            "h5": self.head_5(h).squeeze(-1),
            "h10": self.head_10(h).squeeze(-1),
        }


class AttentionGRU(nn.Module):
    """Variant C: Self-attention on top of GRU outputs."""
    def __init__(self, n_features, hidden_size=64, num_layers=2, dropout=0.3, n_heads=4):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=n_features, hidden_size=hidden_size, num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0, batch_first=True,
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=n_heads, dropout=dropout, batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size), nn.Dropout(dropout),
            nn.Linear(hidden_size, 32), nn.GELU(),
            nn.Dropout(dropout * 0.5), nn.Linear(32, 1),
        )

    def forward(self, x):
        rnn_out, _ = self.rnn(x)  # (B, T, H)
        # Self-attention over timesteps
        attn_out, _ = self.attn(rnn_out, rnn_out, rnn_out)
        attn_out = self.attn_norm(rnn_out + attn_out)  # residual
        # Use last timestep after attention
        return self.head(attn_out[:, -1, :]).squeeze(-1)


class BidirGRU(nn.Module):
    """Variant D: Bidirectional GRU for training, forward-only for inference."""
    def __init__(self, n_features, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(
            input_size=n_features, hidden_size=hidden_size, num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0, batch_first=True, bidirectional=True,
        )
        # Projection from bidir (2*H) to H, then head
        self.proj = nn.Linear(hidden_size * 2, hidden_size)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size), nn.Dropout(dropout),
            nn.Linear(hidden_size, 32), nn.GELU(),
            nn.Dropout(dropout * 0.5), nn.Linear(32, 1),
        )
        self._inference_mode = False

    def set_inference_mode(self, mode=True):
        """In inference mode, only use forward direction output."""
        self._inference_mode = mode

    def forward(self, x):
        out, _ = self.rnn(x)  # (B, T, 2*H)
        last = out[:, -1, :]  # (B, 2*H)
        if self._inference_mode:
            # Only use forward direction (first H dims)
            last = last[:, :self.hidden_size]
            # Pad with zeros for projection
            last = torch.cat([last, torch.zeros_like(last)], dim=-1)
        h = self.proj(last)
        return self.head(h).squeeze(-1)


# ─── TRANSITION DETECTION UTILS ──────────────────────────────────────────────

def compute_transition_weights(targets_raw, threshold=0.20, vol_change_thresh=0.03,
                                window=10, transition_radius=5, transition_weight=3.0):
    """Compute sample weights with higher weight near regime transitions.

    A transition is defined as: vol level changes by > vol_change_thresh (3pp) within `window` days.
    """
    weights = np.ones(len(targets_raw))

    # Identify transitions by vol level change
    for i in range(window, len(targets_raw)):
        vol_change = abs(targets_raw[i] - targets_raw[i - window])
        if vol_change > vol_change_thresh:
            # Mark surrounding samples
            lo = max(0, i - transition_radius)
            hi = min(len(targets_raw), i + transition_radius + 1)
            weights[lo:hi] = np.maximum(weights[lo:hi], transition_weight)

    # Also detect threshold crossings
    above = targets_raw > threshold
    crossings = np.where(np.diff(above.astype(int)) != 0)[0]
    for c in crossings:
        lo = max(0, c - transition_radius)
        hi = min(len(targets_raw), c + transition_radius + 1)
        weights[lo:hi] = np.maximum(weights[lo:hi], transition_weight)

    n_trans = (weights > 1.0).sum()
    print(f"    Transition samples: {n_trans}/{len(weights)} ({100*n_trans/len(weights):.1f}%)")
    return weights


def compute_transition_metrics(all_preds, all_actuals, threshold=0.20):
    """Compute transition-specific metrics."""
    pred_high = all_preds > threshold
    actual_high = all_actuals > threshold
    regime_accuracy = (pred_high == actual_high).mean()

    # Transition recall: of actual transitions, how many did we catch within ±5 days?
    actual_transitions = np.diff(actual_high.astype(int)) != 0
    pred_transitions = np.diff(pred_high.astype(int)) != 0
    actual_trans_idx = np.where(actual_transitions)[0]
    pred_trans_idx = np.where(pred_transitions)[0]

    if len(actual_trans_idx) == 0:
        return {
            "regime_accuracy": regime_accuracy,
            "transition_recall": float("nan"),
            "transition_precision": float("nan"),
            "n_actual_transitions": 0,
            "n_pred_transitions": len(pred_trans_idx),
        }

    # Recall: what fraction of actual transitions did we predict?
    recall_hits = 0
    for at in actual_trans_idx:
        if any(abs(at - pt) <= 5 for pt in pred_trans_idx):
            recall_hits += 1
    transition_recall = recall_hits / len(actual_trans_idx)

    # Precision: what fraction of predicted transitions were real?
    precision_hits = 0
    for pt in pred_trans_idx:
        if any(abs(pt - at) <= 5 for at in actual_trans_idx):
            precision_hits += 1
    transition_precision = precision_hits / max(len(pred_trans_idx), 1)

    # Early detection: for matched transitions, how many days early/late?
    lead_times = []
    for at in actual_trans_idx:
        matches = [pt for pt in pred_trans_idx if abs(at - pt) <= 5]
        if matches:
            closest = min(matches, key=lambda pt: abs(at - pt))
            lead_times.append(at - closest)  # positive = we predicted early

    avg_lead = np.mean(lead_times) if lead_times else 0

    return {
        "regime_accuracy": regime_accuracy,
        "transition_recall": transition_recall,
        "transition_precision": transition_precision,
        "transition_f1": 2 * transition_recall * transition_precision / max(transition_recall + transition_precision, 1e-8),
        "avg_lead_days": avg_lead,
        "n_actual_transitions": len(actual_trans_idx),
        "n_pred_transitions": len(pred_trans_idx),
    }


# ─── TRAINING FUNCTIONS ──────────────────────────────────────────────────────

def train_one_epoch_weighted(model, loader, optimizer, has_weights=False):
    """Train with optional per-sample weights."""
    model.train()
    total_loss, n = 0, 0
    criterion = nn.HuberLoss(delta=1.0, reduction='none')
    for batch in loader:
        if has_weights:
            X, y, w = batch
            X, y, w = X.to(DEVICE), y.to(DEVICE), w.to(DEVICE)
        else:
            X, y = batch
            X, y = X.to(DEVICE), y.to(DEVICE)
            w = None

        optimizer.zero_grad()
        pred = model(X)
        loss_raw = criterion(pred, y)
        if w is not None:
            loss = (loss_raw * w).mean()
        else:
            loss = loss_raw.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
        n += len(y)
    return total_loss / max(n, 1)


def train_one_epoch_multi(model, loader, optimizer):
    """Train multi-horizon model (variant B)."""
    model.train()
    total_loss, n = 0, 0
    criterion = nn.HuberLoss(delta=1.0)
    # Weight horizons: more weight on shorter (more useful for transitions)
    horizon_weights = {"h1": 1.0, "h5": 1.5, "h10": 2.0}  # More weight on further = catch transitions
    for X, y_dict in loader:
        X = X.to(DEVICE)
        y_dict = {k: v.to(DEVICE) for k, v in y_dict.items()}
        optimizer.zero_grad()
        preds = model(X)
        loss = sum(horizon_weights[k] * criterion(preds[k], y_dict[k]) for k in preds)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(X)
        n += len(X)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate_model(model, loader, is_multi=False):
    """Evaluate model, return loss and predictions."""
    model.eval()
    criterion = nn.HuberLoss(delta=1.0)
    preds_all, actuals_all = [], []
    total_loss, n = 0, 0

    for batch in loader:
        if is_multi:
            X, y_dict = batch
            X = X.to(DEVICE)
            y_dict = {k: v.to(DEVICE) for k, v in y_dict.items()}
            out = model(X)
            # Use h1 (next-day prediction) as the primary prediction
            pred = out["h1"]
            y = y_dict["h1"]
            loss = sum(criterion(out[k], y_dict[k]) for k in out)
        else:
            if len(batch) == 3:
                X, y, _ = batch
            else:
                X, y = batch
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            loss = criterion(pred, y)

        total_loss += loss.item() * len(pred)
        n += len(pred)
        preds_all.append(pred.cpu().numpy())
        actuals_all.append(y.cpu().numpy())

    preds_all = np.concatenate(preds_all)
    actuals_all = np.concatenate(actuals_all)
    mae = mean_absolute_error(actuals_all, preds_all)
    r2 = r2_score(actuals_all, preds_all) if len(actuals_all) > 2 else 0
    corr = np.corrcoef(actuals_all, preds_all)[0, 1] if len(actuals_all) > 2 else 0
    return total_loss / max(n, 1), mae, r2, corr, preds_all, actuals_all


# ─── WALK-FORWARD ENGINE ─────────────────────────────────────────────────────

def run_variant(variant_name, combined, feat_cols, cfg):
    """Run a single variant through walk-forward."""
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
    fold_starts = list(range(0, total_samples - min_required, step_days))
    n_folds = len(fold_starts)

    is_multi = (variant_name == "B_multi_horizon")
    has_weights = (variant_name == "A_transition_weighted")
    is_bidir = (variant_name == "D_bidirectional")

    print(f"\n{'='*60}")
    print(f"VARIANT {variant_name}: {n_folds} folds")
    print(f"{'='*60}")

    all_oot_preds, all_oot_actuals, all_oot_dates = [], [], []
    fold_metrics = []

    # Build multi-horizon targets if needed
    multi_targets = {}
    if is_multi:
        fwd_ret = _col(pd.DataFrame({"r": combined.index.to_series().diff()}, index=combined.index), "r")
        spy_ret = combined[feat_cols].iloc[:, 0]  # not correct, recompute from data
        # Actually rebuild from raw target column — use shifted versions
        t1 = targets_raw  # already fwd_rvol_20d
        # For t+5 and t+10, shift targets
        t5 = np.roll(targets_raw, -4)  # 5 days ahead
        t10 = np.roll(targets_raw, -9)  # 10 days ahead
        # Zero out the rolled portion
        t5[-4:] = t5[-5]
        t10[-9:] = t10[-10]
        multi_targets = {"h1": t1, "h5": t5, "h10": t10}

    for fold_idx, fold_start in enumerate(fold_starts):
        train_end = fold_start + train_days
        val_end = train_end + val_days
        if val_end > total_samples:
            break

        train_feat = features_raw[fold_start:train_end]
        train_tgt = targets_raw[fold_start:train_end]
        val_feat = features_raw[train_end:val_end]
        val_tgt = targets_raw[train_end:val_end]

        # Scaler
        scaler = RobustScaler()
        train_feat_scaled = scaler.fit_transform(train_feat)
        val_feat_scaled = scaler.transform(val_feat)

        tgt_mean = train_tgt.mean()
        tgt_std = train_tgt.std() + 1e-8
        train_tgt_norm = (train_tgt - tgt_mean) / tgt_std
        val_tgt_norm = (val_tgt - tgt_mean) / tgt_std

        # Create sequences
        if is_multi:
            train_mt = {k: (v[fold_start:train_end] - tgt_mean) / tgt_std for k, v in multi_targets.items()}
            val_mt = {k: (v[train_end:val_end] - tgt_mean) / tgt_std for k, v in multi_targets.items()}

            X_train, y_train_dict = create_sequences_multi(train_feat_scaled, train_mt, lookback)

            # Val with lookback context
            val_feat_ctx = np.vstack([train_feat_scaled[-lookback:], val_feat_scaled])
            val_mt_ctx = {k: np.concatenate([v[-lookback:], val_v]) for k, v, val_v
                          in [(k, train_mt[k], val_mt[k]) for k in train_mt]}
            X_val, y_val_dict = create_sequences_multi(val_feat_ctx, val_mt_ctx, lookback)
            X_val = X_val[lookback:]
            y_val_dict = {k: v[lookback:] for k, v in y_val_dict.items()}

            if len(X_train) < 50 or len(X_val) < 10:
                continue

            train_ds = MultiHorizonDataset(X_train, y_train_dict)
            val_ds = MultiHorizonDataset(X_val, y_val_dict)

        elif has_weights:
            # Compute transition weights
            weights = compute_transition_weights(train_tgt)
            X_train, y_train, w_train = create_sequences_weighted(train_feat_scaled, train_tgt_norm, weights, lookback)

            val_feat_ctx = np.vstack([train_feat_scaled[-lookback:], val_feat_scaled])
            val_tgt_ctx = np.concatenate([train_tgt_norm[-lookback:], val_tgt_norm])
            val_w_ctx = np.ones(len(val_tgt_ctx))
            X_val, y_val, _ = create_sequences_weighted(val_feat_ctx, val_tgt_ctx, val_w_ctx, lookback)
            X_val, y_val = X_val[lookback:], y_val[lookback:]

            if len(X_train) < 50 or len(X_val) < 10:
                continue

            train_ds = RegimeDataset(X_train, y_train, w_train)
            val_ds = RegimeDataset(X_val, y_val)

        else:
            X_train, y_train = create_sequences(train_feat_scaled, train_tgt_norm, lookback)
            val_feat_ctx = np.vstack([train_feat_scaled[-lookback:], val_feat_scaled])
            val_tgt_ctx = np.concatenate([train_tgt_norm[-lookback:], val_tgt_norm])
            X_val, y_val = create_sequences(val_feat_ctx, val_tgt_ctx, lookback)
            X_val, y_val = X_val[lookback:], y_val[lookback:]

            if len(X_train) < 50 or len(X_val) < 10:
                continue

            train_ds = RegimeDataset(X_train, y_train)
            val_ds = RegimeDataset(X_val, y_val)

        train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                                  num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                                num_workers=4, pin_memory=True)

        # Create model
        n_feat = len(feat_cols)
        if variant_name == "A_transition_weighted":
            model = RegimeGRU(n_feat, cfg["hidden_size"], cfg["num_layers"], cfg["dropout"]).to(DEVICE)
        elif variant_name == "B_multi_horizon":
            model = MultiHorizonGRU(n_feat, cfg["hidden_size"], cfg["num_layers"], cfg["dropout"]).to(DEVICE)
        elif variant_name == "C_attention_gru":
            model = AttentionGRU(n_feat, cfg["hidden_size"], cfg["num_layers"], cfg["dropout"]).to(DEVICE)
        elif variant_name == "D_bidirectional":
            model = BidirGRU(n_feat, cfg["hidden_size"], cfg["num_layers"], cfg["dropout"]).to(DEVICE)
        elif variant_name == "E_changepoint_features":
            model = RegimeGRU(n_feat, cfg["hidden_size"], cfg["num_layers"], cfg["dropout"]).to(DEVICE)
        else:
            model = RegimeGRU(n_feat, cfg["hidden_size"], cfg["num_layers"], cfg["dropout"]).to(DEVICE)

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"], eta_min=cfg["min_lr"])

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(cfg["epochs"]):
            if is_multi:
                train_loss = train_one_epoch_multi(model, train_loader, optimizer)
            else:
                train_loss = train_one_epoch_weighted(model, train_loader, optimizer, has_weights=has_weights)

            val_loss, val_mae, val_r2, val_corr, _, _ = evaluate_model(model, val_loader, is_multi=is_multi)
            scheduler.step()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= cfg["patience"]:
                    break

        # Load best and evaluate
        if best_state is None:
            continue
        model.load_state_dict(best_state)

        # For bidir variant, switch to inference mode for final eval
        if is_bidir:
            model.set_inference_mode(True)

        val_loss, val_mae, val_r2, val_corr, val_preds_norm, val_actuals_norm = evaluate_model(
            model, val_loader, is_multi=is_multi
        )

        # Inverse normalize
        val_preds = val_preds_norm * tgt_std + tgt_mean
        val_actuals = val_actuals_norm * tgt_std + tgt_mean

        val_mae_raw = mean_absolute_error(val_actuals, val_preds)
        val_r2_raw = r2_score(val_actuals, val_preds) if len(val_actuals) > 2 else 0
        val_corr_raw = np.corrcoef(val_actuals, val_preds)[0, 1] if len(val_actuals) > 2 else 0

        # Collect OOT predictions
        val_date_start_idx = train_end + lookback
        val_date_end_idx = val_end
        if val_date_end_idx <= len(dates):
            oot_dates = dates[val_date_start_idx:val_date_end_idx]
            min_len = min(len(val_preds), len(oot_dates))
            all_oot_preds.extend(val_preds[:min_len].tolist())
            all_oot_actuals.extend(val_actuals[:min_len].tolist())
            all_oot_dates.extend(oot_dates[:min_len].tolist())

        fold_metrics.append({
            "fold": fold_idx, "val_mae": val_mae_raw, "val_r2": val_r2_raw, "val_corr": val_corr_raw,
            "best_epoch": cfg["epochs"] - patience_counter,
        })

        if fold_idx % 20 == 0 or fold_idx == n_folds - 1:
            print(f"  Fold {fold_idx}/{n_folds}: MAE={val_mae_raw:.4f}, R2={val_r2_raw:.3f}, "
                  f"Corr={val_corr_raw:.3f}")

    # ─── AGGREGATE ────────────────────────────────────────────────────────
    all_oot_preds = np.array(all_oot_preds)
    all_oot_actuals = np.array(all_oot_actuals)

    results = {"variant": variant_name, "n_predictions": len(all_oot_preds)}

    if len(all_oot_preds) > 10:
        concat_mae = mean_absolute_error(all_oot_actuals, all_oot_preds)
        concat_r2 = r2_score(all_oot_actuals, all_oot_preds)
        concat_corr = np.corrcoef(all_oot_actuals, all_oot_preds)[0, 1]

        trans_metrics = compute_transition_metrics(all_oot_preds, all_oot_actuals)

        results.update({
            "concat_mae": concat_mae,
            "concat_r2": concat_r2,
            "concat_corr": concat_corr,
            **trans_metrics,
        })

        print(f"\n  RESULTS for {variant_name}:")
        print(f"    MAE:                {concat_mae:.4f}")
        print(f"    R2:                 {concat_r2:.4f}")
        print(f"    Correlation:        {concat_corr:.4f}")
        print(f"    Regime Accuracy:    {trans_metrics['regime_accuracy']:.3f}")
        print(f"    Transition Recall:  {trans_metrics['transition_recall']:.3f}")
        print(f"    Transition Prec:    {trans_metrics['transition_precision']:.3f}")
        print(f"    Transition F1:      {trans_metrics.get('transition_f1', 0):.3f}")
        print(f"    Avg Lead (days):    {trans_metrics.get('avg_lead_days', 0):.1f}")
        print(f"    Actual transitions: {trans_metrics['n_actual_transitions']}")
        print(f"    Pred transitions:   {trans_metrics['n_pred_transitions']}")

        # Save predictions
        date_strings = [str(d.date()) if hasattr(d, 'date') else str(d) for d in all_oot_dates]
        npz_path = output_dir / f"regime_predictions_{variant_name}.npz"
        np.savez(npz_path, predictions=all_oot_preds, actuals=all_oot_actuals,
                 dates=np.array(date_strings), threshold=0.20,
                 regime_scores=np.clip(all_oot_preds / 0.40, 0, 1))
        print(f"  [SAVED] {npz_path}")

        # Save fold metrics
        fold_df = pd.DataFrame(fold_metrics)
        fold_df.to_csv(output_dir / "fold_metrics.csv", index=False)

    return results


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print(f"{'='*60}")
    print(f"REGIME DETECTOR v2 — Transition Detection Improvement")
    print(f"{'='*60}")
    print(f"Started: {dt.datetime.now().isoformat()}")

    # 1. Download data once
    data = download_data(BASE_CFG)
    if "SPY" not in data or "^VIX" not in data:
        print("[FATAL] Missing SPY or VIX data. Aborting.")
        sys.exit(1)

    # 2. Build features (base and with change-point)
    combined_base, feat_cols_base = build_features(data, BASE_CFG, add_changepoint=False)
    combined_cp, feat_cols_cp = build_features(data, BASE_CFG, add_changepoint=True)

    print(f"\n[INFO] Base features: {len(feat_cols_base)}, Change-point features: {len(feat_cols_cp)}")

    # 3. MLflow setup
    mlflow.set_tracking_uri(BASE_CFG["mlflow_uri"])
    mlflow.set_experiment(BASE_CFG["experiment_name"])

    # 4. Run all 5 variants
    variants = [
        ("A_transition_weighted", combined_base, feat_cols_base),
        ("B_multi_horizon", combined_base, feat_cols_base),
        ("C_attention_gru", combined_base, feat_cols_base),
        ("D_bidirectional", combined_base, feat_cols_base),
        ("E_changepoint_features", combined_cp, feat_cols_cp),
    ]

    all_results = []
    for variant_name, combined, feat_cols in variants:
        with mlflow.start_run(run_name=f"v2_{variant_name}_{dt.datetime.now().strftime('%Y%m%d_%H%M')}"):
            mlflow.log_params({
                "variant": variant_name,
                "hidden_size": BASE_CFG["hidden_size"],
                "num_layers": BASE_CFG["num_layers"],
                "lookback": BASE_CFG["lookback"],
                "train_days": BASE_CFG["train_days"],
                "val_days": BASE_CFG["val_days"],
                "step_days": BASE_CFG["step_days"],
                "n_features": len(feat_cols),
                "epochs": BASE_CFG["epochs"],
                "batch_size": BASE_CFG["batch_size"],
            })

            results = run_variant(variant_name, combined, feat_cols, BASE_CFG)
            all_results.append(results)

            # Log metrics to MLflow
            for k, v in results.items():
                if isinstance(v, (int, float)) and not np.isnan(v) if isinstance(v, float) else True:
                    try:
                        mlflow.log_metric(k, v)
                    except Exception:
                        pass

    # 5. Summary comparison
    print(f"\n{'='*70}")
    print(f"VARIANT COMPARISON (v1 baseline: Accuracy=83.9%, Trans Recall=29.3%)")
    print(f"{'='*70}")
    print(f"{'Variant':<30} {'Accuracy':>8} {'Trans Rec':>10} {'Trans Prec':>10} {'F1':>6} {'R2':>6} {'MAE':>6}")
    print(f"{'-'*30} {'-'*8} {'-'*10} {'-'*10} {'-'*6} {'-'*6} {'-'*6}")

    for r in all_results:
        name = r.get("variant", "?")
        acc = r.get("regime_accuracy", 0)
        tr = r.get("transition_recall", 0)
        tp = r.get("transition_precision", 0)
        f1 = r.get("transition_f1", 0)
        r2 = r.get("concat_r2", 0)
        mae = r.get("concat_mae", 0)
        # Flag if meets success criteria
        flag = " ***" if (tr > 0.50 and acc > 0.80) else ""
        print(f"{name:<30} {acc:>8.3f} {tr:>10.3f} {tp:>10.3f} {f1:>6.3f} {r2:>6.3f} {mae:>6.4f}{flag}")

    # Winners
    winners = [r for r in all_results if r.get("transition_recall", 0) > 0.50 and r.get("regime_accuracy", 0) > 0.80]
    if winners:
        best = max(winners, key=lambda r: r.get("transition_f1", 0))
        print(f"\n  BEST VARIANT: {best['variant']} (F1={best.get('transition_f1',0):.3f})")
    else:
        # Best transition recall regardless
        best = max(all_results, key=lambda r: r.get("transition_recall", 0))
        print(f"\n  No variant met both criteria. Best transition recall: {best['variant']} ({best.get('transition_recall',0):.3f})")

    elapsed = time.time() - t0
    print(f"\n[DONE] Total time: {elapsed/60:.1f} minutes ({elapsed/3600:.1f} hours)")
    print(f"Completed: {dt.datetime.now().isoformat()}")

    # Save summary
    summary_path = Path(BASE_CFG["output_dir"]) / "variant_comparison.csv"
    pd.DataFrame(all_results).to_csv(summary_path, index=False)
    print(f"[SAVED] Summary: {summary_path}")


if __name__ == "__main__":
    main()
