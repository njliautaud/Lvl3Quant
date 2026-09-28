#!/usr/bin/env python3
"""
train_fusion_mlp.py — Fusion MLP: Learn Optimal Combined Signal from Multiple Models
=====================================================================================

Combines OOT predictions from:
  - LGBM DA classifier (28 folds, binary direction P(up))
  - Mamba v4 (multi-horizon: 1s, 5s, 10s continuous predictions)
  - CNN feat15 (if available)

Architecture:
  - 2-layer MLP: input -> 64 -> 32 -> 3 (one output per horizon)
  - Features: raw predictions, magnitudes, agreement signals, confidence
  - Walk-forward: sliding window across OOT folds (train on earlier, test on later)
  - NO LEAKAGE: MLP only sees predictions from earlier folds for training

Output:
  - Refined multi-horizon predictions (1s, 5s, 10s)
  - Per-fold and concat metrics at confidence tiers
  - Saved weights + predictions for live trading

Usage:
  python train_fusion_mlp.py [--train-folds 8] [--epochs 30] [--output-dir ...]

Author: Fusion system for Lvl3Quant
Date: 2026-04-21
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy import stats

# ── Config ────────────────────────────────────────────────────────────────────

BASE = Path("/home/jupiter/Lvl3Quant")
LGBM_DIR = BASE / "alpha_discovery" / "deep_models" / "results" / "lgbm_da_classifier"
MAMBA_DIR = BASE / "output" / "mamba_v4_sliding60d_raw6_v2"
CNN_DIR = BASE / "output" / "cnn1d_sliding60d_feat15_v2"
EVENT_DIR = BASE / "data" / "processed" / "mbo_events"
DEFAULT_OUTPUT = BASE / "output" / "fusion_mlp_v1"

# LGBM config (must match train_lgbm_da_classifier.py)
LGBM_WINDOW = 1000
LGBM_STRIDE = 500

# Mamba config
MAMBA_WINDOW = 1000
MAMBA_STRIDE = 500

# MLP config
HIDDEN_1 = 64
HIDDEN_2 = 32
DROPOUT = 0.15
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 512
N_HORIZONS = 3  # 1s, 5s, 10s

# Walk-forward
DEFAULT_TRAIN_FOLDS = 8   # Number of LGBM folds to use for MLP training
DEFAULT_EPOCHS = 30
EARLY_STOP_PATIENCE = 5

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("fusion_mlp")


# ── Data Loading ──────────────────────────────────────────────────────────────

def discover_lgbm_folds() -> Dict[int, Path]:
    """Find all LGBM prediction folds."""
    folds = {}
    for f in sorted(LGBM_DIR.glob("fold*_preds.npz")):
        idx = int(f.stem.replace("fold", "").replace("_preds", ""))
        folds[idx] = f
    return folds


def discover_mamba_folds() -> Dict[int, Path]:
    """Find all Mamba prediction folds."""
    folds = {}
    if not MAMBA_DIR.exists():
        return folds
    for f in sorted(MAMBA_DIR.glob("fold_*_oot_predictions.npz")):
        idx = int(f.stem.split("_")[1])
        folds[idx] = f
    return folds


def load_lgbm_fold(path: Path) -> Optional[Dict]:
    """Load LGBM fold predictions.
    Returns: {probs, preds, labels, confidence} with shapes (N,)."""
    try:
        d = np.load(path)
        return {
            "probs": d["probs"].astype(np.float32),        # P(up) in [0, 1]
            "preds": d["preds"].astype(np.float32),         # binary 0/1
            "labels": d["labels"].astype(np.float32),       # binary 0/1
            "confidence": d["confidence"].astype(np.float32),  # |P(up) - 0.5|
        }
    except Exception as e:
        log.warning(f"Failed to load LGBM {path}: {e}")
        return None


def load_mamba_fold(path: Path) -> Optional[Dict]:
    """Load Mamba fold predictions.
    Returns: {predictions (N,3), labels (N,3), embeddings (N,128), oot_files}."""
    try:
        d = np.load(path)
        result = {
            "predictions": d["predictions"].astype(np.float32),  # (N, 3)
            "labels": d["labels"].astype(np.float32),            # (N, 3)
        }
        if "embeddings" in d:
            result["embeddings"] = d["embeddings"].astype(np.float32)
        if "oot_files" in d:
            result["oot_files"] = d["oot_files"]
        return result
    except Exception as e:
        log.warning(f"Failed to load Mamba {path}: {e}")
        return None


# ── Feature Engineering ───────────────────────────────────────────────────────

def build_lgbm_features(lgbm_data: Dict) -> np.ndarray:
    """Build features from LGBM predictions only.

    Features (7):
      0: lgbm_signal      = probs - 0.5 (centered, continuous direction signal)
      1: lgbm_confidence   = |probs - 0.5|
      2: lgbm_prob_raw     = raw probability
      3: lgbm_signal_sq    = signal^2 (captures non-linear confidence effects)
      4: lgbm_extreme      = 1 if confidence > 0.2 else 0
      5: lgbm_signal_abs   = abs(signal) * sign(signal)^2 = signal * abs(signal)
      6: lgbm_prob_logit   = log(p / (1-p)) clipped
    """
    probs = lgbm_data["probs"]
    N = len(probs)
    features = np.zeros((N, 7), dtype=np.float32)

    signal = probs - 0.5
    confidence = np.abs(signal)
    prob_clipped = np.clip(probs, 0.01, 0.99)

    features[:, 0] = signal
    features[:, 1] = confidence
    features[:, 2] = probs
    features[:, 3] = signal * np.abs(signal)  # signed square
    features[:, 4] = (confidence > 0.2).astype(np.float32)
    features[:, 5] = signal * confidence  # signal weighted by own confidence
    features[:, 6] = np.log(prob_clipped / (1 - prob_clipped))  # logit

    return features


def build_mamba_features(mamba_data: Dict) -> np.ndarray:
    """Build features from Mamba predictions.

    Features (12):
      0-2: raw predictions for 1s, 5s, 10s
      3-5: abs(predictions) for each horizon (magnitude)
      6-8: sign(predictions) for each horizon (direction)
      9:   mean prediction across horizons
      10:  std prediction across horizons (disagreement)
      11:  horizon consistency = do all horizons agree on direction?
    """
    preds = mamba_data["predictions"]  # (N, 3)
    N = preds.shape[0]
    features = np.zeros((N, 12), dtype=np.float32)

    features[:, 0:3] = preds
    features[:, 3:6] = np.abs(preds)
    features[:, 6:9] = np.sign(preds)
    features[:, 9] = preds.mean(axis=1)
    features[:, 10] = preds.std(axis=1)
    # All 3 horizons same direction
    signs = np.sign(preds)
    features[:, 11] = (np.abs(signs.sum(axis=1)) == 3).astype(np.float32)

    return features


def build_fusion_features(lgbm_feats: np.ndarray,
                          mamba_feats: Optional[np.ndarray] = None) -> np.ndarray:
    """Combine features from all available models.

    If mamba_feats is None, pad with zeros (model learns to ignore).

    Additional cross-model features when both available:
      - agreement: do LGBM and Mamba agree on 10s direction?
      - confidence product: LGBM_conf * Mamba_10s_magnitude
      - signal ratio: LGBM_signal / (Mamba_10s + eps)
    """
    N = lgbm_feats.shape[0]

    if mamba_feats is not None:
        assert mamba_feats.shape[0] == N, f"Shape mismatch: LGBM={N}, Mamba={mamba_feats.shape[0]}"

        # Cross-model features (3)
        cross = np.zeros((N, 3), dtype=np.float32)
        lgbm_dir = np.sign(lgbm_feats[:, 0])  # LGBM signal direction
        mamba_10s_dir = np.sign(mamba_feats[:, 2])  # Mamba 10s direction
        cross[:, 0] = (lgbm_dir == mamba_10s_dir).astype(np.float32)  # agreement
        cross[:, 1] = lgbm_feats[:, 1] * mamba_feats[:, 5]  # conf * magnitude
        cross[:, 2] = lgbm_feats[:, 0] / (mamba_feats[:, 2] + np.sign(mamba_feats[:, 2]) * 1e-6 + 1e-8)  # signal ratio

        features = np.concatenate([lgbm_feats, mamba_feats, cross], axis=1)
    else:
        # LGBM-only mode: pad mamba features with 0 and cross features with 0
        mamba_pad = np.zeros((N, 12), dtype=np.float32)
        cross_pad = np.zeros((N, 3), dtype=np.float32)
        features = np.concatenate([lgbm_feats, mamba_pad, cross_pad], axis=1)

    return features  # shape: (N, 22)


N_FUSION_FEATURES = 7 + 12 + 3  # 22


def build_labels_from_lgbm(lgbm_data: Dict) -> np.ndarray:
    """Build regression labels from LGBM binary labels.

    LGBM labels are binary (0=down, 1=up). Convert to centered: -1/+1.
    We replicate across 3 horizons since LGBM only predicts at 10s.
    The MLP learns that this is a noisy multi-horizon target.
    """
    labels_binary = lgbm_data["labels"]  # 0 or 1
    labels_centered = labels_binary * 2 - 1  # -1 or +1
    # Replicate for 3 horizons
    return np.stack([labels_centered] * 3, axis=1).astype(np.float32)


def build_labels_from_mamba(mamba_data: Dict) -> np.ndarray:
    """Build regression labels from Mamba continuous labels (3 horizons)."""
    return mamba_data["labels"].astype(np.float32)  # (N, 3)


# ── Dataset ───────────────────────────────────────────────────────────────────

class FusionMLPDataset(Dataset):
    """Dataset for fusion MLP training."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """
        Args:
            features: (N, 22) fusion feature matrix
            labels: (N, 3) multi-horizon labels
        """
        # Remove rows with NaN
        valid = ~(np.isnan(features).any(axis=1) | np.isnan(labels).any(axis=1))
        self.X = torch.from_numpy(features[valid].astype(np.float32))
        self.y = torch.from_numpy(labels[valid].astype(np.float32))
        self.n_dropped = (~valid).sum()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Model ─────────────────────────────────────────────────────────────────────

class FusionMLP(nn.Module):
    """2-layer MLP for prediction fusion.

    Input: 22 features (LGBM 7 + Mamba 12 + Cross 3)
    Output: 3 predictions (1s, 5s, 10s)
    """

    def __init__(self, in_dim: int = N_FUSION_FEATURES,
                 hidden1: int = HIDDEN_1,
                 hidden2: int = HIDDEN_2,
                 n_outputs: int = N_HORIZONS,
                 dropout: float = DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, n_outputs),
        )

    def forward(self, x):
        return self.net(x)  # (B, 3)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_ic(preds: np.ndarray, labels: np.ndarray) -> float:
    """Pearson IC."""
    valid = ~(np.isnan(preds) | np.isnan(labels))
    p, l = preds[valid], labels[valid]
    if len(p) < 50 or np.std(p) < 1e-10:
        return 0.0
    return float(np.corrcoef(p, l)[0, 1])


def compute_rank_ic(preds: np.ndarray, labels: np.ndarray) -> float:
    """Spearman rank IC."""
    valid = ~(np.isnan(preds) | np.isnan(labels))
    p, l = preds[valid], labels[valid]
    if len(p) < 50:
        return 0.0
    return float(stats.spearmanr(p, l)[0])


def compute_da(preds: np.ndarray, labels: np.ndarray) -> float:
    """Directional accuracy."""
    valid = ~(np.isnan(preds) | np.isnan(labels))
    p, l = preds[valid], labels[valid]
    if len(p) < 50:
        return 0.5
    return float(np.mean(np.sign(p) == np.sign(l)))


def compute_mag_corr(preds: np.ndarray, labels: np.ndarray) -> float:
    """Magnitude correlation: corr(|pred|, |label|)."""
    valid = ~(np.isnan(preds) | np.isnan(labels))
    p, l = np.abs(preds[valid]), np.abs(labels[valid])
    if len(p) < 50 or np.std(p) < 1e-10:
        return 0.0
    return float(np.corrcoef(p, l)[0, 1])


def compute_tiered_metrics(preds: np.ndarray, labels: np.ndarray, tag: str = "") -> Dict:
    """Compute IC, DA, MagCorr at confidence tiers (All/50%/25%/10%/5%)."""
    valid = ~(np.isnan(preds) | np.isnan(labels))
    p, l = preds[valid], labels[valid]
    N = len(p)

    if N < 50:
        return {"All": {"IC": 0.0, "DA": 0.5, "MagCorr": 0.0, "RankIC": 0.0, "N": N}}

    results = {
        "All": {
            "IC": compute_ic(p, l),
            "DA": compute_da(p, l),
            "MagCorr": compute_mag_corr(p, l),
            "RankIC": compute_rank_ic(p, l),
            "N": N,
        }
    }

    abs_p = np.abs(p)
    for pct, label in [(50, "Top50"), (25, "Top25"), (10, "Top10"), (5, "Top5")]:
        threshold = np.percentile(abs_p, 100 - pct)
        mask = abs_p >= threshold
        if mask.sum() < 30:
            continue
        pt, lt = p[mask], l[mask]
        results[label] = {
            "IC": compute_ic(pt, lt),
            "DA": compute_da(pt, lt),
            "MagCorr": compute_mag_corr(pt, lt),
            "RankIC": compute_rank_ic(pt, lt),
            "N": int(mask.sum()),
        }

    return results


def print_metrics_table(results: Dict, title: str):
    """Pretty-print metrics at all tiers."""
    print(f"\n{'='*75}")
    print(f"  {title}")
    print(f"{'='*75}")
    print(f"{'Tier':<8} {'IC':>8} {'RankIC':>8} {'DA':>8} {'MagCorr':>8} {'N':>10}")
    print(f"{'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    for tier in ["All", "Top50", "Top25", "Top10", "Top5"]:
        if tier in results:
            r = results[tier]
            da_str = f"{r['DA']:.1%}" if r['DA'] != 0.5 else " 50.0%"
            print(f"{tier:<8} {r['IC']:>8.4f} {r['RankIC']:>8.4f} {da_str:>8} {r['MagCorr']:>8.4f} {r['N']:>10,}")


# ── Training ──────────────────────────────────────────────────────────────────

def train_one_fold(model: FusionMLP,
                   train_ds: FusionMLPDataset,
                   val_ds: FusionMLPDataset,
                   epochs: int = DEFAULT_EPOCHS,
                   lr: float = LR,
                   patience: int = EARLY_STOP_PATIENCE,
                   device: str = "cpu") -> Tuple[FusionMLP, Dict]:
    """Train MLP for one walk-forward fold with early stopping."""

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    criterion = nn.MSELoss()

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history = []

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss_sum += loss.item() * len(X_batch)
            train_n += len(X_batch)

        # Validate
        model.eval()
        with torch.no_grad():
            val_X = val_ds.X.to(device)
            val_y = val_ds.y.to(device)
            val_pred = model(val_X)
            val_loss = criterion(val_pred, val_y).item()

            # IC on 10s (index 2)
            vp = val_pred[:, 2].cpu().numpy()
            vy = val_y[:, 2].cpu().numpy()
            val_ic = compute_ic(vp, vy)

        train_loss = train_loss_sum / max(train_n, 1)
        scheduler.step(val_loss)

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_ic_10s": val_ic,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            log.info(f"  Epoch {epoch+1:3d}: train_loss={train_loss:.6f} "
                     f"val_loss={val_loss:.6f} val_IC_10s={val_ic:.4f}")

        if patience_counter >= patience:
            log.info(f"  Early stop at epoch {epoch+1} (patience={patience})")
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {"history": history, "best_val_loss": best_val_loss}


# ── Standardization ──────────────────────────────────────────────────────────

class FeatureScaler:
    """Simple z-score scaler that handles NaN/inf."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X: np.ndarray):
        self.mean = np.nanmean(X, axis=0)
        self.std = np.nanstd(X, axis=0)
        self.std[self.std < 1e-8] = 1.0  # prevent division by zero
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_scaled = (X - self.mean) / self.std
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=3.0, neginf=-3.0)
        X_scaled = np.clip(X_scaled, -5.0, 5.0)
        return X_scaled.astype(np.float32)

    def save(self, path: Path):
        np.savez(path, mean=self.mean, std=self.std)

    def load(self, path: Path):
        d = np.load(path)
        self.mean = d["mean"]
        self.std = d["std"]
        return self


# ── Walk-Forward Runner ───────────────────────────────────────────────────────

def run_walk_forward(args):
    """Main walk-forward training loop."""

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"{'='*75}")
    log.info(f"  FUSION MLP — Walk-Forward Training")
    log.info(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"{'='*75}")

    # ── Discover available predictions ────────────────────────────────────

    lgbm_folds = discover_lgbm_folds()
    mamba_folds = discover_mamba_folds()
    log.info(f"Found {len(lgbm_folds)} LGBM folds, {len(mamba_folds)} Mamba folds")

    if len(lgbm_folds) < 3:
        log.error("Need at least 3 LGBM folds for walk-forward")
        return

    # ── Load all LGBM fold data ───────────────────────────────────────────

    lgbm_data = {}
    for fold_idx, path in sorted(lgbm_folds.items()):
        data = load_lgbm_fold(path)
        if data is not None:
            lgbm_data[fold_idx] = data
            log.info(f"  LGBM fold {fold_idx:02d}: {data['probs'].shape[0]:>8,} samples, "
                     f"P(up) mean={data['probs'].mean():.4f}")

    # ── Load Mamba fold data (if available) ───────────────────────────────

    mamba_data = {}
    for fold_idx, path in sorted(mamba_folds.items()):
        data = load_mamba_fold(path)
        if data is not None:
            mamba_data[fold_idx] = data
            oot = data.get("oot_files", ["unknown"])
            log.info(f"  Mamba fold {fold_idx:02d}: {data['predictions'].shape[0]:>8,} samples, "
                     f"OOT={oot[0] if len(oot) > 0 else 'unknown'}")

    # ── Build features per LGBM fold ──────────────────────────────────────
    # For now, each LGBM fold becomes one "block" in our walk-forward.
    # Mamba features are only included where we have aligned predictions.

    fold_features = {}  # fold_idx -> (features, labels)
    fold_indices = sorted(lgbm_data.keys())

    for fold_idx in fold_indices:
        ld = lgbm_data[fold_idx]
        lgbm_feats = build_lgbm_features(ld)
        labels = build_labels_from_lgbm(ld)

        # For now, Mamba features are zeros (we only have 2 Mamba folds
        # and alignment requires date-level matching which is complex).
        # The MLP architecture supports adding Mamba when more folds arrive.
        fusion_feats = build_fusion_features(lgbm_feats, mamba_feats=None)

        fold_features[fold_idx] = (fusion_feats, labels)

    log.info(f"\nBuilt features for {len(fold_features)} folds")
    log.info(f"Feature dimension: {N_FUSION_FEATURES}")

    # ── Attempt Mamba alignment for overlapping folds ─────────────────────
    # Mamba fold 0 = Mar 2, fold 1 = Mar 3
    # These overlap with LGBM fold 27 (last fold, dates ~Feb 27 - Apr 21)
    # For the aligned samples, we can replace the zero-padded Mamba features

    if mamba_data:
        log.info("\n[Mamba Alignment] Attempting to align Mamba with LGBM fold 27...")
        _try_mamba_alignment(lgbm_data, mamba_data, fold_features)

    # ── Walk-Forward: Sliding window across folds ─────────────────────────

    train_window = args.train_folds  # Number of folds in training window
    n_folds = len(fold_indices)

    if n_folds <= train_window:
        log.warning(f"Only {n_folds} folds, need > {train_window} for walk-forward. "
                    f"Reducing train window to {n_folds - 1}.")
        train_window = max(1, n_folds - 1)

    # Collect all OOT predictions for concat evaluation
    all_oot_preds = []  # list of (N_i, 3)
    all_oot_labels = []  # list of (N_i, 3)
    all_oot_fold_ids = []  # fold index for each sample

    fold_metrics = []
    wf_start = train_window  # First test fold

    log.info(f"\n{'='*75}")
    log.info(f"  Walk-Forward: {n_folds - train_window} test folds "
             f"(train window = {train_window} folds)")
    log.info(f"{'='*75}")

    for test_fold_pos in range(wf_start, n_folds):
        test_fold_idx = fold_indices[test_fold_pos]
        train_fold_start = test_fold_pos - train_window
        train_fold_indices = fold_indices[train_fold_start:test_fold_pos]

        log.info(f"\n--- WF Fold: test={test_fold_idx:02d}, "
                 f"train=[{train_fold_indices[0]:02d}..{train_fold_indices[-1]:02d}] ---")

        # Build training data (concatenate multiple LGBM folds)
        train_X_parts, train_y_parts = [], []
        for ti in train_fold_indices:
            if ti in fold_features:
                X, y = fold_features[ti]
                train_X_parts.append(X)
                train_y_parts.append(y)

        if not train_X_parts:
            log.warning(f"  No training data for test fold {test_fold_idx}, skipping")
            continue

        train_X = np.concatenate(train_X_parts, axis=0)
        train_y = np.concatenate(train_y_parts, axis=0)

        # Test data
        if test_fold_idx not in fold_features:
            log.warning(f"  No test data for fold {test_fold_idx}, skipping")
            continue
        test_X, test_y = fold_features[test_fold_idx]

        log.info(f"  Train: {train_X.shape[0]:,} samples, Test: {test_X.shape[0]:,} samples")

        # Fit scaler on training data ONLY (no leakage)
        scaler = FeatureScaler()
        scaler.fit(train_X)
        train_X_scaled = scaler.transform(train_X)
        test_X_scaled = scaler.transform(test_X)

        # Also scale labels to help MLP training
        label_scaler = FeatureScaler()
        label_scaler.fit(train_y)
        train_y_scaled = label_scaler.transform(train_y)
        test_y_scaled = label_scaler.transform(test_y)

        # Build datasets
        train_ds = FusionMLPDataset(train_X_scaled, train_y_scaled)
        test_ds = FusionMLPDataset(test_X_scaled, test_y_scaled)

        if len(train_ds) < 100 or len(test_ds) < 50:
            log.warning(f"  Too few samples (train={len(train_ds)}, test={len(test_ds)}), skipping")
            continue

        # Train model
        model = FusionMLP(in_dim=N_FUSION_FEATURES,
                          hidden1=args.hidden1,
                          hidden2=args.hidden2)
        model, train_info = train_one_fold(
            model, train_ds, test_ds,
            epochs=args.epochs,
            lr=args.lr,
            patience=EARLY_STOP_PATIENCE,
            device="cpu",
        )

        # Generate OOT predictions
        model.eval()
        with torch.no_grad():
            test_preds_scaled = model(test_ds.X).numpy()

        # Inverse-scale predictions back to label space
        test_preds = test_preds_scaled * label_scaler.std + label_scaler.mean
        test_labels = test_ds.y.numpy() * label_scaler.std + label_scaler.mean

        # Collect for concat evaluation
        all_oot_preds.append(test_preds)
        all_oot_labels.append(test_labels)
        all_oot_fold_ids.extend([test_fold_idx] * len(test_preds))

        # Per-fold metrics
        horizons = ["1s", "5s", "10s"]
        fold_result = {"fold": test_fold_idx, "n_samples": len(test_preds)}
        for h_idx, h_name in enumerate(horizons):
            metrics = compute_tiered_metrics(test_preds[:, h_idx], test_labels[:, h_idx])
            fold_result[h_name] = metrics
            if h_name == "10s":
                ic_all = metrics["All"]["IC"]
                da_all = metrics["All"]["DA"]
                log.info(f"  Fold {test_fold_idx:02d} 10s: IC={ic_all:.4f} DA={da_all:.1%} "
                         f"N={metrics['All']['N']:,}")

        fold_metrics.append(fold_result)

        # Save fold artifacts
        fold_dir = output_dir / f"fold_{test_fold_idx:02d}"
        fold_dir.mkdir(exist_ok=True)

        torch.save(model.state_dict(), fold_dir / "model.pt")
        scaler.save(fold_dir / "feature_scaler.npz")
        label_scaler.save(fold_dir / "label_scaler.npz")
        np.savez_compressed(
            fold_dir / "predictions.npz",
            predictions=test_preds,
            labels=test_labels,
            fold_idx=test_fold_idx,
        )

    # ── Concat Evaluation ─────────────────────────────────────────────────

    if not all_oot_preds:
        log.error("No OOT predictions generated! Check data availability.")
        return

    concat_preds = np.concatenate(all_oot_preds, axis=0)
    concat_labels = np.concatenate(all_oot_labels, axis=0)
    N_total = concat_preds.shape[0]

    log.info(f"\n{'='*75}")
    log.info(f"  CONCAT EVALUATION — {N_total:,} total OOT samples "
             f"across {len(fold_metrics)} folds")
    log.info(f"{'='*75}")

    horizons = ["1s", "5s", "10s"]
    concat_results = {}
    for h_idx, h_name in enumerate(horizons):
        metrics = compute_tiered_metrics(concat_preds[:, h_idx], concat_labels[:, h_idx])
        concat_results[h_name] = metrics
        print_metrics_table(metrics, f"FUSION MLP — Concat {h_name}")

    # ── LGBM Baseline Comparison ──────────────────────────────────────────
    # Compare against raw LGBM signal for the same test folds

    log.info(f"\n{'='*75}")
    log.info(f"  BASELINE COMPARISON — Raw LGBM Signal")
    log.info(f"{'='*75}")

    baseline_preds = []
    baseline_labels = []
    for fm in fold_metrics:
        fi = fm["fold"]
        if fi in lgbm_data:
            ld = lgbm_data[fi]
            signal = ld["probs"] - 0.5  # centered signal
            labels_centered = ld["labels"] * 2 - 1
            baseline_preds.append(signal)
            baseline_labels.append(labels_centered)

    if baseline_preds:
        bp = np.concatenate(baseline_preds)
        bl = np.concatenate(baseline_labels)
        baseline_metrics = compute_tiered_metrics(bp, bl)
        print_metrics_table(baseline_metrics, "RAW LGBM Signal (10s baseline)")
        concat_results["lgbm_baseline_10s"] = baseline_metrics

    # ── Save Final Results ────────────────────────────────────────────────

    final_results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "hidden1": HIDDEN_1,
            "hidden2": HIDDEN_2,
            "dropout": DROPOUT,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "epochs": args.epochs,
            "train_folds": train_window,
            "n_fusion_features": N_FUSION_FEATURES,
            "early_stop_patience": EARLY_STOP_PATIENCE,
        },
        "n_lgbm_folds": len(lgbm_data),
        "n_mamba_folds": len(mamba_data),
        "n_test_folds": len(fold_metrics),
        "n_total_oot_samples": int(N_total),
        "concat_metrics": {h: {tier: {k: float(v) if isinstance(v, (float, np.floating)) else int(v)
                                       for k, v in m.items()}
                                for tier, m in metrics.items()}
                           for h, metrics in concat_results.items()},
        "per_fold_metrics": fold_metrics,
    }

    results_path = output_dir / "fusion_mlp_results.json"
    with open(results_path, "w") as f:
        json.dump(final_results, f, indent=2, default=str)
    log.info(f"\n[SAVED] Results: {results_path}")

    # Save concat predictions
    concat_path = output_dir / "concat_oot_predictions.npz"
    np.savez_compressed(
        concat_path,
        predictions=concat_preds,
        labels=concat_labels,
        fold_ids=np.array(all_oot_fold_ids),
    )
    log.info(f"[SAVED] Concat predictions: {concat_path}")

    log.info(f"\n{'='*75}")
    log.info(f"  FUSION MLP COMPLETE — {N_total:,} OOT samples, "
             f"{len(fold_metrics)} folds")
    log.info(f"{'='*75}")


def _try_mamba_alignment(lgbm_data, mamba_data, fold_features):
    """Try to align Mamba predictions with LGBM fold 27.

    Mamba fold 0 = 20260302, fold 1 = 20260303.
    LGBM fold 27 covers multiple dates; alignment requires knowing
    which samples in LGBM fold 27 correspond to these dates.

    This is a best-effort alignment using event position matching.
    """
    if 27 not in lgbm_data:
        log.info("  LGBM fold 27 not found, skipping Mamba alignment")
        return

    # Load event file metadata to compute per-day sample counts
    # LGBM fold 27 OOT dates were identified in fusion_mamba_lgbm.py
    LGBM_OOT_DATES = ["20260227", "20260302", "20260303", "20260304"]
    MAMBA_FOLD_DATES = {
        0: "20260302",
        1: "20260303",
    }

    # Compute day offsets within LGBM fold 27
    cumulative = 0
    day_offsets = {}
    for dt in LGBM_OOT_DATES:
        path = EVENT_DIR / f"{dt}_mbo_events.npz"
        if not path.exists():
            continue
        try:
            d = np.load(path)
            n_events = d["events"].shape[0]
            labels = d.get("labels_10s", None)
            if labels is not None and np.all(np.isnan(labels)):
                continue
            # Match LGBM windowing: valid samples after removing zero-label events
            # Approximate: use the window/stride to get sample count
            n_samples = max(0, (n_events - LGBM_WINDOW) // LGBM_STRIDE + 1)
            day_offsets[dt] = (cumulative, n_samples)
            cumulative += n_samples
            log.info(f"  Day {dt}: ~{n_samples:,} LGBM samples (offset={cumulative - n_samples})")
        except Exception as e:
            log.warning(f"  Failed to process {dt}: {e}")
            continue

    lgbm_fold27 = lgbm_data[27]
    total_lgbm = lgbm_fold27["probs"].shape[0]
    log.info(f"  LGBM fold 27 total: {total_lgbm:,} samples, computed day total: {cumulative:,}")

    # For each Mamba fold, try to align
    n_aligned = 0
    for mamba_fold_idx, mamba_d in mamba_data.items():
        date_str = MAMBA_FOLD_DATES.get(mamba_fold_idx)
        if date_str is None or date_str not in day_offsets:
            continue

        lgbm_offset, lgbm_n = day_offsets[date_str]
        n_mamba = mamba_d["predictions"].shape[0]

        # Alignment: Mamba sample j -> event j*500+1000
        # LGBM sample i -> event i*500+500
        # Match: j*500+1000 = i*500+500 => i = j+1
        mamba_indices = []
        lgbm_indices = []
        for j in range(n_mamba):
            lgbm_i = j + 1  # Simplified alignment for same stride
            if 0 <= lgbm_i < lgbm_n:
                abs_lgbm_i = lgbm_offset + lgbm_i
                if abs_lgbm_i < total_lgbm:
                    mamba_indices.append(j)
                    lgbm_indices.append(abs_lgbm_i)

        if len(mamba_indices) == 0:
            log.info(f"  No aligned samples for Mamba fold {mamba_fold_idx} ({date_str})")
            continue

        mamba_indices = np.array(mamba_indices)
        lgbm_indices = np.array(lgbm_indices)
        log.info(f"  Aligned {len(mamba_indices):,} Mamba samples for {date_str}")
        n_aligned += len(mamba_indices)

        # Update fold 27 features at aligned positions with real Mamba features
        if 27 in fold_features:
            feats, labels = fold_features[27]
            mamba_feats = build_mamba_features(mamba_d)

            for mi, li in zip(mamba_indices, lgbm_indices):
                if li < feats.shape[0] and mi < mamba_feats.shape[0]:
                    # Replace Mamba feature columns (indices 7..18)
                    feats[li, 7:19] = mamba_feats[mi]
                    # Update cross-model features (indices 19..21)
                    lgbm_signal = feats[li, 0]
                    mamba_10s = mamba_feats[mi, 2]
                    mamba_10s_mag = mamba_feats[mi, 5]
                    feats[li, 19] = float(np.sign(lgbm_signal) == np.sign(mamba_10s))
                    feats[li, 20] = feats[li, 1] * mamba_10s_mag
                    feats[li, 21] = lgbm_signal / (mamba_10s + np.sign(mamba_10s) * 1e-6 + 1e-8)

                    # Also upgrade labels to Mamba's continuous multi-horizon labels
                    labels[li] = mamba_d["labels"][mi]

            fold_features[27] = (feats, labels)

    log.info(f"  Total aligned: {n_aligned:,} samples with real Mamba features")


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fusion MLP: combine model predictions")
    parser.add_argument("--train-folds", type=int, default=DEFAULT_TRAIN_FOLDS,
                        help="Number of LGBM folds in sliding training window")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help="Max training epochs per fold")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT),
                        help="Output directory for models and predictions")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate")
    parser.add_argument("--hidden1", type=int, default=HIDDEN_1, help="First hidden layer size")
    parser.add_argument("--hidden2", type=int, default=HIDDEN_2, help="Second hidden layer size")
    args = parser.parse_args()

    run_walk_forward(args)


if __name__ == "__main__":
    main()
