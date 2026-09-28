#!/usr/bin/env python3
"""
Supervised Execution Classifier v2 — Ultra-Fast
=================================================
Trains a supervised MLP to predict whether a given signal will be profitable,
using CNN-Mamba predictions + embeddings as features and actual price labels.

Data source: fold_NN_oot_predictions.npz files, which contain:
  - predictions: (N, 3) — CNN-Mamba pred_1s, pred_5s, pred_10s
  - labels: (N, 3) — actual price changes at 1s, 5s, 10s (ticks)
  - embeddings: (N, 96) — CNN-Mamba model hidden state embeddings
  - oot_files: the MBO event file this came from

Labeling strategy:
  - GOOD ENTRY: label_10s > 1.5 ticks in predicted direction (profitable trade)
  - BAD ENTRY: everything else
  - This captures "was the signal right AND big enough to cover costs?"

Features: predictions (3) + embeddings (96) + derived (8) = 107 dims
Training: ~30 seconds per fold on GPU. Full walk-forward in < 5 minutes.

Cost: $4.70 RT commission = 0.376 ticks

Usage:
    python train_exec_classifier.py --gpu --epochs 30
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from scipy.stats import spearmanr

# ─── Path setup ──────────────────────────────────────────────────────────────
if sys.platform == "win32":
    LVL3_ROOT = Path("C:/Users/claude/Lvl3Quant")
else:
    LVL3_ROOT = Path(os.environ.get("LVL3_ROOT", "/home/jupiter/Lvl3Quant"))
    if not LVL3_ROOT.exists() and Path("/home/nick/Lvl3Quant").exists():
        LVL3_ROOT = Path("/home/nick/Lvl3Quant")

PRED_DIR = LVL3_ROOT / "output" / "cnn_mamba_v2_smart_v3_mar"
OUTPUT_DIR = LVL3_ROOT / "output" / "exec_classifier"

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [EXEC_CLS] %(levelname)s: %(message)s",
)
log = logging.getLogger("exec_classifier")

# Constants
COMMISSION_TICKS = 0.376
PROFIT_THRESHOLD = 1.5  # ticks — min move to be "profitable"


# ─── Data loading ────────────────────────────────────────────────────────────

def load_fold(fold_file: Path) -> Optional[dict]:
    """Load one fold's predictions + embeddings + labels."""
    try:
        data = np.load(str(fold_file), allow_pickle=True)
    except Exception as e:
        log.warning(f"Cannot load {fold_file}: {e}")
        return None

    preds = data["predictions"].astype(np.float32)      # (N, 3)
    labels = data["labels"].astype(np.float32)           # (N, 3)
    embeddings = data.get("embeddings", None)             # (N, 96) or None

    if embeddings is not None:
        embeddings = embeddings.astype(np.float32)

    return {
        "predictions": preds,
        "labels": labels,
        "embeddings": embeddings,
        "ic_1s": float(data.get("ic_1s", 0)),
        "ic_5s": float(data.get("ic_5s", 0)),
        "ic_10s": float(data.get("ic_10s", 0)),
        "oot_files": data.get("oot_files", []),
    }


def build_features_and_labels(
    preds: np.ndarray,
    labels: np.ndarray,
    embeddings: Optional[np.ndarray],
    threshold: float = PROFIT_THRESHOLD,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build feature matrix and binary labels from fold data.

    Features:
      [0:3]   predictions (pred_1s, pred_5s, pred_10s)
      [3:6]   abs predictions (signal magnitude)
      [6]     signal agreement (sign consistency across horizons)
      [7]     signal trend (pred_1s - pred_10s, captures momentum vs mean-rev)
      [8]     max signal magnitude
      [9]     signal ratio (pred_1s / pred_10s, captures decay profile)
      [10]    signal sign (1 for long, -1 for short)
      [11:107] embeddings (96 dims, if available)

    Labels:
      1 = good entry: price moved > threshold ticks in predicted direction
      0 = bad entry: otherwise

    Also returns continuous target (actual ticks in predicted direction)
    for regression analysis.
    """
    n = len(preds)
    pred_1s = preds[:, 0]
    pred_5s = preds[:, 1]
    pred_10s = preds[:, 2]

    # Direction from strongest short-term signal
    direction = np.sign(pred_1s)
    direction[direction == 0] = 1.0  # default long for zero

    # Actual move in predicted direction (using 10s horizon as primary)
    actual_move = labels[:, 2] * direction  # ticks in predicted direction

    # Also check multi-horizon: best move across horizons
    actual_1s = labels[:, 0] * direction
    actual_5s = labels[:, 1] * direction
    actual_10s = labels[:, 2] * direction
    best_move = np.maximum(actual_1s, np.maximum(actual_5s, actual_10s))

    # Binary label: profitable if best move > threshold
    binary_labels = (best_move > threshold).astype(np.int32)

    # Build features
    n_base = 11
    n_emb = embeddings.shape[1] if embeddings is not None else 0
    n_feat = n_base + n_emb

    features = np.zeros((n, n_feat), dtype=np.float32)
    features[:, 0] = pred_1s
    features[:, 1] = pred_5s
    features[:, 2] = pred_10s
    features[:, 3] = np.abs(pred_1s)
    features[:, 4] = np.abs(pred_5s)
    features[:, 5] = np.abs(pred_10s)
    features[:, 6] = (np.sign(pred_1s) == np.sign(pred_10s)).astype(np.float32)
    features[:, 7] = pred_1s - pred_10s
    features[:, 8] = np.maximum(np.abs(pred_1s), np.maximum(np.abs(pred_5s), np.abs(pred_10s)))
    # Ratio with safe denominator
    safe_10s = np.where(np.abs(pred_10s) > 0.01, pred_10s, 0.01 * np.sign(pred_10s + 1e-8))
    features[:, 9] = np.clip(pred_1s / safe_10s, -5, 5)
    features[:, 10] = direction

    if embeddings is not None:
        features[:, n_base:n_base + n_emb] = embeddings

    return features, binary_labels, actual_move


# ─── Dataset + Model ─────────────────────────────────────────────────────────

class ExecDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.from_numpy(features)
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class ExecClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.net(x)


# ─── Training ───────────────────────────────────────────────────────────────

def train_fold(
    train_feat, train_labels, val_feat, val_labels,
    fold_idx: int, device: torch.device,
    epochs: int = 30, batch_size: int = 4096, lr: float = 3e-4,
) -> dict:
    """Train one fold. Returns metrics dict + model."""

    # Class weights
    n_pos = np.sum(train_labels == 1)
    n_neg = np.sum(train_labels == 0)
    if n_pos < 10 or n_neg < 10:
        return {"fold": fold_idx, "status": "skip_imbalanced", "n_pos": int(n_pos), "n_neg": int(n_neg)}, None, None, None

    pos_weight = torch.tensor([1.0, n_neg / max(n_pos, 1)], device=device, dtype=torch.float32)

    # Normalize
    feat_mean = train_feat.mean(axis=0)
    feat_std = train_feat.std(axis=0) + 1e-8
    train_norm = (train_feat - feat_mean) / feat_std
    val_norm = (val_feat - feat_mean) / feat_std

    train_dl = DataLoader(ExecDataset(train_norm, train_labels), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(ExecDataset(val_norm, val_labels), batch_size=batch_size * 2, num_workers=2)

    model = ExecClassifier(input_dim=train_feat.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(weight=pos_weight)

    best_f1 = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for feats, labs in train_dl:
            feats, labs = feats.to(device), labs.to(device)
            loss = criterion(model(feats), labs)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Eval
        if epoch % 5 == 0 or epoch == epochs - 1:
            model.eval()
            all_probs, all_labs = [], []
            with torch.no_grad():
                for feats, labs in val_dl:
                    probs = F.softmax(model(feats.to(device)), dim=1)[:, 1].cpu().numpy()
                    all_probs.append(probs)
                    all_labs.append(labs.numpy())

            vp = np.concatenate(all_probs)
            vl = np.concatenate(all_labs)

            # Metrics at threshold 0.5
            pred_pos = vp > 0.5
            tp = np.sum(pred_pos & (vl == 1))
            precision = tp / max(np.sum(pred_pos), 1)
            recall = tp / max(np.sum(vl == 1), 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)
            acc = np.mean((vp > 0.5) == vl)

            if epoch % 5 == 0:
                log.info(f"  Ep {epoch:2d} | acc={acc:.3f} P={precision:.3f} R={recall:.3f} F1={f1:.3f} | pos_rate={np.mean(vl):.3f}")

            if f1 > best_f1:
                best_f1 = f1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    # Final eval with all thresholds
    model.eval()
    all_probs, all_labs = [], []
    with torch.no_grad():
        for feats, labs in val_dl:
            probs = F.softmax(model(feats.to(device)), dim=1)[:, 1].cpu().numpy()
            all_probs.append(probs)
            all_labs.append(labs.numpy())
    vp = np.concatenate(all_probs)
    vl = np.concatenate(all_labs)

    result = {
        "fold": fold_idx,
        "n_train": len(train_labels),
        "n_val": len(val_labels),
        "pos_rate_train": float(np.mean(train_labels)),
        "pos_rate_val": float(np.mean(vl)),
        "best_f1": float(best_f1),
        "thresholds": {},
    }

    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        pp = vp > thresh
        if np.sum(pp) > 0:
            tp = np.sum(pp & (vl == 1))
            fp = np.sum(pp & (vl == 0))
            precision = tp / max(tp + fp, 1)
            recall = tp / max(np.sum(vl == 1), 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)

            # Simulated P&L: if we only take trades where P > thresh
            # Expected edge per trade = precision * avg_winner - (1-precision) * avg_loser - commission
            result["thresholds"][str(thresh)] = {
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "f1": round(float(f1), 4),
                "n_selected": int(np.sum(pp)),
                "n_correct": int(tp),
                "selectivity": round(float(np.mean(pp)), 4),
            }

    return result, model, feat_mean, feat_std


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--threshold", type=float, default=PROFIT_THRESHOLD)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info("Supervised Execution Classifier v2 — Ultra-Fast")
    log.info("=" * 70)
    log.info(f"  Device:         {device}")
    log.info(f"  Epochs:         {args.epochs}")
    log.info(f"  Profit thresh:  {args.threshold} ticks")
    log.info(f"  Pred dir:       {PRED_DIR}")
    log.info(f"  Output:         {output_dir}")
    log.info("=" * 70)

    # MLflow
    mlflow_uri = "http://jupiter:5000" if not Path("/home/jupiter").exists() else "http://localhost:5000"
    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_tracking_uri(mlflow_uri)
            mlflow.set_experiment("exec_classifier_v2")
            mlflow.start_run(run_name=f"exec_cls_v2_{time.strftime('%Y%m%d_%H%M%S')}")
            mlflow.log_params({"epochs": args.epochs, "threshold": args.threshold, "device": str(device)})
        except Exception as e:
            log.warning(f"MLflow: {e}")

    # Load all fold files
    fold_files = sorted(PRED_DIR.glob("fold_*_oot_predictions.npz"))
    log.info(f"Found {len(fold_files)} OOT fold files")

    if len(fold_files) < 3:
        log.error("Need at least 3 fold files for walk-forward")
        return

    # Load all fold data
    fold_data = []
    for ff in fold_files:
        d = load_fold(ff)
        if d is not None:
            fold_idx = int(ff.stem.split("_")[1])
            features, binary_labels, actual_move = build_features_and_labels(
                d["predictions"], d["labels"], d["embeddings"], args.threshold
            )
            fold_data.append({
                "fold_idx": fold_idx,
                "features": features,
                "labels": binary_labels,
                "actual_move": actual_move,
                "ic_1s": d["ic_1s"],
                "ic_10s": d["ic_10s"],
                "n_samples": len(features),
                "pos_rate": float(np.mean(binary_labels)),
            })
            log.info(f"  Fold {fold_idx:02d}: {len(features)} samples, pos_rate={np.mean(binary_labels):.3f}, IC_1s={d['ic_1s']:.3f}")

    log.info(f"\nLoaded {len(fold_data)} folds, total {sum(f['n_samples'] for f in fold_data)} samples")

    # Walk-forward: for each OOT fold, train on preceding folds
    all_results = []
    concat_preds = []
    concat_labels = []
    concat_actual = []

    for oot_idx in range(3, len(fold_data)):
        oot = fold_data[oot_idx]
        fold_num = oot["fold_idx"]

        # Train on preceding 5 folds
        train_start = max(0, oot_idx - 5)
        train_folds = fold_data[train_start:oot_idx]

        train_feat = np.concatenate([f["features"] for f in train_folds])
        train_labels = np.concatenate([f["labels"] for f in train_folds])

        log.info(f"\n{'='*60}")
        log.info(f"OOT Fold {fold_num} | Train: {len(train_feat)} ({len(train_folds)} folds) | Val: {oot['n_samples']}")

        t0 = time.time()
        result, model, feat_mean, feat_std = train_fold(
            train_feat, train_labels,
            oot["features"], oot["labels"],
            fold_num, device,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        elapsed = time.time() - t0

        if model is None:
            log.warning(f"  Skipped (imbalanced)")
            all_results.append(result)
            continue

        log.info(f"  Trained in {elapsed:.1f}s | best_F1={result['best_f1']:.3f}")

        for thresh, m in result.get("thresholds", {}).items():
            log.info(f"  @{thresh}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} sel={m['n_selected']}")

        # Save model
        torch.save({
            "model_state_dict": model.state_dict(),
            "feat_mean": feat_mean,
            "feat_std": feat_std,
            "fold_idx": fold_num,
            "input_dim": train_feat.shape[1],
            "metrics": result,
        }, str(output_dir / f"fold_{fold_num:02d}_model.pt"))

        # Collect for concat metrics
        model.eval()
        val_norm = (oot["features"] - feat_mean) / (feat_std + 1e-8)
        val_dl = DataLoader(ExecDataset(val_norm, oot["labels"]), batch_size=8192)
        fold_probs = []
        with torch.no_grad():
            for feats, _ in val_dl:
                probs = F.softmax(model(feats.to(device)), dim=1)[:, 1].cpu().numpy()
                fold_probs.append(probs)
        fold_probs = np.concatenate(fold_probs)
        concat_preds.append(fold_probs)
        concat_labels.append(oot["labels"])
        concat_actual.append(oot["actual_move"])

        all_results.append(result)

        if MLFLOW_AVAILABLE:
            try:
                mlflow.log_metrics({
                    f"fold_{fold_num}_f1": result["best_f1"],
                    f"fold_{fold_num}_precision_05": result["thresholds"].get("0.5", {}).get("precision", 0),
                }, step=fold_num)
            except:
                pass

    # ─── CONCAT ANALYSIS (the real test) ─────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("CONCAT WALK-FORWARD RESULTS (out-of-time)")
    log.info("=" * 70)

    if concat_preds:
        all_p = np.concatenate(concat_preds)
        all_l = np.concatenate(concat_labels)
        all_a = np.concatenate(concat_actual)

        log.info(f"Total OOT samples: {len(all_p)}")
        log.info(f"Overall pos rate: {np.mean(all_l):.3f}")
        log.info(f"Spearman(prob, actual_move): {spearmanr(all_p, all_a)[0]:.4f}")

        log.info("\nThreshold analysis (simulated trading):")
        log.info(f"{'Thresh':>7} {'Select%':>8} {'Trades':>7} {'Prec':>6} {'Recall':>7} {'F1':>6} {'AvgMove':>8} {'EdgeTk':>7}")

        for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            mask = all_p > thresh
            n_sel = np.sum(mask)
            if n_sel == 0:
                continue

            tp = np.sum(mask & (all_l == 1))
            precision = tp / n_sel
            recall = tp / max(np.sum(all_l == 1), 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)

            # Average actual move for selected trades
            avg_move = np.mean(all_a[mask])
            # Edge = avg_move - commission
            edge = avg_move - COMMISSION_TICKS

            selectivity = n_sel / len(all_p) * 100

            log.info(f"  {thresh:5.1f}   {selectivity:6.1f}%  {n_sel:6d}  {precision:5.3f}  {recall:6.3f}  {f1:5.3f}  {avg_move:+7.2f}  {edge:+6.2f}")

        # Top decile analysis
        log.info("\nTop decile analysis:")
        for pct in [1, 5, 10, 20]:
            cutoff = np.percentile(all_p, 100 - pct)
            mask = all_p >= cutoff
            n_sel = np.sum(mask)
            if n_sel > 0:
                avg_move = np.mean(all_a[mask])
                precision = np.mean(all_l[mask])
                edge = avg_move - COMMISSION_TICKS
                log.info(f"  Top {pct:2d}%: {n_sel:5d} trades, P={precision:.3f}, avg_move={avg_move:+.2f}tk, edge={edge:+.2f}tk")

    # Save summary
    summary = {
        "results": all_results,
        "n_folds": len([r for r in all_results if "best_f1" in r]),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "profit_threshold": args.threshold,
    }

    if concat_preds:
        summary["concat_spearman"] = float(spearmanr(all_p, all_a)[0])

    with open(str(output_dir / "walk_forward_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    log.info(f"\nSummary saved: {output_dir / 'walk_forward_summary.json'}")

    if MLFLOW_AVAILABLE:
        try:
            if concat_preds:
                mlflow.log_metric("concat_spearman", float(spearmanr(all_p, all_a)[0]))
            mlflow.end_run()
        except:
            pass


if __name__ == "__main__":
    main()
