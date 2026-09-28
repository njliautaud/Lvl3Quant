#!/usr/bin/env python3
"""
Confluence-Aware Meta-Model v4
==============================
MLP (256->128->64, dropout 0.2) that combines multi-head predictions from
CNN-Mamba v2 + PatchTST + MFE/MAE features to predict realized 5s price moves.

Key improvements over v3:
  - Cross-model confluence features (CM x PatchTST agreement)
  - MFE/MAE as INPUT features (not target) — model learns which MFE patterns
    lead to profitable moves
  - Head entropy & triplet agreement features
  - 36 engineered features total

Walk-forward: 15-date sliding train, 1-date OOT.
Target: CM label_5s (actual 5s price move in ticks).

Runs on Razer (RTX 3070 8GB). Logs to MLflow on Jupiter.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Optional: MLflow
try:
    import mlflow
    import mlflow.pytorch
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

# Optional: sklearn StandardScaler
try:
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ============================================================================
# Constants
# ============================================================================
ES_RT_COMMISSION_TICKS = 0.376  # AMP round-trip commission in ticks

# Auto-detect platform
if sys.platform == 'win32':
    _BASE = Path(r'C:\Users\claude\Lvl3Quant')
else:
    _BASE = Path('/home/jupiter/Lvl3Quant')

DEFAULT_CM_DIR   = _BASE / 'output' / 'cnn_mamba_v2_bulk_oot'
DEFAULT_MFE_DIR  = _BASE / 'output' / 'mfe_mae_labels_v1'
DEFAULT_PT_DIR   = _BASE / 'output' / 'patchtst_bulk_oot'
DEFAULT_OUT_DIR  = _BASE / 'output' / 'confluence_meta_v4'

MLFLOW_URI = 'http://jupiter:5000'
EXPERIMENT_NAME = 'confluence-meta-v4'


# ============================================================================
# Model
# ============================================================================
class ConfluenceMetaMLP(nn.Module):
    """MLP predicting 5s price move from confluence features."""

    def __init__(self, input_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ============================================================================
# Feature Engineering
# ============================================================================
def compute_head_entropy(signs):
    """Entropy of sign distribution across heads. 0 = all agree."""
    n = len(signs)
    if n == 0:
        return 0.0
    pos = np.sum(signs > 0, axis=-1).astype(np.float32)
    neg = n - pos
    p_pos = pos / n
    p_neg = neg / n
    # Handle log(0) safely
    ent = np.zeros_like(p_pos)
    mask_pos = p_pos > 0
    mask_neg = p_neg > 0
    ent[mask_pos] -= p_pos[mask_pos] * np.log2(p_pos[mask_pos])
    ent[mask_neg] -= p_neg[mask_neg] * np.log2(p_neg[mask_neg])
    return ent


def build_features(cm_preds, pt_preds, mfe_data):
    """
    Build 36-dim feature vector from multi-head predictions + MFE/MAE.

    Args:
        cm_preds: (N, 3) CNN-Mamba predictions [1s, 5s, 10s]
        pt_preds: (N, 3) PatchTST predictions [1s, 5s, 10s] or None
        mfe_data: dict with mfe_1s, mae_1s, ... keys (N,) each, or None

    Returns:
        features: (N, 36) float32 array
    """
    N = cm_preds.shape[0]

    if pt_preds is None:
        pt_preds = np.zeros((N, 3), dtype=np.float32)

    # --- Base features (9) ---
    cm_1s, cm_5s, cm_10s = cm_preds[:, 0], cm_preds[:, 1], cm_preds[:, 2]
    pt_1s, pt_5s, pt_10s = pt_preds[:, 0], pt_preds[:, 1], pt_preds[:, 2]

    # --- Pair confluence (9) ---
    sign_agree_cm_1s_5s  = (np.sign(cm_1s) == np.sign(cm_5s)).astype(np.float32)
    sign_agree_cm_1s_10s = (np.sign(cm_1s) == np.sign(cm_10s)).astype(np.float32)
    sign_agree_cm_5s_10s = (np.sign(cm_5s) == np.sign(cm_10s)).astype(np.float32)

    sign_agree_cross_1s = (np.sign(cm_1s) == np.sign(pt_1s)).astype(np.float32)
    sign_agree_cross_5s = (np.sign(cm_5s) == np.sign(pt_5s)).astype(np.float32)
    sign_agree_cross_10s = (np.sign(cm_10s) == np.sign(pt_10s)).astype(np.float32)

    mag_ratio_1s_5s = np.clip(
        np.abs(cm_1s) / (np.abs(cm_5s) + 1e-8), 0, 5
    )
    mag_ratio_5s_10s = np.clip(
        np.abs(cm_5s) / (np.abs(cm_10s) + 1e-8), 0, 5
    )

    spread_cm = np.max(cm_preds, axis=1) - np.min(cm_preds, axis=1)

    # --- Triplet confluence (3) ---
    cm_signs = np.sign(cm_preds)  # (N, 3)
    all_3_agree_cm = (
        (cm_signs[:, 0] == cm_signs[:, 1]) &
        (cm_signs[:, 1] == cm_signs[:, 2])
    ).astype(np.float32)

    all_signs = np.column_stack([np.sign(cm_preds), np.sign(pt_preds)])  # (N, 6)
    all_6_agree = np.all(all_signs == all_signs[:, :1], axis=1).astype(np.float32)

    head_entropy = compute_head_entropy(all_signs)

    # --- Confidence (3) ---
    abs_cm_1s  = np.abs(cm_1s)
    abs_cm_5s  = np.abs(cm_5s)
    abs_cm_10s = np.abs(cm_10s)

    # --- MFE/MAE features (12) ---
    if mfe_data is not None:
        mfe_1s  = mfe_data['mfe_1s']
        mae_1s  = mfe_data['mae_1s']
        mfe_5s  = mfe_data['mfe_5s']
        mae_5s  = mfe_data['mae_5s']
        mfe_10s = mfe_data['mfe_10s']
        mae_10s = mfe_data['mae_10s']
        mfe_30s = mfe_data['mfe_30s']
        mae_30s = mfe_data['mae_30s']
        edge_ratio_5s  = mfe_5s / (mae_5s + 1e-8)
        edge_ratio_10s = mfe_10s / (mae_10s + 1e-8)
        mfe_skew_5s  = mfe_5s - mae_5s
        mfe_skew_10s = mfe_10s - mae_10s
    else:
        mfe_1s = mae_1s = mfe_5s = mae_5s = np.zeros(N, dtype=np.float32)
        mfe_10s = mae_10s = mfe_30s = mae_30s = np.zeros(N, dtype=np.float32)
        edge_ratio_5s = edge_ratio_10s = np.zeros(N, dtype=np.float32)
        mfe_skew_5s = mfe_skew_10s = np.zeros(N, dtype=np.float32)

    # Stack all 36 features
    features = np.column_stack([
        # Base (9)
        cm_1s, cm_5s, cm_10s,
        pt_1s, pt_5s, pt_10s,
        # Pair confluence (9)
        sign_agree_cm_1s_5s, sign_agree_cm_1s_10s, sign_agree_cm_5s_10s,
        sign_agree_cross_1s, sign_agree_cross_5s, sign_agree_cross_10s,
        mag_ratio_1s_5s, mag_ratio_5s_10s,
        spread_cm,
        # Triplet confluence (3)
        all_3_agree_cm, all_6_agree, head_entropy,
        # Confidence (3)
        abs_cm_1s, abs_cm_5s, abs_cm_10s,
        # MFE/MAE (12)
        mfe_1s, mae_1s, mfe_5s, mae_5s, mfe_10s, mae_10s, mfe_30s, mae_30s,
        edge_ratio_5s, edge_ratio_10s, mfe_skew_5s, mfe_skew_10s,
    ]).astype(np.float32)

    return features


FEATURE_NAMES = [
    'cm_pred_1s', 'cm_pred_5s', 'cm_pred_10s',
    'pt_pred_1s', 'pt_pred_5s', 'pt_pred_10s',
    'sign_agree_cm_1s_5s', 'sign_agree_cm_1s_10s', 'sign_agree_cm_5s_10s',
    'sign_agree_cross_1s', 'sign_agree_cross_5s', 'sign_agree_cross_10s',
    'mag_ratio_1s_5s', 'mag_ratio_5s_10s', 'spread_cm',
    'all_3_agree_cm', 'all_6_agree', 'head_entropy',
    'abs_cm_1s', 'abs_cm_5s', 'abs_cm_10s',
    'mfe_1s', 'mae_1s', 'mfe_5s', 'mae_5s',
    'mfe_10s', 'mae_10s', 'mfe_30s', 'mae_30s',
    'edge_ratio_5s', 'edge_ratio_10s', 'mfe_skew_5s', 'mfe_skew_10s',
]
assert len(FEATURE_NAMES) == 33, f"Expected 33 feature names, got {len(FEATURE_NAMES)}"


# ============================================================================
# Data Loading
# ============================================================================
def discover_dates(cm_dir, mfe_dir, pt_dir):
    """Find dates where all 3 data sources overlap."""
    cm_dates = set()
    for f in sorted(Path(cm_dir).glob('*_predictions.npz')):
        date_str = f.stem.replace('_predictions', '')
        cm_dates.add(date_str)

    mfe_dates = set()
    for f in sorted(Path(mfe_dir).glob('*_mfe_mae.npz')):
        date_str = f.stem.replace('_mfe_mae', '')
        mfe_dates.add(date_str)

    pt_dates = set()
    for f in sorted(Path(pt_dir).glob('*_predictions.npz')):
        date_str = f.stem.replace('_predictions', '')
        pt_dates.add(date_str)

    # Intersection of all 3
    overlap_all3 = sorted(cm_dates & mfe_dates & pt_dates)
    # Also allow CM + MFE (without PT) — we zero-fill PT
    overlap_cm_mfe = sorted(cm_dates & mfe_dates)

    print(f"[DATA] CM dates: {len(cm_dates)}, MFE dates: {len(mfe_dates)}, "
          f"PT dates: {len(pt_dates)}", flush=True)
    print(f"[DATA] All-3 overlap: {len(overlap_all3)} dates", flush=True)
    print(f"[DATA] CM+MFE overlap (PT optional): {len(overlap_cm_mfe)} dates", flush=True)

    return overlap_cm_mfe, pt_dates


def load_day(date_str, cm_dir, mfe_dir, pt_dir, has_pt):
    """
    Load one day of data. Returns (features, labels) or None on failure.

    Alignment strategy:
      - CM predictions are the base (N samples)
      - MFE has denser stride (M >> N): subsample with linspace indices
      - PatchTST has ~same count as CM: use directly or zero-fill
    """
    cm_file = Path(cm_dir) / f'{date_str}_predictions.npz'
    mfe_file = Path(mfe_dir) / f'{date_str}_mfe_mae.npz'

    if not cm_file.exists():
        print(f"  [SKIP] No CM file for {date_str}", flush=True)
        return None
    if not mfe_file.exists():
        print(f"  [SKIP] No MFE file for {date_str}", flush=True)
        return None

    try:
        cm_data = np.load(cm_file, allow_pickle=True)
        cm_preds = cm_data['predictions'].astype(np.float32)   # (N, 3)
        cm_labels = cm_data['labels'].astype(np.float32)         # (N, 3)
        N = cm_preds.shape[0]

        if N < 100:
            print(f"  [SKIP] {date_str}: only {N} CM samples", flush=True)
            return None

        # Load MFE/MAE and subsample to match CM count
        mfe_raw = np.load(mfe_file, allow_pickle=True)
        M = len(mfe_raw['mfe_5s'])

        if M >= N:
            # Subsample MFE to match CM count
            indices = np.linspace(0, M - 1, N, dtype=int)
        else:
            # MFE is smaller — take what we have and pad
            indices = np.arange(M)

        mfe_keys = ['mfe_1s', 'mae_1s', 'mfe_5s', 'mae_5s',
                     'mfe_10s', 'mae_10s', 'mfe_30s', 'mae_30s']

        mfe_data = {}
        for key in mfe_keys:
            if key in mfe_raw:
                arr = mfe_raw[key].astype(np.float32)
                if M >= N:
                    mfe_data[key] = arr[indices]
                else:
                    # Pad with zeros
                    padded = np.zeros(N, dtype=np.float32)
                    padded[:M] = arr
                    mfe_data[key] = padded
            else:
                mfe_data[key] = np.zeros(N, dtype=np.float32)

        # Load PatchTST predictions
        pt_preds = None
        if has_pt:
            pt_file = Path(pt_dir) / f'{date_str}_predictions.npz'
            if pt_file.exists():
                pt_data = np.load(pt_file, allow_pickle=True)
                pt_raw = pt_data['predictions'].astype(np.float32)  # (P, 3)
                P = pt_raw.shape[0]
                if P == N:
                    pt_preds = pt_raw
                elif P > N:
                    # Subsample PT to match CM
                    pt_indices = np.linspace(0, P - 1, N, dtype=int)
                    pt_preds = pt_raw[pt_indices]
                else:
                    # PT is smaller — pad
                    pt_preds = np.zeros((N, 3), dtype=np.float32)
                    pt_preds[:P] = pt_raw

        # Build features
        features = build_features(cm_preds, pt_preds, mfe_data)

        # Target: 5s horizon label (column 1)
        target = cm_labels[:, 1].astype(np.float32)

        # Replace NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=5.0, neginf=-5.0)
        target = np.nan_to_num(target, nan=0.0)

        return features, target

    except Exception as e:
        print(f"  [ERROR] {date_str}: {e}", flush=True)
        return None


# ============================================================================
# Training
# ============================================================================
def train_one_fold(model, train_features, train_targets, val_features, val_targets,
                   epochs, batch_size, lr, weight_decay, patience, device):
    """Train one fold with early stopping. Returns best model state dict and val predictions."""
    # Scale features
    if HAS_SKLEARN:
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features)
        val_features = scaler.transform(val_features)
    else:
        # Manual standardization
        mu = train_features.mean(axis=0)
        std = train_features.std(axis=0) + 1e-8
        train_features = (train_features - mu) / std
        val_features = (val_features - mu) / std

    train_X = torch.tensor(train_features, dtype=torch.float32)
    train_y = torch.tensor(train_targets, dtype=torch.float32)
    val_X = torch.tensor(val_features, dtype=torch.float32)
    val_y = torch.tensor(val_targets, dtype=torch.float32)

    train_ds = TensorDataset(train_X, train_y)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=0, pin_memory=True, drop_last=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=False
    )
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss_sum += loss.item() * xb.shape[0]
            train_count += xb.shape[0]

        train_loss = train_loss_sum / max(train_count, 1)

        # --- Validate ---
        model.eval()
        with torch.no_grad():
            val_X_dev = val_X.to(device)
            val_pred = model(val_X_dev).cpu()
            val_loss = criterion(val_pred, val_y).item()

        scheduler.step(val_loss)

        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"    Epoch {epoch+1:3d}/{epochs}: "
                  f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}", flush=True)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch+1} "
                      f"(best val_loss={best_val_loss:.6f})", flush=True)
                break

    # Restore best model and get final predictions
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_pred = model(val_X.to(device)).cpu().numpy()

    return best_state, val_pred, best_val_loss


def compute_metrics(predictions, labels):
    """Compute IC, Sharpe, and commission-adjusted metrics."""
    N = len(predictions)
    if N < 10:
        return {}

    # Correlation (IC)
    ic = np.corrcoef(predictions, labels)[0, 1] if np.std(predictions) > 1e-10 else 0.0

    # Rank IC
    from scipy.stats import spearmanr
    rank_ic, _ = spearmanr(predictions, labels)

    # Simple directional accuracy
    correct_sign = np.mean(np.sign(predictions) == np.sign(labels))

    # Sharpe of predicted-direction P&L
    # If pred > 0, go long (profit = label), if pred < 0, go short (profit = -label)
    direction = np.sign(predictions)
    pnl_per_trade = direction * labels  # ticks per trade (raw)
    pnl_passive = pnl_per_trade - ES_RT_COMMISSION_TICKS  # net after passive commission

    sharpe_raw = (np.mean(pnl_per_trade) / (np.std(pnl_per_trade) + 1e-8)) * np.sqrt(252)
    sharpe_net = (np.mean(pnl_passive) / (np.std(pnl_passive) + 1e-8)) * np.sqrt(252)

    mean_pnl_raw = float(np.mean(pnl_per_trade))
    mean_pnl_net = float(np.mean(pnl_passive))

    # Win rate
    wr_raw = float(np.mean(pnl_per_trade > 0))
    wr_net = float(np.mean(pnl_passive > 0))

    # Profit factor
    gross_profit = float(np.sum(pnl_passive[pnl_passive > 0]))
    gross_loss = float(np.abs(np.sum(pnl_passive[pnl_passive < 0])))
    pf = gross_profit / (gross_loss + 1e-8)

    # Top-decile analysis (strongest signals)
    abs_pred = np.abs(predictions)
    top10_mask = abs_pred >= np.percentile(abs_pred, 90)
    if np.sum(top10_mask) > 5:
        top10_pnl = pnl_passive[top10_mask]
        top10_sharpe = (np.mean(top10_pnl) / (np.std(top10_pnl) + 1e-8)) * np.sqrt(252)
        top10_wr = float(np.mean(top10_pnl > 0))
        top10_mean = float(np.mean(top10_pnl))
    else:
        top10_sharpe = top10_wr = top10_mean = 0.0

    return {
        'ic': float(ic),
        'rank_ic': float(rank_ic) if not np.isnan(rank_ic) else 0.0,
        'directional_acc': float(correct_sign),
        'sharpe_raw': float(sharpe_raw),
        'sharpe_net': float(sharpe_net),
        'mean_pnl_raw_ticks': mean_pnl_raw,
        'mean_pnl_net_ticks': mean_pnl_net,
        'wr_raw': wr_raw,
        'wr_net': wr_net,
        'profit_factor': pf,
        'n_samples': int(N),
        'top10_sharpe_net': float(top10_sharpe),
        'top10_wr_net': float(top10_wr),
        'top10_mean_net_ticks': float(top10_mean),
    }


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Confluence Meta-Model v4 Training')
    parser.add_argument('--cm-dir', type=str, default=str(DEFAULT_CM_DIR),
                        help='CNN-Mamba v2 predictions directory')
    parser.add_argument('--mfe-dir', type=str, default=str(DEFAULT_MFE_DIR),
                        help='MFE/MAE labels directory')
    parser.add_argument('--pt-dir', type=str, default=str(DEFAULT_PT_DIR),
                        help='PatchTST predictions directory')
    parser.add_argument('--out-dir', type=str, default=str(DEFAULT_OUT_DIR),
                        help='Output directory')
    parser.add_argument('--train-window', type=int, default=15,
                        help='Number of training dates per fold')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Max epochs per fold')
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience')
    parser.add_argument('--no-mlflow', action='store_true',
                        help='Disable MLflow logging')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Self-logging: always redirect stdout/stderr to files on Windows (background execution)
    log_stdout = out_dir / 'train_stdout.log'
    log_stderr = out_dir / 'train_stderr.log'
    if sys.platform == 'win32':
        sys.stdout = open(log_stdout, 'w', buffering=1)
        sys.stderr = open(log_stderr, 'w', buffering=1)
        print(f"[SELF-LOG] Logging to {log_stdout}", flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INIT] Device: {device}", flush=True)
    print(f"[INIT] CM dir: {args.cm_dir}", flush=True)
    print(f"[INIT] MFE dir: {args.mfe_dir}", flush=True)
    print(f"[INIT] PT dir: {args.pt_dir}", flush=True)
    print(f"[INIT] Output: {out_dir}", flush=True)
    print(f"[INIT] Train window: {args.train_window} dates, "
          f"Epochs: {args.epochs}, BS: {args.batch_size}", flush=True)

    # --- MLflow setup ---
    use_mlflow = HAS_MLFLOW and not args.no_mlflow
    if use_mlflow:
        try:
            mlflow.set_tracking_uri(MLFLOW_URI)
            mlflow.set_experiment(EXPERIMENT_NAME)
            run = mlflow.start_run(run_name=f'v4_wf{args.train_window}')
            mlflow.log_params({
                'train_window': args.train_window,
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'lr': args.lr,
                'weight_decay': args.weight_decay,
                'dropout': args.dropout,
                'patience': args.patience,
                'architecture': '256-128-64',
                'target': 'label_5s',
            })
            print("[INIT] MLflow run started", flush=True)
        except Exception as e:
            print(f"[WARN] MLflow setup failed: {e}. Continuing without.", flush=True)
            use_mlflow = False

    # --- Discover dates ---
    available_dates, pt_dates = discover_dates(args.cm_dir, args.mfe_dir, args.pt_dir)

    if len(available_dates) < args.train_window + 1:
        print(f"[FATAL] Need at least {args.train_window + 1} dates, "
              f"have {len(available_dates)}. Exiting.", flush=True)
        sys.exit(1)

    # --- Load all days ---
    print(f"\n[LOAD] Loading {len(available_dates)} dates...", flush=True)
    day_data = {}
    for date_str in available_dates:
        has_pt = date_str in pt_dates
        result = load_day(date_str, args.cm_dir, args.mfe_dir, args.pt_dir, has_pt)
        if result is not None:
            day_data[date_str] = result
            features, target = result
            print(f"  {date_str}: {features.shape[0]} samples, "
                  f"{features.shape[1]} features, "
                  f"target_mean={target.mean():.4f} target_std={target.std():.4f}"
                  f"{' +PT' if has_pt else ''}", flush=True)

    valid_dates = sorted(day_data.keys())
    print(f"\n[LOAD] Successfully loaded {len(valid_dates)} dates", flush=True)

    if len(valid_dates) < args.train_window + 1:
        print(f"[FATAL] Need at least {args.train_window + 1} valid dates, "
              f"have {len(valid_dates)}. Exiting.", flush=True)
        sys.exit(1)

    input_dim = day_data[valid_dates[0]][0].shape[1]
    print(f"[INIT] Input dimension: {input_dim}", flush=True)

    # --- Walk-forward training ---
    n_folds = len(valid_dates) - args.train_window
    print(f"\n{'='*60}", flush=True)
    print(f"[WF] Starting walk-forward: {n_folds} folds "
          f"(train={args.train_window} dates, predict=1 date)", flush=True)
    print(f"{'='*60}\n", flush=True)

    all_fold_predictions = []
    all_fold_labels = []
    all_fold_features = []
    fold_results = []

    total_start = time.time()

    for fold_idx in range(n_folds):
        fold_start = time.time()
        train_dates = valid_dates[fold_idx : fold_idx + args.train_window]
        val_date = valid_dates[fold_idx + args.train_window]

        print(f"\n[FOLD {fold_idx+1}/{n_folds}] "
              f"Train: {train_dates[0]}..{train_dates[-1]} -> Val: {val_date}",
              flush=True)

        # Assemble training data
        train_feats_list = []
        train_targets_list = []
        for d in train_dates:
            if d in day_data:
                feats, tgt = day_data[d]
                train_feats_list.append(feats)
                train_targets_list.append(tgt)

        if not train_feats_list:
            print(f"  [SKIP] No training data for fold {fold_idx+1}", flush=True)
            continue

        train_features = np.concatenate(train_feats_list, axis=0)
        train_targets = np.concatenate(train_targets_list, axis=0)

        val_features, val_targets = day_data[val_date]

        print(f"  Train: {train_features.shape[0]} samples, "
              f"Val: {val_features.shape[0]} samples", flush=True)

        # Create fresh model per fold
        model = ConfluenceMetaMLP(input_dim=input_dim, dropout=args.dropout).to(device)

        # Train
        best_state, val_pred, best_val_loss = train_one_fold(
            model=model,
            train_features=train_features,
            train_targets=train_targets,
            val_features=val_features,
            val_targets=val_targets,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            device=device,
        )

        # Metrics
        metrics = compute_metrics(val_pred, val_targets)
        fold_time = time.time() - fold_start

        print(f"  [RESULT] IC={metrics.get('ic', 0):.4f}  "
              f"Rank_IC={metrics.get('rank_ic', 0):.4f}  "
              f"Sharpe_net={metrics.get('sharpe_net', 0):.2f}  "
              f"WR_net={metrics.get('wr_net', 0):.3f}  "
              f"PF={metrics.get('profit_factor', 0):.3f}  "
              f"Top10_Sharpe={metrics.get('top10_sharpe_net', 0):.2f}  "
              f"Time={fold_time:.1f}s", flush=True)

        # Store
        fold_info = {
            'fold_idx': fold_idx,
            'val_date': val_date,
            'train_dates': train_dates,
            'n_train': train_features.shape[0],
            'n_val': val_features.shape[0],
            'best_val_loss': float(best_val_loss),
            'wall_time_s': round(fold_time, 1),
            **metrics,
        }
        fold_results.append(fold_info)

        all_fold_predictions.append(val_pred)
        all_fold_labels.append(val_targets)
        all_fold_features.append(val_features)

        # Save per-fold predictions
        np.savez_compressed(
            out_dir / f'fold_{fold_idx:03d}_predictions.npz',
            predictions=val_pred,
            labels=val_targets,
            features=val_features,
            fold_idx=fold_idx,
            val_date=val_date,
        )

        # Save model checkpoint
        torch.save(best_state, out_dir / f'fold_{fold_idx:03d}_model.pt')

        # Log to MLflow
        if use_mlflow:
            try:
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(f'fold_{k}', v, step=fold_idx)
            except Exception:
                pass

    # --- Concat all OOT predictions ---
    if not all_fold_predictions:
        print("[FATAL] No folds completed successfully.", flush=True)
        sys.exit(1)

    concat_predictions = np.concatenate(all_fold_predictions)
    concat_labels = np.concatenate(all_fold_labels)
    concat_features = np.concatenate(all_fold_features)

    np.savez_compressed(
        out_dir / 'concat_predictions.npz',
        predictions=concat_predictions,
        labels=concat_labels,
        features=concat_features,
    )

    # --- Overall metrics ---
    overall_metrics = compute_metrics(concat_predictions, concat_labels)

    total_time = time.time() - total_start

    print(f"\n{'='*60}", flush=True)
    print(f"[FINAL] Walk-forward complete: {len(fold_results)} folds, "
          f"{total_time:.0f}s total", flush=True)
    print(f"[FINAL] Total OOT samples: {len(concat_predictions)}", flush=True)
    print(f"[FINAL] Concat IC:          {overall_metrics.get('ic', 0):.4f}", flush=True)
    print(f"[FINAL] Concat Rank IC:     {overall_metrics.get('rank_ic', 0):.4f}", flush=True)
    print(f"[FINAL] Concat Sharpe (raw): {overall_metrics.get('sharpe_raw', 0):.3f}", flush=True)
    print(f"[FINAL] Concat Sharpe (net): {overall_metrics.get('sharpe_net', 0):.3f}", flush=True)
    print(f"[FINAL] Win Rate (net):      {overall_metrics.get('wr_net', 0):.3f}", flush=True)
    print(f"[FINAL] Profit Factor:       {overall_metrics.get('profit_factor', 0):.3f}", flush=True)
    print(f"[FINAL] Mean PnL (raw):      {overall_metrics.get('mean_pnl_raw_ticks', 0):.4f} ticks", flush=True)
    print(f"[FINAL] Mean PnL (net):      {overall_metrics.get('mean_pnl_net_ticks', 0):.4f} ticks", flush=True)
    print(f"[FINAL] Top-10% Sharpe(net): {overall_metrics.get('top10_sharpe_net', 0):.3f}", flush=True)
    print(f"[FINAL] Top-10% WR(net):     {overall_metrics.get('top10_wr_net', 0):.3f}", flush=True)
    print(f"[FINAL] Top-10% Mean(net):   {overall_metrics.get('top10_mean_net_ticks', 0):.4f} ticks", flush=True)
    print(f"{'='*60}\n", flush=True)

    # --- Per-fold summary table ---
    print(f"{'Fold':>4} {'Date':>10} {'IC':>7} {'RankIC':>7} {'Sharpe':>7} "
          f"{'WR':>6} {'PF':>6} {'T10_S':>6} {'Time':>6}", flush=True)
    print('-' * 70, flush=True)
    for r in fold_results:
        print(f"{r['fold_idx']+1:>4} {r['val_date']:>10} "
              f"{r.get('ic', 0):>7.4f} {r.get('rank_ic', 0):>7.4f} "
              f"{r.get('sharpe_net', 0):>7.2f} {r.get('wr_net', 0):>6.3f} "
              f"{r.get('profit_factor', 0):>6.3f} "
              f"{r.get('top10_sharpe_net', 0):>6.2f} "
              f"{r['wall_time_s']:>5.0f}s", flush=True)

    # --- Save results ---
    results = {
        'experiment': 'confluence-meta-v4',
        'architecture': '256-128-64',
        'input_dim': input_dim,
        'train_window': args.train_window,
        'n_folds': len(fold_results),
        'total_oot_samples': int(len(concat_predictions)),
        'total_wall_time_s': round(total_time, 1),
        'overall_metrics': overall_metrics,
        'per_fold': fold_results,
        'hyperparams': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'dropout': args.dropout,
            'patience': args.patience,
            'optimizer': 'AdamW',
            'loss': 'MSE',
            'scheduler': 'ReduceLROnPlateau',
        },
        'cost_assumptions': {
            'commission_rt_ticks': ES_RT_COMMISSION_TICKS,
            'execution': 'passive_limit',
        },
    }

    results_file = out_dir / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[SAVE] Results saved to {results_file}", flush=True)

    # --- Log overall to MLflow ---
    if use_mlflow:
        try:
            for k, v in overall_metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f'overall_{k}', v)
            mlflow.log_metric('total_wall_time_s', total_time)
            mlflow.log_metric('n_folds', len(fold_results))
            mlflow.log_artifact(str(results_file))
            mlflow.end_run()
            print("[MLFLOW] Run logged and closed", flush=True)
        except Exception as e:
            print(f"[WARN] MLflow final logging failed: {e}", flush=True)

    print(f"\n[DONE] All outputs saved to {out_dir}", flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        # Emergency crash log for debugging pythonw/background launches
        import traceback
        crash_log = Path(str(DEFAULT_OUT_DIR)) / 'CRASH.log'
        crash_log.parent.mkdir(parents=True, exist_ok=True)
        with open(crash_log, 'w') as f:
            f.write(f"CRASH at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            traceback.print_exc(file=f)
        raise
