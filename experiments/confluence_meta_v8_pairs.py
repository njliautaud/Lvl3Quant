#!/usr/bin/env python3
"""
Meta-Model v8: Confluence-Aware Pairs (CNN-Mamba x PatchTST)

Hypothesis: PatchTST individually showed no lift in v5 ablation, but INTERACTION
features (agreement, disagreement, magnitude ratios) between CNN-Mamba and PatchTST
may capture complementary information that raw predictions miss.

Features (~55-60 total):
  Group 1 - CNN-Mamba raw predictions (3): 1s, 5s, 10s
  Group 2 - CNN-Mamba confidence (3): abs(pred) per horizon
  Group 3 - PatchTST raw predictions (3): 1s, 5s, 10s
  Group 4 - PatchTST confidence (3): abs(pred) per horizon
  Group 5 - PAIR features (13 NEW):
    - Sign agreement per horizon (3): +1 if agree, -1 if disagree
    - Magnitude ratio per horizon (3): cm / (pt + eps)
    - Prediction difference per horizon (3): cm - pt
    - Agreement strength per horizon (3): cm * pt
    - Cross-horizon sign agreement count (1): 0-3
  Group 6 - TRIPLET features (9 NEW):
    - Mean prediction across models (3)
    - Std prediction across models (3): disagreement magnitude
    - Max confidence across models (3): max(|cm|, |pt|)
  Group 7 - MBO microstructure (25): pass-through from events

Architecture: MLP 256->128->64, BatchNorm+GELU+Dropout(0.2)
Walk-forward: SLIDING 10d train / 3d eval (matches v7 prod)
Target: labels_1s (1-second mid-price movement, Huber loss)
v7 baseline: Spearman 0.308

Target platform: Razer (Windows, RTX 3070 8GB)
"""

import os
import sys
import json
import time
import zipfile
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
from datetime import datetime

# MLflow (optional)
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# ============================================================
# Configuration
# ============================================================
CONFIG = {
    # Data paths (Razer Windows)
    'cm_dir': r'C:\Users\claude\Lvl3Quant\output\cnn_mamba_v2_bulk_oot_v2',
    'pt_dir': r'C:\Users\claude\Lvl3Quant\output\hc470_dense_patchtst_s5',
    'mbo_dir': r'C:\Users\claude\Lvl3Quant\data\processed\mbo_events_smart_v3',
    'output_dir': r'C:\Users\claude\Lvl3Quant\output\meta_v8_confluence_pairs',
    # Walk-forward (SLIDING only, never expanding)
    'train_days': 10,
    'eval_days': 3,
    # Model
    'hidden_dims': [256, 128, 64],
    'dropout': 0.2,
    'batch_size': 2048,
    'epochs': 25,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'patience': 6,
    # Cost constants (canonical)
    'tick_value': 12.50,
    'commission_ticks': 0.376,
    # CNN-Mamba v2 parameters
    'cm_window_size': 3000,
    'cm_stride': 250,
    # v7 baseline
    'v7_spearman': 0.308,
    # Target
    'target_horizon': '1s',
    # Windows: no multiprocessing in DataLoader
    'num_workers': 0,
    # NaN handling
    'nan_fill': 0.0,
    'posinf_fill': 5.0,
    'neginf_fill': -5.0,
}

# ============================================================
# Logging
# ============================================================
os.makedirs(CONFIG['output_dir'], exist_ok=True)
log_path = os.path.join(CONFIG['output_dir'], 'training.log')


class MetaV8Formatter(logging.Formatter):
    def format(self, record):
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return f"{ts} [META_V8_PAIRS] {record.levelname}: {record.msg}"


logger = logging.getLogger('meta_v8_pairs')
logger.setLevel(logging.INFO)
logger.propagate = False

# File handler
fh = logging.FileHandler(log_path, encoding='utf-8')
fh.setFormatter(MetaV8Formatter())
logger.addHandler(fh)

# Console handler with flush
class FlushStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

sh = FlushStreamHandler(sys.stdout)
sh.setFormatter(MetaV8Formatter())
logger.addHandler(sh)


# ============================================================
# PatchTST Format Probe
# ============================================================
def probe_patchtst_format(pt_dir):
    """
    Load one PatchTST npz file and print all keys + shapes.
    Returns a dict describing the format for downstream use.
    """
    files = sorted([f for f in os.listdir(pt_dir)
                    if f.endswith('.npz') and f[0] == '2'])
    if not files:
        logger.error(f"No .npz files found in {pt_dir}")
        return None

    sample_file = os.path.join(pt_dir, files[0])
    logger.info(f"Probing PatchTST format from: {files[0]}")

    try:
        data = np.load(sample_file, allow_pickle=True)
    except Exception as e:
        logger.error(f"Failed to load PatchTST file: {e}")
        return None

    fmt = {'file': files[0], 'keys': {}}
    for key in data.files:
        arr = data[key]
        fmt['keys'][key] = {
            'shape': arr.shape if hasattr(arr, 'shape') else 'scalar',
            'dtype': str(arr.dtype) if hasattr(arr, 'dtype') else type(arr).__name__,
        }
        logger.info(f"  Key '{key}': shape={fmt['keys'][key]['shape']}, "
                    f"dtype={fmt['keys'][key]['dtype']}")

    # Determine prediction key naming convention
    pred_keys = {}
    for horizon in ['1s', '5s', '10s']:
        candidates = [
            f'predictions_{horizon}',      # CNN-Mamba style
            f'dense_predictions_{horizon}', # dense variant
            f'preds_{horizon}',
            f'pred_{horizon}',
        ]
        for c in candidates:
            if c in data.files:
                pred_keys[horizon] = c
                break

    # Also check for a combined 'predictions' array (N, 3) like CNN-Mamba
    if not pred_keys and 'predictions' in data.files:
        arr = data['predictions']
        if len(arr.shape) == 2 and arr.shape[1] >= 3:
            pred_keys = {'combined': 'predictions'}
            logger.info(f"  Found combined predictions array: shape {arr.shape}")

    # Check for dense_predictions combined
    if not pred_keys and 'dense_predictions' in data.files:
        arr = data['dense_predictions']
        if len(arr.shape) == 2 and arr.shape[1] >= 3:
            pred_keys = {'combined': 'dense_predictions'}
            logger.info(f"  Found combined dense_predictions array: shape {arr.shape}")

    fmt['pred_keys'] = pred_keys

    # Check for event indices or timestamps
    fmt['has_event_indices'] = 'event_indices' in data.files
    fmt['has_timestamps'] = any(k in data.files for k in ['timestamps', 'timestamp', 'times'])

    # Check for window/stride info
    for k in ['window_size', 'stride']:
        if k in data.files:
            fmt[k] = int(data[k])
            logger.info(f"  {k}: {fmt[k]}")

    logger.info(f"  Prediction keys found: {pred_keys}")
    logger.info(f"  Has event_indices: {fmt['has_event_indices']}")
    logger.info(f"  Has timestamps: {fmt['has_timestamps']}")

    data.close()
    return fmt


# ============================================================
# Model Definition
# ============================================================
class ConfluencePairsMLP(nn.Module):
    """
    Deeper confluence-aware MLP with BatchNorm before each layer.
    Input: concatenated feature vector (~55-60 dims)
    Output: scalar prediction (1s target)
    """
    def __init__(self, input_dim, hidden_dims=(256, 128, 64), dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ============================================================
# Data Loading & Feature Engineering
# ============================================================
def extract_date_from_filename(filename):
    """Extract 8-digit date string from filename like 20260401_predictions.npz."""
    basename = os.path.basename(filename)
    return basename[:8] if len(basename) >= 8 and basename[:8].isdigit() else None


def get_file_dates(directory, suffix):
    """Get sorted set of date strings from files matching suffix."""
    if not os.path.isdir(directory):
        logger.error(f"Directory not found: {directory}")
        return set()
    dates = set()
    for f in os.listdir(directory):
        if f.endswith(suffix) and len(f) >= 8 and f[:8].isdigit():
            dates.add(f[:8])
    return dates


def get_overlapping_dates():
    """Find dates present in ALL 3 data sources."""
    cm_dates = get_file_dates(CONFIG['cm_dir'], '_predictions.npz')
    pt_dates = get_file_dates(CONFIG['pt_dir'], '.npz')
    mbo_dates = get_file_dates(CONFIG['mbo_dir'], '_mbo_events.npz')

    overlap = sorted(cm_dates & pt_dates & mbo_dates)
    logger.info(f"Date counts: CNN-Mamba={len(cm_dates)}, PatchTST={len(pt_dates)}, "
                f"MBO={len(mbo_dates)}")
    logger.info(f"Three-way overlap: {len(overlap)} dates")

    if len(overlap) == 0:
        # Show what's missing for debugging
        cm_pt = sorted(cm_dates & pt_dates)
        cm_mbo = sorted(cm_dates & mbo_dates)
        pt_mbo = sorted(pt_dates & mbo_dates)
        logger.warning(f"  CM & PT overlap: {len(cm_pt)}")
        logger.warning(f"  CM & MBO overlap: {len(cm_mbo)}")
        logger.warning(f"  PT & MBO overlap: {len(pt_mbo)}")

    return overlap


def load_cnn_mamba_predictions(date_str):
    """Load CNN-Mamba predictions and compute event indices."""
    path = os.path.join(CONFIG['cm_dir'], f'{date_str}_predictions.npz')
    try:
        data = np.load(path, allow_pickle=True)
    except Exception as e:
        logger.warning(f"  Failed to load CNN-Mamba {date_str}: {e}")
        return None

    # Try different key formats
    if 'predictions' in data.files:
        preds = data['predictions']  # (N, 3) for 1s/5s/10s
    else:
        # Try per-horizon keys
        horizons = []
        for h in ['predictions_1s', 'predictions_5s', 'predictions_10s']:
            if h in data.files:
                horizons.append(data[h])
        if horizons:
            preds = np.column_stack(horizons)
        else:
            logger.warning(f"  {date_str}: No prediction keys in CNN-Mamba npz. "
                          f"Keys: {data.files}")
            return None

    n_preds = len(preds)

    # Event indices: either stored or computed from window/stride
    if 'event_indices' in data.files:
        event_idx = data['event_indices'].astype(np.int64)
    else:
        window = int(data['window_size']) if 'window_size' in data.files else CONFIG['cm_window_size']
        stride = int(data['stride']) if 'stride' in data.files else CONFIG['cm_stride']
        event_idx = np.array([window + i * stride for i in range(n_preds)], dtype=np.int64)

    # Ensure preds is 2D with 3 columns
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)
    if preds.shape[1] < 3:
        # Pad with zeros for missing horizons
        pad = np.zeros((len(preds), 3 - preds.shape[1]), dtype=preds.dtype)
        preds = np.column_stack([preds, pad])

    data.close()
    return {'preds': preds[:, :3], 'event_idx': event_idx}


def load_patchtst_predictions(date_str, pt_format):
    """
    Load PatchTST predictions for a date. Adapts to probed format.
    Returns dict with 'preds' (N, 3) and optionally 'event_idx'.
    """
    # Try common filename patterns
    candidates = [
        f'{date_str}_dense_predictions.npz',
        f'{date_str}_predictions.npz',
        f'{date_str}.npz',
    ]
    path = None
    for c in candidates:
        p = os.path.join(CONFIG['pt_dir'], c)
        if os.path.exists(p):
            path = p
            break

    if path is None:
        # Fallback: find any npz with this date
        for f in os.listdir(CONFIG['pt_dir']):
            if f.startswith(date_str) and f.endswith('.npz'):
                path = os.path.join(CONFIG['pt_dir'], f)
                break

    if path is None:
        return None

    try:
        data = np.load(path, allow_pickle=True)
    except Exception as e:
        logger.warning(f"  Failed to load PatchTST {date_str}: {e}")
        return None

    result = {}

    # Extract predictions based on probed format
    pred_keys = pt_format.get('pred_keys', {}) if pt_format else {}

    if 'combined' in pred_keys:
        key = pred_keys['combined']
        preds = data[key]
        if preds.ndim == 2:
            preds = preds[:, :3]  # Take first 3 columns (1s/5s/10s)
        else:
            preds = preds.reshape(-1, 1)
    else:
        # Try per-horizon keys
        horizon_preds = []
        for horizon in ['1s', '5s', '10s']:
            if horizon in pred_keys:
                horizon_preds.append(data[pred_keys[horizon]])
            else:
                # Try common key patterns directly
                for candidate_key in [f'predictions_{horizon}', f'dense_predictions_{horizon}',
                                      f'preds_{horizon}', f'pred_{horizon}']:
                    if candidate_key in data.files:
                        horizon_preds.append(data[candidate_key])
                        break

        if horizon_preds:
            preds = np.column_stack(horizon_preds)
        elif 'predictions' in data.files:
            preds = data['predictions']
            if preds.ndim == 1:
                preds = preds.reshape(-1, 1)
            preds = preds[:, :min(3, preds.shape[1])]
        elif 'dense_predictions' in data.files:
            preds = data['dense_predictions']
            if preds.ndim == 1:
                preds = preds.reshape(-1, 1)
            preds = preds[:, :min(3, preds.shape[1])]
        else:
            logger.warning(f"  {date_str}: Cannot find PatchTST predictions. "
                          f"Keys: {data.files}")
            data.close()
            return None

    # Pad to 3 columns if needed
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)
    if preds.shape[1] < 3:
        pad = np.zeros((len(preds), 3 - preds.shape[1]), dtype=preds.dtype)
        preds = np.column_stack([preds, pad])

    result['preds'] = preds[:, :3].astype(np.float32)

    # Event indices for alignment
    if 'event_indices' in data.files:
        result['event_idx'] = data['event_indices'].astype(np.int64)
    else:
        result['event_idx'] = None  # Will use positional alignment

    data.close()
    return result


def build_pair_features(cm_preds, pt_preds):
    """
    Build PAIR and TRIPLET interaction features between CNN-Mamba and PatchTST.

    Args:
        cm_preds: (N, 3) CNN-Mamba predictions [1s, 5s, 10s]
        pt_preds: (N, 3) PatchTST predictions [1s, 5s, 10s]

    Returns:
        pair_features: (N, 22) interaction features
    """
    eps = 1e-8
    n_horizons = min(cm_preds.shape[1], pt_preds.shape[1], 3)

    features = []

    # --- PAIR features (13) ---

    # 1. Sign agreement per horizon (3): +1 if same sign, -1 if different
    cm_sign = np.sign(cm_preds[:, :n_horizons])
    pt_sign = np.sign(pt_preds[:, :n_horizons])
    sign_agree = np.where(cm_sign == pt_sign, 1.0, -1.0)  # (N, n_horizons)
    features.append(sign_agree)

    # 2. Magnitude ratio per horizon (3): cm / (pt + eps)
    mag_ratio = cm_preds[:, :n_horizons] / (pt_preds[:, :n_horizons] + eps)
    mag_ratio = np.clip(mag_ratio, -10.0, 10.0)  # Bound extreme ratios
    features.append(mag_ratio)

    # 3. Prediction difference per horizon (3): cm - pt
    pred_diff = cm_preds[:, :n_horizons] - pt_preds[:, :n_horizons]
    features.append(pred_diff)

    # 4. Agreement strength per horizon (3): cm * pt (positive=agree, negative=disagree)
    agree_strength = cm_preds[:, :n_horizons] * pt_preds[:, :n_horizons]
    features.append(agree_strength)

    # 5. Cross-horizon sign agreement count (1): how many of 3 horizons agree
    cross_agree = np.sum(sign_agree > 0, axis=1, keepdims=True).astype(np.float32)  # (N, 1), range 0-3
    features.append(cross_agree)

    # Pad if fewer than 3 horizons (ensure consistent feature count)
    if n_horizons < 3:
        pad_cols = 3 - n_horizons
        for i in range(4):  # sign_agree, mag_ratio, pred_diff, agree_strength
            features.insert(len(features) - 1, np.zeros((cm_preds.shape[0], pad_cols), dtype=np.float32))

    # --- TRIPLET features (9) ---

    # 6. Mean prediction across models per horizon (3)
    mean_pred = (cm_preds[:, :n_horizons] + pt_preds[:, :n_horizons]) / 2.0
    features.append(mean_pred)
    if n_horizons < 3:
        features.append(np.zeros((cm_preds.shape[0], 3 - n_horizons), dtype=np.float32))

    # 7. Std prediction across models per horizon (3): disagreement magnitude
    stacked = np.stack([cm_preds[:, :n_horizons], pt_preds[:, :n_horizons]], axis=0)  # (2, N, h)
    std_pred = np.std(stacked, axis=0)  # (N, h)
    features.append(std_pred)
    if n_horizons < 3:
        features.append(np.zeros((cm_preds.shape[0], 3 - n_horizons), dtype=np.float32))

    # 8. Max confidence across models per horizon (3): max(|cm|, |pt|)
    max_conf = np.maximum(np.abs(cm_preds[:, :n_horizons]), np.abs(pt_preds[:, :n_horizons]))
    features.append(max_conf)
    if n_horizons < 3:
        features.append(np.zeros((cm_preds.shape[0], 3 - n_horizons), dtype=np.float32))

    return np.column_stack(features).astype(np.float32)


def load_date_features(date_str, pt_format):
    """
    Load and align all three data sources for a single date.
    Build the full feature vector: CM + PT + PAIR + TRIPLET + MBO.

    Returns (features, target) or (None, None) on failure.
    """
    # 1. Load CNN-Mamba predictions
    cm = load_cnn_mamba_predictions(date_str)
    if cm is None:
        return None, None
    cm_preds = cm['preds']           # (N_cm, 3)
    cm_event_idx = cm['event_idx']   # (N_cm,)

    # 2. Load PatchTST predictions
    pt = load_patchtst_predictions(date_str, pt_format)
    if pt is None:
        logger.warning(f"  {date_str}: PatchTST load failed, skipping")
        return None, None
    pt_preds = pt['preds']           # (N_pt, 3)
    pt_event_idx = pt.get('event_idx')  # may be None

    # 3. Load MBO events
    mbo_path = os.path.join(CONFIG['mbo_dir'], f'{date_str}_mbo_events.npz')
    try:
        mbo_data = np.load(mbo_path)
    except (zipfile.BadZipFile, Exception) as e:
        logger.warning(f"  {date_str}: Bad MBO file ({e}), skip")
        return None, None
    mbo_events = mbo_data['events']       # (N_mbo, 25)
    mbo_labels_1s = mbo_data['labels_1s']  # (N_mbo,) — 1s target
    n_mbo = len(mbo_events)

    # 4. Alignment: CNN-Mamba event indices are the anchor
    #    Clamp to valid MBO range
    valid_cm = cm_event_idx < n_mbo
    cm_event_idx = cm_event_idx[valid_cm]
    cm_preds = cm_preds[valid_cm]
    n_cm = len(cm_preds)

    if n_cm < 10:
        logger.warning(f"  {date_str}: Only {n_cm} valid CNN-Mamba rows, skipping")
        return None, None

    # 5. Align PatchTST to CNN-Mamba positions
    if pt_event_idx is not None:
        # PatchTST has event indices: find overlap with CNN-Mamba
        pt_idx_set = {}
        for i, idx in enumerate(pt_event_idx):
            pt_idx_set[int(idx)] = i

        # For each CM event position, find the closest PT prediction
        aligned_pt_preds = np.zeros((n_cm, 3), dtype=np.float32)
        pt_found_count = 0
        for i, cm_idx in enumerate(cm_event_idx):
            cm_idx_int = int(cm_idx)
            if cm_idx_int in pt_idx_set:
                aligned_pt_preds[i] = pt_preds[pt_idx_set[cm_idx_int]]
                pt_found_count += 1
            else:
                # Find nearest PT index within a tolerance window
                best_dist = float('inf')
                best_j = -1
                for pt_idx_val, pt_j in pt_idx_set.items():
                    dist = abs(cm_idx_int - pt_idx_val)
                    if dist < best_dist:
                        best_dist = dist
                        best_j = pt_j
                # Only align if within reasonable tolerance (500 events ~ 2 seconds)
                if best_dist <= 500 and best_j >= 0:
                    aligned_pt_preds[i] = pt_preds[best_j]
                    pt_found_count += 1
                # else: stays zero

        if pt_found_count < n_cm * 0.3:
            logger.warning(f"  {date_str}: Poor PT alignment ({pt_found_count}/{n_cm}), "
                          "falling back to positional")
            # Fallback to positional alignment
            n_shared = min(n_cm, len(pt_preds))
            aligned_pt_preds = np.zeros((n_cm, 3), dtype=np.float32)
            aligned_pt_preds[:n_shared] = pt_preds[:n_shared, :3]
        else:
            logger.info(f"  {date_str}: PT aligned by event_idx: {pt_found_count}/{n_cm} matched")
    else:
        # No event indices in PatchTST: use positional alignment
        n_shared = min(n_cm, len(pt_preds))
        aligned_pt_preds = np.zeros((n_cm, 3), dtype=np.float32)
        aligned_pt_preds[:n_shared] = pt_preds[:n_shared, :3]
        logger.info(f"  {date_str}: PT positional alignment: {n_shared}/{n_cm} covered")

    # 6. Extract features at CNN-Mamba event positions
    mbo_at_cm = mbo_events[cm_event_idx]  # (n_cm, 25)
    target = mbo_labels_1s[cm_event_idx]   # (n_cm,)

    # 7. Build feature groups
    cm_confidence = np.abs(cm_preds)       # (n_cm, 3)
    pt_confidence = np.abs(aligned_pt_preds)  # (n_cm, 3)

    # Pair + triplet interaction features
    pair_features = build_pair_features(cm_preds, aligned_pt_preds)  # (n_cm, 22)

    # 8. Concatenate all features
    features = np.column_stack([
        cm_preds,           # 3: CNN-Mamba raw predictions
        cm_confidence,      # 3: CNN-Mamba confidence
        aligned_pt_preds,   # 3: PatchTST raw predictions
        pt_confidence,      # 3: PatchTST confidence
        pair_features,      # 22: PAIR + TRIPLET interaction features
        mbo_at_cm,          # 25: MBO microstructure
    ]).astype(np.float32)

    return features, target.astype(np.float32)


# ============================================================
# Training
# ============================================================
def train_fold(model, train_X, train_y, val_X, val_y, device, fold_idx):
    """Train one fold with early stopping on validation loss."""
    train_ds = TensorDataset(
        torch.tensor(train_X, dtype=torch.float32),
        torch.tensor(train_y, dtype=torch.float32)
    )
    val_ds = TensorDataset(
        torch.tensor(val_X, dtype=torch.float32),
        torch.tensor(val_y, dtype=torch.float32)
    )
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'],
                              shuffle=True,
                              num_workers=CONFIG['num_workers'],
                              pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'] * 2,
                            shuffle=False,
                            num_workers=CONFIG['num_workers'],
                            pin_memory=(device.type == 'cuda'))

    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'],
                                  weight_decay=CONFIG['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG['epochs'])

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(CONFIG['epochs']):
        # --- Train ---
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss_sum += loss.item() * len(y_batch)
            train_n += len(y_batch)

        scheduler.step()

        # --- Validate ---
        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        val_preds_list = []
        val_labels_list = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                val_loss_sum += loss.item() * len(y_batch)
                val_n += len(y_batch)
                val_preds_list.append(pred.cpu().numpy())
                val_labels_list.append(y_batch.cpu().numpy())

        train_loss = train_loss_sum / max(train_n, 1)
        val_loss = val_loss_sum / max(val_n, 1)

        val_preds_arr = np.concatenate(val_preds_list)
        val_labels_arr = np.concatenate(val_labels_list)
        sp_r, _ = stats.spearmanr(val_preds_arr, val_labels_arr)

        if epoch % 5 == 0 or epoch == CONFIG['epochs'] - 1:
            logger.info(f"  Fold {fold_idx} Epoch {epoch}: "
                        f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, "
                        f"spearman={sp_r:.4f}, lr={scheduler.get_last_lr()[0]:.6f}")
            sys.stdout.flush()

        # Early stopping on val loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['patience']:
                logger.info(f"  Fold {fold_idx}: Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_val_loss


def evaluate_fold(model, eval_X, eval_y, device):
    """Generate predictions on OOT data."""
    model.eval()
    ds = TensorDataset(
        torch.tensor(eval_X, dtype=torch.float32),
        torch.tensor(eval_y, dtype=torch.float32)
    )
    loader = DataLoader(ds, batch_size=CONFIG['batch_size'] * 2,
                        shuffle=False, num_workers=0)
    all_preds = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            pred = model(X_batch)
            all_preds.append(pred.cpu().numpy())
    return np.concatenate(all_preds)


def compute_metrics(predictions, labels, prefix=""):
    """Compute Spearman, Pearson, and top-N% trade metrics."""
    sp_r, _ = stats.spearmanr(predictions, labels)
    pe_r, _ = stats.pearsonr(predictions, labels)

    comm = CONFIG['commission_ticks']
    results = {
        f'{prefix}spearman': float(sp_r),
        f'{prefix}pearson': float(pe_r),
        f'{prefix}n_samples': int(len(predictions)),
    }

    for pct_name, pct in [('top5', 95), ('top10', 90), ('top20', 80)]:
        threshold = np.percentile(np.abs(predictions), pct)
        mask = np.abs(predictions) >= threshold
        n_trades = mask.sum()

        if n_trades < 5:
            results[f'{prefix}{pct_name}_net_ticks'] = 0.0
            results[f'{prefix}{pct_name}_n_trades'] = 0
            results[f'{prefix}{pct_name}_wr'] = 0.0
            results[f'{prefix}{pct_name}_avg_ticks'] = 0.0
            results[f'{prefix}{pct_name}_precision'] = 0.0
            continue

        trade_pnl = np.sign(predictions[mask]) * labels[mask] - comm
        net_ticks = trade_pnl.sum()
        avg_ticks = trade_pnl.mean()
        win_rate = (trade_pnl > 0).mean()
        direction_correct = np.sign(predictions[mask]) == np.sign(labels[mask])
        precision = direction_correct.mean()

        results[f'{prefix}{pct_name}_net_ticks'] = float(net_ticks)
        results[f'{prefix}{pct_name}_avg_ticks'] = float(avg_ticks)
        results[f'{prefix}{pct_name}_n_trades'] = int(n_trades)
        results[f'{prefix}{pct_name}_wr'] = float(win_rate)
        results[f'{prefix}{pct_name}_precision'] = float(precision)

    return results


# ============================================================
# Main Walk-Forward Loop
# ============================================================
def main():
    start_time = time.time()

    logger.info("=" * 70)
    logger.info("Meta-Model v8: Confluence-Aware PAIRS (CNN-Mamba x PatchTST)")
    logger.info("=" * 70)
    logger.info("HYPOTHESIS: PatchTST adds value through INTERACTION features")
    logger.info("(agreement, disagreement, magnitude ratios) even though raw")
    logger.info("predictions showed no individual lift in v5 ablation.")
    logger.info("")
    logger.info("Features: CM preds (3) + CM conf (3) + PT preds (3) + PT conf (3)")
    logger.info("          + PAIR features (13) + TRIPLET features (9) + MBO (25)")
    logger.info(f"Walk-forward: SLIDING {CONFIG['train_days']}d train / "
                f"{CONFIG['eval_days']}d eval")
    logger.info(f"Target: {CONFIG['target_horizon']} mid-price movement")
    logger.info(f"v7 baseline Spearman: {CONFIG['v7_spearman']}")
    logger.info(f"Config: {json.dumps(CONFIG, indent=2, default=str)}")
    sys.stdout.flush()

    # --- MLflow ---
    mlflow_active = False
    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_tracking_uri("http://localhost:5000")
            mlflow.set_experiment("meta_v8_confluence_pairs")
            mlflow.start_run(
                run_name=f"v8_pairs_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            mlflow.log_params({
                'target_horizon': CONFIG['target_horizon'],
                'train_days': CONFIG['train_days'],
                'eval_days': CONFIG['eval_days'],
                'hidden_dims': str(CONFIG['hidden_dims']),
                'dropout': CONFIG['dropout'],
                'batch_size': CONFIG['batch_size'],
                'epochs': CONFIG['epochs'],
                'lr': CONFIG['lr'],
                'weight_decay': CONFIG['weight_decay'],
                'patience': CONFIG['patience'],
                'commission_ticks': CONFIG['commission_ticks'],
                'v7_baseline_spearman': CONFIG['v7_spearman'],
            })
            mlflow_active = True
            logger.info("MLflow tracking enabled")
        except Exception as e:
            logger.warning(f"MLflow setup failed ({e}), continuing without tracking")
    else:
        logger.info("MLflow not installed, training without tracking")

    # --- Device ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        logger.warning("No CUDA device found - training will be slow on CPU")
    sys.stdout.flush()

    # --- Probe PatchTST format ---
    logger.info("\n--- Probing PatchTST npz format ---")
    pt_format = probe_patchtst_format(CONFIG['pt_dir'])
    if pt_format is None:
        logger.error("FATAL: Cannot determine PatchTST format. Check pt_dir path.")
        if mlflow_active:
            mlflow.log_param('error', 'patchtst_format_probe_failed')
            mlflow.end_run(status='FAILED')
        return
    logger.info(f"PatchTST format probe complete: {len(pt_format.get('pred_keys', {}))} "
                f"prediction key patterns found")
    sys.stdout.flush()

    # --- Get overlapping dates ---
    dates = get_overlapping_dates()
    min_dates = CONFIG['train_days'] + CONFIG['eval_days']
    if len(dates) < min_dates:
        logger.error(f"Not enough overlapping dates ({len(dates)}) for walk-forward "
                     f"(need {min_dates})")
        if mlflow_active:
            mlflow.log_param('error', f'insufficient_dates_{len(dates)}')
            mlflow.end_run(status='FAILED')
        return

    # --- Pre-load all date data ---
    logger.info(f"\nLoading data for {len(dates)} overlapping dates...")
    sys.stdout.flush()

    date_data = {}
    valid_dates = []
    feature_dim_verified = False
    expected_dim = None

    for d in dates:
        features, target = load_date_features(d, pt_format)
        if features is None:
            continue

        # Reject zero-variance target
        if np.nanstd(target) < 1e-6:
            logger.warning(f"  {d}: ZERO-VARIANCE target, REJECTED")
            continue

        # Reject excessive NaN targets
        valid_mask = ~np.isnan(target)
        if valid_mask.mean() < 0.5:
            logger.warning(f"  {d}: Too many NaN targets ({(~valid_mask).sum()}/{len(target)}), "
                          "REJECTED")
            continue

        # Filter out NaN targets
        features = features[valid_mask]
        target = target[valid_mask]

        # NaN handling for features
        features = np.nan_to_num(features, nan=CONFIG['nan_fill'],
                                 posinf=CONFIG['posinf_fill'],
                                 neginf=CONFIG['neginf_fill'])

        # Verify consistent feature dimension
        if not feature_dim_verified:
            expected_dim = features.shape[1]
            feature_dim_verified = True
            logger.info(f"  Feature dimension: {expected_dim}")
        elif features.shape[1] != expected_dim:
            logger.warning(f"  {d}: Feature dim mismatch ({features.shape[1]} vs "
                          f"{expected_dim}), REJECTED")
            continue

        date_data[d] = (features, target)
        valid_dates.append(d)
        logger.info(f"  {d}: {len(features):,} samples, "
                    f"target std={np.std(target):.4f}, "
                    f"features shape={features.shape}")
        sys.stdout.flush()

    logger.info(f"\nValid dates after filtering: {len(valid_dates)}")

    if len(valid_dates) < min_dates:
        logger.error(f"Not enough valid dates ({len(valid_dates)}) for walk-forward")
        if mlflow_active:
            mlflow.log_param('error', f'insufficient_valid_dates_{len(valid_dates)}')
            mlflow.end_run(status='FAILED')
        return

    input_dim = expected_dim
    logger.info(f"Input dimension: {input_dim}")
    logger.info(f"Feature breakdown: CM_preds(3) + CM_conf(3) + PT_preds(3) + "
                f"PT_conf(3) + PAIR(13) + TRIPLET(9) + MBO(25) = 59 expected, "
                f"got {input_dim}")
    sys.stdout.flush()

    # --- Walk-forward setup ---
    train_days = CONFIG['train_days']
    eval_days = CONFIG['eval_days']
    n_folds = (len(valid_dates) - train_days) // eval_days

    logger.info(f"\nWalk-forward: {n_folds} folds, "
                f"{train_days}d train / {eval_days}d eval (SLIDING)")
    sys.stdout.flush()

    if mlflow_active:
        mlflow.log_metrics({
            'n_valid_dates': len(valid_dates),
            'n_folds': n_folds,
            'input_dim': input_dim,
        })

    # --- Walk-forward loop ---
    all_oot_preds = []
    all_oot_labels = []
    all_oot_dates = []
    fold_results = []

    for fold_idx in range(n_folds):
        fold_start = fold_idx * eval_days  # SLIDING: each fold starts eval_days later
        train_slice = valid_dates[fold_start:fold_start + train_days]
        eval_start = fold_start + train_days
        eval_end = min(eval_start + eval_days, len(valid_dates))
        eval_slice = valid_dates[eval_start:eval_end]

        if len(eval_slice) == 0:
            break

        logger.info(f"\n{'='*50}")
        logger.info(f"Fold {fold_idx}/{n_folds-1}: "
                     f"Train [{train_slice[0]}..{train_slice[-1]}] ({len(train_slice)}d) -> "
                     f"Eval [{eval_slice[0]}..{eval_slice[-1]}] ({len(eval_slice)}d)")
        sys.stdout.flush()

        # Assemble train data
        train_X = np.concatenate([date_data[d][0] for d in train_slice])
        train_y = np.concatenate([date_data[d][1] for d in train_slice])

        # Feature normalization (z-score from train set only — no leakage)
        feat_mean = train_X.mean(axis=0)
        feat_std = train_X.std(axis=0) + 1e-8
        train_X_norm = (train_X - feat_mean) / feat_std

        # Use last 20% of train as validation for early stopping
        val_split = int(len(train_X_norm) * 0.8)
        val_X = train_X_norm[val_split:]
        val_y = train_y[val_split:]

        logger.info(f"  Train: {len(train_X):,} samples, "
                    f"Val: {len(val_X):,} samples")

        # Build model
        model = ConfluencePairsMLP(
            input_dim=input_dim,
            hidden_dims=CONFIG['hidden_dims'],
            dropout=CONFIG['dropout']
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters())
        if fold_idx == 0:
            logger.info(f"  Model params: {n_params:,}")

        # Train
        best_val_loss = train_fold(
            model, train_X_norm, train_y,
            val_X, val_y,
            device, fold_idx
        )

        # Save model weights + normalization stats
        weight_path = os.path.join(CONFIG['output_dir'],
                                   f'fold_{fold_idx:02d}_model.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'feat_mean': feat_mean,
            'feat_std': feat_std,
            'input_dim': input_dim,
            'hidden_dims': CONFIG['hidden_dims'],
            'dropout': CONFIG['dropout'],
            'train_dates': train_slice,
            'eval_dates': eval_slice,
            'fold_idx': fold_idx,
        }, weight_path)

        norm_path = os.path.join(CONFIG['output_dir'],
                                 f'fold_{fold_idx:02d}_norm_stats.npz')
        np.savez_compressed(norm_path, feat_mean=feat_mean, feat_std=feat_std)

        # Evaluate on OOT dates
        fold_preds = []
        fold_labels = []
        fold_date_labels = []

        for eval_date in eval_slice:
            eval_X_raw, eval_y = date_data[eval_date]
            eval_X_norm = (eval_X_raw - feat_mean) / feat_std
            preds = evaluate_fold(model, eval_X_norm, eval_y, device)
            fold_preds.append(preds)
            fold_labels.append(eval_y)
            fold_date_labels.extend([eval_date] * len(preds))

        fold_preds = np.concatenate(fold_preds)
        fold_labels = np.concatenate(fold_labels)

        # Save fold OOT predictions
        pred_path = os.path.join(CONFIG['output_dir'],
                                 f'fold_{fold_idx:02d}_oot_predictions.npz')
        np.savez_compressed(pred_path,
                            predictions=fold_preds,
                            labels=fold_labels,
                            dates=np.array(fold_date_labels),
                            train_dates=np.array(train_slice),
                            eval_dates=np.array(eval_slice),
                            feat_mean=feat_mean,
                            feat_std=feat_std)

        # Compute fold metrics
        metrics = compute_metrics(fold_preds, fold_labels, prefix=f'fold{fold_idx}_')
        fold_results.append(metrics)

        fold_sp = metrics[f'fold{fold_idx}_spearman']
        logger.info(f"  Fold {fold_idx} OOT: Spearman={fold_sp:.4f}, "
                     f"Pearson={metrics[f'fold{fold_idx}_pearson']:.4f}, "
                     f"n={metrics[f'fold{fold_idx}_n_samples']:,}")
        for pct in ['top5', 'top10', 'top20']:
            key_net = f'fold{fold_idx}_{pct}_net_ticks'
            key_wr = f'fold{fold_idx}_{pct}_wr'
            key_avg = f'fold{fold_idx}_{pct}_avg_ticks'
            key_n = f'fold{fold_idx}_{pct}_n_trades'
            logger.info(f"    {pct}: net={metrics.get(key_net, 0):.1f}t, "
                        f"avg={metrics.get(key_avg, 0):.4f}t, "
                        f"WR={metrics.get(key_wr, 0):.3f}, "
                        f"n={metrics.get(key_n, 0)}")
        sys.stdout.flush()

        if mlflow_active:
            mlflow.log_metrics({
                f'fold_{fold_idx}_spearman': float(fold_sp),
                f'fold_{fold_idx}_val_loss': float(best_val_loss),
            }, step=fold_idx)

        all_oot_preds.append(fold_preds)
        all_oot_labels.append(fold_labels)
        all_oot_dates.extend(fold_date_labels)

        # Free GPU memory between folds
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # ============================================================
    # Concat OOT Analysis
    # ============================================================
    if not all_oot_preds:
        logger.error("No OOT predictions generated. Check data availability.")
        if mlflow_active:
            mlflow.end_run(status='FAILED')
        return

    concat_preds = np.concatenate(all_oot_preds)
    concat_labels = np.concatenate(all_oot_labels)
    concat_dates = np.array(all_oot_dates)

    logger.info(f"\n{'='*70}")
    logger.info("CONCAT OOT RESULTS (all folds combined)")
    logger.info(f"{'='*70}")

    concat_metrics = compute_metrics(concat_preds, concat_labels, prefix='concat_')

    logger.info(f"Total OOT samples: {concat_metrics['concat_n_samples']:,}")
    logger.info(f"Concat Spearman:   {concat_metrics['concat_spearman']:.4f}")
    logger.info(f"Concat Pearson:    {concat_metrics['concat_pearson']:.4f}")

    for pct in ['top5', 'top10', 'top20']:
        logger.info(f"  {pct}: net={concat_metrics.get(f'concat_{pct}_net_ticks', 0):.1f}t, "
                    f"avg={concat_metrics.get(f'concat_{pct}_avg_ticks', 0):.4f}t/trade, "
                    f"WR={concat_metrics.get(f'concat_{pct}_wr', 0):.3f}, "
                    f"precision={concat_metrics.get(f'concat_{pct}_precision', 0):.3f}, "
                    f"n={concat_metrics.get(f'concat_{pct}_n_trades', 0):,}")

    # ============================================================
    # v7 vs v8 Comparison
    # ============================================================
    v7_sp = CONFIG['v7_spearman']
    v8_sp = concat_metrics['concat_spearman']
    delta = v8_sp - v7_sp
    pct_change = (delta / abs(v7_sp)) * 100 if v7_sp != 0 else 0.0

    logger.info(f"\n{'='*70}")
    logger.info("v7 (CM + MBO only) vs v8 (CM + PT PAIRS + MBO) COMPARISON")
    logger.info(f"{'='*70}")
    logger.info(f"v7 Spearman (31 features): {v7_sp:.4f}")
    logger.info(f"v8 Spearman (~{input_dim} features): {v8_sp:.4f}")
    logger.info(f"Delta: {delta:+.4f} ({pct_change:+.1f}%)")

    if delta > 0.01:
        logger.info("RESULT: v8 PAIR features ADD LIFT - PatchTST interactions help")
    elif delta > -0.01:
        logger.info("RESULT: v8 MATCHES v7 - PAIR features are neutral")
    else:
        logger.info("RESULT: v8 UNDERPERFORMS v7 - PAIR features hurt, revert to v7")

    # ============================================================
    # Per-Date OOT Breakdown
    # ============================================================
    logger.info(f"\nPer-date OOT breakdown:")
    unique_dates = sorted(set(all_oot_dates))
    date_sharpes = []
    positive_sp_dates = 0
    positive_sharpe_dates = 0

    for d in unique_dates:
        mask = concat_dates == d
        d_preds = concat_preds[mask]
        d_labels = concat_labels[mask]
        d_sp, _ = stats.spearmanr(d_preds, d_labels)

        if d_sp > 0:
            positive_sp_dates += 1

        # Daily Sharpe at top20%
        threshold = np.percentile(np.abs(d_preds), 80)
        d_mask = np.abs(d_preds) >= threshold
        if d_mask.sum() > 5:
            d_pnl = np.sign(d_preds[d_mask]) * d_labels[d_mask] - CONFIG['commission_ticks']
            d_sharpe = d_pnl.mean() / (d_pnl.std() + 1e-8) * np.sqrt(252)
            date_sharpes.append(d_sharpe)
            if d_sharpe > 0:
                positive_sharpe_dates += 1
        else:
            d_sharpe = 0.0
            date_sharpes.append(0.0)

        logger.info(f"  {d}: n={mask.sum():,}, spearman={d_sp:.4f}, "
                    f"daily_sharpe_top20={d_sharpe:.2f}")

    avg_sharpe = np.mean(date_sharpes) if date_sharpes else 0.0
    median_sharpe = np.median(date_sharpes) if date_sharpes else 0.0

    logger.info(f"\nAvg daily Sharpe (top20%):    {avg_sharpe:.2f}")
    logger.info(f"Median daily Sharpe (top20%): {median_sharpe:.2f}")
    logger.info(f"Positive Spearman days:       {positive_sp_dates}/{len(unique_dates)}")
    logger.info(f"Positive Sharpe days:         {positive_sharpe_dates}/{len(unique_dates)}")
    sys.stdout.flush()

    # ============================================================
    # Per-fold Spearman summary
    # ============================================================
    logger.info(f"\nPer-fold Spearman summary:")
    fold_spearmans = []
    for i, fr in enumerate(fold_results):
        sp = fr[f'fold{i}_spearman']
        fold_spearmans.append(sp)
        logger.info(f"  Fold {i}: {sp:.4f}")
    logger.info(f"  Mean: {np.mean(fold_spearmans):.4f}, "
                f"Std: {np.std(fold_spearmans):.4f}, "
                f"Min: {np.min(fold_spearmans):.4f}, "
                f"Max: {np.max(fold_spearmans):.4f}")

    # ============================================================
    # Save outputs
    # ============================================================

    # Concat predictions
    concat_path = os.path.join(CONFIG['output_dir'], 'concat_oot_predictions.npz')
    np.savez_compressed(concat_path,
                        predictions=concat_preds,
                        labels=concat_labels,
                        dates=concat_dates)

    # Training summary JSON
    summary = {
        'version': 'v8_confluence_pairs',
        'description': ('Meta-model v8: CNN-Mamba + PatchTST PAIR/TRIPLET interaction '
                        'features + MBO microstructure. Tests hypothesis that PatchTST '
                        'helps through interaction features even if raw preds are weak.'),
        'target_horizon': CONFIG['target_horizon'],
        'input_dim': input_dim,
        'n_folds': len(fold_results),
        'n_valid_dates': len(valid_dates),
        'n_oot_dates': len(unique_dates),
        'walk_forward': f"SLIDING {train_days}d train / {eval_days}d eval",
        'v7_baseline_spearman': v7_sp,
        'v8_spearman': float(v8_sp),
        'v8_vs_v7_delta': float(delta),
        'v8_vs_v7_pct_change': float(pct_change),
        'avg_daily_sharpe_top20': float(avg_sharpe),
        'median_daily_sharpe_top20': float(median_sharpe),
        'positive_spearman_days': f"{positive_sp_dates}/{len(unique_dates)}",
        'positive_sharpe_days': f"{positive_sharpe_dates}/{len(unique_dates)}",
        'fold_spearmans': [float(s) for s in fold_spearmans],
        'concat_metrics': {k: (float(v) if isinstance(v, (float, np.floating)) else int(v))
                          for k, v in concat_metrics.items()},
        'date_sharpes': {d: float(s) for d, s in zip(unique_dates, date_sharpes)},
        'config': {k: (str(v) if not isinstance(v, (int, float, str, list)) else v)
                   for k, v in CONFIG.items()},
        'patchtst_format': {
            'pred_keys': pt_format.get('pred_keys', {}),
            'has_event_indices': pt_format.get('has_event_indices', False),
        },
        'timestamp': datetime.now().isoformat(),
    }

    summary_path = os.path.join(CONFIG['output_dir'], 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nSaved concat predictions and training summary")

    # MLflow logging
    if mlflow_active:
        try:
            mlflow.log_metrics({
                'concat_spearman': float(v8_sp),
                'concat_pearson': float(concat_metrics['concat_pearson']),
                'v7_v8_delta': float(delta),
                'n_oot_dates': len(unique_dates),
                'positive_spearman_days': positive_sp_dates,
                'avg_daily_sharpe_top20': float(avg_sharpe),
                'median_daily_sharpe_top20': float(median_sharpe),
            })
            mlflow.log_artifact(summary_path)
            mlflow.log_artifact(log_path)
            mlflow.end_run()
            logger.info("MLflow run completed")
        except Exception as e:
            logger.warning(f"MLflow finalization error: {e}")

    # ============================================================
    # Final Summary
    # ============================================================
    elapsed = time.time() - start_time
    logger.info(f"\n{'='*70}")
    logger.info("META-MODEL v8 CONFLUENCE PAIRS -- TRAINING COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Runtime: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    logger.info(f"Folds: {len(fold_results)}")
    logger.info(f"OOT dates: {len(unique_dates)}")
    logger.info(f"OOT samples: {len(concat_preds):,}")
    logger.info(f"")
    logger.info(f"  CONCAT SPEARMAN:  {v8_sp:.4f}  (v7 baseline: {v7_sp:.4f})")
    logger.info(f"  DELTA vs v7:      {delta:+.4f}  ({pct_change:+.1f}%)")
    logger.info(f"  Top5%  WR: {concat_metrics.get('concat_top5_wr', 0):.3f}  "
                f"avg: {concat_metrics.get('concat_top5_avg_ticks', 0):+.4f}t")
    logger.info(f"  Top10% WR: {concat_metrics.get('concat_top10_wr', 0):.3f}  "
                f"avg: {concat_metrics.get('concat_top10_avg_ticks', 0):+.4f}t")
    logger.info(f"  Top20% WR: {concat_metrics.get('concat_top20_wr', 0):.3f}  "
                f"avg: {concat_metrics.get('concat_top20_avg_ticks', 0):+.4f}t")
    logger.info(f"")
    if delta > 0.01:
        logger.info("VERDICT: PatchTST PAIR features provide incremental lift.")
        logger.info("Next: test in fill sim to verify exploitability.")
    elif delta > -0.01:
        logger.info("VERDICT: Neutral. PAIR features neither help nor hurt.")
        logger.info("Next: consider dropping PT if no other evidence of value.")
    else:
        logger.info("VERDICT: PAIR features HURT. Revert to v7 (CM + MBO only).")
    logger.info(f"{'='*70}")
    sys.stdout.flush()


if __name__ == '__main__':
    main()
