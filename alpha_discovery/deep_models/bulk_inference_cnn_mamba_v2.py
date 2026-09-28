"""
CNN-Mamba v2 Bulk Inference Script
===================================
Generates predictions + embeddings for ALL 248 dates using trained fold weights.
Uses ensemble of folds 08, 09, 10 (same architecture: d_model=96, d_state=32, n_layers=3).

Output: /home/nick/Lvl3Quant/output/cnn_mamba_v2_bulk_inference/YYYYMMDD_predictions.npz
Each file contains:
  - predictions: (N, 3) float32 -- predicted price changes for 1s, 5s, 10s
  - embeddings: (N, 96) float32 -- model state embeddings
  - labels: (N, 3) float32 -- actual labels (if available)
  - horizons: ['1s', '5s', '10s']
  - date: YYYYMMDD string
  - data_type: 'IS' (in-sample), 'OOT' (walk-forward out-of-time), or 'OOS' (post-training)
  - fold_weights: list of fold indices used for ensemble
  - n_windows: number of prediction windows
  - window_size: 1000
  - stride: 83

This is INFERENCE ONLY -- no training, no gradient computation.
"""

import os
import sys
import gc
import time
import logging
import warnings
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import scipy.stats

# ============================================================
# Logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ============================================================
# Config -- must match training exactly
# ============================================================
WINDOW_SIZE = 1000
STRIDE = 83  # 1000 // 12
BATCH_SIZE = 256  # larger for inference (no gradients = less VRAM)
HORIZONS = ["1s", "5s", "10s"]
N_TOTAL_FEATURES = 25  # smart_v3

# Architecture for folds 08, 09, 10
ARCH = {
    "d_model": 96,
    "d_state": 32,
    "n_layers": 3,
    "dt_rank": 16,
    "d_conv": 4,
    "dropout": 0.1,
}

FOLD_INDICES = [8, 9, 10]  # ensemble these folds
WEIGHTS_DIR = Path("/home/nick/Lvl3Quant/output/cnn_mamba_v2_smart_v3_mar")
DATA_DIR = Path("/home/nick/Lvl3Quant/data/processed/mbo_events_smart_v3")
OUTPUT_DIR = Path("/home/nick/Lvl3Quant/output/cnn_mamba_v2_bulk_inference")

# Walk-forward OOT dates (from existing fold predictions)
OOT_DATE_MAP = {
    "20260223": 0, "20260224": 1, "20260225": 2, "20260226": 3,
    "20260227": 4, "20260301": 5, "20260302": 6, "20260303": 7,
    "20260304": 8, "20260305": 9,
}
# Dates AFTER 20260305 are truly OOS (no fold trained on or tested on them)
LAST_OOT_DATE = "20260305"

# Try CUDA mamba_ssm kernels
try:
    from mamba_ssm import Mamba as CUDAMamba
    USE_CUDA_MAMBA = True
    logger.info("Using mamba_ssm CUDA kernels (FAST)")
except ImportError:
    USE_CUDA_MAMBA = False
    logger.info("Using pure PyTorch Mamba (SLOW)")

# ============================================================
# Import model from training script
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault('MAMBA_FEATURE_SET', 'smart_v3')
os.environ.setdefault('SKIP_NORMALIZE', '1')
import train_cnn_mamba_v2 as _T
CNNMambaV2 = _T.CNNMambaV2

def load_model(fold_idx, device):
    """Load a trained model from checkpoint."""
    ckpt_path = WEIGHTS_DIR / ("fold_%02d_best.pt" % fold_idx)
    logger.info("Loading fold %02d weights from %s" % (fold_idx, ckpt_path))
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    arch = ckpt["arch"]
    model = CNNMambaV2(
        d_model=arch["d_model"],
        d_state=arch["d_state"],
        n_layers=arch["n_layers"],
        dt_rank=arch.get("dt_rank", ARCH["dt_rank"]),
        d_conv=arch.get("d_conv", ARCH["d_conv"]),
        dropout=arch.get("dropout", ARCH["dropout"]),
    )
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("  Fold %02d: %s params, val_ic_10s=%s" % (fold_idx, format(n_params, ","), ckpt.get("val_ic_10s", "?")))
    return model


def classify_date(date_str):
    """Classify a date as IS (in-sample), OOT (walk-forward), or OOS (post-training)."""
    if date_str in OOT_DATE_MAP:
        return "OOT"
    elif date_str > LAST_OOT_DATE:
        return "OOS"
    else:
        return "IS"


def run_inference_single_model(model, loader, device):
    """Run inference with a single model, return preds, labels, embeddings."""
    all_preds = []
    all_labels = []
    all_embeds = []

    with torch.no_grad(), torch.amp.autocast("cuda"):
        for events, labels in loader:
            events = events.to(device, non_blocking=True)
            preds, emb = model(events, return_embedding=True)
            all_preds.append(preds.float().cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_embeds.append(emb.float().cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    embeds = np.concatenate(all_embeds, axis=0)
    return preds, labels, embeds


def main():
    logger.info("=" * 70)
    logger.info("CNN-Mamba v2 BULK INFERENCE")
    logger.info("Ensemble folds: %s" % FOLD_INDICES)
    logger.info("Window: %d, Stride: %d, Batch: %d" % (WINDOW_SIZE, STRIDE, BATCH_SIZE))
    logger.info("Data dir: %s" % DATA_DIR)
    logger.info("Output dir: %s" % OUTPUT_DIR)
    logger.info("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s" % device)
    if device.type == "cuda":
        logger.info("GPU: %s" % torch.cuda.get_device_name(0))
        logger.info("VRAM: %.1f GB" % (torch.cuda.get_device_properties(0).total_memory / 1e9))

    # Create output dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load all 3 models
    models = []
    for fold_idx in FOLD_INDICES:
        model = load_model(fold_idx, device)
        models.append(model)
    logger.info("Loaded %d models for ensemble" % len(models))

    # Get all data files sorted
    data_files = sorted(DATA_DIR.glob("*.npz"))
    logger.info("Found %d data files" % len(data_files))

    # Track stats
    total_windows = 0
    is_count = 0
    oot_count = 0
    oos_count = 0
    skipped = 0
    start_time = time.time()

    for i, npz_path in enumerate(data_files):
        date_str = npz_path.stem.split("_")[0]  # e.g., "20260223"
        out_path = OUTPUT_DIR / ("%s_predictions.npz" % date_str)

        # Skip if already processed
        if out_path.exists():
            logger.info("[%d/%d] %s -- SKIPPED (already exists)" % (i + 1, len(data_files), date_str))
            skipped += 1
            continue

        data_type = classify_date(date_str)

        # Create dataset and loader
        try:
            dataset = SingleDateDataset(npz_path, window_size=WINDOW_SIZE, stride=STRIDE)
        except Exception as e:
            logger.error("[%d/%d] %s -- FAILED to load: %s" % (i + 1, len(data_files), date_str, e))
            continue

        if len(dataset) == 0:
            logger.warning("[%d/%d] %s -- 0 windows, skipping" % (i + 1, len(data_files), date_str))
            continue

        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            persistent_workers=False,
        )

        # Ensemble inference: average predictions, average embeddings
        ensemble_preds = None
        ensemble_embeds = None
        labels = None

        for model in models:
            preds, lab, embeds = run_inference_single_model(model, loader, device)
            if ensemble_preds is None:
                ensemble_preds = preds
                ensemble_embeds = embeds
                labels = lab
            else:
                ensemble_preds += preds
                ensemble_embeds += embeds

        ensemble_preds /= len(models)
        ensemble_embeds /= len(models)

        n_windows = len(ensemble_preds)
        total_windows += n_windows

        if data_type == "IS":
            is_count += 1
        elif data_type == "OOT":
            oot_count += 1
        else:
            oos_count += 1

        # Compute IC for this date
        ic_strs = []
        for h_idx, h in enumerate(HORIZONS):
            valid = ~np.isnan(labels[:, h_idx])
            if valid.sum() >= 20:
                ic = scipy.stats.spearmanr(ensemble_preds[valid, h_idx], labels[valid, h_idx]).correlation
                ic_strs.append("%s=%.3f" % (h, ic))
            else:
                ic_strs.append("%s=N/A" % h)

        # Save
        np.savez_compressed(
            out_path,
            predictions=ensemble_preds.astype(np.float32),
            embeddings=ensemble_embeds.astype(np.float32),
            labels=labels.astype(np.float32),
            horizons=np.array(HORIZONS),
            date=date_str,
            data_type=data_type,
            fold_weights=np.array(FOLD_INDICES),
            n_windows=n_windows,
            window_size=WINDOW_SIZE,
            stride=STRIDE,
        )

        elapsed = time.time() - start_time
        rate = (i + 1 - skipped) / max(elapsed, 1) * 60  # dates per minute

        if (i + 1) % 10 == 0 or i == 0:
            logger.info(
                "[%d/%d] %s (%s) | %s windows | IC: %s | Total: %s windows | Rate: %.1f dates/min | Elapsed: %.0fs"
                % (i + 1, len(data_files), date_str, data_type, format(n_windows, ","),
                   ", ".join(ic_strs), format(total_windows, ","), rate, elapsed)
            )
        else:
            logger.info(
                "[%d/%d] %s (%s) | %s windows | IC: %s"
                % (i + 1, len(data_files), date_str, data_type, format(n_windows, ","),
                   ", ".join(ic_strs))
            )

        # Clean up to avoid memory buildup
        del dataset, loader, ensemble_preds, ensemble_embeds, labels
        gc.collect()

    # Final summary
    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info("BULK INFERENCE COMPLETE")
    logger.info("Total dates processed: %d" % (len(data_files) - skipped))
    logger.info("Total windows: %s" % format(total_windows, ","))
    logger.info("Date breakdown: IS=%d, OOT=%d, OOS=%d" % (is_count, oot_count, oos_count))
    logger.info("Skipped (already existed): %d" % skipped)
    logger.info("Total time: %.0fs (%.1fmin)" % (elapsed, elapsed / 60))
    logger.info("Output: %s" % OUTPUT_DIR)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
