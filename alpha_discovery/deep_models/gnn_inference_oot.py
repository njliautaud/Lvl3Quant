#!/usr/bin/env python3
"""
GNN Inference-Only: Load pre-trained fold checkpoints and generate OOT predictions.
No IS data needed -- just loads checkpoints and runs inference on OOT book cache.

Ensembles predictions from all available fold checkpoints (mean).

Usage:
    python3 gnn_inference_oot.py
    python3 gnn_inference_oot.py --ckpt-dir checkpoints --oot-dir /path/to/oot
"""

import argparse
import gc
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT_DIR = Path(__file__).resolve().parent.parent.parent  # Lvl3Quant root
MODELS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(MODELS_DIR))

from book_gnn import BookGNN, count_parameters

# Use all available CPU threads
torch.set_num_threads(min(32, torch.get_num_threads() * 2))

logging.basicConfig(
    format="%(asctime)s [gnn_inf] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("gnn_inf")


class GNNInferenceDataset(Dataset):
    def __init__(self, tensors, n_bars, window_size=20):
        self.window_size = window_size
        self.tensors = tensors.astype(np.float32).copy()
        self.tensors[:, :, 1] = np.log1p(self.tensors[:, :, 1])
        self.tensors[:, :, 2] = np.log1p(self.tensors[:, :, 2])
        self.tensors[:, :, 3] = np.log1p(self.tensors[:, :, 3])
        self.valid_indices = np.arange(window_size - 1, n_bars, dtype=np.int64)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        i = int(self.valid_indices[idx])
        window = self.tensors[i - self.window_size + 1 : i + 1]
        return torch.from_numpy(window.copy()), i


def collate_inference(batch):
    windows = torch.stack([b[0] for b in batch])
    indices = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return windows, indices


def load_model(ckpt_path, device):
    ck = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    args = ck.get("args", {})

    model = BookGNN(
        window_size=args.get("window_size", 20),
        num_nodes=20,
        in_features=4,
        node_features=5,
        hidden_dim=args.get("hidden", 64),
        num_gcn_layers=args.get("layers", 3),
        temporal_dim=args.get("temporal_dim", 128),
        dropout=0.0,  # no dropout at inference
        num_classes=1,
        use_gat=args.get("use_gat", False),
    ).to(device)

    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model, ck.get("ic", 0), ck.get("fold", 0), ck.get("epoch", 0)


@torch.no_grad()
def predict_day(model, book_tensors, n_bars, device, window_size=20, batch_size=512):
    predictions = np.zeros(n_bars, dtype=np.float64)
    dataset = GNNInferenceDataset(book_tensors, n_bars, window_size=window_size)
    if len(dataset) == 0:
        return predictions
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                       num_workers=0, collate_fn=collate_inference)
    for windows, indices in loader:
        try:
            windows = windows.to(device)
            preds = model(windows).squeeze(-1)
            preds = preds.cpu().float().numpy()
            indices = indices.numpy()
            predictions[indices] = preds
        except RuntimeError as e:
            if "CUDA" in str(e):
                # Fall back to CPU for this batch
                windows_cpu = windows.cpu()
                model_cpu = model.cpu()
                preds = model_cpu(windows_cpu).squeeze(-1).numpy()
                model.to(device)
                indices = indices.numpy() if torch.is_tensor(indices) else indices
                predictions[indices] = preds
            else:
                raise
    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", type=str,
                       default=str(MODELS_DIR / "checkpoints"))
    parser.add_argument("--oot-dir", type=str,
                       default=str(ROOT_DIR / "data" / "processed" / "dl_book_cache_oot"))
    parser.add_argument("--output-dir", type=str,
                       default=str(MODELS_DIR / "results"))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--top-n-models", type=int, default=0,
                       help="Use only top N models by val IC (0=all)")
    parser.add_argument("--resume-from", type=str, default=None,
                       help="Resume from partial NPZ file")
    args = parser.parse_args()

    device = torch.device(args.device)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / ("gnn_inference_oot_" + timestamp + ".log")
    fh = logging.FileHandler(str(log_path), mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s [gnn_inf] %(levelname)s: %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(fh)

    logger.info("=" * 70)
    logger.info("GNN Inference-Only OOT Prediction Generation")
    logger.info("  Checkpoints: " + args.ckpt_dir)
    logger.info("  OOT data:    " + args.oot_dir)
    logger.info("  Device:      " + str(device))
    logger.info("=" * 70)

    # -- Find best checkpoint per fold --
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_files = sorted(ckpt_dir.glob("gnn_fold*.pt"))

    fold_best = defaultdict(lambda: (0, None))
    for f in ckpt_files:
        parts = f.stem.split("_")
        fold = int(parts[1].replace("fold", ""))
        epoch = int(parts[2].replace("epoch", ""))
        if epoch > fold_best[fold][0]:
            fold_best[fold] = (epoch, f)

    logger.info("Found " + str(len(fold_best)) + " fold checkpoints")

    # -- Load all models --
    models = []
    for fold in sorted(fold_best.keys()):
        epoch, ckpt_path = fold_best[fold]
        model, ic, f_num, ep = load_model(ckpt_path, device)
        models.append((model, fold, ic))
        logger.info("  Fold " + str(fold) + ": epoch " + str(epoch) + ", val IC=" + format(ic, "+.4f"))

    # Optionally keep only top N models by IC
    if args.top_n_models > 0 and args.top_n_models < len(models):
        models.sort(key=lambda x: x[2], reverse=True)
        models = models[:args.top_n_models]
        logger.info("Using top " + str(args.top_n_models) + " models: folds " +
                    str([m[1] for m in models]))

    logger.info("Loaded " + str(len(models)) + " models for ensemble")

    # -- Load OOT data --
    oot_dir = Path(args.oot_dir)
    oot_files = sorted(oot_dir.glob("*_book_tensors.npz"))
    logger.info("OOT: " + str(len(oot_files)) + " days")

    # -- Resume from partial results if available --
    pred_data = {}
    completed_dates = set()
    if args.resume_from and Path(args.resume_from).exists():
        logger.info("Resuming from " + args.resume_from)
        old = np.load(args.resume_from)
        for key in old.files:
            pred_data[key] = old[key]
            if key.endswith("_preds"):
                completed_dates.add(key.replace("_preds", ""))
        logger.info("  Loaded " + str(len(completed_dates)) + " completed days")

    # -- Generate predictions --
    t_start = time.time()

    for i, f in enumerate(oot_files):
        date = f.name.replace("_book_tensors.npz", "")
        if date in completed_dates:
            logger.info("  Skipping " + date + " (already done)")
            continue
        npz = np.load(str(f))
        book_tensors = npz["book_tensors"]
        mid_prices = npz["mid_prices"]
        n_bars = len(mid_prices)

        # Ensemble: average predictions from all fold models
        all_preds = []
        for model, fold, ic in models:
            preds = predict_day(model, book_tensors, n_bars, device,
                              window_size=20, batch_size=args.batch_size)
            all_preds.append(preds)

        ensemble_preds = np.mean(all_preds, axis=0)
        pred_data[date + "_preds"] = ensemble_preds.astype(np.float64)
        pred_data[date + "_mid"] = mid_prices

        if (i + 1) % 5 == 0 or i == 0 or i == len(oot_files) - 1:
            elapsed = time.time() - t_start
            done_count = len(pred_data) // 2
            rate = done_count / elapsed if elapsed > 0 else 1
            remaining = len(oot_files) - done_count
            eta = remaining / rate if rate > 0 else 0
            logger.info("  [" + str(done_count) + "/" + str(len(oot_files)) + "] " + date + ": " +
                       str(n_bars) + " bars, " +
                       "pred range [" + format(ensemble_preds.min(), ".3f") + ", " +
                       format(ensemble_preds.max(), ".3f") + "], " +
                       "ETA " + format(eta, ".0f") + "s")
            # Flush log
            for handler in logger.handlers:
                handler.flush()

        # Save checkpoint every 20 days
        if (i + 1) % 20 == 0:
            partial_file = output_dir / ("gnn_oot_partial_" + timestamp + ".npz")
            np.savez_compressed(str(partial_file), **pred_data)
            logger.info("  Saved partial: " + str(partial_file))

        del npz, book_tensors, all_preds
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # -- Save predictions --
    pred_file = output_dir / ("oos_predictions_gnn_oot_" + timestamp + ".npz")
    np.savez_compressed(str(pred_file), **pred_data)

    elapsed = time.time() - t_start
    logger.info("")
    logger.info("Done! " + str(len(oot_files)) + " days in " + format(elapsed, ".1f") + "s")
    logger.info("Predictions saved: " + str(pred_file))
    logger.info("Keys: " + str(len(pred_data)) + " (" + str(len(oot_files)) + " days x 2)")

    # -- Summary --
    logger.info("")
    logger.info("=" * 70)
    logger.info("COMPLETE")
    logger.info("  Output: " + str(pred_file))
    logger.info("  Days:   " + str(len(oot_files)))
    logger.info("  Models: " + str(len(models)) + " fold ensemble")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
