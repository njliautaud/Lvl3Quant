#!/usr/bin/env python3
"""
regen_embeddings.py — Regenerate CNN-Mamba predictions WITH embeddings
======================================================================
Re-runs inference on dates that have predictions but NO embeddings,
saving fold-format NPZ files with 96-dim embeddings included.

Uses the ORIGINAL CNNMamba class from train_cnn_mamba.py to match
the checkpoint architecture exactly.

Usage:
    python -m alpha_discovery.execution.regen_embeddings \
        --pred-dir output/cnn_mamba_v2_all_oot \
        --weights output/cnn_mamba_v2_smart_v3_mar/fold_10_best.pt \
        --device cuda
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import scipy.stats

# Add project root to path
if sys.platform == "win32":
    LVL3_ROOT = Path("C:/Users/claude/Lvl3Quant")
else:
    LVL3_ROOT = Path(os.environ.get("LVL3_ROOT", "/home/jupiter/Lvl3Quant"))
    if not LVL3_ROOT.exists() and Path("/home/nick/Lvl3Quant").exists():
        LVL3_ROOT = Path("/home/nick/Lvl3Quant")

sys.path.insert(0, str(LVL3_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [REGEN] %(levelname)s: %(message)s",
)
log = logging.getLogger("regen_embeddings")

# Import Mamba building blocks from the training script
from alpha_discovery.deep_models.train_cnn_mamba import MambaBlock

MBO_DIR = LVL3_ROOT / "data" / "processed" / "mbo_events_smart_v3"
WINDOW_SIZE = 1000
STRIDE = 500
BATCH_SIZE = 256


class CNNMambaCompat(torch.nn.Module):
    """
    CNN-Mamba model that matches the EXACT checkpoint architecture.

    Checkpoint uses nn.Sequential for CNN (cnn.0, cnn.3, cnn.6),
    a cnn_residual 1x1 conv, and cnn_to_model projection.
    This is the architecture that produced fold_10_best.pt.
    """
    def __init__(self, n_features=25, cnn_channels=64, cnn_kernel=5,
                 d_model=96, d_state=32, n_layers=3, dt_rank=6,
                 d_conv=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # CNN front-end (Sequential style matching checkpoint)
        pad = cnn_kernel // 2
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv1d(n_features, cnn_channels, cnn_kernel, padding=pad),
            torch.nn.BatchNorm1d(cnn_channels),
            torch.nn.GELU(),
            torch.nn.Conv1d(cnn_channels, cnn_channels, cnn_kernel, padding=pad),
            torch.nn.BatchNorm1d(cnn_channels),
            torch.nn.GELU(),
            torch.nn.Conv1d(cnn_channels, cnn_channels, cnn_kernel, padding=pad),
            torch.nn.BatchNorm1d(cnn_channels),
            torch.nn.GELU(),
        )
        # 1x1 conv for residual from input to cnn output
        self.cnn_residual = torch.nn.Conv1d(n_features, cnn_channels, 1)

        # Projection to d_model
        self.cnn_to_model = torch.nn.Sequential(
            torch.nn.Linear(cnn_channels, d_model),
            torch.nn.LayerNorm(d_model),
        )

        # Mamba backbone
        self.blocks = torch.nn.ModuleList([
            MambaBlock(d_model=d_model, d_state=d_state, dt_rank=dt_rank,
                      d_conv=d_conv, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Final norm + head
        self.final_norm = torch.nn.LayerNorm(d_model)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(d_model, 3),
        )

    def forward(self, events, return_embedding=False):
        B, L, _ = events.shape
        time_delta = events[:, :, 0]

        # CNN
        x = events.transpose(1, 2).contiguous()
        residual = self.cnn_residual(x)
        x = self.cnn(x)
        x = x + residual
        x = x.transpose(1, 2).contiguous()

        # Project
        x = self.cnn_to_model(x)

        # Mamba
        for block in self.blocks:
            x = block(x, time_delta=time_delta)

        # Output
        embedding = self.final_norm(x[:, -1, :])
        preds = self.head(embedding)

        if return_embedding:
            return preds, embedding
        return preds


def load_model(weights_path: str, device: str = "cpu") -> CNNMambaCompat:
    """Load CNN-Mamba from checkpoint, matching exact architecture."""
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    state = ckpt["model_state"]
    arch = ckpt.get("arch", {})

    # Infer architecture from state dict
    d_model = arch.get("d_model", 96)
    d_state = state["blocks.0.ssm.A_log"].shape[1]
    x_proj = state["blocks.0.ssm.x_proj.weight"]
    dt_rank = x_proj.shape[0] - 2 * d_state
    n_layers = sum(1 for k in state if k.endswith(".ssm.A_log"))
    d_conv = arch.get("d_conv", 4)
    dropout = arch.get("dropout", 0.1)
    n_features = state["cnn.0.weight"].shape[1]
    cnn_channels = state["cnn.0.weight"].shape[0]
    cnn_kernel = state["cnn.0.weight"].shape[2]

    model = CNNMambaCompat(
        n_features=n_features,
        cnn_channels=cnn_channels,
        cnn_kernel=cnn_kernel,
        d_model=d_model,
        d_state=d_state,
        n_layers=n_layers,
        dt_rank=dt_rank,
        d_conv=d_conv,
        dropout=dropout,
    )

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        log.info(f"Missing keys (expected): {missing}")
    if unexpected:
        log.warning(f"Unexpected keys: {unexpected}")

    model = model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Loaded CNN-Mamba: {n_params:,} params, n_features={n_features}, "
             f"d_model={d_model}, fold={ckpt.get('fold')}, "
             f"IC_10s={ckpt.get('val_ic_10s', 0):.4f}")
    return model


@torch.no_grad()
def run_inference_on_day(
    model: CNNMambaCompat,
    events: np.ndarray,
    labels_1s: np.ndarray,
    labels_5s: np.ndarray,
    labels_10s: np.ndarray,
    device: str = "cpu",
    batch_size: int = BATCH_SIZE,
):
    """Run sliding-window inference, returning predictions + labels + embeddings."""
    n_events = len(events)
    valid_starts = [
        s for s in range(0, n_events - WINDOW_SIZE + 1, STRIDE)
        if (not np.isnan(labels_1s[s + WINDOW_SIZE - 1])
            and not np.isnan(labels_5s[s + WINDOW_SIZE - 1])
            and not np.isnan(labels_10s[s + WINDOW_SIZE - 1]))
    ]

    N = len(valid_starts)
    if N == 0:
        return (np.zeros((0, 3), np.float32),
                np.zeros((0, 3), np.float32),
                np.zeros((0, model.d_model), np.float32))

    all_preds, all_embeds, all_labels = [], [], []
    events_t = torch.from_numpy(events).to(device)

    for i in range(0, N, batch_size):
        batch_starts = valid_starts[i:i + batch_size]
        batch = torch.stack([events_t[s:s + WINDOW_SIZE] for s in batch_starts])

        preds, embeds = model(batch, return_embedding=True)
        all_preds.append(preds.cpu().numpy())
        all_embeds.append(embeds.cpu().numpy())

        all_labels.append(np.array([
            [labels_1s[s + WINDOW_SIZE - 1],
             labels_5s[s + WINDOW_SIZE - 1],
             labels_10s[s + WINDOW_SIZE - 1]]
            for s in batch_starts
        ], dtype=np.float32))

    return (np.concatenate(all_preds, 0),
            np.concatenate(all_labels, 0),
            np.concatenate(all_embeds, 0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", type=str,
                        default=str(LVL3_ROOT / "output" / "cnn_mamba_v2_all_oot"))
    parser.add_argument("--weights", type=str,
                        default=str(LVL3_ROOT / "output" / "cnn_mamba_v2_smart_v3_mar" / "fold_10_best.pt"))
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    # Find per-date files WITHOUT embeddings
    import re
    dates_needing_embeddings = []
    for f in sorted(pred_dir.glob("[0-9]*_predictions.npz")):
        match = re.match(r"(\d{8})_predictions", f.stem)
        if match:
            data = np.load(str(f), allow_pickle=True)
            if "embeddings" not in data:
                dates_needing_embeddings.append(match.group(1))

    # Also check fold files
    fold_dates_with_embeddings = set()
    for f in sorted(pred_dir.glob("fold_*_oot_predictions.npz")):
        data = np.load(str(f), allow_pickle=True)
        if "embeddings" in data and "oot_files" in data:
            oot = data["oot_files"].tolist()
            fn = str(oot[0]).replace("\\", "/").split("/")[-1] if oot else ""
            date = fn.split("_")[0]
            if len(date) == 8:
                fold_dates_with_embeddings.add(date)

    # Remove dates already covered by fold files
    dates_needing_embeddings = [d for d in dates_needing_embeddings
                                 if d not in fold_dates_with_embeddings]

    log.info(f"Found {len(dates_needing_embeddings)} dates needing embeddings")
    log.info(f"Already covered by fold files: {len(fold_dates_with_embeddings)}")

    if not dates_needing_embeddings:
        log.info("All dates have embeddings! Nothing to do.")
        return

    # Load model
    model = load_model(args.weights, device)

    # Find next fold index
    existing_folds = sorted(pred_dir.glob("fold_*_oot_predictions.npz"))
    next_fold = 0
    if existing_folds:
        try:
            next_fold = int(existing_folds[-1].name.split("_")[1]) + 1
        except:
            next_fold = len(existing_folds)

    # Process each date
    t_total = time.time()
    success = 0
    for i, date_str in enumerate(dates_needing_embeddings):
        mbo_path = MBO_DIR / f"{date_str}_mbo_events.npz"
        if not mbo_path.exists():
            log.warning(f"  {date_str}: MBO file not found, skipping")
            continue

        t0 = time.time()
        try:
            data = np.load(str(mbo_path), allow_pickle=True)
            events = data["events"].astype(np.float32)
            labels_1s = data["labels_1s"].astype(np.float32)
            labels_5s = data["labels_5s"].astype(np.float32)
            labels_10s = data["labels_10s"].astype(np.float32)

            preds, labels, embeds = run_inference_on_day(
                model, events, labels_1s, labels_5s, labels_10s,
                device=device, batch_size=args.batch_size,
            )

            if len(preds) == 0:
                log.warning(f"  {date_str}: 0 valid windows, skipping")
                continue

            # Compute ICs
            ics = []
            for col in range(3):
                valid = ~(np.isnan(preds[:, col]) | np.isnan(labels[:, col]))
                if valid.sum() < 10:
                    ics.append(float("nan"))
                else:
                    ic, _ = scipy.stats.spearmanr(preds[valid, col], labels[valid, col])
                    ics.append(float(ic))

            # Save as fold file WITH embeddings
            fold_idx = next_fold + i
            out_path = pred_dir / f"fold_{fold_idx:02d}_oot_predictions.npz"
            np.savez_compressed(
                out_path,
                predictions=preds.astype(np.float32),
                labels=labels.astype(np.float32),
                horizons=np.array(["1s", "5s", "10s"]),
                ic_1s=np.float64(ics[0]),
                ic_5s=np.float64(ics[1]),
                ic_10s=np.float64(ics[2]),
                oot_files=np.array([str(mbo_path)]),
                embeddings=embeds.astype(np.float32),
            )

            elapsed = time.time() - t0
            log.info(f"  {date_str}: {len(preds):,} windows, "
                     f"IC=[{ics[0]:.4f}, {ics[1]:.4f}, {ics[2]:.4f}], "
                     f"emb={embeds.shape}, {elapsed:.1f}s → fold_{fold_idx:02d}")
            success += 1

        except Exception as e:
            log.error(f"  {date_str}: FAILED — {e}")
            import traceback
            traceback.print_exc()

    total_time = time.time() - t_total
    log.info(f"\nDone: {success}/{len(dates_needing_embeddings)} dates in {total_time:.1f}s")


if __name__ == "__main__":
    main()
