#!/usr/bin/env python3
"""
Microstructure Feature Encoder v1 — Self-supervised pretraining (HC #469 R5h, e4)
==================================================================================
Masked-feature reconstruction (tabular MLM) on the 41 microstructure features
from /home/nick/Lvl3Quant/output/exec_features_v1/*_exec_features.npz.

NO labels needed. Encoder weights are a portable artifact for downstream use:
  (a) queue-position v1 (incoming)
  (b) fill-prob v3 on MBO-walker labels
  (c) optionally aux features for v3.5

Architecture (small ~50-200k params):
  Input: 41 micro feats + 41 mask bits = 82 dim
  Encoder: Linear(82->128) + GELU + LN + Linear(128->64) + GELU + LN + Linear(64->32)
  Decoder: Linear(32->64) + GELU + LN + Linear(64->128) + GELU + LN + Linear(128->41)
  Loss: MSE on masked positions only

Mask rate: 15%. Batch 4096. Adam lr=1e-3. 1 epoch. Wall-clock cap ~25 min.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

LVL3_ROOT = Path("/home/nick/Lvl3Quant")
EXEC_DIR = LVL3_ROOT / "output" / "exec_features_v1"
OUTPUT_DIR = LVL3_ROOT / "output" / "microstructure_encoder_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [MICRO_ENC] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("micro_enc")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MLFLOW_URI = "http://localhost:5000"
EXPERIMENT_NAME = "microstructure_encoder_v1"

N_FEATURES = 41
MASK_RATE = 0.15
EMBED_DIM = 32


# ---------- Data ----------
def load_all_micro_features() -> tuple[np.ndarray, list[str]]:
    """Load and concatenate the 41 micro features from every date npz.

    Drops the 3 fill_prob_* columns (potential leakage, matches v2.1 behavior).
    Returns (X (N, 41) float32, feature_names list).
    """
    files = sorted(EXEC_DIR.glob("*_exec_features.npz"))
    if not files:
        raise FileNotFoundError(f"No exec_features npz files in {EXEC_DIR}")
    log.info("Found %d date files in %s", len(files), EXEC_DIR)

    chunks = []
    names = None
    for i, p in enumerate(files):
        d = np.load(str(p), allow_pickle=True)
        feats = d["features"]  # (n_windows, 44)
        cur_names = [str(n) for n in d["feature_names"]]
        keep_idx = [j for j, n in enumerate(cur_names) if not n.startswith("fill_prob_")]
        assert len(keep_idx) == N_FEATURES, (
            f"expected {N_FEATURES} micro feats, got {len(keep_idx)} in {p.name}"
        )
        if names is None:
            names = [cur_names[j] for j in keep_idx]
        chunks.append(feats[:, keep_idx].astype(np.float32))
        if (i + 1) % 50 == 0:
            log.info("  loaded %d/%d files", i + 1, len(files))

    X = np.concatenate(chunks, axis=0)
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    log.info("Loaded %d total exec-window rows, %d features", X.shape[0], X.shape[1])
    return X, names


def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-feature z-score normalization. Returns (X_norm, mean, std)."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    X_norm = ((X - mean) / std).astype(np.float32)
    # Clip extreme outliers to stabilize MSE
    X_norm = np.clip(X_norm, -10.0, 10.0)
    return X_norm, mean.astype(np.float32), std.astype(np.float32)


# ---------- Model ----------
class MaskedAutoEncoder(nn.Module):
    def __init__(self, n_feat: int = N_FEATURES, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.n_feat = n_feat
        # Input = features + mask bits
        self.encoder = nn.Sequential(
            nn.Linear(n_feat * 2, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, embed_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, n_feat),
        )

    def encode(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x_masked: zero out masked positions
        x_in = x * (1.0 - mask)
        h = torch.cat([x_in, mask], dim=-1)
        return self.encoder(h)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        z = self.encode(x, mask)
        recon = self.decoder(z)
        return recon


def make_mask(batch_size: int, n_feat: int, mask_rate: float, device) -> torch.Tensor:
    """Bernoulli mask, ensure at least 1 masked per row."""
    m = (torch.rand(batch_size, n_feat, device=device) < mask_rate).float()
    # If any row has zero masks, force one position
    row_sum = m.sum(dim=1, keepdim=True)
    needs_mask = (row_sum == 0).squeeze(-1)
    if needs_mask.any():
        idx = torch.randint(0, n_feat, (int(needs_mask.sum().item()),), device=device)
        m[needs_mask, idx] = 1.0
    return m


# ---------- Train ----------
def masked_mse(recon: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    err = (recon - target) ** 2
    err = err * mask  # zero unmasked positions
    n_masked = mask.sum().clamp_min(1.0)
    return err.sum() / n_masked


def train_one_epoch(model, loader, optimizer, device, max_batches=None, log_every=50):
    model.train()
    total_loss = 0.0
    n_batches = 0
    t0 = time.time()
    for i, (xb,) in enumerate(loader):
        xb = xb.to(device, non_blocking=True)
        mask = make_mask(xb.size(0), xb.size(1), MASK_RATE, device)
        recon = model(xb, mask)
        loss = masked_mse(recon, xb, mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
        if (i + 1) % log_every == 0:
            elapsed = time.time() - t0
            log.info("  batch %d  loss=%.5f  rate=%.0f rows/s",
                     i + 1, total_loss / n_batches, (i + 1) * xb.size(0) / elapsed)
        if max_batches is not None and (i + 1) >= max_batches:
            break
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def eval_loss(model, loader, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for (xb,) in loader:
        xb = xb.to(device, non_blocking=True)
        mask = make_mask(xb.size(0), xb.size(1), MASK_RATE, device)
        recon = model(xb, mask)
        loss = masked_mse(recon, xb, mask)
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def embedding_sanity_check(model, X_tensor, device, n_sample=1000):
    """Return per-dim mean/std and pairwise cosine sim p50/p90."""
    model.eval()
    n = X_tensor.size(0)
    idx = torch.randperm(n)[:n_sample]
    x = X_tensor[idx].to(device)
    mask = torch.zeros_like(x)  # no masking for inference embedding
    z = model.encode(x, mask).cpu().numpy()
    per_dim_mean = z.mean(axis=0).tolist()
    per_dim_std = z.std(axis=0).tolist()
    # cosine sim on a random pair subset (n_sample x n_sample is too big — sample 2000 pairs)
    n_z = z.shape[0]
    rng = np.random.default_rng(0)
    a_idx = rng.integers(0, n_z, 2000)
    b_idx = rng.integers(0, n_z, 2000)
    a = z[a_idx]
    b = z[b_idx]
    eps = 1e-8
    cos = (a * b).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + eps)
    return {
        "embed_mean_per_dim": per_dim_mean,
        "embed_std_per_dim": per_dim_std,
        "cos_sim_p50": float(np.percentile(cos, 50)),
        "cos_sim_p90": float(np.percentile(cos, 90)),
        "cos_sim_mean": float(cos.mean()),
        "cos_sim_min": float(cos.min()),
        "cos_sim_max": float(cos.max()),
        "embed_norm_mean": float(np.linalg.norm(z, axis=1).mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Smoke test: 1000 batches only")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    log.info("Device: %s", DEVICE)
    if DEVICE.type == "cuda":
        log.info("GPU: %s", torch.cuda.get_device_name(0))

    # ---- Load data ----
    t0 = time.time()
    X_raw, feat_names = load_all_micro_features()
    log.info("Data load time: %.1fs", time.time() - t0)

    X_norm, mu, sd = standardize(X_raw)
    log.info("Standardized: mean range [%.3f, %.3f], std range [%.3f, %.3f]",
             float(X_norm.mean(axis=0).min()), float(X_norm.mean(axis=0).max()),
             float(X_norm.std(axis=0).min()), float(X_norm.std(axis=0).max()))

    X_tensor = torch.from_numpy(X_norm)
    dataset = TensorDataset(X_tensor)
    n_total = len(dataset)
    n_val = int(n_total * args.val_frac)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
    )
    log.info("Train rows: %d, Val rows: %d", n_train, n_val)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=(DEVICE.type == "cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=(DEVICE.type == "cuda"),
    )

    # ---- Model ----
    model = MaskedAutoEncoder(N_FEATURES, EMBED_DIM).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model params: %d", n_params)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ---- MLflow ----
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    run_name = f"micro_enc_v1_{time.strftime('%Y%m%d_%H%M')}"
    mlflow.start_run(run_name=run_name)
    mlflow.log_params({
        "n_features": N_FEATURES,
        "embed_dim": EMBED_DIM,
        "mask_rate": MASK_RATE,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "n_train": n_train,
        "n_val": n_val,
        "n_params": n_params,
        "smoke": args.smoke,
    })

    # ---- Initial loss (before training) ----
    init_loss = eval_loss(model, val_loader, DEVICE)
    log.info("Initial val loss (random init): %.5f", init_loss)
    mlflow.log_metric("val_loss_init", init_loss)

    train_curve = []
    val_curve = []

    max_batches = 1000 if args.smoke else None

    for epoch in range(args.epochs):
        log.info("=== Epoch %d ===", epoch + 1)
        t_ep = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, DEVICE,
            max_batches=max_batches, log_every=100,
        )
        val_loss = eval_loss(model, val_loader, DEVICE)
        log.info("Epoch %d done in %.1fs. train_loss=%.5f val_loss=%.5f",
                 epoch + 1, time.time() - t_ep, train_loss, val_loss)
        train_curve.append(train_loss)
        val_curve.append(val_loss)
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)

    # ---- Sanity check ----
    log.info("Running embedding sanity check...")
    sanity = embedding_sanity_check(model, X_tensor, DEVICE, n_sample=1000)
    log.info("Cosine sim p50=%.4f p90=%.4f mean=%.4f norm_mean=%.4f",
             sanity["cos_sim_p50"], sanity["cos_sim_p90"],
             sanity["cos_sim_mean"], sanity["embed_norm_mean"])
    mlflow.log_metric("cos_sim_p50", sanity["cos_sim_p50"])
    mlflow.log_metric("cos_sim_p90", sanity["cos_sim_p90"])
    mlflow.log_metric("embed_norm_mean", sanity["embed_norm_mean"])

    collapsed = sanity["cos_sim_p50"] > 0.95
    if collapsed:
        log.warning("ENCODER COLLAPSED: cos_sim_p50=%.4f > 0.95", sanity["cos_sim_p50"])
        mlflow.set_tag("warning", "encoder_collapsed")

    # ---- Save artifacts ----
    enc_path = OUTPUT_DIR / "encoder.pt"
    full_path = OUTPUT_DIR / "full_model.pt"
    summary_path = OUTPUT_DIR / "summary.json"

    torch.save(model.encoder.state_dict(), str(enc_path))
    torch.save({
        "model_state": model.state_dict(),
        "n_features": N_FEATURES,
        "embed_dim": EMBED_DIM,
        "feature_names": feat_names,
        "feature_mean": mu.tolist(),
        "feature_std": sd.tolist(),
    }, str(full_path))

    summary = {
        "run_name": run_name,
        "mlflow_experiment": EXPERIMENT_NAME,
        "n_params": n_params,
        "n_features": N_FEATURES,
        "embed_dim": EMBED_DIM,
        "mask_rate": MASK_RATE,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "n_train": n_train,
        "n_val": n_val,
        "feature_names": feat_names,
        "feature_mean": mu.tolist(),
        "feature_std": sd.tolist(),
        "init_val_loss": init_loss,
        "train_loss_curve": train_curve,
        "val_loss_curve": val_curve,
        "final_train_loss": train_curve[-1] if train_curve else None,
        "final_val_loss": val_curve[-1] if val_curve else None,
        "sanity": sanity,
        "collapsed_warning": collapsed,
        "smoke": args.smoke,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Saved encoder -> %s", enc_path)
    log.info("Saved full model -> %s", full_path)
    log.info("Saved summary -> %s", summary_path)

    mlflow.log_artifact(str(enc_path))
    mlflow.log_artifact(str(full_path))
    mlflow.log_artifact(str(summary_path))
    mlflow.end_run()

    log.info("DONE. final_val_loss=%.5f cos_sim_p50=%.4f",
             val_curve[-1] if val_curve else float("nan"), sanity["cos_sim_p50"])


if __name__ == "__main__":
    main()
