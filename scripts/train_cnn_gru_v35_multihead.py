#!/usr/bin/env python3
"""
train_cnn_gru_v35_multihead.py — HC #499 Razer-friendly continuation trainer.

Self-contained CNN+GRU multi-head trainer for the v3.5 continuation/pressure
objective (HC #497 R5 + HC #499). Built for Razer (Windows, RTX 3070 8GB)
because mamba_ssm cannot be installed on Windows.

Architecture is a drop-in substitute for the CNN-Mamba v35 trainer:
  - 1D causal CNN feature extractor (3 layers, dilated)
  - 2-layer GRU on top (hidden=128, last hidden state = embedding)
  - Same 4 task heads (A: 4 binary, B: scalar, C: 3×3 softmax, D: 3-class)
  - Same multi_head_loss
  - Same MultiHeadDataset (loads v35 .npz files)

Authorization: HC #420 — user's own legitimate quant research codebase.

Env vars (subset):
  V35_DATA_DIR       (default ./data/processed/mbo_events_smart_v3_v35)
  V35_OUTPUT_DIR     (default ./output/cnn_gru_v35_multihead)
  EVENT_WINDOW_SIZE  (default 1500)
  EVENT_STRIDE       (default 250)
  EVENT_BATCH_SIZE   (default 32 for 3070-8GB; bump to 64 on bigger GPU)
  EVENT_LR           (default 3e-4)
  EVENT_EPOCHS       (default 3)
  EVENT_N_FOLDS      (default 4)
  WF_WINDOW_DAYS     (default 20, sliding)
  HEAD_WEIGHTS       (default "1,1,1,1")
  MLFLOW_EXPERIMENT  (default "cnn_gru_v35_multihead")
  MLFLOW_TRACKING_URI (default http://jupiter:5000 — Jupiter tailscale)
  V35_SMOKE          (set 1 for tiny smoke)
"""
from __future__ import annotations

import math
import os
import sys
import time
import logging
import socket
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ── MLflow ───────────────────────────────────────────────────────────────────
try:
    import mlflow
    MLFLOW_AVAILABLE = True
    if os.environ.get("DISABLE_MLFLOW", "0") == "1":
        MLFLOW_AVAILABLE = False
except ImportError:
    MLFLOW_AVAILABLE = False

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("train_v35_gru")

# ── Config ───────────────────────────────────────────────────────────────────
V35_SMOKE = int(os.environ.get("V35_SMOKE", "0")) == 1

# Default data dir resolves to ~/Lvl3Quant on Razer/Jupiter
_DEFAULT_DATA = str(Path.home() / "Lvl3Quant" / "data" / "processed" / "mbo_events_smart_v3_v35")
V35_DATA_DIR = Path(os.environ.get("V35_DATA_DIR", _DEFAULT_DATA))

HEAD_WEIGHTS_STR = os.environ.get("HEAD_WEIGHTS", "1,1,1,1")
HEAD_WEIGHTS = [float(x) for x in HEAD_WEIGHTS_STR.split(",")]
assert len(HEAD_WEIGHTS) == 4, "HEAD_WEIGHTS must be 4 comma-separated floats"

WINDOW_SIZE = int(os.environ.get("EVENT_WINDOW_SIZE", "1500" if not V35_SMOKE else "256"))
STRIDE = int(os.environ.get("EVENT_STRIDE", "250" if not V35_SMOKE else "128"))
BATCH_SIZE = int(os.environ.get("EVENT_BATCH_SIZE", "32" if not V35_SMOKE else "8"))
LR = float(os.environ.get("EVENT_LR", "3e-4"))
EPOCHS_PER_FOLD = int(os.environ.get("EVENT_EPOCHS", "3" if not V35_SMOKE else "1"))
N_FOLDS = int(os.environ.get("EVENT_N_FOLDS", "4" if not V35_SMOKE else "1"))
WF_WINDOW_DAYS = int(os.environ.get("WF_WINDOW_DAYS", "20"))
GRAD_CLIP = float(os.environ.get("EVENT_GRAD_CLIP", "1.0"))
WARMUP_STEPS = int(os.environ.get("EVENT_WARMUP", "300"))
HORIZONS_A_K = [5, 10, 30, 60]
HEAD_C_K = [2, 4, 8]
N_FEATURES = int(os.environ.get("N_FEATURES", "25"))  # smart_v3
D_MODEL = int(os.environ.get("D_MODEL", "128"))

MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "cnn_gru_v35_multihead")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://jupiter:5000")

OUTPUT_DIR = Path(os.environ.get(
    "V35_OUTPUT_DIR",
    str(Path.home() / "Lvl3Quant" / "output" / "cnn_gru_v35_multihead"),
))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Dataset (identical to v35 mamba trainer) ─────────────────────────────────
class MultiHeadDataset(Dataset):
    def __init__(self, npz_files: List[Path], window_size: int = WINDOW_SIZE, stride: int = STRIDE):
        self.window_size = window_size
        self.stride = stride
        self.npz_files = list(npz_files)
        self.day_data: List[Dict[str, np.ndarray]] = []
        self.sample_index: List[Tuple[int, int]] = []
        for day_idx, f in enumerate(self.npz_files):
            with np.load(f, allow_pickle=False) as z:
                day = {
                    "events": z["events"].astype(np.float32),
                    "A_5s": z["label_A_5s"].astype(np.float32),
                    "A_10s": z["label_A_10s"].astype(np.float32),
                    "A_30s": z["label_A_30s"].astype(np.float32),
                    "A_60s": z["label_A_60s"].astype(np.float32),
                    "B_p": z["label_B_persistence_s"].astype(np.float32),
                    "B_c": z["label_B_censored"].astype(np.uint8),
                    "C_K2": z["label_C_K2"].astype(np.int8),
                    "C_K4": z["label_C_K4"].astype(np.int8),
                    "C_K8": z["label_C_K8"].astype(np.int8),
                    "D": z["label_D_regime_60s"].astype(np.int8),
                }
            n_events = day["events"].shape[0]
            for s in range(0, n_events - window_size + 1, stride):
                self.sample_index.append((day_idx, s))
            self.day_data.append(day)
        log.info(f"MultiHeadDataset: {len(self.npz_files)} days, "
                 f"{len(self.sample_index)} samples (W={window_size}, S={stride})")

    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, idx):
        day_idx, start = self.sample_index[idx]
        end = start + self.window_size
        d = self.day_data[day_idx]
        events = torch.from_numpy(d["events"][start:end])  # (W, F)
        last = end - 1
        labels = {
            "A_5s": float(d["A_5s"][last]),
            "A_10s": float(d["A_10s"][last]),
            "A_30s": float(d["A_30s"][last]),
            "A_60s": float(d["A_60s"][last]),
            "B_p": float(d["B_p"][last]),
            "B_c": int(d["B_c"][last]),
            "C_K2": int(d["C_K2"][last]) + 1,
            "C_K4": int(d["C_K4"][last]) + 1,
            "C_K8": int(d["C_K8"][last]) + 1,
            "D": int(d["D"][last]),
        }
        return events, labels


def _collate(batch):
    events = torch.stack([b[0] for b in batch], dim=0)
    keys = batch[0][1].keys()
    out = {}
    for k in keys:
        vals = [b[1][k] for b in batch]
        if k.startswith("A_") or k == "B_p":
            out[k] = torch.tensor(vals, dtype=torch.float32)
        else:
            out[k] = torch.tensor(vals, dtype=torch.long)
    return events, out


# ── Causal CNN block (mirrors v2 CNN frontend, no padding leak) ──────────────
class CausalConv1d(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel_size: int = 5, dilation: int = 1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(c_in, c_out, kernel_size, dilation=dilation, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T)
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class CNNGRUEncoder(nn.Module):
    """Replacement for CNN-Mamba v2 backbone.

    Input:  (B, W, F)  — F = N_FEATURES
    Output: (B, D)     — D = D_MODEL  (last GRU hidden state of last layer)
    """
    def __init__(self, n_features: int = N_FEATURES, d_model: int = D_MODEL):
        super().__init__()
        self.d_model = d_model
        c1 = d_model // 2
        c2 = d_model
        self.cnn = nn.Sequential(
            CausalConv1d(n_features, c1, kernel_size=5, dilation=1),
            nn.GELU(),
            nn.LayerNorm([c1]) if False else nn.Identity(),  # avoid (B,C,T) LN shape mismatch
            CausalConv1d(c1, c1, kernel_size=5, dilation=2),
            nn.GELU(),
            CausalConv1d(c1, c2, kernel_size=5, dilation=4),
            nn.GELU(),
        )
        self.gru = nn.GRU(
            input_size=c2,
            hidden_size=d_model,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        # events: (B, W, F)
        x = events.transpose(1, 2)            # (B, F, W)
        x = self.cnn(x)                       # (B, C2, W)
        x = x.transpose(1, 2)                 # (B, W, C2)
        out, h = self.gru(x)                  # h: (num_layers, B, D)
        emb = self.norm(h[-1])                # (B, D)  — last layer last hidden
        return emb


# ── Multi-head model ─────────────────────────────────────────────────────────
class CNNGRUV35MultiHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = CNNGRUEncoder(N_FEATURES, D_MODEL)
        d = self.encoder.d_model
        self.head_A = nn.Linear(d, len(HORIZONS_A_K))
        self.head_B = nn.Linear(d, 1)
        self.head_C = nn.Linear(d, len(HEAD_C_K) * 3)
        self.head_D = nn.Linear(d, 3)

    def forward(self, events: torch.Tensor) -> Dict[str, torch.Tensor]:
        emb = self.encoder(events)
        return {
            "logits_A": self.head_A(emb),
            "pred_B": self.head_B(emb).squeeze(-1),
            "logits_C": self.head_C(emb).view(emb.shape[0], len(HEAD_C_K), 3),
            "logits_D": self.head_D(emb),
        }


# ── Multi-task loss (identical to v35 mamba) ─────────────────────────────────
def multi_head_loss(outputs, labels, weights=HEAD_WEIGHTS):
    w_A, w_B, w_C, w_D = weights
    losses = {}

    A_targets = torch.stack([labels["A_5s"], labels["A_10s"],
                             labels["A_30s"], labels["A_60s"]], dim=1)
    A_logits = outputs["logits_A"]
    A_valid = ~torch.isnan(A_targets)
    if A_valid.any():
        loss_A = F.binary_cross_entropy_with_logits(
            A_logits[A_valid], A_targets[A_valid], reduction="mean")
    else:
        loss_A = torch.tensor(0.0, device=A_logits.device)
    losses["A"] = float(loss_A.detach().item())

    B_pred = outputs["pred_B"]
    B_target = labels["B_p"]
    B_cens = labels["B_c"].bool()
    B_valid = (~B_cens) & (~torch.isnan(B_target))
    if B_valid.any():
        loss_B = F.smooth_l1_loss(B_pred[B_valid], B_target[B_valid])
    else:
        loss_B = torch.tensor(0.0, device=B_pred.device)
    losses["B"] = float(loss_B.detach().item())

    C_logits = outputs["logits_C"]
    C_targets = torch.stack([labels["C_K2"], labels["C_K4"], labels["C_K8"]], dim=1)
    B_, K_, C_ = C_logits.shape
    loss_C = F.cross_entropy(C_logits.reshape(B_ * K_, C_),
                             C_targets.reshape(B_ * K_))
    losses["C"] = float(loss_C.detach().item())

    D_logits = outputs["logits_D"]
    D_targets = labels["D"]
    loss_D = F.cross_entropy(D_logits, D_targets)
    losses["D"] = float(loss_D.detach().item())

    total = w_A * loss_A + w_B * loss_B + w_C * loss_C + w_D * loss_D
    return total, losses


# ── Warmup + cosine LR scheduler ─────────────────────────────────────────────
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.05):
        self.opt = optimizer
        self.warmup = max(1, warmup_steps)
        self.total = max(self.warmup + 1, total_steps)
        self.min_ratio = min_lr_ratio
        self.step_n = 0
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]

    def step(self):
        self.step_n += 1
        if self.step_n < self.warmup:
            factor = self.step_n / self.warmup
        else:
            progress = (self.step_n - self.warmup) / max(1, self.total - self.warmup)
            progress = min(1.0, progress)
            factor = self.min_ratio + (1.0 - self.min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
        for g, base in zip(self.opt.param_groups, self.base_lrs):
            g["lr"] = base * factor


# ── Fold builder (identical to v35 mamba) ────────────────────────────────────
def build_folds(npz_files, n_folds, wf_window_days, smoke=False):
    npz_files = sorted(npz_files)
    n = len(npz_files)
    if smoke:
        mid = max(1, n // 2)
        return [(0, list(range(mid)), list(range(mid, n)))]
    min_train = max(5, n - n_folds)
    folds = []
    for f in range(n_folds):
        train_end = min_train + f
        oot_start = train_end
        oot_end = min(oot_start + max(1, (n - min_train) // n_folds), n)
        if oot_start >= n:
            break
        train_start = max(0, train_end - wf_window_days) if wf_window_days > 0 else 0
        folds.append((f, list(range(train_start, train_end)), list(range(oot_start, oot_end))))
    return folds


# ── Train one fold ───────────────────────────────────────────────────────────
def train_one_fold(model, train_loader, val_loader, fold_idx, output_dir,
                   device, total_train_steps, mlflow_run=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    scheduler = WarmupCosineScheduler(optimizer, WARMUP_STEPS, total_train_steps)

    best_val_loss = float("inf")
    step = 0
    for epoch in range(EPOCHS_PER_FOLD):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()
        for events, labels in train_loader:
            events = events.to(device, non_blocking=True)
            labels = {k: v.to(device, non_blocking=True) for k, v in labels.items()}
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = model(events)
                    total_loss, per_head = multi_head_loss(outputs, labels)
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(events)
                total_loss, per_head = multi_head_loss(outputs, labels)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
            scheduler.step()
            epoch_loss += float(total_loss.detach().item())
            n_batches += 1
            step += 1
            if n_batches % 50 == 0:
                log.info(
                    f"  Fold {fold_idx} Ep {epoch+1} Batch {n_batches}/{len(train_loader)} "
                    f"| total={epoch_loss/n_batches:.4f} "
                    f"A={per_head['A']:.3f} B={per_head['B']:.3f} "
                    f"C={per_head['C']:.3f} D={per_head['D']:.3f}"
                )

        avg = epoch_loss / max(n_batches, 1)
        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for events, labels in val_loader:
                events = events.to(device, non_blocking=True)
                labels = {k: v.to(device, non_blocking=True) for k, v in labels.items()}
                if use_amp:
                    with torch.amp.autocast("cuda"):
                        outputs = model(events)
                        total_loss, _ = multi_head_loss(outputs, labels)
                else:
                    outputs = model(events)
                    total_loss, _ = multi_head_loss(outputs, labels)
                val_loss_sum += float(total_loss.detach().item())
                val_n += 1
        val_loss = val_loss_sum / max(val_n, 1)
        log.info(
            f"Fold {fold_idx} Epoch {epoch+1}/{EPOCHS_PER_FOLD} | "
            f"train={avg:.4f} val={val_loss:.4f} "
            f"time={time.time()-t0:.1f}s"
        )
        if MLFLOW_AVAILABLE and mlflow_run is not None:
            try:
                mlflow.log_metrics({
                    f"fold{fold_idx}_train_loss": avg,
                    f"fold{fold_idx}_val_loss": val_loss,
                }, step=epoch)
            except Exception as e:
                log.warning(f"MLflow log failed: {e}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = output_dir / f"fold_{fold_idx:02d}_best.pt"
            torch.save({"model_state": model.state_dict(),
                        "val_loss": val_loss,
                        "fold": fold_idx,
                        "epoch": epoch}, ckpt_path)
            log.info(f"  Saved best ckpt -> {ckpt_path}")
    return {"best_val_loss": best_val_loss}


def run_oot_inference(model, loader, device):
    model.eval()
    all_A, all_B, all_C, all_D = [], [], [], []
    with torch.no_grad():
        for events, _labels in loader:
            events = events.to(device, non_blocking=True)
            outputs = model(events)
            all_A.append(torch.sigmoid(outputs["logits_A"]).cpu().numpy())
            all_B.append(outputs["pred_B"].cpu().numpy())
            all_C.append(F.softmax(outputs["logits_C"], dim=-1).cpu().numpy())
            all_D.append(F.softmax(outputs["logits_D"], dim=-1).cpu().numpy())
    return {
        "prob_A": np.concatenate(all_A, axis=0) if all_A else np.empty((0, 4)),
        "pred_B": np.concatenate(all_B, axis=0) if all_B else np.empty((0,)),
        "prob_C": np.concatenate(all_C, axis=0) if all_C else np.empty((0, 3, 3)),
        "prob_D": np.concatenate(all_D, axis=0) if all_D else np.empty((0, 3)),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device} | smoke={V35_SMOKE} | head_weights={HEAD_WEIGHTS}")
    log.info(f"Data dir: {V35_DATA_DIR}")
    log.info(f"Output dir: {OUTPUT_DIR}")

    npz_files = sorted(V35_DATA_DIR.glob("*_v35.npz"))
    if not npz_files:
        log.error(f"No _v35.npz files in {V35_DATA_DIR}.")
        sys.exit(2)
    log.info(f"Found {len(npz_files)} v3.5 day files")

    if V35_SMOKE and len(npz_files) >= 2:
        npz_files = npz_files[:2]
        log.info(f"SMOKE: trimmed to {len(npz_files)} files: {[f.name for f in npz_files]}")

    folds = build_folds(npz_files, N_FOLDS, WF_WINDOW_DAYS, smoke=V35_SMOKE)
    log.info(f"Built {len(folds)} fold(s)")

    mlflow_run = None
    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(MLFLOW_EXPERIMENT)
            mlflow_run = mlflow.start_run(
                run_name=f"v35_gru_multihead_{time.strftime('%Y%m%d_%H%M')}"
            )
            mlflow.log_params({
                "window_size": WINDOW_SIZE, "stride": STRIDE,
                "batch_size": BATCH_SIZE, "lr": LR,
                "epochs_per_fold": EPOCHS_PER_FOLD,
                "n_folds": len(folds),
                "head_weights": HEAD_WEIGHTS_STR,
                "smoke": int(V35_SMOKE),
                "node": socket.gethostname(),
                "v35_data_dir": str(V35_DATA_DIR),
                "encoder": "CNN+GRU",
                "d_model": D_MODEL,
            })
        except Exception as e:
            log.warning(f"MLflow init failed: {e}")
            mlflow_run = None

    try:
        for fold_idx, train_idx, oot_idx in folds:
            train_files = [npz_files[i] for i in train_idx]
            oot_files = [npz_files[i] for i in oot_idx]
            log.info(f"FOLD {fold_idx}: train={len(train_files)} oot={len(oot_files)}")
            train_ds = MultiHeadDataset(train_files, WINDOW_SIZE, STRIDE)
            oot_ds = MultiHeadDataset(oot_files, WINDOW_SIZE, STRIDE)
            train_loader = DataLoader(
                train_ds, batch_size=BATCH_SIZE, shuffle=True,
                num_workers=0, collate_fn=_collate, pin_memory=(device.type == "cuda"),
            )
            oot_loader = DataLoader(
                oot_ds, batch_size=BATCH_SIZE, shuffle=False,
                num_workers=0, collate_fn=_collate, pin_memory=(device.type == "cuda"),
            )
            model = CNNGRUV35MultiHead().to(device)
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            log.info(f"  model params: {n_params:,}")

            total_steps = max(1, len(train_loader) * EPOCHS_PER_FOLD)
            result = train_one_fold(
                model, train_loader, oot_loader,
                fold_idx=fold_idx, output_dir=OUTPUT_DIR, device=device,
                total_train_steps=total_steps, mlflow_run=mlflow_run,
            )
            log.info(f"FOLD {fold_idx} DONE: best_val_loss={result['best_val_loss']:.4f}")

            oot_preds = run_oot_inference(model, oot_loader, device)
            pred_path = OUTPUT_DIR / f"fold_{fold_idx:02d}_oot_predictions.npz"
            np.savez(pred_path, **oot_preds)
            log.info(f"  OOT predictions saved -> {pred_path}")

    finally:
        if MLFLOW_AVAILABLE and mlflow_run is not None:
            try:
                mlflow.end_run()
            except Exception:
                pass
    log.info("DONE")


if __name__ == "__main__":
    main()
