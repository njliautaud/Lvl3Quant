#!/usr/bin/env python3
"""
train_cnn_mamba_v35_multihead.py — HC #497 v3.5 multi-head trainer (scaffold).

Reuses the CNN-Mamba backbone defined in
  /home/nick/Lvl3Quant/alpha_discovery/deep_models/train_cnn_mamba_v2.py
(via direct import — that file is NOT modified per HC #497 constraint).

Replaces the single multi-target regression head with four task heads:

  Head-A (pressure-direction):   1 logit per K_a in {5,10,30,60}s (BCE loss,
                                  skip events with NaN label)
  Head-B (pressure-persistence): scalar regression of persistence-seconds
                                  (Huber loss on UN-CENSORED examples only)
  Head-C (cum-K-tick first-pass): 3-class softmax per K in {2,4,8} (CE loss)
  Head-D (regime):               3-class softmax over (trend / MR / noise) (CE loss)

Multi-task loss:
  total = sum_h(w_h * loss_h)   where w_h defaults to uniform 1.0 (override
                                 via HEAD_WEIGHTS env: comma-separated
                                 A,B,C,D, e.g. "1.0,0.5,1.0,0.5").
  Monday-tuning task: replace with uncertainty-weighted MTL or grid search.

Input data:
  mbo_events_smart_v3_v35/*_v35.npz  (built by scripts/build_continuation_labels.py)

Env vars (subset; remaining shared with v2):
  EVENT_WINDOW_SIZE  (default 3000)
  EVENT_STRIDE       (default WINDOW_SIZE // 12)
  EVENT_BATCH_SIZE   (default 128)
  EVENT_LR           (default 3e-4)
  EVENT_EPOCHS       (default 5)
  EVENT_N_FOLDS      (default 5)
  WF_WINDOW_DAYS     (default 60, sliding)
  HEAD_WEIGHTS       (default "1,1,1,1")
  MLFLOW_EXPERIMENT  (default "cnn_mamba_v35_multihead_smoke")
  V35_DATA_DIR       (default /home/nick/Lvl3Quant/data/processed/mbo_events_smart_v3_v35)
  V35_SMOKE          (default 0 — set 1 for: 1 fold, 1 epoch, tiny window)

Smoke run example:
  V35_SMOKE=1 EVENT_N_FOLDS=1 EVENT_EPOCHS=1 EVENT_WINDOW_SIZE=256 \\
  EVENT_STRIDE=128 EVENT_BATCH_SIZE=16 \\
  python3 train_cnn_mamba_v35_multihead.py
"""
from __future__ import annotations

import os
import sys
import time
import logging
import socket
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ── Locate the v2 backbone module on this host ───────────────────────────────
V2_PATHS = [
    Path(os.environ["V2_BACKBONE_DIR"]) if os.environ.get("V2_BACKBONE_DIR") else None,
    Path("/home/nick/Lvl3Quant/alpha_discovery/deep_models"),
    Path("/home/jupiter/Lvl3Quant/alpha_discovery/deep_models"),
    Path(r"C:\Users\claude\Lvl3Quant"),
    Path(r"C:\Users\claude\Lvl3Quant\alpha_discovery\deep_models"),
    Path(r"C:\Users\claude\Lvl3Quant\live_trading"),
]
V2_PATHS = [p for p in V2_PATHS if p is not None]
for p in V2_PATHS:
    if (p / "train_cnn_mamba_v2.py").exists():
        sys.path.insert(0, str(p))
        break
else:
    raise SystemExit(
        "Could not find train_cnn_mamba_v2.py in any of: " + ", ".join(map(str, V2_PATHS))
    )

# v3.5 uses smart_v3 features (25 columns). Set this BEFORE importing the
# backbone so its module-level config picks up the correct N_TOTAL_FEATURES.
os.environ.setdefault("MAMBA_FEATURE_SET", "smart_v3")

# Defer the heavy imports to inside main() so smoke-help works without torch.
import train_cnn_mamba_v2 as v2backbone  # noqa: E402

# Re-use backbone, derived feature helper, scheduler
CNNMambaV2 = v2backbone.CNNMambaV2
WarmupCosineScheduler = v2backbone.WarmupCosineScheduler
MambaBlock = v2backbone.MambaBlock

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
log = logging.getLogger("train_v35")

# ── Config ───────────────────────────────────────────────────────────────────
V35_SMOKE = int(os.environ.get("V35_SMOKE", "0")) == 1
V35_DATA_DIR = Path(os.environ.get(
    "V35_DATA_DIR",
    "/home/nick/Lvl3Quant/data/processed/mbo_events_smart_v3_v35",
))
HEAD_WEIGHTS_STR = os.environ.get("HEAD_WEIGHTS", "1,1,1,1")
HEAD_WEIGHTS = [float(x) for x in HEAD_WEIGHTS_STR.split(",")]
assert len(HEAD_WEIGHTS) == 4, "HEAD_WEIGHTS must be 4 comma-separated floats"

WINDOW_SIZE = int(os.environ.get("EVENT_WINDOW_SIZE", "3000" if not V35_SMOKE else "256"))
STRIDE = int(os.environ.get("EVENT_STRIDE", str(WINDOW_SIZE // 12 if not V35_SMOKE else 128)))
BATCH_SIZE = int(os.environ.get("EVENT_BATCH_SIZE", "128" if not V35_SMOKE else "16"))
LR = float(os.environ.get("EVENT_LR", "3e-4"))
EPOCHS_PER_FOLD = int(os.environ.get("EVENT_EPOCHS", "5" if not V35_SMOKE else "1"))
N_FOLDS = int(os.environ.get("EVENT_N_FOLDS", "5" if not V35_SMOKE else "1"))
WF_WINDOW_DAYS = int(os.environ.get("WF_WINDOW_DAYS", "60"))
GRAD_CLIP = float(os.environ.get("EVENT_GRAD_CLIP", "1.0"))
WARMUP_STEPS = int(os.environ.get("EVENT_WARMUP", "300"))
HORIZONS_A_K = [5, 10, 30, 60]
HEAD_C_K = [2, 4, 8]

MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "cnn_mamba_v35_multihead_smoke")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")

OUTPUT_DIR = Path(os.environ.get(
    "V35_OUTPUT_DIR",
    str(Path.home() / "Lvl3Quant" / "output" / "cnn_mamba_v35_multihead"),
))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Dataset (v3.5: returns one sample with ALL head labels at last position) ──
class MultiHeadDataset(Dataset):
    """Loads a list of _v35.npz files and serves (events_window, label_dict).

    Each sample is a sliding window of length WINDOW_SIZE; the label for the
    sample is taken from the LAST position in the window (causal, same as v2).

    The label dict contains:
      A_5s, A_10s, A_30s, A_60s  (float, NaN-allowed)
      B_persistence_s            (float, NaN-allowed)
      B_censored                 (uint8 -> float)
      C_K2, C_K4, C_K8           (int -1/0/+1 -> {0,1,2})
      D_regime                   (int 0/1/2)
    """

    def __init__(
        self,
        npz_files: List[Path],
        window_size: int = WINDOW_SIZE,
        stride: int = STRIDE,
    ):
        self.window_size = window_size
        self.stride = stride
        self.npz_files = list(npz_files)
        # Build per-day index of (day_idx, start)
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
        events = torch.from_numpy(d["events"][start:end])  # (W, 25)
        last = end - 1
        labels = {
            "A_5s": float(d["A_5s"][last]),
            "A_10s": float(d["A_10s"][last]),
            "A_30s": float(d["A_30s"][last]),
            "A_60s": float(d["A_60s"][last]),
            "B_p": float(d["B_p"][last]),
            "B_c": int(d["B_c"][last]),
            "C_K2": int(d["C_K2"][last]) + 1,  # -1/0/+1 → 0/1/2
            "C_K4": int(d["C_K4"][last]) + 1,
            "C_K8": int(d["C_K8"][last]) + 1,
            "D": int(d["D"][last]),
        }
        return events, labels


def _collate(batch):
    """Collate (events, labels_dict) into batched tensors + label dict of tensors."""
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


# ── Multi-head model (wraps CNN-Mamba v2 backbone, returns embedding) ────────
class CNNMambaV35MultiHead(nn.Module):
    """CNN-Mamba v2 backbone -> 4 task heads:

        Head-A: 4-logit binary per K_a (5,10,30,60)
        Head-B: scalar (persistence seconds, regressed)
        Head-C: 3-class softmax per K (2,4,8)  -> 3*3=9 logits
        Head-D: 3-class softmax (regime)       -> 3 logits

    The backbone is reused (with n_targets=1 — we ignore its head output and
    extract the embedding via return_embedding=True).
    """

    def __init__(self):
        super().__init__()
        # Backbone produces (B, d_model) embedding when return_embedding=True
        # n_targets=1 is unused (we won't call its head).
        self.backbone = CNNMambaV2(n_targets=1)
        d = self.backbone.d_model
        # Head-A: 4 binary heads
        self.head_A = nn.Linear(d, len(HORIZONS_A_K))
        # Head-B: scalar
        self.head_B = nn.Linear(d, 1)
        # Head-C: 3 × 3-class softmax = 9 logits, reshape to (B, 3, 3)
        self.head_C = nn.Linear(d, len(HEAD_C_K) * 3)
        # Head-D: 3-class softmax
        self.head_D = nn.Linear(d, 3)

    def forward(self, events: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Backbone returns (preds, embedding) when return_embedding=True
        _, emb = self.backbone(events, return_embedding=True)  # (B, d_model)
        logits_A = self.head_A(emb)                            # (B, 4)
        pred_B = self.head_B(emb).squeeze(-1)                  # (B,)
        logits_C = self.head_C(emb).view(emb.shape[0], len(HEAD_C_K), 3)  # (B, 3, 3)
        logits_D = self.head_D(emb)                            # (B, 3)
        return {
            "logits_A": logits_A,
            "pred_B": pred_B,
            "logits_C": logits_C,
            "logits_D": logits_D,
        }


# ── Multi-task loss ──────────────────────────────────────────────────────────
def multi_head_loss(
    outputs: Dict[str, torch.Tensor],
    labels: Dict[str, torch.Tensor],
    weights: List[float] = HEAD_WEIGHTS,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Weighted sum of 4 head losses. Returns (total_loss, per_head_dict)."""
    w_A, w_B, w_C, w_D = weights
    losses = {}

    # Head-A: BCE per K, average over K, skip NaN labels
    A_targets = torch.stack(
        [labels["A_5s"], labels["A_10s"], labels["A_30s"], labels["A_60s"]],
        dim=1,
    )  # (B, 4)
    A_logits = outputs["logits_A"]  # (B, 4)
    A_valid = ~torch.isnan(A_targets)
    if A_valid.any():
        loss_A = F.binary_cross_entropy_with_logits(
            A_logits[A_valid], A_targets[A_valid], reduction="mean"
        )
    else:
        loss_A = torch.tensor(0.0, device=A_logits.device)
    losses["A"] = float(loss_A.detach().item())

    # Head-B: Huber on un-censored examples
    B_pred = outputs["pred_B"]              # (B,)
    B_target = labels["B_p"]                # (B,)
    B_cens = labels["B_c"].bool()
    B_valid = (~B_cens) & (~torch.isnan(B_target))
    if B_valid.any():
        loss_B = F.smooth_l1_loss(B_pred[B_valid], B_target[B_valid])
    else:
        loss_B = torch.tensor(0.0, device=B_pred.device)
    losses["B"] = float(loss_B.detach().item())

    # Head-C: CE per K, average
    C_logits = outputs["logits_C"]          # (B, 3, 3)  [K_idx, class]
    C_targets = torch.stack(
        [labels["C_K2"], labels["C_K4"], labels["C_K8"]], dim=1
    )  # (B, 3)
    # Reshape for CE: (B*3, 3) vs (B*3,)
    B_, K_, C_ = C_logits.shape
    loss_C = F.cross_entropy(
        C_logits.reshape(B_ * K_, C_),
        C_targets.reshape(B_ * K_),
    )
    losses["C"] = float(loss_C.detach().item())

    # Head-D: CE
    loss_D = F.cross_entropy(outputs["logits_D"], labels["D"])
    losses["D"] = float(loss_D.detach().item())

    total = w_A * loss_A + w_B * loss_B + w_C * loss_C + w_D * loss_D
    losses["total"] = float(total.detach().item())
    return total, losses


# ── Walk-forward (sliding) fold builder (re-used from v2 semantics) ──────────
def build_folds(
    npz_files: List[Path], n_folds: int, wf_window_days: int,
    smoke: bool = False,
) -> List[Tuple[int, List[int], List[int]]]:
    npz_files = sorted(npz_files)
    n = len(npz_files)
    if smoke:
        # Smoke: split available files 50/50 into train/oot
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
def train_one_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    fold_idx: int,
    output_dir: Path,
    device: torch.device,
    total_train_steps: int,
    mlflow_run=None,
) -> Dict:
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_steps=WARMUP_STEPS, total_steps=total_train_steps,
    )
    amp_ctx = (
        torch.amp.autocast("cuda") if device.type == "cuda"
        else torch.amp.autocast("cpu", enabled=False)
    )

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
            with amp_ctx:
                outputs = model(events)
                total_loss, per_head = multi_head_loss(outputs, labels)
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
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
                with amp_ctx:
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
            mlflow.log_metrics({
                f"fold{fold_idx}_train_loss": avg,
                f"fold{fold_idx}_val_loss": val_loss,
            }, step=epoch)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = output_dir / f"fold_{fold_idx:02d}_best.pt"
            torch.save({"model_state": model.state_dict(),
                        "val_loss": val_loss,
                        "fold": fold_idx,
                        "epoch": epoch}, ckpt_path)
            log.info(f"  Saved best ckpt → {ckpt_path}")
    return {"best_val_loss": best_val_loss}


# ── Run inference + dump per-event predictions (one OOT day) ─────────────────
def run_oot_inference(
    model: nn.Module, loader: DataLoader, device: torch.device,
) -> Dict[str, np.ndarray]:
    """Inference: return dict of head outputs (logits/probabilities)."""
    model.eval()
    all_A = []
    all_B = []
    all_C = []
    all_D = []
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


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device} | smoke={V35_SMOKE} | head_weights={HEAD_WEIGHTS}")
    log.info(f"Output dir: {OUTPUT_DIR}")

    npz_files = sorted(V35_DATA_DIR.glob("*_v35.npz"))
    if not npz_files:
        log.error(f"No _v35.npz files in {V35_DATA_DIR}.  "
                  "Run scripts/build_continuation_labels.py first.")
        sys.exit(2)
    log.info(f"Found {len(npz_files)} v3.5 day files")

    if V35_SMOKE and len(npz_files) >= 2:
        # Smoke: just 2 files (1 train, 1 oot)
        npz_files = npz_files[:2]
        log.info(f"SMOKE: trimmed to {len(npz_files)} files: {[f.name for f in npz_files]}")

    folds = build_folds(npz_files, N_FOLDS, WF_WINDOW_DAYS, smoke=V35_SMOKE)
    log.info(f"Built {len(folds)} fold(s)")

    # MLflow
    mlflow_run = None
    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(MLFLOW_EXPERIMENT)
            mlflow_run = mlflow.start_run(
                run_name=f"v35_multihead_{time.strftime('%Y%m%d_%H%M')}"
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
            model = CNNMambaV35MultiHead().to(device)
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            log.info(f"  model params: {n_params:,}")

            total_steps = max(1, len(train_loader) * EPOCHS_PER_FOLD)
            result = train_one_fold(
                model, train_loader, oot_loader,
                fold_idx=fold_idx, output_dir=OUTPUT_DIR, device=device,
                total_train_steps=total_steps, mlflow_run=mlflow_run,
            )
            log.info(f"FOLD {fold_idx} DONE: best_val_loss={result['best_val_loss']:.4f}")

            # Inference dump (smoke too — verifies inference path works)
            oot_preds = run_oot_inference(model, oot_loader, device)
            pred_path = OUTPUT_DIR / f"fold_{fold_idx:02d}_oot_predictions.npz"
            np.savez(pred_path, **oot_preds)
            log.info(f"  OOT predictions saved → {pred_path}")

    finally:
        if MLFLOW_AVAILABLE and mlflow_run is not None:
            mlflow.end_run()
    log.info("DONE")


if __name__ == "__main__":
    main()
