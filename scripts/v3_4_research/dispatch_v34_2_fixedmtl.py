"""
dispatch_v34_2_fixedmtl.py — v3.4.2 FIXED-WEIGHT MTL dispatch

Built on v3.4.1's gated-residual book-CNN architecture (CNNMambaV341BookResidual)
but REPLACING the Kendall uncertainty-weighted loss with a static fixed-weight MTL.

WHY: v3.4 / v3.4.1 failed 4x because Kendall log_sigma collapsed on 3/6 heads
(log_sigma 3.0-7.6), starving the IC_1s head of gradient → final IC_1s = 0.04-0.10
versus the v3.3 baseline 0.286 and gate >= 0.296.

THIS DISPATCHER:
  - Model: CNNMambaV341BookResidual (UNCHANGED — gated residual book CNN over v3.3 trunk).
  - Loss : FixedWeightMultiHeadLoss (NEW, inline) — Σ w_h * L_h, where:
        L_h = MSE for log_ret_* heads
        L_h = BCEWithLogits for p_up_*, p_reversal_*, _hit_tp heads
        L_h = Huber(delta=2.0) for pred_*_ticks / _net / time_to_mfe / realized_vol
        L_h = Pinball for quantile heads
    No log_sigma. No Kendall. No uncertainty term.

  - Static weights (per user 2026-05-15 spec; canonical head names below):
        log_ret_1s              : 1.0   (PRIMARY)
        log_ret_5s              : 0.7
        log_ret_10s             : 0.5
        log_ret_30s             : 0.3
        log_ret_60s             : 0.3
        log_ret_5min            : 0.2
        p_reversal_60s          : 0.2
        pred_mfe_30s_ticks      : 0.2
        pred_mae_30s_ticks      : 0.2
        all other heads         : 0.1

  - Training: custom train_one_fold_v342 mirroring v3.3's loop but WITHOUT the
    loss params in the optimizer (loss has no learnable params now).
  - Everything else (data, schedule, warmstart, MLflow logging) identical to v3.4.1.

Usage (Neptune):
  V32_BATCH_SIZE=16 V32_WF_TRAIN_DAYS=10 PYTHONPATH=/home/nick/Lvl3Quant \
    /home/nick/miniconda3/envs/py311-train/bin/python -u -X faulthandler \
    scripts/v3_4_research/dispatch_v34_2_fixedmtl.py --device cuda --n-folds 1
"""
import argparse
import json
import os as _os
_os.environ.setdefault("MAMBA_FEATURE_SET", "smart_v3")
_os.environ.setdefault("EVENT_WINDOW_SIZE", "1500")
_os.environ.setdefault("EVENT_STRIDE", "250")
_os.environ.setdefault("MLFLOW_TRACKING_URI", "http://jupiter:5000")

import logging
import os
import socket
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Path bootstrap
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# v3.2 (model + dataset + helpers)
from alpha_discovery.deep_models.train_cnn_mamba_v3_2 import (  # noqa: E402
    CNNMambaV32,
    SmartV32Dataset,
    ALL_HEAD_NAMES,
    QUANTILE_TARGETS,
    DIR_REG_HEADS,
    PinballLoss,
    build_weekly_fold_schedule,
    date_from_path,
    evaluate_v32,
    WINDOW_SIZE_T1, WINDOW_SIZE_T2, WINDOW_SIZE_T3,
    STRIDE, BATCH_SIZE, EPOCHS_PER_FOLD, WF_TRAIN_DAYS,
    TRUNK_DIM,
    N_T1_FEATURES, N_T2_FEATURES, N_T3_FEATURES,
    DEFAULT_DATA_DIR, DEFAULT_FIFO_LABEL_DIR, DEFAULT_ALPHA_LABEL_DIR,
    DEFAULT_PT_PRED_DIR, DEFAULT_TIER2_PARQUET_ROOT, DEFAULT_TIER3_PARQUET_ROOT,
)
# v3.4 (book dataset wrapper + collate + Book2DCNN)
from alpha_discovery.deep_models.train_cnn_mamba_v3_4 import (  # noqa: E402
    Book2DCNN,
    SmartV34DualTrunkDataset,
    collate_v34,
    DEFAULT_BOOK_FEATURES_DIR,
    N_BOOK_LEVELS,
    N_BOOK_FEATURES_PER_LEVEL,
)
# Scheduler from base trainer
from alpha_discovery.deep_models.train_cnn_mamba import (  # noqa: E402
    WarmupCosineScheduler,
)

try:
    import mlflow  # noqa: E402
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


logger = logging.getLogger("v342")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# Hyperparams (mirror v3.3 / base trainer)
LR = float(os.environ.get("EVENT_LR", 3e-4))
WARMUP_STEPS = int(os.environ.get("EVENT_WARMUP", 300))
GRAD_CLIP = float(os.environ.get("EVENT_GRAD_CLIP", 1.0))


PROJECT_ROOT = Path(__file__).resolve().parents[2]
V33_WARMSTART_DEFAULT = str(
    PROJECT_ROOT / "output" / "cnn_mamba_v3_3_uncertainty_weighted" / "fold_00_intra_ckpt.pt"
)
DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / "output" / "cnn_mamba_v3_4_2_fixedmtl")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://jupiter:5000")
MLFLOW_EXPERIMENT = "CNNMamba_v3_4_2_fixed_mtl"


# ============================================================
# FIXED-WEIGHT MTL HEAD WEIGHTS
# Canonical head names in ALL_HEAD_NAMES (no "pred_" prefix on log_ret heads):
#   log_ret_1s, log_ret_5s, log_ret_10s, log_ret_30s, log_ret_60s, log_ret_5min
#   p_up_*, p_reversal_*, log_ret_*_qNN, pred_mfe_*_ticks, pred_mae_*_ticks,
#   pred_time_to_mfe_secs, pred_realized_vol_30s_ticks,
#   fifo_tp4sl3_net, fifo_tp8sl5_net, *_hit_tp
# ============================================================
HEAD_WEIGHTS_V342: Dict[str, float] = {
    # ─── HC #454 PHASE 2 — STREAM-COHERENT TARGETS (primary, weight 1.0) ───
    # Per HC #453 R1: persistence + MFE-MAE + pressure ARE the new "what we want
    # the model to predict". Per HC #455 R2: every head we expect to learn must be
    # >=1.0 in this dict, otherwise the default-0.1 gradient floor collapses it
    # to a constant (the failure mode that wasted HC #455 R6a).
    "p_persistence_1s_10s":          1.0,  # HEAD A — binary persistence (BCE)
    "pred_mfe_minus_mae_10s_ticks":  1.0,  # HEAD B — realized (MFE - |MAE|) over 10s (Huber)
    "pred_pressure_score_30s":       1.0,  # HEAD D — cumulative pressure score 30s (Huber)

    # ─── PREVIOUSLY-DEFAULTED HEADS PROMOTED TO 1.0 (per HC #455 R2) ───
    # These were the "smooth" heads in HC #453 R6a — they're proven smoothness-friendly
    # at the label level but collapsed in v3.4.2 training because their loss weight was
    # the 0.1 default. Promote so they actually learn this retrain.
    "fifo_tp4sl3_net":     1.0,
    "fifo_tp8sl5_net":     1.0,
    "fifo_tp4sl3_hit_tp":  1.0,
    "fifo_tp8sl5_hit_tp":  1.0,
    "p_up_60s":            1.0,
    "pred_realized_vol_30s_ticks": 1.0,

    # ─── HC #453 R1 HEAD F — LEGACY SNAPSHOT (DEMOTED to 0.2 auxiliary) ───
    # The snapshot heads worked at the label level (concat IC 0.092) but produce flicker
    # at the prediction level (lag-1 autocorr 0.019). Keep gradient signal alive but
    # prevent the model from prioritizing snapshot over the new pressure heads.
    "log_ret_1s":    0.2,
    "log_ret_5s":    0.2,
    "log_ret_10s":   0.2,
    "log_ret_30s":   0.2,
    "log_ret_60s":   0.2,
    "log_ret_5min":  0.2,

    # ─── REVERSAL + PATH HEADS (kept at 0.2 per prior v3.4.2 spec) ───
    "p_reversal_60s":      0.2,
    "pred_mfe_30s_ticks":  0.2,
    "pred_mae_30s_ticks":  0.2,
}
DEFAULT_HEAD_WEIGHT_V342 = 0.1  # Quantiles, p_up_5s/10s/30s, time-to-MFE, etc. — auxiliary only.


def get_head_weight(name: str) -> float:
    return HEAD_WEIGHTS_V342.get(name, DEFAULT_HEAD_WEIGHT_V342)


# ============================================================
# FIXED-WEIGHT MULTI-HEAD LOSS
# No learnable parameters. Static weights from HEAD_WEIGHTS_V342.
# ============================================================
class FixedWeightMultiHeadLoss(nn.Module):
    """
    Total loss = Σ_h w_h * L_h(pred_h, target_h, mask_h)

    Where L_h is per-head appropriate (MSE / BCE / Huber / Pinball) — same families
    as v3.2/v3.3 — but the head weight w_h is STATIC (no Kendall log_sigma).
    """

    def __init__(
        self,
        head_names: Optional[List[str]] = None,
        head_weights: Optional[Dict[str, float]] = None,
        default_weight: float = DEFAULT_HEAD_WEIGHT_V342,
        huber_delta: float = 2.0,
    ):
        super().__init__()
        self.head_names: List[str] = list(head_names or ALL_HEAD_NAMES)
        self._weight_overrides = dict(head_weights or HEAD_WEIGHTS_V342)
        self._default_weight = float(default_weight)
        self.huber = nn.HuberLoss(reduction="none", delta=huber_delta)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.mse = nn.MSELoss(reduction="none")
        self.pinball = {
            name: PinballLoss(QUANTILE_TARGETS[name][1])
            for name in QUANTILE_TARGETS
        }

    def get_weight(self, name: str) -> float:
        return float(self._weight_overrides.get(name, self._default_weight))

    def weights_dict(self) -> Dict[str, float]:
        return {n: self.get_weight(n) for n in self.head_names}

    def _per_head_loss(
        self,
        name: str,
        pred: torch.Tensor,
        targets: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if name in QUANTILE_TARGETS:
            tgt_name, _q = QUANTILE_TARGETS[name]
            if tgt_name not in targets:
                return None
            t = targets[tgt_name]
            m = masks.get(tgt_name, torch.ones_like(pred))
            loss_per = self.pinball[name](pred, t)
        elif (name.endswith("_hit_tp") or name.startswith("p_up")
              or name.startswith("p_reversal")
              or name.startswith("p_persistence")):  # HC #454 Phase 2 — BCE on binary persistence
            if name not in targets:
                return None
            t = targets[name]
            m = masks.get(name, torch.ones_like(pred))
            loss_per = self.bce(pred, t)
        elif (name.startswith("pred_") or name.endswith("_net")
              or name == "pred_realized_vol_30s_ticks"
              or name.endswith("_ticks") or name == "pred_time_to_mfe_secs"):
            if name not in targets:
                return None
            t = targets[name]
            m = masks.get(name, torch.ones_like(pred))
            loss_per = self.huber(pred, t)
        elif name.startswith("log_ret"):
            if name not in targets:
                return None
            t = targets[name]
            m = masks.get(name, torch.ones_like(pred))
            loss_per = self.mse(pred, t)
        else:
            return None

        denom = m.sum().clamp_min(1.0)
        return (loss_per * m).sum() / denom

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        components: Dict[str, float] = {}
        total: Optional[torch.Tensor] = None

        for name in self.head_names:
            if name not in preds:
                continue
            base = self._per_head_loss(name, preds[name], targets, masks)
            if base is None:
                continue
            w = self.get_weight(name)
            contribution = w * base
            components[name] = float(base.detach().item())
            total = contribution if total is None else total + contribution

        if total is None:
            # No usable heads in this batch — return a zero tensor with grad
            device = next(iter(preds.values())).device if preds else torch.device("cpu")
            total = torch.zeros((), device=device, requires_grad=True)
        return total, components


# ============================================================
# v3.4.1 model: v3.3 trunk EXACTLY + gated residual book CNN
# (copied inline so this dispatcher is self-contained per user spec)
# ============================================================
class CNNMambaV341BookResidual(nn.Module):
    HEAD_NAMES = ALL_HEAD_NAMES

    def __init__(self, book_emb_dim: int = TRUNK_DIM, dropout: float = 0.1):
        super().__init__()
        self.v32_core = CNNMambaV32()
        self.book_cnn = Book2DCNN(
            in_features=N_BOOK_FEATURES_PER_LEVEL,
            in_levels=N_BOOK_LEVELS,
            out_dim=book_emb_dim,
            dropout=dropout,
        )
        # Scalar gate, tanh(0)=0 ⇒ book residual contributes ZERO at init
        self.book_gate = nn.Parameter(torch.zeros(1))

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        c = self.v32_core
        e1 = torch.nan_to_num(batch["events_t1"], nan=0.0, posinf=10.0, neginf=-10.0)
        x1 = c.t1_adapter(e1)
        _, emb1 = c.t1_backbone(x1, return_embedding=True)

        e2 = torch.nan_to_num(batch["events_t2"], nan=0.0, posinf=10.0, neginf=-10.0)
        x2 = c.t2_adapter(e2)
        _, emb2 = c.t2_backbone(x2, return_embedding=True)

        e3 = torch.nan_to_num(batch["events_t3"], nan=0.0, posinf=10.0, neginf=-10.0)
        x3 = c.t3_adapter(e3)
        _, emb3 = c.t3_backbone(x3, return_embedding=True)

        emb = torch.cat([emb1, emb2, emb3], dim=-1)
        trunk_out = c.trunk(emb)

        book_emb = self.book_cnn(batch["book_pyramid"])
        gate = torch.tanh(self.book_gate)
        trunk_out = trunk_out + gate * book_emb

        return {name: head(trunk_out).squeeze(-1) for name, head in c.heads.items()}

    def load_v33_warmstart(self, ckpt_path: str, device: torch.device) -> Dict[str, int]:
        stats = {"loaded": 0, "skipped": 0, "total_v33": 0}
        if not Path(ckpt_path).exists():
            print(f">>> v3.4.2 WARN: v3.3 ckpt not found at {ckpt_path} — fully random init",
                  flush=True)
            return stats
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        except Exception as e:
            print(f">>> v3.4.2 WARN: torch.load failed: {e} — random init", flush=True)
            return stats
        src = ckpt.get("model_state", ckpt)
        stats["total_v33"] = len(src)
        my_state = self.state_dict()
        for k, v in src.items():
            target = f"v32_core.{k}"
            if target in my_state and my_state[target].shape == v.shape:
                my_state[target].copy_(v.to(device))
                stats["loaded"] += 1
            else:
                stats["skipped"] += 1
        bp_params = sum(p.numel() for p in self.book_cnn.parameters()) + self.book_gate.numel()
        print(f">>> v3.4.2 warmstart: loaded={stats['loaded']}/{stats['total_v33']} v3.3 tensors, "
              f"skipped={stats['skipped']}, book_pathway_init_params={bp_params}", flush=True)
        return stats


# ============================================================
# Custom fold training loop — mirrors train_one_fold_v33 but uses fixed-weight loss.
# Differences vs v3.3:
#   * loss_fn = FixedWeightMultiHeadLoss(...) -- NO log_sigma
#   * Optimizer has ONLY model params (no loss params)
#   * No sigma logging (logs weights once at start instead)
# ============================================================
def train_one_fold_v342(
    model: nn.Module,
    train_loader: DataLoader,
    oot_loader: DataLoader,
    fold_idx: int,
    output_dir: Path,
    mlflow_run,
    device: torch.device,
    total_train_steps: int,
    use_amp: bool = True,
) -> Dict:
    loss_fn = FixedWeightMultiHeadLoss(
        head_names=ALL_HEAD_NAMES,
        head_weights=HEAD_WEIGHTS_V342,
        default_weight=DEFAULT_HEAD_WEIGHT_V342,
        huber_delta=2.0,
    ).to(device)

    weights_for_log = loss_fn.weights_dict()
    print(">>> v3.4.2 fixed head weights:", flush=True)
    for n in ALL_HEAD_NAMES:
        print(f"     {n:32s} w={weights_for_log[n]:.2f}", flush=True)

    optimizer = torch.optim.AdamW(
        [{"params": model.parameters(), "lr": LR, "weight_decay": 1e-4}]
    )
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_steps=WARMUP_STEPS, total_steps=total_train_steps
    )

    amp_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if use_amp and device.type == "cuda"
        else torch.amp.autocast("cpu", enabled=False)
    )

    best_val_loss = float("inf")
    global_step = 0
    total_batches = len(train_loader)

    print(
        f">>> v3.4.2 train_one_fold {fold_idx}: {EPOCHS_PER_FOLD} epochs x {total_batches} batches",
        flush=True,
    )

    if MLFLOW_AVAILABLE and mlflow_run is not None and fold_idx == 0:
        try:
            mlflow.log_params({f"hw_{k}": v for k, v in weights_for_log.items()})
        except Exception:
            pass

    for epoch in range(EPOCHS_PER_FOLD):
        model.train()
        loss_fn.train()
        epoch_loss = 0.0
        n_batches = 0
        epoch_start = time.time()
        comp_acc = defaultdict(float)

        for events, targets, masks in train_loader:
            events = {k: v.to(device, non_blocking=True) for k, v in events.items()}
            targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}
            masks = {k: v.to(device, non_blocking=True) for k, v in masks.items()}

            optimizer.zero_grad(set_to_none=True)
            with amp_ctx:
                preds = model(events)
                loss, components = loss_fn(preds, targets, masks)

            loss_finite = bool(torch.isfinite(loss).item())
            if loss_finite:
                loss.backward()
                for p in model.parameters():
                    if p.grad is not None:
                        torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
            else:
                if (n_batches % 100) == 0:
                    logger.warning(
                        f"  Fold {fold_idx} Ep {epoch+1} batch {n_batches}: "
                        f"non-finite loss — skipping step"
                    )
            scheduler.step()

            if loss_finite:
                epoch_loss += float(loss.item())
                for k, v in components.items():
                    comp_acc[k] += v
            n_batches += 1
            global_step += 1

            if n_batches % 100 == 0:
                elapsed = time.time() - epoch_start
                eta = elapsed / n_batches * (total_batches - n_batches)
                msg = (
                    f"  Fold {fold_idx} Ep {epoch+1} Batch {n_batches}/{total_batches} | "
                    f"Loss: {epoch_loss/n_batches:.4f} | Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s"
                )
                print(msg, flush=True)
                logger.info(msg)

            if n_batches % 500 == 0:
                ckpt = {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_step": getattr(scheduler, "_step_count", global_step),
                    "scheduler_state": (
                        scheduler.state_dict()
                        if hasattr(scheduler, "state_dict") else None
                    ),
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": (
                        torch.cuda.get_rng_state() if torch.cuda.is_available() else None
                    ),
                    "fold": fold_idx, "epoch": epoch, "batch": n_batches,
                    "global_step": global_step, "best_val_loss": best_val_loss,
                    "ckpt_version": 2,
                    "head_weights": weights_for_log,
                }
                torch.save(ckpt, output_dir / f"fold_{fold_idx:02d}_intra_ckpt.pt")

        avg_loss = epoch_loss / max(n_batches, 1)
        epoch_time = time.time() - epoch_start
        comp_avg = {k: v / max(n_batches, 1) for k, v in comp_acc.items()}

        # OOT eval (reuse v3.2 evaluator — it just calls loss_fn(preds, targets, masks))
        val_metrics, oot_preds, oot_targets, oot_masks = evaluate_v32(
            model, oot_loader, loss_fn, device, use_amp=use_amp
        )
        # HC #410 — persist OOT predictions for downstream analysis
        # (conf×vol filter, MFE-trigger sweeps, trade-economics matrix, fill-sim).
        # Saved per-epoch so we can compare ep1 vs ep2 vs … without re-running inference.
        try:
            npz_path = output_dir / f"fold_{fold_idx:02d}_ep{epoch+1}_oot.npz"
            save_dict = {"metrics_loss": val_metrics["loss"]}
            for k, v in oot_preds.items():
                save_dict[f"pred_{k}"] = v
            for k, v in oot_targets.items():
                save_dict[f"target_{k}"] = v
            for k, v in oot_masks.items():
                save_dict[f"mask_{k}"] = v
            for k, v in val_metrics.items():
                if k != "loss" and isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)):
                    save_dict[f"metric_{k}"] = float(v)
            np.savez_compressed(npz_path, **save_dict)
            logger.info(f"Saved OOT NPZ -> {npz_path.name} (heads={len(oot_preds)})")
            if MLFLOW_AVAILABLE and mlflow_run is not None:
                try:
                    mlflow.log_artifact(str(npz_path))
                except Exception as _e:
                    logger.warning(f"MLflow OOT NPZ artifact log failed: {_e}")
        except Exception as _save_e:
            logger.warning(f"OOT NPZ save failed: {_save_e}")

        msg = (
            f"Fold {fold_idx:02d} Ep {epoch+1}/{EPOCHS_PER_FOLD} | "
            f"TrLoss {avg_loss:.4f} | OOT Loss {val_metrics['loss']:.4f} | "
            f"IC 1s/5s/10s/30s = "
            f"{val_metrics.get('ic_log_ret_1s', float('nan')):.4f}/"
            f"{val_metrics.get('ic_log_ret_5s', float('nan')):.4f}/"
            f"{val_metrics.get('ic_log_ret_10s', float('nan')):.4f}/"
            f"{val_metrics.get('ic_log_ret_30s', float('nan')):.4f} | "
            f"LR {scheduler.get_lr():.2e} | T {epoch_time:.1f}s | "
            f"book_gate={torch.tanh(model.book_gate).item():.4f}"
        )
        print(msg, flush=True)
        logger.info(msg)

        if MLFLOW_AVAILABLE and mlflow_run is not None:
            try:
                step = fold_idx * EPOCHS_PER_FOLD + epoch
                metrics_to_log = {
                    f"f{fold_idx:02d}_train_loss": avg_loss,
                    f"f{fold_idx:02d}_oot_loss": val_metrics["loss"],
                    f"f{fold_idx:02d}_book_gate_tanh": float(torch.tanh(model.book_gate).item()),
                }
                for k, v in val_metrics.items():
                    if k != "loss" and not np.isnan(v):
                        metrics_to_log[f"f{fold_idx:02d}_{k}"] = float(v)
                for k, v in comp_avg.items():
                    metrics_to_log[f"f{fold_idx:02d}_train_loss_{k}"] = float(v)
                mlflow.log_metrics(metrics_to_log, step=step)
            except Exception as e:
                logger.warning(f"MLflow log failed: {e}")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save({
                "model_state": model.state_dict(),
                "fold": fold_idx, "epoch": epoch,
                "val_loss": val_metrics["loss"],
                "val_metrics": val_metrics,
                "head_weights": weights_for_log,
                "arch": {
                    "model": "CNNMambaV341BookResidual",
                    "trainer": "v3.4.2_fixed_weight_mtl",
                    "loss_kind": "fixed_weight_mtl",
                    "window_t1": WINDOW_SIZE_T1,
                    "window_t2": WINDOW_SIZE_T2,
                    "window_t3": WINDOW_SIZE_T3,
                    "n_t1_features": N_T1_FEATURES,
                    "n_t2_features": N_T2_FEATURES,
                    "n_t3_features": N_T3_FEATURES,
                    "head_names": ALL_HEAD_NAMES,
                },
            }, output_dir / f"fold_{fold_idx:02d}_best.pt")

    return {"best_val_loss": best_val_loss, "head_weights": weights_for_log}


# ============================================================
# Date alignment helper (same as v3.4.1)
# ============================================================
def discover_aligned_dates(data_dir: Path, book_dir: Path):
    event_npz = sorted(data_dir.glob("*_mbo_events.npz"))
    event_dates = set()
    for p in event_npz:
        d = date_from_path(p)
        if d:
            event_dates.add(d)
    book_dates = set()
    for p in book_dir.glob("*_book_features.npz"):
        stem = p.stem.split("_")[0]
        if len(stem) == 8 and stem.isdigit():
            book_dates.add(stem)
    return sorted(event_dates & book_dates)


def main():
    p = argparse.ArgumentParser(description="v3.4.2 FIXED-WEIGHT MTL dispatch")
    p.add_argument("--n-folds", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--warmstart-ckpt", default=V33_WARMSTART_DEFAULT)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--book-features-dir", default=DEFAULT_BOOK_FEATURES_DIR)
    p.add_argument("--fifo-label-dir", default=str(DEFAULT_FIFO_LABEL_DIR))
    p.add_argument("--alpha-label-dir", default=str(DEFAULT_ALPHA_LABEL_DIR))
    p.add_argument("--pt-pred-dir", default=str(DEFAULT_PT_PRED_DIR))
    p.add_argument("--tier2-parquet-root", default=str(DEFAULT_TIER2_PARQUET_ROOT))
    p.add_argument("--tier3-parquet-root", default=str(DEFAULT_TIER3_PARQUET_ROOT))
    p.add_argument("--no-mlflow", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        print(">>> v3.4.2: CUDA unavailable — CPU fallback", flush=True)
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    use_amp = (device.type == "cuda") and not args.no_amp

    print(f">>> v3.4.2 device={device} amp={use_amp} output_dir={output_dir}", flush=True)
    print(f">>> v3.4.2 warmstart={args.warmstart_ckpt}", flush=True)

    # Smoke test
    if args.smoke_test:
        print(">>> v3.4.2 SMOKE TEST", flush=True)
        m = CNNMambaV341BookResidual().to(device)
        m.load_v33_warmstart(args.warmstart_ckpt, device)
        loss_fn = FixedWeightMultiHeadLoss().to(device)
        n_params = sum(p.numel() for p in m.parameters())
        loss_params = sum(p.numel() for p in loss_fn.parameters())
        print(f">>> v3.4.2 model params: {n_params/1e6:.3f}M  loss_params={loss_params} (must be 0)",
              flush=True)
        assert loss_params == 0, "Fixed-weight loss must have NO learnable params"
        return 0

    # Discover aligned event+book dates
    data_dir = Path(args.data_dir)
    book_dir = Path(args.book_features_dir)
    aligned = discover_aligned_dates(data_dir, book_dir)
    if not aligned:
        print(">>> v3.4.2 ERROR: no aligned dates", flush=True)
        return 1
    print(f">>> v3.4.2 aligned dates: {len(aligned)} ({aligned[0]} → {aligned[-1]})", flush=True)

    train_days_env = int(os.environ.get("V32_WF_TRAIN_DAYS", WF_TRAIN_DAYS))
    folds = build_weekly_fold_schedule(aligned, n_folds=args.n_folds, train_days=train_days_env)
    print(f">>> v3.4.2 built {len(folds)} fold(s) | train_days={train_days_env}", flush=True)
    for f in folds:
        print(f"  Fold {f['fold']}: train {f['train_start']}→{f['train_end']} "
              f"({len(f['train_dates'])}d) | OOT {f['oot_start']}→{f['oot_end']} "
              f"({len(f['oot_dates'])}d)", flush=True)
    with open(output_dir / "fold_schedule.json", "w") as fh:
        json.dump(folds, fh, indent=2, default=str)

    # MLflow
    mlflow_run = None
    if MLFLOW_AVAILABLE and not args.no_mlflow:
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(MLFLOW_EXPERIMENT)
            gpu_name = (torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu")
            mlflow_run = mlflow.start_run(
                run_name=f"v3.4.2_fixedmtl_{time.strftime('%Y%m%d_%H%M')}_{socket.gethostname()}",
                tags={
                    "model_family": "cnn_mamba_v3.4.2_fixed_mtl",
                    "version": "3.4.2",
                    "arch": "v3.3+gated_book_residual",
                    "loss_kind": "fixed_weight_mtl",
                    "init_strategy": "v33_full_warmstart_gate_zero",
                },
            )
            mlflow.log_params({
                "model": "CNNMambaV341BookResidual",
                "trainer": "train_one_fold_v342_fixed_mtl",
                "book_gate_init": 0.0,
                "book_emb_dim": TRUNK_DIM,
                "batch_size": int(os.environ.get("V32_BATCH_SIZE", BATCH_SIZE)),
                "epochs_per_fold": EPOCHS_PER_FOLD,
                "warmstart_ckpt": args.warmstart_ckpt,
                "wf_train_days": train_days_env,
                "n_folds_planned": len(folds),
                "node": socket.gethostname(),
                "gpu": gpu_name,
                "mixed_precision": "bf16" if use_amp else "none",
                "n_heads": len(ALL_HEAD_NAMES),
                "default_head_weight": DEFAULT_HEAD_WEIGHT_V342,
                "loss_has_learnable_params": False,
            })
            print(f">>> v3.4.2 MLflow run: {mlflow_run.info.run_id}", flush=True)
        except Exception as e:
            print(f">>> v3.4.2 MLflow init failed: {e}", flush=True)
            mlflow_run = None

    bs = int(os.environ.get("V32_BATCH_SIZE", BATCH_SIZE))
    num_workers = int(os.environ.get("V32_NUM_WORKERS", "0"))

    try:
        for f_info in folds:
            fold_idx = f_info["fold"]
            print("=" * 60, flush=True)
            print(f"FOLD {fold_idx} | train {f_info['train_start']}→{f_info['train_end']} "
                  f"| OOT {f_info['oot_start']}→{f_info['oot_end']}", flush=True)
            print("=" * 60, flush=True)

            train_inner = SmartV32Dataset(
                data_dir=data_dir,
                fifo_label_dir=Path(args.fifo_label_dir),
                alpha_label_dir=Path(args.alpha_label_dir),
                pt_pred_dir=Path(args.pt_pred_dir),
                tier2_parquet_root=Path(args.tier2_parquet_root),
                tier3_parquet_root=Path(args.tier3_parquet_root),
                dates=f_info["train_dates"],
                window_t1=WINDOW_SIZE_T1, window_t2=WINDOW_SIZE_T2, window_t3=WINDOW_SIZE_T3,
                stride=STRIDE, feature_stats=None,
                cache_size=1, require_alpha_labels=False,
            )
            feature_stats = train_inner.get_feature_stats()
            np.savez(
                output_dir / f"fold_{fold_idx:02d}_feature_stats.npz",
                mean_t1=feature_stats["mean_t1"], std_t1=feature_stats["std_t1"],
            )
            oot_inner = SmartV32Dataset(
                data_dir=data_dir,
                fifo_label_dir=Path(args.fifo_label_dir),
                alpha_label_dir=Path(args.alpha_label_dir),
                pt_pred_dir=Path(args.pt_pred_dir),
                tier2_parquet_root=Path(args.tier2_parquet_root),
                tier3_parquet_root=Path(args.tier3_parquet_root),
                dates=f_info["oot_dates"],
                window_t1=WINDOW_SIZE_T1, window_t2=WINDOW_SIZE_T2, window_t3=WINDOW_SIZE_T3,
                stride=STRIDE, feature_stats=feature_stats,
                cache_size=1, require_alpha_labels=False,
            )
            train_ds = SmartV34DualTrunkDataset(train_inner, book_features_dir=str(book_dir))
            oot_ds = SmartV34DualTrunkDataset(oot_inner, book_features_dir=str(book_dir))

            train_loader = DataLoader(
                train_ds, batch_size=bs, shuffle=False, num_workers=num_workers,
                pin_memory=True, drop_last=True, collate_fn=collate_v34,
                persistent_workers=False,
                prefetch_factor=(2 if num_workers > 0 else None),
            )
            oot_loader = DataLoader(
                oot_ds, batch_size=bs * 2, shuffle=False, num_workers=num_workers,
                pin_memory=True, collate_fn=collate_v34,
                persistent_workers=False,
                prefetch_factor=(2 if num_workers > 0 else None),
            )

            model = CNNMambaV341BookResidual().to(device)
            n_params = sum(p.numel() for p in model.parameters())
            print(f">>> v3.4.2 Model parameters: {n_params:,} "
                  f"(v3.3 ≈ 1.63M; delta = book CNN + 1 gate)", flush=True)
            if MLFLOW_AVAILABLE and mlflow_run is not None and fold_idx == 0:
                try:
                    mlflow.log_params({"model_params": n_params})
                except Exception:
                    pass

            ws_stats = model.load_v33_warmstart(args.warmstart_ckpt, device)
            if MLFLOW_AVAILABLE and mlflow_run is not None and fold_idx == 0:
                try:
                    mlflow.log_params({
                        "warmstart_loaded_tensors": ws_stats["loaded"],
                        "warmstart_skipped": ws_stats["skipped"],
                        "warmstart_total_v33": ws_stats["total_v33"],
                    })
                except Exception:
                    pass

            total_steps = EPOCHS_PER_FOLD * len(train_loader)
            print(f">>> v3.4.2 fold {fold_idx} batches: train={len(train_loader)} "
                  f"oot={len(oot_loader)} total_steps={total_steps} bs={bs} num_workers={num_workers}",
                  flush=True)

            result = train_one_fold_v342(
                model, train_loader, oot_loader,
                fold_idx, output_dir, mlflow_run, device,
                total_train_steps=total_steps, use_amp=use_amp,
            )
            print(f">>> v3.4.2 fold {fold_idx} done | "
                  f"best_val_loss={result.get('best_val_loss'):.4f} | "
                  f"book_gate={torch.tanh(model.book_gate).item():.4f}", flush=True)
            if MLFLOW_AVAILABLE and mlflow_run is not None:
                try:
                    mlflow.log_metric(f"f{fold_idx:02d}_final_book_gate_tanh",
                                      float(torch.tanh(model.book_gate).item()))
                except Exception:
                    pass

    finally:
        if MLFLOW_AVAILABLE and mlflow_run is not None:
            try:
                mlflow.end_run()
            except Exception:
                pass

    print(">>> v3.4.2 DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
