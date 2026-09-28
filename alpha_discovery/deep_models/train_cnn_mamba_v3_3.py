"""
CNN-Mamba v3.3 — UNCERTAINTY-WEIGHTED Multi-Task Trainer
(Kendall, Gal & Cipolla, 2018: "Multi-Task Learning Using Uncertainty to Weigh Losses")

Motivation
----------
v3.2 used hand-tuned static lambdas (LOSS_LAMBDA dict) to balance the 28 heads.
Empirically the heads with the smallest residual variance (typically the very-short-
horizon log_ret_1s/5s) dominated the gradient and the longer-horizon heads
(log_ret_30s/60s/5min, MFE/MAE) silently regressed compared to v3.

v3.3 replaces the static lambdas with learnable log-variance parameters (log_sigma_i)
per head, optimised jointly with the model. Each head's contribution to the loss is

    L_total += (1 / (2 * exp(2 * log_sigma_i))) * L_i + log_sigma_i

so the optimiser is free to *down-weight* noisy heads by raising their log_sigma,
trading a bit of regularisation against a large precision factor on noisier targets.
The training loop, data pipeline, walk-forward schedule, intra-ckpt resume logic,
and v3 warmstart are all reused unchanged via import from v3.2.

The ONLY behavioural change is the loss class. Everything else is identical to v3.2
so OOT IC numbers are directly comparable head-for-head.

Author: Claude (head-of-quant), 2026-05-13.
"""

# Must set BEFORE importing v3.2 trainer (which reads these on import)
import os as _os
_os.environ.setdefault("MAMBA_FEATURE_SET", "smart_v3")
_os.environ.setdefault("EVENT_WINDOW_SIZE", "1500")
_os.environ.setdefault("EVENT_STRIDE", "250")
_os.environ.setdefault("MLFLOW_TRACKING_URI", "http://jupiter:5000")

import os
import gc
import time
import json
import logging
import argparse
import socket
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ------------------------------------------------------------------
# Reuse EVERYTHING from v3.2 except the loss class
# ------------------------------------------------------------------
from alpha_discovery.deep_models.train_cnn_mamba_v3_2 import (
    # Model + data
    CNNMambaV32,
    SmartV32Dataset,
    collate_v32,
    # Head taxonomy
    ALL_HEAD_NAMES,
    DIR_REG_HEADS,
    P_UP_HEADS,
    QUANTILE_HEADS,
    PATH_HEADS,
    TIME_HEADS,
    REVERSAL_HEADS,
    VOL_HEADS,
    LEGACY_AUX_HEADS,
    QUANTILE_TARGETS,
    LOSS_LAMBDA,
    PinballLoss,
    # Eval + metrics
    evaluate_v32,
    compute_ic,
    # Walk-forward + paths
    build_weekly_fold_schedule,
    date_from_path,
    DEFAULT_DATA_DIR,
    DEFAULT_FIFO_LABEL_DIR,
    DEFAULT_ALPHA_LABEL_DIR,
    DEFAULT_PT_PRED_DIR,
    DEFAULT_TIER2_PARQUET_ROOT,
    DEFAULT_TIER3_PARQUET_ROOT,
    V3_WARMSTART_CKPT,
    FIRST_OOT_MONDAY,
    N_FOLDS,
    WF_TRAIN_DAYS,
    EPOCHS_PER_FOLD,
    BATCH_SIZE,
    STRIDE,
    WINDOW_SIZE_T1,
    WINDOW_SIZE_T2,
    WINDOW_SIZE_T3,
    N_T1_FEATURES,
    N_T2_FEATURES,
    N_T3_FEATURES,
)

# v2 utilities (LR schedule, param counter, hyperparams)
from alpha_discovery.deep_models.train_cnn_mamba import (
    WarmupCosineScheduler,
    count_parameters,
    LR,
    WARMUP_STEPS,
    GRAD_CLIP,
)

try:
    import mlflow
    MLFLOW_AVAILABLE = True
    if os.environ.get("DISABLE_MLFLOW", "0") == "1":
        MLFLOW_AVAILABLE = False
except ImportError:
    MLFLOW_AVAILABLE = False


# ============================================================
# Logging
# ============================================================
LOG_DIR = Path(__file__).parent / "results"
LOG_DIR.mkdir(exist_ok=True)
log_path = LOG_DIR / "cnn_mamba_v3_3.log"
_v33_handler = logging.FileHandler(log_path)
_v33_handler.setLevel(logging.INFO)
_v33_handler.setFormatter(logging.Formatter("%(asctime)s [v3.3 %(levelname)s] %(message)s"))
logging.root.addHandler(_v33_handler)
logger = logging.getLogger("cnn_mamba_v3_3")
logger.setLevel(logging.INFO)
print(">>> train_cnn_mamba_v3_3.py loaded", flush=True)


# ============================================================
# v3.3-specific config
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / "output" / "cnn_mamba_v3_3_uncertainty_weighted")

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://jupiter:5000")
MLFLOW_EXPERIMENT = "CNNMamba_v3_3_uncertainty_weighted"

# Initial log_sigma per head. 0.0 => sigma=1.0 => precision (1/2σ²)=0.5,
# which is in the same ballpark as v3.2's static lambdas (0.1 .. 1.0).
LOG_SIGMA_INIT = float(os.environ.get("V33_LOG_SIGMA_INIT", 0.0))


# ============================================================
# Uncertainty-Weighted Joint Multi-Head Loss
# ============================================================
class JointMultiHeadLossV33_UncertaintyWeighted(nn.Module):
    """
    Per-head learnable homoscedastic uncertainty weighting (Kendall et al. 2018).

    For each head i we keep a learnable scalar log_sigma_i. The head's
    contribution to the total loss is

        (1 / (2 * exp(2 * log_sigma_i))) * L_i + log_sigma_i

    Where L_i is:
      * MSE for log_ret_* regression heads
      * Huber for tick-valued regression heads (MFE/MAE/vol/time/legacy net)
      * BCE for binary heads (p_up_*, p_reversal_*, _hit_tp)
      * Pinball for quantile heads
    (Same per-head loss families as JointMultiHeadLossV32, only the head weighting changes.)

    log_sigma_i is a torch.nn.Parameter and is registered for optimisation.
    """

    def __init__(
        self,
        head_names: List[str] = None,
        huber_delta: float = 2.0,
        log_sigma_init: float = LOG_SIGMA_INIT,
        log_sigma_min: float = -3.0,   # ~sigma 0.05  -> precision ~ 100
        log_sigma_max: float = 3.0,    # ~sigma 20    -> precision ~ 0.00125
    ):
        super().__init__()
        self.head_names: List[str] = list(head_names or ALL_HEAD_NAMES)
        self.huber = nn.HuberLoss(reduction="none", delta=huber_delta)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.mse = nn.MSELoss(reduction="none")
        self.pinball = {
            name: PinballLoss(QUANTILE_TARGETS[name][1])
            for name in QUANTILE_TARGETS
        }
        self.log_sigma_min = log_sigma_min
        self.log_sigma_max = log_sigma_max

        # One learnable log_sigma per head. ParameterDict keys can't contain ".".
        # All current ALL_HEAD_NAMES entries are dot-free, but sanitize defensively.
        self.log_sigma = nn.ParameterDict({
            self._safe_key(name): nn.Parameter(torch.tensor(float(log_sigma_init)))
            for name in self.head_names
        })

    @staticmethod
    def _safe_key(name: str) -> str:
        return name.replace(".", "_")

    def _per_head_loss(
        self,
        name: str,
        pred: torch.Tensor,
        targets: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """
        Returns a scalar tensor = masked mean loss for this head, or None if
        the head has no usable target in this batch.
        """
        if name in QUANTILE_TARGETS:
            tgt_name, _q = QUANTILE_TARGETS[name]
            if tgt_name not in targets:
                return None
            t = targets[tgt_name]
            m = masks.get(tgt_name, torch.ones_like(pred))
            loss_per = self.pinball[name](pred, t)
        elif (name.endswith("_hit_tp") or name.startswith("p_up")
              or name.startswith("p_reversal")):
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

            key = self._safe_key(name)
            ls = self.log_sigma[key].clamp(self.log_sigma_min, self.log_sigma_max)
            precision = torch.exp(-2.0 * ls)
            head_loss = 0.5 * precision * base + ls

            components[name] = float(base.detach().item())
            components[f"sigma_{name}"] = float(torch.exp(ls).detach().item())

            total = head_loss if total is None else total + head_loss

        if total is None:
            # Defensive: shouldn't happen with non-empty preds, but keep
            # the graph alive via a zero tensor that depends on log_sigmas
            # so the optimiser step is well-defined.
            total = sum(p.sum() * 0.0 for p in self.log_sigma.values())

        return total, components

    def get_sigma_dict(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for name in self.head_names:
            ls = self.log_sigma[self._safe_key(name)].detach()
            out[name] = float(torch.exp(ls).item())
        return out


# ============================================================
# Train one fold (v3.3) — same control flow as v3.2, new loss
# ============================================================
def train_one_fold_v33(
    model: CNNMambaV32,
    train_loader: DataLoader,
    oot_loader: DataLoader,
    fold_idx: int,
    output_dir: Path,
    mlflow_run,
    device: torch.device,
    total_train_steps: int,
    use_amp: bool = True,
    resume_state: Optional[Dict] = None,
) -> Dict:
    # Loss owns learnable log_sigmas — include in optimiser param groups.
    loss_fn = JointMultiHeadLossV33_UncertaintyWeighted(head_names=ALL_HEAD_NAMES).to(device)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": LR, "weight_decay": 1e-4},
            # log_sigmas: NO weight decay (Kendall et al.), slightly higher LR ok
            {"params": list(loss_fn.parameters()), "lr": LR, "weight_decay": 0.0},
        ]
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

    resume_epoch = 0
    resume_batch = 0
    if resume_state is not None and resume_state.get("ckpt_version", 1) >= 2:
        try:
            optimizer.load_state_dict(resume_state["optimizer_state"])
            if resume_state.get("scheduler_state") is not None and hasattr(scheduler, "load_state_dict"):
                scheduler.load_state_dict(resume_state["scheduler_state"])
            else:
                for _ in range(int(resume_state.get("scheduler_step", 0))):
                    scheduler.step()
            if resume_state.get("torch_rng_state") is not None:
                _trs = resume_state["torch_rng_state"]
                if hasattr(_trs, "cpu"):
                    _trs = _trs.cpu()
                if hasattr(_trs, "to"):
                    _trs = _trs.to(torch.uint8)
                torch.set_rng_state(_trs)
            if resume_state.get("cuda_rng_state") is not None and torch.cuda.is_available():
                _crs = resume_state["cuda_rng_state"]
                if hasattr(_crs, "cpu"):
                    _crs = _crs.cpu()
                if hasattr(_crs, "to"):
                    _crs = _crs.to(torch.uint8)
                torch.cuda.set_rng_state(_crs)
            if "loss_state" in resume_state and resume_state["loss_state"] is not None:
                loss_fn.load_state_dict(resume_state["loss_state"], strict=False)
            resume_epoch = int(resume_state.get("epoch", 0))
            resume_batch = int(resume_state.get("batch", 0))
            global_step = int(resume_state.get("global_step", 0))
            best_val_loss = float(resume_state.get("best_val_loss", float("inf")))
            logger.info(
                f"  Fold {fold_idx} RESUMED v3.3: ep={resume_epoch} batch={resume_batch} "
                f"global_step={global_step}"
            )
            print(
                f">>> v3.3 RESUME fold {fold_idx} from Ep {resume_epoch+1} batch {resume_batch}",
                flush=True,
            )
        except Exception as e:
            logger.warning(f"v3.3 resume failed ({e}); starting fold from scratch")
            resume_epoch = 0
            resume_batch = 0
            global_step = 0
            best_val_loss = float("inf")

    print(
        f">>> v3.3 train_one_fold {fold_idx}: {EPOCHS_PER_FOLD} epochs x {total_batches} batches",
        flush=True,
    )

    for epoch in range(EPOCHS_PER_FOLD):
        if epoch < resume_epoch:
            continue
        model.train()
        loss_fn.train()
        epoch_loss = 0.0
        n_batches = 0
        epoch_start = time.time()
        comp_acc = defaultdict(float)

        skip_until = resume_batch if epoch == resume_epoch else 0
        if skip_until > 0:
            logger.info(
                f"  Fold {fold_idx} Ep {epoch+1}: fast-forwarding loader to batch {skip_until}"
            )

        for events, targets, masks in train_loader:
            if n_batches < skip_until:
                n_batches += 1
                continue
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
                # sanitize NaN grads on both model + loss params
                for p in list(model.parameters()) + list(loss_fn.parameters()):
                    if p.grad is not None:
                        torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(loss_fn.parameters()),
                    GRAD_CLIP,
                )
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
                    "loss_state": loss_fn.state_dict(),
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
                }
                torch.save(ckpt, output_dir / f"fold_{fold_idx:02d}_intra_ckpt.pt")

        avg_loss = epoch_loss / max(n_batches, 1)
        epoch_time = time.time() - epoch_start
        comp_avg = {k: v / max(n_batches, 1) for k, v in comp_acc.items()}

        # OOT eval — reuses v3.2 evaluator (it just calls loss_fn(preds, targets, masks))
        val_metrics, _, _, _ = evaluate_v32(model, oot_loader, loss_fn, device, use_amp=use_amp)
        sigmas = loss_fn.get_sigma_dict()
        # Top-3 highest and lowest sigmas for human-readable logging
        sigma_items = sorted(sigmas.items(), key=lambda kv: kv[1])
        hi = sigma_items[-3:]
        lo = sigma_items[:3]
        msg = (
            f"Fold {fold_idx:02d} Ep {epoch+1}/{EPOCHS_PER_FOLD} | "
            f"TrLoss {avg_loss:.4f} | OOT Loss {val_metrics['loss']:.4f} | "
            f"IC 1s/5s/10s/30s = "
            f"{val_metrics.get('ic_log_ret_1s', float('nan')):.4f}/"
            f"{val_metrics.get('ic_log_ret_5s', float('nan')):.4f}/"
            f"{val_metrics.get('ic_log_ret_10s', float('nan')):.4f}/"
            f"{val_metrics.get('ic_log_ret_30s', float('nan')):.4f} | "
            f"LR {scheduler.get_lr():.2e} | T {epoch_time:.1f}s | "
            f"σ lo {lo} σ hi {hi}"
        )
        print(msg, flush=True)
        logger.info(msg)

        if MLFLOW_AVAILABLE and mlflow_run is not None:
            try:
                step = fold_idx * EPOCHS_PER_FOLD + epoch
                metrics_to_log = {
                    f"f{fold_idx:02d}_train_loss": avg_loss,
                    f"f{fold_idx:02d}_oot_loss": val_metrics["loss"],
                }
                for k, v in val_metrics.items():
                    if k != "loss" and not np.isnan(v):
                        metrics_to_log[f"f{fold_idx:02d}_{k}"] = float(v)
                for k, v in comp_avg.items():
                    metrics_to_log[f"f{fold_idx:02d}_train_loss_{k}"] = float(v)
                for h_name, sig in sigmas.items():
                    metrics_to_log[f"f{fold_idx:02d}_sigma_{h_name}"] = float(sig)
                mlflow.log_metrics(metrics_to_log, step=step)
            except Exception as e:
                logger.warning(f"MLflow log failed: {e}")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save({
                "model_state": model.state_dict(),
                "loss_state": loss_fn.state_dict(),
                "fold": fold_idx, "epoch": epoch,
                "val_loss": val_metrics["loss"],
                "val_metrics": val_metrics,
                "sigmas": sigmas,
                "arch": {
                    "model": "CNNMambaV32",  # arch unchanged; only loss differs
                    "trainer": "v3.3_uncertainty_weighted",
                    "window_t1": WINDOW_SIZE_T1,
                    "window_t2": WINDOW_SIZE_T2,
                    "window_t3": WINDOW_SIZE_T3,
                    "n_t1_features": N_T1_FEATURES,
                    "n_t2_features": N_T2_FEATURES,
                    "n_t3_features": N_T3_FEATURES,
                    "head_names": ALL_HEAD_NAMES,
                    "log_sigma_init": LOG_SIGMA_INIT,
                },
            }, output_dir / f"fold_{fold_idx:02d}_best.pt")

    return {"best_val_loss": best_val_loss, "final_sigmas": loss_fn.get_sigma_dict()}


# ============================================================
# Walk-forward driver (v3.3)
# ============================================================
def run_weekly_wf_v33(
    data_dir: Path,
    fifo_label_dir: Path,
    alpha_label_dir: Path,
    pt_pred_dir: Path,
    output_dir: Path,
    device: torch.device,
    n_folds: int = N_FOLDS,
    train_days: int = WF_TRAIN_DAYS,
    warmstart_ckpt: str = V3_WARMSTART_CKPT,
    tier2_parquet_root: Optional[Path] = None,
    tier3_parquet_root: Optional[Path] = None,
    resume_from_intra_ckpt: Optional[Path] = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(data_dir.glob("*_mbo_events.npz"))
    available_dates = sorted([date_from_path(p) for p in npz_files if date_from_path(p)])
    if not available_dates:
        logger.error(f"No MBO events found in {data_dir}")
        return {}
    logger.info(f"Found {len(available_dates)} dates: {available_dates[0]} → {available_dates[-1]}")

    folds = build_weekly_fold_schedule(available_dates, n_folds=n_folds, train_days=train_days)
    logger.info(f"Built {len(folds)} weekly folds (anchor={FIRST_OOT_MONDAY})")
    for f in folds:
        logger.info(
            f"  Fold {f['fold']}: train {f['train_start']}→{f['train_end']} ({len(f['train_dates'])}d) "
            f"OOT {f['oot_start']}→{f['oot_end']} ({len(f['oot_dates'])}d)"
        )
    with open(output_dir / "fold_schedule.json", "w") as fh:
        json.dump(folds, fh, indent=2, default=str)

    use_amp = device.type == "cuda"

    mlflow_run = None
    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(MLFLOW_EXPERIMENT)
            mlflow_run = mlflow.start_run(
                run_name=f"v3.3_{time.strftime('%Y%m%d_%H%M')}_{socket.gethostname()}",
                tags={"model_family": "cnn_mamba_v3.3", "version": "3.3",
                      "loss_kind": "uncertainty_weighted_kendall2018"},
            )
            gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
            mlflow.log_params({
                "model": "CNNMambaV32",
                "trainer": "v3.3_uncertainty_weighted",
                "window_t1": WINDOW_SIZE_T1,
                "window_t2": WINDOW_SIZE_T2,
                "window_t3": WINDOW_SIZE_T3,
                "stride": STRIDE,
                "n_t1_features": N_T1_FEATURES,
                "n_t2_features": N_T2_FEATURES,
                "n_t3_features": N_T3_FEATURES,
                "n_heads": len(ALL_HEAD_NAMES),
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "epochs_per_fold": EPOCHS_PER_FOLD,
                "warmup_steps": WARMUP_STEPS,
                "grad_clip": GRAD_CLIP,
                "n_folds_planned": len(folds),
                "wf_train_days": train_days,
                "first_oot_monday": FIRST_OOT_MONDAY,
                "warmstart_ckpt": warmstart_ckpt,
                "node": socket.gethostname(),
                "gpu": gpu_name,
                "mixed_precision": "bf16" if use_amp else "none",
                "log_sigma_init": LOG_SIGMA_INIT,
                "loss_kind": "uncertainty_weighted_kendall2018",
                # We retain the v3.2 lambdas in metadata for reference, but the
                # v3.3 loss does NOT use them — it uses learnable log_sigmas.
                "v32_reference_lambdas": json.dumps(LOSS_LAMBDA),
            })
            logger.info(f"MLflow run started: {mlflow_run.info.run_id}")
        except Exception as e:
            logger.warning(f"MLflow init failed: {e} — continuing without tracking")
            mlflow_run = None

    concat_results = {h: {"preds": [], "targets": [], "masks": []} for h in ALL_HEAD_NAMES}

    try:
        start_fold = int(os.environ.get("V32_START_FOLD", os.environ.get("V33_START_FOLD", 0)))
        for f_info in folds:
            fold_idx = f_info["fold"]
            if fold_idx < start_fold:
                logger.info(f"Skipping fold {fold_idx} (start_fold={start_fold})")
                continue
            logger.info("=" * 60)
            logger.info(f"FOLD {fold_idx} | train {f_info['train_start']}→{f_info['train_end']} "
                        f"| OOT {f_info['oot_start']}→{f_info['oot_end']}")
            logger.info("=" * 60)

            train_ds = SmartV32Dataset(
                data_dir=data_dir, fifo_label_dir=fifo_label_dir,
                alpha_label_dir=alpha_label_dir, pt_pred_dir=pt_pred_dir,
                tier2_parquet_root=tier2_parquet_root,
                tier3_parquet_root=tier3_parquet_root,
                dates=f_info["train_dates"],
                window_t1=WINDOW_SIZE_T1, window_t2=WINDOW_SIZE_T2, window_t3=WINDOW_SIZE_T3,
                stride=STRIDE, feature_stats=None,
                cache_size=1, require_alpha_labels=False,
            )
            feature_stats = train_ds.get_feature_stats()
            np.savez(
                output_dir / f"fold_{fold_idx:02d}_feature_stats.npz",
                mean_t1=feature_stats["mean_t1"], std_t1=feature_stats["std_t1"],
            )
            oot_ds = SmartV32Dataset(
                data_dir=data_dir, fifo_label_dir=fifo_label_dir,
                alpha_label_dir=alpha_label_dir, pt_pred_dir=pt_pred_dir,
                tier2_parquet_root=tier2_parquet_root,
                tier3_parquet_root=tier3_parquet_root,
                dates=f_info["oot_dates"],
                window_t1=WINDOW_SIZE_T1, window_t2=WINDOW_SIZE_T2, window_t3=WINDOW_SIZE_T3,
                stride=STRIDE, feature_stats=feature_stats,
                cache_size=1, require_alpha_labels=False,
            )

            train_loader = DataLoader(
                train_ds, batch_size=BATCH_SIZE, shuffle=False,
                num_workers=2, pin_memory=True, drop_last=True,
                persistent_workers=False, prefetch_factor=2,
                collate_fn=collate_v32,
            )
            oot_loader = DataLoader(
                oot_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                num_workers=1, pin_memory=True,
                persistent_workers=False, prefetch_factor=2,
                collate_fn=collate_v32,
            )

            model = CNNMambaV32().to(device)
            if fold_idx == 0:
                logger.info(f"Model parameters: {count_parameters(model):,}")
            warm_stats = model.load_v3_warmstart(warmstart_ckpt, device)
            if MLFLOW_AVAILABLE and mlflow_run is not None and fold_idx == 0:
                try:
                    mlflow.log_params({
                        "warmstart_loaded_tensors": warm_stats["loaded"],
                        "warmstart_skipped_keys": warm_stats["skipped"],
                        "warmstart_init_random_keys": warm_stats["init_random"],
                    })
                except Exception as e:
                    logger.warning(f"MLflow warmstart log failed: {e}")

            total_steps = EPOCHS_PER_FOLD * len(train_loader)

            fold_resume_state = None
            if resume_from_intra_ckpt is not None:
                rp = Path(resume_from_intra_ckpt)
                if rp.exists():
                    try:
                        rckpt = torch.load(rp, map_location=device, weights_only=False)
                        rfold = int(rckpt.get("fold", fold_idx))
                        if rfold == fold_idx:
                            model.load_state_dict(rckpt["model_state"], strict=False)
                            ver = int(rckpt.get("ckpt_version", 1))
                            if ver >= 2:
                                fold_resume_state = rckpt
                                logger.info(
                                    f"  Fold {fold_idx} loaded RESUME ckpt v{ver} from {rp.name} "
                                    f"(ep={rckpt.get('epoch')} batch={rckpt.get('batch')})"
                                )
                            else:
                                logger.info(
                                    f"  Fold {fold_idx} loaded LEGACY ckpt v1 from {rp.name} "
                                    f"as warmstart"
                                )
                        else:
                            logger.info(
                                f"  Fold {fold_idx} skipping resume ckpt (saved for fold {rfold})"
                            )
                    except Exception as e:
                        logger.warning(f"Resume ckpt load failed ({e}); proceeding without resume")
                else:
                    logger.warning(f"Resume ckpt path does not exist: {rp}")

            fold_result = train_one_fold_v33(
                model, train_loader, oot_loader,
                fold_idx, output_dir, mlflow_run, device,
                total_train_steps=total_steps, use_amp=use_amp,
                resume_state=fold_resume_state,
            )

            # Reload best for OOT inference
            ckpt_path = output_dir / f"fold_{fold_idx:02d}_best.pt"
            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                model.load_state_dict(ckpt["model_state"])
                logger.info(f"Loaded best (val_loss={ckpt['val_loss']:.4f})")

            eval_loss_fn = JointMultiHeadLossV33_UncertaintyWeighted(
                head_names=ALL_HEAD_NAMES
            ).to(device)
            # Restore the *trained* log_sigmas so reported OOT loss matches the
            # weighting that produced the best checkpoint.
            if "loss_state" in ckpt:
                try:
                    eval_loss_fn.load_state_dict(ckpt["loss_state"], strict=False)
                except Exception as e:
                    logger.warning(f"Could not restore loss_state for OOT eval: {e}")

            metrics, preds, targets, masks = evaluate_v32(
                model, oot_loader, eval_loss_fn, device, use_amp=use_amp
            )
            logger.info(
                f"Fold {fold_idx} OOT FINAL | "
                f"IC 1s={metrics.get('ic_log_ret_1s', float('nan')):.4f} | "
                f"IC 5s={metrics.get('ic_log_ret_5s', float('nan')):.4f} | "
                f"IC 10s={metrics.get('ic_log_ret_10s', float('nan')):.4f} | "
                f"IC 30s={metrics.get('ic_log_ret_30s', float('nan')):.4f}"
            )

            save_dict = {
                "fold_idx": np.array(fold_idx),
                "oot_dates": np.array(f_info["oot_dates"]),
            }
            for h in ALL_HEAD_NAMES:
                if h in preds:
                    save_dict[f"pred_{h}"] = preds[h]
                    save_dict[f"target_{h}"] = targets[h]
                    save_dict[f"mask_{h}"] = masks[h]
            np.savez_compressed(output_dir / f"fold_{fold_idx:02d}_oot_predictions.npz", **save_dict)

            for h in ALL_HEAD_NAMES:
                if h in preds:
                    concat_results[h]["preds"].append(preds[h])
                    concat_results[h]["targets"].append(targets[h])
                    concat_results[h]["masks"].append(masks[h])

            fold_summary = {
                "fold": fold_idx,
                "train_window": [f_info["train_start"], f_info["train_end"], len(f_info["train_dates"])],
                "oot_window": [f_info["oot_start"], f_info["oot_end"], len(f_info["oot_dates"])],
                "n_train_samples": len(train_ds),
                "n_oot_samples": len(oot_ds),
                "metrics": {k: float(v) if not np.isnan(v) else None for k, v in metrics.items()},
                "final_sigmas": fold_result.get("final_sigmas", {}),
            }
            with open(output_dir / f"fold_{fold_idx:02d}_analysis.json", "w") as fh:
                json.dump(fold_summary, fh, indent=2, default=str)

            if MLFLOW_AVAILABLE and mlflow_run is not None:
                try:
                    final_metrics = {
                        f"oot_final_f{fold_idx:02d}_{k}": float(v)
                        for k, v in metrics.items() if not np.isnan(v)
                    }
                    mlflow.log_metrics(final_metrics, step=fold_idx)
                except Exception as e:
                    logger.warning(f"MLflow final log failed: {e}")

            del train_ds, oot_ds, train_loader, oot_loader, model
            gc.collect()
            torch.cuda.empty_cache()

        # Concat IC across folds
        logger.info("=" * 60)
        logger.info("CONCAT RESULTS (all folds combined)")
        logger.info("=" * 60)
        concat_summary = {}
        for h in ALL_HEAD_NAMES:
            if not concat_results[h]["preds"]:
                continue
            p = np.concatenate(concat_results[h]["preds"])
            t = np.concatenate(concat_results[h]["targets"])
            m = np.concatenate(concat_results[h]["masks"])
            valid = m > 0
            if h.startswith("log_ret"):
                ic = compute_ic(p[valid], t[valid])
                concat_summary[f"concat_ic_{h}"] = ic
                logger.info(f"  Concat IC {h}: {ic:.4f}  (n={int(valid.sum())})")
            elif h.startswith("pred_") or h.endswith("_ticks"):
                corr = compute_ic(p[valid], t[valid])
                concat_summary[f"concat_corr_{h}"] = corr
                logger.info(f"  Concat corr {h}: {corr:.4f}  (n={int(valid.sum())})")

        with open(output_dir / "concat_summary.json", "w") as fh:
            json.dump(
                {k: (float(v) if not np.isnan(v) else None) for k, v in concat_summary.items()},
                fh, indent=2,
            )

        if MLFLOW_AVAILABLE and mlflow_run is not None:
            try:
                mlflow.log_metrics({k: float(v) for k, v in concat_summary.items() if not np.isnan(v)})
            except Exception as e:
                logger.warning(f"MLflow concat log failed: {e}")

        return concat_summary
    finally:
        if MLFLOW_AVAILABLE and mlflow_run is not None:
            try:
                mlflow.end_run()
            except Exception:
                pass


# ============================================================
# CLI
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description="CNN-Mamba v3.3 uncertainty-weighted multi-task trainer")
    p.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    p.add_argument("--fifo-label-dir", type=str, default=DEFAULT_FIFO_LABEL_DIR)
    p.add_argument("--alpha-label-dir", type=str, default=DEFAULT_ALPHA_LABEL_DIR)
    p.add_argument("--pt-pred-dir", type=str, default=DEFAULT_PT_PRED_DIR)
    p.add_argument("--tier2-parquet-root", type=str, default=DEFAULT_TIER2_PARQUET_ROOT)
    p.add_argument("--tier3-parquet-root", type=str, default=DEFAULT_TIER3_PARQUET_ROOT)
    p.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--n-folds", type=int, default=N_FOLDS)
    p.add_argument("--train-days", type=int, default=WF_TRAIN_DAYS)
    p.add_argument("--warmstart-ckpt", type=str, default=V3_WARMSTART_CKPT)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--resume-from-intra-ckpt", type=str, default=None,
        help="Path to fold_NN_intra_ckpt.pt to resume from. v2 ckpts restore full "
             "state (model + loss log_sigmas + optimizer + scheduler + RNG)."
    )
    return p.parse_args()


def main():
    args = parse_args()
    logger.info("=" * 60)
    logger.info("CNN-Mamba v3.3 Uncertainty-Weighted Multi-Task Training")
    logger.info("=" * 60)
    logger.info(f"data_dir         = {args.data_dir}")
    logger.info(f"fifo_label_dir   = {args.fifo_label_dir}")
    logger.info(f"alpha_label_dir  = {args.alpha_label_dir}")
    logger.info(f"pt_pred_dir      = {args.pt_pred_dir}")
    logger.info(f"tier2_parquet    = {args.tier2_parquet_root}")
    logger.info(f"tier3_parquet    = {args.tier3_parquet_root}")
    logger.info(f"output_dir       = {args.output_dir}")
    logger.info(f"n_folds          = {args.n_folds}")
    logger.info(f"train_days       = {args.train_days}")
    logger.info(f"warmstart_ckpt   = {args.warmstart_ckpt}")
    logger.info(f"log_sigma_init   = {LOG_SIGMA_INIT}")
    logger.info(f"MLflow URI       = {MLFLOW_TRACKING_URI}")
    logger.info(f"MLflow exp       = {MLFLOW_EXPERIMENT}")

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"device           = {device}")
    if device.type == "cuda":
        logger.info(f"GPU              = {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM             = {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    resume_path = (
        Path(args.resume_from_intra_ckpt)
        if getattr(args, "resume_from_intra_ckpt", None) else None
    )
    if resume_path is not None:
        logger.info(f"resume_intra_ckpt = {resume_path}")

    summary = run_weekly_wf_v33(
        data_dir=Path(args.data_dir),
        fifo_label_dir=Path(args.fifo_label_dir),
        alpha_label_dir=Path(args.alpha_label_dir),
        pt_pred_dir=Path(args.pt_pred_dir),
        output_dir=Path(args.output_dir),
        device=device,
        n_folds=args.n_folds,
        train_days=args.train_days,
        warmstart_ckpt=args.warmstart_ckpt,
        tier2_parquet_root=Path(args.tier2_parquet_root),
        tier3_parquet_root=Path(args.tier3_parquet_root),
        resume_from_intra_ckpt=resume_path,
    )

    logger.info("=" * 60)
    logger.info("FINAL SUMMARY (v3.3)")
    logger.info("=" * 60)
    for k, v in (summary or {}).items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
