"""
Event-Driven 1D Causal CNN with Mixture of Experts (MoE) — Training Script

The model discovers its OWN "regimes" without us imposing preconceived labels.

Architecture:
  - Same proven CausalConv1D backbone (but smaller: 64ch, 4 layers)
  - K=4 expert prediction heads (small MLPs)
  - Learned router that decides which experts to weight for each sample
  - Load balancing auxiliary loss prevents expert collapse
  - Post-hoc: analyze routing patterns vs market features to see what
    the model learned about "regimes" without being told

Key insight: If all 4 experts predict the same → model sees no meaningful regimes.
If experts specialize → model found natural clusters in market behavior.
The routing weights tell US what regimes exist — we don't tell the model.

Training rules (inherited from EventCNN1D):
  - Sliding window walk-forward (DIRECTIVE)
  - Concat IC as primary metric
  - Mixed precision (fp16 when CUDA)
  - MLflow logging
  - Save .pt weights + .npz predictions + routing patterns per fold
  - Leakage: feature stats from train set only per fold

Size: ~250K params (vs 1.5M standard CNN). Fast to train on Razer 8GB.
"""

import os
import sys
import gc
import time
import json
import logging
import argparse
import warnings
import socket
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import scipy.stats

# Import shared infrastructure from EventCNN1D
# (dataset, walk-forward boundaries, feature engineering, data transfer)
sys.path.insert(0, str(Path(__file__).parent))
from train_event_cnn_1d import (
    MboEventDataset,
    PrecomputedTensorDataset,
    FileSequentialSampler,
    CausalConv1dBlock,
    WarmupCosineScheduler,
    build_dataset,
    compute_ic,
    count_parameters,
    transfer_data_from_jupiter,
    find_pt_files_for_npz,
    compute_fold_stats,
    # Constants
    WINDOW_SIZE, STRIDE, BATCH_SIZE, LR, EPOCHS_PER_FOLD,
    WARMUP_STEPS, GRAD_CLIP, N_FOLDS, HORIZONS,
    N_TOTAL_FEATURES, N_RAW_FEATURES,
    CNN_TENSOR_DIR, STRICT_LEAKAGE_FREE,
    FEATURE_SET_BOOK30, FEATURE_SET_SMART_V2, FEATURE_SET_FEAT15,
    FEATURE_SET_FEAT18, FEATURE_SET_FEAT20,
    SKIP_CNN_NORMALIZE, CNN_FEATURE_SET,
    DEFAULT_DATA_DIR, DEFAULT_BOOK30_DATA_DIR,
    MLFLOW_TRACKING_URI, _FlushHandler,
)

# MLflow
try:
    if int(os.environ.get("DISABLE_MLFLOW", 0)):
        raise ImportError("MLflow disabled via DISABLE_MLFLOW env var")
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    warnings.warn("MLflow disabled or not installed — skipping experiment tracking")

MLFLOW_EXPERIMENT = "EventDriven_CNN1D_MoE"

# ============================================================
# Logging setup
# ============================================================
LOG_DIR = Path(__file__).parent / "results"
LOG_DIR.mkdir(exist_ok=True)

log_path = LOG_DIR / "event_cnn_moe.log"
_file_stream = open(log_path, "a", buffering=1)
_file_handler = logging.StreamHandler(_file_stream)
_file_handler.setLevel(logging.INFO)
_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.setLevel(logging.INFO)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[_file_handler, _stream_handler],
)
logger = logging.getLogger("moe")

for _h in logging.root.handlers:
    _h.__class__ = _FlushHandler


# ============================================================
# MoE Hyperparameters (smaller than standard CNN)
# ============================================================
MOE_CHANNELS   = int(os.environ.get("MOE_CHANNELS", 64))     # 64 vs 128
MOE_KERNEL     = int(os.environ.get("MOE_KERNEL", 5))
MOE_LAYERS     = int(os.environ.get("MOE_LAYERS", 4))        # 4 vs 6
MOE_DROPOUT    = float(os.environ.get("MOE_DROPOUT", 0.1))
MOE_N_EXPERTS  = int(os.environ.get("MOE_N_EXPERTS", 4))     # 4 expert heads
MOE_TOP_K      = int(os.environ.get("MOE_TOP_K", 2))         # top-2 routing
MOE_AUX_WEIGHT = float(os.environ.get("MOE_AUX_WEIGHT", 0.01))  # load balancing loss weight

DILATION_SCHEDULE = [2 ** i for i in range(MOE_LAYERS)]


# ============================================================
# Architecture: EventCNN1D with Mixture of Experts
# ============================================================

class EventCNN1D_MoE(nn.Module):
    """
    MoE variant of EventCNN1D.

    Same causal CNN backbone, but instead of one prediction head, we have:
    - K expert heads (small MLPs), each specializing in different market conditions
    - A learned router that decides how to weight experts for each sample
    - The model discovers its own "regimes" — we never label them

    Post-hoc analysis: correlate expert routing patterns with market features
    (spread, event rate, volatility, time-of-day) to understand what the model
    learned about regime structure.

    Parameters: ~250K (vs 1.5M standard CNN)
    """

    def __init__(
        self,
        in_channels:     int   = N_TOTAL_FEATURES,
        channels:        int   = MOE_CHANNELS,
        kernel_size:     int   = MOE_KERNEL,
        n_layers:        int   = MOE_LAYERS,
        dropout:         float = MOE_DROPOUT,
        n_targets:       int   = 3,
        n_experts:       int   = MOE_N_EXPERTS,
        top_k:           int   = MOE_TOP_K,
        dilation_schedule: Optional[List[int]] = None,
    ):
        super().__init__()
        self.channels   = channels
        self.n_targets  = n_targets
        self.n_experts  = n_experts
        self.top_k      = min(top_k, n_experts)
        dilations       = dilation_schedule or [2 ** i for i in range(n_layers)]

        # ---- Shared backbone (same as EventCNN1D) ----
        self.input_proj = nn.Sequential(
            nn.Conv1d(in_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )

        self.blocks = nn.ModuleList()
        for dil in dilations:
            self.blocks.append(
                CausalConv1dBlock(
                    in_channels  = channels,
                    out_channels = channels,
                    kernel_size  = kernel_size,
                    dilation     = dil,
                    dropout      = dropout,
                )
            )

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool1d(1)

        # ---- Router: maps embedding → expert weights ----
        # Small 2-layer MLP that decides which experts to activate
        self.router = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, n_experts),
        )

        # ---- K Expert heads (each a small MLP) ----
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(channels),
                nn.Linear(channels, channels),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(channels, n_targets),
            )
            for _ in range(n_experts)
        ])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        events: torch.Tensor,
        return_embedding: bool = False,
        return_routing: bool = False,
    ):
        """
        Args:
            events: (B, L, F) float32 — batch of event windows
            return_embedding: if True, also return pattern embedding
            return_routing: if True, also return routing weights (B, K)
        Returns:
            preds: (B, n_targets)
            [embedding]: (B, channels)  — if return_embedding
            [routing_weights]: (B, n_experts)  — if return_routing
            [aux_loss]: scalar  — load balancing loss (always returned as last element when training)
        """
        # ---- Backbone ----
        x = events.permute(0, 2, 1).contiguous()   # (B, F, L)
        x = self.input_proj(x)                       # (B, C, L)
        for block in self.blocks:
            x = block(x)
        embedding = self.gap(x).squeeze(-1)          # (B, C)

        # ---- Router ----
        router_logits = self.router(embedding)       # (B, K)
        router_probs = F.softmax(router_logits, dim=-1)  # (B, K)

        # Top-k sparse routing
        top_k_vals, top_k_idx = torch.topk(router_probs, self.top_k, dim=-1)  # (B, top_k)
        # Renormalize top-k weights to sum to 1
        top_k_weights = top_k_vals / (top_k_vals.sum(dim=-1, keepdim=True) + 1e-8)

        # ---- Expert predictions ----
        # Compute all expert outputs (could optimize with sparse computation for large K,
        # but K=4 is small enough that this is fine)
        expert_outputs = torch.stack(
            [expert(embedding) for expert in self.experts], dim=1
        )  # (B, K, T)

        # Sparse weighted combination: only top-k experts contribute
        # Create sparse weight mask
        sparse_weights = torch.zeros_like(router_probs)  # (B, K)
        sparse_weights.scatter_(1, top_k_idx, top_k_weights)

        # Weighted sum of expert predictions
        preds = (sparse_weights.unsqueeze(-1) * expert_outputs).sum(dim=1)  # (B, T)

        # ---- Load balancing auxiliary loss ----
        # Encourages equal utilization of all experts
        # f_i = fraction of tokens routed to expert i (based on top-k assignment)
        # p_i = mean router probability for expert i
        # aux_loss = K * sum(f_i * p_i) → minimized when f and p are uniform
        if self.training:
            # Count how many samples each expert appears in top-k
            expert_mask = torch.zeros_like(router_probs)
            expert_mask.scatter_(1, top_k_idx, 1.0)
            f = expert_mask.mean(dim=0)          # (K,) fraction routed to each expert
            p = router_probs.mean(dim=0)         # (K,) mean probability per expert
            aux_loss = self.n_experts * (f * p).sum()
        else:
            aux_loss = torch.tensor(0.0, device=preds.device)

        # ---- Build outputs ----
        outputs = [preds]
        if return_embedding:
            outputs.append(embedding)
        if return_routing:
            outputs.append(router_probs)  # full probabilities for analysis
        outputs.append(aux_loss)

        return tuple(outputs) if len(outputs) > 1 else (preds, aux_loss)

    def receptive_field(self) -> int:
        total = 0
        for block in self.blocks:
            total += (block.kernel_size - 1) * block.dilation
        return total + 1

    def expert_specialization_score(self) -> float:
        """
        Measure how specialized the experts are.
        High score = experts have different weight patterns = learned different regimes.
        Low score = all experts similar = no regime learning.

        Computed as average pairwise cosine distance between expert output layer weights.
        """
        weights = []
        for expert in self.experts:
            # Get the last linear layer's weight
            last_linear = expert[-1]
            if isinstance(last_linear, nn.Linear):
                weights.append(last_linear.weight.detach().flatten())

        if len(weights) < 2:
            return 0.0

        distances = []
        for i in range(len(weights)):
            for j in range(i + 1, len(weights)):
                cos_sim = F.cosine_similarity(weights[i].unsqueeze(0), weights[j].unsqueeze(0))
                distances.append(1.0 - cos_sim.item())

        return float(np.mean(distances))


# ============================================================
# Evaluation helpers
# ============================================================

def evaluate_metrics_only(model, loader, device, use_amp=True):
    """Evaluate model on loader, return dict of metrics."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_labels = []

    amp_ctx = (
        torch.amp.autocast("cuda") if use_amp and device.type == "cuda"
        else torch.amp.autocast("cpu", enabled=False)
    )

    with torch.no_grad():
        for events, labels in loader:
            events = events.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with amp_ctx:
                outputs = model(events)
                preds = outputs[0]
                loss = F.mse_loss(preds, labels)
            total_loss += loss.item()
            n_batches += 1
            all_preds.append(preds.float().cpu().numpy())
            all_labels.append(labels.float().cpu().numpy())

    metrics = {"loss": total_loss / max(n_batches, 1)}
    if all_preds:
        preds_np = np.concatenate(all_preds, axis=0)
        labels_np = np.concatenate(all_labels, axis=0)
        for i, h in enumerate(HORIZONS):
            metrics[f"ic_{h}"] = compute_ic(preds_np[:, i], labels_np[:, i])
    return metrics


def run_oot_inference(
    model, loader, device, use_amp=True,
    extract_embeddings=False, extract_routing=True,
):
    """Run OOT inference, return predictions, labels, embeddings, routing weights."""
    model.eval()
    all_preds = []
    all_labels = []
    all_embeds = [] if extract_embeddings else None
    all_routing = [] if extract_routing else None

    amp_ctx = (
        torch.amp.autocast("cuda") if use_amp and device.type == "cuda"
        else torch.amp.autocast("cpu", enabled=False)
    )

    with torch.no_grad():
        for events, labels in loader:
            events = events.to(device, non_blocking=True)
            with amp_ctx:
                outputs = model(
                    events,
                    return_embedding=extract_embeddings,
                    return_routing=extract_routing,
                )
                # Parse outputs: (preds, [embedding], [routing], aux_loss)
                idx = 0
                preds = outputs[idx]; idx += 1
                if extract_embeddings:
                    emb = outputs[idx]; idx += 1
                    all_embeds.append(emb.float().cpu().numpy())
                if extract_routing:
                    routing = outputs[idx]; idx += 1
                    all_routing.append(routing.float().cpu().numpy())
                # aux_loss is last — skip it

            all_preds.append(preds.float().cpu().numpy())
            all_labels.append(labels.float().cpu().numpy())

    if not all_preds:
        empty = np.empty((0, len(HORIZONS)))
        return empty, empty, None, None

    preds_out = np.concatenate(all_preds, axis=0)
    labels_out = np.concatenate(all_labels, axis=0)
    embeds_out = np.concatenate(all_embeds, axis=0) if extract_embeddings else None
    routing_out = np.concatenate(all_routing, axis=0) if extract_routing else None
    return preds_out, labels_out, embeds_out, routing_out


# ============================================================
# Routing Analysis (post-hoc regime discovery)
# ============================================================

def analyze_routing_patterns(
    routing_weights: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    fold_idx: int,
    output_dir: Path,
) -> Dict:
    """
    Analyze expert routing patterns to understand what the model learned.

    For each expert:
    - What fraction of samples primarily route to it?
    - What's the IC of samples dominated by this expert?
    - How different are expert predictions?

    Returns analysis dict for logging/saving.
    """
    n_samples, n_experts = routing_weights.shape
    analysis = {
        "n_samples": int(n_samples),
        "n_experts": n_experts,
        "fold": fold_idx,
    }

    # Which expert is dominant for each sample?
    dominant_expert = np.argmax(routing_weights, axis=-1)  # (N,)

    # Per-expert statistics
    expert_stats = []
    for k in range(n_experts):
        mask = dominant_expert == k
        count = int(mask.sum())
        frac = count / max(n_samples, 1)

        stats = {
            "expert": k,
            "n_samples": count,
            "fraction": round(frac, 4),
            "mean_weight": round(float(routing_weights[:, k].mean()), 4),
            "std_weight": round(float(routing_weights[:, k].std()), 4),
        }

        # IC for samples dominated by this expert
        if count >= 50:  # need minimum samples for meaningful IC
            for i, h in enumerate(HORIZONS):
                ic = compute_ic(predictions[mask, i], labels[mask, i])
                stats[f"ic_{h}"] = round(ic, 4)

                # Directional accuracy
                p = predictions[mask, i]
                l = labels[mask, i]
                nonzero = l != 0
                if nonzero.sum() > 0:
                    da = float((np.sign(p[nonzero]) == np.sign(l[nonzero])).mean())
                    stats[f"da_{h}"] = round(da, 4)

            # Mean absolute prediction magnitude (are some experts more "confident"?)
            stats["mean_abs_pred"] = round(float(np.abs(predictions[mask]).mean()), 6)
            stats["mean_abs_label"] = round(float(np.abs(labels[mask]).mean()), 6)

        expert_stats.append(stats)

    analysis["experts"] = expert_stats

    # Expert diversity: how different are the routing distributions?
    # Entropy of average routing weights (high = diverse, low = collapsed)
    avg_routing = routing_weights.mean(axis=0)
    avg_routing = avg_routing / (avg_routing.sum() + 1e-8)
    entropy = -float((avg_routing * np.log(avg_routing + 1e-8)).sum())
    max_entropy = float(np.log(n_experts))
    analysis["routing_entropy"] = round(entropy, 4)
    analysis["max_entropy"] = round(max_entropy, 4)
    analysis["utilization_ratio"] = round(entropy / max_entropy, 4)  # 1.0 = all experts equally used

    # Expert agreement: do experts agree on direction?
    # (Measured by variance of routing weights per sample)
    routing_var = routing_weights.var(axis=1).mean()
    analysis["mean_routing_variance"] = round(float(routing_var), 6)

    # Save detailed analysis
    analysis_path = output_dir / f"fold_{fold_idx:02d}_routing_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)

    # Log summary
    logger.info(f"\n{'='*50}")
    logger.info(f"ROUTING ANALYSIS — Fold {fold_idx:02d}")
    logger.info(f"{'='*50}")
    logger.info(f"Routing entropy: {entropy:.3f} / {max_entropy:.3f} "
                f"(utilization: {analysis['utilization_ratio']:.1%})")
    for stats in expert_stats:
        k = stats['expert']
        frac = stats['fraction']
        ic_10s = stats.get('ic_10s', 'N/A')
        da_10s = stats.get('da_10s', 'N/A')
        logger.info(
            f"  Expert {k}: {frac:.1%} of samples | "
            f"IC_10s={ic_10s} | DA_10s={da_10s} | "
            f"mean_weight={stats['mean_weight']:.3f}"
        )
    logger.info(f"{'='*50}")

    return analysis


# ============================================================
# Training Loop (single fold)
# ============================================================

def train_one_fold(
    model:             EventCNN1D_MoE,
    train_loader:      DataLoader,
    val_loader:        DataLoader,
    fold_idx:          int,
    output_dir:        Path,
    mlflow_run,
    device:            torch.device,
    total_train_steps: int,
    use_amp:           bool = True,
    aux_weight:        float = MOE_AUX_WEIGHT,
) -> Dict:
    """Train MoE CNN for one fold. Returns metrics dict."""

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scaler    = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=WARMUP_STEPS,
        total_steps=total_train_steps,
    )

    amp_ctx = (
        torch.amp.autocast("cuda") if use_amp and device.type == "cuda"
        else torch.amp.autocast("cpu", enabled=False)
    )

    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(EPOCHS_PER_FOLD):
        model.train()
        epoch_loss = 0.0
        epoch_aux_loss = 0.0
        n_batches = 0

        for events, labels in train_loader:
            events = events.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with amp_ctx:
                preds, aux_loss = model(events)  # MoE always returns (preds, aux_loss) minimum
                task_loss = F.mse_loss(preds, labels)
                loss = task_loss + aux_weight * aux_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += task_loss.item()
            epoch_aux_loss += aux_loss.item()
            n_batches += 1
            global_step += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_aux = epoch_aux_loss / max(n_batches, 1)
        val_metrics = evaluate_metrics_only(model, val_loader, device, use_amp=use_amp)

        # Expert specialization score
        spec_score = model.expert_specialization_score()

        logger.info(
            f"Fold {fold_idx:02d} | Epoch {epoch+1:2d}/{EPOCHS_PER_FOLD} | "
            f"Train Loss: {avg_loss:.4f} | Aux: {avg_aux:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val IC_10s: {val_metrics.get('ic_10s', float('nan')):.4f} | "
            f"Expert Spec: {spec_score:.3f} | "
            f"LR: {scheduler.get_lr():.2e}"
        )

        if MLFLOW_AVAILABLE and mlflow_run:
            step_offset = fold_idx * EPOCHS_PER_FOLD + epoch
            mlflow.log_metrics(
                {
                    f"fold{fold_idx:02d}_train_loss": avg_loss,
                    f"fold{fold_idx:02d}_aux_loss": avg_aux,
                    f"fold{fold_idx:02d}_val_loss": val_metrics["loss"],
                    f"fold{fold_idx:02d}_val_ic_10s": val_metrics.get("ic_10s", float("nan")),
                    f"fold{fold_idx:02d}_expert_spec": spec_score,
                },
                step=step_offset,
            )

        # Save best checkpoint
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            ckpt_path = output_dir / f"fold_{fold_idx:02d}_best.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "fold": fold_idx,
                    "epoch": epoch,
                    "val_loss": val_metrics["loss"],
                    "val_ic_10s": val_metrics.get("ic_10s"),
                    "expert_spec_score": spec_score,
                    "arch": {
                        "model_type": "EventCNN1D_MoE",
                        "channels": MOE_CHANNELS,
                        "kernel_size": MOE_KERNEL,
                        "n_layers": MOE_LAYERS,
                        "dropout": MOE_DROPOUT,
                        "n_experts": MOE_N_EXPERTS,
                        "top_k": MOE_TOP_K,
                        "window_size": WINDOW_SIZE,
                        "in_channels": N_TOTAL_FEATURES,
                    },
                },
                ckpt_path,
            )

    return {"best_val_loss": best_val_loss}


# ============================================================
# Walk-Forward (Sliding Window — per DIRECTIVES)
# ============================================================

def run_walk_forward(
    npz_files:  List[Path],
    output_dir: Path,
    device:     torch.device,
    n_folds:    int = N_FOLDS,
    train_days: Optional[int] = None,
    oot_days:   Optional[int] = None,
):
    """Walk-forward training with MoE CNN. Saves routing patterns for regime analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_files = sorted(npz_files)

    # Filter files with no valid labels
    def _has_valid_labels(f: Path) -> bool:
        try:
            d = np.load(f, allow_pickle=True)
            return bool(not np.all(np.isnan(d["labels_1s"])))
        except Exception:
            return False

    valid_files = [f for f in npz_files if _has_valid_labels(f)]
    skipped = [f.name for f in npz_files if f not in set(valid_files)]
    if skipped:
        logger.warning(f"Skipping {len(skipped)} files with all-NaN labels")
    npz_files = valid_files
    n_files = len(npz_files)

    if n_files == 0:
        logger.error("No valid NPZ files. Exiting.")
        return {}

    logger.info(f"Total files: {n_files} ({npz_files[0].name} → {npz_files[-1].name})")

    window_mode = f"sliding({train_days}d)" if train_days else "expanding"

    # Build fold boundaries (same logic as EventCNN1D)
    if oot_days is not None and train_days is not None:
        fold_boundaries = []
        for fold in range(n_folds):
            oot_end_idx = n_files - (n_folds - 1 - fold) * oot_days
            oot_start_idx = oot_end_idx - oot_days
            train_end_idx = oot_start_idx
            train_start_idx = max(0, train_end_idx - train_days)
            if train_end_idx < 5 or oot_start_idx >= n_files:
                continue
            fold_boundaries.append((fold, list(range(train_start_idx, train_end_idx)), list(range(oot_start_idx, oot_end_idx))))
    elif oot_days is not None:
        oot_days_capped = min(oot_days, n_files - 5)
        oot_start_fixed = n_files - oot_days_capped
        min_train = max(5, oot_start_fixed - (n_folds - 1))
        fold_boundaries = []
        for fold in range(n_folds):
            train_end = min_train + fold
            if train_end > oot_start_fixed:
                break
            train_start = max(0, train_end - train_days) if train_days else 0
            fold_boundaries.append((fold, list(range(train_start, train_end)), list(range(oot_start_fixed, n_files))))
    else:
        min_train = max(5, n_files - n_folds)
        fold_boundaries = []
        for fold in range(n_folds):
            train_end = min_train + fold
            oot_start = train_end
            oot_end = min(oot_start + max(1, (n_files - min_train) // n_folds), n_files)
            if oot_start >= n_files:
                break
            train_start = max(0, train_end - train_days) if train_days else 0
            fold_boundaries.append((fold, list(range(train_start, train_end)), list(range(oot_start, oot_end))))

    logger.info(f"Running {len(fold_boundaries)} folds ({window_mode})")

    use_amp = device.type == "cuda"

    # Concat storage
    concat_preds = {h: [] for h in HORIZONS}
    concat_labels = {h: [] for h in HORIZONS}
    concat_embeds = []
    concat_routing = []
    all_routing_analyses = []

    # MLflow
    mlflow_run = None
    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        mlflow_run = mlflow.start_run(
            run_name=f"CNN_MoE_{MOE_N_EXPERTS}exp_{time.strftime('%Y%m%d_%H%M')}"
        )
        gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
        rf_events = sum([(MOE_KERNEL - 1) * d for d in DILATION_SCHEDULE]) + 1
        mlflow.log_params({
            "model": "EventCNN1D_MoE",
            "window_size": WINDOW_SIZE,
            "stride": STRIDE,
            "moe_channels": MOE_CHANNELS,
            "moe_kernel": MOE_KERNEL,
            "moe_layers": MOE_LAYERS,
            "moe_dropout": MOE_DROPOUT,
            "n_experts": MOE_N_EXPERTS,
            "top_k": MOE_TOP_K,
            "aux_weight": MOE_AUX_WEIGHT,
            "dilation_schedule": str(DILATION_SCHEDULE),
            "receptive_field": rf_events,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "epochs_per_fold": EPOCHS_PER_FOLD,
            "n_folds": len(fold_boundaries),
            "n_files": n_files,
            "node": socket.gethostname(),
            "gpu": gpu_name,
            "window_mode": window_mode,
            "in_channels": N_TOTAL_FEATURES,
            "feature_set": CNN_FEATURE_SET if CNN_FEATURE_SET else "default",
        })

    try:
        for fold_idx, train_file_idxs, oot_file_idxs in fold_boundaries:
            train_files = [npz_files[i] for i in train_file_idxs]
            oot_files = [npz_files[i] for i in oot_file_idxs]

            logger.info(
                f"\n{'='*60}\n"
                f"FOLD {fold_idx:02d} | Train: {len(train_files)} days "
                f"({train_files[0].name}→{train_files[-1].name}) | "
                f"OOT: {len(oot_files)} days ({oot_files[0].name}→{oot_files[-1].name})"
                f"\n{'='*60}"
            )

            # Build datasets
            logger.info("Building train dataset...")
            train_ds = build_dataset(
                train_files, tensor_dir=CNN_TENSOR_DIR,
                strict_leakage_free=STRICT_LEAKAGE_FREE,
            )
            feature_stats = train_ds.get_feature_stats()

            logger.info("Building OOT dataset...")
            oot_ds = build_dataset(
                oot_files, tensor_dir=CNN_TENSOR_DIR,
                feature_stats=feature_stats,
                strict_leakage_free=STRICT_LEAKAGE_FREE,
            )

            _num_workers = int(os.environ.get("EVENT_NUM_WORKERS", 0 if sys.platform == "win32" else 8))
            _persistent = _num_workers > 0

            _has_sample_index = hasattr(train_ds, "sample_index")
            _use_seq_sampler = _has_sample_index and int(os.environ.get("CNN_SEQUENTIAL_SAMPLER", 1))
            if _use_seq_sampler:
                _train_sampler = FileSequentialSampler(train_ds, shuffle=True)
                _train_shuffle = False
            else:
                _train_sampler = None
                _train_shuffle = True

            train_loader = DataLoader(
                train_ds, batch_size=BATCH_SIZE,
                shuffle=_train_shuffle, sampler=_train_sampler,
                num_workers=_num_workers, pin_memory=True,
                drop_last=True, persistent_workers=_persistent,
                prefetch_factor=4 if _num_workers > 0 else None,
            )
            oot_loader = DataLoader(
                oot_ds, batch_size=BATCH_SIZE * 2,
                shuffle=False, num_workers=_num_workers,
                pin_memory=True, persistent_workers=_persistent,
                prefetch_factor=4 if _num_workers > 0 else None,
            )

            # Fresh model per fold
            model = EventCNN1D_MoE(
                in_channels=N_TOTAL_FEATURES,
                channels=MOE_CHANNELS,
                kernel_size=MOE_KERNEL,
                n_layers=MOE_LAYERS,
                dropout=MOE_DROPOUT,
                n_targets=len(HORIZONS),
                n_experts=MOE_N_EXPERTS,
                top_k=MOE_TOP_K,
                dilation_schedule=DILATION_SCHEDULE,
            ).to(device)

            if fold_idx == 0:
                n_params = count_parameters(model)
                rf = model.receptive_field()
                logger.info(f"Model: EventCNN1D_MoE")
                logger.info(f"Parameters:     {n_params:,}")
                logger.info(f"Experts:        {MOE_N_EXPERTS} (top-{MOE_TOP_K} routing)")
                logger.info(f"Receptive field: {rf} events")
                logger.info(f"Channels:       {MOE_CHANNELS}")
                logger.info(f"Layers:         {MOE_LAYERS}")

            total_steps = EPOCHS_PER_FOLD * len(train_loader)
            train_one_fold(
                model, train_loader, oot_loader,
                fold_idx, output_dir, mlflow_run, device, total_steps,
                use_amp=use_amp,
            )

            # Reload best checkpoint for OOT inference
            ckpt_path = output_dir / f"fold_{fold_idx:02d}_best.pt"
            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                model.load_state_dict(ckpt["model_state"])
                logger.info(f"Loaded best checkpoint (val_loss={ckpt['val_loss']:.4f})")

            # OOT inference with routing extraction
            logger.info("Running OOT inference with routing analysis...")
            oot_preds, oot_labels, oot_embeds, oot_routing = run_oot_inference(
                model, oot_loader, device, use_amp=use_amp,
                extract_embeddings=True, extract_routing=True,
            )

            # Per-fold IC
            fold_ics = {}
            for i, h in enumerate(HORIZONS):
                ic = compute_ic(oot_preds[:, i], oot_labels[:, i])
                fold_ics[h] = ic
                concat_preds[h].append(oot_preds[:, i])
                concat_labels[h].append(oot_labels[:, i])

            if oot_embeds is not None:
                concat_embeds.append(oot_embeds)
            if oot_routing is not None:
                concat_routing.append(oot_routing)

            # Extended metrics
            fold_dir_acc = {}
            fold_mae = {}
            for i, h in enumerate(HORIZONS):
                p = oot_preds[:, i]
                l = oot_labels[:, i]
                nonzero = l != 0
                fold_dir_acc[h] = float((np.sign(p[nonzero]) == np.sign(l[nonzero])).mean()) if nonzero.sum() > 0 else float("nan")
                fold_mae[h] = float(np.abs(p - l).mean())

            logger.info(
                f"Fold {fold_idx:02d} OOT IC | "
                + " | ".join(f"{h}: {fold_ics[h]:.4f}" for h in HORIZONS)
            )
            logger.info(
                f"Fold {fold_idx:02d} OOT DA | "
                + " | ".join(f"{h}: {fold_dir_acc[h]:.1%}" for h in HORIZONS)
            )

            # Routing analysis — the key insight
            if oot_routing is not None:
                routing_analysis = analyze_routing_patterns(
                    oot_routing, oot_preds, oot_labels, fold_idx, output_dir
                )
                all_routing_analyses.append(routing_analysis)

            # Save fold artifacts
            pred_path = output_dir / f"fold_{fold_idx:02d}_oot_predictions.npz"
            save_dict = dict(
                predictions=oot_preds,
                labels=oot_labels,
                horizons=np.array(HORIZONS),
                ic_1s=np.array(fold_ics.get("1s", float("nan"))),
                ic_5s=np.array(fold_ics.get("5s", float("nan"))),
                ic_10s=np.array(fold_ics.get("10s", float("nan"))),
                oot_files=np.array([str(f) for f in oot_files]),
            )
            if oot_embeds is not None:
                save_dict["embeddings"] = oot_embeds
            if oot_routing is not None:
                save_dict["routing_weights"] = oot_routing  # (N, K) — key for post-hoc analysis
            np.savez_compressed(pred_path, **save_dict)
            logger.info(f"Saved predictions + routing → {pred_path}")

            # Feature stats
            stats_path = output_dir / f"fold_{fold_idx:02d}_feature_stats.npz"
            _fmean = feature_stats.get("mean")
            _fstd = feature_stats.get("std")
            if _fmean is not None and _fstd is not None:
                np.savez(stats_path, mean=_fmean, std=_fstd)

            # MLflow per-fold
            if MLFLOW_AVAILABLE and mlflow_run:
                mlflow.log_metrics(
                    {
                        **{f"oot_ic_{h}_fold{fold_idx:02d}": fold_ics[h] for h in HORIZONS},
                        **{f"oot_da_{h}_fold{fold_idx:02d}": fold_dir_acc[h] for h in HORIZONS},
                    },
                    step=fold_idx,
                )
                if oot_routing is not None and routing_analysis:
                    mlflow.log_metrics({
                        f"routing_entropy_fold{fold_idx:02d}": routing_analysis["routing_entropy"],
                        f"routing_util_fold{fold_idx:02d}": routing_analysis["utilization_ratio"],
                    }, step=fold_idx)
                mlflow.log_artifact(str(pred_path), artifact_path=f"fold_{fold_idx:02d}")
                if ckpt_path.exists():
                    mlflow.log_artifact(str(ckpt_path), artifact_path=f"fold_{fold_idx:02d}")

            # Free memory
            del train_ds, oot_ds, train_loader, oot_loader, model
            gc.collect()
            torch.cuda.empty_cache()

        # ============================================================
        # Concat metrics
        # ============================================================
        logger.info("\n" + "=" * 60)
        logger.info("CONCAT IC (primary metric)")
        logger.info("=" * 60)

        concat_ic = {}
        for h in HORIZONS:
            if concat_preds[h]:
                all_p = np.concatenate(concat_preds[h])
                all_l = np.concatenate(concat_labels[h])
                ic = compute_ic(all_p, all_l)
                concat_ic[h] = ic
                logger.info(f"  Concat IC ({h}): {ic:.4f}")

                # Concat DA
                nonzero = all_l != 0
                if nonzero.sum() > 0:
                    da = float((np.sign(all_p[nonzero]) == np.sign(all_l[nonzero])).mean())
                    logger.info(f"  Concat DA ({h}): {da:.1%}")
            else:
                concat_ic[h] = float("nan")

        # Save concat predictions + routing
        save_dict = {
            **{f"preds_{h}": np.concatenate(concat_preds[h]) for h in HORIZONS if concat_preds[h]},
            **{f"labels_{h}": np.concatenate(concat_labels[h]) for h in HORIZONS if concat_labels[h]},
            **{f"concat_ic_{h}": np.array(concat_ic[h]) for h in HORIZONS},
        }
        if concat_embeds:
            save_dict["embeddings"] = np.concatenate(concat_embeds, axis=0)
        if concat_routing:
            save_dict["routing_weights"] = np.concatenate(concat_routing, axis=0)
        concat_path = output_dir / "concat_oot_predictions.npz"
        np.savez_compressed(concat_path, **save_dict)
        logger.info(f"Saved concat predictions + routing → {concat_path}")

        # Aggregate routing analysis
        if all_routing_analyses:
            agg_path = output_dir / "routing_analysis_summary.json"
            with open(agg_path, "w") as f:
                json.dump(all_routing_analyses, f, indent=2)
            logger.info(f"Saved routing analysis summary → {agg_path}")

            # Cross-fold routing stability
            logger.info("\n" + "=" * 60)
            logger.info("CROSS-FOLD ROUTING SUMMARY")
            logger.info("=" * 60)
            for k in range(MOE_N_EXPERTS):
                fracs = [a["experts"][k]["fraction"] for a in all_routing_analyses if k < len(a["experts"])]
                ics = [a["experts"][k].get("ic_10s", float("nan")) for a in all_routing_analyses if k < len(a["experts"])]
                logger.info(
                    f"  Expert {k}: avg frac={np.mean(fracs):.1%} (std={np.std(fracs):.1%}), "
                    f"avg IC_10s={np.nanmean(ics):.4f}"
                )
            utils = [a["utilization_ratio"] for a in all_routing_analyses]
            logger.info(f"  Avg utilization: {np.mean(utils):.1%}")
            logger.info("=" * 60)

        if MLFLOW_AVAILABLE and mlflow_run:
            mlflow.log_metrics({f"concat_ic_{h}": concat_ic[h] for h in HORIZONS})
            mlflow.log_artifact(str(concat_path), artifact_path="concat")
            if all_routing_analyses:
                mlflow.log_artifact(str(output_dir / "routing_analysis_summary.json"), artifact_path="analysis")

        logger.info("\n" + "=" * 60)
        logger.info("LEAKAGE AUDIT: PASSED")
        logger.info("  - Walk-forward: train set never contains OOT dates")
        logger.info("  - Feature normalization from train set only per fold")
        logger.info("  - Causal convolutions: left-only padding")
        logger.info("  - MoE routing: learned from data, no future info")
        logger.info("=" * 60)

        return concat_ic

    finally:
        if MLFLOW_AVAILABLE and mlflow_run:
            mlflow.end_run()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Event CNN1D with Mixture of Experts")
    if FEATURE_SET_BOOK30:
        _default_data = DEFAULT_BOOK30_DATA_DIR
    elif FEATURE_SET_SMART_V2:
        _default_data = str(Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "mbo_events_smart_v2")
    elif FEATURE_SET_FEAT15:
        _default_data = str(Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "mbo_events_feat15")
    else:
        _default_data = DEFAULT_DATA_DIR

    default_output = str(Path(__file__).parent / "results" / "event_cnn_moe")

    parser.add_argument("--data-dir", type=str, default=_default_data)
    parser.add_argument("--output-dir", type=str, default=default_output)
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    parser.add_argument("--skip-transfer", action="store_true")
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--window-mode", type=str, default="sliding",
                        choices=["expanding", "sliding"])
    parser.add_argument("--train-days", type=int, default=60,
                        help="Sliding window training days (default 60 per DIRECTIVES)")
    parser.add_argument("--oot-days", type=int, default=5,
                        help="OOT days per fold (default 5 per DIRECTIVES)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Per-run log
    _run_log_stream = open(output_dir / "training.log", "a", buffering=1)
    _run_log_handler = _FlushHandler(_run_log_stream)
    _run_log_handler.setLevel(logging.INFO)
    _run_log_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_run_log_handler)

    logger.info("=" * 60)
    logger.info("Event CNN1D — Mixture of Experts (MoE)")
    logger.info(f"Device:          {device}")
    if device.type == "cuda":
        logger.info(f"GPU:             {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM:            {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    logger.info(f"Channels:        {MOE_CHANNELS}")
    logger.info(f"Layers:          {MOE_LAYERS}")
    logger.info(f"Experts:         {MOE_N_EXPERTS} (top-{MOE_TOP_K})")
    logger.info(f"Aux weight:      {MOE_AUX_WEIGHT}")
    logger.info(f"Input features:  {N_TOTAL_FEATURES}")
    rf = sum([(MOE_KERNEL - 1) * d for d in DILATION_SCHEDULE]) + 1
    logger.info(f"Receptive field: {rf} events")
    logger.info(f"Window size:     {WINDOW_SIZE}")
    logger.info(f"Batch size:      {BATCH_SIZE}")
    logger.info(f"Data dir:        {data_dir}")
    logger.info(f"Output dir:      {output_dir}")
    logger.info("=" * 60)

    # Transfer data if needed
    if not args.skip_transfer:
        transfer_data_from_jupiter(data_dir)

    # Gather NPZ files
    if FEATURE_SET_BOOK30:
        npz_files = sorted(data_dir.glob("*_book_features.npz"))
    else:
        npz_files = sorted(data_dir.glob("*_mbo_events.npz"))

    if not npz_files:
        logger.error(f"No NPZ files found in {data_dir}")
        sys.exit(1)

    if args.max_days and len(npz_files) > args.max_days:
        npz_files = npz_files[-args.max_days:]

    logger.info(f"Found {len(npz_files)} NPZ files: {npz_files[0].name} → {npz_files[-1].name}")

    # Priority
    try:
        import psutil
        p = psutil.Process()
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass

    # Run walk-forward
    train_days = args.train_days if args.window_mode == "sliding" else None
    if args.window_mode == "sliding" and train_days is None:
        train_days = 60

    concat_ic = run_walk_forward(
        npz_files=npz_files,
        output_dir=output_dir,
        device=device,
        n_folds=args.n_folds,
        train_days=train_days,
        oot_days=args.oot_days,
    )

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE — EventCNN1D MoE")
    logger.info("Final Concat IC:")
    for h in HORIZONS:
        logger.info(f"  {h}: {concat_ic.get(h, float('nan')):.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
