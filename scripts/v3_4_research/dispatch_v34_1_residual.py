"""
dispatch_v34_1_residual.py — v3.4.1 dispatch (GATED RESIDUAL BOOK-CNN)

Spec per user (2026-05-15, recovery #81):
  "v3.4 = v3.3 same as the v3.3 just with an added CNN for interpreting book.
   Book CNN added."

Architecture (literal interpretation, NO MTL changes, NO trunk widening):
  v3.4.1 model = v3.3 model (CNNMambaV32) UNCHANGED
               + Book2DCNN(out_dim=TRUNK_DIM)
               + scalar gate (init=0)
  forward:
    trunk_out = v3.3_forward_to_trunk_out(events_t1, events_t2, events_t3)  # IDENTICAL to v3.3
    book_emb  = book_cnn(book_pyramid)                                       # (B, TRUNK_DIM)
    fused_trunk_out = trunk_out + tanh(book_gate) * book_emb                # gate init=0 ⇒ identity
    return heads(fused_trunk_out)

Why this fixes the v3.4 failure modes:
  1. tanh(book_gate=0)=0 ⇒ at init, v3.4.1 IS v3.3 exactly. Heads see same distribution.
  2. v3.3 warmstart loads with 100% match into v32_core (no partial-Linear copy hacks).
  3. log_sigma MTL params have NO new untrained capacity to balance ⇒ no σ-collapse.
     The book pathway just adds a small residual that the optimiser can grow if it
     helps any of the 32 heads — gradient flow is gated naturally.
  4. Parameter delta vs v3.3: ~50K (book CNN) + 1 (gate). Tiny.

Loss: JointMultiHeadLossV33_UncertaintyWeighted (UNCHANGED from v3.3).
Training loop: train_one_fold_v33 (UNCHANGED from v3.3, just our new model).

Usage (Neptune):
  PYTHONPATH=/home/nick/Lvl3Quant V32_BATCH_SIZE=16 V32_WF_TRAIN_DAYS=30 \\
    /home/nick/miniconda3/envs/py311-train/bin/python -u -X faulthandler \\
    scripts/v3_4_research/dispatch_v34_1_residual.py --device cuda --n-folds 1
"""
import argparse
import json
import os as _os
_os.environ.setdefault("MAMBA_FEATURE_SET", "smart_v3")
_os.environ.setdefault("EVENT_WINDOW_SIZE", "1500")
_os.environ.setdefault("EVENT_STRIDE", "250")
_os.environ.setdefault("MLFLOW_TRACKING_URI", "http://jupiter:5000")

import os
import socket
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Path bootstrap
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# v3.3 trainer (loss + per-fold training loop)
from alpha_discovery.deep_models.train_cnn_mamba_v3_3 import (  # noqa: E402
    JointMultiHeadLossV33_UncertaintyWeighted,
    train_one_fold_v33,
    LOG_SIGMA_INIT,
)
# v3.2 (model + dataset)
from alpha_discovery.deep_models.train_cnn_mamba_v3_2 import (  # noqa: E402
    CNNMambaV32,
    SmartV32Dataset,
    ALL_HEAD_NAMES,
    build_weekly_fold_schedule,
    date_from_path,
    evaluate_v32,
    WINDOW_SIZE_T1, WINDOW_SIZE_T2, WINDOW_SIZE_T3,
    STRIDE, BATCH_SIZE, EPOCHS_PER_FOLD, WF_TRAIN_DAYS,
    TRUNK_DIM, MAMBA_D_MODEL,
    DEFAULT_DATA_DIR, DEFAULT_FIFO_LABEL_DIR, DEFAULT_ALPHA_LABEL_DIR,
    DEFAULT_PT_PRED_DIR, DEFAULT_TIER2_PARQUET_ROOT, DEFAULT_TIER3_PARQUET_ROOT,
)
# v3.4 (book dataset wrapper + collate — reuse, they work)
from alpha_discovery.deep_models.train_cnn_mamba_v3_4 import (  # noqa: E402
    Book2DCNN,
    SmartV34DualTrunkDataset,
    collate_v34,
    DEFAULT_BOOK_FEATURES_DIR,
    N_BOOK_LEVELS,
    N_BOOK_FEATURES_PER_LEVEL,
)

try:
    import mlflow  # noqa: E402
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


PROJECT_ROOT = Path(__file__).resolve().parents[2]
V33_WARMSTART_DEFAULT = str(
    PROJECT_ROOT / "output" / "cnn_mamba_v3_3_uncertainty_weighted" / "fold_00_intra_ckpt.pt"
)
DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / "output" / "cnn_mamba_v3_4_1_residual")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://jupiter:5000")
MLFLOW_EXPERIMENT = "CNNMamba_v3_4_1_book_residual"


# ============================================================
# v3.4.1 model: v3.3 trunk EXACTLY + gated residual book CNN
# ============================================================
class CNNMambaV341BookResidual(nn.Module):
    """
    Wraps an UNCHANGED CNNMambaV32 (the v3.3 architecture).
    Adds a Book2DCNN whose output is added to v3.3's trunk_out via a learnable
    scalar gate initialised to zero. At init the model is EXACTLY v3.3.
    """
    HEAD_NAMES = ALL_HEAD_NAMES

    def __init__(self, book_emb_dim: int = TRUNK_DIM, dropout: float = 0.1):
        super().__init__()
        # FULL v3.3 model — branches, trunk, heads — all untouched
        self.v32_core = CNNMambaV32()
        # Book pathway
        self.book_cnn = Book2DCNN(
            in_features=N_BOOK_FEATURES_PER_LEVEL,
            in_levels=N_BOOK_LEVELS,
            out_dim=book_emb_dim,
            dropout=dropout,
        )
        # Scalar gate: tanh(0)=0 ⇒ book residual contributes ZERO at init
        # ⇒ model output is IDENTICAL to v3.3 at init ⇒ heads + loss σ stay calibrated
        self.book_gate = nn.Parameter(torch.zeros(1))

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        c = self.v32_core
        # Mirror CNNMambaV32.forward up to trunk_out (do NOT call c.forward — it
        # would apply heads early)
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

        # Book residual — init contribution is ZERO (tanh(0)=0)
        book_emb = self.book_cnn(batch["book_pyramid"])
        gate = torch.tanh(self.book_gate)
        trunk_out = trunk_out + gate * book_emb

        return {name: head(trunk_out).squeeze(-1) for name, head in c.heads.items()}

    def load_v33_warmstart(self, ckpt_path: str, device: torch.device) -> Dict[str, int]:
        """
        Load v3.3 fold_00_intra_ckpt.pt — every v3.3 weight matches a v32_core.* key here.
        book_cnn + book_gate stay at init.
        Returns {loaded, skipped, total_v33}.
        """
        stats = {"loaded": 0, "skipped": 0, "total_v33": 0}
        if not Path(ckpt_path).exists():
            print(f">>> v3.4.1 WARN: v3.3 ckpt not found at {ckpt_path} — fully random init",
                  flush=True)
            return stats
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        except Exception as e:
            print(f">>> v3.4.1 WARN: torch.load failed: {e} — random init", flush=True)
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
        # Confirm book pathway stays at init
        bp_params = sum(p.numel() for p in self.book_cnn.parameters()) + self.book_gate.numel()
        print(f">>> v3.4.1 warmstart: loaded={stats['loaded']}/{stats['total_v33']} v3.3 tensors, "
              f"skipped={stats['skipped']}, book_pathway_init_params={bp_params}", flush=True)
        return stats


# ============================================================
# Dataset adapter — SmartV34DualTrunkDataset works as-is; we only need to ensure
# events dict keys match what CNNMambaV341BookResidual.forward expects:
#   events_t1, events_t2, events_t3, book_pyramid  ✓ (matches v3.4 dataset output)
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
    p = argparse.ArgumentParser(description="v3.4.1 gated-residual book-CNN dispatch")
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
        print(">>> v3.4.1: CUDA unavailable — CPU fallback", flush=True)
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    use_amp = (device.type == "cuda") and not args.no_amp

    print(f">>> v3.4.1 device={device} amp={use_amp} output_dir={output_dir}", flush=True)
    print(f">>> v3.4.1 warmstart={args.warmstart_ckpt}", flush=True)

    # Smoke test for sanity (CPU-fast)
    if args.smoke_test:
        print(">>> v3.4.1 SMOKE TEST", flush=True)
        m = CNNMambaV341BookResidual().to(device)
        m.load_v33_warmstart(args.warmstart_ckpt, device)
        n_params = sum(p.numel() for p in m.parameters())
        print(f">>> v3.4.1 params: {n_params/1e6:.3f}M", flush=True)
        B = 2
        dummy = {
            "events_t1": torch.randn(B, WINDOW_SIZE_T1, 39, device=device),
            "events_t2": torch.randn(B, WINDOW_SIZE_T2, 14, device=device),
            "events_t3": torch.randn(B, WINDOW_SIZE_T3, 25, device=device),
            "book_pyramid": torch.randn(B, WINDOW_SIZE_T2, N_BOOK_LEVELS,
                                        N_BOOK_FEATURES_PER_LEVEL, device=device),
        }
        out = m(dummy)
        print(f">>> v3.4.1 SMOKE OK: n_heads={len(out)} sample={list(out.keys())[:3]}", flush=True)
        # Confirm gate is exactly 0 ⇒ book contributes nothing
        with torch.no_grad():
            m.book_gate.fill_(0.0)
            out_zero = m(dummy)
            print(f">>> v3.4.1 gate=0 sanity: log_ret_1s={out_zero['log_ret_1s'].mean().item():.6f}",
                  flush=True)
        return 0

    # Discover aligned event+book dates
    data_dir = Path(args.data_dir)
    book_dir = Path(args.book_features_dir)
    aligned = discover_aligned_dates(data_dir, book_dir)
    if not aligned:
        print(">>> v3.4.1 ERROR: no aligned dates", flush=True)
        return 1
    print(f">>> v3.4.1 aligned dates: {len(aligned)} ({aligned[0]} → {aligned[-1]})", flush=True)

    train_days_env = int(os.environ.get("V32_WF_TRAIN_DAYS", WF_TRAIN_DAYS))
    folds = build_weekly_fold_schedule(aligned, n_folds=args.n_folds, train_days=train_days_env)
    print(f">>> v3.4.1 built {len(folds)} fold(s) | train_days={train_days_env}", flush=True)
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
                run_name=f"v3.4.1_residual_{time.strftime('%Y%m%d_%H%M')}_{socket.gethostname()}",
                tags={
                    "model_family": "cnn_mamba_v3.4.1_book_residual",
                    "version": "3.4.1",
                    "arch": "v3.3+gated_book_residual",
                    "loss_kind": "uncertainty_weighted_kendall2018",
                    "init_strategy": "v33_full_warmstart_gate_zero",
                },
            )
            mlflow.log_params({
                "model": "CNNMambaV341BookResidual",
                "trainer": "train_one_fold_v33_unchanged",
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
            })
            print(f">>> v3.4.1 MLflow run: {mlflow_run.info.run_id}", flush=True)
        except Exception as e:
            print(f">>> v3.4.1 MLflow init failed: {e}", flush=True)
            mlflow_run = None

    bs = int(os.environ.get("V32_BATCH_SIZE", BATCH_SIZE))
    num_workers = int(os.environ.get("V32_NUM_WORKERS", "0"))  # default 0 to avoid OOM-by-fork

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
            print(f">>> v3.4.1 Model parameters: {n_params:,} "
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
            print(f">>> v3.4.1 fold {fold_idx} batches: train={len(train_loader)} "
                  f"oot={len(oot_loader)} total_steps={total_steps} bs={bs} num_workers={num_workers}",
                  flush=True)

            # Delegate to v3.3's training loop UNCHANGED. It will:
            #   - Build a fresh JointMultiHeadLossV33_UncertaintyWeighted (log_sigma_init=0)
            #   - Run EPOCHS_PER_FOLD epochs
            #   - Save fold_{N:02d}_intra_ckpt.pt every 500 batches
            #   - Save fold_{N:02d}_best.pt on each OOT-loss improvement
            #   - Log everything to MLflow
            result = train_one_fold_v33(
                model, train_loader, oot_loader,
                fold_idx, output_dir, mlflow_run, device,
                total_train_steps=total_steps, use_amp=use_amp,
                resume_state=None,
            )

            print(f">>> v3.4.1 fold {fold_idx} done | "
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

    print(">>> v3.4.1 DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
