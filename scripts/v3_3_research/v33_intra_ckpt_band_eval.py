"""
v3.3 intra_ckpt forward-inference on OOT day 20260223.

Per HC #343 (script spec) + HC #348 (MagCorr-ticks).
Per HC #325: Neptune GPU is OFF-LIMITS. This runs on Jupiter CPU.
Per HC #307D malware-guard: NEW file under scripts/v3_3_research/. Imports
trainer modules as library — does NOT modify them.

v3.3 reuses the v3.2 model class (CNNMambaV32) and dataset (SmartV32Dataset)
verbatim — only the loss class is new. Inference therefore needs only:
  1. CNNMambaV32() with the v3.3 intra_ckpt model_state loaded.
  2. SmartV32Dataset on OOT date(s) with v3.3 fold_00_feature_stats.
  3. evaluate_v32() forward-pass on CPU.

Produces a predictions.npz keyed identically to the v3.2 OOT NPZ so that
v3_band_comparison_*.py can ingest both with the same load helper.

Usage:
    python v33_intra_ckpt_band_eval.py \
        --ckpt /tmp/v33_intra_ckpt.pt \
        --feature-stats /tmp/v33_feature_stats.npz \
        --fold-schedule /tmp/v33_fold_schedule.json \
        --output /home/jupiter/Lvl3Quant/output/v3_3_oot_20260223/predictions.npz \
        --dates 20260223 \
        --device cpu --batch-size 8 --num-workers 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import resource
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Prefer the Neptune trainer copy (rsync'd to /tmp/v33_trainer_readonly/) so we
# instantiate the SAME architecture that produced fold_00_intra_ckpt.pt. The
# Jupiter copy of train_cnn_mamba.py drifted (different CNN module layout), so
# importing the project-root version would yield a non-loadable model.
NEPTUNE_TRAINER_RO = Path("/tmp/v33_trainer_readonly")
sys.path.insert(0, str(PROJECT_ROOT))
if (NEPTUNE_TRAINER_RO / "alpha_discovery/deep_models/train_cnn_mamba_v3_2.py").exists():
    # Insert AFTER project-root so that this one is searched FIRST. (sys.path is
    # searched in order; insert(0, X) makes X first; doing it last makes it win.)
    sys.path.insert(0, str(NEPTUNE_TRAINER_RO))

# Match Neptune training defaults BEFORE any trainer import so module-level
# constants (MAMBA_D_MODEL, MAMBA_D_STATE, MAMBA_N_LAYERS, MAMBA_DT_RANK) match
# the architecture that produced fold_00_intra_ckpt.pt.
# Neptune trained with USE_CUDA_MAMBA=True → CUDAMamba auto-computes
# dt_rank = ceil(d_model/16). On Jupiter CPU we hit the SelectiveSSM fallback
# which honours module-level MAMBA_DT_RANK, so we set it to 8 here (d_model=128
# → dt_rank=8; T3's d_model=64 → dt_rank=4, fixed up below).
os.environ.setdefault("MAMBA_FEATURE_SET", "smart_v3")
os.environ.setdefault("MAMBA_D_MODEL", "128")
os.environ.setdefault("MAMBA_D_STATE", "64")
os.environ.setdefault("MAMBA_N_LAYERS", "4")
os.environ.setdefault("MAMBA_DT_RANK", "8")
os.environ.setdefault("MAMBA_D_CONV", "4")
os.environ.setdefault("CNN_CHANNELS", "64")
os.environ.setdefault("CNN_KERNEL", "5")
os.environ.setdefault("CNN_LAYERS", "3")
os.environ.setdefault("DISABLE_MLFLOW", "1")

# Library import only — no modification of trainer modules.
from alpha_discovery.deep_models.train_cnn_mamba import CNNMamba  # noqa: E402
from alpha_discovery.deep_models.train_cnn_mamba_v3_2 import (  # noqa: E402
    CNNMambaV32,
    SmartV32Dataset,
    evaluate_v32,
    JointMultiHeadLossV32,
    LOSS_LAMBDA,
    ALL_HEAD_NAMES,
    DEFAULT_TIER2_PARQUET_ROOT,
    DEFAULT_TIER3_PARQUET_ROOT,
    T3_D_MODEL,
    T3_N_LAYERS,
    CNN_CHANNELS,
    CNN_KERNEL,
    CNN_LAYERS,
    MAMBA_D_STATE,
    MAMBA_D_CONV,
    MAMBA_DROPOUT,
)


def rebuild_t3_backbone_with_smaller_dt_rank(model: "CNNMambaV32") -> None:
    """
    Neptune's CUDAMamba auto-sets dt_rank = ceil(d_model/16). For T3 with
    d_model=64 that means dt_rank=4. The pure-PyTorch SelectiveSSM fallback
    uses module-level MAMBA_DT_RANK uniformly (we set =8 to match T1/T2).
    We rebuild ONLY t3_backbone with explicit dt_rank=4 so the v3.3 ckpt
    loads cleanly. This is a runtime re-instantiation, not a code change.
    """
    # Mirror the v3.2 t3 ctor exactly (see train_cnn_mamba_v3_2.py:380-388)
    new_t3 = CNNMamba(
        d_model=T3_D_MODEL,
        d_state=MAMBA_D_STATE,
        n_layers=T3_N_LAYERS,
        dt_rank=4,                       # <-- d_model_t3 / 16
        d_conv=MAMBA_D_CONV,
        dropout=MAMBA_DROPOUT,
        n_targets=3,
        cnn_channels=max(CNN_CHANNELS // 2, 16),
        cnn_kernel=CNN_KERNEL,
        cnn_layers=max(CNN_LAYERS - 1, 1),
    )
    import torch.nn as nn
    new_t3.head = nn.Identity()
    model.t3_backbone = new_t3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--feature-stats", required=True, type=Path)
    p.add_argument("--fold-schedule", required=True, type=Path)
    p.add_argument("--fold-idx", default=0, type=int)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--data-dir", default=PROJECT_ROOT / "data/processed/mbo_events_smart_v3", type=Path)
    p.add_argument("--fifo-label-dir", default=PROJECT_ROOT / "data/processed/mbo_events_smart_v3_fifo_labels", type=Path)
    p.add_argument("--alpha-label-dir", default=PROJECT_ROOT / "data/processed/mbo_events_smart_v3_alpha_labels", type=Path)
    p.add_argument("--pt-pred-dir", default=PROJECT_ROOT / "data/processed/mbo_events_smart_v3_pt_pred", type=Path)
    p.add_argument("--tier2-root", default=str(PROJECT_ROOT / "data/derived/tier2_orderflow_features_v1.parquet"), type=str)
    p.add_argument("--tier3-root", default=str(PROJECT_ROOT / "data/derived/tier3_session_features_v1.parquet"), type=str)
    p.add_argument("--dates", nargs="*", default=None,
                   help="Override OOT dates (YYYYMMDD). If None, read from fold-schedule.")
    p.add_argument("--batch-size", default=8, type=int)
    p.add_argument("--num-workers", default=2, type=int)
    p.add_argument("--device", default="cpu", type=str)
    p.add_argument("--cpu-threads", default=8, type=int)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.device == "cpu":
        torch.set_num_threads(args.cpu_threads)
        print(f"[v33_oot] torch.set_num_threads({args.cpu_threads})", flush=True)

    output_dir = args.output.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve OOT dates
    with open(args.fold_schedule) as f:
        schedule = json.load(f)
    if isinstance(schedule, list):
        fold_info = next((x for x in schedule if x.get("fold") == args.fold_idx), schedule[0])
    else:
        fold_info = schedule
    oot_dates = args.dates if args.dates else fold_info["oot_dates"]
    print(f"[v33_oot] fold {args.fold_idx} oot_dates={oot_dates}", flush=True)

    # Feature stats
    raw_stats = np.load(args.feature_stats, allow_pickle=True)
    feature_stats = {k: raw_stats[k] for k in raw_stats.files}
    print(f"[v33_oot] feature_stats keys={list(feature_stats.keys())}", flush=True)

    device = torch.device(args.device)

    # Build model + load v3.3 ckpt (v3.3 reuses CNNMambaV32; only loss differs)
    model = CNNMambaV32()
    # T3 backbone has dt_rank=4 in the ckpt (CUDA auto-scaling). Rebuild it
    # in-place with explicit dt_rank=4 to match. T1/T2 already correct via env.
    rebuild_t3_backbone_with_smaller_dt_rank(model)
    model = model.to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[v33_oot] ckpt loaded. fold={ckpt.get('fold')} epoch={ckpt.get('epoch')} "
          f"batch={ckpt.get('batch')} step={ckpt.get('global_step')} "
          f"ckpt_version={ckpt.get('ckpt_version')}", flush=True)
    print(f"[v33_oot] state_dict load: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if missing:
        print(f"  missing[:5]={missing[:5]}", flush=True)
    if unexpected:
        print(f"  unexpected[:5]={unexpected[:5]}", flush=True)
    model.eval()

    # Dataset + loader
    t_ds = time.time()
    ds = SmartV32Dataset(
        data_dir=args.data_dir,
        fifo_label_dir=args.fifo_label_dir,
        alpha_label_dir=args.alpha_label_dir,
        pt_pred_dir=args.pt_pred_dir,
        dates=list(oot_dates),
        tier2_parquet_root=args.tier2_root,
        tier3_parquet_root=args.tier3_root,
        feature_stats=feature_stats,
        cache_size=1,
        require_alpha_labels=False,
    )
    print(f"[v33_oot] dataset built. n_samples={len(ds)} (in {time.time()-t_ds:.1f}s)", flush=True)

    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=False,
    )

    loss_fn = JointMultiHeadLossV32(LOSS_LAMBDA)
    t0 = time.time()
    metrics, preds, targets, masks = evaluate_v32(
        model, loader, loss_fn, device, use_amp=False,
    )
    elapsed = time.time() - t0
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"[v33_oot] eval done in {elapsed:.1f}s "
          f"({len(ds)/max(elapsed,1):.1f} samples/sec, peak RSS={peak_kb/1e6:.2f} GB)", flush=True)
    print(f"[v33_oot] metrics keys={list(metrics.keys())}", flush=True)
    for k in list(metrics.keys()):
        if k.startswith("ic_") or k.startswith("corr_"):
            print(f"  {k} = {metrics[k]:.4f}", flush=True)

    save_dict = {
        "fold_idx": np.array(args.fold_idx),
        "oot_dates": np.array(list(oot_dates)),
        "n_samples": np.array(len(ds)),
        "elapsed_sec": np.array(elapsed),
        "ckpt_epoch": np.array(ckpt.get("epoch", -1)),
        "ckpt_batch": np.array(ckpt.get("batch", -1)),
        "ckpt_global_step": np.array(ckpt.get("global_step", -1)),
    }
    for h in ALL_HEAD_NAMES:
        if h in preds:
            save_dict[f"pred_{h}"] = preds[h]
            save_dict[f"target_{h}"] = targets[h]
            save_dict[f"mask_{h}"] = masks[h]

    np.savez_compressed(args.output, **save_dict)
    print(f"[v33_oot] wrote {args.output} ({args.output.stat().st_size/1e6:.1f} MB)", flush=True)

    metrics_json = {}
    for k, v in metrics.items():
        try:
            fv = float(v)
            metrics_json[k] = fv if np.isfinite(fv) else None
        except Exception:
            metrics_json[k] = None
    metrics_json["_meta"] = {
        "ckpt": str(args.ckpt),
        "fold_idx": args.fold_idx,
        "oot_dates": list(oot_dates),
        "n_samples": len(ds),
        "elapsed_sec": elapsed,
        "device": str(device),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "ckpt_epoch": int(ckpt.get("epoch", -1)),
        "ckpt_batch": int(ckpt.get("batch", -1)),
        "ckpt_global_step": int(ckpt.get("global_step", -1)),
        "ckpt_version": int(ckpt.get("ckpt_version", -1)),
    }
    with open(args.output.with_suffix(".metrics.json"), "w") as f:
        json.dump(metrics_json, f, indent=2)

    print(f"[v33_oot] DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
