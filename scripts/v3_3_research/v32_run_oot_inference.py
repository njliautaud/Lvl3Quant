"""
v3.2 OOT Inference — read-only inference dispatcher that reuses the existing
trainer module as a library. NO modifications to train_cnn_mamba_v3_2.py.

Loads a checkpoint (default: fold_00_intra_ckpt.pt), builds an OOT dataloader
on the OOT dates from fold_schedule.json, runs forward pass on the GPU, dumps
a comprehensive predictions.npz suitable for downstream confidence-band,
DA, price-path, MFE/MAE, rolling-avg-exit-confluence, and multi-head agreement
analysis.

Designed to share GPU with the active training run safely:
  - torch.cuda.set_per_process_memory_fraction(0.10) keeps us under ~2.4GB
    on a 24GB 3090 (training uses ~12-13GB)
  - batch_size=1 to minimize transient VRAM spikes
  - torch.cuda.empty_cache() after every batch
  - eval() mode + no_grad

Usage:
    python v32_run_oot_inference.py \
        --ckpt /home/nick/Lvl3Quant/output/cnn_mamba_v3_2_long_context/fold_00_intra_ckpt.pt \
        --fold-schedule /home/nick/Lvl3Quant/output/cnn_mamba_v3_2_long_context/fold_schedule.json \
        --output /home/nick/Lvl3Quant/output/v3_2_deep_sim_20260512/fold_00_oot_predictions.npz \
        --feature-stats /home/nick/Lvl3Quant/output/cnn_mamba_v3_2_long_context/fold_00_feature_stats.npz \
        --max-vram-frac 0.10 \
        --batch-size 1 \
        --dates 20260223 20260224 20260225 20260226 20260227
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Resolve project root + import trainer module as library (read-only)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from alpha_discovery.deep_models.train_cnn_mamba_v3_2 import (  # type: ignore  # noqa: E402
    CNNMambaV32,
    SmartV32Dataset,
    evaluate_v32,
    JointMultiHeadLossV32,
    LOSS_LAMBDA,
    ALL_HEAD_NAMES,
    DEFAULT_TIER2_PARQUET_ROOT,
    DEFAULT_TIER3_PARQUET_ROOT,
)


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
    p.add_argument("--tier2-root", default=DEFAULT_TIER2_PARQUET_ROOT, type=str)
    p.add_argument("--tier3-root", default=DEFAULT_TIER3_PARQUET_ROOT, type=str)
    p.add_argument("--dates", nargs="*", default=None,
                   help="Override OOT dates (YYYYMMDD). If None, read from fold-schedule.")
    p.add_argument("--batch-size", default=1, type=int)
    p.add_argument("--num-workers", default=0, type=int,
                   help="Workers for DataLoader. 0 = main process only (safest for shared GPU).")
    p.add_argument("--max-vram-frac", default=0.10, type=float,
                   help="Cap process VRAM fraction. 0.10 on 24GB 3090 = 2.4GB.")
    p.add_argument("--device", default="cuda", type=str)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    output_dir = args.output.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve OOT dates
    with open(args.fold_schedule) as f:
        schedule = json.load(f)
    fold_info = next((x for x in schedule if x.get("fold") == args.fold_idx), schedule[0])
    oot_dates = args.dates or fold_info["oot_dates"]
    print(f"[v32_oot_inference] fold {args.fold_idx} oot_dates={oot_dates}", flush=True)

    # Feature stats (mandatory for OOT — must match training-time normalization)
    raw_stats = np.load(args.feature_stats, allow_pickle=True)
    feature_stats = {k: raw_stats[k] for k in raw_stats.files}
    print(f"[v32_oot_inference] loaded feature_stats keys={list(feature_stats.keys())}", flush=True)

    # Cap VRAM
    device = torch.device(args.device if ":" in args.device else f"{args.device}:0" if args.device == "cuda" else args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.max_vram_frac, device=device.index or 0)
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info(device.index or 0)
        print(f"[v32_oot_inference] VRAM cap={args.max_vram_frac:.2f}, free={free/1e9:.2f}GB total={total/1e9:.2f}GB", flush=True)

    # Build model + load ckpt
    model = CNNMambaV32().to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[v32_oot_inference] ckpt loaded. missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if missing:
        print(f"  missing[:5]={missing[:5]}", flush=True)
    model.eval()

    # Dataset + loader
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
    print(f"[v32_oot_inference] dataset built. n_samples={len(ds)}", flush=True)

    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=False,
    )

    # Run inference via existing evaluate_v32 (read-only call)
    loss_fn = JointMultiHeadLossV32(LOSS_LAMBDA)
    t0 = time.time()
    metrics, preds, targets, masks = evaluate_v32(
        model, loader, loss_fn, device, use_amp=(device.type == "cuda"),
    )
    elapsed = time.time() - t0
    print(f"[v32_oot_inference] eval done in {elapsed:.1f}s ({len(ds)/max(elapsed,1):.1f} samples/sec)", flush=True)
    print(f"[v32_oot_inference] metrics keys={list(metrics.keys())}", flush=True)

    # Dump predictions.npz with full coverage
    save_dict = {
        "fold_idx": np.array(args.fold_idx),
        "oot_dates": np.array(list(oot_dates)),
        "n_samples": np.array(len(ds)),
        "elapsed_sec": np.array(elapsed),
    }
    for h in ALL_HEAD_NAMES:
        if h in preds:
            save_dict[f"pred_{h}"] = preds[h]
            save_dict[f"target_{h}"] = targets[h]
            save_dict[f"mask_{h}"] = masks[h]

    np.savez_compressed(args.output, **save_dict)
    print(f"[v32_oot_inference] wrote {args.output} ({args.output.stat().st_size/1e6:.1f} MB)", flush=True)

    # Write metrics.json sidecar
    metrics_json = {k: (float(v) if isinstance(v, (int, float, np.floating)) and not np.isnan(v) else None)
                    for k, v in metrics.items()}
    metrics_json["_meta"] = {
        "ckpt": str(args.ckpt),
        "fold_idx": args.fold_idx,
        "oot_dates": list(oot_dates),
        "n_samples": len(ds),
        "elapsed_sec": elapsed,
        "vram_cap_frac": args.max_vram_frac,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
    }
    with open(args.output.with_suffix(".metrics.json"), "w") as f:
        json.dump(metrics_json, f, indent=2)

    print(f"[v32_oot_inference] DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
