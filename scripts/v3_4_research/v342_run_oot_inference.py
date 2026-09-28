"""
v3.4.2 OOT Inference — standalone read-only inference dispatcher.

Loads fold_00_intra_ckpt.pt and runs OOT eval on the fold-0 OOT dates, producing
an NPZ file with the EXACT schema written by the HC #410 save block in
dispatch_v34_2_fixedmtl.py (lines ~460-485).

Mirrors v32_run_oot_inference.py structure but for the v3.4.2 dual-trunk model.

The model class CNNMambaV341BookResidual is defined INLINE inside
scripts/v3_4_research/dispatch_v34_2_fixedmtl.py (no __init__.py in that
directory), so we import that module by file path via importlib.

Usage (Neptune):
  cd /home/nick/Lvl3Quant && V32_BATCH_SIZE=16 V32_WF_TRAIN_DAYS=10 \
    PYTHONPATH=/home/nick/Lvl3Quant \
    /home/nick/miniconda3/envs/py311-train/bin/python -u -X faulthandler \
    scripts/v3_4_research/v342_run_oot_inference.py \
    --device cuda \
    --ckpt output/cnn_mamba_v3_4_2_fixedmtl/fold_00_intra_ckpt.pt \
    --output output/cnn_mamba_v3_4_2_fixedmtl/fold_00_ep1_oot_inference.npz
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os as _os
import sys
import time
from pathlib import Path

# Mirror dispatch env defaults BEFORE importing trainer modules
_os.environ.setdefault("MAMBA_FEATURE_SET", "smart_v3")
_os.environ.setdefault("EVENT_WINDOW_SIZE", "1500")
_os.environ.setdefault("EVENT_STRIDE", "250")

import numpy as np
import torch
from torch.utils.data import DataLoader

# Resolve project root + path bootstrap
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# v3.2 (model + dataset + helpers)
from alpha_discovery.deep_models.train_cnn_mamba_v3_2 import (  # noqa: E402
    SmartV32Dataset,
    evaluate_v32,
    WINDOW_SIZE_T1, WINDOW_SIZE_T2, WINDOW_SIZE_T3,
    STRIDE,
    DEFAULT_DATA_DIR, DEFAULT_FIFO_LABEL_DIR, DEFAULT_ALPHA_LABEL_DIR,
    DEFAULT_PT_PRED_DIR, DEFAULT_TIER2_PARQUET_ROOT, DEFAULT_TIER3_PARQUET_ROOT,
)
# v3.4 (book dataset wrapper + collate)
from alpha_discovery.deep_models.train_cnn_mamba_v3_4 import (  # noqa: E402
    SmartV34DualTrunkDataset,
    collate_v34,
    DEFAULT_BOOK_FEATURES_DIR,
)


def _import_dispatch_module():
    """Import dispatch_v34_2_fixedmtl.py by file path (no __init__.py)."""
    dispatch_path = PROJECT_ROOT / "scripts" / "v3_4_research" / "dispatch_v34_2_fixedmtl.py"
    spec = importlib.util.spec_from_file_location(
        "dispatch_v34_2_fixedmtl", str(dispatch_path)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dispatch_v34_2_fixedmtl"] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="v3.4.2 OOT standalone inference")
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--batch-size", default=16, type=int)
    p.add_argument("--num-workers", default=0, type=int)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--feature-stats", default=None, type=Path,
                   help="Path to feature_stats.npz (default: alongside ckpt)")
    p.add_argument("--fold-schedule", default=None, type=Path,
                   help="Path to fold_schedule.json (default: alongside ckpt)")
    p.add_argument("--fold-idx", default=0, type=int)
    p.add_argument("--dates", nargs="*", default=None,
                   help="Override OOT dates (YYYYMMDD)")
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--book-features-dir", default=DEFAULT_BOOK_FEATURES_DIR)
    p.add_argument("--fifo-label-dir", default=str(DEFAULT_FIFO_LABEL_DIR))
    p.add_argument("--alpha-label-dir", default=str(DEFAULT_ALPHA_LABEL_DIR))
    p.add_argument("--pt-pred-dir", default=str(DEFAULT_PT_PRED_DIR))
    p.add_argument("--tier2-parquet-root", default=str(DEFAULT_TIER2_PARQUET_ROOT))
    p.add_argument("--tier3-parquet-root", default=str(DEFAULT_TIER3_PARQUET_ROOT))
    p.add_argument("--no-amp", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    output_dir = args.output.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = args.ckpt.parent
    feature_stats_path = args.feature_stats or (ckpt_dir / "fold_00_feature_stats.npz")
    fold_schedule_path = args.fold_schedule or (ckpt_dir / "fold_schedule.json")

    # Resolve device
    if args.device == "cuda" and not torch.cuda.is_available():
        print(">>> [v342_oot_inf] CUDA unavailable — CPU fallback", flush=True)
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    use_amp = (device.type == "cuda") and not args.no_amp
    print(f">>> [v342_oot_inf] device={device} amp={use_amp}", flush=True)

    # Import dispatch module to grab the model class + loss
    dispatch_mod = _import_dispatch_module()
    CNNMambaV341BookResidual = dispatch_mod.CNNMambaV341BookResidual
    FixedWeightMultiHeadLoss = dispatch_mod.FixedWeightMultiHeadLoss
    print(">>> [v342_oot_inf] dispatch module imported (model + loss)", flush=True)

    # Resolve OOT dates
    with open(fold_schedule_path) as f:
        schedule = json.load(f)
    fold_info = next((x for x in schedule if x.get("fold") == args.fold_idx), schedule[0])
    oot_dates = args.dates or fold_info["oot_dates"]
    print(f">>> [v342_oot_inf] fold {args.fold_idx} oot_dates={oot_dates}", flush=True)

    if args.smoke_test:
        # Limit to first 1 OOT date for smoke
        oot_dates = list(oot_dates)[:1]
        print(f">>> [v342_oot_inf] SMOKE TEST — limiting OOT to {oot_dates}", flush=True)

    # Feature stats (mandatory — must match training-time normalization)
    raw_stats = np.load(feature_stats_path, allow_pickle=True)
    feature_stats = {k: raw_stats[k] for k in raw_stats.files}
    print(f">>> [v342_oot_inf] loaded feature_stats keys={list(feature_stats.keys())} "
          f"shapes={ {k: feature_stats[k].shape for k in feature_stats} }", flush=True)

    # Build inner dataset (event-level, v3.2 style)
    oot_inner = SmartV32Dataset(
        data_dir=Path(args.data_dir),
        fifo_label_dir=Path(args.fifo_label_dir),
        alpha_label_dir=Path(args.alpha_label_dir),
        pt_pred_dir=Path(args.pt_pred_dir),
        tier2_parquet_root=Path(args.tier2_parquet_root),
        tier3_parquet_root=Path(args.tier3_parquet_root),
        dates=list(oot_dates),
        window_t1=WINDOW_SIZE_T1, window_t2=WINDOW_SIZE_T2, window_t3=WINDOW_SIZE_T3,
        stride=STRIDE, feature_stats=feature_stats,
        cache_size=1, require_alpha_labels=False,
    )
    # Wrap in dual-trunk dataset to add book pyramid
    oot_ds = SmartV34DualTrunkDataset(oot_inner, book_features_dir=str(args.book_features_dir))
    print(f">>> [v342_oot_inf] dataset built. n_samples={len(oot_ds)}", flush=True)

    # Loader mirrors dispatch's OOT loader (batch_size * 2 in dispatch; we just use --batch-size)
    loader = DataLoader(
        oot_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_v34, persistent_workers=False,
        prefetch_factor=(2 if args.num_workers > 0 else None),
    )

    # Build model + load ckpt
    model = CNNMambaV341BookResidual().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f">>> [v342_oot_inf] model params: {n_params/1e6:.3f}M", flush=True)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt))
    try:
        missing, unexpected = model.load_state_dict(state_dict, strict=True)
        print(f">>> [v342_oot_inf] ckpt loaded STRICT. "
              f"missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    except RuntimeError as e:
        print(f">>> [v342_oot_inf] strict load failed: {e}", flush=True)
        print(">>> [v342_oot_inf] retrying with strict=False", flush=True)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f">>> [v342_oot_inf] non-strict load: "
              f"missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        if missing:
            print(f"   missing[:5]={missing[:5]}", flush=True)
        if unexpected:
            print(f"   unexpected[:5]={unexpected[:5]}", flush=True)
    print(f">>> [v342_oot_inf] ckpt meta: "
          f"fold={ckpt.get('fold')} epoch={ckpt.get('epoch')} "
          f"batch={ckpt.get('batch')} global_step={ckpt.get('global_step')} "
          f"best_val_loss={ckpt.get('best_val_loss')}",
          flush=True)
    if "book_gate" in state_dict:
        bg = float(state_dict["book_gate"].item() if hasattr(state_dict["book_gate"], "item")
                   else state_dict["book_gate"])
        print(f">>> [v342_oot_inf] ckpt book_gate raw={bg:.6f} tanh={np.tanh(bg):.6f}",
              flush=True)
    model.eval()

    # Build loss (no learnable params — only used for eval bookkeeping via evaluate_v32)
    loss_fn = FixedWeightMultiHeadLoss().to(device)

    # Smoke-test: cap loader iterations
    if args.smoke_test:
        original_loader = loader

        class _LimitedLoader:
            def __init__(self, base, n_max):
                self.base = base
                self.n_max = n_max
                # evaluate_v32 may call len() or iter()
            def __iter__(self):
                for i, b in enumerate(self.base):
                    if i >= self.n_max:
                        break
                    yield b
            def __len__(self):
                return min(self.n_max, len(self.base))

        loader = _LimitedLoader(original_loader, n_max=10)
        print(">>> [v342_oot_inf] SMOKE TEST — capped to 10 batches", flush=True)

    # Run inference via existing evaluate_v32 (read-only)
    t0 = time.time()
    val_metrics, oot_preds, oot_targets, oot_masks = evaluate_v32(
        model, loader, loss_fn, device, use_amp=use_amp,
    )
    elapsed = time.time() - t0
    n_eff = sum(v.size for v in oot_preds.values()) // max(len(oot_preds), 1)
    print(f">>> [v342_oot_inf] eval done in {elapsed:.1f}s "
          f"({n_eff} samples/head, {n_eff/max(elapsed,1):.1f} samples/sec)", flush=True)
    print(f">>> [v342_oot_inf] metrics keys={sorted(val_metrics.keys())[:20]}...", flush=True)

    # Print key IC values for sanity
    for h in ("log_ret_1s", "log_ret_5s", "log_ret_10s", "log_ret_30s"):
        ic_key = f"ic_{h}"
        if ic_key in val_metrics:
            v = val_metrics[ic_key]
            print(f">>> [v342_oot_inf] IC {h:12s} = {float(v):.6f}", flush=True)

    # Save NPZ with EXACT HC #410 schema
    save_dict = {"metrics_loss": float(val_metrics["loss"])}
    for k, v in oot_preds.items():
        save_dict[f"pred_{k}"] = v
    for k, v in oot_targets.items():
        save_dict[f"target_{k}"] = v
    for k, v in oot_masks.items():
        save_dict[f"mask_{k}"] = v
    for k, v in val_metrics.items():
        if k != "loss" and isinstance(v, (int, float, np.floating)):
            vf = float(v)
            if not (isinstance(vf, float) and np.isnan(vf)):
                save_dict[f"metric_{k}"] = vf

    # HC #432: save oot_dates + per-sample date mapping for 47-day stratified validation
    save_dict["oot_dates"] = np.array([str(d) for d in oot_dates])
    try:
        per_sample_dates = np.array([s[0] for s in oot_inner.sample_index], dtype="U8")
        save_dict["sample_dates"] = per_sample_dates
        print(f">>> [v342_oot_inf] HC#432: saved oot_dates ({len(oot_dates)}) + sample_dates ({len(per_sample_dates)})", flush=True)
    except Exception as _e:
        print(f">>> [v342_oot_inf] HC#432 WARN: could not derive per-sample dates: {_e}", flush=True)
    np.savez_compressed(args.output, **save_dict)
    sz_mb = args.output.stat().st_size / 1e6
    print(f">>> [v342_oot_inf] wrote {args.output} ({sz_mb:.1f} MB) "
          f"keys={len(save_dict)}", flush=True)
    # Show a few keys
    key_sample = sorted(save_dict.keys())[:25]
    print(f">>> [v342_oot_inf] sample keys: {key_sample}", flush=True)

    print(">>> [v342_oot_inf] DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
