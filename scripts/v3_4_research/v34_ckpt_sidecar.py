"""
v34_ckpt_sidecar.py — HC #367 PART A satisfaction via NEW tooling (HC #307D compliant).

PROBLEM:
  HC #367 requires v3.4 ckpts every 500 batches with THREE artifacts:
    (1) model.pt  — full state_dict + opt + sched + arch + step
    (2) embeddings.npz  — last-layer pre-head embeddings on fixed OOT minibatch
    (3) predictions.npz — full 32-head predictions on fixed OOT minibatch

  The v3.4 trainer ALREADY writes (1) every 500 batches to
  `output/cnn_mamba_v3_4_dual_trunk/fold_NN_intra_ckpt.pt` (overwrites in-place).

  Per malware-guard + HC #307D: NO trainer modifications. Sidecar pattern instead.

SOLUTION (this script):
  Long-running daemon that:
    1. Polls fold_NN_intra_ckpt.pt mtime every POLL_SEC seconds.
    2. On mtime change → load ckpt (CPU), read (global_step, epoch, fold) from inside.
    3. Snapshot raw .pt → ckpt_batch_<global_step>/model.pt (versioned dir).
    4. Build model on chosen device, load state, register pre-head trunk hook.
    5. Run forward on a FIXED deterministic OOT minibatch (seed = fold_id, N=SAMPLES).
    6. Save embeddings.npz (N, trunk_dim) + predictions.npz (N, 32 heads) into ckpt dir.
    7. Append jsonl line to train_log_sidecar.jsonl.
    8. Rotate keep-last-N ckpt dirs (oldest deleted, best/intra_best/final retained).

Authorized: HC #367 PART A (user verbatim 2026-05-14 23:25 ET) + HC #366 (autonomous
lean-and-execute) + HC #307D (analysis tooling under scripts/v3_4_research/).

Usage (Neptune, runs alongside the active trainer):
  cd /home/nick/Lvl3Quant
  PYTHONPATH=/home/nick/Lvl3Quant nohup \\
    /home/nick/miniconda3/envs/py311-train/bin/python -u \\
    scripts/v3_4_research/v34_ckpt_sidecar.py \\
      --fold-idx 0 \\
      --device cpu \\
      --poll-sec 30 \\
      --samples 1024 \\
      --keep-last 20 \\
    > logs/v3_4/sidecar_fold0.log 2>&1 &

Run on CPU to avoid contention with trainer's CUDA usage. On Neptune (32GB RAM)
CPU inference on a 1024-sample OOT minibatch takes ~60-90s — well under the
500-batch trainer cadence (~5min at 1.5 batch/s).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# Path bootstrap so this script can be run from anywhere
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Re-use existing dataset + model classes (do NOT modify them)
from alpha_discovery.deep_models.train_cnn_mamba_v3_2 import (  # noqa: E402
    SmartV32Dataset,
    date_from_path,
    WINDOW_SIZE_T1,
    WINDOW_SIZE_T2,
    WINDOW_SIZE_T3,
    STRIDE,
    DEFAULT_DATA_DIR,
    DEFAULT_FIFO_LABEL_DIR,
    DEFAULT_ALPHA_LABEL_DIR,
    DEFAULT_PT_PRED_DIR,
    DEFAULT_TIER2_PARQUET_ROOT,
    DEFAULT_TIER3_PARQUET_ROOT,
)
from alpha_discovery.deep_models.train_cnn_mamba_v3_4 import (  # noqa: E402
    CNNMambaV34DualTrunk,
    SmartV34DualTrunkDataset,
    collate_v34,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_BOOK_FEATURES_DIR,
    ALL_HEAD_NAMES,
)


# ============================================================
# Helpers
# ============================================================
def _log(msg: str) -> None:
    print(f"[sidecar {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _discover_aligned_oot_dates(
    data_dir: Path,
    book_dir: Path,
    fold_schedule_path: Path,
    fold_idx: int,
) -> List[str]:
    """Read OOT dates from fold_schedule.json the dispatcher writes at launch."""
    with open(fold_schedule_path) as fh:
        folds = json.load(fh)
    for f in folds:
        if int(f["fold"]) == fold_idx:
            return list(f["oot_dates"])
    raise ValueError(f"fold_idx {fold_idx} not in {fold_schedule_path}")


def _build_oot_loader(
    data_dir: Path,
    book_dir: Path,
    fifo_label_dir: Path,
    alpha_label_dir: Path,
    pt_pred_dir: Path,
    tier2_parquet_root: Path,
    tier3_parquet_root: Path,
    feature_stats_path: Path,
    oot_dates: List[str],
    samples: int,
    batch_size: int,
    seed: int,
) -> DataLoader:
    """Build a deterministic fixed OOT minibatch loader (seed=fold_idx).

    Uses Subset over a seed-derived random index to ensure the SAME samples
    are evaluated across every ckpt snapshot — critical for downstream
    embedding-evolution / prediction-trajectory analysis.
    """
    feat = np.load(feature_stats_path)
    feature_stats = {"mean_t1": feat["mean_t1"], "std_t1": feat["std_t1"]}

    inner = SmartV32Dataset(
        data_dir=data_dir,
        fifo_label_dir=fifo_label_dir,
        alpha_label_dir=alpha_label_dir,
        pt_pred_dir=pt_pred_dir,
        tier2_parquet_root=tier2_parquet_root,
        tier3_parquet_root=tier3_parquet_root,
        dates=oot_dates,
        window_t1=WINDOW_SIZE_T1,
        window_t2=WINDOW_SIZE_T2,
        window_t3=WINDOW_SIZE_T3,
        stride=STRIDE,
        feature_stats=feature_stats,
        cache_size=1,
        require_alpha_labels=False,
    )
    ds = SmartV34DualTrunkDataset(inner, book_features_dir=str(book_dir))

    rng = np.random.default_rng(seed)
    total = len(ds)
    n = min(samples, total)
    idx = rng.choice(total, size=n, replace=False)
    idx.sort()  # sort for cache locality
    subset = torch.utils.data.Subset(ds, idx.tolist())
    return DataLoader(
        subset, batch_size=batch_size, shuffle=False, num_workers=0,
        pin_memory=False, collate_fn=collate_v34,
    )


def _read_ckpt_meta(ckpt_path: Path) -> Dict:
    """Load ckpt to CPU and return its metadata (global_step, epoch, fold)."""
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return {
        "global_step": int(blob.get("global_step", 0)),
        "epoch": int(blob.get("epoch", 0)),
        "batch": int(blob.get("batch", 0)),
        "fold": int(blob.get("fold", 0)),
        "best_val_loss": float(blob.get("best_val_loss", float("inf"))),
        "ckpt_version": int(blob.get("ckpt_version", 1)),
    }


def _load_model_from_ckpt(ckpt_path: Path, device: torch.device) -> CNNMambaV34DualTrunk:
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = CNNMambaV34DualTrunk().to(device)
    state = blob.get("model_state", blob)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        _log(f"  WARN: missing keys on load: {len(missing)} (first 3: {missing[:3]})")
    if unexpected:
        _log(f"  WARN: unexpected keys on load: {len(unexpected)} (first 3: {unexpected[:3]})")
    model.eval()
    return model


def _run_inference_with_embeddings(
    model: CNNMambaV34DualTrunk,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Run forward over loader, capturing pre-head trunk embeddings + 32-head preds.

    Uses a forward hook on model.trunk to capture the (B, trunk_dim) embedding
    BEFORE it hits the head linears. No model modification required.
    """
    captured: List[torch.Tensor] = []

    def _hook(_module, _inp, out):
        captured.append(out.detach().cpu())

    handle = model.trunk.register_forward_hook(_hook)
    all_preds: Dict[str, List[torch.Tensor]] = {h: [] for h in ALL_HEAD_NAMES}

    try:
        with torch.no_grad():
            for events, _targets, _masks in loader:
                events = {k: v.to(device, non_blocking=False) for k, v in events.items()}
                preds = model(events)
                for h in ALL_HEAD_NAMES:
                    all_preds[h].append(preds[h].detach().cpu())
    finally:
        handle.remove()

    emb = torch.cat(captured, dim=0).numpy().astype(np.float32)
    pred_dict = {h: torch.cat(v, dim=0).numpy().astype(np.float32) for h, v in all_preds.items()}
    return emb, pred_dict


def _rotate_ckpt_dirs(fold_dir: Path, keep_last: int) -> None:
    """Keep newest `keep_last` ckpt_batch_* dirs. Always retain
    fold_NN_best.pt, fold_NN_intra_best.pt, fold_NN_final.pt parents."""
    dirs = sorted(
        [p for p in fold_dir.glob("ckpt_batch_*") if p.is_dir()],
        key=lambda p: int(p.name.split("_")[-1]),
    )
    if len(dirs) <= keep_last:
        return
    to_delete = dirs[:-keep_last]
    for d in to_delete:
        try:
            shutil.rmtree(d)
            _log(f"  rotated (deleted) {d.name}")
        except Exception as e:
            _log(f"  WARN: rotation failed for {d.name}: {e}")


def _snapshot_ckpt(
    ckpt_path: Path,
    fold_dir: Path,
    loader: DataLoader,
    device: torch.device,
    keep_last: int,
    log_path: Path,
) -> Optional[int]:
    """One snapshot pass: load → version → infer → write artifacts → rotate.

    Returns the global_step on success, None on failure.
    """
    t0 = time.time()
    meta = _read_ckpt_meta(ckpt_path)
    gstep = meta["global_step"]
    ckpt_dir = fold_dir / f"ckpt_batch_{gstep:08d}"
    if ckpt_dir.exists() and (ckpt_dir / "model.pt").exists() \
            and (ckpt_dir / "embeddings.npz").exists() \
            and (ckpt_dir / "predictions.npz").exists():
        return None  # already snapshotted this gstep

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # (1) raw .pt snapshot
    shutil.copy2(ckpt_path, ckpt_dir / "model.pt")
    # (2) model load + inference
    model = _load_model_from_ckpt(ckpt_path, device)
    emb, preds = _run_inference_with_embeddings(model, loader, device)
    # (3) write artifacts
    np.savez_compressed(ckpt_dir / "embeddings.npz", embeddings=emb)
    np.savez_compressed(
        ckpt_dir / "predictions.npz",
        **preds,
    )
    # (4) jsonl log line
    log_line = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "fold": meta["fold"],
        "epoch": meta["epoch"],
        "batch": meta["batch"],
        "global_step": gstep,
        "ckpt_version": meta["ckpt_version"],
        "best_val_loss": meta["best_val_loss"],
        "n_samples_infered": int(emb.shape[0]),
        "trunk_dim": int(emb.shape[1]),
        "snapshot_sec": round(time.time() - t0, 2),
    }
    with open(log_path, "a") as fh:
        fh.write(json.dumps(log_line) + "\n")
    # (5) rotate
    _rotate_ckpt_dirs(fold_dir, keep_last)
    # cleanup
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    _log(f"  snapshot gstep={gstep} ep={meta['epoch']} batch={meta['batch']} "
         f"sec={log_line['snapshot_sec']} → {ckpt_dir.name}")
    return gstep


# ============================================================
# Main loop
# ============================================================
def main():
    p = argparse.ArgumentParser(description="HC #367 v3.4 ckpt sidecar")
    p.add_argument("--fold-idx", type=int, default=0)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--poll-sec", type=int, default=30)
    p.add_argument("--samples", type=int, default=1024,
                   help="size of fixed OOT minibatch for embeddings+preds")
    p.add_argument("--keep-last", type=int, default=20,
                   help="keep newest N ckpt_batch_* dirs per fold")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--book-features-dir", default=DEFAULT_BOOK_FEATURES_DIR)
    p.add_argument("--fifo-label-dir", default=str(DEFAULT_FIFO_LABEL_DIR))
    p.add_argument("--alpha-label-dir", default=str(DEFAULT_ALPHA_LABEL_DIR))
    p.add_argument("--pt-pred-dir", default=str(DEFAULT_PT_PRED_DIR))
    p.add_argument("--tier2-parquet-root", default=str(DEFAULT_TIER2_PARQUET_ROOT))
    p.add_argument("--tier3-parquet-root", default=str(DEFAULT_TIER3_PARQUET_ROOT))
    p.add_argument("--max-iters", type=int, default=0,
                   help="0 = run forever, else stop after N polls")
    p.add_argument("--exit-on-final-ckpt", action="store_true",
                   help="exit after seeing fold_NN_final.pt (signals fold complete)")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    fold_dir = output_dir / f"fold_{args.fold_idx:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    intra_ckpt_path = output_dir / f"fold_{args.fold_idx:02d}_intra_ckpt.pt"
    final_ckpt_path = output_dir / f"fold_{args.fold_idx:02d}_final.pt"
    fold_schedule_path = output_dir / "fold_schedule.json"
    feat_stats_path = output_dir / f"fold_{args.fold_idx:02d}_feature_stats.npz"
    log_path = fold_dir / "train_log_sidecar.jsonl"

    _log(f"start fold={args.fold_idx} device={args.device} poll={args.poll_sec}s "
         f"samples={args.samples} keep_last={args.keep_last}")
    _log(f"watching: {intra_ckpt_path}")
    _log(f"writing into: {fold_dir}")

    # Wait for fold_schedule.json + feature_stats.npz to land (trainer writes both early)
    waited = 0
    while not (fold_schedule_path.exists() and feat_stats_path.exists()):
        if waited == 0:
            _log("  waiting for fold_schedule.json + feature_stats.npz from trainer...")
        time.sleep(args.poll_sec)
        waited += args.poll_sec
        if args.max_iters and waited >= args.max_iters * args.poll_sec:
            _log("  max_iters reached before trainer prerequisites — exiting")
            return 1

    # Build fixed OOT loader ONCE (deterministic, reused across all snapshots)
    oot_dates = _discover_aligned_oot_dates(
        data_dir=Path(args.data_dir),
        book_dir=Path(args.book_features_dir),
        fold_schedule_path=fold_schedule_path,
        fold_idx=args.fold_idx,
    )
    _log(f"  OOT dates for fold {args.fold_idx}: {oot_dates[0]}→{oot_dates[-1]} "
         f"({len(oot_dates)} days)")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        _log("  CUDA unavailable; falling back to CPU")
        device = torch.device("cpu")

    loader = _build_oot_loader(
        data_dir=Path(args.data_dir),
        book_dir=Path(args.book_features_dir),
        fifo_label_dir=Path(args.fifo_label_dir),
        alpha_label_dir=Path(args.alpha_label_dir),
        pt_pred_dir=Path(args.pt_pred_dir),
        tier2_parquet_root=Path(args.tier2_parquet_root),
        tier3_parquet_root=Path(args.tier3_parquet_root),
        feature_stats_path=feat_stats_path,
        oot_dates=oot_dates,
        samples=args.samples,
        batch_size=args.batch_size,
        seed=args.fold_idx,
    )

    last_mtime: Optional[float] = None
    iters = 0
    last_snapshot_gstep: Optional[int] = None

    while True:
        iters += 1
        if args.max_iters and iters > args.max_iters:
            _log(f"max_iters {args.max_iters} reached — exiting")
            break

        # Exit-on-fold-complete signal
        if args.exit_on_final_ckpt and final_ckpt_path.exists():
            _log(f"final ckpt {final_ckpt_path.name} detected — taking one last snapshot then exiting")
            try:
                _snapshot_ckpt(
                    ckpt_path=final_ckpt_path,
                    fold_dir=fold_dir,
                    loader=loader,
                    device=device,
                    keep_last=args.keep_last,
                    log_path=log_path,
                )
            except Exception as e:
                _log(f"final snapshot failed: {e}\n{traceback.format_exc()}")
            break

        if not intra_ckpt_path.exists():
            time.sleep(args.poll_sec)
            continue

        try:
            mtime = intra_ckpt_path.stat().st_mtime
        except OSError:
            time.sleep(args.poll_sec)
            continue

        if last_mtime is not None and mtime <= last_mtime:
            time.sleep(args.poll_sec)
            continue

        # New ckpt detected — snapshot
        try:
            gstep = _snapshot_ckpt(
                ckpt_path=intra_ckpt_path,
                fold_dir=fold_dir,
                loader=loader,
                device=device,
                keep_last=args.keep_last,
                log_path=log_path,
            )
            if gstep is not None:
                last_snapshot_gstep = gstep
            last_mtime = mtime
        except Exception as e:
            _log(f"snapshot error (will retry next poll): {e}\n{traceback.format_exc()}")
            # do NOT update last_mtime so we retry

        time.sleep(args.poll_sec)

    _log(f"sidecar exit clean. iters={iters} last_gstep={last_snapshot_gstep}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
