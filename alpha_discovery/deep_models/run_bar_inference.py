"""
Bar-indexed inference for CNN-GRU S250 fold checkpoints.
Produces event-stride-aligned predictions for fill_sim compatibility.

Fixes vs prior draft:
  1. Loads raw 'events' key and applies compute_derived_features (not 'features' key)
  2. Normalizes using compute_stats (or pre-saved stats file)
  3. REMOVED x.permute(0,2,1) — model transposes internally (B,W,F) → (B,F,W)
  4. Fixed output key: pred_10s not preds_10s
  5. Output: event-stride predictions with timestamps for fill_sim alignment

Usage:
  python3 run_bar_inference.py \
    --checkpoint fold00_weights.pt \
    --data-dir /path/to/mbo_events/ \
    --test-start 20250922 --test-end 20251008 \
    --train-start 20250714 --train-end 20250921 \
    --output fold00_bar_preds.npz [--device cuda]

Output NPZ keys per date:
  {date}_pred_10s  (N_windows,) float32 — one per stride
  {date}_pred_5s   (N_windows,) float32
  {date}_pred_30s  (N_windows,) float32
  {date}_event_idx (N_windows,) int64   — event index of window END (last event)
  {date}_timestamp (N_windows,) int64   — timestamp (ns) of window END event
"""
import argparse, sys, os, json
import numpy as np
from pathlib import Path
import torch

# ── import from training script in same dir ──────────────────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from train_cnn_mamba import (
    CnnMambaModel, compute_derived_features, compute_stats,
    N_FEATURES, USE_DERIVED,
)

WINDOW   = 1000
STRIDE   = 250
CHANNELS = 64
N_LAYERS = 4
KERNEL   = 3
BATCH    = 512


def load_model(ckpt_path: Path, device: torch.device) -> CnnMambaModel:
    model = CnnMambaModel(N_FEATURES, CHANNELS, N_LAYERS, KERNEL)
    sd = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    model.load_state_dict(sd)
    model.eval().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[load_model] {ckpt_path.name}  params={n_params:,}  features={N_FEATURES}")
    return model


def get_stats(stats_file, train_files):
    """Load pre-saved stats or compute from train files."""
    if stats_file and Path(stats_file).exists():
        d = np.load(stats_file)
        print(f"[stats] Loaded from {stats_file}")
        return {"mean": d["mean"], "std": d["std"]}
    print(f"[stats] Computing from {len(train_files)} train files...")
    stats = compute_stats(train_files)
    # Optionally save alongside checkpoint
    return stats


def infer_file(model, npz_path: Path, stats: dict, device: torch.device):
    """
    Run stride-aligned inference for one NPZ file.
    Returns arrays aligned by stride: pred_5s, pred_10s, pred_30s, event_idx, timestamps.
    """
    data = np.load(npz_path, allow_pickle=True)
    ev_raw = (data["features"] if "features" in data else data["events"]).astype(np.float32)
    timestamps = data["timestamps"] if "timestamps" in data else np.zeros(len(ev_raw), dtype=np.int64)

    # Feature extraction + normalization
    if USE_DERIVED:
        file_date = int(npz_path.stem[:8]) if len(npz_path.stem) >= 8 else 0
        ev = compute_derived_features(ev_raw, file_date=file_date)
    else:
        ev = ev_raw
    mean = stats["mean"]; std = stats["std"]
    ev = (ev - mean) / (std + 1e-8)

    T = len(ev)
    pred_5s_list, pred_10s_list, pred_30s_list = [], [], []
    event_idx_list, ts_list = [], []

    batch_x, batch_ends = [], []

    def flush_batch():
        if not batch_x:
            return
        x = torch.tensor(np.array(batch_x, dtype=np.float32), dtype=torch.float32).to(device)
        # Shape: (B, W, F) — model transposes internally, do NOT permute here
        with torch.no_grad():
            out = model(x)
        if isinstance(out, dict):
            p5  = out.get("pred_5s",  out.get("5s",  list(out.values())[0])).cpu().numpy().ravel()
            p10 = out.get("pred_10s", out.get("10s", list(out.values())[1])).cpu().numpy().ravel()
            p30 = out.get("pred_30s", out.get("30s", list(out.values())[2])).cpu().numpy().ravel()
        elif isinstance(out, (list, tuple)):
            p5, p10, p30 = out[0].cpu().numpy().ravel(), out[1].cpu().numpy().ravel(), out[2].cpu().numpy().ravel()
        else:
            p5 = p10 = p30 = out.cpu().numpy().ravel()
        pred_5s_list.extend(p5.tolist())
        pred_10s_list.extend(p10.tolist())
        pred_30s_list.extend(p30.tolist())
        batch_x.clear(); batch_ends.clear()

    for start in range(0, T - WINDOW + 1, STRIDE):
        end = start + WINDOW
        win = ev[start:end]          # (W, N_FEATURES)
        end_idx = end - 1            # index of last event in window
        batch_x.append(win)
        batch_ends.append(end_idx)
        event_idx_list.append(end_idx)
        ts_list.append(int(timestamps[end_idx]) if end_idx < len(timestamps) else 0)
        if len(batch_x) >= BATCH:
            flush_batch()

    flush_batch()

    return {
        "pred_5s":    np.array(pred_5s_list,  dtype=np.float32),
        "pred_10s":   np.array(pred_10s_list, dtype=np.float32),
        "pred_30s":   np.array(pred_30s_list, dtype=np.float32),
        "event_idx":  np.array(event_idx_list, dtype=np.int64),
        "timestamps": np.array(ts_list,        dtype=np.int64),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",   required=True, help="fold*_weights.pt path")
    ap.add_argument("--data-dir",     required=True, help="dir with *.npz event files")
    ap.add_argument("--test-start",   required=True, help="YYYYMMDD")
    ap.add_argument("--test-end",     required=True, help="YYYYMMDD")
    ap.add_argument("--train-start",  default=None,  help="YYYYMMDD — for stats computation")
    ap.add_argument("--train-end",    default=None,  help="YYYYMMDD")
    ap.add_argument("--stats-file",   default=None,  help="pre-saved stats .npz (mean/std)")
    ap.add_argument("--output",       required=True)
    ap.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"[main] device={device}  N_FEATURES={N_FEATURES}  USE_DERIVED={USE_DERIVED}")

    ckpt = Path(args.checkpoint)
    data_dir = Path(args.data_dir)
    all_files = sorted(data_dir.glob("*_mbo_events.npz"))
    test_files  = [f for f in all_files if args.test_start  <= f.stem[:8] <= args.test_end]
    train_files = [f for f in all_files
                   if args.train_start and args.train_end
                   and args.train_start <= f.stem[:8] <= args.train_end] or all_files

    print(f"[main] test={len(test_files)} files  train={len(train_files)} for stats")
    stats = get_stats(args.stats_file, train_files)
    model = load_model(ckpt, device)

    out = {}
    for npz in test_files:
        date_str = f"{npz.stem[:4]}-{npz.stem[4:6]}-{npz.stem[6:8]}"
        print(f"  {date_str}...", end="", flush=True)
        res = infer_file(model, npz, stats, device)
        n = len(res["pred_10s"])
        nonzero = int(np.count_nonzero(res["pred_10s"]))
        print(f" {n} windows ({nonzero} nonzero pred_10s)")
        for k, v in res.items():
            out[f"{date_str}_{k}"] = v

    np.savez_compressed(args.output, **out)
    n_days = len(test_files)
    total_windows = sum(len(v) for k, v in out.items() if k.endswith("_pred_10s"))
    print(f"[main] Saved {n_days} days / {total_windows} total windows → {args.output}")


if __name__ == "__main__":
    main()
