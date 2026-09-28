#!/usr/bin/env python3
"""Run fold_00_best.pt inference over all 56 OOT days (Feb 24 -> Apr 29).

Reuses model class + dataset + run_oot_inference from train_cnn_mamba_v2.
Output: one combined NPZ with per-day predictions + labels + day index.
"""
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/home/nick/Lvl3Quant")
from alpha_discovery.deep_models.train_cnn_mamba_v2 import (
    CNNMambaV2,
    LazyMboEventDataset,
    run_oot_inference,
    HORIZONS,
    WINDOW_SIZE,
    STRIDE,
)

DATA_DIR = Path("/home/nick/Lvl3Quant/data/processed/mbo_events_smart_v3")
MODEL_DIR = Path("/home/nick/Lvl3Quant/output/cnn_mamba_v3_h30s_oot_apr")
OUT_DIR = Path("/home/nick/Lvl3Quant/output/cnn_mamba_v3_h30s_56day_inference")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OOT_START = "20260224"
OOT_END = "20260429"
BATCH_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    t0 = time.time()
    # Load model
    ckpt = torch.load(MODEL_DIR / "fold_00_best.pt", map_location=DEVICE, weights_only=False)
    model = CNNMambaV2(n_targets=len(HORIZONS)).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded fold_00_best.pt val_loss={ckpt.get('val_loss','?')}", flush=True)

    # Load feature stats for normalization (a dict from .npz)
    stats_npz = np.load(MODEL_DIR / "fold_00_feature_stats.npz")
    feat_stats = {k: stats_npz[k] for k in stats_npz.files}
    print(f"Feature stats keys: {list(feat_stats.keys())}", flush=True)

    # Iterate OOT days
    npz_files = sorted(DATA_DIR.glob("*_mbo_events.npz"))
    oot_files = [f for f in npz_files if OOT_START <= f.stem.split("_")[0] <= OOT_END]
    print(f"OOT days: {len(oot_files)} (range {oot_files[0].stem} -> {oot_files[-1].stem})", flush=True)

    all_preds = []
    all_labels = []
    all_day_idx = []
    all_day_names = []

    for day_i, fp in enumerate(oot_files):
        day = fp.stem.split("_")[0]
        ds = LazyMboEventDataset(
            [fp], window_size=WINDOW_SIZE, stride=STRIDE,
            horizons=HORIZONS, normalize_features=True,
            feature_stats=feat_stats,
        )
        if len(ds) == 0:
            print(f"  {day}: 0 samples, skip", flush=True)
            continue
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)
        preds, labels, _ = run_oot_inference(model, loader, DEVICE, use_amp=True,
                                              extract_embeddings=False)
        n = len(preds)
        all_preds.append(preds)
        all_labels.append(labels)
        all_day_idx.append(np.full(n, day_i, dtype=np.int32))
        all_day_names.append(day)
        print(f"  {day}: n={n}  IC@1s={np.corrcoef(preds[:,0], labels[:,0])[0,1]:.4f}", flush=True)

    if not all_preds:
        print("No predictions produced", flush=True)
        return

    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    day_idx = np.concatenate(all_day_idx, axis=0)

    np.savez_compressed(
        OUT_DIR / "predictions_56day.npz",
        predictions=preds,
        labels=labels,
        day_idx=day_idx,
        day_names=np.array(all_day_names),
        horizons=np.array(HORIZONS),
    )
    print(f"\nDone. {len(preds):,} samples across {len(all_day_names)} days, elapsed {time.time()-t0:.1f}s", flush=True)
    # Top-level concat IC per horizon
    for i, h in enumerate(HORIZONS):
        ic = np.corrcoef(preds[:, i], labels[:, i])[0, 1]
        print(f"  Concat IC@{h}: {ic:.4f}", flush=True)


if __name__ == "__main__":
    main()
