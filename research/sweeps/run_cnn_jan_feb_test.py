"""Quick CNN test: Train on January 2026, test on February 2026.
Directly tests if the model works on the "hard" recent dates.
Uses sliding window approach (single fold, no warm start needed).
"""
import os, sys

# Single-fold: train Jan, test Feb (first 10 days)
os.environ["CNN_WF_MODE"] = "expanding"  # Use expanding for single manual fold
os.environ["CNN_STRIDE"] = "250"
os.environ["CNN_EPOCHS"] = "15"
os.environ["CNN_BATCH"] = "128"
os.environ["CNN_DERIVED_FEATURES"] = "0"
os.environ["STRICT_LEAKAGE_FREE"] = "1"
os.environ["CNN_CACHE_DAYS"] = "5"
os.environ["CNN_OUTPUT_DIR"] = r"C:\Users\Footb\Documents\Github\Lvl3Quant\results\cnn_jan_train_feb_test"
os.environ["PYTHONUNBUFFERED"] = "1"

# We'll manually control the fold — just need to filter files
# Train: Jan 2026 files (idx 146-170, ~25 days)
# Test: Feb 1-10 2026 (first 10 Feb files)

output_dir = os.environ["CNN_OUTPUT_DIR"]
os.makedirs(output_dir, exist_ok=True)

log_path = os.path.join(output_dir, "train.log")
sys.stdout = open(log_path, "w", buffering=1)
sys.stderr = sys.stdout

os.chdir(r"C:\Users\Footb\Documents\Github\Lvl3Quant")

# Override the training to use specific date ranges
# We'll do this by importing the script's components and running manually
import numpy as np
import torch
import time
import json
from pathlib import Path
from scipy.stats import spearmanr

# Set all CNN config
os.environ["CNN_FOLDS"] = "1"  # Just one fold
os.environ["START_FOLD"] = "0"

# Get file list
data_dir = Path("data/processed/mbo_events")
all_files = sorted(data_dir.glob("*_mbo_events.npz"))

# Split: Jan for train, first 10 Feb days for test
jan_files = [f for f in all_files if f.name.startswith("202601")]
feb_files = [f for f in all_files if f.name.startswith("202602")][:10]

print(f"Train: {len(jan_files)} Jan files ({jan_files[0].name} → {jan_files[-1].name})")
print(f"Test: {len(feb_files)} Feb files ({feb_files[0].name} → {feb_files[-1].name})")
print(f"Config: stride=250, epochs=15, batch=128, raw6 features")
print()

# Now run the CNN training using the script's functions
sys.path.insert(0, "alpha_discovery/deep_models")
os.chdir("alpha_discovery/deep_models")

# Import what we need from the training script
exec("""
import importlib.util
spec = importlib.util.spec_from_file_location("cnn", "train_cnn_clean_s84.py")
""")

# Actually just run the training directly with manual fold control
# Simpler: exec the whole script but override the fold building
# Even simpler: use the training functions directly

# Let's just do it inline — load data, build model, train, eval
from train_cnn_clean_s84 import (
    EventCNN1D, MboDataset, FileSequentialSampler,
    compute_feature_stats, compute_ic, compute_long_short_ic,
    WINDOW_SIZE, STRIDE, BATCH_SIZE, EPOCHS, LR, GRAD_CLIP,
    CNN_CHANNELS, CNN_KERNEL, CNN_LAYERS, CNN_DROPOUT, DILATIONS,
    N_FEATURES, HORIZONS, log
)
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Compute feature stats from train (Jan) only
log("Computing feature stats from Jan files...")
feat_mean, feat_std = compute_feature_stats(jan_files)

# Build datasets
log("Loading Jan train data...")
train_ds = MboDataset(jan_files, feat_mean, feat_std)
log(f"Train samples: {len(train_ds)}")

log("Loading Feb test data...")
test_ds = MboDataset(feb_files, feat_mean, feat_std)
log(f"Test samples: {len(test_ds)}")

# DataLoaders
train_sampler = FileSequentialSampler(train_ds, shuffle_files=True)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=0, drop_last=True)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=0)

# Model
model = EventCNN1D(n_features=N_FEATURES).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

log(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
log(f"Training {EPOCHS} epochs...")

for epoch in range(EPOCHS):
    model.train()
    total_loss, batches = 0.0, 0
    t0 = time.time()
    for xb, yb, wb in train_dl:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=True):
            out = model(xb)
            loss = F.mse_loss(out, yb)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        batches += 1
    elapsed = time.time() - t0
    log(f"Epoch {epoch+1}/{EPOCHS} loss={total_loss/max(batches,1):.6f} t={elapsed:.1f}s")

# Evaluate on Feb
log("Evaluating on February data...")
model.eval()
all_preds = {h: [] for h in HORIZONS}
all_labels = {h: [] for h in HORIZONS}
with torch.no_grad():
    for xb, yb, _wb in test_dl:
        xb = xb.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=True):
            out = model(xb).cpu().float().numpy()
        yb_np = yb.numpy()
        for i, h in enumerate(HORIZONS):
            all_preds[h].append(out[:, i])
            all_labels[h].append(yb_np[:, i])

preds_cat = {h: np.concatenate(all_preds[h]) for h in HORIZONS}
labels_cat = {h: np.concatenate(all_labels[h]) for h in HORIZONS}

log("=" * 60)
log("RESULTS: Train on Jan 2026, Test on Feb 1-10 2026")
log("=" * 60)
metrics = {}
for h in HORIZONS:
    ic = compute_ic(preds_cat[h], labels_cat[h])
    ic_l, ic_s = compute_long_short_ic(preds_cat[h], labels_cat[h])
    n_long = (preds_cat[h] > 0).sum()
    n_short = (preds_cat[h] < 0).sum()
    metrics[h] = {"ic": ic, "ic_long": ic_l, "ic_short": ic_s, "n_long": int(n_long), "n_short": int(n_short), "n_total": len(preds_cat[h])}
    log(f"IC_{h} = {ic:.4f}  | Long={ic_l:.4f}(n={n_long})  Short={ic_s:.4f}(n={n_short})")

# Save
out_dir = Path(output_dir)
np.savez(out_dir / "jan_feb_preds.npz", **{f"preds_{h}": preds_cat[h] for h in HORIZONS}, **{f"labels_{h}": labels_cat[h] for h in HORIZONS})
with open(out_dir / "jan_feb_metrics.json", "w") as f:
    json.dump({"train": "Jan 2026", "test": "Feb 1-10 2026", "metrics": metrics}, f, indent=2)
log("Results saved.")
