"""
IC Anomaly Investigation: Fold 75 vs Fold 76 and decay analysis.
"""
import numpy as np
import torch
import sys
from scipy.stats import spearmanr
sys.path.insert(0, 'C:/Users/Footb/Documents/Github/Lvl3Quant/alpha_discovery/deep_models')
from train_walkforward import compute_mfe_net
from book_spatial_cnn import BookSpatialCNN

class WiderBookSpatialCNN(BookSpatialCNN):
    def __init__(self, **kwargs):
        kwargs['spatial_channels'] = (64, 128, 256, 512)
        kwargs['temporal_channels'] = 512
        kwargs.setdefault('dropout', 0.15)
        super().__init__(**kwargs)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

DATA_DIR = 'C:/Users/Footb/Documents/Github/Lvl3Quant/data/processed/dl_book_cache'
CKPT_DIR = 'C:/Users/Footb/Documents/Github/Lvl3Quant/alpha_discovery/deep_models/results/wider_cnn/checkpoints'


def run_inference_on_date(weights_path, date_str, batch_size=256):
    """Load model and run inference on a specific date. Returns (IC, n_samples)."""
    npz = np.load(f'{DATA_DIR}/{date_str}_book_tensors.npz', allow_pickle=True)
    books = npz['book_tensors'].astype(np.float32)
    # Apply log-transform (same as BarDataset preprocessing)
    np.log1p(books[:, :, 1], out=books[:, :, 1])
    np.log1p(books[:, :, 2], out=books[:, :, 2])
    np.log1p(books[:, :, 3], out=books[:, :, 3])
    mids = npz['mid_prices']
    target = compute_mfe_net(mids, [0, len(mids)])
    valid_mask = np.isfinite(target)

    model = WiderBookSpatialCNN(window_size=20, num_levels=20, num_features=4, num_classes=1)
    state = torch.load(weights_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    window_size = 20
    valid_indices = np.where(valid_mask)[0]
    valid_indices = valid_indices[valid_indices >= window_size - 1]

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for start in range(0, len(valid_indices), batch_size):
            batch_idx = valid_indices[start:start+batch_size]
            windows = np.stack([books[i-window_size+1:i+1] for i in batch_idx])
            windows_t = torch.from_numpy(windows).to(device)
            preds = model(windows_t).squeeze(-1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(target[batch_idx])

    preds_all = np.concatenate(all_preds)
    tgts_all = np.concatenate(all_targets)
    ic, _ = spearmanr(preds_all, tgts_all)
    return ic, len(preds_all)


print("=" * 70)
print("TASK 1: Fold 75 vs Fold 76 models on 2025-11-04")
print("Checkpoint naming: fold_74_2025-11-03.pt = actual fold 75 (pre-jump, IC=0.1293)")
print("                   fold_75_2025-11-04.pt = actual fold 76 (post-jump, IC=0.6929)")
print("=" * 70)

# fold_74 = trained as fold 75 (test=2025-11-03), weights saved AFTER fold 75 training
fold75_weights = f'{CKPT_DIR}/fold_74_2025-11-03.pt'
fold76_weights = f'{CKPT_DIR}/fold_75_2025-11-04.pt'

print(f"\nRunning fold 75 model (pre-jump) on 2025-11-04...")
ic_f75_on_1104, n = run_inference_on_date(fold75_weights, '2025-11-04')
print(f"  IC: {ic_f75_on_1104:+.4f}  (n={n:,})")

print(f"\nRunning fold 76 model (post-jump) on 2025-11-04...")
ic_f76_on_1104, n = run_inference_on_date(fold76_weights, '2025-11-04')
print(f"  IC: {ic_f76_on_1104:+.4f}  (n={n:,})")

print(f"\n--- COMPARISON ---")
print(f"Fold 75 model on 2025-11-04: {ic_f75_on_1104:+.4f}")
print(f"Fold 76 model on 2025-11-04: {ic_f76_on_1104:+.4f}")
print(f"Delta: {ic_f76_on_1104 - ic_f75_on_1104:+.4f}")

if ic_f76_on_1104 > 0.4:
    print("VERDICT: Fold 76 model achieves IC > 0.4 on OOT date → CONFIRMED LEAKAGE")
elif ic_f76_on_1104 > 0.2:
    print("VERDICT: Fold 76 model IC is elevated but plausible → uncertain, check further")
else:
    print("VERDICT: IC is normal on OOT date → fold 76 IC was training artifact, not real")

print()
print("=" * 70)
print("TASK 2: Fold 75 model decay across time")
print("Using fold_74_2025-11-03.pt (fold 75, trained through 2025-10-31)")
print("=" * 70)

decay_dates = ['2025-12-01', '2026-01-15', '2026-02-15', '2026-03-01']
print()
for date in decay_dates:
    try:
        ic, n = run_inference_on_date(fold75_weights, date)
        print(f"  {date}: IC={ic:+.4f}  (n={n:,})")
    except Exception as e:
        print(f"  {date}: ERROR - {e}")

print()
print("Done.")
