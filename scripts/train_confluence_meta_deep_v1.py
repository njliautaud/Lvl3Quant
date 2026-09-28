"""
Deep Confluence Meta-Model v1 (HC #469 e1)
-------------------------------------------
Trains an MLP (256->128->64, dropout 0.2) on stacked predictions from:
  - CNN-Mamba v2 (bulk_oot + bulk_oot_v2)
  - PatchTST (bulk_oot)
Features: raw predictions per model per horizon, magnitudes, signs,
          pairwise agreements, pairwise products, spread features.
Target: realized labels (direction) from the NPZ files.
Walk-forward: sliding window (20 days train, 1 day OOT).
Cost: 0.376 ticks (HC #512).
No Mamba/SSM-CUDA (HC #464 R1) — pure MLP.
"""

import argparse
import os
import sys
import json
import glob
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict
from datetime import datetime


class DeepConfluenceMLP(nn.Module):
    """MLP 256->128->64 with dropout for confluence meta-learning."""
    def __init__(self, input_dim, n_horizons=3, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_horizons),  # predict each horizon
        )

    def forward(self, x):
        return self.net(x)


def load_model_predictions(base_dir, model_dirs):
    """Load predictions from multiple model output dirs, keyed by date."""
    model_data = {}
    for mname, mdir in model_dirs.items():
        full_path = os.path.join(base_dir, mdir)
        if not os.path.isdir(full_path):
            print(f"  SKIP {mname}: dir not found")
            continue
        npz_files = sorted(glob.glob(os.path.join(full_path, "*_predictions.npz")))
        if not npz_files:
            print(f"  SKIP {mname}: no NPZ files")
            continue
        model_data[mname] = {}
        for f in npz_files:
            date_str = os.path.basename(f).split("_")[0]
            try:
                d = np.load(f, allow_pickle=True)
                model_data[mname][date_str] = {
                    'predictions': d['predictions'],
                    'labels': d['labels'],
                    'n_samples': d['predictions'].shape[0],
                }
            except Exception as e:
                print(f"  WARN: {mname}/{date_str}: {e}")
        print(f"  {mname}: {len(model_data[mname])} dates loaded")
    return model_data


def find_overlapping_dates(model_data):
    """Find dates where ALL models have predictions."""
    if not model_data:
        return []
    date_sets = [set(v.keys()) for v in model_data.values()]
    common = date_sets[0]
    for ds in date_sets[1:]:
        common = common.intersection(ds)
    return sorted(common)


def build_features_for_date(model_data, date, model_names):
    """
    Build feature matrix for a single date from all models.
    Aligns by min sample count across models.

    Features per sample:
    - Raw predictions from each model (N_models * N_horizons)
    - Absolute predictions (N_models * N_horizons)
    - Sign indicators (N_models * N_horizons)
    - Pairwise products for each horizon (C(N_models,2) * N_horizons)
    - Pairwise agreement (sign match) for each horizon (C(N_models,2) * N_horizons)
    - Mean prediction per horizon (N_horizons)
    - Std prediction per horizon (N_horizons)
    - Max-min spread per horizon (N_horizons)
    """
    # Align sample counts
    min_n = min(model_data[m][date]['n_samples'] for m in model_names)

    all_preds = []
    for m in model_names:
        p = model_data[m][date]['predictions'][:min_n]  # (N, 3)
        all_preds.append(p)

    # Labels from first model (should be identical across models)
    labels = model_data[model_names[0]][date]['labels'][:min_n]

    # Stack: (N, n_models, 3)
    stacked = np.stack(all_preds, axis=1)
    N, n_models, n_h = stacked.shape

    features = []

    # 1. Raw predictions flattened: (N, n_models*3)
    features.append(stacked.reshape(N, -1))

    # 2. Absolute values: (N, n_models*3)
    features.append(np.abs(stacked).reshape(N, -1))

    # 3. Sign indicators: (N, n_models*3)
    features.append(np.sign(stacked).reshape(N, -1))

    # 4. Pairwise products and agreements
    pair_prods = []
    pair_agree = []
    for i in range(n_models):
        for j in range(i+1, n_models):
            # Product: (N, 3)
            pair_prods.append(stacked[:, i, :] * stacked[:, j, :])
            # Agreement: same sign = 1, different = 0
            pair_agree.append((np.sign(stacked[:, i, :]) == np.sign(stacked[:, j, :])).astype(np.float32))

    if pair_prods:
        features.append(np.concatenate(pair_prods, axis=1))
        features.append(np.concatenate(pair_agree, axis=1))

    # 5. Per-horizon stats across models
    features.append(np.mean(stacked, axis=1))  # (N, 3)
    features.append(np.std(stacked, axis=1))   # (N, 3)
    features.append(np.max(stacked, axis=1) - np.min(stacked, axis=1))  # (N, 3)

    # 6. Per-model confidence rank (normalized magnitude rank across models per horizon)
    abs_stacked = np.abs(stacked)
    # Rank within models for each sample: higher abs = higher rank
    ranks = np.zeros_like(abs_stacked)
    for h in range(n_h):
        for row in range(N):
            order = np.argsort(abs_stacked[row, :, h])
            ranks[row, order, h] = np.arange(n_models, dtype=np.float32) / max(n_models - 1, 1)
    features.append(ranks.reshape(N, -1))

    X = np.concatenate(features, axis=1).astype(np.float32)
    y = labels.astype(np.float32)

    return X, y


def train_one_fold(model, train_X, train_y, val_X, val_y, device, epochs=30, lr=1e-3, batch_size=4096):
    """Train model on one WF fold, return val metrics."""
    model.to(device)
    model.train()

    train_ds = TensorDataset(torch.tensor(train_X), torch.tensor(train_y))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=0, pin_memory=True)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()

    # Validate
    model.eval()
    with torch.no_grad():
        val_xt = torch.tensor(val_X).to(device)
        val_pred = model(val_xt).cpu().numpy()

    # Compute per-horizon IC (Spearman approx via Pearson on ranks)
    from scipy.stats import spearmanr
    ics = []
    for h in range(val_y.shape[1]):
        if np.std(val_pred[:, h]) < 1e-8 or np.std(val_y[:, h]) < 1e-8:
            ics.append(0.0)
        else:
            ic, _ = spearmanr(val_pred[:, h], val_y[:, h])
            ics.append(float(ic))

    # Directional accuracy per horizon
    accs = []
    for h in range(val_y.shape[1]):
        pred_dir = (val_pred[:, h] > 0).astype(int)
        true_dir = (val_y[:, h] > 0.25).astype(int)  # label > 0.25 = up
        acc = np.mean(pred_dir == true_dir)
        accs.append(float(acc))

    # Simulated P&L: top 5% confidence signals, net of 0.376 ticks cost
    pnl_per_h = []
    for h in range(val_y.shape[1]):
        abs_pred = np.abs(val_pred[:, h])
        thresh = np.percentile(abs_pred, 95)
        mask = abs_pred >= thresh
        if mask.sum() < 10:
            pnl_per_h.append(0.0)
            continue
        # Direction: sign of prediction
        directions = np.sign(val_pred[:, h][mask])
        # Realized move in ticks (labels are 0-1 scale, convert)
        # Labels are binary (0 or 0.5 or 1) — use raw for correlation
        # For P&L we need actual tick moves — use labels as proxy
        # label > 0.5 = up move, label < 0.5 = down move
        realized = (val_y[:, h][mask] - 0.5) * 2  # scale to [-1, 1]
        gross_ticks = np.mean(directions * realized)
        net_ticks = gross_ticks - 0.376
        pnl_per_h.append(float(net_ticks))

    return {
        'ic': ics,
        'acc': accs,
        'pnl_top5': pnl_per_h,
        'val_pred': val_pred,
        'val_y': val_y,
        'n_train': len(train_X),
        'n_val': len(val_X),
    }


def main():
    parser = argparse.ArgumentParser(description='Deep Confluence Meta-Model v1')
    parser.add_argument('--output-dir', default='output/confluence_meta_deep_v1')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--wf-window', type=int, default=20, help='WF training window in days')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--dropout', type=float, default=0.2)
    args = parser.parse_args()

    # Detect base dir robustly (works even when exec'd without __file__)
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        base_dir = os.path.abspath(os.getcwd())
    output_base = os.path.join(base_dir, 'output')
    out_dir = os.path.join(base_dir, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"=== Deep Confluence Meta-Model v1 ===")
    print(f"Device: {args.device}")
    print(f"WF window: {args.wf_window} days")
    print(f"Epochs: {args.epochs}")
    print(f"Base dir: {base_dir}")
    print()

    # Define model sources
    model_dirs = {
        'cm_v2': 'output/cnn_mamba_v2_bulk_oot',
        'cm_v2b': 'output/cnn_mamba_v2_bulk_oot_v2',
        'ptst': 'output/patchtst_bulk_oot',
    }

    print("Loading model predictions...")
    model_data = load_model_predictions(base_dir, model_dirs)

    if len(model_data) < 2:
        print("ERROR: Need at least 2 models with predictions. Exiting.")
        sys.exit(1)

    model_names = sorted(model_data.keys())
    print(f"\nModels loaded: {model_names}")

    # Find overlapping dates
    dates = find_overlapping_dates(model_data)
    print(f"Overlapping dates: {len(dates)}")

    if len(dates) < args.wf_window + 5:
        print(f"ERROR: Need at least {args.wf_window + 5} overlapping dates, got {len(dates)}")
        sys.exit(1)

    # Build features for all dates
    print("\nBuilding feature matrices...")
    date_features = {}
    for date in dates:
        X, y = build_features_for_date(model_data, date, model_names)
        date_features[date] = (X, y)
        print(f"  {date}: {X.shape[0]} samples, {X.shape[1]} features")

    input_dim = date_features[dates[0]][0].shape[1]
    print(f"\nTotal feature dimension: {input_dim}")

    # Walk-forward sliding window
    results = []
    all_val_preds = []
    all_val_labels = []

    n_folds = len(dates) - args.wf_window
    print(f"\nRunning {n_folds} walk-forward folds...")
    print(f"{'Fold':>4} {'Date':>10} {'IC_1s':>7} {'IC_5s':>7} {'IC_10s':>7} {'Acc_1s':>7} {'PnL_1s':>8}")
    print("-" * 60)

    t0 = time.time()

    for fold_idx in range(n_folds):
        train_dates = dates[fold_idx:fold_idx + args.wf_window]
        val_date = dates[fold_idx + args.wf_window]

        # Stack training data
        train_Xs = [date_features[d][0] for d in train_dates]
        train_ys = [date_features[d][1] for d in train_dates]
        train_X = np.concatenate(train_Xs, axis=0)
        train_y = np.concatenate(train_ys, axis=0)

        val_X, val_y = date_features[val_date]

        # Fresh model each fold
        model = DeepConfluenceMLP(input_dim, n_horizons=3, dropout=args.dropout)

        fold_result = train_one_fold(
            model, train_X, train_y, val_X, val_y,
            device=args.device, epochs=args.epochs,
            lr=args.lr, batch_size=args.batch_size
        )
        fold_result['date'] = val_date
        fold_result['fold'] = fold_idx
        results.append(fold_result)

        all_val_preds.append(fold_result['val_pred'])
        all_val_labels.append(fold_result['val_y'])

        # Save fold weights
        torch.save(model.state_dict(), os.path.join(out_dir, f'fold_{fold_idx:03d}_{val_date}.pt'))

        ic = fold_result['ic']
        acc = fold_result['acc']
        pnl = fold_result['pnl_top5']
        print(f"{fold_idx:4d} {val_date:>10} {ic[0]:7.4f} {ic[1]:7.4f} {ic[2]:7.4f} {acc[0]:7.3f} {pnl[0]:8.4f}")

    elapsed = time.time() - t0
    print(f"\nCompleted {n_folds} folds in {elapsed:.1f}s ({elapsed/n_folds:.1f}s/fold)")

    # Concat results
    all_preds = np.concatenate(all_val_preds, axis=0)
    all_labels = np.concatenate(all_val_labels, axis=0)

    # Save predictions
    np.savez(os.path.join(out_dir, 'concat_predictions.npz'),
             predictions=all_preds, labels=all_labels)

    # Summary stats
    from scipy.stats import spearmanr
    print("\n=== CONCAT OOT RESULTS ===")
    horizons = ['1s', '5s', '10s']
    for h_idx, h_name in enumerate(horizons):
        ic, _ = spearmanr(all_preds[:, h_idx], all_labels[:, h_idx])
        pred_dir = (all_preds[:, h_idx] > 0).astype(int)
        true_dir = (all_labels[:, h_idx] > 0.25).astype(int)
        acc = np.mean(pred_dir == true_dir)
        print(f"  {h_name}: concat_IC={ic:.4f}, accuracy={acc:.3f}")

    # Per-fold summary
    mean_ics = np.mean([r['ic'] for r in results], axis=0)
    std_ics = np.std([r['ic'] for r in results], axis=0)
    print(f"\n  Mean IC:  1s={mean_ics[0]:.4f}±{std_ics[0]:.4f}  5s={mean_ics[1]:.4f}±{std_ics[1]:.4f}  10s={mean_ics[2]:.4f}±{std_ics[2]:.4f}")

    mean_pnl = np.mean([r['pnl_top5'] for r in results], axis=0)
    print(f"  Mean PnL (top5%, net 0.376t): 1s={mean_pnl[0]:.4f}  5s={mean_pnl[1]:.4f}  10s={mean_pnl[2]:.4f}")

    # Save full results
    summary = {
        'n_folds': n_folds,
        'n_models': len(model_names),
        'model_names': model_names,
        'input_dim': input_dim,
        'dates': dates,
        'wf_window': args.wf_window,
        'epochs': args.epochs,
        'elapsed_s': elapsed,
        'concat_ic': {h: float(spearmanr(all_preds[:, i], all_labels[:, i])[0]) for i, h in enumerate(horizons)},
        'mean_ic': {h: float(mean_ics[i]) for i, h in enumerate(horizons)},
        'mean_pnl_top5': {h: float(mean_pnl[i]) for i, h in enumerate(horizons)},
        'per_fold': [{
            'fold': r['fold'],
            'date': r['date'],
            'ic': r['ic'],
            'acc': r['acc'],
            'pnl_top5': r['pnl_top5'],
            'n_train': r['n_train'],
            'n_val': r['n_val'],
        } for r in results],
    }

    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {out_dir}")
    print("DONE")


if __name__ == '__main__':
    main()
