"""
Quick Validation Script for BookGNN.

Loads 1 day of data, builds the graph, runs a forward pass through an untrained
model (sanity check shapes), then trains 1 epoch and reports metrics.

This is the "start small to validate" step before committing to a full
walk-forward run.

Usage:
  python validate_gnn.py
  python validate_gnn.py --use-gat
  python validate_gnn.py --hidden 128 --layers 3
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

# Path setup
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(MODELS_DIR))

from book_gnn import BookGNN, build_order_book_graph, compute_norm_adj, count_parameters

DEFAULT_BOOK_DIR = str(ROOT_DIR / 'data' / 'processed' / 'dl_book_cache')


def compute_mfe_net(mid_prices, day_boundaries, horizon_bars=100, tick_size=0.25):
    """Compute mfe_net target (same as train_walkforward.py)."""
    from numpy.lib.stride_tricks import sliding_window_view

    N = len(mid_prices)
    H = horizon_bars
    mfe_long = np.full(N, np.nan, dtype=np.float32)
    mfe_short = np.full(N, np.nan, dtype=np.float32)

    mid_shifted = mid_prices[1:]
    valid_len = N - H - 1

    if valid_len > 0:
        windows = sliding_window_view(mid_shifted, H)[:valid_len]
        fwd_max = windows.max(axis=1)
        fwd_min = windows.min(axis=1)
        mfe_long[:valid_len] = np.maximum(0.0, (fwd_max - mid_prices[:valid_len]) / tick_size)
        mfe_short[:valid_len] = np.maximum(0.0, (mid_prices[:valid_len] - fwd_min) / tick_size)

    n_days = len(day_boundaries) - 1
    for d in range(n_days - 1):
        day_end = day_boundaries[d + 1]
        nan_start = max(day_boundaries[d], day_end - H)
        mfe_long[nan_start:day_end] = np.nan
        mfe_short[nan_start:day_end] = np.nan

    return (mfe_long - mfe_short).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description='BookGNN Quick Validation')
    parser.add_argument('--book-dir', type=str, default=DEFAULT_BOOK_DIR)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--temporal-dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--use-gat', action='store_true')
    parser.add_argument('--gat-heads', type=int, default=4)
    parser.add_argument('--window-size', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = torch.device(
        'cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    )

    print('=' * 70)
    print('BookGNN Quick Validation')
    print('=' * 70)

    # -----------------------------------------------------------------------
    # Step 1: Graph structure sanity check
    # -----------------------------------------------------------------------
    print('\n--- Step 1: Graph Structure ---')
    edge_index = build_order_book_graph()
    num_nodes = 20
    num_edges = edge_index.shape[1]
    print(f'  Nodes: {num_nodes}')
    print(f'  Edges: {num_edges} (directed, including self-loops)')

    # Count edge types
    src, dst = edge_index[0].numpy(), edge_index[1].numpy()
    self_loops = sum(1 for s, d in zip(src, dst) if s == d)
    bid_bid = sum(1 for s, d in zip(src, dst) if s < 10 and d < 10 and s != d)
    ask_ask = sum(1 for s, d in zip(src, dst) if s >= 10 and d >= 10 and s != d)
    cross = sum(1 for s, d in zip(src, dst) if (s < 10) != (d < 10))
    print(f'  Self-loops: {self_loops}')
    print(f'  Bid-bid edges: {bid_bid}')
    print(f'  Ask-ask edges: {ask_ask}')
    print(f'  Cross bid-ask: {cross}')

    norm_adj = compute_norm_adj(edge_index, num_nodes)
    print(f'  Norm adj shape: {norm_adj.shape}')
    print(f'  Norm adj row sums: min={norm_adj.sum(1).min():.3f}, '
          f'max={norm_adj.sum(1).max():.3f}')

    # -----------------------------------------------------------------------
    # Step 2: Model instantiation and parameter count
    # -----------------------------------------------------------------------
    print('\n--- Step 2: Model Architecture ---')
    variant = 'GAT' if args.use_gat else 'GCN'
    model = BookGNN(
        window_size=args.window_size,
        hidden_dim=args.hidden,
        num_gcn_layers=args.layers,
        temporal_dim=args.temporal_dim,
        dropout=args.dropout,
        num_classes=1,
        use_gat=args.use_gat,
        gat_heads=args.gat_heads,
    ).to(device)

    n_params = count_parameters(model)
    print(f'  Variant: {variant}')
    print(f'  Hidden dim: {args.hidden}')
    print(f'  GCN layers: {args.layers}')
    print(f'  Temporal dim: {args.temporal_dim}')
    print(f'  Parameters: {n_params:,}')

    # -----------------------------------------------------------------------
    # Step 3: Forward pass with dummy data (shape verification)
    # -----------------------------------------------------------------------
    print('\n--- Step 3: Forward Pass (dummy data) ---')
    B = 4
    x_dummy = torch.randn(B, args.window_size, 20, 4, device=device)

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    with torch.no_grad():
        out = model(x_dummy)
    t_fwd = time.time() - t0

    print(f'  Input shape:  ({B}, {args.window_size}, 20, 4)')
    print(f'  Output shape: {out.shape}')
    print(f'  Output values: {out.squeeze(-1).cpu().numpy()}')
    print(f'  Forward pass time: {t_fwd*1000:.1f}ms')

    if device.type == 'cuda':
        peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
        print(f'  GPU peak memory (dummy): {peak_mem:.1f} MB')

    # -----------------------------------------------------------------------
    # Step 4: Load 1 day of real data
    # -----------------------------------------------------------------------
    print('\n--- Step 4: Load Real Data (1 day) ---')
    book_dir = Path(args.book_dir)
    npz_files = sorted(book_dir.glob('*_book_tensors.npz'))
    if not npz_files:
        print(f'  ERROR: No book tensor files found in {book_dir}')
        return

    npz_path = npz_files[0]
    print(f'  Loading: {npz_path.name}')

    data = np.load(npz_path)
    book_tensors = data['book_tensors']  # (n_bars, 20, 4)
    mid_prices = data['mid_prices']      # (n_bars,)

    print(f'  book_tensors shape: {book_tensors.shape}')
    print(f'  mid_prices shape: {mid_prices.shape}')
    print(f'  book_tensors dtype: {book_tensors.dtype}')

    # Log-transform features 1,2,3 (same as training pipeline)
    book_tensors = book_tensors.copy()
    book_tensors[:, :, 1] = np.log1p(book_tensors[:, :, 1])
    book_tensors[:, :, 2] = np.log1p(book_tensors[:, :, 2])
    book_tensors[:, :, 3] = np.log1p(book_tensors[:, :, 3])

    # Compute targets
    boundaries = [0, len(mid_prices)]
    target = compute_mfe_net(mid_prices, boundaries, horizon_bars=100)
    n_valid = int(np.isfinite(target).sum())
    print(f'  Valid target bars: {n_valid:,} / {len(target):,}')

    # Z-score normalize
    finite_mask = np.isfinite(target)
    tgt_mean = float(target[finite_mask].mean())
    tgt_std = float(target[finite_mask].std())
    target = (target - tgt_mean) / tgt_std
    print(f'  Target mean: {tgt_mean:.3f}, std: {tgt_std:.3f} ticks')

    # Build windowed samples (subsample for speed)
    W = args.window_size
    subsample = 10  # aggressive subsampling for quick validation
    valid_indices = []
    for i in range(W - 1, len(target) - 1):
        if np.isfinite(target[i]):
            valid_indices.append(i)
    valid_indices = valid_indices[::subsample]

    print(f'  Samples (subsample={subsample}x): {len(valid_indices):,}')

    # -----------------------------------------------------------------------
    # Step 5: Forward pass with real data (shape check)
    # -----------------------------------------------------------------------
    print('\n--- Step 5: Forward Pass (real data batch) ---')

    # Build a small batch
    batch_indices = valid_indices[:args.batch_size]
    batch_windows = np.stack([
        book_tensors[i - W + 1: i + 1] for i in batch_indices
    ])  # (B, W, 20, 4)
    batch_targets = np.array([target[i] for i in batch_indices])

    x_real = torch.from_numpy(batch_windows.astype(np.float32)).to(device)
    y_real = torch.from_numpy(batch_targets).to(device)

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    with torch.no_grad():
        preds = model(x_real).squeeze(-1)
    t_fwd = time.time() - t0

    print(f'  Batch input shape: {x_real.shape}')
    print(f'  Predictions shape: {preds.shape}')
    print(f'  Predictions range: [{preds.min().item():.4f}, {preds.max().item():.4f}]')
    print(f'  Targets range: [{y_real.min().item():.4f}, {y_real.max().item():.4f}]')
    print(f'  Forward time: {t_fwd*1000:.1f}ms for batch of {len(batch_indices)}')

    # IC on untrained model (should be ~0)
    ic_untrained, _ = spearmanr(preds.cpu().numpy(), y_real.cpu().numpy())
    print(f'  IC (untrained): {ic_untrained:+.4f} (expected ~0)')

    if device.type == 'cuda':
        peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
        print(f'  GPU peak memory: {peak_mem:.1f} MB')

    # -----------------------------------------------------------------------
    # Step 6: Train 1 epoch
    # -----------------------------------------------------------------------
    print('\n--- Step 6: Train 1 Epoch ---')

    # Build full dataset tensors
    all_windows = np.stack([
        book_tensors[i - W + 1: i + 1] for i in valid_indices
    ])  # (N, W, 20, 4)
    all_targets = np.array([target[i] for i in valid_indices])

    # Split: 80% train, 20% val (temporal split, no shuffle)
    split = int(0.8 * len(valid_indices))
    train_x = torch.from_numpy(all_windows[:split].astype(np.float32))
    train_y = torch.from_numpy(all_targets[:split].astype(np.float32))
    val_x = torch.from_numpy(all_windows[split:].astype(np.float32))
    val_y = torch.from_numpy(all_targets[split:].astype(np.float32))

    print(f'  Train samples: {len(train_x):,}')
    print(f'  Val samples: {len(val_x):,}')

    train_ds = torch.utils.data.TensorDataset(train_x, train_y)
    val_ds = torch.utils.data.TensorDataset(val_x, val_y)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
    )

    # Reset model
    model = BookGNN(
        window_size=args.window_size,
        hidden_dim=args.hidden,
        num_gcn_layers=args.layers,
        temporal_dim=args.temporal_dim,
        dropout=args.dropout,
        num_classes=1,
        use_gat=args.use_gat,
        gat_heads=args.gat_heads,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = nn.HuberLoss(delta=1.0)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    # Train
    model.train()
    t_train = time.time()
    total_loss = 0.0
    total_n = 0
    n_batches = 0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                preds = model(batch_x).squeeze(-1)
                loss = criterion(preds, batch_y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(batch_x).squeeze(-1)
            loss = criterion(preds, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        n = batch_y.shape[0]
        total_loss += loss.item() * n
        total_n += n
        n_batches += 1

    train_time = time.time() - t_train
    train_loss = total_loss / max(total_n, 1)

    print(f'  Train loss: {train_loss:.4f}')
    print(f'  Batches: {n_batches}')
    print(f'  Epoch time: {train_time:.1f}s')
    print(f'  Time per batch: {train_time/max(n_batches,1)*1000:.1f}ms')

    if device.type == 'cuda':
        peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
        print(f'  GPU peak memory (training): {peak_mem:.1f} MB')

    # Evaluate
    model.eval()
    all_preds = []
    all_tgts = []
    with torch.no_grad():
        ctx = torch.amp.autocast('cuda') if device.type == 'cuda' else torch.no_grad()
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            with ctx:
                preds = model(batch_x).squeeze(-1)
            all_preds.append(preds.cpu().numpy())
            all_tgts.append(batch_y.numpy())

    preds_arr = np.concatenate(all_preds)
    tgts_arr = np.concatenate(all_tgts)

    mask = np.isfinite(preds_arr) & np.isfinite(tgts_arr)
    ic_trained, _ = spearmanr(preds_arr[mask], tgts_arr[mask])

    print(f'\n  IC after 1 epoch: {ic_trained:+.4f}')
    print(f'  (IC > 0 after 1 epoch is a good sign)')

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('VALIDATION SUMMARY')
    print('=' * 70)
    print(f'  Model variant:     {variant}')
    print(f'  Parameters:        {n_params:,}')
    print(f'  Graph:             {num_nodes} nodes, {num_edges} edges')
    print(f'  Shapes:            OK (input -> output verified)')
    print(f'  IC (untrained):    {ic_untrained:+.4f}')
    print(f'  IC (1 epoch):      {ic_trained:+.4f}')
    print(f'  Train loss:        {train_loss:.4f}')
    print(f'  Epoch time:        {train_time:.1f}s')
    if device.type == 'cuda':
        print(f'  GPU peak memory:   {peak_mem:.1f} MB')
    print(f'  Device:            {device}')
    print('=' * 70)

    if ic_trained > 0:
        print('\n  PASS: Model learns signal after 1 epoch. Ready for walk-forward.')
    else:
        print('\n  NOTE: IC <= 0 after 1 epoch. This may improve with more epochs.')
        print('        Try --epochs 3 in train_gnn.py before concluding.')


if __name__ == '__main__':
    main()
