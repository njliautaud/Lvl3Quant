#!/usr/bin/env python3
"""
Longer-Horizon Neural Network Model v1 (HC #637 Phase 2)
=========================================================

Deeper model for Neptune GPU. Uses the same hourly feature pipeline
as LightGBM v1, but with a temporal neural network that can capture:
  - Nonlinear feature interactions
  - Sequence patterns across hours (attention over hourly bars)
  - Better calibrated confidence scores

Architecture: Temporal MLP with hour-level attention
  - Input: last 6 hourly feature bars (full trading day context)
  - 3-layer MLP per bar → attention pooling → prediction head
  - Multi-task: predict 2h + 4h + EOD simultaneously
  - Output: directional score + calibrated confidence

Training: Walk-forward sliding 60d, daily retrain.
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LOG_DIR = ROOT / "logs"
OUTPUT_DIR = ROOT / "output" / "longer_horizon_nn_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    format='%(asctime)s [LH-NN] %(levelname)s %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_DIR / "longer_horizon_nn_v1.log")),
    ],
)
log = logging.getLogger('LH-NN')

# ─────────────────────────────────────────────
#  COST CONSTANTS
# ─────────────────────────────────────────────
ES_TICK_VALUE = 12.50
COST_RT_TICKS = 2.376

# ─────────────────────────────────────────────
#  DATA LOADING (reuse from v1)
# ─────────────────────────────────────────────
from alpha_discovery.longer_horizon_directional_v1 import (
    load_all_minute_bars,
    compute_hourly_features,
    add_rolling_context,
    add_macro_features,
    add_forward_labels,
    get_feature_columns,
    simulate_trades,
)


# ─────────────────────────────────────────────
#  PYTORCH MODEL
# ─────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    log.warning("PyTorch not available — install with: pip install torch")


if HAS_TORCH:

    class HourlySequenceDataset(Dataset):
        """Dataset that provides sequences of hourly bars."""

        def __init__(self, features: np.ndarray, targets: np.ndarray,
                     dates: np.ndarray, seq_len: int = 6):
            self.features = torch.FloatTensor(features)
            self.targets = torch.FloatTensor(targets)
            self.dates = dates
            self.seq_len = seq_len

            # Build valid indices (sequences within same day or consecutive days)
            self.valid_indices = []
            for i in range(seq_len - 1, len(features)):
                if not np.isnan(targets[i]):
                    self.valid_indices.append(i)

        def __len__(self):
            return len(self.valid_indices)

        def __getitem__(self, idx):
            i = self.valid_indices[idx]
            start = max(0, i - self.seq_len + 1)
            seq = self.features[start:i+1]

            # Pad if needed
            if len(seq) < self.seq_len:
                pad = torch.zeros(self.seq_len - len(seq), seq.shape[1])
                seq = torch.cat([pad, seq], dim=0)

            return seq, self.targets[i]

    class TemporalDirectionalModel(nn.Module):
        """
        Temporal MLP with attention for hourly directional prediction.

        Architecture:
          - Per-bar MLP: input_dim → hidden → hidden
          - Temporal attention: weight each bar by learned importance
          - Prediction head: attended repr → directional score
        """

        def __init__(self, input_dim: int, hidden_dim: int = 128,
                     n_heads: int = 4, dropout: float = 0.2, seq_len: int = 6):
            super().__init__()

            # Per-bar feature extractor
            self.bar_encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

            # Temporal attention
            self.attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.attn_norm = nn.LayerNorm(hidden_dim)

            # Position encoding
            self.pos_embed = nn.Embedding(seq_len, hidden_dim)

            # Prediction heads (multi-task: 2h, 4h, EOD)
            self.head_2h = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )
            self.head_4h = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )
            self.head_eod = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )

        def forward(self, x):
            """
            x: (batch, seq_len, input_dim)
            Returns: dict of predictions per horizon
            """
            B, S, D = x.shape

            # Encode each bar
            encoded = self.bar_encoder(x)  # (B, S, hidden)

            # Add position encoding
            positions = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)
            encoded = encoded + self.pos_embed(positions)

            # Self-attention over the sequence
            attn_out, _ = self.attn(encoded, encoded, encoded)
            encoded = self.attn_norm(encoded + attn_out)

            # Pool: take the last position (most recent bar)
            pooled = encoded[:, -1, :]  # (B, hidden)

            return {
                '2h': self.head_2h(pooled).squeeze(-1),
                '4h': self.head_4h(pooled).squeeze(-1),
                'eod': self.head_eod(pooled).squeeze(-1),
            }


# ─────────────────────────────────────────────
#  WALK-FORWARD TRAINING
# ─────────────────────────────────────────────

def train_nn_walk_forward(
    hourly_df: pd.DataFrame,
    feature_cols: List[str],
    train_days: int = 60,
    test_days: int = 5,
    seq_len: int = 6,
    hidden_dim: int = 128,
    epochs_per_fold: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = 'auto',
) -> Dict:
    """Walk-forward training of the temporal NN."""

    if not HAS_TORCH:
        log.error("PyTorch required for NN training")
        return {}

    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log.info(f"Device: {device}")

    dates = sorted(hourly_df['date'].unique())
    log.info(f"Date range: {dates[0]} → {dates[-1]} ({len(dates)} days)")

    # Prepare targets
    targets_2h = hourly_df['fwd_ticks_2h'].values
    targets_4h = hourly_df['fwd_ticks_4h'].values if 'fwd_ticks_4h' in hourly_df.columns else np.full(len(hourly_df), np.nan)
    targets_eod = hourly_df['fwd_ticks_eod'].values

    features = hourly_df[feature_cols].values.astype(np.float32)
    features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)
    date_arr = hourly_df['date'].values

    input_dim = len(feature_cols)

    # Results storage
    all_results = {h: {'preds': [], 'actuals': [], 'dates': []}
                   for h in ['2h', '4h', 'eod']}

    n_folds = 0
    for fold_start in range(train_days, len(dates) - test_days + 1, test_days):
        train_dates_set = set(dates[fold_start - train_days : fold_start])
        test_dates_set = set(dates[fold_start : fold_start + test_days])

        train_mask = np.isin(date_arr, list(train_dates_set))
        test_mask = np.isin(date_arr, list(test_dates_set))

        if train_mask.sum() < 100 or test_mask.sum() < 10:
            continue

        # Normalize features (fit on train)
        train_mean = features[train_mask].mean(axis=0)
        train_std = features[train_mask].std(axis=0)
        train_std[train_std < 1e-8] = 1.0

        norm_features = (features - train_mean) / train_std

        # Create datasets
        train_dataset = HourlySequenceDataset(
            norm_features[train_mask], targets_2h[train_mask],
            date_arr[train_mask], seq_len=seq_len
        )
        test_dataset = HourlySequenceDataset(
            norm_features[test_mask], targets_2h[test_mask],
            date_arr[test_mask], seq_len=seq_len
        )

        if len(train_dataset) < 50 or len(test_dataset) < 5:
            continue

        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=0)

        # Build model
        model = TemporalDirectionalModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            seq_len=seq_len,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_per_fold)

        # Train
        best_loss = float('inf')
        patience_counter = 0
        for epoch in range(epochs_per_fold):
            model.train()
            train_loss = 0
            n_batches = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                preds = model(X_batch)
                loss = F.mse_loss(preds['2h'], y_batch)
                # Could add 4h and EOD losses here for multi-task

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()
                n_batches += 1

            scheduler.step()

            avg_loss = train_loss / max(n_batches, 1)
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    break

        # Evaluate
        model.eval()
        fold_preds = {h: [] for h in ['2h', '4h', 'eod']}
        fold_actuals_2h = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                preds = model(X_batch)
                for h in ['2h', '4h', 'eod']:
                    fold_preds[h].extend(preds[h].cpu().numpy().tolist())
                fold_actuals_2h.extend(y_batch.numpy().tolist())

        # Record
        test_dates_list = sorted(test_dates_set)
        for h in ['2h']:  # Primary horizon for now
            if fold_preds[h]:
                all_results[h]['preds'].extend(fold_preds[h])
                all_results[h]['actuals'].extend(fold_actuals_2h)
                all_results[h]['dates'].extend([test_dates_list[0]] * len(fold_preds[h]))

        n_folds += 1
        if n_folds % 5 == 0:
            p = np.array(all_results['2h']['preds'])
            a = np.array(all_results['2h']['actuals'])
            if len(p) > 10:
                ic = np.corrcoef(p, a)[0, 1]
                log.info(f"Fold {n_folds}: cumulative 2h IC={ic:.4f}, n={len(p)}")

        # Cleanup
        del model, optimizer, scheduler
        if device == 'cuda':
            torch.cuda.empty_cache()

    # Final results
    final_results = {}
    for h in ['2h']:
        p = np.array(all_results[h]['preds'])
        a = np.array(all_results[h]['actuals'])
        d = np.array(all_results[h]['dates'])
        if len(p) < 10:
            continue

        from scipy import stats
        ic = np.corrcoef(p, a)[0, 1]
        rank_ic = stats.spearmanr(p, a)[0]
        dir_acc = np.mean(np.sign(p) == np.sign(a))

        # Quantile analysis
        for q in [0.10, 0.20, 0.30]:
            top = p >= np.quantile(p, 1-q)
            bot = p <= np.quantile(p, q)
            log.info(f"  {h} top_{int(q*100)}%: mean={a[top].mean():.1f}tk, "
                     f"WR={np.mean(a[top]>0):.1%}, n={top.sum()}")
            log.info(f"  {h} bot_{int(q*100)}%: mean={a[bot].mean():.1f}tk, "
                     f"WR={np.mean(a[bot]<0):.1%}, n={bot.sum()}")

        # Trading sim
        sim_20 = simulate_trades(p, a, confidence_threshold=0.20)

        final_results[h] = {
            'ic': ic,
            'rank_ic': rank_ic,
            'dir_acc': dir_acc,
            'n_preds': len(p),
            'n_folds': n_folds,
            'sim_top20': sim_20,
        }

        log.info(f"\n{'='*60}")
        log.info(f"{h} FINAL: IC={ic:.4f}, RankIC={rank_ic:.4f}, "
                 f"DirAcc={dir_acc:.1%}, n={len(p)}")
        if sim_20:
            log.info(f"  Sim top20%: Sharpe={sim_20['sharpe']:.2f}, "
                     f"WR={sim_20['win_rate']:.1%}, "
                     f"PF={sim_20['profit_factor']:.2f}, "
                     f"trades={sim_20['n_trades']}")

    return final_results


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Longer-Horizon NN v1')
    parser.add_argument('--train-days', type=int, default=60)
    parser.add_argument('--test-days', type=int, default=5)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seq-len', type=int, default=6)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--mlflow', action='store_true')
    parser.add_argument('--experiment-name', default='longer_horizon_nn_v1')
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("LONGER-HORIZON NEURAL NET v1 (HC #637 Phase 2)")
    log.info("=" * 60)

    # MLflow
    mlflow = None
    if args.mlflow:
        try:
            import mlflow as _mlflow
            mlflow = _mlflow
            mlflow.set_tracking_uri("http://localhost:5000")
            mlflow.set_experiment(args.experiment_name)
            mlflow.start_run(run_name=f"lh_nn_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            mlflow.log_params({
                'train_days': args.train_days,
                'test_days': args.test_days,
                'hidden_dim': args.hidden_dim,
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'lr': args.lr,
                'seq_len': args.seq_len,
                'model': 'temporal_mlp_attn',
            })
        except Exception as e:
            log.warning(f"MLflow failed: {e}")

    # Load data (same pipeline as LightGBM v1)
    log.info("Loading data...")
    minute_df = load_all_minute_bars()
    hourly_df = compute_hourly_features(minute_df)
    del minute_df; gc.collect()
    hourly_df = add_rolling_context(hourly_df)
    hourly_df = add_macro_features(hourly_df)

    horizons = {'2h': 2, '4h': 4, 'eod': -1}
    hourly_df = add_forward_labels(hourly_df, horizons)

    feature_cols = get_feature_columns(hourly_df, clean_mode=True)
    log.info(f"Features: {len(feature_cols)}, samples: {len(hourly_df)}")

    # Train
    results = train_nn_walk_forward(
        hourly_df,
        feature_cols=feature_cols,
        train_days=args.train_days,
        test_days=args.test_days,
        seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
        epochs_per_fold=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )

    # Log to MLflow
    if mlflow and results:
        for h, r in results.items():
            mlflow.log_metrics({
                f'{h}_ic': r['ic'],
                f'{h}_rank_ic': r['rank_ic'],
                f'{h}_dir_acc': r['dir_acc'],
            })
            if r.get('sim_top20'):
                mlflow.log_metrics({
                    f'{h}_sharpe': r['sim_top20']['sharpe'],
                    f'{h}_wr': r['sim_top20']['win_rate'],
                })
        mlflow.end_run()

    # Save results
    with open(OUTPUT_DIR / "results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

    log.info(f"\nResults saved to {OUTPUT_DIR / 'results.json'}")


if __name__ == '__main__':
    main()
