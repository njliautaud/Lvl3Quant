#!/usr/bin/env python3
"""
Trade Quality Filter v1 — MLP classifier to predict which trades will need market exit.

The blended strategy earns +0.32 ticks/trade but 10% of trades need market exit
(avg -7.5 tick loss). If we can predict which trades will need market exit and
SKIP those, the remaining trades average much higher.

Features: 25 smart_v3 microstructure features at prediction time
Label: binary — 1 if trade needs market exit (price moves against within 10s), 0 if passive exit fills

Walk-forward: 10-date sliding train, 1-date test across OOT dates.
Model: MLP 25 → 128 → 64 → 1 (binary), trained with BCE + class weights.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

if sys.platform == 'win32':
    DATA_ROOT = Path(r"C:\Users\claude\Lvl3Quant")
else:
    DATA_ROOT = Path("/home/jupiter/Lvl3Quant")

PRED_DIR = DATA_ROOT / "output" / "cnn_mamba_v2_bulk_oot_v2"
MBO_DIR = DATA_ROOT / "data" / "processed" / "mbo_events_smart_v3"
OUT_DIR = DATA_ROOT / "output" / "trade_filter_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 3000
STRIDE = 250
TRAIN_WINDOW = 10
CONFIDENCE_PCT = 2  # Top 2% short signals
COMMISSION_RT = 0.376

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class TradeFilterMLP(nn.Module):
    def __init__(self, in_dim=25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_date(date_str):
    """Load predictions + MBO features for one date, return aligned data."""
    pred_path = PRED_DIR / f"{date_str}_predictions.npz"
    mbo_path = MBO_DIR / f"{date_str}_mbo_events.npz"

    if not pred_path.exists() or not mbo_path.exists():
        return None

    pred_data = np.load(pred_path)
    mbo_data = np.load(mbo_path)

    predictions = pred_data['predictions']  # (N_pred, 3)
    labels = pred_data['labels']  # (N_pred, 3) — 1s, 5s, 10s returns
    events = mbo_data['events']  # (N_events, 25) — smart_v3 features

    ws = int(pred_data['window_size'])
    stride = int(pred_data['stride'])
    n_preds = len(predictions)

    # Map predictions to event indices
    event_indices = np.arange(ws, ws + n_preds * stride, stride)

    # Filter valid indices
    valid = event_indices < len(events)
    event_indices = event_indices[valid]
    predictions = predictions[valid]
    labels = labels[valid]

    # Get features at prediction points
    features = events[event_indices]  # (N, 25)

    # Remove NaN rows
    mask = (~np.any(np.isnan(features), axis=1) &
            ~np.any(np.isnan(predictions), axis=1) &
            ~np.any(np.isnan(labels), axis=1))

    features = features[mask]
    predictions = predictions[mask]
    labels = labels[mask]

    # Subsample to keep data manageable on CPU (every 4th event)
    stride_sub = 4
    idx = np.arange(0, len(features), stride_sub)
    features = features[idx]
    predictions = predictions[idx]
    labels = labels[idx]

    return {
        'features': features,
        'predictions': predictions,
        'labels': labels,
        'n': len(features)
    }


def create_trade_labels(data, confidence_pct=2):
    """
    For top confidence_pct% SHORT signals, create binary labels:
    0 = passive exit works (price moves favorably within 10s)
    1 = needs market exit (price stays adverse through 10s)
    """
    preds = data['predictions'][:, 0]  # 1s predictions
    labels = data['labels']  # (N, 3) at 1s, 5s, 10s
    features = data['features']

    # Standardize predictions within date
    p_std = (preds - preds.mean()) / (preds.std() + 1e-8)
    threshold = np.percentile(p_std, confidence_pct)
    trade_mask = p_std <= threshold

    trade_features = features[trade_mask]
    trade_labels_raw = labels[trade_mask]

    # For each trade, determine if it needs market exit
    # Short trade: profit if label < 0 (price goes down)
    pnl_1s = -trade_labels_raw[:, 0]
    pnl_5s = -trade_labels_raw[:, 1]
    pnl_10s = -trade_labels_raw[:, 2]

    # Passive exit works if any horizon shows favorable price
    passive_ok = (pnl_1s >= 0) | (pnl_5s >= 0) | (pnl_10s >= 0)
    needs_market = ~passive_ok  # Binary label: 1 = needs market exit

    return trade_features, needs_market.astype(np.float32), pnl_1s, pnl_5s, pnl_10s


def main():
    print(f"{'='*60}")
    print(f"Trade Quality Filter v1")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"{'='*60}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Discover dates
    pred_dates = sorted([f.stem.replace('_predictions', '') for f in PRED_DIR.glob('*_predictions.npz')])
    mbo_dates = sorted([f.stem.replace('_mbo_events', '') for f in MBO_DIR.glob('*_mbo_events.npz')])
    common_dates = sorted(set(pred_dates) & set(mbo_dates))
    print(f"Common dates: {len(common_dates)}")

    # Load all dates
    all_data = {}
    for d in common_dates:
        data = load_date(d)
        if data is not None and data['n'] > 500:
            all_data[d] = data
            print(f"  {d}: N={data['n']:,}")

    dates = sorted(all_data.keys())
    print(f"\nLoaded {len(dates)} dates")

    # Walk-forward
    results = []
    all_test_probs = []
    all_test_labels = []
    all_test_pnls = []

    for i in range(TRAIN_WINDOW, len(dates)):
        test_date = dates[i]
        train_dates = dates[i - TRAIN_WINDOW:i]

        # Gather training data
        train_feats_list, train_labels_list = [], []
        for d in train_dates:
            feats, labels, _, _, _ = create_trade_labels(all_data[d], CONFIDENCE_PCT)
            if len(feats) > 0:
                train_feats_list.append(feats)
                train_labels_list.append(labels)

        if not train_feats_list:
            continue

        train_X = np.concatenate(train_feats_list)
        train_Y = np.concatenate(train_labels_list)

        # Test data
        test_feats, test_labels, pnl_1s, pnl_5s, pnl_10s = create_trade_labels(
            all_data[test_date], CONFIDENCE_PCT)

        if len(test_feats) < 10:
            continue

        # Standardize features
        mu = train_X.mean(axis=0)
        sigma = train_X.std(axis=0) + 1e-8
        train_X_std = (train_X - mu) / sigma
        test_X_std = (test_feats - mu) / sigma

        # Class weights (market exit is rare ~10%)
        pos_weight = (1 - train_Y.mean()) / (train_Y.mean() + 1e-8)

        # Train
        model = TradeFilterMLP(in_dim=25).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.BCELoss(weight=None)

        train_ds = TensorDataset(
            torch.FloatTensor(train_X_std).to(device),
            torch.FloatTensor(train_Y).to(device)
        )
        loader = DataLoader(train_ds, batch_size=2048, shuffle=True)

        model.train()
        for epoch in range(20):
            for X_batch, Y_batch in loader:
                optimizer.zero_grad()
                pred = model(X_batch)
                # Weighted BCE
                weights = torch.where(Y_batch == 1, pos_weight, 1.0)
                loss = nn.functional.binary_cross_entropy(pred, Y_batch, weight=weights)
                loss.backward()
                optimizer.step()

        # Predict on test
        model.eval()
        with torch.no_grad():
            test_probs = model(torch.FloatTensor(test_X_std).to(device)).cpu().numpy()

        # Evaluate
        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(test_labels, test_probs)
        except:
            auc = 0.5

        # Compute blended P&L with and without filter
        # Without filter
        trade_pnls = []
        for j in range(len(test_feats)):
            if pnl_1s[j] >= 0: pnl = pnl_1s[j] - COMMISSION_RT
            elif pnl_5s[j] >= 0: pnl = pnl_5s[j] - COMMISSION_RT
            elif pnl_10s[j] >= 0: pnl = pnl_10s[j] - COMMISSION_RT
            else: pnl = pnl_10s[j] - COMMISSION_RT - 1.0
            trade_pnls.append(pnl)
        trade_pnls = np.array(trade_pnls)

        # With filter: skip trades where model predicts high market-exit probability
        for threshold in [0.15, 0.20, 0.25, 0.30]:
            keep = test_probs < threshold
            if keep.sum() < 5:
                continue
            filtered_pnls = trade_pnls[keep]
            n_kept = keep.sum()
            n_total = len(keep)

            if i == TRAIN_WINDOW:  # Print header once
                pass

        # Store for concat analysis
        all_test_probs.append(test_probs)
        all_test_labels.append(test_labels)
        all_test_pnls.append(trade_pnls)

        market_rate = test_labels.mean()
        results.append({
            'date': test_date,
            'fold': i - TRAIN_WINDOW,
            'auc': auc,
            'n_trades': len(test_feats),
            'market_rate': float(market_rate),
            'avg_pnl_unfiltered': float(trade_pnls.mean()),
        })

        print(f"Fold {i-TRAIN_WINDOW:2d} | {test_date} | N={len(test_feats):,} | "
              f"AUC={auc:.4f} | MktRate={market_rate:.1%} | "
              f"Avg PnL={trade_pnls.mean():+.4f}")

    # Concat analysis
    if all_test_probs:
        concat_probs = np.concatenate(all_test_probs)
        concat_labels = np.concatenate(all_test_labels)
        concat_pnls = np.concatenate(all_test_pnls)

        try:
            concat_auc = roc_auc_score(concat_labels, concat_probs)
        except:
            concat_auc = 0.5

        print(f"\n{'='*60}")
        print(f"CONCAT RESULTS ({len(concat_pnls):,} trades)")
        print(f"{'='*60}")
        print(f"AUC: {concat_auc:.4f}")
        print(f"Market exit rate: {concat_labels.mean():.1%}")
        print(f"Unfiltered avg PnL: {concat_pnls.mean():+.4f}")

        print(f"\nFILTER SWEEP (skip trades with high market-exit probability):")
        print(f"{'Threshold':>10} {'Kept':>6} {'Skipped':>8} {'Avg PnL':>8} {'WR':>6} {'Lift':>7}")

        for thr in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50]:
            keep = concat_probs < thr
            if keep.sum() < 100:
                continue
            filtered = concat_pnls[keep]
            lift = filtered.mean() - concat_pnls.mean()
            wr = (filtered > 0).mean()
            print(f"{thr:10.2f} {keep.sum():6,} ({keep.mean():.0%})  "
                  f"{filtered.mean():+8.4f} {wr:6.1%} {lift:+7.4f}")

    # Save results
    summary = {
        'timestamp': datetime.now().isoformat(),
        'concat_auc': float(concat_auc) if all_test_probs else None,
        'n_folds': len(results),
        'fold_results': results,
    }
    with open(OUT_DIR / 'results.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nResults saved to {OUT_DIR}")
    print(f"Completed: {datetime.now().isoformat()}")


if __name__ == '__main__':
    main()
