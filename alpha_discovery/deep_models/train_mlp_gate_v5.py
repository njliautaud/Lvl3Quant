#!/usr/bin/env python3
"""
train_mlp_gate_v5.py — MLP Confidence Gate using CNN Embeddings + Vol Predictions
==================================================================================
Predicts P(CNN signal correct) using:
  - CNN-Mamba v2 embeddings (96d)
  - CNN-Mamba v2 raw predictions (3 horizons)
  - Vol LGBM v3 predictions (volatility)
  - Derived features (|pred|, pred sign agreement across horizons, etc.)

Uses CORRECT costs: $4.70 RT = 0.376 ticks (HC #52).

Walk-forward: leave-one-date-out cross-validation on 10 OOT dates.
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from scipy.stats import spearmanr

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from constants import COMMISSION_TICKS, TICK_VALUE

# ─── Config ───────────────────────────────────────────────────────────────────
CNN_PRED_DIR = "output/cnn_mamba_v2_smart_v3_mar"
VOL_PRED_DIR = "output/vol_lgbm_v3"
OUTPUT_DIR = "output/mlp_gate_v5"

HORIZONS = ['1s', '5s', '10s']
N_FOLDS = 10
EPOCHS = 50
LR = 1e-3
BATCH_SIZE = 1024
HIDDEN_DIMS = [128, 64, 32]
DROPOUT = 0.3
DEVICE = "cpu"  # Jupiter is CPU-only

# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_fold_data(fold_idx: int) -> dict:
    """Load CNN embeddings, predictions, and vol predictions for one OOT fold."""
    cnn_file = os.path.join(CNN_PRED_DIR, f"fold_{fold_idx:02d}_oot_predictions.npz")
    if not os.path.exists(cnn_file):
        return None

    cnn_data = np.load(cnn_file, allow_pickle=True)
    embeddings = cnn_data['embeddings']       # (N, 96)
    predictions = cnn_data['predictions']     # (N, 3) = 1s/5s/10s
    labels = cnn_data['labels']               # (N, 3) = 1s/5s/10s

    # Get OOT date
    oot_files = cnn_data['oot_files']
    if hasattr(oot_files, 'tolist'):
        oot_files = oot_files.tolist()
    date_str = str(oot_files[0]).split('/')[-1][:8] if isinstance(oot_files, list) else str(oot_files).split('/')[-1][:8]

    # Load vol predictions for this date
    vol_file = os.path.join(VOL_PRED_DIR, f"vol_v3_{date_str}_predictions.npz")
    vol_pred = None
    if os.path.exists(vol_file):
        vd = np.load(vol_file, allow_pickle=True)
        vol_keys = list(vd.keys())
        # Try common key names
        for k in ['predictions', 'vol_pred', 'y_pred']:
            if k in vd:
                vp = vd[k]
                # Align lengths — vol may have different sampling
                if len(vp) == len(embeddings):
                    vol_pred = vp
                else:
                    # Simple interpolation to match CNN length
                    from scipy.interpolate import interp1d
                    x_vol = np.linspace(0, 1, len(vp))
                    x_cnn = np.linspace(0, 1, len(embeddings))
                    if vp.ndim == 1:
                        f_interp = interp1d(x_vol, vp, kind='nearest', fill_value='extrapolate')
                        vol_pred = f_interp(x_cnn)
                    else:
                        vol_pred = np.zeros((len(embeddings), vp.shape[1]))
                        for j in range(vp.shape[1]):
                            f_interp = interp1d(x_vol, vp[:, j], kind='nearest', fill_value='extrapolate')
                            vol_pred[:, j] = f_interp(x_cnn)
                break

    return {
        'embeddings': embeddings,
        'predictions': predictions,
        'labels': labels,
        'vol_pred': vol_pred,
        'date': date_str,
        'n_samples': len(embeddings)
    }


def build_features(data: dict) -> np.ndarray:
    """Build feature matrix from CNN embeddings, predictions, and vol."""
    parts = []

    # 1. CNN embeddings (96d)
    parts.append(data['embeddings'])

    # 2. Raw CNN predictions (3 horizons)
    parts.append(data['predictions'])

    # 3. Absolute predictions (conviction strength)
    parts.append(np.abs(data['predictions']))

    # 4. Sign agreement across horizons (do all horizons agree?)
    signs = np.sign(data['predictions'])
    sign_agreement = np.mean(signs == signs[:, :1], axis=1, keepdims=True)  # fraction agreeing with 1s
    parts.append(sign_agreement)

    # 5. Prediction z-scores (normalized conviction)
    pred_std = np.std(data['predictions'], axis=0, keepdims=True)
    pred_std = np.where(pred_std < 1e-8, 1, pred_std)
    parts.append(data['predictions'] / pred_std)

    # 6. Vol predictions if available
    if data['vol_pred'] is not None:
        vp = data['vol_pred']
        if vp.ndim == 1:
            vp = vp.reshape(-1, 1)
        parts.append(vp)

    return np.concatenate(parts, axis=1).astype(np.float32)


def build_labels(data: dict, horizon_idx: int = 0) -> np.ndarray:
    """Binary label: was the CNN prediction correct (direction match)?"""
    pred_sign = np.sign(data['predictions'][:, horizon_idx])
    label_sign = np.sign(data['labels'][:, horizon_idx])
    correct = (pred_sign == label_sign).astype(np.float32)
    return correct


# ─── Model ────────────────────────────────────────────────────────────────────

class MLPGate(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, dropout: float = 0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hd in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hd),
                nn.BatchNorm1d(hd),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hd
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ─── Training ────────────────────────────────────────────────────────────────

def train_gate(X_train, y_train, X_val, y_val, epochs=50, lr=1e-3):
    """Train MLP gate with early stopping."""
    input_dim = X_train.shape[1]
    model = MLPGate(input_dim, HIDDEN_DIMS, DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.BCELoss()

    X_tr = torch.from_numpy(X_train).to(DEVICE)
    y_tr = torch.from_numpy(y_train).to(DEVICE)
    X_vl = torch.from_numpy(X_val).to(DEVICE)
    y_vl = torch.from_numpy(y_val).to(DEVICE)

    best_val_loss = float('inf')
    best_state = None
    patience = 10
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        # Mini-batch training
        perm = torch.randperm(len(X_tr))
        total_loss = 0
        n_batches = 0
        for i in range(0, len(X_tr), BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            pred = model(X_tr[idx])
            loss = criterion(pred, y_tr[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_vl)
            val_loss = criterion(val_pred, y_vl).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    model.load_state_dict(best_state)
    return model, best_val_loss


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 80)
    print("MLP Gate v5 — CNN Embeddings + Vol Predictions")
    print(f"Commission: ${COMMISSION_TICKS * TICK_VALUE:.2f} RT = {COMMISSION_TICKS:.3f} ticks (HC #52)")
    print("=" * 80)

    # Load all fold data
    all_data = []
    for fold in range(N_FOLDS):
        d = load_fold_data(fold)
        if d is not None:
            all_data.append(d)
            print(f"Fold {fold}: date={d['date']}, n={d['n_samples']}, vol={'yes' if d['vol_pred'] is not None else 'no'}")

    if len(all_data) < 3:
        print("ERROR: Not enough folds to train. Need at least 3.")
        return

    # For each horizon, run leave-one-date-out CV
    for h_idx, h_name in enumerate(HORIZONS):
        print(f"\n{'='*60}")
        print(f"HORIZON: {h_name}")
        print(f"{'='*60}")

        all_oot_results = []

        for val_idx in range(len(all_data)):
            # Train on all dates except val_idx
            train_X = []
            train_y = []
            for i, d in enumerate(all_data):
                if i == val_idx:
                    continue
                X = build_features(d)
                y = build_labels(d, h_idx)
                # Replace NaN/inf
                X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
                train_X.append(X)
                train_y.append(y)

            train_X = np.concatenate(train_X)
            train_y = np.concatenate(train_y)

            # Val data
            val_d = all_data[val_idx]
            val_X = build_features(val_d)
            val_X = np.nan_to_num(val_X, nan=0, posinf=0, neginf=0)
            val_y = build_labels(val_d, h_idx)

            # Normalize features
            mean = train_X.mean(axis=0)
            std = train_X.std(axis=0)
            std = np.where(std < 1e-8, 1, std)
            train_X = (train_X - mean) / std
            val_X = (val_X - mean) / std

            # Train
            model, val_loss = train_gate(train_X, train_y, val_X, val_y, EPOCHS, LR)

            # Evaluate
            model.eval()
            with torch.no_grad():
                gate_probs = model(torch.from_numpy(val_X).to(DEVICE)).cpu().numpy()

            # CNN predictions for this date
            cnn_preds = val_d['predictions'][:, h_idx]
            cnn_labels = val_d['labels'][:, h_idx]

            # Evaluate at different gate thresholds
            abs_cnn = np.abs(cnn_preds)

            print(f"\n  Val date: {val_d['date']} (n={len(val_X)})")
            print(f"  {'Threshold':>10s} | {'Coverage':>8s} | {'DA_gated':>8s} | {'DA_ungated':>10s} | {'Lift':>6s} | {'Sortino':>8s} | {'AvgPnL':>8s}")
            print(f"  {'-'*70}")

            baseline_da = np.mean(np.sign(cnn_preds) == np.sign(cnn_labels))

            for gate_thresh in [0.40, 0.45, 0.50, 0.55, 0.60]:
                gate_mask = gate_probs >= gate_thresh
                if gate_mask.sum() < 20:
                    continue

                coverage = gate_mask.mean()
                gated_preds = cnn_preds[gate_mask]
                gated_labels = cnn_labels[gate_mask]

                da_gated = np.mean(np.sign(gated_preds) == np.sign(gated_labels))
                lift = da_gated - baseline_da

                # PnL with correct costs
                direction = np.sign(gated_preds)
                pnl = direction * gated_labels - COMMISSION_TICKS
                avg_pnl = np.mean(pnl) * TICK_VALUE

                down = pnl[pnl < 0]
                down_std = np.std(down) if len(down) > 0 else 1e-8
                sortino = np.mean(pnl) / down_std * np.sqrt(252 * 6.5 * 60)

                print(f"  {gate_thresh:>10.2f} | {coverage:>7.1%} | {da_gated:>7.1%} | {baseline_da:>9.1%} | {lift:>+5.1%} | {sortino:>8.2f} | ${avg_pnl:>7.2f}")

            # Also test: gate + confidence combo (top 5% conviction + gate)
            top5_mask = abs_cnn >= np.percentile(abs_cnn, 95)
            combo_mask = top5_mask & (gate_probs >= 0.50)
            if combo_mask.sum() >= 10:
                gated_preds = cnn_preds[combo_mask]
                gated_labels = cnn_labels[combo_mask]
                da = np.mean(np.sign(gated_preds) == np.sign(gated_labels))
                direction = np.sign(gated_preds)
                pnl = direction * gated_labels - COMMISSION_TICKS
                avg_pnl = np.mean(pnl) * TICK_VALUE
                down = pnl[pnl < 0]
                down_std = np.std(down) if len(down) > 0 else 1e-8
                sortino = np.mean(pnl) / down_std * np.sqrt(252 * 6.5 * 60)
                print(f"  {'Top5%+Gate':>10s} | {combo_mask.mean():>7.1%} | {da:>7.1%} | {baseline_da:>9.1%} | {da-baseline_da:>+5.1%} | {sortino:>8.2f} | ${avg_pnl:>7.2f}")

            all_oot_results.append({
                'date': val_d['date'],
                'gate_probs': gate_probs,
                'cnn_preds': cnn_preds,
                'cnn_labels': cnn_labels,
                'baseline_da': float(baseline_da),
            })

        # Concat all OOT results
        print(f"\n{'='*60}")
        print(f"CONCAT RESULTS — {h_name}")
        print(f"{'='*60}")

        all_gate = np.concatenate([r['gate_probs'] for r in all_oot_results])
        all_cpreds = np.concatenate([r['cnn_preds'] for r in all_oot_results])
        all_clabels = np.concatenate([r['cnn_labels'] for r in all_oot_results])
        abs_all = np.abs(all_cpreds)

        baseline_da = np.mean(np.sign(all_cpreds) == np.sign(all_clabels))
        print(f"Total predictions: {len(all_gate):,}")
        print(f"Baseline DA: {baseline_da:.1%}")
        print(f"\n{'Threshold':>10s} | {'Coverage':>8s} | {'DA':>6s} | {'Lift':>6s} | {'Sortino':>8s} | {'AvgPnL':>8s} | {'PF':>5s}")
        print(f"{'-'*65}")

        for gate_thresh in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
            gate_mask = all_gate >= gate_thresh
            n = gate_mask.sum()
            if n < 50:
                continue

            gp = all_cpreds[gate_mask]
            gl = all_clabels[gate_mask]
            da = np.mean(np.sign(gp) == np.sign(gl))

            direction = np.sign(gp)
            pnl = direction * gl - COMMISSION_TICKS
            avg_pnl = np.mean(pnl) * TICK_VALUE

            down = pnl[pnl < 0]
            down_std = np.std(down) if len(down) > 0 else 1e-8
            sortino = np.mean(pnl) / down_std * np.sqrt(252 * 6.5 * 60)

            wins = pnl[pnl > 0]
            losses = pnl[pnl < 0]
            pf = abs(wins.sum() / losses.sum()) if len(losses) > 0 and losses.sum() != 0 else float('inf')

            print(f"{gate_thresh:>10.2f} | {gate_mask.mean():>7.1%} | {da:>5.1%} | {da-baseline_da:>+5.1%} | {sortino:>8.2f} | ${avg_pnl:>7.2f} | {pf:>5.2f}")

        # Combo: top N% conviction + gate
        for pct_name, pct in [('Top10%', 90), ('Top5%', 95), ('Top1%', 99)]:
            thresh = np.percentile(abs_all, pct)
            conv_mask = abs_all >= thresh
            combo = conv_mask & (all_gate >= 0.50)
            n = combo.sum()
            if n < 20:
                continue
            gp = all_cpreds[combo]
            gl = all_clabels[combo]
            da = np.mean(np.sign(gp) == np.sign(gl))
            direction = np.sign(gp)
            pnl = direction * gl - COMMISSION_TICKS
            avg_pnl = np.mean(pnl) * TICK_VALUE
            down = pnl[pnl < 0]
            down_std = np.std(down) if len(down) > 0 else 1e-8
            sortino = np.mean(pnl) / down_std * np.sqrt(252 * 6.5 * 60)
            wins = pnl[pnl > 0]
            losses = pnl[pnl < 0]
            pf = abs(wins.sum() / losses.sum()) if len(losses) > 0 and losses.sum() != 0 else float('inf')
            print(f"{pct_name+'+Gate':>10s} | {combo.mean():>7.1%} | {da:>5.1%} | {da-baseline_da:>+5.1%} | {sortino:>8.2f} | ${avg_pnl:>7.2f} | {pf:>5.2f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
