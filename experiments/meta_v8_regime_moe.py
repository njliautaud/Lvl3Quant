"""
Meta-Model v8: Regime-Conditioned Mixture-of-Experts

Key change from v7: Instead of a single MLP, use a Mixture-of-Experts with
regime-conditioned gating. Two expert MLPs (high-vol specialist, low-vol specialist)
with a gating network that routes based on regime features.

v7 baseline: Spearman 0.308, 27/27 positive days

Features:
  Expert input (31 total, same as v7):
    - CNN-Mamba predictions (3): 1s, 5s, 10s horizons
    - CNN-Mamba confidence (3): abs(prediction) per horizon
    - Microstructure from MBO events (25): spread, depth imbalance, OFI, etc.

  Gating input (8 regime features, computed from MBO events):
    - Rolling realized vol (50/200/500 event windows) — from col 20 (realized_volatility z-score)
    - Trend indicator: signed cumulative price momentum — from col 9 (price_mom_10)
    - Volume acceleration: event density ratio — from col 8 (event_density_20)
    - Spread regime: spread vs session avg — from col 5 (spread_ticks)
    - OFI trend: long-term OFI for directional regime — from col 23 (ofi_long_2000)
    - Momentum divergence: short vs long momentum — from col 16 (mom_divergence)

Architecture:
  Expert 1: MLP 128->64 (low-vol specialist)
  Expert 2: MLP 128->64 (high-vol specialist)
  Gating: Linear(8) -> softmax -> 2 weights
  Output: weighted sum of expert outputs
  ~50K params (similar to v7)

Walk-forward: SLIDING 10d train / 3d eval (match v7 prod)
Target: 1s mid-price movement (Huber loss)
"""

import os
import sys
import json
import time
import logging
import zipfile
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
from datetime import datetime

# MLflow
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# ============================================================
# Configuration
# ============================================================
CONFIG = {
    'output_dir': '/home/nick/Lvl3Quant/output/meta_v8_regime_moe',
    'cm_dir': '/home/nick/Lvl3Quant/output/cnn_mamba_v2_bulk_oot',
    'mbo_dir': '/home/nick/Lvl3Quant/data/processed/mbo_events_smart_v3',
    # Walk-forward (match v7 prod: 10d train / 3d eval)
    'train_days': 10,
    'eval_days': 3,
    # Model — MoE
    'expert_hidden_dims': [128, 64],
    'n_experts': 2,
    'n_regime_features': 8,
    'dropout': 0.2,
    'batch_size': 2048,
    'epochs': 25,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'patience': 5,
    # Auxiliary loss weight for gating diversity
    'gate_entropy_weight': 0.01,
    # Cost constants
    'tick_value': 12.50,
    'commission_ticks': 0.376,
    # v7 baseline
    'v7_spearman': 0.308,
    'target_horizon': '1s',
}

# MBO event column indices (from streaming_features_smart_v3.py)
COL_SPREAD = 5        # spread_ticks: clip [0,20], /5.0
COL_OFI_500 = 7       # rolling_ofi_500: z-score
COL_EVENT_DENSITY = 8  # event_density_20: clamp [0,4], /2.0
COL_PRICE_MOM = 9      # price_mom_10: z-score
COL_MOM_DIV = 16       # mom_divergence: *5 + clip
COL_REALIZED_VOL = 20  # realized_volatility: z-score
COL_OFI_LONG = 23      # ofi_long_2000: z-score
COL_OFI_ACCEL = 24     # ofi_acceleration: z-score

# ============================================================
# Logging
# ============================================================
os.makedirs(CONFIG['output_dir'], exist_ok=True)
log_path = os.path.join(CONFIG['output_dir'], 'training.log')

class MetaV8Formatter(logging.Formatter):
    def format(self, record):
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return f"{ts} [META_V8_MOE] {record.levelname}: {record.msg}"

logger = logging.getLogger('meta_v8')
logger.setLevel(logging.INFO)
logger.propagate = False

fh = logging.FileHandler(log_path)
fh.setFormatter(MetaV8Formatter())
logger.addHandler(fh)

class FlushStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

sh = FlushStreamHandler(sys.stdout)
sh.setFormatter(MetaV8Formatter())
logger.addHandler(sh)


# ============================================================
# Regime Feature Extraction
# ============================================================
def compute_regime_features(mbo_events, event_indices):
    """
    Compute 8 regime features at CNN-Mamba event positions from MBO events.

    The MBO events columns are already z-scored/normalized at the event level.
    For regime features, we compute ROLLING STATISTICS over windows of events
    to capture the current regime state (not individual event features).

    Features (8):
      0: vol_regime_short  — rolling mean of realized_vol over last 50 events
      1: vol_regime_med    — rolling mean of realized_vol over last 200 events
      2: vol_regime_long   — rolling mean of realized_vol over last 500 events
      3: trend_indicator   — rolling mean of price_mom over last 100 events (signed)
      4: trend_long        — rolling mean of price_mom over last 500 events
      5: volume_accel      — event_density at position / rolling mean event_density(200)
      6: spread_regime     — rolling mean spread over last 200 events vs overall
      7: ofi_regime        — rolling mean of ofi_long over last 200 events
    """
    n_events = len(event_indices)
    regime = np.zeros((n_events, 8), dtype=np.float32)

    # Pre-extract relevant columns for all MBO events
    vol_col = mbo_events[:, COL_REALIZED_VOL]
    pmom_col = mbo_events[:, COL_PRICE_MOM]
    density_col = mbo_events[:, COL_EVENT_DENSITY]
    spread_col = mbo_events[:, COL_SPREAD]
    ofi_long_col = mbo_events[:, COL_OFI_LONG]

    # Use cumulative sums for efficient rolling mean computation
    vol_cumsum = np.concatenate([[0], np.cumsum(vol_col)])
    pmom_cumsum = np.concatenate([[0], np.cumsum(pmom_col)])
    density_cumsum = np.concatenate([[0], np.cumsum(density_col)])
    spread_cumsum = np.concatenate([[0], np.cumsum(spread_col)])
    ofi_cumsum = np.concatenate([[0], np.cumsum(ofi_long_col)])

    def rolling_mean(cumsum, idx, window):
        """Compute rolling mean of previous `window` values ending at idx (exclusive of idx)."""
        start = max(0, idx - window)
        if idx <= start:
            return 0.0
        return (cumsum[idx] - cumsum[start]) / (idx - start)

    for i, eidx in enumerate(event_indices):
        # Vol regime at 3 scales
        regime[i, 0] = rolling_mean(vol_cumsum, eidx, 50)
        regime[i, 1] = rolling_mean(vol_cumsum, eidx, 200)
        regime[i, 2] = rolling_mean(vol_cumsum, eidx, 500)

        # Trend at 2 scales
        regime[i, 3] = rolling_mean(pmom_cumsum, eidx, 100)
        regime[i, 4] = rolling_mean(pmom_cumsum, eidx, 500)

        # Volume acceleration: current density vs rolling mean
        curr_density = density_col[eidx] if eidx < len(density_col) else 0.0
        avg_density = rolling_mean(density_cumsum, eidx, 200)
        regime[i, 5] = (curr_density / (avg_density + 1e-6)) - 1.0  # centered at 0

        # Spread regime
        regime[i, 6] = rolling_mean(spread_cumsum, eidx, 200)

        # OFI regime (directional pressure)
        regime[i, 7] = rolling_mean(ofi_cumsum, eidx, 200)

    # Clip extreme values
    regime = np.clip(regime, -5.0, 5.0)

    return regime


# ============================================================
# Model Definition — Mixture of Experts
# ============================================================
class ExpertMLP(nn.Module):
    """Single expert network."""
    def __init__(self, input_dim, hidden_dims=[128, 64], dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class RegimeMoE(nn.Module):
    """
    Mixture-of-Experts with regime-conditioned gating.

    Expert inputs: 31 features (CNN-Mamba + microstructure)
    Gating inputs: 8 regime features
    Output: weighted combination of expert predictions
    """
    def __init__(self, expert_input_dim, regime_dim, n_experts=2,
                 expert_hidden_dims=[128, 64], dropout=0.2):
        super().__init__()
        self.n_experts = n_experts

        # Expert networks
        self.experts = nn.ModuleList([
            ExpertMLP(expert_input_dim, expert_hidden_dims, dropout)
            for _ in range(n_experts)
        ])

        # Gating network: regime features -> expert weights
        # Small network to avoid overfitting on regime features
        self.gate = nn.Sequential(
            nn.Linear(regime_dim, 16),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(16, n_experts),
        )

    def forward(self, expert_input, regime_input):
        """
        expert_input: (B, 31) — CNN-Mamba + microstructure
        regime_input: (B, 8)  — regime features for gating
        Returns: predictions (B,), gate_weights (B, n_experts)
        """
        # Gating weights
        gate_logits = self.gate(regime_input)          # (B, n_experts)
        gate_weights = torch.softmax(gate_logits, dim=-1)  # (B, n_experts)

        # Expert predictions
        expert_outputs = torch.stack([
            expert(expert_input) for expert in self.experts
        ], dim=-1)  # (B, n_experts)

        # Weighted combination
        output = (expert_outputs * gate_weights).sum(dim=-1)  # (B,)

        return output, gate_weights

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# Data Loading & Alignment
# ============================================================
def get_overlapping_dates():
    """Find dates where CNN-Mamba AND MBO both have data."""
    cm_dates = set(f[:8] for f in os.listdir(CONFIG['cm_dir'])
                   if f.endswith('_predictions.npz') and f[0] == '2')
    mbo_dates = set(f[:8] for f in os.listdir(CONFIG['mbo_dir'])
                    if f.endswith('_mbo_events.npz'))
    overlap = sorted(cm_dates & mbo_dates)
    logger.info(f"CNN-Mamba dates: {len(cm_dates)}, MBO: {len(mbo_dates)}, "
                f"overlap: {len(overlap)}")
    return overlap


def load_date_features(date_str):
    """
    Load CNN-Mamba + MBO features + regime features for a single date.

    Returns:
      expert_features: (N, 31) — same as v7
      regime_features: (N, 8)  — new regime conditioning features
      target: (N,)             — 1s mid-price move
    """
    # Load CNN-Mamba predictions
    cm_path = os.path.join(CONFIG['cm_dir'], f'{date_str}_predictions.npz')
    cm_data = np.load(cm_path, allow_pickle=True)
    cm_preds = cm_data['predictions']  # (N_cm, 3) for 1s/5s/10s
    cm_n = len(cm_preds)
    cm_window = int(cm_data['window_size'])
    cm_stride = int(cm_data['stride'])

    # CNN-Mamba event indices
    cm_event_idx = np.array([cm_window + i * cm_stride for i in range(cm_n)])

    # Load MBO events
    mbo_path = os.path.join(CONFIG['mbo_dir'], f'{date_str}_mbo_events.npz')
    try:
        mbo_data = np.load(mbo_path)
    except (zipfile.BadZipFile, Exception) as e:
        logger.warning(f"  {date_str}: Bad MBO file ({e}), skip")
        return None, None, None
    mbo_events = mbo_data['events']        # (N_mbo, 25)
    mbo_labels_1s = mbo_data['labels_1s']  # target
    n_mbo = len(mbo_events)

    # Clamp indices
    valid_cm = cm_event_idx < n_mbo
    cm_event_idx = cm_event_idx[valid_cm]
    cm_preds = cm_preds[valid_cm]
    cm_n = len(cm_preds)

    if cm_n < 10:
        logger.warning(f"  {date_str}: Only {cm_n} valid rows, skipping")
        return None, None, None

    # Extract microstructure at CNN-Mamba positions
    mbo_at_cm = mbo_events[cm_event_idx]  # (cm_n, 25)

    # Target
    target = mbo_labels_1s[cm_event_idx]

    # CNN-Mamba confidence
    cm_confidence = np.abs(cm_preds)

    # Expert features (31 total, same as v7)
    expert_features = np.column_stack([
        cm_preds,       # 3
        cm_confidence,  # 3
        mbo_at_cm,      # 25
    ]).astype(np.float32)

    # Regime features (8) — computed from rolling stats of MBO events
    regime_features = compute_regime_features(mbo_events, cm_event_idx)

    return expert_features, regime_features, target.astype(np.float32)


# ============================================================
# Training
# ============================================================
def train_fold(model, train_expert, train_regime, train_y,
               val_expert, val_regime, val_y, device, fold_idx):
    """Train one fold with early stopping."""
    train_ds = TensorDataset(
        torch.tensor(train_expert, dtype=torch.float32),
        torch.tensor(train_regime, dtype=torch.float32),
        torch.tensor(train_y, dtype=torch.float32)
    )
    val_ds = TensorDataset(
        torch.tensor(val_expert, dtype=torch.float32),
        torch.tensor(val_regime, dtype=torch.float32),
        torch.tensor(val_y, dtype=torch.float32)
    )
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'],
                               shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'] * 2,
                             shuffle=False, num_workers=8, pin_memory=True)

    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'],
                                   weight_decay=CONFIG['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(CONFIG['epochs']):
        # Train
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        gate_entropy_sum = 0.0
        for X_exp, X_reg, y_batch in train_loader:
            X_exp = X_exp.to(device)
            X_reg = X_reg.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            pred, gate_w = model(X_exp, X_reg)

            # Main loss
            loss = criterion(pred, y_batch)

            # Auxiliary: encourage gating diversity (entropy bonus)
            # Higher entropy = more balanced expert usage = less mode collapse
            gate_ent = -(gate_w * torch.log(gate_w + 1e-8)).sum(dim=-1).mean()
            total_loss = loss - CONFIG['gate_entropy_weight'] * gate_ent

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += loss.item() * len(y_batch)
            gate_entropy_sum += gate_ent.item() * len(y_batch)
            train_n += len(y_batch)

        scheduler.step()

        # Validate
        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        val_preds_list = []
        val_labels_list = []
        val_gate_w_list = []
        with torch.no_grad():
            for X_exp, X_reg, y_batch in val_loader:
                X_exp = X_exp.to(device)
                X_reg = X_reg.to(device)
                y_batch = y_batch.to(device)
                pred, gate_w = model(X_exp, X_reg)
                loss = criterion(pred, y_batch)
                val_loss_sum += loss.item() * len(y_batch)
                val_n += len(y_batch)
                val_preds_list.append(pred.cpu().numpy())
                val_labels_list.append(y_batch.cpu().numpy())
                val_gate_w_list.append(gate_w.cpu().numpy())

        train_loss = train_loss_sum / max(train_n, 1)
        val_loss = val_loss_sum / max(val_n, 1)
        avg_gate_ent = gate_entropy_sum / max(train_n, 1)

        val_preds_arr = np.concatenate(val_preds_list)
        val_labels_arr = np.concatenate(val_labels_list)
        val_gate_arr = np.concatenate(val_gate_w_list)
        spearman_r, _ = stats.spearmanr(val_preds_arr, val_labels_arr)

        # Expert usage stats
        expert_usage = val_gate_arr.mean(axis=0)

        if epoch % 5 == 0 or epoch == CONFIG['epochs'] - 1:
            usage_str = '/'.join(f'{u:.2f}' for u in expert_usage)
            logger.info(f"  Fold {fold_idx} Epoch {epoch}: "
                        f"train={train_loss:.6f}, val={val_loss:.6f}, "
                        f"spearman={spearman_r:.4f}, gate_ent={avg_gate_ent:.4f}, "
                        f"expert_usage=[{usage_str}]")
            sys.stdout.flush()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['patience']:
                logger.info(f"  Fold {fold_idx}: Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_val_loss


def evaluate_fold(model, eval_expert, eval_regime, eval_y, device):
    """Evaluate model on OOT data. Returns predictions and gate weights."""
    model.eval()
    ds = TensorDataset(
        torch.tensor(eval_expert, dtype=torch.float32),
        torch.tensor(eval_regime, dtype=torch.float32),
        torch.tensor(eval_y, dtype=torch.float32)
    )
    loader = DataLoader(ds, batch_size=CONFIG['batch_size'] * 2,
                         shuffle=False, num_workers=0)
    all_preds = []
    all_gates = []
    with torch.no_grad():
        for X_exp, X_reg, y_batch in loader:
            X_exp = X_exp.to(device)
            X_reg = X_reg.to(device)
            pred, gate_w = model(X_exp, X_reg)
            all_preds.append(pred.cpu().numpy())
            all_gates.append(gate_w.cpu().numpy())
    return np.concatenate(all_preds), np.concatenate(all_gates)


def compute_metrics(predictions, labels, prefix=""):
    """Compute evaluation metrics (same as v7)."""
    spearman_r, _ = stats.spearmanr(predictions, labels)
    pearson_r, _ = stats.pearsonr(predictions, labels)

    comm = CONFIG['commission_ticks']
    results = {
        f'{prefix}spearman': spearman_r,
        f'{prefix}pearson': pearson_r,
        f'{prefix}n_samples': len(predictions),
    }

    for pct_name, pct in [('top5', 95), ('top10', 90), ('top20', 80), ('top50', 50)]:
        threshold = np.percentile(np.abs(predictions), pct)
        mask = np.abs(predictions) >= threshold
        n_trades = mask.sum()
        if n_trades < 5:
            results[f'{prefix}{pct_name}_net_ticks'] = 0.0
            results[f'{prefix}{pct_name}_n_trades'] = 0
            results[f'{prefix}{pct_name}_wr'] = 0.0
            continue

        trade_pnl = np.sign(predictions[mask]) * labels[mask] - comm
        net_ticks = trade_pnl.sum()
        avg_ticks = trade_pnl.mean()
        win_rate = (trade_pnl > 0).mean()
        direction_correct = (np.sign(predictions[mask]) == np.sign(labels[mask]))
        precision = direction_correct.mean()

        results[f'{prefix}{pct_name}_net_ticks'] = float(net_ticks)
        results[f'{prefix}{pct_name}_avg_ticks'] = float(avg_ticks)
        results[f'{prefix}{pct_name}_n_trades'] = int(n_trades)
        results[f'{prefix}{pct_name}_wr'] = float(win_rate)
        results[f'{prefix}{pct_name}_precision'] = float(precision)

    return results


# ============================================================
# Regime-Stratified Analysis
# ============================================================
def regime_stratified_analysis(concat_preds, concat_labels, concat_regime, concat_dates):
    """
    Stratify results by regime (high-vol vs low-vol) for HC #428 compliance.
    """
    logger.info(f"\n{'='*60}")
    logger.info("REGIME-STRATIFIED ANALYSIS (HC #428 R1)")
    logger.info(f"{'='*60}")

    # Use vol_regime_med (col 1 of regime features) as the regime indicator
    vol_regime = concat_regime[:, 1]  # rolling 200-event vol
    vol_median = np.median(vol_regime)

    high_vol_mask = vol_regime >= vol_median
    low_vol_mask = vol_regime < vol_median

    for name, mask in [('HIGH_VOL', high_vol_mask), ('LOW_VOL', low_vol_mask)]:
        if mask.sum() < 100:
            logger.info(f"  {name}: too few samples ({mask.sum()}), skip")
            continue

        m_preds = concat_preds[mask]
        m_labels = concat_labels[mask]
        m_metrics = compute_metrics(m_preds, m_labels, prefix=f'{name}_')

        sp = m_metrics[f'{name}_spearman']
        logger.info(f"  {name}: n={mask.sum()}, Spearman={sp:.4f}")

        # Compute Sharpe for top20
        threshold = np.percentile(np.abs(m_preds), 80)
        trade_mask = np.abs(m_preds) >= threshold
        if trade_mask.sum() > 10:
            pnl = np.sign(m_preds[trade_mask]) * m_labels[trade_mask] - CONFIG['commission_ticks']
            sharpe = pnl.mean() / (pnl.std() + 1e-8) * np.sqrt(252)
            wr = (pnl > 0).mean()
            logger.info(f"    top20: Sharpe={sharpe:.2f}, WR={wr:.3f}, n={trade_mask.sum()}")
        else:
            sharpe = 0.0

    # Gate analysis by regime
    logger.info(f"\nGating behavior by regime:")
    for name, mask in [('HIGH_VOL', high_vol_mask), ('LOW_VOL', low_vol_mask)]:
        if mask.sum() < 10:
            continue
        gate_in_regime = concat_regime[mask]
        # We stored full gate weights separately, but here we just show regime feature stats
        logger.info(f"  {name}: avg regime features = {concat_regime[mask].mean(axis=0)}")


# ============================================================
# Main Walk-Forward Loop
# ============================================================
def main():
    logger.info("=" * 60)
    logger.info("Meta-Model v8: Regime-Conditioned Mixture-of-Experts")
    logger.info("=" * 60)
    logger.info("v8 KEY: MoE with regime gating — two expert MLPs routed by vol/trend/spread regime")
    logger.info("v7 baseline: Spearman 0.308, 27/27 positive days")
    logger.info(f"Expert features: 31 (same as v7) | Regime features: {CONFIG['n_regime_features']}")
    logger.info(f"Walk-forward: {CONFIG['train_days']}d train / {CONFIG['eval_days']}d eval (SLIDING)")
    logger.info(f"Config: {json.dumps(CONFIG, indent=2, default=str)}")
    sys.stdout.flush()

    # MLflow setup
    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("meta_v8_regime_moe")
        mlflow.start_run(run_name=f"meta_v8_moe_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        mlflow.log_params({
            'model_type': 'mixture_of_experts',
            'target_horizon': '1s',
            'train_days': CONFIG['train_days'],
            'eval_days': CONFIG['eval_days'],
            'expert_hidden_dims': str(CONFIG['expert_hidden_dims']),
            'n_experts': CONFIG['n_experts'],
            'n_regime_features': CONFIG['n_regime_features'],
            'dropout': CONFIG['dropout'],
            'batch_size': CONFIG['batch_size'],
            'epochs': CONFIG['epochs'],
            'lr': CONFIG['lr'],
            'weight_decay': CONFIG['weight_decay'],
            'patience': CONFIG['patience'],
            'gate_entropy_weight': CONFIG['gate_entropy_weight'],
            'commission_ticks': CONFIG['commission_ticks'],
            'v7_baseline_spearman': CONFIG['v7_spearman'],
        })
        logger.info("MLflow tracking enabled")
    else:
        logger.warning("MLflow not available — training without tracking")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("No CUDA device — training will be slow")
    sys.stdout.flush()

    # Get overlapping dates
    dates = get_overlapping_dates()
    if len(dates) < CONFIG['train_days'] + CONFIG['eval_days']:
        logger.error(f"Not enough dates ({len(dates)}) for walk-forward")
        return

    # Pre-load all date data
    logger.info("Loading all date data (expert + regime features)...")
    sys.stdout.flush()
    date_data = {}
    valid_dates = []
    for d in dates:
        expert_feat, regime_feat, target = load_date_features(d)
        if expert_feat is None:
            continue
        if np.nanstd(target) < 1e-6:
            logger.warning(f"  {d}: ZERO-VARIANCE target, REJECTED")
            continue
        valid_mask = ~np.isnan(target)
        if valid_mask.mean() < 0.5:
            logger.warning(f"  {d}: Too many NaN targets, REJECTED")
            continue
        expert_feat = expert_feat[valid_mask]
        regime_feat = regime_feat[valid_mask]
        target = target[valid_mask]
        expert_feat = np.nan_to_num(expert_feat, nan=0.0, posinf=5.0, neginf=-5.0)
        regime_feat = np.nan_to_num(regime_feat, nan=0.0, posinf=5.0, neginf=-5.0)
        date_data[d] = (expert_feat, regime_feat, target)
        valid_dates.append(d)
        logger.info(f"  Loaded {d}: {len(expert_feat)} samples, "
                    f"target_std={np.std(target):.4f}, "
                    f"regime_vol_mean={np.mean(regime_feat[:,1]):.3f}")
        sys.stdout.flush()

    logger.info(f"Valid dates after filtering: {len(valid_dates)}")

    if len(valid_dates) < CONFIG['train_days'] + CONFIG['eval_days']:
        logger.error(f"Not enough valid dates ({len(valid_dates)})")
        return

    expert_dim = date_data[valid_dates[0]][0].shape[1]
    regime_dim = date_data[valid_dates[0]][1].shape[1]
    logger.info(f"Expert input dim: {expert_dim} (expected 31)")
    logger.info(f"Regime input dim: {regime_dim} (expected {CONFIG['n_regime_features']})")
    sys.stdout.flush()

    # Walk-forward: SLIDING window
    train_days = CONFIG['train_days']
    eval_days = CONFIG['eval_days']
    n_folds = (len(valid_dates) - train_days) // eval_days

    logger.info(f"Walk-forward: {n_folds} folds, {train_days}d train / {eval_days}d eval")
    sys.stdout.flush()

    if MLFLOW_AVAILABLE:
        mlflow.log_metrics({
            'n_valid_dates': len(valid_dates),
            'n_folds': n_folds,
            'expert_dim': expert_dim,
            'regime_dim': regime_dim,
        })

    all_oot_preds = []
    all_oot_labels = []
    all_oot_dates = []
    all_oot_gates = []
    all_oot_regime = []
    fold_results = []

    for fold_idx in range(n_folds):
        fold_start = fold_idx * eval_days
        train_date_slice = valid_dates[fold_start:fold_start + train_days]
        eval_start = fold_start + train_days
        eval_end = min(eval_start + eval_days, len(valid_dates))
        eval_date_slice = valid_dates[eval_start:eval_end]

        if len(eval_date_slice) == 0:
            break

        logger.info(f"\n{'='*40}")
        logger.info(f"Fold {fold_idx}: Train {train_date_slice[0]}..{train_date_slice[-1]} "
                     f"({len(train_date_slice)}d) -> Eval {eval_date_slice[0]}..{eval_date_slice[-1]} "
                     f"({len(eval_date_slice)}d)")
        sys.stdout.flush()

        # Assemble train data
        train_expert = np.concatenate([date_data[d][0] for d in train_date_slice])
        train_regime = np.concatenate([date_data[d][1] for d in train_date_slice])
        train_y = np.concatenate([date_data[d][2] for d in train_date_slice])

        # Normalize expert features using train statistics
        expert_mean = train_expert.mean(axis=0)
        expert_std = train_expert.std(axis=0) + 1e-8
        train_expert_norm = (train_expert - expert_mean) / expert_std

        # Normalize regime features separately
        regime_mean = train_regime.mean(axis=0)
        regime_std = train_regime.std(axis=0) + 1e-8
        train_regime_norm = (train_regime - regime_mean) / regime_std

        logger.info(f"  Train: {len(train_expert)} samples")

        # Build and train model
        model = RegimeMoE(
            expert_input_dim=expert_dim,
            regime_dim=regime_dim,
            n_experts=CONFIG['n_experts'],
            expert_hidden_dims=CONFIG['expert_hidden_dims'],
            dropout=CONFIG['dropout']
        ).to(device)

        if fold_idx == 0:
            n_params = model.count_parameters()
            logger.info(f"  Model params: {n_params:,}")
            if MLFLOW_AVAILABLE:
                mlflow.log_metric('model_params', n_params)

        # Use last 20% of train as validation for early stopping
        val_split = len(train_expert) // 5
        best_val_loss = train_fold(
            model,
            train_expert_norm, train_regime_norm, train_y,
            train_expert_norm[-val_split:], train_regime_norm[-val_split:], train_y[-val_split:],
            device, fold_idx
        )

        # Save model weights
        weight_path = os.path.join(CONFIG['output_dir'], f'fold_{fold_idx:02d}_model.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'expert_mean': expert_mean,
            'expert_std': expert_std,
            'regime_mean': regime_mean,
            'regime_std': regime_std,
            'expert_dim': expert_dim,
            'regime_dim': regime_dim,
            'config': CONFIG,
            'train_dates': train_date_slice,
            'eval_dates': eval_date_slice,
        }, weight_path)

        # Evaluate on OOT dates
        fold_preds = []
        fold_labels = []
        fold_date_labels = []
        fold_gates = []
        fold_regime = []
        for eval_date in eval_date_slice:
            eval_expert_raw, eval_regime_raw, eval_y = date_data[eval_date]
            eval_expert_norm = (eval_expert_raw - expert_mean) / expert_std
            eval_regime_norm = (eval_regime_raw - regime_mean) / regime_std
            preds, gates = evaluate_fold(model, eval_expert_norm, eval_regime_norm, eval_y, device)
            fold_preds.append(preds)
            fold_labels.append(eval_y)
            fold_date_labels.extend([eval_date] * len(preds))
            fold_gates.append(gates)
            fold_regime.append(eval_regime_raw)  # raw for stratification

        fold_preds = np.concatenate(fold_preds)
        fold_labels = np.concatenate(fold_labels)
        fold_gates = np.concatenate(fold_gates)
        fold_regime_arr = np.concatenate(fold_regime)

        # Save fold OOT predictions
        pred_path = os.path.join(CONFIG['output_dir'], f'fold_{fold_idx:02d}_oot_predictions.npz')
        np.savez_compressed(pred_path,
                            predictions=fold_preds,
                            labels=fold_labels,
                            dates=np.array(fold_date_labels),
                            gate_weights=fold_gates,
                            regime_features=fold_regime_arr,
                            expert_mean=expert_mean,
                            expert_std=expert_std,
                            regime_mean=regime_mean,
                            regime_std=regime_std)

        # Metrics
        metrics = compute_metrics(fold_preds, fold_labels, prefix=f'fold{fold_idx}_')
        fold_results.append(metrics)

        fold_sp = metrics[f'fold{fold_idx}_spearman']
        expert_usage = fold_gates.mean(axis=0)
        usage_str = '/'.join(f'{u:.3f}' for u in expert_usage)
        logger.info(f"  Fold {fold_idx} OOT: Spearman={fold_sp:.4f}, "
                     f"expert_usage=[{usage_str}], "
                     f"n={metrics[f'fold{fold_idx}_n_samples']}")
        for pct in ['top5', 'top10', 'top20']:
            key = f'fold{fold_idx}_{pct}_net_ticks'
            if key in metrics:
                logger.info(f"    {pct}: net={metrics[key]:.2f}t, "
                            f"WR={metrics.get(f'fold{fold_idx}_{pct}_wr', 0):.3f}, "
                            f"n={metrics.get(f'fold{fold_idx}_{pct}_n_trades', 0)}")
        sys.stdout.flush()

        if MLFLOW_AVAILABLE:
            mlflow.log_metrics({
                f'fold_{fold_idx}_spearman': fold_sp,
                f'fold_{fold_idx}_val_loss': best_val_loss,
                f'fold_{fold_idx}_expert0_usage': float(expert_usage[0]),
                f'fold_{fold_idx}_expert1_usage': float(expert_usage[1]),
            }, step=fold_idx)

        all_oot_preds.append(fold_preds)
        all_oot_labels.append(fold_labels)
        all_oot_dates.extend(fold_date_labels)
        all_oot_gates.append(fold_gates)
        all_oot_regime.append(fold_regime_arr)

    # ============================================================
    # Concat OOT Analysis
    # ============================================================
    if all_oot_preds:
        concat_preds = np.concatenate(all_oot_preds)
        concat_labels = np.concatenate(all_oot_labels)
        concat_dates = np.array(all_oot_dates)
        concat_gates = np.concatenate(all_oot_gates)
        concat_regime = np.concatenate(all_oot_regime)

        logger.info(f"\n{'='*60}")
        logger.info("CONCAT OOT RESULTS (all folds)")
        logger.info(f"{'='*60}")

        concat_metrics = compute_metrics(concat_preds, concat_labels, prefix='concat_')

        logger.info(f"Total samples: {concat_metrics['concat_n_samples']}")
        logger.info(f"Spearman: {concat_metrics['concat_spearman']:.4f}")
        logger.info(f"Pearson:  {concat_metrics['concat_pearson']:.4f}")

        # Expert usage summary
        avg_gates = concat_gates.mean(axis=0)
        logger.info(f"Avg expert usage: Expert0={avg_gates[0]:.3f}, Expert1={avg_gates[1]:.3f}")

        for pct in ['top5', 'top10', 'top20', 'top50']:
            key = f'concat_{pct}_net_ticks'
            if key in concat_metrics:
                logger.info(f"{pct}: net={concat_metrics[key]:.2f}t, "
                            f"avg={concat_metrics.get(f'concat_{pct}_avg_ticks', 0):.4f}, "
                            f"WR={concat_metrics.get(f'concat_{pct}_wr', 0):.3f}, "
                            f"precision={concat_metrics.get(f'concat_{pct}_precision', 0):.3f}, "
                            f"n={concat_metrics.get(f'concat_{pct}_n_trades', 0)}")

        # v7 vs v8 comparison
        v7_spearman = CONFIG['v7_spearman']
        v8_spearman = concat_metrics['concat_spearman']
        delta = v8_spearman - v7_spearman
        pct_change = (delta / abs(v7_spearman)) * 100 if v7_spearman != 0 else 0.0

        logger.info(f"\n{'='*60}")
        logger.info("v7 (single MLP) vs v8 (regime MoE) COMPARISON")
        logger.info(f"{'='*60}")
        logger.info(f"v7 Spearman: {v7_spearman:.4f}")
        logger.info(f"v8 Spearman: {v8_spearman:.4f}")
        logger.info(f"Delta: {delta:+.4f} ({pct_change:+.1f}%)")
        if delta > 0:
            logger.info("RESULT: v8 MoE BEATS v7 — regime conditioning adds value")
        elif delta > -0.02:
            logger.info("RESULT: v8 roughly MATCHES v7 — check regime-stratified metrics")
        else:
            logger.info(f"RESULT: v8 underperforms v7 by {abs(delta):.4f}")

        # Regime-stratified analysis (HC #428 R1)
        regime_stratified_analysis(concat_preds, concat_labels, concat_regime, concat_dates)

        # Per-date breakdown
        logger.info(f"\nPer-date OOT breakdown:")
        unique_dates = sorted(set(all_oot_dates))
        date_sharpes = []
        positive_dates = 0
        for d in unique_dates:
            mask = concat_dates == d
            d_preds = concat_preds[mask]
            d_labels = concat_labels[mask]
            d_metrics = compute_metrics(d_preds, d_labels)
            d_spearman = d_metrics.get('spearman', 0)

            threshold = np.percentile(np.abs(d_preds), 80)
            d_mask = np.abs(d_preds) >= threshold
            if d_mask.sum() > 5:
                d_pnl = np.sign(d_preds[d_mask]) * d_labels[d_mask] - CONFIG['commission_ticks']
                d_sharpe = d_pnl.mean() / (d_pnl.std() + 1e-8) * np.sqrt(252)
                date_sharpes.append(d_sharpe)
                if d_spearman > 0:
                    positive_dates += 1

                # Gate usage for this day
                d_gates = concat_gates[mask]
                d_gate_avg = d_gates.mean(axis=0)
            else:
                d_sharpe = 0.0
                d_gate_avg = np.zeros(CONFIG['n_experts'])

            logger.info(f"  {d}: n={mask.sum()}, spearman={d_spearman:.4f}, "
                        f"sharpe={d_sharpe:.2f}, "
                        f"gates=[{d_gate_avg[0]:.2f}/{d_gate_avg[1]:.2f}]")

        if date_sharpes:
            avg_sharpe = np.mean(date_sharpes)
            logger.info(f"\nAvg daily Sharpe (top20%): {avg_sharpe:.2f}")
            logger.info(f"Positive Spearman days: {positive_dates}/{len(unique_dates)}")
            pos_sharpe_days = sum(1 for s in date_sharpes if s > 0)
            logger.info(f"Positive Sharpe days: {pos_sharpe_days}/{len(date_sharpes)}")

        sys.stdout.flush()

        if MLFLOW_AVAILABLE:
            mlflow.log_metrics({
                'concat_spearman': float(v8_spearman),
                'concat_pearson': float(concat_metrics['concat_pearson']),
                'v7_v8_delta': float(delta),
                'n_oot_dates': len(unique_dates),
                'positive_spearman_dates': positive_dates,
                'avg_daily_sharpe_top20': float(avg_sharpe) if date_sharpes else 0.0,
                'avg_expert0_usage': float(avg_gates[0]),
                'avg_expert1_usage': float(avg_gates[1]),
            })

        # Save concat predictions
        concat_path = os.path.join(CONFIG['output_dir'], 'concat_oot_predictions.npz')
        np.savez_compressed(concat_path,
                            predictions=concat_preds,
                            labels=concat_labels,
                            dates=concat_dates,
                            gate_weights=concat_gates,
                            regime_features=concat_regime)

        # Save summary
        summary = {
            'version': 'v8_regime_moe',
            'description': 'Meta-model v8: Mixture-of-Experts with regime-conditioned gating',
            'target_horizon': '1s',
            'expert_input_dim': expert_dim,
            'regime_input_dim': regime_dim,
            'n_experts': CONFIG['n_experts'],
            'config': {k: str(v) if not isinstance(v, (int, float, str, list)) else v
                       for k, v in CONFIG.items()},
            'n_folds': len(fold_results),
            'n_valid_dates': len(valid_dates),
            'n_oot_dates': len(unique_dates),
            'v7_baseline_spearman': v7_spearman,
            'v8_spearman': float(v8_spearman),
            'v8_vs_v7_delta': float(delta),
            'avg_expert_usage': [float(g) for g in avg_gates],
            'concat_metrics': {k: float(v) if isinstance(v, (float, np.floating)) else v
                               for k, v in concat_metrics.items()},
            'fold_results': [{k: float(v) if isinstance(v, (float, np.floating)) else v
                              for k, v in fr.items()} for fr in fold_results],
            'date_sharpes': {d: float(s) for d, s in zip(unique_dates, date_sharpes)} if date_sharpes else {},
            'positive_spearman_days': positive_dates,
            'total_oot_days': len(unique_dates),
            'timestamp': datetime.now().isoformat(),
        }
        summary_path = os.path.join(CONFIG['output_dir'], 'training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"\nSaved summary and concat predictions")

        if MLFLOW_AVAILABLE:
            mlflow.log_artifact(summary_path)
            mlflow.log_artifact(log_path)

    if MLFLOW_AVAILABLE:
        mlflow.end_run()

    logger.info("\n" + "=" * 60)
    logger.info("Meta-Model v8 (Regime MoE) — Training Complete")
    logger.info("=" * 60)
    sys.stdout.flush()


if __name__ == '__main__':
    t0 = time.time()
    main()
    elapsed = time.time() - t0
    logger.info(f"Total runtime: {elapsed:.1f}s ({elapsed/60:.1f}m)")
    sys.stdout.flush()
