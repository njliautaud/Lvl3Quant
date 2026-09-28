#!/usr/bin/env python3
"""
queue_entry_selector_mlp_v1.py — MLP Queue Entry Selection (GPU)
================================================================

PyTorch MLP version of queue_entry_selector_v2, designed for Neptune RTX 3090.
Same data loading, feature engineering, and walk-forward setup as v2,
with additional time-of-day + side features.

Architecture: [input -> 128 -> BN -> ReLU -> Drop(0.3)
                     -> 64  -> BN -> ReLU -> Drop(0.2)
                     -> 32  -> BN -> ReLU -> 1 -> Sigmoid]

Walk-forward: 25d train, 5d OOT, slide 5d (SLIDING per HC #0)
Regime gate: |Sharpe_green - Sharpe_red| / max(...) <= 0.50 (HC #428)
Commission: 0.376 ticks RT
"""
import os
import sys
import json
import logging
import warnings
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score
from scipy import stats as scipy_stats

warnings.filterwarnings('ignore')

# ── Paths ──
if Path("/home/nick/Lvl3Quant").exists():
    BASE = Path("/home/nick/Lvl3Quant")
elif Path("/home/jupiter/Lvl3Quant").exists():
    BASE = Path("/home/jupiter/Lvl3Quant")
else:
    raise RuntimeError("Cannot find Lvl3Quant directory")

QUEUE_DIR = BASE / "output" / "queue_features_universal"
FIFO_DIR = BASE / "data" / "processed" / "mbo_events_smart_v3_fifo_labels"
OUTPUT_DIR = BASE / "output" / "queue_entry_selector_mlp_v1"
LOG_PATH = BASE / "logs" / "queue_entry_selector_mlp_v1.log"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(BASE / "logs").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)

# ── Config ──
ES_TICK_SIZE = 0.25
COMMISSION_RT_TICKS = 0.376

FIFO_CONFIGS = ['tp4sl3', 'tp8sl5']

# Walk-forward
TRAIN_DAYS = 25
OOT_DAYS = 5
SLIDE_DAYS = 5

# MLP training
BATCH_SIZE = 4096
MAX_EPOCHS = 50
EARLY_STOP_PATIENCE = 10
LR = 1e-3
VAL_FRAC = 0.20  # last 20% of training data for validation
SEED = 42

# RTH session boundaries (nanoseconds offset from midnight ET)
RTH_OPEN_NS = int(9.5 * 3600 * 1e9)   # 9:30 ET
RTH_CLOSE_NS = int(16 * 3600 * 1e9)    # 16:00 ET
RTH_DURATION_NS = RTH_CLOSE_NS - RTH_OPEN_NS

# Features from v2
RAW_FEATURES = [
    'ofi_10s', 'ofi_5s', 'ofi_1s',
    'top_imbalance', 'microprice_offset_ticks',
    'bid_qty_at_touch', 'ask_qty_at_touch',
    'bid_trade_rate_1s', 'ask_trade_rate_1s',
    'bid_add_rate_1s', 'ask_add_rate_1s',
    'bid_cancel_rate_1s', 'ask_cancel_rate_1s',
    'bid_level_age_s', 'ask_level_age_s',
]

FLIP_FEATURES = ['ofi_10s', 'ofi_5s', 'ofi_1s', 'top_imbalance', 'microprice_offset_ticks',
                  'queue_diff', 'net_flow_diff', 'trade_imbalance', 'ofi_momentum']
SWAP_PAIRS = [
    ('bid_qty_at_touch', 'ask_qty_at_touch'),
    ('bid_trade_rate_1s', 'ask_trade_rate_1s'),
    ('bid_add_rate_1s', 'ask_add_rate_1s'),
    ('bid_cancel_rate_1s', 'ask_cancel_rate_1s'),
    ('bid_level_age_s', 'ask_level_age_s'),
]

THRESHOLDS = [0.50, 0.52, 0.55, 0.58, 0.60, 0.65]


# ─────────────────────────────────────
# MLP Model
# ─────────────────────────────────────

class QueueEntryMLP(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ─────────────────────────────────────
# Data Loading (same as v2)
# ─────────────────────────────────────

def load_queue_features(date_str):
    path = QUEUE_DIR / f"features_{date_str}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if 'ts_ns' not in df.columns or len(df) < 100:
        return None
    df = df.sort_values('ts_ns').reset_index(drop=True)
    return df


def load_fifo_labels(date_str, fifo_config):
    path = FIFO_DIR / f"{date_str}_fifo_labels.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)

    prefix = fifo_config
    required = [f'{prefix}_long_filled', f'{prefix}_long_net_ticks',
                f'{prefix}_short_filled', f'{prefix}_short_net_ticks']

    if not all(k in data for k in ['ts_ns'] + required):
        return None

    cols = {'ts_ns': data['ts_ns']}
    for side in ['long', 'short']:
        for suffix in ['filled', 'net_ticks', 'hit_tp', 'exit_reason']:
            key = f'{prefix}_{side}_{suffix}'
            if key in data:
                cols[f'{side}_{suffix}'] = data[key]

    return pd.DataFrame(cols)


def join_queue_fifo(queue_df, fifo_df):
    queue_df = queue_df.sort_values('ts_ns').reset_index(drop=True)
    fifo_df = fifo_df.sort_values('ts_ns').reset_index(drop=True)

    merged = pd.merge_asof(
        fifo_df, queue_df,
        on='ts_ns',
        direction='backward',
        tolerance=1_500_000_000
    )

    n_before = len(merged)
    merged = merged.dropna(subset=['ofi_10s'])
    n_after = len(merged)
    return merged, n_before, n_after


def engineer_features(df):
    bid_q = df['bid_qty_at_touch'].clip(lower=1)
    ask_q = df['ask_qty_at_touch'].clip(lower=1)
    df['queue_ratio'] = bid_q / (bid_q + ask_q)
    df['queue_diff'] = df['bid_qty_at_touch'] - df['ask_qty_at_touch']

    bid_net = df.get('bid_add_rate_1s', 0) - df.get('bid_cancel_rate_1s', 0)
    ask_net = df.get('ask_add_rate_1s', 0) - df.get('ask_cancel_rate_1s', 0)
    df['net_flow_diff'] = bid_net - ask_net

    df['trade_imbalance'] = df['bid_trade_rate_1s'] - df['ask_trade_rate_1s']
    df['ofi_momentum'] = df['ofi_10s'] - df['ofi_1s']
    df['level_age_diff'] = df['bid_level_age_s'] - df['ask_level_age_s']
    return df


def compute_time_features(ts_ns_series, date_str):
    """Compute time_of_day (seconds since 9:30 ET) and session_pct (0-1)."""
    # Parse date to get midnight reference
    dt = pd.Timestamp(date_str, tz='US/Eastern')
    midnight_ns = int(dt.value)  # ns since epoch at midnight ET

    # Time of day in seconds since 9:30 ET
    rth_open_sec = 9.5 * 3600  # 9:30 = 34200 seconds from midnight
    elapsed_ns = ts_ns_series - midnight_ns
    elapsed_sec = elapsed_ns / 1e9
    time_of_day = (elapsed_sec - rth_open_sec).clip(lower=0)

    # Session percentage (0 at 9:30, 1 at 16:00)
    session_duration_sec = 6.5 * 3600  # 6.5 hours
    session_pct = (time_of_day / session_duration_sec).clip(0, 1)

    return time_of_day.astype(np.float32), session_pct.astype(np.float32)


def prepare_training_data(merged_df, date_str):
    rows = []
    for side in ['long', 'short']:
        filled_col = f'{side}_filled'
        net_col = f'{side}_net_ticks'

        mask = merged_df[filled_col] == True
        subset = merged_df[mask].copy()
        if len(subset) == 0:
            continue

        subset['target'] = (subset[net_col] > 0).astype(int)
        subset['net_ticks'] = subset[net_col].astype(float)
        subset['side'] = side
        subset['side_indicator'] = 0 if side == 'short' else 1
        subset['date'] = date_str

        # Time features
        tod, spct = compute_time_features(subset['ts_ns'], date_str)
        subset['time_of_day'] = tod
        subset['session_pct'] = spct

        feature_cols = RAW_FEATURES + ['queue_ratio', 'queue_diff', 'net_flow_diff',
                                        'trade_imbalance', 'ofi_momentum', 'level_age_diff']

        keep_cols = (['ts_ns', 'target', 'net_ticks', 'side', 'side_indicator', 'date',
                      'time_of_day', 'session_pct'] +
                     [c for c in feature_cols if c in subset.columns])
        row = subset[keep_cols].copy()

        if side == 'short':
            for feat in FLIP_FEATURES:
                if feat in row.columns:
                    row[feat] = -row[feat]
            for bid_feat, ask_feat in SWAP_PAIRS:
                if bid_feat in row.columns and ask_feat in row.columns:
                    row[bid_feat], row[ask_feat] = row[ask_feat].copy(), row[bid_feat].copy()

        rows.append(row)

    if not rows:
        return None
    return pd.concat(rows, ignore_index=True)


def classify_day_regime(queue_df):
    if queue_df is None or 'mid_price' not in queue_df.columns or len(queue_df) < 10:
        return 'unknown'
    day_return = queue_df['mid_price'].iloc[-1] - queue_df['mid_price'].iloc[0]
    if day_return > 2 * ES_TICK_SIZE:
        return 'green'
    elif day_return < -2 * ES_TICK_SIZE:
        return 'red'
    return 'flat'


def compute_regime_gap(trades_df):
    if trades_df is None or len(trades_df) == 0:
        return 999.0, {}, {}

    daily = trades_df.groupby('date')['net_ticks'].sum()
    regimes = trades_df.groupby('date')['regime'].first()

    green = daily[regimes == 'green']
    red = daily[regimes == 'red']

    sharpe_green = float(green.mean() / green.std()) if len(green) > 1 and green.std() > 0 else 0
    sharpe_red = float(red.mean() / red.std()) if len(red) > 1 and red.std() > 0 else 0

    denom = max(abs(sharpe_green), abs(sharpe_red), 1e-6)
    gap = abs(sharpe_green - sharpe_red) / denom

    return float(gap), \
           {'sharpe': sharpe_green, 'n_days': len(green), 'n_trades': int(len(trades_df[trades_df['regime']=='green']))}, \
           {'sharpe': sharpe_red, 'n_days': len(red), 'n_trades': int(len(trades_df[trades_df['regime']=='red']))}


# ─────────────────────────────────────
# Training utilities
# ─────────────────────────────────────

def train_mlp_fold(X_train, y_train, X_val, y_val, n_features, device):
    """Train MLP for one fold. Returns trained model and best val AUC."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = QueueEntryMLP(n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()

    # DataLoaders
    train_ds = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).float()
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(y_val).float()
    )

    use_pin = device.type == 'cuda'
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              pin_memory=use_pin, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                            pin_memory=use_pin, num_workers=0)

    best_val_auc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):
        # ── Train ──
        model.train()
        train_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        train_loss /= max(n_batches, 1)

        # ── Validate ──
        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                pred = model(xb)
                val_preds.append(pred.cpu().numpy())
                val_labels.append(yb.numpy())

        val_preds = np.concatenate(val_preds)
        val_labels = np.concatenate(val_labels)

        if len(np.unique(val_labels)) > 1:
            val_auc = roc_auc_score(val_labels, val_preds)
        else:
            val_auc = 0.5

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOP_PATIENCE:
            break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    return model, best_val_auc, epoch + 1


def predict_mlp(model, X, device):
    """Get predictions from trained model."""
    model.eval()
    ds = TensorDataset(torch.from_numpy(X).float())
    loader = DataLoader(ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                        pin_memory=(device.type == 'cuda'), num_workers=0)
    preds = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device, non_blocking=True)
            pred = model(xb)
            preds.append(pred.cpu().numpy())
    return np.concatenate(preds)


def compute_gradient_importance(model, X_sample, device):
    """Compute gradient-based feature importance using a sample of data."""
    model.eval()
    # Use up to 5000 samples
    n = min(len(X_sample), 5000)
    X_t = torch.from_numpy(X_sample[:n]).float().to(device)
    X_t.requires_grad = True

    pred = model(X_t)
    pred.sum().backward()

    # Mean absolute gradient per feature
    importance = X_t.grad.abs().mean(dim=0).cpu().numpy()
    return importance


# ─────────────────────────────────────
# Main
# ─────────────────────────────────────

def run_config(fifo_config, device, all_config_results):
    """Run full walk-forward for one FIFO config."""
    log.info(f"\n{'='*70}")
    log.info(f"FIFO CONFIG: {fifo_config}")
    log.info(f"{'='*70}")

    # ── 1. Discover dates ──
    q_files = sorted(QUEUE_DIR.glob("features_*.parquet"))
    q_dates = {f.stem.replace('features_', '') for f in q_files}

    f_files = sorted(FIFO_DIR.glob("*_fifo_labels.npz"))
    f_dates = {f.stem.replace('_fifo_labels', '') for f in f_files}

    overlap = sorted(q_dates & f_dates)
    log.info(f"Queue dates: {len(q_dates)}, FIFO dates: {len(f_dates)}, Overlap: {len(overlap)}")

    if len(overlap) < TRAIN_DAYS + OOT_DAYS:
        log.error(f"Need >= {TRAIN_DAYS + OOT_DAYS} dates, got {len(overlap)}")
        return None

    # ── 2. Load and join all data ──
    log.info("Loading and joining data...")
    all_data = {}
    date_regimes = {}

    for date_str in overlap:
        queue = load_queue_features(date_str)
        fifo = load_fifo_labels(date_str, fifo_config)

        if queue is None or fifo is None:
            continue

        queue = engineer_features(queue)
        merged, n_before, n_after = join_queue_fifo(queue, fifo)

        if n_after < 50:
            continue

        training = prepare_training_data(merged, date_str)
        if training is None or len(training) < 50:
            continue

        regime = classify_day_regime(queue)
        date_regimes[date_str] = regime
        training['regime'] = regime
        all_data[date_str] = training

        wr = training['target'].mean()
        log.info(f"  {date_str}: {len(training)} entries (L:{(training['side']=='long').sum()} "
                 f"S:{(training['side']=='short').sum()}), WR={wr:.3f}, regime={regime}")

    valid_dates = sorted(all_data.keys())
    log.info(f"Valid dates: {len(valid_dates)}")
    log.info(f"Regimes: {pd.Series(date_regimes).value_counts().to_dict()}")

    if len(valid_dates) < TRAIN_DAYS + OOT_DAYS:
        log.error(f"Only {len(valid_dates)} valid dates")
        return None

    # ── 3. Define feature columns ──
    # v2 features + time_of_day + session_pct + side_indicator
    sample = list(all_data.values())[0]
    v2_features = [c for c in RAW_FEATURES + ['queue_ratio', 'queue_diff', 'net_flow_diff',
                    'trade_imbalance', 'ofi_momentum', 'level_age_diff']
                    if c in sample.columns]
    mlp_extra_features = ['time_of_day', 'session_pct', 'side_indicator']
    feature_cols = v2_features + mlp_extra_features
    log.info(f"Feature columns ({len(feature_cols)}): {feature_cols}")

    n_features = len(feature_cols)

    # ── 4. Walk-forward ──
    log.info(f"Walk-forward: {TRAIN_DAYS}d train, {OOT_DAYS}d OOT, {SLIDE_DAYS}d slide (SLIDING)")

    all_trades = []
    fold_results = []
    fi_accum = np.zeros(n_features)
    fi_count = 0

    fold_num = 0
    start_idx = TRAIN_DAYS

    while start_idx + OOT_DAYS <= len(valid_dates):
        train_dates = valid_dates[start_idx - TRAIN_DAYS : start_idx]
        oot_dates = valid_dates[start_idx : start_idx + OOT_DAYS]

        train_df = pd.concat([all_data[d] for d in train_dates if d in all_data], ignore_index=True)
        oot_df = pd.concat([all_data[d] for d in oot_dates if d in all_data], ignore_index=True)

        if len(train_df) < 200 or len(oot_df) < 50:
            start_idx += SLIDE_DAYS
            continue

        X_train_raw = train_df[feature_cols].values.astype(np.float32)
        y_train_raw = train_df['target'].values.astype(np.float32)
        X_oot_raw = oot_df[feature_cols].values.astype(np.float32)

        # Replace inf/nan
        X_train_raw = np.nan_to_num(X_train_raw, nan=0., posinf=100., neginf=-100.)
        X_oot_raw = np.nan_to_num(X_oot_raw, nan=0., posinf=100., neginf=-100.)

        # Per-fold z-score normalization (fit on train, apply to OOT)
        train_mean = X_train_raw.mean(axis=0)
        train_std = X_train_raw.std(axis=0)
        train_std[train_std < 1e-8] = 1.0  # avoid division by zero

        X_train_norm = (X_train_raw - train_mean) / train_std
        X_oot_norm = (X_oot_raw - train_mean) / train_std

        # Clip extreme values after normalization
        X_train_norm = np.clip(X_train_norm, -10, 10)
        X_oot_norm = np.clip(X_oot_norm, -10, 10)

        # Split train into train/val (last 20% for early stopping)
        val_split = int(len(X_train_norm) * (1 - VAL_FRAC))
        X_tr = X_train_norm[:val_split]
        y_tr = y_train_raw[:val_split]
        X_val = X_train_norm[val_split:]
        y_val = y_train_raw[val_split:]

        # Train MLP
        t0 = time.time()
        model, best_val_auc, n_epochs = train_mlp_fold(X_tr, y_tr, X_val, y_val, n_features, device)
        train_time = time.time() - t0

        # Predict on OOT
        pred_prob = predict_mlp(model, X_oot_norm, device)
        oot_df = oot_df.copy()
        oot_df['pred_prob'] = pred_prob

        # Gradient-based feature importance
        try:
            gi = compute_gradient_importance(model, X_oot_norm, device)
            fi_accum += gi
            fi_count += 1
        except Exception:
            pass

        # Evaluate thresholds
        y_oot = oot_df['target'].values
        baseline_wr = float(y_oot.mean())
        baseline_pnl = float(oot_df['net_ticks'].sum())
        baseline_n = len(oot_df)

        best_threshold = 0.5
        best_pf = 0
        threshold_results = {}

        for thresh in THRESHOLDS:
            selected = oot_df[oot_df['pred_prob'] >= thresh]
            if len(selected) < 5:
                continue
            sel_wr = float(selected['target'].mean())
            sel_pnl = float(selected['net_ticks'].sum())
            winners = selected[selected['net_ticks'] > 0]['net_ticks'].sum()
            losers = abs(selected[selected['net_ticks'] < 0]['net_ticks'].sum())
            pf = float(winners / max(losers, 1e-6))

            threshold_results[str(thresh)] = {
                'n_trades': int(len(selected)),
                'wr': sel_wr,
                'pnl': sel_pnl,
                'pf': pf,
            }
            if pf > best_pf:
                best_pf = pf
                best_threshold = thresh

        oot_auc = float(roc_auc_score(y_oot, pred_prob)) if len(np.unique(y_oot)) > 1 else 0.5

        fold_res = {
            'fold': fold_num,
            'train_dates': train_dates,
            'oot_dates': oot_dates,
            'oot_regime': oot_df.groupby('date')['regime'].first().value_counts().to_dict(),
            'n_train': len(train_df),
            'n_oot': len(oot_df),
            'baseline_wr': baseline_wr,
            'baseline_pnl': baseline_pnl,
            'baseline_n': baseline_n,
            'val_auc': float(best_val_auc),
            'oot_auc': oot_auc,
            'n_epochs': n_epochs,
            'train_time_s': round(train_time, 1),
            'thresholds': threshold_results,
            'best_threshold': best_threshold,
            'best_pf': float(best_pf),
        }
        fold_results.append(fold_res)

        oot_df['fold'] = fold_num
        all_trades.append(oot_df)

        best_t = threshold_results.get(str(best_threshold), {})
        log.info(f"  Fold {fold_num:2d} | OOT={','.join(oot_dates)} | "
                 f"valAUC={best_val_auc:.3f}, ootAUC={oot_auc:.3f}, epochs={n_epochs}, "
                 f"t={train_time:.1f}s | "
                 f"base: WR={baseline_wr:.3f}, PnL={baseline_pnl:+.1f}t | "
                 f"best@{best_threshold}: n={best_t.get('n_trades','?')}, "
                 f"WR={best_t.get('wr',0):.3f}, PF={best_t.get('pf',0):.2f}")

        fold_num += 1
        start_idx += SLIDE_DAYS

    # ── 5. Aggregate ──
    log.info(f"\n{'='*70}")
    log.info(f"AGGREGATE RESULTS — {fifo_config} ({fold_num} folds)")
    log.info(f"{'='*70}")

    if not all_trades:
        log.error("No valid folds completed")
        return None

    all_trades_df = pd.concat(all_trades, ignore_index=True)

    total_n = len(all_trades_df)
    total_wr = float(all_trades_df['target'].mean())
    total_pnl = float(all_trades_df['net_ticks'].sum())

    log.info(f"\n--- BASELINE (no filter) ---")
    log.info(f"  Total entries: {total_n}")
    log.info(f"  WR: {total_wr:.3f}")
    log.info(f"  Total PnL: {total_pnl:+.1f} ticks")
    log.info(f"  Per-trade: {total_pnl/max(total_n,1):+.3f} ticks")

    # Aggregate thresholds
    log.info(f"\n--- FILTERED ENTRY PERFORMANCE ---")
    agg_results = {}
    for thresh in THRESHOLDS + [0.70]:
        sel = all_trades_df[all_trades_df['pred_prob'] >= thresh]
        if len(sel) < 10:
            continue
        sel_wr = float(sel['target'].mean())
        sel_pnl = float(sel['net_ticks'].sum())
        winners = sel[sel['net_ticks'] > 0]['net_ticks'].sum()
        losers = abs(sel[sel['net_ticks'] < 0]['net_ticks'].sum())
        pf = float(winners / max(losers, 1e-6))
        per_trade = sel_pnl / len(sel)

        daily = sel.groupby('date')['net_ticks'].sum()
        sharpe = float(daily.mean() / daily.std()) if len(daily) > 1 and daily.std() > 0 else 0
        downside = daily[daily < 0]
        sortino = float(daily.mean() / downside.std()) if len(downside) > 1 and downside.std() > 0 else (
            999.0 if daily.mean() > 0 else 0)

        agg_results[str(thresh)] = {
            'n_trades': int(len(sel)),
            'wr': sel_wr,
            'pnl': sel_pnl,
            'per_trade': float(per_trade),
            'pf': pf,
            'sharpe': sharpe,
            'sortino': sortino,
        }
        log.info(f"  thresh={thresh:.2f}: n={len(sel):5d}, WR={sel_wr:.3f}, "
                 f"PnL={sel_pnl:+.1f}t, per_trade={per_trade:+.3f}t, "
                 f"PF={pf:.3f}, Sharpe={sharpe:.2f}, Sortino={sortino:.2f}")

    # Per-side breakdown
    best_agg_thresh = max(agg_results.items(), key=lambda x: x[1]['pf'])[0] if agg_results else '0.55'
    best_thresh_val = float(best_agg_thresh)
    log.info(f"\n--- PER-SIDE BREAKDOWN (best threshold = {best_agg_thresh}) ---")
    side_stats = {}
    for side in ['long', 'short']:
        side_df = all_trades_df[(all_trades_df['side'] == side) &
                                (all_trades_df['pred_prob'] >= best_thresh_val)]
        if len(side_df) < 5:
            log.info(f"  {side.upper()}: insufficient trades ({len(side_df)})")
            continue
        sw = float(side_df['target'].mean())
        sp = float(side_df['net_ticks'].sum())
        swin = side_df[side_df['net_ticks'] > 0]['net_ticks'].sum()
        slos = abs(side_df[side_df['net_ticks'] < 0]['net_ticks'].sum())
        spf = float(swin / max(slos, 1e-6))
        side_stats[side] = {'n': int(len(side_df)), 'wr': sw, 'pnl': sp, 'pf': spf}
        log.info(f"  {side.upper()}: n={len(side_df)}, WR={sw:.3f}, PnL={sp:+.1f}t, PF={spf:.3f}")

    # ── 6. Regime analysis (HC #428) ──
    log.info(f"\n--- REGIME ANALYSIS (HC #428) ---")
    filtered = all_trades_df[all_trades_df['pred_prob'] >= best_thresh_val]
    if len(filtered) > 10:
        gap, green_stats, red_stats = compute_regime_gap(filtered)
        log.info(f"  At threshold {best_thresh_val}:")
        log.info(f"  Green: Sharpe={green_stats.get('sharpe',0):.2f}, n_days={green_stats.get('n_days',0)}")
        log.info(f"  Red:   Sharpe={red_stats.get('sharpe',0):.2f}, n_days={red_stats.get('n_days',0)}")
        regime_pass = gap <= 0.50
        log.info(f"  Regime gap: {gap:.3f} {'PASS' if regime_pass else 'FAIL'}")
    else:
        gap = 999.0
        green_stats, red_stats = {}, {}
        regime_pass = False
        log.info("  Insufficient trades for regime analysis")

    # ── 7. Feature importance ──
    log.info(f"\n--- GRADIENT-BASED FEATURE IMPORTANCE ---")
    fi_summary = {}
    if fi_count > 0:
        fi_mean = fi_accum / fi_count
        fi_sorted_idx = np.argsort(fi_mean)[::-1]
        for rank, idx in enumerate(fi_sorted_idx):
            feat = feature_cols[idx]
            fi_summary[feat] = {'mean_grad': float(fi_mean[idx]), 'rank': rank + 1}
            if rank < 15:
                log.info(f"  {rank+1:2d}. {feat:35s} grad={fi_mean[idx]:.4f}")

    # ── 8. IC analysis ──
    ic = float(np.corrcoef(all_trades_df['pred_prob'], all_trades_df['net_ticks'])[0, 1])
    rank_ic = float(scipy_stats.spearmanr(all_trades_df['pred_prob'], all_trades_df['net_ticks'])[0])
    log.info(f"\n--- INFORMATION COEFFICIENT ---")
    log.info(f"  Pearson IC (prob vs net_ticks): {ic:.4f}")
    log.info(f"  Rank IC (Spearman): {rank_ic:.4f}")

    # ── 9. Mean val/oot AUC ──
    mean_val_auc = float(np.mean([f['val_auc'] for f in fold_results]))
    mean_oot_auc = float(np.mean([f['oot_auc'] for f in fold_results]))
    log.info(f"\n--- AUC ---")
    log.info(f"  Mean val AUC: {mean_val_auc:.4f}")
    log.info(f"  Mean OOT AUC: {mean_oot_auc:.4f}")

    # Build result dict
    config_result = {
        'fifo_config': fifo_config,
        'n_dates': len(valid_dates),
        'n_folds': fold_num,
        'n_features': n_features,
        'feature_cols': feature_cols,
        'walk_forward': {
            'train_days': TRAIN_DAYS,
            'oot_days': OOT_DAYS,
            'slide_days': SLIDE_DAYS,
            'method': 'SLIDING',
        },
        'mlp_config': {
            'architecture': '128-BN-ReLU-Drop0.3 -> 64-BN-ReLU-Drop0.2 -> 32-BN-ReLU -> 1-Sigmoid',
            'lr': LR,
            'batch_size': BATCH_SIZE,
            'max_epochs': MAX_EPOCHS,
            'early_stop_patience': EARLY_STOP_PATIENCE,
            'val_fraction': VAL_FRAC,
            'normalization': 'per-fold z-score',
        },
        'baseline': {
            'n_trades': total_n,
            'wr': total_wr,
            'pnl': total_pnl,
            'per_trade': float(total_pnl / max(total_n, 1)),
        },
        'filtered_performance': agg_results,
        'side_breakdown': side_stats,
        'regime_analysis': {
            'threshold': best_thresh_val,
            'gap': float(gap),
            'green': green_stats,
            'red': red_stats,
            'pass_gate': regime_pass,
        },
        'feature_importance': fi_summary,
        'ic': {'pearson': ic, 'spearman': rank_ic},
        'auc': {'mean_val': mean_val_auc, 'mean_oot': mean_oot_auc},
        'fold_results': fold_results,
        'date_regimes': date_regimes,
    }

    # Save trades for this config
    save_cols = [c for c in all_trades_df.columns if all_trades_df[c].dtype.kind in 'iufbOS']
    trades_path = OUTPUT_DIR / f"all_oot_trades_{fifo_config}.parquet"
    all_trades_df[save_cols].to_parquet(trades_path, index=False)
    log.info(f"Saved {len(all_trades_df)} OOT trades to {trades_path}")

    return config_result


def main():
    t_start = time.time()
    log.info("=" * 70)
    log.info("Queue Entry Selector MLP v1 — PyTorch GPU")
    log.info("=" * 70)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Device: {device}")
    if device.type == 'cuda':
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
        log.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    all_results = {}

    for fifo_config in FIFO_CONFIGS:
        result = run_config(fifo_config, device, all_results)
        if result is not None:
            all_results[fifo_config] = result

    # ── Save combined results ──
    combined = {
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'model': 'MLP_v1',
        'total_time_s': round(time.time() - t_start, 1),
        'configs': all_results,
    }

    out_path = OUTPUT_DIR / "results.json"
    with open(out_path, 'w') as f:
        json.dump(combined, f, indent=2, default=str)
    log.info(f"\nCombined results saved to {out_path}")

    # ── MLflow logging ──
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("queue_entry_selector")

        for fifo_config, result in all_results.items():
            run_name = f"mlp_v1_{fifo_config}_{datetime.now().strftime('%Y%m%d_%H%M')}"
            with mlflow.start_run(run_name=run_name):
                mlflow.log_params({
                    'version': 'mlp_v1',
                    'model_type': 'MLP',
                    'fifo_config': fifo_config,
                    'n_features': result['n_features'],
                    'n_dates': result['n_dates'],
                    'n_folds': result['n_folds'],
                    'train_days': TRAIN_DAYS,
                    'oot_days': OOT_DAYS,
                    'lr': LR,
                    'batch_size': BATCH_SIZE,
                    'architecture': '128-64-32-1',
                    'normalization': 'per-fold_zscore',
                })
                mlflow.log_metrics({
                    'baseline_wr': result['baseline']['wr'],
                    'baseline_pnl': result['baseline']['pnl'],
                    'ic_pearson': result['ic']['pearson'],
                    'ic_spearman': result['ic']['spearman'],
                    'mean_val_auc': result['auc']['mean_val'],
                    'mean_oot_auc': result['auc']['mean_oot'],
                    'regime_gap': result['regime_analysis']['gap'],
                })
                # Log best threshold metrics
                if result['filtered_performance']:
                    best_key = max(result['filtered_performance'].items(),
                                   key=lambda x: x[1]['pf'])[0]
                    best = result['filtered_performance'][best_key]
                    mlflow.log_metrics({
                        'best_thresh': float(best_key),
                        'best_wr': best['wr'],
                        'best_pf': best['pf'],
                        'best_sharpe': best['sharpe'],
                        'best_sortino': best['sortino'],
                        'best_n_trades': best['n_trades'],
                    })
                # Log feature importance
                if result['feature_importance']:
                    for feat, fi in result['feature_importance'].items():
                        mlflow.log_metric(f'fi_{feat}', fi['mean_grad'])
                mlflow.log_artifact(str(out_path))

        log.info("MLflow runs logged")
    except Exception as e:
        log.warning(f"MLflow logging failed: {e}")

    total_time = time.time() - t_start
    log.info(f"\nTotal runtime: {total_time:.1f}s ({total_time/60:.1f}m)")
    log.info("Done.")


if __name__ == '__main__':
    main()
