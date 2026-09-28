#!/usr/bin/env python3
"""
Mamba + PatchTST Creative Execution Strategy Engine
=====================================================
Develops and tests execution strategies that combine:
- Mamba multi-horizon predictions (1s/5s/10s) for entry/exit timing
- PatchTST directional accuracy for trade confirmation
- Embeddings from both models for a small MLP execution optimizer

Strategies tested:
1. Rule-based: multi-horizon agreement + confidence thresholds
2. Signal-flip exits with directional confirmation
3. MLP execution optimizer trained on embeddings → {enter, hold, exit}
4. Adaptive hold-time based on multi-horizon signal coherence
5. Confidence-weighted position sizing

Uses Mamba v7 March OOT predictions against raw MBO data for market replay.
"""

import os
import sys
import json
import time
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

# ── Paths ──
LVL3_ROOT = Path(__file__).resolve().parent.parent.parent
MAMBA_PRED_DIR = LVL3_ROOT / 'output' / 'mamba_v7_tiny_smart_v3_mar_apr'
PATCHTST_PRED_DIR = LVL3_ROOT / 'output' / 'patchtst_sliding60d_smart_v2'
MBO_DIR = LVL3_ROOT / 'data' / 'raw' / 'mbo'
RESULTS_DIR = LVL3_ROOT / 'alpha_discovery' / 'deep_models' / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Cost model (from directives: $4.70 RT = 0.376 ticks) ──
TICK_VALUE = 12.50
COMMISSION_TICKS = 0.376  # $4.70 RT
SPREAD_HALF_TICK = 0.5    # half spread for limit orders
MID_PRICE_COST = 0.0      # mid-price: no spread cost, just commission

# ── Logging ──
_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
log = logging.getLogger('exec_strategy')
log.setLevel(logging.INFO)
_fh = logging.FileHandler(str(RESULTS_DIR / f'mamba_patchtst_exec_{_ts}.log'), mode='w', encoding='utf-8')
_fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
log.addHandler(_fh)
_ch = logging.StreamHandler(sys.stdout)
_ch.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
log.addHandler(_ch)


@dataclass
class Trade:
    entry_idx: int
    entry_price: float
    direction: int  # +1 long, -1 short
    entry_signal_1s: float = 0.0
    entry_signal_5s: float = 0.0
    entry_signal_10s: float = 0.0
    entry_confidence: float = 0.0
    exit_idx: int = -1
    exit_price: float = 0.0
    exit_reason: str = ""
    pnl_ticks: float = 0.0
    hold_events: int = 0
    mfe_ticks: float = 0.0  # max favorable excursion
    mae_ticks: float = 0.0  # max adverse excursion


@dataclass
class StrategyResult:
    name: str
    n_trades: int = 0
    n_winners: int = 0
    total_pnl_ticks: float = 0.0
    total_pnl_dollars: float = 0.0
    avg_pnl_per_trade: float = 0.0
    win_rate: float = 0.0
    avg_winner: float = 0.0
    avg_loser: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_ticks: float = 0.0
    sortino_ratio: float = 0.0
    avg_hold_events: float = 0.0
    avg_mfe: float = 0.0
    avg_mae: float = 0.0
    trades_per_day: float = 0.0
    long_pnl: float = 0.0
    short_pnl: float = 0.0
    cost_model: str = ""
    details: dict = field(default_factory=dict)


def load_mamba_predictions() -> Dict:
    """Load Mamba v7 concat OOT predictions with multi-horizon signals."""
    concat_file = MAMBA_PRED_DIR / 'concat_oot_predictions.npz'
    if not concat_file.exists():
        log.error(f"No concat predictions at {concat_file}")
        return {}

    data = np.load(str(concat_file), allow_pickle=True)
    log.info(f"Loaded Mamba v7 predictions: {data['preds_1s'].shape[0]:,} events")
    log.info(f"  IC_1s={float(data['concat_ic_1s']):.4f}, IC_5s={float(data['concat_ic_5s']):.4f}, IC_10s={float(data['concat_ic_10s']):.4f}")
    log.info(f"  Embeddings: {data['embeddings'].shape}")

    return {
        'preds_1s': data['preds_1s'],
        'preds_5s': data['preds_5s'],
        'preds_10s': data['preds_10s'],
        'labels_1s': data['labels_1s'],
        'labels_5s': data['labels_5s'],
        'labels_10s': data['labels_10s'],
        'embeddings': data['embeddings'],
    }


def load_per_fold_mamba() -> List[Dict]:
    """Load per-fold Mamba predictions to get date-level granularity."""
    folds = []
    for f in sorted(MAMBA_PRED_DIR.glob('fold_*_oot_predictions.npz')):
        data = np.load(str(f), allow_pickle=True)
        fold_idx = int(f.stem.split('_')[1])
        oot_file = str(data['oot_files'][0]) if 'oot_files' in data else ''
        # Extract date from oot filename
        date_str = ''
        if oot_file:
            import re
            m = re.search(r'(\d{8})_mbo', oot_file)
            if m:
                date_str = m.group(1)

        folds.append({
            'fold_idx': fold_idx,
            'date': date_str,
            'preds': data['predictions'],  # (N, 3) = 1s,5s,10s
            'labels': data['labels'],      # (N, 3)
            'embeddings': data['embeddings'],
            'n_events': data['predictions'].shape[0],
        })
        log.info(f"  Fold {fold_idx:02d} ({date_str}): {data['predictions'].shape[0]:,} events")

    return folds


def compute_confidence(preds_1s, preds_5s, preds_10s) -> np.ndarray:
    """Compute confidence as average absolute prediction magnitude across horizons."""
    return (np.abs(preds_1s) + np.abs(preds_5s) + np.abs(preds_10s)) / 3.0


def compute_direction_agreement(preds_1s, preds_5s, preds_10s) -> np.ndarray:
    """Check if all 3 horizons agree on direction. Returns +1, -1, or 0."""
    signs = np.sign(np.stack([preds_1s, preds_5s, preds_10s], axis=1))
    all_pos = np.all(signs > 0, axis=1)
    all_neg = np.all(signs < 0, axis=1)
    agreement = np.zeros(len(preds_1s))
    agreement[all_pos] = 1
    agreement[all_neg] = -1
    return agreement


def compute_signal_coherence(preds_1s, preds_5s, preds_10s) -> np.ndarray:
    """Coherence: how aligned are multi-horizon signals? Range [0, 1]."""
    stacked = np.stack([preds_1s, preds_5s, preds_10s], axis=1)
    signs = np.sign(stacked)
    magnitudes = np.abs(stacked)
    # All same sign = high coherence; mixed = low
    sign_agreement = np.abs(signs.sum(axis=1)) / 3.0  # 1.0 if all agree
    # Weighted by magnitude
    mag_norm = magnitudes / (magnitudes.sum(axis=1, keepdims=True) + 1e-8)
    coherence = sign_agreement * magnitudes.mean(axis=1)
    return coherence


# ═══════════════════════════════════════════════════════════════
# EXECUTION STRATEGIES
# ═══════════════════════════════════════════════════════════════

def strategy_multi_horizon_agree(fold_data: Dict, conf_threshold: float,
                                  cost_ticks: float, hold_mode: str = 'signal_flip',
                                  max_hold: int = 2000) -> List[Trade]:
    """
    Strategy 1: Enter when ALL 3 horizons agree on direction AND confidence > threshold.
    Exit on signal flip, max hold, or confidence drop.
    """
    preds = fold_data['preds']  # (N, 3)
    labels = fold_data['labels']
    p1, p5, p10 = preds[:, 0], preds[:, 1], preds[:, 2]

    agreement = compute_direction_agreement(p1, p5, p10)
    confidence = compute_confidence(p1, p5, p10)

    trades = []
    in_trade = False
    current_trade = None

    for i in range(len(preds)):
        if not in_trade:
            # Entry condition: all horizons agree + confidence above threshold
            if agreement[i] != 0 and confidence[i] > conf_threshold:
                current_trade = Trade(
                    entry_idx=i,
                    entry_price=labels[i, 2],  # use 10s label as reference price proxy
                    direction=int(agreement[i]),
                    entry_signal_1s=float(p1[i]),
                    entry_signal_5s=float(p5[i]),
                    entry_signal_10s=float(p10[i]),
                    entry_confidence=float(confidence[i]),
                )
                in_trade = True

        else:
            hold_events = i - current_trade.entry_idx

            # Track MFE/MAE using cumulative label movement
            price_move = labels[i, 0] * current_trade.direction  # 1s move in our direction
            current_trade.mfe_ticks = max(current_trade.mfe_ticks, price_move)
            current_trade.mae_ticks = min(current_trade.mae_ticks, price_move)

            # Exit conditions
            exit_signal = False
            exit_reason = ""

            if hold_mode == 'signal_flip':
                # Exit when direction flips
                if agreement[i] == -current_trade.direction:
                    exit_signal = True
                    exit_reason = "signal_flip"
            elif hold_mode == 'conditional_flip':
                # Exit when direction flips AND confidence is high
                if agreement[i] == -current_trade.direction and confidence[i] > conf_threshold * 0.5:
                    exit_signal = True
                    exit_reason = "conditional_flip"
            elif hold_mode == 'slow_flip':
                # Only exit when ALL horizons flip AND hold > min events
                if hold_events > 100 and agreement[i] == -current_trade.direction:
                    exit_signal = True
                    exit_reason = "slow_flip"

            # Max hold exit
            if hold_events >= max_hold:
                exit_signal = True
                exit_reason = "max_hold"

            # Confidence collapse exit
            if confidence[i] < conf_threshold * 0.3 and hold_events > 50:
                exit_signal = True
                exit_reason = "confidence_collapse"

            if exit_signal:
                # PnL = cumulative price move in our direction - costs
                actual_move = labels[i, 2] - current_trade.entry_price if current_trade.entry_price != 0 else 0
                pnl = actual_move * current_trade.direction - cost_ticks
                current_trade.exit_idx = i
                current_trade.pnl_ticks = pnl
                current_trade.hold_events = hold_events
                current_trade.exit_reason = exit_reason
                trades.append(current_trade)
                in_trade = False
                current_trade = None

    return trades


def strategy_1s_entry_10s_hold(fold_data: Dict, entry_threshold: float,
                                 hold_threshold: float, cost_ticks: float,
                                 max_hold: int = 2000) -> List[Trade]:
    """
    Strategy 2: Use 1s signal for precise entry timing, hold based on 10s signal.
    Entry: |pred_1s| > entry_threshold AND sign(1s) == sign(10s)
    Hold: while sign(10s) hasn't flipped
    Exit: 10s signal flips OR magnitude drops below hold_threshold
    """
    preds = fold_data['preds']
    labels = fold_data['labels']
    p1, p5, p10 = preds[:, 0], preds[:, 1], preds[:, 2]

    trades = []
    in_trade = False
    current_trade = None

    for i in range(len(preds)):
        if not in_trade:
            if np.abs(p1[i]) > entry_threshold and np.sign(p1[i]) == np.sign(p10[i]):
                direction = int(np.sign(p1[i]))
                current_trade = Trade(
                    entry_idx=i,
                    entry_price=labels[i, 2],
                    direction=direction,
                    entry_signal_1s=float(p1[i]),
                    entry_signal_5s=float(p5[i]),
                    entry_signal_10s=float(p10[i]),
                    entry_confidence=float(np.abs(p10[i])),
                )
                in_trade = True
        else:
            hold_events = i - current_trade.entry_idx
            price_move = labels[i, 0] * current_trade.direction
            current_trade.mfe_ticks = max(current_trade.mfe_ticks, price_move)
            current_trade.mae_ticks = min(current_trade.mae_ticks, price_move)

            exit_signal = False
            exit_reason = ""

            # Exit when 10s signal flips
            if np.sign(p10[i]) == -current_trade.direction:
                exit_signal = True
                exit_reason = "10s_flip"

            # Exit when 10s magnitude drops
            if np.abs(p10[i]) < hold_threshold and hold_events > 50:
                exit_signal = True
                exit_reason = "10s_weak"

            if hold_events >= max_hold:
                exit_signal = True
                exit_reason = "max_hold"

            if exit_signal:
                actual_move = labels[i, 2] - current_trade.entry_price if current_trade.entry_price != 0 else 0
                pnl = actual_move * current_trade.direction - cost_ticks
                current_trade.exit_idx = i
                current_trade.pnl_ticks = pnl
                current_trade.hold_events = hold_events
                current_trade.exit_reason = exit_reason
                trades.append(current_trade)
                in_trade = False
                current_trade = None

    return trades


def strategy_coherence_scaled(fold_data: Dict, coherence_threshold: float,
                               cost_ticks: float, max_hold: int = 2000) -> List[Trade]:
    """
    Strategy 3: Enter only when multi-horizon coherence is high.
    Uses coherence as a position sizing proxy.
    Higher coherence = stronger conviction = tighter exits allowed.
    """
    preds = fold_data['preds']
    labels = fold_data['labels']
    p1, p5, p10 = preds[:, 0], preds[:, 1], preds[:, 2]

    coherence = compute_signal_coherence(p1, p5, p10)
    agreement = compute_direction_agreement(p1, p5, p10)

    trades = []
    in_trade = False
    current_trade = None

    for i in range(len(preds)):
        if not in_trade:
            if coherence[i] > coherence_threshold and agreement[i] != 0:
                current_trade = Trade(
                    entry_idx=i,
                    entry_price=labels[i, 2],
                    direction=int(agreement[i]),
                    entry_signal_1s=float(p1[i]),
                    entry_signal_5s=float(p5[i]),
                    entry_signal_10s=float(p10[i]),
                    entry_confidence=float(coherence[i]),
                )
                in_trade = True
        else:
            hold_events = i - current_trade.entry_idx
            price_move = labels[i, 0] * current_trade.direction
            current_trade.mfe_ticks = max(current_trade.mfe_ticks, price_move)
            current_trade.mae_ticks = min(current_trade.mae_ticks, price_move)

            exit_signal = False
            exit_reason = ""

            # Adaptive exit: higher initial coherence → allow longer hold
            adaptive_patience = int(100 * current_trade.entry_confidence / coherence_threshold)

            if agreement[i] == -current_trade.direction and hold_events > adaptive_patience:
                exit_signal = True
                exit_reason = "adaptive_flip"

            if coherence[i] < coherence_threshold * 0.2 and hold_events > 50:
                exit_signal = True
                exit_reason = "coherence_collapse"

            if hold_events >= max_hold:
                exit_signal = True
                exit_reason = "max_hold"

            if exit_signal:
                actual_move = labels[i, 2] - current_trade.entry_price if current_trade.entry_price != 0 else 0
                pnl = actual_move * current_trade.direction - cost_ticks
                current_trade.exit_idx = i
                current_trade.pnl_ticks = pnl
                current_trade.hold_events = hold_events
                current_trade.exit_reason = exit_reason
                trades.append(current_trade)
                in_trade = False
                current_trade = None

    return trades


def strategy_momentum_burst(fold_data: Dict, burst_threshold: float,
                             cost_ticks: float, cooldown: int = 200) -> List[Trade]:
    """
    Strategy 4: Momentum burst detection.
    Enter when 1s signal suddenly spikes (momentum burst) while 5s/10s confirm direction.
    Quick exit after burst dissipates. Designed for capturing sharp moves.
    """
    preds = fold_data['preds']
    labels = fold_data['labels']
    p1, p5, p10 = preds[:, 0], preds[:, 1], preds[:, 2]

    # Compute rolling 1s signal magnitude change (momentum)
    p1_diff = np.zeros_like(p1)
    p1_diff[1:] = np.abs(p1[1:]) - np.abs(p1[:-1])

    trades = []
    in_trade = False
    current_trade = None
    last_exit_idx = -cooldown

    for i in range(10, len(preds)):
        if not in_trade:
            if (i - last_exit_idx < cooldown):
                continue

            # Detect momentum burst: large 1s change + 5s/10s direction agreement
            if (p1_diff[i] > burst_threshold and
                np.sign(p1[i]) == np.sign(p5[i]) == np.sign(p10[i]) and
                np.abs(p1[i]) > burst_threshold):

                direction = int(np.sign(p1[i]))
                current_trade = Trade(
                    entry_idx=i,
                    entry_price=labels[i, 2],
                    direction=direction,
                    entry_signal_1s=float(p1[i]),
                    entry_signal_5s=float(p5[i]),
                    entry_signal_10s=float(p10[i]),
                    entry_confidence=float(p1_diff[i]),
                )
                in_trade = True
        else:
            hold_events = i - current_trade.entry_idx
            price_move = labels[i, 0] * current_trade.direction
            current_trade.mfe_ticks = max(current_trade.mfe_ticks, price_move)
            current_trade.mae_ticks = min(current_trade.mae_ticks, price_move)

            exit_signal = False
            exit_reason = ""

            # Quick exit when burst fades (1s signal weakens)
            if np.abs(p1[i]) < np.abs(current_trade.entry_signal_1s) * 0.3 and hold_events > 20:
                exit_signal = True
                exit_reason = "burst_fade"

            # Or when direction reverses
            if np.sign(p1[i]) == -current_trade.direction and hold_events > 10:
                exit_signal = True
                exit_reason = "reversal"

            if hold_events >= 500:
                exit_signal = True
                exit_reason = "max_hold"

            if exit_signal:
                actual_move = labels[i, 2] - current_trade.entry_price if current_trade.entry_price != 0 else 0
                pnl = actual_move * current_trade.direction - cost_ticks
                current_trade.exit_idx = i
                current_trade.pnl_ticks = pnl
                current_trade.hold_events = hold_events
                current_trade.exit_reason = exit_reason
                trades.append(current_trade)
                in_trade = False
                current_trade = None
                last_exit_idx = i

    return trades


def strategy_embedding_mlp(fold_data: Dict, cost_ticks: float,
                            train_ratio: float = 0.5) -> Tuple[List[Trade], dict]:
    """
    Strategy 5: Small MLP trained on Mamba embeddings to predict optimal trade action.
    Target: whether holding for 10s from this point is profitable after costs.
    Uses first half of fold for training, second half for testing (no leakage).
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
    except ImportError:
        log.warning("PyTorch not available — skipping MLP strategy")
        return [], {}

    preds = fold_data['preds']
    labels = fold_data['labels']
    embeddings = fold_data['embeddings']
    p1, p5, p10 = preds[:, 0], preds[:, 1], preds[:, 2]

    n = len(preds)
    train_n = int(n * train_ratio)

    # Create target: is the 10s move profitable after costs?
    # For each event: if we went in the predicted direction, would we profit?
    direction = np.sign(p10)
    profit = labels[:, 2] * direction - cost_ticks  # 10s pnl after costs
    target = (profit > 0).astype(np.float32)

    # Features: embeddings + multi-horizon preds + confidence metrics
    confidence = compute_confidence(p1, p5, p10)
    coherence = compute_signal_coherence(p1, p5, p10)
    features = np.hstack([
        embeddings,                            # 96-dim
        preds,                                 # 3 (1s, 5s, 10s)
        confidence.reshape(-1, 1),             # 1
        coherence.reshape(-1, 1),              # 1
        np.abs(preds),                         # 3 (absolute magnitudes)
    ])  # Total: 96 + 3 + 1 + 1 + 3 = 104

    # Train/test split (temporal, no leakage)
    X_train = torch.FloatTensor(features[:train_n])
    y_train = torch.FloatTensor(target[:train_n])
    X_test = torch.FloatTensor(features[train_n:])
    y_test = torch.FloatTensor(target[train_n:])

    # Small MLP
    input_dim = features.shape[1]
    model = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(32, 1),
        nn.Sigmoid(),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCELoss()

    # Train
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=512, shuffle=True)

    model.train()
    for epoch in range(10):
        epoch_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb).squeeze()
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_probs = model(X_test).squeeze().numpy()

    # Generate trades from MLP predictions
    test_preds = preds[train_n:]
    test_labels = labels[train_n:]
    test_p10 = test_preds[:, 2]

    trades = []
    in_trade = False
    current_trade = None

    for i in range(len(test_probs)):
        if not in_trade:
            # Enter when MLP says profitable AND confidence > 0.6
            if test_probs[i] > 0.6 and np.abs(test_p10[i]) > 0.01:
                direction = int(np.sign(test_p10[i]))
                current_trade = Trade(
                    entry_idx=train_n + i,
                    entry_price=test_labels[i, 2],
                    direction=direction,
                    entry_signal_1s=float(test_preds[i, 0]),
                    entry_signal_5s=float(test_preds[i, 1]),
                    entry_signal_10s=float(test_preds[i, 2]),
                    entry_confidence=float(test_probs[i]),
                )
                in_trade = True
        else:
            hold_events = (train_n + i) - current_trade.entry_idx
            price_move = test_labels[i, 0] * current_trade.direction
            current_trade.mfe_ticks = max(current_trade.mfe_ticks, price_move)
            current_trade.mae_ticks = min(current_trade.mae_ticks, price_move)

            # Exit when MLP says unprofitable OR signal flips
            exit_signal = False
            exit_reason = ""

            if test_probs[i] < 0.4 and hold_events > 20:
                exit_signal = True
                exit_reason = "mlp_bearish"

            if np.sign(test_p10[i]) == -current_trade.direction and hold_events > 10:
                exit_signal = True
                exit_reason = "signal_flip"

            if hold_events >= 2000:
                exit_signal = True
                exit_reason = "max_hold"

            if exit_signal:
                actual_move = test_labels[i, 2] - current_trade.entry_price if current_trade.entry_price != 0 else 0
                pnl = actual_move * current_trade.direction - cost_ticks
                current_trade.exit_idx = train_n + i
                current_trade.pnl_ticks = pnl
                current_trade.hold_events = hold_events
                current_trade.exit_reason = exit_reason
                trades.append(current_trade)
                in_trade = False
                current_trade = None

    mlp_stats = {
        'train_n': train_n,
        'test_n': len(test_probs),
        'mlp_mean_prob': float(test_probs.mean()),
        'mlp_trades_above_06': int((test_probs > 0.6).sum()),
        'target_positive_rate': float(target[train_n:].mean()),
    }

    return trades, mlp_stats


def evaluate_trades(trades: List[Trade], strategy_name: str, cost_model: str,
                    total_events: int = 0, n_days: int = 1) -> StrategyResult:
    """Compute comprehensive metrics from a list of trades."""
    result = StrategyResult(name=strategy_name, cost_model=cost_model)

    if not trades:
        return result

    result.n_trades = len(trades)
    pnls = np.array([t.pnl_ticks for t in trades])
    holds = np.array([t.hold_events for t in trades])
    mfes = np.array([t.mfe_ticks for t in trades])
    maes = np.array([t.mae_ticks for t in trades])

    result.n_winners = int((pnls > 0).sum())
    result.total_pnl_ticks = float(pnls.sum())
    result.total_pnl_dollars = result.total_pnl_ticks * TICK_VALUE
    result.avg_pnl_per_trade = float(pnls.mean())
    result.win_rate = result.n_winners / result.n_trades if result.n_trades > 0 else 0
    result.avg_hold_events = float(holds.mean())
    result.avg_mfe = float(mfes.mean())
    result.avg_mae = float(maes.mean())
    result.trades_per_day = result.n_trades / max(1, n_days)

    winners = pnls[pnls > 0]
    losers = pnls[pnls <= 0]
    result.avg_winner = float(winners.mean()) if len(winners) > 0 else 0
    result.avg_loser = float(losers.mean()) if len(losers) > 0 else 0

    gross_profit = winners.sum() if len(winners) > 0 else 0
    gross_loss = abs(losers.sum()) if len(losers) > 0 else 1e-8
    result.profit_factor = float(gross_profit / gross_loss)

    # Sortino ratio (annualized, using downside deviation)
    if len(pnls) > 1:
        downside = pnls[pnls < 0]
        downside_std = np.std(downside) if len(downside) > 0 else 1e-8
        daily_return = result.total_pnl_ticks / max(1, n_days)
        result.sortino_ratio = float(daily_return / downside_std) if downside_std > 0 else 0

    # Max drawdown
    cumsum = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumsum)
    drawdown = running_max - cumsum
    result.max_drawdown_ticks = float(drawdown.max()) if len(drawdown) > 0 else 0

    # Long/short breakdown
    long_pnls = [t.pnl_ticks for t in trades if t.direction == 1]
    short_pnls = [t.pnl_ticks for t in trades if t.direction == -1]
    result.long_pnl = float(sum(long_pnls))
    result.short_pnl = float(sum(short_pnls))

    # Exit reason breakdown
    exit_reasons = defaultdict(int)
    for t in trades:
        exit_reasons[t.exit_reason] += 1
    result.details = {
        'exit_reasons': dict(exit_reasons),
        'n_long': len(long_pnls),
        'n_short': len(short_pnls),
        'long_win_rate': sum(1 for p in long_pnls if p > 0) / max(1, len(long_pnls)),
        'short_win_rate': sum(1 for p in short_pnls if p > 0) / max(1, len(short_pnls)),
    }

    return result


def print_results_table(results: List[StrategyResult]):
    """Print formatted comparison table."""
    log.info("\n" + "=" * 120)
    log.info("EXECUTION STRATEGY COMPARISON — Mamba v7 March OOT Predictions")
    log.info("=" * 120)

    header = f"{'Strategy':<35} {'Trades':>7} {'WinRate':>8} {'PnL($)':>10} {'$/Trade':>9} {'PF':>6} {'Sortino':>8} {'AvgHold':>8} {'MFE':>7} {'MAE':>7}"
    log.info(header)
    log.info("-" * 120)

    for r in sorted(results, key=lambda x: x.total_pnl_dollars, reverse=True):
        line = (f"{r.name:<35} {r.n_trades:>7} {r.win_rate:>7.1%} "
                f"${r.total_pnl_dollars:>9,.0f} ${r.avg_pnl_per_trade * TICK_VALUE:>8.2f} "
                f"{r.profit_factor:>5.2f} {r.sortino_ratio:>7.3f} "
                f"{r.avg_hold_events:>7.0f} {r.avg_mfe:>6.3f} {r.avg_mae:>6.3f}")
        log.info(line)

    log.info("=" * 120)

    # Best strategy details
    best = max(results, key=lambda x: x.sortino_ratio) if results else None
    if best and best.n_trades > 0:
        log.info(f"\nBest by Sortino: {best.name}")
        log.info(f"  Trades: {best.n_trades} ({best.trades_per_day:.1f}/day)")
        log.info(f"  Win rate: {best.win_rate:.1%} | PF: {best.profit_factor:.2f}")
        log.info(f"  Total P&L: ${best.total_pnl_dollars:,.0f} | Per trade: ${best.avg_pnl_per_trade * TICK_VALUE:.2f}")
        log.info(f"  Long: ${best.long_pnl * TICK_VALUE:,.0f} | Short: ${best.short_pnl * TICK_VALUE:,.0f}")
        log.info(f"  Avg MFE: {best.avg_mfe:.3f} | Avg MAE: {best.avg_mae:.3f}")
        log.info(f"  Max DD: ${best.max_drawdown_ticks * TICK_VALUE:,.0f}")
        log.info(f"  Exit reasons: {best.details.get('exit_reasons', {})}")


def main():
    log.info("=" * 80)
    log.info("MAMBA + PATCHTST CREATIVE EXECUTION STRATEGY ENGINE")
    log.info(f"  Cost model: commission={COMMISSION_TICKS:.3f} ticks ($4.70 RT)")
    log.info("=" * 80)

    # Load per-fold Mamba predictions (for date-level testing)
    log.info("\nLoading Mamba v7 per-fold predictions...")
    folds = load_per_fold_mamba()

    if not folds:
        log.error("No Mamba predictions found!")
        return

    # Also load concat for MLP training (needs more data)
    log.info("\nLoading concat predictions for MLP...")
    concat_data = load_mamba_predictions()

    all_results = []
    cost_models = {
        'mid_price': COMMISSION_TICKS,                    # 0.376 ticks (best case)
        'limit_order': COMMISSION_TICKS + SPREAD_HALF_TICK * 0.3,  # partial fill cost
        'market_order': COMMISSION_TICKS + 1.0,           # commission + full spread
    }

    # Test all strategies across all folds, for each cost model
    for cost_name, cost_ticks in cost_models.items():
        log.info(f"\n{'='*60}")
        log.info(f"COST MODEL: {cost_name} ({cost_ticks:.3f} ticks)")
        log.info(f"{'='*60}")

        for fold in folds:
            if fold['n_events'] < 1000:
                continue  # Skip tiny folds

            fold_tag = f"F{fold['fold_idx']:02d}_{fold['date']}"
            log.info(f"\n--- {fold_tag} ({fold['n_events']:,} events) ---")

            # Strategy 1: Multi-horizon agreement (multiple conf thresholds)
            for conf in [0.05, 0.10, 0.15, 0.20]:
                for hold_mode in ['signal_flip', 'conditional_flip', 'slow_flip']:
                    trades = strategy_multi_horizon_agree(
                        fold, conf_threshold=conf, cost_ticks=cost_ticks,
                        hold_mode=hold_mode)
                    name = f"MH_agree_{hold_mode}_c{conf:.2f}"
                    result = evaluate_trades(trades, name, cost_name, fold['n_events'])
                    result.details['fold'] = fold_tag
                    if result.n_trades > 0:
                        all_results.append(result)

            # Strategy 2: 1s entry / 10s hold
            for entry_thresh in [0.05, 0.10, 0.15]:
                trades = strategy_1s_entry_10s_hold(
                    fold, entry_threshold=entry_thresh,
                    hold_threshold=0.02, cost_ticks=cost_ticks)
                name = f"1s_entry_10s_hold_e{entry_thresh:.2f}"
                result = evaluate_trades(trades, name, cost_name, fold['n_events'])
                result.details['fold'] = fold_tag
                if result.n_trades > 0:
                    all_results.append(result)

            # Strategy 3: Coherence-scaled
            for coh_thresh in [0.03, 0.05, 0.08]:
                trades = strategy_coherence_scaled(
                    fold, coherence_threshold=coh_thresh, cost_ticks=cost_ticks)
                name = f"coherence_scaled_c{coh_thresh:.2f}"
                result = evaluate_trades(trades, name, cost_name, fold['n_events'])
                result.details['fold'] = fold_tag
                if result.n_trades > 0:
                    all_results.append(result)

            # Strategy 4: Momentum burst
            for burst in [0.02, 0.05, 0.08]:
                trades = strategy_momentum_burst(
                    fold, burst_threshold=burst, cost_ticks=cost_ticks)
                name = f"momentum_burst_b{burst:.2f}"
                result = evaluate_trades(trades, name, cost_name, fold['n_events'])
                result.details['fold'] = fold_tag
                if result.n_trades > 0:
                    all_results.append(result)

    # Strategy 5: MLP execution optimizer (uses concat data, needs more events)
    if concat_data:
        log.info(f"\n{'='*60}")
        log.info("STRATEGY 5: EMBEDDING MLP EXECUTION OPTIMIZER")
        log.info(f"{'='*60}")

        # Reshape concat data into fold format for MLP
        mlp_fold = {
            'preds': np.stack([concat_data['preds_1s'],
                              concat_data['preds_5s'],
                              concat_data['preds_10s']], axis=1),
            'labels': np.stack([concat_data['labels_1s'],
                               concat_data['labels_5s'],
                               concat_data['labels_10s']], axis=1),
            'embeddings': concat_data['embeddings'],
        }

        for cost_name, cost_ticks in cost_models.items():
            trades, mlp_stats = strategy_embedding_mlp(
                mlp_fold, cost_ticks=cost_ticks, train_ratio=0.5)
            name = f"embedding_mlp"
            n_days = len(folds)
            result = evaluate_trades(trades, name, cost_name, mlp_fold['preds'].shape[0], n_days)
            result.details['mlp_stats'] = mlp_stats
            if result.n_trades > 0:
                all_results.append(result)
                log.info(f"  MLP ({cost_name}): {result.n_trades} trades, "
                        f"WR={result.win_rate:.1%}, PnL=${result.total_pnl_dollars:,.0f}, "
                        f"Sortino={result.sortino_ratio:.3f}")

    # Aggregate results by strategy (across folds)
    log.info("\n\n" + "=" * 120)
    log.info("AGGREGATE RESULTS BY STRATEGY (across all folds, mid_price cost)")
    log.info("=" * 120)

    # Group by strategy name for mid_price cost model
    mid_results = [r for r in all_results if r.cost_model == 'mid_price']
    strategy_groups = defaultdict(list)
    for r in mid_results:
        strategy_groups[r.name].append(r)

    agg_results = []
    for name, group in strategy_groups.items():
        agg = StrategyResult(name=name, cost_model='mid_price')
        agg.n_trades = sum(r.n_trades for r in group)
        agg.n_winners = sum(r.n_winners for r in group)
        agg.total_pnl_ticks = sum(r.total_pnl_ticks for r in group)
        agg.total_pnl_dollars = agg.total_pnl_ticks * TICK_VALUE
        agg.avg_pnl_per_trade = agg.total_pnl_ticks / max(1, agg.n_trades)
        agg.win_rate = agg.n_winners / max(1, agg.n_trades)
        agg.trades_per_day = agg.n_trades / max(1, len(group))

        all_pnls = []
        for r in group:
            # Approximate per-trade PnLs from aggregate
            if r.n_trades > 0:
                all_pnls.extend([r.avg_pnl_per_trade] * r.n_trades)

        if all_pnls:
            pnls_arr = np.array(all_pnls)
            winners = pnls_arr[pnls_arr > 0]
            losers = pnls_arr[pnls_arr <= 0]
            agg.avg_winner = float(winners.mean()) if len(winners) > 0 else 0
            agg.avg_loser = float(losers.mean()) if len(losers) > 0 else 0
            gross_profit = winners.sum() if len(winners) > 0 else 0
            gross_loss = abs(losers.sum()) if len(losers) > 0 else 1e-8
            agg.profit_factor = float(gross_profit / gross_loss)

        agg.avg_mfe = np.mean([r.avg_mfe for r in group if r.n_trades > 0]) if group else 0
        agg.avg_mae = np.mean([r.avg_mae for r in group if r.n_trades > 0]) if group else 0
        agg.avg_hold_events = np.mean([r.avg_hold_events for r in group if r.n_trades > 0]) if group else 0
        agg.long_pnl = sum(r.long_pnl for r in group)
        agg.short_pnl = sum(r.short_pnl for r in group)

        agg_results.append(agg)

    print_results_table(agg_results)

    # Save all results
    save_path = RESULTS_DIR / f'mamba_patchtst_exec_{_ts}.json'
    serializable = []
    for r in all_results:
        d = asdict(r)
        # Convert numpy types
        for k, v in d.items():
            if isinstance(v, (np.floating, np.integer)):
                d[k] = float(v)
        serializable.append(d)

    with open(save_path, 'w') as f:
        json.dump({
            'timestamp': _ts,
            'n_strategies_tested': len(set(r.name for r in all_results)),
            'n_folds': len(folds),
            'cost_models': {k: float(v) for k, v in cost_models.items()},
            'results': serializable,
        }, f, indent=2, default=str)

    log.info(f"\nResults saved: {save_path}")
    log.info(f"Log saved: {RESULTS_DIR / f'mamba_patchtst_exec_{_ts}.log'}")


if __name__ == '__main__':
    main()
