#!/usr/bin/env python3
"""
Advanced Execution Strategies v3 — April 2026
==============================================
Builds on the top strategies from multi_model_execution_sweep.py:
  1. mamba_z3.0_slowflip_long: +$4,796, 312 trades, 64.1% WR, 11s avg hold
  2. mamba_z3.0_signalflip:    +$4,039, 372 trades, 63.2% WR, 5s avg hold
  3. agree_z2.0_signalflip:    +$3,336, 1031 trades, 57.8% WR

NEW strategies tested:
  1. Confidence-weighted position sizing (variable lots at Top0.1% vs Top1%)
  2. Time-of-day filter (midday 11am-2pm best window from v6 testing)
  3. Regime detection (rolling volatility → different z-thresholds)
  4. Multi-model cascade (Mamba entry → CNN-Mamba confirmation → trade)
  5. Adaptive z-threshold (start z=2.0, increase to z=3.0 after losses)
  6. Momentum burst (3+ consecutive agreeing signals → increase aggression)
  7. Anti-correlation exit (exit when models DISAGREE)

Data: Mamba v7 + CNN-Mamba v2, folds 5-9 (March 1-5)
Cost model: $4.70 RT = 0.376 ticks, plus 0.5 tick spread
"""

import sys
import gc
import json
import logging
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

import numpy as np

# ── Paths ──
LVL3_ROOT = Path(__file__).resolve().parent.parent
MAMBA_PRED_DIR = LVL3_ROOT / 'output' / 'mamba_v7_tiny_smart_v3_mar_apr'
CNN_MAMBA_PRED_DIR = LVL3_ROOT / 'output' / 'cnn_mamba_v2_smart_v3_mar'
RESULTS_DIR = LVL3_ROOT / 'execution' / 'results' / 'advanced_v3'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── NQ Futures Constants ──
TICK_SIZE = 0.25
TICK_VALUE = 5.00       # NQ = $5/tick (not ES)
POINT_VALUE = 20.00     # NQ = $20/point
COMMISSION_RT = 4.70    # $4.70 round trip
COMMISSION_TICKS = COMMISSION_RT / TICK_VALUE  # 0.94 ticks for NQ
SPREAD_COST_TICKS = 0.5

# ── Model Constants ──
WINDOW = 1000
STRIDE = 500
BARS_PER_SEC = 10       # ~10 events per second
BAR_NS = 100_000_000    # 100ms per bar

# ── Fold-to-date mapping (March 2026) ──
FOLD_DATES = {
    5: '20260301',
    6: '20260302',
    7: '20260303',
    8: '20260304',
    9: '20260305',
}

# Approximate time-of-day bins (event index → hour EST)
# Trading day ~6:30am-5pm ET, most active 9:30am-4pm
# With ~30k events/day over ~10.5 hrs ≈ 2857 events/hr
# Index 0 = market open (~6:30am ET for futures)
EVENTS_PER_HOUR = 2857  # approximate

# ── Logging ──
_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
log = logging.getLogger('advanced_v3')
log.setLevel(logging.INFO)
_fh = logging.FileHandler(str(RESULTS_DIR / f'advanced_v3_{_ts}.log'), mode='w')
_fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
log.addHandler(_fh)
_ch = logging.StreamHandler(sys.stdout)
_ch.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
log.addHandler(_ch)


# ============================================================
# Data loading
# ============================================================
def load_fold_predictions(fold_idx: int) -> Optional[dict]:
    """Load both Mamba v7 and CNN-Mamba v2 predictions for a fold."""
    date_str = FOLD_DATES.get(fold_idx, f'fold{fold_idx}')

    mamba_path = MAMBA_PRED_DIR / f'fold_{fold_idx:02d}_oot_predictions.npz'
    cnn_mamba_path = CNN_MAMBA_PRED_DIR / f'fold_{fold_idx:02d}_oot_predictions.npz'

    if not mamba_path.exists():
        log.warning(f"Mamba predictions not found: {mamba_path}")
        return None
    if not cnn_mamba_path.exists():
        log.warning(f"CNN-Mamba predictions not found: {cnn_mamba_path}")
        return None

    try:
        mamba_data = np.load(mamba_path, allow_pickle=True)
        cnn_data = np.load(cnn_mamba_path, allow_pickle=True)

        mamba_preds = mamba_data['predictions']  # (N, 3) = 1s, 5s, 10s
        cnn_preds = cnn_data['predictions']      # (N, 3) = 1s, 5s, 10s
        labels = mamba_data['labels']            # (N, 3)

        if mamba_preds.shape[0] != cnn_preds.shape[0]:
            log.warning(f"Fold {fold_idx}: shape mismatch {mamba_preds.shape} vs {cnn_preds.shape}, skipping")
            return None

        N = mamba_preds.shape[0]
        if N < 100:
            log.warning(f"Fold {fold_idx}: only {N} samples, skipping")
            return None

        log.info(f"Fold {fold_idx} ({date_str}): {N} samples loaded")

        return {
            'date': date_str,
            'fold': fold_idx,
            'mamba_preds': mamba_preds,
            'cnn_preds': cnn_preds,
            'labels': labels,
            'n_samples': N,
        }
    except Exception as e:
        log.error(f"Failed to load fold {fold_idx}: {e}")
        return None


# ============================================================
# Z-score computation (expanding, causal)
# ============================================================
def compute_z_scores(preds: np.ndarray) -> np.ndarray:
    """Expanding z-score normalization (causal -- no lookahead).
    Vectorized with cumulative stats."""
    N, H = preds.shape
    z = np.zeros_like(preds)
    for h in range(H):
        col = preds[:, h].astype(np.float64)
        cs = np.cumsum(col)
        css = np.cumsum(col ** 2)
        ns = np.arange(1, N + 1, dtype=np.float64)
        means = cs / ns
        vars_ = np.maximum(css / ns - means ** 2, 1e-12)
        stds = np.sqrt(vars_)
        mask = stds > 1e-8
        z[mask, h] = ((col[mask] - means[mask]) / stds[mask]).astype(np.float32)
    return z


# ============================================================
# Rolling volatility for regime detection
# ============================================================
def compute_rolling_vol(labels: np.ndarray, window: int = 500) -> np.ndarray:
    """Rolling std of 10s label returns as volatility proxy. Causal."""
    returns_10s = labels[:, 2]  # 10s forward return
    N = len(returns_10s)
    vol = np.zeros(N, dtype=np.float32)
    # Use expanding then rolling
    for i in range(N):
        start = max(0, i - window + 1)
        chunk = returns_10s[start:i+1]
        if len(chunk) > 10:
            vol[i] = np.std(chunk)
    return vol


# ============================================================
# Strategy 1: BASELINE (mamba_z3.0_slowflip_long)
# ============================================================
def strategy_baseline(data: dict) -> List[dict]:
    """Baseline: mamba_z3.0_slowflip_long — the current champion."""
    mamba_z = compute_z_scores(data['mamba_preds'])
    N = data['n_samples']
    z_thresh = 3.0
    min_hold = 100
    max_hold = 6000
    horizon = 2  # 10s

    trades = []
    position = 0
    entry_idx = 0
    entry_signal = 0.0

    for i in range(N):
        if position == 0:
            if abs(mamba_z[i, horizon]) > z_thresh:
                position = int(np.sign(mamba_z[i, horizon]))
                entry_idx = i
                entry_signal = float(mamba_z[i, horizon])
        else:
            held = i - entry_idx
            should_exit = False
            exit_reason = ''

            if held >= min_hold and i >= 2:
                dirs = [np.sign(mamba_z[i-j, 2]) for j in range(3)]
                if all(d == -position for d in dirs):
                    should_exit = True
                    exit_reason = 'slow_flip_3'
            if held >= max_hold:
                should_exit = True
                exit_reason = 'max_hold'

            if should_exit:
                trades.append({
                    'entry_idx': entry_idx, 'exit_idx': i,
                    'direction': position, 'hold_events': held,
                    'entry_signal': entry_signal, 'exit_reason': exit_reason,
                    'size': 1,
                })
                position = 0

    if position != 0:
        trades.append({
            'entry_idx': entry_idx, 'exit_idx': N-1,
            'direction': position, 'hold_events': N-1-entry_idx,
            'entry_signal': entry_signal, 'exit_reason': 'eod', 'size': 1,
        })
    return trades


# ============================================================
# Strategy 2: Confidence-weighted position sizing
# ============================================================
def strategy_confidence_sizing(data: dict) -> List[dict]:
    """Variable position size based on signal confidence.
    Top0.1% → 3 lots, Top1% → 2 lots, Top5% → 1 lot.
    Uses mamba z3.0 slowflip as base."""
    mamba_z = compute_z_scores(data['mamba_preds'])
    N = data['n_samples']

    # Compute z-score percentiles from data so far (expanding)
    abs_z_10s = np.abs(mamba_z[:, 2])

    z_thresh = 3.0
    min_hold = 100
    max_hold = 6000

    trades = []
    position = 0
    entry_idx = 0
    entry_signal = 0.0
    entry_size = 1

    for i in range(N):
        if position == 0:
            z_val = abs(mamba_z[i, 2])
            if z_val > z_thresh:
                position = int(np.sign(mamba_z[i, 2]))
                entry_idx = i
                entry_signal = float(mamba_z[i, 2])

                # Determine size based on percentile rank of this z among all seen so far
                if i > 100:
                    seen = abs_z_10s[:i]
                    pct = (seen < z_val).sum() / len(seen) * 100
                    if pct >= 99.9:
                        entry_size = 3
                    elif pct >= 99.0:
                        entry_size = 2
                    else:
                        entry_size = 1
                else:
                    entry_size = 1
        else:
            held = i - entry_idx
            should_exit = False
            exit_reason = ''

            if held >= min_hold and i >= 2:
                dirs = [np.sign(mamba_z[i-j, 2]) for j in range(3)]
                if all(d == -position for d in dirs):
                    should_exit = True
                    exit_reason = 'slow_flip_3'
            if held >= max_hold:
                should_exit = True
                exit_reason = 'max_hold'

            if should_exit:
                trades.append({
                    'entry_idx': entry_idx, 'exit_idx': i,
                    'direction': position, 'hold_events': held,
                    'entry_signal': entry_signal, 'exit_reason': exit_reason,
                    'size': entry_size,
                })
                position = 0

    if position != 0:
        trades.append({
            'entry_idx': entry_idx, 'exit_idx': N-1,
            'direction': position, 'hold_events': N-1-entry_idx,
            'entry_signal': entry_signal, 'exit_reason': 'eod', 'size': entry_size,
        })
    return trades


# ============================================================
# Strategy 3: Time-of-day filter (midday 11am-2pm)
# ============================================================
def strategy_time_of_day_filter(data: dict) -> List[dict]:
    """Only trade during 11am-2pm ET window (best from v6 testing).
    Futures data starts ~6:30am ET, so 11am ≈ index 4.5*EVENTS_PER_HOUR.
    Uses mamba z3.0 slowflip as base."""
    mamba_z = compute_z_scores(data['mamba_preds'])
    N = data['n_samples']

    # Midday window: ~11am-2pm ET = 4.5-7.5 hours from 6:30am open
    midday_start = int(4.5 * EVENTS_PER_HOUR)
    midday_end = int(7.5 * EVENTS_PER_HOUR)

    z_thresh = 3.0
    min_hold = 100
    max_hold = 6000

    trades = []
    position = 0
    entry_idx = 0
    entry_signal = 0.0

    for i in range(N):
        if position == 0:
            # Only enter during midday window
            if midday_start <= i <= midday_end:
                if abs(mamba_z[i, 2]) > z_thresh:
                    position = int(np.sign(mamba_z[i, 2]))
                    entry_idx = i
                    entry_signal = float(mamba_z[i, 2])
        else:
            held = i - entry_idx
            should_exit = False
            exit_reason = ''

            if held >= min_hold and i >= 2:
                dirs = [np.sign(mamba_z[i-j, 2]) for j in range(3)]
                if all(d == -position for d in dirs):
                    should_exit = True
                    exit_reason = 'slow_flip_3'
            if held >= max_hold:
                should_exit = True
                exit_reason = 'max_hold'

            if should_exit:
                trades.append({
                    'entry_idx': entry_idx, 'exit_idx': i,
                    'direction': position, 'hold_events': held,
                    'entry_signal': entry_signal, 'exit_reason': exit_reason,
                    'size': 1,
                })
                position = 0

    if position != 0:
        trades.append({
            'entry_idx': entry_idx, 'exit_idx': N-1,
            'direction': position, 'hold_events': N-1-entry_idx,
            'entry_signal': entry_signal, 'exit_reason': 'eod', 'size': 1,
        })
    return trades


# ============================================================
# Strategy 4: Regime detection (vol-adaptive thresholds)
# ============================================================
def strategy_regime_detection(data: dict) -> List[dict]:
    """Different z-thresholds for high vs low volatility regimes.
    High vol (top 30% rolling vol): z_thresh=3.5 (more selective, bigger moves)
    Low vol (bottom 30%): z_thresh=2.5 (lower bar since moves are smaller)
    Medium: z_thresh=3.0 (baseline)
    Uses slowflip exit."""
    mamba_z = compute_z_scores(data['mamba_preds'])
    vol = compute_rolling_vol(data['labels'], window=500)
    N = data['n_samples']

    min_hold = 100
    max_hold = 6000

    trades = []
    position = 0
    entry_idx = 0
    entry_signal = 0.0

    for i in range(500, N):  # skip first 500 for vol warmup
        # Determine regime
        if i > 500:
            vol_pct = (vol[:i] < vol[i]).sum() / i * 100
        else:
            vol_pct = 50

        if vol_pct >= 70:
            z_thresh = 3.5  # high vol → more selective
        elif vol_pct <= 30:
            z_thresh = 2.5  # low vol → lower threshold
        else:
            z_thresh = 3.0  # medium

        if position == 0:
            if abs(mamba_z[i, 2]) > z_thresh:
                position = int(np.sign(mamba_z[i, 2]))
                entry_idx = i
                entry_signal = float(mamba_z[i, 2])
        else:
            held = i - entry_idx
            should_exit = False
            exit_reason = ''

            if held >= min_hold and i >= 2:
                dirs = [np.sign(mamba_z[i-j, 2]) for j in range(3)]
                if all(d == -position for d in dirs):
                    should_exit = True
                    exit_reason = 'slow_flip_3'
            if held >= max_hold:
                should_exit = True
                exit_reason = 'max_hold'

            if should_exit:
                trades.append({
                    'entry_idx': entry_idx, 'exit_idx': i,
                    'direction': position, 'hold_events': held,
                    'entry_signal': entry_signal, 'exit_reason': exit_reason,
                    'size': 1, 'regime_vol_pct': round(vol_pct, 1),
                })
                position = 0

    if position != 0:
        trades.append({
            'entry_idx': entry_idx, 'exit_idx': N-1,
            'direction': position, 'hold_events': N-1-entry_idx,
            'entry_signal': entry_signal, 'exit_reason': 'eod', 'size': 1,
        })
    return trades


# ============================================================
# Strategy 5: Multi-model cascade
# ============================================================
def strategy_cascade(data: dict) -> List[dict]:
    """Mamba triggers entry candidate, CNN-Mamba must confirm within 50 events.
    Entry only when BOTH models agree. Mamba slowflip exit.
    More selective than simple agreement — requires Mamba to fire first, then
    CNN-Mamba must independently confirm within a short window."""
    mamba_z = compute_z_scores(data['mamba_preds'])
    cnn_z = compute_z_scores(data['cnn_preds'])
    N = data['n_samples']

    z_thresh_mamba = 3.0
    z_thresh_cnn = 2.0  # lower bar for confirmation
    confirm_window = 50  # 5 seconds to confirm
    min_hold = 100
    max_hold = 6000

    trades = []
    position = 0
    entry_idx = 0
    entry_signal = 0.0
    pending_dir = 0
    pending_idx = -1

    for i in range(N):
        if position == 0:
            # Check for Mamba trigger
            if abs(mamba_z[i, 2]) > z_thresh_mamba:
                pending_dir = int(np.sign(mamba_z[i, 2]))
                pending_idx = i

            # Check for CNN-Mamba confirmation
            if pending_dir != 0 and (i - pending_idx) <= confirm_window:
                cnn_dir = np.sign(cnn_z[i, 2])
                if cnn_dir == pending_dir and abs(cnn_z[i, 2]) > z_thresh_cnn:
                    position = pending_dir
                    entry_idx = i
                    entry_signal = float(mamba_z[pending_idx, 2])
                    pending_dir = 0
            elif pending_dir != 0 and (i - pending_idx) > confirm_window:
                pending_dir = 0  # expired
        else:
            held = i - entry_idx
            should_exit = False
            exit_reason = ''

            if held >= min_hold and i >= 2:
                dirs = [np.sign(mamba_z[i-j, 2]) for j in range(3)]
                if all(d == -position for d in dirs):
                    should_exit = True
                    exit_reason = 'slow_flip_3'
            if held >= max_hold:
                should_exit = True
                exit_reason = 'max_hold'

            if should_exit:
                trades.append({
                    'entry_idx': entry_idx, 'exit_idx': i,
                    'direction': position, 'hold_events': held,
                    'entry_signal': entry_signal, 'exit_reason': exit_reason,
                    'size': 1,
                })
                position = 0

    if position != 0:
        trades.append({
            'entry_idx': entry_idx, 'exit_idx': N-1,
            'direction': position, 'hold_events': N-1-entry_idx,
            'entry_signal': entry_signal, 'exit_reason': 'eod', 'size': 1,
        })
    return trades


# ============================================================
# Strategy 6: Adaptive z-threshold (increase after losses)
# ============================================================
def strategy_adaptive_z(data: dict) -> List[dict]:
    """Start day at z=2.0, increase to z=3.0 after consecutive losses.
    After 2 consecutive losses: z=2.5
    After 3 consecutive losses: z=3.0
    After a win: reset to z=2.0
    Uses slowflip exit."""
    mamba_z = compute_z_scores(data['mamba_preds'])
    labels = data['labels']
    N = data['n_samples']

    base_z = 2.0
    consecutive_losses = 0
    min_hold = 100
    max_hold = 6000

    trades = []
    position = 0
    entry_idx = 0
    entry_signal = 0.0

    for i in range(N):
        # Adaptive threshold
        if consecutive_losses >= 3:
            z_thresh = 3.0
        elif consecutive_losses >= 2:
            z_thresh = 2.5
        else:
            z_thresh = base_z

        if position == 0:
            if abs(mamba_z[i, 2]) > z_thresh:
                position = int(np.sign(mamba_z[i, 2]))
                entry_idx = i
                entry_signal = float(mamba_z[i, 2])
        else:
            held = i - entry_idx
            should_exit = False
            exit_reason = ''

            if held >= min_hold and i >= 2:
                dirs = [np.sign(mamba_z[i-j, 2]) for j in range(3)]
                if all(d == -position for d in dirs):
                    should_exit = True
                    exit_reason = 'slow_flip_3'
            if held >= max_hold:
                should_exit = True
                exit_reason = 'max_hold'

            if should_exit:
                trade = {
                    'entry_idx': entry_idx, 'exit_idx': i,
                    'direction': position, 'hold_events': held,
                    'entry_signal': entry_signal, 'exit_reason': exit_reason,
                    'size': 1, 'z_thresh_used': z_thresh,
                }
                trades.append(trade)

                # Quick P&L check to update consecutive losses
                if entry_idx < len(labels):
                    if held <= 15:
                        pnl = position * labels[entry_idx, 0]
                    elif held <= 60:
                        pnl = position * labels[entry_idx, 1]
                    else:
                        pnl = position * labels[entry_idx, 2]
                    if pnl > SPREAD_COST_TICKS + COMMISSION_TICKS:
                        consecutive_losses = 0
                    else:
                        consecutive_losses += 1

                position = 0

    if position != 0:
        trades.append({
            'entry_idx': entry_idx, 'exit_idx': N-1,
            'direction': position, 'hold_events': N-1-entry_idx,
            'entry_signal': entry_signal, 'exit_reason': 'eod', 'size': 1,
        })
    return trades


# ============================================================
# Strategy 7: Momentum burst (consecutive signals → aggression)
# ============================================================
def strategy_momentum_burst(data: dict) -> List[dict]:
    """When 3+ consecutive signal events agree on direction, increase aggression.
    Normal: z=3.0 for entry
    Burst mode: z=2.0 for entry + 2 lots (signal aligned 3+ times in last 10 events)
    Uses slowflip exit."""
    mamba_z = compute_z_scores(data['mamba_preds'])
    N = data['n_samples']

    min_hold = 100
    max_hold = 6000
    lookback = 10  # check last 10 events for streak

    trades = []
    position = 0
    entry_idx = 0
    entry_signal = 0.0
    entry_size = 1

    for i in range(lookback, N):
        if position == 0:
            # Check for momentum streak
            recent_dirs = [np.sign(mamba_z[i-j, 2]) for j in range(min(lookback, i+1))]
            current_dir = np.sign(mamba_z[i, 2])

            if current_dir != 0:
                streak = sum(1 for d in recent_dirs if d == current_dir)

                if streak >= 3 and abs(mamba_z[i, 2]) > 2.0:
                    # Burst mode: lower threshold, bigger size
                    position = int(current_dir)
                    entry_idx = i
                    entry_signal = float(mamba_z[i, 2])
                    entry_size = 2
                elif abs(mamba_z[i, 2]) > 3.0:
                    # Normal mode
                    position = int(current_dir)
                    entry_idx = i
                    entry_signal = float(mamba_z[i, 2])
                    entry_size = 1
        else:
            held = i - entry_idx
            should_exit = False
            exit_reason = ''

            if held >= min_hold and i >= 2:
                dirs = [np.sign(mamba_z[i-j, 2]) for j in range(3)]
                if all(d == -position for d in dirs):
                    should_exit = True
                    exit_reason = 'slow_flip_3'
            if held >= max_hold:
                should_exit = True
                exit_reason = 'max_hold'

            if should_exit:
                trades.append({
                    'entry_idx': entry_idx, 'exit_idx': i,
                    'direction': position, 'hold_events': held,
                    'entry_signal': entry_signal, 'exit_reason': exit_reason,
                    'size': entry_size,
                })
                position = 0

    if position != 0:
        trades.append({
            'entry_idx': entry_idx, 'exit_idx': N-1,
            'direction': position, 'hold_events': N-1-entry_idx,
            'entry_signal': entry_signal, 'exit_reason': 'eod', 'size': entry_size,
        })
    return trades


# ============================================================
# Strategy 8: Anti-correlation exit
# ============================================================
def strategy_anti_correlation_exit(data: dict) -> List[dict]:
    """Enter on Mamba z>3.0. Exit when models DISAGREE (not signal flip).
    Instead of waiting for Mamba to flip, exit immediately when CNN-Mamba
    and Mamba predict opposite directions. Hypothesis: disagreement signals
    uncertainty, better to exit early than wait for flip."""
    mamba_z = compute_z_scores(data['mamba_preds'])
    cnn_z = compute_z_scores(data['cnn_preds'])
    N = data['n_samples']

    z_thresh = 3.0
    min_hold = 50   # shorter min hold since exit is earlier
    max_hold = 3000  # shorter max hold too

    trades = []
    position = 0
    entry_idx = 0
    entry_signal = 0.0

    for i in range(N):
        if position == 0:
            if abs(mamba_z[i, 2]) > z_thresh:
                position = int(np.sign(mamba_z[i, 2]))
                entry_idx = i
                entry_signal = float(mamba_z[i, 2])
        else:
            held = i - entry_idx
            should_exit = False
            exit_reason = ''

            if held >= min_hold:
                # Anti-correlation: models disagree
                mamba_dir = np.sign(mamba_z[i, 2])
                cnn_dir = np.sign(cnn_z[i, 2])
                if mamba_dir != 0 and cnn_dir != 0 and mamba_dir != cnn_dir:
                    # Only exit if both have strong enough signals
                    if abs(mamba_z[i, 2]) > 1.0 and abs(cnn_z[i, 2]) > 1.0:
                        should_exit = True
                        exit_reason = 'anti_correlation'

            if held >= max_hold:
                should_exit = True
                exit_reason = 'max_hold'

            if should_exit:
                trades.append({
                    'entry_idx': entry_idx, 'exit_idx': i,
                    'direction': position, 'hold_events': held,
                    'entry_signal': entry_signal, 'exit_reason': exit_reason,
                    'size': 1,
                })
                position = 0

    if position != 0:
        trades.append({
            'entry_idx': entry_idx, 'exit_idx': N-1,
            'direction': position, 'hold_events': N-1-entry_idx,
            'entry_signal': entry_signal, 'exit_reason': 'eod', 'size': 1,
        })
    return trades


# ============================================================
# P&L simulation from labels
# ============================================================
def simulate_pnl(trades: List[dict], data: dict) -> List[dict]:
    """Estimate P&L from forward labels. Accounts for position size."""
    labels = data['labels']

    for trade in trades:
        entry_idx = trade['entry_idx']
        direction = trade['direction']
        hold_events = trade['hold_events']
        size = trade.get('size', 1)

        if entry_idx < len(labels):
            if hold_events <= 15:
                pnl_ticks = direction * labels[entry_idx, 0]
            elif hold_events <= 60:
                pnl_ticks = direction * labels[entry_idx, 1]
            else:
                pnl_ticks = direction * labels[entry_idx, 2]
        else:
            pnl_ticks = 0

        total_cost = SPREAD_COST_TICKS + COMMISSION_TICKS
        net_pnl_ticks = pnl_ticks - total_cost

        # Scale by position size
        trade['pnl_ticks'] = float(pnl_ticks)
        trade['net_pnl_ticks'] = float(net_pnl_ticks)
        trade['pnl_dollars'] = float(net_pnl_ticks * TICK_VALUE * size)
        trade['cost_ticks'] = float(total_cost)
        trade['label_horizon'] = '1s' if hold_events <= 15 else ('5s' if hold_events <= 60 else '10s')

        # MFE/MAE proxy
        if entry_idx < len(labels):
            forward_moves = direction * labels[entry_idx]
            trade['mfe_ticks'] = float(max(0, np.max(forward_moves)))
            trade['mae_ticks'] = float(min(0, np.min(forward_moves)))
        else:
            trade['mfe_ticks'] = 0.0
            trade['mae_ticks'] = 0.0

    return trades


# ============================================================
# Performance analysis
# ============================================================
def analyze_performance(trades: List[dict], n_days: int) -> dict:
    """Compute comprehensive performance metrics."""
    if not trades:
        return {'n_trades': 0, 'total_pnl': 0, 'win_rate': 0, 'sortino': 0}

    pnls = np.array([t.get('pnl_dollars', 0) for t in trades])
    holds = np.array([t['hold_events'] for t in trades])
    sizes = np.array([t.get('size', 1) for t in trades])

    n_trades = len(trades)
    total_pnl = float(np.sum(pnls))
    wins = (pnls > 0).sum()
    losses = (pnls <= 0).sum()
    win_rate = float(wins / n_trades) if n_trades > 0 else 0

    # Sortino ratio
    neg_returns = pnls[pnls < 0]
    downside_std = np.std(neg_returns) if len(neg_returns) > 1 else 1e-8
    sortino = float(np.mean(pnls) / downside_std) if downside_std > 1e-8 else 0.0

    # Profit factor
    gross_profit = float(np.sum(pnls[pnls > 0]))
    gross_loss = float(abs(np.sum(pnls[pnls < 0])))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Average hold time
    avg_hold_events = float(np.mean(holds))
    avg_hold_seconds = avg_hold_events / BARS_PER_SEC

    # Trades per day
    trades_per_day = n_trades / max(n_days, 1)

    # Max drawdown (cumulative)
    cum_pnl = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum_pnl)
    drawdown = peak - cum_pnl
    max_drawdown = float(np.max(drawdown)) if len(drawdown) > 0 else 0

    # Average size
    avg_size = float(np.mean(sizes))

    # Exit reason breakdown
    exit_reasons = {}
    for t in trades:
        r = t.get('exit_reason', 'unknown')
        exit_reasons[r] = exit_reasons.get(r, 0) + 1

    # Long/short breakdown
    longs = [t for t in trades if t['direction'] == 1]
    shorts = [t for t in trades if t['direction'] == -1]
    long_pnl = sum(t.get('pnl_dollars', 0) for t in longs)
    short_pnl = sum(t.get('pnl_dollars', 0) for t in shorts)

    return {
        'n_trades': n_trades,
        'total_pnl': round(total_pnl, 2),
        'avg_pnl': round(float(np.mean(pnls)), 2),
        'win_rate': round(win_rate, 4),
        'sortino': round(sortino, 3),
        'profit_factor': round(profit_factor, 3),
        'avg_hold_seconds': round(avg_hold_seconds, 1),
        'trades_per_day': round(trades_per_day, 1),
        'max_drawdown': round(max_drawdown, 2),
        'avg_size': round(avg_size, 2),
        'gross_profit': round(gross_profit, 2),
        'gross_loss': round(gross_loss, 2),
        'long_trades': len(longs),
        'long_pnl': round(long_pnl, 2),
        'short_trades': len(shorts),
        'short_pnl': round(short_pnl, 2),
        'exit_reasons': exit_reasons,
    }


# ============================================================
# Confidence tier drill-down
# ============================================================
def tier_analysis(trades: List[dict]) -> dict:
    """Performance at different confidence tiers."""
    if not trades:
        return {}

    signals = np.array([abs(t['entry_signal']) for t in trades])
    pnls = np.array([t.get('pnl_dollars', 0) for t in trades])

    tiers = {'All': 0, 'Top25%': 75, 'Top10%': 90, 'Top5%': 95, 'Top1%': 99}
    results = {}

    for tier_name, pct in tiers.items():
        if pct == 0:
            mask = np.ones(len(signals), dtype=bool)
        else:
            threshold = np.percentile(signals, pct)
            mask = signals >= threshold

        if mask.sum() == 0:
            continue

        tier_pnls = pnls[mask]
        n = len(tier_pnls)
        total = float(np.sum(tier_pnls))
        wr = float((tier_pnls > 0).sum() / n) if n > 0 else 0
        neg = tier_pnls[tier_pnls < 0]
        ds = np.std(neg) if len(neg) > 1 else 1e-8
        sort_r = float(np.mean(tier_pnls) / ds) if ds > 1e-8 else 0

        results[tier_name] = {
            'n_trades': int(n),
            'total_pnl': round(total, 2),
            'win_rate': round(wr, 4),
            'sortino': round(sort_r, 3),
        }

    return results


# ============================================================
# Main
# ============================================================
STRATEGY_RUNNERS = {
    '0_baseline_mamba_z3_slowflip_long': strategy_baseline,
    '1_confidence_sizing': strategy_confidence_sizing,
    '2_time_of_day_midday': strategy_time_of_day_filter,
    '3_regime_detection': strategy_regime_detection,
    '4_cascade_mamba_cnn': strategy_cascade,
    '5_adaptive_z_threshold': strategy_adaptive_z,
    '6_momentum_burst': strategy_momentum_burst,
    '7_anti_correlation_exit': strategy_anti_correlation_exit,
}


def main():
    log.info("=" * 80)
    log.info("ADVANCED EXECUTION STRATEGIES v3")
    log.info(f"Models: Mamba v7 + CNN-Mamba v2")
    log.info(f"Folds: {sorted(FOLD_DATES.keys())} ({len(FOLD_DATES)} days)")
    log.info(f"Strategies: {len(STRATEGY_RUNNERS)}")
    log.info(f"Cost model: ES ${COMMISSION_RT} RT = {COMMISSION_TICKS:.3f} ticks + {SPREAD_COST_TICKS} spread")
    log.info(f"Tick value: ${TICK_VALUE}")
    log.info("=" * 80)

    # Load all fold data
    all_data = {}
    for fold_idx in sorted(FOLD_DATES.keys()):
        data = load_fold_predictions(fold_idx)
        if data is not None:
            all_data[fold_idx] = data
        gc.collect()

    if not all_data:
        log.error("No data loaded! Check prediction paths.")
        return

    n_days = len(all_data)
    log.info(f"\nLoaded {n_days} days of data")

    # Run each strategy
    all_results = []

    for strat_name, strat_func in STRATEGY_RUNNERS.items():
        log.info(f"\n{'=' * 60}")
        log.info(f"Strategy: {strat_name}")
        log.info(f"{'=' * 60}")

        strat_trades = []
        daily_pnls = {}

        for fold_idx, data in sorted(all_data.items()):
            date_str = data['date']

            # Run strategy
            trades = strat_func(data)

            # Simulate P&L
            trades = simulate_pnl(trades, data)

            day_pnl = sum(t.get('pnl_dollars', 0) for t in trades)
            daily_pnls[date_str] = round(day_pnl, 2)
            strat_trades.extend(trades)

            n_trades = len(trades)
            log.info(f"  {date_str}: {n_trades} trades, P&L=${day_pnl:+.2f}")

        # Analyze
        perf = analyze_performance(strat_trades, n_days)
        tiers = tier_analysis(strat_trades)

        result = {
            'strategy': strat_name,
            'performance': perf,
            'daily_pnls': daily_pnls,
            'tier_analysis': tiers,
            'profitable_days': sum(1 for v in daily_pnls.values() if v > 0),
            'total_days': n_days,
        }
        all_results.append(result)

        log.info(f"  TOTAL: {perf['n_trades']} trades, P&L=${perf['total_pnl']:+.2f}, "
                 f"WR={perf['win_rate']:.1%}, Sortino={perf['sortino']:.3f}, "
                 f"AvgHold={perf['avg_hold_seconds']:.0f}s, "
                 f"Trades/day={perf['trades_per_day']:.0f}, "
                 f"AvgSize={perf['avg_size']:.1f}")

    # ── Sort by total P&L ──
    all_results.sort(key=lambda r: r['performance']['total_pnl'], reverse=True)

    # ── Final ranking ──
    log.info("\n" + "=" * 100)
    log.info("STRATEGY RANKING (by total P&L)")
    log.info("=" * 100)
    header = (f"{'#':<3} {'Strategy':<40} {'Trades':>6} {'P&L':>10} {'WR':>6} "
              f"{'Sortino':>8} {'AvgHold':>8} {'Tr/Day':>7} {'PF':>7} {'MaxDD':>8} {'AvgSz':>6}")
    log.info(header)
    log.info("-" * 100)

    baseline_pnl = None
    for i, r in enumerate(all_results):
        p = r['performance']
        if 'baseline' in r['strategy']:
            baseline_pnl = p['total_pnl']

        log.info(f"{i+1:<3} {r['strategy']:<40} {p['n_trades']:>6} ${p['total_pnl']:>+9.2f} "
                 f"{p['win_rate']:>5.1%} {p['sortino']:>8.3f} {p['avg_hold_seconds']:>6.0f}s "
                 f"{p['trades_per_day']:>6.0f} {p['profit_factor']:>7.3f} "
                 f"${p['max_drawdown']:>7.2f} {p['avg_size']:>5.1f}")

    # ── Delta vs baseline ──
    if baseline_pnl is not None:
        log.info(f"\n{'Delta vs baseline':>43}")
        log.info("-" * 60)
        for r in all_results:
            p = r['performance']
            delta = p['total_pnl'] - baseline_pnl
            marker = " <-- BASELINE" if 'baseline' in r['strategy'] else ""
            log.info(f"  {r['strategy']:<40} ${delta:>+9.2f}{marker}")

    # ── Daily P&L breakdown ──
    log.info(f"\n{'Daily P&L by strategy':>43}")
    log.info("-" * 80)
    dates = sorted(set(d for r in all_results for d in r['daily_pnls'].keys()))
    header = f"{'Strategy':<40} " + " ".join(f"{d[-4:]:>10}" for d in dates) + f" {'Total':>10}"
    log.info(header)
    for r in all_results:
        vals = " ".join(f"${r['daily_pnls'].get(d, 0):>+9.2f}" for d in dates)
        log.info(f"{r['strategy']:<40} {vals} ${r['performance']['total_pnl']:>+9.2f}")

    # ── Confidence tier summary for top 3 ──
    log.info(f"\n{'Confidence Tier Analysis (Top 3 strategies)':>50}")
    log.info("-" * 80)
    for r in all_results[:3]:
        log.info(f"\n  {r['strategy']}:")
        tiers = r.get('tier_analysis', {})
        for tier_name, tv in tiers.items():
            log.info(f"    {tier_name:<10}: {tv['n_trades']:>4} trades, "
                     f"P&L=${tv['total_pnl']:>+8.2f}, "
                     f"WR={tv['win_rate']:.1%}, Sortino={tv['sortino']:.3f}")

    # ── Save results ──
    out_path = RESULTS_DIR / f'advanced_v3_results_{_ts}.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info(f"\nResults saved to: {out_path}")

    # ── Key findings ──
    log.info("\n" + "=" * 80)
    log.info("KEY FINDINGS")
    log.info("=" * 80)
    best = all_results[0]
    bp = best['performance']
    log.info(f"Best strategy: {best['strategy']}")
    log.info(f"  P&L: ${bp['total_pnl']:+.2f}, {bp['n_trades']} trades, "
             f"WR: {bp['win_rate']:.1%}, Sortino: {bp['sortino']:.3f}")

    # Check which strategies beat baseline
    if baseline_pnl is not None:
        beaten = [r for r in all_results if r['performance']['total_pnl'] > baseline_pnl
                  and 'baseline' not in r['strategy']]
        if beaten:
            log.info(f"\n{len(beaten)} strategies BEAT the baseline (+${baseline_pnl:.2f}):")
            for r in beaten:
                delta = r['performance']['total_pnl'] - baseline_pnl
                log.info(f"  {r['strategy']}: +${delta:.2f} improvement")
        else:
            log.info(f"\nNo strategies beat the baseline. Current champion holds.")

    log.info("\nDone!")
    return all_results


if __name__ == '__main__':
    main()
