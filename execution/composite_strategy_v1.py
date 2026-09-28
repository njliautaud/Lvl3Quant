#!/usr/bin/env python3
"""
Composite Execution Strategy v1 — April 2026
=============================================
Combines the 3 best-performing elements from advanced_execution_v3:

  1. Confidence-weighted sizing: 3 lots at Top0.1%, 2 at Top1%, 1 otherwise
     (+$2,878 vs $1,011 baseline — best v3 strategy)
  2. Cascade entry: CNN-Mamba v2 confirms Mamba v7 direction within 5 seconds
     (filtered ~5 bad trades/day)
  3. Anti-correlation exit: exit when Mamba and CNN-Mamba DISAGREE
     (68% WR, highest Sortino, avg hold 242s)

Plus NEW:
  4. PatchTST confluence gate: PatchTST smart_v3 Top10% must agree with
     Mamba direction. PatchTST at Top10% has 71.4% DA at 1s.

Full composite pipeline:
  Mamba z>=3.0 → CNN-Mamba confirms within 5s → PatchTST Top10% agrees →
  confidence-weighted sizing → anti-correlation exit (slowflip fallback)

Variants tested:
  A. Full composite (all 4 elements)
  B. Without PatchTST gate (cascade + sizing + anti-corr)
  C. Without anti-corr exit (full entry + slowflip exit)
  D. Without sizing (full entry/exit + flat 1-lot)
  E. Baseline: Mamba z3.0 slowflip 1-lot (reference)
  F. Cascade + sizing only (no PatchTST, slowflip exit)
  G. Full composite with relaxed PatchTST (Top20% instead of Top10%)

Data alignment:
  Mamba v7 and CNN-Mamba v2 share sample indices (same stride=500).
  PatchTST uses stride=250 → 2x samples. Alignment: PatchTST[2 + 2*i] = Mamba[i].

Overlapping dates (all 3 models):
  20260302 (Mamba fold_06, CNN fold_06, PatchTST fold_10)
  20260303 (Mamba fold_07, CNN fold_07, PatchTST fold_11)
  20260304 (Mamba fold_08, CNN fold_08, PatchTST fold_12)

Additional dates (Mamba + CNN-Mamba only, no PatchTST):
  20260301 (fold_05), 20260305 (fold_09)

Cost model: ES $5/tick, $4.70 RT commission = 0.94 ticks, 0.5 tick spread
"""

import sys
import gc
import json
import logging
import os
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import numpy as np

# ── Paths ──
LVL3_ROOT = Path(__file__).resolve().parent.parent
MAMBA_PRED_DIR = LVL3_ROOT / 'output' / 'mamba_v7_tiny_smart_v3_mar_apr'
CNN_MAMBA_PRED_DIR = LVL3_ROOT / 'output' / 'cnn_mamba_v2_smart_v3_mar'
PATCHTST_PRED_DIR = LVL3_ROOT / 'output' / 'patchtst_smart_v3_mar'
RESULTS_DIR = LVL3_ROOT / 'execution' / 'results' / 'composite_v1'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── NQ Futures Constants ──
TICK_SIZE = 0.25
TICK_VALUE = 5.00
POINT_VALUE = 20.00
COMMISSION_RT = 4.70
COMMISSION_TICKS = COMMISSION_RT / TICK_VALUE  # 0.94
SPREAD_COST_TICKS = 0.5
TOTAL_COST_TICKS = COMMISSION_TICKS + SPREAD_COST_TICKS  # 1.44

# ── Model Constants ──
BARS_PER_SEC = 10
PATCHTST_OFFSET = 2       # PatchTST[2 + 2*i] aligns with Mamba/CNN[i]
PATCHTST_STRIDE_RATIO = 2  # PatchTST has 2x samples

# ── Fold-to-date mapping ──
# All 3 models overlap on these dates
TRIPLE_OVERLAP_FOLDS = {
    # date: (mamba_fold, cnn_fold, patchtst_fold)
    '20260302': (6, 6, 10),
    '20260303': (7, 7, 11),
    '20260304': (8, 8, 12),
}

# Mamba + CNN-Mamba only (for "without PatchTST" variants — extra days)
DUAL_ONLY_FOLDS = {
    '20260301': (5, 5),
    '20260305': (9, 9),
}

# ── Logging ──
_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
log = logging.getLogger('composite_v1')
log.setLevel(logging.INFO)
_fh = logging.FileHandler(str(RESULTS_DIR / f'composite_v1_{_ts}.log'), mode='w')
_fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
log.addHandler(_fh)
_ch = logging.StreamHandler(sys.stdout)
_ch.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
log.addHandler(_ch)


# ============================================================
# Data loading
# ============================================================
def load_triple_fold(date_str: str) -> Optional[dict]:
    """Load all 3 models for an overlapping date."""
    mamba_fold, cnn_fold, ptst_fold = TRIPLE_OVERLAP_FOLDS[date_str]

    mamba_path = MAMBA_PRED_DIR / f'fold_{mamba_fold:02d}_oot_predictions.npz'
    cnn_path = CNN_MAMBA_PRED_DIR / f'fold_{cnn_fold:02d}_oot_predictions.npz'
    ptst_path = PATCHTST_PRED_DIR / f'fold_{ptst_fold:02d}_oot_predictions.npz'

    for p, name in [(mamba_path, 'Mamba'), (cnn_path, 'CNN-Mamba'), (ptst_path, 'PatchTST')]:
        if not p.exists():
            log.warning(f"{name} predictions not found: {p}")
            return None

    try:
        mamba_data = np.load(mamba_path, allow_pickle=True)
        cnn_data = np.load(cnn_path, allow_pickle=True)
        ptst_data = np.load(ptst_path, allow_pickle=True)

        mamba_preds = mamba_data['predictions']   # (N, 3): 1s, 5s, 10s
        cnn_preds = cnn_data['predictions']       # (N, 3)
        ptst_preds_raw = ptst_data['predictions'] # (2N+2, 3)
        labels = mamba_data['labels']             # (N, 3)

        N = mamba_preds.shape[0]
        if cnn_preds.shape[0] != N:
            log.warning(f"{date_str}: Mamba/CNN shape mismatch {N} vs {cnn_preds.shape[0]}")
            return None

        # Align PatchTST: ptst[2 + 2*i] -> mamba[i]
        ptst_indices = PATCHTST_OFFSET + np.arange(N) * PATCHTST_STRIDE_RATIO
        valid = ptst_indices < ptst_preds_raw.shape[0]
        if valid.sum() < N * 0.95:
            log.warning(f"{date_str}: PatchTST alignment covers only {valid.sum()}/{N} samples")

        # Trim to valid range
        n_valid = int(valid.sum())
        ptst_preds = np.zeros((N, 3), dtype=np.float32)
        ptst_preds[:n_valid] = ptst_preds_raw[ptst_indices[:n_valid]]

        if N < 100:
            log.warning(f"{date_str}: only {N} samples, skipping")
            return None

        log.info(f"{date_str}: {N} samples (triple: Mamba+CNN+PatchTST)")
        return {
            'date': date_str,
            'mamba_preds': mamba_preds,
            'cnn_preds': cnn_preds,
            'ptst_preds': ptst_preds,
            'labels': labels,
            'n_samples': N,
            'has_patchtst': True,
        }
    except Exception as e:
        log.error(f"Failed to load {date_str}: {e}")
        return None


def load_dual_fold(date_str: str) -> Optional[dict]:
    """Load Mamba + CNN-Mamba only (no PatchTST)."""
    mamba_fold, cnn_fold = DUAL_ONLY_FOLDS[date_str]

    mamba_path = MAMBA_PRED_DIR / f'fold_{mamba_fold:02d}_oot_predictions.npz'
    cnn_path = CNN_MAMBA_PRED_DIR / f'fold_{cnn_fold:02d}_oot_predictions.npz'

    for p, name in [(mamba_path, 'Mamba'), (cnn_path, 'CNN-Mamba')]:
        if not p.exists():
            log.warning(f"{name} predictions not found: {p}")
            return None

    try:
        mamba_data = np.load(mamba_path, allow_pickle=True)
        cnn_data = np.load(cnn_path, allow_pickle=True)

        mamba_preds = mamba_data['predictions']
        cnn_preds = cnn_data['predictions']
        labels = mamba_data['labels']

        N = mamba_preds.shape[0]
        if cnn_preds.shape[0] != N:
            log.warning(f"{date_str}: shape mismatch {N} vs {cnn_preds.shape[0]}")
            return None

        if N < 100:
            log.warning(f"{date_str}: only {N} samples, skipping")
            return None

        log.info(f"{date_str}: {N} samples (dual: Mamba+CNN only)")
        return {
            'date': date_str,
            'mamba_preds': mamba_preds,
            'cnn_preds': cnn_preds,
            'ptst_preds': None,
            'labels': labels,
            'n_samples': N,
            'has_patchtst': False,
        }
    except Exception as e:
        log.error(f"Failed to load {date_str}: {e}")
        return None


# ============================================================
# Z-score computation (expanding, causal)
# ============================================================
def compute_z_scores(preds: np.ndarray) -> np.ndarray:
    """Expanding z-score normalization (causal — no lookahead)."""
    N, H = preds.shape
    z = np.zeros_like(preds, dtype=np.float64)
    for h in range(H):
        col = preds[:, h].astype(np.float64)
        cs = np.cumsum(col)
        css = np.cumsum(col ** 2)
        ns = np.arange(1, N + 1, dtype=np.float64)
        means = cs / ns
        vars_ = np.maximum(css / ns - means ** 2, 1e-12)
        stds = np.sqrt(vars_)
        mask = stds > 1e-8
        z[mask, h] = (col[mask] - means[mask]) / stds[mask]
    return z.astype(np.float32)


def compute_expanding_percentile_rank(values: np.ndarray) -> np.ndarray:
    """For each index i, compute the percentile rank of values[i] among values[:i+1].
    Returns array of percentiles (0-100). Causal."""
    N = len(values)
    pct = np.zeros(N, dtype=np.float32)
    for i in range(1, N):
        pct[i] = (values[:i] < values[i]).sum() / i * 100.0
    return pct


# ============================================================
# Composite strategy engine
# ============================================================
@dataclass
class StrategyConfig:
    """Configuration for composite strategy variants."""
    name: str
    # Entry
    mamba_z_thresh: float = 3.0
    use_cascade: bool = True
    cascade_window: int = 50       # 5 seconds at 10 events/s
    cascade_cnn_z_thresh: float = 2.0
    use_patchtst_gate: bool = True
    patchtst_pct_thresh: float = 90.0  # Top10% = 90th percentile
    # Sizing
    use_confidence_sizing: bool = True
    # Exit
    use_anti_corr_exit: bool = True
    anti_corr_min_hold: int = 50
    anti_corr_min_z: float = 0.3     # lowered from 1.0 — models rarely disagree at z>1
    anti_corr_horizon: str = 'mixed' # 'mixed' = Mamba 10s vs CNN 1s, '10s' = both 10s
    # Fallback exit (slowflip)
    slowflip_min_hold: int = 100
    slowflip_lookback: int = 3
    max_hold: int = 6000


def run_composite_strategy(data: dict, cfg: StrategyConfig) -> List[dict]:
    """Run composite strategy with given configuration."""
    mamba_z = compute_z_scores(data['mamba_preds'])
    cnn_z = compute_z_scores(data['cnn_preds'])
    N = data['n_samples']

    # PatchTST z-scores and percentile rank (if available and enabled)
    ptst_z = None
    ptst_abs_pct = None
    if cfg.use_patchtst_gate and data.get('has_patchtst') and data['ptst_preds'] is not None:
        ptst_z = compute_z_scores(data['ptst_preds'])
        # Percentile rank of absolute 1s z-score (for top-N% filtering)
        ptst_abs_1s = np.abs(ptst_z[:, 0])
        ptst_abs_pct = compute_expanding_percentile_rank(ptst_abs_1s)

    # Mamba absolute z percentile for confidence sizing
    mamba_abs_10s = np.abs(mamba_z[:, 2])
    mamba_abs_pct = compute_expanding_percentile_rank(mamba_abs_10s)

    trades = []
    position = 0
    entry_idx = 0
    entry_signal = 0.0
    entry_size = 1
    pending_dir = 0
    pending_idx = -1
    pending_signal = 0.0

    for i in range(N):
        if position == 0:
            # ── STAGE 1: Mamba trigger ──
            if abs(mamba_z[i, 2]) >= cfg.mamba_z_thresh:
                trigger_dir = int(np.sign(mamba_z[i, 2]))
                trigger_signal = float(mamba_z[i, 2])

                if cfg.use_cascade:
                    # Set pending, wait for CNN-Mamba confirmation
                    pending_dir = trigger_dir
                    pending_idx = i
                    pending_signal = trigger_signal
                else:
                    # No cascade — direct entry (check PatchTST gate first)
                    enter = True
                    if cfg.use_patchtst_gate and ptst_z is not None and i > 100:
                        ptst_dir = np.sign(ptst_z[i, 0])
                        ptst_strong = ptst_abs_pct[i] >= cfg.patchtst_pct_thresh
                        if not (ptst_dir == trigger_dir and ptst_strong):
                            enter = False

                    if enter:
                        position = trigger_dir
                        entry_idx = i
                        entry_signal = trigger_signal
                        # Confidence sizing
                        if cfg.use_confidence_sizing and i > 100:
                            pct = mamba_abs_pct[i]
                            if pct >= 99.9:
                                entry_size = 3
                            elif pct >= 99.0:
                                entry_size = 2
                            else:
                                entry_size = 1
                        else:
                            entry_size = 1

            # ── STAGE 2: CNN-Mamba confirmation (cascade mode) ──
            if cfg.use_cascade and pending_dir != 0:
                elapsed = i - pending_idx
                if elapsed <= cfg.cascade_window:
                    cnn_dir = np.sign(cnn_z[i, 2])
                    if cnn_dir == pending_dir and abs(cnn_z[i, 2]) >= cfg.cascade_cnn_z_thresh:
                        # CNN-Mamba confirms! Now check PatchTST gate
                        enter = True
                        if cfg.use_patchtst_gate and ptst_z is not None and i > 100:
                            ptst_dir = np.sign(ptst_z[i, 0])
                            ptst_strong = ptst_abs_pct[i] >= cfg.patchtst_pct_thresh
                            if not (ptst_dir == pending_dir and ptst_strong):
                                enter = False

                        if enter:
                            position = pending_dir
                            entry_idx = i
                            entry_signal = pending_signal
                            # Confidence sizing based on original Mamba signal
                            if cfg.use_confidence_sizing and pending_idx > 100:
                                pct = mamba_abs_pct[pending_idx]
                                if pct >= 99.9:
                                    entry_size = 3
                                elif pct >= 99.0:
                                    entry_size = 2
                                else:
                                    entry_size = 1
                            else:
                                entry_size = 1
                            pending_dir = 0
                elif elapsed > cfg.cascade_window:
                    pending_dir = 0  # expired

        else:
            # ── EXIT LOGIC ──
            held = i - entry_idx
            should_exit = False
            exit_reason = ''

            # Anti-correlation exit: models disagree on direction
            if cfg.use_anti_corr_exit and held >= cfg.anti_corr_min_hold:
                # Position direction tracked on Mamba 10s
                mamba_dir = np.sign(mamba_z[i, 2])
                # CNN exit signal: use 1s for faster reaction ('mixed') or 10s
                if cfg.anti_corr_horizon == 'mixed':
                    cnn_dir = np.sign(cnn_z[i, 0])   # 1s horizon — faster
                    cnn_strength = abs(cnn_z[i, 0])
                else:
                    cnn_dir = np.sign(cnn_z[i, 2])   # 10s horizon
                    cnn_strength = abs(cnn_z[i, 2])
                mamba_strength = abs(mamba_z[i, 2])

                if (mamba_dir != 0 and cnn_dir != 0 and mamba_dir != cnn_dir
                        and mamba_strength > cfg.anti_corr_min_z
                        and cnn_strength > cfg.anti_corr_min_z):
                    should_exit = True
                    exit_reason = 'anti_correlation'

            # Slowflip fallback exit
            if not should_exit and held >= cfg.slowflip_min_hold and i >= cfg.slowflip_lookback:
                dirs = [np.sign(mamba_z[i - j, 2]) for j in range(cfg.slowflip_lookback)]
                if all(d == -position for d in dirs):
                    should_exit = True
                    exit_reason = 'slow_flip'

            # Max hold
            if not should_exit and held >= cfg.max_hold:
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

    # Close any open position at EOD
    if position != 0:
        trades.append({
            'entry_idx': entry_idx, 'exit_idx': N - 1,
            'direction': position, 'hold_events': N - 1 - entry_idx,
            'entry_signal': entry_signal, 'exit_reason': 'eod',
            'size': entry_size,
        })

    return trades


# ============================================================
# P&L simulation
# ============================================================
def simulate_pnl(trades: List[dict], data: dict) -> List[dict]:
    """Estimate P&L from forward labels, accounting for position size."""
    labels = data['labels']

    for trade in trades:
        entry_idx = trade['entry_idx']
        direction = trade['direction']
        hold_events = trade['hold_events']
        size = trade.get('size', 1)

        if entry_idx < len(labels):
            # Select label horizon based on actual hold time
            if hold_events <= 15:
                pnl_ticks = direction * labels[entry_idx, 0]   # 1s
            elif hold_events <= 60:
                pnl_ticks = direction * labels[entry_idx, 1]   # 5s
            else:
                pnl_ticks = direction * labels[entry_idx, 2]   # 10s
        else:
            pnl_ticks = 0

        net_pnl_ticks = pnl_ticks - TOTAL_COST_TICKS
        trade['pnl_ticks'] = float(pnl_ticks)
        trade['net_pnl_ticks'] = float(net_pnl_ticks)
        trade['pnl_dollars'] = float(net_pnl_ticks * TICK_VALUE * size)
        trade['cost_ticks'] = float(TOTAL_COST_TICKS)
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
    """Comprehensive performance metrics."""
    if not trades:
        return {
            'n_trades': 0, 'total_pnl': 0, 'avg_pnl': 0, 'win_rate': 0,
            'sortino': 0, 'profit_factor': 0, 'avg_hold_seconds': 0,
            'trades_per_day': 0, 'max_drawdown': 0, 'avg_size': 1,
            'gross_profit': 0, 'gross_loss': 0, 'long_trades': 0,
            'long_pnl': 0, 'short_trades': 0, 'short_pnl': 0,
            'exit_reasons': {},
        }

    pnls = np.array([t.get('pnl_dollars', 0) for t in trades])
    holds = np.array([t['hold_events'] for t in trades])
    sizes = np.array([t.get('size', 1) for t in trades])

    n_trades = len(trades)
    total_pnl = float(np.sum(pnls))
    wins = (pnls > 0).sum()
    win_rate = float(wins / n_trades)

    # Sortino
    neg_returns = pnls[pnls < 0]
    downside_std = np.std(neg_returns) if len(neg_returns) > 1 else 1e-8
    sortino = float(np.mean(pnls) / downside_std) if downside_std > 1e-8 else 0.0

    # Profit factor
    gross_profit = float(np.sum(pnls[pnls > 0]))
    gross_loss = float(abs(np.sum(pnls[pnls < 0])))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Drawdown
    cum_pnl = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum_pnl)
    max_drawdown = float(np.max(peak - cum_pnl)) if len(cum_pnl) > 0 else 0

    # Exit reasons
    exit_reasons = {}
    for t in trades:
        r = t.get('exit_reason', 'unknown')
        exit_reasons[r] = exit_reasons.get(r, 0) + 1

    # Long/short
    longs = [t for t in trades if t['direction'] == 1]
    shorts = [t for t in trades if t['direction'] == -1]

    # Size distribution
    size_dist = {}
    for s in [1, 2, 3]:
        s_trades = [t for t in trades if t.get('size', 1) == s]
        if s_trades:
            s_pnls = [t['pnl_dollars'] for t in s_trades]
            size_dist[f'{s}_lot'] = {
                'count': len(s_trades),
                'total_pnl': round(sum(s_pnls), 2),
                'wr': round(sum(1 for p in s_pnls if p > 0) / len(s_pnls), 4),
            }

    return {
        'n_trades': n_trades,
        'total_pnl': round(total_pnl, 2),
        'avg_pnl': round(float(np.mean(pnls)), 2),
        'win_rate': round(win_rate, 4),
        'sortino': round(sortino, 3),
        'profit_factor': round(profit_factor, 3),
        'avg_hold_seconds': round(float(np.mean(holds)) / BARS_PER_SEC, 1),
        'trades_per_day': round(n_trades / max(n_days, 1), 1),
        'max_drawdown': round(max_drawdown, 2),
        'avg_size': round(float(np.mean(sizes)), 2),
        'gross_profit': round(gross_profit, 2),
        'gross_loss': round(gross_loss, 2),
        'long_trades': len(longs),
        'long_pnl': round(sum(t['pnl_dollars'] for t in longs), 2),
        'short_trades': len(shorts),
        'short_pnl': round(sum(t['pnl_dollars'] for t in shorts), 2),
        'exit_reasons': exit_reasons,
        'size_distribution': size_dist,
    }


# ============================================================
# Strategy variant definitions
# ============================================================
def get_strategy_variants() -> List[StrategyConfig]:
    return [
        # A: Full composite — all 4 elements
        StrategyConfig(
            name='A_full_composite',
            use_cascade=True, use_patchtst_gate=True,
            use_confidence_sizing=True, use_anti_corr_exit=True,
        ),

        # B: No PatchTST gate — cascade + sizing + anti-corr
        StrategyConfig(
            name='B_no_patchtst',
            use_cascade=True, use_patchtst_gate=False,
            use_confidence_sizing=True, use_anti_corr_exit=True,
        ),

        # C: No anti-corr exit — full entry pipeline + slowflip exit
        StrategyConfig(
            name='C_no_anticorr_exit',
            use_cascade=True, use_patchtst_gate=True,
            use_confidence_sizing=True, use_anti_corr_exit=False,
        ),

        # D: No sizing — full entry/exit + flat 1-lot
        StrategyConfig(
            name='D_no_sizing',
            use_cascade=True, use_patchtst_gate=True,
            use_confidence_sizing=False, use_anti_corr_exit=True,
        ),

        # E: Baseline — Mamba z3.0 slowflip 1-lot (reference)
        StrategyConfig(
            name='E_baseline_mamba_z3_slowflip',
            use_cascade=False, use_patchtst_gate=False,
            use_confidence_sizing=False, use_anti_corr_exit=False,
        ),

        # F: Cascade + sizing (no PatchTST, slowflip exit)
        StrategyConfig(
            name='F_cascade_sizing_slowflip',
            use_cascade=True, use_patchtst_gate=False,
            use_confidence_sizing=True, use_anti_corr_exit=False,
        ),

        # G: Full composite with relaxed PatchTST (Top20%)
        StrategyConfig(
            name='G_full_composite_ptst_top20',
            use_cascade=True, use_patchtst_gate=True,
            patchtst_pct_thresh=80.0,
            use_confidence_sizing=True, use_anti_corr_exit=True,
        ),

        # H: Anti-corr exit only (no cascade, no PatchTST, sizing on) — mixed horizon
        StrategyConfig(
            name='H_anticorr_sizing_mixed',
            use_cascade=False, use_patchtst_gate=False,
            use_confidence_sizing=True, use_anti_corr_exit=True,
            anti_corr_horizon='mixed', anti_corr_min_z=0.3,
        ),

        # I: Anti-corr with 10s horizon (both models 10s)
        StrategyConfig(
            name='I_anticorr_sizing_10s',
            use_cascade=False, use_patchtst_gate=False,
            use_confidence_sizing=True, use_anti_corr_exit=True,
            anti_corr_horizon='10s', anti_corr_min_z=0.3,
        ),

        # J: Full composite with mixed-horizon anti-corr (best combo candidate)
        StrategyConfig(
            name='J_cascade_ptst_sizing_anticorr_mixed',
            use_cascade=True, use_patchtst_gate=True,
            use_confidence_sizing=True, use_anti_corr_exit=True,
            anti_corr_horizon='mixed', anti_corr_min_z=0.3,
        ),

        # K: Cascade + sizing + mixed anti-corr (no PatchTST) — full 5-day test
        StrategyConfig(
            name='K_cascade_sizing_anticorr_mixed',
            use_cascade=True, use_patchtst_gate=False,
            use_confidence_sizing=True, use_anti_corr_exit=True,
            anti_corr_horizon='mixed', anti_corr_min_z=0.3,
        ),
    ]


# ============================================================
# Main
# ============================================================
def main():
    log.info("=" * 90)
    log.info("COMPOSITE EXECUTION STRATEGY v1")
    log.info("Combining: Confidence sizing + Cascade entry + Anti-corr exit + PatchTST gate")
    log.info(f"Cost model: ES ${COMMISSION_RT} RT = {COMMISSION_TICKS:.3f} ticks + {SPREAD_COST_TICKS} spread = {TOTAL_COST_TICKS:.3f} total")
    log.info(f"Tick value: ${TICK_VALUE}")
    log.info("=" * 90)

    # ── Load data ──
    all_data = {}

    # Triple-overlap dates (all 3 models)
    for date_str in sorted(TRIPLE_OVERLAP_FOLDS.keys()):
        data = load_triple_fold(date_str)
        if data is not None:
            all_data[date_str] = data
        gc.collect()

    # Dual-only dates (Mamba + CNN-Mamba)
    for date_str in sorted(DUAL_ONLY_FOLDS.keys()):
        data = load_dual_fold(date_str)
        if data is not None:
            all_data[date_str] = data
        gc.collect()

    if not all_data:
        log.error("No data loaded! Check prediction paths.")
        return

    n_days_total = len(all_data)
    n_days_triple = sum(1 for d in all_data.values() if d['has_patchtst'])
    log.info(f"\nLoaded {n_days_total} days total ({n_days_triple} with PatchTST)")

    # ── Run all strategy variants ──
    variants = get_strategy_variants()
    all_results = []

    for cfg in variants:
        log.info(f"\n{'=' * 70}")
        log.info(f"Strategy: {cfg.name}")
        log.info(f"  cascade={cfg.use_cascade}, patchtst={cfg.use_patchtst_gate}"
                 f"{'(Top' + str(100-cfg.patchtst_pct_thresh) + '%)' if cfg.use_patchtst_gate else ''}"
                 f", sizing={cfg.use_confidence_sizing}, anti_corr={cfg.use_anti_corr_exit}")
        log.info(f"{'=' * 70}")

        strat_trades = []
        daily_pnls = {}

        for date_str, data in sorted(all_data.items()):
            # If strategy needs PatchTST but this date doesn't have it:
            # skip this date for PatchTST-gated strategies, include for non-PatchTST
            if cfg.use_patchtst_gate and not data['has_patchtst']:
                log.info(f"  {date_str}: SKIPPED (no PatchTST)")
                continue

            trades = run_composite_strategy(data, cfg)
            trades = simulate_pnl(trades, data)

            day_pnl = sum(t.get('pnl_dollars', 0) for t in trades)
            daily_pnls[date_str] = round(day_pnl, 2)
            strat_trades.extend(trades)

            n_t = len(trades)
            sizes = [t.get('size', 1) for t in trades]
            avg_sz = np.mean(sizes) if sizes else 1
            log.info(f"  {date_str}: {n_t} trades, P&L=${day_pnl:+.2f}, avg_size={avg_sz:.1f}")

        n_days_used = len(daily_pnls)
        perf = analyze_performance(strat_trades, n_days_used)

        result = {
            'strategy': cfg.name,
            'config': {
                'cascade': cfg.use_cascade,
                'patchtst_gate': cfg.use_patchtst_gate,
                'patchtst_pct_thresh': cfg.patchtst_pct_thresh,
                'confidence_sizing': cfg.use_confidence_sizing,
                'anti_corr_exit': cfg.use_anti_corr_exit,
            },
            'performance': perf,
            'daily_pnls': daily_pnls,
            'n_days_used': n_days_used,
            'profitable_days': sum(1 for v in daily_pnls.values() if v > 0),
        }
        all_results.append(result)

        log.info(f"  TOTAL: {perf['n_trades']} trades, P&L=${perf['total_pnl']:+.2f}, "
                 f"WR={perf['win_rate']:.1%}, Sortino={perf['sortino']:.3f}, "
                 f"PF={perf['profit_factor']:.2f}, AvgHold={perf['avg_hold_seconds']:.0f}s, "
                 f"Tr/day={perf['trades_per_day']:.0f}, AvgSz={perf['avg_size']:.1f}")

    # ── Sort by total P&L ──
    all_results.sort(key=lambda r: r['performance']['total_pnl'], reverse=True)

    # ── RESULTS TABLE ──
    log.info("\n" + "=" * 120)
    log.info("COMPOSITE STRATEGY RANKING (by total P&L)")
    log.info("=" * 120)
    header = (f"{'#':<3} {'Strategy':<35} {'Days':>4} {'Trades':>6} {'P&L':>10} "
              f"{'WR':>6} {'Sortino':>8} {'PF':>7} {'AvgHold':>8} {'Tr/Day':>7} "
              f"{'MaxDD':>8} {'AvgSz':>6}")
    log.info(header)
    log.info("-" * 120)

    baseline_pnl = None
    for i, r in enumerate(all_results):
        p = r['performance']
        if 'baseline' in r['strategy']:
            baseline_pnl = p['total_pnl']
        pf_str = f"{p['profit_factor']:.2f}" if p['profit_factor'] < 100 else "inf"
        log.info(f"{i+1:<3} {r['strategy']:<35} {r['n_days_used']:>4} {p['n_trades']:>6} "
                 f"${p['total_pnl']:>+9.2f} {p['win_rate']:>5.1%} {p['sortino']:>8.3f} "
                 f"{pf_str:>7} {p['avg_hold_seconds']:>6.0f}s {p['trades_per_day']:>6.0f} "
                 f"${p['max_drawdown']:>7.2f} {p['avg_size']:>5.1f}")

    # ── Delta vs baseline ──
    if baseline_pnl is not None:
        log.info(f"\n{'Delta vs baseline ($' + str(round(baseline_pnl, 2)) + ')':>50}")
        log.info("-" * 70)
        for r in all_results:
            p = r['performance']
            delta = p['total_pnl'] - baseline_pnl
            marker = " <-- BASELINE" if 'baseline' in r['strategy'] else ""
            log.info(f"  {r['strategy']:<35} ${delta:>+9.2f}  "
                     f"(WR: {p['win_rate']:.1%}, Sortino: {p['sortino']:.3f}){marker}")

    # ── Daily P&L breakdown ──
    log.info(f"\n{'Daily P&L Breakdown':>40}")
    log.info("-" * 100)
    all_dates = sorted(set(d for r in all_results for d in r['daily_pnls'].keys()))
    header = f"{'Strategy':<35} " + " ".join(f"{d[-4:]:>10}" for d in all_dates) + f" {'Total':>10}"
    log.info(header)
    for r in all_results:
        vals = " ".join(
            f"${r['daily_pnls'].get(d, 0):>+9.2f}" if d in r['daily_pnls'] else f"{'---':>10}"
            for d in all_dates
        )
        log.info(f"{r['strategy']:<35} {vals} ${r['performance']['total_pnl']:>+9.2f}")

    # ── Exit reason analysis for top strategies ──
    log.info(f"\nExit Reason Breakdown (top 3)")
    log.info("-" * 70)
    for r in all_results[:3]:
        log.info(f"  {r['strategy']}:")
        exits = r['performance'].get('exit_reasons', {})
        total = sum(exits.values()) or 1
        for reason, count in sorted(exits.items(), key=lambda x: -x[1]):
            log.info(f"    {reason:<20}: {count:>4} ({count/total*100:.1f}%)")

    # ── Size distribution for sized strategies ──
    log.info(f"\nPosition Size Distribution")
    log.info("-" * 70)
    for r in all_results:
        sd = r['performance'].get('size_distribution', {})
        if sd and any(k != '1_lot' for k in sd):
            log.info(f"  {r['strategy']}:")
            for sz_name, sv in sorted(sd.items()):
                log.info(f"    {sz_name}: {sv['count']} trades, "
                         f"P&L=${sv['total_pnl']:+.2f}, WR={sv['wr']:.1%}")

    # ── Long/Short breakdown ──
    log.info(f"\nLong vs Short Breakdown")
    log.info("-" * 80)
    log.info(f"{'Strategy':<35} {'Long#':>6} {'LongPnL':>10} {'Short#':>7} {'ShortPnL':>10}")
    for r in all_results:
        p = r['performance']
        log.info(f"{r['strategy']:<35} {p['long_trades']:>6} ${p['long_pnl']:>+9.2f} "
                 f"{p['short_trades']:>7} ${p['short_pnl']:>+9.2f}")

    # ── Per-day per-trade P&L consistency ──
    log.info(f"\nPer-Day Normalized P&L (P&L / trades)")
    log.info("-" * 80)
    for r in all_results[:4]:
        p = r['performance']
        avg_daily = p['total_pnl'] / max(r['n_days_used'], 1)
        log.info(f"  {r['strategy']:<35}: ${avg_daily:+.2f}/day, "
                 f"${p['avg_pnl']:+.2f}/trade")

    # ── Save results ──
    out_path = RESULTS_DIR / f'composite_v1_results_{_ts}.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info(f"\nResults saved to: {out_path}")

    # ── KEY FINDINGS ──
    log.info("\n" + "=" * 90)
    log.info("KEY FINDINGS")
    log.info("=" * 90)

    best = all_results[0]
    bp = best['performance']
    log.info(f"BEST STRATEGY: {best['strategy']}")
    log.info(f"  P&L: ${bp['total_pnl']:+.2f} over {best['n_days_used']} days "
             f"(${bp['total_pnl']/max(best['n_days_used'],1):+.2f}/day)")
    log.info(f"  {bp['n_trades']} trades, WR: {bp['win_rate']:.1%}, "
             f"Sortino: {bp['sortino']:.3f}, PF: {bp['profit_factor']:.2f}")
    log.info(f"  Avg hold: {bp['avg_hold_seconds']:.0f}s, Avg size: {bp['avg_size']:.1f}")

    if baseline_pnl is not None:
        improvement = bp['total_pnl'] - baseline_pnl
        pct_improvement = (improvement / abs(baseline_pnl) * 100) if baseline_pnl != 0 else float('inf')
        log.info(f"  vs Baseline: +${improvement:+.2f} ({pct_improvement:+.0f}%)")

    # Component contribution analysis
    log.info(f"\nCOMPONENT CONTRIBUTION ANALYSIS:")

    # Find matching pairs to isolate each component's contribution
    result_map = {r['strategy']: r['performance']['total_pnl'] for r in all_results}

    if 'A_full_composite' in result_map and 'B_no_patchtst' in result_map:
        delta = result_map['A_full_composite'] - result_map['B_no_patchtst']
        log.info(f"  PatchTST gate:      ${delta:+.2f} (A vs B)")

    if 'B_no_patchtst' in result_map and 'C_no_anticorr_exit' in result_map:
        # B has anti-corr but no PatchTST; C has PatchTST but no anti-corr
        # Compare both to full to isolate
        pass

    if 'A_full_composite' in result_map and 'C_no_anticorr_exit' in result_map:
        delta = result_map['A_full_composite'] - result_map['C_no_anticorr_exit']
        log.info(f"  Anti-corr exit:     ${delta:+.2f} (A vs C)")

    if 'A_full_composite' in result_map and 'D_no_sizing' in result_map:
        delta = result_map['A_full_composite'] - result_map['D_no_sizing']
        log.info(f"  Confidence sizing:  ${delta:+.2f} (A vs D)")

    if 'E_baseline_mamba_z3_slowflip' in result_map and 'F_cascade_sizing_slowflip' in result_map:
        delta = result_map['F_cascade_sizing_slowflip'] - result_map['E_baseline_mamba_z3_slowflip']
        log.info(f"  Cascade+Sizing:     ${delta:+.2f} (F vs E)")

    log.info("\nDone!")
    return all_results


if __name__ == '__main__':
    main()
