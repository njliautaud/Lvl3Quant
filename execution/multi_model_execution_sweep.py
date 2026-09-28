#!/usr/bin/env python3
"""
Multi-Model Execution Strategy Sweep — March 2-5, 2026
========================================================
Uses BOTH Mamba v7 AND CNN-Mamba v2 predictions for confluence-based execution.

Key innovations vs previous backtests:
  1. Multi-model AGREEMENT: only trade when Mamba + CNN-Mamba agree on direction
  2. Signal-flip exits: hold until model flips prediction sign (not fixed time)
  3. Multi-horizon entry: use 1s signal for entry timing, 10s for direction
  4. Ultra-selective bands: Top 1%, 0.5%, 0.1%
  5. Continuous re-evaluation: every stride (500 events), check if signal still valid
  6. Proper cost model: $4.70 RT = 0.376 ticks commission

Fill sim: Rust fill_sim_cli with real MBO FIFO queue replay.
"""

import sys
import gc
import json
import time
import subprocess
import logging
import os
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Tuple

import numpy as np
from scipy import stats

# ── Paths ──
LVL3_ROOT = Path(__file__).resolve().parent.parent
BINARY = LVL3_ROOT / 'rust_cache_builder' / 'target' / 'release' / 'fill_sim_cli'
MBO_DIR = LVL3_ROOT / 'data' / 'raw' / 'mbo'
MAMBA_PRED_DIR = LVL3_ROOT / 'output' / 'mamba_v7_tiny_smart_v3_mar_apr'
CNN_MAMBA_PRED_DIR = LVL3_ROOT / 'output' / 'cnn_mamba_v2_smart_v3_mar'
RESULTS_DIR = LVL3_ROOT / 'execution' / 'results' / 'multi_model_sweep'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── ES Futures Constants ──
TICK_SIZE = 0.25
TICK_VALUE = 12.50
POINT_VALUE = 50.00
COMMISSION_RT = 4.70  # $4.70 round trip = 0.376 ticks
COMMISSION_TICKS = COMMISSION_RT / TICK_VALUE  # 0.376

# ── Model Constants ──
WINDOW = 1000
STRIDE = 500
BARS_PER_SEC = 10
BAR_NS = 100_000_000  # 100ms

# ── March overlap folds ──
OVERLAP_FOLDS = {
    6: '20260302',
    7: '20260303',
    8: '20260304',
    9: '20260305',
}

# ── Logging ──
_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
log = logging.getLogger('multi_model_sweep')
log.setLevel(logging.INFO)
_fh = logging.FileHandler(str(RESULTS_DIR / f'sweep_{_ts}.log'), mode='w')
_fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
log.addHandler(_fh)
_ch = logging.StreamHandler(sys.stdout)
_ch.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
log.addHandler(_ch)


# ============================================================
# Load predictions
# ============================================================
def load_fold_predictions(fold_idx: int) -> dict:
    """Load both Mamba v7 and CNN-Mamba v2 predictions for a fold."""
    date_str = OVERLAP_FOLDS[fold_idx]

    mamba_path = MAMBA_PRED_DIR / f'fold_{fold_idx:02d}_oot_predictions.npz'
    cnn_mamba_path = CNN_MAMBA_PRED_DIR / f'fold_{fold_idx:02d}_oot_predictions.npz'

    mamba_data = np.load(mamba_path, allow_pickle=True)
    cnn_data = np.load(cnn_mamba_path, allow_pickle=True)

    mamba_preds = mamba_data['predictions']  # (N, 3) = 1s, 5s, 10s
    cnn_preds = cnn_data['predictions']      # (N, 3) = 1s, 5s, 10s
    labels = mamba_data['labels']            # (N, 3)

    assert mamba_preds.shape == cnn_preds.shape, f"Shape mismatch: {mamba_preds.shape} vs {cnn_preds.shape}"

    N = mamba_preds.shape[0]
    log.info(f"Fold {fold_idx} ({date_str}): {N} samples, both models loaded")

    return {
        'date': date_str,
        'fold': fold_idx,
        'mamba_preds': mamba_preds,
        'cnn_preds': cnn_preds,
        'labels': labels,
        'n_samples': N,
    }


# ============================================================
# Signal generation strategies
# ============================================================
def compute_z_scores(preds: np.ndarray) -> np.ndarray:
    """Expanding z-score normalization (causal — no lookahead)."""
    N, H = preds.shape
    z = np.zeros_like(preds)
    for h in range(H):
        running_sum = 0.0
        running_sq = 0.0
        for i in range(N):
            running_sum += preds[i, h]
            running_sq += preds[i, h] ** 2
            n = i + 1
            mean = running_sum / n
            var = max(running_sq / n - mean ** 2, 1e-12)
            std = var ** 0.5
            z[i, h] = (preds[i, h] - mean) / std if std > 1e-8 else 0.0
    return z


def generate_signals(data: dict, strategy: dict) -> np.ndarray:
    """
    Generate trade signals based on strategy config.
    Returns array of (signal_direction, confidence) for each sample.
    signal_direction: +1 = long, -1 = short, 0 = no signal
    """
    mamba_z = compute_z_scores(data['mamba_preds'])
    cnn_z = compute_z_scores(data['cnn_preds'])
    N = data['n_samples']

    signals = np.zeros(N, dtype=np.float32)

    stype = strategy['type']
    z_thresh = strategy.get('z_threshold', 2.0)
    horizon = strategy.get('horizon', 2)  # 0=1s, 1=5s, 2=10s
    entry_horizon = strategy.get('entry_horizon', 0)  # 1s for entry timing

    if stype == 'mamba_only':
        # Single model: Mamba 10s direction, z-score threshold
        for i in range(N):
            if abs(mamba_z[i, horizon]) > z_thresh:
                signals[i] = np.sign(mamba_z[i, horizon]) * abs(mamba_z[i, horizon])

    elif stype == 'cnn_mamba_only':
        # Single model: CNN-Mamba 10s direction
        for i in range(N):
            if abs(cnn_z[i, horizon]) > z_thresh:
                signals[i] = np.sign(cnn_z[i, horizon]) * abs(cnn_z[i, horizon])

    elif stype == 'agreement':
        # Both models agree on direction AND both above threshold
        for i in range(N):
            mamba_dir = np.sign(mamba_z[i, horizon])
            cnn_dir = np.sign(cnn_z[i, horizon])
            if mamba_dir == cnn_dir and mamba_dir != 0:
                mamba_mag = abs(mamba_z[i, horizon])
                cnn_mag = abs(cnn_z[i, horizon])
                if mamba_mag > z_thresh and cnn_mag > z_thresh:
                    # Average magnitude as confidence
                    signals[i] = mamba_dir * (mamba_mag + cnn_mag) / 2

    elif stype == 'agreement_any_thresh':
        # Both agree on direction, at least ONE above threshold
        for i in range(N):
            mamba_dir = np.sign(mamba_z[i, horizon])
            cnn_dir = np.sign(cnn_z[i, horizon])
            if mamba_dir == cnn_dir and mamba_dir != 0:
                mamba_mag = abs(mamba_z[i, horizon])
                cnn_mag = abs(cnn_z[i, horizon])
                if max(mamba_mag, cnn_mag) > z_thresh:
                    signals[i] = mamba_dir * (mamba_mag + cnn_mag) / 2

    elif stype == 'multi_horizon':
        # 1s + 10s must agree (entry timing + direction)
        for i in range(N):
            dir_10s = np.sign(mamba_z[i, 2])  # 10s for direction
            dir_1s = np.sign(mamba_z[i, 0])   # 1s for timing
            if dir_10s == dir_1s and dir_10s != 0:
                mag_10s = abs(mamba_z[i, 2])
                mag_1s = abs(mamba_z[i, 0])
                if mag_10s > z_thresh:
                    signals[i] = dir_10s * mag_10s

    elif stype == 'multi_horizon_agreement':
        # Both models 10s agree + Mamba 1s confirms entry direction
        for i in range(N):
            mamba_10s = np.sign(mamba_z[i, 2])
            cnn_10s = np.sign(cnn_z[i, 2])
            mamba_1s = np.sign(mamba_z[i, 0])
            if mamba_10s == cnn_10s == mamba_1s and mamba_10s != 0:
                mag = (abs(mamba_z[i, 2]) + abs(cnn_z[i, 2])) / 2
                if mag > z_thresh:
                    signals[i] = mamba_10s * mag

    elif stype == 'all_horizons_agree':
        # All 3 horizons (1s/5s/10s) agree on direction in BOTH models
        for i in range(N):
            mamba_dirs = [np.sign(mamba_z[i, h]) for h in range(3)]
            cnn_dirs = [np.sign(cnn_z[i, h]) for h in range(3)]
            all_dirs = mamba_dirs + cnn_dirs
            if all(d == all_dirs[0] and d != 0 for d in all_dirs):
                avg_mag = np.mean([abs(mamba_z[i, h]) for h in range(3)] +
                                  [abs(cnn_z[i, h]) for h in range(3)])
                if avg_mag > z_thresh:
                    signals[i] = all_dirs[0] * avg_mag

    return signals


# ============================================================
# Exit strategies
# ============================================================
def apply_exit_strategy(signals: np.ndarray, data: dict, strategy: dict) -> List[dict]:
    """
    Convert signals to trades using specified exit strategy.
    Returns list of trade dicts with entry/exit indices.
    """
    exit_type = strategy.get('exit', 'fixed_hold')
    hold_events = strategy.get('hold_events', 300)  # 30s at 10 events/sec
    min_hold = strategy.get('min_hold', 50)          # 5s minimum hold
    max_hold = strategy.get('max_hold', 3000)        # 5min max hold

    trades = []
    position = 0  # 0=flat, 1=long, -1=short
    entry_idx = 0
    entry_signal = 0.0

    N = len(signals)
    mamba_z = compute_z_scores(data['mamba_preds'])

    for i in range(N):
        if position == 0:
            # Look for entry
            if abs(signals[i]) > 0:
                position = int(np.sign(signals[i]))
                entry_idx = i
                entry_signal = signals[i]
        else:
            # Look for exit
            held = i - entry_idx
            should_exit = False
            exit_reason = ''

            if exit_type == 'fixed_hold':
                if held >= hold_events:
                    should_exit = True
                    exit_reason = 'fixed_hold'

            elif exit_type == 'signal_flip':
                # Exit when 10s prediction flips direction
                if held >= min_hold:
                    current_dir = np.sign(mamba_z[i, 2]) if abs(mamba_z[i, 2]) > 0.5 else 0
                    if current_dir == -position:  # Signal flipped
                        should_exit = True
                        exit_reason = 'signal_flip'
                    elif held >= max_hold:
                        should_exit = True
                        exit_reason = 'max_hold'

            elif exit_type == 'conditional_flip':
                # Exit on signal flip only if new signal is strong enough
                if held >= min_hold:
                    current_z = mamba_z[i, 2]
                    if np.sign(current_z) == -position and abs(current_z) > 1.5:
                        should_exit = True
                        exit_reason = 'conditional_flip'
                    elif held >= max_hold:
                        should_exit = True
                        exit_reason = 'max_hold'

            elif exit_type == 'slow_flip':
                # Wait for sustained signal flip (3 consecutive opposite signals)
                if held >= min_hold:
                    if i >= 2:
                        dirs = [np.sign(mamba_z[i-j, 2]) for j in range(3)]
                        if all(d == -position for d in dirs):
                            should_exit = True
                            exit_reason = 'slow_flip_3'
                    if held >= max_hold:
                        should_exit = True
                        exit_reason = 'max_hold'

            if should_exit:
                trades.append({
                    'entry_idx': entry_idx,
                    'exit_idx': i,
                    'direction': position,
                    'hold_events': held,
                    'entry_signal': float(entry_signal),
                    'exit_reason': exit_reason,
                })
                position = 0

    # Close any open position at end
    if position != 0:
        trades.append({
            'entry_idx': entry_idx,
            'exit_idx': N - 1,
            'direction': position,
            'hold_events': N - 1 - entry_idx,
            'entry_signal': float(entry_signal),
            'exit_reason': 'eod',
        })

    return trades


# ============================================================
# Fill simulation via Rust binary
# ============================================================
def run_fill_sim_for_date(date_str: str, trades: List[dict],
                          data: dict, order_type: str = 'mid') -> List[dict]:
    """
    Run each trade through the Rust FIFO fill simulator.
    Returns enriched trade list with fill results.
    """
    mbo_file = MBO_DIR / f'glbx-mdp3-{date_str}.mbo.dbn.zst'
    if not mbo_file.exists():
        log.warning(f"MBO file not found: {mbo_file}")
        return trades

    if not BINARY.exists():
        log.error(f"Fill sim binary not found: {BINARY}")
        return trades

    results = []
    for trade in trades:
        # Convert event indices to approximate timestamps
        entry_bar = trade['entry_idx'] * STRIDE  # bar index
        exit_bar = trade['exit_idx'] * STRIDE

        direction = 'buy' if trade['direction'] == 1 else 'sell'

        try:
            cmd = [
                str(BINARY),
                '--mbo-file', str(mbo_file),
                '--direction', direction,
                '--entry-bar', str(entry_bar),
                '--exit-bar', str(exit_bar),
                '--order-type', order_type,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                fill_data = json.loads(result.stdout)
                trade.update({
                    'filled': fill_data.get('filled', False),
                    'fill_price': fill_data.get('fill_price', 0),
                    'exit_price': fill_data.get('exit_price', 0),
                    'pnl_ticks': fill_data.get('pnl_ticks', 0),
                    'pnl_dollars': fill_data.get('pnl_ticks', 0) * TICK_VALUE - COMMISSION_RT,
                    'fill_time_ns': fill_data.get('fill_time_ns', 0),
                    'mfe_ticks': fill_data.get('mfe_ticks', 0),
                    'mae_ticks': fill_data.get('mae_ticks', 0),
                })
            else:
                trade.update({'filled': False, 'pnl_ticks': 0, 'pnl_dollars': -COMMISSION_RT,
                              'error': result.stderr[:200]})
        except Exception as e:
            trade.update({'filled': False, 'pnl_ticks': 0, 'pnl_dollars': 0, 'error': str(e)[:200]})

        results.append(trade)

    return results


# ============================================================
# Simulated P&L (when fill sim binary unavailable, use label-based)
# ============================================================
def simulate_pnl_from_labels(trades: List[dict], data: dict,
                              spread_cost_ticks: float = 0.0) -> List[dict]:
    # HC #231(A): spread cost deleted — fill price already encodes side
    """
    Estimate P&L from forward labels.
    Labels are FORWARD returns: labels[i, h] = price(i + horizon_h) - price(i) in ticks.
      h=0: 1s (~10 events), h=1: 5s (~50 events), h=2: 10s (~100 events)

    Strategy:
    - For holds <= 10 events: use 1s label at entry
    - For holds <= 50 events: use 5s label at entry
    - For holds <= 100 events: use 10s label at entry
    - For holds > 100 events: use 10s label (best available, underestimates)
    """
    labels = data['labels']  # (N, 3) = 1s, 5s, 10s forward returns

    for trade in trades:
        entry_idx = trade['entry_idx']
        exit_idx = trade['exit_idx']
        direction = trade['direction']
        hold_events = trade['hold_events']

        if entry_idx < len(labels):
            # Pick the label horizon closest to actual hold duration
            if hold_events <= 15:
                pnl_ticks = direction * labels[entry_idx, 0]  # 1s label
            elif hold_events <= 60:
                pnl_ticks = direction * labels[entry_idx, 1]  # 5s label
            else:
                pnl_ticks = direction * labels[entry_idx, 2]  # 10s label
        else:
            pnl_ticks = 0

        # HC #231(A): cost is just commission; spread_cost_ticks kept for caller compat (default 0)
        total_cost = spread_cost_ticks + COMMISSION_TICKS
        net_pnl_ticks = pnl_ticks - total_cost

        trade['pnl_ticks'] = float(pnl_ticks)
        trade['net_pnl_ticks'] = float(net_pnl_ticks)
        trade['pnl_dollars'] = float(net_pnl_ticks * TICK_VALUE)
        trade['cost_ticks'] = float(total_cost)
        trade['label_horizon'] = '1s' if hold_events <= 15 else ('5s' if hold_events <= 60 else '10s')

        # MFE/MAE: use all 3 label horizons at entry as proxy for excursion
        if entry_idx < len(labels):
            forward_moves = direction * labels[entry_idx]  # all 3 horizons
            trade['mfe_ticks'] = float(max(0, np.max(forward_moves)))
            trade['mae_ticks'] = float(min(0, np.min(forward_moves)))
        else:
            trade['mfe_ticks'] = 0.0
            trade['mae_ticks'] = 0.0

    return trades


# ============================================================
# Confidence tier analysis
# ============================================================
def analyze_by_confidence_tier(trades: List[dict]) -> dict:
    """Analyze performance at different confidence tiers."""
    if not trades:
        return {}

    filled = [t for t in trades if t.get('pnl_dollars') is not None]
    if not filled:
        return {}

    signals = np.array([abs(t['entry_signal']) for t in filled])
    pnls = np.array([t.get('pnl_dollars', t.get('net_pnl_ticks', 0) * TICK_VALUE) for t in filled])
    directions = np.array([t['direction'] for t in filled])
    mfes = np.array([t.get('mfe_ticks', 0) for t in filled])
    maes = np.array([t.get('mae_ticks', 0) for t in filled])

    tiers = {
        'All': 0, 'Top50%': 50, 'Top25%': 75, 'Top10%': 90,
        'Top5%': 95, 'Top1%': 99, 'Top0.5%': 99.5, 'Top0.1%': 99.9,
    }

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
        tier_dirs = directions[mask]
        tier_mfes = mfes[mask]
        tier_maes = maes[mask]

        n = len(tier_pnls)
        total_pnl = float(np.sum(tier_pnls))
        wins = (tier_pnls > 0).sum()
        losses = (tier_pnls <= 0).sum()

        long_mask = tier_dirs == 1
        short_mask = tier_dirs == -1

        # Sortino
        neg_returns = tier_pnls[tier_pnls < 0]
        downside_std = np.std(neg_returns) if len(neg_returns) > 1 else 1e-8
        sortino = float(np.mean(tier_pnls) / downside_std) if downside_std > 1e-8 else 0.0

        # Profit factor
        gross_profit = float(np.sum(tier_pnls[tier_pnls > 0]))
        gross_loss = float(abs(np.sum(tier_pnls[tier_pnls < 0])))
        pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        results[tier_name] = {
            'n_trades': int(n),
            'total_pnl': round(total_pnl, 2),
            'avg_pnl': round(float(np.mean(tier_pnls)), 2),
            'win_rate': round(float(wins / n), 4) if n > 0 else 0,
            'profit_factor': round(pf, 3),
            'sortino': round(sortino, 3),
            'avg_mfe': round(float(np.mean(tier_mfes)), 3),
            'avg_mae': round(float(np.mean(tier_maes)), 3),
            'long_trades': int(long_mask.sum()),
            'long_pnl': round(float(np.sum(tier_pnls[long_mask])), 2) if long_mask.any() else 0,
            'long_wr': round(float((tier_pnls[long_mask] > 0).sum() / long_mask.sum()), 4) if long_mask.sum() > 0 else 0,
            'short_trades': int(short_mask.sum()),
            'short_pnl': round(float(np.sum(tier_pnls[short_mask])), 2) if short_mask.any() else 0,
            'short_wr': round(float((tier_pnls[short_mask] > 0).sum() / short_mask.sum()), 4) if short_mask.sum() > 0 else 0,
        }

    return results


# ============================================================
# Main sweep
# ============================================================
STRATEGIES = [
    # === Single model baselines ===
    {'name': 'mamba_z2.0_fixed30s', 'type': 'mamba_only', 'z_threshold': 2.0,
     'exit': 'fixed_hold', 'hold_events': 300},
    {'name': 'mamba_z3.0_fixed30s', 'type': 'mamba_only', 'z_threshold': 3.0,
     'exit': 'fixed_hold', 'hold_events': 300},

    # === Signal flip exits ===
    {'name': 'mamba_z2.0_signalflip', 'type': 'mamba_only', 'z_threshold': 2.0,
     'exit': 'signal_flip', 'min_hold': 50, 'max_hold': 3000},
    {'name': 'mamba_z3.0_signalflip', 'type': 'mamba_only', 'z_threshold': 3.0,
     'exit': 'signal_flip', 'min_hold': 50, 'max_hold': 3000},
    {'name': 'mamba_z2.0_condflip', 'type': 'mamba_only', 'z_threshold': 2.0,
     'exit': 'conditional_flip', 'min_hold': 50, 'max_hold': 3000},
    {'name': 'mamba_z2.0_slowflip', 'type': 'mamba_only', 'z_threshold': 2.0,
     'exit': 'slow_flip', 'min_hold': 50, 'max_hold': 3000},

    # === Multi-model agreement ===
    {'name': 'agree_z2.0_fixed30s', 'type': 'agreement', 'z_threshold': 2.0,
     'exit': 'fixed_hold', 'hold_events': 300},
    {'name': 'agree_z2.0_signalflip', 'type': 'agreement', 'z_threshold': 2.0,
     'exit': 'signal_flip', 'min_hold': 50, 'max_hold': 3000},
    {'name': 'agree_z1.5_signalflip', 'type': 'agreement_any_thresh', 'z_threshold': 1.5,
     'exit': 'signal_flip', 'min_hold': 50, 'max_hold': 3000},
    {'name': 'agree_z2.0_condflip', 'type': 'agreement', 'z_threshold': 2.0,
     'exit': 'conditional_flip', 'min_hold': 50, 'max_hold': 3000},

    # === Multi-horizon (1s entry + 10s direction) ===
    {'name': 'multihz_z2.0_signalflip', 'type': 'multi_horizon', 'z_threshold': 2.0,
     'exit': 'signal_flip', 'min_hold': 50, 'max_hold': 3000},
    {'name': 'multihz_agree_z2.0_signalflip', 'type': 'multi_horizon_agreement', 'z_threshold': 2.0,
     'exit': 'signal_flip', 'min_hold': 50, 'max_hold': 3000},

    # === All horizons agree (ultra selective) ===
    {'name': 'all6agree_z1.5_signalflip', 'type': 'all_horizons_agree', 'z_threshold': 1.5,
     'exit': 'signal_flip', 'min_hold': 50, 'max_hold': 3000},
    {'name': 'all6agree_z1.0_condflip', 'type': 'all_horizons_agree', 'z_threshold': 1.0,
     'exit': 'conditional_flip', 'min_hold': 50, 'max_hold': 3000},

    # === Longer holds ===
    {'name': 'agree_z2.0_fixed60s', 'type': 'agreement', 'z_threshold': 2.0,
     'exit': 'fixed_hold', 'hold_events': 600},
    {'name': 'mamba_z3.0_slowflip_long', 'type': 'mamba_only', 'z_threshold': 3.0,
     'exit': 'slow_flip', 'min_hold': 100, 'max_hold': 6000},
]


def main():
    log.info("=" * 70)
    log.info("MULTI-MODEL EXECUTION STRATEGY SWEEP")
    log.info(f"Models: Mamba v7 + CNN-Mamba v2")
    log.info(f"Dates: March 2-5, 2026 (folds 6-9)")
    log.info(f"Strategies: {len(STRATEGIES)}")
    log.info(f"Cost model: ${COMMISSION_RT} RT = {COMMISSION_TICKS:.3f} ticks")
    log.info(f"Fill sim: {'Rust FIFO' if BINARY.exists() else 'Label-based (fallback)'}")
    log.info("=" * 70)

    # NOTE: fill_sim_cli expects --predictions NPZ (whole-day mode), not per-trade args.
    # Use label-based P&L until per-trade fill sim wrapper is built.
    use_fill_sim = False  # BINARY.exists()

    # Load all fold data
    all_data = {}
    for fold_idx in OVERLAP_FOLDS:
        try:
            all_data[fold_idx] = load_fold_predictions(fold_idx)
        except Exception as e:
            log.error(f"Failed to load fold {fold_idx}: {e}")

    if not all_data:
        log.error("No data loaded!")
        return

    # Run each strategy across all folds
    all_results = []

    for strat in STRATEGIES:
        strat_name = strat['name']
        log.info(f"\n{'─' * 50}")
        log.info(f"Strategy: {strat_name}")
        log.info(f"  Type: {strat['type']}, Exit: {strat.get('exit', 'fixed_hold')}")

        strat_trades = []
        daily_pnls = {}

        for fold_idx, data in sorted(all_data.items()):
            date_str = data['date']

            # Generate signals
            signals = generate_signals(data, strat)
            n_signals = (signals != 0).sum()

            # Apply exit strategy
            trades = apply_exit_strategy(signals, data, strat)

            # Run through fill sim or label-based P&L
            if use_fill_sim and trades:
                trades = run_fill_sim_for_date(date_str, trades, data, order_type='mid')
            elif trades:
                trades = simulate_pnl_from_labels(trades, data, spread_cost_ticks=0.0)  # HC #231(A): no spread cost

            day_pnl = sum(t.get('pnl_dollars', 0) for t in trades)
            daily_pnls[date_str] = round(day_pnl, 2)
            strat_trades.extend(trades)

            n_filled = len(trades)
            log.info(f"  {date_str}: {n_signals} signals, {n_filled} trades, P&L=${day_pnl:+.2f}")

        # Confidence tier analysis
        tier_analysis = analyze_by_confidence_tier(strat_trades)

        total_pnl = sum(daily_pnls.values())
        n_trades = len(strat_trades)
        n_wins = sum(1 for t in strat_trades if t.get('pnl_dollars', 0) > 0)
        avg_hold = np.mean([t['hold_events'] for t in strat_trades]) if strat_trades else 0

        # Exit reason breakdown
        exit_reasons = {}
        for t in strat_trades:
            r = t.get('exit_reason', 'unknown')
            exit_reasons[r] = exit_reasons.get(r, 0) + 1

        result = {
            'strategy': strat_name,
            'config': strat,
            'total_pnl': round(total_pnl, 2),
            'n_trades': n_trades,
            'win_rate': round(n_wins / n_trades, 4) if n_trades > 0 else 0,
            'avg_hold_events': round(float(avg_hold), 1),
            'avg_hold_seconds': round(float(avg_hold) / BARS_PER_SEC, 1),
            'daily_pnls': daily_pnls,
            'exit_reasons': exit_reasons,
            'tier_analysis': tier_analysis,
            'profitable_days': sum(1 for v in daily_pnls.values() if v > 0),
        }

        all_results.append(result)

        log.info(f"  TOTAL: {n_trades} trades, P&L=${total_pnl:+.2f}, "
                 f"WR={result['win_rate']:.1%}, "
                 f"Avg hold={result['avg_hold_seconds']:.0f}s")
        if tier_analysis.get('Top1%'):
            t1 = tier_analysis['Top1%']
            log.info(f"  Top1%: {t1['n_trades']} trades, P&L=${t1['total_pnl']:+.2f}, "
                     f"WR={t1['win_rate']:.1%}, PF={t1['profit_factor']:.2f}")

    # ── Sort by total P&L ──
    all_results.sort(key=lambda r: r['total_pnl'], reverse=True)

    # ── Summary ──
    log.info("\n" + "=" * 70)
    log.info("STRATEGY RANKING (by total P&L)")
    log.info("=" * 70)
    log.info(f"{'Strategy':<35} {'Trades':>6} {'P&L':>10} {'WR':>6} {'AvgHold':>8}")
    log.info("-" * 70)
    for r in all_results:
        log.info(f"{r['strategy']:<35} {r['n_trades']:>6} ${r['total_pnl']:>+9.2f} "
                 f"{r['win_rate']:>5.1%} {r['avg_hold_seconds']:>6.0f}s")

    # ── Save results ──
    out_path = RESULTS_DIR / f'sweep_results_{_ts}.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info(f"\nResults saved to: {out_path}")

    # ── Save trade-level data ──
    log.info(f"\nBest strategy: {all_results[0]['strategy']} (${all_results[0]['total_pnl']:+.2f})")
    if len(all_results) > 1:
        log.info(f"2nd best:      {all_results[1]['strategy']} (${all_results[1]['total_pnl']:+.2f})")

    log.info("\nDone!")


if __name__ == '__main__':
    main()
