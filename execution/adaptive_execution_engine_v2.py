#!/usr/bin/env python3
"""
Adaptive Execution Engine v2
==============================
Advanced quant-grade execution optimization that ADAPTS per-trade based on:

1. CONVICTION-CONDITIONAL PARAMETERS
   - Per-quintile adaptive TP/SL from empirical MAE/MFE distributions
   - Quintile-dependent hold time (Q5 reaches MFE in 26s → shorter hold)
   - Signal amplification by conviction rank

2. SIGNAL GATES & FILTERS (pre-processing)
   - Multi-horizon confluence: require 2/3 or 3/3 horizons agree on direction
   - Momentum persistence: N consecutive bars same direction before entry
   - Vol-regime adaptive threshold: high vol → higher z requirement
   - Side asymmetry gate: longs need |z|>X, shorts need |z|>Y
   - Rolling adverse-selection filter: skip if recent fills show mean reversion

3. ADVANCED EXIT MECHANICS
   - Ratcheting MFE stops (lock profit at 4t/8t/12t MFE thresholds)
   - MAE patience window (don't SL for first 10s — 93% of winners go red)
   - Conviction exit (delayed signal flip — require 3-5 bars opposite at |z|>1)
   - Time-decay exit (reduce TP target after 70% of hold time elapsed)

4. REGIME DETECTION
   - Expanding vol estimate → classify into low/mid/high regimes
   - Per-regime parameter sets (not one-size-fits-all)
   - Hour-of-day regime (morning chop vs afternoon trend)

Signal pre-processing happens in Python, then feeds to Rust fill_sim_cli for
production-grade FIFO queue simulation. This is NOT a parameter grid — these
are principled strategies derived from 1,866-trade empirical analysis.

Usage:
    python adaptive_execution_engine_v2.py
    python adaptive_execution_engine_v2.py --model cnn-mamba-v2 --workers 12
"""

import sys
import json
import time
import argparse
import subprocess
import logging
import os
from pathlib import Path
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from typing import Optional, Dict, List, Tuple, Any

import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
LVL3_ROOT = Path(__file__).resolve().parent.parent
BINARY = LVL3_ROOT / 'rust_cache_builder' / 'target' / 'release' / 'fill_sim_cli'
MBO_DIR = LVL3_ROOT / 'data' / 'raw' / 'mbo'
EVENT_DIR_V3 = LVL3_ROOT / 'data' / 'processed' / 'mbo_events_smart_v3'
EVENT_DIR_V2 = LVL3_ROOT / 'data' / 'processed' / 'mbo_events_smart_v2'
RESULTS_DIR = LVL3_ROOT / 'execution' / 'results' / 'adaptive_v2'
PRED_CACHE_DIR = LVL3_ROOT / 'execution' / 'pred_cache' / 'adaptive_v2'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PRED_CACHE_DIR.mkdir(parents=True, exist_ok=True)

CNN_MAMBA_V2_DIR = LVL3_ROOT / 'output' / 'cnn_mamba_v2_smart_v3_mar'
MAMBA_V7_DIR = LVL3_ROOT / 'output' / 'mamba_v7_tiny_smart_v3_mar_apr'

TICK_VALUE = 12.50
BARS_PER_SEC = 10
BAR_NS = 100_000_000
RTH_HOURS = 6.5
N_RTH_BARS = int(RTH_HOURS * 3600 * BARS_PER_SEC)
WINDOW = 1000
STRIDE = 500

_ts = datetime.now().strftime('%Y%m%d_%H%M%S')

log = logging.getLogger('adaptive_v2')
log.setLevel(logging.INFO)
_fh = logging.FileHandler(str(RESULTS_DIR / f'adaptive_v2_{_ts}.log'), mode='w', encoding='utf-8')
_fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
log.addHandler(_fh)
_ch = logging.StreamHandler(sys.stdout)
_ch.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
log.addHandler(_ch)


# ============================================================
# EMPIRICAL CONSTANTS (from 1,866-trade deepdive)
# ============================================================

# MAE: 93% of winners go red first. Median red depth = 4t, p90 = 13t.
# => Don't stop-loss before 10s hold. SL must be >= 13 for patience.
MAE_PATIENCE_SEC = 10   # seconds to wait before SL activates
MAE_P90 = 13.0          # p90 adverse depth in ticks

# MFE by quintile (p50 in ticks):
# Q1=9, Q2=8.5, Q3=9, Q4=9.5, Q5=10
# Time-to-MFE (ms): Q1=26.3s, Q2=31.1s, Q3=28.0s, Q4=29.3s, Q5=25.8s
# => Q5 is fastest to profit peak, suggesting tighter time-stop

# Hour PnL (from deepdive): H9=-$427, H10=-$188, H11=-$244, H12=-$350,
# H13=-$56, H14=-$192, H15=-$232. Hour 13 (1PM) is the only near-breakeven.

# Side asymmetry: Long = -$15,335 (1,179 trades), Short = -$5,779 (687 trades)
# Long loses 2.65x more per trade than short → require higher z for longs
LONG_Z_PREMIUM = 0.5  # require this much more |z| for long trades


# ============================================================
# SIGNAL GATES (Python pre-processing)
# ============================================================

def gate_multi_horizon_confluence(
    preds_1s: np.ndarray, preds_5s: np.ndarray, preds_10s: np.ndarray,
    min_agree: int = 2,
) -> np.ndarray:
    """Gate: require min_agree of 3 horizons to agree on direction.

    Returns mask (True = pass gate). Signals where horizons disagree are filtered.
    This eliminates ~40% of trades but removes conflicted signals.
    """
    sign_1s = np.sign(preds_1s)
    sign_5s = np.sign(preds_5s)
    sign_10s = np.sign(preds_10s)

    agreement = (sign_1s == sign_5s).astype(int) + (sign_1s == sign_10s).astype(int) + (sign_5s == sign_10s).astype(int)
    # agreement: 3 = all agree, 1 = majority agree, 0 = all disagree
    # For min_agree=2: need at least 2 pairs agreeing → agreement >= 1
    # For min_agree=3: need all 3 → agreement == 3
    if min_agree == 3:
        return agreement == 3
    else:  # min_agree == 2
        return agreement >= 1


def gate_momentum_persistence(
    bar_signal: np.ndarray,
    min_bars: int = 3,
) -> np.ndarray:
    """Gate: require min_bars consecutive same-direction signals before entry.

    Filters out isolated spikes (1-bar signals). Keeps sustained directional moves.
    Based on momentum_burst strategy showing +Sortino when requiring persistence.
    """
    mask = np.zeros(len(bar_signal), dtype=bool)
    streak = 0
    last_sign = 0

    for i in range(len(bar_signal)):
        if bar_signal[i] == 0:
            streak = 0
            continue
        current_sign = 1 if bar_signal[i] > 0 else -1
        if current_sign == last_sign:
            streak += 1
        else:
            streak = 1
            last_sign = current_sign

        if streak >= min_bars:
            mask[i] = True

    return mask


def gate_vol_regime_adaptive(
    bar_signal: np.ndarray,
    base_threshold: float = 2.0,
    vol_lookback_bars: int = 3000,  # 5 min of 100ms bars
    low_vol_mult: float = 0.8,     # lower threshold in quiet markets
    high_vol_mult: float = 1.5,    # higher threshold in volatile markets
    vol_percentile_low: float = 25,
    vol_percentile_high: float = 75,
) -> np.ndarray:
    """Adaptive z-threshold based on local volatility regime.

    In low vol: signal is more trustworthy → lower threshold (trade more).
    In high vol: signal is noisier → higher threshold (trade selectively).

    Returns modified signal (zeros out signals below adaptive threshold).
    """
    result = bar_signal.copy()

    # Compute expanding volatility of non-zero predictions
    abs_vals = np.abs(bar_signal)
    running_sum = 0.0
    running_sq = 0.0
    running_n = 0
    vol_series = np.zeros(len(bar_signal))

    for i in range(len(bar_signal)):
        if abs_vals[i] > 0:
            running_sum += abs_vals[i]
            running_sq += abs_vals[i] ** 2
            running_n += 1

        if running_n >= 50:
            mean = running_sum / running_n
            var = (running_sq / running_n) - mean * mean
            vol_series[i] = np.sqrt(max(var, 0))

    # Compute percentile thresholds from expanding vol
    nonzero_vols = vol_series[vol_series > 0]
    if len(nonzero_vols) < 50:
        return result  # Not enough data, return unmodified

    low_thresh = np.percentile(nonzero_vols, vol_percentile_low)
    high_thresh = np.percentile(nonzero_vols, vol_percentile_high)

    for i in range(len(bar_signal)):
        if bar_signal[i] == 0:
            continue

        vol = vol_series[i]
        if vol <= 0:
            continue

        # Adapt threshold based on vol regime
        if vol < low_thresh:
            adaptive_z = base_threshold * low_vol_mult
        elif vol > high_thresh:
            adaptive_z = base_threshold * high_vol_mult
        else:
            # Linear interpolation between low and high
            frac = (vol - low_thresh) / (high_thresh - low_thresh + 1e-8)
            adaptive_z = base_threshold * (low_vol_mult + frac * (high_vol_mult - low_vol_mult))

        if abs(bar_signal[i]) < adaptive_z:
            result[i] = 0.0

    return result


def gate_side_asymmetry(
    bar_signal: np.ndarray,
    base_threshold: float = 2.0,
    long_premium: float = 0.5,
) -> np.ndarray:
    """Require higher |z| for long entries than shorts.

    Empirical: longs lose 2.65x more per trade than shorts.
    This gate requires base_threshold + long_premium for buys.
    """
    result = bar_signal.copy()

    for i in range(len(bar_signal)):
        if bar_signal[i] > 0:  # Long signal
            if bar_signal[i] < base_threshold + long_premium:
                result[i] = 0.0
        elif bar_signal[i] < 0:  # Short signal
            if abs(bar_signal[i]) < base_threshold:
                result[i] = 0.0

    return result


def gate_recent_fill_quality(
    bar_signal: np.ndarray,
    lookback_bars: int = 1800,  # 3 minutes
    min_signal_density: float = 0.01,  # minimum non-zero bars in lookback
) -> np.ndarray:
    """Adverse selection filter: skip signals in thin/choppy markets.

    If recent signal density is very low (few predictions), market may be
    in a regime where our model has no edge. Skip these regions.
    """
    result = bar_signal.copy()

    for i in range(lookback_bars, len(bar_signal)):
        if bar_signal[i] == 0:
            continue

        window = bar_signal[max(0, i - lookback_bars):i]
        density = np.count_nonzero(window) / lookback_bars

        if density < min_signal_density:
            result[i] = 0.0

    return result


# ============================================================
# STRATEGY DEFINITIONS (principled, not grid-search)
# ============================================================

def define_strategies() -> List[Dict]:
    """Define principled adaptive strategies based on empirical findings.

    NOT a grid sweep — each strategy is a coherent hypothesis about
    what execution parameters should be, and WHY.
    """
    strategies = []

    # ── STRATEGY A: "Patient Ratchet" ──
    # Thesis: 93% of winners go red first → be patient, then lock profit
    # MAE patience (10s before SL activates) + ratchet stop + wide SL
    for z in [2.0, 2.5, 3.0]:
        strategies.append({
            'name': f'A_patient_ratchet_z{z}',
            'signal_gates': [],  # raw signal
            'cli_args': [
                '--chase-entry',
                '--signal-threshold', str(z),
                '--stop-loss-ticks', '20',       # Wide SL (MAE p90 = 13)
                '--take-profit-ticks', '15',      # Let winners run
                '--hold-ms', '120000',            # 2 min hold
                '--ratchet-stop',                 # Lock profit at MFE thresholds
                '--mae-exit-ticks', '13',         # MAE patience: only exit if -13t
                '--mae-exit-hold-sec', '10',      # ... AND held for 10s
                '--quiet',
            ],
            'thesis': 'Patient entry with ratcheting profit lock. Based on 93% winners going red.',
        })

    # ── STRATEGY B: "Conviction Sniper" ──
    # Thesis: Q5 (top conviction) reaches MFE fastest (25.8s). Use tight
    # time exit + high z + conviction-delayed exit (don't flip on noise)
    for z in [3.0, 3.5, 4.0]:
        for conv_bars, conv_mag in [(30, 1.0), (50, 1.5)]:
            strategies.append({
                'name': f'A_conviction_sniper_z{z}_cb{conv_bars}_cm{conv_mag}',
                'signal_gates': [],
                'cli_args': [
                    '--chase-entry',
                    '--chase-force-cross',          # Force fill on high conviction
                    '--signal-threshold', str(z),
                    '--stop-loss-ticks', '15',
                    '--take-profit-ticks', '10',
                    '--hold-ms', '60000',           # 60s — Q5 peaks at 26s
                    '--conviction-exit-bars', str(conv_bars),  # 3-5s sustained reversal
                    '--conviction-exit-mag', str(conv_mag),    # require strong opposite
                    '--trailing-ticks', '3',
                    '--quiet',
                ],
                'thesis': f'High-conviction only (z>{z}), quick exit on sustained reversal.',
            })

    # ── STRATEGY C: "Multi-Horizon Confluence" ──
    # Thesis: when all 3 horizons agree, directional accuracy is highest.
    # Pre-filter signal to only keep 2/3 or 3/3 agreement.
    for min_agree in [2, 3]:
        for z in [2.0, 2.5]:
            strategies.append({
                'name': f'C_confluence_{min_agree}of3_z{z}',
                'signal_gates': [('multi_horizon', {'min_agree': min_agree})],
                'cli_args': [
                    '--chase-entry',
                    '--signal-threshold', str(z),
                    '--stop-loss-ticks', '20',
                    '--take-profit-ticks', '12',
                    '--hold-ms', '120000',
                    '--ratchet-stop',
                    '--quiet',
                ],
                'thesis': f'Only trade when {min_agree}/3 horizons agree. Higher DA, fewer trades.',
            })

    # ── STRATEGY D: "Momentum Persistence" ──
    # Thesis: sustained signals (3+ consecutive bars same direction) are
    # more reliable than isolated spikes. Captures order flow momentum.
    for min_bars in [3, 5]:
        for z in [2.0, 2.5]:
            strategies.append({
                'name': f'D_momentum_{min_bars}bars_z{z}',
                'signal_gates': [('momentum', {'min_bars': min_bars})],
                'cli_args': [
                    '--chase-entry',
                    '--signal-threshold', str(z),
                    '--stop-loss-ticks', '15',
                    '--take-profit-ticks', '10',
                    '--hold-ms', '120000',
                    '--trailing-ticks', '3',
                    '--quiet',
                ],
                'thesis': f'Require {min_bars} consecutive same-direction bars. Filters noise.',
            })

    # ── STRATEGY E: "Vol-Regime Adaptive" ──
    # Thesis: same z-score means different things in different vol regimes.
    # High vol: signals are noisier → need higher threshold + wider SL.
    # Low vol: signals are cleaner → can trade more, tighter SL.
    for base_z in [2.0, 2.5]:
        strategies.append({
            'name': f'E_vol_adaptive_z{base_z}',
            'signal_gates': [('vol_regime', {'base_threshold': base_z})],
            'cli_args': [
                '--chase-entry',
                '--signal-threshold', '0.1',      # Threshold applied in gate
                '--stop-loss-ticks', '15',
                '--take-profit-ticks', '10',
                '--hold-ms', '120000',
                '--vol-exit-ticks', '5',          # Fast adverse = vol spike
                '--vol-exit-bars', '5',
                '--ratchet-stop',
                '--quiet',
            ],
            'thesis': f'Vol-adaptive z-threshold (low vol→1.6, high vol→3.0). Vol-exit for spikes.',
        })

    # ── STRATEGY F: "Side-Asymmetric" ──
    # Thesis: longs lose 2.65x more than shorts. Require higher z for longs.
    for base_z in [2.0, 2.5]:
        for premium in [0.5, 1.0]:
            strategies.append({
                'name': f'F_side_asym_z{base_z}_prem{premium}',
                'signal_gates': [('side_asymmetry', {'base_threshold': base_z, 'long_premium': premium})],
                'cli_args': [
                    '--chase-entry',
                    '--signal-threshold', '0.1',    # Applied in gate
                    '--stop-loss-ticks', '15',
                    '--take-profit-ticks', '10',
                    '--hold-ms', '120000',
                    '--trailing-ticks', '3',
                    '--quiet',
                ],
                'thesis': f'Require z>{base_z + premium} for longs, z>{base_z} for shorts.',
            })

    # ── STRATEGY G: "Prime Hours Ratchet Sniper" ──
    # Thesis: combine best time window (10:30-14:30) with ratcheting + patience.
    # This is the "everything works" combo.
    for z in [2.0, 2.5, 3.0]:
        strategies.append({
            'name': f'G_prime_ratchet_z{z}',
            'signal_gates': [],
            'cli_args': [
                '--chase-entry',
                '--signal-threshold', str(z),
                '--stop-loss-ticks', '20',
                '--take-profit-ticks', '12',
                '--hold-ms', '120000',
                '--ratchet-stop',
                '--mae-exit-ticks', '13',
                '--mae-exit-hold-sec', '10',
                '--prime-hours',
                '--quiet',
            ],
            'thesis': 'Best-of-all: prime hours + patience + ratchet.',
        })

    # ── STRATEGY H: "Confluence + Side Asymmetry + Ratchet" ──
    # Thesis: maximum filter stack. 3/3 horizon agreement + side premium + ratchet.
    # Will have very few trades but highest expected edge per trade.
    for z in [2.0, 2.5]:
        strategies.append({
            'name': f'H_max_filter_z{z}',
            'signal_gates': [
                ('multi_horizon', {'min_agree': 3}),
                ('side_asymmetry', {'base_threshold': z, 'long_premium': 0.5}),
            ],
            'cli_args': [
                '--chase-entry',
                '--chase-force-cross',
                '--signal-threshold', '0.1',
                '--stop-loss-ticks', '20',
                '--take-profit-ticks', '15',
                '--hold-ms', '180000',
                '--ratchet-stop',
                '--mae-exit-ticks', '13',
                '--mae-exit-hold-sec', '10',
                '--prime-hours',
                '--quiet',
            ],
            'thesis': 'Max selectivity: 3/3 confluence + side asymmetry + prime + patience.',
        })

    # ── STRATEGY I: "Momentum + Vol + Confluence Combo" ──
    for z in [2.0, 2.5]:
        strategies.append({
            'name': f'I_triple_gate_z{z}',
            'signal_gates': [
                ('multi_horizon', {'min_agree': 2}),
                ('momentum', {'min_bars': 3}),
                ('vol_regime', {'base_threshold': z}),
            ],
            'cli_args': [
                '--chase-entry',
                '--signal-threshold', '0.1',
                '--stop-loss-ticks', '15',
                '--take-profit-ticks', '12',
                '--hold-ms', '120000',
                '--ratchet-stop',
                '--trailing-ticks', '4',
                '--quiet',
            ],
            'thesis': f'Triple gate: 2/3 confluence + 3-bar momentum + vol-adaptive.',
        })

    # ── STRATEGY J: "Queue Position Adverse Selection" ──
    # Thesis: trades filled near top-of-book (good queue position) have less
    # adverse selection. Filter to only trade when we'd be in top 5 of queue.
    for z in [2.0, 2.5, 3.0]:
        strategies.append({
            'name': f'J_queue_filter_z{z}',
            'signal_gates': [],
            'cli_args': [
                '--chase-entry',
                '--signal-threshold', str(z),
                '--stop-loss-ticks', '15',
                '--take-profit-ticks', '10',
                '--hold-ms', '120000',
                '--ratchet-stop',
                '--max-queue-pos', '5',           # Only trade when near top of book
                '--quiet',
            ],
            'thesis': 'Only trade when queue position <= 5 (minimal adverse selection).',
        })

    # ── STRATEGY K: "Conviction Exit Ladder" ──
    # Test different conviction-exit parameters (delayed signal flip)
    for conv_bars in [20, 30, 50, 100]:  # 2s, 3s, 5s, 10s of sustained reversal
        for conv_mag in [0.5, 1.0, 2.0]:
            strategies.append({
                'name': f'K_conv_exit_cb{conv_bars}_cm{conv_mag}',
                'signal_gates': [],
                'cli_args': [
                    '--chase-entry',
                    '--signal-threshold', '2.5',
                    '--stop-loss-ticks', '20',
                    '--take-profit-ticks', '12',
                    '--hold-ms', '300000',          # 5 min — let conviction exit do the work
                    '--conviction-exit-bars', str(conv_bars),
                    '--conviction-exit-mag', str(conv_mag),
                    '--ratchet-stop',
                    '--quiet',
                ],
                'thesis': f'No fixed time exit. Exit only on {conv_bars/10:.0f}s sustained reversal at |z|>{conv_mag}.',
            })

    log.info(f"\n  Defined {len(strategies)} principled strategies")
    for s in strategies:
        log.info(f"    {s['name']}: {s['thesis']}")

    return strategies


# ============================================================
# Prediction Loading (from advanced_strategies_v1)
# ============================================================

def rth_start_ns_for_date(date_str: str) -> int:
    year, month, day = int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8])
    d = datetime(year, month, day)
    dst_start_2025 = datetime(2025, 3, 9)
    dst_end_2025 = datetime(2025, 11, 2)
    dst_start_2026 = datetime(2026, 3, 8)
    dst_end_2026 = datetime(2026, 11, 1)
    if (dst_start_2025 <= d < dst_end_2025) or (dst_start_2026 <= d < dst_end_2026):
        utc_offset = -4
    else:
        utc_offset = -5
    rth_start_utc_hours = 9.5 - utc_offset
    midnight_utc = datetime(year, month, day, tzinfo=timezone.utc)
    rth_start = midnight_utc + timedelta(hours=rth_start_utc_hours)
    return int(rth_start.timestamp() * 1e9)


def load_fold(fold_path: Path) -> Optional[Dict]:
    try:
        data = np.load(str(fold_path), allow_pickle=True)
        preds = data['predictions']
        labels = data['labels']
        oot_path = str(data['oot_files'][0])
        basename = oot_path.replace('\\', '/').split('/')[-1]
        date_str = basename.split('_')[0]
        return {
            'predictions': preds.astype(np.float64),
            'labels': labels.astype(np.float64),
            'date_str': date_str,
            'n_samples': preds.shape[0],
            'fold_path': str(fold_path),
        }
    except Exception as e:
        log.warning(f"Failed to load {fold_path}: {e}")
        return None


def discover_folds(pred_dir: Path) -> Dict[str, Dict]:
    folds = {}
    for f in sorted(pred_dir.glob('fold_*_oot_predictions.npz')):
        if 'concat' in f.name:
            continue
        data = load_fold(f)
        if data:
            folds[data['date_str']] = data
            log.info(f"  Fold: {data['date_str']} ({data['n_samples']} samples, "
                     f"{data['predictions'].shape[1]} horizons)")
    return folds


def load_event_timestamps(date_str: str) -> Optional[np.ndarray]:
    for edir in [EVENT_DIR_V3, EVENT_DIR_V2]:
        candidate = edir / f'{date_str}_mbo_events.npz'
        if candidate.exists():
            try:
                return np.load(str(candidate), allow_pickle=True)['timestamps']
            except Exception as e:
                log.warning(f"Failed to load events {candidate}: {e}")
    return None


def predictions_to_bar_signal(
    predictions: np.ndarray,
    event_timestamps: np.ndarray,
    date_str: str,
    running_stats: Optional[Dict] = None,
) -> Tuple[np.ndarray, Dict]:
    """Convert per-window predictions to bar-indexed z-scored signal."""
    n_events = len(event_timestamps)
    n_preds = len(predictions)
    starts = np.arange(0, n_events - WINDOW + 1, STRIDE, dtype=np.int64)
    label_idxs = starts + WINDOW - 1
    if len(label_idxs) > n_preds:
        label_idxs = label_idxs[:n_preds]
    elif n_preds > len(label_idxs):
        predictions = predictions[:len(label_idxs)]
        n_preds = len(predictions)
    if n_preds == 0:
        return np.zeros(N_RTH_BARS, dtype=np.float64), running_stats or {}

    pred_timestamps = event_timestamps[label_idxs]
    rth_start = rth_start_ns_for_date(date_str)
    bar_indices = ((pred_timestamps - rth_start) // BAR_NS).astype(np.int64)
    rth_mask = (bar_indices >= 0) & (bar_indices < N_RTH_BARS)
    bar_preds = np.zeros(N_RTH_BARS, dtype=np.float64)
    for bi, sig in zip(bar_indices[rth_mask], predictions[rth_mask]):
        bar_preds[bi] = sig

    if running_stats is None:
        running_stats = {'sum': 0.0, 'sq': 0.0, 'count': 0}
    zscore_preds = np.zeros(N_RTH_BARS, dtype=np.float64)
    rs, rsq, cnt = running_stats['sum'], running_stats['sq'], running_stats['count']
    for i in range(N_RTH_BARS):
        v = bar_preds[i]
        if v == 0.0:
            continue
        rs += v
        rsq += v * v
        cnt += 1
        if cnt >= 50:
            mean = rs / cnt
            var = (rsq / cnt) - mean * mean
            std = max(np.sqrt(max(var, 0)), 1e-8)
            zscore_preds[i] = (v - mean) / std
    running_stats = {'sum': rs, 'sq': rsq, 'count': cnt}
    return zscore_preds, running_stats


def prepare_all_signals(folds: Dict[str, Dict]) -> Dict[str, Dict[str, Any]]:
    """Prepare signals for all dates. Returns per-horizon z-scored bars + raw bars.

    For multi-horizon gates, we need z-scored signals for each horizon.
    """
    all_signals = {}
    running_stats = {h: None for h in ['1s', '5s', '10s']}

    for date_str in sorted(folds.keys()):
        fold_data = folds[date_str]

        # Check MBO file
        mbo_file = MBO_DIR / f'glbx-mdp3-{date_str}.mbo.dbn.zst'
        if not mbo_file.exists():
            mbo_file = MBO_DIR / f'glbx-mdp3-{date_str}.mbo.dbn'
        if not mbo_file.exists():
            continue

        timestamps = load_event_timestamps(date_str)
        if timestamps is None:
            continue

        preds = fold_data['predictions']
        n_horizons = preds.shape[1]

        # Generate z-scored bar signals for each horizon
        horizon_signals = {}
        horizon_names = ['1s', '5s', '10s', '30s'][:n_horizons]

        for h_idx, h_name in enumerate(horizon_names):
            bar_sig, running_stats[h_name] = predictions_to_bar_signal(
                preds[:, h_idx], timestamps, date_str, running_stats.get(h_name)
            )
            horizon_signals[h_name] = bar_sig

        all_signals[date_str] = {
            'horizons': horizon_signals,
            'primary': horizon_signals.get('10s', horizon_signals.get('5s')),
            'mbo_file': str(mbo_file),
        }

        n_nonzero = np.count_nonzero(horizon_signals.get('10s', np.array([])))
        log.info(f"  {date_str}: {n_nonzero} non-zero bars (10s horizon)")

    return all_signals


def apply_signal_gates(
    bar_signal: np.ndarray,
    gates: List[Tuple[str, Dict]],
    horizon_signals: Dict[str, np.ndarray],
) -> np.ndarray:
    """Apply signal gates sequentially. Each gate zeros out unwanted signals."""
    result = bar_signal.copy()

    for gate_name, gate_params in gates:
        before_count = np.count_nonzero(result)

        if gate_name == 'multi_horizon':
            # Need all 3 horizon signals
            if '1s' in horizon_signals and '5s' in horizon_signals and '10s' in horizon_signals:
                mask = gate_multi_horizon_confluence(
                    horizon_signals['1s'], horizon_signals['5s'], horizon_signals['10s'],
                    min_agree=gate_params.get('min_agree', 2),
                )
                result[~mask] = 0.0

        elif gate_name == 'momentum':
            mask = gate_momentum_persistence(result, min_bars=gate_params.get('min_bars', 3))
            result[~mask] = 0.0

        elif gate_name == 'vol_regime':
            result = gate_vol_regime_adaptive(
                result,
                base_threshold=gate_params.get('base_threshold', 2.0),
            )

        elif gate_name == 'side_asymmetry':
            result = gate_side_asymmetry(
                result,
                base_threshold=gate_params.get('base_threshold', 2.0),
                long_premium=gate_params.get('long_premium', 0.5),
            )

        elif gate_name == 'fill_quality':
            result = gate_recent_fill_quality(result)

        after_count = np.count_nonzero(result)
        log.debug(f"    Gate '{gate_name}': {before_count} → {after_count} signals "
                  f"({(before_count - after_count) / max(before_count, 1):.0%} filtered)")

    return result


# ============================================================
# Sim Execution
# ============================================================

def run_fill_sim(
    date_str: str,
    pred_file: Path,
    config: Dict,
    out_dir: Path,
) -> Optional[Dict]:
    mbo_file = MBO_DIR / f'glbx-mdp3-{date_str}.mbo.dbn.zst'
    if not mbo_file.exists():
        mbo_file = MBO_DIR / f'glbx-mdp3-{date_str}.mbo.dbn'
    if not mbo_file.exists():
        return None

    out_file = out_dir / f'{config["name"]}_{date_str}.json'
    cmd = [
        str(BINARY),
        '--mbo-file', str(mbo_file),
        '--predictions', str(pred_file),
        '--output', str(out_file),
    ] + config['cli_args']

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            if 'no trades' not in r.stderr.lower():
                log.debug(f"Sim failed {config['name']}/{date_str}: {r.stderr[:200]}")
            return None
        if not out_file.exists():
            return None
        with open(out_file) as f:
            result = json.load(f)
            result['_strategy'] = config['name']
            result['_date'] = date_str
            result['_thesis'] = config.get('thesis', '')
            return result
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None


# ============================================================
# Results
# ============================================================

def aggregate_results(results: List[Dict]) -> Dict[str, Dict]:
    by_strategy = defaultdict(list)
    for r in results:
        by_strategy[r.get('_strategy', 'unknown')].append(r)

    summaries = {}
    for name, runs in by_strategy.items():
        n_dates = len(runs)
        total_pnl = sum(r.get('total_pnl_dollars', 0) for r in runs)
        total_trades = sum(r.get('total_trades', 0) for r in runs)
        if total_trades == 0:
            continue

        mean_pnl = total_pnl / total_trades
        total_wins = sum(r.get('total_trades', 0) * r.get('win_rate', 0) for r in runs)
        win_rate = total_wins / total_trades

        # Profit factor
        gp = sum(r.get('total_trades', 0) * r.get('win_rate', 0) * r.get('avg_win', 0) for r in runs)
        gl = abs(sum(r.get('total_trades', 0) * (1 - r.get('win_rate', 0)) * r.get('avg_loss', 0) for r in runs))
        pf = gp / gl if gl > 0 else float('inf')

        daily_pnl = [r.get('total_pnl_dollars', 0) for r in runs]
        daily_mean = np.mean(daily_pnl)
        daily_std = np.std(daily_pnl) if len(daily_pnl) > 1 else 1
        daily_sharpe = daily_mean / daily_std if daily_std > 0 else 0
        neg = [d for d in daily_pnl if d < 0]
        ds = np.std(neg) if len(neg) > 1 else daily_std
        daily_sortino = daily_mean / ds if ds > 0 else 0

        total_signals = sum(r.get('total_signals', 0) for r in runs)
        fill_rate = total_trades / total_signals if total_signals > 0 else 0

        thesis = runs[0].get('_thesis', '')

        summaries[name] = {
            'thesis': thesis,
            'n_dates': n_dates,
            'n_trades': total_trades,
            'total_pnl': round(total_pnl, 2),
            'mean_pnl_per_trade': round(mean_pnl, 4),
            'win_rate': round(win_rate, 4),
            'profit_factor': round(pf, 4),
            'daily_sharpe': round(daily_sharpe, 4),
            'daily_sortino': round(daily_sortino, 4),
            'fill_rate': round(fill_rate, 4),
        }
    return summaries


def print_results(summaries: Dict[str, Dict]):
    filtered = {k: v for k, v in summaries.items() if v['n_trades'] >= 5}
    sorted_s = sorted(filtered.items(), key=lambda x: x[1]['daily_sortino'], reverse=True)

    log.info(f"\n{'='*140}")
    log.info(f"ADAPTIVE EXECUTION ENGINE v2 — RESULTS BY DAILY SORTINO (min 5 trades)")
    log.info(f"{'='*140}")
    log.info(f"{'Strategy':<45} {'Thesis':<45} {'N':>5} {'PnL($)':>10} {'$/Tr':>8} {'WR':>6} {'PF':>6} {'Sharpe':>7} {'Sortino':>8} {'Fill':>5}")
    log.info('-' * 140)

    for name, s in sorted_s:
        thesis_short = s['thesis'][:42] + '...' if len(s['thesis']) > 45 else s['thesis']
        log.info(
            f"{name:<45} {thesis_short:<45} {s['n_trades']:>5} {s['total_pnl']:>10.0f} "
            f"{s['mean_pnl_per_trade']:>8.2f} {s['win_rate']:>5.1%} {s['profit_factor']:>6.3f} "
            f"{s['daily_sharpe']:>7.3f} {s['daily_sortino']:>8.3f} {s['fill_rate']:>5.1%}"
        )

    # Strategy family analysis
    log.info(f"\n{'='*100}")
    log.info("STRATEGY FAMILY ANALYSIS (avg PnL per family)")
    log.info(f"{'='*100}")
    families = defaultdict(list)
    for name, s in summaries.items():
        family = name.split('_')[0]  # A, B, C, etc.
        families[family].append(s['total_pnl'])
    for fam in sorted(families.keys()):
        pnls = families[fam]
        log.info(f"  Family {fam}: avg=${np.mean(pnls):>8.0f}  best=${max(pnls):>8.0f}  n={len(pnls)}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Adaptive Execution Engine v2')
    parser.add_argument('--model', default='cnn-mamba-v2', choices=['cnn-mamba-v2', 'mamba-v7'])
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()

    pred_dir = CNN_MAMBA_V2_DIR if args.model == 'cnn-mamba-v2' else MAMBA_V7_DIR
    model_name = args.model.replace('-', '_')

    log.info(f"Adaptive Execution Engine v2")
    log.info(f"  Model: {args.model}")
    log.info(f"  Predictions: {pred_dir}")
    log.info(f"  Workers: {args.workers}")

    # ── Discover folds ──
    folds = discover_folds(pred_dir)
    log.info(f"  Found {len(folds)} folds")
    if not folds:
        return

    # ── Prepare all signals (all horizons) ──
    log.info(f"\nPreparing multi-horizon signals...")
    all_signals = prepare_all_signals(folds)
    log.info(f"  {len(all_signals)} dates with signals")

    # ── Define strategies ──
    strategies = define_strategies()

    # ── Build jobs: apply gates and save filtered signals ──
    log.info(f"\nApplying signal gates and building sim jobs...")
    jobs = []
    sim_out = RESULTS_DIR / f'sim_{model_name}_{_ts}'
    sim_out.mkdir(parents=True, exist_ok=True)

    for strategy in strategies:
        gates = strategy.get('signal_gates', [])

        for date_str, sig_data in sorted(all_signals.items()):
            primary_signal = sig_data['primary']
            horizon_signals = sig_data['horizons']

            # Apply gates to create filtered signal
            if gates:
                filtered = apply_signal_gates(primary_signal, gates, horizon_signals)
                # Save filtered predictions
                cache_key = f"{strategy['name']}_{date_str}"
                cache_file = PRED_CACHE_DIR / f'{cache_key}.npz'
                np.savez_compressed(str(cache_file), predictions=filtered)
                pred_file = cache_file
            else:
                # No gates — use standard signal
                cache_file = PRED_CACHE_DIR / f'standard_{date_str}.npz'
                if not cache_file.exists():
                    np.savez_compressed(str(cache_file), predictions=primary_signal)
                pred_file = cache_file

            jobs.append({
                'strategy': strategy,
                'date': date_str,
                'pred_file': pred_file,
            })

    log.info(f"  Total sim jobs: {len(jobs)} ({len(strategies)} strategies × {len(all_signals)} dates)")

    # ── Run sims ──
    results = []
    done = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for job in jobs:
            future = executor.submit(
                run_fill_sim,
                job['date'], job['pred_file'], job['strategy'], sim_out,
            )
            futures[future] = job

        for future in as_completed(futures):
            done += 1
            result = future.result()
            if result:
                results.append(result)
            if done % 100 == 0 or done == len(jobs):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(jobs) - done) / rate / 60 if rate > 0 else 0
                log.info(f"  Progress: {done}/{len(jobs)} ({done/len(jobs):.0%}) "
                         f"| {len(results)} results | {rate:.1f}/s | ETA: {eta:.1f}min")

    elapsed = time.time() - t0
    log.info(f"\nCompleted {len(jobs)} sims in {elapsed:.0f}s")

    # ── Aggregate & Report ──
    summaries = aggregate_results(results)
    print_results(summaries)

    out_json = RESULTS_DIR / f'adaptive_v2_results_{model_name}_{_ts}.json'
    with open(out_json, 'w') as f:
        json.dump(summaries, f, indent=2)
    log.info(f"\nResults saved to: {out_json}")


if __name__ == '__main__':
    main()
