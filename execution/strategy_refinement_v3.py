#!/usr/bin/env python3
"""
Strategy Refinement v3 — CNN-Mamba v2 Execution Optimization
=============================================================
Purpose: Find profitable execution strategies for Monday paper trading.

Key insight from v1/v2: CNN-Mamba v2 has real alpha (17/27 profitable on labels)
but execution destroys it. The ONE profitable strategy was ultra-selective z>=5.0
(7 trades, 71.4% WR). We need MORE profitable strategies by fixing:
  1. Exit cost: signal-flip exits use market orders eating the spread -> use limit exits
  2. Signal quality: simulate PatchTST DA filter as proxy for future combined signal

Strategy Groups:
  G1: Smart Limit-Order Exits (trailing stops, bracket orders)
  G2: Simulated PatchTST DA Filter (Monte Carlo)
  G3: Time-Weighted Strategies (TOD filtering)
  G4: Position Sizing by Confidence (stop management per tier)

Run: python strategy_refinement_v3.py
"""

import sys
import gc
import json
import time
import argparse
import subprocess
import logging
import os
from pathlib import Path
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Tuple, Any
from collections import defaultdict

import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
LVL3_ROOT = Path(__file__).resolve().parent.parent
BINARY = LVL3_ROOT / 'rust_cache_builder' / 'target' / 'release' / 'fill_sim_cli'
MBO_DIR = LVL3_ROOT / 'data' / 'raw' / 'mbo'
EVENT_DIR_V3 = LVL3_ROOT / 'data' / 'processed' / 'mbo_events_smart_v3'
PRED_DIR = LVL3_ROOT / 'output' / 'cnn_mamba_v2_smart_v3_mar'
RESULTS_DIR = LVL3_ROOT / 'execution' / 'results' / 'refinement_v3'
PRED_CACHE_DIR = LVL3_ROOT / 'execution' / 'pred_cache' / 'refinement_v3'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PRED_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── ES Futures Constants ──────────────────────────────────────────────────
TICK_VALUE = 12.50
POINT_VALUE = 50.00
COMMISSION_RT = 4.70
COMMISSION_TICKS = COMMISSION_RT / TICK_VALUE

# ── Bar/Timing Constants ──
BARS_PER_SEC = 10
BAR_NS = 100_000_000
RTH_HOURS = 6.5
N_RTH_BARS = int(RTH_HOURS * 3600 * BARS_PER_SEC)  # 234000

# ── Model Constants ──
WINDOW = 1000
STRIDE = 500

# ── Fold Mapping (CNN-Mamba v2) ──
FOLD_DATES = {
    0: '20260223',
    1: '20260224',
    2: '20260225',
    3: '20260226',
}

# ── Timestamp ──
_ts = datetime.now().strftime('%Y%m%d_%H%M%S')

# ── Logging ──
_log_file = str(RESULTS_DIR / f'refinement_v3_{_ts}.log')
log = logging.getLogger('refinement_v3')
log.setLevel(logging.INFO)
_fh = logging.FileHandler(_log_file, mode='w', encoding='utf-8')
_fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
log.addHandler(_fh)
_ch = logging.StreamHandler(sys.stdout)
_ch.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
log.addHandler(_ch)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# ============================================================
# RTH Timestamp Utilities
# ============================================================

def rth_start_ns_for_date(date_str: str) -> int:
    year, month, day = int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8])
    d = datetime(year, month, day)
    dst_start_2026 = datetime(2026, 3, 8)
    dst_end_2026 = datetime(2026, 11, 1)
    dst_start_2025 = datetime(2025, 3, 9)
    dst_end_2025 = datetime(2025, 11, 2)
    if (dst_start_2025 <= d < dst_end_2025) or (dst_start_2026 <= d < dst_end_2026):
        utc_offset = -4
    else:
        utc_offset = -5
    rth_start_utc_hours = 9.5 - utc_offset
    midnight_utc = datetime(year, month, day, tzinfo=timezone.utc)
    rth_start = midnight_utc + timedelta(hours=rth_start_utc_hours)
    return int(rth_start.timestamp() * 1e9)


# ============================================================
# Prediction Loading & Signal Generation
# ============================================================

def load_fold(fold_idx: int) -> Optional[Dict]:
    """Load CNN-Mamba v2 fold predictions."""
    fold_path = PRED_DIR / f'fold_{fold_idx:02d}_oot_predictions.npz'
    if not fold_path.exists():
        log.warning(f"Fold {fold_idx} not found: {fold_path}")
        return None
    data = np.load(str(fold_path), allow_pickle=True)
    return {
        'predictions': data['predictions'].astype(np.float64),  # (N, 3)
        'labels': data['labels'].astype(np.float64),            # (N, 3)
        'date_str': FOLD_DATES[fold_idx],
        'n_samples': data['predictions'].shape[0],
    }


def convert_predictions_to_bar_signal(
    predictions: np.ndarray,
    event_timestamps: np.ndarray,
    date_str: str,
    running_stats: Optional[Dict] = None,
) -> Tuple[np.ndarray, Dict]:
    """Convert per-window predictions to bar-indexed z-scored signal.

    Same logic as combined_execution_v1.py for consistency.
    """
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
    bar_indices_rth = bar_indices[rth_mask]
    predictions_rth = predictions[rth_mask]

    bar_preds = np.zeros(N_RTH_BARS, dtype=np.float64)
    for bi, sig in zip(bar_indices_rth, predictions_rth):
        bar_preds[bi] = sig

    # Expanding z-score
    if running_stats is None:
        running_stats = {'sum': 0.0, 'sq': 0.0, 'count': 0}

    zscore_preds = np.zeros(N_RTH_BARS, dtype=np.float64)
    rs = running_stats['sum']
    rsq = running_stats['sq']
    cnt = running_stats['count']

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


def prepare_all_predictions() -> Dict[str, Path]:
    """Prepare bar-indexed prediction files for all 4 folds."""
    log.info("Preparing prediction files...")
    saved = {}
    running_stats = None  # Walk-forward across folds

    for fold_idx in sorted(FOLD_DATES.keys()):
        date_str = FOLD_DATES[fold_idx]
        cache_file = PRED_CACHE_DIR / f'cnn_mamba_v2_{date_str}.npz'

        if cache_file.exists():
            saved[date_str] = cache_file
            log.info(f"  {date_str}: cached")
            # Still need to advance running stats for walk-forward
            data = np.load(str(cache_file))
            bar_preds = data['predictions']
            nonzero = bar_preds[bar_preds != 0]
            if running_stats is None:
                running_stats = {'sum': 0.0, 'sq': 0.0, 'count': 0}
            running_stats['sum'] += float(np.sum(nonzero))
            running_stats['sq'] += float(np.sum(nonzero**2))
            running_stats['count'] += len(nonzero)
            continue

        fold_data = load_fold(fold_idx)
        if fold_data is None:
            continue

        # Load event timestamps
        ev_file = EVENT_DIR_V3 / f'{date_str}_mbo_events.npz'
        if not ev_file.exists():
            log.warning(f"  {date_str}: no event file")
            continue

        ev_data = np.load(str(ev_file), allow_pickle=True)
        timestamps = ev_data['timestamps']

        # Use 10s horizon (conviction) as primary signal
        preds_10s = fold_data['predictions'][:, 2]

        bar_signal, running_stats = convert_predictions_to_bar_signal(
            preds_10s, timestamps, date_str, running_stats
        )

        n_nonzero = int(np.count_nonzero(bar_signal))
        np.savez_compressed(str(cache_file), predictions=bar_signal)
        saved[date_str] = cache_file
        log.info(f"  {date_str}: {n_nonzero} bar signals")

        del ev_data, timestamps
        gc.collect()

    log.info(f"  Prepared {len(saved)} prediction files")
    return saved


# ============================================================
# Strategy Definitions
# ============================================================

@dataclass
class Strategy:
    """Execution strategy for fill_sim_cli."""
    label: str
    description: str
    group: str

    # Entry
    market_entry: bool = False
    chase_entry: bool = False
    chase_max_ticks: int = 1
    chase_max_reprices: int = 3
    chase_force_cross: bool = False

    # Signal
    signal_threshold: float = 2.0

    # Exit
    hold_ms: int = 10000
    signal_flip_exit: bool = False
    take_profit_ticks: Optional[int] = None
    stop_loss_ticks: Optional[int] = None
    trailing_ticks: Optional[int] = None
    ratchet_stop: bool = False
    conviction_exit_bars: int = 0
    conviction_exit_mag: float = 0.0

    # Time filters
    prime_hours: bool = False
    time_window_start: str = ""
    time_window_end: str = ""

    def to_cli_args(self) -> List[str]:
        args = []
        if self.market_entry:
            args.append('--market-entry')
        elif self.chase_entry:
            args.append('--chase-entry')
            args.extend(['--chase-max-ticks', str(self.chase_max_ticks)])
            args.extend(['--chase-max-reprices', str(self.chase_max_reprices)])
            if self.chase_force_cross:
                args.append('--chase-force-cross')

        args.extend(['--signal-threshold', str(self.signal_threshold)])
        args.extend(['--hold-ms', str(self.hold_ms)])

        if self.signal_flip_exit:
            args.append('--signal-flip-exit')
        if self.take_profit_ticks is not None:
            args.extend(['--take-profit-ticks', str(self.take_profit_ticks)])
        if self.stop_loss_ticks is not None:
            args.extend(['--stop-loss-ticks', str(self.stop_loss_ticks)])
        if self.trailing_ticks is not None:
            args.extend(['--trailing-ticks', str(self.trailing_ticks)])
        if self.ratchet_stop:
            args.append('--ratchet-stop')
        if self.conviction_exit_bars > 0:
            args.extend(['--conviction-exit-bars', str(self.conviction_exit_bars)])
            args.extend(['--conviction-exit-mag', str(self.conviction_exit_mag)])
        if self.prime_hours:
            args.append('--prime-hours')
        if self.time_window_start:
            args.extend(['--time-window-start', self.time_window_start])
            args.extend(['--time-window-end', self.time_window_end])
        args.append('--quiet')
        return args


def build_all_strategies() -> List[Strategy]:
    """Build complete strategy universe for refinement sweep."""
    strategies = []

    # ================================================================
    # GROUP 1: Smart Limit-Order Exits
    # The key insight: signal-flip exits are market orders (pay spread).
    # Trailing stops and bracket orders exit passively (save the spread).
    # ================================================================

    z_levels = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
    hold_times = [5000, 10000, 30000, 60000]

    # G1a: Trailing stops at various widths
    for z in z_levels:
        for trail in [2, 3, 4]:
            for hold in hold_times:
                hold_label = f'{hold//1000}s'
                strategies.append(Strategy(
                    label=f'trail{trail}t_z{z:.1f}_{hold_label}',
                    description=f'Chase 1x3, trail {trail}t, z>{z}, hold {hold_label}',
                    group='G1_trailing',
                    chase_entry=True, signal_threshold=z,
                    hold_ms=hold, trailing_ticks=trail,
                ))

    # G1b: Fixed bracket orders (SL + TP, both passive)
    bracket_combos = [
        (2, 3, 'tight'),
        (3, 5, 'medium'),
        (4, 8, 'wide'),
        (3, 4, 'balanced'),
        (2, 5, 'asym_tight'),
        (3, 8, 'asym_wide'),
    ]
    for z in z_levels:
        for sl, tp, bname in bracket_combos:
            for hold in [30000, 60000]:
                hold_label = f'{hold//1000}s'
                strategies.append(Strategy(
                    label=f'bracket_{bname}_z{z:.1f}_{hold_label}',
                    description=f'Chase 1x3, SL={sl}t TP={tp}t, z>{z}, hold {hold_label}',
                    group='G1_bracket',
                    chase_entry=True, signal_threshold=z,
                    hold_ms=hold, stop_loss_ticks=sl, take_profit_ticks=tp,
                ))

    # G1c: Trailing + stop-loss combo
    for z in [2.5, 3.0, 3.5, 4.0, 5.0]:
        for trail in [2, 3]:
            for sl in [3, 4, 5]:
                strategies.append(Strategy(
                    label=f'trail{trail}t_sl{sl}t_z{z:.1f}_60s',
                    description=f'Chase 1x3, trail {trail}t + SL {sl}t, z>{z}, hold 60s',
                    group='G1_trail_sl',
                    chase_entry=True, signal_threshold=z,
                    hold_ms=60000, trailing_ticks=trail, stop_loss_ticks=sl,
                ))

    # G1d: Ratchet stop (progressive profit lock-in)
    for z in [2.5, 3.0, 3.5, 4.0, 5.0]:
        for hold in [30000, 60000]:
            hold_label = f'{hold//1000}s'
            strategies.append(Strategy(
                label=f'ratchet_z{z:.1f}_{hold_label}',
                description=f'Chase 1x3, ratchet stop, z>{z}, hold {hold_label}',
                group='G1_ratchet',
                chase_entry=True, signal_threshold=z,
                hold_ms=hold, ratchet_stop=True,
            ))

    # ================================================================
    # GROUP 3: Time-Weighted Strategies
    # ================================================================

    # Open (first 30 min) and close (last 30 min) — highest volatility
    for z in [2.0, 2.5, 3.0, 3.5]:
        for trail in [None, 2, 3]:
            trail_label = f'_trail{trail}t' if trail else ''
            trail_desc = f', trail {trail}t' if trail else ''
            # Open only
            strategies.append(Strategy(
                label=f'open30_z{z:.1f}{trail_label}_30s',
                description=f'Open 30min, z>{z}{trail_desc}, hold 30s',
                group='G3_time',
                chase_entry=True, signal_threshold=z,
                hold_ms=30000, trailing_ticks=trail,
                time_window_start='09:30', time_window_end='10:00',
            ))
            # Close only
            strategies.append(Strategy(
                label=f'close30_z{z:.1f}{trail_label}_30s',
                description=f'Close 30min, z>{z}{trail_desc}, hold 30s',
                group='G3_time',
                chase_entry=True, signal_threshold=z,
                hold_ms=30000, trailing_ticks=trail,
                time_window_start='15:30', time_window_end='16:00',
            ))

    # Midday (lower vol, potentially cleaner signal)
    for z in [2.0, 2.5, 3.0]:
        for trail in [None, 2]:
            trail_label = f'_trail{trail}t' if trail else ''
            trail_desc = f', trail {trail}t' if trail else ''
            strategies.append(Strategy(
                label=f'midday_z{z:.1f}{trail_label}_30s',
                description=f'Midday 11-14, z>{z}{trail_desc}, hold 30s',
                group='G3_time',
                chase_entry=True, signal_threshold=z,
                hold_ms=30000, trailing_ticks=trail,
                time_window_start='11:00', time_window_end='14:00',
            ))

    # Exclude first 5 and last 5 min
    for z in [2.5, 3.0, 3.5]:
        strategies.append(Strategy(
            label=f'core_z{z:.1f}_trail2t_30s',
            description=f'Core hours 9:35-15:55, z>{z}, trail 2t, hold 30s',
            group='G3_time',
            chase_entry=True, signal_threshold=z,
            hold_ms=30000, trailing_ticks=2,
            time_window_start='09:35', time_window_end='15:55',
        ))

    # Prime hours (10:30-14:30)
    for z in [2.0, 2.5, 3.0, 3.5]:
        for trail in [None, 2, 3]:
            trail_label = f'_trail{trail}t' if trail else ''
            trail_desc = f', trail {trail}t' if trail else ''
            strategies.append(Strategy(
                label=f'prime_z{z:.1f}{trail_label}_30s',
                description=f'Prime 10:30-14:30, z>{z}{trail_desc}, hold 30s',
                group='G3_time',
                chase_entry=True, signal_threshold=z,
                hold_ms=30000, trailing_ticks=trail, prime_hours=True,
            ))

    # ================================================================
    # GROUP 4: Position Sizing by Confidence (stop management per tier)
    # z=2.0-3.0: tight stops (protect capital on lower conviction)
    # z=3.0-4.0: medium stops
    # z=4.0+: wide stops (let winners run on high conviction)
    # ================================================================

    # Low conviction: tight bracket
    for z in [2.0, 2.5]:
        strategies.append(Strategy(
            label=f'tier_low_z{z:.1f}_sl2_tp3_30s',
            description=f'Low conviction z>{z}, SL=2t TP=3t, 30s',
            group='G4_sizing',
            chase_entry=True, signal_threshold=z,
            hold_ms=30000, stop_loss_ticks=2, take_profit_ticks=3,
        ))
        strategies.append(Strategy(
            label=f'tier_low_z{z:.1f}_trail2t_30s',
            description=f'Low conviction z>{z}, trail 2t, 30s',
            group='G4_sizing',
            chase_entry=True, signal_threshold=z,
            hold_ms=30000, trailing_ticks=2,
        ))

    # Medium conviction: balanced bracket
    for z in [3.0, 3.5]:
        strategies.append(Strategy(
            label=f'tier_med_z{z:.1f}_sl3_tp5_60s',
            description=f'Med conviction z>{z}, SL=3t TP=5t, 60s',
            group='G4_sizing',
            chase_entry=True, signal_threshold=z,
            hold_ms=60000, stop_loss_ticks=3, take_profit_ticks=5,
        ))
        strategies.append(Strategy(
            label=f'tier_med_z{z:.1f}_trail3t_60s',
            description=f'Med conviction z>{z}, trail 3t, 60s',
            group='G4_sizing',
            chase_entry=True, signal_threshold=z,
            hold_ms=60000, trailing_ticks=3,
        ))

    # High conviction: wide stops, let winners run
    for z in [4.0, 5.0]:
        strategies.append(Strategy(
            label=f'tier_high_z{z:.1f}_sl4_tp8_60s',
            description=f'High conviction z>{z}, SL=4t TP=8t, 60s',
            group='G4_sizing',
            chase_entry=True, signal_threshold=z,
            hold_ms=60000, stop_loss_ticks=4, take_profit_ticks=8,
        ))
        strategies.append(Strategy(
            label=f'tier_high_z{z:.1f}_trail4t_60s',
            description=f'High conviction z>{z}, trail 4t, 60s',
            group='G4_sizing',
            chase_entry=True, signal_threshold=z,
            hold_ms=60000, trailing_ticks=4,
        ))
        strategies.append(Strategy(
            label=f'tier_high_z{z:.1f}_ratchet_60s',
            description=f'High conviction z>{z}, ratchet stop, 60s',
            group='G4_sizing',
            chase_entry=True, signal_threshold=z,
            hold_ms=60000, ratchet_stop=True,
        ))

    # ================================================================
    # Baseline reference strategies (for comparison)
    # ================================================================
    # The known profitable one: z>=5.0 simple hold
    for hold in [5000, 10000, 30000]:
        hold_label = f'{hold//1000}s'
        strategies.append(Strategy(
            label=f'baseline_z5.0_{hold_label}',
            description=f'Baseline: chase 1x3, z>5.0, hold {hold_label}',
            group='baseline',
            chase_entry=True, signal_threshold=5.0, hold_ms=hold,
        ))

    # Simple hold at various thresholds (baseline comparison)
    for z in [2.0, 2.5, 3.0, 3.5, 4.0]:
        for hold in [10000, 30000]:
            hold_label = f'{hold//1000}s'
            strategies.append(Strategy(
                label=f'baseline_z{z:.1f}_{hold_label}',
                description=f'Baseline: chase 1x3, z>{z}, hold {hold_label}',
                group='baseline',
                chase_entry=True, signal_threshold=z, hold_ms=hold,
            ))

    log.info(f"Built {len(strategies)} strategies across groups: "
             f"{len(set(s.group for s in strategies))} groups")
    return strategies


# ============================================================
# Fill Simulator Interface
# ============================================================

def run_fill_sim(
    date_str: str,
    pred_file: Path,
    strategy: Strategy,
    out_dir: Path,
) -> Optional[Dict]:
    """Run Rust fill_sim_cli for a single day + strategy."""
    mbo_file = MBO_DIR / f'glbx-mdp3-{date_str}.mbo.dbn.zst'
    if not mbo_file.exists():
        mbo_file = MBO_DIR / f'glbx-mdp3-{date_str}.mbo.dbn'
    if not mbo_file.exists():
        return None

    out_file = out_dir / f'{strategy.label}_{date_str}.json'

    cmd = [
        str(BINARY),
        '--mbo-file', str(mbo_file),
        '--predictions', str(pred_file),
        '--output', str(out_file),
    ] + strategy.to_cli_args()

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            return None
        if not out_file.exists():
            return None
        with open(out_file) as f:
            return json.load(f)
    except Exception:
        return None


def run_strategy_sweep(
    pred_files: Dict[str, Path],
    strategies: List[Strategy],
    workers: int = 8,
) -> Dict[str, Dict[str, Dict]]:
    """Run all strategies across all days in parallel."""
    sim_out = RESULTS_DIR / f'sim_{_ts}'
    sim_out.mkdir(parents=True, exist_ok=True)

    jobs = []
    for date_str, pred_file in sorted(pred_files.items()):
        for strategy in strategies:
            jobs.append({
                'date': date_str,
                'pred_file': pred_file,
                'strategy': strategy,
            })

    log.info(f"\nRunning {len(jobs)} sim jobs ({workers} workers)")
    log.info(f"  Days: {len(pred_files)}, Strategies: {len(strategies)}")

    results = {}
    done = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for job in jobs:
            future = executor.submit(
                run_fill_sim,
                job['date'], job['pred_file'], job['strategy'], sim_out,
            )
            futures[future] = job

        for future in as_completed(futures):
            done += 1
            job = futures[future]
            try:
                result = future.result()
                if result:
                    label = job['strategy'].label
                    if label not in results:
                        results[label] = {}
                    results[label][job['date']] = result
            except Exception as e:
                pass

            if done % 50 == 0 or done == len(jobs):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                remaining = (len(jobs) - done) / max(rate, 0.01)
                log.info(f"  [{done}/{len(jobs)}] {rate:.1f} jobs/s, "
                         f"~{remaining:.0f}s remaining")

    elapsed = time.time() - t0
    log.info(f"Sweep done: {done} jobs in {elapsed:.1f}s")
    return results


# ============================================================
# Group 2: Simulated PatchTST DA Filter (Monte Carlo)
# ============================================================

def simulate_patchtst_filter(
    sim_results: Dict[str, Dict[str, Dict]],
    da_accuracy: float = 0.65,
    conservative_factor: float = 0.5,
    n_simulations: int = 100,
) -> Dict[str, Dict]:
    """Simulate PatchTST DA filter effect on existing strategy results.

    Logic: PatchTST has DA_1s=65%. We simulate what happens if we could
    filter OUT (1-DA)% of trades that would have been losers.
    Conservative: only filter 50% of what DA implies (account for correlation).

    For each strategy, Monte Carlo 100 runs:
    - For each losing trade, with prob = DA * conservative_factor, remove it
    - Report median and 95% CI of resulting P&L, Sortino, etc.
    """
    log.info(f"\n{'='*60}")
    log.info(f"  Simulating PatchTST DA Filter (Monte Carlo)")
    log.info(f"  DA accuracy: {da_accuracy:.1%}, conservative factor: {conservative_factor}")
    log.info(f"  Simulations: {n_simulations}")
    log.info(f"{'='*60}")

    # Effective filter probability: probability of correctly filtering a losing trade
    filter_prob = da_accuracy * conservative_factor

    mc_results = {}

    for label, date_results in sim_results.items():
        all_trades = []
        for date_str, res in sorted(date_results.items()):
            if 'trades' not in res:
                continue
            for trade in res['trades']:
                tc = dict(trade)
                tc['date'] = date_str
                all_trades.append(tc)

        if len(all_trades) < 3:
            continue

        # Monte Carlo
        total_pnls = []
        win_rates = []
        sortinos = []
        profit_factors = []
        trade_counts = []

        rng = np.random.RandomState(42)

        for sim in range(n_simulations):
            filtered_trades = []
            for trade in all_trades:
                pnl = trade.get('pnl_dollars', 0)
                if pnl < 0:
                    # DA might filter this losing trade
                    if rng.random() < filter_prob:
                        continue  # Filtered out
                filtered_trades.append(trade)

            if not filtered_trades:
                continue

            pnls = np.array([t.get('pnl_dollars', 0) for t in filtered_trades])
            total_pnls.append(float(pnls.sum()))
            win_rates.append(float(np.mean(pnls > 0)))
            trade_counts.append(len(filtered_trades))

            gross_profit = float(sum(p for p in pnls if p > 0))
            gross_loss = abs(float(sum(p for p in pnls if p < 0)))
            profit_factors.append(gross_profit / max(gross_loss, 0.01))

            # Daily P&L for Sortino
            daily_pnl = defaultdict(float)
            for t in filtered_trades:
                daily_pnl[t.get('date', 'unknown')] += t.get('pnl_dollars', 0)
            daily_vals = list(daily_pnl.values())
            if len(daily_vals) > 1:
                avg_daily = np.mean(daily_vals)
                downside = [min(0, x) for x in daily_vals]
                downside_std = np.std(downside)
                sortinos.append((avg_daily / max(downside_std, 1e-8)) * np.sqrt(252))
            else:
                sortinos.append(0.0)

        if not total_pnls:
            continue

        mc_results[label] = {
            'original_trades': len(all_trades),
            'original_pnl': sum(t.get('pnl_dollars', 0) for t in all_trades),
            'da_filtered': {
                'pnl_median': round(float(np.median(total_pnls)), 2),
                'pnl_p5': round(float(np.percentile(total_pnls, 5)), 2),
                'pnl_p95': round(float(np.percentile(total_pnls, 95)), 2),
                'win_rate_median': round(float(np.median(win_rates)), 4),
                'sortino_median': round(float(np.median(sortinos)), 2),
                'sortino_p5': round(float(np.percentile(sortinos, 5)), 2),
                'sortino_p95': round(float(np.percentile(sortinos, 95)), 2),
                'pf_median': round(float(np.median(profit_factors)), 2),
                'trades_median': int(np.median(trade_counts)),
                'pnl_improvement': round(
                    float(np.median(total_pnls)) -
                    sum(t.get('pnl_dollars', 0) for t in all_trades), 2
                ),
            },
        }

    return mc_results


# ============================================================
# Analysis & Reporting
# ============================================================

def aggregate_results(
    results: Dict[str, Dict[str, Dict]],
    strategy_map: Dict[str, Strategy],
) -> List[Dict]:
    """Aggregate per-day sim results into per-strategy summaries."""
    summaries = []

    for label, date_results in results.items():
        total_pnl = 0.0
        total_trades = 0
        total_signals = 0
        total_filled = 0
        total_wins = 0
        daily_pnls = []
        all_trade_pnls = []
        dates = []
        fill_times = []

        for date_str, res in sorted(date_results.items()):
            dates.append(date_str)
            day_pnl = res.get('total_pnl_dollars', 0)
            total_pnl += day_pnl
            total_trades += res.get('total_trades', 0)
            total_signals += res.get('total_signals', 0)
            total_filled += res.get('total_filled', 0)
            daily_pnls.append(day_pnl)

            if 'trades' in res:
                for trade in res['trades']:
                    pnl = trade.get('pnl_dollars', 0)
                    all_trade_pnls.append(pnl)
                    if pnl > 0:
                        total_wins += 1
                    if 'time_to_fill_ms' in trade:
                        fill_times.append(trade['time_to_fill_ms'])

        n_days = len(date_results)
        if n_days == 0 or total_trades == 0:
            continue

        win_rate = total_wins / max(total_trades, 1)
        fill_rate = total_filled / max(total_signals, 1)
        avg_daily = np.mean(daily_pnls) if daily_pnls else 0

        # Sortino
        if len(daily_pnls) > 1:
            downside = [min(0, x) for x in daily_pnls]
            downside_std = np.std(downside)
            sortino = (avg_daily / max(downside_std, 1e-8)) * np.sqrt(252)
        else:
            sortino = 0.0

        # Profit factor
        gross_profit = sum(p for p in all_trade_pnls if p > 0)
        gross_loss = abs(sum(p for p in all_trade_pnls if p < 0))
        profit_factor = gross_profit / max(gross_loss, 0.01)

        avg_trade_pnl = np.mean(all_trade_pnls) if all_trade_pnls else 0
        avg_fill_time = np.mean(fill_times) if fill_times else 0

        # Max drawdown
        cum = np.cumsum(daily_pnls) if daily_pnls else np.array([0])
        peak = np.maximum.accumulate(cum)
        max_dd = abs(float((cum - peak).min())) if len(cum) > 0 else 0

        strategy = strategy_map.get(label)
        group = strategy.group if strategy else 'unknown'
        description = strategy.description if strategy else label

        summaries.append({
            'label': label,
            'description': description,
            'group': group,
            'total_pnl': round(total_pnl, 2),
            'n_days': n_days,
            'n_trades': total_trades,
            'n_signals': total_signals,
            'trades_per_day': round(total_trades / max(n_days, 1), 1),
            'fill_rate': round(fill_rate, 4),
            'win_rate': round(win_rate, 4),
            'sortino': round(sortino, 2),
            'profit_factor': round(profit_factor, 2),
            'avg_daily_pnl': round(avg_daily, 2),
            'avg_trade_pnl': round(avg_trade_pnl, 2),
            'avg_trade_ticks': round(avg_trade_pnl / TICK_VALUE, 3),
            'max_dd': round(max_dd, 2),
            'avg_fill_time_ms': round(avg_fill_time, 1),
            'daily_pnls': {d: round(p, 2) for d, p in zip(dates, daily_pnls)},
        })

    summaries.sort(key=lambda x: x['sortino'], reverse=True)
    return summaries


def print_results_table(summaries: List[Dict], title: str = "Results",
                        max_rows: int = 50):
    """Print formatted results table."""
    log.info(f"\n{'='*160}")
    log.info(f" {title}")
    log.info(f"{'='*160}")

    if not summaries:
        log.info("  No results.")
        return

    header = (
        f"{'Strategy':<40} "
        f"{'Group':<12} "
        f"{'Total P&L':>10} "
        f"{'Trades':>7} "
        f"{'T/Day':>6} "
        f"{'FillR':>6} "
        f"{'WinR':>6} "
        f"{'AvgTrd':>8} "
        f"{'Sortino':>8} "
        f"{'PF':>5} "
        f"{'MaxDD':>8} "
        f"{'FillMs':>7}"
    )
    log.info(header)
    log.info("-" * 160)

    for s in summaries[:max_rows]:
        pnl_marker = '+' if s['total_pnl'] > 0 else ' '
        line = (
            f"{s['label']:<40} "
            f"{s['group']:<12} "
            f"{pnl_marker}${abs(s['total_pnl']):>8,.0f} "
            f"{s['n_trades']:>7} "
            f"{s['trades_per_day']:>5.1f} "
            f"{s['fill_rate']:>5.1%} "
            f"{s['win_rate']:>5.1%} "
            f"${s['avg_trade_pnl']:>7.2f} "
            f"{s['sortino']:>8.2f} "
            f"{s['profit_factor']:>5.2f} "
            f"${s['max_dd']:>7,.0f} "
            f"{s['avg_fill_time_ms']:>6.0f}ms"
        )
        log.info(line)

    n_days = summaries[0]['n_days'] if summaries else 0
    log.info(f"\n  ES Futures | Tick=$12.50 | Commission=$4.70 RT | {n_days} OOT days")


def print_profitable_strategies(summaries: List[Dict]):
    """Print only profitable strategies with details."""
    profitable = [s for s in summaries if s['total_pnl'] > 0]

    log.info(f"\n{'='*80}")
    log.info(f"  PROFITABLE STRATEGIES: {len(profitable)} / {len(summaries)}")
    log.info(f"{'='*80}")

    if not profitable:
        log.info("  NONE profitable on real fills across all 4 days.")
        log.info("  Showing top 10 by Sortino (least negative):")
        print_results_table(summaries[:10], "Top 10 by Sortino (all negative)")
        return

    for rank, s in enumerate(profitable):
        log.info(f"\n  #{rank+1} {s['label']}")
        log.info(f"  {s['description']}")
        log.info(f"  Group:         {s['group']}")
        log.info(f"  Total P&L:     ${s['total_pnl']:,.2f}")
        log.info(f"  Trades:        {s['n_trades']} ({s['trades_per_day']:.1f}/day)")
        log.info(f"  Fill rate:     {s['fill_rate']:.1%}")
        log.info(f"  Win rate:      {s['win_rate']:.1%}")
        log.info(f"  Avg trade:     ${s['avg_trade_pnl']:.2f} ({s['avg_trade_ticks']:.2f} ticks)")
        log.info(f"  Sortino:       {s['sortino']:.2f}")
        log.info(f"  Profit factor: {s['profit_factor']:.2f}")
        log.info(f"  Max DD:        ${s['max_dd']:,.2f}")
        log.info(f"  Daily P&L:")
        for date, pnl in sorted(s['daily_pnls'].items()):
            marker = '+' if pnl >= 0 else '-'
            log.info(f"    {date}: {marker}${abs(pnl):,.2f}")


def print_monte_carlo_results(mc_results: Dict[str, Dict], summaries: List[Dict]):
    """Print Monte Carlo PatchTST DA filter simulation results."""
    # Only show for strategies that were close to profitable or profitable
    interesting = []
    summary_map = {s['label']: s for s in summaries}

    for label, mc in mc_results.items():
        s = summary_map.get(label)
        if not s:
            continue
        # Show if original was somewhat close to profitable or MC makes it profitable
        if (mc['da_filtered']['pnl_median'] > 0 or
            s['total_pnl'] > -100 or
            mc['da_filtered']['sortino_median'] > 0):
            interesting.append((label, mc, s))

    interesting.sort(key=lambda x: x[1]['da_filtered']['sortino_median'], reverse=True)

    log.info(f"\n{'='*120}")
    log.info(f"  SIMULATED PatchTST DA FILTER — Monte Carlo Results")
    log.info(f"  (Conservative: 50% of DA_1s=65% filter rate on losing trades)")
    log.info(f"{'='*120}")

    if not interesting:
        log.info("  No strategies improved enough to be interesting.")
        return

    header = (
        f"{'Strategy':<40} "
        f"{'Orig P&L':>10} "
        f"{'MC P&L':>10} "
        f"{'P&L 5%':>10} "
        f"{'P&L 95%':>10} "
        f"{'MC WinR':>7} "
        f"{'MC Sort':>8} "
        f"{'MC PF':>6} "
        f"{'Trades':>7} "
        f"{'Improve':>10}"
    )
    log.info(header)
    log.info("-" * 120)

    for label, mc, s in interesting[:30]:
        df = mc['da_filtered']
        line = (
            f"{label:<40} "
            f"${mc['original_pnl']:>9,.0f} "
            f"${df['pnl_median']:>9,.0f} "
            f"${df['pnl_p5']:>9,.0f} "
            f"${df['pnl_p95']:>9,.0f} "
            f"{df['win_rate_median']:>6.1%} "
            f"{df['sortino_median']:>8.2f} "
            f"{df['pf_median']:>5.2f} "
            f"{df['trades_median']:>7} "
            f"${df['pnl_improvement']:>9,.0f}"
        )
        log.info(line)


def select_paper_trading_candidates(
    summaries: List[Dict],
    mc_results: Dict[str, Dict],
) -> List[Dict]:
    """Select 3-5 best strategies for Monday paper trading."""
    candidates = []

    # Criteria 1: Profitable on real fills
    profitable = [s for s in summaries if s['total_pnl'] > 0]
    for s in profitable[:3]:
        candidates.append({
            **s,
            'selection_reason': 'Profitable on real fills',
            'mc_improvement': mc_results.get(s['label'], {}).get(
                'da_filtered', {}).get('pnl_improvement', 0),
        })

    # Criteria 2: Best Sortino even if slightly negative, with >=5 trades
    for s in summaries:
        if s['label'] in [c['label'] for c in candidates]:
            continue
        if s['n_trades'] >= 5 and s['sortino'] > 0:
            candidates.append({
                **s,
                'selection_reason': 'Positive Sortino with enough trades',
                'mc_improvement': mc_results.get(s['label'], {}).get(
                    'da_filtered', {}).get('pnl_improvement', 0),
            })
            if len(candidates) >= 5:
                break

    # Criteria 3: MC simulation shows strong improvement
    if len(candidates) < 5:
        mc_candidates = []
        for label, mc in mc_results.items():
            if label in [c['label'] for c in candidates]:
                continue
            df = mc['da_filtered']
            if df['pnl_median'] > 0 and df['sortino_median'] > 0:
                s = next((x for x in summaries if x['label'] == label), None)
                if s and s['n_trades'] >= 5:
                    mc_candidates.append((label, mc, s))

        mc_candidates.sort(key=lambda x: x[1]['da_filtered']['sortino_median'],
                          reverse=True)
        for label, mc, s in mc_candidates[:5 - len(candidates)]:
            candidates.append({
                **s,
                'selection_reason': f"MC shows profitable with DA filter (median ${mc['da_filtered']['pnl_median']:,.0f})",
                'mc_improvement': mc['da_filtered']['pnl_improvement'],
            })

    # If still not enough, take best Sortinos
    if len(candidates) < 3:
        for s in summaries:
            if s['label'] in [c['label'] for c in candidates]:
                continue
            if s['n_trades'] >= 3:
                candidates.append({
                    **s,
                    'selection_reason': 'Best available Sortino',
                    'mc_improvement': mc_results.get(s['label'], {}).get(
                        'da_filtered', {}).get('pnl_improvement', 0),
                })
                if len(candidates) >= 5:
                    break

    return candidates[:5]


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Strategy Refinement v3')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--groups', type=str, default='all',
                        help='Strategy groups: G1_trailing,G1_bracket,G1_trail_sl,'
                             'G1_ratchet,G3_time,G4_sizing,baseline or all')
    parser.add_argument('--clear-cache', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--mc-sims', type=int, default=100,
                        help='Monte Carlo simulations for PatchTST filter')
    args = parser.parse_args()

    log.info("=" * 80)
    log.info("STRATEGY REFINEMENT v3 — CNN-Mamba v2 Execution Optimization")
    log.info("=" * 80)
    log.info(f"  Goal: Find 3-5 strategies for Monday paper trading")
    log.info(f"  Model: CNN-Mamba v2 (4 folds, Feb 23-26)")
    log.info(f"  Known: z>=5.0 ultra-selective was only profitable strategy")
    log.info(f"  Focus: Smart exits (trailing/bracket) + time filters + DA simulation")
    log.info(f"  Workers: {args.workers}")
    log.info("=" * 80)

    # Clear cache if requested
    if args.clear_cache:
        import shutil
        for d in [PRED_CACHE_DIR, RESULTS_DIR]:
            if d.exists():
                shutil.rmtree(d)
                d.mkdir(parents=True, exist_ok=True)
        log.info("  Cache cleared")

    # Build strategies
    all_strategies = build_all_strategies()

    # Filter by group
    if args.groups != 'all':
        groups = set(args.groups.split(','))
        all_strategies = [s for s in all_strategies if s.group in groups]
        log.info(f"  Filtered to {len(all_strategies)} strategies in groups: {groups}")

    strategy_map = {s.label: s for s in all_strategies}

    if args.dry_run:
        log.info(f"\n--- DRY RUN: {len(all_strategies)} strategies ---")
        for group in sorted(set(s.group for s in all_strategies)):
            group_strats = [s for s in all_strategies if s.group == group]
            log.info(f"\n  {group}: {len(group_strats)} strategies")
            for s in group_strats[:5]:
                log.info(f"    {s.label:<45} {s.description}")
            if len(group_strats) > 5:
                log.info(f"    ... and {len(group_strats)-5} more")
        return

    # Check binary
    if not BINARY.exists():
        log.error(f"fill_sim_cli not found: {BINARY}")
        sys.exit(1)

    # Prepare predictions
    pred_files = prepare_all_predictions()
    if not pred_files:
        log.error("No prediction files prepared.")
        sys.exit(1)

    # ── Phase 1: Run all strategies through fill_sim_cli ──
    log.info(f"\n{'='*60}")
    log.info(f"  PHASE 1: Fill Simulation Sweep")
    log.info(f"  {len(all_strategies)} strategies x {len(pred_files)} days = "
             f"{len(all_strategies) * len(pred_files)} jobs")
    log.info(f"{'='*60}")

    results = run_strategy_sweep(pred_files, all_strategies, workers=args.workers)

    if not results:
        log.error("No sim results.")
        sys.exit(1)

    # ── Aggregate ──
    summaries = aggregate_results(results, strategy_map)

    # ── Phase 2: Monte Carlo PatchTST DA filter simulation ──
    log.info(f"\n{'='*60}")
    log.info(f"  PHASE 2: Monte Carlo PatchTST DA Filter Simulation")
    log.info(f"{'='*60}")

    mc_results = simulate_patchtst_filter(
        results,
        da_accuracy=0.65,
        conservative_factor=0.5,
        n_simulations=args.mc_sims,
    )

    # ── Reporting ──

    # By group
    for group in sorted(set(s['group'] for s in summaries)):
        group_summaries = [s for s in summaries if s['group'] == group]
        if group_summaries:
            print_results_table(
                group_summaries[:20],
                f"Group: {group} ({len(group_summaries)} strategies)"
            )

    # Overall top 30
    print_results_table(summaries[:30], "ALL STRATEGIES — Top 30 by Sortino")

    # Profitable strategies detail
    print_profitable_strategies(summaries)

    # Monte Carlo results
    print_monte_carlo_results(mc_results, summaries)

    # ── Phase 3: Select paper trading candidates ──
    log.info(f"\n{'='*80}")
    log.info(f"  PHASE 3: Paper Trading Candidate Selection")
    log.info(f"{'='*80}")

    candidates = select_paper_trading_candidates(summaries, mc_results)

    for i, c in enumerate(candidates):
        log.info(f"\n  CANDIDATE #{i+1}: {c['label']}")
        log.info(f"  Reason: {c['selection_reason']}")
        log.info(f"  Description: {c['description']}")
        log.info(f"  Total P&L:     ${c['total_pnl']:,.2f}")
        log.info(f"  Trades:        {c['n_trades']} ({c['trades_per_day']:.1f}/day)")
        log.info(f"  Win rate:      {c['win_rate']:.1%}")
        log.info(f"  Sortino:       {c['sortino']:.2f}")
        log.info(f"  Profit factor: {c['profit_factor']:.2f}")
        if c.get('mc_improvement'):
            log.info(f"  MC DA improvement: ${c['mc_improvement']:,.2f}")
        log.info(f"  Daily P&L:")
        for date, pnl in sorted(c['daily_pnls'].items()):
            marker = '+' if pnl >= 0 else '-'
            log.info(f"    {date}: {marker}${abs(pnl):,.2f}")

    # ── Save results ──
    out_file = RESULTS_DIR / f'refinement_v3_results_{_ts}.json'
    save_data = {
        'timestamp': _ts,
        'instrument': 'ES',
        'tick_value': TICK_VALUE,
        'commission_rt': COMMISSION_RT,
        'n_days': len(pred_files),
        'dates': sorted(pred_files.keys()),
        'n_strategies_tested': len(all_strategies),
        'n_profitable': len([s for s in summaries if s['total_pnl'] > 0]),
        'strategies': [{k: v for k, v in s.items()} for s in summaries],
        'monte_carlo_da_filter': mc_results,
        'paper_trading_candidates': [
            {k: v for k, v in c.items() if k != 'daily_pnls'}
            for c in candidates
        ],
    }
    with open(out_file, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    log.info(f"\n{'='*80}")
    log.info(f"  REFINEMENT v3 COMPLETE")
    log.info(f"{'='*80}")
    log.info(f"  Strategies tested:  {len(all_strategies)}")
    log.info(f"  Profitable:         {len([s for s in summaries if s['total_pnl'] > 0])}")
    log.info(f"  MC-profitable:      {len([l for l, mc in mc_results.items() if mc['da_filtered']['pnl_median'] > 0])}")
    log.info(f"  Paper candidates:   {len(candidates)}")
    if summaries:
        best = summaries[0]
        log.info(f"  Best strategy:      {best['label']} "
                 f"(Sortino={best['sortino']:.2f}, P&L=${best['total_pnl']:,.0f})")
    log.info(f"  Results saved:      {out_file}")
    log.info(f"  Log file:           {_log_file}")
    log.info(f"{'='*80}")


if __name__ == '__main__':
    main()
