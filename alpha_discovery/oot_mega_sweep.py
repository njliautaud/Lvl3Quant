#!/usr/bin/env python3
"""
OOT Mega Sweep — Comprehensive Out-of-Time Validation for CNN/GNN/Ensemble
================================================================================
Runs 9 sweep tiers through the Rust MBO fill simulator to validate predictions
on new Dec-Mar data (out-of-time, never seen during training).

Sweeps:
  1. CNN Chase Mode (81 configs)         — vol × conv × chase × hold
  2. CNN Passive Limit (27 configs)      — vol × conv × hold
  3. CNN Entry Timing (20 configs)       — time × day gates on best chase
  4. CNN Exit Strategies (50 configs)    — fixed/trailing/target combos
  5. CNN Signal Threshold (42 configs)   — conv × vol sensitivity
  6. CNN Long/Short Asymmetry (18 configs) — side × thresholds
  7. GNN Chase Mode (81 configs)         — same grid as S1, GNN predictions
  8. GNN Passive Limit (27 configs)      — same grid as S2, GNN predictions
  9. CNN+GNN Ensemble (27 configs)       — agreement modes × conv × vol

Usage:
    python alpha_discovery/oot_mega_sweep.py --data-dir /path/to/oot/mbo
    python alpha_discovery/oot_mega_sweep.py --data-dir ~/Lvl3Quant/mbo_oot --workers 12
    python alpha_discovery/oot_mega_sweep.py --data-dir ~/mbo_oot --sweep 1 --workers 8
    python alpha_discovery/oot_mega_sweep.py --data-dir ~/mbo_oot --sweep 1 --config-range 0-40
    python alpha_discovery/oot_mega_sweep.py --data-dir ~/mbo_oot --sweep 7 --gnn-pred-dir ~/Lvl3Quant/data/processed/gnn_oot_predictions
    python alpha_discovery/oot_mega_sweep.py --data-dir ~/mbo_oot --sweep 9 --gnn-pred-dir ~/gnn_preds --ensemble-pred-dir ~/ensemble_preds
"""

import sys
import json
import time
import os
import logging
import argparse
import subprocess
import hashlib
import numpy as np
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product

# ─── Paths ───────────────────────────────────────────────────────────────────
LVL3_ROOT = Path(__file__).resolve().parent.parent

import platform
_bin_name = 'fill_sim_cli.exe' if platform.system() == 'Windows' else 'fill_sim_cli'
BINARY = LVL3_ROOT / 'rust_cache_builder' / 'target' / 'release' / _bin_name

RESULTS_DIR = LVL3_ROOT / 'alpha_discovery' / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TICK_VALUE = 12.50
COMMISSION_TICKS = 0.24  # per side, 0.48 RT

_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
_log_file = str(RESULTS_DIR / f'oot_mega_sweep_{_ts}.log')

log = logging.getLogger('oot_mega_sweep')
log.setLevel(logging.INFO)
_fh = logging.FileHandler(_log_file, mode='w', encoding='utf-8')
_fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
log.addHandler(_fh)
_ch = logging.StreamHandler(sys.stdout)
_ch.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
log.addHandler(_ch)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# ─── Sweep Config Generation ────────────────────────────────────────────────

def generate_sweep_1_chase():
    """SWEEP 1: Chase Mode — 81 configs.
    vol_gates × conv_thresholds × chase_configs × hold_periods
    3 × 3 × 3 × 3 = 81
    """
    configs = []
    vol_gates = [50, 70, 80]
    conv_thresholds = [1.5, 2.0, 2.5]
    chase_configs = [(1, 3), (2, 5), (3, 7)]  # (max_ticks, max_reprices)
    hold_periods_min = [10, 20, 30]

    for vol, conv, (chase_ticks, chase_reprices), hold_min in product(
        vol_gates, conv_thresholds, chase_configs, hold_periods_min
    ):
        hold_ms = hold_min * 60 * 1000
        label = f"S1_chase_v{vol}_c{conv}_ct{chase_ticks}r{chase_reprices}_h{hold_min}m"
        configs.append({
            'sweep': 1,
            'label': label,
            'vol_gate': vol,
            'signal_threshold': conv,
            'hold_ms': hold_ms,
            'latency_ms': 0,
            'entry_mode': 'chase',
            'chase_max_ticks': chase_ticks,
            'chase_max_reprices': chase_reprices,
            'chase_force_cross': False,
            'chase_interval_ms': 100,
            'trailing_ticks': None,
            'take_profit_ticks': None,
            'prime_hours': False,
            'signal_flip_exit': False,
            'side_filter': 'both',
            'model_type': 'cnn',
        })
    return configs


def generate_sweep_2_passive():
    """SWEEP 2: Passive Limit — 27 configs.
    vol_gates × conv_thresholds × hold_periods
    3 × 3 × 3 = 27
    """
    configs = []
    vol_gates = [50, 70, 80]
    conv_thresholds = [1.5, 2.0, 2.5]
    hold_periods_min = [10, 20, 30]

    for vol, conv, hold_min in product(vol_gates, conv_thresholds, hold_periods_min):
        hold_ms = hold_min * 60 * 1000
        label = f"S2_passive_v{vol}_c{conv}_h{hold_min}m"
        configs.append({
            'sweep': 2,
            'label': label,
            'vol_gate': vol,
            'signal_threshold': conv,
            'hold_ms': hold_ms,
            'latency_ms': 0,
            'entry_mode': 'passive',
            'chase_max_ticks': None,
            'chase_max_reprices': None,
            'chase_force_cross': False,
            'chase_interval_ms': None,
            'trailing_ticks': None,
            'take_profit_ticks': None,
            'prime_hours': False,
            'signal_flip_exit': False,
            'side_filter': 'both',
            'model_type': 'cnn',
        })
    return configs


def generate_sweep_3_entry_timing():
    """SWEEP 3: Entry Timing — 20 configs.
    Uses best chase config from IS. Tests time and day gates.
    time_gates × day_gates = 5 × 4 = 20
    (day_gates reduced: all_days, mon_wed_fri, tue_thu, fri_only)
    """
    configs = []
    # Best IS chase: vol70, conv2.5, chase 1t/3r, 30min hold
    base_vol = 70
    base_conv = 2.5
    base_chase_ticks = 1
    base_chase_reprices = 3
    base_hold_ms = 1800000

    # Time gates: we'll implement via prime_hours or post-filter
    # RTH_full = no filter, first_2hr = 9:30-11:30, last_2hr = 14:00-16:00,
    # midday = 11:00-14:00, overnight = not applicable for MBO sim (RTH only)
    time_gates = ['rth_full', 'first_2hr', 'last_2hr', 'midday', 'prime_hours']
    day_gates = ['all_days', 'mon_wed_fri', 'tue_thu', 'fri_only']

    for tg, dg in product(time_gates, day_gates):
        label = f"S3_timing_v{base_vol}_c{base_conv}_{tg}_{dg}"
        configs.append({
            'sweep': 3,
            'label': label,
            'vol_gate': base_vol,
            'signal_threshold': base_conv,
            'hold_ms': base_hold_ms,
            'latency_ms': 0,
            'entry_mode': 'chase',
            'chase_max_ticks': base_chase_ticks,
            'chase_max_reprices': base_chase_reprices,
            'chase_force_cross': False,
            'chase_interval_ms': 100,
            'trailing_ticks': None,
            'take_profit_ticks': None,
            'prime_hours': (tg == 'prime_hours'),
            'signal_flip_exit': False,
            'side_filter': 'both',
            'time_gate': tg,
            'day_gate': dg,
            'model_type': 'cnn',
        })
    return configs


def generate_sweep_4_exit():
    """SWEEP 4: Exit Strategies — 50 configs.
    Fixed hold (7) + trailing stops (5) + profit targets (4) +
    trail+target combos (5x4=20) + signal_flip (2) + combined special (12)
    """
    configs = []
    base_vol = 70
    base_conv = 2.5
    base_chase_ticks = 1
    base_chase_reprices = 3

    base = {
        'sweep': 4,
        'vol_gate': base_vol,
        'signal_threshold': base_conv,
        'latency_ms': 0,
        'entry_mode': 'chase',
        'chase_max_ticks': base_chase_ticks,
        'chase_max_reprices': base_chase_reprices,
        'chase_force_cross': False,
        'chase_interval_ms': 100,
        'prime_hours': False,
        'side_filter': 'both',
        'time_gate': None,
        'day_gate': None,
        'model_type': 'cnn',
    }

    # Fixed hold periods: 5, 10, 15, 20, 30, 45, 60 min
    for hold_min in [5, 10, 15, 20, 30, 45, 60]:
        c = dict(base)
        c['label'] = f"S4_exit_hold{hold_min}m"
        c['hold_ms'] = hold_min * 60 * 1000
        c['trailing_ticks'] = None
        c['take_profit_ticks'] = None
        c['signal_flip_exit'] = False
        configs.append(c)

    # Trailing stops only (with 30min max hold)
    for trail in [3, 5, 8, 10, 15]:
        c = dict(base)
        c['label'] = f"S4_exit_trail{trail}t_h30m"
        c['hold_ms'] = 1800000
        c['trailing_ticks'] = trail
        c['take_profit_ticks'] = None
        c['signal_flip_exit'] = False
        configs.append(c)

    # Profit targets only (with 30min max hold)
    for target in [5, 10, 15, 20]:
        c = dict(base)
        c['label'] = f"S4_exit_target{target}t_h30m"
        c['hold_ms'] = 1800000
        c['trailing_ticks'] = None
        c['take_profit_ticks'] = target
        c['signal_flip_exit'] = False
        configs.append(c)

    # Trail + target combos (top combinations)
    trail_target_combos = [
        (3, 5), (3, 10), (5, 10), (5, 15), (5, 20),
        (8, 10), (8, 15), (8, 20), (10, 15), (10, 20),
        (3, 15), (3, 20), (5, 5), (8, 5), (10, 10),
        (15, 15), (15, 20), (10, 5), (15, 10), (15, 5),
    ]
    for trail, target in trail_target_combos:
        c = dict(base)
        c['label'] = f"S4_exit_trail{trail}t_target{target}t_h30m"
        c['hold_ms'] = 1800000
        c['trailing_ticks'] = trail
        c['take_profit_ticks'] = target
        c['signal_flip_exit'] = False
        configs.append(c)

    # Signal flip exit
    for hold_min in [30, 60]:
        c = dict(base)
        c['label'] = f"S4_exit_sigflip_h{hold_min}m"
        c['hold_ms'] = hold_min * 60 * 1000
        c['trailing_ticks'] = None
        c['take_profit_ticks'] = None
        c['signal_flip_exit'] = True
        configs.append(c)

    return configs


def generate_sweep_5_signal_sensitivity():
    """SWEEP 5: Signal Threshold Sensitivity — 42 configs.
    conv × vol = 7 × 6 = 42
    """
    configs = []
    conv_values = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    vol_values = [30, 50, 60, 70, 80, 90]

    for conv, vol in product(conv_values, vol_values):
        label = f"S5_signal_v{vol}_c{conv}"
        configs.append({
            'sweep': 5,
            'label': label,
            'vol_gate': vol,
            'signal_threshold': conv,
            'hold_ms': 1800000,
            'latency_ms': 0,
            'entry_mode': 'chase',
            'chase_max_ticks': 1,
            'chase_max_reprices': 3,
            'chase_force_cross': False,
            'chase_interval_ms': 100,
            'trailing_ticks': None,
            'take_profit_ticks': None,
            'prime_hours': False,
            'signal_flip_exit': False,
            'side_filter': 'both',
            'model_type': 'cnn',
        })
    return configs


def generate_sweep_6_long_short():
    """SWEEP 6: Long/Short Asymmetry — 18 configs.
    side × conv_threshold = 3 × 6 = 18
    """
    configs = []
    sides = ['long_only', 'short_only', 'both']
    conv_values = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

    for side, conv in product(sides, conv_values):
        label = f"S6_asym_{side}_c{conv}"
        configs.append({
            'sweep': 6,
            'label': label,
            'vol_gate': 70,
            'signal_threshold': conv,
            'hold_ms': 1800000,
            'latency_ms': 0,
            'entry_mode': 'chase',
            'chase_max_ticks': 1,
            'chase_max_reprices': 3,
            'chase_force_cross': False,
            'chase_interval_ms': 100,
            'trailing_ticks': None,
            'take_profit_ticks': None,
            'prime_hours': False,
            'signal_flip_exit': False,
            'side_filter': side,
            'model_type': 'cnn',
        })
    return configs


def generate_sweep_7_gnn_chase():
    """SWEEP 7: GNN Chase Mode — 81 configs.
    Same grid as Sweep 1 but using GNN predictions instead of CNN.
    vol_gates x conv_thresholds x chase_configs x hold_periods
    3 x 3 x 3 x 3 = 81
    """
    configs = []
    vol_gates = [50, 70, 80]
    conv_thresholds = [1.5, 2.0, 2.5]
    chase_configs = [(1, 3), (2, 5), (3, 7)]  # (max_ticks, max_reprices)
    hold_periods_min = [10, 20, 30]

    for vol, conv, (chase_ticks, chase_reprices), hold_min in product(
        vol_gates, conv_thresholds, chase_configs, hold_periods_min
    ):
        hold_ms = hold_min * 60 * 1000
        label = f"S7_gnn_chase_v{vol}_c{conv}_ct{chase_ticks}r{chase_reprices}_h{hold_min}m"
        configs.append({
            'sweep': 7,
            'label': label,
            'vol_gate': vol,
            'signal_threshold': conv,
            'hold_ms': hold_ms,
            'latency_ms': 0,
            'entry_mode': 'chase',
            'chase_max_ticks': chase_ticks,
            'chase_max_reprices': chase_reprices,
            'chase_force_cross': False,
            'chase_interval_ms': 100,
            'trailing_ticks': None,
            'take_profit_ticks': None,
            'prime_hours': False,
            'signal_flip_exit': False,
            'side_filter': 'both',
            'model_type': 'gnn',
        })
    return configs


def generate_sweep_8_gnn_passive():
    """SWEEP 8: GNN Passive Limit — 27 configs.
    Same as Sweep 2 but using GNN predictions.
    vol_gates x conv_thresholds x hold_periods
    3 x 3 x 3 = 27
    """
    configs = []
    vol_gates = [50, 70, 80]
    conv_thresholds = [1.5, 2.0, 2.5]
    hold_periods_min = [10, 20, 30]

    for vol, conv, hold_min in product(vol_gates, conv_thresholds, hold_periods_min):
        hold_ms = hold_min * 60 * 1000
        label = f"S8_gnn_passive_v{vol}_c{conv}_h{hold_min}m"
        configs.append({
            'sweep': 8,
            'label': label,
            'vol_gate': vol,
            'signal_threshold': conv,
            'hold_ms': hold_ms,
            'latency_ms': 0,
            'entry_mode': 'passive',
            'chase_max_ticks': None,
            'chase_max_reprices': None,
            'chase_force_cross': False,
            'chase_interval_ms': None,
            'trailing_ticks': None,
            'take_profit_ticks': None,
            'prime_hours': False,
            'signal_flip_exit': False,
            'side_filter': 'both',
            'model_type': 'gnn',
        })
    return configs


def generate_sweep_9_ensemble():
    """SWEEP 9: CNN+GNN Ensemble — 27 configs.
    Average CNN and GNN prediction scores. Trade when both models agree
    (both above threshold), or use average/max signal.

    Uses best chase config from Sweep 1 as base (1t/3r, 30min hold).
    Varies: agreement_mode, conv thresholds, vol gates

    agreement_modes:
        both_agree:  only trade if BOTH |cnn| > thresh AND |gnn| > thresh
        average:     trade on mean(cnn, gnn) signal, threshold applied to average
        max_signal:  trade on max(|cnn|, |gnn|), takes the stronger signal's direction

    3 agreement_modes x 3 conv_thresholds x 3 vol_gates = 27 configs
    """
    configs = []
    agreement_modes = ['both_agree', 'average', 'max_signal']
    conv_thresholds = [1.5, 2.0, 2.5]
    vol_gates = [50, 70, 80]

    for agree_mode, conv, vol in product(agreement_modes, conv_thresholds, vol_gates):
        label = f"S9_ensemble_{agree_mode}_v{vol}_c{conv}"
        configs.append({
            'sweep': 9,
            'label': label,
            'vol_gate': vol,
            'signal_threshold': conv,
            'hold_ms': 1800000,  # 30min (best from IS)
            'latency_ms': 0,
            'entry_mode': 'chase',
            'chase_max_ticks': 1,
            'chase_max_reprices': 3,
            'chase_force_cross': False,
            'chase_interval_ms': 100,
            'trailing_ticks': None,
            'take_profit_ticks': None,
            'prime_hours': False,
            'signal_flip_exit': False,
            'side_filter': 'both',
            'model_type': 'ensemble',
            'agreement_mode': agree_mode,
        })
    return configs


def get_all_configs(sweep_filter=None):
    """Generate all configs, optionally filtered by sweep number."""
    generators = {
        1: generate_sweep_1_chase,
        2: generate_sweep_2_passive,
        3: generate_sweep_3_entry_timing,
        4: generate_sweep_4_exit,
        5: generate_sweep_5_signal_sensitivity,
        6: generate_sweep_6_long_short,
        7: generate_sweep_7_gnn_chase,
        8: generate_sweep_8_gnn_passive,
        9: generate_sweep_9_ensemble,
    }

    configs = []
    for sweep_num, gen_fn in generators.items():
        if sweep_filter is not None and sweep_num != sweep_filter:
            continue
        sweep_configs = gen_fn()
        configs.extend(sweep_configs)
        log.info(f"  Sweep {sweep_num}: {len(sweep_configs)} configs")

    return configs


# ─── Prediction File Discovery ──────────────────────────────────────────────

def discover_prediction_files(pred_dir):
    """Find all prediction .npz files (CNN or GNN — same format).
    Expected naming: YYYY-MM-DD_volNN_morning_afternoon.npz
    Returns dict of (date_str, vol_gate) -> Path
    """
    pred_dir = Path(pred_dir)
    if not pred_dir.exists():
        log.error(f"Prediction directory not found: {pred_dir}")
        return {}

    files = {}
    for f in pred_dir.glob('*.npz'):
        stem = f.stem
        # Parse: 2025-08-19_vol50_morning_afternoon
        parts = stem.split('_', 2)
        if len(parts) < 2:
            continue
        date_str = parts[0]
        vol_str = parts[1]
        try:
            vol_gate = int(vol_str.replace('vol', ''))
        except ValueError:
            continue
        files[(date_str, vol_gate)] = f

    return files


def discover_mbo_files(mbo_dir):
    """Find all MBO .dbn or .dbn.zst files.
    Returns dict of date_str -> Path
    """
    mbo_dir = Path(mbo_dir)
    if not mbo_dir.exists():
        log.error(f"MBO directory not found: {mbo_dir}")
        return {}

    files = {}
    for ext in ['*.mbo.dbn.zst', '*.mbo.dbn']:
        for f in mbo_dir.glob(ext):
            # Parse: glbx-mdp3-20250819.mbo.dbn
            stem = f.name.split('.')[0]  # glbx-mdp3-20250819
            nodash = stem.split('-')[-1]  # 20250819
            if len(nodash) == 8:
                date_str = f"{nodash[:4]}-{nodash[4:6]}-{nodash[6:8]}"
                if date_str not in files:  # prefer .zst
                    files[date_str] = f

    return files


# ─── Ensemble Prediction Generation ─────────────────────────────────────────

def generate_ensemble_predictions(cnn_pred_dir, gnn_pred_dir, ensemble_pred_dir, agreement_mode='average'):
    """Generate ensemble prediction files by combining CNN and GNN predictions.

    For each (date, vol_gate) pair where BOTH CNN and GNN predictions exist,
    creates an ensemble prediction file.

    Args:
        cnn_pred_dir: Directory with CNN prediction .npz files
        gnn_pred_dir: Directory with GNN prediction .npz files
        ensemble_pred_dir: Output directory for ensemble predictions
        agreement_mode: How to combine signals ('both_agree', 'average', 'max_signal')

    Returns:
        dict of (date_str, vol_gate) -> Path for generated ensemble files
    """
    cnn_pred_dir = Path(cnn_pred_dir)
    gnn_pred_dir = Path(gnn_pred_dir)
    ensemble_pred_dir = Path(ensemble_pred_dir)
    ensemble_pred_dir.mkdir(parents=True, exist_ok=True)

    cnn_files = discover_prediction_files(cnn_pred_dir)
    gnn_files = discover_prediction_files(gnn_pred_dir)

    if not cnn_files:
        log.error(f"No CNN predictions found in {cnn_pred_dir}")
        return {}
    if not gnn_files:
        log.error(f"No GNN predictions found in {gnn_pred_dir}")
        return {}

    # Find intersection of dates+vol_gates
    common_keys = set(cnn_files.keys()) & set(gnn_files.keys())
    log.info(f"Ensemble: {len(cnn_files)} CNN, {len(gnn_files)} GNN, {len(common_keys)} overlap")

    if not common_keys:
        log.error("No overlapping (date, vol_gate) pairs between CNN and GNN predictions")
        return {}

    ensemble_files = {}
    for key in sorted(common_keys):
        date_str, vol_gate = key
        out_file = ensemble_pred_dir / f"{date_str}_vol{vol_gate}_morning_afternoon.npz"

        # Skip if already generated
        if out_file.exists():
            ensemble_files[key] = out_file
            continue

        try:
            cnn_data = np.load(str(cnn_files[key]))
            gnn_data = np.load(str(gnn_files[key]))

            cnn_preds = cnn_data['predictions']
            gnn_preds = gnn_data['predictions']

            # Align lengths (use shorter)
            min_len = min(len(cnn_preds), len(gnn_preds))
            cnn_preds = cnn_preds[:min_len]
            gnn_preds = gnn_preds[:min_len]

            if agreement_mode == 'both_agree':
                # Both must have same sign and both above threshold (threshold applied later by sim)
                # Here we output the average signal, but zero it where they disagree on direction
                agree_mask = (np.sign(cnn_preds) == np.sign(gnn_preds)) & (cnn_preds != 0) & (gnn_preds != 0)
                ensemble_preds = np.where(agree_mask, (cnn_preds + gnn_preds) / 2.0, 0.0)

            elif agreement_mode == 'average':
                # Simple average of both signals
                ensemble_preds = (cnn_preds + gnn_preds) / 2.0

            elif agreement_mode == 'max_signal':
                # Take the signal with larger absolute value, keeping its sign
                abs_cnn = np.abs(cnn_preds)
                abs_gnn = np.abs(gnn_preds)
                ensemble_preds = np.where(abs_cnn >= abs_gnn, cnn_preds, gnn_preds)

            else:
                log.warning(f"Unknown agreement_mode: {agreement_mode}, using average")
                ensemble_preds = (cnn_preds + gnn_preds) / 2.0

            # Pad back to original CNN length if needed
            if min_len < len(cnn_data['predictions']):
                padded = np.zeros(len(cnn_data['predictions']))
                padded[:min_len] = ensemble_preds
                ensemble_preds = padded

            np.savez_compressed(str(out_file), predictions=ensemble_preds)
            ensemble_files[key] = out_file

        except Exception as e:
            log.warning(f"Failed to generate ensemble for {key}: {e}")

    log.info(f"Generated {len(ensemble_files)} ensemble prediction files in {ensemble_pred_dir}")
    return ensemble_files


# ─── Time/Day Gating (Post-Filter) ──────────────────────────────────────────

def should_include_trade(trade, time_gate=None, day_gate=None):
    """Filter trades by time-of-day and day-of-week gates.
    Uses signal_time_ns from the trade record.
    """
    if time_gate is None and day_gate is None:
        return True

    sig_ns = trade.get('signal_time_ns', 0)
    if sig_ns == 0:
        return True

    # Convert nanosecond epoch to hour (ET approximation: UTC-5 during EST, UTC-4 EDT)
    # MBO timestamps are UTC. ES RTH: 9:30-16:00 ET = 14:30-21:00 UTC (EST)
    # or 13:30-20:00 UTC (EDT). We'll use the signal time modulo day.
    import datetime as dt
    ts = dt.datetime.utcfromtimestamp(sig_ns / 1e9)
    utc_hour = ts.hour + ts.minute / 60.0
    weekday = ts.weekday()  # 0=Mon, 4=Fri

    # Approximate ET (assume EDT for summer, EST for winter)
    # Aug-Nov = EDT (UTC-4), Dec-Mar = EST (UTC-5)
    month = ts.month
    if month >= 3 and month <= 11:
        et_hour = utc_hour - 4
    else:
        et_hour = utc_hour - 5
    if et_hour < 0:
        et_hour += 24

    # Time gates
    if time_gate == 'first_2hr' and not (9.5 <= et_hour < 11.5):
        return False
    elif time_gate == 'last_2hr' and not (14.0 <= et_hour < 16.0):
        return False
    elif time_gate == 'midday' and not (11.0 <= et_hour < 14.0):
        return False
    elif time_gate == 'prime_hours' and not (10.5 <= et_hour < 14.5):
        return False
    # 'rth_full' = no filter

    # Day gates
    if day_gate == 'mon_wed_fri' and weekday not in (0, 2, 4):
        return False
    elif day_gate == 'tue_thu' and weekday not in (1, 3):
        return False
    elif day_gate == 'fri_only' and weekday != 4:
        return False
    elif day_gate == 'mon_only' and weekday != 0:
        return False
    # 'all_days' = no filter

    return True


# ─── Single Sim Job ─────────────────────────────────────────────────────────

def config_hash(config):
    """Short hash for a config to use in filenames."""
    key = json.dumps(config, sort_keys=True)
    return hashlib.md5(key.encode()).hexdigest()[:8]


def run_single_sim(mbo_file, pred_file, config, out_dir):
    """Run a single Rust MBO sim job.
    Returns parsed JSON result or None on failure.
    """
    out_file = Path(out_dir) / f"{config['label']}_{Path(mbo_file).name.split('.')[0].split('-')[-1]}.json"

    # Check if already completed (incremental save)
    if out_file.exists():
        try:
            with open(out_file) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass

    cmd = [
        str(BINARY),
        '--mbo-file', str(mbo_file),
        '--predictions', str(pred_file),
        '--output', str(out_file),
        '--hold-ms', str(config['hold_ms']),
        '--signal-threshold', str(config['signal_threshold']),
        '--latency-ms', str(config.get('latency_ms', 0)),
        '--quiet',
    ]

    # Entry mode
    entry_mode = config.get('entry_mode', 'passive')
    if entry_mode == 'chase':
        cmd.append('--chase-entry')
        if config.get('chase_max_ticks') is not None:
            cmd.extend(['--chase-max-ticks', str(config['chase_max_ticks'])])
        if config.get('chase_max_reprices') is not None:
            cmd.extend(['--chase-max-reprices', str(config['chase_max_reprices'])])
        if config.get('chase_force_cross'):
            cmd.append('--chase-force-cross')
        if config.get('chase_interval_ms') is not None:
            cmd.extend(['--chase-interval-ms', str(config['chase_interval_ms'])])
    elif entry_mode == 'market':
        cmd.append('--market-entry')

    # Exit options
    if config.get('trailing_ticks') is not None:
        cmd.extend(['--trailing-ticks', str(config['trailing_ticks'])])
    if config.get('take_profit_ticks') is not None:
        cmd.extend(['--take-profit-ticks', str(config['take_profit_ticks'])])
    if config.get('signal_flip_exit'):
        cmd.append('--signal-flip-exit')
    if config.get('prime_hours'):
        cmd.append('--prime-hours')

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        if r.returncode != 0:
            return None
        with open(out_file) as f:
            return json.load(f)
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None


# ─── Stats Computation ───────────────────────────────────────────────────────

def compute_comprehensive_stats(config, date_results):
    """Compute comprehensive statistics from per-date sim results.
    Returns a rich stats dict for the config.
    """
    all_trades = []
    daily_pnls = []
    daily_trade_counts = []
    total_signals = 0
    total_filled = 0
    total_cancelled = 0
    hourly_pnl = {}
    weekly_data = {}

    time_gate = config.get('time_gate')
    day_gate = config.get('day_gate')
    side_filter = config.get('side_filter', 'both')

    for date_str in sorted(date_results.keys()):
        res = date_results[date_str]
        total_signals += res.get('total_signals', 0)
        total_filled += res.get('total_filled', 0)
        total_cancelled += res.get('total_cancelled', 0)

        day_pnl = 0.0
        day_trades = 0

        trades = res.get('trades', [])
        for trade in trades:
            # Apply side filter
            side = trade.get('side', 'BUY')
            if side_filter == 'long_only' and side != 'BUY':
                continue
            if side_filter == 'short_only' and side != 'SELL':
                continue

            # Apply time/day gate
            if not should_include_trade(trade, time_gate, day_gate):
                continue

            pnl = trade.get('pnl_dollars', 0)
            pnl_ticks = trade.get('pnl_ticks', 0)
            day_pnl += pnl
            day_trades += 1
            all_trades.append(trade)

            # Hourly heatmap
            sig_ns = trade.get('signal_time_ns', 0)
            if sig_ns > 0:
                import datetime as dt
                ts = dt.datetime.utcfromtimestamp(sig_ns / 1e9)
                month = ts.month
                et_offset = 4 if (3 <= month <= 11) else 5
                et_hour = (ts.hour - et_offset) % 24
                hourly_pnl[et_hour] = hourly_pnl.get(et_hour, 0) + pnl

            # Weekly breakdown
            # Use ISO week
            import datetime as dt
            ts = dt.datetime.utcfromtimestamp(sig_ns / 1e9) if sig_ns > 0 else None
            if ts:
                week_key = ts.strftime('%Y-W%W')
                if week_key not in weekly_data:
                    weekly_data[week_key] = {'pnl': 0, 'trades': 0, 'wins': 0}
                weekly_data[week_key]['pnl'] += pnl
                weekly_data[week_key]['trades'] += 1
                if pnl > 0:
                    weekly_data[week_key]['wins'] += 1

        daily_pnls.append({'date': date_str, 'pnl': day_pnl, 'trades': day_trades})
        daily_trade_counts.append(day_trades)

    n_trades = len(all_trades)
    if n_trades == 0:
        return {
            'config': config['label'],
            'sweep': config['sweep'],
            'model_type': config.get('model_type', 'cnn'),
            'total_pnl': 0, 'sharpe': 0, 'n_trades': 0,
            'error': 'no_trades',
        }

    # P&L arrays
    trade_pnls = np.array([t.get('pnl_dollars', 0) for t in all_trades])
    trade_pnl_ticks = np.array([t.get('pnl_ticks', 0) for t in all_trades])
    daily_pnl_values = np.array([d['pnl'] for d in daily_pnls])

    # Win/loss decomposition
    wins = trade_pnls[trade_pnls > 0]
    losses = trade_pnls[trade_pnls <= 0]
    win_count = len(wins)
    loss_count = len(losses)
    win_rate = win_count / n_trades

    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0
    avg_win_ticks = float(np.mean(trade_pnl_ticks[trade_pnls > 0])) if win_count > 0 else 0
    avg_loss_ticks = float(np.mean(trade_pnl_ticks[trade_pnls <= 0])) if loss_count > 0 else 0

    profit_factor = float(abs(np.sum(wins)) / abs(np.sum(losses))) if np.sum(losses) != 0 else float('inf')

    # Sharpe / Sortino / Calmar
    total_pnl = float(np.sum(trade_pnls))
    avg_daily = float(np.mean(daily_pnl_values))
    std_daily = float(np.std(daily_pnl_values)) if len(daily_pnl_values) > 1 else 1e-8
    sharpe = (avg_daily / std_daily) * np.sqrt(252) if std_daily > 1e-10 else 0

    downside = daily_pnl_values[daily_pnl_values < 0]
    downside_std = float(np.std(downside)) if len(downside) > 1 else 1e-8
    sortino = (avg_daily / downside_std) * np.sqrt(252) if downside_std > 1e-10 else 0

    cum = np.cumsum(daily_pnl_values)
    peak = np.maximum.accumulate(cum)
    drawdowns = cum - peak
    max_dd = float(abs(drawdowns.min())) if len(drawdowns) > 0 else 0
    calmar = (total_pnl / max_dd) if max_dd > 0 else 0
    recovery_factor = (total_pnl / max_dd) if max_dd > 0 else 0

    # Long/short decomposition
    long_trades = [t for t in all_trades if t.get('side') == 'BUY']
    short_trades = [t for t in all_trades if t.get('side') == 'SELL']
    long_pnls = [t.get('pnl_dollars', 0) for t in long_trades]
    short_pnls = [t.get('pnl_dollars', 0) for t in short_trades]

    long_wr = sum(1 for p in long_pnls if p > 0) / len(long_pnls) if long_pnls else 0
    short_wr = sum(1 for p in short_pnls if p > 0) / len(short_pnls) if short_pnls else 0

    # Fill rate
    fill_rate = total_filled / total_signals if total_signals > 0 else 0
    cancel_rate = total_cancelled / (total_filled + total_cancelled) if (total_filled + total_cancelled) > 0 else 0

    # Fill latency
    fill_lats = [t.get('fill_latency_ns', 0) / 1e6 for t in all_trades if t.get('fill_latency_ns', 0) > 0]
    avg_fill_latency_ms = float(np.mean(fill_lats)) if fill_lats else 0

    # MFE/MAE (from pnl_ticks proxy — actual MFE/MAE not in sim output)
    mfes = []
    maes = []
    hold_durations = []
    for t in all_trades:
        pnl_t = t.get('pnl_ticks', 0)
        if pnl_t > 0:
            mfes.append(pnl_t)
            maes.append(0)
        else:
            mfes.append(0)
            maes.append(abs(pnl_t))

        hold_ns = t.get('hold_duration_ns', 0)
        if hold_ns > 0:
            hold_durations.append(hold_ns / 1e9 / 60.0)  # minutes

    # Consecutive wins/losses
    signs = np.sign(trade_pnls)
    max_consec_wins = 0
    max_consec_losses = 0
    curr_wins = 0
    curr_losses = 0
    for s in signs:
        if s > 0:
            curr_wins += 1
            curr_losses = 0
            max_consec_wins = max(max_consec_wins, curr_wins)
        elif s <= 0:
            curr_losses += 1
            curr_wins = 0
            max_consec_losses = max(max_consec_losses, curr_losses)

    # Weekly breakdown
    weekly_breakdown = []
    for week_key in sorted(weekly_data.keys()):
        wd = weekly_data[week_key]
        wr = wd['wins'] / wd['trades'] if wd['trades'] > 0 else 0
        weekly_breakdown.append({
            'week': week_key,
            'pnl': round(wd['pnl'], 2),
            'trades': wd['trades'],
            'wr': round(wr, 4),
        })

    # Per-trade log
    per_trade_log = []
    for t in all_trades:
        per_trade_log.append({
            'entry_time': t.get('signal_time_ns', 0),
            'exit_time': t.get('exit_time_ns', 0),
            'side': t.get('side', ''),
            'entry_price': t.get('entry_price', 0),
            'exit_price': t.get('exit_price', 0),
            'pnl_ticks': t.get('pnl_ticks', 0),
            'pnl_dollars': t.get('pnl_dollars', 0),
            'fill_latency_ms': t.get('fill_latency_ns', 0) / 1e6,
            'hold_duration_min': t.get('hold_duration_ns', 0) / 1e9 / 60.0,
            'exit_reason': t.get('exit_reason', ''),
            'signal_strength': t.get('signal_strength', 0),
        })

    stats = {
        'config': config['label'],
        'sweep': config['sweep'],
        'model_type': config.get('model_type', 'cnn'),
        'config_params': {k: v for k, v in config.items() if k not in ('label', 'sweep')},

        # Core metrics
        'total_pnl': round(total_pnl, 2),
        'sharpe': round(sharpe, 4),
        'sortino': round(sortino, 4),
        'calmar': round(calmar, 4),
        'max_dd': round(max_dd, 2),
        'recovery_factor': round(recovery_factor, 4),

        # Trade counts
        'n_trades': n_trades,
        'n_days': len(daily_pnls),
        'win_rate': round(win_rate, 4),
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'avg_win_ticks': round(avg_win_ticks, 2),
        'avg_loss_ticks': round(avg_loss_ticks, 2),
        'profit_factor': round(profit_factor, 4) if profit_factor != float('inf') else 999.0,

        # Long/short decomposition
        'long_count': len(long_trades),
        'long_wr': round(long_wr, 4),
        'long_avg_pnl': round(float(np.mean(long_pnls)), 2) if long_pnls else 0,
        'short_count': len(short_trades),
        'short_wr': round(short_wr, 4),
        'short_avg_pnl': round(float(np.mean(short_pnls)), 2) if short_pnls else 0,

        # Fill statistics
        'fill_rate': round(fill_rate, 4),
        'cancel_rate': round(cancel_rate, 4),
        'avg_fill_latency_ms': round(avg_fill_latency_ms, 2),

        # MFE/MAE approximations
        'mfe_mean': round(float(np.mean(mfes)), 2) if mfes else 0,
        'mfe_median': round(float(np.median(mfes)), 2) if mfes else 0,
        'mae_mean': round(float(np.mean(maes)), 2) if maes else 0,
        'mae_median': round(float(np.median(maes)), 2) if maes else 0,

        # Extremes
        'best_day_pnl': round(float(max(daily_pnl_values)), 2),
        'worst_day_pnl': round(float(min(daily_pnl_values)), 2),
        'best_trade_pnl': round(float(max(trade_pnls)), 2),
        'worst_trade_pnl': round(float(min(trade_pnls)), 2),

        # Streak
        'max_consec_wins': int(max_consec_wins),
        'max_consec_losses': int(max_consec_losses),

        # Series data
        'daily_pnl_series': daily_pnls,
        'hourly_pnl_heatmap': {str(k): round(v, 2) for k, v in sorted(hourly_pnl.items())},
        'weekly_breakdown': weekly_breakdown,

        # Per-trade log
        'per_trade_log': per_trade_log,
    }

    return stats


# ─── Main Sweep Runner ──────────────────────────────────────────────────────

def run_sweep(configs, mbo_dir, pred_dir, out_dir, workers=8,
              gnn_pred_dir=None, ensemble_pred_dir=None):
    """Run the full sweep: configs x dates.

    For sweeps 7-8 (GNN), uses gnn_pred_dir instead of pred_dir.
    For sweep 9 (ensemble), generates ensemble predictions first, then runs sim.
    """
    mbo_files = discover_mbo_files(mbo_dir)

    if not mbo_files:
        log.error(f"No MBO files found in {mbo_dir}")
        return {}

    # Discover prediction files for each model type
    cnn_pred_files = discover_prediction_files(pred_dir)
    gnn_pred_files = {}
    ensemble_pred_files = {}

    if gnn_pred_dir:
        gnn_pred_files = discover_prediction_files(gnn_pred_dir)
        log.info(f"Found {len(gnn_pred_files)} GNN prediction files")

    # Check if any ensemble configs need predictions generated
    has_ensemble = any(c.get('model_type') == 'ensemble' for c in configs)
    if has_ensemble and gnn_pred_dir and ensemble_pred_dir:
        # For each agreement_mode, generate ensemble predictions
        agreement_modes = set(
            c.get('agreement_mode', 'average')
            for c in configs if c.get('model_type') == 'ensemble'
        )
        for mode in agreement_modes:
            mode_dir = Path(ensemble_pred_dir) / mode
            ensemble_files_for_mode = generate_ensemble_predictions(
                pred_dir, gnn_pred_dir, str(mode_dir), agreement_mode=mode
            )
            # Tag by mode so we can look them up later
            for key, path in ensemble_files_for_mode.items():
                ensemble_pred_files[(key[0], key[1], mode)] = path

    log.info(f"Found {len(mbo_files)} MBO dates")
    log.info(f"  CNN predictions: {len(cnn_pred_files)}")
    log.info(f"  GNN predictions: {len(gnn_pred_files)}")
    log.info(f"  Ensemble predictions: {len(ensemble_pred_files)}")

    # Build job list: each job = (config, date, mbo_file, pred_file)
    jobs = []
    skipped_no_pred = 0
    for config in configs:
        vol_gate = config['vol_gate']
        model_type = config.get('model_type', 'cnn')

        for date_str, mbo_file in mbo_files.items():
            pred_key = (date_str, vol_gate)

            if model_type == 'gnn':
                if pred_key not in gnn_pred_files:
                    skipped_no_pred += 1
                    continue
                pred_file = str(gnn_pred_files[pred_key])
            elif model_type == 'ensemble':
                agree_mode = config.get('agreement_mode', 'average')
                ens_key = (date_str, vol_gate, agree_mode)
                if ens_key not in ensemble_pred_files:
                    skipped_no_pred += 1
                    continue
                pred_file = str(ensemble_pred_files[ens_key])
            else:
                # CNN (default)
                if pred_key not in cnn_pred_files:
                    skipped_no_pred += 1
                    continue
                pred_file = str(cnn_pred_files[pred_key])

            jobs.append((config, date_str, str(mbo_file), pred_file))

    if skipped_no_pred > 0:
        log.info(f"  Skipped {skipped_no_pred} jobs (no matching prediction file)")

    log.info(f"Total jobs: {len(jobs)} ({len(configs)} configs x available dates)")

    if not jobs:
        log.error("No jobs to run. Check prediction directories.")
        return {}

    # Per-config sim output directory
    sim_out = Path(out_dir) / 'sim_outputs'
    sim_out.mkdir(parents=True, exist_ok=True)

    # Run jobs with multiprocessing
    results_by_config = {}
    done = 0
    t0 = time.time()
    failed = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for config, date_str, mbo_file, pred_file in jobs:
            future = executor.submit(run_single_sim, mbo_file, pred_file, config, str(sim_out))
            futures[future] = (config, date_str)

        for future in as_completed(futures):
            done += 1
            config, date_str = futures[future]
            try:
                result = future.result()
                if result:
                    label = config['label']
                    if label not in results_by_config:
                        results_by_config[label] = {'config': config, 'dates': {}}
                    results_by_config[label]['dates'][date_str] = result
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                if done <= 5:
                    log.warning(f"Job error: {e}")

            if done % 100 == 0 or done == len(jobs):
                elapsed = time.time() - t0
                rate = done / elapsed
                remaining = (len(jobs) - done) / max(rate, 0.01)
                log.info(
                    f"  [{done}/{len(jobs)}] {rate:.1f} jobs/s, "
                    f"~{remaining/60:.1f}m remaining, {failed} failed"
                )

                # Incremental save of progress
                _save_progress(results_by_config, out_dir)

    elapsed = time.time() - t0
    log.info(f"Sweep complete: {done} jobs in {elapsed:.1f}s ({failed} failed)")
    return results_by_config


def _save_progress(results_by_config, out_dir):
    """Save intermediate progress so we don't lose work on interruption."""
    progress_file = Path(out_dir) / 'sweep_progress.json'
    summary = {}
    for label, data in results_by_config.items():
        summary[label] = {
            'n_dates': len(data['dates']),
            'sweep': data['config']['sweep'],
            'model_type': data['config'].get('model_type', 'cnn'),
        }
    try:
        with open(progress_file, 'w') as f:
            json.dump(summary, f, indent=1)
    except Exception:
        pass


def compute_all_stats(results_by_config, out_dir):
    """Compute comprehensive stats for all configs and save."""
    log.info("Computing comprehensive statistics...")
    all_stats = []

    for label, data in results_by_config.items():
        config = data['config']
        date_results = data['dates']
        stats = compute_comprehensive_stats(config, date_results)
        all_stats.append(stats)

    # Sort by Sharpe
    all_stats.sort(key=lambda x: x.get('sharpe', 0), reverse=True)

    # Save full results
    full_file = Path(out_dir) / f'oot_mega_sweep_full_{_ts}.json'
    with open(full_file, 'w') as f:
        json.dump(all_stats, f, indent=2, default=str)
    log.info(f"Full results saved: {full_file}")

    # Save summary (without per-trade logs for quick viewing)
    summary_stats = []
    for s in all_stats:
        summary = {k: v for k, v in s.items()
                   if k not in ('per_trade_log', 'daily_pnl_series', 'weekly_breakdown', 'hourly_pnl_heatmap')}
        summary_stats.append(summary)

    summary_file = Path(out_dir) / f'oot_mega_sweep_summary_{_ts}.json'
    with open(summary_file, 'w') as f:
        json.dump(summary_stats, f, indent=2, default=str)
    log.info(f"Summary saved: {summary_file}")

    return all_stats


def print_leaderboard(all_stats, top_n=30):
    """Print leaderboard of top configs."""
    log.info("\n" + "=" * 130)
    log.info("OOT MEGA SWEEP LEADERBOARD — Top Configs by Sharpe")
    log.info("=" * 130)
    log.info(
        f"{'#':>3} {'Config':<55} {'Model':>8} {'P&L':>10} {'Sharpe':>7} {'Trades':>7} "
        f"{'WR':>6} {'FillR':>6} {'MaxDD':>8} {'PF':>6} {'Sortino':>8}"
    )
    log.info("-" * 130)

    for i, s in enumerate(all_stats[:top_n]):
        log.info(
            f"{i+1:>3} {s['config']:<55} "
            f"{s.get('model_type', 'cnn'):>8} "
            f"${s['total_pnl']:>9,.0f} "
            f"{s['sharpe']:>7.2f} "
            f"{s['n_trades']:>7} "
            f"{s['win_rate']:>5.1%} "
            f"{s['fill_rate']:>5.1%} "
            f"${s['max_dd']:>7,.0f} "
            f"{s['profit_factor']:>6.2f} "
            f"{s['sortino']:>8.2f}"
        )


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='OOT Mega Sweep — Comprehensive Out-of-Time Validation (CNN/GNN/Ensemble)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all sweeps on OOT data (CNN only, sweeps 1-6)
  python alpha_discovery/oot_mega_sweep.py --data-dir ~/Lvl3Quant/mbo_oot

  # Run only sweep 1 (chase mode) with 12 workers
  python alpha_discovery/oot_mega_sweep.py --data-dir ~/mbo_oot --sweep 1 --workers 12

  # Run configs 0-40 of sweep 1 (for splitting across machines)
  python alpha_discovery/oot_mega_sweep.py --data-dir ~/mbo_oot --sweep 1 --config-range 0-40

  # Run GNN chase sweep (sweep 7) with GNN predictions
  python alpha_discovery/oot_mega_sweep.py --data-dir ~/mbo_oot --sweep 7 \\
      --gnn-pred-dir ~/Lvl3Quant/data/processed/gnn_oot_predictions

  # Run ensemble sweep (sweep 9) — needs both CNN and GNN predictions
  python alpha_discovery/oot_mega_sweep.py --data-dir ~/mbo_oot --sweep 9 \\
      --gnn-pred-dir ~/Lvl3Quant/data/processed/gnn_oot_predictions \\
      --ensemble-pred-dir ~/Lvl3Quant/data/processed/ensemble_oot_predictions

  # Run all sweeps including GNN + ensemble
  python alpha_discovery/oot_mega_sweep.py --data-dir ~/mbo_oot --workers 12 \\
      --gnn-pred-dir ~/Lvl3Quant/data/processed/gnn_oot_predictions \\
      --ensemble-pred-dir ~/Lvl3Quant/data/processed/ensemble_oot_predictions

  # Use existing IS predictions (same model, new data dates)
  python alpha_discovery/oot_mega_sweep.py --data-dir ~/mbo_oot --pred-dir ~/Lvl3Quant/data/processed/cnn_oot_predictions
        """
    )
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing OOT MBO .dbn/.dbn.zst files')
    parser.add_argument('--pred-dir', type=str, default=None,
                        help='Directory containing CNN prediction .npz files '
                             '(default: LVL3_ROOT/data/processed/cnn_oot_predictions)')
    parser.add_argument('--gnn-pred-dir', type=str, default=None,
                        help='Directory containing GNN prediction .npz files '
                             '(default: LVL3_ROOT/data/processed/gnn_oot_predictions). '
                             'Required for sweeps 7, 8, 9.')
    parser.add_argument('--ensemble-pred-dir', type=str, default=None,
                        help='Directory to store generated ensemble prediction .npz files '
                             '(default: LVL3_ROOT/data/processed/ensemble_oot_predictions). '
                             'Required for sweep 9.')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='Output directory (default: alpha_discovery/results/oot)')
    parser.add_argument('--sweep', type=int, default=None,
                        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9],
                        help='Run only this sweep number (default: all)')
    parser.add_argument('--config-range', type=str, default=None,
                        help='Config index range, e.g. "0-40" (for splitting across machines)')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers (default: 8)')
    parser.add_argument('--list-configs', action='store_true',
                        help='Just list all configs and exit (dry run)')

    args = parser.parse_args()

    # Prediction directories
    pred_dir = args.pred_dir or str(LVL3_ROOT / 'data' / 'processed' / 'cnn_oot_predictions')
    gnn_pred_dir = args.gnn_pred_dir or str(LVL3_ROOT / 'data' / 'processed' / 'gnn_oot_predictions')
    ensemble_pred_dir = args.ensemble_pred_dir or str(LVL3_ROOT / 'data' / 'processed' / 'ensemble_oot_predictions')

    # Output directory
    out_dir = args.out_dir or str(RESULTS_DIR / 'oot')
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    log.info("=" * 80)
    log.info("OOT MEGA SWEEP — Comprehensive Out-of-Time Validation (CNN/GNN/Ensemble)")
    log.info("=" * 80)
    log.info(f"  Binary:          {BINARY}")
    log.info(f"  MBO data:        {args.data_dir}")
    log.info(f"  CNN predictions: {pred_dir}")
    log.info(f"  GNN predictions: {gnn_pred_dir}")
    log.info(f"  Ensemble dir:    {ensemble_pred_dir}")
    log.info(f"  Output:          {out_dir}")
    log.info(f"  Workers:         {args.workers}")
    log.info(f"  Sweep:           {'all' if args.sweep is None else args.sweep}")
    log.info(f"  Config range:    {args.config_range or 'all'}")
    log.info("")

    # Generate configs
    configs = get_all_configs(sweep_filter=args.sweep)
    log.info(f"Total configs generated: {len(configs)}")

    # Apply config range filter
    if args.config_range:
        start, end = map(int, args.config_range.split('-'))
        configs = configs[start:end]
        log.info(f"  Filtered to range [{start}:{end}] = {len(configs)} configs")

    # Validate: if running GNN/ensemble sweeps, check prediction dirs
    has_gnn = any(c.get('model_type') == 'gnn' for c in configs)
    has_ensemble = any(c.get('model_type') == 'ensemble' for c in configs)

    if has_gnn and not Path(gnn_pred_dir).exists():
        log.warning(f"GNN prediction dir does not exist: {gnn_pred_dir}")
        log.warning("GNN sweeps (7, 8) will produce 0 jobs unless predictions are generated first.")
        log.warning("Use gnn_rust_sim_validation.py to generate GNN predictions, or --gnn-pred-dir.")

    if has_ensemble and not Path(gnn_pred_dir).exists():
        log.warning(f"Ensemble requires GNN predictions but dir does not exist: {gnn_pred_dir}")
        log.warning("Sweep 9 will produce 0 jobs. Generate GNN predictions first.")

    if args.list_configs:
        log.info("\nConfig listing:")
        for i, c in enumerate(configs):
            model = c.get('model_type', 'cnn')
            log.info(f"  [{i:>3}] [{model:>8}] {c['label']}")
        log.info(f"\nTotal: {len(configs)} configs")

        # Count by model type
        from collections import Counter
        model_counts = Counter(c.get('model_type', 'cnn') for c in configs)
        for model, count in model_counts.items():
            log.info(f"  {model}: {count} configs")
        return

    # Run sweep
    results = run_sweep(
        configs, args.data_dir, pred_dir, out_dir, workers=args.workers,
        gnn_pred_dir=gnn_pred_dir, ensemble_pred_dir=ensemble_pred_dir
    )

    if not results:
        log.error("No results produced. Check data paths and predictions.")
        return

    # Compute stats
    all_stats = compute_all_stats(results, out_dir)

    # Print leaderboard
    print_leaderboard(all_stats)

    log.info(f"\nDone. Results in: {out_dir}")


if __name__ == '__main__':
    main()
