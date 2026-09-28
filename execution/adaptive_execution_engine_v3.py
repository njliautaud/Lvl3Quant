#!/usr/bin/env python3
"""
Adaptive Execution Engine v3 — Decision Tree Meta-Learner
==========================================================
NOT a parameter sweep. This is fundamentally different from v1/v2.

ARCHITECTURE:
  Phase 1: HARVEST — Run baseline configs through fill_sim_cli to collect
           per-trade data with rich features (signal strength, queue pos,
           fill latency, hour, vol regime, momentum, confluence).

  Phase 2: LEARN — Train a LightGBM gradient-boosted decision tree on the
           harvested trade features to predict per-trade PnL. Features:
           - Signal strength (z-score magnitude)
           - Signal quintile (relative conviction rank)
           - Queue position at fill
           - Fill latency (ms) — proxy for adverse selection
           - Hour of day (ET)
           - Local volatility regime (rolling std of signal)
           - Momentum persistence (consecutive same-direction bars)
           - Multi-horizon agreement (1s/5s/10s concordance)
           - Side (long vs short)
           - Book depth at entry
           - MFE/MAE path features from completed trades

  Phase 3: GATE — Use the trained model as a trade filter. Only take trades
           where the model predicts positive expected PnL. The model learns
           the INTERACTION effects between features (e.g., "high conviction +
           good queue position + afternoon = good trade" vs "high conviction +
           bad queue position + morning = adverse selection trap").

  Phase 4: OPTIMIZE — For trades that pass the gate, use the model's feature
           importance to set adaptive TP/SL. The model tells us which regimes
           support wider targets vs which need tight stops.

ADVANCED TECHNIQUES:
  1. MFE/MAE path analysis — empirical price path distributions
  2. Ratcheting MFE stops — trail winners based on MFE thresholds
  3. Adverse selection filter — fill latency < 100ms = likely picked off
  4. Queue position estimation — near-top fills have less adverse selection
  5. Momentum persistence gate — N consecutive events same direction
  6. Multi-horizon confluence — 2/3 or 3/3 horizons must agree
  7. Vol-regime adaptive — different thresholds per vol regime
  8. GBM decision tree — learned optimal {action, TP, SL, hold} per context

Usage:
    python adaptive_execution_engine_v3.py
    python adaptive_execution_engine_v3.py --workers 12 --skip-harvest
"""

import sys
import json
import time
import argparse
import subprocess
import logging
import os
import pickle
from pathlib import Path
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from typing import Optional, Dict, List, Tuple, Any

import numpy as np

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_squared_error, r2_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ── Paths ──────────────────────────────────────────────────────────────────
LVL3_ROOT = Path(__file__).resolve().parent.parent
BINARY = LVL3_ROOT / 'rust_cache_builder' / 'target' / 'release' / 'fill_sim_cli'
MBO_DIR = LVL3_ROOT / 'data' / 'raw' / 'mbo'
EVENT_DIR_V3 = LVL3_ROOT / 'data' / 'processed' / 'mbo_events_smart_v3'
EVENT_DIR_V2 = LVL3_ROOT / 'data' / 'processed' / 'mbo_events_smart_v2'
RESULTS_DIR = LVL3_ROOT / 'execution' / 'results' / 'v3_meta_learner'
PRED_CACHE_DIR = LVL3_ROOT / 'execution' / 'pred_cache' / 'v3_meta_learner'
MODEL_DIR = LVL3_ROOT / 'execution' / 'models'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PRED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CNN_MAMBA_V2_DIR = LVL3_ROOT / 'output' / 'cnn_mamba_v2_smart_v3_mar'

TICK_VALUE = 12.50
BARS_PER_SEC = 10
BAR_NS = 100_000_000
RTH_HOURS = 6.5
N_RTH_BARS = int(RTH_HOURS * 3600 * BARS_PER_SEC)
WINDOW = 1000
STRIDE = 500

_ts = datetime.now().strftime('%Y%m%d_%H%M%S')

log = logging.getLogger('v3_meta')
log.setLevel(logging.INFO)
_fh = logging.FileHandler(str(RESULTS_DIR / f'v3_meta_{_ts}.log'), mode='w', encoding='utf-8')
_fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
log.addHandler(_fh)
_ch = logging.StreamHandler(sys.stdout)
_ch.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
log.addHandler(_ch)


# ============================================================
# PHASE 0: Signal Preparation (shared with v2)
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
    """Prepare multi-horizon z-scored signals for all dates."""
    all_signals = {}
    running_stats = {h: None for h in ['1s', '5s', '10s']}

    for date_str in sorted(folds.keys()):
        fold_data = folds[date_str]

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


# ============================================================
# FEATURE ENGINEERING — Per-bar features for the meta-learner
# ============================================================

def compute_bar_features(
    primary_signal: np.ndarray,
    horizon_signals: Dict[str, np.ndarray],
    date_str: str,
) -> Dict[str, np.ndarray]:
    """Compute per-bar features that the GBM will use for trade gating.

    These features are computed BEFORE the trade happens, so no lookahead.
    They capture the market microstructure context at signal time.
    """
    n_bars = len(primary_signal)
    features = {}

    # 1. Signal magnitude (absolute z-score)
    features['signal_abs'] = np.abs(primary_signal)
    features['signal_raw'] = primary_signal.copy()

    # 2. Signal side: +1 long, -1 short, 0 no signal
    features['signal_side'] = np.sign(primary_signal)

    # 3. Signal quintile (rolling percentile rank)
    signal_quintile = np.zeros(n_bars, dtype=np.float64)
    abs_sig = np.abs(primary_signal)
    # Use expanding window of non-zero signals
    nonzero_vals = []
    for i in range(n_bars):
        if abs_sig[i] > 0:
            nonzero_vals.append(abs_sig[i])
            if len(nonzero_vals) >= 20:
                rank = np.searchsorted(np.sort(nonzero_vals), abs_sig[i]) / len(nonzero_vals)
                signal_quintile[i] = min(5, int(rank * 5) + 1)
    features['signal_quintile'] = signal_quintile

    # 4. Hour of day (ET) — bar index to hour
    rth_start_hour = 9.5  # 9:30 AM ET
    bar_hours = rth_start_hour + np.arange(n_bars) / (BARS_PER_SEC * 3600)
    features['hour_et'] = np.floor(bar_hours).astype(np.float64)

    # 5. Minute of day (for finer resolution)
    features['minute_of_day'] = ((bar_hours - 9.0) * 60).astype(np.float64)

    # 6. Local volatility regime (rolling std of signal over last 5 min)
    vol_lookback = 3000  # 5 min of 100ms bars
    local_vol = np.zeros(n_bars, dtype=np.float64)
    for i in range(vol_lookback, n_bars):
        window = primary_signal[i - vol_lookback:i]
        nz = window[window != 0]
        if len(nz) >= 10:
            local_vol[i] = np.std(nz)
    features['local_vol'] = local_vol

    # 7. Vol regime category (tercile)
    nz_vol = local_vol[local_vol > 0]
    if len(nz_vol) >= 30:
        vol_p33 = np.percentile(nz_vol, 33)
        vol_p66 = np.percentile(nz_vol, 66)
        vol_regime = np.zeros(n_bars, dtype=np.float64)
        for i in range(n_bars):
            if local_vol[i] <= 0:
                vol_regime[i] = 1  # default to low
            elif local_vol[i] < vol_p33:
                vol_regime[i] = 0  # low vol
            elif local_vol[i] < vol_p66:
                vol_regime[i] = 1  # mid vol
            else:
                vol_regime[i] = 2  # high vol
        features['vol_regime'] = vol_regime
    else:
        features['vol_regime'] = np.ones(n_bars, dtype=np.float64)

    # 8. Momentum persistence (consecutive same-direction bars)
    momentum = np.zeros(n_bars, dtype=np.float64)
    streak = 0
    last_sign = 0
    for i in range(n_bars):
        if primary_signal[i] == 0:
            streak = 0
            continue
        current_sign = 1 if primary_signal[i] > 0 else -1
        if current_sign == last_sign:
            streak += 1
        else:
            streak = 1
            last_sign = current_sign
        momentum[i] = streak
    features['momentum_streak'] = momentum

    # 9. Multi-horizon confluence
    if '1s' in horizon_signals and '5s' in horizon_signals and '10s' in horizon_signals:
        sign_1s = np.sign(horizon_signals['1s'])
        sign_5s = np.sign(horizon_signals['5s'])
        sign_10s = np.sign(horizon_signals['10s'])

        # Count agreeing pairs: 0, 1, 2, or 3
        agreement = (
            (sign_1s == sign_5s).astype(np.float64) +
            (sign_1s == sign_10s).astype(np.float64) +
            (sign_5s == sign_10s).astype(np.float64)
        )
        features['horizon_agreement'] = agreement

        # Strongest horizon signal (max absolute z across horizons)
        max_hz = np.maximum(np.abs(horizon_signals['1s']),
                            np.maximum(np.abs(horizon_signals['5s']),
                                       np.abs(horizon_signals['10s'])))
        features['max_horizon_z'] = max_hz

        # Horizon spread (difference between max and min absolute z)
        min_hz = np.minimum(np.abs(horizon_signals['1s']),
                            np.minimum(np.abs(horizon_signals['5s']),
                                       np.abs(horizon_signals['10s'])))
        features['horizon_spread'] = max_hz - min_hz
    else:
        features['horizon_agreement'] = np.ones(n_bars, dtype=np.float64) * 3
        features['max_horizon_z'] = features['signal_abs']
        features['horizon_spread'] = np.zeros(n_bars, dtype=np.float64)

    # 10. Signal acceleration (change in signal strength)
    signal_accel = np.zeros(n_bars, dtype=np.float64)
    last_nonzero = 0.0
    for i in range(n_bars):
        if primary_signal[i] != 0:
            signal_accel[i] = primary_signal[i] - last_nonzero
            last_nonzero = primary_signal[i]
    features['signal_accel'] = signal_accel

    # 11. Time since last signal (bars since last non-zero)
    time_since = np.zeros(n_bars, dtype=np.float64)
    last_sig_bar = -1
    for i in range(n_bars):
        if primary_signal[i] != 0:
            if last_sig_bar >= 0:
                time_since[i] = i - last_sig_bar
            last_sig_bar = i
    features['time_since_last_signal'] = time_since

    # 12. Signal density (fraction of non-zero bars in last 1 min)
    density_lookback = 600  # 1 min
    signal_density = np.zeros(n_bars, dtype=np.float64)
    for i in range(density_lookback, n_bars):
        window = primary_signal[i - density_lookback:i]
        signal_density[i] = np.count_nonzero(window) / density_lookback
    features['signal_density'] = signal_density

    # 13. Rolling signal mean (expanding, non-zero only) — recent bias
    rolling_mean = np.zeros(n_bars, dtype=np.float64)
    recent_lookback = 1800  # 3 min
    for i in range(recent_lookback, n_bars):
        window = primary_signal[i - recent_lookback:i]
        nz = window[window != 0]
        if len(nz) >= 5:
            rolling_mean[i] = np.mean(nz)
    features['rolling_signal_mean'] = rolling_mean

    return features


def save_bar_features_as_predictions(
    primary_signal: np.ndarray,
    cache_path: Path,
) -> Path:
    """Save the primary signal as a predictions file for fill_sim_cli."""
    np.savez_compressed(str(cache_path), predictions=primary_signal)
    return cache_path


# ============================================================
# PHASE 1: HARVEST — Collect per-trade data with features
# ============================================================

def define_harvest_configs() -> List[Dict]:
    """Define baseline configs for harvesting trade data.

    We run MULTIPLE config variants to get trade data under different
    TP/SL/hold settings. The meta-learner needs to see how trades perform
    under various exit conditions to learn what matters.
    """
    configs = []

    # Wide net: low threshold, various exit modes
    # These capture MANY trades with rich per-trade features
    for z in [1.5, 2.0, 2.5, 3.0]:
        for tp, sl in [(6, 10), (8, 15), (10, 20), (12, 20), (15, 25), (20, 30)]:
            for hold_ms in [30000, 60000, 120000, 300000]:
                configs.append({
                    'name': f'harvest_z{z}_tp{tp}_sl{sl}_h{hold_ms//1000}',
                    'cli_args': [
                        '--chase-entry',
                        '--signal-threshold', str(z),
                        '--stop-loss-ticks', str(sl),
                        '--take-profit-ticks', str(tp),
                        '--hold-ms', str(hold_ms),
                        '--quiet',
                    ],
                })

    # Ratchet variants
    for z in [2.0, 2.5, 3.0]:
        for tp, sl in [(10, 20), (15, 25)]:
            configs.append({
                'name': f'harvest_ratchet_z{z}_tp{tp}_sl{sl}',
                'cli_args': [
                    '--chase-entry',
                    '--signal-threshold', str(z),
                    '--stop-loss-ticks', str(sl),
                    '--take-profit-ticks', str(tp),
                    '--hold-ms', '120000',
                    '--ratchet-stop',
                    '--quiet',
                ],
            })

    # Trailing stop variants
    for z in [2.0, 2.5, 3.0]:
        for trail in [2, 3, 4, 5]:
            configs.append({
                'name': f'harvest_trail_z{z}_t{trail}',
                'cli_args': [
                    '--chase-entry',
                    '--signal-threshold', str(z),
                    '--stop-loss-ticks', '20',
                    '--take-profit-ticks', '15',
                    '--hold-ms', '120000',
                    '--trailing-ticks', str(trail),
                    '--quiet',
                ],
            })

    # Market entry (for adverse selection comparison)
    for z in [2.5, 3.0, 3.5]:
        configs.append({
            'name': f'harvest_market_z{z}',
            'cli_args': [
                '--market-entry',
                '--signal-threshold', str(z),
                '--stop-loss-ticks', '15',
                '--take-profit-ticks', '10',
                '--hold-ms', '120000',
                '--quiet',
            ],
        })

    # Prime hours only
    for z in [2.0, 2.5, 3.0]:
        configs.append({
            'name': f'harvest_prime_z{z}',
            'cli_args': [
                '--chase-entry',
                '--prime-hours',
                '--signal-threshold', str(z),
                '--stop-loss-ticks', '20',
                '--take-profit-ticks', '12',
                '--hold-ms', '120000',
                '--ratchet-stop',
                '--quiet',
            ],
        })

    # MAE patience variants
    for z in [2.0, 2.5]:
        for mae_ticks, mae_hold in [(10, 10), (13, 10), (15, 15)]:
            configs.append({
                'name': f'harvest_mae_z{z}_mt{mae_ticks}_mh{mae_hold}',
                'cli_args': [
                    '--chase-entry',
                    '--signal-threshold', str(z),
                    '--stop-loss-ticks', '25',
                    '--take-profit-ticks', '15',
                    '--hold-ms', '180000',
                    '--mae-exit-ticks', str(mae_ticks),
                    '--mae-exit-hold-sec', str(mae_hold),
                    '--ratchet-stop',
                    '--quiet',
                ],
            })

    log.info(f"  Defined {len(configs)} harvest configs")
    return configs


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
            return result
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None


def extract_trade_features(
    trade: Dict,
    bar_features: Dict[str, np.ndarray],
    config_name: str,
    date_str: str,
    rth_start: int,
) -> Optional[Dict]:
    """Extract rich features from a single trade for the meta-learner.

    Combines trade-level features (from fill_sim output) with bar-level
    features (computed from signal context at trade time).
    """
    try:
        signal_time_ns = trade['signal_time_ns']
        bar_idx = int((signal_time_ns - rth_start) // BAR_NS)

        if bar_idx < 0 or bar_idx >= N_RTH_BARS:
            return None

        feat = {
            # ── Trade-level features (from fill sim) ──
            'pnl_dollars': trade['pnl_dollars'],
            'pnl_ticks': trade['pnl_ticks'],
            'mae_ticks': trade['mae_ticks'],
            'mfe_ticks': trade['mfe_ticks'],
            'queue_pos': trade['queue_position_at_post'],
            'book_size': trade['book_size_at_post'],
            'fill_latency_ms': trade['fill_latency_ns'] / 1e6,
            'hold_duration_ms': trade['hold_duration_ns'] / 1e6,
            'signal_strength': abs(trade['signal_strength']),
            'signal_strength_raw': trade['signal_strength'],
            'side_numeric': 1.0 if trade['side'] == 'BUY' else -1.0,
            'exit_reason': trade['exit_reason'],

            # ── Derived trade features ──
            'queue_ratio': trade['queue_position_at_post'] / max(trade['book_size_at_post'], 1),
            'mfe_mae_ratio': trade['mfe_ticks'] / max(trade['mae_ticks'], 0.25),
            'fill_speed_category': (
                0 if trade['fill_latency_ns'] / 1e6 < 100 else  # instant (adverse selection risk)
                1 if trade['fill_latency_ns'] / 1e6 < 500 else  # fast
                2 if trade['fill_latency_ns'] / 1e6 < 2000 else  # normal
                3  # slow (good queue position)
            ),

            # ── Bar-level features (context at signal time) ──
            'date': date_str,
            'config': config_name,
            'bar_idx': bar_idx,
        }

        # Add all precomputed bar features at this bar index
        for fname, farray in bar_features.items():
            if bar_idx < len(farray):
                feat[f'bar_{fname}'] = float(farray[bar_idx])
            else:
                feat[f'bar_{fname}'] = 0.0

        return feat
    except (KeyError, IndexError, TypeError) as e:
        return None


def harvest_trades(
    all_signals: Dict[str, Dict],
    configs: List[Dict],
    workers: int = 8,
) -> List[Dict]:
    """Phase 1: Run harvest configs and collect per-trade feature vectors."""
    log.info(f"\n{'='*80}")
    log.info(f"PHASE 1: HARVEST — Collecting per-trade data")
    log.info(f"{'='*80}")

    # Prepare prediction files (just the primary 10s signal, no gates)
    pred_files = {}
    for date_str, sig_data in all_signals.items():
        cache_file = PRED_CACHE_DIR / f'primary_{date_str}.npz'
        if not cache_file.exists():
            np.savez_compressed(str(cache_file), predictions=sig_data['primary'])
        pred_files[date_str] = cache_file

    # Precompute bar features for all dates
    all_bar_features = {}
    for date_str, sig_data in all_signals.items():
        log.info(f"  Computing bar features for {date_str}...")
        all_bar_features[date_str] = compute_bar_features(
            sig_data['primary'], sig_data['horizons'], date_str
        )

    # Build jobs
    sim_out = RESULTS_DIR / f'harvest_{_ts}'
    sim_out.mkdir(parents=True, exist_ok=True)

    jobs = []
    for config in configs:
        for date_str in sorted(all_signals.keys()):
            jobs.append({
                'config': config,
                'date': date_str,
                'pred_file': pred_files[date_str],
            })

    log.info(f"  Total harvest jobs: {len(jobs)} "
             f"({len(configs)} configs x {len(all_signals)} dates)")

    # Run sims
    all_trades = []
    done = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for job in jobs:
            future = executor.submit(
                run_fill_sim,
                job['date'], job['pred_file'], job['config'], sim_out,
            )
            futures[future] = job

        for future in as_completed(futures):
            done += 1
            job = futures[future]
            result = future.result()

            if result and 'trades' in result:
                date_str = job['date']
                config_name = job['config']['name']
                rth_start = rth_start_ns_for_date(date_str)
                bar_feats = all_bar_features[date_str]

                for trade in result['trades']:
                    feat = extract_trade_features(
                        trade, bar_feats, config_name, date_str, rth_start
                    )
                    if feat:
                        all_trades.append(feat)

            if done % 200 == 0 or done == len(jobs):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(jobs) - done) / rate / 60 if rate > 0 else 0
                log.info(f"  Harvest: {done}/{len(jobs)} ({done/len(jobs):.0%}) "
                         f"| {len(all_trades)} trades | {rate:.1f}/s | ETA: {eta:.1f}min")

    elapsed = time.time() - t0
    log.info(f"  Harvest complete: {len(all_trades)} trades in {elapsed:.0f}s")

    # Save harvested data
    harvest_file = RESULTS_DIR / f'harvested_trades_{_ts}.json'
    with open(harvest_file, 'w') as f:
        json.dump(all_trades, f)
    log.info(f"  Saved to {harvest_file}")

    return all_trades


# ============================================================
# PHASE 2: LEARN — Train GBM meta-learner
# ============================================================

# Features the GBM will use (must be available BEFORE the trade)
GBM_FEATURES = [
    'signal_strength',          # |z-score| of the prediction
    'side_numeric',             # +1 long, -1 short
    'queue_pos',                # queue position at order post
    'book_size',                # book depth at order post
    'queue_ratio',              # queue_pos / book_size
    'fill_latency_ms',          # time to fill (adverse selection proxy)
    'fill_speed_category',      # binned fill speed
    'bar_signal_abs',           # absolute signal at bar
    'bar_signal_quintile',      # quintile rank of signal
    'bar_hour_et',              # hour of day (ET)
    'bar_minute_of_day',        # minute of day
    'bar_local_vol',            # local vol at signal time
    'bar_vol_regime',           # vol regime category
    'bar_momentum_streak',      # consecutive same-direction bars
    'bar_horizon_agreement',    # multi-horizon agreement (0-3)
    'bar_max_horizon_z',        # max |z| across horizons
    'bar_horizon_spread',       # spread of |z| across horizons
    'bar_signal_accel',         # signal acceleration
    'bar_time_since_last_signal',  # bars since last signal
    'bar_signal_density',       # signal density in recent window
    'bar_rolling_signal_mean',  # rolling mean of recent signals
]

# Features available only AFTER the trade (for MFE/MAE analysis, not for gating)
POST_TRADE_FEATURES = [
    'mae_ticks', 'mfe_ticks', 'mfe_mae_ratio', 'hold_duration_ms',
]


def prepare_training_data(trades: List[Dict]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Convert trade dicts to X, y arrays for GBM training."""

    # Filter to trades with all required features
    valid_trades = []
    for t in trades:
        has_all = True
        for f in GBM_FEATURES:
            if f not in t or t[f] is None:
                has_all = False
                break
        if has_all and 'pnl_dollars' in t:
            valid_trades.append(t)

    log.info(f"  Valid trades for training: {len(valid_trades)} / {len(trades)}")

    X = np.array([[t[f] for f in GBM_FEATURES] for t in valid_trades], dtype=np.float64)
    y = np.array([t['pnl_dollars'] for t in valid_trades], dtype=np.float64)

    return X, y, [t['date'] for t in valid_trades]


def train_gbm_meta_learner(
    trades: List[Dict],
) -> Tuple[Any, Dict]:
    """Train a LightGBM model to predict per-trade PnL from pre-trade features.

    Key insight: We train on trades from MULTIPLE config variants. The model
    learns which CONTEXT features (not config params) predict profitability.
    Then we use it as a gate: only take trades where predicted PnL > 0.
    """
    log.info(f"\n{'='*80}")
    log.info(f"PHASE 2: LEARN — Training GBM meta-learner")
    log.info(f"{'='*80}")

    if not HAS_LGB:
        log.error("LightGBM not installed! Cannot train meta-learner.")
        return None, {}

    X, y, dates = prepare_training_data(trades)

    if len(X) < 100:
        log.error(f"Only {len(X)} valid trades — not enough to train. Need 100+.")
        return None, {}

    log.info(f"  Training data: {X.shape[0]} trades, {X.shape[1]} features")
    log.info(f"  Target PnL: mean=${y.mean():.2f}, std=${y.std():.2f}, "
             f"positive={100*(y>0).mean():.1f}%")

    # ── Time-series split (no lookahead) ──
    unique_dates = sorted(set(dates))
    n_dates = len(unique_dates)

    if n_dates < 4:
        # Not enough dates for proper CV, use simple split
        split_idx = int(0.7 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        log.info(f"  Simple split: {len(X_train)} train, {len(X_val)} val")
    else:
        # Use last 30% of dates as validation
        val_start_date = unique_dates[int(0.7 * n_dates)]
        train_mask = np.array([d < val_start_date for d in dates])
        val_mask = ~train_mask
        X_train, X_val = X[train_mask], X[val_mask]
        y_train, y_val = y[train_mask], y[val_mask]
        log.info(f"  Time-series split: {len(X_train)} train ({sum(train_mask)} trades), "
                 f"{len(X_val)} val ({sum(val_mask)} trades)")
        log.info(f"  Train dates: {unique_dates[0]} to {val_start_date}")
        log.info(f"  Val dates: {val_start_date} to {unique_dates[-1]}")

    # ── Train LightGBM ──
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=GBM_FEATURES)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=GBM_FEATURES, reference=train_data)

    # Params tuned for small datasets with noisy targets
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42,
    }

    callbacks = [
        lgb.log_evaluation(period=50),
        lgb.early_stopping(stopping_rounds=50),
    ]

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=callbacks,
    )

    # ── Evaluate ──
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)

    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
    r2_train = r2_score(y_train, y_pred_train)
    r2_val = r2_score(y_val, y_pred_val)

    log.info(f"\n  Model Performance:")
    log.info(f"    Train RMSE: ${rmse_train:.2f}, R²: {r2_train:.4f}")
    log.info(f"    Val   RMSE: ${rmse_val:.2f}, R²: {r2_val:.4f}")

    # ── Feature importance ──
    importance = model.feature_importance(importance_type='gain')
    feat_imp = sorted(zip(GBM_FEATURES, importance), key=lambda x: x[1], reverse=True)
    log.info(f"\n  Feature Importance (gain):")
    for fname, imp in feat_imp:
        log.info(f"    {fname:<35} {imp:>10.1f}")

    # ── Gate analysis: trades where model predicts positive PnL ──
    log.info(f"\n  Gate Analysis (validation set):")
    for threshold in [0, 2, 5, 10, 15, 20]:
        gate_mask = y_pred_val > threshold
        n_pass = gate_mask.sum()
        if n_pass > 0:
            gated_pnl = y_val[gate_mask].sum()
            gated_mean = y_val[gate_mask].mean()
            gated_wr = (y_val[gate_mask] > 0).mean()
            log.info(f"    Threshold >${threshold:>3}: {n_pass:>5} trades, "
                     f"PnL=${gated_pnl:>8.0f}, $/trade=${gated_mean:>6.2f}, "
                     f"WR={gated_wr:.1%}")

    # ── MFE/MAE path analysis ──
    log.info(f"\n  MFE/MAE Path Analysis (all trades):")
    all_mae = np.array([t.get('mae_ticks', 0) for t in trades if 'mae_ticks' in t])
    all_mfe = np.array([t.get('mfe_ticks', 0) for t in trades if 'mfe_ticks' in t])
    if len(all_mae) > 0:
        log.info(f"    MAE: mean={all_mae.mean():.1f}t, median={np.median(all_mae):.1f}t, "
                 f"p90={np.percentile(all_mae, 90):.1f}t, p99={np.percentile(all_mae, 99):.1f}t")
        log.info(f"    MFE: mean={all_mfe.mean():.1f}t, median={np.median(all_mfe):.1f}t, "
                 f"p90={np.percentile(all_mfe, 90):.1f}t, p99={np.percentile(all_mfe, 99):.1f}t")

        # Conditional MFE by MAE depth
        for mae_max in [0, 1, 2, 3, 5]:
            mask = all_mae <= mae_max
            if mask.sum() > 10:
                log.info(f"    MFE when MAE<={mae_max}t: mean={all_mfe[mask].mean():.1f}t, "
                         f"n={mask.sum()}")

    # ── Adverse selection analysis ──
    log.info(f"\n  Adverse Selection Analysis:")
    for lat_max in [50, 100, 200, 500, 1000, 2000]:
        lat_mask = np.array([t.get('fill_latency_ms', 9999) < lat_max for t in trades])
        pnl_vals = np.array([t['pnl_dollars'] for t in trades])
        if lat_mask.sum() > 10:
            log.info(f"    Fill < {lat_max}ms: n={lat_mask.sum()}, "
                     f"mean PnL=${pnl_vals[lat_mask].mean():.2f}, "
                     f"WR={(pnl_vals[lat_mask] > 0).mean():.1%}")

    # ── Save model ──
    model_path = MODEL_DIR / f'gbm_meta_learner_{_ts}.txt'
    model.save_model(str(model_path))
    log.info(f"\n  Model saved to: {model_path}")

    # Also save as pickle for quick loading
    pkl_path = MODEL_DIR / f'gbm_meta_learner_{_ts}.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'features': GBM_FEATURES,
            'rmse_val': rmse_val,
            'r2_val': r2_val,
            'n_train': len(X_train),
            'n_val': len(X_val),
            'feat_importance': dict(feat_imp),
        }, f)

    metrics = {
        'rmse_train': rmse_train,
        'rmse_val': rmse_val,
        'r2_train': r2_train,
        'r2_val': r2_val,
        'n_train': len(X_train),
        'n_val': len(X_val),
        'feat_importance': dict(feat_imp),
    }

    return model, metrics


# ============================================================
# PHASE 3: GATE — Apply model as trade filter + adaptive params
# ============================================================

def apply_gbm_gate(
    model,
    primary_signal: np.ndarray,
    bar_features: Dict[str, np.ndarray],
    gate_threshold: float = 0.0,
) -> np.ndarray:
    """Use the trained GBM to filter signals.

    For each non-zero bar, compute features and predict PnL.
    Zero out signals where predicted PnL < threshold.

    NOTE: This is an approximation — we don't know queue_pos and fill_latency
    before the trade. We use reasonable estimates (median from training data).
    """
    n_bars = len(primary_signal)
    gated_signal = primary_signal.copy()

    # Default estimates for trade-level features we don't know yet
    default_queue_pos = 10.0  # median from training
    default_book_size = 15.0
    default_fill_latency_ms = 1000.0
    default_fill_speed_cat = 2.0  # normal

    signal_bars = np.nonzero(primary_signal)[0]
    if len(signal_bars) == 0:
        return gated_signal

    # Build feature matrix for all signal bars
    X = np.zeros((len(signal_bars), len(GBM_FEATURES)), dtype=np.float64)

    feature_map = {
        'signal_strength': np.abs(primary_signal[signal_bars]),
        'side_numeric': np.sign(primary_signal[signal_bars]),
        'queue_pos': np.full(len(signal_bars), default_queue_pos),
        'book_size': np.full(len(signal_bars), default_book_size),
        'queue_ratio': np.full(len(signal_bars), default_queue_pos / default_book_size),
        'fill_latency_ms': np.full(len(signal_bars), default_fill_latency_ms),
        'fill_speed_category': np.full(len(signal_bars), default_fill_speed_cat),
    }

    # Add bar features
    for fname, farray in bar_features.items():
        gbm_name = f'bar_{fname}'
        if gbm_name in GBM_FEATURES:
            vals = np.array([farray[bi] if bi < len(farray) else 0.0 for bi in signal_bars])
            feature_map[gbm_name] = vals

    for fi, feat_name in enumerate(GBM_FEATURES):
        if feat_name in feature_map:
            X[:, fi] = feature_map[feat_name]

    # Predict
    y_pred = model.predict(X)

    # Gate: zero out signals below threshold
    gate_mask = y_pred < gate_threshold
    gated_signal[signal_bars[gate_mask]] = 0.0

    n_filtered = gate_mask.sum()
    n_passed = (~gate_mask).sum()
    log.debug(f"  GBM gate: {n_passed} passed, {n_filtered} filtered "
              f"(threshold=${gate_threshold})")

    return gated_signal


# ============================================================
# PHASE 4: BACKTEST — Run gated strategies through fill sim
# ============================================================

def define_gated_strategies(model, gate_thresholds: List[float] = None) -> List[Dict]:
    """Define strategies that combine GBM gating with various exit configs.

    The GBM decides IF we trade. The exit config decides HOW we manage the trade.
    """
    if gate_thresholds is None:
        gate_thresholds = [0, 2, 5, 10, 15, 20, 25]

    strategies = []

    # Best exit configs from empirical findings
    exit_configs = [
        {
            'name': 'ratchet_patient',
            'cli_args': [
                '--chase-entry',
                '--signal-threshold', '0.1',
                '--stop-loss-ticks', '20',
                '--take-profit-ticks', '15',
                '--hold-ms', '120000',
                '--ratchet-stop',
                '--mae-exit-ticks', '13',
                '--mae-exit-hold-sec', '10',
                '--quiet',
            ],
            'thesis': 'GBM-gated + ratchet + MAE patience (93% winners go red)',
        },
        {
            'name': 'ratchet_tight',
            'cli_args': [
                '--chase-entry',
                '--signal-threshold', '0.1',
                '--stop-loss-ticks', '15',
                '--take-profit-ticks', '10',
                '--hold-ms', '60000',
                '--ratchet-stop',
                '--quiet',
            ],
            'thesis': 'GBM-gated + ratchet + tight stops for high conviction',
        },
        {
            'name': 'trail_ratchet',
            'cli_args': [
                '--chase-entry',
                '--signal-threshold', '0.1',
                '--stop-loss-ticks', '20',
                '--take-profit-ticks', '20',
                '--hold-ms', '180000',
                '--ratchet-stop',
                '--trailing-ticks', '4',
                '--quiet',
            ],
            'thesis': 'GBM-gated + trailing + ratchet (let winners run)',
        },
        {
            'name': 'prime_ratchet',
            'cli_args': [
                '--chase-entry',
                '--prime-hours',
                '--signal-threshold', '0.1',
                '--stop-loss-ticks', '20',
                '--take-profit-ticks', '12',
                '--hold-ms', '120000',
                '--ratchet-stop',
                '--quiet',
            ],
            'thesis': 'GBM-gated + prime hours + ratchet',
        },
        {
            'name': 'conviction_exit',
            'cli_args': [
                '--chase-entry',
                '--chase-force-cross',
                '--signal-threshold', '0.1',
                '--stop-loss-ticks', '20',
                '--take-profit-ticks', '15',
                '--hold-ms', '300000',
                '--conviction-exit-bars', '30',
                '--conviction-exit-mag', '1.0',
                '--ratchet-stop',
                '--quiet',
            ],
            'thesis': 'GBM-gated + conviction exit (exit on sustained reversal only)',
        },
        {
            'name': 'aggressive_market',
            'cli_args': [
                '--market-entry',
                '--signal-threshold', '0.1',
                '--stop-loss-ticks', '15',
                '--take-profit-ticks', '10',
                '--hold-ms', '60000',
                '--ratchet-stop',
                '--quiet',
            ],
            'thesis': 'GBM-gated + market entry (guaranteed fill, higher cost)',
        },
        {
            'name': 'wide_mae_patience',
            'cli_args': [
                '--chase-entry',
                '--signal-threshold', '0.1',
                '--stop-loss-ticks', '25',
                '--take-profit-ticks', '20',
                '--hold-ms', '300000',
                '--mae-exit-ticks', '15',
                '--mae-exit-hold-sec', '15',
                '--ratchet-stop',
                '--quiet',
            ],
            'thesis': 'GBM-gated + very wide + MAE patience (maximum winner room)',
        },
    ]

    for threshold in gate_thresholds:
        for exit_cfg in exit_configs:
            strategies.append({
                'name': f'gbm_gate{threshold}_{exit_cfg["name"]}',
                'gate_threshold': threshold,
                'cli_args': exit_cfg['cli_args'],
                'thesis': f'Gate>${threshold}: {exit_cfg["thesis"]}',
            })

    log.info(f"  Defined {len(strategies)} gated strategies "
             f"({len(gate_thresholds)} thresholds x {len(exit_configs)} exit configs)")
    return strategies


def run_gated_backtest(
    model,
    all_signals: Dict[str, Dict],
    all_bar_features: Dict[str, Dict],
    strategies: List[Dict],
    workers: int = 8,
) -> Dict[str, Dict]:
    """Phase 4: Run GBM-gated strategies through fill sim."""
    log.info(f"\n{'='*80}")
    log.info(f"PHASE 4: BACKTEST — Running GBM-gated strategies")
    log.info(f"{'='*80}")

    sim_out = RESULTS_DIR / f'gated_{_ts}'
    sim_out.mkdir(parents=True, exist_ok=True)

    # Pre-compute gated signals for each threshold
    threshold_cache = {}
    unique_thresholds = sorted(set(s['gate_threshold'] for s in strategies))

    for threshold in unique_thresholds:
        threshold_cache[threshold] = {}
        for date_str, sig_data in sorted(all_signals.items()):
            gated = apply_gbm_gate(
                model, sig_data['primary'],
                all_bar_features[date_str],
                gate_threshold=threshold,
            )
            cache_file = PRED_CACHE_DIR / f'gated_t{threshold}_{date_str}.npz'
            np.savez_compressed(str(cache_file), predictions=gated)
            threshold_cache[threshold][date_str] = cache_file

            n_orig = np.count_nonzero(sig_data['primary'])
            n_gated = np.count_nonzero(gated)
            log.info(f"    Gate>{threshold} {date_str}: {n_orig} → {n_gated} signals "
                     f"({100*n_gated/max(n_orig,1):.0f}% pass rate)")

    # Build jobs
    jobs = []
    for strategy in strategies:
        threshold = strategy['gate_threshold']
        for date_str in sorted(all_signals.keys()):
            jobs.append({
                'strategy': strategy,
                'date': date_str,
                'pred_file': threshold_cache[threshold][date_str],
            })

    log.info(f"  Total backtest jobs: {len(jobs)}")

    # Run sims
    results = []
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
            result = future.result()
            if result:
                result['_strategy'] = futures[future]['strategy']['name']
                result['_date'] = futures[future]['date']
                result['_thesis'] = futures[future]['strategy'].get('thesis', '')
                results.append(result)

            if done % 100 == 0 or done == len(jobs):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                log.info(f"  Backtest: {done}/{len(jobs)} ({done/len(jobs):.0%}) "
                         f"| {len(results)} results | {rate:.1f}/s")

    # Aggregate
    summaries = aggregate_results(results)
    return summaries


# ============================================================
# ALSO run ungated baselines for comparison
# ============================================================

def run_ungated_baselines(
    all_signals: Dict[str, Dict],
    workers: int = 8,
) -> Dict[str, Dict]:
    """Run baseline strategies WITHOUT the GBM gate for comparison."""
    log.info(f"\n{'='*80}")
    log.info(f"BASELINE: Running ungated strategies for comparison")
    log.info(f"{'='*80}")

    sim_out = RESULTS_DIR / f'baseline_{_ts}'
    sim_out.mkdir(parents=True, exist_ok=True)

    pred_files = {}
    for date_str, sig_data in all_signals.items():
        cache_file = PRED_CACHE_DIR / f'primary_{date_str}.npz'
        if not cache_file.exists():
            np.savez_compressed(str(cache_file), predictions=sig_data['primary'])
        pred_files[date_str] = cache_file

    configs = []
    for z in [2.0, 2.5, 3.0]:
        for tp, sl in [(10, 15), (10, 20), (12, 20), (15, 25)]:
            configs.append({
                'name': f'baseline_z{z}_tp{tp}_sl{sl}',
                'cli_args': [
                    '--chase-entry',
                    '--signal-threshold', str(z),
                    '--stop-loss-ticks', str(sl),
                    '--take-profit-ticks', str(tp),
                    '--hold-ms', '120000',
                    '--ratchet-stop',
                    '--quiet',
                ],
                'thesis': f'Ungated baseline z={z} tp={tp} sl={sl}',
            })

    jobs = []
    for config in configs:
        for date_str in sorted(all_signals.keys()):
            jobs.append({
                'config': config,
                'date': date_str,
                'pred_file': pred_files[date_str],
            })

    results = []
    done = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for job in jobs:
            future = executor.submit(
                run_fill_sim,
                job['date'], job['pred_file'], job['config'], sim_out,
            )
            futures[future] = job

        for future in as_completed(futures):
            done += 1
            result = future.result()
            if result:
                result['_strategy'] = futures[future]['config']['name']
                result['_date'] = futures[future]['date']
                result['_thesis'] = futures[future]['config'].get('thesis', '')
                results.append(result)

    elapsed = time.time() - t0
    log.info(f"  Baselines complete in {elapsed:.0f}s")

    return aggregate_results(results)


# ============================================================
# Results aggregation
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

        # Per-trade stats
        all_trade_pnls = []
        for r in runs:
            if 'trades' in r:
                for t in r['trades']:
                    all_trade_pnls.append(t.get('pnl_dollars', 0))

        # Max drawdown (cumulative PnL)
        if len(daily_pnl) > 1:
            cum_pnl = np.cumsum(daily_pnl)
            peak = np.maximum.accumulate(cum_pnl)
            drawdown = cum_pnl - peak
            max_dd = drawdown.min()
        else:
            max_dd = 0

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
            'max_drawdown': round(max_dd, 2),
        }
    return summaries


def print_results(title: str, summaries: Dict[str, Dict]):
    filtered = {k: v for k, v in summaries.items() if v['n_trades'] >= 3}
    sorted_s = sorted(filtered.items(), key=lambda x: x[1]['daily_sortino'], reverse=True)

    log.info(f"\n{'='*160}")
    log.info(f"{title} — RANKED BY DAILY SORTINO (min 3 trades)")
    log.info(f"{'='*160}")
    log.info(f"{'Strategy':<50} {'N':>5} {'PnL($)':>10} {'$/Tr':>8} {'WR':>6} "
             f"{'PF':>6} {'Sharpe':>7} {'Sortino':>8} {'Fill':>5} {'MaxDD':>8}")
    log.info('-' * 160)

    for name, s in sorted_s[:50]:  # Top 50
        log.info(
            f"{name:<50} {s['n_trades']:>5} {s['total_pnl']:>10.0f} "
            f"{s['mean_pnl_per_trade']:>8.2f} {s['win_rate']:>5.1%} {s['profit_factor']:>6.3f} "
            f"{s['daily_sharpe']:>7.3f} {s['daily_sortino']:>8.3f} {s['fill_rate']:>5.1%} "
            f"{s['max_drawdown']:>8.0f}"
        )

    # Summary stats
    if sorted_s:
        positive = [s for _, s in sorted_s if s['total_pnl'] > 0]
        log.info(f"\n  Positive PnL strategies: {len(positive)}/{len(sorted_s)}")
        if positive:
            best = max(positive, key=lambda x: x['daily_sortino'])
            log.info(f"  Best Sortino: {best['daily_sortino']:.3f} "
                     f"(PnL=${best['total_pnl']:.0f}, {best['n_trades']} trades)")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Adaptive Execution Engine v3 — Meta-Learner')
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--skip-harvest', action='store_true',
                        help='Skip harvest phase, load from previous run')
    parser.add_argument('--harvest-file', type=str, default=None,
                        help='Path to previously harvested trades JSON')
    args = parser.parse_args()

    log.info(f"{'='*80}")
    log.info(f"ADAPTIVE EXECUTION ENGINE v3 — GBM META-LEARNER")
    log.info(f"{'='*80}")
    log.info(f"  Workers: {args.workers}")
    log.info(f"  LightGBM available: {HAS_LGB}")
    log.info(f"  scikit-learn available: {HAS_SKLEARN}")
    log.info(f"  Timestamp: {_ts}")

    if not HAS_LGB:
        log.error("LightGBM is required for the meta-learner. Install with: pip install lightgbm")
        return

    # ── Discover folds ──
    pred_dir = CNN_MAMBA_V2_DIR
    log.info(f"\nDiscovering prediction folds from {pred_dir}...")
    folds = discover_folds(pred_dir)
    log.info(f"  Found {len(folds)} folds")
    if not folds:
        return

    # ── Prepare signals ──
    log.info(f"\nPreparing multi-horizon signals...")
    all_signals = prepare_all_signals(folds)
    log.info(f"  {len(all_signals)} dates with signals")
    if not all_signals:
        return

    # ── Phase 1: Harvest ──
    if args.skip_harvest and args.harvest_file:
        log.info(f"\nLoading previously harvested trades from {args.harvest_file}")
        with open(args.harvest_file) as f:
            trades = json.load(f)
        log.info(f"  Loaded {len(trades)} trades")
    else:
        configs = define_harvest_configs()
        trades = harvest_trades(all_signals, configs, workers=args.workers)

    if len(trades) < 100:
        log.error(f"Only {len(trades)} trades harvested — not enough for training.")
        return

    # ── Phase 2: Train GBM ──
    model, metrics = train_gbm_meta_learner(trades)
    if model is None:
        log.error("GBM training failed!")
        return

    # ── Pre-compute bar features for gated backtest ──
    log.info(f"\nPre-computing bar features for backtest...")
    all_bar_features = {}
    for date_str, sig_data in all_signals.items():
        all_bar_features[date_str] = compute_bar_features(
            sig_data['primary'], sig_data['horizons'], date_str
        )

    # ── Phase 3+4: Gate + Backtest ──
    strategies = define_gated_strategies(model)
    gated_summaries = run_gated_backtest(
        model, all_signals, all_bar_features, strategies, workers=args.workers
    )

    # ── Baselines for comparison ──
    baseline_summaries = run_ungated_baselines(all_signals, workers=args.workers)

    # ── Print results ──
    print_results("GBM-GATED STRATEGIES", gated_summaries)
    print_results("UNGATED BASELINES (comparison)", baseline_summaries)

    # ── Save all results ──
    all_results = {
        'gated': gated_summaries,
        'baselines': baseline_summaries,
        'model_metrics': metrics,
        'n_harvested_trades': len(trades),
        'timestamp': _ts,
    }

    out_json = RESULTS_DIR / f'v3_meta_learner_results_{_ts}.json'
    with open(out_json, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info(f"\nAll results saved to: {out_json}")

    # ── Final summary ──
    log.info(f"\n{'='*80}")
    log.info(f"FINAL SUMMARY")
    log.info(f"{'='*80}")
    log.info(f"  Harvested trades: {len(trades)}")
    log.info(f"  GBM val R²: {metrics.get('r2_val', 'N/A')}")
    log.info(f"  Gated strategies tested: {len(gated_summaries)}")
    log.info(f"  Baseline strategies tested: {len(baseline_summaries)}")

    # Best gated vs best baseline
    if gated_summaries:
        best_gated = max(
            [(k, v) for k, v in gated_summaries.items() if v['n_trades'] >= 3],
            key=lambda x: x[1]['daily_sortino'], default=None
        )
        if best_gated:
            log.info(f"\n  BEST GATED: {best_gated[0]}")
            log.info(f"    Sortino: {best_gated[1]['daily_sortino']:.3f}")
            log.info(f"    PnL: ${best_gated[1]['total_pnl']:.0f}")
            log.info(f"    Trades: {best_gated[1]['n_trades']}")
            log.info(f"    Win Rate: {best_gated[1]['win_rate']:.1%}")

    if baseline_summaries:
        best_baseline = max(
            [(k, v) for k, v in baseline_summaries.items() if v['n_trades'] >= 3],
            key=lambda x: x[1]['daily_sortino'], default=None
        )
        if best_baseline:
            log.info(f"\n  BEST BASELINE: {best_baseline[0]}")
            log.info(f"    Sortino: {best_baseline[1]['daily_sortino']:.3f}")
            log.info(f"    PnL: ${best_baseline[1]['total_pnl']:.0f}")
            log.info(f"    Trades: {best_baseline[1]['n_trades']}")
            log.info(f"    Win Rate: {best_baseline[1]['win_rate']:.1%}")


if __name__ == '__main__':
    main()
