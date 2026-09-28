#!/usr/bin/env python3
"""
Vol-Gated Composite Strategy Test — Saturn Version
=====================================================
Tailored for Saturn's actual data layout:
  - signal_predictions/composite_*.npz (50 days, Jul-Sep 2025)
  - signal_predictions/lgbm_5m_*.npz, lgbm_15m_*.npz etc. (85 days)
  - signal_predictions/meta_global_10s_*.npz (80 days)
  - alpha_discovery/results/vol_pred_rvol_10s_20260227_201022.json (fold ICs)

Strategy:
  1. Build composite mean z-score from all available signals (>=40 days coverage)
  2. Regenerate vol predictions inline (LightGBM rvol@10s walk-forward)
     — saves bar-level predictions to data/processed/vol_pred_cache/
  3. Gate: only take composite signal trades when vol pred is in top 50/30/20/10%
  4. Sim with Python market-order sim (300s, 600s, 900s hold times)
  5. Report PnL, Sharpe, trade count, win rate, both-halves-positive

IC=0.674 vol prediction signal should filter noisy low-vol trades
and keep the high-vol-move trades that actually overcome spread+commission.

Usage:
    python alpha_discovery/vol_gated_composite_saturn.py
    python alpha_discovery/vol_gated_composite_saturn.py --regen-vol
    python alpha_discovery/vol_gated_composite_saturn.py --n-days 50
    python alpha_discovery/vol_gated_composite_saturn.py --use-composite-only
"""

import argparse
import gc
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr

# ============================================================================
# Paths and constants
# ============================================================================

# Detect Linux/Saturn
_linux_root = Path.home() / 'lvl3quant'
LVL3_ROOT = _linux_root if _linux_root.exists() else Path(__file__).resolve().parent.parent

MBO_DIR = LVL3_ROOT / 'data' / 'processed' / 'mbo_features_cache'
SIG_DIR = LVL3_ROOT / 'data' / 'processed' / 'signal_predictions'
VOL_CACHE_DIR = LVL3_ROOT / 'data' / 'processed' / 'vol_pred_cache'
RESULTS_DIR = LVL3_ROOT / 'alpha_discovery' / 'results'

VOL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ES constants
TICK = 0.25
TICK_VAL = 12.50
COMM_TICKS = 3.00 / TICK_VAL  # 0.24 ticks commission
BARS_PER_SEC = 10             # 100ms bars

# Vol prediction (rvol @ 10s)
VOL_HORIZON_BARS = 1000
VOL_SUBSAMPLE = 100
EXCLUDE_FEATURES = [0, 3, 8, 9]  # mid, microprice, bid, ask
MIN_TRAIN_DAYS_VOL = 15

# Simulation parameters
VOL_GATES = [0.50, 0.70, 0.80, 0.90]   # only trade when vol_pred rank >= this
HOLD_SECONDS = [300, 600, 900]          # 5min, 10min, 15min
THRESHOLDS = [0.3, 0.5, 0.75, 1.0, 1.5, 2.0]  # composite signal magnitude threshold
COOLDOWN_BARS = 50                       # 5s cooldown between trades

logging.basicConfig(
    format='%(asctime)s [vol_gated] %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger('vol_gated')


# ============================================================================
# Data loading
# ============================================================================

def find_mbo_days() -> List[Tuple[str, Path]]:
    files = sorted(MBO_DIR.glob('*_mbo_features.npz'))
    return [(f.stem.replace('_mbo_features', ''), f) for f in files]


def load_mid_spread(fpath: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    try:
        data = np.load(str(fpath))
        raw = data['mbo_features']
        mid = raw[:, 0].astype(np.float32).copy()
        spread = raw[:, 1].astype(np.float32).copy()
        mask = np.isnan(mid)
        if mask.any():
            first_valid = int(np.argmax(~mask))
            mid[:first_valid] = mid[first_valid] if not mask[first_valid] else 5000.0
        del raw
        return mid, spread
    except Exception as e:
        logger.warning(f'Failed loading {fpath.name}: {e}')
        return None


def load_mbo_features(fpath: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load (features_336, mid) for vol prediction training."""
    try:
        data = np.load(str(fpath))
        raw = data['mbo_features']
        mid = raw[:, 0].astype(np.float32).copy()
        mask = np.isnan(mid)
        if mask.any():
            first_valid = int(np.argmax(~mask))
            mid[:first_valid] = mid[first_valid] if not mask[first_valid] else 5000.0
        keep = [i for i in range(raw.shape[1]) if i not in EXCLUDE_FEATURES]
        features = raw[:, keep].astype(np.float32)
        np.nan_to_num(features, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        del raw
        return features, mid
    except Exception as e:
        logger.warning(f'Failed loading features {fpath.name}: {e}')
        return None


# ============================================================================
# Composite signal builder
# ============================================================================

def build_composite_signals(all_dates: List[str], min_days: int = 40) -> Dict[str, np.ndarray]:
    """Build mean z-score composite from all available signal predictions."""
    logger.info('Discovering signal prediction files...')

    # Map signal_name -> set of dates
    sig_map = defaultdict(set)
    for f in SIG_DIR.glob('*.npz'):
        stem = f.stem
        # Try to split at date pattern
        for sep_idx, part in enumerate(stem.split('_')):
            if part.startswith('2025') or part.startswith('2024') or part.startswith('2026'):
                sig_name = '_'.join(stem.split('_')[:sep_idx])
                date = '_'.join(stem.split('_')[sep_idx:])
                if sig_name:
                    sig_map[sig_name].add(date)
                break

    available = {k: v for k, v in sig_map.items() if len(v) >= min_days}
    logger.info(f'Signals with >={min_days} days: {len(available)}')
    for name in sorted(available.keys()):
        logger.info(f'  {name}: {len(available[name])} days')

    if not available:
        logger.warning('No signals found — check signal_predictions directory')
        return {}

    # Build composite per date
    composite = {}
    for date in all_dates:
        mbo_path = MBO_DIR / f'{date}_mbo_features.npz'
        if not mbo_path.exists():
            continue
        try:
            n_bars = int(np.load(str(mbo_path))['mbo_features'].shape[0])
        except Exception:
            continue

        sig_sum = np.zeros(n_bars, dtype=np.float64)
        n_loaded = 0

        for sig_name in available:
            # Try different filename patterns
            candidates = [
                SIG_DIR / f'{sig_name}_{date}.npz',
            ]
            for path in candidates:
                if not path.exists():
                    continue
                try:
                    arr = np.load(str(path))
                    # Try common key names
                    for key in ['predictions', 'signal', 'pred', 'data']:
                        if key in arr:
                            preds = arr[key].astype(np.float64).ravel()
                            break
                    else:
                        preds = arr[arr.files[0]].astype(np.float64).ravel()

                    if len(preds) > n_bars:
                        preds = preds[:n_bars]
                    elif len(preds) < n_bars:
                        preds = np.pad(preds, (0, n_bars - len(preds)))

                    # Z-score normalize (clip extreme values)
                    std = np.std(preds)
                    if std > 1e-10:
                        preds = np.clip(preds / std, -5, 5)

                    np.nan_to_num(preds, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                    sig_sum += preds
                    n_loaded += 1
                    break
                except Exception as ex:
                    logger.debug(f'  Skip {sig_name} {date}: {ex}')

        if n_loaded > 0:
            composite[date] = (sig_sum / n_loaded).astype(np.float32)

    n_sigs = len(available)
    n_days = len(composite)
    logger.info(f'Composite built: {n_days} days, {n_sigs} signals averaged')
    return composite


# ============================================================================
# Vol prediction (LightGBM walk-forward, rvol@10s)
# ============================================================================

def compute_rvol_target(mid: np.ndarray, horizon: int = VOL_HORIZON_BARS) -> np.ndarray:
    N = len(mid)
    mid_d = mid.astype(np.float64)
    returns = np.diff(mid_d, prepend=mid_d[0]) / np.where(mid_d > 0, mid_d, 1.0)
    fwd_rvol = np.full(N, np.nan, dtype=np.float64)
    r = returns[1:]
    if len(r) < horizon:
        return fwd_rvol
    cs = np.cumsum(r)
    cs2 = np.cumsum(r ** 2)
    sa = np.concatenate(([0.0], cs))
    s2a = np.concatenate(([0.0], cs2))
    n_valid = N - horizon
    idx = np.arange(n_valid)
    s = sa[idx + horizon] - sa[idx]
    s2 = s2a[idx + horizon] - s2a[idx]
    mean_r = s / horizon
    var = s2 / horizon - mean_r ** 2
    var = np.clip(var, 0, None)
    fwd_rvol[:n_valid] = np.sqrt(var)
    return fwd_rvol


def get_vol_predictions(
    all_days: List[Tuple[str, Path]],
    regen: bool = False,
) -> Dict[str, np.ndarray]:
    """Walk-forward vol predictions (rvol@10s). Caches per-bar predictions."""
    try:
        import lightgbm as lgb
    except ImportError:
        logger.error('lightgbm not installed — cannot generate vol predictions')
        sys.exit(1)

    logger.info('=' * 60)
    logger.info('Vol Prediction: rvol@10s LightGBM walk-forward')
    logger.info('=' * 60)

    vol_preds_by_date: Dict[str, np.ndarray] = {}
    cached_count = 0

    # Load day data for training (subsampled)
    day_data = []
    for date_str, fpath in all_days:
        cache_path = VOL_CACHE_DIR / f'volpred_rvol_10s_{date_str}.npz'

        # If cached, load and skip training for this day
        if not regen and cache_path.exists():
            try:
                d = np.load(str(cache_path))
                vol_preds_by_date[date_str] = d['vol_pred']
                cached_count += 1
                continue
            except Exception:
                pass

        result = load_mbo_features(fpath)
        if result is None:
            continue
        features, mid = result

        target = compute_rvol_target(mid, VOL_HORIZON_BARS)
        indices = np.arange(VOL_SUBSAMPLE - 1, len(features), VOL_SUBSAMPLE)
        X = features[indices]
        y = target[indices]
        valid = ~np.isnan(y) & np.all(np.isfinite(X), axis=1)
        X_sub = X[valid].astype(np.float32)
        y_sub = y[valid].astype(np.float32)

        day_data.append({
            'date': date_str,
            'fpath': fpath,
            'X_sub': X_sub,
            'y_sub': y_sub,
            'n_bars': len(mid),
        })
        del features, mid, target
        gc.collect()

    if cached_count > 0:
        logger.info(f'Loaded {cached_count} cached vol predictions')

    if not day_data:
        logger.info('All days loaded from cache.')
        return vol_preds_by_date

    logger.info(f'Walk-forward on {len(day_data)} uncached days (cached: {cached_count})')

    lgbm_params = {
        'objective': 'regression',
        'metric': 'mse',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': 6,
        'min_child_samples': 200,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42,
    }

    all_ics = []
    t_start = time.time()

    for test_idx in range(MIN_TRAIN_DAYS_VOL, len(day_data)):
        train_days = day_data[:test_idx]
        test_day = day_data[test_idx]
        date_str = test_day['date']

        X_train = np.vstack([d['X_sub'] for d in train_days]).astype(np.float32)
        y_train = np.concatenate([d['y_sub'] for d in train_days]).astype(np.float32)
        np.nan_to_num(X_train, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        X_test_sub = test_day['X_sub'].astype(np.float32)
        np.nan_to_num(X_test_sub, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        dtrain = lgb.Dataset(X_train, label=y_train)
        model = lgb.train(lgbm_params, dtrain, num_boost_round=200)

        # Predict on full-resolution MBO features
        result = load_mbo_features(test_day['fpath'])
        if result is None:
            del dtrain, model
            continue
        features_full, _ = result
        np.nan_to_num(features_full, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        vol_pred_full = model.predict(features_full).astype(np.float32)

        # IC on subsampled test
        preds_sub = model.predict(X_test_sub)
        if len(preds_sub) > 0 and len(test_day['y_sub']) > 10:
            try:
                ic, _ = spearmanr(preds_sub, test_day['y_sub'])
                if np.isfinite(ic):
                    all_ics.append(ic)
            except Exception:
                pass

        if test_idx % 5 == 0 or test_idx == len(day_data) - 1:
            mean_ic = float(np.nanmean(all_ics)) if all_ics else float('nan')
            elapsed = time.time() - t_start
            logger.info(f'  [{test_idx+1}/{len(day_data)}] {date_str}  '
                       f'mean_IC={mean_ic:+.4f}  elapsed={elapsed:.0f}s')

        # Cache
        cache_path = VOL_CACHE_DIR / f'volpred_rvol_10s_{date_str}.npz'
        np.savez_compressed(str(cache_path), vol_pred=vol_pred_full)
        vol_preds_by_date[date_str] = vol_pred_full

        del X_train, y_train, dtrain, model, features_full
        gc.collect()

    if all_ics:
        ics = np.array(all_ics)
        mean_ic = float(np.mean(ics))
        std_ic = float(np.std(ics))
        t_stat = mean_ic / (std_ic / np.sqrt(len(ics))) if std_ic > 0 else 0.0
        logger.info(f'\nVol Prediction Summary:')
        logger.info(f'  n_folds={len(ics)}  mean_IC={mean_ic:+.4f}  '
                   f'std={std_ic:.4f}  t={t_stat:.2f}  '
                   f'pct+={np.mean(ics > 0) * 100:.1f}%')

    return vol_preds_by_date


# ============================================================================
# Rolling vol rank (causal, fast)
# ============================================================================

def compute_vol_rank_fast(vol_arr: np.ndarray, window: int = 5000) -> np.ndarray:
    """Causal rolling percentile rank — approximated via stride sampling."""
    N = len(vol_arr)
    out = np.full(N, 0.5, dtype=np.float32)
    x = np.nan_to_num(vol_arr.astype(np.float64), nan=0.0)
    stride = max(1, window // 50)
    for i in range(window - 1, N, stride):
        w_start = max(0, i - window + 1)
        wnd = x[w_start: i + 1]
        pct = float(np.sum(wnd < x[i])) / max(len(wnd), 1)
        fill_end = min(i + stride, N)
        out[i: fill_end] = np.float32(pct)
    return out


# ============================================================================
# Market order simulator
# ============================================================================

def sim_day_gated(
    mid: np.ndarray,
    spread: np.ndarray,
    composite: np.ndarray,
    vol_pred: np.ndarray,
    vol_gate_pct: float,
    thresh: float,
    hold_bars: int,
    cooldown: int = COOLDOWN_BARS,
    vol_rank_window: int = 5000,
) -> Tuple[float, int, int]:
    """Simulate one day with vol-gated composite signal."""
    n = min(len(mid), len(composite), len(vol_pred))
    if n < 200:
        return 0.0, 0, 0

    vol_rank = compute_vol_rank_fast(vol_pred[:n], window=vol_rank_window)

    total_pnl = 0.0
    trades = 0
    wins = 0
    last_exit = -cooldown
    in_pos = False
    entry_price = 0.0
    direction = 0
    entry_bar = 0

    m = mid[:n]
    sp = spread[:n]
    comp = composite[:n]

    for i in range(n):
        if in_pos:
            if direction == 1:
                unr = (m[i] - entry_price) / TICK
            else:
                unr = (entry_price - m[i]) / TICK
            if i - entry_bar >= hold_bars:
                pnl = unr - sp[i] / 2.0 / TICK - COMM_TICKS
                total_pnl += pnl
                trades += 1
                wins += 1 if pnl > 0 else 0
                in_pos = False
                last_exit = i
        else:
            if (vol_rank[i] >= vol_gate_pct and
                    abs(comp[i]) > thresh and
                    (i - last_exit) >= cooldown):
                d = 1 if comp[i] > 0 else -1
                entry_price = m[i] + d * sp[i] / 2.0
                direction = d
                in_pos = True
                entry_bar = i

    if in_pos:
        i = n - 1
        if direction == 1:
            unr = (m[i] - entry_price) / TICK
        else:
            unr = (entry_price - m[i]) / TICK
        pnl = unr - sp[i] / 2.0 / TICK - COMM_TICKS
        total_pnl += pnl
        trades += 1
        wins += 1 if pnl > 0 else 0

    return total_pnl * TICK_VAL, trades, wins


def sim_day_baseline(
    mid: np.ndarray,
    spread: np.ndarray,
    composite: np.ndarray,
    thresh: float,
    hold_bars: int,
    cooldown: int = COOLDOWN_BARS,
) -> Tuple[float, int, int]:
    """Baseline sim — no vol gate."""
    n = min(len(mid), len(composite))
    if n < 200:
        return 0.0, 0, 0

    total_pnl = 0.0
    trades = 0
    wins = 0
    last_exit = -cooldown
    in_pos = False
    entry_price = 0.0
    direction = 0
    entry_bar = 0

    m = mid[:n]
    sp = spread[:n]
    comp = composite[:n]

    for i in range(n):
        if in_pos:
            if direction == 1:
                unr = (m[i] - entry_price) / TICK
            else:
                unr = (entry_price - m[i]) / TICK
            if i - entry_bar >= hold_bars:
                pnl = unr - sp[i] / 2.0 / TICK - COMM_TICKS
                total_pnl += pnl
                trades += 1
                wins += 1 if pnl > 0 else 0
                in_pos = False
                last_exit = i
        else:
            if abs(comp[i]) > thresh and (i - last_exit) >= cooldown:
                d = 1 if comp[i] > 0 else -1
                entry_price = m[i] + d * sp[i] / 2.0
                direction = d
                in_pos = True
                entry_bar = i

    if in_pos:
        i = n - 1
        if direction == 1:
            unr = (m[i] - entry_price) / TICK
        else:
            unr = (entry_price - m[i]) / TICK
        pnl = unr - sp[i] / 2.0 / TICK - COMM_TICKS
        total_pnl += pnl
        trades += 1
        wins += 1 if pnl > 0 else 0

    return total_pnl * TICK_VAL, trades, wins


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Vol-Gated Composite Strategy Test (Saturn)')
    parser.add_argument('--n-days', type=int, default=0)
    parser.add_argument('--regen-vol', action='store_true', help='Regenerate vol predictions')
    parser.add_argument('--no-baseline', action='store_true', help='Skip baseline sim')
    parser.add_argument('--min-signal-days', type=int, default=40,
                        help='Min days a signal must be available to be included in composite')
    args = parser.parse_args()

    ts_start = time.strftime('%Y%m%d_%H%M%S')
    logger.info('=' * 70)
    logger.info('Vol-Gated Composite Strategy Test — Saturn')
    logger.info('=' * 70)
    logger.info(f'LVL3_ROOT: {LVL3_ROOT}')
    logger.info(f'MBO_DIR:   {MBO_DIR}')
    logger.info(f'SIG_DIR:   {SIG_DIR}')
    logger.info(f'VOL_CACHE: {VOL_CACHE_DIR}')
    logger.info('')

    # Discover MBO days
    all_days = find_mbo_days()
    if not all_days:
        logger.error(f'No MBO feature files found in {MBO_DIR}')
        sys.exit(1)

    if args.n_days > 0:
        all_days = all_days[:args.n_days]

    all_dates = [d for d, _ in all_days]
    logger.info(f'MBO days found: {len(all_days)} ({all_dates[0]} .. {all_dates[-1]})')

    # Step 1: Build composite signals
    logger.info('')
    logger.info('Step 1: Building composite signals...')
    composite_signals = build_composite_signals(all_dates, min_days=args.min_signal_days)

    if not composite_signals:
        logger.error('No composite signals built — check signal_predictions directory')
        sys.exit(1)

    # Step 2: Get vol predictions
    logger.info('')
    logger.info('Step 2: Getting vol predictions (rvol@10s)...')
    vol_preds = get_vol_predictions(all_days, regen=args.regen_vol)

    # Find common dates
    comp_dates = set(composite_signals.keys())
    vol_dates = set(vol_preds.keys())
    common_dates = sorted(comp_dates & vol_dates)

    logger.info('')
    logger.info(f'Composite signal coverage: {len(comp_dates)} days')
    logger.info(f'Vol prediction coverage:   {len(vol_dates)} days')
    logger.info(f'Common (both available):   {len(common_dates)} days')

    if len(common_dates) < 15:
        logger.warning(f'Only {len(common_dates)} common days — results may be noisy')
        if len(common_dates) < 5:
            logger.error('Insufficient data. Run with --regen-vol or check data.')
            sys.exit(1)

    # Step 3: Load mid/spread
    logger.info('')
    logger.info('Step 3: Loading mid/spread...')
    mid_spread = {}
    for date, fpath in all_days:
        if date not in common_dates:
            continue
        result = load_mid_spread(fpath)
        if result is not None:
            mid_spread[date] = {'mid': result[0], 'spread': result[1]}

    sim_dates = [d for d in common_dates if d in mid_spread]
    logger.info(f'Simulation dates: {len(sim_dates)}')
    if not sim_dates:
        logger.error('No dates available for simulation')
        sys.exit(1)

    h_mid = len(sim_dates) // 2
    first_half = sim_dates[:h_mid]
    second_half = sim_dates[h_mid:]
    logger.info(f'  H1: {first_half[0]} .. {first_half[-1]} ({len(first_half)} days)')
    logger.info(f'  H2: {second_half[0]} .. {second_half[-1]} ({len(second_half)} days)')

    # Step 4: Baseline (no vol gate)
    baseline_results = []
    if not args.no_baseline:
        logger.info('')
        logger.info('=' * 70)
        logger.info('BASELINE: No Vol Gate')
        logger.info('=' * 70)

        for thresh in THRESHOLDS:
            for hold_s in HOLD_SECONDS:
                hold_bars = hold_s * BARS_PER_SEC
                day_pnls = []
                tot_trades = tot_wins = 0

                for date in sim_dates:
                    pnl, t, w = sim_day_baseline(
                        mid_spread[date]['mid'],
                        mid_spread[date]['spread'],
                        composite_signals[date],
                        thresh, hold_bars,
                    )
                    day_pnls.append(pnl)
                    tot_trades += t
                    tot_wins += w

                arr = np.array(day_pnls)
                total = float(arr.sum())
                std = float(arr.std()) if len(arr) > 1 else 1.0
                sharpe = float(arr.mean() / max(std, 0.01) * np.sqrt(252))
                h1 = float(arr[:h_mid].sum())
                h2 = float(arr[h_mid:].sum())

                baseline_results.append({
                    'vol_gate': 'NONE',
                    'thresh': thresh,
                    'hold_s': hold_s,
                    'total_pnl': total,
                    'h1_pnl': h1,
                    'h2_pnl': h2,
                    'sharpe': sharpe,
                    'trades': tot_trades,
                    'win_rate': tot_wins / max(tot_trades, 1),
                    'profit_days': int((arr > 0).sum()),
                    'n_days': len(day_pnls),
                    'both_positive': h1 > 0 and h2 > 0,
                })

        baseline_results.sort(key=lambda x: x['total_pnl'], reverse=True)
        logger.info(f"\n{'Config':<35} {'Sharpe':>7} {'PnL':>10} {'H1':>8} {'H2':>8} "
                   f"{'Trades':>7} {'Win%':>6} {'Days+':>6}")
        logger.info('-' * 90)
        for r in baseline_results[:10]:
            cfg = f"NONE|t{r['thresh']}|h{r['hold_s']}s"
            both = ' BOTH+' if r['both_positive'] else ''
            logger.info(f"{cfg:<35} {r['sharpe']:>+7.2f} ${r['total_pnl']:>+9,.0f} "
                       f"${r['h1_pnl']:>+7,.0f} ${r['h2_pnl']:>+7,.0f} "
                       f"{r['trades']:>7} {r['win_rate']:>5.1%} "
                       f"{r['profit_days']:>3}/{r['n_days']}{both}")

    # Step 5: Vol-gated simulation
    logger.info('')
    logger.info('=' * 70)
    logger.info('VOL-GATED COMPOSITE SIM')
    logger.info(f'Gates: {[f"top {int((1-g)*100)}%" for g in VOL_GATES]}')
    logger.info(f'Holds: {HOLD_SECONDS}s  |  Thresholds: {THRESHOLDS}')
    logger.info('=' * 70)

    all_results = []
    total_combos = len(VOL_GATES) * len(THRESHOLDS) * len(HOLD_SECONDS)
    combo_idx = 0
    t_sim_start = time.time()

    for vol_gate in VOL_GATES:
        gate_label = f'top{int((1-vol_gate)*100)}pct'

        for thresh in THRESHOLDS:
            for hold_s in HOLD_SECONDS:
                hold_bars = hold_s * BARS_PER_SEC
                day_pnls = []
                tot_trades = tot_wins = 0

                for date in sim_dates:
                    pnl, t, w = sim_day_gated(
                        mid_spread[date]['mid'],
                        mid_spread[date]['spread'],
                        composite_signals[date],
                        vol_preds[date],
                        vol_gate_pct=vol_gate,
                        thresh=thresh,
                        hold_bars=hold_bars,
                    )
                    day_pnls.append(pnl)
                    tot_trades += t
                    tot_wins += w

                arr = np.array(day_pnls)
                total = float(arr.sum())
                std = float(arr.std()) if len(arr) > 1 else 1.0
                sharpe = float(arr.mean() / max(std, 0.01) * np.sqrt(252))
                h1 = float(arr[:h_mid].sum())
                h2 = float(arr[h_mid:].sum())

                all_results.append({
                    'vol_gate': vol_gate,
                    'vol_gate_label': gate_label,
                    'thresh': thresh,
                    'hold_s': hold_s,
                    'total_pnl': total,
                    'h1_pnl': h1,
                    'h2_pnl': h2,
                    'sharpe': sharpe,
                    'trades': tot_trades,
                    'win_rate': tot_wins / max(tot_trades, 1),
                    'profit_days': int((arr > 0).sum()),
                    'n_days': len(day_pnls),
                    'both_positive': h1 > 0 and h2 > 0,
                })

                combo_idx += 1
                if combo_idx % 12 == 0:
                    elapsed = time.time() - t_sim_start
                    pct = combo_idx / total_combos * 100
                    logger.info(f'  Sim progress: {combo_idx}/{total_combos} ({pct:.0f}%)  '
                               f'elapsed={elapsed:.0f}s')

    # Step 6: Report
    all_results.sort(key=lambda x: x['total_pnl'], reverse=True)

    logger.info('')
    logger.info('=' * 115)
    logger.info('TOP 20 VOL-GATED COMBOS (by total PnL)')
    logger.info('=' * 115)
    logger.info(f"{'Config':<45} {'Sharpe':>7} {'PnL':>10} {'H1':>8} {'H2':>8} "
               f"{'Trades':>7} {'Win%':>6} {'Days+':>6} {'Both':>5}")
    logger.info('-' * 115)
    for r in all_results[:20]:
        cfg = f"{r['vol_gate_label']}|t{r['thresh']}|h{r['hold_s']}s"
        both = 'YES' if r['both_positive'] else ''
        logger.info(f"{cfg:<45} {r['sharpe']:>+7.2f} ${r['total_pnl']:>+9,.0f} "
                   f"${r['h1_pnl']:>+7,.0f} ${r['h2_pnl']:>+7,.0f} "
                   f"{r['trades']:>7} {r['win_rate']:>5.1%} "
                   f"{r['profit_days']:>3}/{r['n_days']} {both:>5}")

    # Both-halves positive
    both_pos = [r for r in all_results if r['both_positive'] and r['total_pnl'] > 0]
    logger.info('')
    logger.info(f'BOTH-HALVES POSITIVE: {len(both_pos)} combos')
    for r in both_pos[:10]:
        cfg = f"{r['vol_gate_label']}|t{r['thresh']}|h{r['hold_s']}s"
        logger.info(f'  {cfg:<45} ${r["total_pnl"]:>+9,.0f}  H1=${r["h1_pnl"]:>+8,.0f}  '
                   f'H2=${r["h2_pnl"]:>+8,.0f}  Sharpe={r["sharpe"]:+.2f}  '
                   f'trades={r["trades"]}  win={r["win_rate"]:.1%}')

    # Per-gate best
    logger.info('')
    logger.info('BEST RESULT PER VOL GATE:')
    for gate in VOL_GATES:
        gr = [r for r in all_results if abs(r['vol_gate'] - gate) < 0.01]
        if gr:
            best = gr[0]
            pct = int((1 - gate) * 100)
            logger.info(f'  Top {pct:2d}%  → ${best["total_pnl"]:>+9,.0f}  '
                       f'Sharpe={best["sharpe"]:+.2f}  trades={best["trades"]}  '
                       f'config: t{best["thresh"]}|h{best["hold_s"]}s  '
                       f'both+={best["both_positive"]}')

    # Baseline comparison
    if baseline_results:
        bb = baseline_results[0]
        bg = all_results[0]
        delta = bg['total_pnl'] - bb['total_pnl']
        logger.info('')
        logger.info('BASELINE VS BEST VOL-GATED:')
        logger.info(f'  Baseline:    ${bb["total_pnl"]:>+9,.0f}  '
                   f'Sharpe={bb["sharpe"]:+.2f}  trades={bb["trades"]}')
        logger.info(f'  Vol-gated:   ${bg["total_pnl"]:>+9,.0f}  '
                   f'Sharpe={bg["sharpe"]:+.2f}  trades={bg["trades"]}  '
                   f'gate={bg["vol_gate_label"]}  t={bg["thresh"]}  h={bg["hold_s"]}s')
        logger.info(f'  Delta:       ${delta:>+9,.0f}  '
                   f'({"improvement" if delta > 0 else "regression"})')

    # Profitable configs summary
    n_profitable = sum(1 for r in all_results if r['total_pnl'] > 0)
    logger.info('')
    logger.info(f'SUMMARY: {n_profitable}/{len(all_results)} configs profitable  |  '
               f'{len(both_pos)} both-halves-positive')

    # Save results
    out_path = RESULTS_DIR / f'vol_gated_composite_{ts_start}.json'
    output = {
        'timestamp': ts_start,
        'n_sim_days': len(sim_dates),
        'sim_date_range': [sim_dates[0], sim_dates[-1]],
        'h1_dates': [first_half[0], first_half[-1]],
        'h2_dates': [second_half[0], second_half[-1]],
        'vol_gates_tested': VOL_GATES,
        'thresholds_tested': THRESHOLDS,
        'hold_seconds_tested': HOLD_SECONDS,
        'top20_results': all_results[:20],
        'both_positive_results': both_pos,
        'baseline_top10': baseline_results[:10],
        'summary': {
            'best_pnl': all_results[0]['total_pnl'] if all_results else 0.0,
            'best_config': all_results[0] if all_results else {},
            'n_both_positive': len(both_pos),
            'n_profitable': n_profitable,
            'n_total_configs': len(all_results),
        },
    }
    with open(str(out_path), 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f'\nResults saved: {out_path}')
    logger.info('Done.')


if __name__ == '__main__':
    main()
