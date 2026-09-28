#!/usr/bin/env python3
"""
High-Conviction Trading Strategy — Multi-Model Ensemble with Regime Filters
============================================================================

Goal: Turn REAL but sub-cost-threshold signals into profitable trades by:
  1. Ensemble agreement: only trade when multiple models agree
  2. Confidence gating: only trade extreme predictions
  3. Longer holds: 5min, 10min, 30min, 1hr (reduce cost-per-dollar-traded)
  4. Vol gating: only trade when predicted vol is high (bigger moves)
  5. Time-of-day filtering: focus on highest-signal periods

Data sources:
  - CNN (BookSpatialCNN): 94 OOS days, IC=0.130, date-keyed .npz
  - ET (EventTransformer): 45 OOS days, IC=0.145, index-based .npz
  - Combinator (LightGBM multi-horizon): 70 days, dir+mag at 3s/10s/30s
  - MBO features cache: 100 days, 340 features including mid, spread
  - Vol prediction: use mag_10s from combinator (IC~0.27 on magnitude)

All data at 100ms bars, 234000 bars/day (6.5hr RTH session).

Walk-forward: first 20 OOS days for parameter tuning, remaining for TRUE OOS.
"""

import sys
import json
import time
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# ── Paths ──
LVL3_ROOT = Path(__file__).parent.parent
FEAT_CACHE = LVL3_ROOT / 'data' / 'processed' / 'mbo_features_cache'
CNN_PRED = LVL3_ROOT / 'alpha_discovery' / 'deep_models' / 'results' / 'oos_predictions_book_20260303_234725.npz'
ET_PRED = LVL3_ROOT / 'alpha_discovery' / 'results' / 'predictions_ret_10s_20260221_200936.npz'
COMBO_PRED = LVL3_ROOT / 'alpha_discovery' / 'results' / 'combinator_preds_20260224_175256.npz'
RESULTS_DIR = LVL3_ROOT / 'alpha_discovery' / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──
TICK = 0.25          # ES tick size in points
TICK_VAL = 12.50     # $ per tick
BARS_PER_SEC = 10    # 100ms bars
BARS_PER_MIN = 600
BARS_PER_DAY = 234000
RTH_HOURS = 6.5

# Hold periods to test (in bars)
HOLD_PERIODS = {
    '5min': 3000,
    '10min': 6000,
    '30min': 18000,
    '1hr': 36000,
}

# Cost structures (all in ticks for 1 ES contract equivalent)
COST_STRUCTURES = {
    'ES_futures': {
        'spread_ticks': 1.0,    # 1 tick spread (pay half each way)
        'comm_ticks': 0.24,     # $3.00 RT / $12.50
        'multiplier': 50.0,     # $/point for 1 ES
        'label': 'ES 1-lot',
    },
    'SPY_500sh_IBKR': {
        'spread_ticks': 0.032,  # ~$0.01 spread on $600 SPY = 0.008 pts / 0.25 tick
        'comm_ticks': 0.40,     # ~$5.00 RT / $12.50 tick_val equiv
        'multiplier': 50.0,     # 500 SPY shares ≈ 1 ES
        'label': 'SPY 500sh IBKR',
    },
    'SPY_100sh_free': {
        'spread_ticks': 0.032,
        'comm_ticks': 0.0,      # Robinhood/Schwab zero-commission
        'multiplier': 10.0,     # 100 shares
        'label': 'SPY 100sh free',
    },
}

# Walk-forward split for parameter tuning
PARAM_TUNE_DAYS = 20  # first 20 overlapping OOS days for param search
# Remaining days = TRUE OOS

# ── Logging ──
_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(str(RESULTS_DIR / f'high_conviction_{_ts}.log'), mode='w'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger('hcs')


# ============================================================================
# DATA LOADING
# ============================================================================

def load_all_dates():
    """Return sorted list of all 100 trading dates."""
    return sorted([f.stem.replace('_mbo_features', '') for f in FEAT_CACHE.glob('*.npz')])


def load_mbo_day(date_str):
    """Load MBO features for a single day. Returns (mid, spread).
    Only loads columns 0-1 to save memory (~640MB/day avoided)."""
    path = FEAT_CACHE / f'{date_str}_mbo_features.npz'
    if not path.exists():
        return None
    d = np.load(str(path), mmap_mode='r')
    feats = d['mbo_features']
    mid = feats[:, 0].astype(np.float64)
    spread = feats[:, 1].astype(np.float64)
    return mid, spread


def load_cnn_predictions():
    """Load CNN OOS predictions. Returns {date: (preds, targets)}."""
    if not CNN_PRED.exists():
        log.warning(f"CNN predictions not found: {CNN_PRED}")
        return {}
    d = np.load(str(CNN_PRED), allow_pickle=True)
    dates = sorted(set(k.rsplit('_', 1)[0] for k in d.keys()))
    result = {}
    for date in dates:
        p = d[f'{date}_preds'].astype(np.float64)
        t = d[f'{date}_targets'].astype(np.float64)
        result[date] = (p, t)
    return result


def load_et_predictions(all_dates):
    """Load ET predictions. Returns {date: (preds, targets, mag_preds, mag_targets)}."""
    if not ET_PRED.exists():
        log.warning(f"ET predictions not found: {ET_PRED}")
        return {}
    d = np.load(str(ET_PRED), allow_pickle=True)
    dir_preds = d['direction_preds']
    dir_targets = d['direction_target']
    mag_preds = d['magnitude_preds']
    mag_targets = d['magnitude_target']
    bounds = d['day_boundaries']

    result = {}
    n_days = min(len(bounds) - 1, len(all_dates))
    for i in range(n_days):
        s, e = int(bounds[i]), int(bounds[i + 1])
        dp = dir_preds[s:e].astype(np.float64)
        dt = dir_targets[s:e].astype(np.float64)
        mp = mag_preds[s:e].astype(np.float64)
        mt = mag_targets[s:e].astype(np.float64)
        if np.all(np.isnan(dp)):
            continue  # training fold
        result[all_dates[i]] = (dp, dt, mp, mt)
    return result


def load_combo_predictions(all_dates):
    """Load combinator multi-horizon predictions. Returns {date: dict}."""
    if not COMBO_PRED.exists():
        log.warning(f"Combinator predictions not found: {COMBO_PRED}")
        return {}
    d = np.load(str(COMBO_PRED), allow_pickle=True)
    bounds = d['day_boundaries']
    result = {}
    n_days = min(len(bounds) - 1, len(all_dates))
    for i in range(n_days):
        s, e = int(bounds[i]), int(bounds[i + 1])
        day_data = {}
        for key in ['dir_3s', 'dir_10s', 'dir_30s', 'mag_3s', 'mag_10s', 'mag_30s']:
            arr = d[key][s:e].astype(np.float64)
            if np.all(np.isnan(arr)):
                continue
            day_data[key] = arr
        if day_data:
            day_data['mid'] = d['mid_prices'][s:e].astype(np.float64)
            result[all_dates[i]] = day_data
    return result


def load_aligned_data():
    """Load all data sources and align by date. Returns list of day dicts."""
    log.info("Loading all prediction sources...")
    t0 = time.time()

    all_dates = load_all_dates()
    log.info(f"  {len(all_dates)} total trading dates")

    cnn_data = load_cnn_predictions()
    log.info(f"  CNN: {len(cnn_data)} days")

    et_data = load_et_predictions(all_dates)
    log.info(f"  ET: {len(et_data)} days")

    combo_data = load_combo_predictions(all_dates)
    log.info(f"  Combinator: {len(combo_data)} days")

    # Find dates where we have CNN + at least one other source + MBO features
    days = []
    for date in all_dates:
        mbo = load_mbo_day(date)
        if mbo is None:
            continue
        mid, spread = mbo
        n_bars = len(mid)

        day = {
            'date': date,
            'mid': mid,
            'spread': spread,
            'n_bars': n_bars,
        }

        # CNN predictions — offset by CNN_WINDOW_SIZE-1 bars for proper alignment.
        # CNN valid_indices start at bar (window_size-1)=99, so pred[0] → bar 99.
        # We pad with NaN at the front and trim to match MBO bar count.
        CNN_OFFSET = 99  # window_size(100) - 1
        if date in cnn_data:
            cp, ct = cnn_data[date]
            # Prepend NaN padding so pred[0] aligns to MBO bar 99
            cp_aligned = np.full(n_bars, np.nan)
            ct_aligned = np.full(n_bars, np.nan)
            end_idx = min(CNN_OFFSET + len(cp), n_bars)
            cp_aligned[CNN_OFFSET:end_idx] = cp[:end_idx - CNN_OFFSET]
            ct_aligned[CNN_OFFSET:end_idx] = ct[:end_idx - CNN_OFFSET]
            day['cnn_preds'] = cp_aligned
            day['cnn_targets'] = ct_aligned

        # ET predictions
        if date in et_data:
            dp, dt, mp, mt = et_data[date]
            n = min(len(dp), n_bars)
            day['et_dir_preds'] = dp[:n]
            day['et_dir_targets'] = dt[:n]
            day['et_mag_preds'] = mp[:n]
            day['et_mag_targets'] = mt[:n]

        # Combinator predictions
        if date in combo_data:
            cd = combo_data[date]
            for key in ['dir_3s', 'dir_10s', 'dir_30s', 'mag_3s', 'mag_10s', 'mag_30s']:
                if key in cd:
                    n = min(len(cd[key]), n_bars)
                    day[f'combo_{key}'] = cd[key][:n]

        # Count available models
        has_cnn = 'cnn_preds' in day
        has_et = 'et_dir_preds' in day
        has_combo = 'combo_dir_10s' in day
        day['n_models'] = sum([has_cnn, has_et, has_combo])
        day['models'] = []
        if has_cnn: day['models'].append('CNN')
        if has_et: day['models'].append('ET')
        if has_combo: day['models'].append('COMBO')

        if day['n_models'] >= 1:  # keep all days with at least 1 model
            days.append(day)

    # Free raw prediction data to save memory
    del cnn_data, et_data, combo_data
    import gc; gc.collect()

    log.info(f"  Loaded {len(days)} days with predictions in {time.time()-t0:.1f}s")

    # Show model coverage
    coverage = defaultdict(int)
    for d in days:
        key = '+'.join(d['models'])
        coverage[key] += 1
    for k, v in sorted(coverage.items()):
        log.info(f"    {k}: {v} days")

    return days


# ============================================================================
# SIGNAL CONSTRUCTION
# ============================================================================

def zscore_per_day(arr):
    """Expanding-window z-score — NO look-ahead bias.
    At bar i, uses only mean/std of arr[0:i+1].
    Falls back to 0.0 until 100 valid observations accumulated.
    Vectorized with numpy cumulative sums for performance."""
    n = len(arr)
    out = np.zeros(n, dtype=np.float64)

    # Replace NaN with 0 for cumsum, track valid count separately
    valid = ~np.isnan(arr)
    clean = np.where(valid, arr, 0.0)

    cumsum = np.cumsum(clean)
    cumsum2 = np.cumsum(clean ** 2)
    cumcount = np.cumsum(valid.astype(np.float64))

    # Only compute where we have >= 100 valid observations
    mask = cumcount >= 100
    idx = np.where(mask & valid)[0]

    if len(idx) == 0:
        return out

    c = cumcount[idx]
    m = cumsum[idx] / c
    var = cumsum2[idx] / c - m * m
    s = np.sqrt(np.maximum(var, 0.0))

    nonzero_s = s > 1e-10
    out[idx[nonzero_s]] = (arr[idx[nonzero_s]] - m[nonzero_s]) / s[nonzero_s]

    return out


def compute_trailing_vol(mid, window=3000):
    """Compute trailing realized vol (std of 1s returns) over window bars.
    Window=3000 = 5 minutes of 100ms bars."""
    # 1-second returns (every 10 bars)
    ret_1s = np.zeros(len(mid))
    ret_1s[10:] = (mid[10:] - mid[:-10]) / mid[:-10] * 10000  # in bps

    # Rolling std
    vol = np.full(len(mid), np.nan)
    cumsum = np.cumsum(ret_1s)
    cumsum2 = np.cumsum(ret_1s ** 2)

    for i in range(window, len(mid)):
        s = cumsum[i] - cumsum[i - window]
        s2 = cumsum2[i] - cumsum2[i - window]
        mean = s / window
        var = s2 / window - mean ** 2
        vol[i] = np.sqrt(max(var, 0))

    return vol


def compute_time_features(n_bars):
    """Compute time-of-day features for a trading day.
    Returns (minutes_since_open, is_first_30min, is_last_30min, is_power_hour)."""
    # RTH: 9:30 - 16:00 ET = 6.5 hours = 23400 seconds = 234000 bars
    seconds = np.arange(n_bars) / BARS_PER_SEC
    minutes = seconds / 60.0

    first_30 = minutes < 30
    last_30 = minutes > (RTH_HOURS * 60 - 30)
    power_hour = minutes > (RTH_HOURS * 60 - 60)
    lunch_dead = (minutes > 120) & (minutes < 240)  # 11:30 - 1:30
    morning_afternoon = (minutes < 120) | ((minutes >= 240) & (minutes < 330))  # 9:30-11:30 + 1:30-3:00

    return minutes, first_30, last_30, power_hour, lunch_dead, morning_afternoon


def build_ensemble_signal(day, method='rank_average'):
    """Build ensemble signal from available models.

    Methods:
      - rank_average: z-score each model, average
      - agreement: sign agreement (returns -1, 0, +1)
      - conviction: average z-score, weighted by agreement count

    Returns (signal, conviction, n_agreeing) arrays.
    """
    n = day['n_bars']
    signals = []

    # CNN: predictions are in z-score-like space already
    if 'cnn_preds' in day:
        p = day['cnn_preds']
        # Pad to n_bars if shorter
        if len(p) < n:
            p = np.concatenate([p, np.full(n - len(p), np.nan)])
        signals.append(('CNN', zscore_per_day(p[:n])))

    # ET: predictions are in tick-space, z-score them
    if 'et_dir_preds' in day:
        p = day['et_dir_preds']
        if len(p) < n:
            p = np.concatenate([p, np.full(n - len(p), np.nan)])
        signals.append(('ET', zscore_per_day(p[:n])))

    # Combinator dir_10s: z-score
    if 'combo_dir_10s' in day:
        p = day['combo_dir_10s']
        if len(p) < n:
            p = np.concatenate([p, np.full(n - len(p), np.nan)])
        signals.append(('COMBO', zscore_per_day(p[:n])))

    if not signals:
        return np.zeros(n), np.zeros(n), np.zeros(n, dtype=int)

    # Stack signals
    names = [s[0] for s in signals]
    stack = np.array([s[1] for s in signals])  # (n_models, n_bars)

    # Average z-score
    avg_signal = np.nanmean(stack, axis=0)

    # Agreement: count how many models agree on direction
    signs = np.sign(stack)
    pos_count = np.sum(signs > 0, axis=0)
    neg_count = np.sum(signs < 0, axis=0)
    n_models = len(signals)

    # n_agreeing = max of pos/neg count
    n_agreeing = np.maximum(pos_count, neg_count)

    # Agreement direction: +1 if majority positive, -1 if majority negative, 0 if tied
    agreement_dir = np.zeros(n, dtype=np.float64)
    agreement_dir[pos_count > neg_count] = 1.0
    agreement_dir[neg_count > pos_count] = -1.0

    # Conviction = |avg_signal| * agreement_fraction
    agreement_frac = n_agreeing / n_models
    conviction = np.abs(avg_signal) * agreement_frac

    # Final signal: avg_signal (keeps direction)
    return avg_signal, conviction, n_agreeing


def get_vol_prediction(day):
    """Get volatility prediction for the day.
    Uses combo mag_10s if available, else trailing realized vol."""
    n = day['n_bars']
    if 'combo_mag_10s' in day:
        v = day['combo_mag_10s']
        if len(v) < n:
            v = np.concatenate([v, np.full(n - len(v), np.nan)])
        return v[:n]
    # Fallback: trailing vol
    return compute_trailing_vol(day['mid'])


# ============================================================================
# TRADING SIMULATION
# ============================================================================

def simulate_trades(day, signal, conviction, n_agreeing, vol_pred,
                    hold_bars, cost_spread_ticks, cost_comm_ticks,
                    conviction_threshold=1.5, min_agreement=2,
                    vol_percentile_min=50, time_filter='none',
                    cooldown_bars=None, max_trades_per_day=50):
    """
    Simulate trading for one day.

    Entry conditions (ALL must be true):
      1. |signal| > conviction_threshold (in z-score units)
      2. n_agreeing >= min_agreement (model agreement)
      3. vol_pred >= vol_percentile_min-th percentile (vol gating)
      4. Time filter passes
      5. Not in cooldown from previous trade
      6. Not too close to session end for hold period

    Exit: fixed hold period (no early exit)

    Returns dict with trade results.
    """
    mid = day['mid']
    spread = day['spread']
    n = day['n_bars']

    if cooldown_bars is None:
        cooldown_bars = hold_bars  # don't overlap trades

    # Cost per round trip in points
    rt_cost_points = (cost_spread_ticks + cost_comm_ticks) * TICK

    # Vol gating: use pre-computed expanding-window percentile (no look-ahead)
    vol_expanding_threshold = np.full(len(vol_pred), -np.inf)
    if vol_percentile_min > 0 and '_vol_pct_thresholds' in day:
        # Find closest pre-computed percentile
        available = sorted(day['_vol_pct_thresholds'].keys())
        closest = min(available, key=lambda x: abs(x - vol_percentile_min))
        vol_expanding_threshold = day['_vol_pct_thresholds'][closest]

    # Time filter
    minutes, first_30, last_30, power_hour, lunch_dead, morning_afternoon = compute_time_features(n)
    if time_filter == 'edges':
        time_ok = first_30 | last_30
    elif time_filter == 'no_lunch':
        time_ok = ~lunch_dead
    elif time_filter == 'power_hour':
        time_ok = power_hour
    elif time_filter == 'first_hour':
        time_ok = minutes < 60
    elif time_filter == 'morning_afternoon':
        time_ok = morning_afternoon
    else:
        time_ok = np.ones(n, dtype=bool)

    # Simulate
    trades = []
    last_exit_bar = -1

    for i in range(100, n - hold_bars):  # skip first 100 bars for warmup
        # Cooldown check
        if i < last_exit_bar + cooldown_bars:
            continue

        # Max trades check
        if len(trades) >= max_trades_per_day:
            break

        # Time filter
        if not time_ok[i]:
            continue

        # Signal strength (conviction threshold)
        if np.isnan(signal[i]) or abs(signal[i]) < conviction_threshold:
            continue

        # Model agreement
        if n_agreeing[i] < min_agreement:
            continue

        # Vol gating (expanding window — no look-ahead)
        if np.isnan(vol_pred[i]) or vol_pred[i] < vol_expanding_threshold[i]:
            continue

        # ENTRY
        direction = 1.0 if signal[i] > 0 else -1.0
        entry_price = mid[i]
        exit_bar = i + hold_bars
        exit_price = mid[exit_bar]

        # P&L in points (before costs)
        gross_pnl_points = direction * (exit_price - entry_price)
        net_pnl_points = gross_pnl_points - rt_cost_points

        # P&L in ticks
        gross_pnl_ticks = gross_pnl_points / TICK
        net_pnl_ticks = net_pnl_points / TICK

        trades.append({
            'entry_bar': i,
            'exit_bar': exit_bar,
            'direction': direction,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'gross_pnl_ticks': gross_pnl_ticks,
            'net_pnl_ticks': net_pnl_ticks,
            'signal_strength': abs(signal[i]),
            'conviction': conviction[i],
            'n_agree': n_agreeing[i],
            'vol_pred': vol_pred[i],
            'minute': minutes[i],
        })

        last_exit_bar = exit_bar

    return trades


def compute_metrics(all_trades, multiplier=50.0, label=''):
    """Compute strategy metrics from list of trades across all days."""
    if not all_trades:
        return {
            'label': label,
            'total_trades': 0,
            'net_pnl_ticks': 0,
            'net_pnl_dollars': 0,
            'sharpe': 0,
            'win_rate': 0,
            'avg_pnl_per_trade_ticks': 0,
            'max_drawdown_ticks': 0,
            'profitable': False,
        }

    pnls = np.array([t['net_pnl_ticks'] for t in all_trades])
    gross = np.array([t['gross_pnl_ticks'] for t in all_trades])

    # Daily aggregation for Sharpe
    daily_pnl = defaultdict(float)
    daily_count = defaultdict(int)
    for t in all_trades:
        # Use entry_bar to identify day (approximate)
        day_key = t.get('date', 'unknown')
        daily_pnl[day_key] += t['net_pnl_ticks']
        daily_count[day_key] += 1

    daily_returns = np.array(list(daily_pnl.values()))
    n_days = len(daily_returns)

    # Sharpe (annualized, assuming 252 trading days)
    if n_days > 1 and daily_returns.std() > 0:
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Max drawdown
    cumsum = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumsum)
    drawdown = running_max - cumsum
    max_dd = drawdown.max() if len(drawdown) > 0 else 0

    # Win rate
    win_rate = (pnls > 0).mean() * 100

    # Trades per day
    trades_per_day = len(all_trades) / max(n_days, 1)

    return {
        'label': label,
        'total_trades': len(all_trades),
        'n_days': n_days,
        'trades_per_day': round(trades_per_day, 1),
        'net_pnl_ticks': round(float(pnls.sum()), 1),
        'net_pnl_dollars': round(float(pnls.sum() * TICK_VAL), 2),
        'gross_pnl_ticks': round(float(gross.sum()), 1),
        'avg_pnl_per_trade_ticks': round(float(pnls.mean()), 3),
        'avg_gross_per_trade_ticks': round(float(gross.mean()), 3),
        'win_rate': round(win_rate, 1),
        'sharpe': round(sharpe, 2),
        'max_drawdown_ticks': round(float(max_dd), 1),
        'max_drawdown_dollars': round(float(max_dd * TICK_VAL), 2),
        'best_trade_ticks': round(float(pnls.max()), 1),
        'worst_trade_ticks': round(float(pnls.min()), 1),
        'profitable': bool(pnls.sum() > 0),
    }


# ============================================================================
# PRE-COMPUTATION (avoid recomputing signals for every config)
# ============================================================================

def _precompute_vol_percentiles(vol_pred, percentiles=(50, 60, 70, 80, 90)):
    """Pre-compute expanding-window vol percentile thresholds for a day.
    Returns dict {pct: threshold_array} — no look-ahead bias.
    Uses sorted insertion + binary search for O(n log n) per day."""
    import bisect
    n = len(vol_pred)
    result = {p: np.full(n, -np.inf) for p in percentiles}
    sorted_vals = []
    for i in range(n):
        if not np.isnan(vol_pred[i]):
            bisect.insort(sorted_vals, vol_pred[i])
        if len(sorted_vals) >= 100:
            for p in percentiles:
                idx = min(int(len(sorted_vals) * p / 100), len(sorted_vals) - 1)
                result[p][i] = sorted_vals[idx]
    return result


def precompute_signals(days):
    """Pre-compute ensemble signals, vol predictions, and vol percentiles for all days."""
    log.info("Pre-computing ensemble signals for all days...")
    for day in days:
        if '_signal' not in day:
            signal, conviction, n_agree = build_ensemble_signal(day)
            vol_pred = get_vol_prediction(day)
            day['_signal'] = signal
            day['_conviction'] = conviction
            day['_n_agree'] = n_agree
            day['_vol_pred'] = vol_pred
            # Pre-compute expanding vol percentile thresholds (no look-ahead)
            day['_vol_pct_thresholds'] = _precompute_vol_percentiles(vol_pred)
    log.info(f"  Done. {len(days)} days pre-computed.")


def get_precomputed(day):
    """Get pre-computed signal data for a day."""
    return day['_signal'], day['_conviction'], day['_n_agree'], day['_vol_pred']


# ============================================================================
# APPROACH A: ENSEMBLE AGREEMENT + CONFIDENCE GATING
# ============================================================================

def run_approach_a(days, cost_key='ES_futures', verbose=True):
    """
    High-conviction ensemble: only trade when multiple models agree strongly.

    Sweep over:
      - conviction_threshold: [1.0, 1.5, 2.0, 2.5, 3.0]
      - hold_period: [5min, 10min, 30min, 1hr]
      - min_agreement: [2, 3] (when 3 models available)
    """
    log.info("\n" + "="*80)
    log.info("APPROACH A: ENSEMBLE AGREEMENT + CONFIDENCE GATING")
    log.info("="*80)

    cost = COST_STRUCTURES[cost_key]

    # Split: first PARAM_TUNE_DAYS for tuning, rest for OOS
    tune_days = days[:PARAM_TUNE_DAYS]
    oos_days = days[PARAM_TUNE_DAYS:]

    log.info(f"Tuning on {len(tune_days)} days, OOS on {len(oos_days)} days")
    log.info(f"Cost structure: {cost['label']}")

    # Parameter grid
    conv_thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    hold_keys = list(HOLD_PERIODS.keys())
    min_agreements = [1, 2]  # 2 means need at least 2 models to agree

    best_config = None
    best_sharpe = -999
    all_results = []

    for conv_thr in conv_thresholds:
        for hold_key in hold_keys:
            hold_bars = HOLD_PERIODS[hold_key]
            for min_agree in min_agreements:
                tune_trades = []
                for day in tune_days:
                    signal, conviction, n_agree, vol_pred = get_precomputed(day)
                    trades = simulate_trades(
                        day, signal, conviction, n_agree, vol_pred,
                        hold_bars=hold_bars,
                        cost_spread_ticks=cost['spread_ticks'],
                        cost_comm_ticks=cost['comm_ticks'],
                        conviction_threshold=conv_thr,
                        min_agreement=min_agree,
                        vol_percentile_min=0,  # no vol filter in approach A
                        time_filter='none',
                    )
                    for t in trades:
                        t['date'] = day['date']
                    tune_trades.extend(trades)

                m = compute_metrics(tune_trades, label=f'A|conv={conv_thr}|hold={hold_key}|agree={min_agree}')
                all_results.append(m)

                if m['sharpe'] > best_sharpe and m['total_trades'] >= 10:
                    best_sharpe = m['sharpe']
                    best_config = {
                        'conviction_threshold': conv_thr,
                        'hold_key': hold_key,
                        'min_agreement': min_agree,
                    }

    # Report tuning results
    if verbose:
        log.info("\n--- Tuning Results (top 10 by Sharpe) ---")
        sorted_results = sorted(all_results, key=lambda x: x['sharpe'], reverse=True)
        for r in sorted_results[:10]:
            log.info(f"  {r['label']:50s} | trades={r['total_trades']:4d} | "
                    f"net={r['net_pnl_ticks']:+8.1f}t | sharpe={r['sharpe']:+6.2f} | "
                    f"win={r['win_rate']:5.1f}%")

    if best_config is None:
        log.info("No valid configuration found in tuning!")
        return None

    log.info(f"\nBest config: {best_config} (tune Sharpe={best_sharpe:.2f})")

    # Run on TRUE OOS
    oos_trades = []
    for day in oos_days:
        signal, conviction, n_agree, vol_pred = get_precomputed(day)
        trades = simulate_trades(
            day, signal, conviction, n_agree, vol_pred,
            hold_bars=HOLD_PERIODS[best_config['hold_key']],
            cost_spread_ticks=cost['spread_ticks'],
            cost_comm_ticks=cost['comm_ticks'],
            conviction_threshold=best_config['conviction_threshold'],
            min_agreement=best_config['min_agreement'],
            vol_percentile_min=0,
            time_filter='none',
        )
        for t in trades:
            t['date'] = day['date']
        oos_trades.extend(trades)

    oos_metrics = compute_metrics(oos_trades, label=f'A_OOS|{cost_key}')
    log.info(f"\n*** APPROACH A OOS RESULTS ({cost_key}) ***")
    for k, v in oos_metrics.items():
        log.info(f"  {k}: {v}")

    return {
        'approach': 'A_ensemble_agreement',
        'best_config': best_config,
        'tune_sharpe': best_sharpe,
        'oos_metrics': oos_metrics,
        'oos_trades': oos_trades,
        'all_tune_results': all_results,
    }


# ============================================================================
# APPROACH B: VOL-GATED TRADING
# ============================================================================

def run_approach_b(days, cost_key='ES_futures', verbose=True):
    """
    Vol-gated: only trade when predicted vol is HIGH.
    High vol = bigger moves = more gross P&L per trade to cover costs.

    Sweep over:
      - vol_percentile_min: [50, 60, 70, 80, 90]
      - conviction_threshold: [0.5, 1.0, 1.5, 2.0]
      - hold_period: [5min, 10min, 30min]
    """
    log.info("\n" + "="*80)
    log.info("APPROACH B: VOL-GATED TRADING")
    log.info("="*80)

    cost = COST_STRUCTURES[cost_key]
    tune_days = days[:PARAM_TUNE_DAYS]
    oos_days = days[PARAM_TUNE_DAYS:]

    log.info(f"Tuning on {len(tune_days)} days, OOS on {len(oos_days)} days")

    vol_pctiles = [50, 60, 70, 80, 90]
    conv_thresholds = [0.5, 1.0, 1.5, 2.0]
    hold_keys = ['5min', '10min', '30min']

    best_config = None
    best_sharpe = -999
    all_results = []

    for vol_pct in vol_pctiles:
        for conv_thr in conv_thresholds:
            for hold_key in hold_keys:
                hold_bars = HOLD_PERIODS[hold_key]
                tune_trades = []
                for day in tune_days:
                    signal, conviction, n_agree, vol_pred = get_precomputed(day)
                    trades = simulate_trades(
                        day, signal, conviction, n_agree, vol_pred,
                        hold_bars=hold_bars,
                        cost_spread_ticks=cost['spread_ticks'],
                        cost_comm_ticks=cost['comm_ticks'],
                        conviction_threshold=conv_thr,
                        min_agreement=1,
                        vol_percentile_min=vol_pct,
                        time_filter='none',
                    )
                    for t in trades:
                        t['date'] = day['date']
                    tune_trades.extend(trades)

                m = compute_metrics(tune_trades, label=f'B|vol>={vol_pct}th|conv={conv_thr}|hold={hold_key}')
                all_results.append(m)

                if m['sharpe'] > best_sharpe and m['total_trades'] >= 10:
                    best_sharpe = m['sharpe']
                    best_config = {
                        'vol_percentile_min': vol_pct,
                        'conviction_threshold': conv_thr,
                        'hold_key': hold_key,
                    }

    if verbose:
        log.info("\n--- Tuning Results (top 10 by Sharpe) ---")
        sorted_results = sorted(all_results, key=lambda x: x['sharpe'], reverse=True)
        for r in sorted_results[:10]:
            log.info(f"  {r['label']:50s} | trades={r['total_trades']:4d} | "
                    f"net={r['net_pnl_ticks']:+8.1f}t | sharpe={r['sharpe']:+6.2f} | "
                    f"win={r['win_rate']:5.1f}%")

    if best_config is None:
        log.info("No valid configuration found in tuning!")
        return None

    log.info(f"\nBest config: {best_config} (tune Sharpe={best_sharpe:.2f})")

    # OOS
    oos_trades = []
    for day in oos_days:
        signal, conviction, n_agree, vol_pred = get_precomputed(day)
        trades = simulate_trades(
            day, signal, conviction, n_agree, vol_pred,
            hold_bars=HOLD_PERIODS[best_config['hold_key']],
            cost_spread_ticks=cost['spread_ticks'],
            cost_comm_ticks=cost['comm_ticks'],
            conviction_threshold=best_config['conviction_threshold'],
            min_agreement=1,
            vol_percentile_min=best_config['vol_percentile_min'],
            time_filter='none',
        )
        for t in trades:
            t['date'] = day['date']
        oos_trades.extend(trades)

    oos_metrics = compute_metrics(oos_trades, label=f'B_OOS|{cost_key}')
    log.info(f"\n*** APPROACH B OOS RESULTS ({cost_key}) ***")
    for k, v in oos_metrics.items():
        log.info(f"  {k}: {v}")

    return {
        'approach': 'B_vol_gated',
        'best_config': best_config,
        'tune_sharpe': best_sharpe,
        'oos_metrics': oos_metrics,
        'oos_trades': oos_trades,
        'all_tune_results': all_results,
    }


# ============================================================================
# APPROACH C: TIME-OF-DAY FILTER
# ============================================================================

def run_approach_c(days, cost_key='ES_futures', verbose=True):
    """
    Time-of-day filtering: trade only during high-signal periods.
    First 30min and last 30min typically have highest vol + signal.

    Sweep over:
      - time_filter: [edges, no_lunch, first_hour, power_hour]
      - conviction_threshold: [0.5, 1.0, 1.5, 2.0]
      - hold_period: [5min, 10min, 30min]
    """
    log.info("\n" + "="*80)
    log.info("APPROACH C: TIME-OF-DAY FILTER")
    log.info("="*80)

    cost = COST_STRUCTURES[cost_key]
    tune_days = days[:PARAM_TUNE_DAYS]
    oos_days = days[PARAM_TUNE_DAYS:]

    time_filters = ['edges', 'no_lunch', 'first_hour', 'power_hour']
    conv_thresholds = [0.5, 1.0, 1.5, 2.0]
    hold_keys = ['5min', '10min', '30min']

    best_config = None
    best_sharpe = -999
    all_results = []

    for tf in time_filters:
        for conv_thr in conv_thresholds:
            for hold_key in hold_keys:
                hold_bars = HOLD_PERIODS[hold_key]
                tune_trades = []
                for day in tune_days:
                    signal, conviction, n_agree, vol_pred = get_precomputed(day)
                    trades = simulate_trades(
                        day, signal, conviction, n_agree, vol_pred,
                        hold_bars=hold_bars,
                        cost_spread_ticks=cost['spread_ticks'],
                        cost_comm_ticks=cost['comm_ticks'],
                        conviction_threshold=conv_thr,
                        min_agreement=1,
                        vol_percentile_min=0,
                        time_filter=tf,
                    )
                    for t in trades:
                        t['date'] = day['date']
                    tune_trades.extend(trades)

                m = compute_metrics(tune_trades, label=f'C|time={tf}|conv={conv_thr}|hold={hold_key}')
                all_results.append(m)

                if m['sharpe'] > best_sharpe and m['total_trades'] >= 10:
                    best_sharpe = m['sharpe']
                    best_config = {
                        'time_filter': tf,
                        'conviction_threshold': conv_thr,
                        'hold_key': hold_key,
                    }

    if verbose:
        log.info("\n--- Tuning Results (top 10 by Sharpe) ---")
        sorted_results = sorted(all_results, key=lambda x: x['sharpe'], reverse=True)
        for r in sorted_results[:10]:
            log.info(f"  {r['label']:50s} | trades={r['total_trades']:4d} | "
                    f"net={r['net_pnl_ticks']:+8.1f}t | sharpe={r['sharpe']:+6.2f} | "
                    f"win={r['win_rate']:5.1f}%")

    if best_config is None:
        log.info("No valid configuration found in tuning!")
        return None

    log.info(f"\nBest config: {best_config} (tune Sharpe={best_sharpe:.2f})")

    # OOS
    oos_trades = []
    for day in oos_days:
        signal, conviction, n_agree, vol_pred = get_precomputed(day)
        trades = simulate_trades(
            day, signal, conviction, n_agree, vol_pred,
            hold_bars=HOLD_PERIODS[best_config['hold_key']],
            cost_spread_ticks=cost['spread_ticks'],
            cost_comm_ticks=cost['comm_ticks'],
            conviction_threshold=best_config['conviction_threshold'],
            min_agreement=1,
            vol_percentile_min=0,
            time_filter=best_config['time_filter'],
        )
        for t in trades:
            t['date'] = day['date']
        oos_trades.extend(trades)

    oos_metrics = compute_metrics(oos_trades, label=f'C_OOS|{cost_key}')
    log.info(f"\n*** APPROACH C OOS RESULTS ({cost_key}) ***")
    for k, v in oos_metrics.items():
        log.info(f"  {k}: {v}")

    return {
        'approach': 'C_time_filter',
        'best_config': best_config,
        'tune_sharpe': best_sharpe,
        'oos_metrics': oos_metrics,
        'oos_trades': oos_trades,
        'all_tune_results': all_results,
    }


# ============================================================================
# APPROACH D: KITCHEN SINK — COMBINE ALL FILTERS
# ============================================================================

def run_approach_d(days, cost_key='ES_futures', verbose=True):
    """
    Combine: ensemble agreement + vol gating + time filter.
    Use best sub-parameters from A/B/C or sweep a focused grid.
    """
    log.info("\n" + "="*80)
    log.info("APPROACH D: COMBINED (ENSEMBLE + VOL + TIME)")
    log.info("="*80)

    cost = COST_STRUCTURES[cost_key]
    tune_days = days[:PARAM_TUNE_DAYS]
    oos_days = days[PARAM_TUNE_DAYS:]

    # Focused grid combining the filters
    configs = []
    for conv_thr in [1.0, 1.5, 2.0, 2.5]:
        for hold_key in ['5min', '10min', '30min']:
            for vol_pct in [0, 50, 70, 80]:
                for tf in ['none', 'no_lunch', 'edges']:
                    for min_agree in [1, 2]:
                        configs.append({
                            'conviction_threshold': conv_thr,
                            'hold_key': hold_key,
                            'vol_percentile_min': vol_pct,
                            'time_filter': tf,
                            'min_agreement': min_agree,
                        })

    log.info(f"Testing {len(configs)} configurations...")

    best_config = None
    best_sharpe = -999
    all_results = []

    for ci, cfg in enumerate(configs):
        hold_bars = HOLD_PERIODS[cfg['hold_key']]
        tune_trades = []
        for day in tune_days:
            signal, conviction, n_agree, vol_pred = get_precomputed(day)
            trades = simulate_trades(
                day, signal, conviction, n_agree, vol_pred,
                hold_bars=hold_bars,
                cost_spread_ticks=cost['spread_ticks'],
                cost_comm_ticks=cost['comm_ticks'],
                conviction_threshold=cfg['conviction_threshold'],
                min_agreement=cfg['min_agreement'],
                vol_percentile_min=cfg['vol_percentile_min'],
                time_filter=cfg['time_filter'],
            )
            for t in trades:
                t['date'] = day['date']
            tune_trades.extend(trades)

        label = (f"D|conv={cfg['conviction_threshold']}|hold={cfg['hold_key']}|"
                f"vol>={cfg['vol_percentile_min']}|time={cfg['time_filter']}|"
                f"agree={cfg['min_agreement']}")
        m = compute_metrics(tune_trades, label=label)
        all_results.append(m)

        if m['sharpe'] > best_sharpe and m['total_trades'] >= 5:
            best_sharpe = m['sharpe']
            best_config = cfg

    if verbose:
        log.info("\n--- Tuning Results (top 15 by Sharpe) ---")
        sorted_results = sorted(all_results, key=lambda x: x['sharpe'], reverse=True)
        for r in sorted_results[:15]:
            log.info(f"  {r['label']:70s} | trades={r['total_trades']:4d} | "
                    f"net={r['net_pnl_ticks']:+8.1f}t | sharpe={r['sharpe']:+6.2f} | "
                    f"win={r['win_rate']:5.1f}%")

    if best_config is None:
        log.info("No valid configuration found in tuning!")
        return None

    log.info(f"\nBest config: {best_config} (tune Sharpe={best_sharpe:.2f})")

    # OOS
    oos_trades = []
    for day in oos_days:
        signal, conviction, n_agree, vol_pred = get_precomputed(day)
        trades = simulate_trades(
            day, signal, conviction, n_agree, vol_pred,
            hold_bars=HOLD_PERIODS[best_config['hold_key']],
            cost_spread_ticks=cost['spread_ticks'],
            cost_comm_ticks=cost['comm_ticks'],
            conviction_threshold=best_config['conviction_threshold'],
            min_agreement=best_config['min_agreement'],
            vol_percentile_min=best_config['vol_percentile_min'],
            time_filter=best_config['time_filter'],
        )
        for t in trades:
            t['date'] = day['date']
        oos_trades.extend(trades)

    oos_metrics = compute_metrics(oos_trades, label=f'D_OOS|{cost_key}')
    log.info(f"\n*** APPROACH D OOS RESULTS ({cost_key}) ***")
    for k, v in oos_metrics.items():
        log.info(f"  {k}: {v}")

    return {
        'approach': 'D_combined',
        'best_config': best_config,
        'tune_sharpe': best_sharpe,
        'oos_metrics': oos_metrics,
        'oos_trades': oos_trades,
        'all_tune_results': all_results,
    }


# ============================================================================
# APPROACH E: HOLD-UNTIL-FLIP (REDUCE TURNOVER)
# ============================================================================

def run_approach_e(days, cost_key='ES_futures', verbose=True):
    """
    Hold-until-flip: enter on strong conviction, hold until signal flips.
    This is the most promising for reducing costs — trade count drops dramatically.

    Entry: ensemble signal crosses above +threshold or below -threshold
    Exit: signal crosses zero OR reaches max_hold
    """
    log.info("\n" + "="*80)
    log.info("APPROACH E: HOLD-UNTIL-FLIP (ADAPTIVE EXIT)")
    log.info("="*80)

    cost = COST_STRUCTURES[cost_key]
    tune_days = days[:PARAM_TUNE_DAYS]
    oos_days = days[PARAM_TUNE_DAYS:]

    def _vectorized_ema(signal, span=100):
        """Vectorized EMA using scipy's lfilter (no per-bar Python loop)."""
        from scipy.signal import lfilter
        alpha = 2.0 / (span + 1.0)
        # Replace NaN with 0 for filtering
        clean = np.where(np.isnan(signal), 0.0, signal)
        b = [alpha]
        a = [1, -(1 - alpha)]
        return lfilter(b, a, clean)

    def sim_hold_until_flip(day, entry_threshold, exit_threshold,
                            max_hold_bars, min_agree, vol_pct_min,
                            time_filter, cost_spread, cost_comm):
        """Simulate hold-until-flip strategy for one day (vectorized EMA)."""
        signal, conviction, n_agree, vol_pred = get_precomputed(day)
        mid = day['mid']
        n = day['n_bars']

        # Vol threshold (pre-computed expanding window — no look-ahead)
        vol_expanding_thr = np.full(n, -np.inf)
        if vol_pct_min > 0 and '_vol_pct_thresholds' in day:
            available = sorted(day['_vol_pct_thresholds'].keys())
            closest = min(available, key=lambda x: abs(x - vol_pct_min))
            vol_expanding_thr = day['_vol_pct_thresholds'][closest]

        # Time filter
        minutes, first_30, last_30, power_hour, lunch_dead, morning_afternoon = compute_time_features(n)
        if time_filter == 'no_lunch':
            time_ok = ~lunch_dead
        elif time_filter == 'edges':
            time_ok = first_30 | last_30
        elif time_filter == 'morning_afternoon':
            time_ok = morning_afternoon
        else:
            time_ok = np.ones(n, dtype=bool)

        # Vectorized EMA smoothing
        smooth = _vectorized_ema(signal, span=100)

        # Build entry-eligible mask (vectorized)
        entry_eligible = np.zeros(n, dtype=bool)
        entry_eligible[200:n-100] = True
        entry_eligible &= time_ok
        entry_eligible &= ~np.isnan(signal)
        entry_eligible &= (np.abs(smooth) >= entry_threshold)
        entry_eligible &= (n_agree >= min_agree)
        entry_eligible &= (~np.isnan(vol_pred)) & (vol_pred >= vol_expanding_thr)

        # Sequential trade simulation (fast: only iterates over eligible bars)
        eligible_indices = np.where(entry_eligible)[0]
        trades = []
        next_allowed = 200

        for idx in eligible_indices:
            if idx < next_allowed:
                continue

            # Enter trade
            direction = 1.0 if smooth[idx] > 0 else -1.0
            entry_price = mid[idx]
            entry_bar = idx

            # Find exit: scan forward for flip or timeout
            exit_bar = min(idx + max_hold_bars, n - 1)
            exit_reason = 'timeout'

            # Check for flip in the hold window
            hold_end = min(idx + max_hold_bars, n - 100)
            if direction > 0:
                flip_mask = smooth[idx+1:hold_end] < exit_threshold
            else:
                flip_mask = smooth[idx+1:hold_end] > -exit_threshold

            flip_positions = np.where(flip_mask)[0]
            if len(flip_positions) > 0:
                exit_bar = idx + 1 + flip_positions[0]
                exit_reason = 'flip'

            # Near close check
            if exit_bar >= n - 200:
                exit_bar = n - 200
                exit_reason = 'close'

            exit_price = mid[exit_bar]
            gross = direction * (exit_price - entry_price) / TICK
            net = gross - (cost_spread + cost_comm)
            held = exit_bar - entry_bar

            trades.append({
                'entry_bar': entry_bar,
                'exit_bar': exit_bar,
                'direction': direction,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'gross_pnl_ticks': gross,
                'net_pnl_ticks': net,
                'hold_bars': held,
                'hold_seconds': held / BARS_PER_SEC,
                'signal_strength': abs(signal[entry_bar]) if not np.isnan(signal[entry_bar]) else 0,
                'exit_reason': exit_reason,
                'minute': minutes[entry_bar],
            })

            next_allowed = exit_bar + 1  # no overlapping trades

        return trades

    # Parameter grid
    configs = []
    for entry_thr in [1.0, 1.5, 2.0, 2.5]:
        for exit_thr in [0.0, 0.3, 0.5]:
            for max_hold in ['10min', '30min', '1hr']:
                for vol_pct in [0, 50, 70]:
                    for tf in ['none', 'no_lunch']:
                        for min_agree in [1, 2]:
                            configs.append({
                                'entry_threshold': entry_thr,
                                'exit_threshold': exit_thr,
                                'max_hold': max_hold,
                                'vol_pct': vol_pct,
                                'time_filter': tf,
                                'min_agreement': min_agree,
                            })

    log.info(f"Testing {len(configs)} configurations...")

    best_config = None
    best_sharpe = -999
    all_results = []

    for cfg in configs:
        tune_trades = []
        for day in tune_days:
            trades = sim_hold_until_flip(
                day,
                entry_threshold=cfg['entry_threshold'],
                exit_threshold=cfg['exit_threshold'],
                max_hold_bars=HOLD_PERIODS[cfg['max_hold']],
                min_agree=cfg['min_agreement'],
                vol_pct_min=cfg['vol_pct'],
                time_filter=cfg['time_filter'],
                cost_spread=cost['spread_ticks'],
                cost_comm=cost['comm_ticks'],
            )
            for t in trades:
                t['date'] = day['date']
            tune_trades.extend(trades)

        label = (f"E|entry={cfg['entry_threshold']}|exit={cfg['exit_threshold']}|"
                f"max={cfg['max_hold']}|vol>={cfg['vol_pct']}|"
                f"time={cfg['time_filter']}|agree={cfg['min_agreement']}")
        m = compute_metrics(tune_trades, label=label)
        all_results.append(m)

        if m['sharpe'] > best_sharpe and m['total_trades'] >= 5:
            best_sharpe = m['sharpe']
            best_config = cfg

    if verbose:
        log.info("\n--- Tuning Results (top 15 by Sharpe) ---")
        sorted_results = sorted(all_results, key=lambda x: x['sharpe'], reverse=True)
        for r in sorted_results[:15]:
            log.info(f"  {r['label']:75s} | trades={r['total_trades']:4d} | "
                    f"net={r['net_pnl_ticks']:+8.1f}t | sharpe={r['sharpe']:+6.2f} | "
                    f"win={r['win_rate']:5.1f}%")

    if best_config is None:
        log.info("No valid configuration found!")
        return None

    log.info(f"\nBest config: {best_config} (tune Sharpe={best_sharpe:.2f})")

    # OOS
    oos_trades = []
    for day in oos_days:
        trades = sim_hold_until_flip(
            day,
            entry_threshold=best_config['entry_threshold'],
            exit_threshold=best_config['exit_threshold'],
            max_hold_bars=HOLD_PERIODS[best_config['max_hold']],
            min_agree=best_config['min_agreement'],
            vol_pct_min=best_config['vol_pct'],
            time_filter=best_config['time_filter'],
            cost_spread=cost['spread_ticks'],
            cost_comm=cost['comm_ticks'],
        )
        for t in trades:
            t['date'] = day['date']
        oos_trades.extend(trades)

    oos_metrics = compute_metrics(oos_trades, label=f'E_OOS|{cost_key}')
    log.info(f"\n*** APPROACH E OOS RESULTS ({cost_key}) ***")
    for k, v in oos_metrics.items():
        log.info(f"  {k}: {v}")

    # Additional: show hold time distribution
    if oos_trades:
        holds = [t['hold_seconds'] for t in oos_trades]
        log.info(f"\n  Hold time stats: mean={np.mean(holds):.0f}s, "
                f"median={np.median(holds):.0f}s, "
                f"min={np.min(holds):.0f}s, max={np.max(holds):.0f}s")
        exits = defaultdict(int)
        for t in oos_trades:
            exits[t['exit_reason']] += 1
        log.info(f"  Exit reasons: {dict(exits)}")

    return {
        'approach': 'E_hold_until_flip',
        'best_config': best_config,
        'tune_sharpe': best_sharpe,
        'oos_metrics': oos_metrics,
        'oos_trades': oos_trades,
        'all_tune_results': all_results,
    }


# ============================================================================
# CROSS-COST COMPARISON
# ============================================================================

def run_best_across_costs(days, best_approach_fn, best_config, approach_name):
    """Re-run the best approach configuration across all cost structures."""
    log.info(f"\n{'='*80}")
    log.info(f"CROSS-COST COMPARISON FOR {approach_name}")
    log.info(f"{'='*80}")

    oos_days = days[PARAM_TUNE_DAYS:]
    results = {}

    for cost_key, cost in COST_STRUCTURES.items():
        log.info(f"\n--- {cost['label']} ---")
        oos_trades = []
        for day in oos_days:
            signal, conviction, n_agree, vol_pred = get_precomputed(day)

            if 'hold_key' in best_config:
                hold_bars = HOLD_PERIODS[best_config['hold_key']]
            else:
                hold_bars = HOLD_PERIODS.get(best_config.get('max_hold', '10min'), 6000)

            trades = simulate_trades(
                day, signal, conviction, n_agree, vol_pred,
                hold_bars=hold_bars,
                cost_spread_ticks=cost['spread_ticks'],
                cost_comm_ticks=cost['comm_ticks'],
                conviction_threshold=best_config.get('conviction_threshold',
                                                     best_config.get('entry_threshold', 1.5)),
                min_agreement=best_config.get('min_agreement', 1),
                vol_percentile_min=best_config.get('vol_percentile_min',
                                                   best_config.get('vol_pct', 0)),
                time_filter=best_config.get('time_filter', 'none'),
            )
            for t in trades:
                t['date'] = day['date']
            oos_trades.extend(trades)

        m = compute_metrics(oos_trades, label=f'{approach_name}|{cost_key}')
        results[cost_key] = m
        log.info(f"  Trades: {m['total_trades']}, Net P&L: {m['net_pnl_ticks']:+.1f} ticks "
                f"(${m['net_pnl_dollars']:+.2f}), Sharpe: {m['sharpe']:+.2f}, "
                f"Win: {m['win_rate']:.1f}%")

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    log.info("="*80)
    log.info("HIGH-CONVICTION TRADING STRATEGY — MULTI-APPROACH BACKTEST")
    log.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("="*80)

    # Load data
    days = load_aligned_data()
    if not days:
        log.error("No data loaded! Check paths.")
        return

    # Sort by date
    days.sort(key=lambda d: d['date'])

    log.info(f"\nTotal days with predictions: {len(days)}")
    log.info(f"Date range: {days[0]['date']} to {days[-1]['date']}")
    log.info(f"Param tuning: first {PARAM_TUNE_DAYS} days ({days[0]['date']} to {days[min(PARAM_TUNE_DAYS-1, len(days)-1)]['date']})")
    log.info(f"TRUE OOS: remaining {len(days) - PARAM_TUNE_DAYS} days")

    # Show model availability by period
    n_3model = sum(1 for d in days if d['n_models'] >= 3)
    n_2model = sum(1 for d in days if d['n_models'] >= 2)
    log.info(f"Days with 3+ models: {n_3model}")
    log.info(f"Days with 2+ models: {n_2model}")

    # Pre-compute signals once (huge speedup)
    precompute_signals(days)

    # Primary cost structure for optimization
    primary_cost = 'ES_futures'

    # Run all approaches
    results = {}

    t0 = time.time()
    results['A'] = run_approach_a(days, cost_key=primary_cost)
    log.info(f"\nApproach A completed in {time.time()-t0:.1f}s")

    t0 = time.time()
    results['B'] = run_approach_b(days, cost_key=primary_cost)
    log.info(f"\nApproach B completed in {time.time()-t0:.1f}s")

    t0 = time.time()
    results['C'] = run_approach_c(days, cost_key=primary_cost)
    log.info(f"\nApproach C completed in {time.time()-t0:.1f}s")

    t0 = time.time()
    results['D'] = run_approach_d(days, cost_key=primary_cost)
    log.info(f"\nApproach D completed in {time.time()-t0:.1f}s")

    t0 = time.time()
    results['E'] = run_approach_e(days, cost_key=primary_cost)
    log.info(f"\nApproach E completed in {time.time()-t0:.1f}s")

    # ── Final Summary ──
    log.info("\n" + "="*80)
    log.info("FINAL SUMMARY — ALL APPROACHES (ES Futures costs)")
    log.info("="*80)

    summary_rows = []
    for key in ['A', 'B', 'C', 'D', 'E']:
        r = results.get(key)
        if r and r.get('oos_metrics'):
            m = r['oos_metrics']
            summary_rows.append({
                'approach': r['approach'],
                'trades': m['total_trades'],
                'trades_per_day': m['trades_per_day'],
                'net_pnl_ticks': m['net_pnl_ticks'],
                'net_pnl_dollars': m['net_pnl_dollars'],
                'sharpe': m['sharpe'],
                'win_rate': m['win_rate'],
                'avg_pnl_tick': m['avg_pnl_per_trade_ticks'],
                'max_dd_ticks': m['max_drawdown_ticks'],
                'config': r['best_config'],
            })

    for row in sorted(summary_rows, key=lambda x: x['sharpe'], reverse=True):
        log.info(f"\n  {row['approach']}:")
        log.info(f"    Trades: {row['trades']} ({row['trades_per_day']}/day)")
        log.info(f"    Net P&L: {row['net_pnl_ticks']:+.1f} ticks (${row['net_pnl_dollars']:+.2f})")
        log.info(f"    Sharpe: {row['sharpe']:+.2f}")
        log.info(f"    Win rate: {row['win_rate']:.1f}%")
        log.info(f"    Avg P&L/trade: {row['avg_pnl_tick']:+.3f} ticks")
        log.info(f"    Max DD: {row['max_dd_ticks']:.1f} ticks (${row['max_dd_ticks'] * TICK_VAL:.0f})")
        log.info(f"    Config: {row['config']}")

    # ── Cross-cost comparison for best approach ──
    best_approach = max(summary_rows, key=lambda x: x['sharpe']) if summary_rows else None
    if best_approach:
        log.info(f"\n\nBest approach: {best_approach['approach']} (Sharpe={best_approach['sharpe']:+.2f})")

        # Find the approach key
        for key in ['A', 'B', 'C', 'D', 'E']:
            r = results.get(key)
            if r and r['approach'] == best_approach['approach']:
                cross = run_best_across_costs(days, None, r['best_config'], r['approach'])

                log.info("\n--- Cross-Cost Summary ---")
                for ck, cm in cross.items():
                    status = "PROFITABLE" if cm['profitable'] else "UNPROFITABLE"
                    log.info(f"  {COST_STRUCTURES[ck]['label']:25s}: "
                            f"net={cm['net_pnl_ticks']:+8.1f}t "
                            f"(${cm['net_pnl_dollars']:+10.2f}) "
                            f"Sharpe={cm['sharpe']:+6.2f} "
                            f"[{status}]")
                break

    # ── Save results ──
    save_data = {
        'timestamp': datetime.now().isoformat(),
        'n_days': len(days),
        'param_tune_days': PARAM_TUNE_DAYS,
        'approaches': {},
    }
    for key in ['A', 'B', 'C', 'D', 'E']:
        r = results.get(key)
        if r:
            save_data['approaches'][key] = {
                'approach': r['approach'],
                'best_config': r['best_config'],
                'tune_sharpe': r['tune_sharpe'],
                'oos_metrics': r['oos_metrics'],
                # Don't save individual trades to keep file size small
            }

    out_path = RESULTS_DIR / f'high_conviction_{_ts}.json'
    with open(str(out_path), 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    log.info(f"\nResults saved to: {out_path}")

    log.info(f"\n{'='*80}")
    log.info(f"COMPLETED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"{'='*80}")


if __name__ == '__main__':
    main()
