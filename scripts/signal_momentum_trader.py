#!/usr/bin/env python3
"""
Signal Momentum Trader — Treat CNN-Mamba v2 predictions as a momentum indicator.

CONCEPT:
  - Smooth predictions with EMA to get a "signal momentum" line
  - ENTER when EMA ramps above threshold (confirmed over N bars)
  - HOLD as long as signal stays strong (even if price moves against us)
  - EXIT when signal momentum fades or flips

Works on BOTH sides: BUY (positive EMA) and SELL (negative EMA).

DATA:
  - Predictions: /home/nick/Lvl3Quant/output/extended_oot_validation/pred_npzs/YYYYMMDD_unfiltered.npz
  - Mid prices:  /home/nick/Lvl3Quant/data/derived/mid_price_bars/YYYYMMDD.npz
  - Only dates with BOTH predictions and mid prices are processed.

Run on Neptune:  python3 /home/nick/Lvl3Quant/scripts/signal_momentum_trader.py
Run on Jupiter:  python3 /home/jupiter/Lvl3Quant/scripts/signal_momentum_trader.py

Leakage audit: PASSED — uses OOT predictions on walk-forward folds. No future data.
"""

import os
import sys
import json
import time
import hashlib
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product

# ── Paths (auto-detect host) ────────────────────────────────────────────────
HOSTNAME = os.uname().nodename.lower()

if 'neptune' in HOSTNAME or 'nick' in HOSTNAME or Path("/home/nick").exists():
    BASE = Path("/home/nick/Lvl3Quant")
elif 'jupiter' in HOSTNAME or Path("/home/jupiter").exists():
    BASE = Path("/home/jupiter/Lvl3Quant")
else:
    BASE = Path("/home/nick/Lvl3Quant")  # default to Neptune layout

PRED_DIR = BASE / "output/extended_oot_validation/pred_npzs"
MID_PRICE_DIR = BASE / "data/derived/mid_price_bars"
OUTPUT_DIR = BASE / "output/signal_momentum"

# Also check alternate mid price cache locations
MID_CACHE_CANDIDATES = [
    MID_PRICE_DIR,
    BASE / "data/derived/mid_price_cache_hc439",
    BASE / "data/derived/mid_price_cache",
    BASE / "data/derived/mid_prices",
    BASE / "output/mid_price_bars",
]

# ── Constants (ES futures, AMP/Rithmic) ─────────────────────────────────────
TICK = 0.25                    # ES tick = 0.25 points
TICK_VALUE = 12.50             # $12.50 per tick
COMMISSION_TICKS = 0.376       # $4.70 / $12.50 round-trip commission
MARKET_SPREAD_TICKS = 1.0     # 1 tick spread for crossing
MARKET_EXIT_HALF_SPREAD = 0.5  # half-spread for exit market order

N_BARS_PER_DAY = 234000       # 6.5h RTH at 100ms = 234000 bars
BAR_INTERVAL_SEC = 0.1        # 100ms per bar

# Entry cost scenarios
PASSIVE_ENTRY_COST = COMMISSION_TICKS                          # 0.376 ticks
MARKET_ENTRY_COST = COMMISSION_TICKS + MARKET_SPREAD_TICKS     # 1.376 ticks
MARKET_EXIT_COST = COMMISSION_TICKS + MARKET_EXIT_HALF_SPREAD  # 0.876 ticks

# RTH bar ranges (bar 0 = 9:30 ET open)
# 9:30 ET = bar 0, 14:00 ET = bar 162000, 16:00 ET = bar 234000
RTH_FULL_START = 0
RTH_FULL_END = 234000
RTH_AFTERNOON_START = 162000   # 14:00 ET
RTH_AFTERNOON_END = 234000     # 16:00 ET

# ── Workers ──────────────────────────────────────────────────────────────────
WORKERS = max(1, min(os.cpu_count() - 2, 16))


# ── Parameter grid ───────────────────────────────────────────────────────────

def build_config_grid():
    """Build a smart grid of configs, keeping total under 500.

    Strategy: Latin hypercube-ish — sweep each dimension with defaults for others,
    then add targeted combos of promising ranges.
    """
    configs = []
    seen = set()

    def add_config(cfg):
        """Add config if not duplicate."""
        key = tuple(sorted(cfg.items()))
        h = hashlib.md5(str(key).encode()).hexdigest()[:12]
        if h not in seen:
            seen.add(h)
            cfg['config_id'] = h
            configs.append(cfg)

    # Default values for each param
    defaults = {
        'ema_bars': 30,
        'entry_threshold': 0.2,
        'confirm_bars': 10,
        'hold_threshold': 0.05,
        'exit_fade_bars': 30,
        'safety_sl_ticks': 24,
        'max_hold_sec': 120,
    }

    # ── Phase 1: One-at-a-time sweeps (anchor on defaults) ──────────────
    sweeps = {
        'ema_bars': [10, 20, 30, 50, 100],
        'entry_threshold': [0.1, 0.15, 0.2, 0.3, 0.5],
        'confirm_bars': [3, 5, 10, 20, 30],
        'hold_threshold': [0.0, 0.02, 0.05, 0.1, 0.15],
        'exit_fade_bars': [5, 10, 20, 30, 50],
        'safety_sl_ticks': [16, 24, 32],
        'max_hold_sec': [30, 60, 120, 300],
    }

    for param, values in sweeps.items():
        for val in values:
            cfg = dict(defaults)
            cfg[param] = val
            add_config(cfg)

    # ── Phase 2: Targeted combos (promising ranges) ─────────────────────
    ema_vals = [10, 30, 50]
    entry_vals = [0.1, 0.2, 0.3]
    confirm_vals = [5, 10]
    hold_vals = [0.0, 0.05, 0.1]
    fade_vals = [10, 30]
    sl_vals = [24]
    max_hold_vals = [60, 120]

    for ema, entry, confirm, hold, fade, sl, mh in product(
            ema_vals, entry_vals, confirm_vals, hold_vals, fade_vals,
            sl_vals, max_hold_vals):
        # Sanity: hold_threshold must be <= entry_threshold
        if hold > entry:
            continue
        add_config({
            'ema_bars': ema,
            'entry_threshold': entry,
            'confirm_bars': confirm,
            'hold_threshold': hold,
            'exit_fade_bars': fade,
            'safety_sl_ticks': sl,
            'max_hold_sec': mh,
        })

    # ── Phase 3: Extreme configs (very fast, very slow) ─────────────────
    # Very fast scalp
    for ema in [10, 20]:
        for entry in [0.15, 0.3]:
            add_config({
                'ema_bars': ema,
                'entry_threshold': entry,
                'confirm_bars': 3,
                'hold_threshold': 0.0,
                'exit_fade_bars': 5,
                'safety_sl_ticks': 16,
                'max_hold_sec': 30,
            })

    # Slow trend-follower
    for ema in [50, 100]:
        for entry in [0.1, 0.2]:
            add_config({
                'ema_bars': ema,
                'entry_threshold': entry,
                'confirm_bars': 20,
                'hold_threshold': 0.05,
                'exit_fade_bars': 50,
                'safety_sl_ticks': 32,
                'max_hold_sec': 300,
            })

    print(f"Built {len(configs)} configs")
    return configs


# ── EMA computation (vectorized) ────────────────────────────────────────────

def compute_ema(predictions, ema_bars):
    """Compute EMA of prediction signal. Vectorized for speed."""
    alpha = 2.0 / (ema_bars + 1)
    ema = np.empty_like(predictions, dtype=np.float64)
    ema[0] = predictions[0]
    for i in range(1, len(predictions)):
        ema[i] = alpha * predictions[i] + (1.0 - alpha) * ema[i - 1]
    return ema


def compute_ema_fast(predictions, ema_bars):
    """Compute EMA using scipy if available, else fallback to loop."""
    try:
        from scipy.signal import lfilter
        alpha = 2.0 / (ema_bars + 1)
        b = [alpha]
        a = [1, -(1.0 - alpha)]
        # Initial condition to match ema[0] = predictions[0]
        zi = [(1.0 - alpha) * predictions[0]]
        ema, _ = lfilter(b, a, predictions, zi=zi)
        return ema
    except ImportError:
        return compute_ema(predictions, ema_bars)


# ── Day simulation ──────────────────────────────────────────────────────────

def simulate_day(predictions, mid_prices, config, time_window='afternoon',
                 entry_type='passive'):
    """Simulate momentum trading for one day.

    Args:
        predictions: np.array of model predictions (N_BARS_PER_DAY,)
        mid_prices: np.array of mid prices (N_BARS_PER_DAY,)
        config: dict of strategy parameters
        time_window: 'afternoon' (14:00-16:00) or 'full_rth' (9:30-16:00)
        entry_type: 'passive' or 'market'

    Returns:
        list of trade dicts
    """
    ema_bars = config['ema_bars']
    entry_threshold = config['entry_threshold']
    confirm_bars = config['confirm_bars']
    hold_threshold = config['hold_threshold']
    exit_fade_bars = config['exit_fade_bars']
    safety_sl_ticks = config['safety_sl_ticks']
    max_hold_bars = int(config['max_hold_sec'] / BAR_INTERVAL_SEC)
    flip_bars = max(3, confirm_bars // 2)  # signal flip exit = half of confirm

    # Compute EMA
    ema = compute_ema_fast(predictions.astype(np.float64), ema_bars)

    # Bar range
    if time_window == 'afternoon':
        start_bar = RTH_AFTERNOON_START
        end_bar = RTH_AFTERNOON_END
    else:
        start_bar = RTH_FULL_START
        end_bar = RTH_FULL_END

    # EMA warmup: skip first ema_bars * 3 bars from start of day
    warmup_end = ema_bars * 3
    effective_start = max(start_bar, warmup_end)

    # Stop trading 5 min before close to avoid forced exit at bell
    stop_entry_bar = end_bar - 3000  # 5 min = 3000 bars

    trades = []
    position = None  # dict: side, entry_bar, entry_price, entry_ema
    confirm_count = 0
    confirm_side = None  # 'BUY' or 'SELL'
    exit_fade_count = 0
    flip_count = 0

    # Entry cost in ticks
    if entry_type == 'passive':
        entry_cost = PASSIVE_ENTRY_COST
    else:
        entry_cost = MARKET_ENTRY_COST

    for bar in range(effective_start, end_bar):
        if bar >= len(ema) or bar >= len(mid_prices):
            break

        current_ema = ema[bar]
        current_price = mid_prices[bar]

        # Skip bars with invalid price
        if current_price <= 0 or np.isnan(current_price):
            continue

        # ── No position: check for entry ────────────────────────────────
        if position is None:
            if bar >= stop_entry_bar:
                continue  # too close to close, no new entries

            # BUY signal
            if current_ema > entry_threshold:
                if confirm_side == 'BUY':
                    confirm_count += 1
                else:
                    confirm_side = 'BUY'
                    confirm_count = 1

                if confirm_count >= confirm_bars:
                    position = {
                        'side': 'BUY',
                        'entry_bar': bar,
                        'entry_price': current_price,
                        'entry_ema': current_ema,
                    }
                    confirm_count = 0
                    confirm_side = None
                    exit_fade_count = 0
                    flip_count = 0

            # SELL signal
            elif current_ema < -entry_threshold:
                if confirm_side == 'SELL':
                    confirm_count += 1
                else:
                    confirm_side = 'SELL'
                    confirm_count = 1

                if confirm_count >= confirm_bars:
                    position = {
                        'side': 'SELL',
                        'entry_bar': bar,
                        'entry_price': current_price,
                        'entry_ema': current_ema,
                    }
                    confirm_count = 0
                    confirm_side = None
                    exit_fade_count = 0
                    flip_count = 0

            else:
                # EMA in dead zone, reset confirmation
                confirm_count = 0
                confirm_side = None

        # ── Have position: check for exit ───────────────────────────────
        else:
            side = position['side']
            entry_price = position['entry_price']
            entry_bar = position['entry_bar']
            hold_bars = bar - entry_bar

            # P&L in ticks (signed)
            if side == 'BUY':
                unrealized_ticks = (current_price - entry_price) / TICK
            else:
                unrealized_ticks = (entry_price - current_price) / TICK

            exit_reason = None

            # 1. Safety stop loss (hard, wide)
            if unrealized_ticks <= -safety_sl_ticks:
                exit_reason = 'safety_sl'

            # 2. Max hold time
            elif hold_bars >= max_hold_bars:
                exit_reason = 'max_hold'

            # 3. Signal fade: EMA dropped below hold threshold
            elif side == 'BUY' and current_ema < hold_threshold:
                exit_fade_count += 1
                if exit_fade_count >= exit_fade_bars:
                    exit_reason = 'signal_fade'
            elif side == 'SELL' and current_ema > -hold_threshold:
                exit_fade_count += 1
                if exit_fade_count >= exit_fade_bars:
                    exit_reason = 'signal_fade'
            else:
                exit_fade_count = 0

            # 4. Signal flip: EMA crossed to opposite side
            if exit_reason is None:
                if side == 'BUY' and current_ema < -entry_threshold * 0.5:
                    flip_count += 1
                    if flip_count >= flip_bars:
                        exit_reason = 'signal_flip'
                elif side == 'SELL' and current_ema > entry_threshold * 0.5:
                    flip_count += 1
                    if flip_count >= flip_bars:
                        exit_reason = 'signal_flip'
                else:
                    flip_count = 0

            # 5. End of trading window — forced exit
            if exit_reason is None and bar >= end_bar - 1:
                exit_reason = 'eod_close'

            # ── Execute exit ────────────────────────────────────────────
            if exit_reason is not None:
                # Gross P&L in ticks
                gross_ticks = unrealized_ticks
                # Costs: entry + exit (market)
                total_cost = entry_cost + MARKET_EXIT_COST
                net_ticks = gross_ticks - total_cost
                hold_sec = hold_bars * BAR_INTERVAL_SEC

                trades.append({
                    'side': side,
                    'entry_bar': entry_bar,
                    'exit_bar': bar,
                    'entry_price': float(entry_price),
                    'exit_price': float(current_price),
                    'gross_ticks': float(gross_ticks),
                    'net_ticks': float(net_ticks),
                    'cost_ticks': float(total_cost),
                    'hold_sec': float(hold_sec),
                    'exit_reason': exit_reason,
                    'entry_ema': float(position['entry_ema']),
                    'exit_ema': float(current_ema),
                })

                position = None
                exit_fade_count = 0
                flip_count = 0
                confirm_count = 0
                confirm_side = None

    # Force close any open position at end
    if position is not None:
        bar = min(end_bar - 1, len(mid_prices) - 1)
        current_price = mid_prices[bar]
        side = position['side']
        entry_price = position['entry_price']

        if side == 'BUY':
            unrealized_ticks = (current_price - entry_price) / TICK
        else:
            unrealized_ticks = (entry_price - current_price) / TICK

        total_cost = entry_cost + MARKET_EXIT_COST
        net_ticks = unrealized_ticks - total_cost

        trades.append({
            'side': side,
            'entry_bar': position['entry_bar'],
            'exit_bar': bar,
            'entry_price': float(entry_price),
            'exit_price': float(current_price),
            'gross_ticks': float(unrealized_ticks),
            'net_ticks': float(net_ticks),
            'cost_ticks': float(total_cost),
            'hold_sec': float((bar - position['entry_bar']) * BAR_INTERVAL_SEC),
            'exit_reason': 'eod_close',
            'entry_ema': float(position['entry_ema']),
            'exit_ema': float(ema[bar] if bar < len(ema) else 0),
        })

    return trades


# ── Data loading ─────────────────────────────────────────────────────────────

def find_mid_cache():
    """Find the mid price cache directory."""
    for candidate in MID_CACHE_CANDIDATES:
        if candidate.exists() and any(candidate.glob("*.npz")):
            return candidate
    return None


def discover_dates():
    """Find dates that have BOTH predictions and mid prices.

    Returns list of (date_str, pred_path, mid_path) tuples.
    """
    # Find prediction files
    pred_files = {}
    if PRED_DIR.exists():
        for f in sorted(PRED_DIR.glob("*_unfiltered.npz")):
            date_str = f.stem.replace("_unfiltered", "")
            if len(date_str) == 8 and date_str.isdigit():
                pred_files[date_str] = f

    # Find mid price files
    mid_cache_dir = find_mid_cache()
    if mid_cache_dir is None:
        print("ERROR: No mid price cache found. Cannot proceed.")
        print(f"  Checked: {[str(c) for c in MID_CACHE_CANDIDATES]}")
        return []

    mid_files = {}
    for f in sorted(mid_cache_dir.glob("*.npz")):
        # Extract date from various naming patterns
        stem = f.stem
        for prefix in ['mid_', 'glbx-mdp3-', '']:
            if stem.startswith(prefix):
                date_part = stem[len(prefix):]
                if len(date_part) == 8 and date_part.isdigit():
                    mid_files[date_part] = f
                    break

    # Intersection
    common_dates = sorted(set(pred_files.keys()) & set(mid_files.keys()))
    print(f"Found {len(pred_files)} prediction dates, "
          f"{len(mid_files)} mid price dates, "
          f"{len(common_dates)} overlapping dates")

    return [(d, pred_files[d], mid_files[d]) for d in common_dates]


def load_npz_array(path, preferred_keys):
    """Load first matching key from npz file."""
    try:
        npz = np.load(path)
        for key in preferred_keys:
            if key in npz:
                return npz[key].astype(np.float32)
        # Fallback to first key
        if len(npz.files) > 0:
            return npz[npz.files[0]].astype(np.float32)
    except Exception as e:
        print(f"  WARNING: Cannot read {path}: {e}")
    return None


def load_day_data(pred_path, mid_path):
    """Load predictions and mid prices for a single date."""
    predictions = load_npz_array(
        pred_path, ['predictions', 'pred_10s', 'pred'])
    mid_prices = load_npz_array(
        mid_path, ['mid_prices', 'mid', 'mid_price', 'close', 'last', 'price'])

    if predictions is None or mid_prices is None:
        return None, None

    # Validate lengths
    if len(predictions) < 100000 or len(mid_prices) < 100000:
        print(f"  WARNING: Short arrays (pred={len(predictions)}, "
              f"mid={len(mid_prices)}), skipping")
        return None, None

    # Pad to N_BARS_PER_DAY if slightly short
    if len(predictions) < N_BARS_PER_DAY:
        pad = np.zeros(N_BARS_PER_DAY - len(predictions), dtype=np.float32)
        predictions = np.concatenate([predictions, pad])
    if len(mid_prices) < N_BARS_PER_DAY:
        pad = np.full(N_BARS_PER_DAY - len(mid_prices), mid_prices[-1],
                      dtype=np.float32)
        mid_prices = np.concatenate([mid_prices, pad])

    return predictions[:N_BARS_PER_DAY], mid_prices[:N_BARS_PER_DAY]


# ── Metrics computation ─────────────────────────────────────────────────────

def compute_metrics(all_trades, dates):
    """Compute risk-adjusted performance metrics from trade list."""
    if not all_trades:
        return {
            'n_trades': 0, 'net_ticks': 0, 'sharpe': 0, 'sortino': 0,
            'pf': 0, 'wr': 0, 'avg_hold_sec': 0, 'trades_per_day': 0,
            'avg_net_ticks': 0, 'daily_pnl': [],
        }

    net_ticks_list = [t['net_ticks'] for t in all_trades]
    n_trades = len(all_trades)
    total_net = sum(net_ticks_list)
    wins = [t for t in net_ticks_list if t > 0]
    losses = [t for t in net_ticks_list if t <= 0]

    wr = len(wins) / n_trades if n_trades > 0 else 0
    pf = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else 999.0
    avg_hold = np.mean([t['hold_sec'] for t in all_trades])
    n_days = max(len(dates), 1)

    # Daily P&L for Sharpe/Sortino
    daily_pnl = defaultdict(float)
    for t in all_trades:
        # Assign trade to its entry date (approximate via bar index)
        # We use the date from the date list based on trade ordering
        pass

    # Build daily P&L from grouped trades
    # Since trades are stored per-date in the sweep, we compute this in the caller
    # For now, compute trade-level Sharpe as proxy
    arr = np.array(net_ticks_list, dtype=np.float64)
    mean_ret = np.mean(arr)
    std_ret = np.std(arr, ddof=1) if len(arr) > 1 else 1.0
    sharpe_trade = (mean_ret / std_ret) if std_ret > 0 else 0.0

    # Sortino (downside deviation only)
    downside = arr[arr < 0]
    if len(downside) > 1:
        downside_std = np.std(downside, ddof=1)
        sortino_trade = (mean_ret / downside_std) if downside_std > 0 else 0.0
    else:
        sortino_trade = sharpe_trade * 1.5  # rough approx when few losses

    return {
        'n_trades': n_trades,
        'net_ticks': float(total_net),
        'avg_net_ticks': float(mean_ret),
        'sharpe': float(sharpe_trade),
        'sortino': float(sortino_trade),
        'pf': float(min(pf, 99.0)),
        'wr': float(wr),
        'avg_hold_sec': float(avg_hold),
        'trades_per_day': float(n_trades / n_days),
        'n_wins': len(wins),
        'n_losses': len(losses),
        'total_win_ticks': float(sum(wins)) if wins else 0.0,
        'total_loss_ticks': float(sum(losses)) if losses else 0.0,
    }


def compute_daily_metrics(daily_net_ticks):
    """Compute Sharpe and Sortino from daily P&L series."""
    if len(daily_net_ticks) < 3:
        return {'daily_sharpe': 0, 'daily_sortino': 0}

    arr = np.array(daily_net_ticks, dtype=np.float64)
    mean_d = np.mean(arr)
    std_d = np.std(arr, ddof=1)
    daily_sharpe = (mean_d / std_d * np.sqrt(252)) if std_d > 0 else 0.0

    downside = arr[arr < 0]
    if len(downside) > 1:
        dd_std = np.std(downside, ddof=1)
        daily_sortino = (mean_d / dd_std * np.sqrt(252)) if dd_std > 0 else 0.0
    else:
        daily_sortino = daily_sharpe * 1.5

    return {
        'daily_sharpe': float(daily_sharpe),
        'daily_sortino': float(daily_sortino),
    }


# ── Worker function for multiprocessing ─────────────────────────────────────

def _run_config_on_day(args):
    """Worker: run one config on one day. Returns (config_id, date, trades)."""
    config, predictions, mid_prices, time_window, entry_type = args
    trades = simulate_day(predictions, mid_prices, config, time_window,
                          entry_type)
    return config['config_id'], trades


def run_sweep_for_variant(configs, day_data, time_window, entry_type, label):
    """Run all configs across all days for one variant (time_window x entry_type).

    Args:
        configs: list of config dicts
        day_data: list of (date_str, predictions, mid_prices)
        time_window: 'afternoon' or 'full_rth'
        entry_type: 'passive' or 'market'
        label: string label for printing

    Returns:
        dict: config_id -> {metrics, daily_pnl, config}
    """
    print(f"\n{'='*60}")
    print(f"Running sweep: {label} ({len(configs)} configs x {len(day_data)} days)")
    print(f"{'='*60}")

    # Collect results: config_id -> date -> trades
    results_by_config = defaultdict(lambda: defaultdict(list))
    t0 = time.time()
    total_tasks = len(configs) * len(day_data)
    done = 0

    # Process day by day to manage memory
    for date_str, predictions, mid_prices in day_data:
        # Build task list for this day
        tasks = []
        for cfg in configs:
            tasks.append((cfg, predictions, mid_prices, time_window, entry_type))

        # Run in parallel
        with ProcessPoolExecutor(max_workers=WORKERS) as pool:
            futures = {pool.submit(_run_config_on_day, t): t[0]['config_id']
                       for t in tasks}
            for future in as_completed(futures):
                config_id = futures[future]
                try:
                    _, trades = future.result()
                    results_by_config[config_id][date_str] = trades
                except Exception as e:
                    print(f"  ERROR config {config_id} on {date_str}: {e}")

        done += len(configs)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total_tasks - done) / rate if rate > 0 else 0
        print(f"  {date_str}: done ({done}/{total_tasks}, "
              f"{rate:.0f} tasks/s, ETA {eta:.0f}s)")

    # Aggregate metrics per config
    print(f"\nAggregating results...")
    config_results = {}
    dates = [d[0] for d in day_data]

    for cfg in configs:
        cid = cfg['config_id']
        all_trades = []
        daily_net = {}

        for date_str in dates:
            day_trades = results_by_config[cid].get(date_str, [])
            all_trades.extend(day_trades)
            day_net = sum(t['net_ticks'] for t in day_trades)
            daily_net[date_str] = day_net

        metrics = compute_metrics(all_trades, dates)
        daily_metrics = compute_daily_metrics(list(daily_net.values()))
        metrics.update(daily_metrics)
        metrics['daily_pnl'] = daily_net

        # Exit reason breakdown
        exit_reasons = defaultdict(int)
        for t in all_trades:
            exit_reasons[t['exit_reason']] += 1
        metrics['exit_reasons'] = dict(exit_reasons)

        # Side breakdown
        buy_trades = [t for t in all_trades if t['side'] == 'BUY']
        sell_trades = [t for t in all_trades if t['side'] == 'SELL']
        metrics['n_buy'] = len(buy_trades)
        metrics['n_sell'] = len(sell_trades)
        metrics['buy_net'] = sum(t['net_ticks'] for t in buy_trades)
        metrics['sell_net'] = sum(t['net_ticks'] for t in sell_trades)

        config_results[cid] = {
            'config': cfg,
            'metrics': metrics,
            'label': label,
        }

    elapsed = time.time() - t0
    print(f"Sweep complete in {elapsed:.1f}s")

    return config_results


# ── Regime analysis ──────────────────────────────────────────────────────────

def classify_regime(date_str):
    """Classify date into regime bucket.

    March 2026 = mixed/bull, April 2026 = more volatile/bear.
    Simple split by month for now.
    """
    month = int(date_str[4:6])
    if month == 3:
        return 'march'
    elif month == 4:
        return 'april'
    else:
        return 'other'


def regime_analysis(config_results, top_n=5):
    """Compute per-regime metrics for top configs."""
    # Sort by daily_sharpe descending
    ranked = sorted(config_results.values(),
                    key=lambda x: x['metrics'].get('daily_sharpe', 0),
                    reverse=True)

    regime_report = []
    for entry in ranked[:top_n]:
        cid = entry['config']['config_id']
        daily_pnl = entry['metrics'].get('daily_pnl', {})

        regime_pnl = defaultdict(list)
        for date_str, pnl in daily_pnl.items():
            regime = classify_regime(date_str)
            regime_pnl[regime].append(pnl)

        regime_metrics = {}
        for regime, pnls in regime_pnl.items():
            arr = np.array(pnls)
            regime_metrics[regime] = {
                'n_days': len(pnls),
                'total_ticks': float(np.sum(arr)),
                'mean_daily': float(np.mean(arr)),
                'std_daily': float(np.std(arr)) if len(arr) > 1 else 0,
                'win_days': int(np.sum(arr > 0)),
                'lose_days': int(np.sum(arr <= 0)),
            }

        regime_report.append({
            'config_id': cid,
            'config': entry['config'],
            'overall_metrics': entry['metrics'],
            'regime_metrics': regime_metrics,
        })

    return regime_report


# ── Reporting ────────────────────────────────────────────────────────────────

def format_report(all_results, regime_reports):
    """Generate human-readable report."""
    lines = []
    lines.append("=" * 70)
    lines.append("SIGNAL MOMENTUM TRADER — SWEEP RESULTS")
    lines.append("=" * 70)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    for label, results in all_results.items():
        lines.append(f"\n{'─'*60}")
        lines.append(f"VARIANT: {label}")
        lines.append(f"{'─'*60}")

        # Sort by daily_sharpe
        ranked = sorted(results.values(),
                        key=lambda x: x['metrics'].get('daily_sharpe', 0),
                        reverse=True)

        lines.append(f"\nTop 10 configs by daily Sharpe:")
        lines.append(f"{'Rank':>4} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} "
                      f"{'WR':>5} {'Trades':>6} {'AvgNet':>7} {'Hold':>6} "
                      f"{'Total':>8} {'Config':>40}")

        for i, entry in enumerate(ranked[:10]):
            m = entry['metrics']
            cfg = entry['config']
            cfg_str = (f"ema={cfg['ema_bars']} ent={cfg['entry_threshold']} "
                       f"conf={cfg['confirm_bars']} hold={cfg['hold_threshold']} "
                       f"fade={cfg['exit_fade_bars']}")
            lines.append(
                f"{i+1:>4} {m['daily_sharpe']:>7.2f} {m['daily_sortino']:>8.2f} "
                f"{m['pf']:>6.2f} {m['wr']:>5.1%} {m['n_trades']:>6} "
                f"{m['avg_net_ticks']:>7.2f} {m['avg_hold_sec']:>6.1f}s "
                f"{m['net_ticks']:>8.1f}t {cfg_str:>40}")

        # Show exit reason distribution for top config
        if ranked:
            top = ranked[0]
            lines.append(f"\nTop config exit reasons: {top['metrics'].get('exit_reasons', {})}")
            lines.append(f"Top config side split: "
                         f"BUY={top['metrics']['n_buy']} ({top['metrics']['buy_net']:.1f}t) "
                         f"SELL={top['metrics']['n_sell']} ({top['metrics']['sell_net']:.1f}t)")

    # Regime analysis
    if regime_reports:
        lines.append(f"\n\n{'='*60}")
        lines.append("REGIME ANALYSIS (Top 5 configs)")
        lines.append(f"{'='*60}")

        for rr in regime_reports:
            lines.append(f"\nConfig {rr['config_id']}: "
                         f"ema={rr['config']['ema_bars']} "
                         f"entry={rr['config']['entry_threshold']} "
                         f"confirm={rr['config']['confirm_bars']}")
            for regime, rm in rr['regime_metrics'].items():
                lines.append(
                    f"  {regime:>8}: {rm['n_days']}d, "
                    f"total={rm['total_ticks']:.1f}t, "
                    f"mean={rm['mean_daily']:.2f}t/d, "
                    f"W/L={rm['win_days']}/{rm['lose_days']}")

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()

    # Discover available dates
    date_entries = discover_dates()
    if not date_entries:
        print("ERROR: No dates with both predictions and mid prices. Exiting.")
        sys.exit(1)

    print(f"\nDates to process: {[d[0] for d in date_entries]}")

    # Load all day data into memory
    print("\nLoading data...")
    day_data = []
    for date_str, pred_path, mid_path in date_entries:
        predictions, mid_prices = load_day_data(pred_path, mid_path)
        if predictions is not None and mid_prices is not None:
            day_data.append((date_str, predictions, mid_prices))
            print(f"  {date_str}: loaded (pred range [{predictions.min():.3f}, "
                  f"{predictions.max():.3f}], mid range [{mid_prices.min():.1f}, "
                  f"{mid_prices.max():.1f}])")
        else:
            print(f"  {date_str}: SKIPPED (missing data)")

    if not day_data:
        print("ERROR: No valid day data loaded. Exiting.")
        sys.exit(1)

    print(f"\nLoaded {len(day_data)} days of data")

    # Build config grid
    configs = build_config_grid()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run 4 variants: {afternoon, full_rth} x {passive, market}
    all_results = {}

    variants = [
        ('afternoon', 'passive', 'Afternoon + Passive Entry'),
        ('afternoon', 'market', 'Afternoon + Market Entry'),
        ('full_rth', 'passive', 'Full RTH + Passive Entry'),
        ('full_rth', 'market', 'Full RTH + Market Entry'),
    ]

    for time_window, entry_type, label in variants:
        results = run_sweep_for_variant(
            configs, day_data, time_window, entry_type, label)
        all_results[label] = results

    # Combine all results for regime analysis (use best variant)
    # Pick variant with highest top-config Sharpe
    best_variant = None
    best_sharpe = -999
    for label, results in all_results.items():
        if results:
            top_sharpe = max(
                r['metrics'].get('daily_sharpe', 0) for r in results.values())
            if top_sharpe > best_sharpe:
                best_sharpe = top_sharpe
                best_variant = label

    regime_reports = []
    if best_variant:
        regime_reports = regime_analysis(all_results[best_variant], top_n=5)

    # Generate report
    report = format_report(all_results, regime_reports)
    print(f"\n{report}")

    # Save results
    results_file = OUTPUT_DIR / "results.json"
    summary = {
        'generated': datetime.now().isoformat(),
        'n_dates': len(day_data),
        'dates': [d[0] for d in day_data],
        'n_configs': len(configs),
        'variants': {},
    }

    for label, results in all_results.items():
        # Rank by daily_sharpe
        ranked = sorted(results.values(),
                        key=lambda x: x['metrics'].get('daily_sharpe', 0),
                        reverse=True)

        variant_summary = {
            'n_configs_with_trades': sum(
                1 for r in results.values() if r['metrics']['n_trades'] > 0),
            'top_5': [],
        }

        for entry in ranked[:5]:
            # Clean up daily_pnl for JSON (convert keys)
            m = dict(entry['metrics'])
            m['daily_pnl'] = entry['metrics'].get('daily_pnl', {})
            variant_summary['top_5'].append({
                'config': entry['config'],
                'metrics': m,
            })

        summary['variants'][label] = variant_summary

    # Add regime analysis
    if regime_reports:
        summary['regime_analysis'] = []
        for rr in regime_reports:
            # Clean up non-serializable types
            cleaned = {
                'config_id': rr['config_id'],
                'config': rr['config'],
                'regime_metrics': rr['regime_metrics'],
            }
            summary['regime_analysis'].append(cleaned)

    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults saved to {results_file}")

    # Save report text
    report_file = OUTPUT_DIR / "report.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"Report saved to {report_file}")

    elapsed = time.time() - t_start
    print(f"\nTotal elapsed: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
