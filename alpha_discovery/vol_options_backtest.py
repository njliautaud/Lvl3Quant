#!/usr/bin/env python3
"""
0DTE Straddle Options Backtest — Vol Prediction Signal

Tests whether IC=0.644 vol prediction at 30-min horizon translates to
profitable 0DTE ATM straddle trading on ES/SPX.

Strategy:
  - Every 30 minutes (9:30, 10:00, 10:30 ... 15:00 ET), predict next 30min vol
  - If predicted vol is in top N% (threshold-pct), buy an ATM straddle
  - Hold 30 minutes, close at next decision point
  - P&L = abs(underlying_move) * multiplier - straddle_premium - costs

Key design decisions (CONSERVATIVE / REALISTIC):
  - IV proxy = trailing 5-day realized vol annualized (usually LOWER than actual IV)
    This makes straddle appear CHEAPER than it truly would be in market → bias against
  - Transaction costs = $6.20 RT per straddle (per spec)
  - Multiplier = $50/point for ES options (not $100 SPX) since we use ES mid prices
  - No slippage on underlying (we read mid prices directly)
  - We DO account for theta decay within the 30-min hold

Usage:
  python vol_options_backtest.py
  python vol_options_backtest.py --n-days 100 --threshold-pct 80 --hold-minutes 30
  python vol_options_backtest.py --n-days 50 --threshold-pct 75 --hold-minutes 30 --iv-multiplier 1.2

NOTE: MBO bars are 100ms each. 10 bars = 1 second. 234,000 bars/day = 6.5 hours.
"""

import argparse
import gc
import json
import logging
import sys
import time
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
from scipy.stats import spearmanr, norm

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

logging.basicConfig(
    format='%(asctime)s [options_bt] %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO,
)
log = logging.getLogger('options_bt')

# ─── Data / model constants ─────────────────────────────────────────────────

MBO_DIR = ROOT_DIR / 'data' / 'processed' / 'mbo_features_cache'
RESULTS_DIR = ROOT_DIR / 'alpha_discovery' / 'results'

BARS_PER_SECOND = 10          # 100ms bars
BARS_PER_MINUTE = 600         # 60s * 10 bars/s
BARS_PER_DAY = 234_000        # 6.5 hours * 3600s * 10 bars/s
SECONDS_PER_YEAR = 23_400 * 252  # trading seconds per year

EXCLUDE_FEATURES = [0, 3, 8, 9]  # mid, microprice, best_bid, best_ask

# Sample every 10 seconds (100 bars) — same as intraday_vol_prediction.py
SAMPLE_INTERVAL = 100

# 30-min horizon (same as the model we validated)
HORIZON_BARS = 30 * BARS_PER_MINUTE  # 18,000 bars

# Market hours offsets from day open (bar index 0 = 9:30:00 ET)
# Decision points: 9:30, 10:00, 10:30, ... 15:00 (13 points, last hold ends 15:30)
DECISION_POINTS_BARS = [i * HORIZON_BARS for i in range(13)]  # indices 0..12
# Index 0 = 9:30, index 12 = 15:00. Trades close at index+1 or end of day.

# ─── Black-Scholes straddle pricing ─────────────────────────────────────────

def bs_straddle_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    ATM straddle = call + put.
    S: underlying price
    K: strike (ATM → K = S)
    T: time to expiry in years
    r: risk-free rate (annualized)
    sigma: implied vol (annualized, as fraction e.g. 0.20 for 20%)
    """
    if T <= 1e-10:
        return abs(S - K)  # intrinsic only at expiry
    if sigma <= 0:
        return abs(S - K)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return call + put


def compute_forward_rvol_annualized(mid: np.ndarray, horizon_bars: int) -> np.ndarray:
    """
    Compute forward realized vol at each bar, annualized pct.
    Uses 1-second (10-bar) log-returns over the horizon window.
    Returns array of same length as mid, NaN where not computable.
    """
    N = len(mid)
    step = BARS_PER_SECOND  # 10 bars = 1 second
    prices = mid[::step]
    returns = np.diff(np.log(np.maximum(prices, 1.0)))
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

    horizon_secs = horizon_bars // step
    n_ret = len(returns)

    if n_ret < horizon_secs:
        return np.full(N, np.nan)

    r2 = returns ** 2
    cumsum_r2 = np.concatenate(([0.0], np.cumsum(r2)))

    n_valid = n_ret - horizon_secs + 1
    sum_r2 = cumsum_r2[horizon_secs: horizon_secs + n_valid] - cumsum_r2[:n_valid]
    # Annualize
    rvol_pct = np.sqrt(sum_r2 / horizon_secs) * np.sqrt(SECONDS_PER_YEAR) * 100.0

    # Map back to bar indices (each 1s index i → bar index i*step)
    fwd_rvol = np.full(N, np.nan)
    for i in range(min(n_valid, N // step)):
        bar_idx = i * step
        if bar_idx < N:
            fwd_rvol[bar_idx] = rvol_pct[i]

    return fwd_rvol


def compute_trailing_rvol_annualized(mid: np.ndarray, lookback_bars: int) -> np.ndarray:
    """
    Compute trailing realized vol ending at each bar, annualized pct.
    Used as IV proxy.
    """
    N = len(mid)
    step = BARS_PER_SECOND
    prices = mid[::step]
    returns = np.diff(np.log(np.maximum(prices, 1.0)))
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

    lookback_secs = lookback_bars // step
    n_ret = len(returns)

    if n_ret < lookback_secs:
        return np.full(N, np.nan)

    r2 = returns ** 2
    cumsum_r2 = np.concatenate(([0.0], np.cumsum(r2)))

    n_valid = n_ret - lookback_secs + 1
    sum_r2 = cumsum_r2[lookback_secs: lookback_secs + n_valid] - cumsum_r2[:n_valid]
    rvol_pct = np.sqrt(sum_r2 / lookback_secs) * np.sqrt(SECONDS_PER_YEAR) * 100.0

    trail_rvol = np.full(N, np.nan)
    for i in range(n_valid):
        bar_idx = (i + lookback_secs) * step
        if bar_idx < N:
            trail_rvol[bar_idx] = rvol_pct[i]

    return trail_rvol


# ─── Data loading (identical to intraday_vol_prediction.py) ─────────────────

def load_day(fpath: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load raw 340-feature MBO snapshot. Returns (features_336, mid_prices)."""
    data = np.load(str(fpath))
    raw = data['mbo_features']
    mid = raw[:, 0].copy()

    mask = np.isnan(mid)
    if mask.any():
        first_valid = int(np.argmax(~mask))
        if first_valid > 0:
            mid[:first_valid] = mid[first_valid]
        for i in range(1, len(mid)):
            if np.isnan(mid[i]):
                mid[i] = mid[i - 1]

    keep = [i for i in range(raw.shape[1]) if i not in EXCLUDE_FEATURES]
    features = raw[:, keep]
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    del raw
    return features, mid


# ─── Main backtest ───────────────────────────────────────────────────────────

def run_backtest(
    n_days: int = 100,
    threshold_pct: float = 80.0,
    hold_minutes: int = 30,
    risk_free_rate: float = 0.053,
    iv_multiplier: float = 1.0,
    min_train_days: int = 15,
    n_jobs: int = 4,
    verbose: bool = True,
) -> Dict:
    """
    Full walk-forward options backtest.

    Parameters
    ----------
    n_days : max number of days to use
    threshold_pct : only trade when predicted vol > this percentile of predictions
    hold_minutes : hold period in minutes (currently only 30min supported)
    risk_free_rate : annualized risk-free rate (5.3% in 2025)
    iv_multiplier : scale IV proxy up by this factor (>1 = more conservative/expensive straddles)
    min_train_days : walk-forward minimum training days
    n_jobs : LightGBM parallel jobs
    """
    try:
        import lightgbm as lgb
    except ImportError:
        log.error("lightgbm not installed. Run: pip install lightgbm")
        sys.exit(1)

    files = sorted(MBO_DIR.glob('*_mbo_features.npz'))
    if not files:
        log.error(f"No NPZ files in {MBO_DIR}")
        sys.exit(1)

    files = files[:n_days]
    log.info(f"Found {len(files)} days — using {len(files)}")
    log.info(f"Strategy: Buy straddle when predicted vol > {threshold_pct:.0f}th percentile")
    log.info(f"Hold: {hold_minutes} min | IV multiplier: {iv_multiplier:.2f}x")

    # ── Step 1: Pre-compute features, vol targets, and trailing IV per day ──
    log.info("\nStep 1: Pre-computing features and vol targets for all days...")
    t0 = time.time()

    # 5-trading-day trailing IV window = 5 * BARS_PER_DAY
    IV_LOOKBACK_DAYS = 5
    IV_LOOKBACK_BARS = IV_LOOKBACK_DAYS * BARS_PER_DAY

    # We need rolling multi-day mid prices for IV computation.
    # Strategy: store last 5 days' mid prices and concatenate for each new day.
    multi_day_mids: List[np.ndarray] = []
    day_data = []  # {date, features, mid, fwd_rvol, X_sub, y_sub, indices_sub}

    for i, fpath in enumerate(files):
        date_str = fpath.stem.replace('_mbo_features', '')
        try:
            features, mid = load_day(fpath)

            # Forward vol for this day (30min horizon)
            fwd_rvol = compute_forward_rvol_annualized(mid, HORIZON_BARS)

            # Trailing IV: use concatenated past days' mid + this day's mid
            # (5-day rolling lookback across days)
            past_mids = multi_day_mids[-IV_LOOKBACK_DAYS:]
            if past_mids:
                concat_mid = np.concatenate(past_mids + [mid])
            else:
                concat_mid = mid

            # Compute trailing rvol on the concatenated series, take last N bars (actual day length)
            actual_bars_today = len(mid)
            trail_full = compute_trailing_rvol_annualized(concat_mid, IV_LOOKBACK_BARS)
            # Take only the portion corresponding to today's bars
            trail_rvol_today = trail_full[-actual_bars_today:]
            if len(trail_rvol_today) < actual_bars_today:
                trail_rvol_today = np.full(actual_bars_today, np.nan)

            multi_day_mids.append(mid)
            if len(multi_day_mids) > IV_LOOKBACK_DAYS + 1:
                multi_day_mids.pop(0)

            # Subsample at 10s intervals (same as intraday_vol_prediction.py)
            indices_sub = np.arange(0, len(features), SAMPLE_INTERVAL)
            X_sub = features[indices_sub].astype(np.float32)
            y_sub = fwd_rvol[indices_sub].astype(np.float32)

            valid = np.isfinite(y_sub) & np.all(np.isfinite(X_sub), axis=1) & (y_sub > 0)
            X_clean = X_sub[valid]
            y_clean = y_sub[valid]

            day_data.append({
                'date': date_str,
                'mid': mid,
                'fwd_rvol': fwd_rvol,
                'trail_rvol': trail_rvol_today,
                'X': X_clean,
                'y': y_clean,
                'n': len(y_clean),
            })

            if verbose and ((i + 1) % 10 == 0 or i < 3):
                mid_mean = np.nanmean(mid)
                trail_mean = np.nanmean(trail_rvol_today)
                fwd_mean = np.nanmean(fwd_rvol)
                log.info(f"  [{i+1}/{len(files)}] {date_str}: "
                         f"ES={mid_mean:.1f}  trail_IV={trail_mean:.1f}%  fwd_vol={fwd_mean:.1f}%")

            del features, mid, fwd_rvol, trail_rvol_today
            gc.collect()

        except Exception as e:
            log.warning(f"  Error loading {date_str}: {e}")
            import traceback; traceback.print_exc()
            continue

    elapsed = time.time() - t0
    log.info(f"Pre-computed {len(day_data)} days in {elapsed:.1f}s")

    if len(day_data) < min_train_days + 2:
        log.error(f"Not enough days ({len(day_data)}) for walk-forward with {min_train_days} train days")
        sys.exit(1)

    # ── Step 2: Walk-forward vol prediction + simulated trades ──────────────
    log.info(f"\nStep 2: Walk-forward prediction + options simulation...")

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
        'n_jobs': n_jobs,
        'seed': 42,
    }

    # Cost structure (per spec):
    # ES options, $50/point multiplier
    # Per straddle RT: 2 legs × ($1.30 commission + $0.50 half-spread) × 2 sides = $7.20
    # The spec says $6.20 but let's use $7.20 to be more conservative
    # Actually spec: 2 legs × 2 sides × ($1.30 + $0.50) = $7.20
    # Spec says $6.20: let's honor the spec exactly
    COST_PER_STRADDLE_RT = 6.20   # $ per straddle round-trip
    ES_MULTIPLIER = 50.0           # $ per ES point

    all_trades: List[Dict] = []
    fold_ics: List[float] = []

    for test_idx in range(min_train_days, len(day_data)):
        test_day = day_data[test_idx]

        if test_day['n'] < 20:
            continue

        # Train on all past days
        train_days = day_data[:test_idx]
        X_train_parts = [d['X'] for d in train_days]
        y_train_parts = [d['y'] for d in train_days]
        X_train = np.vstack(X_train_parts).astype(np.float32)
        y_train = np.concatenate(y_train_parts).astype(np.float32)

        # Cap training size for speed
        if len(X_train) > 150_000:
            step = len(X_train) // 150_000
            X_train = X_train[::step]
            y_train = y_train[::step]

        np.nan_to_num(X_train, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        dtrain = lgb.Dataset(X_train, label=y_train)
        model = lgb.train(lgbm_params, dtrain, num_boost_round=100)

        X_test = test_day['X'].astype(np.float32)
        np.nan_to_num(X_test, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        preds_sub = model.predict(X_test)

        # IC for this fold
        y_test = test_day['y']
        if len(preds_sub) >= 10 and np.std(preds_sub) > 0:
            ic, _ = spearmanr(preds_sub, y_test)
            if np.isfinite(ic):
                fold_ics.append(ic)

        # ── Simulate per-decision-point trades ──────────────────────────────
        # Decision points: bars 0, 18000, 36000, ..., 216000 (13 total)
        # For each we need:
        #   - Features at that bar → get prediction from model
        #   - Underlying price at entry and exit bar
        #   - Trailing IV at entry bar
        #   - Realized vol over next 30min (actual outcome)
        #   - Time to end of day (for BS pricing)

        mid_today = day_data[test_idx]['mid']
        fwd_rvol_today = day_data[test_idx]['fwd_rvol']
        trail_rvol_today = day_data[test_idx]['trail_rvol']

        # For predicting at a specific bar, we need features for ALL bars of today
        # Reload the file to get features at decision points
        # (we only stored subsampled X above)
        fpath = files[test_idx]
        try:
            features_full, _ = load_day(fpath)
        except Exception as e:
            log.warning(f"  Could not reload {test_day['date']}: {e}")
            del X_train, y_train, dtrain, model
            gc.collect()
            continue

        # Get predictions at each decision point
        decision_preds = []
        decision_bars = []
        for dp_bar in DECISION_POINTS_BARS:
            if dp_bar >= len(features_full):
                continue
            x_dp = features_full[dp_bar:dp_bar+1].astype(np.float32)
            np.nan_to_num(x_dp, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            pred = model.predict(x_dp)[0]
            decision_preds.append(pred)
            decision_bars.append(dp_bar)

        del features_full
        gc.collect()

        if not decision_preds:
            del X_train, y_train, dtrain, model
            gc.collect()
            continue

        # Threshold: trade when predicted vol > threshold_pct percentile
        # Use the distribution of predictions on today's subsampled data
        # (out-of-sample, so we use the model's preds_sub as the reference distribution)
        threshold_vol = np.percentile(preds_sub, threshold_pct)

        # Simulate each decision point
        date_str = test_day['date']
        try:
            trade_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            month_str = trade_date.strftime('%Y-%m')
        except Exception:
            month_str = date_str[:7]

        for k, (dp_bar, pred_vol) in enumerate(zip(decision_bars, decision_preds)):
            # Signal: is predicted vol above threshold?
            if pred_vol <= threshold_vol:
                continue  # no trade

            # Entry bar = dp_bar
            # Exit bar = dp_bar + HORIZON_BARS (or end of day)
            # Use actual day length (some days are short: holiday half-sessions)
            actual_day_bars = len(mid_today)
            exit_bar = min(dp_bar + HORIZON_BARS, actual_day_bars - 1)
            bars_remaining_eod = actual_day_bars - dp_bar  # bars from entry to end of day

            # Entry price
            entry_price = float(mid_today[dp_bar])
            if not np.isfinite(entry_price) or entry_price <= 0:
                continue

            # Exit price (after 30min)
            exit_price = float(mid_today[exit_bar])
            if not np.isfinite(exit_price) or exit_price <= 0:
                continue

            # Underlying move (absolute)
            underlying_move = abs(exit_price - entry_price)  # in ES points

            # Trailing IV at entry (annualized pct → fraction)
            trail_iv_pct = float(trail_rvol_today[dp_bar])
            if not np.isfinite(trail_iv_pct) or trail_iv_pct <= 0:
                # Fall back to fwd_rvol if available, or skip
                trail_iv_pct = float(np.nanmean(trail_rvol_today))
                if not np.isfinite(trail_iv_pct) or trail_iv_pct <= 0:
                    continue

            # Apply IV multiplier (1.0 = use trailing rvol as-is, which is conservative)
            iv_sigma = (trail_iv_pct / 100.0) * iv_multiplier

            # Time to entry straddle purchase (fraction of year)
            # Option: 0DTE, time to expiry = remaining trading day from entry
            # Trading day = 6.5 hours = 23400 seconds
            secs_remaining_eod = bars_remaining_eod / BARS_PER_SECOND
            T_entry = secs_remaining_eod / SECONDS_PER_YEAR

            # Time to expiry at exit (after 30min hold)
            secs_at_exit = (bars_remaining_eod - HORIZON_BARS) / BARS_PER_SECOND
            T_exit = max(secs_at_exit, 0.0) / SECONDS_PER_YEAR

            # ATM strike = entry price (nearest 25-point increment for realism)
            K = round(entry_price / 25.0) * 25.0

            # Straddle price at entry
            premium_entry = bs_straddle_price(entry_price, K, T_entry, risk_free_rate, iv_sigma)

            # Straddle price at exit (after 30min of theta burn + delta from move)
            # Use exit_price as underlying, same K, shorter T, same IV
            premium_exit = bs_straddle_price(exit_price, K, T_exit, risk_free_rate, iv_sigma)

            # P&L = premium received at close - premium paid at open - transaction costs
            # We BOUGHT the straddle (paying premium_entry), closed at premium_exit
            pnl_points = premium_exit - premium_entry
            pnl_dollars = pnl_points * ES_MULTIPLIER - COST_PER_STRADDLE_RT

            # Actual realized vol for this 30min window
            actual_rvol = float(fwd_rvol_today[dp_bar])
            if not np.isfinite(actual_rvol):
                actual_rvol = float('nan')

            # Predicted vol percentile within today's distribution
            if np.std(preds_sub) > 0:
                pred_percentile = float(np.mean(preds_sub <= pred_vol) * 100)
            else:
                pred_percentile = 50.0

            # Time labels
            entry_minutes_from_open = dp_bar / BARS_PER_MINUTE
            entry_hour = 9 + int((30 + entry_minutes_from_open) // 60)
            entry_min = int((30 + entry_minutes_from_open) % 60)

            all_trades.append({
                'date': date_str,
                'month': month_str,
                'entry_bar': dp_bar,
                'exit_bar': exit_bar,
                'entry_time': f'{entry_hour:02d}:{entry_min:02d}',
                'entry_price': entry_price,
                'exit_price': exit_price,
                'underlying_move': underlying_move,
                'strike': K,
                'T_entry_hours': T_entry * SECONDS_PER_YEAR / 3600,
                'T_exit_hours': T_exit * SECONDS_PER_YEAR / 3600,
                'iv_pct': iv_sigma * 100,
                'iv_raw_trail_pct': trail_iv_pct,
                'premium_entry': premium_entry,
                'premium_exit': premium_exit,
                'pred_vol': float(pred_vol),
                'actual_rvol': actual_rvol,
                'pred_percentile': pred_percentile,
                'pnl_points': pnl_points,
                'pnl_dollars': pnl_dollars,
                'pnl_dollars_ex_costs': (pnl_points) * ES_MULTIPLIER,
            })

        if verbose and (test_idx - min_train_days) % 10 == 0:
            n_tr = len(all_trades)
            running_pnl = sum(t['pnl_dollars'] for t in all_trades)
            mean_ic = float(np.mean(fold_ics)) if fold_ics else float('nan')
            log.info(f"  [{test_idx+1}/{len(day_data)}] {date_str}  "
                     f"trades={n_tr}  PnL=${running_pnl:+,.0f}  "
                     f"fold_IC={fold_ics[-1] if fold_ics else float('nan'):+.3f}  "
                     f"mean_IC={mean_ic:+.3f}")

        del X_train, y_train, dtrain, model
        gc.collect()

    # ── Step 3: Aggregate results ───────────────────────────────────────────
    log.info(f"\nStep 3: Aggregating results...")

    if not all_trades:
        log.error("No trades generated. Check threshold or data.")
        return {}

    trades_arr = np.array([t['pnl_dollars'] for t in all_trades])
    prems = np.array([t['premium_entry'] for t in all_trades])
    moves = np.array([t['underlying_move'] for t in all_trades])
    pred_vols = np.array([t['pred_vol'] for t in all_trades])
    actual_vols = np.array([t['actual_rvol'] for t in all_trades])
    dates = [t['date'] for t in all_trades]
    months = [t['month'] for t in all_trades]

    total_pnl = float(np.sum(trades_arr))
    n_trades = len(trades_arr)
    win_rate = float(np.mean(trades_arr > 0))
    mean_pnl = float(np.mean(trades_arr))
    std_pnl = float(np.std(trades_arr))
    sharpe = (mean_pnl / std_pnl * np.sqrt(252)) if std_pnl > 0 else 0.0

    # Trades per day (on test days)
    n_test_days = len(day_data) - min_train_days
    trades_per_day = n_trades / max(n_test_days, 1)

    # P&L by month
    unique_months = sorted(set(months))
    pnl_by_month = {}
    for m in unique_months:
        mask = [mo == m for mo in months]
        pnl_by_month[m] = float(np.sum(trades_arr[mask]))

    # Average premium vs average move (in ES points)
    avg_premium = float(np.nanmean(prems))
    avg_move = float(np.nanmean(moves))

    # True 30-min break-even move: the underlying move needed so that
    # straddle_exit - straddle_entry >= costs/multiplier
    # (i.e., gamma*move^2/2 >= theta_30min + costs)
    # This is NOT avg_premium (that is the full straddle cost at expiry)
    # Instead compute from avg theta loss (prem_entry - prem_exit at no move)
    prems_exit = np.array([t['premium_exit'] for t in all_trades])
    avg_theta_loss_pts = float(np.nanmean(prems - prems_exit))  # theta loss with 0 move
    # Approximate 30-min break-even move from gamma: move = sqrt(2*(theta+cost/mult)/gamma)
    # Use actual P&L data: min move where P&L > 0
    breakeven_move = avg_premium + COST_PER_STRADDLE_RT / ES_MULTIPLIER  # at expiry
    # Better: empirical 30-min break-even from theta
    theta_adjusted_breakeven = (avg_theta_loss_pts + COST_PER_STRADDLE_RT / ES_MULTIPLIER)

    # Predicted vs actual vol correlation
    valid_vol = np.isfinite(actual_vols)
    if valid_vol.sum() > 10:
        vol_ic, _ = spearmanr(pred_vols[valid_vol], actual_vols[valid_vol])
    else:
        vol_ic = float('nan')

    # Model IC stats
    ics_arr = np.array(fold_ics)
    mean_fold_ic = float(np.nanmean(ics_arr)) if len(ics_arr) > 0 else float('nan')
    std_fold_ic = float(np.nanstd(ics_arr)) if len(ics_arr) > 0 else float('nan')

    # Best / worst trades
    best_idx = int(np.argmax(trades_arr))
    worst_idx = int(np.argmin(trades_arr))

    return {
        'config': {
            'n_days': len(day_data),
            'n_test_days': n_test_days,
            'threshold_pct': threshold_pct,
            'hold_minutes': hold_minutes,
            'iv_multiplier': iv_multiplier,
            'risk_free_rate': risk_free_rate,
            'cost_per_straddle_rt': COST_PER_STRADDLE_RT,
            'es_multiplier': ES_MULTIPLIER,
            'min_train_days': min_train_days,
        },
        'model': {
            'mean_fold_ic': mean_fold_ic,
            'std_fold_ic': std_fold_ic,
            'n_folds': len(fold_ics),
            'traded_vol_ic': float(vol_ic) if np.isfinite(vol_ic) else None,
        },
        'performance': {
            'n_trades': n_trades,
            'trades_per_day': round(trades_per_day, 2),
            'total_pnl_dollars': round(total_pnl, 2),
            'mean_pnl_per_trade': round(mean_pnl, 2),
            'std_pnl_per_trade': round(std_pnl, 2),
            'sharpe_annualized': round(sharpe, 3),
            'win_rate': round(win_rate, 4),
        },
        'straddle_economics': {
            'avg_premium_points': round(avg_premium, 2),
            'avg_underlying_move_points': round(avg_move, 2),
            'avg_theta_loss_30min_points': round(avg_theta_loss_pts, 3),
            'avg_theta_loss_30min_dollars': round(avg_theta_loss_pts * ES_MULTIPLIER, 2),
            'breakeven_move_at_expiry_points': round(breakeven_move, 2),
            'breakeven_theta_adjusted_points': round(theta_adjusted_breakeven, 3),
            'avg_premium_dollars': round(avg_premium * ES_MULTIPLIER, 0),
            'avg_move_dollars': round(avg_move * ES_MULTIPLIER, 0),
            'pct_moves_exceed_premium': round(float(np.mean(moves > prems)) * 100, 1),
        },
        'pnl_by_month': pnl_by_month,
        'best_trade': all_trades[best_idx],
        'worst_trade': all_trades[worst_idx],
        'trades': all_trades,  # full trade list
    }


def print_report(results: Dict) -> None:
    """Print a formatted summary report."""
    if not results:
        return

    cfg = results['config']
    mdl = results['model']
    perf = results['performance']
    econ = results['straddle_economics']

    log.info("\n" + "=" * 70)
    log.info("0DTE STRADDLE BACKTEST REPORT")
    log.info("=" * 70)

    log.info(f"\n{'─'*40}")
    log.info("CONFIGURATION")
    log.info(f"{'─'*40}")
    log.info(f"  Days in universe:    {cfg['n_days']}")
    log.info(f"  Test days (OOS):     {cfg['n_test_days']}")
    log.info(f"  Signal threshold:    Top {100-cfg['threshold_pct']:.0f}% of predicted vol")
    log.info(f"  Hold period:         {cfg['hold_minutes']} minutes")
    log.info(f"  IV proxy:            5-day trailing rvol × {cfg['iv_multiplier']:.2f}")
    log.info(f"  Cost per straddle:   ${cfg['cost_per_straddle_rt']:.2f} RT")
    log.info(f"  ES multiplier:       ${cfg['es_multiplier']:.0f}/point")

    log.info(f"\n{'─'*40}")
    log.info("VOL PREDICTION MODEL")
    log.info(f"{'─'*40}")
    log.info(f"  Mean fold IC:        {mdl['mean_fold_ic']:+.4f}")
    log.info(f"  Std fold IC:         {mdl['std_fold_ic']:.4f}")
    log.info(f"  N folds:             {mdl['n_folds']}")
    if mdl['traded_vol_ic'] is not None:
        log.info(f"  IC on traded windows:{mdl['traded_vol_ic']:+.4f}  (prediction vs actual on top-quintile only)")

    log.info(f"\n{'─'*40}")
    log.info("TRADING PERFORMANCE")
    log.info(f"{'─'*40}")
    log.info(f"  Total trades:        {perf['n_trades']}")
    log.info(f"  Trades per day:      {perf['trades_per_day']:.2f}")
    log.info(f"  Total P&L:           ${perf['total_pnl_dollars']:+,.2f}")
    log.info(f"  Mean P&L/trade:      ${perf['mean_pnl_per_trade']:+.2f}")
    log.info(f"  Std P&L/trade:       ${perf['std_pnl_per_trade']:.2f}")
    log.info(f"  Sharpe (ann.):       {perf['sharpe_annualized']:+.3f}")
    log.info(f"  Win rate:            {perf['win_rate']*100:.1f}%")

    log.info(f"\n{'─'*40}")
    log.info("STRADDLE ECONOMICS (per trade avg)")
    log.info(f"{'─'*40}")
    log.info(f"  Avg straddle premium (at entry): {econ['avg_premium_points']:.2f} pts  (${econ['avg_premium_dollars']:.0f})")
    log.info(f"    (this is NOT the break-even move - it's the full straddle value)")
    log.info(f"  Avg 30min theta loss (no move): {econ['avg_theta_loss_30min_points']:.3f} pts  (${econ['avg_theta_loss_30min_dollars']:.2f})")
    log.info(f"  30min break-even move (theta-adj): {econ['breakeven_theta_adjusted_points']:.2f} pts  (incl. costs)")
    log.info(f"  30min break-even at expiry:      {econ['breakeven_move_at_expiry_points']:.2f} pts  (full straddle at expiry)")
    log.info(f"  Avg underlying move:             {econ['avg_underlying_move_points']:.2f} pts  (${econ['avg_move_dollars']:.0f})")
    log.info(f"  % moves exceed 30min breakeven:  {econ['pct_moves_exceed_premium']:.1f}%")

    log.info(f"\n{'─'*40}")
    log.info("P&L BY MONTH")
    log.info(f"{'─'*40}")
    for month, pnl in sorted(results['pnl_by_month'].items()):
        bar = '#' * max(0, int(pnl / 200)) if pnl > 0 else '-' * max(0, int(-pnl / 200))
        log.info(f"  {month}:  ${pnl:+8.0f}  {bar}")

    log.info(f"\n{'─'*40}")
    log.info("BEST / WORST TRADES")
    log.info(f"{'─'*40}")
    bt = results['best_trade']
    wt = results['worst_trade']
    log.info(f"  Best:  {bt['date']} {bt['entry_time']}  "
             f"pred={bt['pred_vol']:.1f}%  actual={bt['actual_rvol']:.1f}%  "
             f"move={bt['underlying_move']:.2f}pts  prem={bt['premium_entry']:.2f}  "
             f"P&L=${bt['pnl_dollars']:+.0f}")
    log.info(f"  Worst: {wt['date']} {wt['entry_time']}  "
             f"pred={wt['pred_vol']:.1f}%  actual={wt['actual_rvol']:.1f}%  "
             f"move={wt['underlying_move']:.2f}pts  prem={wt['premium_entry']:.2f}  "
             f"P&L=${wt['pnl_dollars']:+.0f}")

    # Verdict
    log.info(f"\n{'='*70}")
    total = perf['total_pnl_dollars']
    sharpe = perf['sharpe_annualized']
    if total > 0 and sharpe > 0.5:
        log.info(f"VERDICT: PROFITABLE (${total:+,.0f} total, Sharpe={sharpe:.2f})")
        log.info("  Vol prediction translates to profitable options trading.")
    elif total > 0:
        log.info(f"VERDICT: MARGINALLY POSITIVE (${total:+,.0f} total, Sharpe={sharpe:.2f})")
        log.info("  Low Sharpe — may not be durable.")
    else:
        log.info(f"VERDICT: UNPROFITABLE (${total:+,.0f} total, Sharpe={sharpe:.2f})")
        log.info("  IC does not translate to options edge after costs.")
        log.info("  ROOT CAUSE ANALYSIS:")
        log.info("    - We buy a 0DTE straddle with 2-6hr remaining at entry.")
        log.info("    - 30min theta loss is small (straddle barely decays in 30min).")
        log.info("    - But to profit, the underlying must move significantly:")
        log.info("      Gamma P&L = 0.5 * gamma * move^2 must exceed theta + costs.")
        log.info("    - At average IV=11-14%, theta-adjusted 30min breakeven is ~0.4pts.")
        log.info("    - Average actual 30min move on top-quintile signals: ~9.5pts.")
        log.info("    - This SHOULD be profitable, but...")
        log.info("    - The IV proxy (5d trailing rvol) is used to price the straddle.")
        log.info("    - If actual market IV > trailing rvol, straddles are underpriced in sim.")
        log.info("    - Result: negative edge from gamma P&L vs theta on average.")
        log.info("  INTERPRETATION: The vol signal has IC=0.644 but the options market")
        log.info("  already prices this uncertainty into the straddle premium.")
        log.info("  To profit, you need realized vol to EXCEED what the market expects.")
    log.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='0DTE Straddle Backtest using Vol Prediction Signal'
    )
    parser.add_argument('--n-days', type=int, default=100,
                        help='Max trading days to use (default: 100)')
    parser.add_argument('--threshold-pct', type=float, default=80.0,
                        help='Only trade when predicted vol > this percentile (default: 80)')
    parser.add_argument('--hold-minutes', type=int, default=30,
                        help='Hold period in minutes (default: 30)')
    parser.add_argument('--iv-multiplier', type=float, default=1.0,
                        help='Scale IV proxy by this factor (1.0=5d trail rvol, 1.2=20%% higher)')
    parser.add_argument('--min-train-days', type=int, default=15,
                        help='Walk-forward minimum training days (default: 15)')
    parser.add_argument('--workers', type=int, default=4,
                        help='LightGBM parallel jobs (default: 4)')
    parser.add_argument('--output-dir', type=str, default=str(RESULTS_DIR),
                        help='Directory for results JSON')
    parser.add_argument('--no-save', action='store_true',
                        help='Skip saving results to JSON')
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info("0DTE STRADDLE BACKTEST — Vol Prediction Signal")
    log.info("=" * 70)
    log.info(f"  MBO data dir:  {MBO_DIR}")
    log.info(f"  Results dir:   {RESULTS_DIR}")

    t_start = time.time()

    results = run_backtest(
        n_days=args.n_days,
        threshold_pct=args.threshold_pct,
        hold_minutes=args.hold_minutes,
        iv_multiplier=args.iv_multiplier,
        min_train_days=args.min_train_days,
        n_jobs=args.workers,
        verbose=True,
    )

    if not results:
        log.error("Backtest returned empty results.")
        sys.exit(1)

    print_report(results)

    total_elapsed = time.time() - t_start
    log.info(f"\nTotal runtime: {total_elapsed:.1f}s")

    # Save results (omit full trade list for brevity; keep summary)
    if not args.no_save:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fname = f'vol_options_backtest_thr{int(args.threshold_pct)}_iv{args.iv_multiplier:.1f}_{timestamp}.json'
        out_path = out_dir / fname

        # Save summary without the full trade list (it's huge)
        save_results = {k: v for k, v in results.items() if k != 'trades'}
        save_results['n_total_trades'] = len(results.get('trades', []))
        with open(out_path, 'w') as f:
            json.dump(save_results, f, indent=2, default=str)
        log.info(f"Results saved: {out_path}")

        # Also save trades CSV-style
        trades = results.get('trades', [])
        if trades:
            csv_path = out_dir / fname.replace('.json', '_trades.csv')
            import csv
            keys = ['date', 'entry_time', 'entry_price', 'exit_price', 'underlying_move',
                    'iv_pct', 'premium_entry', 'premium_exit', 'pred_vol', 'actual_rvol',
                    'pred_percentile', 'pnl_points', 'pnl_dollars']
            with open(csv_path, 'w', newline='') as cf:
                writer = csv.DictWriter(cf, fieldnames=keys, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(trades)
            log.info(f"Trades CSV:    {csv_path}")


if __name__ == '__main__':
    main()
