#!/usr/bin/env python3
"""
Multi-DTE Straddle Options Backtest — Vol Prediction Signal

Extends the 0DTE straddle backtest to test longer DTEs (0, 1, 3, 5, 7 days to expiry).

Key insight:
  - 0DTE failed because straddle costs ~44pts but avg top-quintile move = 9.5pts
  - With longer DTE, we buy MORE time value but LESS theta decay per 30min hold
  - With longer DTE, vega exposure increases — if realized vol > IV, straddle gains even without move

Strategy for each DTE:
  - Entry: Buy straddle priced with T = DTE * 6.5hr + remaining_trading_day_hours
  - Hold: 30 minutes (same 13 decision points per day, 9:30-15:00)
  - Exit: Re-price straddle with T = (DTE * 6.5hr + remaining_eod_hours - 0.5hr)
          i.e., exactly 30min less time value
  - Entry signal: same as before — predicted vol in top 20%

P&L components:
  1. Gamma P&L: underlying moves in our favor
  2. Theta loss: LESS per 30min with longer DTE (lower theta per unit time)
  3. Vega P&L: if realized vol exceeds trailing-rvol IV proxy, straddle gains value

Usage:
  python vol_options_backtest_multi_dte.py
  python vol_options_backtest_multi_dte.py --n-days 100 --threshold-pct 80
  python vol_options_backtest_multi_dte.py --dtes 0 1 3 5 7 --iv-multiplier 1.2

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
    format='%(asctime)s [multi_dte_bt] %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO,
)
log = logging.getLogger('multi_dte_bt')

# ─── Data / model constants ─────────────────────────────────────────────────

MBO_DIR = ROOT_DIR / 'data' / 'processed' / 'mbo_features_cache'
RESULTS_DIR = ROOT_DIR / 'alpha_discovery' / 'results'

BARS_PER_SECOND = 10          # 100ms bars
BARS_PER_MINUTE = 600         # 60s * 10 bars/s
BARS_PER_DAY = 234_000        # 6.5 hours * 3600s * 10 bars/s
SECONDS_PER_YEAR = 23_400 * 252  # trading seconds per year
HOURS_PER_TRADING_DAY = 6.5
SECS_PER_TRADING_DAY = HOURS_PER_TRADING_DAY * 3600

EXCLUDE_FEATURES = [0, 3, 8, 9]  # mid, microprice, best_bid, best_ask

# Sample every 10 seconds (100 bars) — same as intraday_vol_prediction.py
SAMPLE_INTERVAL = 100

# 30-min horizon (same as the model we validated)
HORIZON_BARS = 30 * BARS_PER_MINUTE  # 18,000 bars

# Market hours offsets from day open (bar index 0 = 9:30:00 ET)
# Decision points: 9:30, 10:00, 10:30, ... 15:00 (13 points, last hold ends 15:30)
DECISION_POINTS_BARS = [i * HORIZON_BARS for i in range(13)]


# ─── Black-Scholes straddle pricing ─────────────────────────────────────────

def bs_straddle_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    ATM straddle = call + put.
    S: underlying price
    K: strike (ATM -> K = S)
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
    rvol_pct = np.sqrt(sum_r2 / horizon_secs) * np.sqrt(SECONDS_PER_YEAR) * 100.0

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


# ─── Data loading ─────────────────────────────────────────────────────────

def load_day(fpath: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load raw MBO snapshot. Returns (features_336, mid_prices)."""
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


# ─── Simulate trades for one DTE ─────────────────────────────────────────

def simulate_trades_for_dte(
    dte: int,
    test_day: Dict,
    features_full: np.ndarray,
    model,
    preds_sub: np.ndarray,
    threshold_vol: float,
    risk_free_rate: float,
    iv_multiplier: float,
    cost_per_straddle_rt: float,
    es_multiplier: float,
) -> List[Dict]:
    """
    Simulate all trades for a single day at a given DTE.

    DTE mechanics:
      - We enter a straddle with DTE calendar days remaining until expiry
      - The option has T_entry = DTE * 6.5hr + remaining_today_hours of time value
      - We hold for 30min, then exit
      - T_exit = T_entry - 30min (exactly 30min less time value)
      - This is the key change vs 0DTE: T is much larger, so theta loss per 30min is smaller,
        but premium is much higher (more time value to buy)

    For DTE=0: T_entry = remaining today (same as original script)
    For DTE=1: T_entry = 6.5hr + remaining today (option expires tomorrow EOD)
    For DTE=3: T_entry = 3 * 6.5hr + remaining today
    etc.
    """
    mid_today = test_day['mid']
    fwd_rvol_today = test_day['fwd_rvol']
    trail_rvol_today = test_day['trail_rvol']
    date_str = test_day['date']

    try:
        trade_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        month_str = trade_date.strftime('%Y-%m')
    except Exception:
        month_str = date_str[:7]

    # Extra time value from DTE (in seconds of trading time)
    # DTE=0 -> 0 extra seconds, DTE=1 -> 23400 extra seconds, etc.
    extra_secs_from_dte = dte * SECS_PER_TRADING_DAY

    trades = []

    for dp_bar in DECISION_POINTS_BARS:
        if dp_bar >= len(features_full):
            continue

        x_dp = features_full[dp_bar:dp_bar+1].astype(np.float32)
        np.nan_to_num(x_dp, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        pred_vol = float(model.predict(x_dp)[0])

        # Signal filter
        if pred_vol <= threshold_vol:
            continue

        actual_day_bars = len(mid_today)
        exit_bar = min(dp_bar + HORIZON_BARS, actual_day_bars - 1)
        bars_remaining_eod = actual_day_bars - dp_bar

        entry_price = float(mid_today[dp_bar])
        if not np.isfinite(entry_price) or entry_price <= 0:
            continue

        exit_price = float(mid_today[exit_bar])
        if not np.isfinite(exit_price) or exit_price <= 0:
            continue

        underlying_move = abs(exit_price - entry_price)

        # Trailing IV at entry
        trail_iv_pct = float(trail_rvol_today[dp_bar])
        if not np.isfinite(trail_iv_pct) or trail_iv_pct <= 0:
            trail_iv_pct = float(np.nanmean(trail_rvol_today))
            if not np.isfinite(trail_iv_pct) or trail_iv_pct <= 0:
                continue

        iv_sigma = (trail_iv_pct / 100.0) * iv_multiplier

        # Time to expiry at entry:
        #   secs_remaining_today + extra from DTE
        secs_remaining_today = bars_remaining_eod / BARS_PER_SECOND
        T_entry_secs = extra_secs_from_dte + secs_remaining_today
        T_entry = T_entry_secs / SECONDS_PER_YEAR

        # Time to expiry at exit (30 min later):
        #   T_entry minus exactly 30 minutes
        HOLD_SECS = 30 * 60  # 1800 seconds
        T_exit_secs = max(T_entry_secs - HOLD_SECS, 0.0)
        T_exit = T_exit_secs / SECONDS_PER_YEAR

        # ATM strike
        K = round(entry_price / 25.0) * 25.0

        # Straddle prices
        premium_entry = bs_straddle_price(entry_price, K, T_entry, risk_free_rate, iv_sigma)
        premium_exit = bs_straddle_price(exit_price, K, T_exit, risk_free_rate, iv_sigma)

        pnl_points = premium_exit - premium_entry
        pnl_dollars = pnl_points * es_multiplier - cost_per_straddle_rt

        actual_rvol = float(fwd_rvol_today[dp_bar])
        if not np.isfinite(actual_rvol):
            actual_rvol = float('nan')

        if np.std(preds_sub) > 0:
            pred_percentile = float(np.mean(preds_sub <= pred_vol) * 100)
        else:
            pred_percentile = 50.0

        entry_minutes_from_open = dp_bar / BARS_PER_MINUTE
        entry_hour = 9 + int((30 + entry_minutes_from_open) // 60)
        entry_min = int((30 + entry_minutes_from_open) % 60)

        trades.append({
            'date': date_str,
            'month': month_str,
            'dte': dte,
            'entry_bar': dp_bar,
            'exit_bar': exit_bar,
            'entry_time': f'{entry_hour:02d}:{entry_min:02d}',
            'entry_price': entry_price,
            'exit_price': exit_price,
            'underlying_move': underlying_move,
            'strike': K,
            'T_entry_hours': T_entry_secs / 3600,
            'T_exit_hours': T_exit_secs / 3600,
            'iv_pct': iv_sigma * 100,
            'iv_raw_trail_pct': trail_iv_pct,
            'premium_entry': premium_entry,
            'premium_exit': premium_exit,
            'pred_vol': pred_vol,
            'actual_rvol': actual_rvol,
            'pred_percentile': pred_percentile,
            'pnl_points': pnl_points,
            'pnl_dollars': pnl_dollars,
            'pnl_dollars_ex_costs': pnl_points * es_multiplier,
        })

    return trades


# ─── Main backtest ───────────────────────────────────────────────────────────

def run_multi_dte_backtest(
    dtes: List[int] = None,
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
    Walk-forward options backtest across multiple DTEs.

    All DTEs share the same model training (trained once per test day).
    This is fair because the vol prediction model is agnostic to DTE —
    it predicts 30min forward vol, which is the entry signal for all DTEs.

    The key difference is in option pricing:
      - Longer DTE = higher premium (more time value)
      - Longer DTE = less theta loss per 30min hold (theta scales as ~1/T)
      - The profit condition: gamma_pnl > theta_loss + costs

    Break-even analysis:
      For an ATM straddle:
        Theta per day ~ sigma * S / (2 * sqrt(T_years)) * (1/sqrt(2*pi))
        Theta per 30min ~ theta_per_day / (2 * 6.5 hours per day / 0.5 hours)
                        ~ theta_per_day / 26
      So longer DTE has much less theta per 30min — BUT much higher premium.
      The gamma (and thus underlying breakeven move) is roughly the same for ATM options
      regardless of DTE (gamma ~ 1/(S*sigma*sqrt(T))), so longer DTE options have
      LOWER gamma — you need BIGGER moves to profit per dollar of premium.

    Key question: does the lower theta/cost ratio outweigh the lower gamma?
    """
    if dtes is None:
        dtes = [0, 1, 3, 5, 7]

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
    log.info(f"Testing DTEs: {dtes}")
    log.info(f"Strategy: Buy straddle when predicted vol > {threshold_pct:.0f}th percentile")
    log.info(f"Hold: {hold_minutes} min | IV multiplier: {iv_multiplier:.2f}x")

    # ── Step 1: Pre-compute features, vol targets, and trailing IV ──────────
    log.info("\nStep 1: Pre-computing features and vol targets for all days...")
    t0 = time.time()

    IV_LOOKBACK_DAYS = 5
    IV_LOOKBACK_BARS = IV_LOOKBACK_DAYS * BARS_PER_DAY

    multi_day_mids: List[np.ndarray] = []
    day_data = []

    for i, fpath in enumerate(files):
        date_str = fpath.stem.replace('_mbo_features', '')
        try:
            features, mid = load_day(fpath)

            fwd_rvol = compute_forward_rvol_annualized(mid, HORIZON_BARS)

            past_mids = multi_day_mids[-IV_LOOKBACK_DAYS:]
            if past_mids:
                concat_mid = np.concatenate(past_mids + [mid])
            else:
                concat_mid = mid

            actual_bars_today = len(mid)
            trail_full = compute_trailing_rvol_annualized(concat_mid, IV_LOOKBACK_BARS)
            trail_rvol_today = trail_full[-actual_bars_today:]
            if len(trail_rvol_today) < actual_bars_today:
                trail_rvol_today = np.full(actual_bars_today, np.nan)

            multi_day_mids.append(mid)
            if len(multi_day_mids) > IV_LOOKBACK_DAYS + 1:
                multi_day_mids.pop(0)

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
    log.info(f"\nStep 2: Walk-forward prediction + multi-DTE options simulation...")

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

    COST_PER_STRADDLE_RT = 6.20
    ES_MULTIPLIER = 50.0

    # Per-DTE trade lists
    all_trades_by_dte: Dict[int, List[Dict]] = {dte: [] for dte in dtes}
    fold_ics: List[float] = []

    for test_idx in range(min_train_days, len(day_data)):
        test_day = day_data[test_idx]
        date_str = test_day['date']

        if test_day['n'] < 20:
            continue

        # Train once per test day (shared across all DTEs)
        train_days = day_data[:test_idx]
        X_train_parts = [d['X'] for d in train_days]
        y_train_parts = [d['y'] for d in train_days]
        X_train = np.vstack(X_train_parts).astype(np.float32)
        y_train = np.concatenate(y_train_parts).astype(np.float32)

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

        y_test = test_day['y']
        if len(preds_sub) >= 10 and np.std(preds_sub) > 0:
            ic, _ = spearmanr(preds_sub, y_test)
            if np.isfinite(ic):
                fold_ics.append(ic)

        # Signal threshold (from today's distribution of predictions)
        threshold_vol = np.percentile(preds_sub, threshold_pct)

        # Load full features for this test day (needed for per-decision-point predictions)
        fpath = files[test_idx]
        try:
            features_full, _ = load_day(fpath)
        except Exception as e:
            log.warning(f"  Could not reload {date_str}: {e}")
            del X_train, y_train, dtrain, model
            gc.collect()
            continue

        # Simulate for each DTE (model and features are shared)
        for dte in dtes:
            day_trades = simulate_trades_for_dte(
                dte=dte,
                test_day=test_day,
                features_full=features_full,
                model=model,
                preds_sub=preds_sub,
                threshold_vol=threshold_vol,
                risk_free_rate=risk_free_rate,
                iv_multiplier=iv_multiplier,
                cost_per_straddle_rt=COST_PER_STRADDLE_RT,
                es_multiplier=ES_MULTIPLIER,
            )
            all_trades_by_dte[dte].extend(day_trades)

        del features_full
        gc.collect()

        if verbose and (test_idx - min_train_days) % 10 == 0:
            n_tr_0dte = len(all_trades_by_dte[dtes[0]])
            running_pnl_0dte = sum(t['pnl_dollars'] for t in all_trades_by_dte[dtes[0]])
            mean_ic = float(np.mean(fold_ics)) if fold_ics else float('nan')
            log.info(f"  [{test_idx+1}/{len(day_data)}] {date_str}  "
                     f"0dte_trades={n_tr_0dte}  0dte_PnL=${running_pnl_0dte:+,.0f}  "
                     f"fold_IC={fold_ics[-1] if fold_ics else float('nan'):+.3f}  "
                     f"mean_IC={mean_ic:+.3f}")

        del X_train, y_train, dtrain, model
        gc.collect()

    # ── Step 3: Aggregate results per DTE ───────────────────────────────────
    log.info(f"\nStep 3: Aggregating results per DTE...")

    n_test_days = len(day_data) - min_train_days
    ics_arr = np.array(fold_ics)
    mean_fold_ic = float(np.nanmean(ics_arr)) if len(ics_arr) > 0 else float('nan')
    std_fold_ic = float(np.nanstd(ics_arr)) if len(ics_arr) > 0 else float('nan')

    results_by_dte = {}

    for dte in dtes:
        trades_list = all_trades_by_dte[dte]
        if not trades_list:
            log.warning(f"  DTE={dte}: No trades generated.")
            results_by_dte[dte] = None
            continue

        trades_arr = np.array([t['pnl_dollars'] for t in trades_list])
        prems = np.array([t['premium_entry'] for t in trades_list])
        prems_exit = np.array([t['premium_exit'] for t in trades_list])
        moves = np.array([t['underlying_move'] for t in trades_list])
        pred_vols = np.array([t['pred_vol'] for t in trades_list])
        actual_vols = np.array([t['actual_rvol'] for t in trades_list])
        t_entry_hrs = np.array([t['T_entry_hours'] for t in trades_list])
        months = [t['month'] for t in trades_list]

        total_pnl = float(np.sum(trades_arr))
        n_trades = len(trades_arr)
        win_rate = float(np.mean(trades_arr > 0))
        mean_pnl = float(np.mean(trades_arr))
        std_pnl = float(np.std(trades_arr))
        sharpe = (mean_pnl / std_pnl * np.sqrt(252)) if std_pnl > 0 else 0.0
        trades_per_day = n_trades / max(n_test_days, 1)

        unique_months = sorted(set(months))
        pnl_by_month = {}
        for m in unique_months:
            mask = [mo == m for mo in months]
            pnl_by_month[m] = float(np.sum(trades_arr[mask]))

        avg_premium = float(np.nanmean(prems))
        avg_move = float(np.nanmean(moves))
        avg_theta_loss_pts = float(np.nanmean(prems - prems_exit))  # with 0 underlying move
        avg_t_entry_hrs = float(np.nanmean(t_entry_hrs))

        # Traded window IC
        valid_vol = np.isfinite(actual_vols)
        if valid_vol.sum() > 10:
            vol_ic, _ = spearmanr(pred_vols[valid_vol], actual_vols[valid_vol])
        else:
            vol_ic = float('nan')

        best_idx = int(np.argmax(trades_arr))
        worst_idx = int(np.argmin(trades_arr))

        results_by_dte[dte] = {
            'dte': dte,
            'n_trades': n_trades,
            'trades_per_day': round(trades_per_day, 2),
            'total_pnl_dollars': round(total_pnl, 2),
            'mean_pnl_per_trade': round(mean_pnl, 2),
            'std_pnl_per_trade': round(std_pnl, 2),
            'sharpe_annualized': round(sharpe, 3),
            'win_rate': round(win_rate, 4),
            'avg_premium_points': round(avg_premium, 2),
            'avg_premium_dollars': round(avg_premium * 50, 0),
            'avg_underlying_move_points': round(avg_move, 2),
            'avg_theta_loss_30min_points': round(avg_theta_loss_pts, 4),
            'avg_theta_loss_30min_dollars': round(avg_theta_loss_pts * 50, 2),
            'avg_t_entry_hours': round(avg_t_entry_hrs, 1),
            'traded_vol_ic': float(vol_ic) if np.isfinite(vol_ic) else None,
            'pnl_by_month': pnl_by_month,
            'best_trade': trades_list[best_idx],
            'worst_trade': trades_list[worst_idx],
            'trades': trades_list,
        }

    return {
        'config': {
            'n_days': len(day_data),
            'n_test_days': n_test_days,
            'dtes': dtes,
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
        },
        'results_by_dte': results_by_dte,
    }


def print_report(results: Dict) -> None:
    """Print a formatted summary report with comparison table."""
    if not results:
        return

    cfg = results['config']
    mdl = results['model']
    rbd = results['results_by_dte']

    log.info("\n" + "=" * 80)
    log.info("MULTI-DTE STRADDLE BACKTEST REPORT — Vol Prediction Signal")
    log.info("=" * 80)

    log.info(f"\nCONFIGURATION")
    log.info(f"  Days in universe:    {cfg['n_days']}")
    log.info(f"  Test days (OOS):     {cfg['n_test_days']}")
    log.info(f"  Signal threshold:    Top {100-cfg['threshold_pct']:.0f}% of predicted vol")
    log.info(f"  Hold period:         {cfg['hold_minutes']} minutes")
    log.info(f"  IV proxy:            5-day trailing rvol × {cfg['iv_multiplier']:.2f}")
    log.info(f"  Cost per straddle:   ${cfg['cost_per_straddle_rt']:.2f} RT")
    log.info(f"  ES multiplier:       ${cfg['es_multiplier']:.0f}/point")

    log.info(f"\nVOL PREDICTION MODEL (shared across all DTEs)")
    log.info(f"  Mean fold IC:        {mdl['mean_fold_ic']:+.4f}")
    log.info(f"  Std fold IC:         {mdl['std_fold_ic']:.4f}")
    log.info(f"  N folds:             {mdl['n_folds']}")

    # ── Comparison table ──────────────────────────────────────────────────
    log.info(f"\n{'─'*80}")
    log.info("DTE COMPARISON TABLE")
    log.info(f"{'─'*80}")
    header = (f"{'DTE':>5} | {'Trades':>7} | {'Total P&L':>10} | {'Per Trade':>9} | "
              f"{'Win%':>6} | {'Sharpe':>7} | {'Avg Prem':>9} | {'Theta/30m':>10} | {'Avg Move':>9}")
    log.info(header)
    log.info("─" * 80)

    for dte in sorted(rbd.keys()):
        r = rbd[dte]
        if r is None:
            log.info(f"  DTE={dte}: No results")
            continue
        log.info(
            f"{dte:>5} | {r['n_trades']:>7} | "
            f"${r['total_pnl_dollars']:>+9,.0f} | "
            f"${r['mean_pnl_per_trade']:>+8.2f} | "
            f"{r['win_rate']*100:>5.1f}% | "
            f"{r['sharpe_annualized']:>+7.3f} | "
            f"{r['avg_premium_points']:>7.2f}pt | "
            f"{r['avg_theta_loss_30min_points']:>8.4f}pt | "
            f"{r['avg_underlying_move_points']:>7.2f}pt"
        )

    log.info("─" * 80)
    log.info("  (Avg Prem = straddle entry price in ES pts | Theta/30m = time decay per hold)")
    log.info("  (Avg Move = average abs underlying move in 30min window)")
    log.info(f"{'─'*80}")

    # ── Detailed breakdown per DTE ────────────────────────────────────────
    log.info(f"\n{'─'*80}")
    log.info("DETAILED BREAKDOWN BY DTE")
    log.info(f"{'─'*80}")

    for dte in sorted(rbd.keys()):
        r = rbd[dte]
        if r is None:
            continue

        label = f"DTE={dte}" + (" (baseline)" if dte == 0 else "")
        log.info(f"\n  [{label}]  avg T_entry={r['avg_t_entry_hours']:.1f}hr")
        log.info(f"    Trades:        {r['n_trades']} ({r['trades_per_day']:.2f}/day)")
        log.info(f"    Total P&L:     ${r['total_pnl_dollars']:+,.2f}")
        log.info(f"    Mean P&L/trade:${r['mean_pnl_per_trade']:+.2f}  (std=${r['std_pnl_per_trade']:.2f})")
        log.info(f"    Sharpe (ann.): {r['sharpe_annualized']:+.3f}")
        log.info(f"    Win rate:      {r['win_rate']*100:.1f}%")
        log.info(f"    Avg premium:   {r['avg_premium_points']:.2f}pts  (${r['avg_premium_dollars']:.0f})")
        log.info(f"    Theta/30min:   {r['avg_theta_loss_30min_points']:.4f}pts  (${r['avg_theta_loss_30min_dollars']:.2f})")
        log.info(f"    Avg move:      {r['avg_underlying_move_points']:.2f}pts")
        if r['traded_vol_ic'] is not None:
            log.info(f"    Traded IC:     {r['traded_vol_ic']:+.4f}")

        log.info(f"    P&L by month:")
        for month, pnl in sorted(r['pnl_by_month'].items()):
            bar = '#' * max(0, int(pnl / 200)) if pnl > 0 else '-' * max(0, int(-pnl / 200))
            log.info(f"      {month}:  ${pnl:+8.0f}  {bar}")

        bt = r['best_trade']
        wt = r['worst_trade']
        log.info(f"    Best:  {bt['date']} {bt['entry_time']}  "
                 f"move={bt['underlying_move']:.2f}pts  prem={bt['premium_entry']:.2f}  "
                 f"P&L=${bt['pnl_dollars']:+.0f}")
        log.info(f"    Worst: {wt['date']} {wt['entry_time']}  "
                 f"move={wt['underlying_move']:.2f}pts  prem={wt['premium_entry']:.2f}  "
                 f"P&L=${wt['pnl_dollars']:+.0f}")

    # ── Final verdict ─────────────────────────────────────────────────────
    log.info(f"\n{'='*80}")
    log.info("VERDICT")
    log.info(f"{'='*80}")

    best_dte = None
    best_sharpe = -999.0
    for dte in sorted(rbd.keys()):
        r = rbd[dte]
        if r is not None and r['sharpe_annualized'] > best_sharpe:
            best_sharpe = r['sharpe_annualized']
            best_dte = dte

    profitable_dtes = [dte for dte in sorted(rbd.keys())
                       if rbd[dte] is not None and rbd[dte]['total_pnl_dollars'] > 0]

    if profitable_dtes:
        log.info(f"  PROFITABLE DTEs: {profitable_dtes}")
        log.info(f"  Best DTE by Sharpe: DTE={best_dte} (Sharpe={best_sharpe:+.3f})")
        if best_sharpe > 0.5:
            log.info("  STRONG EDGE: Longer DTE significantly improves profitability.")
            log.info("  RECOMMENDATION: Trade with DTE={best_dte} straddles.")
        else:
            log.info("  MARGINAL EDGE: Positive but low Sharpe. Regime-dependent.")
    else:
        log.info("  NO DTE IS PROFITABLE after costs.")
        log.info("  ROOT CAUSE:")
        log.info("  - Theta savings from longer DTE are real but premium is proportionally larger.")
        log.info("  - For an ATM straddle, premium ~ sigma*S*sqrt(T).")
        log.info("  - Theta per 30min ~ premium * (30min / (T * 2)) for large T.")
        log.info("  - So theta/premium ratio is constant regardless of DTE!")
        log.info("  - What changes: gamma (lower for longer DTE) — need bigger moves to profit.")
        log.info("  - IMPLICATION: The fundamental problem is moves < breakeven, not DTE.")
        log.info("  - NEXT STEP: Need realized vol to systematically exceed IV,")
        log.info("    not just predict high vol — that only works if our IV proxy underestimates.")

    log.info("=" * 80)


def format_discord_summary(results: Dict) -> str:
    """Format a compact summary for Discord."""
    cfg = results['config']
    mdl = results['model']
    rbd = results['results_by_dte']

    lines = [
        "**Multi-DTE Straddle Backtest Results**",
        f"Signal: top {100-cfg['threshold_pct']:.0f}% predicted vol | Model IC={mdl['mean_fold_ic']:+.4f}",
        f"Hold: {cfg['hold_minutes']}min | IV proxy: 5d trail rvol x{cfg['iv_multiplier']:.1f} | Cost: ${cfg['cost_per_straddle_rt']:.2f}/straddle",
        "",
        "```",
        f"{'DTE':>4} | {'Trades':>6} | {'Total P&L':>10} | {'Mean/trade':>10} | {'Win%':>5} | {'Sharpe':>7} | {'Prem(pt)':>8} | {'Theta(pt)':>9}",
        "─" * 75,
    ]

    for dte in sorted(rbd.keys()):
        r = rbd[dte]
        if r is None:
            lines.append(f"  DTE={dte}: No results")
            continue
        verdict = " <-- BEST" if r['sharpe_annualized'] == max(
            rbd[d]['sharpe_annualized'] for d in sorted(rbd.keys()) if rbd[d] is not None
        ) else ""
        lines.append(
            f"{dte:>4} | {r['n_trades']:>6} | "
            f"${r['total_pnl_dollars']:>+9,.0f} | "
            f"${r['mean_pnl_per_trade']:>+9.2f} | "
            f"{r['win_rate']*100:>4.1f}% | "
            f"{r['sharpe_annualized']:>+7.3f} | "
            f"{r['avg_premium_points']:>7.2f}  | "
            f"{r['avg_theta_loss_30min_points']:>8.4f}"
            f"{verdict}"
        )

    lines.append("```")

    # Verdict
    profitable_dtes = [dte for dte in sorted(rbd.keys())
                       if rbd[dte] is not None and rbd[dte]['total_pnl_dollars'] > 0]
    if profitable_dtes:
        best_dte = max(profitable_dtes, key=lambda d: rbd[d]['sharpe_annualized'])
        lines.append(f"\nProfitable DTEs: {profitable_dtes}")
        lines.append(f"Best DTE: **{best_dte}** (Sharpe={rbd[best_dte]['sharpe_annualized']:+.3f}, total=${rbd[best_dte]['total_pnl_dollars']:+,.0f})")
    else:
        lines.append("\nNo DTE is profitable after costs. Vol signal does not overcome theta + costs.")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Multi-DTE Straddle Backtest using Vol Prediction Signal'
    )
    parser.add_argument('--n-days', type=int, default=100,
                        help='Max trading days to use (default: 100)')
    parser.add_argument('--threshold-pct', type=float, default=80.0,
                        help='Only trade when predicted vol > this percentile (default: 80)')
    parser.add_argument('--hold-minutes', type=int, default=30,
                        help='Hold period in minutes (default: 30)')
    parser.add_argument('--iv-multiplier', type=float, default=1.0,
                        help='Scale IV proxy by this factor')
    parser.add_argument('--min-train-days', type=int, default=15,
                        help='Walk-forward minimum training days (default: 15)')
    parser.add_argument('--workers', type=int, default=4,
                        help='LightGBM parallel jobs (default: 4)')
    parser.add_argument('--dtes', type=int, nargs='+', default=[0, 1, 3, 5, 7],
                        help='DTEs to test (default: 0 1 3 5 7)')
    parser.add_argument('--output-dir', type=str, default=str(RESULTS_DIR),
                        help='Directory for results JSON')
    parser.add_argument('--no-save', action='store_true',
                        help='Skip saving results to JSON')
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 80)
    log.info("MULTI-DTE STRADDLE BACKTEST — Vol Prediction Signal")
    log.info("=" * 80)
    log.info(f"  MBO data dir:  {MBO_DIR}")
    log.info(f"  Results dir:   {RESULTS_DIR}")
    log.info(f"  DTEs to test:  {args.dtes}")

    # Send initial Discord notification
    try:
        import subprocess
        # We'll send updates through print statements captured by the calling process
    except Exception:
        pass

    t_start = time.time()

    results = run_multi_dte_backtest(
        dtes=args.dtes,
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

    # Save results
    if not args.no_save:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fname = f'vol_options_backtest_multichte_thr{int(args.threshold_pct)}_iv{args.iv_multiplier:.1f}_{timestamp}.json'
        out_path = out_dir / fname

        # Save summary (no full trade lists)
        save_results = {
            'config': results['config'],
            'model': results['model'],
            'results_by_dte': {
                str(dte): {k: v for k, v in r.items() if k != 'trades'}
                for dte, r in results['results_by_dte'].items()
                if r is not None
            },
        }
        with open(out_path, 'w') as f:
            json.dump(save_results, f, indent=2, default=str)
        log.info(f"Results saved: {out_path}")

        # Save per-DTE trade CSVs
        import csv
        for dte, r in results['results_by_dte'].items():
            if r is None:
                continue
            trades = r.get('trades', [])
            if not trades:
                continue
            csv_fname = fname.replace('.json', f'_dte{dte}_trades.csv')
            csv_path = out_dir / csv_fname
            keys = ['date', 'entry_time', 'dte', 'entry_price', 'exit_price',
                    'underlying_move', 'T_entry_hours', 'iv_pct', 'premium_entry',
                    'premium_exit', 'pred_vol', 'actual_rvol', 'pred_percentile',
                    'pnl_points', 'pnl_dollars']
            with open(csv_path, 'w', newline='') as cf:
                writer = csv.DictWriter(cf, fieldnames=keys, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(trades)
            log.info(f"DTE={dte} trades CSV: {csv_path}")

    # Print Discord summary
    discord_msg = format_discord_summary(results)
    print("\n" + "=" * 80)
    print("DISCORD SUMMARY:")
    print(discord_msg)
    print("=" * 80)


if __name__ == '__main__':
    main()
