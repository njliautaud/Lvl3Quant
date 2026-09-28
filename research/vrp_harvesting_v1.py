#!/usr/bin/env python3
"""
Volatility Risk Premium (VRP) Harvesting Strategy v1
=====================================================
Systematic short-vol strategy on SPY via strangle selling.

Core thesis: Implied volatility (VIX) systematically exceeds realized
volatility. The difference (VRP = IV - RV) is a well-documented risk
premium that can be harvested by selling options.

Strategy logic:
  - Measure VRP = VIX / 100 - 20d realized vol of SPY
  - When VRP > threshold → sell 30-DTE, 10-delta strangle on SPY
  - When VRP < threshold → stay flat (premium too thin)
  - Position sizing: fixed notional per trade
  - Risk management: close at 2x premium loss (stop-loss)
  - Roll monthly: close at 7 DTE or 50% profit

Anti-lookahead:
  - All signals use T-1 data (lag=1)
  - Entry on T+1 open after signal
  - Walk-forward validation with expanding train, rolling OOS

Adversarial gates (HC #718, HC #428):
  G1: Permutation test (p < 0.05)
  G2: Regime-agnostic (|Sharpe_green - Sharpe_red| / max < 0.50)
  G3: Sub-period stability (no sub-period Sharpe < 0)
  G4: Outlier sensitivity (Sharpe w/o top 5 trades > 50% of full)

Cost assumptions:
  - SPY option bid-ask spread: ~3-5% of premium (liquid)
  - Commission: $0.65/contract
  - Slippage: 1% of premium
  Total cost: ~5% of premium collected

Author: Claude (Head of Quant)
Date: 2026-07-24
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from datetime import datetime, timedelta
import json
import warnings
import sys
import time
import traceback

warnings.filterwarnings('ignore')
np.random.seed(42)

# ============================================================
# CONSTANTS
# ============================================================
RISK_FREE_RATE = 0.04
TRADING_DAYS = 252
COST_PCT_OF_PREMIUM = 0.05  # 5% of premium for round-trip costs (SPY is very liquid)
STOP_LOSS_MULT = 2.0  # Close if loss > 2x premium
PROFIT_TAKE_PCT = 0.50  # Close at 50% of max profit
DTE_ENTRY = 30  # Days to expiration at entry
DTE_ROLL = 7  # Roll when DTE falls to this
DELTA_TARGET = 0.10  # ~10-delta strangle (5% OTM each side for typical vol)
VRP_THRESHOLD_DEFAULT = 0.02  # Minimum VRP to sell (2% annualized)
NOTIONAL_PER_TRADE = 10000  # $10k notional per strangle

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def download_with_retry(tickers, start, end, retries=3):
    """Download data with retry logic."""
    for attempt in range(retries):
        try:
            data = yf.download(tickers, start=start, end=end, progress=False)
            if data is not None and len(data) > 50:
                return data
        except Exception as e:
            print(f"  Download attempt {attempt+1} failed: {e}")
            time.sleep(2)
    raise RuntimeError(f"Failed to download {tickers} after {retries} attempts")


def bs_call_price(S, K, T, sigma, r=RISK_FREE_RATE):
    """Black-Scholes call price."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(S - K, 0) if T <= 0 else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(-d2)


def bs_put_price(S, K, T, sigma, r=RISK_FREE_RATE):
    """Black-Scholes put price."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(K - S, 0) if T <= 0 else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_delta_call(S, K, T, sigma, r=RISK_FREE_RATE):
    """Black-Scholes call delta."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)


def bs_delta_put(S, K, T, sigma, r=RISK_FREE_RATE):
    """Black-Scholes put delta."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return -1.0 if S < K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) - 1.0


def find_strike_for_delta(S, T, sigma, target_delta, is_call=True, r=RISK_FREE_RATE):
    """Binary search for strike that gives target delta."""
    if T <= 0 or sigma <= 0:
        return S

    lo, hi = S * 0.70, S * 1.30
    for _ in range(50):
        mid = (lo + hi) / 2
        if is_call:
            d = bs_delta_call(S, mid, T, sigma, r)
            if d > target_delta:
                lo = mid
            else:
                hi = mid
        else:
            d = abs(bs_delta_put(S, mid, T, sigma, r))
            if d > target_delta:
                hi = mid
            else:
                lo = mid
    return round((lo + hi) / 2, 2)


def calc_metrics(returns, rf_annual=0.04, periods_per_year=252):
    """Calculate risk-adjusted metrics."""
    if len(returns) < 10 or returns.std() == 0:
        return {'sharpe': 0, 'sortino': 0, 'pf': 0, 'wr': 0,
                'cagr': 0, 'max_dd': 0, 'total_ret': 0, 'n_periods': 0,
                'annual_vol': 0, 'calmar': 0}

    rf_per = rf_annual / periods_per_year
    excess = returns - rf_per

    ann_ret = returns.mean() * periods_per_year
    ann_vol = returns.std() * np.sqrt(periods_per_year)
    sharpe = excess.mean() / returns.std() * np.sqrt(periods_per_year) if returns.std() > 0 else 0

    downside = returns[returns < 0].std()
    sortino = excess.mean() / downside * np.sqrt(periods_per_year) if (downside is not None and downside > 0) else 0

    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    pf = gains / losses if losses > 0 else float('inf')

    wr = (returns > 0).mean()

    cum = (1 + returns).cumprod()
    n_years = len(returns) / periods_per_year
    cagr = (cum.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 and cum.iloc[-1] > 0 else 0

    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    total_ret = cum.iloc[-1] - 1

    return {
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'pf': round(pf, 3),
        'wr': round(wr, 4),
        'cagr': round(cagr, 4),
        'max_dd': round(max_dd, 4),
        'total_ret': round(total_ret, 4),
        'annual_vol': round(ann_vol, 4),
        'calmar': round(calmar, 3),
        'n_periods': len(returns)
    }


def regime_split(returns, spy_returns):
    """Split returns by SPY regime."""
    green = returns[spy_returns > 0.002]
    red = returns[spy_returns < -0.002]
    flat = returns[(spy_returns >= -0.002) & (spy_returns <= 0.002)]
    return {
        'green': calc_metrics(green) if len(green) > 10 else None,
        'red': calc_metrics(red) if len(red) > 10 else None,
        'flat': calc_metrics(flat) if len(flat) > 10 else None,
    }


def regime_agnostic_check(regime_results):
    """HC #428 R1: |Sharpe_green - Sharpe_red| / max(...) <= 0.50"""
    g = regime_results.get('green')
    r = regime_results.get('red')
    if g is None or r is None:
        return True, 0.0
    sg, sr = g['sharpe'], r['sharpe']
    denom = max(abs(sg), abs(sr))
    if denom == 0:
        return True, 0.0
    ratio = abs(sg - sr) / denom
    return ratio <= 0.50, round(ratio, 3)


def permutation_test(returns, n_perms=200):
    """
    Permutation test: randomly shuffle trade-level returns to break
    the signal-timing link. Tests whether VRP timing adds value vs random entry.
    """
    actual_sharpe = calc_metrics(returns)['sharpe']
    null_sharpes = []
    ret_vals = returns.values.copy()
    n = len(ret_vals)
    for _ in range(n_perms):
        np.random.shuffle(ret_vals)
        shuffled = pd.Series(ret_vals, index=returns.index)
        null_sharpes.append(calc_metrics(shuffled)['sharpe'])
    null_sharpes = np.array(null_sharpes)
    p_value = (null_sharpes >= actual_sharpe).mean()
    return actual_sharpe, p_value, null_sharpes.mean(), null_sharpes.std()


def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_metrics(metrics, prefix="  "):
    print(f"{prefix}Sharpe: {metrics['sharpe']:.3f}  |  Sortino: {metrics['sortino']:.3f}  |  "
          f"PF: {metrics['pf']:.3f}  |  WR: {metrics['wr']:.1%}")
    print(f"{prefix}CAGR: {metrics['cagr']:.2%}  |  MaxDD: {metrics['max_dd']:.2%}  |  "
          f"Calmar: {metrics['calmar']:.3f}  |  Vol: {metrics['annual_vol']:.2%}")


# ============================================================
# DATA LOADING
# ============================================================

def load_data():
    """Load SPY and VIX data."""
    print_section("DATA LOADING")

    # SPY prices
    spy_raw = download_with_retry('SPY', '2006-01-01', '2026-07-24')
    if isinstance(spy_raw.columns, pd.MultiIndex):
        spy_raw.columns = spy_raw.columns.get_level_values(0)
    spy = spy_raw[['Close', 'High', 'Low']].dropna()
    spy.columns = ['close', 'high', 'low']
    spy.index = pd.to_datetime(spy.index)
    if spy.index.tz is not None:
        spy.index = spy.index.tz_localize(None)

    # VIX (implied vol proxy)
    vix_raw = download_with_retry('^VIX', '2006-01-01', '2026-07-24')
    if isinstance(vix_raw.columns, pd.MultiIndex):
        vix_raw.columns = vix_raw.columns.get_level_values(0)
    vix = vix_raw[['Close']].dropna()
    vix.columns = ['vix']
    vix.index = pd.to_datetime(vix.index)
    if vix.index.tz is not None:
        vix.index = vix.index.tz_localize(None)

    # Merge
    df = spy.join(vix, how='inner')
    df = df.dropna()

    # Compute realized vol (20-day)
    df['ret'] = df['close'].pct_change()
    df['rv_20d'] = df['ret'].rolling(20).std() * np.sqrt(TRADING_DAYS)  # annualized

    # VRP = IV - RV (in annualized decimal terms)
    df['iv'] = df['vix'] / 100.0  # VIX is in percentage points
    df['vrp'] = df['iv'] - df['rv_20d']

    # Also compute 5d, 10d, 60d RV for comparison
    df['rv_5d'] = df['ret'].rolling(5).std() * np.sqrt(TRADING_DAYS)
    df['rv_10d'] = df['ret'].rolling(10).std() * np.sqrt(TRADING_DAYS)
    df['rv_60d'] = df['ret'].rolling(60).std() * np.sqrt(TRADING_DAYS)

    # VRP z-score (rolling 60d mean/std of VRP)
    df['vrp_mean_60d'] = df['vrp'].rolling(60).mean()
    df['vrp_std_60d'] = df['vrp'].rolling(60).std()
    df['vrp_z'] = (df['vrp'] - df['vrp_mean_60d']) / df['vrp_std_60d'].replace(0, np.nan)

    df = df.dropna()

    print(f"  SPY data: {df.index[0].date()} to {df.index[-1].date()}, {len(df)} trading days")
    print(f"  VIX stats: mean={df['vix'].mean():.1f}, median={df['vix'].median():.1f}")
    print(f"  VRP stats: mean={df['vrp'].mean():.4f}, median={df['vrp'].median():.4f}, "
          f"std={df['vrp'].std():.4f}")
    print(f"  VRP > 0 (IV > RV): {(df['vrp'] > 0).mean():.1%} of days")
    print(f"  VRP > 2%: {(df['vrp'] > 0.02).mean():.1%} of days")
    print(f"  VRP > 5%: {(df['vrp'] > 0.05).mean():.1%} of days")

    return df


# ============================================================
# STRANGLE SIMULATION ENGINE
# ============================================================

def simulate_strangle_strategy(df, vrp_threshold=0.02, delta_target=0.10,
                                dte=30, roll_dte=7, stop_mult=2.0,
                                profit_take=0.50, cost_pct=0.05,
                                lag=1, use_vrp_sizing=False):
    """
    Simulate selling monthly strangles on SPY filtered by VRP.

    Logic:
      - Each trading day, check if VRP(T-lag) > threshold
      - If yes and no position open → sell strangle (put + call at target delta)
      - Mark to market daily using BS repricing
      - Close if: DTE <= roll_dte, P&L >= profit_take * premium, or loss >= stop_mult * premium
      - After close, check for new entry next day

    Returns a daily P&L series.
    """
    dates = df.index.tolist()
    n = len(dates)

    daily_pnl = pd.Series(0.0, index=df.index)
    trades = []

    # State
    in_position = False
    entry_date = None
    entry_spy = 0
    call_strike = 0
    put_strike = 0
    call_premium = 0
    put_premium = 0
    total_premium = 0
    entry_iv = 0
    days_held = 0
    remaining_dte = 0

    for i in range(max(lag, 1), n):
        today = dates[i]
        S = df.loc[today, 'close']
        iv_today = df.loc[today, 'iv']

        if in_position:
            days_held += 1
            remaining_dte -= 1
            if remaining_dte < 0:
                remaining_dte = 0

            T_remaining = remaining_dte / TRADING_DAYS

            # Reprice options with current spot, IV, and remaining time
            # Use current IV for repricing (vol moves affect P&L)
            if T_remaining > 0.001:
                call_val = bs_call_price(S, call_strike, T_remaining, iv_today)
                put_val = bs_put_price(S, put_strike, T_remaining, iv_today)
            else:
                call_val = max(S - call_strike, 0)
                put_val = max(put_strike - S, 0)

            current_value = call_val + put_val
            position_pnl = total_premium - current_value  # Short: profit when value decreases

            # Per-unit daily P&L for return calculation
            # Premium = max risk for a strangle is theoretically unlimited,
            # but we use the margin requirement as denominator
            # Approximate margin: max(put_strike, call_strike) * 0.20 (typical)
            margin = max(put_strike, call_strike) * 0.20
            if margin > 0:
                daily_ret = position_pnl / margin  # Return on margin
            else:
                daily_ret = 0

            # Check close conditions
            close_reason = None
            if remaining_dte <= roll_dte:
                close_reason = 'roll'
            elif position_pnl >= profit_take * total_premium:
                close_reason = 'profit_take'
            elif position_pnl <= -stop_mult * total_premium:
                close_reason = 'stop_loss'

            if close_reason:
                # Close: P&L = premium collected - cost to close - entry costs - exit costs
                close_cost = current_value * cost_pct  # Cost to buy back
                entry_cost = total_premium * cost_pct  # Cost at entry
                net_pnl = position_pnl - entry_cost - close_cost
                net_ret = net_pnl / margin if margin > 0 else 0

                trades.append({
                    'entry_date': entry_date,
                    'exit_date': today,
                    'days_held': days_held,
                    'entry_spy': entry_spy,
                    'exit_spy': S,
                    'call_strike': call_strike,
                    'put_strike': put_strike,
                    'premium': total_premium,
                    'close_value': current_value,
                    'gross_pnl': position_pnl,
                    'net_pnl': net_pnl,
                    'net_ret': net_ret,
                    'close_reason': close_reason,
                    'entry_vrp': df.loc[entry_date, 'vrp'] if entry_date in df.index else 0,
                    'entry_iv': entry_iv,
                    'margin': margin,
                })

                daily_pnl.loc[today] = net_ret
                in_position = False
                continue

            # Still holding — record mark-to-market change
            if i > 0:
                prev_date = dates[i - 1]
                S_prev = df.loc[prev_date, 'close']
                iv_prev = df.loc[prev_date, 'iv']
                T_prev = (remaining_dte + 1) / TRADING_DAYS

                if T_prev > 0.001:
                    prev_call_val = bs_call_price(S_prev, call_strike, T_prev, iv_prev)
                    prev_put_val = bs_put_price(S_prev, put_strike, T_prev, iv_prev)
                else:
                    prev_call_val = max(S_prev - call_strike, 0)
                    prev_put_val = max(put_strike - S_prev, 0)

                prev_value = prev_call_val + prev_put_val
                daily_change = prev_value - current_value  # Short: profit when value drops
                daily_pnl.loc[today] = daily_change / margin if margin > 0 else 0

        else:
            # Check for entry signal (using T-lag data)
            signal_idx = i - lag
            if signal_idx < 0:
                continue
            signal_date = dates[signal_idx]
            vrp_signal = df.loc[signal_date, 'vrp']

            if vrp_signal > vrp_threshold:
                # Enter strangle
                entry_date = today
                entry_spy = S
                entry_iv = iv_today
                remaining_dte = dte
                days_held = 0

                T = dte / TRADING_DAYS

                # Find strikes for target delta
                call_strike = find_strike_for_delta(S, T, iv_today, delta_target, is_call=True)
                put_strike = find_strike_for_delta(S, T, iv_today, delta_target, is_call=False)

                # Price the strangle
                call_premium = bs_call_price(S, call_strike, T, iv_today)
                put_premium = bs_put_price(S, put_strike, T, iv_today)
                total_premium = call_premium + put_premium

                # Size: VRP-weighted if enabled
                if use_vrp_sizing:
                    size_mult = min(vrp_signal / 0.05, 2.0)  # Scale up to 2x for very high VRP
                    total_premium *= size_mult
                    call_premium *= size_mult
                    put_premium *= size_mult

                in_position = True

    # Build daily return series from trades
    # For walk-forward: use the trade-level returns, distributed to exit dates
    trade_returns = pd.Series(0.0, index=df.index)
    for t in trades:
        trade_returns.loc[t['exit_date']] = t['net_ret']

    return daily_pnl, trade_returns, trades


# ============================================================
# WALK-FORWARD VALIDATION
# ============================================================

def walk_forward_backtest(df, vrp_threshold=0.02, train_months=24, oos_months=3):
    """
    Walk-forward validation: train VRP threshold on in-sample,
    test on out-of-sample period. Rolling window.

    In-sample optimization: find best VRP threshold from [0.01, 0.02, 0.03, 0.04, 0.05]
    Out-of-sample: use that threshold and record performance.
    """
    dates = df.index
    start = dates[0]
    end = dates[-1]

    oos_returns_all = []
    oos_trades_all = []
    wf_results = []

    thresholds_to_try = [0.01, 0.015, 0.02, 0.03, 0.04, 0.05]

    current = start + pd.DateOffset(months=train_months)

    while current + pd.DateOffset(months=oos_months) <= end:
        train_start = current - pd.DateOffset(months=train_months)
        train_end = current
        oos_start = current
        oos_end = current + pd.DateOffset(months=oos_months)

        df_train = df[(df.index >= train_start) & (df.index < train_end)]
        df_oos = df[(df.index >= oos_start) & (df.index < oos_end)]

        if len(df_train) < 200 or len(df_oos) < 20:
            current += pd.DateOffset(months=oos_months)
            continue

        # In-sample: find best threshold
        best_sharpe = -999
        best_thresh = 0.02
        for thresh in thresholds_to_try:
            _, trade_rets, trades = simulate_strangle_strategy(
                df_train, vrp_threshold=thresh, lag=1
            )
            if len(trades) < 5:
                continue
            m = calc_metrics(trade_rets[trade_rets != 0])
            if m['sharpe'] > best_sharpe:
                best_sharpe = m['sharpe']
                best_thresh = thresh

        # Out-of-sample: apply best threshold
        _, oos_trade_rets, oos_trades = simulate_strangle_strategy(
            df_oos, vrp_threshold=best_thresh, lag=1
        )

        oos_nonzero = oos_trade_rets[oos_trade_rets != 0]
        if len(oos_nonzero) > 0:
            oos_returns_all.append(oos_nonzero)
            oos_trades_all.extend(oos_trades)

        m_oos = calc_metrics(oos_nonzero) if len(oos_nonzero) > 2 else {'sharpe': 0}
        wf_results.append({
            'oos_start': oos_start.strftime('%Y-%m-%d'),
            'oos_end': oos_end.strftime('%Y-%m-%d'),
            'best_thresh': best_thresh,
            'is_sharpe': round(best_sharpe, 3),
            'oos_sharpe': m_oos['sharpe'],
            'n_trades': len([t for t in oos_trades]),
        })

        current += pd.DateOffset(months=oos_months)

    if oos_returns_all:
        combined = pd.concat(oos_returns_all).sort_index()
    else:
        combined = pd.Series(dtype=float)

    return combined, oos_trades_all, wf_results


# ============================================================
# ADVERSARIAL GATES
# ============================================================

def run_adversarial_gates(returns, spy_returns, trades):
    """Run all 4 adversarial gates."""
    gates = {}

    # G1: Permutation test
    print("\n  Gate 1: Permutation Test (200 permutations)")
    actual_s, p_val, null_mean, null_std = permutation_test(returns, n_perms=200)
    g1_pass = p_val < 0.05
    gates['G1_permutation'] = {
        'pass': g1_pass,
        'actual_sharpe': actual_s,
        'p_value': round(p_val, 4),
        'null_mean': round(null_mean, 3),
        'null_std': round(null_std, 3),
    }
    print(f"    Actual Sharpe: {actual_s:.3f} | Null: {null_mean:.3f} +/- {null_std:.3f} | "
          f"p={p_val:.4f} | {'PASS' if g1_pass else 'FAIL'}")

    # G2: Regime-agnostic (HC #428)
    print("\n  Gate 2: Regime-Agnostic Test")
    regimes = regime_split(returns, spy_returns)
    agnostic_pass, regime_ratio = regime_agnostic_check(regimes)
    gates['G2_regime'] = {
        'pass': agnostic_pass,
        'ratio': regime_ratio,
        'green_sharpe': regimes['green']['sharpe'] if regimes['green'] else None,
        'red_sharpe': regimes['red']['sharpe'] if regimes['red'] else None,
        'flat_sharpe': regimes['flat']['sharpe'] if regimes['flat'] else None,
    }
    for rname, rm in regimes.items():
        if rm:
            print(f"    {rname.upper():5s}: Sharpe={rm['sharpe']:.3f}  PF={rm['pf']:.3f}  WR={rm['wr']:.1%}  N={rm['n_periods']}")
    print(f"    Regime ratio: {regime_ratio:.3f} | {'PASS' if agnostic_pass else 'FAIL'}")

    # G3: Sub-period stability (split into 4 quarters)
    print("\n  Gate 3: Sub-Period Stability (4 periods)")
    n = len(returns)
    chunk = n // 4
    sub_sharpes = []
    all_positive = True
    for q in range(4):
        start = q * chunk
        end = start + chunk if q < 3 else n
        sub = returns.iloc[start:end]
        m = calc_metrics(sub)
        sub_sharpes.append(m['sharpe'])
        if m['sharpe'] < 0:
            all_positive = False
        print(f"    Q{q+1}: Sharpe={m['sharpe']:.3f}  PF={m['pf']:.3f}  WR={m['wr']:.1%}  N={m['n_periods']}")

    gates['G3_subperiod'] = {
        'pass': all_positive,
        'sub_sharpes': [round(s, 3) for s in sub_sharpes],
        'min_sharpe': round(min(sub_sharpes), 3),
    }
    print(f"    All sub-periods positive: {'PASS' if all_positive else 'FAIL'}")

    # G4: Outlier sensitivity (remove top 5 winning trades)
    print("\n  Gate 4: Outlier Sensitivity (remove top 5 trades)")
    if len(trades) >= 10:
        trade_pnls = sorted([t['net_ret'] for t in trades], reverse=True)
        full_sharpe = calc_metrics(returns)['sharpe']

        # Remove top 5 trades
        top5 = set()
        sorted_trades = sorted(trades, key=lambda x: x['net_ret'], reverse=True)
        for t in sorted_trades[:5]:
            top5.add(t['exit_date'])

        filtered_returns = returns[~returns.index.isin(top5)]
        filtered_m = calc_metrics(filtered_returns)

        ratio = filtered_m['sharpe'] / full_sharpe if full_sharpe > 0 else 0
        g4_pass = ratio >= 0.50

        gates['G4_outlier'] = {
            'pass': g4_pass,
            'full_sharpe': full_sharpe,
            'filtered_sharpe': filtered_m['sharpe'],
            'ratio': round(ratio, 3),
        }
        print(f"    Full Sharpe: {full_sharpe:.3f} | Without top 5: {filtered_m['sharpe']:.3f} | "
              f"Ratio: {ratio:.3f} | {'PASS' if g4_pass else 'FAIL'}")
    else:
        gates['G4_outlier'] = {'pass': False, 'reason': 'insufficient trades'}
        print(f"    Insufficient trades ({len(trades)}) | FAIL")

    # Summary
    n_pass = sum(1 for g in gates.values() if g.get('pass', False))
    print(f"\n  GATES PASSED: {n_pass}/4")

    return gates, n_pass


# ============================================================
# VARIANT STRATEGIES
# ============================================================

def run_variants(df):
    """Run multiple VRP harvesting variants and compare."""
    print_section("STRATEGY VARIANTS")

    variants = {
        'baseline': {'vrp_threshold': 0.02, 'delta_target': 0.10, 'stop_mult': 2.0, 'profit_take': 0.50},
        'tight_filter': {'vrp_threshold': 0.04, 'delta_target': 0.10, 'stop_mult': 2.0, 'profit_take': 0.50},
        'wide_strangle': {'vrp_threshold': 0.02, 'delta_target': 0.05, 'stop_mult': 2.0, 'profit_take': 0.50},
        'narrow_strangle': {'vrp_threshold': 0.02, 'delta_target': 0.15, 'stop_mult': 2.0, 'profit_take': 0.50},
        'tight_stop': {'vrp_threshold': 0.02, 'delta_target': 0.10, 'stop_mult': 1.5, 'profit_take': 0.50},
        'vrp_sized': {'vrp_threshold': 0.02, 'delta_target': 0.10, 'stop_mult': 2.0, 'profit_take': 0.50, 'use_vrp_sizing': True},
    }

    results = {}
    for name, params in variants.items():
        _, trade_rets, trades = simulate_strangle_strategy(df, lag=1, **params)
        nonzero = trade_rets[trade_rets != 0]
        if len(nonzero) < 10:
            print(f"  {name}: insufficient trades ({len(trades)})")
            continue

        m = calc_metrics(nonzero)
        results[name] = {
            'metrics': m,
            'n_trades': len(trades),
            'avg_days_held': np.mean([t['days_held'] for t in trades]),
            'win_rate': np.mean([1 for t in trades if t['net_pnl'] > 0]) / len(trades) if trades else 0,
            'avg_pnl': np.mean([t['net_ret'] for t in trades]),
            'trades': trades,
            'returns': nonzero,
        }

        print(f"\n  {name}:")
        print(f"    Trades: {len(trades)} | Avg hold: {results[name]['avg_days_held']:.1f}d | "
              f"WR: {results[name]['win_rate']:.1%}")
        print_metrics(m, prefix="    ")

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("  VRP HARVESTING STRATEGY v1 — SPY STRANGLE SELLING")
    print("  Volatility Risk Premium Systematic Harvesting")
    print("=" * 70)

    # 1. Load data
    df = load_data()

    # 2. Full-sample backtest (all variants)
    variant_results = run_variants(df)

    if not variant_results:
        print("\n*** No variant produced enough trades. Exiting. ***")
        return

    # 3. Pick best variant
    best_name = max(variant_results, key=lambda k: variant_results[k]['metrics']['sharpe'])
    best = variant_results[best_name]
    print_section(f"BEST VARIANT: {best_name}")
    print_metrics(best['metrics'])
    print(f"  Trades: {best['n_trades']} | Avg hold: {best['avg_days_held']:.1f}d")

    # 4. Walk-forward validation on best variant config
    print_section("WALK-FORWARD VALIDATION (24m train, 3m OOS)")

    # Get best threshold from variant
    best_params = {
        'baseline': 0.02, 'tight_filter': 0.04, 'wide_strangle': 0.02,
        'narrow_strangle': 0.02, 'tight_stop': 0.02, 'vrp_sized': 0.02,
    }

    wf_returns, wf_trades, wf_results = walk_forward_backtest(
        df, vrp_threshold=best_params.get(best_name, 0.02),
        train_months=24, oos_months=3
    )

    if len(wf_returns) < 10:
        print("  Walk-forward produced insufficient OOS returns. Using full-sample.")
        final_returns = best['returns']
        final_trades = best['trades']
        wf_used = False
    else:
        final_returns = wf_returns
        final_trades = wf_trades
        wf_used = True

        wf_metrics = calc_metrics(wf_returns)
        print(f"\n  Walk-Forward OOS Results:")
        print_metrics(wf_metrics)
        print(f"  OOS trades: {len(wf_trades)}")

        # Print WF windows
        print(f"\n  Walk-forward windows: {len(wf_results)}")
        for wf in wf_results:
            print(f"    {wf['oos_start']} to {wf['oos_end']}: thresh={wf['best_thresh']:.3f}, "
                  f"IS Sharpe={wf['is_sharpe']:.3f}, OOS Sharpe={wf['oos_sharpe']:.3f}, "
                  f"trades={wf['n_trades']}")

        oos_sharpes = [w['oos_sharpe'] for w in wf_results if w['n_trades'] > 0]
        if oos_sharpes:
            print(f"\n  OOS Sharpe: mean={np.mean(oos_sharpes):.3f}, "
                  f"median={np.median(oos_sharpes):.3f}, "
                  f"min={np.min(oos_sharpes):.3f}, max={np.max(oos_sharpes):.3f}")
            print(f"  Positive OOS windows: {sum(1 for s in oos_sharpes if s > 0)}/{len(oos_sharpes)}")

    # 5. Adversarial gates
    print_section("ADVERSARIAL GATES")
    spy_daily = df['ret'].reindex(final_returns.index).fillna(0)
    gates, n_pass = run_adversarial_gates(final_returns, spy_daily, final_trades)

    # 6. Trade analysis
    print_section("TRADE ANALYSIS")
    if final_trades:
        trade_df = pd.DataFrame(final_trades)
        print(f"  Total trades: {len(trade_df)}")
        print(f"  Win rate: {(trade_df['net_pnl'] > 0).mean():.1%}")
        print(f"  Avg net P&L per trade: {trade_df['net_ret'].mean():.4f}")
        print(f"  Avg days held: {trade_df['days_held'].mean():.1f}")

        # By close reason
        print("\n  By close reason:")
        for reason in ['profit_take', 'roll', 'stop_loss']:
            subset = trade_df[trade_df['close_reason'] == reason]
            if len(subset) > 0:
                wr = (subset['net_pnl'] > 0).mean()
                avg_ret = subset['net_ret'].mean()
                print(f"    {reason:12s}: {len(subset):4d} trades | WR: {wr:.1%} | Avg ret: {avg_ret:+.4f}")

        # By VRP regime at entry
        print("\n  By entry VRP level:")
        for lo, hi, label in [(0.01, 0.03, 'Low VRP'), (0.03, 0.06, 'Mid VRP'), (0.06, 1.0, 'High VRP')]:
            subset = trade_df[(trade_df['entry_vrp'] >= lo) & (trade_df['entry_vrp'] < hi)]
            if len(subset) > 0:
                wr = (subset['net_pnl'] > 0).mean()
                avg_ret = subset['net_ret'].mean()
                print(f"    {label:12s}: {len(subset):4d} trades | WR: {wr:.1%} | Avg ret: {avg_ret:+.4f}")

        # Worst trades
        print("\n  5 Worst trades:")
        worst = trade_df.nsmallest(5, 'net_ret')
        for _, t in worst.iterrows():
            print(f"    {t['entry_date'].strftime('%Y-%m-%d') if hasattr(t['entry_date'], 'strftime') else t['entry_date']} "
                  f"→ {t['exit_date'].strftime('%Y-%m-%d') if hasattr(t['exit_date'], 'strftime') else t['exit_date']} "
                  f"| ret={t['net_ret']:+.4f} | reason={t['close_reason']} | SPY: {t['entry_spy']:.0f}→{t['exit_spy']:.0f}")

    # 7. Lag sensitivity
    print_section("LAG SENSITIVITY")
    _, t0_rets, t0_trades = simulate_strangle_strategy(df, vrp_threshold=0.02, lag=0)
    _, t1_rets, t1_trades = simulate_strangle_strategy(df, vrp_threshold=0.02, lag=1)
    _, t2_rets, t2_trades = simulate_strangle_strategy(df, vrp_threshold=0.02, lag=2)

    for lag_val, rets, trs in [(0, t0_rets, t0_trades), (1, t1_rets, t1_trades), (2, t2_rets, t2_trades)]:
        nonzero = rets[rets != 0]
        m = calc_metrics(nonzero) if len(nonzero) > 2 else {'sharpe': 0, 'wr': 0}
        print(f"  Lag={lag_val}: Sharpe={m['sharpe']:.3f} | WR={m['wr']:.1%} | Trades={len(trs)}")

    m0 = calc_metrics(t0_rets[t0_rets != 0])
    m1 = calc_metrics(t1_rets[t1_rets != 0])
    if m0['sharpe'] != 0:
        deg = (m0['sharpe'] - m1['sharpe']) / abs(m0['sharpe']) * 100
        print(f"  Degradation T0→T1: {deg:+.1f}%")
        print(f"  Verdict: {'FRAGILE - lookahead dependent!' if abs(deg) > 50 else 'ROBUST'}")

    # 8. Final summary
    final_m = calc_metrics(final_returns)
    print_section("FINAL SUMMARY")
    print(f"  Strategy: VRP Harvesting via SPY Strangle Selling")
    print(f"  Best variant: {best_name}")
    print(f"  Walk-forward validated: {'Yes' if wf_used else 'No (full-sample only)'}")
    print(f"\n  Performance ({('WF OOS' if wf_used else 'full-sample')}):")
    print_metrics(final_m)
    print(f"\n  Gates passed: {n_pass}/4")
    for gname, gval in gates.items():
        status = 'PASS' if gval.get('pass') else 'FAIL'
        print(f"    {gname}: {status}")

    # 9. Save results
    output = {
        'strategy': 'VRP Harvesting v1 — SPY Strangle Selling',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'best_variant': best_name,
        'walk_forward_used': wf_used,
        'metrics': final_m,
        'gates_passed': n_pass,
        'gates': {k: {kk: (str(vv) if isinstance(vv, (pd.Timestamp, datetime)) else vv)
                       for kk, vv in v.items()} for k, v in gates.items()},
        'n_trades': len(final_trades),
        'trade_stats': {
            'avg_hold_days': float(np.mean([t['days_held'] for t in final_trades])) if final_trades else 0,
            'win_rate': float(np.mean([1 for t in final_trades if t['net_pnl'] > 0]) / len(final_trades)) if final_trades else 0,
            'avg_net_ret': float(np.mean([t['net_ret'] for t in final_trades])) if final_trades else 0,
        },
        'variant_comparison': {
            name: {
                'sharpe': r['metrics']['sharpe'],
                'sortino': r['metrics']['sortino'],
                'pf': r['metrics']['pf'],
                'wr': r['metrics']['wr'],
                'cagr': r['metrics']['cagr'],
                'max_dd': r['metrics']['max_dd'],
                'n_trades': r['n_trades'],
            }
            for name, r in variant_results.items()
        },
        'wf_windows': wf_results if wf_used else [],
    }

    out_path = '/home/jupiter/Lvl3Quant/research/findings/vrp_harvesting_v1_results.json'
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to findings/vrp_harvesting_v1_results.json")

    return output


if __name__ == '__main__':
    try:
        results = main()
    except Exception as e:
        print(f"\n*** ERROR: {e} ***")
        traceback.print_exc()
        sys.exit(1)
