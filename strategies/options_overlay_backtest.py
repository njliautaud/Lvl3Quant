#!/usr/bin/env python3
"""
Options Overlay Backtest — Amplify Validated Equity Signals with Options
=========================================================================

Our validated strategies (Sentiment Contrarian Sharpe 2.73, Macro Regime Rotation
Sharpe 4.39, Sector Rotation Sharpe 2.87) generate reliable BUY signals on sector
ETFs but only return 7-10% because they trade small and sit in cash.

This backtest replaces share purchases with ATM/slightly-OTM call options
(delta 0.40-0.60, 30-45 DTE) to get 3-5x leverage with defined risk.

Signal generation (simplified versions of our validated signals):
  1. Quality Dip: RSI(14) < 30 + price > 200-SMA
  2. Volume Spike Reversal: volume > 2x 20d avg + price dip > -1%
  3. Sector Relative Strength Flip: worst-performing → showing improvement

Options pricing: Black-Scholes with IV = 20d realized vol * 1.2
Exit rules:
  - Underlying +3% gain → sell option
  - Option premium -50% → stop loss
  - 21 DTE remaining → theta decay exit
  - Trailing stop: 30% from peak option value

Walk-forward: 2020-2026, monthly OOS windows
Comparison: options overlay vs shares-only on same signals
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import math
import sys
import json
import time
from datetime import datetime, timedelta
from scipy import stats

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance required. pip install yfinance")
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

SECTOR_ETFS = ['XLK', 'XLE', 'XLF', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU',
               'XLC', 'XLRE', 'XLB']

START_DATE = '2019-01-01'   # warmup start (need 200+ days for SMA200)
OOS_START  = '2020-01-01'   # backtest start
OOS_END    = '2026-08-22'   # backtest end

INITIAL_CAPITAL = 10000.0
MAX_ALLOC_PER_TRADE = 0.07   # 7% of portfolio per option trade
MAX_CONCURRENT = 4           # max open positions
COMMISSION_PER_CONTRACT = 0.65  # per contract, per leg

# Options parameters
DTE_TARGET = 37              # ~5 weeks to expiry
DTE_MIN = 30
DTE_MAX = 45
TARGET_DELTA = 0.50          # ATM
RISK_FREE_RATE = 0.045
IV_MULTIPLIER = 1.20         # IV premium over realized vol
CONTRACT_MULTIPLIER = 100    # 1 option = 100 shares

# Exit rules
UNDERLYING_TP_PCT = 0.03     # sell when underlying +3%
OPTION_SL_PCT = -0.50        # stop loss at -50% of premium
THETA_EXIT_DTE = 21          # exit at 21 DTE (avoid theta crush)
TRAILING_STOP_PCT = 0.25     # 25% from peak option value

# Walk-forward
WF_TRAIN_MONTHS = 6
WF_OOS_MONTHS = 1

# Shares-only comparison
SHARES_ALLOC_PER_TRADE = 0.15  # 15% of portfolio per share trade
SHARES_TP_PCT = 0.03
SHARES_SL_PCT = -0.02
SHARES_MAX_HOLD_DAYS = 10


# =============================================================================
# BLACK-SCHOLES
# =============================================================================

def norm_cdf(x):
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def norm_pdf(x):
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def bs_call(S, K, T, r, sigma):
    """Black-Scholes call price."""
    if T <= 1e-8 or sigma <= 1e-8:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)

def bs_call_delta(S, K, T, r, sigma):
    """Call delta."""
    if T <= 1e-8 or sigma <= 1e-8:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return norm_cdf(d1)

def bs_call_gamma(S, K, T, r, sigma):
    """Call gamma."""
    if T <= 1e-8 or sigma <= 1e-8:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return norm_pdf(d1) / (S * sigma * math.sqrt(T))

def bs_call_theta(S, K, T, r, sigma):
    """Call theta (per day)."""
    if T <= 1e-8 or sigma <= 1e-8:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    theta = (-S * norm_pdf(d1) * sigma / (2 * math.sqrt(T))
             - r * K * math.exp(-r * T) * norm_cdf(d2))
    return theta / 365.0  # per calendar day

def find_strike_for_delta(S, T, r, sigma, target_delta=0.50):
    """
    Find the strike that gives approximately the target delta.
    Uses bisection between deep ITM and deep OTM.
    """
    K_low = S * 0.80
    K_high = S * 1.20
    for _ in range(50):
        K_mid = (K_low + K_high) / 2.0
        d = bs_call_delta(S, K_mid, T, r, sigma)
        if d > target_delta:
            K_low = K_mid
        else:
            K_high = K_mid
    # Round to nearest dollar (ETF options have $1 strikes)
    return round((K_low + K_high) / 2.0)

def compute_iv_series(close_series, window=20):
    """Compute IV estimate: realized vol * IV_MULTIPLIER, floored/capped."""
    log_ret = np.log(close_series / close_series.shift(1))
    rv = log_ret.rolling(window).std() * np.sqrt(252)
    iv = rv * IV_MULTIPLIER
    return iv.clip(lower=0.10, upper=1.50)


# =============================================================================
# DATA DOWNLOAD
# =============================================================================

def download_data():
    """Download OHLCV data for all sector ETFs + SPY."""
    tickers = SECTOR_ETFS + ['SPY']
    print(f"Downloading {len(tickers)} tickers from {START_DATE} to {OOS_END}...")

    raw = yf.download(tickers, start=START_DATE, end=OOS_END, auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw['Close'].copy()
        volume = raw['Volume'].copy()
        high = raw['High'].copy()
        low = raw['Low'].copy()
    else:
        close = raw[['Close']].copy()
        volume = raw[['Volume']].copy()
        high = raw[['High']].copy()
        low = raw[['Low']].copy()

    # Flatten any remaining MultiIndex
    for df in [close, volume, high, low]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    close = close.ffill().dropna(how='all')
    volume = volume.ffill().fillna(0)
    high = high.ffill().dropna(how='all')
    low = low.ffill().dropna(how='all')

    print(f"Data range: {close.index.min().date()} to {close.index.max().date()}")
    print(f"Trading days: {len(close)}")
    return close, volume, high, low


# =============================================================================
# SIGNAL GENERATION
# =============================================================================

def compute_rsi(close_arr, period=14):
    """Compute RSI series."""
    delta = np.diff(close_arr, prepend=close_arr[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).rolling(period).mean().values
    avg_loss = pd.Series(loss).rolling(period).mean().values
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
    return 100 - 100 / (1 + rs)

def generate_signals(close, volume, high, low):
    """
    Generate BUY CALL signals for each sector ETF using three signal types:

    1. Quality Dip: RSI(14) < 30 AND price > 200-SMA (quality in a dip)
    2. Volume Spike Reversal: volume > 2x 20d avg AND daily return < -1%
    3. Sector Relative Strength Flip: worst 3 sectors (21d) showing 5d improvement

    Returns list of dicts: {date, ticker, signal_type, strength}
    """
    spy = close['SPY'] if 'SPY' in close.columns else None
    etfs = [e for e in SECTOR_ETFS if e in close.columns]
    signals = []

    oos_mask = close.index >= OOS_START

    # Precompute indicators per ETF
    for etf in etfs:
        c = close[etf].values.astype(float)
        v = volume[etf].values.astype(float) if etf in volume.columns else np.ones(len(c))
        n = len(c)
        dates = close.index

        # RSI
        rsi = compute_rsi(c)

        # SMA200
        sma200 = pd.Series(c).rolling(200).mean().values

        # SMA50
        sma50 = pd.Series(c).rolling(50).mean().values

        # Volume 20d avg
        vol_avg20 = pd.Series(v).rolling(20).mean().values

        # Daily returns
        rets = np.zeros(n)
        rets[1:] = (c[1:] - c[:-1]) / c[:-1]

        # 5d return
        ret5d = np.zeros(n)
        ret5d[5:] = (c[5:] - c[:-5]) / c[:-5]

        # 21d return
        ret21d = np.zeros(n)
        ret21d[21:] = (c[21:] - c[:-21]) / c[:-21]

        for i in range(210, n):
            if not oos_mask[i]:
                continue
            date = dates[i]

            # ----- Signal 1: Quality Dip -----
            # RSI < 30 AND price above 200-SMA (uptrend dip)
            if (not np.isnan(sma200[i]) and rsi[i] < 30 and c[i] > sma200[i]):
                strength = (30 - rsi[i]) / 30.0  # stronger when RSI is lower
                signals.append({
                    'date': date, 'ticker': etf,
                    'signal_type': 'quality_dip',
                    'strength': strength,
                    'entry_price': c[i]
                })

            # ----- Signal 2: Volume Spike Reversal -----
            # Volume > 2x 20d avg AND price dropped > 1%
            if (not np.isnan(vol_avg20[i]) and vol_avg20[i] > 0
                    and v[i] > 2.0 * vol_avg20[i]
                    and rets[i] < -0.01):
                vol_ratio = v[i] / vol_avg20[i]
                strength = min(vol_ratio / 4.0, 1.0) * abs(rets[i]) / 0.03
                strength = min(strength, 1.0)
                signals.append({
                    'date': date, 'ticker': etf,
                    'signal_type': 'volume_spike_reversal',
                    'strength': strength,
                    'entry_price': c[i]
                })

            # ----- Signal 3: Sector RS Flip -----
            # Check if this ETF is in bottom 3 on 21d but improving on 5d
            # (we only add if relative to SPY it's bouncing)
            if spy is not None and etf in close.columns:
                spy_ret21 = 0.0
                spy_idx = close.index.get_loc(date)
                if spy_idx >= 21:
                    spy_ret21 = (spy.iloc[spy_idx] - spy.iloc[spy_idx - 21]) / spy.iloc[spy_idx - 21]
                rel_str_21d = ret21d[i] - spy_ret21
                # If among worst performers (rel str < -3%) but bouncing strongly (5d > 1%)
                if rel_str_21d < -0.03 and ret5d[i] > 0.01:
                    # Quality filter: above SMA50, RSI not overbought
                    if (not np.isnan(sma50[i]) and c[i] > sma50[i] * 0.97
                            and rsi[i] < 60 and rsi[i] > 25):
                        strength = min(abs(rel_str_21d) * 8 * ret5d[i] * 15, 1.0)
                        # Only include if strength is meaningful
                        if strength >= 0.25:
                            signals.append({
                                'date': date, 'ticker': etf,
                                'signal_type': 'rs_flip',
                                'strength': strength,
                                'entry_price': c[i]
                            })

    print(f"Generated {len(signals)} raw signals across {len(etfs)} ETFs")
    # Count by type
    type_counts = {}
    for s in signals:
        t = s['signal_type']
        type_counts[t] = type_counts.get(t, 0) + 1
    for t, cnt in sorted(type_counts.items()):
        print(f"  {t}: {cnt}")

    return signals


# =============================================================================
# OPTIONS OVERLAY BACKTEST
# =============================================================================

def run_options_overlay(signals, close, volume):
    """
    Backtest: buy ATM call options when signal fires.

    For each signal:
    - Find strike near delta 0.50 using BS
    - Price the call with BS (IV = 20d realized vol * 1.2)
    - Buy contracts worth up to MAX_ALLOC_PER_TRADE of current equity
    - Track daily mark-to-market
    - Exit on: underlying +3%, premium -50%, 21 DTE, or trailing stop 30%
    """
    # Precompute IV for all ETFs
    iv_dict = {}
    for etf in SECTOR_ETFS:
        if etf in close.columns:
            iv_dict[etf] = compute_iv_series(close[etf])

    # Sort signals by date
    signals = sorted(signals, key=lambda x: x['date'])

    equity = INITIAL_CAPITAL
    equity_curve = []
    positions = []  # open positions
    trades = []     # closed trades
    daily_equity = {}

    all_dates = close.index[close.index >= OOS_START]

    for date in all_dates:
        # ---- Mark existing positions to market ----
        closed_today = []
        for pos in positions:
            etf = pos['ticker']
            if etf not in close.columns:
                continue

            current_price = close[etf].get(date, np.nan)
            if np.isnan(current_price):
                continue

            # Days elapsed
            days_held = (date - pos['entry_date']).days
            T_remain = max((pos['expiry_dte'] - days_held) / 365.0, 1e-8)
            dte_remain = pos['expiry_dte'] - days_held

            # Current IV (use latest available)
            current_iv = iv_dict[etf].get(date, pos['entry_iv'])
            if np.isnan(current_iv):
                current_iv = pos['entry_iv']

            # Reprice option
            current_option_price = bs_call(
                current_price, pos['strike'], T_remain,
                RISK_FREE_RATE, current_iv
            )

            pos['current_option_price'] = current_option_price
            pos['current_underlying'] = current_price
            current_value = current_option_price * CONTRACT_MULTIPLIER * pos['num_contracts']
            pos['current_value'] = current_value

            # Track peak for trailing stop
            if current_value > pos.get('peak_value', 0):
                pos['peak_value'] = current_value

            # ---- EXIT LOGIC ----
            exit_reason = None
            underlying_return = (current_price - pos['entry_underlying']) / pos['entry_underlying']
            option_return = (current_option_price - pos['entry_option_price']) / pos['entry_option_price']

            # 1. Underlying TP: +3%
            if underlying_return >= UNDERLYING_TP_PCT:
                exit_reason = 'underlying_tp'

            # 2. Option SL: -50% of premium
            elif option_return <= OPTION_SL_PCT:
                exit_reason = 'option_sl'

            # 3. Theta decay exit: 21 DTE
            elif dte_remain <= THETA_EXIT_DTE:
                exit_reason = 'theta_exit'

            # 4. Trailing stop: 30% from peak value
            elif pos.get('peak_value', 0) > 0:
                drawdown_from_peak = (current_value - pos['peak_value']) / pos['peak_value']
                if drawdown_from_peak <= -TRAILING_STOP_PCT:
                    exit_reason = 'trailing_stop'

            if exit_reason:
                # Close position
                proceeds = current_value - COMMISSION_PER_CONTRACT * pos['num_contracts']
                pnl = proceeds - pos['cost_basis']
                pnl_pct = pnl / pos['cost_basis'] if pos['cost_basis'] > 0 else 0

                equity += proceeds
                trades.append({
                    'entry_date': pos['entry_date'],
                    'exit_date': date,
                    'ticker': etf,
                    'signal_type': pos['signal_type'],
                    'strike': pos['strike'],
                    'entry_underlying': pos['entry_underlying'],
                    'exit_underlying': current_price,
                    'underlying_return': underlying_return,
                    'entry_option_price': pos['entry_option_price'],
                    'exit_option_price': current_option_price,
                    'option_return': option_return,
                    'num_contracts': pos['num_contracts'],
                    'cost_basis': pos['cost_basis'],
                    'proceeds': proceeds,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'days_held': days_held,
                    'exit_reason': exit_reason,
                })
                closed_today.append(pos)

        # Remove closed positions
        for pos in closed_today:
            positions.remove(pos)

        # ---- Open new positions ----
        todays_signals = [s for s in signals if s['date'] == date]
        # Sort by strength descending, take best
        todays_signals.sort(key=lambda x: x['strength'], reverse=True)

        # Deduplicate: only one signal per ticker per day
        seen_tickers = set(p['ticker'] for p in positions)
        for sig in todays_signals:
            if len(positions) >= MAX_CONCURRENT:
                break
            if sig['ticker'] in seen_tickers:
                continue

            etf = sig['ticker']
            if etf not in close.columns or etf not in iv_dict:
                continue

            current_price = close[etf].get(date, np.nan)
            if np.isnan(current_price) or current_price <= 0:
                continue

            iv = iv_dict[etf].get(date, np.nan)
            if np.isnan(iv) or iv <= 0:
                continue

            T = DTE_TARGET / 365.0

            # Find strike for target delta
            strike = find_strike_for_delta(current_price, T, RISK_FREE_RATE, iv, TARGET_DELTA)

            # Price the call
            option_price = bs_call(current_price, strike, T, RISK_FREE_RATE, iv)
            if option_price < 0.10:
                continue  # too cheap, probably bad pricing

            delta = bs_call_delta(current_price, strike, T, RISK_FREE_RATE, iv)

            # Determine position size: allocate up to MAX_ALLOC_PER_TRADE of equity
            alloc = equity * MAX_ALLOC_PER_TRADE
            cost_per_contract = option_price * CONTRACT_MULTIPLIER + COMMISSION_PER_CONTRACT
            num_contracts = max(1, int(alloc / cost_per_contract))

            # Cap at what we can afford
            total_cost = cost_per_contract * num_contracts
            while total_cost > alloc and num_contracts > 1:
                num_contracts -= 1
                total_cost = cost_per_contract * num_contracts

            if total_cost > equity * 0.25:
                continue  # safety: never risk >25% of equity on one trade

            # Effective leverage
            notional = current_price * CONTRACT_MULTIPLIER * num_contracts
            leverage = notional / total_cost

            # Open position
            equity -= total_cost
            positions.append({
                'ticker': etf,
                'signal_type': sig['signal_type'],
                'entry_date': date,
                'entry_underlying': current_price,
                'strike': strike,
                'entry_option_price': option_price,
                'entry_iv': iv,
                'entry_delta': delta,
                'num_contracts': num_contracts,
                'cost_basis': total_cost,
                'expiry_dte': DTE_TARGET,
                'current_value': total_cost,
                'peak_value': total_cost,
                'leverage': leverage,
            })
            seen_tickers.add(etf)

        # Daily equity = cash + sum of open position values
        pos_value = sum(p.get('current_value', p['cost_basis']) for p in positions)
        total_equity = equity + pos_value
        daily_equity[date] = total_equity
        equity_curve.append({'date': date, 'equity': total_equity, 'cash': equity,
                             'positions': len(positions)})

    # Close any remaining positions at last date
    last_date = all_dates[-1]
    for pos in positions:
        etf = pos['ticker']
        current_price = close[etf].get(last_date, pos['entry_underlying'])
        proceeds = pos.get('current_value', 0) - COMMISSION_PER_CONTRACT * pos['num_contracts']
        pnl = proceeds - pos['cost_basis']
        equity += proceeds
        trades.append({
            'entry_date': pos['entry_date'],
            'exit_date': last_date,
            'ticker': etf,
            'signal_type': pos['signal_type'],
            'strike': pos['strike'],
            'entry_underlying': pos['entry_underlying'],
            'exit_underlying': current_price,
            'underlying_return': (current_price - pos['entry_underlying']) / pos['entry_underlying'],
            'entry_option_price': pos['entry_option_price'],
            'exit_option_price': pos.get('current_option_price', 0),
            'option_return': (pos.get('current_option_price', 0) - pos['entry_option_price']) / pos['entry_option_price'],
            'num_contracts': pos['num_contracts'],
            'cost_basis': pos['cost_basis'],
            'proceeds': proceeds,
            'pnl': pnl,
            'pnl_pct': pnl / pos['cost_basis'] if pos['cost_basis'] > 0 else 0,
            'days_held': (last_date - pos['entry_date']).days,
            'exit_reason': 'end_of_backtest',
        })

    return trades, equity_curve


# =============================================================================
# SHARES-ONLY BACKTEST (COMPARISON)
# =============================================================================

def run_shares_only(signals, close):
    """
    Same signals, but buy shares instead of options.
    Simple long-only: buy shares, TP +3%, SL -2%, max hold 10 days.
    """
    signals = sorted(signals, key=lambda x: x['date'])

    equity = INITIAL_CAPITAL
    equity_curve = []
    positions = []
    trades = []

    all_dates = close.index[close.index >= OOS_START]

    for date in all_dates:
        # Mark to market and check exits
        closed_today = []
        for pos in positions:
            etf = pos['ticker']
            current_price = close[etf].get(date, np.nan)
            if np.isnan(current_price):
                continue

            ret = (current_price - pos['entry_price']) / pos['entry_price']
            days_held = (date - pos['entry_date']).days

            exit_reason = None
            if ret >= SHARES_TP_PCT:
                exit_reason = 'tp'
            elif ret <= SHARES_SL_PCT:
                exit_reason = 'sl'
            elif days_held >= SHARES_MAX_HOLD_DAYS:
                exit_reason = 'time_stop'

            if exit_reason:
                proceeds = pos['shares'] * current_price
                pnl = proceeds - pos['cost_basis']
                equity += proceeds
                trades.append({
                    'entry_date': pos['entry_date'],
                    'exit_date': date,
                    'ticker': etf,
                    'entry_price': pos['entry_price'],
                    'exit_price': current_price,
                    'return': ret,
                    'pnl': pnl,
                    'pnl_pct': pnl / pos['cost_basis'],
                    'days_held': days_held,
                    'exit_reason': exit_reason,
                })
                closed_today.append(pos)

        for pos in closed_today:
            positions.remove(pos)

        # New positions
        todays_signals = [s for s in signals if s['date'] == date]
        todays_signals.sort(key=lambda x: x['strength'], reverse=True)
        seen = set(p['ticker'] for p in positions)

        for sig in todays_signals:
            if len(positions) >= MAX_CONCURRENT:
                break
            if sig['ticker'] in seen:
                continue

            etf = sig['ticker']
            current_price = close[etf].get(date, np.nan)
            if np.isnan(current_price) or current_price <= 0:
                continue

            alloc = equity * SHARES_ALLOC_PER_TRADE
            shares = int(alloc / current_price)
            if shares < 1:
                continue

            cost = shares * current_price
            equity -= cost
            positions.append({
                'ticker': etf,
                'entry_date': date,
                'entry_price': current_price,
                'shares': shares,
                'cost_basis': cost,
            })
            seen.add(etf)

        pos_val = sum(p['shares'] * close[p['ticker']].get(date, p['entry_price'])
                      for p in positions)
        equity_curve.append({'date': date, 'equity': equity + pos_val})

    # Close remaining
    last_date = all_dates[-1]
    for pos in positions:
        current_price = close[pos['ticker']].get(last_date, pos['entry_price'])
        proceeds = pos['shares'] * current_price
        pnl = proceeds - pos['cost_basis']
        equity += proceeds
        trades.append({
            'entry_date': pos['entry_date'],
            'exit_date': last_date,
            'ticker': pos['ticker'],
            'entry_price': pos['entry_price'],
            'exit_price': current_price,
            'return': (current_price - pos['entry_price']) / pos['entry_price'],
            'pnl': pnl,
            'pnl_pct': pnl / pos['cost_basis'],
            'days_held': (last_date - pos['entry_date']).days,
            'exit_reason': 'end_of_backtest',
        })

    return trades, equity_curve


# =============================================================================
# WALK-FORWARD ANALYSIS
# =============================================================================

def walk_forward_analysis(signals, close, volume):
    """
    Walk-forward: train on 6 months, test on 1 month.
    In-sample: tune signal strength threshold.
    OOS: trade with that threshold.
    Returns OOS-only trades and equity curve.
    """
    all_dates = close.index[close.index >= OOS_START]
    if len(all_dates) == 0:
        return [], []

    start = all_dates[0]
    end = all_dates[-1]

    # Generate monthly windows
    windows = []
    current = pd.Timestamp(OOS_START)
    while current < pd.Timestamp(OOS_END):
        train_start = current - pd.DateOffset(months=WF_TRAIN_MONTHS)
        train_end = current - pd.DateOffset(days=1)
        oos_start = current
        oos_end = current + pd.DateOffset(months=WF_OOS_MONTHS) - pd.DateOffset(days=1)
        windows.append((train_start, train_end, oos_start, min(oos_end, pd.Timestamp(OOS_END))))
        current += pd.DateOffset(months=WF_OOS_MONTHS)

    print(f"\nWalk-forward: {len(windows)} monthly windows")

    # For WF, filter signals by strength threshold optimized IS
    oos_signals = []
    for train_s, train_e, oos_s, oos_e in windows:
        # IS signals
        is_sigs = [s for s in signals if train_s <= s['date'] <= train_e]

        # Find optimal strength threshold (maximize IS win rate)
        best_thresh = 0.3
        best_score = -1
        for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
            filtered = [s for s in is_sigs if s['strength'] >= thresh]
            if len(filtered) < 5:
                continue
            # Quick score: count signals where underlying moved +1% within 5 days
            wins = 0
            for s in filtered:
                etf = s['ticker']
                if etf not in close.columns:
                    continue
                idx = close.index.get_loc(s['date'])
                if idx + 5 < len(close):
                    future_ret = (close[etf].iloc[idx + 5] - close[etf].iloc[idx]) / close[etf].iloc[idx]
                    if future_ret > 0.01:
                        wins += 1
            wr = wins / len(filtered) if filtered else 0
            score = wr * len(filtered)  # balance WR and trade count
            if score > best_score:
                best_score = score
                best_thresh = thresh

        # Apply threshold to OOS signals
        oos_sigs = [s for s in signals if oos_s <= s['date'] <= oos_e and s['strength'] >= best_thresh]
        oos_signals.extend(oos_sigs)

    print(f"WF filtered signals: {len(oos_signals)} (from {len(signals)} raw)")
    return oos_signals


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(trades, equity_curve, label="Strategy"):
    """Compute and print performance metrics."""
    if not trades or not equity_curve:
        print(f"\n{label}: No trades")
        return {}

    eq = pd.DataFrame(equity_curve)
    eq['date'] = pd.to_datetime(eq['date'])
    eq = eq.set_index('date').sort_index()

    # Daily returns
    eq['daily_ret'] = eq['equity'].pct_change().fillna(0)

    # Time in market
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    if years <= 0:
        years = 1.0

    # CAGR
    final_equity = eq['equity'].iloc[-1]
    initial_equity = eq['equity'].iloc[0]
    cagr = (final_equity / initial_equity) ** (1 / years) - 1

    # Sharpe
    daily_rets = eq['daily_ret'].values
    if daily_rets.std() > 0:
        sharpe = np.sqrt(252) * daily_rets.mean() / daily_rets.std()
    else:
        sharpe = 0.0

    # Sortino
    downside = daily_rets[daily_rets < 0]
    if len(downside) > 0 and downside.std() > 0:
        sortino = np.sqrt(252) * daily_rets.mean() / downside.std()
    else:
        sortino = 0.0

    # Max drawdown
    cummax = eq['equity'].cummax()
    drawdown = (eq['equity'] - cummax) / cummax
    max_dd = drawdown.min()

    # Trade stats
    pnls = [t['pnl'] for t in trades]
    pnl_pcts = [t['pnl_pct'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    win_rate = len(wins) / len(pnls) if pnls else 0
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf')
    avg_return = np.mean(pnl_pcts) if pnl_pcts else 0
    avg_days = np.mean([t['days_held'] for t in trades]) if trades else 0

    # Total return
    total_return = (final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL

    metrics = {
        'label': label,
        'total_return': total_return,
        'cagr': cagr,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_dd': max_dd,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_return_per_trade': avg_return,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'num_trades': len(trades),
        'avg_days_held': avg_days,
        'final_equity': final_equity,
    }

    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")
    print(f"  Total Return:     {total_return*100:>8.1f}%")
    print(f"  CAGR:             {cagr*100:>8.1f}%")
    print(f"  Sharpe:           {sharpe:>8.2f}")
    print(f"  Sortino:          {sortino:>8.2f}")
    print(f"  Max Drawdown:     {max_dd*100:>8.1f}%")
    print(f"  Win Rate:         {win_rate*100:>8.1f}%")
    print(f"  Profit Factor:    {profit_factor:>8.2f}")
    print(f"  Avg Return/Trade: {avg_return*100:>8.2f}%")
    print(f"  Avg Win:          ${avg_win:>8.2f}")
    print(f"  Avg Loss:         ${avg_loss:>8.2f}")
    print(f"  Num Trades:       {len(trades):>8d}")
    print(f"  Avg Days Held:    {avg_days:>8.1f}")
    print(f"  Final Equity:     ${final_equity:>10,.2f}")
    print(f"{'='*65}")

    # Exit reason breakdown
    exit_counts = {}
    for t in trades:
        r = t.get('exit_reason', 'unknown')
        exit_counts[r] = exit_counts.get(r, 0) + 1
    print(f"\n  Exit Reasons:")
    for reason, cnt in sorted(exit_counts.items(), key=lambda x: -x[1]):
        print(f"    {reason:<20s} {cnt:>5d}  ({cnt/len(trades)*100:.1f}%)")

    # Per signal type (options only)
    if 'signal_type' in trades[0]:
        print(f"\n  Performance by Signal Type:")
        sig_types = set(t.get('signal_type', '') for t in trades)
        for st in sorted(sig_types):
            st_trades = [t for t in trades if t.get('signal_type') == st]
            st_pnls = [t['pnl'] for t in st_trades]
            st_wins = len([p for p in st_pnls if p > 0])
            st_wr = st_wins / len(st_pnls) if st_pnls else 0
            st_avg = np.mean([t['pnl_pct'] for t in st_trades])
            print(f"    {st:<25s}  N={len(st_trades):>4d}  WR={st_wr*100:.1f}%  AvgRet={st_avg*100:.2f}%")

    return metrics


# =============================================================================
# LEVERAGE ANALYSIS
# =============================================================================

def analyze_leverage(options_trades):
    """Analyze effective leverage achieved through options."""
    if not options_trades:
        return

    leverages = []
    for t in options_trades:
        if 'entry_underlying' in t and 'entry_option_price' in t and t['entry_option_price'] > 0:
            # Effective leverage = underlying price / option premium
            eff_lev = t['entry_underlying'] / t['entry_option_price']
            leverages.append(eff_lev)

    if leverages:
        print(f"\n  Leverage Analysis:")
        print(f"    Avg Effective Leverage:  {np.mean(leverages):.1f}x")
        print(f"    Min Leverage:            {np.min(leverages):.1f}x")
        print(f"    Max Leverage:            {np.max(leverages):.1f}x")
        print(f"    Median Leverage:         {np.median(leverages):.1f}x")

    # Gamma/convexity benefit: big moves amplified
    big_winners = [t for t in options_trades if t['pnl_pct'] > 0.50]
    if big_winners:
        avg_underlying_move = np.mean([t['underlying_return'] for t in big_winners])
        avg_option_move = np.mean([t['option_return'] for t in big_winners])
        print(f"\n    Big Winners (>50% option return):")
        print(f"      Count:                 {len(big_winners)}")
        print(f"      Avg Underlying Move:   {avg_underlying_move*100:.2f}%")
        print(f"      Avg Option Return:     {avg_option_move*100:.1f}%")
        print(f"      Amplification:         {avg_option_move/avg_underlying_move:.1f}x" if avg_underlying_move > 0 else "")


# =============================================================================
# YEARLY BREAKDOWN
# =============================================================================

def yearly_breakdown(equity_curve, label="Strategy"):
    """Show year-by-year performance."""
    if not equity_curve:
        return

    eq = pd.DataFrame(equity_curve)
    eq['date'] = pd.to_datetime(eq['date'])
    eq = eq.set_index('date').sort_index()
    eq['daily_ret'] = eq['equity'].pct_change().fillna(0)

    print(f"\n  Yearly Breakdown — {label}:")
    print(f"  {'Year':<6} {'Return':>8} {'Sharpe':>8} {'MaxDD':>8} {'EndEquity':>12}")
    print(f"  {'-'*46}")

    for year in sorted(eq.index.year.unique()):
        yr_data = eq[eq.index.year == year]
        if len(yr_data) < 2:
            continue
        yr_ret = (yr_data['equity'].iloc[-1] / yr_data['equity'].iloc[0]) - 1
        yr_rets = yr_data['daily_ret'].values
        yr_sharpe = np.sqrt(252) * yr_rets.mean() / yr_rets.std() if yr_rets.std() > 0 else 0
        cummax = yr_data['equity'].cummax()
        yr_dd = ((yr_data['equity'] - cummax) / cummax).min()
        print(f"  {year:<6} {yr_ret*100:>7.1f}% {yr_sharpe:>8.2f} {yr_dd*100:>7.1f}% ${yr_data['equity'].iloc[-1]:>10,.0f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  OPTIONS OVERLAY BACKTEST — Amplifying Validated Equity Signals")
    print("=" * 70)
    print(f"\n  Capital: ${INITIAL_CAPITAL:,.0f}")
    print(f"  Period: {OOS_START} to {OOS_END}")
    print(f"  ETFs: {', '.join(SECTOR_ETFS)}")
    print(f"  Options: {DTE_TARGET} DTE, delta ~{TARGET_DELTA}, BS pricing")
    print(f"  Max alloc/trade: {MAX_ALLOC_PER_TRADE*100:.0f}%  Max concurrent: {MAX_CONCURRENT}")

    # Download data
    close, volume, high, low = download_data()

    # Generate signals
    print("\n--- Signal Generation ---")
    all_signals = generate_signals(close, volume, high, low)

    if not all_signals:
        print("ERROR: No signals generated. Check data.")
        return

    # Walk-forward filtering
    print("\n--- Walk-Forward Signal Filtering ---")
    wf_signals = walk_forward_analysis(all_signals, close, volume)
    if not wf_signals:
        print("WARNING: WF filtering removed all signals. Using raw signals.")
        wf_signals = all_signals

    # Run OPTIONS overlay
    print("\n--- Running Options Overlay Backtest ---")
    opt_trades, opt_equity = run_options_overlay(wf_signals, close, volume)
    opt_metrics = compute_metrics(opt_trades, opt_equity, "OPTIONS OVERLAY (Calls, delta ~0.50)")
    analyze_leverage(opt_trades)
    yearly_breakdown(opt_equity, "Options Overlay")

    # Run SHARES-ONLY comparison
    print("\n--- Running Shares-Only Comparison ---")
    share_trades, share_equity = run_shares_only(wf_signals, close)
    share_metrics = compute_metrics(share_trades, share_equity, "SHARES ONLY (Same Signals)")
    yearly_breakdown(share_equity, "Shares Only")

    # Comparison summary
    print("\n" + "=" * 70)
    print("  COMPARISON: OPTIONS vs SHARES")
    print("=" * 70)
    if opt_metrics and share_metrics:
        print(f"\n  {'Metric':<25s} {'Options':>12s} {'Shares':>12s} {'Amplification':>15s}")
        print(f"  {'-'*66}")
        for key, fmt, label in [
            ('cagr', '.1f', 'CAGR (%)'),
            ('sharpe', '.2f', 'Sharpe'),
            ('sortino', '.2f', 'Sortino'),
            ('max_dd', '.1f', 'Max DD (%)'),
            ('win_rate', '.1f', 'Win Rate (%)'),
            ('profit_factor', '.2f', 'Profit Factor'),
            ('num_trades', 'd', 'Num Trades'),
            ('total_return', '.1f', 'Total Return (%)'),
        ]:
            o_val = opt_metrics.get(key, 0)
            s_val = share_metrics.get(key, 0)
            if key in ('cagr', 'max_dd', 'win_rate', 'total_return'):
                o_str = f"{o_val*100:{fmt}}%"
                s_str = f"{s_val*100:{fmt}}%"
            elif key == 'num_trades':
                o_str = f"{o_val:{fmt}}"
                s_str = f"{s_val:{fmt}}"
            else:
                o_str = f"{o_val:{fmt}}"
                s_str = f"{s_val:{fmt}}"

            if s_val != 0 and key in ('cagr', 'total_return'):
                amp = f"{o_val/s_val:.1f}x"
            else:
                amp = ""
            print(f"  {label:<25s} {o_str:>12s} {s_str:>12s} {amp:>15s}")

    # Save results
    results = {
        'options': opt_metrics,
        'shares': share_metrics,
        'config': {
            'initial_capital': INITIAL_CAPITAL,
            'dte_target': DTE_TARGET,
            'target_delta': TARGET_DELTA,
            'max_alloc_per_trade': MAX_ALLOC_PER_TRADE,
            'underlying_tp': UNDERLYING_TP_PCT,
            'option_sl': OPTION_SL_PCT,
            'theta_exit_dte': THETA_EXIT_DTE,
            'trailing_stop': TRAILING_STOP_PCT,
        },
        'run_date': datetime.now().isoformat(),
    }

    results_path = '/home/jupiter/Lvl3Quant/strategies/options_overlay_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    return results


if __name__ == '__main__':
    main()
