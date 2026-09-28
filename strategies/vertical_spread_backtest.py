#!/usr/bin/env python3
"""
Vertical Spread Backtest for Sector ETFs
=========================================

Tests bull call spreads and bear put spreads on sector ETFs using
synthetic signals derived from momentum/mean-reversion indicators.

Compares single-leg options vs vertical spreads for a small account ($751).

Key parameters:
  - Strike widths: $1, $2, $3, $5
  - Entry: aggregator confidence >= 78%
  - Exit: TP +30%, SL -25%, time stop 5 days, trailing stop 15%
  - Commission: $0.53/leg ($2.12 RT for spread, $1.06 RT for single-leg)
  - Options pricing: Black-Scholes with IV = 20-day realized vol * 1.2
  - DTE: 21-35 days (standard monthly expiry range)

5-Gate validation:
  G1: Sharpe > 0.5
  G2: Permutation test p < 0.05
  G3: Regime gap < 0.50
  G4: Survives 20bps additional cost
  G5: At least 30 trades
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import math
import sys
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

SECTOR_ETFS = ['XLE', 'XLU', 'XLK', 'XLI', 'SMH', 'XLV', 'XLP', 'XLF', 'XLY', 'XLC', 'XLRE']
START_DATE = '2020-01-01'
END_DATE = '2026-08-15'

ACCOUNT_SIZE = 751.0
MAX_RISK_PCT = 0.20  # 20% per trade = $150.20 max risk
COMMISSION_PER_LEG = 0.53  # Robinhood-like
SPREAD_RT_COMMISSION = 4 * COMMISSION_PER_LEG  # 4 legs RT (open + close both legs)
SINGLE_RT_COMMISSION = 2 * COMMISSION_PER_LEG  # 2 legs RT (open + close single)

# Signal thresholds
CONFIDENCE_THRESHOLD = 0.78  # 78% aggregator confidence

# Exit rules
TP_PCT = 0.30        # Take profit at +30%
SL_PCT = -0.25       # Stop loss at -25%
TIME_STOP_DAYS = 5   # Max hold 5 trading days
TRAILING_STOP_PCT = 0.15  # 15% trailing stop from peak

# Strike widths to test
STRIKE_WIDTHS = [1, 2, 3, 5]

# Options parameters
DTE_TARGET = 28  # ~4 weeks to expiry
RISK_FREE_RATE = 0.045  # ~4.5% (recent T-bill)
IV_MULTIPLIER = 1.2  # IV premium over realized vol

# Permutation test
N_PERMUTATIONS = 1000

# =============================================================================
# BLACK-SCHOLES
# =============================================================================

def norm_cdf(x):
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_call(S, K, T, r, sigma):
    """Black-Scholes call price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)

def bs_put(S, K, T, r, sigma):
    """Black-Scholes put price."""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)

def bs_delta_call(S, K, T, r, sigma):
    """Call delta."""
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return norm_cdf(d1)

def bs_delta_put(S, K, T, r, sigma):
    """Put delta."""
    return bs_delta_call(S, K, T, r, sigma) - 1.0

def compute_iv(prices, window=20):
    """Compute implied vol estimate: 20-day realized vol * IV_MULTIPLIER."""
    log_ret = np.log(prices / prices.shift(1))
    rv = log_ret.rolling(window).std() * np.sqrt(252)
    iv = rv * IV_MULTIPLIER
    return iv.clip(lower=0.10, upper=1.50)  # Floor 10%, cap 150%


# =============================================================================
# SIGNAL GENERATION (Synthetic aggregator mimicking our live system)
# =============================================================================

def generate_signals(df):
    """
    Generate synthetic aggregator signals from price data.
    Mimics our multi-indicator confluence system:
      - RSI oversold/overbought
      - MACD cross
      - Bollinger Band touch
      - Volume surge
      - 20-day momentum

    Returns DataFrame with 'signal' (-1, 0, +1) and 'confidence' (0-1).
    """
    close = df['Close'].values.astype(float)
    high = df['High'].values.astype(float)
    low = df['Low'].values.astype(float)
    volume = df['Volume'].values.astype(float)
    n = len(close)

    signals = np.zeros(n)
    confidences = np.zeros(n)

    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).rolling(14).mean().values
    avg_loss = pd.Series(loss).rolling(14).mean().values
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
    rsi = 100 - 100 / (1 + rs)

    # MACD
    ema12 = pd.Series(close).ewm(span=12).mean().values
    ema26 = pd.Series(close).ewm(span=26).mean().values
    macd_line = ema12 - ema26
    macd_signal = pd.Series(macd_line).ewm(span=9).mean().values
    macd_hist = macd_line - macd_signal

    # Bollinger Bands (20, 2)
    sma20 = pd.Series(close).rolling(20).mean().values
    std20 = pd.Series(close).rolling(20).std().values
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20

    # Volume surge (vs 20-day avg)
    vol_avg = pd.Series(volume).rolling(20).mean().values
    with np.errstate(divide='ignore', invalid='ignore'):
        vol_ratio = np.where(vol_avg > 0, volume / vol_avg, 1.0)

    # 20-day momentum
    mom20 = np.zeros(n)
    mom20[20:] = (close[20:] - close[:-20]) / close[:-20]

    # SMA crossovers (additional signal)
    sma5 = pd.Series(close).rolling(5).mean().values
    sma10 = pd.Series(close).rolling(10).mean().values

    # Stochastic RSI
    rsi_series = pd.Series(rsi)
    rsi_low14 = rsi_series.rolling(14).min().values
    rsi_high14 = rsi_series.rolling(14).max().values
    with np.errstate(divide='ignore', invalid='ignore'):
        stoch_rsi = np.where(rsi_high14 - rsi_low14 > 0,
                             (rsi - rsi_low14) / (rsi_high14 - rsi_low14), 0.5)

    # ADX proxy: directional movement strength
    atr14 = pd.Series(high - low).rolling(14).mean().values

    # Aggregate signals (start after warmup)
    for i in range(30, n):
        bull_score = 0
        bear_score = 0
        total_weight = 8  # More indicators = more granular confidence

        # RSI (widened thresholds)
        if rsi[i] < 35:
            bull_score += 1.0
        elif rsi[i] < 45:
            bull_score += 0.4
        elif rsi[i] > 65:
            bear_score += 1.0
        elif rsi[i] > 55:
            bear_score += 0.4

        # Stochastic RSI
        if stoch_rsi[i] < 0.20:
            bull_score += 0.8
        elif stoch_rsi[i] > 0.80:
            bear_score += 0.8

        # MACD cross
        if macd_hist[i] > 0 and macd_hist[i-1] <= 0:
            bull_score += 1.5
        elif macd_hist[i] < 0 and macd_hist[i-1] >= 0:
            bear_score += 1.5
        elif macd_hist[i] > 0:
            bull_score += 0.5
        elif macd_hist[i] < 0:
            bear_score += 0.5

        # Bollinger Band (widened: within 0.5 std of band)
        if not np.isnan(bb_lower[i]) and not np.isnan(std20[i]) and std20[i] > 0:
            if close[i] <= bb_lower[i]:
                bull_score += 1.2
            elif close[i] <= bb_lower[i] + 0.5 * std20[i]:
                bull_score += 0.5
            elif close[i] >= bb_upper[i]:
                bear_score += 1.2
            elif close[i] >= bb_upper[i] - 0.5 * std20[i]:
                bear_score += 0.5

        # SMA crossover
        if not np.isnan(sma5[i]) and not np.isnan(sma10[i]):
            if sma5[i] > sma10[i] and sma5[i-1] <= sma10[i-1]:
                bull_score += 1.0
            elif sma5[i] < sma10[i] and sma5[i-1] >= sma10[i-1]:
                bear_score += 1.0
            elif sma5[i] > sma10[i]:
                bull_score += 0.3
            elif sma5[i] < sma10[i]:
                bear_score += 0.3

        # Volume surge confirms
        if vol_ratio[i] > 1.3:
            if close[i] > close[i-1]:
                bull_score += 0.8
            else:
                bear_score += 0.8

        # Momentum (multi-timeframe)
        if mom20[i] < -0.03:
            bull_score += 0.5  # Mean-reversion opportunity
        elif mom20[i] > 0.08:
            bull_score += 0.3  # Trend continuation
            bear_score += 0.2  # Overextension risk

        # 5-day momentum
        if i >= 5:
            mom5 = (close[i] - close[i-5]) / close[i-5]
            if mom5 < -0.02:
                bull_score += 0.4
            elif mom5 > 0.03:
                bull_score += 0.3

        # Price vs SMA20
        if not np.isnan(sma20[i]) and sma20[i] > 0:
            pct_from_sma = (close[i] - sma20[i]) / sma20[i]
            if pct_from_sma < -0.03:
                bull_score += 0.5
            elif pct_from_sma > 0.03:
                bear_score += 0.3

        # Calculate confidence
        net = bull_score - bear_score
        max_possible = total_weight
        raw_conf = abs(net) / max_possible
        # Scale to 0.60-0.95 range with noise
        confidence = 0.60 + 0.35 * min(raw_conf, 1.0)
        # Add small noise to avoid clustering
        confidence = np.clip(confidence + np.random.normal(0, 0.04), 0.55, 0.99)

        if net > 0.2:
            signals[i] = 1
            confidences[i] = confidence
        elif net < -0.2:
            signals[i] = -1
            confidences[i] = confidence

    return signals, confidences


# =============================================================================
# SPREAD PRICING
# =============================================================================

def price_bull_call_spread(S, width, T, r, sigma):
    """
    Price a bull call spread: buy ATM call, sell OTM call.
    Returns (debit, max_profit, breakeven).
    """
    K_long = round(S)  # ATM
    K_short = K_long + width  # OTM

    long_premium = bs_call(S, K_long, T, r, sigma)
    short_premium = bs_call(S, K_short, T, r, sigma)

    debit = long_premium - short_premium  # Net debit (cost)
    max_profit = width - debit
    breakeven = K_long + debit

    return debit, max_profit, breakeven, K_long, K_short

def price_bear_put_spread(S, width, T, r, sigma):
    """
    Price a bear put spread: buy ATM put, sell OTM put.
    Returns (debit, max_profit, breakeven).
    """
    K_long = round(S)  # ATM
    K_short = K_long - width  # OTM (lower strike)

    long_premium = bs_put(S, K_long, T, r, sigma)
    short_premium = bs_put(S, K_short, T, r, sigma)

    debit = long_premium - short_premium  # Net debit (cost)
    max_profit = width - debit
    breakeven = K_long - debit

    return debit, max_profit, breakeven, K_long, K_short

def price_single_call(S, T, r, sigma):
    """Price a single ATM call."""
    K = round(S)
    premium = bs_call(S, K, T, r, sigma)
    return premium, K

def price_single_put(S, T, r, sigma):
    """Price a single ATM put."""
    K = round(S)
    premium = bs_put(S, K, T, r, sigma)
    return premium, K


def spread_value_at_time(S_now, K_long, K_short, T_remain, r, sigma, spread_type):
    """
    Mark-to-market value of spread at some point before expiry.
    spread_type: 'bull_call' or 'bear_put'
    """
    if spread_type == 'bull_call':
        long_val = bs_call(S_now, K_long, T_remain, r, sigma)
        short_val = bs_call(S_now, K_short, T_remain, r, sigma)
    else:  # bear_put
        long_val = bs_put(S_now, K_long, T_remain, r, sigma)
        short_val = bs_put(S_now, K_short, T_remain, r, sigma)
    return long_val - short_val

def single_option_value_at_time(S_now, K, T_remain, r, sigma, opt_type):
    """Mark-to-market value of a single option."""
    if opt_type == 'call':
        return bs_call(S_now, K, T_remain, r, sigma)
    else:
        return bs_put(S_now, K, T_remain, r, sigma)


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

def run_backtest(etf_data, spy_data, mode='spread', strike_width=2):
    """
    Run backtest across all ETFs.

    mode: 'spread' or 'single'
    strike_width: width for spreads (ignored for single)

    Returns list of trade dicts.
    """
    trades = []
    np.random.seed(42)  # Reproducible signal noise

    for ticker, df in etf_data.items():
        if len(df) < 60:
            continue

        # Flatten multi-level columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        close = df['Close'].values.astype(float)
        iv_series = compute_iv(df['Close'])
        signals, confidences = generate_signals(df)
        dates = df.index

        i = 30  # Start after warmup
        while i < len(df) - TIME_STOP_DAYS - 1:
            sig = signals[i]
            conf = confidences[i]

            if sig == 0 or conf < CONFIDENCE_THRESHOLD:
                i += 1
                continue

            S = close[i]
            iv = iv_series.iloc[i] if not np.isnan(iv_series.iloc[i]) else 0.25
            T = DTE_TARGET / 365.0

            direction = 'bull' if sig == 1 else 'bear'

            if mode == 'spread':
                if direction == 'bull':
                    debit, max_prof, beven, K_long, K_short = price_bull_call_spread(
                        S, strike_width, T, RISK_FREE_RATE, iv)
                    spread_type = 'bull_call'
                else:
                    debit, max_prof, beven, K_long, K_short = price_bear_put_spread(
                        S, strike_width, T, RISK_FREE_RATE, iv)
                    spread_type = 'bear_put'

                # Contract cost (x100 multiplier)
                entry_cost = debit * 100
                max_profit_dollars = max_prof * 100

                if entry_cost <= 0 or entry_cost > ACCOUNT_SIZE * MAX_RISK_PCT:
                    i += 1
                    continue

                # Simulate exit over next TIME_STOP_DAYS
                peak_value = entry_cost
                exit_reason = 'time_stop'
                exit_value = entry_cost  # default
                exit_day = min(i + TIME_STOP_DAYS, len(df) - 1)

                for j in range(1, TIME_STOP_DAYS + 1):
                    if i + j >= len(df):
                        break

                    S_j = close[i + j]
                    T_remain = max((DTE_TARGET - j) / 365.0, 1/365.0)

                    # IV mean-reverts slightly during hold
                    iv_j = iv * (1 - 0.01 * j) if iv > 0.15 else iv

                    current_value = spread_value_at_time(
                        S_j, K_long, K_short, T_remain, RISK_FREE_RATE, iv_j, spread_type) * 100

                    pnl_pct = (current_value - entry_cost) / entry_cost

                    # Track peak for trailing stop
                    if current_value > peak_value:
                        peak_value = current_value

                    # Take profit
                    if pnl_pct >= TP_PCT:
                        exit_value = current_value
                        exit_reason = 'take_profit'
                        exit_day = i + j
                        break

                    # Stop loss
                    if pnl_pct <= SL_PCT:
                        exit_value = current_value
                        exit_reason = 'stop_loss'
                        exit_day = i + j
                        break

                    # Trailing stop
                    if peak_value > entry_cost and (current_value - peak_value) / peak_value < -TRAILING_STOP_PCT:
                        exit_value = current_value
                        exit_reason = 'trailing_stop'
                        exit_day = i + j
                        break

                    # Time stop on last day
                    if j == TIME_STOP_DAYS:
                        exit_value = current_value
                        exit_reason = 'time_stop'
                        exit_day = i + j

                raw_pnl = exit_value - entry_cost
                net_pnl = raw_pnl - SPREAD_RT_COMMISSION

                trades.append({
                    'ticker': ticker,
                    'entry_date': str(dates[i].date()),
                    'exit_date': str(dates[min(exit_day, len(dates)-1)].date()),
                    'direction': direction,
                    'mode': 'spread',
                    'strike_width': strike_width,
                    'entry_cost': round(entry_cost, 2),
                    'exit_value': round(exit_value, 2),
                    'max_profit': round(max_profit_dollars, 2),
                    'raw_pnl': round(raw_pnl, 2),
                    'net_pnl': round(net_pnl, 2),
                    'commission': SPREAD_RT_COMMISSION,
                    'exit_reason': exit_reason,
                    'confidence': round(conf, 3),
                    'iv': round(iv, 3),
                    'underlying_price': round(S, 2),
                    'K_long': K_long,
                    'K_short': K_short,
                    'spy_date': str(dates[i].date()),
                })

            else:  # single-leg
                if direction == 'bull':
                    premium, K = price_single_call(S, T, RISK_FREE_RATE, iv)
                    opt_type = 'call'
                else:
                    premium, K = price_single_put(S, T, RISK_FREE_RATE, iv)
                    opt_type = 'put'

                entry_cost = premium * 100

                if entry_cost <= 0 or entry_cost > ACCOUNT_SIZE * MAX_RISK_PCT:
                    i += 1
                    continue

                # Simulate exit
                peak_value = entry_cost
                exit_reason = 'time_stop'
                exit_value = entry_cost
                exit_day = min(i + TIME_STOP_DAYS, len(df) - 1)

                for j in range(1, TIME_STOP_DAYS + 1):
                    if i + j >= len(df):
                        break

                    S_j = close[i + j]
                    T_remain = max((DTE_TARGET - j) / 365.0, 1/365.0)
                    iv_j = iv * (1 - 0.01 * j) if iv > 0.15 else iv

                    current_value = single_option_value_at_time(
                        S_j, K, T_remain, RISK_FREE_RATE, iv_j, opt_type) * 100

                    pnl_pct = (current_value - entry_cost) / entry_cost if entry_cost > 0 else 0

                    if current_value > peak_value:
                        peak_value = current_value

                    if pnl_pct >= TP_PCT:
                        exit_value = current_value
                        exit_reason = 'take_profit'
                        exit_day = i + j
                        break

                    if pnl_pct <= SL_PCT:
                        exit_value = current_value
                        exit_reason = 'stop_loss'
                        exit_day = i + j
                        break

                    if peak_value > entry_cost and (current_value - peak_value) / peak_value < -TRAILING_STOP_PCT:
                        exit_value = current_value
                        exit_reason = 'trailing_stop'
                        exit_day = i + j
                        break

                    if j == TIME_STOP_DAYS:
                        exit_value = current_value
                        exit_reason = 'time_stop'
                        exit_day = i + j

                raw_pnl = exit_value - entry_cost
                net_pnl = raw_pnl - SINGLE_RT_COMMISSION

                trades.append({
                    'ticker': ticker,
                    'entry_date': str(dates[i].date()),
                    'exit_date': str(dates[min(exit_day, len(dates)-1)].date()),
                    'direction': direction,
                    'mode': 'single',
                    'strike_width': 0,
                    'entry_cost': round(entry_cost, 2),
                    'exit_value': round(exit_value, 2),
                    'max_profit': round(exit_value, 2),  # Unlimited for single
                    'raw_pnl': round(raw_pnl, 2),
                    'net_pnl': round(net_pnl, 2),
                    'commission': SINGLE_RT_COMMISSION,
                    'exit_reason': exit_reason,
                    'confidence': round(conf, 3),
                    'iv': round(iv, 3),
                    'underlying_price': round(S, 2),
                    'K_long': K,
                    'K_short': 0,
                    'spy_date': str(dates[i].date()),
                })

            # Skip forward to avoid overlapping trades on same ticker
            # Use actual hold duration + 1 day cooldown
            hold_days = exit_day - i if exit_day > i else TIME_STOP_DAYS
            i += hold_days + 1
            continue

    return trades


# =============================================================================
# METRICS COMPUTATION
# =============================================================================

def compute_metrics(trades):
    """Compute performance metrics from trade list."""
    if not trades:
        return None

    pnls = np.array([t['net_pnl'] for t in trades])
    costs = np.array([t['entry_cost'] for t in trades])

    n_trades = len(pnls)
    total_pnl = pnls.sum()
    avg_pnl = pnls.mean()
    avg_cost = costs.mean()
    median_cost = np.median(costs)

    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    win_rate = len(wins) / n_trades if n_trades > 0 else 0

    gross_profit = wins.sum() if len(wins) > 0 else 0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Returns per trade (percent of entry cost)
    returns = pnls / np.where(costs > 0, costs, 1)

    # Sharpe (annualized, assuming ~50 trades/year)
    trades_per_year = max(n_trades / 6.5, 1)  # ~6.5 years of data
    if returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(trades_per_year)
    else:
        sharpe = 0

    # Sortino
    downside = returns[returns < 0]
    if len(downside) > 0 and downside.std() > 0:
        sortino = (returns.mean() / downside.std()) * np.sqrt(trades_per_year)
    else:
        sortino = sharpe

    # Max drawdown (cumulative P&L based)
    cum_pnl = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum_pnl)
    drawdown = cum_pnl - peak
    max_dd = drawdown.min()
    max_dd_pct = max_dd / ACCOUNT_SIZE if ACCOUNT_SIZE > 0 else 0

    # Affordability: how many trades could we take with $751?
    affordable_trades = sum(1 for c in costs if c <= ACCOUNT_SIZE * MAX_RISK_PCT)
    affordable_pct = affordable_trades / n_trades if n_trades > 0 else 0

    # Exit reason distribution
    exit_reasons = {}
    for t in trades:
        r = t['exit_reason']
        exit_reasons[r] = exit_reasons.get(r, 0) + 1

    # Direction breakdown
    bull_trades = [t for t in trades if t['direction'] == 'bull']
    bear_trades = [t for t in trades if t['direction'] == 'bear']

    return {
        'n_trades': n_trades,
        'total_pnl': round(total_pnl, 2),
        'avg_pnl': round(avg_pnl, 2),
        'avg_cost': round(avg_cost, 2),
        'median_cost': round(median_cost, 2),
        'win_rate': round(win_rate, 4),
        'profit_factor': round(profit_factor, 3),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'max_drawdown': round(max_dd, 2),
        'max_drawdown_pct': round(max_dd_pct, 4),
        'affordable_trades': affordable_trades,
        'affordable_pct': round(affordable_pct, 4),
        'exit_reasons': exit_reasons,
        'n_bull': len(bull_trades),
        'n_bear': len(bear_trades),
        'avg_bull_pnl': round(np.mean([t['net_pnl'] for t in bull_trades]), 2) if bull_trades else 0,
        'avg_bear_pnl': round(np.mean([t['net_pnl'] for t in bear_trades]), 2) if bear_trades else 0,
    }


def permutation_test(trades, n_perms=N_PERMUTATIONS):
    """Permutation test: is the mean P&L significantly different from random?"""
    if len(trades) < 10:
        return 1.0  # Not enough trades

    pnls = np.array([t['net_pnl'] for t in trades])
    observed = pnls.mean()

    count_extreme = 0
    for _ in range(n_perms):
        # Randomly flip signs (simulate random entry direction)
        shuffled = pnls * np.random.choice([-1, 1], size=len(pnls))
        if shuffled.mean() >= observed:
            count_extreme += 1

    return count_extreme / n_perms


def regime_test(trades, spy_data):
    """
    Test regime robustness: split trades by whether SPY was in
    bull (green) or bear (red) regime on entry date.
    """
    if len(trades) < 20:
        return 1.0, 0, 0  # Not enough, fail

    spy_close = spy_data['Close']
    if isinstance(spy_close, pd.DataFrame):
        spy_close = spy_close.iloc[:, 0]
    spy_sma50 = spy_close.rolling(50).mean()

    green_trades = []
    red_trades = []

    for t in trades:
        entry_date = pd.Timestamp(t['entry_date'])
        if entry_date in spy_close.index and entry_date in spy_sma50.index:
            price = float(spy_close.loc[entry_date])
            sma = float(spy_sma50.loc[entry_date])
            if np.isnan(sma):
                continue
            if price >= sma:
                green_trades.append(t)
            else:
                red_trades.append(t)

    if len(green_trades) < 3 or len(red_trades) < 3:
        return 1.0, 0, 0  # Can't compute

    green_pnls = np.array([t['net_pnl'] for t in green_trades])
    red_pnls = np.array([t['net_pnl'] for t in red_trades])
    costs_g = np.array([t['entry_cost'] for t in green_trades])
    costs_r = np.array([t['entry_cost'] for t in red_trades])

    ret_g = green_pnls / np.where(costs_g > 0, costs_g, 1)
    ret_r = red_pnls / np.where(costs_r > 0, costs_r, 1)

    sharpe_g = ret_g.mean() / ret_g.std() if ret_g.std() > 0 else 0
    sharpe_r = ret_r.mean() / ret_r.std() if ret_r.std() > 0 else 0

    max_sharpe = max(abs(sharpe_g), abs(sharpe_r))
    gap = abs(sharpe_g - sharpe_r) / max_sharpe if max_sharpe > 0 else 0

    return gap, round(sharpe_g, 3), round(sharpe_r, 3)


def cost_sensitivity_test(trades, extra_cost_bps=20):
    """Test if strategy survives with 20bps additional cost per trade."""
    adjusted_pnls = []
    for t in trades:
        extra = t['entry_cost'] * (extra_cost_bps / 10000)
        adjusted_pnls.append(t['net_pnl'] - extra)

    if not adjusted_pnls:
        return 0

    return np.mean(adjusted_pnls) > 0


# =============================================================================
# 5-GATE VALIDATION
# =============================================================================

def validate_5_gates(trades, spy_data, label=""):
    """Run 5-gate validation and print results."""
    metrics = compute_metrics(trades)
    if metrics is None:
        print(f"  {label}: NO TRADES")
        return None, None

    # Gate 1: Sharpe > 0.5
    g1 = metrics['sharpe'] > 0.5

    # Gate 2: Permutation test p < 0.05
    p_val = permutation_test(trades)
    g2 = p_val < 0.05

    # Gate 3: Regime gap < 0.50
    regime_gap, sharpe_g, sharpe_r = regime_test(trades, spy_data)
    g3 = regime_gap < 0.50

    # Gate 4: Survives 20bps additional cost
    g4 = cost_sensitivity_test(trades)

    # Gate 5: At least 30 trades
    g5 = metrics['n_trades'] >= 30

    gates_passed = sum([g1, g2, g3, g4, g5])

    gates = {
        'G1_sharpe': {'pass': g1, 'value': metrics['sharpe']},
        'G2_permutation': {'pass': g2, 'value': round(p_val, 4)},
        'G3_regime': {'pass': g3, 'value': round(regime_gap, 3), 'green_sharpe': sharpe_g, 'red_sharpe': sharpe_r},
        'G4_cost_robust': {'pass': g4, 'value': 'survives' if g4 else 'fails'},
        'G5_trade_count': {'pass': g5, 'value': metrics['n_trades']},
        'total_passed': gates_passed,
    }

    return metrics, gates


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("VERTICAL SPREAD BACKTEST — Sector ETFs")
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Account: ${ACCOUNT_SIZE:.0f} | Max risk/trade: ${ACCOUNT_SIZE * MAX_RISK_PCT:.0f}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD*100:.0f}%")
    print("=" * 80)

    # Download data
    print("\nDownloading market data...")
    all_tickers = SECTOR_ETFS + ['SPY']
    data = {}
    for ticker in all_tickers:
        try:
            df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 60:
                data[ticker] = df
                print(f"  {ticker}: {len(df)} bars")
            else:
                print(f"  {ticker}: insufficient data ({len(df)} bars)")
        except Exception as e:
            print(f"  {ticker}: download failed ({e})")

    spy_data = data.pop('SPY', None)
    etf_data = data

    if spy_data is None or len(etf_data) == 0:
        print("ERROR: Could not download required data.")
        return

    print(f"\nLoaded {len(etf_data)} ETFs + SPY")

    # ─── Run backtests ─────────────────────────────────────────────────────

    results = {}

    # 1. Single-leg baseline
    print("\n" + "─" * 60)
    print("Running SINGLE-LEG options backtest...")
    single_trades = run_backtest(etf_data, spy_data, mode='single')
    single_metrics, single_gates = validate_5_gates(single_trades, spy_data, "Single-Leg")
    results['single'] = {'trades': single_trades, 'metrics': single_metrics, 'gates': single_gates}

    # 2. Spreads at each width
    for width in STRIKE_WIDTHS:
        print(f"\nRunning SPREAD backtest (width=${width})...")
        spread_trades = run_backtest(etf_data, spy_data, mode='spread', strike_width=width)
        spread_metrics, spread_gates = validate_5_gates(spread_trades, spy_data, f"Spread $${width}")
        results[f'spread_{width}'] = {'trades': spread_trades, 'metrics': spread_metrics, 'gates': spread_gates}

    # ─── Print comprehensive results ──────────────────────────────────────

    print("\n" + "=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)

    # Header
    configs = ['single'] + [f'spread_{w}' for w in STRIKE_WIDTHS]
    labels = ['Single-Leg'] + [f'Spread ${w}' for w in STRIKE_WIDTHS]

    print(f"\n{'Metric':<25}", end="")
    for label in labels:
        print(f"{label:>14}", end="")
    print()
    print("-" * (25 + 14 * len(labels)))

    metric_rows = [
        ('Trades', 'n_trades', '{:>14d}'),
        ('Affordable Trades', 'affordable_trades', '{:>14d}'),
        ('Affordable %', 'affordable_pct', '{:>13.1%} '),
        ('Avg Cost/Trade', 'avg_cost', '{:>13.2f} '),
        ('Median Cost', 'median_cost', '{:>13.2f} '),
        ('Total P&L', 'total_pnl', '{:>13.2f} '),
        ('Avg P&L/Trade', 'avg_pnl', '{:>13.2f} '),
        ('Win Rate', 'win_rate', '{:>13.1%} '),
        ('Profit Factor', 'profit_factor', '{:>13.3f} '),
        ('Sharpe', 'sharpe', '{:>13.3f} '),
        ('Sortino', 'sortino', '{:>13.3f} '),
        ('Max Drawdown', 'max_drawdown', '{:>13.2f} '),
        ('Max DD %', 'max_drawdown_pct', '{:>13.1%} '),
        ('Bull Trades', 'n_bull', '{:>14d}'),
        ('Bear Trades', 'n_bear', '{:>14d}'),
        ('Avg Bull P&L', 'avg_bull_pnl', '{:>13.2f} '),
        ('Avg Bear P&L', 'avg_bear_pnl', '{:>13.2f} '),
    ]

    for row_label, key, fmt in metric_rows:
        print(f"{row_label:<25}", end="")
        for cfg in configs:
            m = results[cfg]['metrics']
            if m is None:
                print(f"{'N/A':>14}", end="")
            else:
                val = m.get(key, 0)
                try:
                    print(fmt.format(val), end="")
                except:
                    print(f"{val:>14}", end="")
        print()

    # 5-Gate results
    print(f"\n{'5-GATE VALIDATION':<25}", end="")
    for label in labels:
        print(f"{label:>14}", end="")
    print()
    print("-" * (25 + 14 * len(labels)))

    gate_labels = [
        ('G1: Sharpe > 0.5', 'G1_sharpe'),
        ('G2: Perm p < 0.05', 'G2_permutation'),
        ('G3: Regime gap < 0.50', 'G3_regime'),
        ('G4: Cost robust', 'G4_cost_robust'),
        ('G5: Trades >= 30', 'G5_trade_count'),
    ]

    for glabel, gkey in gate_labels:
        print(f"{glabel:<25}", end="")
        for cfg in configs:
            g = results[cfg]['gates']
            if g is None:
                print(f"{'N/A':>14}", end="")
            else:
                gate = g[gkey]
                status = "PASS" if gate['pass'] else "FAIL"
                val = gate['value']
                print(f"{status + ' (' + str(val) + ')':>14}", end="")
        print()

    # Total gates passed
    print(f"{'GATES PASSED':<25}", end="")
    for cfg in configs:
        g = results[cfg]['gates']
        if g is None:
            print(f"{'N/A':>14}", end="")
        else:
            print(f"{g['total_passed']:>14d}/5", end="")
    print()

    # Regime detail
    print(f"\n{'REGIME DETAIL':<25}", end="")
    for label in labels:
        print(f"{label:>14}", end="")
    print()
    print("-" * (25 + 14 * len(labels)))

    print(f"{'Green Sharpe':<25}", end="")
    for cfg in configs:
        g = results[cfg]['gates']
        if g and g['G3_regime'].get('green_sharpe') is not None:
            print(f"{g['G3_regime']['green_sharpe']:>14.3f}", end="")
        else:
            print(f"{'N/A':>14}", end="")
    print()

    print(f"{'Red Sharpe':<25}", end="")
    for cfg in configs:
        g = results[cfg]['gates']
        if g and g['G3_regime'].get('red_sharpe') is not None:
            print(f"{g['G3_regime']['red_sharpe']:>14.3f}", end="")
        else:
            print(f"{'N/A':>14}", end="")
    print()

    # Exit reason breakdown
    print(f"\n{'EXIT REASONS':<25}", end="")
    for label in labels:
        print(f"{label:>14}", end="")
    print()
    print("-" * (25 + 14 * len(labels)))

    for reason in ['take_profit', 'stop_loss', 'trailing_stop', 'time_stop']:
        print(f"  {reason:<23}", end="")
        for cfg in configs:
            m = results[cfg]['metrics']
            if m:
                count = m['exit_reasons'].get(reason, 0)
                print(f"{count:>14d}", end="")
            else:
                print(f"{'N/A':>14}", end="")
        print()

    # ─── Key insight: affordability comparison ────────────────────────────

    print("\n" + "=" * 80)
    print("KEY INSIGHT: AFFORDABILITY COMPARISON")
    print("=" * 80)

    if results['single']['metrics']:
        sm = results['single']['metrics']
        print(f"\nSingle-leg options:")
        print(f"  Average cost per trade: ${sm['avg_cost']:.2f}")
        print(f"  Affordable trades (under ${ACCOUNT_SIZE * MAX_RISK_PCT:.0f}): {sm['affordable_trades']} / {sm['n_trades']} ({sm['affordable_pct']:.1%})")

    for width in STRIKE_WIDTHS:
        key = f'spread_{width}'
        if results[key]['metrics']:
            m = results[key]['metrics']
            print(f"\n${width}-wide spreads:")
            print(f"  Average cost per trade: ${m['avg_cost']:.2f}")
            print(f"  Affordable trades (under ${ACCOUNT_SIZE * MAX_RISK_PCT:.0f}): {m['affordable_trades']} / {m['n_trades']} ({m['affordable_pct']:.1%})")
            if sm['avg_cost'] > 0:
                print(f"  Cost reduction vs single: {(1 - m['avg_cost']/sm['avg_cost'])*100:.0f}%")

    # ─── Best configuration ──────────────────────────────────────────────

    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    best_cfg = None
    best_score = -999
    for cfg in configs:
        m = results[cfg]['metrics']
        g = results[cfg]['gates']
        if m and g:
            # Score: gates passed * 10 + sharpe * 5 + affordable_pct * 3
            score = g['total_passed'] * 10 + m['sharpe'] * 5 + m['affordable_pct'] * 3 + m['profit_factor'] * 2
            if score > best_score:
                best_score = score
                best_cfg = cfg

    if best_cfg:
        m = results[best_cfg]['metrics']
        g = results[best_cfg]['gates']
        print(f"\nBest configuration: {best_cfg}")
        print(f"  Gates passed: {g['total_passed']}/5")
        print(f"  Sharpe: {m['sharpe']:.3f} | Sortino: {m['sortino']:.3f}")
        print(f"  Win rate: {m['win_rate']:.1%} | Profit factor: {m['profit_factor']:.3f}")
        print(f"  Avg cost: ${m['avg_cost']:.2f} | Affordable: {m['affordable_pct']:.1%}")
        print(f"  Total P&L: ${m['total_pnl']:.2f} over {m['n_trades']} trades")

    # ─── Per-ETF breakdown for best config ────────────────────────────────

    if best_cfg:
        trades = results[best_cfg]['trades']
        print(f"\nPer-ETF breakdown ({best_cfg}):")
        print(f"  {'ETF':<8} {'Trades':>8} {'Win%':>8} {'Avg P&L':>10} {'Total':>10}")
        print(f"  {'-'*44}")

        etf_groups = {}
        for t in trades:
            tk = t['ticker']
            if tk not in etf_groups:
                etf_groups[tk] = []
            etf_groups[tk].append(t)

        for tk in sorted(etf_groups.keys()):
            tlist = etf_groups[tk]
            pnls = [t['net_pnl'] for t in tlist]
            wr = sum(1 for p in pnls if p > 0) / len(pnls)
            print(f"  {tk:<8} {len(tlist):>8d} {wr:>7.1%} {np.mean(pnls):>10.2f} {sum(pnls):>10.2f}")

    # ─── SMH-specific analysis (user's key concern) ─────────────────────

    print("\n" + "=" * 80)
    print("SMH DEEP DIVE (Key signal: 80% confidence)")
    print("=" * 80)

    for cfg in configs:
        trades = results[cfg]['trades']
        smh_trades = [t for t in trades if t['ticker'] == 'SMH']
        if smh_trades:
            m = compute_metrics(smh_trades)
            if m:
                label = cfg.replace('_', ' $').replace('spread $', 'Spread $').replace('single', 'Single-Leg')
                print(f"\n  {label}:")
                print(f"    Trades: {m['n_trades']} | Win Rate: {m['win_rate']:.1%} | PF: {m['profit_factor']:.3f}")
                print(f"    Avg Cost: ${m['avg_cost']:.2f} | Avg P&L: ${m['avg_pnl']:.2f} | Total: ${m['total_pnl']:.2f}")
                if cfg == 'single':
                    smh_expensive = [t for t in smh_trades if t['entry_cost'] > 150]
                    print(f"    Trades too expensive for $751 acct (>${ACCOUNT_SIZE * MAX_RISK_PCT:.0f}): {len(smh_expensive)}/{len(smh_trades)}")

    # ─── High-confidence subset analysis ──────────────────────────────────

    print("\n" + "=" * 80)
    print("HIGH-CONFIDENCE SUBSET (>= 85% confidence)")
    print("=" * 80)

    for cfg in configs:
        trades = results[cfg]['trades']
        hc_trades = [t for t in trades if t['confidence'] >= 0.85]
        if len(hc_trades) >= 5:
            m = compute_metrics(hc_trades)
            if m:
                label = cfg.replace('_', ' $').replace('spread $', 'Spread $').replace('single', 'Single-Leg')
                print(f"\n  {label}: {m['n_trades']} trades")
                print(f"    Win Rate: {m['win_rate']:.1%} | PF: {m['profit_factor']:.3f} | Sharpe: {m['sharpe']:.3f}")
                print(f"    Avg Cost: ${m['avg_cost']:.2f} | Avg P&L: ${m['avg_pnl']:.2f} | Total: ${m['total_pnl']:.2f}")

    # ─── Spread capital efficiency analysis ───────────────────────────────

    print("\n" + "=" * 80)
    print("CAPITAL EFFICIENCY: SPREADS VS SINGLE-LEG")
    print("=" * 80)

    if results['single']['metrics']:
        sm = results['single']['metrics']
        print(f"\n  With ${ACCOUNT_SIZE:.0f} account, max ${ACCOUNT_SIZE * MAX_RISK_PCT:.0f} risk per trade:")
        print(f"\n  Single-leg options:")
        print(f"    Can take: {sm['affordable_trades']} trades")
        print(f"    Avg cost: ${sm['avg_cost']:.2f}")
        print(f"    Max concurrent trades (at avg cost): {int(ACCOUNT_SIZE / sm['avg_cost']) if sm['avg_cost'] > 0 else 0}")

        for width in STRIKE_WIDTHS:
            key = f'spread_{width}'
            m = results[key]['metrics']
            if m and m['avg_cost'] > 0:
                print(f"\n  ${width}-wide spreads:")
                print(f"    Can take: {m['affordable_trades']} trades")
                print(f"    Avg cost: ${m['avg_cost']:.2f}")
                max_concurrent = int(ACCOUNT_SIZE / m['avg_cost'])
                print(f"    Max concurrent trades (at avg cost): {max_concurrent}")
                trade_multiplier = m['affordable_trades'] / sm['affordable_trades'] if sm['affordable_trades'] > 0 else 0
                print(f"    Trade opportunity multiplier vs single: {trade_multiplier:.1f}x")

    print("\n" + "=" * 80)
    print("BACKTEST COMPLETE")
    print("=" * 80)

    return results


if __name__ == '__main__':
    results = main()
