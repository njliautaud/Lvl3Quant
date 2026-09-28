#!/usr/bin/env python3
"""
Options vs Shares Leverage Backtest
====================================
Tests 3 validated dip-buying strategies with 5 option structures each.
Uses Black-Scholes for option pricing (no historical options data available).

Strategies:
1. IV-RV Gap: VIX > SPY 20d RV + 5pts, RSI<40, stock >5% below 20-SMA
2. Bond Yield Signal: 10Y yield drops >0.1% in 5 days, RSI<40, stock >5% below 20-SMA
3. RSI Divergence: Price lower low vs 10d ago but RSI higher low, stock >5% below 20-SMA

Option Structures (per signal):
A. ATM Call (~0.50 delta), 30 DTE
B. OTM Call (~0.30 delta), 30 DTE, 3-5% OTM
C. Bull Call Spread (ATM/+5%), 30 DTE
D. Bull Put Spread (ATM/-5%), 30 DTE (credit)
E. LEAPS Call (~0.70 delta), 180 DTE, deep ITM

HC #767 — Critical research for $750 account 5-10x growth target.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# BLACK-SCHOLES PRICING
# ============================================================

def bs_d1(S, K, T, r, sigma):
    """Black-Scholes d1."""
    if T <= 0 or sigma <= 0:
        return 0.0
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def bs_d2(S, K, T, r, sigma):
    return bs_d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

def bs_call_price(S, K, T, r, sigma):
    """Black-Scholes call price."""
    if T <= 0:
        return max(S - K, 0)
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_put_price(S, K, T, r, sigma):
    """Black-Scholes put price."""
    if T <= 0:
        return max(K - S, 0)
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def bs_call_delta(S, K, T, r, sigma):
    if T <= 0:
        return 1.0 if S > K else 0.0
    return norm.cdf(bs_d1(S, K, T, r, sigma))

def find_strike_for_delta(S, target_delta, T, r, sigma, call=True):
    """Find strike that gives target delta via bisection."""
    lo, hi = S * 0.5, S * 1.5
    for _ in range(50):
        mid = (lo + hi) / 2
        d = bs_call_delta(S, mid, T, r, sigma)
        if call:
            if d > target_delta:
                lo = mid
            else:
                hi = mid
        else:
            if d < target_delta:
                hi = mid
            else:
                lo = mid
    return (lo + hi) / 2


# ============================================================
# DATA DOWNLOAD
# ============================================================

UNIVERSE = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'JPM', 'UNH', 'LLY', 'AVGO', 'AMD']
START = '2019-06-01'  # extra buffer for indicators
END = '2026-07-01'
SIGNAL_START = '2020-01-01'

RISK_FREE = 0.04
MAX_RISK_PER_TRADE = 150.0
STARTING_CAPITAL = 750.0
MAX_CONCURRENT = 2
SLIPPAGE_PCT = 0.05  # 5% on option prices
IV_CRUSH_PCT = 0.15  # IV drops 15% after dip recovery
TP_PCT = 1.0  # 100% gain on premium
SL_PCT = 0.50  # 50% loss on premium
UNDERLYING_TP_PCT = 0.05  # +5% on underlying
MAX_HOLD_DAYS = 21
MIN_DTE_EXIT = 7  # close at 7 DTE remaining

print("="*80)
print("OPTIONS vs SHARES LEVERAGE BACKTEST")
print("="*80)
print(f"\nDownloading data for {len(UNIVERSE)} stocks + VIX + TNX...")

# Download all data
tickers_to_download = UNIVERSE + ['^VIX', '^TNX', 'SPY']
data = {}
for ticker in tickers_to_download:
    try:
        df = yf.download(ticker, start=START, end=END, progress=False, auto_adjust=True)
        if len(df) > 100:
            data[ticker] = df
            print(f"  {ticker}: {len(df)} days")
        else:
            print(f"  {ticker}: INSUFFICIENT DATA ({len(df)} days)")
    except Exception as e:
        print(f"  {ticker}: FAILED - {e}")

vix = data.get('^VIX')
tnx = data.get('^TNX')
spy = data.get('SPY')

if vix is None or tnx is None or spy is None:
    raise RuntimeError("Missing VIX, TNX, or SPY data")

# Flatten multi-level columns if needed
for k in data:
    if isinstance(data[k].columns, pd.MultiIndex):
        data[k].columns = data[k].columns.get_level_values(0)

# ============================================================
# INDICATOR CALCULATIONS
# ============================================================

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_realized_vol(series, window=20):
    """Annualized realized vol from daily returns."""
    return series.pct_change().rolling(window).std() * np.sqrt(252)

# Pre-compute indicators for SPY and each stock
spy_rv = calc_realized_vol(spy['Close'], 20) * 100  # in percentage points

stock_indicators = {}
for ticker in UNIVERSE:
    if ticker not in data:
        continue
    df = data[ticker].copy()
    close = df['Close']
    stock_indicators[ticker] = {
        'close': close,
        'sma20': close.rolling(20).mean(),
        'rsi14': calc_rsi(close, 14),
        'rv20': calc_realized_vol(close, 20),
    }

# ============================================================
# SIGNAL GENERATION
# ============================================================

def generate_signals():
    """Generate buy signals for all 3 strategies across all stocks."""
    signals = []  # list of (date, ticker, strategy_name)

    vix_close = vix['Close']
    tnx_close = tnx['Close']

    for ticker in UNIVERSE:
        if ticker not in stock_indicators:
            continue
        ind = stock_indicators[ticker]
        close = ind['close']
        sma20 = ind['sma20']
        rsi14 = ind['rsi14']
        rv20 = ind['rv20']

        # Common dates
        common_idx = close.index.intersection(vix_close.index).intersection(tnx_close.index)
        common_idx = common_idx[common_idx >= SIGNAL_START]

        for dt in common_idx:
            try:
                s_close = float(close.loc[dt])
                s_sma20 = float(sma20.loc[dt]) if dt in sma20.index and not pd.isna(sma20.loc[dt]) else None
                s_rsi = float(rsi14.loc[dt]) if dt in rsi14.index and not pd.isna(rsi14.loc[dt]) else None
                s_rv = float(rv20.loc[dt]) if dt in rv20.index and not pd.isna(rv20.loc[dt]) else None
                v = float(vix_close.loc[dt]) if not pd.isna(vix_close.loc[dt]) else None

                if s_sma20 is None or s_rsi is None or s_rv is None or v is None:
                    continue

                # Common filter: stock >5% below 20-SMA
                below_sma = (s_close < s_sma20 * 0.95)
                rsi_low = (s_rsi < 40)

                if not below_sma:
                    continue

                # Strategy 1: IV-RV Gap
                spy_rv_val = float(spy_rv.loc[dt]) if dt in spy_rv.index and not pd.isna(spy_rv.loc[dt]) else None
                if spy_rv_val is not None and rsi_low:
                    if v > spy_rv_val + 5:
                        signals.append((dt, ticker, 'IV-RV Gap'))

                # Strategy 2: Bond Yield Signal
                tnx_5d_ago_idx = tnx_close.index.get_indexer([dt - timedelta(days=7)], method='nearest')
                if len(tnx_5d_ago_idx) > 0 and tnx_5d_ago_idx[0] >= 0:
                    tnx_5d_ago = float(tnx_close.iloc[tnx_5d_ago_idx[0]])
                    tnx_now = float(tnx_close.loc[dt])
                    if rsi_low and (tnx_5d_ago - tnx_now) > 0.1:
                        signals.append((dt, ticker, 'Bond Yield Signal'))

                # Strategy 3: RSI Divergence
                close_10d_ago_idx = close.index.get_indexer([dt - timedelta(days=14)], method='nearest')
                rsi_10d_ago_idx = rsi14.index.get_indexer([dt - timedelta(days=14)], method='nearest')
                if (len(close_10d_ago_idx) > 0 and close_10d_ago_idx[0] >= 0 and
                    len(rsi_10d_ago_idx) > 0 and rsi_10d_ago_idx[0] >= 0):
                    close_10d = float(close.iloc[close_10d_ago_idx[0]])
                    rsi_10d = float(rsi14.iloc[rsi_10d_ago_idx[0]])
                    if s_close < close_10d and s_rsi > rsi_10d and rsi_low:
                        signals.append((dt, ticker, 'RSI Divergence'))
            except (KeyError, IndexError, TypeError):
                continue

    return signals

print("\nGenerating signals...")
all_signals = generate_signals()
print(f"Total raw signals: {len(all_signals)}")

# Count by strategy
from collections import Counter
strat_counts = Counter(s[2] for s in all_signals)
for strat, cnt in strat_counts.items():
    print(f"  {strat}: {cnt} signals")

# ============================================================
# OPTION TRADE SIMULATOR
# ============================================================

def get_stock_iv(ticker, date, vix_val):
    """Estimate per-stock IV using VIX as base, scaled by stock's realized vol."""
    ind = stock_indicators.get(ticker)
    if ind is None:
        return vix_val / 100.0
    rv = ind['rv20']
    if date in rv.index and not pd.isna(rv.loc[date]):
        stock_rv = float(rv.loc[date])
    else:
        stock_rv = 0.30  # default

    # SPY realized vol
    spy_rv_val = float(spy_rv.loc[date]) / 100.0 if date in spy_rv.index and not pd.isna(spy_rv.loc[date]) else 0.15
    if spy_rv_val < 0.01:
        spy_rv_val = 0.15

    # Scale VIX by stock vol relative to market vol
    ratio = stock_rv / spy_rv_val
    ratio = np.clip(ratio, 0.5, 3.0)  # cap extremes
    return (vix_val / 100.0) * ratio


def simulate_option_trade(ticker, entry_date, strategy_type, close_series):
    """
    Simulate an option trade from entry to exit.

    Returns dict with trade results or None if trade couldn't be taken.
    strategy_type: 'atm_call', 'otm_call', 'bull_call_spread', 'bull_put_spread', 'leaps_call'
    """
    close = close_series
    entry_idx = close.index.get_loc(entry_date)
    S_entry = float(close.iloc[entry_idx])

    # Get IV at entry
    vix_val = float(vix['Close'].loc[entry_date]) if entry_date in vix['Close'].index else 20.0
    iv_entry = get_stock_iv(ticker, entry_date, vix_val)
    iv_entry = max(iv_entry, 0.10)  # floor at 10%

    # Set up option parameters based on strategy
    if strategy_type == 'atm_call':
        DTE = 30
        K = round(S_entry, 0)  # ATM
        T = DTE / 365.0
        premium = bs_call_price(S_entry, K, T, RISK_FREE, iv_entry)
        premium *= (1 + SLIPPAGE_PCT)  # add slippage on entry
        if premium < 0.10:
            return None
        contracts = max(1, int(MAX_RISK_PER_TRADE / (premium * 100)))
        cost = contracts * premium * 100
        if cost > MAX_RISK_PER_TRADE * 1.5:
            contracts = 1
            cost = premium * 100
        trade_type = 'debit'

    elif strategy_type == 'otm_call':
        DTE = 30
        K = round(S_entry * 1.04, 0)  # ~4% OTM
        T = DTE / 365.0
        premium = bs_call_price(S_entry, K, T, RISK_FREE, iv_entry)
        premium *= (1 + SLIPPAGE_PCT)
        if premium < 0.05:
            return None
        contracts = max(1, int(MAX_RISK_PER_TRADE / (premium * 100)))
        cost = contracts * premium * 100
        if cost > MAX_RISK_PER_TRADE * 1.5:
            contracts = 1
            cost = premium * 100
        trade_type = 'debit'

    elif strategy_type == 'bull_call_spread':
        DTE = 30
        K_long = round(S_entry, 0)  # ATM
        K_short = round(S_entry * 1.05, 0)  # 5% OTM
        T = DTE / 365.0
        long_prem = bs_call_price(S_entry, K_long, T, RISK_FREE, iv_entry)
        short_prem = bs_call_price(S_entry, K_short, T, RISK_FREE, iv_entry)
        net_debit = (long_prem - short_prem) * (1 + SLIPPAGE_PCT)
        if net_debit < 0.05:
            return None
        max_profit_per = (K_short - K_long) - net_debit
        if max_profit_per <= 0:
            return None
        contracts = max(1, int(MAX_RISK_PER_TRADE / (net_debit * 100)))
        cost = contracts * net_debit * 100
        if cost > MAX_RISK_PER_TRADE * 1.5:
            contracts = 1
            cost = net_debit * 100
        premium = net_debit
        trade_type = 'debit_spread'

    elif strategy_type == 'bull_put_spread':
        DTE = 30
        K_short = round(S_entry, 0)  # ATM put sold
        K_long = round(S_entry * 0.95, 0)  # 5% OTM put bought
        T = DTE / 365.0
        short_put_prem = bs_put_price(S_entry, K_short, T, RISK_FREE, iv_entry)
        long_put_prem = bs_put_price(S_entry, K_long, T, RISK_FREE, iv_entry)
        net_credit = (short_put_prem - long_put_prem) * (1 - SLIPPAGE_PCT)  # slippage reduces credit
        if net_credit < 0.05:
            return None
        max_loss_per = (K_short - K_long) - net_credit
        if max_loss_per <= 0:
            return None
        contracts = max(1, int(MAX_RISK_PER_TRADE / (max_loss_per * 100)))
        cost = contracts * max_loss_per * 100  # capital at risk
        credit_received = contracts * net_credit * 100
        if cost > MAX_RISK_PER_TRADE * 1.5:
            contracts = 1
            cost = max_loss_per * 100
            credit_received = net_credit * 100
        premium = net_credit
        trade_type = 'credit_spread'

    elif strategy_type == 'leaps_call':
        DTE = 180
        # Deep ITM for ~0.70 delta
        K = find_strike_for_delta(S_entry, 0.70, DTE/365.0, RISK_FREE, iv_entry, call=True)
        K = round(K, 0)
        T = DTE / 365.0
        premium = bs_call_price(S_entry, K, T, RISK_FREE, iv_entry)
        premium *= (1 + SLIPPAGE_PCT)
        if premium < 0.50:
            return None
        contracts = max(1, int(MAX_RISK_PER_TRADE / (premium * 100)))
        cost = contracts * premium * 100
        if cost > MAX_RISK_PER_TRADE * 1.5:
            contracts = 1
            cost = premium * 100
        trade_type = 'debit'
    else:
        return None

    # Simulate day-by-day through holding period
    entry_premium = premium
    max_hold = min(MAX_HOLD_DAYS, DTE - MIN_DTE_EXIT)
    if max_hold < 1:
        return None

    exit_date = None
    exit_reason = None
    exit_pnl = 0

    for hold_day in range(1, max_hold + 1):
        if entry_idx + hold_day >= len(close):
            break

        current_date = close.index[entry_idx + hold_day]
        S_now = float(close.iloc[entry_idx + hold_day])
        days_remaining = DTE - hold_day
        T_now = days_remaining / 365.0

        # IV at exit: assume gradual IV crush as dip recovers
        pct_move = (S_now - S_entry) / S_entry
        # IV crushes proportional to recovery (max 15% crush at full recovery)
        iv_crush = min(IV_CRUSH_PCT, max(0, pct_move * 2)) * iv_entry
        iv_now = max(iv_entry - iv_crush, 0.08)

        # Calculate current option value
        if strategy_type == 'atm_call' or strategy_type == 'otm_call' or strategy_type == 'leaps_call':
            current_val = bs_call_price(S_now, K, T_now, RISK_FREE, iv_now)
            current_val *= (1 - SLIPPAGE_PCT)  # slippage on exit
            pnl_per = current_val - entry_premium
            pnl_pct = pnl_per / entry_premium if entry_premium > 0 else 0

        elif strategy_type == 'bull_call_spread':
            long_val = bs_call_price(S_now, K_long, T_now, RISK_FREE, iv_now)
            short_val = bs_call_price(S_now, K_short, T_now, RISK_FREE, iv_now)
            current_spread_val = (long_val - short_val) * (1 - SLIPPAGE_PCT)
            pnl_per = current_spread_val - entry_premium
            pnl_pct = pnl_per / entry_premium if entry_premium > 0 else 0

        elif strategy_type == 'bull_put_spread':
            short_put_val = bs_put_price(S_now, K_short, T_now, RISK_FREE, iv_now)
            long_put_val = bs_put_price(S_now, K_long, T_now, RISK_FREE, iv_now)
            current_spread_cost = (short_put_val - long_put_val) * (1 + SLIPPAGE_PCT)
            # For credit spread: profit = credit received - cost to close
            pnl_per = entry_premium - current_spread_cost
            max_loss = (K_short - K_long) - entry_premium
            pnl_pct = pnl_per / max_loss if max_loss > 0 else 0

        # Check exit conditions
        # 1. Underlying +5%
        if pct_move >= UNDERLYING_TP_PCT:
            exit_date = current_date
            exit_reason = 'underlying_tp'
            exit_pnl = pnl_per * contracts * 100
            break

        # 2. Option premium doubles (100% gain)
        if pnl_pct >= TP_PCT:
            exit_date = current_date
            exit_reason = 'premium_tp'
            exit_pnl = pnl_per * contracts * 100
            break

        # 3. Stop loss (50% loss on premium)
        if strategy_type in ('atm_call', 'otm_call', 'leaps_call', 'bull_call_spread'):
            if pnl_pct <= -SL_PCT:
                exit_date = current_date
                exit_reason = 'stop_loss'
                exit_pnl = pnl_per * contracts * 100
                break
        elif strategy_type == 'bull_put_spread':
            # For credit spread, max loss is defined
            if pnl_pct <= -0.80:  # 80% of max loss
                exit_date = current_date
                exit_reason = 'stop_loss'
                exit_pnl = pnl_per * contracts * 100
                break

        # 4. Time exit (7 DTE remaining) or max hold
        if days_remaining <= MIN_DTE_EXIT or hold_day == max_hold:
            exit_date = current_date
            exit_reason = 'time_exit'
            exit_pnl = pnl_per * contracts * 100
            break

    if exit_date is None:
        # Reached end of data
        return None

    return {
        'ticker': ticker,
        'entry_date': entry_date,
        'exit_date': exit_date,
        'entry_price': S_entry,
        'exit_price': S_now,
        'strategy_type': strategy_type,
        'contracts': contracts,
        'entry_premium': entry_premium,
        'cost': cost,
        'pnl': exit_pnl,
        'pnl_pct': exit_pnl / cost if cost > 0 else 0,
        'hold_days': hold_day,
        'exit_reason': exit_reason,
        'iv_entry': iv_entry,
    }


def simulate_share_trade(ticker, entry_date, close_series):
    """Simulate buying $300 worth of shares with same exit rules."""
    close = close_series
    entry_idx = close.index.get_loc(entry_date)
    S_entry = float(close.iloc[entry_idx])

    position_size = 300.0  # $300 per trade in shares
    shares = position_size / S_entry

    for hold_day in range(1, MAX_HOLD_DAYS + 1):
        if entry_idx + hold_day >= len(close):
            break

        S_now = float(close.iloc[entry_idx + hold_day])
        pct_move = (S_now - S_entry) / S_entry

        exit_date = close.index[entry_idx + hold_day]

        # TP at +5%
        if pct_move >= UNDERLYING_TP_PCT:
            pnl = shares * (S_now - S_entry)
            return {
                'pnl': pnl, 'pnl_pct': pnl / position_size,
                'hold_days': hold_day, 'exit_reason': 'tp',
                'entry_date': entry_date, 'exit_date': exit_date,
                'cost': position_size,
            }

        # SL at -5%
        if pct_move <= -0.05:
            pnl = shares * (S_now - S_entry)
            return {
                'pnl': pnl, 'pnl_pct': pnl / position_size,
                'hold_days': hold_day, 'exit_reason': 'sl',
                'entry_date': entry_date, 'exit_date': exit_date,
                'cost': position_size,
            }

    # Time exit
    if entry_idx + MAX_HOLD_DAYS < len(close):
        S_now = float(close.iloc[entry_idx + MAX_HOLD_DAYS])
        pnl = shares * (S_now - S_entry)
        return {
            'pnl': pnl, 'pnl_pct': pnl / position_size,
            'hold_days': MAX_HOLD_DAYS, 'exit_reason': 'time',
            'entry_date': entry_date, 'exit_date': close.index[entry_idx + MAX_HOLD_DAYS],
            'cost': position_size,
        }
    return None


# ============================================================
# RUN BACKTEST
# ============================================================

print("\n" + "="*80)
print("RUNNING BACKTEST...")
print("="*80)

OPTION_STRATEGIES = ['atm_call', 'otm_call', 'bull_call_spread', 'bull_put_spread', 'leaps_call']
OPTION_LABELS = {
    'atm_call': 'A. ATM Call (δ~0.50)',
    'otm_call': 'B. OTM Call (δ~0.30)',
    'bull_call_spread': 'C. Bull Call Spread',
    'bull_put_spread': 'D. Bull Put Spread (credit)',
    'leaps_call': 'E. LEAPS Call (δ~0.70)',
}
SIGNAL_STRATEGIES = ['IV-RV Gap', 'Bond Yield Signal', 'RSI Divergence']

# Group signals by strategy
signals_by_strat = {s: [] for s in SIGNAL_STRATEGIES}
for dt, ticker, strat in all_signals:
    signals_by_strat[strat].append((dt, ticker))

# Sort by date
for strat in signals_by_strat:
    signals_by_strat[strat].sort(key=lambda x: x[0])

results = {}  # (signal_strat, option_strat) -> list of trade dicts

for sig_strat in SIGNAL_STRATEGIES:
    sigs = signals_by_strat[sig_strat]
    print(f"\n--- {sig_strat}: {len(sigs)} signals ---")

    # Shares baseline
    share_trades = []
    for dt, ticker in sigs:
        if ticker not in stock_indicators:
            continue
        close = stock_indicators[ticker]['close']
        result = simulate_share_trade(ticker, dt, close)
        if result:
            share_trades.append(result)
    results[(sig_strat, 'shares')] = share_trades
    print(f"  Shares: {len(share_trades)} trades completed")

    # Each option strategy
    for opt_strat in OPTION_STRATEGIES:
        trades = []
        open_positions = []  # track concurrent positions

        for dt, ticker in sigs:
            if ticker not in stock_indicators:
                continue

            # Check concurrent position limit
            open_positions = [p for p in open_positions if p['exit_date'] > dt]
            if len(open_positions) >= MAX_CONCURRENT:
                continue

            close = stock_indicators[ticker]['close']
            result = simulate_option_trade(ticker, dt, opt_strat, close)
            if result:
                trades.append(result)
                open_positions.append(result)

        results[(sig_strat, opt_strat)] = trades
        print(f"  {OPTION_LABELS[opt_strat]}: {len(trades)} trades completed")


# ============================================================
# CALCULATE METRICS
# ============================================================

def calc_metrics(trades, label, starting_cap=STARTING_CAPITAL):
    """Calculate performance metrics for a list of trades."""
    if not trades:
        return None

    pnls = [t['pnl'] for t in trades]
    costs = [t['cost'] for t in trades]
    hold_days = [t['hold_days'] for t in trades]
    pnl_pcts = [t['pnl_pct'] for t in trades]

    total_pnl = sum(pnls)
    total_return_pct = total_pnl / starting_cap * 100

    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p <= 0]
    win_rate = len(winners) / len(pnls) * 100 if pnls else 0

    avg_winner = np.mean(winners) if winners else 0
    avg_loser = abs(np.mean(losers)) if losers else 1
    win_loss_ratio = avg_winner / avg_loser if avg_loser > 0 else float('inf')

    # Sharpe on per-trade returns
    if len(pnl_pcts) > 1 and np.std(pnl_pcts) > 0:
        # Annualize: assume avg 1 trade per week -> ~52 trades/year
        trades_per_year = 252 / np.mean(hold_days) if np.mean(hold_days) > 0 else 52
        sharpe = (np.mean(pnl_pcts) / np.std(pnl_pcts)) * np.sqrt(min(trades_per_year, 252))
    else:
        sharpe = 0

    # Sortino
    downside = [p for p in pnl_pcts if p < 0]
    if downside and np.std(downside) > 0:
        sortino = (np.mean(pnl_pcts) / np.std(downside)) * np.sqrt(min(trades_per_year, 252))
    else:
        sortino = sharpe * 1.5 if sharpe > 0 else 0

    # Max drawdown (equity curve)
    equity = [starting_cap]
    for p in pnls:
        equity.append(equity[-1] + p)
    equity = np.array(equity)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak * 100
    max_dd = dd.min()

    # CAGR
    if len(trades) > 0:
        first_date = min(t['entry_date'] for t in trades)
        last_date = max(t.get('exit_date', t['entry_date']) for t in trades)
        years = max((last_date - first_date).days / 365.25, 0.5)
        final_equity = starting_cap + total_pnl
        if final_equity > 0:
            cagr = (final_equity / starting_cap) ** (1/years) - 1
        else:
            cagr = -1.0
    else:
        cagr = 0
        years = 1

    return {
        'label': label,
        'n_trades': len(trades),
        'total_pnl': total_pnl,
        'total_return_pct': total_return_pct,
        'cagr': cagr * 100,
        'sharpe': sharpe,
        'sortino': sortino,
        'win_rate': win_rate,
        'win_loss_ratio': win_loss_ratio,
        'max_drawdown': max_dd,
        'avg_hold_days': np.mean(hold_days),
        'avg_pnl_pct': np.mean(pnl_pcts) * 100,
        'final_equity': starting_cap + total_pnl,
        'growth_multiple': (starting_cap + total_pnl) / starting_cap,
    }


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "="*80)
print("RESULTS: OPTIONS vs SHARES LEVERAGE BACKTEST (2020-2026)")
print(f"Starting Capital: ${STARTING_CAPITAL:,.0f} | Max Risk/Trade: ${MAX_RISK_PER_TRADE:,.0f}")
print("="*80)

all_metrics = []

for sig_strat in SIGNAL_STRATEGIES:
    print(f"\n{'='*80}")
    print(f"SIGNAL: {sig_strat}")
    print(f"{'='*80}")

    # Shares baseline
    share_m = calc_metrics(results[(sig_strat, 'shares')], f"{sig_strat} | Shares")
    if share_m:
        all_metrics.append(share_m)
        print(f"\n  SHARES BASELINE ($300/trade):")
        print(f"    Trades: {share_m['n_trades']} | Win Rate: {share_m['win_rate']:.1f}%")
        print(f"    Total Return: {share_m['total_return_pct']:.1f}% | CAGR: {share_m['cagr']:.1f}%")
        print(f"    Sharpe: {share_m['sharpe']:.2f} | Sortino: {share_m['sortino']:.2f}")
        print(f"    Max DD: {share_m['max_drawdown']:.1f}% | W/L Ratio: {share_m['win_loss_ratio']:.2f}")
        print(f"    Final Equity: ${share_m['final_equity']:.0f} ({share_m['growth_multiple']:.1f}x)")

    print(f"\n  OPTIONS STRATEGIES:")
    print(f"  {'Strategy':<30} {'Trades':>6} {'WR%':>6} {'Return%':>9} {'CAGR%':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD%':>7} {'W/L':>5} {'Final$':>8} {'Growth':>7}")
    print(f"  {'-'*29} {'-'*6} {'-'*6} {'-'*9} {'-'*7} {'-'*7} {'-'*8} {'-'*7} {'-'*5} {'-'*8} {'-'*7}")

    for opt_strat in OPTION_STRATEGIES:
        trades = results[(sig_strat, opt_strat)]
        m = calc_metrics(trades, f"{sig_strat} | {OPTION_LABELS[opt_strat]}")
        if m:
            all_metrics.append(m)
            print(f"  {OPTION_LABELS[opt_strat]:<30} {m['n_trades']:>6} {m['win_rate']:>5.1f}% {m['total_return_pct']:>8.1f}% {m['cagr']:>6.1f}% {m['sharpe']:>7.2f} {m['sortino']:>8.2f} {m['max_drawdown']:>6.1f}% {m['win_loss_ratio']:>5.2f} {m['final_equity']:>7.0f} {m['growth_multiple']:>6.1f}x")
        else:
            print(f"  {OPTION_LABELS[opt_strat]:<30} {'NO TRADES':>6}")


# ============================================================
# OVERALL RANKING
# ============================================================

print("\n" + "="*80)
print("OVERALL RANKING (by Growth Multiple)")
print("="*80)

all_metrics.sort(key=lambda x: x['growth_multiple'], reverse=True)

print(f"\n{'Rank':>4} {'Strategy':<50} {'Growth':>7} {'CAGR%':>7} {'Sharpe':>7} {'Sortino':>8} {'WR%':>6} {'MaxDD%':>7} {'Trades':>6}")
print(f"{'-'*4} {'-'*50} {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*6} {'-'*7} {'-'*6}")

for i, m in enumerate(all_metrics, 1):
    print(f"{i:>4} {m['label']:<50} {m['growth_multiple']:>6.1f}x {m['cagr']:>6.1f}% {m['sharpe']:>7.2f} {m['sortino']:>8.2f} {m['win_rate']:>5.1f}% {m['max_drawdown']:>6.1f}% {m['n_trades']:>6}")


# ============================================================
# KEY ANALYSIS
# ============================================================

print("\n" + "="*80)
print("KEY ANALYSIS: Does Options Leverage Enable 5-10x Growth?")
print("="*80)

# Find best option strategy
best = all_metrics[0] if all_metrics else None
if best:
    print(f"\n  BEST OVERALL: {best['label']}")
    print(f"    Growth: {best['growth_multiple']:.1f}x | CAGR: {best['cagr']:.1f}%")
    print(f"    Sharpe: {best['sharpe']:.2f} | Sortino: {best['sortino']:.2f}")
    print(f"    Win Rate: {best['win_rate']:.1f}% | Max DD: {best['max_drawdown']:.1f}%")

# Compare shares vs options for each signal
print("\n  SHARES vs OPTIONS COMPARISON:")
for sig_strat in SIGNAL_STRATEGIES:
    share_m = None
    best_opt = None
    for m in all_metrics:
        if m['label'].startswith(sig_strat):
            if 'Shares' in m['label']:
                share_m = m
            elif best_opt is None or m['growth_multiple'] > best_opt['growth_multiple']:
                best_opt = m

    if share_m and best_opt:
        leverage = best_opt['growth_multiple'] / share_m['growth_multiple'] if share_m['growth_multiple'] > 0 else float('inf')
        print(f"\n  {sig_strat}:")
        print(f"    Shares: {share_m['growth_multiple']:.1f}x growth, Sharpe {share_m['sharpe']:.2f}")
        print(f"    Best Option ({best_opt['label'].split('|')[1].strip()}): {best_opt['growth_multiple']:.1f}x growth, Sharpe {best_opt['sharpe']:.2f}")
        print(f"    Options Leverage Factor: {leverage:.1f}x amplification")

# 5-10x target assessment
hits_5x = [m for m in all_metrics if m['growth_multiple'] >= 5]
hits_10x = [m for m in all_metrics if m['growth_multiple'] >= 10]

print(f"\n  TARGET ASSESSMENT:")
print(f"    Strategies hitting 5x growth: {len(hits_5x)}")
print(f"    Strategies hitting 10x growth: {len(hits_10x)}")

if hits_5x:
    print(f"\n    5x+ strategies:")
    for m in hits_5x:
        print(f"      {m['label']}: {m['growth_multiple']:.1f}x (CAGR {m['cagr']:.1f}%, Sharpe {m['sharpe']:.2f}, MaxDD {m['max_drawdown']:.1f}%)")

if not hits_5x:
    print(f"\n    No single strategy hits 5x alone over the test period.")
    print(f"    Consider: combining strategies, increasing frequency, or portfolio allocation.")

# Risk-adjusted comparison
print(f"\n  RISK-ADJUSTED VIEW (Sharpe > 1.0 strategies):")
good_sharpe = [m for m in all_metrics if m['sharpe'] > 1.0]
good_sharpe.sort(key=lambda x: x['sharpe'], reverse=True)
for m in good_sharpe[:5]:
    print(f"    {m['label']}: Sharpe {m['sharpe']:.2f}, Sortino {m['sortino']:.2f}, Growth {m['growth_multiple']:.1f}x")

# Exit reason analysis
print(f"\n  EXIT REASON DISTRIBUTION:")
for sig_strat in SIGNAL_STRATEGIES:
    for opt_strat in OPTION_STRATEGIES:
        trades = results.get((sig_strat, opt_strat), [])
        if not trades:
            continue
        reasons = Counter(t['exit_reason'] for t in trades)
        label = f"{sig_strat} | {OPTION_LABELS[opt_strat]}"
        reason_str = ", ".join(f"{r}: {c}" for r, c in reasons.most_common())
        if len(trades) >= 5:
            print(f"    {label[:50]:<50} {reason_str}")

# Portfolio simulation: combine top 3
print(f"\n" + "="*80)
print("PORTFOLIO SIMULATION: Top 3 Strategies Combined")
print("="*80)

# Get top 3 by Sharpe (risk-adjusted)
top3_by_sharpe = sorted([m for m in all_metrics if 'Shares' not in m['label']],
                        key=lambda x: x['sharpe'], reverse=True)[:3]

if len(top3_by_sharpe) >= 3:
    # Combine all trades from top 3, sort by date, simulate with $750 capital
    combined_trades = []
    for m in top3_by_sharpe:
        sig = m['label'].split('|')[0].strip()
        opt = None
        for key, label in OPTION_LABELS.items():
            if label in m['label']:
                opt = key
                break
        if opt:
            combined_trades.extend(results.get((sig, opt), []))

    combined_trades.sort(key=lambda x: x['entry_date'])

    # Simple equity curve
    equity = STARTING_CAPITAL
    peak = equity
    max_dd = 0
    equity_curve = [(combined_trades[0]['entry_date'] if combined_trades else datetime.now(), equity)]

    for t in combined_trades:
        equity += t['pnl']
        equity_curve.append((t['exit_date'], equity))
        peak = max(peak, equity)
        dd = (equity - peak) / peak
        max_dd = min(max_dd, dd)

    print(f"\n  Top 3 strategies by Sharpe:")
    for m in top3_by_sharpe:
        print(f"    {m['label']}: Sharpe {m['sharpe']:.2f}")

    print(f"\n  Combined Portfolio Results:")
    print(f"    Starting Capital: ${STARTING_CAPITAL:,.0f}")
    print(f"    Final Equity: ${equity:,.0f}")
    print(f"    Growth Multiple: {equity/STARTING_CAPITAL:.1f}x")
    print(f"    Total Trades: {len(combined_trades)}")
    print(f"    Max Drawdown: {max_dd*100:.1f}%")

    target_5x = STARTING_CAPITAL * 5
    target_10x = STARTING_CAPITAL * 10
    print(f"\n    5x Target (${target_5x:,.0f}): {'ACHIEVED' if equity >= target_5x else 'NOT REACHED'}")
    print(f"    10x Target (${target_10x:,.0f}): {'ACHIEVED' if equity >= target_10x else 'NOT REACHED'}")

print("\n" + "="*80)
print("BACKTEST COMPLETE")
print("="*80)
