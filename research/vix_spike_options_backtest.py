"""
VIX Spike Mean-Reversion Backtest
=================================
Tests three approaches to profiting from VIX spikes (VIX > 30):
1. VIX Puts — buy ATM/OTM puts when VIX spikes, profit as it mean-reverts
2. SVIX Shares — buy short-VIX ETF during spikes, sell on mean-reversion
3. SPY Calls — buy slightly OTM calls during VIX spikes (buy-the-dip proxy)

Also tests staged entries: tranches at VIX>30, >35, >40.

Uses Black-Scholes for option price estimation since we lack historical chains.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# BLACK-SCHOLES OPTION PRICING
# ============================================================

def bs_call(S, K, T, r, sigma):
    """Black-Scholes call price."""
    if T <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put(S, K, T, r, sigma):
    """Black-Scholes put price."""
    if T <= 0:
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# ============================================================
# DATA DOWNLOAD
# ============================================================

def download_data():
    """Download VIX, SPY, and SVIX data."""
    print("Downloading data...")

    vix = yf.download("^VIX", start="2010-01-01", end="2026-07-17", progress=False)
    spy = yf.download("SPY", start="2010-01-01", end="2026-07-17", progress=False)
    svix = yf.download("SVIX", start="2022-03-01", end="2026-07-17", progress=False)

    # Flatten multi-level columns if present
    for df in [vix, spy, svix]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

    print(f"  VIX: {len(vix)} rows ({vix.index[0].strftime('%Y-%m-%d')} to {vix.index[-1].strftime('%Y-%m-%d')})")
    print(f"  SPY: {len(spy)} rows")
    print(f"  SVIX: {len(svix)} rows ({svix.index[0].strftime('%Y-%m-%d')} to {svix.index[-1].strftime('%Y-%m-%d')})")

    return vix, spy, svix


# ============================================================
# IDENTIFY VIX SPIKE EVENTS
# ============================================================

def find_vix_spikes(vix, threshold=30, min_gap_days=5):
    """
    Find distinct VIX spike events where VIX crosses above threshold.
    Groups consecutive days above threshold into single events.
    Returns list of (entry_date, peak_vix, peak_date) tuples.
    """
    close = vix['Close'].squeeze()
    above = close > threshold

    events = []
    in_spike = False
    entry_date = None
    peak_vix = 0
    peak_date = None
    last_exit = None

    for date, val in close.items():
        if above[date] and not in_spike:
            # Check minimum gap from last event
            if last_exit is not None and (date - last_exit).days < min_gap_days:
                continue
            in_spike = True
            entry_date = date
            peak_vix = val
            peak_date = date
        elif in_spike and above[date]:
            if val > peak_vix:
                peak_vix = val
                peak_date = date
        elif in_spike and not above[date]:
            events.append({
                'entry_date': entry_date,
                'peak_vix': peak_vix,
                'peak_date': peak_date,
                'exit_date': date,
                'entry_vix': float(close[entry_date]),
                'days_above': (date - entry_date).days
            })
            in_spike = False
            last_exit = date

    # Handle case where we're still in a spike at end of data
    if in_spike:
        events.append({
            'entry_date': entry_date,
            'peak_vix': peak_vix,
            'peak_date': peak_date,
            'exit_date': close.index[-1],
            'entry_vix': float(close[entry_date]),
            'days_above': (close.index[-1] - entry_date).days
        })

    return events


# ============================================================
# STRATEGY 1: VIX PUTS
# ============================================================

def backtest_vix_puts(vix, events, dte=37, otm_pct=0.0, max_hold=45):
    """
    Buy VIX puts when VIX > 30.
    Strike = VIX level at entry * (1 - otm_pct).
    Use BS to price with IV derived from VIX level itself.
    Exit when VIX < 20 or max_hold days.

    VIX options are European, cash-settled, based on VIX settlement value.
    IV for VIX options (VVIX) is typically 80-120%; we use 90% as default.
    """
    results = []
    close = vix['Close'].squeeze()

    for event in events:
        entry = event['entry_date']
        entry_vix = event['entry_vix']

        # Strike: ATM or slightly OTM
        strike = entry_vix * (1 - otm_pct)

        # VIX option IV (VVIX-derived): typically 80-120%
        # Higher when VIX is elevated
        vvix = 0.90 + 0.005 * (entry_vix - 20)  # Scale with VIX level
        vvix = min(max(vvix, 0.70), 1.50)

        T_entry = dte / 365.0
        r = 0.04  # risk-free rate

        # Entry price
        entry_price = bs_put(entry_vix, strike, T_entry, r, vvix)
        if entry_price < 0.10:
            continue  # Skip if option too cheap

        # Find exit
        exit_price = None
        exit_date = None
        exit_reason = None

        future_dates = close.loc[entry:].index[1:]  # Skip entry day
        for i, date in enumerate(future_dates):
            days_held = (date - entry).days
            T_remaining = max((dte - days_held) / 365.0, 0.001)
            current_vix = float(close[date])

            # Update IV estimate
            current_vvix = 0.90 + 0.005 * (current_vix - 20)
            current_vvix = min(max(current_vvix, 0.70), 1.50)

            if current_vix < 20 or days_held >= max_hold:
                exit_price = bs_put(current_vix, strike, T_remaining, r, current_vvix)
                exit_date = date
                exit_reason = 'vix_below_20' if current_vix < 20 else 'max_hold'
                break

        if exit_price is None:
            # Use last available date
            last_date = future_dates[-1] if len(future_dates) > 0 else entry
            days_held = (last_date - entry).days
            T_remaining = max((dte - days_held) / 365.0, 0.001)
            current_vix = float(close[last_date])
            current_vvix = 0.90 + 0.005 * (current_vix - 20)
            current_vvix = min(max(current_vvix, 0.70), 1.50)
            exit_price = bs_put(current_vix, strike, T_remaining, r, current_vvix)
            exit_date = last_date
            exit_reason = 'end_of_data'

        pnl_per_contract = (exit_price - entry_price) * 100  # VIX options: $100 multiplier
        commission = 2.60  # RT commission estimate
        net_pnl = pnl_per_contract - commission
        ret = (exit_price - entry_price) / entry_price

        results.append({
            'entry_date': entry,
            'exit_date': exit_date,
            'entry_vix': entry_vix,
            'exit_vix': float(close[exit_date]) if exit_date in close.index else np.nan,
            'strike': strike,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl_per_contract': net_pnl,
            'return': ret,
            'days_held': (exit_date - entry).days,
            'exit_reason': exit_reason,
            'peak_vix': event['peak_vix']
        })

    return pd.DataFrame(results)


# ============================================================
# STRATEGY 2: SVIX SHARES
# ============================================================

def backtest_svix(vix, svix, events, exit_vix=20, max_hold=30):
    """
    Buy SVIX when VIX > 30, sell when VIX < exit_vix or max_hold days.
    Only events after SVIX launch (2022-03-22).
    """
    results = []
    vix_close = vix['Close'].squeeze()
    svix_close = svix['Close'].squeeze()
    svix_start = svix_close.index[0]

    for event in events:
        entry = event['entry_date']
        if entry < svix_start:
            continue

        # Find nearest trading day in SVIX
        if entry not in svix_close.index:
            future = svix_close.loc[entry:].index
            if len(future) == 0:
                continue
            entry = future[0]

        entry_price = float(svix_close[entry])
        entry_vix = float(vix_close[entry]) if entry in vix_close.index else event['entry_vix']

        # Find exit
        exit_price = None
        exit_date = None
        exit_reason = None

        future_dates = svix_close.loc[entry:].index[1:]
        for date in future_dates:
            days_held = (date - entry).days
            current_vix = float(vix_close[date]) if date in vix_close.index else 25

            if current_vix < exit_vix or days_held >= max_hold:
                exit_price = float(svix_close[date])
                exit_date = date
                exit_reason = 'vix_below_20' if current_vix < exit_vix else 'max_hold'
                break

        if exit_price is None:
            last = future_dates[-1] if len(future_dates) > 0 else entry
            exit_price = float(svix_close[last])
            exit_date = last
            exit_reason = 'end_of_data'

        # SVIX: assume $10k position size
        shares = int(10000 / entry_price)
        gross_pnl = shares * (exit_price - entry_price)
        commission = 2.0  # minimal for shares
        net_pnl = gross_pnl - commission
        ret = (exit_price - entry_price) / entry_price

        results.append({
            'entry_date': entry,
            'exit_date': exit_date,
            'entry_vix': entry_vix,
            'exit_vix': float(vix_close[exit_date]) if exit_date in vix_close.index else np.nan,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'shares': shares,
            'pnl': net_pnl,
            'return': ret,
            'days_held': (exit_date - entry).days,
            'exit_reason': exit_reason,
            'peak_vix': event['peak_vix']
        })

    return pd.DataFrame(results)


# ============================================================
# STRATEGY 3: SPY CALLS
# ============================================================

def backtest_spy_calls(vix, spy, events, dte=37, otm_pct=0.02, max_hold=45):
    """
    Buy SPY calls (2% OTM, 37 DTE) when VIX > 30.
    Exit when VIX < 20 or max_hold days.
    IV = VIX level / 100 (VIX IS the implied vol of SPY options).
    """
    results = []
    vix_close = vix['Close'].squeeze()
    spy_close = spy['Close'].squeeze()

    for event in events:
        entry = event['entry_date']

        if entry not in spy_close.index:
            future = spy_close.loc[entry:].index
            if len(future) == 0:
                continue
            entry = future[0]

        entry_spy = float(spy_close[entry])
        entry_vix = float(vix_close[entry]) if entry in vix_close.index else event['entry_vix']

        # Strike: slightly OTM call
        strike = entry_spy * (1 + otm_pct)

        # IV = VIX / 100 (VIX is annualized implied vol of SPY)
        iv_entry = entry_vix / 100.0
        T_entry = dte / 365.0
        r = 0.04

        entry_price = bs_call(entry_spy, strike, T_entry, r, iv_entry)
        if entry_price < 0.10:
            continue

        # Find exit
        exit_price = None
        exit_date = None
        exit_reason = None

        future_dates = spy_close.loc[entry:].index[1:]
        for date in future_dates:
            if date not in vix_close.index:
                continue
            days_held = (date - entry).days
            T_remaining = max((dte - days_held) / 365.0, 0.001)
            current_spy = float(spy_close[date])
            current_vix = float(vix_close[date])
            iv_current = current_vix / 100.0

            if current_vix < 20 or days_held >= max_hold:
                exit_price = bs_call(current_spy, strike, T_remaining, r, iv_current)
                exit_date = date
                exit_reason = 'vix_below_20' if current_vix < 20 else 'max_hold'
                break

        if exit_price is None:
            last = future_dates[-1] if len(future_dates) > 0 else entry
            days_held = (last - entry).days
            T_remaining = max((dte - days_held) / 365.0, 0.001)
            current_spy = float(spy_close[last])
            current_vix = float(vix_close[last]) if last in vix_close.index else 20
            iv_current = current_vix / 100.0
            exit_price = bs_call(current_spy, strike, T_remaining, r, iv_current)
            exit_date = last
            exit_reason = 'end_of_data'

        # SPY options: $100 multiplier, assume 1 contract
        pnl_per_contract = (exit_price - entry_price) * 100
        commission = 1.30  # RT commission
        net_pnl = pnl_per_contract - commission
        ret = (exit_price - entry_price) / entry_price

        results.append({
            'entry_date': entry,
            'exit_date': exit_date,
            'entry_vix': entry_vix,
            'exit_vix': float(vix_close[exit_date]) if exit_date in vix_close.index else np.nan,
            'entry_spy': entry_spy,
            'exit_spy': float(spy_close[exit_date]) if exit_date in spy_close.index else np.nan,
            'strike': strike,
            'iv_entry': iv_entry,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl_per_contract': net_pnl,
            'return': ret,
            'days_held': (exit_date - entry).days,
            'exit_reason': exit_reason,
            'peak_vix': event['peak_vix']
        })

    return pd.DataFrame(results)


# ============================================================
# STAGED ENTRY BACKTEST
# ============================================================

def backtest_staged_entries(vix, spy, events, dte=37, otm_pct=0.02, max_hold=45):
    """
    Staged entry SPY calls:
    - Tranche 1: VIX > 30 (1 contract)
    - Tranche 2: VIX > 35 (1 more contract, deeper OTM since SPY lower)
    - Tranche 3: VIX > 40 (1 more contract)
    Exit all when VIX < 20 or max_hold from first entry.
    """
    results = []
    vix_close = vix['Close'].squeeze()
    spy_close = spy['Close'].squeeze()
    r = 0.04

    for event in events:
        entry = event['entry_date']
        if entry not in spy_close.index or entry not in vix_close.index:
            continue

        tranches = []
        total_cost = 0

        # Walk through the spike to add tranches
        future_dates = vix_close.loc[entry:].index
        t1_added = False
        t2_added = False
        t3_added = False
        first_entry = None

        for date in future_dates:
            if date not in spy_close.index:
                continue
            cv = float(vix_close[date])
            cs = float(spy_close[date])

            if cv > 30 and not t1_added:
                strike = cs * (1 + otm_pct)
                iv = cv / 100.0
                price = bs_call(cs, strike, dte / 365.0, r, iv)
                if price >= 0.10:
                    tranches.append({'date': date, 'strike': strike, 'price': price, 'spy': cs, 'vix': cv, 'tranche': 1})
                    total_cost += price * 100
                    t1_added = True
                    first_entry = date

            if cv > 35 and t1_added and not t2_added:
                strike = cs * (1 + otm_pct)
                iv = cv / 100.0
                price = bs_call(cs, strike, dte / 365.0, r, iv)
                if price >= 0.10:
                    tranches.append({'date': date, 'strike': strike, 'price': price, 'spy': cs, 'vix': cv, 'tranche': 2})
                    total_cost += price * 100
                    t2_added = True

            if cv > 40 and t1_added and not t3_added:
                strike = cs * (1 + otm_pct)
                iv = cv / 100.0
                price = bs_call(cs, strike, dte / 365.0, r, iv)
                if price >= 0.10:
                    tranches.append({'date': date, 'strike': strike, 'price': price, 'spy': cs, 'vix': cv, 'tranche': 3})
                    total_cost += price * 100
                    t3_added = True

            # Check exit conditions
            if first_entry is not None:
                days_held = (date - first_entry).days
                if (cv < 20 and days_held > 1) or days_held >= max_hold:
                    # Exit all tranches
                    total_exit_value = 0
                    for t in tranches:
                        t_days = (date - t['date']).days
                        T_rem = max((dte - t_days) / 365.0, 0.001)
                        iv_exit = cv / 100.0
                        exit_p = bs_call(cs, t['strike'], T_rem, r, iv_exit)
                        total_exit_value += exit_p * 100
                        t['exit_price'] = exit_p
                        t['exit_spy'] = cs

                    commission = len(tranches) * 1.30
                    net_pnl = total_exit_value - total_cost - commission

                    results.append({
                        'entry_date': first_entry,
                        'exit_date': date,
                        'num_tranches': len(tranches),
                        'entry_vix': event['entry_vix'],
                        'exit_vix': cv,
                        'peak_vix': event['peak_vix'],
                        'total_cost': total_cost,
                        'total_exit_value': total_exit_value,
                        'net_pnl': net_pnl,
                        'return': (total_exit_value - total_cost) / total_cost if total_cost > 0 else 0,
                        'days_held': days_held,
                        'exit_reason': 'vix_below_20' if cv < 20 else 'max_hold',
                        'tranches_detail': tranches
                    })
                    break

        # If no exit found within loop
        if first_entry is not None and (len(results) == 0 or results[-1]['entry_date'] != first_entry):
            last = future_dates[-1]
            if last in spy_close.index and last in vix_close.index:
                cs = float(spy_close[last])
                cv = float(vix_close[last])
                total_exit_value = 0
                for t in tranches:
                    t_days = (last - t['date']).days
                    T_rem = max((dte - t_days) / 365.0, 0.001)
                    exit_p = bs_call(cs, t['strike'], T_rem, r, cv / 100.0)
                    total_exit_value += exit_p * 100

                commission = len(tranches) * 1.30
                net_pnl = total_exit_value - total_cost - commission
                results.append({
                    'entry_date': first_entry,
                    'exit_date': last,
                    'num_tranches': len(tranches),
                    'entry_vix': event['entry_vix'],
                    'exit_vix': cv,
                    'peak_vix': event['peak_vix'],
                    'total_cost': total_cost,
                    'total_exit_value': total_exit_value,
                    'net_pnl': net_pnl,
                    'return': (total_exit_value - total_cost) / total_cost if total_cost > 0 else 0,
                    'days_held': (last - first_entry).days,
                    'exit_reason': 'end_of_data',
                    'tranches_detail': tranches
                })

    return pd.DataFrame(results)


# ============================================================
# METRICS CALCULATION
# ============================================================

def calc_metrics(df, pnl_col='return', name='Strategy'):
    """Calculate strategy metrics."""
    if len(df) == 0:
        return {'name': name, 'n_events': 0}

    returns = df[pnl_col].values
    n = len(returns)
    wins = np.sum(returns > 0)

    # Annualized Sharpe (assume ~4 events/year average)
    events_per_year = max(n / ((df['exit_date'].max() - df['entry_date'].min()).days / 365.25), 1)
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1) if n > 1 else 1
    sharpe = (mean_ret / std_ret) * np.sqrt(events_per_year) if std_ret > 0 else 0

    # Sortino
    downside = returns[returns < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else std_ret
    sortino = (mean_ret / downside_std) * np.sqrt(events_per_year) if downside_std > 0 else 0

    # Profit factor
    gross_wins = np.sum(returns[returns > 0])
    gross_losses = abs(np.sum(returns[returns < 0]))
    pf = gross_wins / gross_losses if gross_losses > 0 else np.inf

    return {
        'name': name,
        'n_events': n,
        'win_rate': wins / n * 100,
        'avg_return': mean_ret * 100,
        'median_return': np.median(returns) * 100,
        'best_return': np.max(returns) * 100,
        'worst_return': np.min(returns) * 100,
        'sharpe': sharpe,
        'sortino': sortino,
        'profit_factor': pf,
        'avg_days_held': df['days_held'].mean(),
        'total_return': np.sum(returns) * 100,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("VIX SPIKE MEAN-REVERSION BACKTEST")
    print("=" * 80)
    print()

    # Download data
    vix, spy, svix = download_data()
    print()

    # Find VIX spike events
    events = find_vix_spikes(vix, threshold=30, min_gap_days=5)
    print(f"Found {len(events)} VIX spike events (VIX > 30) since 2010:")
    print("-" * 90)
    print(f"{'#':<4} {'Entry Date':<14} {'Entry VIX':>10} {'Peak VIX':>10} {'Peak Date':<14} {'Days Above':>12}")
    print("-" * 90)
    for i, e in enumerate(events):
        print(f"{i+1:<4} {e['entry_date'].strftime('%Y-%m-%d'):<14} {e['entry_vix']:>10.1f} {e['peak_vix']:>10.1f} {e['peak_date'].strftime('%Y-%m-%d'):<14} {e['days_above']:>12}")
    print()

    # ---- Strategy 1: VIX Puts ----
    print("=" * 80)
    print("STRATEGY 1: VIX PUTS (ATM, 37 DTE)")
    print("=" * 80)
    vix_puts_df = backtest_vix_puts(vix, events, dte=37, otm_pct=0.0, max_hold=45)
    if len(vix_puts_df) > 0:
        print(f"\n{'Entry Date':<14} {'VIX In':>8} {'VIX Out':>8} {'Strike':>8} {'Entry$':>8} {'Exit$':>8} {'P&L/K':>10} {'Return':>8} {'Days':>5} {'Exit Reason':<12}")
        print("-" * 110)
        for _, row in vix_puts_df.iterrows():
            print(f"{row['entry_date'].strftime('%Y-%m-%d'):<14} {row['entry_vix']:>8.1f} {row['exit_vix']:>8.1f} {row['strike']:>8.1f} "
                  f"{row['entry_price']:>8.2f} {row['exit_price']:>8.2f} {row['pnl_per_contract']:>10.0f} {row['return']*100:>7.1f}% {row['days_held']:>5} {row['exit_reason']:<12}")
    print()

    # ---- Strategy 2: SVIX Shares ----
    print("=" * 80)
    print("STRATEGY 2: SVIX SHARES ($10k position)")
    print("=" * 80)
    svix_df = backtest_svix(vix, svix, events, exit_vix=20, max_hold=30)
    if len(svix_df) > 0:
        print(f"\n{'Entry Date':<14} {'VIX In':>8} {'VIX Out':>8} {'SVIX In':>8} {'SVIX Out':>9} {'Shares':>7} {'P&L$':>9} {'Return':>8} {'Days':>5} {'Exit Reason':<12}")
        print("-" * 110)
        for _, row in svix_df.iterrows():
            print(f"{row['entry_date'].strftime('%Y-%m-%d'):<14} {row['entry_vix']:>8.1f} {row['exit_vix']:>8.1f} {row['entry_price']:>8.2f} "
                  f"{row['exit_price']:>9.2f} {row['shares']:>7} {row['pnl']:>9.0f} {row['return']*100:>7.1f}% {row['days_held']:>5} {row['exit_reason']:<12}")
    print()

    # ---- Strategy 3: SPY Calls ----
    print("=" * 80)
    print("STRATEGY 3: SPY CALLS (2% OTM, 37 DTE)")
    print("=" * 80)
    spy_calls_df = backtest_spy_calls(vix, spy, events, dte=37, otm_pct=0.02, max_hold=45)
    if len(spy_calls_df) > 0:
        print(f"\n{'Entry Date':<14} {'VIX In':>8} {'SPY In':>8} {'Strike':>8} {'IV':>6} {'Entry$':>8} {'Exit$':>8} {'P&L/K':>10} {'Return':>8} {'Days':>5}")
        print("-" * 110)
        for _, row in spy_calls_df.iterrows():
            print(f"{row['entry_date'].strftime('%Y-%m-%d'):<14} {row['entry_vix']:>8.1f} {row['entry_spy']:>8.1f} {row['strike']:>8.1f} "
                  f"{row['iv_entry']:>5.0%} {row['entry_price']:>8.2f} {row['exit_price']:>8.2f} {row['pnl_per_contract']:>10.0f} {row['return']*100:>7.1f}% {row['days_held']:>5}")
    print()

    # ---- Strategy 4: Staged Entries ----
    print("=" * 80)
    print("STRATEGY 4: STAGED SPY CALLS (Tranches at VIX>30, >35, >40)")
    print("=" * 80)
    staged_df = backtest_staged_entries(vix, spy, events, dte=37, otm_pct=0.02, max_hold=45)
    if len(staged_df) > 0:
        print(f"\n{'Entry Date':<14} {'VIX In':>8} {'Peak VIX':>9} {'#Tranch':>8} {'Cost$':>9} {'Exit$':>9} {'P&L$':>9} {'Return':>8} {'Days':>5}")
        print("-" * 100)
        for _, row in staged_df.iterrows():
            print(f"{row['entry_date'].strftime('%Y-%m-%d'):<14} {row['entry_vix']:>8.1f} {row['peak_vix']:>9.1f} {row['num_tranches']:>8} "
                  f"{row['total_cost']:>9.0f} {row['total_exit_value']:>9.0f} {row['net_pnl']:>9.0f} {row['return']*100:>7.1f}% {row['days_held']:>5}")
    print()

    # ============================================================
    # SUMMARY COMPARISON
    # ============================================================
    print("=" * 80)
    print("STRATEGY COMPARISON SUMMARY")
    print("=" * 80)

    metrics = []
    metrics.append(calc_metrics(vix_puts_df, 'return', 'VIX Puts (ATM, 37DTE)'))
    if len(svix_df) > 0:
        metrics.append(calc_metrics(svix_df, 'return', 'SVIX Shares ($10k)'))
    metrics.append(calc_metrics(spy_calls_df, 'return', 'SPY Calls (2% OTM)'))
    metrics.append(calc_metrics(staged_df, 'return', 'Staged SPY Calls'))

    print(f"\n{'Metric':<25}", end="")
    for m in metrics:
        print(f"{m['name']:>22}", end="")
    print()
    print("-" * (25 + 22 * len(metrics)))

    rows = [
        ('Events', 'n_events', '{:.0f}'),
        ('Win Rate %', 'win_rate', '{:.1f}%'),
        ('Avg Return %', 'avg_return', '{:.1f}%'),
        ('Median Return %', 'median_return', '{:.1f}%'),
        ('Best Return %', 'best_return', '{:.1f}%'),
        ('Worst Return %', 'worst_return', '{:.1f}%'),
        ('Sharpe (ann.)', 'sharpe', '{:.2f}'),
        ('Sortino (ann.)', 'sortino', '{:.2f}'),
        ('Profit Factor', 'profit_factor', '{:.2f}'),
        ('Avg Days Held', 'avg_days_held', '{:.1f}'),
        ('Total Return %', 'total_return', '{:.1f}%'),
    ]

    for label, key, fmt in rows:
        print(f"{label:<25}", end="")
        for m in metrics:
            val = m.get(key, 0)
            if val == np.inf:
                print(f"{'inf':>22}", end="")
            else:
                print(f"{fmt.format(val):>22}", end="")
        print()

    print()
    print("=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    # Find best strategy by Sharpe
    best = max(metrics, key=lambda x: x.get('sharpe', 0))
    worst = min(metrics, key=lambda x: x.get('sharpe', float('inf')) if x.get('n_events', 0) > 0 else float('inf'))

    print(f"\n  Best risk-adjusted strategy: {best['name']}")
    print(f"    Sharpe: {best.get('sharpe', 0):.2f}, Win Rate: {best.get('win_rate', 0):.0f}%, Avg Return: {best.get('avg_return', 0):.1f}%")
    print()

    # VIX put nuance
    if len(vix_puts_df) > 0:
        print("  VIX Puts insight:")
        print(f"    - VIX puts LOSE when VIX drops because you're betting ON the drop (put = short VIX)")
        print(f"    - Actually: VIX put = you profit if VIX FALLS below strike. This IS the mean-reversion bet.")
        print(f"    - Key risk: time decay (theta) eats premium while waiting for mean-reversion")
        vp_wins = vix_puts_df[vix_puts_df['return'] > 0]
        vp_losses = vix_puts_df[vix_puts_df['return'] <= 0]
        if len(vp_wins) > 0:
            print(f"    - Winning trades avg VIX drop: {(vp_wins['entry_vix'] - vp_wins['exit_vix']).mean():.1f} pts in {vp_wins['days_held'].mean():.0f} days")
        if len(vp_losses) > 0:
            print(f"    - Losing trades avg VIX drop: {(vp_losses['entry_vix'] - vp_losses['exit_vix']).mean():.1f} pts in {vp_losses['days_held'].mean():.0f} days")

    print()
    if len(spy_calls_df) > 0:
        print("  SPY Calls insight:")
        print(f"    - Benefits from BOTH SPY recovery AND IV crush as VIX normalizes")
        print(f"    - IV crush helps because you're long calls bought at high IV, which become cheaper to replace")
        print(f"    - Wait: IV crush HURTS long calls (you bought expensive, value drops). Offset by SPY rally.")
        spy_wins = spy_calls_df[spy_calls_df['return'] > 0]
        if len(spy_wins) > 0:
            print(f"    - Winners: SPY rallied avg {((spy_wins['exit_spy'] / spy_wins['entry_spy'] - 1) * 100).mean():.1f}% "
                  f"while VIX dropped {(spy_wins['entry_vix'] - spy_wins['exit_vix']).mean():.1f} pts")

    print()
    if len(staged_df) > 0:
        multi = staged_df[staged_df['num_tranches'] > 1]
        single = staged_df[staged_df['num_tranches'] == 1]
        print(f"  Staged entry insight:")
        print(f"    - {len(multi)} events had multiple tranches (VIX went >35)")
        print(f"    - {len(single)} events only triggered tranche 1 (VIX stayed 30-35)")
        if len(multi) > 0:
            print(f"    - Multi-tranche avg return: {multi['return'].mean()*100:.1f}%")
        if len(single) > 0:
            print(f"    - Single-tranche avg return: {single['return'].mean()*100:.1f}%")

    print()
    print("  CAVEATS:")
    print("  - Option prices are BS-estimated, not actual historical prices")
    print("  - VIX options use European-style cash settlement; model is approximate")
    print("  - No bid-ask spread modeled on options (would reduce returns 5-15%)")
    print("  - SVIX has tracking error and daily rebalancing drag not fully captured")
    print("  - Small sample size (VIX>30 events are rare, ~1-3x/year)")
    print("  - Survivorship/look-ahead bias: we know VIX eventually mean-reverts")
    print()


if __name__ == "__main__":
    main()
