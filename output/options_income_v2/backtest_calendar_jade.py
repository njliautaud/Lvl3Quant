"""
Options Income Strategies v2 — PMCC + Jade Lizard
===================================================
Two creative options income strategies backtested 2015-2026.

Strategy 1: Poor Man's Covered Call (PMCC) on SPY/QQQ
  - Buy deep ITM LEAPS call (delta ~0.80+, 12-18 month expiry)
  - Sell monthly OTM calls against it (~0.30 delta, 30-45 DTE)
  - Lower capital than covered calls (~30-40% of underlying)
  - Roll short call at 75% profit or 7 DTE
  - Roll LEAPS when <6 months remaining

Strategy 2: Jade Lizard (high-IV individual stocks)
  - Sell OTM put + sell OTM call spread (short call + long higher call)
  - Credit received > width of call spread => eliminates upside risk
  - Enter on stocks with IV rank > 50th percentile
  - ~45 DTE cycles, manage at 50% profit or 7 DTE

Anti-lookahead: all signals T-1, BS pricing with bid/ask haircuts, walk-forward only.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from datetime import datetime, timedelta
import json
import os
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/options_income_v2'
os.makedirs(OUTPUT_DIR, exist_ok=True)

COMMISSION_PER_CONTRACT = 0.65
SPY_HAIRCUT = 0.02    # SPY/QQQ very liquid
STOCK_HAIRCUT = 0.08  # individual stock options
RISK_FREE_RATE = 0.04
NUM_CONTRACTS = 1

# =============================================================================
# Black-Scholes
# =============================================================================

def bs_price(S, K, T, r, sigma, option_type='call'):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0) if option_type == 'call' else max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_delta(S, K, T, r, sigma, option_type='call'):
    if T <= 0 or sigma <= 0:
        if option_type == 'call':
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1


def sell_price(mid, haircut=SPY_HAIRCUT):
    return mid * (1 - haircut)


def buy_price(mid, haircut=SPY_HAIRCUT):
    return mid * (1 + haircut)


# =============================================================================
# Data
# =============================================================================

def download_data():
    print("Downloading price data...")

    tickers_etf = {'SPY': None}
    for t in tickers_etf:
        df = yf.download(t, start='2014-01-01', end='2026-07-20', progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = df.index.tz_localize(None) if df.index.tz else df.index
        tickers_etf[t] = df
        print(f"  {t}: {len(df)} days")

    vix = yf.download('^VIX', start='2014-01-01', end='2026-07-20', progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    vix.index = vix.index.tz_localize(None) if vix.index.tz else vix.index
    print(f"  VIX: {len(vix)} days")

    jade_tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA', 'AMD', 'NFLX', 'CRM']
    stock_data = {}
    for ticker in jade_tickers:
        try:
            df = yf.download(ticker, start='2014-01-01', end='2026-07-20', progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = df.index.tz_localize(None) if df.index.tz else df.index
            if len(df) > 500:
                stock_data[ticker] = df
                print(f"  {ticker}: {len(df)} days")
        except Exception as e:
            print(f"  {ticker}: FAILED ({e})")

    return tickers_etf, vix, stock_data


def compute_historical_vol(prices, window=21):
    log_returns = np.log(prices / prices.shift(1))
    hvol = log_returns.rolling(window=window).std() * np.sqrt(252)
    return hvol.shift(1)


def compute_iv_rank(hvol_series, lookback=252):
    def pctile(x):
        if len(x) < 20:
            return 50.0
        current = x.iloc[-1]
        past = x.iloc[:-1]
        return (past < current).sum() / len(past) * 100
    return hvol_series.rolling(window=lookback + 1).apply(pctile, raw=False)


def find_monthly_expiry(date, months_ahead=1):
    target_month = date.month + months_ahead
    target_year = date.year
    while target_month > 12:
        target_month -= 12
        target_year += 1
    first_day = datetime(target_year, target_month, 1)
    first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
    return first_friday + timedelta(days=14)


# =============================================================================
# Strategy 1: Poor Man's Covered Call (PMCC)
# =============================================================================

def backtest_pmcc(etf_data, vix):
    """
    PMCC on SPY and QQQ.

    Structure:
    - Buy deep ITM LEAPS call (~0.80 delta, ~12-month expiry)
    - Sell monthly OTM short call (~0.30 delta, ~30-45 DTE)

    The LEAPS acts as a stock substitute at ~30-40% of the cost.
    The short call generates income like a covered call.

    Profit: short call premium collected when stock stays below short strike.
    Loss: if stock drops significantly (LEAPS loses value) or rockets past short strike.

    Rules:
    - Roll short call at 75% profit or 7 DTE
    - Roll LEAPS when <6 months to expiry (buy new 12-month, sell old)
    - Only enter short calls when 200-SMA trend is bullish (T-1)
    - No short call when VIX > 35 (T-1) — volatility too high, risk of assignment

    Capital: LEAPS cost as base. Income = short call premiums collected.
    """
    print("\n" + "="*70)
    print("STRATEGY 1: POOR MAN'S COVERED CALL (PMCC)")
    print("="*70)

    all_trades = []

    for etf_name, df in etf_data.items():
        print(f"\n  Processing {etf_name}...")
        close = df['Close'].copy()

        # 200-SMA for trend filter (T-1)
        sma200 = close.rolling(200).mean().shift(1)

        # HV for IV proxy
        hv = compute_historical_vol(close, window=21)

        # VIX as IV proxy for SPY (T-1)
        vix_c = vix['Close'].reindex(close.index, method='ffill').shift(1)

        start_date = pd.Timestamp('2015-01-05')
        dates = close.index[close.index >= start_date]

        # State tracking
        leaps_position = None  # {strike, expiry, entry_price, cost_basis}
        short_call = None       # {strike, expiry, premium_received}
        leaps_total_cost = 0
        total_premium_collected = 0
        total_commissions = 0
        short_call_trades = []

        for i, date in enumerate(dates):
            S = float(close.loc[date])
            cur_hv = float(hv.loc[date]) if date in hv.index and not pd.isna(hv.loc[date]) else 0.16
            cur_vix = float(vix_c.loc[date]) if date in vix_c.index and not pd.isna(vix_c.loc[date]) else 16
            cur_sma = float(sma200.loc[date]) if date in sma200.index and not pd.isna(sma200.loc[date]) else None

            if cur_sma is None:
                continue

            # IV proxy
            if etf_name == 'SPY':
                iv = max(cur_vix / 100, 0.10)
            else:
                iv = max(cur_hv * 1.05, 0.12)

            # ---- LEAPS Management ----
            if leaps_position is None:
                # Buy initial LEAPS — deep ITM call, ~12 months out, ~0.80 delta
                leaps_exp = find_monthly_expiry(date, 12)
                leaps_dte = (leaps_exp - date).days
                T_leaps = leaps_dte / 365.0

                # Find strike for ~0.80 delta
                for pct in np.arange(0.05, 0.30, 0.01):
                    k = round(S * (1 - pct), 0)
                    d = bs_delta(S, k, T_leaps, RISK_FREE_RATE, iv, 'call')
                    if d <= 0.82:
                        leaps_strike = k
                        break
                else:
                    leaps_strike = round(S * 0.80, 0)

                leaps_mid = bs_price(S, leaps_strike, T_leaps, RISK_FREE_RATE, iv, 'call')
                leaps_cost = buy_price(leaps_mid, SPY_HAIRCUT)
                leaps_cost += COMMISSION_PER_CONTRACT / 100

                leaps_position = {
                    'strike': leaps_strike,
                    'expiry': leaps_exp,
                    'entry_date': date,
                    'entry_S': S,
                    'cost_basis': leaps_cost,
                    'iv_at_entry': iv,
                }
                leaps_total_cost += leaps_cost * 100
                total_commissions += COMMISSION_PER_CONTRACT

            else:
                # Check if LEAPS needs rolling (< 6 months to expiry)
                leaps_dte = (leaps_position['expiry'] - date).days
                if leaps_dte < 180:
                    # Roll: sell current LEAPS, buy new one
                    T_old = max(leaps_dte / 365.0, 0.001)
                    old_mid = bs_price(S, leaps_position['strike'], T_old, RISK_FREE_RATE, iv, 'call')
                    old_sell = sell_price(old_mid, SPY_HAIRCUT)

                    # New LEAPS
                    new_exp = find_monthly_expiry(date, 12)
                    new_dte = (new_exp - date).days
                    T_new = new_dte / 365.0

                    for pct in np.arange(0.05, 0.30, 0.01):
                        k = round(S * (1 - pct), 0)
                        d = bs_delta(S, k, T_new, RISK_FREE_RATE, iv, 'call')
                        if d <= 0.82:
                            new_strike = k
                            break
                    else:
                        new_strike = round(S * 0.80, 0)

                    new_mid = bs_price(S, new_strike, T_new, RISK_FREE_RATE, iv, 'call')
                    new_buy = buy_price(new_mid, SPY_HAIRCUT)

                    roll_cost = (new_buy - old_sell) * 100
                    leaps_total_cost += roll_cost
                    total_commissions += 2 * COMMISSION_PER_CONTRACT

                    leaps_position = {
                        'strike': new_strike,
                        'expiry': new_exp,
                        'entry_date': date,
                        'entry_S': S,
                        'cost_basis': new_buy,
                        'iv_at_entry': iv,
                    }

            # ---- Short Call Management ----
            if short_call is not None:
                sc_dte = (short_call['expiry'] - date).days

                if sc_dte <= 0:
                    # Expired
                    intrinsic = max(S - short_call['strike'], 0)
                    pnl = (short_call['premium'] - intrinsic) * 100
                    pnl -= COMMISSION_PER_CONTRACT
                    total_commissions += COMMISSION_PER_CONTRACT

                    short_call_trades.append({
                        'ticker': etf_name,
                        'entry_date': short_call['entry_date'],
                        'exit_date': date,
                        'entry_price': short_call['entry_S'],
                        'exit_price': S,
                        'strike': short_call['strike'],
                        'premium': short_call['premium'],
                        'pnl': pnl,
                        'exit_reason': 'expiration',
                        'holding_days': (date - short_call['entry_date']).days,
                    })
                    total_premium_collected += pnl
                    short_call = None
                else:
                    T_sc = sc_dte / 365.0
                    sc_mid = bs_price(S, short_call['strike'], T_sc, RISK_FREE_RATE, iv, 'call')
                    buyback_cost = buy_price(sc_mid, SPY_HAIRCUT)

                    unrealized = short_call['premium'] - sc_mid  # mid-to-mid
                    target = short_call['premium'] * 0.75  # 75% of premium

                    exit_reason = None
                    if unrealized >= target:
                        exit_reason = 'profit_target'
                    elif sc_dte <= 7:
                        exit_reason = 'dte_roll'
                    # Stop if short call goes 200% against
                    elif (buyback_cost - short_call['premium']) > short_call['premium'] * 2:
                        exit_reason = 'stop_loss'

                    if exit_reason:
                        pnl = (short_call['premium'] - buyback_cost) * 100
                        pnl -= COMMISSION_PER_CONTRACT
                        total_commissions += COMMISSION_PER_CONTRACT

                        short_call_trades.append({
                            'ticker': etf_name,
                            'entry_date': short_call['entry_date'],
                            'exit_date': date,
                            'entry_price': short_call['entry_S'],
                            'exit_price': S,
                            'strike': short_call['strike'],
                            'premium': short_call['premium'],
                            'pnl': pnl,
                            'exit_reason': exit_reason,
                            'holding_days': (date - short_call['entry_date']).days,
                        })
                        total_premium_collected += pnl
                        short_call = None

            # ---- Sell new short call ----
            if short_call is None and leaps_position is not None:
                # Trend filter: above 200-SMA
                # VIX filter: < 35 (not crisis)
                if S > cur_sma and cur_vix < 35:
                    # Find ~30 DTE expiry
                    sc_exp = find_monthly_expiry(date, 1)
                    sc_dte = (sc_exp - date).days
                    if sc_dte < 20:
                        sc_exp = find_monthly_expiry(date, 2)
                        sc_dte = (sc_exp - date).days

                    T_sc = sc_dte / 365.0

                    # ~0.30 delta short call
                    sc_strike = None
                    for pct in np.arange(0.01, 0.15, 0.005):
                        k = round(S * (1 + pct), 0)
                        d = bs_delta(S, k, T_sc, RISK_FREE_RATE, iv, 'call')
                        if d <= 0.32:
                            sc_strike = k
                            break
                    if sc_strike is None:
                        sc_strike = round(S * 1.05, 0)

                    # Ensure short call strike > LEAPS strike (avoid negative spread)
                    if sc_strike > leaps_position['strike']:
                        sc_mid = bs_price(S, sc_strike, T_sc, RISK_FREE_RATE, iv, 'call')
                        premium = sell_price(sc_mid, SPY_HAIRCUT)
                        premium -= COMMISSION_PER_CONTRACT / 100

                        if premium > 0.10:  # minimum premium threshold
                            short_call = {
                                'strike': sc_strike,
                                'expiry': sc_exp,
                                'entry_date': date,
                                'entry_S': S,
                                'premium': premium,
                            }
                            total_commissions += COMMISSION_PER_CONTRACT

        # Close remaining positions
        if short_call is not None:
            date = dates[-1]
            S = float(close.loc[date])
            sc_dte = max((short_call['expiry'] - date).days, 0)
            T_sc = max(sc_dte / 365.0, 0.001)
            sc_mid = bs_price(S, short_call['strike'], T_sc, RISK_FREE_RATE, iv, 'call')
            buyback = buy_price(sc_mid, SPY_HAIRCUT)
            pnl = (short_call['premium'] - buyback) * 100 - COMMISSION_PER_CONTRACT
            short_call_trades.append({
                'ticker': etf_name, 'entry_date': short_call['entry_date'], 'exit_date': date,
                'entry_price': short_call['entry_S'], 'exit_price': S,
                'strike': short_call['strike'], 'premium': short_call['premium'],
                'pnl': pnl, 'exit_reason': 'eod_close',
                'holding_days': (date - short_call['entry_date']).days,
            })
            total_premium_collected += pnl

        # Compute LEAPS P&L at end
        if leaps_position is not None:
            date = dates[-1]
            S = float(close.loc[date])
            leaps_dte = max((leaps_position['expiry'] - date).days, 0)
            T_l = max(leaps_dte / 365.0, 0.001)
            leaps_mid = bs_price(S, leaps_position['strike'], T_l, RISK_FREE_RATE, iv, 'call')
            leaps_exit = sell_price(leaps_mid, SPY_HAIRCUT) * 100

        # Add net_credit column for compatibility
        for t in short_call_trades:
            t['net_credit'] = t['premium']

        all_trades.extend(short_call_trades)

        print(f"    {etf_name}: {len(short_call_trades)} short call cycles")
        print(f"    Total short call P&L: ${sum(t['pnl'] for t in short_call_trades):.0f}")
        print(f"    LEAPS total cost: ${leaps_total_cost:.0f}")
        print(f"    Total commissions: ${total_commissions:.0f}")

    return pd.DataFrame(all_trades)


# =============================================================================
# Strategy 2: Jade Lizard
# =============================================================================

def backtest_jade_lizard(stock_data, vix):
    """
    Jade Lizard on high-IV individual stocks.

    Structure: Sell OTM put + sell call spread (short call + long higher call)
    Key: credit > call spread width => NO upside risk.
    Downside risk: naked put below put strike minus credit received.

    Entry: IV rank > 50, ~45 DTE
    Exit: 50% of max profit, or 7 DTE, or expiration
    """
    print("\n" + "="*70)
    print("STRATEGY 2: JADE LIZARD (HIGH-IV STOCKS)")
    print("="*70)

    all_trades = []

    for ticker, df in stock_data.items():
        print(f"\n  Processing {ticker}...")
        close = df['Close'].copy()

        hv21 = compute_historical_vol(close, window=21)
        iv_rank = compute_iv_rank(hv21, lookback=252)

        active_trade = None
        start_date = pd.Timestamp('2015-01-05')
        dates = close.index[close.index >= start_date]

        for i, date in enumerate(dates):
            S = float(close.loc[date])
            cur_hv = float(hv21.loc[date]) if date in hv21.index and not pd.isna(hv21.loc[date]) else None
            cur_rank = float(iv_rank.loc[date]) if date in iv_rank.index and not pd.isna(iv_rank.loc[date]) else None

            if cur_hv is None or cur_rank is None:
                continue

            iv = max(cur_hv * 1.10, 0.15)

            # ---- Manage ----
            if active_trade is not None:
                dte = (active_trade['expiry'] - date).days

                if dte <= 0:
                    put_intr = max(active_trade['put_strike'] - S, 0)
                    sc_intr = max(S - active_trade['short_call_strike'], 0)
                    lc_intr = max(S - active_trade['long_call_strike'], 0)
                    settlement_cost = put_intr + sc_intr - lc_intr
                    pnl = (active_trade['net_credit'] - settlement_cost) * 100
                    pnl -= 3 * COMMISSION_PER_CONTRACT
                    exit_reason = 'expiration'
                else:
                    T = dte / 365.0
                    K_p = active_trade['put_strike']
                    K_sc = active_trade['short_call_strike']
                    K_lc = active_trade['long_call_strike']

                    put_mid = bs_price(S, K_p, T, RISK_FREE_RATE, iv, 'put')
                    sc_mid = bs_price(S, K_sc, T, RISK_FREE_RATE, iv, 'call')
                    lc_mid = bs_price(S, K_lc, T, RISK_FREE_RATE, iv, 'call')

                    close_cost = buy_price(put_mid, STOCK_HAIRCUT) + buy_price(sc_mid, STOCK_HAIRCUT) - sell_price(lc_mid, STOCK_HAIRCUT)
                    pnl = (active_trade['net_credit'] - close_cost) * 100
                    pnl -= 3 * COMMISSION_PER_CONTRACT

                    target = active_trade['net_credit'] * 100 * 0.50
                    exit_reason = None
                    if pnl >= target:
                        exit_reason = 'profit_target'
                    elif dte <= 7:
                        exit_reason = 'dte_exit'

                if exit_reason if dte <= 0 else exit_reason:
                    all_trades.append({
                        'ticker': ticker,
                        'entry_date': active_trade['entry_date'],
                        'exit_date': date,
                        'entry_price': active_trade['entry_price'],
                        'exit_price': S,
                        'put_strike': active_trade['put_strike'],
                        'short_call_strike': active_trade['short_call_strike'],
                        'long_call_strike': active_trade['long_call_strike'],
                        'net_credit': active_trade['net_credit'],
                        'pnl': pnl,
                        'exit_reason': exit_reason if dte > 0 else 'expiration',
                        'holding_days': (date - active_trade['entry_date']).days,
                        'iv_rank_entry': active_trade['iv_rank'],
                        'iv_entry': active_trade['iv'],
                        'spot_move_pct': (S / active_trade['entry_price'] - 1) * 100,
                    })
                    active_trade = None

            # ---- Entry ----
            if active_trade is None and i > 0 and cur_rank > 50:
                expiry = find_monthly_expiry(date, 2)
                dte = (expiry - date).days
                if dte < 30:
                    expiry = find_monthly_expiry(date, 3)
                    dte = (expiry - date).days
                T = dte / 365.0

                # Put strike: ~0.20 delta
                put_strike = None
                for pct in np.arange(0.03, 0.20, 0.01):
                    k = round(S * (1 - pct), 2)
                    d = abs(bs_delta(S, k, T, RISK_FREE_RATE, iv, 'put'))
                    if d <= 0.22:
                        put_strike = k
                        break
                if put_strike is None:
                    put_strike = round(S * 0.90, 2)

                # Short call: ~0.25 delta
                short_call_strike = None
                for pct in np.arange(0.03, 0.20, 0.01):
                    k = round(S * (1 + pct), 2)
                    d = bs_delta(S, k, T, RISK_FREE_RATE, iv, 'call')
                    if d <= 0.27:
                        short_call_strike = k
                        break
                if short_call_strike is None:
                    short_call_strike = round(S * 1.08, 2)

                # Long call: spread width = 5% of S or $5
                width_dollars = max(S * 0.05, 5.0)
                long_call_strike = round(short_call_strike + width_dollars, 2)
                call_spread_width = long_call_strike - short_call_strike

                put_mid = bs_price(S, put_strike, T, RISK_FREE_RATE, iv, 'put')
                sc_mid = bs_price(S, short_call_strike, T, RISK_FREE_RATE, iv, 'call')
                lc_mid = bs_price(S, long_call_strike, T, RISK_FREE_RATE, iv, 'call')

                net_credit = sell_price(put_mid, STOCK_HAIRCUT) + sell_price(sc_mid, STOCK_HAIRCUT) - buy_price(lc_mid, STOCK_HAIRCUT)
                entry_comm = 3 * COMMISSION_PER_CONTRACT / 100
                net_credit -= entry_comm

                if net_credit > call_spread_width and net_credit > 0.50:
                    active_trade = {
                        'entry_date': date,
                        'entry_price': S,
                        'put_strike': put_strike,
                        'short_call_strike': short_call_strike,
                        'long_call_strike': long_call_strike,
                        'expiry': expiry,
                        'net_credit': net_credit,
                        'iv': iv,
                        'iv_rank': cur_rank,
                    }

        # Close remaining
        if active_trade is not None:
            date = dates[-1]
            S = float(close.loc[date])
            dte = max((active_trade['expiry'] - date).days, 0)
            T = max(dte / 365.0, 0.001)
            put_mid = bs_price(S, active_trade['put_strike'], T, RISK_FREE_RATE, iv, 'put')
            sc_mid = bs_price(S, active_trade['short_call_strike'], T, RISK_FREE_RATE, iv, 'call')
            lc_mid = bs_price(S, active_trade['long_call_strike'], T, RISK_FREE_RATE, iv, 'call')
            close_cost = buy_price(put_mid, STOCK_HAIRCUT) + buy_price(sc_mid, STOCK_HAIRCUT) - sell_price(lc_mid, STOCK_HAIRCUT)
            pnl = (active_trade['net_credit'] - close_cost) * 100 - 3 * COMMISSION_PER_CONTRACT
            all_trades.append({
                'ticker': ticker, 'entry_date': active_trade['entry_date'], 'exit_date': date,
                'entry_price': active_trade['entry_price'], 'exit_price': S,
                'put_strike': active_trade['put_strike'],
                'short_call_strike': active_trade['short_call_strike'],
                'long_call_strike': active_trade['long_call_strike'],
                'net_credit': active_trade['net_credit'], 'pnl': pnl,
                'exit_reason': 'eod_close', 'holding_days': (date - active_trade['entry_date']).days,
                'iv_rank_entry': active_trade['iv_rank'], 'iv_entry': active_trade['iv'],
                'spot_move_pct': 0,
            })

    return pd.DataFrame(all_trades)


# =============================================================================
# Analytics
# =============================================================================

def compute_metrics(trades_df, strategy_name, capital=10000):
    if len(trades_df) == 0:
        print(f"\n  {strategy_name}: NO TRADES")
        return {}

    pnls = trades_df['pnl'].values
    n_trades = len(pnls)
    winners = (pnls > 0).sum()
    losers = (pnls <= 0).sum()
    win_rate = winners / n_trades * 100

    total_pnl = pnls.sum()
    avg_pnl = pnls.mean()
    avg_winner = pnls[pnls > 0].mean() if winners > 0 else 0
    avg_loser = pnls[pnls <= 0].mean() if losers > 0 else 0

    equity = np.cumsum(pnls) + capital
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = dd.min() * 100

    total_days = (trades_df['exit_date'].max() - trades_df['entry_date'].min()).days
    years = max(total_days / 365.25, 0.5)
    cagr = ((max(equity[-1], 1) / capital) ** (1 / years) - 1) * 100

    trades_df = trades_df.copy()
    trades_df['exit_month'] = pd.to_datetime(trades_df['exit_date']).dt.to_period('M')
    monthly_pnl = trades_df.groupby('exit_month')['pnl'].sum()
    monthly_ret = monthly_pnl / capital

    if len(monthly_ret) > 2:
        sharpe = monthly_ret.mean() / monthly_ret.std() * np.sqrt(12) if monthly_ret.std() > 0 else 0
        downside = monthly_ret[monthly_ret < 0].std()
        sortino = monthly_ret.mean() / downside * np.sqrt(12) if downside > 0 else 0
    else:
        sharpe = sortino = 0

    gross_profit = pnls[pnls > 0].sum() if winners > 0 else 0
    gross_loss = abs(pnls[pnls <= 0].sum()) if losers > 0 else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    avg_credit = trades_df['net_credit'].mean() if 'net_credit' in trades_df.columns else 0
    avg_premium = trades_df['premium'].mean() if 'premium' in trades_df.columns else avg_credit

    metrics = {
        'strategy': strategy_name,
        'n_trades': int(n_trades),
        'winners': int(winners),
        'losers': int(losers),
        'win_rate': round(win_rate, 1),
        'total_pnl': round(total_pnl, 2),
        'avg_pnl': round(avg_pnl, 2),
        'avg_winner': round(avg_winner, 2),
        'avg_loser': round(avg_loser, 2),
        'avg_premium_or_credit': round(avg_premium, 4),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'cagr_pct': round(cagr, 2),
        'max_dd_pct': round(max_dd, 2),
        'profit_factor': round(profit_factor, 3),
        'years': round(years, 1),
    }

    print(f"\n{'='*55}")
    print(f"  {strategy_name} RESULTS")
    print(f"{'='*55}")
    for k, v in metrics.items():
        print(f"  {k:>25}: {v}")

    return metrics


def regime_analysis(trades_df, spy_close, strategy_name):
    if len(trades_df) == 0:
        return {}

    spy_ret_20d = spy_close.pct_change(20).shift(1)
    trades_df = trades_df.copy()
    trades_df['spy_regime'] = 'neutral'

    for idx, row in trades_df.iterrows():
        entry = row['entry_date']
        if entry in spy_ret_20d.index and not pd.isna(spy_ret_20d.loc[entry]):
            ret = float(spy_ret_20d.loc[entry])
            if ret > 0.02:
                trades_df.at[idx, 'spy_regime'] = 'bull'
            elif ret < -0.02:
                trades_df.at[idx, 'spy_regime'] = 'bear'

    print(f"\n  REGIME ANALYSIS (20d SPY return) - {strategy_name}")
    print(f"  {'='*60}")

    results = {}
    for regime in ['bull', 'bear', 'neutral']:
        subset = trades_df[trades_df['spy_regime'] == regime]
        if len(subset) == 0:
            continue
        wr = (subset['pnl'] > 0).mean() * 100
        avg = subset['pnl'].mean()
        n = len(subset)
        pt_sharpe = subset['pnl'].mean() / subset['pnl'].std() if subset['pnl'].std() > 0 else 0
        results[regime] = {'n': n, 'wr': round(wr, 1), 'avg_pnl': round(avg, 2), 'sharpe': round(pt_sharpe, 3)}
        print(f"  {regime:>8}: n={n:>4}, WR={wr:.1f}%, avg=${avg:.2f}, sharpe={pt_sharpe:.3f}")

    if 'bull' in results and 'bear' in results:
        s_b = results['bull']['sharpe']
        s_r = results['bear']['sharpe']
        denom = max(abs(s_b), abs(s_r), 0.001)
        ratio = abs(s_b - s_r) / denom
        status = "PASS" if ratio < 0.50 else "MARGINAL" if ratio < 0.75 else "FAIL"
        print(f"  Regime Sharpe divergence: {ratio:.2f} ({status})")

    return results


def permutation_test(trades_df, strategy_name, n_perms=200):
    """
    Permutation test: test if the strategy's performance is due to skill or luck.

    Method: Bootstrap the trade Sharpe ratio by resampling trade P&Ls with replacement.
    If the actual Sharpe is in the top 5% of the bootstrapped null distribution,
    the result is statistically significant.

    Note: A simple shuffle of P&Ls preserves the mean (trivially p=0.5).
    Instead we use a bootstrap approach: resample WITH replacement to create
    synthetic equity curves and test the realized Sharpe against this distribution.
    """
    if len(trades_df) < 10:
        print(f"\n  Permutation test: too few trades ({len(trades_df)})")
        return 0, 1.0

    pnls = trades_df['pnl'].values
    actual_mean = pnls.mean()
    actual_sharpe = pnls.mean() / pnls.std() if pnls.std() > 0 else 0

    rng = np.random.RandomState(42)
    boot_sharpes = []
    n = len(pnls)

    for _ in range(n_perms):
        # Resample with replacement (bootstrap)
        sample = rng.choice(pnls, size=n, replace=True)
        s = sample.mean() / sample.std() if sample.std() > 0 else 0
        boot_sharpes.append(s)

    boot_sharpes = np.array(boot_sharpes)

    # Under null (no skill), center the distribution at zero
    # Test: how extreme is our Sharpe vs zero-centered bootstrap?
    centered_sharpes = boot_sharpes - boot_sharpes.mean()
    p_value = (centered_sharpes >= actual_sharpe).mean()

    # Also compute confidence interval
    ci_lo = np.percentile(boot_sharpes, 2.5)
    ci_hi = np.percentile(boot_sharpes, 97.5)

    print(f"\n  PERMUTATION/BOOTSTRAP TEST - {strategy_name}")
    print(f"  Actual mean P&L: ${actual_mean:.2f}, per-trade Sharpe: {actual_sharpe:.4f}")
    print(f"  Bootstrap Sharpe 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  Null p-value: {p_value:.4f} ({'SIGNIFICANT' if p_value < 0.05 else 'NOT significant'})")
    print(f"  Sharpe > 0 in {(np.array(boot_sharpes) > 0).mean()*100:.1f}% of bootstraps")

    return actual_sharpe, p_value


def lag_sensitivity_test(trades_df, strategy_name):
    if len(trades_df) < 10:
        return

    print(f"\n  LAG SENSITIVITY - {strategy_name}")
    print(f"  Original avg P&L: ${trades_df['pnl'].mean():.2f}, WR: {(trades_df['pnl']>0).mean()*100:.1f}%")

    if 'holding_days' in trades_df.columns:
        corr = trades_df[['holding_days', 'pnl']].corr().iloc[0, 1]
        median_hold = trades_df['holding_days'].median()
        early = trades_df[trades_df['holding_days'] <= median_hold]
        late = trades_df[trades_df['holding_days'] > median_hold]
        print(f"  Hold-days vs P&L corr: {corr:.3f}")
        if len(early) > 0 and len(late) > 0:
            print(f"  Early (<= {median_hold:.0f}d): n={len(early)}, avg=${early['pnl'].mean():.2f}, WR={100*(early['pnl']>0).mean():.1f}%")
            print(f"  Late  (>  {median_hold:.0f}d): n={len(late)}, avg=${late['pnl'].mean():.2f}, WR={100*(late['pnl']>0).mean():.1f}%")

    if 'spot_move_pct' in trades_df.columns:
        corr2 = trades_df[['spot_move_pct', 'pnl']].corr().iloc[0, 1]
        print(f"  Spot move vs P&L corr: {corr2:.3f}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("Options Income Strategies v2 Backtest")
    print("="*70)

    etf_data, vix, stock_data = download_data()

    # ---- Strategy 1: PMCC ----
    print("\n\nRunning PMCC backtest...")
    pmcc_trades = backtest_pmcc(etf_data, vix)
    pmcc_metrics = compute_metrics(pmcc_trades, "PMCC (SPY)")

    if len(pmcc_trades) > 0:
        pmcc_trades.to_csv(f'{OUTPUT_DIR}/pmcc_trades.csv', index=False)
        spy_close = etf_data['SPY']['Close']
        regime_pmcc = regime_analysis(pmcc_trades, spy_close, "PMCC")
        perm_sharpe_pmcc, pval_pmcc = permutation_test(pmcc_trades, "PMCC")
        lag_sensitivity_test(pmcc_trades, "PMCC")

        print("\n  EXIT REASONS:")
        print(pmcc_trades['exit_reason'].value_counts().to_string())

        if 'ticker' in pmcc_trades.columns:
            print("\n  PER-ETF BREAKDOWN:")
            for t, g in pmcc_trades.groupby('ticker'):
                wr = (g['pnl'] > 0).mean() * 100
                print(f"    {t:>5}: n={len(g):>3}, WR={wr:.0f}%, avg=${g['pnl'].mean():.0f}, total=${g['pnl'].sum():.0f}")

    # ---- Strategy 2: Jade Lizard ----
    print("\n\nRunning Jade Lizard backtest...")
    jade_trades = backtest_jade_lizard(stock_data, vix)
    jade_metrics = compute_metrics(jade_trades, "Jade Lizard")

    if len(jade_trades) > 0:
        jade_trades.to_csv(f'{OUTPUT_DIR}/jade_lizard_trades.csv', index=False)
        spy_close = etf_data['SPY']['Close']
        regime_jade = regime_analysis(jade_trades, spy_close, "Jade Lizard")
        perm_sharpe_jade, pval_jade = permutation_test(jade_trades, "Jade Lizard")
        lag_sensitivity_test(jade_trades, "Jade Lizard")

        print("\n  PER-TICKER BREAKDOWN:")
        for ticker, group in jade_trades.groupby('ticker'):
            wr = (group['pnl'] > 0).mean() * 100
            print(f"    {ticker:>6}: n={len(group):>3}, WR={wr:.0f}%, avg=${group['pnl'].mean():.0f}, total=${group['pnl'].sum():.0f}")

        print("\n  EXIT REASONS:")
        print(jade_trades['exit_reason'].value_counts().to_string())

    # ---- Comparison ----
    print("\n" + "="*70)
    print("STRATEGY COMPARISON")
    print("="*70)

    comparison = {}
    for name, metrics in [("PMCC", pmcc_metrics), ("Jade Lizard", jade_metrics)]:
        if metrics:
            comparison[name] = metrics

    if comparison:
        comp_df = pd.DataFrame(comparison).T
        cols = ['n_trades', 'win_rate', 'sharpe', 'sortino', 'cagr_pct', 'max_dd_pct', 'profit_factor', 'avg_pnl', 'total_pnl']
        available = [c for c in cols if c in comp_df.columns]
        print(comp_df[available].to_string())

    print("\n  EXISTING STRATEGY BENCHMARKS:")
    print("  Wheel (CSP+CC):  Sharpe=0.365, CAGR=5.4%")
    print("  BPS Conservative: +5.9% in 2 weeks (paper)")

    # Save summary
    summary = {
        'pmcc': pmcc_metrics,
        'jade_lizard': jade_metrics,
        'run_date': datetime.now().isoformat(),
        'backtest_period': '2015-01 to 2026-07',
        'commission': '$0.65/contract/leg',
        'bid_ask_haircut': 'SPY/QQQ: 2% each side, stocks: 8% each side',
        'anti_lookahead': 'all signals T-1, BS pricing with haircuts, 200-SMA trend filter',
    }
    with open(f'{OUTPUT_DIR}/backtest_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nResults saved to {OUTPUT_DIR}/")
    print("DONE.")


if __name__ == '__main__':
    main()
