#!/usr/bin/env python3
"""
Options-Enhanced Growth Strategy Backtest
==========================================
HC #696: High-return growth research — options overlay on growth ETFs.

Prior research: equity momentum (20-30% CAGR), leveraged ETF rotation (35%+),
trend following (15-25%). Options ON growth positions are untested at scale.

Strategies tested:
  1. Wheel on TQQQ — CSPs at delta 25-30, covered calls when assigned
  2. Collar on QQQ — buy QQQ + sell OTM call + buy OTM put
  3. Put selling for entry — only enter TQQQ via put selling
  4. Synthetic long + protection — sell ATM put + buy ATM call + OTM put hedge
  5. Cash-secured puts on growth ETFs — diversified CSPs on QQQ/SMH/XLK/ARKK
  6. Bull call spreads — buy ATM call + sell OTM call, defined risk

Black-Scholes pricing with VIX as IV proxy. Commission-free (HC #694).
Walk-forward: sliding 252d train, monthly test periods.
"""

import math
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

OUT_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research/options_growth")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RISK_FREE = 0.04
STARTING_CAPITAL = 100_000
TRADING_DAYS = 252
BA_SPREAD_FRAC = 0.05  # 5% bid-ask cost on premium (conservative for liquid ETFs)
# Commission: $0 per HC #694 (Robinhood/IBKR commission-free for options)


# ═══════════════════════════════════════════════════════════════════
# Black-Scholes Primitives (from codebase standard)
# ═══════════════════════════════════════════════════════════════════

def _Phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _ndtri(p):
    """Inverse normal CDF (rational approximation)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def bs_price(S, K, T, sigma, r=RISK_FREE, kind="put"):
    """Black-Scholes option price."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(K - S, 0.0) if kind == "put" else max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if kind == "put":
        return K * math.exp(-r * T) * _Phi(-d2) - S * _Phi(-d1)
    return S * _Phi(d1) - K * math.exp(-r * T) * _Phi(d2)


def strike_from_delta(S, T, sigma, target_delta, kind="put", r=RISK_FREE):
    """Find strike that corresponds to a given delta."""
    if T <= 0 or sigma <= 0:
        return S
    target = abs(target_delta)
    p = target if kind == "call" else (1 - target)
    p = min(max(p, 1e-6), 1 - 1e-6)
    d1 = _ndtri(p)
    K = S * math.exp(-(d1 * sigma * math.sqrt(T) - (r + 0.5 * sigma**2) * T))
    return max(0.01, round(K, 2))


def get_iv(vix_val, ticker_vol_mult=1.0):
    """Convert VIX to implied vol for a given ticker.
    VIX is annualized 30d implied vol for SPX. Scale for other tickers."""
    iv = (vix_val / 100.0) * ticker_vol_mult
    return max(0.05, min(iv, 3.0))  # floor 5%, cap 300%


# ═══════════════════════════════════════════════════════════════════
# Data Download
# ═══════════════════════════════════════════════════════════════════

def download_data(start="2010-01-01", end="2026-07-14"):
    """Download daily data for all tickers needed."""
    tickers = ["QQQ", "TQQQ", "SPY", "^VIX", "SMH", "XLK", "ARKK", "SHV"]
    print(f"Downloading {len(tickers)} tickers...")
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data

    # Rename ^VIX
    if "^VIX" in prices.columns:
        prices = prices.rename(columns={"^VIX": "VIX"})

    prices = prices.ffill().dropna(how="all")
    print(f"Data: {prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}, {len(prices)} days")

    # Report availability
    for t in ["QQQ", "TQQQ", "SPY", "VIX", "SMH", "XLK", "ARKK"]:
        if t in prices.columns:
            valid = prices[t].notna().sum()
            start_d = prices[t].dropna().index[0].strftime('%Y-%m-%d')
            print(f"  {t}: {valid} days from {start_d}")
        else:
            print(f"  {t}: MISSING")

    return prices


# ═══════════════════════════════════════════════════════════════════
# IV Multipliers (historical vol ratio vs SPX)
# ═══════════════════════════════════════════════════════════════════

IV_MULT = {
    "QQQ": 1.15,   # slightly higher vol than SPX
    "TQQQ": 3.2,   # ~3x leveraged, IV higher due to vol drag
    "SPY": 1.0,
    "SMH": 1.35,   # semis are volatile
    "XLK": 1.1,    # tech sector, similar to QQQ
    "ARKK": 1.8,   # innovation/growth, much higher vol
}


# ═══════════════════════════════════════════════════════════════════
# Strategy Engines
# ═══════════════════════════════════════════════════════════════════

def strategy_wheel_tqqq(prices, dte_target=35, put_delta=0.27, call_delta=0.30):
    """
    Wheel on TQQQ: sell cash-secured puts. If assigned, sell covered calls.
    Compare total return vs buy-and-hold TQQQ.
    """
    tqqq = prices["TQQQ"].dropna()
    vix = prices["VIX"].reindex(tqqq.index).ffill()
    dates = tqqq.index[TRADING_DAYS:]  # skip warmup

    cash = STARTING_CAPITAL
    shares = 0
    share_basis = 0.0
    nav_series = []
    put_position = None   # {strike, expiry, premium}
    call_position = None  # {strike, expiry, premium}
    trades = []
    monthly_returns = []
    last_month = None

    for date in dates:
        S = tqqq.loc[date]
        v = vix.loc[date] if date in vix.index else 20.0
        iv = get_iv(v, IV_MULT["TQQQ"])
        T_dte = dte_target / 365.0

        nav = cash + shares * S
        nav_series.append({"date": date, "nav": nav, "cash": cash, "shares": shares})

        # Track monthly returns
        cur_month = date.to_period("M")
        if last_month is not None and cur_month != last_month:
            monthly_returns.append(nav)
        last_month = cur_month

        # Check put expiry
        if put_position and date >= put_position["expiry"]:
            if S <= put_position["strike"]:
                # Assigned — buy shares at strike
                n_contracts = put_position["contracts"]
                cost = put_position["strike"] * 100 * n_contracts
                if cost <= cash:
                    shares += 100 * n_contracts
                    cash -= cost
                    share_basis = put_position["strike"]
                    trades.append({"date": str(date.date()), "action": "put_assigned",
                                   "strike": put_position["strike"], "price": S})
            else:
                # Expired worthless — keep premium
                trades.append({"date": str(date.date()), "action": "put_expired_otm",
                               "strike": put_position["strike"], "price": S})
            put_position = None

        # Check call expiry
        if call_position and date >= call_position["expiry"]:
            if S >= call_position["strike"]:
                # Called away — sell shares at strike
                n_contracts = call_position["contracts"]
                proceeds = call_position["strike"] * 100 * n_contracts
                shares -= 100 * n_contracts
                shares = max(shares, 0)
                cash += proceeds
                trades.append({"date": str(date.date()), "action": "call_assigned",
                               "strike": call_position["strike"], "price": S})
            else:
                trades.append({"date": str(date.date()), "action": "call_expired_otm",
                               "strike": call_position["strike"], "price": S})
            call_position = None

        # Sell new options on monthly boundaries
        if put_position is None and call_position is None:
            if shares == 0:
                # Sell cash-secured put
                K = strike_from_delta(S, T_dte, iv, put_delta, kind="put")
                premium = bs_price(S, K, T_dte, iv, kind="put")
                premium *= (1 - BA_SPREAD_FRAC)  # bid-ask cost
                n_contracts = max(1, int(cash / (K * 100)))
                # Size: use up to 90% of cash
                n_contracts = min(n_contracts, int(0.9 * cash / (K * 100)))
                n_contracts = max(1, n_contracts)

                cash += premium * 100 * n_contracts
                expiry = date + pd.Timedelta(days=dte_target)
                put_position = {"strike": K, "expiry": expiry, "premium": premium,
                                "contracts": n_contracts}
                trades.append({"date": str(date.date()), "action": "sell_put",
                               "strike": K, "premium": round(premium, 2),
                               "contracts": n_contracts, "price": S})
            elif shares >= 100:
                # Sell covered call
                K = strike_from_delta(S, T_dte, iv, call_delta, kind="call")
                premium = bs_price(S, K, T_dte, iv, kind="call")
                premium *= (1 - BA_SPREAD_FRAC)
                n_contracts = shares // 100

                cash += premium * 100 * n_contracts
                expiry = date + pd.Timedelta(days=dte_target)
                call_position = {"strike": K, "expiry": expiry, "premium": premium,
                                 "contracts": n_contracts}
                trades.append({"date": str(date.date()), "action": "sell_call",
                               "strike": K, "premium": round(premium, 2),
                               "contracts": n_contracts, "price": S})

    # Build daily return series
    nav_df = pd.DataFrame(nav_series).set_index("date")
    daily_ret = nav_df["nav"].pct_change().dropna()
    return daily_ret, trades


def strategy_collar_qqq(prices, dte_target=35, call_delta=0.30, put_delta=0.20):
    """
    Collar on QQQ: long shares + sell OTM call + buy OTM put.
    Caps upside but protects downside. Monthly roll.
    """
    qqq = prices["QQQ"].dropna()
    vix = prices["VIX"].reindex(qqq.index).ffill()
    dates = qqq.index[TRADING_DAYS:]

    # Buy QQQ with full capital
    S0 = qqq.iloc[TRADING_DAYS]
    shares = int(STARTING_CAPITAL / S0)
    cash = STARTING_CAPITAL - shares * S0
    collar = None  # {call_strike, put_strike, expiry, net_premium}
    nav_series = []
    trades = []
    last_roll = None

    for date in dates:
        S = qqq.loc[date]
        v = vix.loc[date] if date in vix.index else 20.0
        iv = get_iv(v, IV_MULT["QQQ"])
        T_dte = dte_target / 365.0

        nav = cash + shares * S
        nav_series.append({"date": date, "nav": nav})

        # Check expiry
        if collar and date >= collar["expiry"]:
            # Settle
            n_contracts = collar["contracts"]
            if S >= collar["call_strike"]:
                # Called away — sell at call strike, rebuy at market
                cash += collar["call_strike"] * shares
                shares = int((cash) / S)
                cash -= shares * S
            if S <= collar["put_strike"]:
                # Put exercised — sell at put strike, rebuy at market
                cash += collar["put_strike"] * shares
                shares = int((cash) / S)
                cash -= shares * S
            collar = None

        # Roll collar monthly
        should_roll = collar is None
        if not should_roll and last_roll is not None:
            if (date - last_roll).days >= 28:
                should_roll = True

        if should_roll and shares >= 100:
            n_contracts = shares // 100
            call_K = strike_from_delta(S, T_dte, iv, call_delta, kind="call")
            put_K = strike_from_delta(S, T_dte, iv, put_delta, kind="put")

            call_prem = bs_price(S, call_K, T_dte, iv, kind="call") * (1 - BA_SPREAD_FRAC)
            put_cost = bs_price(S, put_K, T_dte, iv, kind="put") * (1 + BA_SPREAD_FRAC)

            net_premium = (call_prem - put_cost) * 100 * n_contracts
            cash += net_premium

            expiry = date + pd.Timedelta(days=dte_target)
            collar = {"call_strike": call_K, "put_strike": put_K, "expiry": expiry,
                       "contracts": n_contracts}
            last_roll = date
            trades.append({"date": str(date.date()), "action": "roll_collar",
                           "call_K": call_K, "put_K": put_K,
                           "net_premium": round(net_premium, 2), "price": S})

    nav_df = pd.DataFrame(nav_series).set_index("date")
    daily_ret = nav_df["nav"].pct_change().dropna()
    return daily_ret, trades


def strategy_put_entry_tqqq(prices, dte_target=35, put_delta=0.27, call_delta=0.30):
    """
    Put selling for entry: only enter TQQQ through put selling.
    When assigned, hold until covered call gets you out. Repeat.
    """
    tqqq = prices["TQQQ"].dropna()
    vix = prices["VIX"].reindex(tqqq.index).ffill()
    dates = tqqq.index[TRADING_DAYS:]

    cash = STARTING_CAPITAL
    shares = 0
    nav_series = []
    option_pos = None  # current option position
    trades = []

    for date in dates:
        S = tqqq.loc[date]
        v = vix.loc[date] if date in vix.index else 20.0
        iv = get_iv(v, IV_MULT["TQQQ"])
        T_dte = dte_target / 365.0

        nav = cash + shares * S
        nav_series.append({"date": date, "nav": nav})

        # Check option expiry
        if option_pos and date >= option_pos["expiry"]:
            if option_pos["kind"] == "put":
                if S <= option_pos["strike"]:
                    # Assigned — acquire shares
                    n = option_pos["contracts"]
                    cost = option_pos["strike"] * 100 * n
                    if cost <= cash + shares * S * 0.5:  # allow some flexibility
                        shares += 100 * n
                        cash -= cost
                        trades.append({"date": str(date.date()), "action": "put_assigned",
                                       "strike": option_pos["strike"]})
                else:
                    trades.append({"date": str(date.date()), "action": "put_expired_otm"})
            elif option_pos["kind"] == "call":
                if S >= option_pos["strike"]:
                    # Called away
                    n = option_pos["contracts"]
                    proceeds = option_pos["strike"] * 100 * n
                    shares -= 100 * n
                    shares = max(shares, 0)
                    cash += proceeds
                    trades.append({"date": str(date.date()), "action": "call_assigned",
                                   "strike": option_pos["strike"]})
                else:
                    trades.append({"date": str(date.date()), "action": "call_expired_otm"})
            option_pos = None

        # Open new position
        if option_pos is None:
            if shares == 0 and cash > S * 10:
                # Sell put to enter
                K = strike_from_delta(S, T_dte, iv, put_delta, kind="put")
                prem = bs_price(S, K, T_dte, iv, kind="put") * (1 - BA_SPREAD_FRAC)
                n = max(1, int(0.9 * cash / (K * 100)))
                cash += prem * 100 * n
                expiry = date + pd.Timedelta(days=dte_target)
                option_pos = {"kind": "put", "strike": K, "expiry": expiry,
                              "contracts": n, "premium": prem}
                trades.append({"date": str(date.date()), "action": "sell_put",
                               "strike": K, "premium": round(prem, 2), "contracts": n})
            elif shares >= 100:
                # Sell covered call to exit
                K = strike_from_delta(S, T_dte, iv, call_delta, kind="call")
                prem = bs_price(S, K, T_dte, iv, kind="call") * (1 - BA_SPREAD_FRAC)
                n = shares // 100
                cash += prem * 100 * n
                expiry = date + pd.Timedelta(days=dte_target)
                option_pos = {"kind": "call", "strike": K, "expiry": expiry,
                              "contracts": n, "premium": prem}
                trades.append({"date": str(date.date()), "action": "sell_call",
                               "strike": K, "premium": round(prem, 2), "contracts": n})

    nav_df = pd.DataFrame(nav_series).set_index("date")
    daily_ret = nav_df["nav"].pct_change().dropna()
    return daily_ret, trades


def strategy_synthetic_long(prices, dte_target=35, hedge_delta=0.10):
    """
    Synthetic long with protection on QQQ.
    Sell ATM put + buy ATM call = synthetic long (leveraged exposure).
    Buy OTM put at delta 10 for tail protection.
    Monthly roll.
    """
    qqq = prices["QQQ"].dropna()
    vix = prices["VIX"].reindex(qqq.index).ffill()
    dates = qqq.index[TRADING_DAYS:]

    cash = STARTING_CAPITAL
    nav_series = []
    position = None
    trades = []
    notional_exposure = 0

    for date in dates:
        S = qqq.loc[date]
        v = vix.loc[date] if date in vix.index else 20.0
        iv = get_iv(v, IV_MULT["QQQ"])
        T_dte = dte_target / 365.0

        # Mark-to-market synthetic position
        if position:
            # P&L from underlying move since position open
            delta_pnl = (S - position["ref_price"]) * 100 * position["contracts"]
            # Time decay on net position (approximate)
            days_held = (date - position["open_date"]).days
            remaining_T = max(0.001, (dte_target - days_held) / 365.0)

            # Current value of legs
            call_val = bs_price(S, position["call_K"], remaining_T, iv, kind="call")
            put_val = bs_price(S, position["put_K"], remaining_T, iv, kind="put")
            hedge_val = bs_price(S, position["hedge_K"], remaining_T, iv, kind="put")

            pos_value = (call_val - put_val + hedge_val) * 100 * position["contracts"]
            nav = cash + pos_value + delta_pnl
        else:
            nav = cash

        nav_series.append({"date": date, "nav": nav})

        # Check expiry / roll
        if position and date >= position["expiry"]:
            # Settle at expiry
            pnl = 0
            n = position["contracts"]
            # Call payoff
            if S > position["call_K"]:
                pnl += (S - position["call_K"]) * 100 * n
            # Put obligation
            if S < position["put_K"]:
                pnl -= (position["put_K"] - S) * 100 * n
            # Hedge payoff
            if S < position["hedge_K"]:
                pnl += (position["hedge_K"] - S) * 100 * n

            cash += pnl + position["net_credit"]
            position = None
            trades.append({"date": str(date.date()), "action": "synthetic_expired",
                           "pnl": round(pnl, 2)})

        # Open new synthetic on monthly boundaries
        if position is None:
            call_K = round(S, 2)  # ATM call
            put_K = round(S, 2)   # ATM put
            hedge_K = strike_from_delta(S, T_dte, iv, hedge_delta, kind="put")

            call_cost = bs_price(S, call_K, T_dte, iv, kind="call") * (1 + BA_SPREAD_FRAC)
            put_prem = bs_price(S, put_K, T_dte, iv, kind="put") * (1 - BA_SPREAD_FRAC)
            hedge_cost = bs_price(S, hedge_K, T_dte, iv, kind="put") * (1 + BA_SPREAD_FRAC)

            net_cost_per_contract = (call_cost - put_prem + hedge_cost) * 100

            # Size: risk no more than 20% of capital per position
            max_loss = (put_K - hedge_K) * 100  # max loss between put sold and hedge
            if max_loss > 0:
                n = max(1, int(0.20 * cash / max_loss))
            else:
                n = max(1, int(0.20 * cash / (S * 10)))

            total_cost = net_cost_per_contract * n
            if total_cost > cash * 0.5:
                n = max(1, int(0.5 * cash / abs(net_cost_per_contract))) if net_cost_per_contract != 0 else 1

            cash -= total_cost
            expiry = date + pd.Timedelta(days=dte_target)
            position = {
                "call_K": call_K, "put_K": put_K, "hedge_K": hedge_K,
                "expiry": expiry, "contracts": n,
                "net_credit": -total_cost,  # negative = net debit
                "ref_price": S, "open_date": date,
            }
            trades.append({"date": str(date.date()), "action": "open_synthetic",
                           "call_K": call_K, "put_K": put_K, "hedge_K": hedge_K,
                           "contracts": n, "net_cost": round(total_cost, 2)})

    nav_df = pd.DataFrame(nav_series).set_index("date")
    daily_ret = nav_df["nav"].pct_change().dropna()
    return daily_ret, trades


def strategy_diversified_csps(prices, dte_target=35, put_delta=0.25):
    """
    Cash-secured puts on multiple growth ETFs: QQQ, SMH, XLK, ARKK.
    Diversified across sectors. When assigned, hold and sell covered calls.
    """
    growth_etfs = ["QQQ", "SMH", "XLK", "ARKK"]
    available = [t for t in growth_etfs if t in prices.columns and prices[t].notna().sum() > TRADING_DAYS]

    if len(available) < 2:
        print(f"  WARNING: Only {len(available)} growth ETFs available for diversified CSPs")
        if len(available) == 0:
            return pd.Series(dtype=float), []

    vix = prices["VIX"].ffill()
    # Use common date range
    common_idx = prices[available].dropna(how="all").index[TRADING_DAYS:]
    dates = common_idx

    cash = STARTING_CAPITAL
    holdings = {t: {"shares": 0, "put": None, "call": None} for t in available}
    nav_series = []
    trades = []
    alloc_per_ticker = 1.0 / len(available)

    for date in dates:
        v = vix.loc[date] if date in vix.index else 20.0

        # Mark-to-market
        total_shares_val = sum(
            holdings[t]["shares"] * prices[t].loc[date]
            for t in available if prices[t].loc[date] == prices[t].loc[date]  # not NaN
        )
        nav = cash + total_shares_val
        nav_series.append({"date": date, "nav": nav})

        for ticker in available:
            if ticker not in prices.columns:
                continue
            S = prices[ticker].loc[date]
            if S != S:  # NaN check
                continue
            iv = get_iv(v, IV_MULT.get(ticker, 1.2))
            T_dte = dte_target / 365.0
            h = holdings[ticker]

            # Check put expiry
            if h["put"] and date >= h["put"]["expiry"]:
                if S <= h["put"]["strike"]:
                    n = h["put"]["contracts"]
                    cost = h["put"]["strike"] * 100 * n
                    if cost <= cash:
                        h["shares"] += 100 * n
                        cash -= cost
                        trades.append({"date": str(date.date()), "action": "put_assigned",
                                       "ticker": ticker, "strike": h["put"]["strike"]})
                else:
                    trades.append({"date": str(date.date()), "action": "put_expired",
                                   "ticker": ticker})
                h["put"] = None

            # Check call expiry
            if h["call"] and date >= h["call"]["expiry"]:
                if S >= h["call"]["strike"]:
                    n = h["call"]["contracts"]
                    proceeds = h["call"]["strike"] * 100 * n
                    h["shares"] -= 100 * n
                    h["shares"] = max(h["shares"], 0)
                    cash += proceeds
                    trades.append({"date": str(date.date()), "action": "call_assigned",
                                   "ticker": ticker})
                else:
                    trades.append({"date": str(date.date()), "action": "call_expired",
                                   "ticker": ticker})
                h["call"] = None

            # Open new positions
            ticker_cash_alloc = nav * alloc_per_ticker

            if h["put"] is None and h["call"] is None:
                if h["shares"] == 0:
                    # Sell put
                    K = strike_from_delta(S, T_dte, iv, put_delta, kind="put")
                    prem = bs_price(S, K, T_dte, iv, kind="put") * (1 - BA_SPREAD_FRAC)
                    n = max(1, int(0.9 * ticker_cash_alloc / (K * 100)))
                    n = min(n, max(1, int(cash * 0.4 / (K * 100))))
                    if K * 100 * n <= cash:
                        cash += prem * 100 * n
                        expiry = date + pd.Timedelta(days=dte_target)
                        h["put"] = {"strike": K, "expiry": expiry, "contracts": n}
                        trades.append({"date": str(date.date()), "action": "sell_put",
                                       "ticker": ticker, "strike": K, "contracts": n})
                elif h["shares"] >= 100:
                    # Sell covered call
                    K = strike_from_delta(S, T_dte, iv, 0.30, kind="call")
                    prem = bs_price(S, K, T_dte, iv, kind="call") * (1 - BA_SPREAD_FRAC)
                    n = h["shares"] // 100
                    cash += prem * 100 * n
                    expiry = date + pd.Timedelta(days=dte_target)
                    h["call"] = {"strike": K, "expiry": expiry, "contracts": n}
                    trades.append({"date": str(date.date()), "action": "sell_call",
                                   "ticker": ticker, "strike": K, "contracts": n})

    nav_df = pd.DataFrame(nav_series).set_index("date")
    daily_ret = nav_df["nav"].pct_change().dropna()
    return daily_ret, trades


def strategy_bull_call_spread(prices, ticker="QQQ", dte_target=35,
                               long_delta=0.50, short_delta=0.25):
    """
    Bull call spreads on a growth ETF.
    Buy ATM call (delta ~50), sell OTM call (delta ~25). Defined risk.
    Monthly rolls. Size to risk ~10% of capital per spread.
    """
    px = prices[ticker].dropna()
    vix = prices["VIX"].reindex(px.index).ffill()
    dates = px.index[TRADING_DAYS:]

    cash = STARTING_CAPITAL
    position = None
    nav_series = []
    trades = []

    for date in dates:
        S = px.loc[date]
        v = vix.loc[date] if date in vix.index else 20.0
        iv = get_iv(v, IV_MULT.get(ticker, 1.15))
        T_dte = dte_target / 365.0

        # MTM
        if position:
            days_held = (date - position["open_date"]).days
            remaining_T = max(0.001, (dte_target - days_held) / 365.0)
            long_val = bs_price(S, position["long_K"], remaining_T, iv, kind="call")
            short_val = bs_price(S, position["short_K"], remaining_T, iv, kind="call")
            spread_val = (long_val - short_val) * 100 * position["contracts"]
            nav = cash + spread_val
        else:
            nav = cash
        nav_series.append({"date": date, "nav": nav})

        # Check expiry
        if position and date >= position["expiry"]:
            n = position["contracts"]
            long_payoff = max(0, S - position["long_K"])
            short_payoff = max(0, S - position["short_K"])
            pnl = (long_payoff - short_payoff) * 100 * n
            cash += pnl
            trades.append({"date": str(date.date()), "action": "spread_expired",
                           "pnl": round(pnl, 2), "price": S})
            position = None

        # Open new spread monthly
        if position is None:
            long_K = strike_from_delta(S, T_dte, iv, long_delta, kind="call")
            short_K = strike_from_delta(S, T_dte, iv, short_delta, kind="call")

            if short_K <= long_K:
                short_K = long_K * 1.05  # ensure short is OTM relative to long

            long_cost = bs_price(S, long_K, T_dte, iv, kind="call") * (1 + BA_SPREAD_FRAC)
            short_prem = bs_price(S, short_K, T_dte, iv, kind="call") * (1 - BA_SPREAD_FRAC)

            net_debit = (long_cost - short_prem) * 100

            if net_debit <= 0:
                # Credit spread — skip (shouldn't happen for bull call)
                continue

            # Max loss = net debit; max gain = (short_K - long_K)*100 - net_debit
            max_loss_per = net_debit
            n = max(1, int(0.10 * cash / max_loss_per))

            total_cost = net_debit * n
            if total_cost > 0.5 * cash:
                n = max(1, int(0.5 * cash / net_debit))
                total_cost = net_debit * n

            cash -= total_cost
            expiry = date + pd.Timedelta(days=dte_target)
            position = {
                "long_K": long_K, "short_K": short_K,
                "expiry": expiry, "contracts": n,
                "net_debit": net_debit, "open_date": date,
            }
            trades.append({"date": str(date.date()), "action": "open_bull_call",
                           "long_K": long_K, "short_K": short_K,
                           "contracts": n, "net_debit": round(net_debit, 2)})

    nav_df = pd.DataFrame(nav_series).set_index("date")
    daily_ret = nav_df["nav"].pct_change().dropna()
    return daily_ret, trades


# ═══════════════════════════════════════════════════════════════════
# Evaluation & Analysis
# ═══════════════════════════════════════════════════════════════════

def evaluate_strategy(daily_returns, label="", spy_returns=None):
    """Compute comprehensive metrics with regime analysis."""
    dr = daily_returns.copy()
    dr = dr.replace([np.inf, -np.inf], 0).fillna(0)

    if len(dr) < TRADING_DAYS or dr.std() == 0:
        return {"label": label, "sharpe": 0, "cagr": 0, "valid": False}

    ann_ret = dr.mean() * TRADING_DAYS
    ann_vol = dr.std() * np.sqrt(TRADING_DAYS)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    downside = dr[dr < 0].std() * np.sqrt(TRADING_DAYS) if (dr < 0).sum() > 10 else ann_vol
    sortino = ann_ret / downside if downside > 0 else 0

    years = len(dr) / TRADING_DAYS
    total_ret = (1 + dr).prod()
    cagr = total_ret ** (1 / years) - 1 if years > 0 else 0

    cum = (1 + dr).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    wr = (dr > 0).mean()

    gains = dr[dr > 0].sum()
    losses = abs(dr[dr < 0].sum())
    pf = gains / losses if losses > 0 else float("inf")

    # Regime analysis — use SPY returns if available
    if spy_returns is not None:
        spy_aligned = spy_returns.reindex(dr.index).fillna(0)
        # Monthly aggregation for regime classification
        monthly_strat = dr.resample("ME").sum()
        monthly_spy = spy_aligned.resample("ME").sum()

        common = monthly_strat.index.intersection(monthly_spy.index)
        if len(common) > 24:
            ms = monthly_strat.loc[common]
            mspy = monthly_spy.loc[common]

            green_mask = mspy > 0
            red_mask = mspy <= 0

            green_rets = ms[green_mask]
            red_rets = ms[red_mask]

            green_sharpe = (green_rets.mean() / green_rets.std() * np.sqrt(12)
                           if len(green_rets) > 3 and green_rets.std() > 0 else 0)
            red_sharpe = (red_rets.mean() / red_rets.std() * np.sqrt(12)
                         if len(red_rets) > 3 and red_rets.std() > 0 else 0)

            max_s = max(abs(green_sharpe), abs(red_sharpe))
            regime_gap = abs(green_sharpe - red_sharpe) / max_s if max_s > 0 else 999
        else:
            green_sharpe = red_sharpe = 0
            regime_gap = 999
    else:
        green_sharpe = red_sharpe = 0
        regime_gap = 999

    return {
        "label": label,
        "cagr": round(cagr * 100, 1),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "max_dd": round(max_dd * 100, 1),
        "calmar": round(calmar, 2),
        "ann_vol": round(ann_vol * 100, 1),
        "wr": round(wr * 100, 1),
        "pf": round(pf, 2),
        "years": round(years, 1),
        "green_sharpe": round(green_sharpe, 2),
        "red_sharpe": round(red_sharpe, 2),
        "regime_gap": round(regime_gap, 3),
        "r1_pass": regime_gap <= 0.50,
        "valid": True,
    }


def run_permutation_test(daily_returns, n_trials=200):
    """Block-bootstrap permutation test."""
    dr = np.array(daily_returns)
    dr = dr[np.isfinite(dr)]
    if len(dr) < TRADING_DAYS:
        return 1.0, []

    real_sharpe = dr.mean() / dr.std() * np.sqrt(TRADING_DAYS) if dr.std() > 0 else 0
    beat_count = 0
    shuffled_sharpes = []

    for _ in range(n_trials):
        monthly_idx = np.arange(0, len(dr), 21)
        blocks = [dr[i:i+21] for i in monthly_idx if i + 21 <= len(dr)]
        if len(blocks) < 12:
            shuf = np.random.permutation(dr)
        else:
            idx = np.random.choice(len(blocks), len(blocks), replace=True)
            shuf = np.concatenate([blocks[i] for i in idx])

        s = shuf.mean() / shuf.std() * np.sqrt(TRADING_DAYS) if shuf.std() > 0 else 0
        shuffled_sharpes.append(s)
        if s >= real_sharpe:
            beat_count += 1

    return beat_count / n_trials, shuffled_sharpes


def walk_forward_evaluate(daily_returns, train_window=252, test_window=21, label=""):
    """Walk-forward sliding window evaluation."""
    dr = daily_returns.values
    n = len(dr)
    if n < train_window + test_window * 6:
        return {"label": label, "valid": False, "reason": "insufficient_data"}

    oot_sharpes = []
    oot_returns = []

    step = 0
    while step + train_window + test_window <= n:
        test_start = step + train_window
        test_end = test_start + test_window
        test_rets = dr[test_start:test_end]

        if len(test_rets) > 0 and np.std(test_rets) > 0:
            s = np.mean(test_rets) / np.std(test_rets) * np.sqrt(TRADING_DAYS)
            oot_sharpes.append(s)
            oot_returns.extend(test_rets.tolist())

        step += test_window

    if len(oot_sharpes) < 12:
        return {"label": label, "valid": False, "reason": "too_few_folds"}

    oot_arr = np.array(oot_returns)
    if oot_arr.std() == 0:
        return {"label": label, "valid": False, "reason": "zero_variance"}

    wf_sharpe = oot_arr.mean() / oot_arr.std() * np.sqrt(TRADING_DAYS)
    wf_cagr = (1 + oot_arr.mean()) ** TRADING_DAYS - 1

    return {
        "label": label,
        "wf_sharpe": round(wf_sharpe, 2),
        "wf_cagr": round(wf_cagr * 100, 1),
        "n_folds": len(oot_sharpes),
        "oot_sharpe_mean": round(np.mean(oot_sharpes), 2),
        "oot_sharpe_std": round(np.std(oot_sharpes), 2),
        "oot_sharpe_min": round(np.min(oot_sharpes), 2),
        "pct_positive_folds": round((np.array(oot_sharpes) > 0).mean() * 100, 1),
        "valid": True,
    }


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 80)
    print("OPTIONS-ENHANCED GROWTH STRATEGY BACKTEST")
    print("HC #696 — Growth Research: Options Overlay on Growth ETFs")
    print("BS pricing with VIX proxy | Commission-free (HC #694)")
    print("=" * 80)

    prices = download_data()
    spy_ret = prices["SPY"].pct_change().dropna() if "SPY" in prices.columns else None

    # ─── Run all strategies with parameter variations ───
    all_results = []
    all_wf = []
    strategy_returns = {}

    strategies = []

    # 1. Wheel on TQQQ — sweep delta/DTE
    if "TQQQ" in prices.columns:
        for put_d in [0.20, 0.25, 0.30]:
            for call_d in [0.25, 0.30, 0.35]:
                for dte in [30, 35, 45]:
                    label = f"wheel_tqqq_pd{int(put_d*100)}_cd{int(call_d*100)}_dte{dte}"
                    strategies.append(("wheel_tqqq", label, {"dte_target": dte,
                                      "put_delta": put_d, "call_delta": call_d}))

    # 2. Collar on QQQ — sweep delta
    if "QQQ" in prices.columns:
        for call_d in [0.25, 0.30, 0.35]:
            for put_d in [0.15, 0.20, 0.25]:
                for dte in [30, 35, 45]:
                    label = f"collar_qqq_cd{int(call_d*100)}_pd{int(put_d*100)}_dte{dte}"
                    strategies.append(("collar_qqq", label, {"dte_target": dte,
                                      "call_delta": call_d, "put_delta": put_d}))

    # 3. Put selling for entry
    if "TQQQ" in prices.columns:
        for put_d in [0.20, 0.25, 0.30]:
            for call_d in [0.25, 0.30, 0.35]:
                for dte in [30, 35, 45]:
                    label = f"put_entry_tqqq_pd{int(put_d*100)}_cd{int(call_d*100)}_dte{dte}"
                    strategies.append(("put_entry", label, {"dte_target": dte,
                                      "put_delta": put_d, "call_delta": call_d}))

    # 4. Synthetic long — sweep hedge delta
    if "QQQ" in prices.columns:
        for hedge_d in [0.05, 0.10, 0.15]:
            for dte in [30, 35, 45]:
                label = f"synthetic_qqq_hd{int(hedge_d*100)}_dte{dte}"
                strategies.append(("synthetic", label, {"dte_target": dte,
                                  "hedge_delta": hedge_d}))

    # 5. Diversified CSPs — sweep delta
    for put_d in [0.20, 0.25, 0.30]:
        for dte in [30, 35, 45]:
            label = f"div_csps_pd{int(put_d*100)}_dte{dte}"
            strategies.append(("div_csps", label, {"dte_target": dte,
                              "put_delta": put_d}))

    # 6. Bull call spreads — QQQ and TQQQ
    for ticker in ["QQQ", "TQQQ"]:
        if ticker not in prices.columns:
            continue
        for long_d in [0.45, 0.50]:
            for short_d in [0.20, 0.25, 0.30]:
                for dte in [30, 35, 45]:
                    label = f"bull_call_{ticker}_ld{int(long_d*100)}_sd{int(short_d*100)}_dte{dte}"
                    strategies.append(("bull_call", label, {"ticker": ticker,
                                      "dte_target": dte, "long_delta": long_d,
                                      "short_delta": short_d}))

    print(f"\nTotal strategy configs: {len(strategies)}")
    print()

    # Run strategies
    for i, (stype, label, params) in enumerate(strategies):
        try:
            if stype == "wheel_tqqq":
                dr, trades = strategy_wheel_tqqq(prices, **params)
            elif stype == "collar_qqq":
                dr, trades = strategy_collar_qqq(prices, **params)
            elif stype == "put_entry":
                dr, trades = strategy_put_entry_tqqq(prices, **params)
            elif stype == "synthetic":
                dr, trades = strategy_synthetic_long(prices, **params)
            elif stype == "div_csps":
                dr, trades = strategy_diversified_csps(prices, **params)
            elif stype == "bull_call":
                dr, trades = strategy_bull_call_spread(prices, **params)
            else:
                continue

            if len(dr) < TRADING_DAYS:
                all_results.append({"label": label, "valid": False, "reason": "insufficient"})
                continue

            result = evaluate_strategy(dr, label=label, spy_returns=spy_ret)
            result["strategy_type"] = stype
            result["params"] = params
            result["n_trades"] = len(trades)
            all_results.append(result)

            # Walk-forward
            wf = walk_forward_evaluate(dr, label=label)
            all_wf.append(wf)

            # Store returns for top configs
            strategy_returns[label] = dr

            if (i + 1) % 25 == 0:
                valid_so_far = [r for r in all_results if r.get("valid")]
                best = max(valid_so_far, key=lambda x: x.get("sharpe", -999)) if valid_so_far else None
                best_s = best["sharpe"] if best else 0
                print(f"  [{i+1}/{len(strategies)}] best Sharpe so far: {best_s:.2f}")

        except Exception as e:
            all_results.append({"label": label, "valid": False, "error": str(e)})

    # ─── Sort and display ───
    valid = [r for r in all_results if r.get("valid")]
    valid.sort(key=lambda x: -x["sharpe"])

    print(f"\n{'='*110}")
    print(f"TOP 25 CONFIGS BY SHARPE (of {len(valid)} valid)")
    print(f"{'='*110}")
    print(f"{'Label':<50} {'Type':<12} {'CAGR':>6} {'Sharpe':>7} {'Sort':>6} {'MaxDD':>7} "
          f"{'Cal':>5} {'WR':>5} {'R1':>4}")
    print("-" * 110)

    for r in valid[:25]:
        r1 = "Y" if r.get("r1_pass") else "N"
        stype = r.get("strategy_type", "?")[:11]
        print(f"{r['label']:<50} {stype:<12} {r['cagr']:>5.1f}% {r['sharpe']:>7.2f} "
              f"{r['sortino']:>6.2f} {r['max_dd']:>6.1f}% {r['calmar']:>5.2f} "
              f"{r['wr']:>4.1f}% {r1:>4}")

    # ─── Benchmarks ───
    print(f"\n{'='*80}")
    print("BENCHMARKS")
    print(f"{'='*80}")

    benchmarks = {}
    for ticker in ["QQQ", "TQQQ", "SPY"]:
        if ticker in prices.columns:
            ret = prices[ticker].pct_change().dropna()
            ret = ret.iloc[TRADING_DAYS:]
            metrics = evaluate_strategy(ret, label=f"{ticker}_buy_hold", spy_returns=spy_ret)
            benchmarks[ticker] = metrics
            print(f"  {ticker} B&H: CAGR={metrics['cagr']:.1f}%, Sharpe={metrics['sharpe']:.2f}, "
                  f"Sortino={metrics['sortino']:.2f}, MaxDD={metrics['max_dd']:.1f}%, "
                  f"Calmar={metrics['calmar']:.2f}")

    # ─── R1 Analysis ───
    r1_passing = [r for r in valid if r.get("r1_pass") and r["sharpe"] > 0.3]
    r1_passing.sort(key=lambda x: -x["sharpe"])

    print(f"\n{'='*80}")
    print(f"R1-PASSING CONFIGS (regime gap <= 0.50): {len(r1_passing)}")
    print(f"{'='*80}")

    for r in r1_passing[:15]:
        print(f"  {r['label']}: CAGR={r['cagr']:.1f}%, Sharpe={r['sharpe']:.2f}, "
              f"MaxDD={r['max_dd']:.1f}%, gap={r['regime_gap']:.3f}, "
              f"green={r['green_sharpe']:.2f}, red={r['red_sharpe']:.2f}")

    # ─── Walk-forward results for top configs ───
    valid_wf = [w for w in all_wf if w.get("valid")]
    valid_wf.sort(key=lambda x: -x.get("wf_sharpe", -999))

    print(f"\n{'='*80}")
    print(f"TOP 15 WALK-FORWARD RESULTS")
    print(f"{'='*80}")

    for w in valid_wf[:15]:
        print(f"  {w['label']}: WF_Sharpe={w['wf_sharpe']:.2f}, WF_CAGR={w['wf_cagr']:.1f}%, "
              f"folds={w['n_folds']}, pct_pos={w['pct_positive_folds']:.0f}%")

    # ─── Strategy type comparison ───
    print(f"\n{'='*80}")
    print("STRATEGY TYPE SUMMARY")
    print(f"{'='*80}")

    type_groups = {}
    for r in valid:
        stype = r.get("strategy_type", "unknown")
        if stype not in type_groups:
            type_groups[stype] = []
        type_groups[stype].append(r)

    print(f"{'Type':<18} {'Count':>5} {'Best CAGR':>10} {'Best Sharpe':>12} {'Med Sharpe':>11} "
          f"{'R1 pass':>8}")
    print("-" * 70)
    for stype, group in sorted(type_groups.items()):
        sharpes = [g["sharpe"] for g in group]
        cagrs = [g["cagr"] for g in group]
        r1_ct = sum(1 for g in group if g.get("r1_pass"))
        print(f"{stype:<18} {len(group):>5} {max(cagrs):>9.1f}% {max(sharpes):>12.2f} "
              f"{np.median(sharpes):>11.2f} {r1_ct:>8}")

    # ─── Permutation test top 3 R1-passing from each strategy type ───
    print(f"\n{'='*80}")
    print("PERMUTATION TESTS — Top R1-passing configs")
    print(f"{'='*80}")

    perm_candidates = []
    tested_types = set()
    for r in r1_passing:
        stype = r.get("strategy_type", "unknown")
        type_count = sum(1 for c in perm_candidates if c.get("strategy_type") == stype)
        if type_count < 1:  # 1 per type, up to ~6 types
            perm_candidates.append(r)
        if len(perm_candidates) >= 6:
            break

    # Also add top 3 overall by Sharpe if not already included
    for r in valid[:3]:
        if r["label"] not in [p["label"] for p in perm_candidates]:
            perm_candidates.append(r)

    perm_candidates = perm_candidates[:8]  # cap at 8

    for r in perm_candidates:
        if r["label"] in strategy_returns:
            dr = strategy_returns[r["label"]]
            p_val, _ = run_permutation_test(dr.values, n_trials=200)
            r["permutation_p"] = p_val
            status = "PASS" if p_val <= 0.05 else "FAIL"
            print(f"  {r['label']}: p={p_val:.3f} {'<-- SIGNIFICANT' if p_val <= 0.05 else ''} ({status})")

    # ─── Regime deep dive on top config ───
    if r1_passing and spy_ret is not None:
        top = r1_passing[0]
        if top["label"] in strategy_returns:
            print(f"\n{'='*80}")
            print(f"REGIME DEEP DIVE: {top['label']}")
            print(f"{'='*80}")

            dr = strategy_returns[top["label"]]
            spy_a = spy_ret.reindex(dr.index).fillna(0)

            # Quarterly regime analysis
            quarterly_strat = dr.resample("QE").sum()
            quarterly_spy = spy_a.resample("QE").sum()

            common = quarterly_strat.index.intersection(quarterly_spy.index)
            if len(common) > 8:
                qs = quarterly_strat.loc[common]
                qspy = quarterly_spy.loc[common]

                print(f"\n  Quarterly performance by SPY regime:")
                print(f"  {'Quarter':<12} {'SPY':>8} {'Strategy':>10} {'Regime':>8}")
                print(f"  {'-'*42}")

                for q in common[-20:]:  # last 20 quarters
                    spy_q = qspy.loc[q] * 100
                    strat_q = qs.loc[q] * 100
                    regime = "GREEN" if spy_q > 0 else "RED"
                    print(f"  {str(q):<12} {spy_q:>7.1f}% {strat_q:>9.1f}% {regime:>8}")

    # ─── VS Benchmark analysis ───
    print(f"\n{'='*80}")
    print("TOP CONFIG VS BENCHMARKS — RISK-ADJUSTED COMPARISON")
    print(f"{'='*80}")

    if valid:
        top3 = valid[:3]
        for r in top3:
            print(f"\n  {r['label']}:")
            print(f"    CAGR={r['cagr']:.1f}%, Sharpe={r['sharpe']:.2f}, "
                  f"Sortino={r['sortino']:.2f}, MaxDD={r['max_dd']:.1f}%")
            for bname, bm in benchmarks.items():
                sharpe_diff = r['sharpe'] - bm['sharpe']
                cagr_diff = r['cagr'] - bm['cagr']
                dd_diff = r['max_dd'] - bm['max_dd']
                print(f"    vs {bname} B&H: Sharpe {'+' if sharpe_diff > 0 else ''}{sharpe_diff:.2f}, "
                      f"CAGR {'+' if cagr_diff > 0 else ''}{cagr_diff:.1f}%, "
                      f"MaxDD {'+' if dd_diff > 0 else ''}{dd_diff:.1f}%")

    # ─── Save results ───
    elapsed = time.time() - t0
    output = {
        "generated": pd.Timestamp.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "n_configs": len(strategies),
        "n_valid": len(valid),
        "n_r1_passing": len(r1_passing),
        "commission": "zero_hc694",
        "pricing": "black_scholes_vix_proxy",
        "ba_spread_frac": BA_SPREAD_FRAC,
        "benchmarks": benchmarks,
        "top_25": valid[:25],
        "r1_passing": r1_passing[:15],
        "walk_forward_top15": valid_wf[:15],
        "strategy_type_summary": {
            stype: {
                "count": len(group),
                "best_sharpe": max(g["sharpe"] for g in group),
                "best_cagr": max(g["cagr"] for g in group),
                "median_sharpe": round(float(np.median([g["sharpe"] for g in group])), 2),
                "r1_pass_count": sum(1 for g in group if g.get("r1_pass")),
            }
            for stype, group in type_groups.items()
        },
        "all_results": valid,
    }

    with open(OUT_DIR / "options_growth_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Save equity curves for top configs
    top_labels = [r["label"] for r in valid[:10]]
    curves = {}
    for label in top_labels:
        if label in strategy_returns:
            cum = (1 + strategy_returns[label]).cumprod()
            curves[label] = {str(k): v for k, v in cum.to_dict().items()}

    with open(OUT_DIR / "equity_curves_top10.json", "w") as f:
        json.dump(curves, f, indent=2, default=str)

    print(f"\n{'='*80}")
    print(f"COMPLETE. {len(valid)} valid configs tested in {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"Results saved to {OUT_DIR}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
