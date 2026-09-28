#!/usr/bin/env python3
"""
Income Portfolio Optimizer v2 — HONEST VERSION
===============================================
Fixes critical flaws from v1:

1. COVERED CALL: Now tracks TOTAL return (stock P&L + premium income).
   CC requires owning 100 shares per contract. Stock price changes are REAL P&L.
   During crashes, portfolio takes full stock drawdown minus tiny premium cushion.
   If called away (price > strike), gain is capped at (strike - entry) + premium.

2. IRON CONDOR IV: Uses per-ticker IV estimation by scaling VIX with a beta factor.
   High-beta stocks (NVDA, META) have higher IV; low-beta (KO, JNJ) have lower IV.

3. DELTA CALCULATION: Fixed to properly target 0.20 delta using correct BS inversion.
   Old code had sign error in strike_from_delta_call.

4. VRP: Adds 5% annual borrow cost for short VXX positions.

5. TAIL CORRELATION: During VIX spikes (>30), all strategies take correlated hits.
   IC losses + CC stock losses happen simultaneously in crashes.

Same 40/40/20 allocation. Same gates. Runs 2018-2026.
"""

import json
import warnings
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INITIAL_CAPITAL = 100_000
ALLOC_IC = 0.40    # 40% margin for iron condors
ALLOC_CSP = 0.00   # 0% CSP — removed
ALLOC_CC = 0.40    # 40% covered calls (HOLDS STOCK — total return)
ALLOC_VRP = 0.20   # 20% VRP overlay

COMMISSION_PER_CONTRACT = 0.65  # per leg per contract
BID_ASK_COST = 0.10             # 10% of premium lost to spread
ASSIGNMENT_RISK = 0.02
TAKE_PROFIT_PCT = 0.50
STOP_LOSS_MULT = 1.5
RISK_FREE_RATE = 0.04

# IC parameters
IC_DTE = 7
IC_DELTA = 0.20
IC_WIDTH_PCT = 0.05
IC_NUM_UNDERLYINGS = 5

# CC parameters
CC_DTE = 30
CC_DELTA = 0.30
CC_SMA_PERIOD = 20

# VRP parameters
VRP_CONTANGO_THRESHOLD = 0.07
VRP_ANNUAL_BORROW_COST = 0.05  # 5% annual borrow cost for short VXX

# Per-ticker IV beta multipliers (scale VIX to approximate individual IV)
# Calibrated from historical IV vs VIX relationships
TICKER_IV_BETA = {
    "SPY": 1.00, "QQQ": 1.15, "IWM": 1.20,
    "AAPL": 1.25, "MSFT": 1.15, "AMZN": 1.40, "GOOGL": 1.30, "META": 1.60,
    "NVDA": 1.80, "JPM": 1.25, "V": 1.10, "JNJ": 0.70, "UNH": 0.90,
    "PG": 0.65, "HD": 1.10, "MA": 1.10, "KO": 0.60, "PEP": 0.60,
    "WMT": 0.70, "COST": 0.85, "ABBV": 0.90, "MRK": 0.80, "LLY": 1.20,
}

IC_TICKERS = ["SPY", "QQQ", "IWM", "AAPL", "MSFT"]

CC_CSP_TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "JPM", "V",
    "JNJ", "UNH", "PG", "HD", "MA", "KO", "PEP", "WMT", "COST",
    "ABBV", "MRK", "LLY"
]

VIX_TICKER = "^VIX"
VXX_TICKER = "VIXY"
SPY_TICKER = "SPY"

# ---------------------------------------------------------------------------
# Black-Scholes helpers (FIXED)
# ---------------------------------------------------------------------------

def bs_call_price(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_put_price(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def bs_call_delta(S, K, T, r, sigma):
    """Call delta = N(d1)."""
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)

def bs_put_delta(S, K, T, r, sigma):
    """Put delta = N(d1) - 1 (negative). Returns absolute value."""
    if T <= 0 or sigma <= 0:
        return 1.0 if S < K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return abs(norm.cdf(d1) - 1.0)

def strike_from_delta_call(S, T, r, sigma, delta_target):
    """Find call strike for a given delta using root finding.

    Call delta = N(d1) = delta_target
    Solve for K such that N(d1(K)) = delta_target.
    Higher K -> lower delta (more OTM).
    For CC with delta=0.30, strike should be above spot.
    """
    if T <= 0 or sigma <= 0:
        return S
    try:
        def objective(K):
            return bs_call_delta(S, K, T, r, sigma) - delta_target
        # Search for strike: for OTM call, K > S
        K_low = S * 0.8
        K_high = S * 1.5
        K = brentq(objective, K_low, K_high)
        return K
    except (ValueError, RuntimeError):
        # Fallback: approximate using inverse normal
        d1_target = norm.ppf(delta_target)
        K = S * np.exp(-d1_target * sigma * np.sqrt(T) + (r + 0.5 * sigma**2) * T)
        return K

def strike_from_delta_put(S, T, r, sigma, delta_target):
    """Find put strike for a given |delta| using root finding.

    |Put delta| = N(-d1) = delta_target
    For OTM put with delta=0.20, strike should be below spot.
    """
    if T <= 0 or sigma <= 0:
        return S
    try:
        def objective(K):
            return bs_put_delta(S, K, T, r, sigma) - delta_target
        K_low = S * 0.5
        K_high = S * 1.1
        K = brentq(objective, K_low, K_high)
        return K
    except (ValueError, RuntimeError):
        d1_target = norm.ppf(1 - delta_target)
        K = S * np.exp(-d1_target * sigma * np.sqrt(T) + (r + 0.5 * sigma**2) * T)
        return K

def get_ticker_iv(vix_val, ticker):
    """Get per-ticker IV by scaling VIX with beta factor."""
    beta = TICKER_IV_BETA.get(ticker, 1.0)
    return (vix_val / 100.0) * beta

def ic_credit_estimate(vix_level, width_pct):
    """Estimate iron condor credit as % of width."""
    base_credit_ratio = 0.25
    vix_scalar = vix_level / 20.0
    credit_ratio = min(base_credit_ratio * vix_scalar, 0.50)
    return credit_ratio


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def download_data(start="2017-06-01", end="2026-07-23"):
    print("Downloading price data...")
    all_tickers = list(set(IC_TICKERS + CC_CSP_TICKERS + [SPY_TICKER, VXX_TICKER]))
    stock_data = yf.download(all_tickers, start=start, end=end, auto_adjust=True, progress=False)
    vix_data = yf.download(VIX_TICKER, start=start, end=end, auto_adjust=True, progress=False)
    vxx_data = yf.download(VXX_TICKER, start=start, end=end, auto_adjust=True, progress=False)
    return stock_data, vix_data, vxx_data


# ---------------------------------------------------------------------------
# Strategy simulators
# ---------------------------------------------------------------------------

class IronCondorStrategy:
    """
    7-day iron condor on liquid underlyings.
    FIXED: Uses per-ticker IV (VIX * beta) instead of raw VIX for all.
    FIXED: Proper delta targeting via root-finding.
    """

    def __init__(self, capital, stock_prices, vix_series):
        self.capital = capital
        self.prices = stock_prices
        self.vix = vix_series
        self.positions = []

    def simulate(self, dates):
        monthly_pnl = {}
        nav = self.capital
        nav_series = []
        prev_vix = None

        for i, date in enumerate(dates):
            if date not in self.vix.index:
                nav_series.append((date, nav))
                continue

            vix_val = self.vix.loc[date]
            if np.isnan(vix_val):
                nav_series.append((date, nav))
                continue

            closed_pnl = self._manage_positions(date, vix_val)
            nav += closed_pnl

            month_key = date.strftime("%Y-%m")
            if month_key not in monthly_pnl:
                monthly_pnl[month_key] = 0
            monthly_pnl[month_key] += closed_pnl

            vix_spike = False
            if prev_vix is not None and (vix_val - prev_vix) > 3:
                vix_spike = True

            if date.weekday() == 0 and len(self.positions) < IC_NUM_UNDERLYINGS and not vix_spike:
                self._open_positions(date, vix_val, nav)

            if date.weekday() == 0:
                prev_vix = vix_val

            nav_series.append((date, nav))

        if self.positions and len(dates) > 0:
            final_pnl = self._close_all(dates[-1])
            nav += final_pnl
            month_key = dates[-1].strftime("%Y-%m")
            if month_key not in monthly_pnl:
                monthly_pnl[month_key] = 0
            monthly_pnl[month_key] += final_pnl

        return monthly_pnl, nav_series

    def _open_positions(self, date, vix_val, current_nav):
        if vix_val > 40:
            return

        per_underlying_capital = self.capital / IC_NUM_UNDERLYINGS
        adaptive_width = IC_WIDTH_PCT * max(1.0, vix_val / 20.0)
        adaptive_width = min(adaptive_width, 0.10)

        opened = 0
        for ticker in IC_TICKERS:
            if opened >= IC_NUM_UNDERLYINGS - len(self.positions):
                break
            if ticker not in self.prices.columns or date not in self.prices.index:
                continue

            price = self.prices.loc[date, ticker]
            if np.isnan(price):
                continue

            # FIXED: Per-ticker IV instead of raw VIX/100
            sigma = get_ticker_iv(vix_val, ticker)
            T = IC_DTE / 365

            # Use proper delta-based strike placement
            call_strike = strike_from_delta_call(price, T, RISK_FREE_RATE, sigma, IC_DELTA)
            put_strike = strike_from_delta_put(price, T, RISK_FREE_RATE, sigma, IC_DELTA)

            # Wing width
            width_per_share = price * adaptive_width

            # Credit = short call premium + short put premium - long wing premiums
            short_call_prem = bs_call_price(price, call_strike, T, RISK_FREE_RATE, sigma)
            long_call_prem = bs_call_price(price, call_strike + width_per_share, T, RISK_FREE_RATE, sigma)
            short_put_prem = bs_put_price(price, put_strike, T, RISK_FREE_RATE, sigma)
            long_put_prem = bs_put_price(price, put_strike - width_per_share, T, RISK_FREE_RATE, sigma)

            credit_per_share = (short_call_prem - long_call_prem) + (short_put_prem - long_put_prem)
            if credit_per_share <= 0:
                continue

            margin_per_contract = width_per_share * 100
            if margin_per_contract <= 0:
                continue
            max_contracts = 3 if vix_val < 25 else 1
            num_contracts = max(1, min(max_contracts, int(per_underlying_capital / margin_per_contract)))

            gross_credit = credit_per_share * 100 * num_contracts
            open_commission = 4 * COMMISSION_PER_CONTRACT * num_contracts
            spread_cost = gross_credit * BID_ASK_COST
            net_credit = gross_credit - open_commission - spread_cost

            if net_credit <= 0:
                continue

            max_loss_total = (width_per_share - credit_per_share) * 100 * num_contracts

            self.positions.append({
                "ticker": ticker,
                "open_date": date,
                "expiry": date + pd.Timedelta(days=IC_DTE),
                "price_at_open": price,
                "call_strike": call_strike,
                "put_strike": put_strike,
                "width_per_share": width_per_share,
                "width_pct": adaptive_width,
                "net_credit": net_credit,
                "credit_per_share": credit_per_share,
                "max_loss": max_loss_total,
                "num_contracts": num_contracts,
                "sigma": sigma,
            })
            opened += 1

    def _manage_positions(self, date, vix_val):
        total_pnl = 0
        remaining = []

        for pos in self.positions:
            ticker = pos["ticker"]
            if ticker not in self.prices.columns or date not in self.prices.index:
                remaining.append(pos)
                continue

            current_price = self.prices.loc[date, ticker]
            if np.isnan(current_price):
                remaining.append(pos)
                continue

            days_held = (date - pos["open_date"]).days
            T_remaining = max((IC_DTE - days_held) / 365, 0.001)

            # Current IV for this ticker
            sigma = get_ticker_iv(vix_val, ticker)

            # Compute current spread values using BS
            call_strike = pos["call_strike"]
            put_strike = pos["put_strike"]
            width = pos["width_per_share"]

            # Call spread value (short call - long call)
            short_call_val = bs_call_price(current_price, call_strike, T_remaining, RISK_FREE_RATE, sigma)
            long_call_val = bs_call_price(current_price, call_strike + width, T_remaining, RISK_FREE_RATE, sigma)
            call_spread_val = short_call_val - long_call_val

            # Put spread value (short put - long put)
            short_put_val = bs_put_price(current_price, put_strike, T_remaining, RISK_FREE_RATE, sigma)
            long_put_val = bs_put_price(current_price, put_strike - width, T_remaining, RISK_FREE_RATE, sigma)
            put_spread_val = short_put_val - long_put_val

            # Buyback cost = current spread value (what we'd pay to close)
            buyback_per_share = (call_spread_val + put_spread_val)
            buyback_total = buyback_per_share * 100 * pos["num_contracts"] * (1 + BID_ASK_COST)
            close_commission = 4 * COMMISSION_PER_CONTRACT * pos["num_contracts"]

            current_pnl = pos["net_credit"] - buyback_total - close_commission
            current_pnl = max(current_pnl, -pos["max_loss"])
            current_pnl = min(current_pnl, pos["net_credit"])

            if current_pnl >= pos["net_credit"] * TAKE_PROFIT_PCT:
                total_pnl += current_pnl
                continue

            if current_pnl <= -pos["net_credit"] * STOP_LOSS_MULT:
                total_pnl += current_pnl
                continue

            if date >= pos["expiry"]:
                total_pnl += current_pnl
                continue

            remaining.append(pos)

        self.positions = remaining
        return total_pnl

    def _close_all(self, date):
        total_pnl = 0
        for pos in self.positions:
            days_held = (date - pos["open_date"]).days
            time_frac = min(days_held / IC_DTE, 1.0)
            pnl = pos["net_credit"] * time_frac * 0.5
            close_commission = 4 * COMMISSION_PER_CONTRACT * pos["num_contracts"]
            pnl -= close_commission
            total_pnl += pnl
        self.positions = []
        return total_pnl


class CoveredCallStrategy:
    """
    HONEST Covered Call: Tracks TOTAL return = stock P&L + option premium.

    CC requires OWNING 100 shares per contract. The portfolio HOLDS these shares.
    - If stock drops $10: loss = $10/share * 100 * contracts, offset by tiny premium
    - If stock rises past strike: gain capped at (strike - entry) + premium
    - If stock rises but stays below strike: gain = (price change) + premium

    During crashes, this strategy takes the FULL stock drawdown minus premium cushion.
    This is the honest reality of covered calls — they are NOT pure income.
    """

    def __init__(self, capital, stock_prices, vix_series):
        self.capital = capital
        self.prices = stock_prices
        self.vix = vix_series
        self.positions = []
        # Track stock positions separately — CC capital is INVESTED in stocks
        self.stock_holdings = {}  # ticker -> {shares, cost_basis}

    def simulate(self, dates):
        monthly_pnl = {}
        nav = self.capital
        nav_series = []

        sma20 = self.prices.rolling(CC_SMA_PERIOD).mean()

        # Track daily NAV properly (stock value + cash)
        cash = self.capital

        for i, date in enumerate(dates):
            if date not in self.vix.index:
                # Still need to mark stocks to market
                stock_value = self._mark_to_market(date)
                nav = cash + stock_value
                nav_series.append((date, nav))
                continue

            vix_val = self.vix.loc[date]
            if np.isnan(vix_val):
                stock_value = self._mark_to_market(date)
                nav = cash + stock_value
                nav_series.append((date, nav))
                continue

            month_key = date.strftime("%Y-%m")
            if month_key not in monthly_pnl:
                monthly_pnl[month_key] = 0

            # Manage existing CC positions (handle expiry/assignment)
            closed_pnl, cash_change = self._manage_positions(date, vix_val)
            cash += cash_change
            monthly_pnl[month_key] += closed_pnl

            # Open new positions on first trading day of month
            if i == 0 or dates[i-1].month != date.month:
                cash_used = self._open_positions(date, vix_val, sma20, cash)
                cash -= cash_used

            # NAV = cash + market value of all stock holdings
            stock_value = self._mark_to_market(date)
            nav = cash + stock_value
            nav_series.append((date, nav))

        # Close everything at end
        if self.positions or self.stock_holdings:
            final_pnl, cash_change = self._close_all(dates[-1], self.vix)
            cash += cash_change
            month_key = dates[-1].strftime("%Y-%m")
            if month_key not in monthly_pnl:
                monthly_pnl[month_key] = 0
            monthly_pnl[month_key] += final_pnl

        # Final NAV after closing
        stock_value = self._mark_to_market(dates[-1])
        nav = cash + stock_value

        # Convert nav_series to monthly PnL
        # monthly_pnl already tracks realized P&L
        # But we also need unrealized stock P&L changes month-over-month
        # Recalculate monthly PnL from NAV changes for accuracy
        monthly_pnl_from_nav = {}
        prev_nav = self.capital
        nav_by_date = {d: n for d, n in nav_series}

        # Group nav_series by month, take last value each month
        month_end_navs = {}
        for d, n in nav_series:
            mk = d.strftime("%Y-%m")
            month_end_navs[mk] = n  # last value wins

        prev_month_nav = self.capital
        for mk in sorted(month_end_navs.keys()):
            monthly_pnl_from_nav[mk] = month_end_navs[mk] - prev_month_nav
            prev_month_nav = month_end_navs[mk]

        return monthly_pnl_from_nav, nav_series

    def _mark_to_market(self, date):
        """Get current market value of all stock holdings."""
        total = 0
        for ticker, holding in self.stock_holdings.items():
            if ticker in self.prices.columns and date in self.prices.index:
                price = self.prices.loc[date, ticker]
                if not np.isnan(price):
                    total += price * holding["shares"]
                else:
                    total += holding["cost_basis"] * holding["shares"]
            else:
                total += holding["cost_basis"] * holding["shares"]
        return total

    def _open_positions(self, date, vix_val, sma20, available_cash):
        """Open CC positions: buy stock + sell call."""
        # Allocate across up to 10 stocks
        per_stock_budget = self.capital / 10
        total_cash_used = 0

        active_tickers = {p["ticker"] for p in self.positions}

        for ticker in CC_CSP_TICKERS[:10]:
            if ticker in active_tickers:
                continue
            if ticker not in self.prices.columns or date not in self.prices.index:
                continue

            price = self.prices.loc[date, ticker]
            if date not in sma20.index or ticker not in sma20.columns:
                continue
            sma_val = sma20.loc[date, ticker]
            if np.isnan(price) or np.isnan(sma_val):
                continue

            # Only sell calls if above 20 SMA
            if price < sma_val:
                continue

            # FIXED: Per-ticker IV
            sigma = get_ticker_iv(vix_val, ticker)
            T = CC_DTE / 365

            # FIXED: Proper delta targeting
            K = strike_from_delta_call(price, T, RISK_FREE_RATE, sigma, CC_DELTA)
            premium_per_share = bs_call_price(price, K, T, RISK_FREE_RATE, sigma)

            # How many contracts can we afford? Need to BUY 100 shares per contract
            cost_per_contract = price * 100  # cost to buy 100 shares
            max_by_budget = int(per_stock_budget / cost_per_contract)
            max_by_cash = int(available_cash / cost_per_contract)
            num_contracts = max(1, min(max_by_budget, max_by_cash))

            if num_contracts < 1 or available_cash < cost_per_contract:
                continue

            shares = num_contracts * 100
            stock_cost = price * shares

            gross_premium = premium_per_share * 100 * num_contracts
            commission = COMMISSION_PER_CONTRACT * num_contracts
            spread_cost = gross_premium * BID_ASK_COST
            net_premium = gross_premium - commission - spread_cost

            if net_premium <= 0:
                continue

            # Buy the stock
            if ticker not in self.stock_holdings:
                self.stock_holdings[ticker] = {"shares": 0, "cost_basis": 0}

            # Update cost basis (weighted average)
            old_shares = self.stock_holdings[ticker]["shares"]
            old_basis = self.stock_holdings[ticker]["cost_basis"]
            new_total_shares = old_shares + shares
            if new_total_shares > 0:
                self.stock_holdings[ticker]["cost_basis"] = (
                    (old_basis * old_shares + price * shares) / new_total_shares
                )
            self.stock_holdings[ticker]["shares"] = new_total_shares

            # Record the call option position
            self.positions.append({
                "ticker": ticker,
                "open_date": date,
                "expiry": date + pd.Timedelta(days=CC_DTE),
                "price_at_open": price,
                "strike": K,
                "premium": net_premium,
                "premium_per_share": premium_per_share,
                "num_contracts": num_contracts,
                "shares": shares,
                "sigma": sigma,
            })

            # Cash flow: spend on stock, receive premium
            cash_used = stock_cost - net_premium
            total_cash_used += cash_used
            available_cash -= cash_used

        return total_cash_used

    def _manage_positions(self, date, vix_val):
        """Manage CC positions. Returns (realized_pnl, cash_change)."""
        total_pnl = 0
        total_cash_change = 0
        remaining = []

        for pos in self.positions:
            ticker = pos["ticker"]
            if ticker not in self.prices.columns or date not in self.prices.index:
                remaining.append(pos)
                continue

            current_price = self.prices.loc[date, ticker]
            if np.isnan(current_price):
                remaining.append(pos)
                continue

            days_held = (date - pos["open_date"]).days
            sigma = get_ticker_iv(vix_val, ticker)
            T_remaining = max((CC_DTE - days_held) / 365, 0.001)

            if date >= pos["expiry"]:
                if current_price <= pos["strike"]:
                    # OTM — call expires worthless, keep premium, keep stock
                    # Realized income = premium. Stock P&L is unrealized (still holding).
                    # But we want to cycle — sell stock and re-enter next month
                    stock_pnl = (current_price - pos["price_at_open"]) * pos["shares"]
                    option_income = pos["premium"]
                    pnl = stock_pnl + option_income

                    # Sell the stock (cash back)
                    cash_back = current_price * pos["shares"]
                    total_cash_change += cash_back

                    # Remove from holdings
                    if ticker in self.stock_holdings:
                        self.stock_holdings[ticker]["shares"] -= pos["shares"]
                        if self.stock_holdings[ticker]["shares"] <= 0:
                            del self.stock_holdings[ticker]
                else:
                    # ITM — called away at strike price
                    # Gain capped at (strike - entry) + premium
                    stock_pnl = (pos["strike"] - pos["price_at_open"]) * pos["shares"]
                    option_income = pos["premium"]
                    pnl = stock_pnl + option_income

                    # Stock sold at strike price
                    cash_back = pos["strike"] * pos["shares"]
                    assignment_cost = COMMISSION_PER_CONTRACT * pos["num_contracts"]
                    cash_back -= assignment_cost
                    total_cash_change += cash_back

                    # Remove from holdings
                    if ticker in self.stock_holdings:
                        self.stock_holdings[ticker]["shares"] -= pos["shares"]
                        if self.stock_holdings[ticker]["shares"] <= 0:
                            del self.stock_holdings[ticker]

                total_pnl += pnl
                continue

            # Early management: buy back call if 50% profit on option
            current_opt_val = bs_call_price(current_price, pos["strike"], T_remaining, RISK_FREE_RATE, sigma)
            buyback_per_share = current_opt_val * (1 + BID_ASK_COST)
            sold_per_share = pos["premium_per_share"] * (1 - BID_ASK_COST)

            if buyback_per_share <= sold_per_share * (1 - TAKE_PROFIT_PCT):
                # Buy back the call only — keep the stock for next cycle
                buyback_total = buyback_per_share * 100 * pos["num_contracts"]
                commission = COMMISSION_PER_CONTRACT * pos["num_contracts"]
                option_pnl = pos["premium"] - buyback_total - commission

                # Stock P&L realized when we sell at cycle end
                stock_pnl = (current_price - pos["price_at_open"]) * pos["shares"]
                pnl = stock_pnl + option_pnl

                # Sell stock, get cash back
                cash_back = current_price * pos["shares"]
                total_cash_change += cash_back

                if ticker in self.stock_holdings:
                    self.stock_holdings[ticker]["shares"] -= pos["shares"]
                    if self.stock_holdings[ticker]["shares"] <= 0:
                        del self.stock_holdings[ticker]

                total_pnl += pnl
                continue

            remaining.append(pos)

        self.positions = remaining
        return total_pnl, total_cash_change

    def _close_all(self, date, vix_series):
        """Close all remaining positions and sell all stock."""
        total_pnl = 0
        total_cash = 0

        for pos in self.positions:
            ticker = pos["ticker"]
            if ticker in self.prices.columns and date in self.prices.index:
                current_price = self.prices.loc[date, ticker]
                if not np.isnan(current_price):
                    stock_pnl = (current_price - pos["price_at_open"]) * pos["shares"]
                    days_held = (date - pos["open_date"]).days
                    theta_frac = min(days_held / CC_DTE, 0.9)
                    option_income = pos["premium"] * theta_frac * 0.6
                    total_pnl += stock_pnl + option_income
                    total_cash += current_price * pos["shares"]

                    if ticker in self.stock_holdings:
                        self.stock_holdings[ticker]["shares"] -= pos["shares"]
                        if self.stock_holdings[ticker]["shares"] <= 0:
                            del self.stock_holdings[ticker]

        # Sell any remaining stock holdings
        for ticker in list(self.stock_holdings.keys()):
            holding = self.stock_holdings[ticker]
            if ticker in self.prices.columns and date in self.prices.index:
                price = self.prices.loc[date, ticker]
                if not np.isnan(price):
                    total_cash += price * holding["shares"]
                    total_pnl += (price - holding["cost_basis"]) * holding["shares"]
            del self.stock_holdings[ticker]

        self.positions = []
        return total_pnl, total_cash


class VRPOverlay:
    """
    Short VXX/VIXY when VIX term structure in contango > 7%.
    FIXED: Adds 5% annual borrow cost for short VXX positions.
    """

    def __init__(self, capital, vix_series, vxx_prices):
        self.capital = capital
        self.vix = vix_series
        self.vxx = vxx_prices
        self.position = None

    def simulate(self, dates):
        monthly_pnl = {}
        nav = self.capital
        nav_series = []

        vix_sma = self.vix.rolling(20).mean()

        for i, date in enumerate(dates):
            month_key = date.strftime("%Y-%m")
            if month_key not in monthly_pnl:
                monthly_pnl[month_key] = 0

            if date not in self.vix.index or date not in self.vxx.index:
                nav_series.append((date, nav))
                continue

            vix_val = self.vix.loc[date]
            vxx_val = self.vxx.loc[date]

            if np.isnan(vix_val) or np.isnan(vxx_val) or vxx_val <= 0:
                nav_series.append((date, nav))
                continue

            if date in vix_sma.index:
                sma_val = vix_sma.loc[date]
                if not np.isnan(sma_val) and sma_val > 0:
                    contango = (sma_val - vix_val) / sma_val
                else:
                    contango = 0
            else:
                contango = 0

            if self.position is not None:
                entry_price = self.position["entry_price"]
                direction = self.position["direction"]

                if direction == -1:
                    pnl_pct = (entry_price - vxx_val) / entry_price
                else:
                    pnl_pct = (vxx_val - entry_price) / entry_price

                days_held = (date - self.position["open_date"]).days

                # FIXED: Deduct borrow cost for short positions
                if direction == -1:
                    daily_borrow = VRP_ANNUAL_BORROW_COST / 252
                    borrow_cost_pct = daily_borrow * days_held
                    pnl_pct -= borrow_cost_pct

                exit = False
                if direction == -1:
                    if contango < 0.02 or pnl_pct < -0.15 or days_held > 30:
                        exit = True
                    elif pnl_pct > 0.10:
                        exit = True
                else:
                    if contango > 0.02 or pnl_pct < -0.10 or days_held > 10:
                        exit = True
                    elif pnl_pct > 0.15:
                        exit = True

                if exit:
                    realized_pnl = pnl_pct * self.position["size"]
                    commission = self.position["size"] * 0.001
                    realized_pnl -= commission
                    nav += realized_pnl
                    monthly_pnl[month_key] += realized_pnl
                    self.position = None

            if self.position is None:
                if contango > VRP_CONTANGO_THRESHOLD:
                    position_size = min(self.capital * 0.5, nav * 0.1)
                    self.position = {
                        "open_date": date,
                        "entry_price": vxx_val,
                        "size": position_size,
                        "direction": -1,
                    }
                elif contango < -0.05:
                    position_size = min(self.capital * 0.3, nav * 0.05)
                    self.position = {
                        "open_date": date,
                        "entry_price": vxx_val,
                        "size": position_size,
                        "direction": 1,
                    }

            nav_series.append((date, nav))

        if self.position is not None and len(dates) > 0:
            last_date = dates[-1]
            if last_date in self.vxx.index:
                vxx_val = self.vxx.loc[last_date]
                entry_price = self.position["entry_price"]
                direction = self.position.get("direction", -1)
                if direction == -1:
                    pnl_pct = (entry_price - vxx_val) / entry_price
                    days_held = (last_date - self.position["open_date"]).days
                    borrow_cost_pct = (VRP_ANNUAL_BORROW_COST / 252) * days_held
                    pnl_pct -= borrow_cost_pct
                else:
                    pnl_pct = (vxx_val - entry_price) / entry_price
                realized_pnl = pnl_pct * self.position["size"]
                nav += realized_pnl
                month_key = last_date.strftime("%Y-%m")
                if month_key not in monthly_pnl:
                    monthly_pnl[month_key] = 0
                monthly_pnl[month_key] += realized_pnl
            self.position = None

        return monthly_pnl, nav_series


# ---------------------------------------------------------------------------
# Tail correlation adjustment
# ---------------------------------------------------------------------------

def apply_tail_correlation(ic_pnl, cc_pnl, vrp_pnl, vix_monthly, all_months):
    """
    During high-VIX months, apply correlation penalty.
    In crashes, IC losses + CC stock losses + VRP losses all happen at once.
    The diversification benefit disappears precisely when you need it most.

    Model: when monthly avg VIX > 25, apply a 10-20% additional loss
    reflecting correlated margin calls, forced liquidations, wider spreads.
    """
    adjusted_ic = dict(ic_pnl)
    adjusted_cc = dict(cc_pnl)
    adjusted_vrp = dict(vrp_pnl)

    for month in all_months:
        if month not in vix_monthly:
            continue
        avg_vix = vix_monthly[month]
        if avg_vix > 30:
            # Severe stress: spreads widen, fills worsen, margin calls
            stress_mult = 1.0 + 0.15 * ((avg_vix - 30) / 20)  # up to 15% extra cost at VIX=50
            stress_mult = min(stress_mult, 1.30)  # cap at 30% extra

            # Only penalize losing months (stress makes losses worse, not gains)
            if month in adjusted_ic and adjusted_ic[month] < 0:
                adjusted_ic[month] *= stress_mult
            if month in adjusted_cc and adjusted_cc[month] < 0:
                adjusted_cc[month] *= stress_mult
            if month in adjusted_vrp and adjusted_vrp[month] < 0:
                adjusted_vrp[month] *= stress_mult
        elif avg_vix > 25:
            # Moderate stress
            stress_mult = 1.0 + 0.05 * ((avg_vix - 25) / 5)
            stress_mult = min(stress_mult, 1.10)

            if month in adjusted_ic and adjusted_ic[month] < 0:
                adjusted_ic[month] *= stress_mult
            if month in adjusted_cc and adjusted_cc[month] < 0:
                adjusted_cc[month] *= stress_mult
            if month in adjusted_vrp and adjusted_vrp[month] < 0:
                adjusted_vrp[month] *= stress_mult

    return adjusted_ic, adjusted_cc, adjusted_vrp


# ---------------------------------------------------------------------------
# Portfolio combiner & analytics (same as v1)
# ---------------------------------------------------------------------------

def combine_strategies(ic_pnl, cc_pnl, csp_pnl, vrp_pnl, all_months):
    combined = {}
    for m in all_months:
        combined[m] = (
            ic_pnl.get(m, 0) +
            cc_pnl.get(m, 0) +
            csp_pnl.get(m, 0) +
            vrp_pnl.get(m, 0)
        )
    return combined


def compute_nav_series(monthly_pnl, initial_capital):
    nav = initial_capital
    nav_series = []
    for month in sorted(monthly_pnl.keys()):
        nav += monthly_pnl[month]
        nav_series.append((month, nav))
    return nav_series


def compute_metrics(monthly_pnl, initial_capital, spy_monthly_returns=None):
    months = sorted(monthly_pnl.keys())
    if not months:
        return {
            "total_return_pct": 0, "cagr_pct": 0, "annualized_income_yield_pct": 0,
            "max_drawdown_pct": 0, "min_nav_pct_of_start": 100,
            "sharpe": 0, "sortino": 0, "positive_months_pct": 0,
            "num_months": 0, "avg_monthly_income": 0,
            "best_month": {"month": "N/A", "pnl": 0},
            "worst_month": {"month": "N/A", "pnl": 0},
            "final_nav": initial_capital, "regime_stats": None,
        }
    pnl_values = [monthly_pnl[m] for m in months]
    if initial_capital == 0:
        initial_capital = 1
    monthly_returns = [p / initial_capital for p in pnl_values]

    nav = initial_capital
    nav_series = []
    peak = initial_capital
    max_dd = 0
    min_nav = initial_capital

    for pnl in pnl_values:
        nav += pnl
        nav_series.append(nav)
        if nav > peak:
            peak = nav
        dd = (peak - nav) / peak
        if dd > max_dd:
            max_dd = dd
        if nav < min_nav:
            min_nav = nav

    num_years = len(months) / 12
    total_return = (nav - initial_capital) / initial_capital
    if num_years > 0 and nav > 0:
        cagr = (nav / initial_capital) ** (1 / num_years) - 1
    else:
        cagr = 0

    avg_monthly = np.mean(monthly_returns)
    std_monthly = np.std(monthly_returns) if len(monthly_returns) > 1 else 0.001

    excess_monthly = avg_monthly - RISK_FREE_RATE / 12
    sharpe = (excess_monthly / std_monthly * np.sqrt(12)) if std_monthly > 0 else 0

    downside_returns = [r for r in monthly_returns if r < 0]
    if downside_returns:
        downside_std = np.std(downside_returns)
        sortino = (excess_monthly / downside_std * np.sqrt(12)) if downside_std > 0 else 0
    else:
        sortino = float("inf")

    positive_months = sum(1 for r in monthly_returns if r > 0)
    positive_pct = positive_months / len(monthly_returns) * 100 if monthly_returns else 0

    best_month_idx = np.argmax(pnl_values)
    worst_month_idx = np.argmin(pnl_values)

    regime_stats = None
    if spy_monthly_returns is not None:
        bull_pnl = []
        bear_pnl = []
        for m in months:
            if m in spy_monthly_returns:
                if spy_monthly_returns[m] > 0:
                    bull_pnl.append(monthly_pnl[m] / initial_capital)
                else:
                    bear_pnl.append(monthly_pnl[m] / initial_capital)

        bull_avg = np.mean(bull_pnl) if bull_pnl else 0
        bear_avg = np.mean(bear_pnl) if bear_pnl else 0
        bull_sharpe = (bull_avg * 12 - RISK_FREE_RATE) / (np.std(bull_pnl) * np.sqrt(12)) if bull_pnl and np.std(bull_pnl) > 0 else 0
        bear_sharpe = (bear_avg * 12 - RISK_FREE_RATE) / (np.std(bear_pnl) * np.sqrt(12)) if bear_pnl and np.std(bear_pnl) > 0 else 0

        max_sharpe = max(abs(bull_sharpe), abs(bear_sharpe))
        regime_gap = abs(bull_sharpe - bear_sharpe) / max_sharpe if max_sharpe > 0 else 0

        regime_stats = {
            "bull_months": len(bull_pnl),
            "bear_months": len(bear_pnl),
            "bull_avg_monthly_return": round(bull_avg * 100, 3),
            "bear_avg_monthly_return": round(bear_avg * 100, 3),
            "bull_sharpe": round(bull_sharpe, 3),
            "bear_sharpe": round(bear_sharpe, 3),
            "regime_gap": round(regime_gap, 3),
        }

    return {
        "total_return_pct": round(total_return * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "annualized_income_yield_pct": round(cagr * 100, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "min_nav_pct_of_start": round(min_nav / initial_capital * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "positive_months_pct": round(positive_pct, 1),
        "num_months": len(months),
        "avg_monthly_income": round(np.mean(pnl_values), 2),
        "best_month": {"month": months[best_month_idx], "pnl": round(pnl_values[best_month_idx], 2)},
        "worst_month": {"month": months[worst_month_idx], "pnl": round(pnl_values[worst_month_idx], 2)},
        "final_nav": round(nav, 2),
        "regime_stats": regime_stats,
    }


def check_gates(metrics):
    gates = {}
    gates["G1_yield_gt_15pct"] = {
        "pass": metrics["annualized_income_yield_pct"] > 15,
        "value": metrics["annualized_income_yield_pct"],
        "threshold": 15,
    }
    gates["G2_max_dd_lt_15pct"] = {
        "pass": metrics["max_drawdown_pct"] < 15,
        "value": metrics["max_drawdown_pct"],
        "threshold": 15,
    }
    gates["G3_positive_months_gt_90pct"] = {
        "pass": metrics["positive_months_pct"] > 90,
        "value": metrics["positive_months_pct"],
        "threshold": 90,
    }
    gates["G4_nav_never_below_90pct"] = {
        "pass": metrics["min_nav_pct_of_start"] >= 90,
        "value": metrics["min_nav_pct_of_start"],
        "threshold": 90,
    }
    if metrics["regime_stats"]:
        regime_gap = metrics["regime_stats"]["regime_gap"]
    else:
        regime_gap = float("nan")
    gates["G5_regime_gap_lt_050"] = {
        "pass": regime_gap < 0.50 if not np.isnan(regime_gap) else False,
        "value": regime_gap,
        "threshold": 0.50,
    }
    gates["all_pass"] = all(g["pass"] for g in gates.values() if isinstance(g, dict))
    return gates


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    np.random.seed(42)

    stock_data, vix_data, vxx_data = download_data(start="2017-06-01", end="2026-07-23")

    # Extract close prices
    if isinstance(stock_data.columns, pd.MultiIndex):
        close_prices = stock_data["Close"]
    else:
        close_prices = stock_data[["Close"]].copy()
        close_prices.columns = [SPY_TICKER]

    if isinstance(vix_data.columns, pd.MultiIndex):
        vix_series = vix_data["Close"].squeeze()
    else:
        vix_series = vix_data["Close"].squeeze()

    if isinstance(vxx_data.columns, pd.MultiIndex):
        vxx_series = vxx_data["Close"].squeeze()
    else:
        vxx_series = vxx_data["Close"].squeeze()

    close_prices.index = pd.to_datetime(close_prices.index)
    vix_series.index = pd.to_datetime(vix_series.index)
    vxx_series.index = pd.to_datetime(vxx_series.index)

    if close_prices.index.tz is not None:
        close_prices.index = close_prices.index.tz_localize(None)
    if vix_series.index.tz is not None:
        vix_series.index = vix_series.index.tz_localize(None)
    if vxx_series.index.tz is not None:
        vxx_series.index = vxx_series.index.tz_localize(None)

    sim_start = pd.Timestamp("2018-01-02")
    dates = close_prices.index[close_prices.index >= sim_start].tolist()

    print(f"Simulation period: {dates[0].date()} to {dates[-1].date()} ({len(dates)} trading days)")
    print(f"Tickers available: {list(close_prices.columns)}")
    print(f"VIX range: {vix_series.min():.1f} - {vix_series.max():.1f}")
    print()

    # SPY monthly returns for regime classification
    spy_monthly = close_prices[SPY_TICKER].resample("ME").last().pct_change()
    spy_monthly_returns = {}
    for idx, val in spy_monthly.items():
        if not np.isnan(val):
            spy_monthly_returns[idx.strftime("%Y-%m")] = val

    # Monthly average VIX for tail correlation
    vix_monthly_avg = {}
    for date in dates:
        if date in vix_series.index:
            mk = date.strftime("%Y-%m")
            if mk not in vix_monthly_avg:
                vix_monthly_avg[mk] = []
            v = vix_series.loc[date]
            if not np.isnan(v):
                vix_monthly_avg[mk].append(v)
    vix_monthly_avg = {k: np.mean(v) for k, v in vix_monthly_avg.items() if v}

    # Strategy allocations
    ic_capital = INITIAL_CAPITAL * ALLOC_IC
    cc_capital = INITIAL_CAPITAL * ALLOC_CC
    vrp_capital = INITIAL_CAPITAL * ALLOC_VRP

    print(f"Allocations: IC=${ic_capital:,.0f}, CC=${cc_capital:,.0f}, VRP=${vrp_capital:,.0f}")
    print(f"CC model: HONEST — tracks stock P&L + premium (total return)")
    print(f"IC model: Per-ticker IV (VIX * beta)")
    print(f"VRP model: 5% annual borrow cost on short positions")
    print()

    # Run each strategy
    print("Running Iron Condor strategy (with per-ticker IV)...")
    ic_strat = IronCondorStrategy(ic_capital, close_prices, vix_series)
    ic_pnl, ic_nav = ic_strat.simulate(dates)
    print(f"  IC months: {len(ic_pnl)}, total PnL: ${sum(ic_pnl.values()):,.2f}")

    print("Running HONEST Covered Call strategy (stock P&L + premium)...")
    cc_strat = CoveredCallStrategy(cc_capital, close_prices, vix_series)
    cc_pnl, cc_nav = cc_strat.simulate(dates)
    print(f"  CC months: {len(cc_pnl)}, total PnL: ${sum(cc_pnl.values()):,.2f}")

    print("Running VRP Overlay (with borrow cost)...")
    vrp_strat = VRPOverlay(vrp_capital, vix_series, vxx_series)
    vrp_pnl, vrp_nav = vrp_strat.simulate(dates)
    print(f"  VRP months: {len(vrp_pnl)}, total PnL: ${sum(vrp_pnl.values()):,.2f}")
    print()

    csp_pnl = {}

    # Apply tail correlation adjustment
    all_months = sorted(set(
        list(ic_pnl.keys()) + list(cc_pnl.keys()) + list(vrp_pnl.keys())
    ))

    print("Applying tail correlation adjustment...")
    ic_pnl_adj, cc_pnl_adj, vrp_pnl_adj = apply_tail_correlation(
        ic_pnl, cc_pnl, vrp_pnl, vix_monthly_avg, all_months
    )

    # Count stress months
    stress_months = sum(1 for m in all_months if m in vix_monthly_avg and vix_monthly_avg[m] > 25)
    print(f"  Stress months (avg VIX > 25): {stress_months} of {len(all_months)}")
    print()

    # Combine with adjusted PnL
    combined_pnl = combine_strategies(ic_pnl_adj, cc_pnl_adj, csp_pnl, vrp_pnl_adj, all_months)

    # Also compute unadjusted for comparison
    combined_pnl_raw = combine_strategies(ic_pnl, cc_pnl, csp_pnl, vrp_pnl, all_months)

    # Compute metrics
    metrics = compute_metrics(combined_pnl, INITIAL_CAPITAL, spy_monthly_returns)
    metrics_raw = compute_metrics(combined_pnl_raw, INITIAL_CAPITAL, spy_monthly_returns)

    # Per-strategy metrics (adjusted)
    ic_metrics = compute_metrics(ic_pnl_adj, ic_capital, spy_monthly_returns)
    cc_metrics = compute_metrics(cc_pnl_adj, cc_capital, spy_monthly_returns)
    vrp_metrics = compute_metrics(vrp_pnl_adj, vrp_capital, spy_monthly_returns)

    gates = check_gates(metrics)

    # Print results
    print("=" * 70)
    print("INCOME PORTFOLIO v2 — HONEST RESULTS")
    print("=" * 70)
    print(f"Period: {dates[0].date()} to {dates[-1].date()} ({metrics['num_months']} months)")
    print(f"Initial Capital: ${INITIAL_CAPITAL:,.0f}")
    print(f"Final NAV: ${metrics['final_nav']:,.0f}")
    print()

    print("--- KEY FIXES APPLIED ---")
    print("  [1] CC tracks TOTAL return (stock + premium), not premium-only")
    print("  [2] Per-ticker IV (VIX * beta), not flat VIX for all")
    print("  [3] Proper delta targeting via Brent root-finding")
    print("  [4] VRP: 5% annual borrow cost on short VXX")
    print("  [5] Tail correlation penalty during VIX > 25/30 months")
    print()

    print("--- COMBINED PORTFOLIO (with tail correlation) ---")
    print(f"  CAGR:              {metrics['cagr_pct']:.1f}%")
    print(f"  Max Drawdown:      {metrics['max_drawdown_pct']:.1f}%")
    print(f"  Sharpe:            {metrics['sharpe']:.3f}")
    print(f"  Sortino:           {metrics['sortino']:.3f}")
    print(f"  Positive Months:   {metrics['positive_months_pct']:.1f}%")
    print(f"  Min NAV %:         {metrics['min_nav_pct_of_start']:.1f}%")
    print(f"  Avg Monthly:       ${metrics['avg_monthly_income']:,.2f}")
    print(f"  Best Month:        {metrics['best_month']['month']} (${metrics['best_month']['pnl']:,.2f})")
    print(f"  Worst Month:       {metrics['worst_month']['month']} (${metrics['worst_month']['pnl']:,.2f})")
    print()

    print("--- WITHOUT TAIL CORRELATION (for comparison) ---")
    print(f"  CAGR:              {metrics_raw['cagr_pct']:.1f}%")
    print(f"  Max Drawdown:      {metrics_raw['max_drawdown_pct']:.1f}%")
    print(f"  Sharpe:            {metrics_raw['sharpe']:.3f}")
    print()

    if metrics["regime_stats"]:
        rs = metrics["regime_stats"]
        print(f"  Regime: Bull({rs['bull_months']}m) avg={rs['bull_avg_monthly_return']:.2f}% | Bear({rs['bear_months']}m) avg={rs['bear_avg_monthly_return']:.2f}%")
        print(f"  Regime Sharpe: Bull={rs['bull_sharpe']:.3f} | Bear={rs['bear_sharpe']:.3f} | Gap={rs['regime_gap']:.3f}")
    print()

    print("--- PER-STRATEGY BREAKDOWN ---")
    for name, m in [("Iron Condor", ic_metrics), ("Covered Call (HONEST)", cc_metrics),
                     ("VRP Overlay", vrp_metrics)]:
        print(f"  {name:25s}: CAGR={m['cagr_pct']:6.1f}%  Sharpe={m['sharpe']:6.3f}  DD={m['max_drawdown_pct']:5.1f}%  +Months={m['positive_months_pct']:.0f}%")
    print()

    print("--- GATE CHECKS ---")
    all_pass = True
    for gate_name, gate in gates.items():
        if gate_name == "all_pass":
            continue
        status = "PASS" if gate["pass"] else "FAIL"
        if not gate["pass"]:
            all_pass = False
        val = gate["value"]
        if isinstance(val, float):
            val = f"{val:.2f}"
        print(f"  {gate_name:35s}: {status}  (value={val}, threshold={gate['threshold']})")

    print(f"\n  ALL GATES: {'PASS' if all_pass else 'FAIL'}")
    print()

    # Build results dict
    results = {
        "metadata": {
            "script": "income_portfolio_v2_honest.py",
            "version": "v2_honest",
            "run_date": dt.datetime.now().isoformat(),
            "sim_start": str(dates[0].date()),
            "sim_end": str(dates[-1].date()),
            "initial_capital": INITIAL_CAPITAL,
            "allocations": {
                "iron_condor": ALLOC_IC,
                "covered_call": ALLOC_CC,
                "cash_secured_put": ALLOC_CSP,
                "vrp_overlay": ALLOC_VRP,
            },
            "fixes_applied": [
                "CC tracks total return (stock P&L + premium), not premium-only",
                "Per-ticker IV estimation (VIX * beta factor per stock)",
                "Proper delta targeting via Brent root-finding instead of approximate inversion",
                "VRP: 5% annual borrow cost on short VXX positions",
                "Tail correlation: stress penalty during VIX > 25/30 months",
            ],
            "assumptions": {
                "commission_per_contract": COMMISSION_PER_CONTRACT,
                "bid_ask_cost_pct": BID_ASK_COST,
                "assignment_risk_pct": ASSIGNMENT_RISK,
                "take_profit_pct": TAKE_PROFIT_PCT,
                "stop_loss_mult": STOP_LOSS_MULT,
                "vrp_annual_borrow_cost": VRP_ANNUAL_BORROW_COST,
            },
            "ticker_iv_betas": TICKER_IV_BETA,
        },
        "combined_metrics": metrics,
        "combined_metrics_no_tail_corr": metrics_raw,
        "per_strategy": {
            "iron_condor": ic_metrics,
            "covered_call_honest": cc_metrics,
            "vrp_overlay": vrp_metrics,
        },
        "gates": gates,
        "monthly_pnl": {m: round(v, 2) for m, v in combined_pnl.items()},
        "per_strategy_monthly_pnl": {
            "iron_condor": {m: round(v, 2) for m, v in ic_pnl_adj.items()},
            "covered_call_honest": {m: round(v, 2) for m, v in cc_pnl_adj.items()},
            "vrp_overlay": {m: round(v, 2) for m, v in vrp_pnl_adj.items()},
        },
        "stress_analysis": {
            "stress_months_count": stress_months,
            "total_months": len(all_months),
            "monthly_vix_averages": {k: round(v, 1) for k, v in vix_monthly_avg.items()},
        },
    }

    output_path = Path("/home/jupiter/Lvl3Quant/findings/income_portfolio_v2_honest_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Results saved to {output_path}")

    return results


if __name__ == "__main__":
    results = main()
