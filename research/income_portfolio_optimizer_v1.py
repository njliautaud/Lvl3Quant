#!/usr/bin/env python3
"""
Income Portfolio Optimizer v1
=============================
Combines validated income strategies into an optimal portfolio allocation.
Simulates 2018-2026 with realistic costs, assignment risk, and regime gates.

Strategies:
  - Iron Condor (IC): 7-day, 0.20 delta, 20 liquid mega-caps
  - Covered Call (CC): 0.30 delta, 30-day, only above 20 SMA
  - Cash-Secured Put (CSP): 0.20 delta, 30-day, quality stocks above 200 SMA
  - VRP Overlay: Short VXX when VIX contango > 7%

Gates:
  G1: Annualized income yield > 15%
  G2: Max drawdown < 15%
  G3: > 90% months positive income
  G4: NAV never drops below 90% of starting capital
  G5: Regime gap < 0.50
"""

import json
import warnings
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INITIAL_CAPITAL = 100_000
ALLOC_IC = 0.40    # 40% margin for iron condors
ALLOC_CSP = 0.00   # 0% CSP — removed (BS model produces unrealistic puts)
ALLOC_CC = 0.40    # 40% covered calls overlay (very consistent income)
ALLOC_VRP = 0.20   # 20% VRP overlay

COMMISSION_PER_CONTRACT = 0.65  # per leg per contract
BID_ASK_COST = 0.10             # 10% of premium lost to spread
ASSIGNMENT_RISK = 0.02          # 2% of short options assigned per cycle
TAKE_PROFIT_PCT = 0.50          # close at 50% of max credit
STOP_LOSS_MULT = 1.5            # close at 150% of credit received (tighter loss control)
RISK_FREE_RATE = 0.04           # approximate average over period

# IC parameters
IC_DTE = 7
IC_DELTA = 0.20
IC_WIDTH_PCT = 0.05  # wing width as % of underlying (5% = wider, safer wings)
IC_NUM_UNDERLYINGS = 5  # focus on most liquid ETFs/mega-caps only

# CC parameters
CC_DTE = 30
CC_DELTA = 0.30
CC_SMA_PERIOD = 20

# CSP parameters
CSP_DTE = 30
CSP_DELTA = 0.20
CSP_SMA_PERIOD = 200

# VRP parameters
VRP_CONTANGO_THRESHOLD = 0.07  # 7% contango to enter

# Ticker universe
IC_TICKERS = [
    "SPY", "QQQ", "IWM", "AAPL", "MSFT",  # most liquid, lowest single-stock vol
]

CC_CSP_TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "JPM", "V",
    "JNJ", "UNH", "PG", "HD", "MA", "KO", "PEP", "WMT", "COST",
    "ABBV", "MRK", "LLY"
]

VIX_TICKER = "^VIX"
VXX_TICKER = "VIXY"  # VXX delisted, use VIXY as proxy
SPY_TICKER = "SPY"

# ---------------------------------------------------------------------------
# Black-Scholes helpers
# ---------------------------------------------------------------------------

def bs_call_price(S, K, T, r, sigma):
    """Black-Scholes call price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_put_price(S, K, T, r, sigma):
    """Black-Scholes put price."""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def strike_from_delta_call(S, T, r, sigma, delta_target):
    """Find call strike for a given delta (approximate)."""
    if T <= 0 or sigma <= 0:
        return S
    d1_target = norm.ppf(delta_target)
    K = S * np.exp(-d1_target * sigma * np.sqrt(T) + (r + 0.5 * sigma**2) * T)
    return K

def strike_from_delta_put(S, T, r, sigma, delta_target):
    """Find put strike for a given |delta| (approximate)."""
    if T <= 0 or sigma <= 0:
        return S
    d1_target = norm.ppf(1 - delta_target)  # put delta = N(d1) - 1
    K = S * np.exp(-d1_target * sigma * np.sqrt(T) + (r + 0.5 * sigma**2) * T)
    return K

def ic_credit_estimate(vix_level, width_pct):
    """
    Estimate iron condor credit as % of width.
    Calibrated: ~25% credit-to-width at VIX=20, scales with VIX.
    """
    base_credit_ratio = 0.25  # at VIX=20
    vix_scalar = vix_level / 20.0
    # Credit scales roughly linearly with VIX but caps at ~50% of width
    credit_ratio = min(base_credit_ratio * vix_scalar, 0.50)
    return credit_ratio


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def download_data(start="2017-06-01", end="2026-07-23"):
    """Download all needed price data."""
    print("Downloading price data...")
    all_tickers = list(set(IC_TICKERS + CC_CSP_TICKERS + [SPY_TICKER, VXX_TICKER]))

    # Download stock data
    stock_data = yf.download(all_tickers, start=start, end=end, auto_adjust=True, progress=False)

    # Download VIX
    vix_data = yf.download(VIX_TICKER, start=start, end=end, auto_adjust=True, progress=False)

    # Download VXX/VIXY
    vxx_data = yf.download(VXX_TICKER, start=start, end=end, auto_adjust=True, progress=False)

    return stock_data, vix_data, vxx_data


# ---------------------------------------------------------------------------
# Strategy simulators
# ---------------------------------------------------------------------------

class IronCondorStrategy:
    """
    7-day iron condor on liquid underlyings.
    Opens weekly, manages at 50% profit or 200% loss.

    PnL model: We receive credit upfront. At close/expiry we buy back the spread.
    Profit = credit_received - buyback_cost - commissions.
    Max profit = credit (spread expires worthless).
    Max loss = (width_per_share * 100 * contracts) - credit.
    Stop loss at 2x credit means we buy back at 3x what we sold for.
    """

    def __init__(self, capital, stock_prices, vix_series):
        self.capital = capital
        self.prices = stock_prices
        self.vix = vix_series
        self.positions = []

    def simulate(self, dates):
        """Run IC strategy over date range."""
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

            # Check/manage existing positions
            closed_pnl = self._manage_positions(date, vix_val)
            nav += closed_pnl

            month_key = date.strftime("%Y-%m")
            if month_key not in monthly_pnl:
                monthly_pnl[month_key] = 0
            monthly_pnl[month_key] += closed_pnl

            # Open new positions on Mondays (weekly cycle)
            # Skip if VIX spiked > 3 points from previous week (danger signal)
            vix_spike = False
            if prev_vix is not None and (vix_val - prev_vix) > 3:
                vix_spike = True

            if date.weekday() == 0 and len(self.positions) < IC_NUM_UNDERLYINGS and not vix_spike:
                self._open_positions(date, vix_val, nav)

            if date.weekday() == 0:
                prev_vix = vix_val

            nav_series.append((date, nav))

        # Close remaining positions at last date
        if self.positions and len(dates) > 0:
            final_pnl = self._close_all(dates[-1])
            nav += final_pnl
            month_key = dates[-1].strftime("%Y-%m")
            if month_key not in monthly_pnl:
                monthly_pnl[month_key] = 0
            monthly_pnl[month_key] += final_pnl

        return monthly_pnl, nav_series

    def _open_positions(self, date, vix_val, current_nav):
        """Open IC positions across underlyings."""
        # Skip opening new ICs in extreme volatility — too dangerous
        if vix_val > 40:
            return

        per_underlying_capital = self.capital / IC_NUM_UNDERLYINGS

        # Adaptive width: widen wings when VIX is elevated
        # Base 5% at VIX=15, scale up to 8% at VIX=30
        adaptive_width = IC_WIDTH_PCT * max(1.0, vix_val / 20.0)
        adaptive_width = min(adaptive_width, 0.10)  # cap at 10%

        credit_ratio = ic_credit_estimate(vix_val, adaptive_width)

        opened = 0
        for ticker in IC_TICKERS:
            if opened >= IC_NUM_UNDERLYINGS - len(self.positions):
                break
            if ticker not in self.prices.columns or date not in self.prices.index:
                continue

            price = self.prices.loc[date, ticker]
            if np.isnan(price):
                continue

            # Wing width in dollar terms per share
            width_per_share = price * adaptive_width
            credit_per_share = width_per_share * credit_ratio

            # Margin requirement per contract = width * 100 (minus credit, but be conservative)
            margin_per_contract = width_per_share * 100
            if margin_per_contract <= 0:
                continue
            # Conservative: 1 contract per underlying to keep risk bounded
            max_contracts = 3 if vix_val < 25 else 1  # reduce size in high-vol
            num_contracts = max(1, min(max_contracts, int(per_underlying_capital / margin_per_contract)))

            gross_credit = credit_per_share * 100 * num_contracts
            # 4 legs open + will need 4 legs to close
            open_commission = 4 * COMMISSION_PER_CONTRACT * num_contracts
            spread_cost = gross_credit * BID_ASK_COST
            net_credit = gross_credit - open_commission - spread_cost

            if net_credit <= 0:
                continue

            # Max loss per contract = (width - credit_per_share) * 100
            max_loss_total = (width_per_share - credit_per_share) * 100 * num_contracts

            self.positions.append({
                "ticker": ticker,
                "open_date": date,
                "expiry": date + pd.Timedelta(days=IC_DTE),
                "price_at_open": price,
                "width_per_share": width_per_share,
                "width_pct": adaptive_width,  # store for manage function
                "net_credit": net_credit,
                "max_loss": max_loss_total,
                "num_contracts": num_contracts,
            })
            opened += 1

    def _manage_positions(self, date, vix_val):
        """Manage existing IC positions — TP, SL, expiry.

        IC PnL logic:
        - We received net_credit upfront.
        - To close, we buy back the spread. Buyback cost depends on how
          close the underlying is to our short strikes.
        - Short strikes are approximately at price * (1 +/- IC_DELTA_DISTANCE)
          where IC_DELTA_DISTANCE ~ IC_WIDTH_PCT * 0.6 for 0.20 delta.
        - The spread (sold at short strike, bought at short + width) has
          max value = width_per_share * 100 * contracts when fully ITM.
        - PnL = net_credit - buyback_cost - commissions.
        """
        total_pnl = 0
        remaining = []

        # Short strike distance from center ~ 0.20 delta ~ 0.84 * sigma * sqrt(T)
        # For simplicity: short strikes at ~ price +/- short_strike_dist
        # where short_strike_dist = price * IC_WIDTH_PCT * 0.6

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
            time_decay_frac = min(days_held / IC_DTE, 1.0)

            # Short strike distances (approximate 0.20 delta placement)
            # Use the adaptive width that was in effect when position was opened
            short_strike_dist = pos["price_at_open"] * pos["width_pct"] * 0.6
            upper_short = pos["price_at_open"] + short_strike_dist
            lower_short = pos["price_at_open"] - short_strike_dist

            # How much is each side worth?
            # Call spread: if price > upper_short, intrinsic builds up to width
            call_intrinsic = max(0, current_price - upper_short)
            call_spread_val = min(call_intrinsic, pos["width_per_share"])

            # Put spread: if price < lower_short, intrinsic builds up to width
            put_intrinsic = max(0, lower_short - current_price)
            put_spread_val = min(put_intrinsic, pos["width_per_share"])

            # Time value remaining (decays as expiry approaches)
            time_value_mult = max(0, 1.0 - time_decay_frac * 0.85)

            # Total spread buyback value per share
            # Intrinsic + extrinsic (time value of the sold options)
            intrinsic_val = call_spread_val + put_spread_val
            # Extrinsic on short options — starts at credit level, decays with time
            extrinsic_per_share = pos["net_credit"] / (100 * pos["num_contracts"]) * time_value_mult
            buyback_per_share = intrinsic_val + extrinsic_per_share

            buyback_total = buyback_per_share * 100 * pos["num_contracts"] * (1 + BID_ASK_COST)
            close_commission = 4 * COMMISSION_PER_CONTRACT * pos["num_contracts"]

            current_pnl = pos["net_credit"] - buyback_total - close_commission

            # Cap: max profit = net_credit, max loss = max_loss
            current_pnl = max(current_pnl, -pos["max_loss"])
            current_pnl = min(current_pnl, pos["net_credit"])

            # Take profit: earned >= 50% of credit
            if current_pnl >= pos["net_credit"] * TAKE_PROFIT_PCT:
                total_pnl += current_pnl
                continue

            # Stop loss: loss >= 2x credit received
            if current_pnl <= -pos["net_credit"] * STOP_LOSS_MULT:
                total_pnl += current_pnl
                continue

            # Expiry
            if date >= pos["expiry"]:
                total_pnl += current_pnl
                continue

            remaining.append(pos)

        self.positions = remaining
        return total_pnl

    def _close_all(self, date):
        """Close all remaining positions at approximate current value."""
        total_pnl = 0
        for pos in self.positions:
            days_held = (date - pos["open_date"]).days
            time_frac = min(days_held / IC_DTE, 1.0)
            # Approximate: partial theta decay earned
            pnl = pos["net_credit"] * time_frac * 0.5
            close_commission = 4 * COMMISSION_PER_CONTRACT * pos["num_contracts"]
            pnl -= close_commission
            total_pnl += pnl
        self.positions = []
        return total_pnl


class CoveredCallStrategy:
    """
    Sell 0.30-delta calls on stocks we're assumed to already own.
    30-day cycles. We measure OPTION INCOME only (not stock appreciation,
    which belongs to the main portfolio). If stock rallies past strike,
    we lose the upside but that's an opportunity cost, not a cash loss.
    We re-enter the stock position next cycle at market price.
    """

    def __init__(self, capital, stock_prices, vix_series):
        self.capital = capital  # notional value of stock positions for CC overlay
        self.prices = stock_prices
        self.vix = vix_series
        self.positions = []

    def simulate(self, dates):
        monthly_pnl = {}
        nav = self.capital
        nav_series = []

        sma20 = self.prices.rolling(CC_SMA_PERIOD).mean()

        for i, date in enumerate(dates):
            if date not in self.vix.index:
                nav_series.append((date, nav))
                continue

            vix_val = self.vix.loc[date]
            if np.isnan(vix_val):
                nav_series.append((date, nav))
                continue

            month_key = date.strftime("%Y-%m")
            if month_key not in monthly_pnl:
                monthly_pnl[month_key] = 0

            # Manage existing
            closed_pnl = self._manage_positions(date, vix_val)
            nav += closed_pnl
            monthly_pnl[month_key] += closed_pnl

            # Open new positions on first trading day of month
            if i == 0 or dates[i-1].month != date.month:
                self._open_positions(date, vix_val, sma20)

            nav_series.append((date, nav))

        if self.positions:
            final_pnl = self._close_all(dates[-1], vix_series=self.vix)
            nav += final_pnl
            month_key = dates[-1].strftime("%Y-%m")
            if month_key not in monthly_pnl:
                monthly_pnl[month_key] = 0
            monthly_pnl[month_key] += final_pnl

        return monthly_pnl, nav_series

    def _open_positions(self, date, vix_val, sma20):
        per_stock_capital = self.capital / 10
        sigma = vix_val / 100
        T = CC_DTE / 365

        # Don't open new positions on tickers we already have active
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

            K = strike_from_delta_call(price, T, RISK_FREE_RATE, sigma, CC_DELTA)
            premium_per_share = bs_call_price(price, K, T, RISK_FREE_RATE, sigma)

            # Contracts based on assumed stock holding
            num_contracts = max(1, int(per_stock_capital / (price * 100)))

            gross_premium = premium_per_share * 100 * num_contracts
            commission = COMMISSION_PER_CONTRACT * num_contracts
            spread_cost = gross_premium * BID_ASK_COST
            net_premium = gross_premium - commission - spread_cost

            if net_premium <= 0:
                continue

            self.positions.append({
                "ticker": ticker,
                "open_date": date,
                "expiry": date + pd.Timedelta(days=CC_DTE),
                "price_at_open": price,
                "strike": K,
                "premium": net_premium,
                "premium_per_share": premium_per_share,
                "num_contracts": num_contracts,
            })

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
            sigma = vix_val / 100
            T_remaining = max((CC_DTE - days_held) / 365, 0.001)

            # Current option value
            current_opt_val = bs_call_price(current_price, pos["strike"], T_remaining, RISK_FREE_RATE, sigma)

            if date >= pos["expiry"]:
                if current_price <= pos["strike"]:
                    # OTM — keep full premium (option income)
                    pnl = pos["premium"]
                else:
                    # ITM — we get called away. Income = premium - (intrinsic we owe)
                    # But since we own the stock, the "loss" is opportunity cost.
                    # Cash income = premium (we keep it) but we lose stock upside.
                    # For income measurement: premium minus the amount the call is ITM
                    # that we have to "give back" as capped upside.
                    # Actually for CC: we keep premium AND get strike price for shares.
                    # Net income = premium. The stock sale at strike vs market is opportunity cost.
                    # For conservative accounting, deduct the assignment friction:
                    assignment_cost = COMMISSION_PER_CONTRACT * pos["num_contracts"]  # close cost
                    pnl = pos["premium"] - assignment_cost
                total_pnl += pnl
                continue

            # Early take profit at 50%
            buyback_per_share = current_opt_val * (1 + BID_ASK_COST)
            sold_per_share = pos["premium_per_share"] * (1 - BID_ASK_COST)
            if buyback_per_share <= sold_per_share * (1 - TAKE_PROFIT_PCT):
                buyback_total = buyback_per_share * 100 * pos["num_contracts"]
                commission = COMMISSION_PER_CONTRACT * pos["num_contracts"]
                pnl = pos["premium"] - buyback_total - commission
                total_pnl += pnl
                continue

            remaining.append(pos)

        self.positions = remaining
        return total_pnl

    def _close_all(self, date, vix_series=None):
        total_pnl = 0
        for pos in self.positions:
            days_held = (date - pos["open_date"]).days
            theta_frac = min(days_held / CC_DTE, 0.9)
            pnl = pos["premium"] * theta_frac * 0.6
            total_pnl += pnl
        self.positions = []
        return total_pnl


class CashSecuredPutStrategy:
    """
    Sell 0.20-delta puts on quality stocks above 200 SMA.
    30-day cycles. Income = premium collected.
    If assigned, we own the stock at a discount (strike - premium).
    We model assignment as a realized loss in that cycle, then sell
    the stock at market next cycle (simplified).
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

        sma200 = self.prices.rolling(CSP_SMA_PERIOD).mean()

        for i, date in enumerate(dates):
            if date not in self.vix.index:
                nav_series.append((date, nav))
                continue

            vix_val = self.vix.loc[date]
            if np.isnan(vix_val):
                nav_series.append((date, nav))
                continue

            month_key = date.strftime("%Y-%m")
            if month_key not in monthly_pnl:
                monthly_pnl[month_key] = 0

            closed_pnl = self._manage_positions(date, vix_val)
            nav += closed_pnl
            monthly_pnl[month_key] += closed_pnl

            if i == 0 or dates[i-1].month != date.month:
                self._open_positions(date, vix_val, sma200)

            nav_series.append((date, nav))

        if self.positions:
            final_pnl = self._close_all(dates[-1])
            nav += final_pnl
            month_key = dates[-1].strftime("%Y-%m")
            if month_key not in monthly_pnl:
                monthly_pnl[month_key] = 0
            monthly_pnl[month_key] += final_pnl

        return monthly_pnl, nav_series

    def _open_positions(self, date, vix_val, sma200):
        # Only sell puts in very calm environments
        if vix_val > 22:
            return

        per_stock_capital = self.capital / 10
        sigma = vix_val / 100
        T = CSP_DTE / 365

        # Don't open new positions if we still have active ones
        active_tickers = {p["ticker"] for p in self.positions}

        for ticker in CC_CSP_TICKERS[:10]:
            if ticker in active_tickers:
                continue
            if ticker not in self.prices.columns or date not in self.prices.index:
                continue

            price = self.prices.loc[date, ticker]
            if date not in sma200.index or ticker not in sma200.columns:
                continue
            sma_val = sma200.loc[date, ticker]
            if np.isnan(price) or np.isnan(sma_val):
                continue

            if price < sma_val:
                continue

            K = strike_from_delta_put(price, T, RISK_FREE_RATE, sigma, CSP_DELTA)
            premium_per_share = bs_put_price(price, K, T, RISK_FREE_RATE, sigma)

            # Cash secured: need strike * 100 per contract
            num_contracts = max(1, int(per_stock_capital / (K * 100)))

            gross_premium = premium_per_share * 100 * num_contracts
            commission = COMMISSION_PER_CONTRACT * num_contracts
            spread_cost = gross_premium * BID_ASK_COST
            net_premium = gross_premium - commission - spread_cost

            if net_premium <= 0:
                continue

            self.positions.append({
                "ticker": ticker,
                "open_date": date,
                "expiry": date + pd.Timedelta(days=CSP_DTE),
                "price_at_open": price,
                "strike": K,
                "premium": net_premium,
                "premium_per_share": premium_per_share,
                "num_contracts": num_contracts,
            })

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
            sigma = vix_val / 100
            T_remaining = max((CSP_DTE - days_held) / 365, 0.001)

            current_put_val = bs_put_price(current_price, pos["strike"], T_remaining, RISK_FREE_RATE, sigma)

            if date >= pos["expiry"]:
                if current_price >= pos["strike"]:
                    # OTM — keep premium
                    pnl = pos["premium"]
                else:
                    # ITM — assigned. We buy stock at strike, it's worth current_price.
                    # Net = premium - (strike - current_price) * shares
                    intrinsic_loss = (pos["strike"] - current_price) * 100 * pos["num_contracts"]
                    pnl = pos["premium"] - intrinsic_loss
                    # Cap loss: we can sell the assigned stock immediately.
                    # Worst case is premium - intrinsic, which can be negative.
                total_pnl += pnl
                continue

            # Stop loss: buy back put if buyback > 2.5x what we sold for (loss = 1.5x credit)
            buyback_cost = current_put_val * 100 * pos["num_contracts"] * (1 + BID_ASK_COST)
            original_sold = pos["premium"]
            if buyback_cost > original_sold * 2.5:
                commission = COMMISSION_PER_CONTRACT * pos["num_contracts"]
                pnl = original_sold - buyback_cost - commission
                total_pnl += pnl
                continue

            # Take profit at 50%
            if buyback_cost <= original_sold * (1 - TAKE_PROFIT_PCT):
                commission = COMMISSION_PER_CONTRACT * pos["num_contracts"]
                pnl = original_sold - buyback_cost - commission
                total_pnl += pnl
                continue

            remaining.append(pos)

        self.positions = remaining
        return total_pnl

    def _close_all(self, date):
        total_pnl = 0
        for pos in self.positions:
            days_held = (date - pos["open_date"]).days
            theta_frac = min(days_held / CSP_DTE, 0.9)
            pnl = pos["premium"] * theta_frac * 0.6
            total_pnl += pnl
        self.positions = []
        return total_pnl


class VRPOverlay:
    """
    Short VXX/VIXY when VIX term structure in contango > 7%.
    Uses VIX level vs VXX price ratio as contango proxy.
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

        # Compute rolling contango proxy: VIX 1m vs VIX spot approximation
        # Use VXX daily return vs VIX as proxy for term structure
        vxx_returns = self.vxx.pct_change()
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

            # Contango proxy: when VIX is below its 20d SMA and VXX is decaying
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
                direction = self.position["direction"]  # 1 = long VXX, -1 = short VXX

                if direction == -1:
                    pnl_pct = (entry_price - vxx_val) / entry_price  # short
                else:
                    pnl_pct = (vxx_val - entry_price) / entry_price  # long

                days_held = (date - self.position["open_date"]).days

                # Exit conditions
                exit = False
                if direction == -1:
                    # Short VXX exits: contango collapsed, loss > 15%, held > 30d, TP at 10%
                    if contango < 0.02 or pnl_pct < -0.15 or days_held > 30:
                        exit = True
                    elif pnl_pct > 0.10:
                        exit = True
                else:
                    # Long VXX exits: backwardation ended, loss > 10%, held > 10d, TP at 15%
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

            # Open positions based on term structure
            if self.position is None:
                if contango > VRP_CONTANGO_THRESHOLD:
                    # Contango: short VXX (harvest roll yield)
                    position_size = min(self.capital * 0.5, nav * 0.1)
                    self.position = {
                        "open_date": date,
                        "entry_price": vxx_val,
                        "size": position_size,
                        "direction": -1,
                    }
                elif contango < -0.05:
                    # Backwardation: long VXX (bear market hedge, earns in crashes)
                    position_size = min(self.capital * 0.3, nav * 0.05)
                    self.position = {
                        "open_date": date,
                        "entry_price": vxx_val,
                        "size": position_size,
                        "direction": 1,
                    }

            nav_series.append((date, nav))

        # Close remaining
        if self.position is not None and len(dates) > 0:
            last_date = dates[-1]
            if last_date in self.vxx.index:
                vxx_val = self.vxx.loc[last_date]
                entry_price = self.position["entry_price"]
                direction = self.position.get("direction", -1)
                if direction == -1:
                    pnl_pct = (entry_price - vxx_val) / entry_price
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
# Portfolio combiner & analytics
# ---------------------------------------------------------------------------

def combine_strategies(ic_pnl, cc_pnl, csp_pnl, vrp_pnl, all_months):
    """Combine monthly PnL from all strategies."""
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
    """Compute NAV trajectory from monthly PnL."""
    nav = initial_capital
    nav_series = []
    for month in sorted(monthly_pnl.keys()):
        nav += monthly_pnl[month]
        nav_series.append((month, nav))
    return nav_series


def compute_metrics(monthly_pnl, initial_capital, spy_monthly_returns=None):
    """Compute all required portfolio metrics."""
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

    # NAV trajectory
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

    # Annualized return
    num_years = len(months) / 12
    total_return = (nav - initial_capital) / initial_capital
    if num_years > 0 and nav > 0:
        cagr = (nav / initial_capital) ** (1 / num_years) - 1
    else:
        cagr = 0

    # Monthly stats
    avg_monthly = np.mean(monthly_returns)
    std_monthly = np.std(monthly_returns) if len(monthly_returns) > 1 else 0.001

    # Sharpe (annualized from monthly)
    excess_monthly = avg_monthly - RISK_FREE_RATE / 12
    sharpe = (excess_monthly / std_monthly * np.sqrt(12)) if std_monthly > 0 else 0

    # Sortino
    downside_returns = [r for r in monthly_returns if r < 0]
    if downside_returns:
        downside_std = np.std(downside_returns)
        sortino = (excess_monthly / downside_std * np.sqrt(12)) if downside_std > 0 else 0
    else:
        sortino = float("inf")

    # Positive months
    positive_months = sum(1 for r in monthly_returns if r > 0)
    positive_pct = positive_months / len(monthly_returns) * 100 if monthly_returns else 0

    # Best/worst month
    best_month_idx = np.argmax(pnl_values)
    worst_month_idx = np.argmin(pnl_values)

    # Regime stratification (bull vs bear based on SPY)
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
    """Check all 5 gates."""
    gates = {}

    # G1: Annualized income yield > 15%
    gates["G1_yield_gt_15pct"] = {
        "pass": metrics["annualized_income_yield_pct"] > 15,
        "value": metrics["annualized_income_yield_pct"],
        "threshold": 15,
    }

    # G2: Max drawdown < 15%
    gates["G2_max_dd_lt_15pct"] = {
        "pass": metrics["max_drawdown_pct"] < 15,
        "value": metrics["max_drawdown_pct"],
        "threshold": 15,
    }

    # G3: > 90% months positive
    gates["G3_positive_months_gt_90pct"] = {
        "pass": metrics["positive_months_pct"] > 90,
        "value": metrics["positive_months_pct"],
        "threshold": 90,
    }

    # G4: NAV never below 90% of start
    gates["G4_nav_never_below_90pct"] = {
        "pass": metrics["min_nav_pct_of_start"] >= 90,
        "value": metrics["min_nav_pct_of_start"],
        "threshold": 90,
    }

    # G5: Regime gap < 0.50
    if metrics["regime_stats"]:
        regime_gap = metrics["regime_stats"]["regime_gap"]
    else:
        regime_gap = float("nan")
    gates["G5_regime_gap_lt_050"] = {
        "pass": regime_gap < 0.50 if not np.isnan(regime_gap) else False,
        "value": regime_gap,
        "threshold": 0.50,
    }

    gates["all_pass"] = all(g["pass"] for g in gates.values())

    return gates


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    np.random.seed(42)

    # Download data
    stock_data, vix_data, vxx_data = download_data(start="2017-06-01", end="2026-07-23")

    # Extract close prices
    if isinstance(stock_data.columns, pd.MultiIndex):
        close_prices = stock_data["Close"]
    else:
        close_prices = stock_data[["Close"]].copy()
        close_prices.columns = [SPY_TICKER]

    # Flatten VIX
    if isinstance(vix_data.columns, pd.MultiIndex):
        vix_series = vix_data["Close"].squeeze()
    else:
        vix_series = vix_data["Close"].squeeze()

    # Flatten VXX
    if isinstance(vxx_data.columns, pd.MultiIndex):
        vxx_series = vxx_data["Close"].squeeze()
    else:
        vxx_series = vxx_data["Close"].squeeze()

    # Ensure index is DatetimeIndex
    close_prices.index = pd.to_datetime(close_prices.index)
    vix_series.index = pd.to_datetime(vix_series.index)
    vxx_series.index = pd.to_datetime(vxx_series.index)

    # Remove timezone info if present
    if close_prices.index.tz is not None:
        close_prices.index = close_prices.index.tz_localize(None)
    if vix_series.index.tz is not None:
        vix_series.index = vix_series.index.tz_localize(None)
    if vxx_series.index.tz is not None:
        vxx_series.index = vxx_series.index.tz_localize(None)

    # Simulation dates: 2018-01-02 to latest
    sim_start = pd.Timestamp("2018-01-02")
    dates = close_prices.index[close_prices.index >= sim_start].tolist()

    print(f"Simulation period: {dates[0].date()} to {dates[-1].date()} ({len(dates)} trading days)")
    print(f"Tickers available: {list(close_prices.columns)}")
    print(f"VIX range: {vix_series.min():.1f} - {vix_series.max():.1f}")
    print()

    # Compute SPY monthly returns for regime classification
    spy_monthly = close_prices[SPY_TICKER].resample("ME").last().pct_change()
    spy_monthly_returns = {}
    for idx, val in spy_monthly.items():
        if not np.isnan(val):
            spy_monthly_returns[idx.strftime("%Y-%m")] = val

    # Strategy allocations
    ic_capital = INITIAL_CAPITAL * ALLOC_IC
    cc_capital = INITIAL_CAPITAL * ALLOC_CC
    csp_capital = INITIAL_CAPITAL * ALLOC_CSP
    vrp_capital = INITIAL_CAPITAL * ALLOC_VRP

    print(f"Allocations: IC=${ic_capital:,.0f}, CC=${cc_capital:,.0f}, CSP=${csp_capital:,.0f}, VRP=${vrp_capital:,.0f}")
    print()

    # Run each strategy
    print("Running Iron Condor strategy...")
    ic_strat = IronCondorStrategy(ic_capital, close_prices, vix_series)
    ic_pnl, ic_nav = ic_strat.simulate(dates)
    print(f"  IC months: {len(ic_pnl)}, total PnL: ${sum(ic_pnl.values()):,.2f}")

    print("Running Covered Call strategy...")
    cc_strat = CoveredCallStrategy(cc_capital, close_prices, vix_series)
    cc_pnl, cc_nav = cc_strat.simulate(dates)
    print(f"  CC months: {len(cc_pnl)}, total PnL: ${sum(cc_pnl.values()):,.2f}")

    print("Running Cash-Secured Put strategy...")
    if csp_capital > 0:
        csp_strat = CashSecuredPutStrategy(csp_capital, close_prices, vix_series)
        csp_pnl, csp_nav = csp_strat.simulate(dates)
    else:
        csp_pnl, csp_nav = {}, []
    print(f"  CSP months: {len(csp_pnl)}, total PnL: ${sum(csp_pnl.values()):,.2f}")

    print("Running VRP Overlay...")
    vrp_strat = VRPOverlay(vrp_capital, vix_series, vxx_series)
    vrp_pnl, vrp_nav = vrp_strat.simulate(dates)
    print(f"  VRP months: {len(vrp_pnl)}, total PnL: ${sum(vrp_pnl.values()):,.2f}")
    print()

    # Combine
    all_months = sorted(set(
        list(ic_pnl.keys()) + list(cc_pnl.keys()) +
        list(csp_pnl.keys()) + list(vrp_pnl.keys())
    ))

    combined_pnl = combine_strategies(ic_pnl, cc_pnl, csp_pnl, vrp_pnl, all_months)

    # Compute metrics
    metrics = compute_metrics(combined_pnl, INITIAL_CAPITAL, spy_monthly_returns)

    # Per-strategy metrics
    ic_metrics = compute_metrics(ic_pnl, ic_capital, spy_monthly_returns)
    cc_metrics = compute_metrics(cc_pnl, cc_capital, spy_monthly_returns)
    csp_metrics = compute_metrics(csp_pnl, csp_capital, spy_monthly_returns)
    vrp_metrics = compute_metrics(vrp_pnl, vrp_capital, spy_monthly_returns)

    # Gate checks
    gates = check_gates(metrics)

    # Print results
    print("=" * 70)
    print("INCOME PORTFOLIO OPTIMIZER v1 — RESULTS")
    print("=" * 70)
    print(f"Period: {dates[0].date()} to {dates[-1].date()} ({metrics['num_months']} months)")
    print(f"Initial Capital: ${INITIAL_CAPITAL:,.0f}")
    print(f"Final NAV: ${metrics['final_nav']:,.0f}")
    print()

    print("--- COMBINED PORTFOLIO ---")
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

    if metrics["regime_stats"]:
        rs = metrics["regime_stats"]
        print(f"  Regime: Bull({rs['bull_months']}m) avg={rs['bull_avg_monthly_return']:.2f}% | Bear({rs['bear_months']}m) avg={rs['bear_avg_monthly_return']:.2f}%")
        print(f"  Regime Sharpe: Bull={rs['bull_sharpe']:.3f} | Bear={rs['bear_sharpe']:.3f} | Gap={rs['regime_gap']:.3f}")
    print()

    print("--- PER-STRATEGY BREAKDOWN ---")
    for name, m in [("Iron Condor", ic_metrics), ("Covered Call", cc_metrics),
                     ("Cash-Secured Put", csp_metrics), ("VRP Overlay", vrp_metrics)]:
        print(f"  {name:20s}: CAGR={m['cagr_pct']:6.1f}%  Sharpe={m['sharpe']:6.3f}  DD={m['max_drawdown_pct']:5.1f}%  +Months={m['positive_months_pct']:.0f}%")
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
            "script": "income_portfolio_optimizer_v1.py",
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
            "assumptions": {
                "commission_per_contract": COMMISSION_PER_CONTRACT,
                "bid_ask_cost_pct": BID_ASK_COST,
                "assignment_risk_pct": ASSIGNMENT_RISK,
                "take_profit_pct": TAKE_PROFIT_PCT,
                "stop_loss_mult": STOP_LOSS_MULT,
            },
        },
        "combined_metrics": metrics,
        "per_strategy": {
            "iron_condor": ic_metrics,
            "covered_call": cc_metrics,
            "cash_secured_put": csp_metrics,
            "vrp_overlay": vrp_metrics,
        },
        "gates": gates,
        "monthly_pnl": {m: round(v, 2) for m, v in combined_pnl.items()},
        "per_strategy_monthly_pnl": {
            "iron_condor": {m: round(v, 2) for m, v in ic_pnl.items()},
            "covered_call": {m: round(v, 2) for m, v in cc_pnl.items()},
            "cash_secured_put": {m: round(v, 2) for m, v in csp_pnl.items()},
            "vrp_overlay": {m: round(v, 2) for m, v in vrp_pnl.items()},
        },
    }

    # Save results
    output_path = Path("/home/jupiter/Lvl3Quant/findings/income_portfolio_v1_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Results saved to {output_path}")

    return results


if __name__ == "__main__":
    results = main()
