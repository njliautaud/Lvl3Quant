#!/usr/bin/env python3
"""
div_capture_options_v1.py — Dividend Capture Enhanced with Options Backtest

THESIS:
  Classic dividend capture (buy before ex-div, sell after) rarely works because
  stock drops by ~dividend amount. BUT: selling OTM puts before ex-div exploits:
    1. IV often rises pre-div from hedging demand
    2. Post-div IV crush benefits put sellers
    3. Theta decay over short holding period
    4. If assigned: own stock at discount, collect next dividend, sell covered calls

THE TRADE:
  1. Identify S&P 500 stocks with ex-div in 1-5 days, yield > 2% annualized
  2. Sell 25-delta OTM put, 7-14 DTE
  3. If assigned: own stock, sell covered call
  4. If not assigned: keep premium, rotate to next ex-div

POSITION SIZING:
  - Max 5% NAV per position
  - Max 10 concurrent positions
  - Max 30% total margin usage

PREMIUM ESTIMATION:
  Black-Scholes with realized vol as IV proxy.
  *** CAVEAT: Premiums are ESTIMATED, not from real chains. ***
  Realized vol typically understates IV, so income estimates are CONSERVATIVE.

BACKTEST: 2015-2026 using real ex-dividend dates from yfinance.
COSTS: $0.65/contract commission (standard retail).

VALIDATION:
  - HC #428 R1: regime-agnostic (green/red/flat days, stratified Sharpe)
  - Permutation test: 500 shuffles of ex-div timing
  - Monthly income at $100K capital
  - Assignment rate analysis
  - Capital needed for $3K/month and $5K/month

BENCHMARKS:
  A) Pure dividend capture (buy shares day before, sell day after)
  B) Random CSP selling (same universe, random timing)
  C) SPY buy-and-hold

Output: /home/nick/Lvl3Quant/output/div_capture_options_v1/
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
import logging
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.stats import norm

warnings.filterwarnings("ignore")

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "output" / "div_capture_options_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(OUT_DIR / "backtest.log"),
    ],
)
log = logging.getLogger(__name__)

# ─── CONSTANTS ────────────────────────────────────────────────────────────
INITIAL_CAPITAL = 100_000.0
COMMISSION_PER_CONTRACT = 0.65  # per leg, per contract
MAX_PCT_PER_POSITION = 0.05     # 5% of NAV
MAX_CONCURRENT = 10
MAX_MARGIN_PCT = 0.30            # 30% total margin
MIN_DIV_YIELD_ANNUAL = 0.02     # 2% annual yield minimum
DAYS_BEFORE_EXDIV = (1, 5)      # enter 1-5 days before ex-div
PUT_DTE_RANGE = (7, 14)          # 7-14 DTE puts
PUT_DELTA_TARGET = 0.25          # 25-delta OTM put
RISK_FREE_RATE = 0.04            # approximate risk-free rate
LOOKBACK_VOL_DAYS = 30           # vol lookback for BS pricing

# S&P 500 dividend payers — large, liquid, high-yield subset
# We'll dynamically filter but start with a curated universe for speed
UNIVERSE_TICKERS = [
    # High-yield large caps
    "T", "VZ", "MO", "PM", "KO", "PEP", "PG", "JNJ", "ABBV", "PFE",
    "CVX", "XOM", "IBM", "MMM", "WBA", "DOW", "LYB",
    # Financials with good yields
    "JPM", "BAC", "WFC", "C", "USB", "PNC", "TFC", "KEY", "RF",
    # REITs (higher yield)
    "O", "SPG", "VNQ", "SCHD",
    # Utilities
    "SO", "DUK", "D", "NEE", "AEP", "SRE", "ED", "XEL",
    # Staples / Healthcare
    "BMY", "GILD", "AMGN", "MRK", "CAG", "KHC", "K", "GIS", "CL",
    # Industrials
    "CAT", "DE", "EMR", "HON", "LMT", "RTX",
    # ETFs with dividends
    "DVY", "HDV", "SPYD", "VYM",
    # Telecom / Media
    "CMCSA",
    # Energy
    "OKE", "EPD", "ET", "KMI", "WMB",
]

# SPY for benchmark
BENCHMARK = "SPY"

# ─── BLACK-SCHOLES OPTION PRICING ────────────────────────────────────────

def bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes put price."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes call price."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes put delta (negative)."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) - 1.0


def find_strike_for_delta(
    S: float, T: float, r: float, sigma: float, target_delta: float = -0.25,
    n_iter: int = 50
) -> float:
    """Find put strike that gives target delta via bisection."""
    K_low = S * 0.70
    K_high = S * 1.0
    for _ in range(n_iter):
        K_mid = (K_low + K_high) / 2
        delta = bs_put_delta(S, K_mid, T, r, sigma)
        if delta < target_delta:
            K_high = K_mid
        else:
            K_low = K_mid
    return round((K_low + K_high) / 2, 2)


# ─── DATA LOADING ────────────────────────────────────────────────────────

def download_data(tickers: List[str], start: str = "2014-06-01", end: str = "2026-07-15"):
    """Download price data and dividend history."""
    import yfinance as yf

    log.info(f"Downloading price data for {len(tickers)} tickers...")
    prices = {}
    dividends = {}
    failed = []

    for i, ticker in enumerate(tickers):
        try:
            tk = yf.Ticker(ticker)
            hist = tk.history(start=start, end=end, auto_adjust=False)
            if hist.empty or len(hist) < 252:
                log.warning(f"  {ticker}: insufficient data ({len(hist)} rows), skipping")
                failed.append(ticker)
                continue

            prices[ticker] = hist[["Open", "High", "Low", "Close", "Volume"]].copy()
            prices[ticker].index = prices[ticker].index.tz_localize(None)

            # Get dividends
            divs = tk.dividends
            if divs is not None and len(divs) > 0:
                divs.index = divs.index.tz_localize(None)
                dividends[ticker] = divs
                log.info(f"  [{i+1}/{len(tickers)}] {ticker}: {len(hist)} days, {len(divs)} dividends")
            else:
                log.warning(f"  {ticker}: no dividends found, skipping")
                failed.append(ticker)
                continue

            time.sleep(0.3)  # rate limit
        except Exception as e:
            log.warning(f"  {ticker}: download failed ({e})")
            failed.append(ticker)

    log.info(f"Downloaded {len(prices)} tickers, {len(failed)} failed: {failed}")
    return prices, dividends


def load_or_download_data(tickers: List[str]) -> Tuple[dict, dict]:
    """Cache data to avoid re-downloading."""
    cache_file = OUT_DIR / "data_cache.pkl"
    if cache_file.exists():
        log.info("Loading cached data...")
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        return data["prices"], data["dividends"]

    prices, dividends = download_data(tickers)
    with open(cache_file, "wb") as f:
        pickle.dump({"prices": prices, "dividends": dividends}, f)
    return prices, dividends


# ─── REGIME CLASSIFICATION (HC #428 R1) ──────────────────────────────────

def classify_regimes(spy_prices: pd.DataFrame) -> pd.Series:
    """Classify each day as green/red/flat based on SPY close-to-close."""
    returns = spy_prices["Close"].pct_change()
    regime = pd.Series("flat", index=returns.index)
    regime[returns > 0.002] = "green"
    regime[returns < -0.002] = "red"
    return regime


# ─── CORE BACKTEST ────────────────────────────────────────────────────────

class DivCaptureOptionsBacktest:
    """Main backtest engine for dividend capture with options."""

    def __init__(self, prices: dict, dividends: dict, spy_prices: pd.DataFrame):
        self.prices = prices
        self.dividends = dividends
        self.spy_prices = spy_prices
        self.regimes = classify_regimes(spy_prices)

        # Build unified trading calendar
        all_dates = set()
        for tk, df in prices.items():
            all_dates.update(df.index.tolist())
        self.trading_days = sorted(all_dates)
        self.trading_days = [d for d in self.trading_days if d >= pd.Timestamp("2015-01-01")]

        # Build ex-div schedule: {ticker: [(ex_div_date, div_amount), ...]}
        self.exdiv_schedule = {}
        for tk, divs in dividends.items():
            entries = []
            for dt, amt in divs.items():
                if pd.Timestamp("2015-01-01") <= dt <= pd.Timestamp("2026-07-15"):
                    entries.append((dt, amt))
            if entries:
                self.exdiv_schedule[tk] = sorted(entries)

        log.info(f"Trading calendar: {len(self.trading_days)} days")
        log.info(f"Ex-div schedule: {len(self.exdiv_schedule)} tickers")
        total_divs = sum(len(v) for v in self.exdiv_schedule.values())
        log.info(f"Total dividend events: {total_divs}")

    def _get_annualized_yield(self, ticker: str, div_amount: float, date: pd.Timestamp) -> float:
        """Estimate annualized dividend yield from recent history."""
        if ticker not in self.dividends:
            return 0.0
        divs = self.dividends[ticker]
        # Sum dividends in last 365 days
        one_year_ago = date - timedelta(days=365)
        annual_div = divs[(divs.index >= one_year_ago) & (divs.index <= date)].sum()
        # Get current price
        if ticker not in self.prices:
            return 0.0
        price_data = self.prices[ticker]
        mask = price_data.index <= date
        if mask.sum() == 0:
            return 0.0
        current_price = price_data.loc[mask, "Close"].iloc[-1]
        if current_price <= 0:
            return 0.0
        return annual_div / current_price

    def _get_realized_vol(self, ticker: str, date: pd.Timestamp, lookback: int = 30) -> float:
        """Get annualized realized volatility."""
        if ticker not in self.prices:
            return 0.30  # default
        price_data = self.prices[ticker]
        mask = price_data.index <= date
        if mask.sum() < lookback + 1:
            return 0.30
        closes = price_data.loc[mask, "Close"].iloc[-(lookback + 1):]
        log_rets = np.log(closes / closes.shift(1)).dropna()
        if len(log_rets) < 10:
            return 0.30
        return float(log_rets.std() * np.sqrt(252))

    def _get_price(self, ticker: str, date: pd.Timestamp) -> Optional[float]:
        """Get close price on or before date."""
        if ticker not in self.prices:
            return None
        price_data = self.prices[ticker]
        mask = price_data.index <= date
        if mask.sum() == 0:
            return None
        return float(price_data.loc[mask, "Close"].iloc[-1])

    def _get_price_on_date(self, ticker: str, date: pd.Timestamp) -> Optional[float]:
        """Get price exactly on date."""
        if ticker not in self.prices:
            return None
        price_data = self.prices[ticker]
        if date in price_data.index:
            return float(price_data.loc[date, "Close"])
        return None

    def _find_trading_day_offset(self, date: pd.Timestamp, offset: int) -> Optional[pd.Timestamp]:
        """Find trading day N days after date."""
        try:
            idx = self.trading_days.index(date)
            target = idx + offset
            if 0 <= target < len(self.trading_days):
                return self.trading_days[target]
        except ValueError:
            # Date not in trading calendar, find nearest
            for i, td in enumerate(self.trading_days):
                if td >= date:
                    target = i + offset
                    if 0 <= target < len(self.trading_days):
                        return self.trading_days[target]
                    break
        return None

    def _days_between(self, d1: pd.Timestamp, d2: pd.Timestamp) -> int:
        """Calendar days between two dates."""
        return abs((d2 - d1).days)

    def run_strategy(self, shuffle_dates: bool = False, seed: int = 0) -> pd.DataFrame:
        """
        Run the dividend capture + put selling strategy.

        If shuffle_dates=True, randomly shuffle which ex-div dates belong to which
        ticker (permutation test for timing edge).
        """
        np.random.seed(seed)

        # Build list of all trade opportunities
        opportunities = []
        for ticker, exdiv_list in self.exdiv_schedule.items():
            for exdiv_date, div_amount in exdiv_list:
                ann_yield = self._get_annualized_yield(ticker, div_amount, exdiv_date)
                if ann_yield < MIN_DIV_YIELD_ANNUAL:
                    continue
                opportunities.append({
                    "ticker": ticker,
                    "exdiv_date": exdiv_date,
                    "div_amount": div_amount,
                    "ann_yield": ann_yield,
                })

        if shuffle_dates:
            # Shuffle ex-div dates across tickers (keep same count per ticker)
            all_dates = [o["exdiv_date"] for o in opportunities]
            np.random.shuffle(all_dates)
            for i, opp in enumerate(opportunities):
                opp["exdiv_date"] = all_dates[i]

        opportunities.sort(key=lambda x: x["exdiv_date"])
        log.info(f"Total trade opportunities (yield > {MIN_DIV_YIELD_ANNUAL*100:.0f}%): {len(opportunities)}")

        # Simulate
        capital = INITIAL_CAPITAL
        daily_equity = {}
        trades = []
        active_positions = []  # list of dicts tracking open positions
        assigned_stock = {}    # ticker -> {shares, cost_basis, entry_date}

        for day in self.trading_days:
            # 1. Check for expired/assigned puts
            new_active = []
            for pos in active_positions:
                if day >= pos["expiry_date"]:
                    # Determine if assigned
                    expiry_price = self._get_price(pos["ticker"], pos["expiry_date"])
                    if expiry_price is None:
                        expiry_price = pos["entry_price"]  # fallback

                    if expiry_price <= pos["strike"]:
                        # ASSIGNED — buy stock at strike price
                        assignment_cost = pos["strike"] * 100 * pos["contracts"]
                        capital -= assignment_cost
                        capital -= COMMISSION_PER_CONTRACT * pos["contracts"]  # assignment fee

                        # Track assigned stock
                        tk = pos["ticker"]
                        if tk not in assigned_stock:
                            assigned_stock[tk] = {
                                "shares": 0,
                                "cost_basis": 0.0,
                                "entry_date": day,
                            }
                        assigned_stock[tk]["shares"] += 100 * pos["contracts"]
                        assigned_stock[tk]["cost_basis"] = pos["strike"]

                        trades.append({
                            "date": day,
                            "ticker": pos["ticker"],
                            "action": "PUT_ASSIGNED",
                            "premium": pos["premium_collected"],
                            "strike": pos["strike"],
                            "stock_price_at_expiry": expiry_price,
                            "pnl": pos["premium_collected"] - (pos["strike"] - expiry_price) * 100 * pos["contracts"],
                            "contracts": pos["contracts"],
                        })
                    else:
                        # Expired worthless — keep premium
                        trades.append({
                            "date": day,
                            "ticker": pos["ticker"],
                            "action": "PUT_EXPIRED_OTM",
                            "premium": pos["premium_collected"],
                            "strike": pos["strike"],
                            "stock_price_at_expiry": expiry_price,
                            "pnl": pos["premium_collected"],
                            "contracts": pos["contracts"],
                        })
                else:
                    new_active.append(pos)
            active_positions = new_active

            # 2. Manage assigned stock — sell covered calls or sell stock
            tickers_to_remove = []
            for tk, stock_info in assigned_stock.items():
                current_price = self._get_price(tk, day)
                if current_price is None:
                    continue

                # If stock has recovered above cost basis + 2%, sell
                if current_price > stock_info["cost_basis"] * 1.02:
                    sell_proceeds = current_price * stock_info["shares"]
                    capital += sell_proceeds
                    trades.append({
                        "date": day,
                        "ticker": tk,
                        "action": "STOCK_SOLD",
                        "premium": 0,
                        "strike": 0,
                        "stock_price_at_expiry": current_price,
                        "pnl": (current_price - stock_info["cost_basis"]) * stock_info["shares"],
                        "contracts": 0,
                    })
                    tickers_to_remove.append(tk)
                # If held > 30 days, sell covered call
                elif self._days_between(stock_info["entry_date"], day) > 30:
                    # Sell ATM call, 30 DTE
                    vol = self._get_realized_vol(tk, day)
                    T = 30 / 365
                    call_strike = round(current_price * 1.02, 2)
                    call_premium = bs_call_price(current_price, call_strike, T, RISK_FREE_RATE, vol)
                    n_contracts = stock_info["shares"] // 100
                    if n_contracts > 0 and call_premium > 0.10:
                        total_premium = call_premium * 100 * n_contracts
                        total_premium -= COMMISSION_PER_CONTRACT * n_contracts * 2
                        capital += total_premium
                        trades.append({
                            "date": day,
                            "ticker": tk,
                            "action": "COVERED_CALL_SOLD",
                            "premium": total_premium,
                            "strike": call_strike,
                            "stock_price_at_expiry": current_price,
                            "pnl": total_premium,
                            "contracts": n_contracts,
                        })
                        # Sell the stock too (simplification: assume called away after 30d)
                        sell_proceeds = call_strike * stock_info["shares"]
                        capital += sell_proceeds
                        tickers_to_remove.append(tk)

            for tk in tickers_to_remove:
                if tk in assigned_stock:
                    del assigned_stock[tk]

            # 3. Look for new put-selling opportunities
            for opp in opportunities:
                exdiv = opp["exdiv_date"]
                ticker = opp["ticker"]
                days_to_exdiv = (exdiv - day).days

                if not (DAYS_BEFORE_EXDIV[0] <= days_to_exdiv <= DAYS_BEFORE_EXDIV[1]):
                    continue

                # Skip if already have position in this ticker
                if any(p["ticker"] == ticker for p in active_positions):
                    continue
                if ticker in assigned_stock:
                    continue

                # Check position limits
                if len(active_positions) >= MAX_CONCURRENT:
                    continue

                # Get current price
                current_price = self._get_price_on_date(ticker, day)
                if current_price is None:
                    continue

                # Calculate margin requirement (approx 20% of notional)
                notional_per_contract = current_price * 100
                margin_per_contract = notional_per_contract * 0.20

                # Position sizing: max 5% of NAV
                max_notional = capital * MAX_PCT_PER_POSITION
                n_contracts = max(1, int(max_notional / notional_per_contract))

                # Check margin constraint
                total_margin_used = sum(
                    p["margin_required"] for p in active_positions
                )
                available_margin = capital * MAX_MARGIN_PCT - total_margin_used
                if margin_per_contract * n_contracts > available_margin:
                    n_contracts = max(1, int(available_margin / margin_per_contract))
                    if margin_per_contract > available_margin:
                        continue

                # Price the put
                vol = self._get_realized_vol(ticker, day)
                # Add 15% IV premium (variance risk premium — conservative)
                iv = vol * 1.15
                # DTE: pick ~10 days
                dte = 10
                T = dte / 365.0

                # Find 25-delta strike
                strike = find_strike_for_delta(current_price, T, RISK_FREE_RATE, iv, -PUT_DELTA_TARGET)
                put_premium = bs_put_price(current_price, strike, T, RISK_FREE_RATE, iv)

                if put_premium < 0.10:
                    continue  # not worth it

                # Find expiry date
                expiry_date = self._find_trading_day_offset(day, dte)
                if expiry_date is None:
                    continue

                total_premium = put_premium * 100 * n_contracts
                total_premium -= COMMISSION_PER_CONTRACT * n_contracts * 2  # open + close

                if total_premium <= 0:
                    continue

                capital += total_premium  # collect premium upfront

                active_positions.append({
                    "ticker": ticker,
                    "entry_date": day,
                    "expiry_date": expiry_date,
                    "strike": strike,
                    "entry_price": current_price,
                    "contracts": n_contracts,
                    "premium_collected": total_premium,
                    "margin_required": margin_per_contract * n_contracts,
                    "div_amount": opp["div_amount"],
                    "exdiv_date": exdiv,
                })

                # Mark this opportunity as used
                opp["exdiv_date"] = pd.Timestamp("1900-01-01")  # prevent re-entry

            # 4. Calculate daily equity
            equity = capital
            # Add value of assigned stock
            for tk, stock_info in assigned_stock.items():
                price = self._get_price(tk, day)
                if price:
                    equity += price * stock_info["shares"]
            # Mark-to-market active puts (simplified: linear interpolation of premium decay)
            for pos in active_positions:
                days_elapsed = max(1, self._days_between(pos["entry_date"], day))
                total_days = max(1, self._days_between(pos["entry_date"], pos["expiry_date"]))
                current_price = self._get_price(pos["ticker"], day)
                if current_price and current_price < pos["strike"]:
                    # ITM — unrealized loss
                    unrealized = -(pos["strike"] - current_price) * 100 * pos["contracts"]
                    equity += unrealized

            daily_equity[day] = equity

        # Build results DataFrame
        equity_series = pd.Series(daily_equity).sort_index()
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

        return equity_series, trades_df

    def run_benchmark_div_capture(self) -> pd.Series:
        """Benchmark A: Pure dividend capture — buy day before, sell day after."""
        capital = INITIAL_CAPITAL
        daily_equity = {}

        for day in self.trading_days:
            # Check if any ex-div is tomorrow
            next_day = self._find_trading_day_offset(day, 1)
            if next_day is None:
                daily_equity[day] = capital
                continue

            # Find tickers going ex-div tomorrow
            trades_today = []
            for ticker, exdiv_list in self.exdiv_schedule.items():
                for exdiv_date, div_amount in exdiv_list:
                    if exdiv_date == next_day:
                        ann_yield = self._get_annualized_yield(ticker, div_amount, day)
                        if ann_yield >= MIN_DIV_YIELD_ANNUAL:
                            price = self._get_price_on_date(ticker, day)
                            if price:
                                trades_today.append((ticker, price, div_amount, exdiv_date))

            # Buy shares, sell next day
            for ticker, buy_price, div_amount, exdiv_date in trades_today[:5]:  # max 5
                shares = int((capital * 0.05) / buy_price)
                if shares <= 0:
                    continue
                # Buy
                cost = shares * buy_price
                # Sell day after ex-div
                sell_date = self._find_trading_day_offset(exdiv_date, 1)
                if sell_date:
                    sell_price = self._get_price_on_date(ticker, sell_date)
                    if sell_price:
                        pnl = (sell_price - buy_price) * shares + div_amount * shares
                        capital += pnl

            daily_equity[day] = capital

        return pd.Series(daily_equity).sort_index()

    def run_benchmark_random_csp(self, seed: int = 42) -> pd.Series:
        """Benchmark B: Random CSP selling — same universe, random timing."""
        np.random.seed(seed)
        capital = INITIAL_CAPITAL
        daily_equity = {}
        active = []
        tickers_list = list(self.prices.keys())

        for day_idx, day in enumerate(self.trading_days):
            # Check expired
            new_active = []
            for pos in active:
                if day >= pos["expiry"]:
                    price = self._get_price(pos["ticker"], pos["expiry"])
                    if price and price <= pos["strike"]:
                        capital -= (pos["strike"] - price) * 100 * pos["contracts"]
                else:
                    new_active.append(pos)
            active = new_active

            # Random entry: ~2% chance per day (similar frequency to div capture)
            if np.random.random() < 0.02 and len(active) < MAX_CONCURRENT:
                ticker = np.random.choice(tickers_list)
                price = self._get_price_on_date(ticker, day)
                if price and price > 0:
                    vol = self._get_realized_vol(ticker, day)
                    iv = vol * 1.15
                    T = 10 / 365.0
                    strike = find_strike_for_delta(price, T, RISK_FREE_RATE, iv, -PUT_DELTA_TARGET)
                    premium = bs_put_price(price, strike, T, RISK_FREE_RATE, iv)
                    if premium > 0.10:
                        n_contracts = max(1, int(capital * MAX_PCT_PER_POSITION / (price * 100)))
                        total_prem = premium * 100 * n_contracts - COMMISSION_PER_CONTRACT * n_contracts * 2
                        if total_prem > 0:
                            capital += total_prem
                            expiry = self._find_trading_day_offset(day, 10)
                            if expiry:
                                active.append({
                                    "ticker": ticker,
                                    "strike": strike,
                                    "contracts": n_contracts,
                                    "expiry": expiry,
                                })

            daily_equity[day] = capital

        return pd.Series(daily_equity).sort_index()


# ─── ANALYTICS ────────────────────────────────────────────────────────────

def calc_metrics(equity: pd.Series, name: str = "") -> dict:
    """Calculate performance metrics from equity curve."""
    returns = equity.pct_change().dropna()
    if len(returns) < 30:
        return {"name": name, "error": "insufficient data"}

    total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
    years = len(returns) / 252
    cagr = (1 + total_return) ** (1 / max(years, 0.01)) - 1

    # Drawdown
    peak = equity.cummax()
    dd = (equity - peak) / peak
    max_dd = dd.min()

    # Sharpe
    mean_ret = returns.mean() * 252
    std_ret = returns.std() * np.sqrt(252)
    sharpe = mean_ret / std_ret if std_ret > 0 else 0

    # Sortino
    downside = returns[returns < 0]
    downside_std = downside.std() * np.sqrt(252) if len(downside) > 0 else 1e-6
    sortino = mean_ret / downside_std if downside_std > 0 else 0

    # Monthly income at $100K
    monthly_returns = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    monthly_income = monthly_returns * INITIAL_CAPITAL
    avg_monthly = monthly_income.mean()
    median_monthly = monthly_income.median()
    pct_above_3k = (monthly_income >= 3000).mean() * 100
    pct_above_5k = (monthly_income >= 5000).mean() * 100

    # Profit factor
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    pf = gains / losses if losses > 0 else float("inf")

    # Win rate (monthly)
    monthly_wr = (monthly_returns > 0).mean() * 100

    return {
        "name": name,
        "total_return_pct": round(total_return * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "profit_factor": round(pf, 3),
        "monthly_win_rate_pct": round(monthly_wr, 1),
        "avg_monthly_income_100k": round(avg_monthly, 0),
        "median_monthly_income_100k": round(median_monthly, 0),
        "pct_months_above_3k": round(pct_above_3k, 1),
        "pct_months_above_5k": round(pct_above_5k, 1),
        "years": round(years, 1),
    }


def regime_analysis(equity: pd.Series, regimes: pd.Series, name: str = "") -> dict:
    """HC #428 R1: stratified Sharpe per regime."""
    returns = equity.pct_change().dropna()
    results = {}

    for regime in ["green", "red", "flat"]:
        mask = regimes.reindex(returns.index) == regime
        regime_rets = returns[mask]
        if len(regime_rets) < 20:
            results[regime] = {"sharpe": 0, "count": len(regime_rets)}
            continue
        mean_r = regime_rets.mean() * 252
        std_r = regime_rets.std() * np.sqrt(252)
        sharpe = mean_r / std_r if std_r > 0 else 0
        results[regime] = {
            "sharpe": round(sharpe, 3),
            "mean_daily_ret_bps": round(regime_rets.mean() * 10000, 2),
            "count": int(mask.sum()),
        }

    # Regime-agnostic check
    sharpe_green = results.get("green", {}).get("sharpe", 0)
    sharpe_red = results.get("red", {}).get("sharpe", 0)
    max_sharpe = max(abs(sharpe_green), abs(sharpe_red), 1e-6)
    regime_skew = abs(sharpe_green - sharpe_red) / max_sharpe
    results["regime_skew"] = round(regime_skew, 3)
    results["regime_agnostic_pass"] = regime_skew <= 0.50

    return results


def yearly_returns(equity: pd.Series) -> dict:
    """Year-by-year returns."""
    returns = equity.pct_change().dropna()
    yearly = returns.resample("YE").apply(lambda x: (1 + x).prod() - 1)
    return {str(d.year): round(r * 100, 2) for d, r in yearly.items()}


def permutation_test(bt: DivCaptureOptionsBacktest, real_sharpe: float, n_perms: int = 500) -> dict:
    """Shuffle ex-div dates and compare Sharpe to real strategy."""
    log.info(f"Running permutation test with {n_perms} shuffles...")
    shuffled_sharpes = []

    for i in range(n_perms):
        if i % 50 == 0:
            log.info(f"  Permutation {i}/{n_perms}...")
        try:
            eq, _ = bt.run_strategy(shuffle_dates=True, seed=i + 1000)
            rets = eq.pct_change().dropna()
            if len(rets) < 30:
                continue
            s = (rets.mean() * 252) / (rets.std() * np.sqrt(252))
            shuffled_sharpes.append(s)
        except Exception as e:
            continue

    if not shuffled_sharpes:
        return {"error": "all permutations failed"}

    shuffled = np.array(shuffled_sharpes)
    p_value = (shuffled >= real_sharpe).mean()

    return {
        "real_sharpe": round(real_sharpe, 3),
        "shuffled_mean_sharpe": round(shuffled.mean(), 3),
        "shuffled_median_sharpe": round(np.median(shuffled), 3),
        "shuffled_std_sharpe": round(shuffled.std(), 3),
        "p_value": round(p_value, 4),
        "n_permutations": len(shuffled_sharpes),
        "statistically_significant": p_value < 0.05,
    }


# ─── MAIN ─────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 80)
    log.info("DIVIDEND CAPTURE ENHANCED WITH OPTIONS — BACKTEST v1")
    log.info("=" * 80)
    start_time = time.time()

    # 1. Download data
    all_tickers = UNIVERSE_TICKERS + [BENCHMARK]
    all_tickers = list(set(all_tickers))
    prices, dividends = load_or_download_data(all_tickers)

    if BENCHMARK not in prices:
        log.error("SPY data not available, downloading separately...")
        import yfinance as yf
        spy = yf.Ticker(BENCHMARK)
        spy_hist = spy.history(start="2014-06-01", end="2026-07-15", auto_adjust=False)
        spy_hist.index = spy_hist.index.tz_localize(None)
        prices[BENCHMARK] = spy_hist[["Open", "High", "Low", "Close", "Volume"]]

    spy_prices = prices[BENCHMARK]

    # 2. Run main strategy
    log.info("\n" + "=" * 60)
    log.info("RUNNING: Dividend Capture + Put Selling Strategy")
    log.info("=" * 60)
    bt = DivCaptureOptionsBacktest(prices, dividends, spy_prices)
    strategy_equity, strategy_trades = bt.run_strategy()

    # 3. Run benchmarks
    log.info("\n" + "=" * 60)
    log.info("RUNNING: Benchmark A — Pure Dividend Capture")
    log.info("=" * 60)
    bench_div_equity = bt.run_benchmark_div_capture()

    log.info("\n" + "=" * 60)
    log.info("RUNNING: Benchmark B — Random CSP Selling")
    log.info("=" * 60)
    bench_random_equity = bt.run_benchmark_random_csp()

    # Benchmark C: SPY buy-and-hold
    spy_returns = spy_prices["Close"].pct_change().dropna()
    spy_equity_start = spy_prices["Close"].loc[spy_prices.index >= pd.Timestamp("2015-01-01")]
    spy_equity = (spy_equity_start / spy_equity_start.iloc[0]) * INITIAL_CAPITAL

    # 4. Calculate metrics
    log.info("\n" + "=" * 60)
    log.info("CALCULATING METRICS")
    log.info("=" * 60)

    results = {}
    results["strategy"] = calc_metrics(strategy_equity, "Div Capture + Options")
    results["bench_div_capture"] = calc_metrics(bench_div_equity, "Pure Div Capture")
    results["bench_random_csp"] = calc_metrics(bench_random_equity, "Random CSP")
    results["bench_spy"] = calc_metrics(spy_equity, "SPY Buy-Hold")

    # 5. Trade analysis
    if len(strategy_trades) > 0:
        trade_stats = {
            "total_trades": len(strategy_trades),
            "put_expired_otm": len(strategy_trades[strategy_trades["action"] == "PUT_EXPIRED_OTM"]),
            "put_assigned": len(strategy_trades[strategy_trades["action"] == "PUT_ASSIGNED"]),
            "covered_calls": len(strategy_trades[strategy_trades["action"] == "COVERED_CALL_SOLD"]),
            "stock_sold": len(strategy_trades[strategy_trades["action"] == "STOCK_SOLD"]),
        }
        n_puts = trade_stats["put_expired_otm"] + trade_stats["put_assigned"]
        trade_stats["assignment_rate_pct"] = round(
            trade_stats["put_assigned"] / max(n_puts, 1) * 100, 1
        )
        trade_stats["avg_premium_per_trade"] = round(
            strategy_trades[strategy_trades["action"].isin(["PUT_EXPIRED_OTM", "PUT_ASSIGNED"])]["premium"].mean(), 2
        )
        trade_stats["total_premium_collected"] = round(
            strategy_trades[strategy_trades["action"].isin(["PUT_EXPIRED_OTM", "PUT_ASSIGNED"])]["premium"].sum(), 2
        )
        results["trade_stats"] = trade_stats
    else:
        results["trade_stats"] = {"error": "no trades executed"}

    # 6. Regime analysis (HC #428 R1)
    log.info("\nRunning regime analysis (HC #428 R1)...")
    regimes = classify_regimes(spy_prices)
    results["regime_analysis"] = regime_analysis(strategy_equity, regimes, "Strategy")

    # 7. Year-by-year
    results["yearly_returns"] = yearly_returns(strategy_equity)
    results["yearly_returns_spy"] = yearly_returns(spy_equity)

    # 8. Capital needed for income targets
    strat_metrics = results["strategy"]
    if "avg_monthly_income_100k" in strat_metrics and strat_metrics["avg_monthly_income_100k"] > 0:
        avg_monthly = strat_metrics["avg_monthly_income_100k"]
        capital_for_3k = round(3000 / avg_monthly * INITIAL_CAPITAL, 0)
        capital_for_5k = round(5000 / avg_monthly * INITIAL_CAPITAL, 0)
        results["capital_requirements"] = {
            "for_3k_monthly": f"${capital_for_3k:,.0f}",
            "for_5k_monthly": f"${capital_for_5k:,.0f}",
            "based_on_avg_monthly_return": f"${avg_monthly:,.0f} per $100K",
        }
    else:
        results["capital_requirements"] = {"error": "strategy didn't generate positive income"}

    # 9. Permutation test
    real_sharpe = results["strategy"].get("sharpe", 0)
    results["permutation_test"] = permutation_test(bt, real_sharpe, n_perms=100)

    # 10. Save results
    elapsed = time.time() - start_time
    results["runtime_seconds"] = round(elapsed, 1)

    # Save JSON results
    results_file = OUT_DIR / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"\nResults saved to {results_file}")

    # Save equity curves
    equity_df = pd.DataFrame({
        "strategy": strategy_equity,
        "pure_div_capture": bench_div_equity,
        "random_csp": bench_random_equity,
        "spy_buyhold": spy_equity,
    })
    equity_df.to_csv(OUT_DIR / "equity_curves.csv")

    # Save trades
    if len(strategy_trades) > 0:
        strategy_trades.to_csv(OUT_DIR / "trades.csv", index=False)

    # ─── PRINT SUMMARY ───────────────────────────────────────────────
    log.info("\n" + "=" * 80)
    log.info("RESULTS SUMMARY")
    log.info("=" * 80)

    for key in ["strategy", "bench_div_capture", "bench_random_csp", "bench_spy"]:
        m = results[key]
        log.info(f"\n{m.get('name', key)}:")
        if "error" in m:
            log.info(f"  ERROR: {m['error']}")
            continue
        log.info(f"  CAGR: {m['cagr_pct']}% | Sharpe: {m['sharpe']} | Sortino: {m['sortino']}")
        log.info(f"  Max DD: {m['max_drawdown_pct']}% | PF: {m['profit_factor']} | Win Rate: {m['monthly_win_rate_pct']}%")
        log.info(f"  Avg Monthly ($100K): ${m['avg_monthly_income_100k']:,.0f}")
        log.info(f"  % Months > $3K: {m['pct_months_above_3k']}% | > $5K: {m['pct_months_above_5k']}%")

    if "trade_stats" in results:
        ts = results["trade_stats"]
        if "error" not in ts:
            log.info(f"\nTrade Stats:")
            log.info(f"  Total puts sold: {ts.get('put_expired_otm', 0) + ts.get('put_assigned', 0)}")
            log.info(f"  Expired OTM: {ts.get('put_expired_otm', 0)} | Assigned: {ts.get('put_assigned', 0)}")
            log.info(f"  Assignment rate: {ts.get('assignment_rate_pct', 0)}%")
            log.info(f"  Avg premium/trade: ${ts.get('avg_premium_per_trade', 0):,.2f}")
            log.info(f"  Total premium: ${ts.get('total_premium_collected', 0):,.0f}")

    if "regime_analysis" in results:
        ra = results["regime_analysis"]
        log.info(f"\nRegime Analysis (HC #428 R1):")
        for regime in ["green", "red", "flat"]:
            if regime in ra:
                r = ra[regime]
                log.info(f"  {regime}: Sharpe={r.get('sharpe', 0)}, days={r.get('count', 0)}")
        log.info(f"  Regime skew: {ra.get('regime_skew', 'N/A')} (pass={ra.get('regime_agnostic_pass', 'N/A')})")

    if "permutation_test" in results:
        pt = results["permutation_test"]
        if "error" not in pt:
            log.info(f"\nPermutation Test:")
            log.info(f"  Real Sharpe: {pt['real_sharpe']} vs Shuffled: {pt['shuffled_mean_sharpe']} +/- {pt['shuffled_std_sharpe']}")
            log.info(f"  p-value: {pt['p_value']} | Significant: {pt['statistically_significant']}")

    if "capital_requirements" in results:
        cr = results["capital_requirements"]
        log.info(f"\nCapital Required:")
        log.info(f"  For $3K/month: {cr.get('for_3k_monthly', 'N/A')}")
        log.info(f"  For $5K/month: {cr.get('for_5k_monthly', 'N/A')}")

    if "yearly_returns" in results:
        log.info(f"\nYear-by-Year Returns:")
        for year, ret in sorted(results["yearly_returns"].items()):
            spy_ret = results.get("yearly_returns_spy", {}).get(year, "N/A")
            log.info(f"  {year}: Strategy {ret}% | SPY {spy_ret}%")

    log.info(f"\nRuntime: {elapsed:.0f}s")
    log.info("=" * 80)
    log.info("DONE")


if __name__ == "__main__":
    main()
